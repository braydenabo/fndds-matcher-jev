"""Evaluation: retrieval recall, system accuracy, prefix accuracy, nutrient distance, calibration."""
from __future__ import annotations

import csv
import hashlib
import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .fndds import FnddsIndex
from .matcher import NONE_OPTION, FnddsMatcher, MatchResult
from .normalize import normalize
from .retrieve import Retriever


@dataclass
class Label:
    food: str
    code: str
    context: dict | None = None
    tier: str = ""


def load_labels(path: str | Path) -> list[Label]:
    """CSV with columns: food, code [, context, tier]. Rows without a numeric code are skipped."""
    out = []
    with open(path, newline="", encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            code = (r.get("code") or "").strip()
            if code.endswith(".0"):
                code = code[:-2]
            if not code.isdigit():
                continue
            ctx = {"note": r["context"].strip()} if (r.get("context") or "").strip() else None
            out.append(Label(r["food"].strip(), code.zfill(8), ctx, (r.get("tier") or "").strip()))
    return out


def recall_at_k(index: FnddsIndex, retriever: Retriever, labels: list[Label], ks=(10, 30, 50, 100)) -> dict:
    """Search at each K separately: fused retrievers rank differently at different depths,
    and the matcher retrieves exactly K, so truncating a deeper list would overstate recall."""
    out = {}
    for k in ks:
        hits = [
            lab.code in {index.foods[i].code for i, _ in retriever.search(lab.food, k)} for lab in labels
        ]
        out[f"recall@{k}"] = sum(hits) / len(hits)
    top1 = [
        bool(r := retriever.search(lab.food, 1)) and index.foods[r[0][0]].code == lab.code
        for lab in labels
    ]
    out["top1_retrieval"] = sum(top1) / len(top1)
    return out


def _prefix_len(a: str, b: str) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def nutrient_distance(index: FnddsIndex, pred: str, true: str) -> dict[str, float]:
    p, t = index.by_code[pred].nutrients, index.by_code[true].nutrients
    return {k: abs(p.get(k, 0.0) - t.get(k, 0.0)) for k in t}


def expected_calibration_error(conf: np.ndarray, correct: np.ndarray, bins: int = 10) -> float:
    if len(conf) == 0:
        return float("nan")
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def coverage_precision(conf: np.ndarray, correct: np.ndarray, thresholds=None) -> list[dict]:
    thresholds = thresholds if thresholds is not None else np.linspace(0.3, 0.95, 14)
    rows = []
    for t in thresholds:
        m = conf >= t
        rows.append({"t": float(t), "coverage": float(m.mean()), "precision": float(correct[m].mean()) if m.any() else float("nan")})
    return rows


def split_of(food: str, fit_frac: float = 0.5, seed: int = 0) -> str:
    """Deterministic fit/test split keyed on the normalized food string, so the same food never
    lands on both sides (repeated strings would otherwise leak between fit and test)."""
    h = int(hashlib.sha1(f"{seed}:{normalize(food)}".encode()).hexdigest(), 16) % 10_000
    return "fit" if h < fit_frac * 10_000 else "test"


def nutrient_close(index: FnddsIndex, pred: str | None, true: str) -> bool:
    """Practically-equivalent: same code, or kcal within max(15, 10%) and protein/carb/fat each
    within max(2 g, 15%) per 100 g. A cheap-error notion, not a claim the foods are the same."""
    if pred is None:
        return False
    if pred == true:
        return True
    p, t = index.by_code[pred].nutrients, index.by_code[true].nutrients
    if not t or not p:
        return False
    if abs(p.get("kcal", 0) - t["kcal"]) > max(15.0, 0.10 * t["kcal"]):
        return False
    return all(abs(p.get(k, 0) - t.get(k, 0)) <= max(2.0, 0.15 * t.get(k, 0)) for k in ("protein_g", "carb_g", "fat_g"))


def _summ(items: list[dict]) -> dict:
    n = len(items)
    if not n:
        return {"n": 0}
    f = lambda k: sum(bool(i[k]) for i in items) / n  # noqa: E731
    return {
        "n": n,
        "exact_top1": f("exact"),
        "top3": f("top3"),
        "nutrient_close_top1": f("near"),
        "retrieval_only_top1": f("retrieval_top1"),
        "candidate_recall": f("in_candidates"),
        "accepted": sum(i["status"] == "accepted" for i in items),
    }


def evaluate_system(
    index: FnddsIndex, matcher: FnddsMatcher, labels: list[Label], batch: int = 20, fit_frac: float = 0.5, seed: int = 0,
    workers: int = 1,
) -> dict:
    """workers > 1 sends batches concurrently (retrieval and the Jev request of different batches overlap).
    Results keep label order. With shuffle-averaging, concurrent batches draw option orders from a shared RNG,
    so orders (not correctness) can differ from a sequential run."""
    chunks = [labels[i : i + batch] for i in range(0, len(labels), batch)]

    def run(chunk: list[Label]) -> list[MatchResult]:
        return matcher.match_many([(l.food, l.context) for l in chunk])

    if workers > 1 and len(chunks) > 1:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            parts = list(ex.map(run, chunks))
    else:
        parts = [run(c) for c in chunks]
    results: list[MatchResult] = [r for part in parts for r in part]

    items, conf, nut = [], [], {k: [] for k in ("kcal", "protein_g", "carb_g", "fat_g")}
    for lab, r in zip(labels, results):
        alt_codes = [a.code for a in r.alternatives]
        pred = alt_codes[0] if alt_codes else None
        true = index.by_code[lab.code]
        probs = r.probabilities
        items.append({
            "food": lab.food, "context": lab.context, "tier": lab.tier,
            "split": split_of(lab.food, fit_frac, seed),
            "true_code": lab.code, "true_description": true.description,
            "pred_code": pred, "pred_description": r.alternatives[0].description if r.alternatives else None,
            "status": r.status, "confidence": r.confidence,
            "p1": probs[0]["p"] if probs else 0.0, "p2": probs[1]["p"] if len(probs) > 1 else 0.0,
            "top_is_none": bool(probs) and probs[0]["option"] == NONE_OPTION,
            "probabilities": probs[:6],
            "exact": pred == lab.code, "top3": lab.code in alt_codes[:3],
            "near": nutrient_close(index, pred, lab.code),
            "in_candidates": lab.code in r.candidate_codes,
            "candidate_rank": r.candidate_codes.index(lab.code) + 1 if lab.code in r.candidate_codes else None,
            "retrieval_top1": bool(r.candidate_codes) and r.candidate_codes[0] == lab.code,
            "same_group": bool(pred) and pred[:1] == lab.code[:1],
            "same_subgroup": bool(pred) and _prefix_len(pred, lab.code) >= 3,
        })
        conf.append(r.confidence)
        if pred:
            for k, v in nutrient_distance(index, pred, lab.code).items():
                nut.setdefault(k, []).append(v)
    conf_a, ok_a = np.array(conf), np.array([i["exact"] for i in items], dtype=float)
    status = [r.status for r in results]
    by = lambda key: {v: _summ([i for i in items if i[key] == v]) for v in sorted({i[key] for i in items})}  # noqa: E731
    return {
        "n": len(labels),
        "recall_at_k": float(np.mean([i["in_candidates"] for i in items])),
        "top1": float(ok_a.mean()),
        "top3": float(np.mean([i["top3"] for i in items])),
        "nutrient_close_top1": float(np.mean([i["near"] for i in items])),
        "retrieval_only_top1": float(np.mean([i["retrieval_top1"] for i in items])),
        "same_group": float(np.mean([i["same_group"] for i in items])),
        "same_subgroup_3digit": float(np.mean([i["same_subgroup"] for i in items])),
        "nutrient_mae_per_100g": {k: float(np.mean(v)) for k, v in nut.items() if v},
        "ece": expected_calibration_error(conf_a, ok_a),
        "coverage_precision": coverage_precision(conf_a, ok_a),
        "status_counts": {s: status.count(s) for s in set(status)},
        "by_split": by("split"),
        "by_tier": by("tier"),
        "cost_usd": float(sum(r.cost_usd for r in results)),
        "items": items,
    }


def _accepted(item: dict, t_high: float, margin: float) -> bool:
    return (not item["top_is_none"]) and item["p1"] >= t_high and item["p1"] - item["p2"] >= margin


def score_thresholds(items: list[dict], t_high: float, margin: float, key: str = "exact") -> dict:
    acc = [i for i in items if _accepted(i, t_high, margin)]
    return {
        "t_high": t_high, "margin": margin, "n": len(items), "accepted": len(acc),
        "coverage": len(acc) / len(items) if items else 0.0,
        "precision": sum(bool(i[key]) for i in acc) / len(acc) if acc else float("nan"),
    }


def fit_thresholds(
    items: list[dict], target_precision: float, key: str = "exact",
    t_grid=tuple(np.round(np.arange(0.3, 0.99, 0.02), 2)), m_grid=(0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5),
) -> dict:
    """Pick the (t_high, margin) with the most coverage whose precision on `items` meets the target.
    `key` is 'exact' (code matches) or 'near' (nutrient-close). Returns None if unreachable."""
    best = None
    for t in t_grid:
        for m in m_grid:
            s = score_thresholds(items, float(t), float(m), key)
            if s["accepted"] and s["precision"] >= target_precision and (best is None or s["coverage"] > best["coverage"]):
                best = s
    return best


ERROR_ORDER = ["exact", "retrieval_miss", "none_despite_candidate", "close_miss", "same_subgroup_miss", "other_subgroup_miss"]


def categorize_error(item: dict) -> str:
    """Mutually exclusive outcome for one food, by precedence: a food whose true code never made
    the shortlist is a retrieval miss even if the answer happens to be nutritionally close."""
    if item["exact"]:
        return "exact"
    if not item["in_candidates"]:
        return "retrieval_miss"
    if item["top_is_none"]:
        return "none_despite_candidate"
    if item["near"]:
        return "close_miss"
    return "same_subgroup_miss" if item["same_subgroup"] else "other_subgroup_miss"


def error_breakdown(items: list[dict]) -> dict[str, int]:
    out = dict.fromkeys(ERROR_ORDER, 0)
    for i in items:
        out[categorize_error(i)] += 1
    return out
