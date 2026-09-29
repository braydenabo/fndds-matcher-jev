"""Evaluation: retrieval recall, system accuracy, prefix accuracy, nutrient distance, calibration."""
from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .fndds import FnddsIndex
from .matcher import FnddsMatcher, MatchResult
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


def evaluate_system(index: FnddsIndex, matcher: FnddsMatcher, labels: list[Label], batch: int = 20) -> dict:
    results: list[MatchResult] = []
    for i in range(0, len(labels), batch):
        chunk = labels[i : i + batch]
        results += matcher.match_many([(l.food, l.context) for l in chunk])

    top1, top3, in_cands, group, subgroup = [], [], [], [], []
    conf, nut = [], {k: [] for k in ("kcal", "protein_g", "carb_g", "fat_g")}
    for lab, r in zip(labels, results):
        alt_codes = [a.code for a in r.alternatives]
        pred = alt_codes[0] if alt_codes else None
        top1.append(pred == lab.code)
        top3.append(lab.code in alt_codes[:3])
        in_cands.append(lab.code in r.candidate_codes)
        group.append(bool(pred) and pred[:1] == lab.code[:1])
        subgroup.append(bool(pred) and _prefix_len(pred, lab.code) >= 3)
        conf.append(r.confidence)
        if pred:
            for k, v in nutrient_distance(index, pred, lab.code).items():
                nut.setdefault(k, []).append(v)
    conf_a, ok_a = np.array(conf), np.array(top1, dtype=float)
    status = [r.status for r in results]
    return {
        "n": len(labels),
        "recall_at_k": float(np.mean(in_cands)),
        "top1": float(np.mean(top1)),
        "top3": float(np.mean(top3)),
        "same_group": float(np.mean(group)),
        "same_subgroup_3digit": float(np.mean(subgroup)),
        "nutrient_mae_per_100g": {k: float(np.mean(v)) for k, v in nut.items() if v},
        "ece": expected_calibration_error(conf_a, ok_a),
        "coverage_precision": coverage_precision(conf_a, ok_a),
        "status_counts": {s: status.count(s) for s in set(status)},
        "cost_usd": float(sum(r.cost_usd for r in results)),
        "results": results,
    }
