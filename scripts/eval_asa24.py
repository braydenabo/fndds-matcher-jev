"""Evaluate on the ASA24-to-FooDB benchmark (Lemay et al., J Nutr 2026;156:101678).

Download the two files once (see README, "Benchmark comparison"), then:

    python scripts/eval_asa24.py data/benchmarks/asa24_foodb                 # retrieval only, no key
    python scripts/eval_asa24.py data/benchmarks/asa24_foodb --mock          # plumbing test, no key/cost
    python scripts/eval_asa24.py data/benchmarks/asa24_foodb --live --out results/asa24.json   # real Jev

Protocol follows the paper: inputs whose id equals the target id are matched by id (counted correct);
the text-matching test is the remaining inputs. Correct = the predicted FooDB entry equals the labeled entry (identified by description text; FooDB ids are unreliable).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from jevmatcher.env import load_dotenv
from jevmatcher.evaluate import evaluate_system, recall_at_k
from jevmatcher.matcher import FnddsMatcher
from jevmatcher.retrieve import (
    EmbeddingRetriever, FastEmbedder, FuzzyRetriever, HybridRetriever, TfidfRetriever, default_retriever,
)
from jevmatcher.targets import load_asa24_foodb

# Semantic-embedding top-K accuracy on the same benchmark's text-only foods, from Lemay et al.
# Table 5 (GTE-large): original ground truth, then the authors' modified ground truth.
# Lemay et al.'s hybrid prompt chose by these criteria, in priority order (adapted to Jev's Choice question).
PAPER_PROMPT = (
    "Which candidate best matches `food`? Decide using these criteria in priority order: "
    "1) same animal or plant source; 2) similar nutritional profile (macronutrients, micronutrients, calories "
    "per serving); 3) same preparation method; 4) semantic and name similarity."
)
PAPER_TOPK = {5: (0.757, 0.854), 10: (0.826, 0.951), 25: (0.865, 0.958), 50: (0.896, 0.976)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--mock", action="store_true", help="simulated Jev (no key, no cost)")
    mode.add_argument("--live", action="store_true", help="real Jev (needs TYPESAFE_API_KEY, spends tokens)")
    ap.add_argument("--k", type=int, default=30)
    ap.add_argument("--paper-prompt", action="store_true", help="instruct Jev with the paper's ranked matching criteria")
    ap.add_argument("--no-none", action="store_true", help="forced choice: drop the 'none of these' option")
    ap.add_argument("--workers", type=int, default=8, help="concurrent Jev requests")
    ap.add_argument("--skip-retrieval-table", action="store_true", help="skip the retrieval-only comparison (slow)")
    ap.add_argument("--out")
    a = ap.parse_args()
    load_dotenv()

    bm = load_asa24_foodb(a.data_dir)
    idx = bm.index
    have = {f.code for f in idx.foods}
    labs = [l for l in bm.text_only if l.code in have]  # a few labels name no target that exists in the list
    n_id, n_txt = len(bm.id_matched), len(labs)
    if len(labs) != len(bm.text_only):
        print(f"dropped {len(bm.text_only) - len(labs)} text-only labels whose target is not in the target list")
    print(f"targets {len(idx)} | inputs {n_id + n_txt}: {n_id} matched by id, {n_txt} text-only (the test)")

    hybrid = default_retriever(idx)
    if not a.skip_retrieval_table:  # about 5 extra searches per food, so it dominates the runtime
        emb = EmbeddingRetriever(idx, FastEmbedder("BAAI/bge-small-en-v1.5", tag=idx.release))
        configs = {
            "bge-small only": emb,
            "Fuzzy + TF-IDF": HybridRetriever([FuzzyRetriever(idx), TfidfRetriever(idx)]),
            "Hybrid (default)": hybrid,
        }
        ks = (1, 5, 10, 25, 50)
        print(f"\nRetrieval on the {n_txt} text-only foods (recall@K = right FooDB entry in top K):")
        print(f"{'':28}" + "".join(f"{'@' + str(k):>10}" for k in ks))
        for name, r in configs.items():
            res = recall_at_k(idx, r, labs, ks=ks)
            print(f"{name:28}" + "".join(f"{res[f'recall@{k}']:>10.3f}" for k in ks))
        print(f"{'paper GTE-large (orig/mod)':28}" + "".join(
            f"{PAPER_TOPK[k][0]:.2f}/{PAPER_TOPK[k][1]:.2f}".rjust(10) if k in PAPER_TOPK else "-".rjust(10) for k in ks))

    if not (a.mock or a.live):
        return
    if a.live:
        from jevmatcher.jev import JevClient

        jev = JevClient()
    else:
        from jevmatcher.mockjev import MockJev

        jev = MockJev(simulate_latency=False)
    m = FnddsMatcher(idx, hybrid, jev, k=a.k, instructions=PAPER_PROMPT if a.paper_prompt else None, allow_none=not a.no_none)
    out = evaluate_system(idx, m, labs, workers=a.workers)
    items = out.pop("items")
    overall = (n_id + sum(i["exact"] for i in items)) / (n_id + n_txt)
    tag = "LIVE Jev" if a.live else "SIMULATED Jev (meaningless accuracy)"
    print(f"\n{tag}, K={a.k}, text-only foods (n={n_txt}):")
    print(f"  retrieval-only top-1  {out['retrieval_only_top1']:.3f}")
    print(f"  pipeline top-1        {out['top1']:.3f}   top-3 {out['top3']:.3f}   candidate recall {out['recall_at_k']:.3f}")
    print(f"  status counts         {out['status_counts']}")
    print(f"  paper-style overall   {overall:.3f}  (id-matched counted correct + text-only exact; paper hybrid: 0.907)")
    print(f"  cost ${out['cost_usd']:.4f}")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps({"summary": out, "items": items}, indent=2, default=str))
        print("per-food results ->", a.out)


if __name__ == "__main__":
    main()
