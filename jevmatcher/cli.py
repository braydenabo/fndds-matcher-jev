from __future__ import annotations

import argparse
import json
from pathlib import Path

from .fndds import FnddsIndex, build_index

DEFAULT_INDEX = Path("data/fndds_index.json")


def _cmd_build(a) -> None:
    src = Path(a.csv_dir)
    idx = build_index(src, a.release)
    idx.save(a.out)
    print(f"{len(idx)} foods, release {idx.release}, no duplicate descriptions -> {a.out}")


def _cmd_download(a) -> None:
    from .fndds import download_release

    out = download_release(a.data_dir)
    idx = FnddsIndex.load(out)
    print(f"{len(idx)} foods, release {idx.release} -> {out}")


def _cmd_recall(a) -> None:
    from .evaluate import load_labels, recall_at_k
    from .retrieve import default_retriever

    idx = FnddsIndex.load(a.index)
    labels = load_labels(a.labels)
    print(json.dumps(recall_at_k(idx, default_retriever(idx), labels), indent=2), f"(n={len(labels)})")


def _cmd_match(a) -> None:
    from .jev import JevClient
    from .matcher import FnddsMatcher
    from .retrieve import default_retriever

    idx = FnddsIndex.load(a.index)
    m = FnddsMatcher(idx, default_retriever(idx), JevClient(), k=a.k, shuffles=a.shuffles, verify=a.verify)
    r = m.match(a.food, {"note": a.context} if a.context else None)
    print(json.dumps(r.__dict__, default=lambda o: o.__dict__, indent=2))


def _cmd_eval(a) -> None:
    from .evaluate import evaluate_system, load_labels
    from .jev import JevClient
    from .matcher import FnddsMatcher
    from .retrieve import default_retriever

    idx = FnddsIndex.load(a.index)
    if a.mock:
        from .mockjev import MockJev

        jev = MockJev(simulate_latency=False)
    else:
        jev = JevClient()
    m = FnddsMatcher(idx, default_retriever(idx), jev, k=a.k, shuffles=a.shuffles, verify=a.verify)
    out = evaluate_system(idx, m, load_labels(a.labels), fit_frac=a.fit_frac, seed=a.seed)
    items = out.pop("items")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps({"settings": vars(a) | {"fn": None}, "summary": out, "items": items}, indent=2, default=str))
        print(f"per-food results -> {a.out}")
    print(json.dumps(out, indent=2))


def _cmd_fit(a) -> None:
    from .evaluate import fit_thresholds, score_thresholds

    items = json.loads(Path(a.results).read_text())["items"]
    fit, test = [i for i in items if i["split"] == "fit"], [i for i in items if i["split"] == "test"]
    for key in ("exact", "near"):
        best = fit_thresholds(fit, a.target, key)
        print(f"\n== precision target {a.target:.0%} on '{key}' correctness (fit n={len(fit)}, test n={len(test)})")
        if best is None:
            print("  no threshold pair reaches the target on the fit split")
            continue
        t, m = best["t_high"], best["margin"]
        print(f"  chosen t_high={t} margin={m}")
        for name, part in (("fit ", fit), ("test", test)):
            s = score_thresholds(part, t, m, key)
            print(f"  {name}: coverage {s['coverage']:.1%}  precision {s['precision']:.1%}  ({s['accepted']}/{s['n']} accepted)")


def main(argv=None) -> None:
    from .env import load_dotenv

    load_dotenv()
    p = argparse.ArgumentParser(prog="jevmatcher")
    p.add_argument("--index", default=str(DEFAULT_INDEX))
    sub = p.add_subparsers(required=True)

    b = sub.add_parser("build-index", help="build the pinned FNDDS table from an extracted FDC survey CSV dir")
    b.add_argument("csv_dir")
    b.add_argument("--release", required=True, help="e.g. fdc-survey-2024-10-31")
    b.add_argument("--out", default=str(DEFAULT_INDEX))
    b.set_defaults(fn=_cmd_build)

    d = sub.add_parser("download-data", help="download the pinned USDA FNDDS release and build the index")
    d.add_argument("--data-dir", default="data")
    d.set_defaults(fn=_cmd_download)

    r = sub.add_parser("recall", help="retrieval-only Recall@K on a labels CSV (no Jev key needed)")
    r.add_argument("labels")
    r.set_defaults(fn=_cmd_recall)

    f = sub.add_parser("fit-thresholds", help="fit accept thresholds on the fit split of saved eval results, report on test")
    f.add_argument("results")
    f.add_argument("--target", type=float, default=0.9, help="minimum precision of accepted answers")
    f.set_defaults(fn=_cmd_fit)

    for name, fn in (("match", _cmd_match), ("eval", _cmd_eval)):
        s = sub.add_parser(name)
        s.add_argument("food" if name == "match" else "labels")
        s.add_argument("--k", type=int, default=30)
        s.add_argument("--shuffles", type=int, default=1)
        s.add_argument("--verify", action="store_true")
        if name == "eval":
            s.add_argument("--out", help="write per-food results (JSON) for error analysis and threshold fitting")
            s.add_argument("--fit-frac", type=float, default=0.5, help="share of foods assigned to the fit split")
            s.add_argument("--seed", type=int, default=0)
            s.add_argument("--mock", action="store_true", help="use the offline simulator (no key, no cost)")
        if name == "match":
            s.add_argument("--context", default="")
        s.set_defaults(fn=fn)

    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
