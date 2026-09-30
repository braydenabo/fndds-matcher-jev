"""Load an arbitrary (id, description) target list as a matcher index, and the ASA24-to-FooDB
benchmark of Lemay et al. (J Nutr 2026;156:101678), https://github.com/dglemay/USDA-Food-Mapping.

Their repo has no license, so the data is downloaded locally (git-ignored), never redistributed here.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from .evaluate import Label
from .fndds import FnddsIndex, Food


def _tsv(path: Path):
    # QUOTE_NONE: descriptions contain literal quote characters that must not be parsed as CSV quoting.
    with open(path, newline="", encoding="utf-8") as fh:
        yield from csv.DictReader(fh, delimiter="\t", quoting=csv.QUOTE_NONE)


def build_target_index(rows: list[tuple[str, str]], release: str) -> FnddsIndex:
    """rows: (target_id, description). Targets are identified by their description text.

    FooDB ids are not usable as identities here: about 12% are blank (1,201 different foods share
    it), some ids carry several descriptions, and a labeled target's id can differ from the id the
    same description has in the target list. Descriptions are also what Jev sees, and they must be
    unique, so repeated descriptions collapse to one target.
    """
    seen, foods = set(), []
    for _, desc in rows:
        if desc not in seen:
            seen.add(desc)
            foods.append(Food(code=desc, description=desc))
    return FnddsIndex(release=release, foods=foods)


@dataclass
class Asa24Benchmark:
    index: FnddsIndex
    id_matched: list[Label]  # the target has the same id as the input (the paper matches these by id)
    text_only: list[Label]  # no id match: the real text-matching test


def load_asa24_foodb(data_dir: str | Path) -> Asa24Benchmark:
    d = Path(data_dir)
    index = build_target_index(
        [(r["target_id"], r["target_desc"]) for r in _tsv(d / "target_desc_fooDB.txt")], "asa24-foodb"
    )
    id_matched, text_only = [], []
    for r in _tsv(d / "groundtruth_ASA24toFooDB.txt"):
        lab = Label(food=r["input_desc"], code=r["target_desc"])
        (id_matched if r["input_id"].strip() and r["input_id"] == r["target_id"] else text_only).append(lab)
    return Asa24Benchmark(index, id_matched, text_only)
