"""FNDDS data layer: build a pinned code table from a FoodData_Central survey CSV release."""
from __future__ import annotations

import csv
import io
import json
import zipfile
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

# Nutrient numbers (this survey release keys food_nutrient by nutrient_nbr, not the 1003-series id).
MACROS = {208: "kcal", 203: "protein_g", 205: "carb_g", 204: "fat_g"}


@dataclass(frozen=True)
class Food:
    code: str
    description: str
    additional: tuple[str, ...] = ()
    common_names: tuple[str, ...] = ()
    wweia_category: str = ""
    wweia_category_description: str = ""
    nutrients: dict[str, float] = field(default_factory=dict)  # per 100 g

    @property
    def group(self) -> str:
        """Major food group: the first digit of the 8-digit FNDDS code."""
        return self.code[:1]

    @property
    def search_text(self) -> str:
        return " | ".join((self.description, *self.additional, *self.common_names))


@dataclass
class FnddsIndex:
    release: str
    foods: list[Food]

    def __post_init__(self) -> None:
        self.by_code = {f.code: f for f in self.foods}
        self.by_description = {f.description: f for f in self.foods}

    def __len__(self) -> int:
        return len(self.foods)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps({"release": self.release, "foods": [asdict(f) for f in self.foods]})
        )

    @classmethod
    def load(cls, path: str | Path) -> "FnddsIndex":
        raw = json.loads(Path(path).read_text())
        foods = [
            Food(
                code=f["code"],
                description=f["description"],
                additional=tuple(f["additional"]),
                common_names=tuple(f["common_names"]),
                wweia_category=f["wweia_category"],
                wweia_category_description=f["wweia_category_description"],
                nutrients=f["nutrients"],
            )
            for f in raw["foods"]
        ]
        return cls(release=raw["release"], foods=foods)


def _rows(path: Path):
    with path.open(newline="", encoding="utf-8") as fh:
        yield from csv.DictReader(fh)


def build_index(csv_dir: str | Path, release: str) -> FnddsIndex:
    """Join the survey CSVs into one row per FNDDS food code.

    Raises if two foods share a description: options sent to Jev must be unique strings.
    """
    d = Path(csv_dir)
    food = {r["fdc_id"]: r["description"] for r in _rows(d / "food.csv")}
    codes = {r["fdc_id"]: r for r in _rows(d / "survey_fndds_food.csv")}
    cats = {r["wweia_food_category"]: r["wweia_food_category_description"] for r in _rows(d / "wweia_food_category.csv")}

    additional: dict[str, list[str]] = defaultdict(list)
    common: dict[str, list[str]] = defaultdict(list)
    for r in _rows(d / "food_attribute.csv"):
        text = r["value"].strip()
        if not text:
            continue
        if r["food_attribute_type_id"] == "1001":
            additional[r["fdc_id"]].append(text)
        elif r["food_attribute_type_id"] == "1000":
            common[r["fdc_id"]].append(text)

    nutrients: dict[str, dict[str, float]] = defaultdict(dict)
    for r in _rows(d / "food_nutrient.csv"):
        name = MACROS.get(int(r["nutrient_id"]))
        if name and r["amount"]:
            nutrients[r["fdc_id"]][name] = float(r["amount"])

    foods = []
    for fdc_id, meta in codes.items():
        cat = meta["wweia_category_number"]
        foods.append(
            Food(
                code=meta["food_code"],
                description=food[fdc_id],
                additional=tuple(additional[fdc_id]),
                common_names=tuple(common[fdc_id]),
                wweia_category=cat,
                wweia_category_description=cats.get(cat, ""),
                nutrients=nutrients[fdc_id],
            )
        )
    foods.sort(key=lambda f: f.code)

    seen: dict[str, str] = {}
    for f in foods:
        if f.description in seen:
            raise ValueError(f"duplicate description {f.description!r}: codes {seen[f.description]} and {f.code}")
        seen[f.description] = f.code
    return FnddsIndex(release=release, foods=foods)


# Pinned FoodData_Central survey (FNDDS 2021-2023) release. Public domain (USDA).
RELEASE_TAG = "fdc-survey-2024-10-31"
RELEASE_URL = "https://fdc.nal.usda.gov/fdc-datasets/FoodData_Central_survey_food_csv_2024-10-31.zip"


def download_release(data_dir: str | Path = "data", url: str = RELEASE_URL, tag: str = RELEASE_TAG) -> Path:
    """Download the pinned USDA release, extract it under data/raw, build data/fndds_index.json."""
    import httpx

    data_dir = Path(data_dir)
    raw = data_dir / "raw" / tag
    raw.mkdir(parents=True, exist_ok=True)
    resp = httpx.get(url, follow_redirects=True, timeout=120)
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
        z.extractall(raw)
    csv_dir = next(raw.rglob("survey_fndds_food.csv")).parent
    out = data_dir / "fndds_index.json"
    build_index(csv_dir, tag).save(out)
    return out
