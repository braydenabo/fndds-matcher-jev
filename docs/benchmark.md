# Comparison with Lemay et al. (ASA24-to-FooDB)

Lemay et al. ([*J Nutr* 2026;156:101678](https://doi.org/10.1016/j.tjnut.2026.101678)) published a benchmark for matching food descriptions to a food database. We ran this project on the part of it that maps 1,199 USDA-style food descriptions to 9,913 FooDB entries, to see how it compares with their method: choose a short list with an embedding model, then have a language model pick one.

The benchmark's repository has no license, so we do not redistribute it. Download it into `data/` (which git ignores):
```bash
mkdir -p data/benchmarks/asa24_foodb && cd data/benchmarks/asa24_foodb
B=https://raw.githubusercontent.com/dglemay/USDA-Food-Mapping/main/benchmark_datasets/ASA24-to-FoodB/dataset
curl -O $B/groundtruth_ASA24toFooDB.txt && curl -O $B/target_desc_fooDB.txt
cd ../../..
```
```bash
.venv/bin/python scripts/eval_asa24.py data/benchmarks/asa24_foodb                      # retrieval only, no key
.venv/bin/python scripts/eval_asa24.py data/benchmarks/asa24_foodb --live --skip-retrieval-table --out results/asa24.json   # real Jev, about 20 s
```
Following the paper, inputs whose id equals the target id (1,014 of 1,198, **84.6% with no model at all**) are matched by id, and the test is the 184 text-only inputs. Targets are identified by description text, because FooDB ids are unreliable here: about 12% are blank and a labeled target's id can differ from the id the same description has in the target list.

**Retrieval** (right entry in the top K, 184 text-only foods), against the paper's GTE-large embedding top-K on the same benchmark:

| | @1 | @5 | @10 | @25 | @50 |
|---|---|---|---|---|---|
| bge-small only | 0.46 | 0.72 | 0.82 | 0.88 | 0.91 |
| Fuzzy + TF-IDF | 0.14 | 0.59 | 0.73 | 0.84 | 0.92 |
| Combined search (fuzzy, TF-IDF, head-noun, bge-small) | 0.29 | 0.70 | 0.84 | 0.90 | 0.92 |
| **GTE-large only** (`--embed-model thenlper/gte-large --dense-only`) | 0.44 | 0.79 | 0.93 | 0.94 | 0.96 |
| Combined search with GTE-large | 0.28 | 0.71 | 0.83 | 0.93 | 0.96 |
| Paper, GTE-large, modified ground truth (the version in this file) | 0.45 | 0.79 | 0.92 | 0.93 | 0.96 |
| Paper, GTE-large, original ground truth | 0.50 | 0.76 | 0.83 | 0.87 | 0.90 |

GTE-large alone reproduces the paper's numbers to within about half a point, which confirms the benchmark set and ground-truth version are the same. On this benchmark, adding the text-matching searches lowers recall at small K. The inputs here are database-style descriptions, which the embedding model handles best on its own.

**End to end with live Jev** (same 184 foods; top-1 is the best real candidate, exact match):

| | Top-1 |
|---|---|
| Combined search, no Jev | 32% |
| Jev over the combined search, K=10 / K=30 | 41.3% / 39.7-40.2% |
| **GTE-large alone, no Jev** | **44.0%** |
| Jev over GTE-large, K=10 / K=30 | 40.2% / 41.3% |
| Paper, embedding alone (Table 5, K=1) | 44.7% |
| Paper, Claude Haiku / Sonnet over the top 5, on its 170 no-id foods | 42.9% / 45.9% |

The paper counts every id-matched food as correct when it reports overall accuracy. Counted the same way, our overall accuracy is 90.6-91.0%, against the paper's 90.7%, but 84.6 points of that come from the id shortcut alone. **In short, on this benchmark neither Jev nor the paper's language models beat the embedding model's own first choice** (about 44%), and this project matches the paper rather than improving on it. Jev looks helpful over our combined search only because that search's first choice is weak. With 184 foods, chance alone moves a result by about 3.6 points, so differences of a few points do not mean much. Prompt wording (the paper's ranked criteria) and forced choice without "none of these" made no difference either.

Caveats: `groundtruth_ASA24toFooDB.txt` differs from the archived original spreadsheet in exactly 43 target entries, which matches the paper's "modified ground truth". The text-only set is not identical to the paper's (184 inputs here vs 170 plus 54 one-to-many there), and some text-only labels look loose (for example a raw cut labeled as the answer for a cooked one), so absolute numbers are uncertain. One ground-truth row with no matching target was dropped.
