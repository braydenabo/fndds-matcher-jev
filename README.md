# fndds-matcher-jev

[![CI](https://github.com/braydenabo/fndds-matcher-jev/actions/workflows/ci.yml/badge.svg)](https://github.com/braydenabo/fndds-matcher-jev/actions/workflows/ci.yml)

Match free-text food descriptions to USDA **FNDDS** food codes, using retrieval to shortlist candidates and [TypeSafe's Jev](https://docs.typesafe.ai/introduction) to pick one. It returns a code, a confidence score, ranked alternatives, or an explicit "no confident match".

The repo includes a small web app that walks through each step (normalize, retrieve, shortlist, Jev, decide) and shows tokens, cost and latency.

```
food string -> normalize -> retrieve top-K -> Jev Choice -> (verify) -> accepted | review | no_match
                            fuzzy + TF-IDF +
                            head-noun + bge-small
```

https://github.com/user-attachments/assets/007fe594-1532-4dde-ad80-9dcbe85c9685

## Results

Evaluated on 279 hand-labeled foods (short names from meal photos, each with a true FNDDS code; the set is private and not in this repo). 

| | Result |
|---|---|
| Retrieval shortlist contains the right code (Recall@30) | 89.6% |
| Top-1 exact code, retrieval alone | 44.1% |
| **Top-1 exact code, with Jev** | **59.1%** (+15 points) |
| Top-1 **nutritionally equivalent** code* | 78.1% |
| Same food group / same 3-digit subgroup | 96.1% / 90.0% |
| Cost for all 279 foods | about $0.01 (about $0.035 per 1,000) |

\*Exact code, or kcal within max(15, 10%) and protein/carbs/fat each within max(2 g, 15%) per 100 g. The tolerances are my own choice, so read this as a rough guide to how costly the misses are.

**Knowing when to trust it.** The foods were split into two halves by a hash of the name, thresholds were fitted on one half and scored on the other.

- Jev's stated confidence is overconfident (expected calibration error 0.22). Even at 95%+ confidence, only about 85% of accepted answers match the label code exactly.
- For nutritionally equivalent answers it is usable: accepting only confidence of at least 0.96 auto-accepts about 41% of foods at 92.9% precision on the held-out half (91.9% on the fit half). Everything else goes to review.

**Caveats.**
- Small sample: the held-out accepted set is 56 foods, so the 92.9% has a 95% interval of roughly 83-97% (Wilson).
- I chose the retrieval methods by scoring on these same labels, so the retrieval numbers are optimistic.

Reproduce on your own labels: see [Evaluating on your own labels](#evaluating-on-your-own-labels).

## Quickstart

Requires Python 3.11+. Every step runs from the repo root.

**1. Clone and install**
```bash
git clone https://github.com/braydenabo/fndds-matcher-jev.git && cd fndds-matcher-jev
python3 -m venv .venv
.venv/bin/pip install -e '.[dev]'
```

**2. Download the FNDDS data** (public USDA data, about 3 MB)
```bash
.venv/bin/jevmatcher download-data
```
This fetches the pinned FoodData Central survey release (FNDDS 2021-2023, release 2024-10-31) into `data/raw/` and builds `data/fndds_index.json` (5,432 foods). Nothing under `data/` is committed.

**3. Get a TypeSafe API key and save it in `.env`**
Sign up at [docs.typesafe.ai](https://docs.typesafe.ai/introduction/quickstart) (Jev is in early access). Then create a `.env` file in the repo root containing your key:
```bash
cp .env.example .env
```
Open `.env` and replace `your-key-here` with your key. `.env` is git-ignored, so it is never committed. The app and CLI load it automatically from the repo root or the current directory. A `TYPESAFE_API_KEY` set in your shell (`export TYPESAFE_API_KEY=...`) also works and takes priority over `.env`.

**4. Run the site**
```bash
.venv/bin/uvicorn jevmatcher.web:create_app --factory --port 8000
```
Open http://localhost:8000, type a food, and step through the pipeline with Next / Back (or the arrow keys). Each lookup sends one request to Jev, or two with "Verify top-1" on. At the vendor-quoted $0.042 per 1M input tokens a lookup is a fraction of a cent.

The first run downloads a small embedding model (about 130 MB) and caches the food vectors in `data/cache/`.

### No key? Use the simulator
```bash
JEV_MOCK=1 .venv/bin/uvicorn jevmatcher.web:create_app --factory --port 8000
```
`mockjev.py` scores options by string similarity and estimates tokens and latency. It is for trying the UI and testing without spending tokens. It says nothing about how accurate Jev is.

## Command line
```bash
.venv/bin/jevmatcher match "peanut butter crackers" --context "52 g"
.venv/bin/jevmatcher match "Lance PB Crakers" --shuffles 3 --verify
```

## Evaluating on your own labels
Create `data/labels.csv` (see `data/labels.example.csv`) with columns `food, code [, context, tier]`. Codes are 8-digit FNDDS codes from the release you downloaded. Rows with a non-numeric code are skipped.

```bash
.venv/bin/jevmatcher recall data/labels.csv                                   # retrieval Recall@K, no key needed
.venv/bin/jevmatcher eval data/labels.csv --out results/run1.json             # full system (needs key)
.venv/bin/jevmatcher fit-thresholds results/run1.json --target 0.9            # pick accept thresholds
```
`eval` reports exact-code top-1/top-3 next to the retrieval-only baseline, a "nutrient-close" top-1 (a wrong code with near-identical per-100 g macros counts as a cheap error), calibration, and coverage vs precision, overall and per easy/medium/hard tier. With `--out` it saves every food's prediction, probabilities and status, so you can read the confidently-wrong cases. Add `--mock` to try the flow without a key (numbers are meaningless).

Foods are split into a **fit** and a **test** half by a hash of the food string, so the same food never lands on both sides. `fit-thresholds` chooses the accept thresholds with the most coverage that still meet the target precision on the fit half, then reports both halves. Report the test half.

Recall@K is the ceiling for the whole system: Jev can only pick from the shortlist.

## How retrieval works
Four independent searches, merged by reciprocal-rank fusion (each list scores 1/(60+rank), scores add):

| Method | Catches |
|---|---|
| Fuzzy (rapidfuzz) | word overlap, typos |
| TF-IDF over 3-5 letter chunks | misspellings, word order |
| Head-noun | generic words ("egg", "carrots") landing on the general entry |
| bge-small embeddings | meaning ("sugar substitute" ~ "Swerve") |


## Benchmark comparison (ASA24-to-FooDB)
The benchmark from Lemay et al. maps 1,199 USDA-style food descriptions to 9,913 FooDB entries. Its repo has no license, so it is not redistributed here; download it locally (git-ignored):
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

| | @5 | @10 | @25 | @50 |
|---|---|---|---|---|
| bge-small only | 0.72 | 0.82 | 0.88 | 0.91 |
| Fuzzy + TF-IDF | 0.59 | 0.73 | 0.84 | 0.92 |
| This repo's hybrid | 0.70 | 0.84 | 0.90 | 0.92 |
| Paper, GTE-large (original / modified ground truth) | 0.76 / 0.85 | 0.83 / 0.95 | 0.87 / 0.96 | 0.90 / 0.98 |

**End to end with live Jev** (same 184 foods; top-1 is the best real candidate, exact match):

| | Top-1 |
|---|---|
| Retrieval only (no Jev) | 32% |
| Jev, K=5 / K=10 / K=30 | 38.6% / 41.3% / 39.7-40.2% (three runs at K=30) |
| Paper, Claude Haiku / Sonnet over the top 5, on its 170 no-id foods | 42.9% / 45.9% |

Paper-style overall accuracy (ids counted correct plus text-only exact) is 90.6-91.0% here against the paper's 90.7%, but 84.6 points of that come from the id shortcut. In short: comparable to the paper, not better. Jev adds about 7-9 points over retrieval alone, and top-1 barely moves with K while candidate recall rises from 73% to 90%, so the choice step, not retrieval, is the limit on this benchmark.

Caveats: the benchmark file differs from the paper's (184 text-only inputs here vs 170 plus 54 one-to-many there), and some text-only labels look loose (for example a raw cut labeled as the answer for a cooked one), so absolute numbers are uncertain. One ground-truth row with no matching target was dropped.

## Related work
The retrieve-then-choose design follows Lemay et al., who found that selecting top candidates with semantic embeddings and then having an LLM pick one of them, or answer "No Match", worked best for mapping dietary data to food databases ([*J Nutr* 2026;156:101678](https://doi.org/10.1016/j.tjnut.2026.101678); their tool is [FoodMapper](https://foodmapper.app/)).

This project differs in three ways: retrieval fuses four methods instead of embeddings alone, the chooser is a model that returns a probability per option (Jev) so that selective acceptance can be measured, and the target is FNDDS with short, noisy inputs. It has **not** been run on their benchmark datasets, so no claim is made that it outperforms their method; the numbers above are not comparable to theirs.

## Layout
```
jevmatcher/   fndds.py (data), normalize.py, retrieve.py, jev.py (API client), matcher.py (pipeline),
              evaluate.py, mockjev.py, web.py (FastAPI app), cli.py
web/          index.html (the site)
tests/        pytest suite (offline; no key needed)
```

## Notes
- `jev.py` follows Jev's public API reference: `POST https://api.typesafe.ai/v1/systemone`, Choice questions are `{type, instructions, criteria}`. Confirm against the live API before trusting timings or calibration.
- Text you type is sent to TypeSafe in live mode. Do not enter anything sensitive.
- The default accept thresholds in `Thresholds` (`t_high=0.6`, `margin=0.2`) are deliberately loose placeholders. On my labels a much higher confidence (about 0.96) was needed for ~93% nutritional precision; fit your own with `jevmatcher fit-thresholds`.


## Development
```bash
.venv/bin/pytest
```

## License
MIT. FNDDS data is public domain (USDA).