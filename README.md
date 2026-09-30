# fndds-matcher-jev

[![CI](https://github.com/braydenabo/fndds-matcher-jev/actions/workflows/ci.yml/badge.svg)](https://github.com/braydenabo/fndds-matcher-jev/actions/workflows/ci.yml)
![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

This project turns a free-text food description, such as "peanut butter crackers", into a food code from the USDA Food and Nutrient Database for Dietary Studies (FNDDS). A search step picks 30 likely foods out of the 5,432 in the database. Then [Jev](https://docs.typesafe.ai/introduction), a hosted model that answers multiple-choice questions and gives a probability for each option, picks one of them or says that none fit.

**Key results** on 279 hand-labeled foods:
- The right code for 59.1% of foods, up from 44.1% with the search step alone.
- A code with almost the same calories and macronutrients for 78.1%.
- Accepting only answers with a confidence of at least 0.96 handled 40.9% of held-out foods at 92.9% precision, with the rest left for a person to review.
- All 279 foods cost about one cent.

https://github.com/user-attachments/assets/007fe594-1532-4dde-ad80-9dcbe85c9685

The full evaluation and error analysis are in the [write-up](paper/main.pdf).

## Contents
[How it works](#how-it-works) · [Results](#results) · [Limitations](#limitations) · [Quickstart](#quickstart) · [Command line](#command-line) · [Evaluating on your own labels](#evaluating-on-your-own-labels) · [Comparison with Lemay et al.](#comparison-with-lemay-et-al) · [Project layout](#project-layout)

## How it works
```
food description -> clean up -> search for 30 candidates -> Jev picks one -> accept, review, or no match
                                fuzzy + TF-IDF +
                                head-noun + bge-small
```
The description is cleaned up (lowercase, abbreviations expanded, common typos fixed). Four searches each rank the foods in the database, and the four rankings are combined by adding 1/(60 + rank) for each food. The top 30 go to Jev as one multiple-choice question, together with a "none of these" option. Jev returns a probability for every option, and a food is accepted when the top option is not "none of these" and its probability is at least the threshold.

| Search | What it catches |
|---|---|
| Fuzzy string match | shared words and typos |
| TF-IDF over 3-5 letter pieces | misspellings and word order |
| Head-noun match | generic words such as "egg" or "carrots" landing on the general entry |
| bge-small embeddings | meaning, such as "sugar substitute" matching "Swerve" |

## Results
The labeled foods are short names from meal photos, each with a true FNDDS code. The set is private and is not in this repository. We ran Jev live, with 30 candidates per food.

| | Result |
|---|---|
| Right code is in the list of 30 candidates | 89.6% |
| Top choice is the right code, search step alone | 44.1% |
| **Top choice is the right code, with Jev** | **59.1%** (+15 points) |
| Top choice is nutritionally equivalent* | 78.1% |
| Same food group / same 3-digit subgroup | 96.1% / 90.0% |
| Cost for all 279 foods | about $0.01 (about $0.035 per 1,000) |

\*The exact code, or one whose calories are within max(15, 10%) and whose protein, carbohydrate and fat are each within max(2 g, 15%) per 100 g. We chose these tolerances ourselves, so read this as a rough measure of how costly the misses are.

**When to trust the answers.** We split the foods into two halves by a hash of the name, fitted thresholds on one half, and scored them on the other.
- Jev's confidence is higher than its accuracy (expected calibration error 0.22). Even at a confidence of 95% or more, only about 85% of accepted answers match the label code exactly.
- For nutritionally equivalent answers it is still useful. Accepting only a confidence of at least 0.96 took 40.9% of the held-out foods at 92.9% precision (91.9% on the half used to pick the threshold). The rest go to a person.

Most mistakes come from inputs that leave out how the food was prepared, such as "carrots" (raw or cooked?), and from foods with many near-identical codes.

## Limitations
- The labeled set is small, with 279 foods and 221 different names, and has no second annotator. The 92.9% rests on 56 accepted foods, so its 95% interval is roughly 83-97%.
- We chose the search methods by scoring them on these same labels, so the search results are somewhat optimistic.
- Some labels are wrong or could have several right answers. Eleven pointed to codes missing from this release, and we replaced them by hand.
- We report one live run with one vendor's early-access model, whose behavior may change. A second run changed the top-1 figure by about a point.
- We tested single foods only, not portions, mixed dishes or whole meals.
- Text you type is sent to TypeSafe in live mode, so do not enter anything sensitive.

## Quickstart
Requires Python 3.11 or newer. Run every step from the repository root.

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
This downloads the FNDDS 2021-2023 release (FoodData Central release 2024-10-31) into `data/raw/` and builds `data/fndds_index.json` with its 5,432 foods. Nothing under `data/` is committed.

**3. Get a TypeSafe API key and save it in `.env`**
Sign up at [docs.typesafe.ai](https://docs.typesafe.ai/introduction/quickstart) (Jev is in early access). Then create a `.env` file in the repository root that holds your key:
```bash
cp .env.example .env
```
Open `.env` and replace `your-key-here` with your key. Git ignores `.env`, so it is never committed. The site and the command line read it automatically. A `TYPESAFE_API_KEY` set in your shell (`export TYPESAFE_API_KEY=...`) also works and takes priority over `.env`.

**4. Run the site**
```bash
.venv/bin/uvicorn jevmatcher.web:create_app --factory --port 8000
```
Open http://localhost:8000, type a food, and step through the matching with Next and Back, or the arrow keys. Each lookup sends one request to Jev (two if "Verify top-1" is on). At the vendor's price of $0.042 per million input tokens, a lookup costs a fraction of a cent.

The first run downloads a small embedding model (about 130 MB) and stores the food vectors in `data/cache/`.

### No key? Use the simulator
```bash
JEV_MOCK=1 .venv/bin/uvicorn jevmatcher.web:create_app --factory --port 8000
```
The simulator scores options by string similarity and estimates tokens and time. It lets you try the site and run tests without spending tokens. It says nothing about how accurate Jev is.

## Command line
```bash
.venv/bin/jevmatcher match "peanut butter crackers" --context "52 g"
.venv/bin/jevmatcher match "Lance PB Crakers" --shuffles 3 --verify
```

## Evaluating on your own labels
Create `data/labels.csv` (see `data/labels.example.csv`) with the columns `food, code [, context, tier]`. Codes are 8-digit FNDDS codes from the release you downloaded. Rows with a non-numeric code are skipped.

```bash
.venv/bin/jevmatcher recall data/labels.csv                          # is the right code in the list? no key needed
.venv/bin/jevmatcher eval data/labels.csv --out results/run1.json    # the full system (needs a key)
.venv/bin/jevmatcher fit-thresholds results/run1.json --target 0.9   # choose the accept threshold
```
The first command shows how often the right code is among the candidates, which is the upper limit for the whole system because Jev can only pick from that list. `eval` reports how often the top choice is right (and top three), next to the search-only baseline, the nutritional measure above, how well the confidence matches accuracy, and how many foods are accepted at each confidence level, overall and by difficulty tier. With `--out` it saves every food's answer and probabilities, so you can read the confident mistakes. Add `--mock` to try it without a key; the numbers then mean nothing.

The foods are split into a fit half and a test half by a hash of the food name, so the same food never lands on both sides. `fit-thresholds` picks the confidence threshold that accepts the most foods while still meeting the target precision on the fit half, then reports both halves. Report the test half.

The accept threshold defaults to 0.96, the value that worked on our labels. Fit your own on yours.

## Comparison with Lemay et al.
The search-then-choose design follows Lemay et al. ([*J Nutr* 2026;156:101678](https://doi.org/10.1016/j.tjnut.2026.101678)), who found that an embedding model followed by a language model that picks one candidate, or answers "No Match", worked best for matching dietary data to food databases. Their tool is [FoodMapper](https://foodmapper.app/).

We differ in three ways. We combine four searches instead of using embeddings alone. The choosing step uses a model that gives a probability for each option, so we can measure how well an accept threshold works. And our target is FNDDS, with short, noisy inputs.

We also ran this project on one of their benchmarks. The embedding model they used reproduces their search results almost exactly, and on that benchmark neither Jev nor their language models beat the embedding model's own first choice. The details are in [docs/benchmark.md](docs/benchmark.md).

## Project layout
```
jevmatcher/   fndds.py (data), normalize.py, retrieve.py (searches), jev.py (Jev client), matcher.py,
              evaluate.py, targets.py (other target lists), mockjev.py, web.py (the site), cli.py
web/          index.html (the site)
scripts/      eval_asa24.py (runs the benchmark in docs/benchmark.md)
docs/         benchmark.md
paper/        the write-up (main.pdf)
tests/        offline tests; no key needed
```
`jev.py` follows Jev's public API reference: `POST https://api.typesafe.ai/v1/systemone`, with questions written as `{type, instructions, criteria}`. Check it against the live API before relying on timing or confidence figures.

## Development
```bash
.venv/bin/pytest
```

## License
MIT. The FNDDS data is in the public domain (USDA).
