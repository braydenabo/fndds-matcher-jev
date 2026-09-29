# fndds-matcher-jev

Match free-text food descriptions to USDA **FNDDS** food codes, using retrieval to shortlist candidates and [TypeSafe's Jev](https://docs.typesafe.ai/introduction) to pick one. It returns a code, a calibrated confidence, ranked alternatives, or an explicit "no confident match".

The repo includes a small web app that walks through each step (normalize, retrieve, shortlist, Jev, decide) and shows tokens, cost and latency.

```
food string -> normalize -> retrieve top-K -> Jev Choice -> (verify) -> accepted | review | no_match
                            fuzzy + TF-IDF +
                            head-noun + bge-small
```





https://github.com/user-attachments/assets/007fe594-1532-4dde-ad80-9dcbe85c9685





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

**3. Get a TypeSafe API key and export it**
Sign up at [docs.typesafe.ai](https://docs.typesafe.ai/introduction/quickstart) (Jev is in early access). Then, in the terminal you will run the site from:
```bash
export TYPESAFE_API_KEY='your-key-here'
```
Keep the key in your environment. Never commit it. `.env` is git-ignored if you prefer a local file, but the app only reads the environment variable.

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
.venv/bin/jevmatcher recall data/labels.csv   # retrieval Recall@K, no key needed
.venv/bin/jevmatcher eval data/labels.csv     # full system: top-1/3, nutrient distance, calibration (needs key)
```
Recall@K is the ceiling for the whole system: Jev can only pick from the shortlist.

## How retrieval works
Four independent searches, merged by reciprocal-rank fusion (each list scores 1/(60+rank), scores add):

| Method | Catches |
|---|---|
| Fuzzy (rapidfuzz) | word overlap, typos |
| TF-IDF over 3-5 letter chunks | misspellings, word order |
| Head-noun | generic words ("egg", "carrots") landing on the general entry |
| bge-small embeddings | meaning ("sugar substitute" ~ "Swerve") |

Each was added because it fixed a class of misses; `jevmatcher/retrieve.py` has the details. Recall depends on your labels, so measure it on your own data.

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
- Thresholds in `Thresholds` (`t_high`, `margin`) are placeholders. Fit them on labeled data.
- FNDDS is versioned about every two years; the release is pinned, and a new one means re-indexing and re-evaluating.

## Development
```bash
.venv/bin/pytest
```

## License
MIT. FNDDS data is public domain (USDA).
