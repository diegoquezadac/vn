# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

VN (Vehicle Normalization) implements a method that transforms raw vehicle representations into a standardized form free from three classes of noise common across automotive data sources:

- **Inconsistent naming** — identical entities represented by different string values (e.g. "mercedes", "mercedes-benz", "Mercedes Benz")
- **Inconsistent granularity** — identical attributes expressed at different levels of detail across sources (e.g. "automatic" vs "7-speed DCT")
- **Semantic overlap** — attribute values that mix distinct vehicle characteristics (e.g. a single field encoding both body style and drivetrain)

The core hypothesis is that this standardized representation improves the performance of downstream pricing ML models. Practically, it enables researchers and engineers in the automotive domain to integrate and compare data across heterogeneous sources.

## Commands

```bash
uv sync                        # install dependencies
uv run python normalize_all.py                          # normalize all datasets
uv run python normalize_all.py --datasets autoscout24  # single dataset
uv run python normalize_all.py --test --test_n 5       # dry-run without saving
uv run python normalize_all.py --fresh_start           # wipe cache and restart
uv run jupyter notebook                                # open notebooks
```

Requires `OPENAI_API_KEY` in a `.env` file.

## Architecture

The pipeline has two modes controlled by `extract_only` on `Normalizer`:

**Extract-only** (`extract_only=True`) — used by `normalize_all.py`:
- LLM parses free-form vehicle text → structured `Vehicle` Pydantic object
- No vector store, no catalog, no matching step
- Results cached per-description in a SQLite `.db` file next to the CSV

**Full pipeline** (`extract_only=False`) — used programmatically:
1. Extract → structured vehicle fields
2. Retrieve → FAISS HNSW similarity search finds top-k candidates from catalog for `brand`, `model`, `submodel`, `trim_level`
3. Match → LLM decides if extracted value is an existing catalog entry or a new one
4. Intra-batch dedup → sequential matching of new values within a batch to avoid synonyms entering the catalog (e.g. "mercedes" and "mercedes benz" in same batch)
5. Catalog update → new canonical values written to `db/catalog.db` (SQLite) + HNSW indexes flushed to `db/`

## Key files

- `src/models.py` — `Vehicle` Pydantic model (all extractable fields) and `Resolution` (match/no-match output). The `EquipmentItem` Literal defines the controlled vocabulary for equipment.
- `src/normalizer.py` — `Normalizer` class + `serialize_row()` which converts a DataFrame row to a `"col: value, ..."` string for LLM input
- `prompts/extraction.j2` — Jinja2 extraction prompt; rendered once at init with few-shot examples baked in, then used as a LangChain `PromptTemplate` with `{x}` as the only variable
- `prompts/matching.j2` — same pattern; variables are `{x}`, `{records}`, `{attribute}`
- `samples/extraction.json` / `samples/matching.json` — few-shot examples rendered into prompts at init time
- `scripts/normalize_all.py` — batch normalization for autoscout24, craigslist, mucars; crash-safe, resumable, analytics tracking
- `scripts/normalize.py` — generic full-pipeline normalization for any CSV with `--cols_to_normalize`
- `scripts/estimate.py` — samples N descriptions and extrapolates cost + time for a full run
- `scripts/audit.py` — runs a small sample through the full pipeline and prints detailed results for debugging
- `scripts/bench_batch.py` — benchmarks batch sizes (with/without prefetch) to find optimal throughput
- `src/train.py` — downstream price regression using CatBoost/XGBoost/LightGBM/stacking on normalized vehicle features
- `vehicle_identity.ipynb` — analysis of which columns define a unique vehicle per dataset vs. listing-instance columns (price, location, etc.)

## Data

Raw CSVs in `data/`: `autoscout24.csv` (251k rows, European), `craigslist.csv` (427k rows, US miles), `mucars.csv` (102k rows, Moroccan, mileage as ranges, Fiscal Power is a tax category not engine power).

Per-dataset normalization cache: `data/<dataset>_mappings.db` (SQLite, description → JSON result).

## Extending the Vehicle model

- **New Literal field** (low cardinality): add to `Vehicle` in `src/models.py`, add examples to `samples/extraction.json`, add rule to `prompts/extraction.j2` if non-obvious.
- **New numerical field with unit**: add a value field + a `_unit` Literal field as a pair (see `engine_power`/`engine_power_unit` pattern). Do not convert units in the prompt — preserve original value and unit.
- **New high-cardinality string field** (e.g. a new catalog attribute): add to `Vehicle`, add to `self.attributes` in `Normalizer.__init__`, create a FAISS HNSW index for it in `db/`, add matching examples to `samples/matching.json`.

## Prompt engineering notes

Jinja2 templates are rendered at `Normalizer.__init__` with samples baked in. The rendered string then has `{`/`}` escaped and a placeholder replaced to become a LangChain `PromptTemplate`. Editing a prompt requires restarting the `Normalizer` — there is no hot reload.

Both chains use `llm.with_structured_output(..., include_raw=True)` so token usage is available on every call via `response["raw"].response_metadata["token_usage"]`.
