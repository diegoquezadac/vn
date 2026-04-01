# Vehicle Normalization (VN)

A Python framework for normalizing raw vehicle representations into a standardized, canonical form using LLM-powered extraction and semantic matching.

## Motivation

Automotive data sourced from different marketplaces suffers from three problems:

- **Inconsistent naming** — identical entities represented by different string values across sources (e.g. `"mercedes"`, `"mercedes-benz"`, `"Mercedes Benz"`)
- **Inconsistent granularity** — identical attributes expressed at varying levels of detail (e.g. `"automatic"` vs `"7-speed DCT"`)
- **Semantic overlap** — attribute values that mix distinct vehicle characteristics into a single field

The hypothesis behind VN is that eliminating these inconsistencies produces a standardized representation that improves the performance of downstream ML models. Additionaly, it enables data integration across heterogeneous automotive sources.

## Pipeline

VN implements two modes controlled by `extract_only` on `Normalizer`:

**Extract-only** (`extract_only=True`) — fast, no catalog or vector store:
1. LLM parses a free-form vehicle description → structured `Vehicle` object (30+ fields)

**Full pipeline** (`extract_only=False`):
1. **Extract** — LLM parses description into structured fields
2. **Retrieve** — FAISS HNSW similarity search finds top-k candidates from catalog for `brand`, `model`, `submodel`, `trim_level`
3. **Match** — LLM decides if extracted value matches an existing catalog entry or is new
4. **Intra-batch dedup** — sequential matching within a batch prevents synonyms from entering the catalog as separate entries
5. **Catalog update** — new canonical values written to `db/catalog.db` (SQLite) + FAISS indexes flushed to `db/`

## Project Structure

```
vn/
├── src/
│   ├── models.py             # Vehicle Pydantic model + Resolution
│   └── normalizer.py         # Normalizer class + serialize_row()
├── prompts/
│   ├── extraction.j2         # Jinja2 extraction prompt (few-shot baked in at init)
│   └── matching.j2           # Jinja2 matching prompt
├── samples/
│   ├── extraction.json       # Few-shot examples for extraction
│   └── matching.json         # Few-shot examples for matching
├── scripts/
│   ├── normalize.py          # Full-pipeline batch normalization for any CSV
│   ├── normalize_all.py      # Batch normalization for autoscout24, craigslist, mucars
│   ├── estimate.py           # Cost + time estimator (samples N descriptions)
│   ├── audit.py              # Inspect pipeline output on a small sample
│   └── bench_batch.py        # Benchmark batch sizes for throughput
├── data/
│   ├── autoscout24.csv       # 251k rows, European marketplace
│   ├── craigslist.csv        # 427k rows, US marketplace (miles)
│   └── mucars.csv            # 102k rows, Moroccan marketplace
└── db/
    ├── catalog.db            # SQLite: canonical values + HNSW value index
    └── hnsw_*.index          # FAISS HNSW binary indexes (one per catalog attribute)
```

## Installation

Requires Python >= 3.10 and an OpenAI API key. Using [uv](https://github.com/astral-sh/uv):

```bash
uv sync
```

Create a `.env` file:

```
OPENAI_API_KEY=your-api-key
```

## Usage

```python
from src.normalizer import Normalizer

# Extract-only (no catalog, no FAISS)
normalizer = Normalizer(extract_only=True)
result = await normalizer.extract("volvo v60 2.4 d6 awd wagon automatic")

# Full pipeline: extract + retrieve + match + catalog update
normalizer = Normalizer(extract_only=False)
vehicles, metrics = await normalizer(["seat arona 2021 ibiza 1.6 bencina blanco"])

# Verbose mode: includes raw extraction and catalog operations per row
vehicles, metrics = await normalizer([...], verbose=True)

# Cost tracking
print(f"${normalizer.get_cost():.4f}")
```

### Recommended workflow

Optimal batch size and prefetch depend on the LLM model, your API tier (TPM/RPM limits), and network latency. The recommended sequence before running a full normalization is:

**Step 1 — Find the optimal batch size and prefetch setting:**
```bash
uv run python scripts/bench_batch.py
```
This tests batch sizes `[20, 30, 40, 50]` with and without `prefetch=1`, measuring DPM/RPM/TPM for each combination. Pick the configuration that maximizes DPM without hitting your TPM ceiling. A 60-second cooldown is applied between configurations for a fair comparison.

**Step 2 — Estimate cost and time for a full run:**
```bash
uv run python scripts/estimate.py --batch_size <optimal> --n 250
```
Samples 250 descriptions per dataset, runs them through the full pipeline, and extrapolates cost and time to the full dataset. Defaults: `batch_size=50`, `prefetch=1`.

**Step 3 — Run normalization with the optimal parameters:**
```bash
uv run python scripts/normalize_all.py --batch_size <optimal>
```
Crash-safe and resumable — re-run the same command to continue from where it left off.

### Other commands

```bash
# Single dataset
uv run python scripts/normalize_all.py --datasets autoscout24

# Test a few samples without saving
uv run python scripts/normalize_all.py --test --test_n 5

# Wipe everything and restart
uv run python scripts/normalize_all.py --fresh_start

# Audit pipeline output on a small sample
uv run python scripts/audit.py --datasets autoscout24 --n 5
```

## Extending the Vehicle Model

- **New low-cardinality field**: add `Literal` field to `Vehicle` in `src/models.py`, add examples to `samples/extraction.json`, add a rule to `prompts/extraction.j2` if non-obvious.
- **New numerical field with unit**: add a value field + `_unit` Literal field as a pair (see `engine_power` / `engine_power_unit` pattern). Preserve original units — do not convert in the prompt.
- **New high-cardinality catalog attribute**: add `str` field to `Vehicle`, add to `self.attributes` in `Normalizer.__init__`, add matching examples to `samples/matching.json`.

## Dependencies

- `langchain` / `langchain-openai` — LLM orchestration and structured output
- `faiss-cpu` — HNSW vector similarity search
- `pydantic` — Vehicle schema and validation
- `jinja2` — Prompt templating
- `openai` — Embeddings API

## License

MIT
