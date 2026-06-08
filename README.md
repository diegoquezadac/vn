# Vehicle Normalization (VN)

Companion code and data for **"Vehicle Data Normalization with Retrieval-Augmented LLMs"**, accepted at **CLEI 2026**.

VN maps a free-text vehicle listing to a normalized, 26-attribute schema, removing three kinds of noise common across automotive data sources:

- **Naming inconsistency** — one entity under different strings (`"mercedes"`, `"mercedes-benz"`, `"Mercedes Benz"`)
- **Granularity inconsistency** — one attribute at different levels of detail (`"automatic"` vs `"7-speed DCT"`)
- **Scope inconsistency** — one field conflating distinct dimensions (a fuel-type column holding both `"petrol"` and `"phev"`)

The method achieves novelty F1 of 0.623 (AutoScout24) and 0.878 (MuCars) on 1,000 human-annotated listings, and reduces downstream price-prediction MAPE by 2.7–5.3% on MuCars across CatBoost, XGBoost, and LightGBM. See `manuscript/` for the full paper.

## Method

Normalization is a mapping `φ: free-text → X̃` over the product space of all attribute domains, run as a three-step pipeline (see `manuscript/`, Fig. 1):

1. **IE** — a single LLM call extracts all attribute values at once into a structured `Vehicle` object.
2. **IR** — each open-domain value (brand, model) is embedded and its top-`k` canonical candidates are retrieved from a per-attribute **knowledge catalog** via FAISS HNSW.
3. **ER** — an LLM resolver compares the full extracted record against each candidate (in similarity order) and stops at the first match. If none matches, the value is a **novel entity** and is appended to the catalog.

Closed-domain attributes (everything except brand and model) are emitted directly by IE, since the prompt constrains them to their fixed value sets.

| Component | Model |
| --- | --- |
| Extraction (IE) | OpenAI `gpt-4.1-nano` |
| Resolution (ER) | OpenAI `gpt-4.1-mini` |
| Embeddings (IR) | OpenAI `text-embedding-3-small` |

## Installation

Requires Python ≥ 3.10 and an OpenAI API key. Using [uv](https://github.com/astral-sh/uv):

```bash
uv sync
echo "OPENAI_API_KEY=sk-..." > .env
```

## Usage

`Normalizer` exposes the three pipeline configurations through `match_mode`:

| `match_mode` | Pipeline | Catalog + FAISS | Per-record LLM calls |
| --- | --- | --- | --- |
| `"off"`       | IE                         | not used | 1 extraction |
| `"threshold"` | IE + IR (top-1 if cosine ≥ `match_threshold`) | required | 1 extraction |
| `"llm"`       | IE + IR + ER (record-level resolver) | required | extraction + resolver calls for novel values |

```python
import asyncio
from src.normalizer import Normalizer


async def main():
    # Extraction only — no catalog needed
    ie = Normalizer(match_mode="off")
    vehicle = await ie.extract("volvo v60 2.4 d6 awd wagon automatic 2016")
    print(vehicle)                       # {'brand': 'volvo', 'model': 'v60', ...}

    # Full pipeline (IE + IR + ER) — needs a seeded catalog in ./db
    # (build it once with: uv run python scripts/catalog.py --out_dir db)
    vn = Normalizer(match_mode="llm", persist_directory="db")
    vehicles, metrics = await vn(["seat arona 2021 1.6 tdi blanco"])
    print(vehicles[0])                   # canonicalized brand/model
    print(f"cost: ${vn.get_cost():.4f}")


asyncio.run(main())
```

Calling the normalizer on a batch returns `(results, metrics)`. Pass `verbose=True` to get per-record `{query, vehicle, extraction, operations}` instead of just the vehicle dicts. In `"llm"` and `"threshold"` modes, novel canonical values are persisted to `{persist_directory}/catalog.db` and the HNSW indexes are flushed to disk, so the catalog grows across runs.

## Reproducibility

This repository ships everything needed to reproduce the **headline evaluation and ablations** without any external download:

- `data/dvm.csv` — DVM-CAR, used to seed the brand/model catalog
- `data/{autoscout24,mucars}_sample.csv` — the 500-listing evaluation samples
- `data/{autoscout24,mucars}_ground_truth.csv` — majority-vote human annotations

**1. Seed the knowledge catalog** (builds `experiments/db/{catalog.db, hnsw_brand.index, hnsw_model.index}` from DVM-CAR):

```bash
uv run python scripts/catalog.py --out_dir experiments/db
```

**2. Main results** — normalization accuracy, novelty detection, and compression for IE / IE+IR / IE+IR+ER (paper Table II):

```bash
uv run python scripts/evaluate_normalizer.py        # writes data/{dataset}_normalizer_report.json
uv run python scripts/summarize_reports.py          # optional: print consolidated tables
```

**3. Ablations** — figures for the cosine threshold `τ` (IE+IR) and retrieval depth `k` (IE+IR+ER):

```bash
uv run python scripts/ablation_threshold.py         # τ sweep → data/ablation_threshold.json + figure
uv run python scripts/ablation_k.py                 # k sweep → data/ablation_k.json + figure
```

The remaining experiments require the **raw full datasets** (not redistributed here — see below):

- **Feasibility / scalability** (paper Fig. 5): `uv run python scripts/feasibility.py`
- **Full-corpus normalization**: `uv run python scripts/normalize.py --datasets mucars` → `data/mucars_mappings.db`
- **Downstream price prediction** (paper Table V): run `notebooks/mucars_eval.ipynb` (and `notebooks/autoscout24_eval.ipynb`), which consume the mappings DB and train the regressors in `src/train.py`.

The evaluation data itself was produced by `scripts/sampling.py` (proportional stratified sampling on brand) followed by independent labeling from three annotators and `scripts/evaluate_annotation.py` (Fleiss' κ + majority-vote ground truth). The per-annotator label files are not redistributed; the resulting ground truth is shipped, so step 2 above runs as-is.

### Datasets

| Dataset | Country | Listings | Role | Source |
| --- | --- | --- | --- | --- |
| DVM-CAR | UK | 268k | Catalog seed (shipped: `data/dvm.csv`) | [Project page](https://deepvisualmarketing.github.io) |
| AutoScout24 | Germany | 251k | Evaluation + feasibility | [Kaggle](https://www.kaggle.com/datasets/wspirat/germany-used-cars-dataset-2023) |
| MuCars | Morocco | 102k | Evaluation + downstream | [Mendeley Data](https://data.mendeley.com/datasets/vjrbcb2rrt/2) |

To run feasibility, full normalization, or the downstream notebooks, download the raw CSVs from their original sources and place them at `data/autoscout24.csv` and `data/mucars.csv`.

## Extensibility

The schema is the `Vehicle` Pydantic model in `src/models.py`. To add an attribute:

- **Closed-domain categorical** (low cardinality): add a `Literal` field to `Vehicle`, add examples to `samples/extraction.json`, and add a rule to `prompts/extraction.j2` if the mapping is non-obvious. Closed-domain values flow straight through IE — no catalog needed.
- **Numeric with a unit**: add a value field plus a paired `_unit` `Literal` field (follow the `engine_power` / `engine_power_unit` pattern). Preserve the original value and unit; do not convert in the prompt.
- **Open-domain (catalog) attribute** (high cardinality, like a new `submodel`): add a `str` field to `Vehicle`, register it in `Normalizer` (the `self.attributes`, `get_empty_catalog`, and the `["brand", "model"]` HNSW lists in `src/normalizer.py`), add matching examples to `samples/matching.json`, and rebuild the catalog with `scripts/catalog.py`.

Prompts are Jinja2 templates rendered once at `Normalizer.__init__` with the few-shot samples baked in, so editing a prompt requires re-instantiating the `Normalizer`.

## Layout

```
vn/
├── src/
│   ├── normalizer.py     # Normalizer (match_mode: off | threshold | llm) + serialize_row
│   ├── models.py         # Vehicle (26-attribute schema) + Resolution
│   └── train.py          # downstream price regressors (CatBoost / XGBoost / LightGBM)
├── prompts/              # extraction.j2, matching.j2 — few-shot baked in at init
├── samples/              # few-shot examples rendered into the prompts
├── scripts/              # reproduction pipeline (see "Reproducing the paper")
├── notebooks/            # downstream price-prediction evaluation
├── data/                 # eval samples, ground truth, DVM-CAR catalog seed, EDA figures
├── db/                   # generated: catalog.db + hnsw_{brand,model}.index
└── manuscript/           # CLEI 2026 paper (LaTeX + figures)
```

## License

MIT — see `LICENSE`.
