# Vehicle Normalization (VN)

A Python framework for normalizing unstructured vehicle information into standardized, canonical formats using LLM-powered extraction and semantic matching.

> **Note:** This version is tailored for a specific research context. The framework is designed to be extensible for different use cases.

## Overview

VN implements a three-stage normalization pipeline:

1. **Extraction** — LLM parses free-form vehicle descriptions into structured data
2. **Retrieval** — Vector similarity search finds candidate matches from a catalog
3. **Matching** — LLM determines if extracted values match existing catalog entries or are new

## Project Structure

```
vn/
├── main.py                 # Entry point
├── src/
│   ├── models.py           # Vehicle Pydantic model
│   └── normalizer.py       # Core normalization engine
├── prompts/
│   ├── extraction.j2       # Extraction prompt template
│   └── matching.j2         # Matching prompt template
├── samples/
│   ├── extraction.json     # Few-shot examples for extraction
│   └── matching.json       # Few-shot examples for matching
├── data/
│   └── catalog.json        # Persisted canonical values
└── db/                     # Chroma vector database
```

## Installation

Requires Python >= 3.10. Using [uv](https://github.com/astral-sh/uv):

```bash
uv sync
```

Or with pip:

```bash
pip install -e .
```

## Configuration

Create a `.env` file with your OpenAI API key:

```
OPENAI_API_KEY=your-api-key
```

## Usage

```python
from src.normalizer import Normalizer

# Initialize the normalizer
vn = Normalizer(
    model="gpt-4.1-nano",           # or "gpt-5-nano", "gpt-5-mini"
    embedding_model="text-embedding-3-small",
    k=3                              # candidates to retrieve
)

# Extract structured data from a description
result = await vn.extract("volvo v60 2.4 d6 awd wagon automatic")

# Full pipeline: extract + retrieve + match
logs = await vn(["seat arona 2021 ibiza 1.6 bencina blanco"])

# Check API costs
print(vn.get_cost())
```

## Extending the Vehicle Model

The `Vehicle` class in [src/models.py](src/models.py) defines the attributes to extract. When adding new attributes, the approach depends on the attribute type:

### 1. Numerical attributes

Straightforward to add. Only requires updating:
- [src/models.py](src/models.py) — add the field to the `Vehicle` class
- [samples/extraction.json](samples/extraction.json) — add few-shot examples

```python
class Vehicle(BaseModel):
    # ... existing fields ...
    mileage: Optional[float] = None
```

### 2. Categorical with low cardinality

Use a `Literal` type to constrain the possible values. Only requires updating:
- [src/models.py](src/models.py) — add the field with `Literal` type
- [samples/extraction.json](samples/extraction.json) — add few-shot examples

```python
class Vehicle(BaseModel):
    # ... existing fields ...
    fuel_type: Optional[Literal["petrol", "diesel", "electric", "hybrid"]] = None
```

### 3. Categorical with high cardinality

More complex. These attributes need to go through the RAG pipeline for matching against a catalog. Requires updating:
- [src/models.py](src/models.py) — add the field as `str`
- [src/normalizer.py](src/normalizer.py) — update retrieval and matching logic
- [samples/extraction.json](samples/extraction.json) — add few-shot examples
- [samples/matching.json](samples/matching.json) — add few-shot examples

```python
class Vehicle(BaseModel):
    # ... existing fields ...
    new_attribute: Optional[str] = None
```

## Adding LLM Providers

Currently OpenAI is supported. The code is designed to be extensible for additional providers. See the `provider` parameter in the `Normalizer` class and the LangChain integrations used throughout.

## Dependencies

- `langchain` / `langchain-openai` — LLM orchestration
- `langchain-chroma` — Vector database
- `pydantic` — Data validation
- `jinja2` — Prompt templating
- `aiofiles` — Async file operations

## License

MIT
