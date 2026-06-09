"""
catalog.py  -  build the knowledge catalog K for brand and model from DVM-CAR.

The catalog instantiates the open-domain attributes K_brand and K_model used
throughout the experiments. For each attribute we:

  1. Extract unique canonical values from data/dvm.csv.
  2. Apply minimal normalization (lowercase + strip).
  3. Embed with OpenAI `text-embedding-3-small` (L2-normalized -> cosine sim).
  4. Build a FAISS HNSW index (M=32, ef_search=64) as reported in the paper.

Output layout (matches the format expected by `src.normalizer.Normalizer`,
so experiments can instantiate `Normalizer(persist_directory=<out_dir>)`):

  <out_dir>/catalog.db          - SQLite: catalog + hnsw_values + mappings tables
  <out_dir>/hnsw_brand.index    - FAISS HNSW index for brand
  <out_dir>/hnsw_model.index    - FAISS HNSW index for model

Usage:
    uv run python scripts/catalog.py
    uv run python scripts/catalog.py --out_dir experiments/db
"""

import argparse
import asyncio
import os
import sqlite3

import faiss
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()

EMBED_DIM = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
}


def preprocess(s) -> str | None:
    """Lowercase and strip. Return None for empty/null values."""
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return None
    v = str(s).strip().lower()
    return v or None


async def embed_values(
    client: AsyncOpenAI,
    model: str,
    attribute: str,
    values: list[str],
    batch_size: int = 1000,
) -> np.ndarray:
    """Embed `{attribute} {value}` strings in chunks, L2-normalized, preserving order."""
    vecs: list[list[float]] = [None] * len(values)  # type: ignore[list-item]
    for i in range(0, len(values), batch_size):
        chunk_idx = list(range(i, min(i + batch_size, len(values))))
        chunk = [f"{attribute} {values[j]}" for j in chunk_idx]
        resp = await client.embeddings.create(input=chunk, model=model)
        for e in resp.data:
            vecs[chunk_idx[e.index]] = e.embedding
        print(f"  [{attribute}] embedded {min(i + batch_size, len(values))}/{len(values)}")
    arr = np.array(vecs, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.maximum(norms, 1e-9)


def build_index(vecs: np.ndarray, dim: int, M: int, ef_search: int) -> faiss.Index:
    idx = faiss.IndexHNSWFlat(dim, M)
    idx.hnsw.efSearch = ef_search
    idx.add(vecs)
    return idx


def persist_sqlite(
    db_path: str,
    values_by_attr: dict[str, list[str]],
) -> None:
    """Write catalog.db with the schema used by src.normalizer.Normalizer."""
    if os.path.exists(db_path):
        os.remove(db_path)
    conn = sqlite3.connect(db_path)
    conn.execute(
        "CREATE TABLE catalog "
        "(attribute TEXT, value TEXT, PRIMARY KEY (attribute, value))"
    )
    conn.execute(
        "CREATE TABLE hnsw_values "
        "(attribute TEXT, position INTEGER, value TEXT, "
        "PRIMARY KEY (attribute, position))"
    )
    conn.execute(
        "CREATE TABLE mappings "
        "(attribute TEXT, brand TEXT, raw_value TEXT, canonical TEXT, "
        "PRIMARY KEY (attribute, brand, raw_value))"
    )
    for attr, values in values_by_attr.items():
        conn.executemany(
            "INSERT INTO catalog VALUES (?, ?)",
            [(attr, v) for v in values],
        )
        conn.executemany(
            "INSERT INTO hnsw_values VALUES (?, ?, ?)",
            [(attr, i, v) for i, v in enumerate(values)],
        )
    conn.commit()
    conn.close()


async def run(args: argparse.Namespace) -> None:
    dim = EMBED_DIM[args.embedding_model]
    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.dvm)

    brands = sorted({preprocess(x) for x in df["Automaker"]} - {None})
    models = sorted({preprocess(x) for x in df["Genmodel"]} - {None})

    print(f"DVM-CAR  rows: {len(df):,}")
    print(f"Unique brands: {len(brands):,}")
    print(f"Unique models: {len(models):,}")
    print(f"Embedding with {args.embedding_model} (dim={dim})")

    client = AsyncOpenAI()
    for attr, values in [("brand", brands), ("model", models)]:
        vecs = await embed_values(client, args.embedding_model, attr, values)
        idx = build_index(vecs, dim, args.M, args.ef_search)
        out_path = os.path.join(args.out_dir, f"hnsw_{attr}.index")
        faiss.write_index(idx, out_path)
        print(f"  [{attr}] wrote {out_path}  ({idx.ntotal} vectors)")

    db_path = os.path.join(args.out_dir, "catalog.db")
    persist_sqlite(db_path, {"brand": brands, "model": models})
    print(f"wrote {db_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build FAISS HNSW indexes for brand and model from DVM-CAR.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--dvm", default="data/dvm.csv")
    parser.add_argument("--out_dir", default="experiments/db")
    parser.add_argument(
        "--embedding_model",
        default="text-embedding-3-small",
        choices=list(EMBED_DIM.keys()),
    )
    parser.add_argument("--M", type=int, default=32)
    parser.add_argument("--ef_search", type=int, default=64)
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
