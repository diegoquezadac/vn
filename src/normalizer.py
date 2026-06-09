import os
import json
import asyncio
import sqlite3
import time as _time
from typing import List, Literal, Dict, Optional, Set
import faiss
import numpy as np
import pandas as pd
from langchain_core.documents import Document
from langchain.chat_models import init_chat_model
from langchain_core.prompts import PromptTemplate
from jinja2 import Environment, FileSystemLoader, select_autoescape
from openai import AsyncOpenAI, RateLimitError, APITimeoutError, APIConnectionError
from tenacity import retry, wait_random_exponential, stop_after_attempt, retry_if_exception_type

# Transient failures worth retrying: rate limits plus connection/read stalls.
# Without retrying timeouts, a silently stalled HTTPS connection hangs the
# event loop indefinitely instead of failing fast and being re-issued.
_RETRYABLE = (RateLimitError, APITimeoutError, APIConnectionError)
from src.models import Vehicle, Resolution

_EMBED_DIM = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
}

# per 1M tokens: (input, cached_input, output)
_MODEL_COST = {
    "gpt-4.1-nano": (0.100, 0.025, 0.400),
    "gpt-4.1-mini": (0.400, 0.100, 1.600),
}

_NULLISH = {"nan", "none", "other", "unknown", "n/a", "na", "", "unspecified", "not specified", "missing"}


def serialize_row(row: pd.Series, columns: List[str] | None = None) -> str:
    """
    Serialize a DataFrame row to a structured key-value string.

    Args:
        row: a pandas Series (one row of a DataFrame).
        columns: optional subset of column names to include.
                 If None, all non-null columns are used.

    Returns:
        Comma-separated 'column: value' pairs, e.g.:
        'brand: mazda, model: mx-5, year: 2019, color: red'.
        List-valued columns are rendered as 'col: a; b; c'.

    Nullish strings ("nan", "other", "unknown", "none", etc.) are treated as
    missing and excluded, collapsing semantically identical combinations.
    """
    if columns is not None:
        row = row[columns]
    parts = []
    for col, v in row.items():
        if isinstance(v, list):
            items = [
                str(x).strip()
                for x in v
                if x is not None and str(x).strip().lower() not in _NULLISH
            ]
            if items:
                parts.append(f"{col}: {'; '.join(items)}")
            continue
        if not pd.notna(v):
            continue
        s = str(v).strip()
        if s.lower() in _NULLISH:
            continue
        parts.append(f"{col}: {s}")
    return ", ".join(parts)


def get_empty_catalog() -> dict:
    return {"brand": [], "model": []}


class Normalizer:
    """
    Vehicle representation normalizer. Three modes controlled by ``match_mode``:

    - ``"off"``      - extraction only (IE). No catalog, no FAISS, no ER.
    - ``"threshold"`` - IE + IR. Top-1 candidate is accepted iff its cosine
                       similarity is >= ``match_threshold``; else novel.
    - ``"llm"``      - Full pipeline. IE -> IR -> LLM entity-resolution with
                       full-vehicle context + intra-batch dedup.

    Catalog and HNSW values are persisted in {persist_directory}/catalog.db (SQLite).
    FAISS indexes are persisted as {persist_directory}/hnsw_{attr}.index binary files.
    """

    def __init__(
        self,
        persist_directory: str = "./db",
        provider: Literal["openai"] = "openai",
        model: Literal["gpt-4.1-nano", "gpt-4.1-mini"] = "gpt-4.1-nano",
        match_model: Literal["gpt-4.1-mini", "gpt-4.1-nano"] = "gpt-4.1-mini",
        match_reasoning_effort: Literal["minimal", "low", "medium", "high"] | None = None,
        embedding_model: Literal["text-embedding-3-small", "text-embedding-3-large"] = "text-embedding-3-small",
        k: int = 5,
        match_mode: Literal["off", "threshold", "llm"] = "llm",
        match_threshold: float = 0.9,
        ef_search: int = 64,
        max_concurrency: int = 50,
        extraction_prompt: str = "extraction.j2",
        extraction_samples: str = "./samples/extraction.json",
        request_timeout: float | None = None,
    ):
        # Language model setup. A finite per-request timeout converts a stalled
        # HTTPS connection into a retryable APITimeoutError instead of an
        # indefinite hang of the asyncio event loop.
        _timeout_kwargs = {"timeout": request_timeout} if request_timeout is not None else {}
        llm = init_chat_model(model, model_provider=provider, temperature=0, **_timeout_kwargs)
        _match_kwargs = {"temperature": 1, **_timeout_kwargs}
        if match_reasoning_effort is not None:
            _match_kwargs["reasoning"] = {"effort": match_reasoning_effort}
        match_llm = init_chat_model(match_model, model_provider=provider, **_match_kwargs)

        # Jinja2 + extraction chain
        env = Environment(loader=FileSystemLoader("./prompts/"), autoescape=select_autoescape())

        with open(extraction_samples) as f:
            extraction_samples_data = json.load(f)

        extraction_template = env.get_template(extraction_prompt).render(
            samples=extraction_samples_data, x="TEMPORARY"
        )
        extraction_template = (
            extraction_template.replace("{", "{{")
            .replace("}", "}}")
            .replace("TEMPORARY", "{x}")
            .replace("\n\n\n", "\n\n")
        )
        extraction_prompt_template = PromptTemplate.from_template(template=extraction_template)
        extraction_chain = extraction_prompt_template | llm.with_structured_output(Vehicle, include_raw=True)

        if match_mode not in ("off", "threshold", "llm"):
            raise ValueError(f"match_mode must be 'off', 'threshold', or 'llm'; got {match_mode!r}")

        if match_mode != "off":
            dim = _EMBED_DIM[embedding_model]
            os.makedirs(persist_directory, exist_ok=True)

            # SQLite: catalog + hnsw_values
            catalog_db_path = os.path.join(persist_directory, "catalog.db")
            catalog_conn = sqlite3.connect(catalog_db_path)
            catalog_conn.execute(
                "CREATE TABLE IF NOT EXISTS catalog "
                "(attribute TEXT, value TEXT, PRIMARY KEY (attribute, value))"
            )
            catalog_conn.execute(
                "CREATE TABLE IF NOT EXISTS hnsw_values "
                "(attribute TEXT, position INTEGER, value TEXT, PRIMARY KEY (attribute, position))"
            )
            catalog_conn.execute(
                "CREATE TABLE IF NOT EXISTS mappings "
                "(attribute TEXT, brand TEXT, raw_value TEXT, canonical TEXT, "
                "PRIMARY KEY (attribute, brand, raw_value))"
            )
            catalog_conn.commit()

            # One-time migration from old catalog.json / hnsw_values.json
            if not catalog_conn.execute("SELECT 1 FROM catalog LIMIT 1").fetchone():
                for old_path in ["./data/catalog.json", "data/catalog.json"]:
                    if os.path.exists(old_path):
                        with open(old_path, encoding="utf-8") as f:
                            old = json.load(f)
                        rows = [(attr, val) for attr, vals in old.items() for val in vals]
                        catalog_conn.executemany("INSERT OR IGNORE INTO catalog VALUES (?, ?)", rows)
                        catalog_conn.commit()
                        os.rename(old_path, old_path + ".bak")
                        print(f"[migration] catalog.json -> catalog.db ({len(rows)} entries)")
                        break

            if not catalog_conn.execute("SELECT 1 FROM hnsw_values LIMIT 1").fetchone():
                old_hnsw = os.path.join(persist_directory, "hnsw_values.json")
                if os.path.exists(old_hnsw):
                    with open(old_hnsw) as f:
                        old = json.load(f)
                    rows = [
                        (attr, pos, val)
                        for attr, vals in old.items()
                        for pos, val in enumerate(vals)
                    ]
                    catalog_conn.executemany("INSERT OR IGNORE INTO hnsw_values VALUES (?, ?, ?)", rows)
                    catalog_conn.commit()
                    os.rename(old_hnsw, old_hnsw + ".bak")
                    print(f"[migration] hnsw_values.json -> catalog.db ({len(rows)} entries)")

            # Load catalog into memory
            catalog = get_empty_catalog()
            for attr, val in catalog_conn.execute("SELECT attribute, value FROM catalog"):
                if attr in catalog:
                    catalog[attr].append(val)

            # Load hnsw_values into memory (ordered by position)
            hnsw_values = {attr: [] for attr in ["brand", "model"]}
            for attr, _, val in catalog_conn.execute(
                "SELECT attribute, position, value FROM hnsw_values ORDER BY attribute, position"
            ):
                if attr in hnsw_values:
                    hnsw_values[attr].append(val)

            # Load mapping cache: (attr, brand, raw_value) -> canonical
            mapping_cache: Dict[tuple, str] = {}
            for attr, brand, raw, canonical in catalog_conn.execute(
                "SELECT attribute, brand, raw_value, canonical FROM mappings"
            ):
                mapping_cache[(attr, brand, raw)] = canonical

            # FAISS HNSW indexes
            hnsw_indexes = {}
            for attr in ["brand", "model"]:
                index_path = os.path.join(persist_directory, f"hnsw_{attr}.index")
                if os.path.exists(index_path):
                    idx = faiss.read_index(index_path)
                else:
                    idx = faiss.IndexHNSWFlat(dim, 32)
                idx.hnsw.efSearch = ef_search
                hnsw_indexes[attr] = idx

            # Matching chain (only needed for LLM mode)
            if match_mode == "llm":
                with open("./samples/matching.json") as f:
                    matching_samples = json.load(f)

                matching_template = env.get_template("matching.j2").render(
                    x="TEMPORARY_1", y="TEMPORARY_2", attribute="TEMPORARY_3",
                    samples=matching_samples,
                )
                matching_template = (
                    matching_template.replace("{", "{{")
                    .replace("}", "}}")
                    .replace("TEMPORARY_1", "{x}")
                    .replace("TEMPORARY_2", "{y}")
                    .replace("TEMPORARY_3", "{attribute}")
                    .replace("\n\n\n", "\n\n")
                )
                matching_prompt = PromptTemplate.from_template(template=matching_template)
                matching_chain = matching_prompt | match_llm.with_structured_output(Resolution, include_raw=True)
            else:
                matching_chain = None
        else:
            hnsw_indexes = None
            hnsw_values = None
            catalog = None
            catalog_conn = None
            matching_chain = None
            mapping_cache = {}

        # per 1M tokens: (input, cached_input, output)
        self.extract_cost = _MODEL_COST[model]
        self.match_cost = _MODEL_COST[match_model]

        self.llm = llm
        self.k = k
        self.match_mode = match_mode
        self.match_threshold = match_threshold
        self.ef_search = ef_search
        self.catalog = catalog
        self.catalog_sets: Dict[str, Set[str]] = (
            {attr: set(catalog[attr]) for attr in ["brand", "model"]}
            if catalog else {}
        )
        self.catalog_conn = catalog_conn
        self.persist_directory = persist_directory
        self.hnsw_indexes = hnsw_indexes
        self.hnsw_values = hnsw_values
        self.matching_chain = matching_chain
        self.extraction_chain = extraction_chain
        self.embedding_model = embedding_model
        self.openai_client = (
            AsyncOpenAI(timeout=request_timeout) if match_mode != "off" else None
        )
        self._sem = asyncio.Semaphore(max_concurrency)
        self._match_cache: Dict[tuple, bool] = {}
        self._mapping_cache: Dict[tuple, str] = mapping_cache
        self.tokens = {
            "extract": {"prompt": 0, "cached": 0, "completion": 0, "count": 0},
            "match":   {"prompt": 0, "cached": 0, "completion": 0, "count": 0},
        }
        self.attributes = ["brand", "model"]

    def get_cost(self) -> float:
        cost = 0.0
        for stage, (inp, cached, out) in [("extract", self.extract_cost), ("match", self.match_cost)]:
            t = self.tokens[stage]
            cost += (t["prompt"] - t["cached"]) * inp + t["cached"] * cached + t["completion"] * out
        return cost / 1e6

    @retry(
        retry=retry_if_exception_type(_RETRYABLE),
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(10),
    )
    async def extract(self, q: str) -> dict:
        """Extract structured information from an open-text vehicle description."""
        async with self._sem:
            response = await self.extraction_chain.ainvoke({"x": q})
        usage = response["raw"].response_metadata["token_usage"]
        self.tokens["extract"]["prompt"] += usage["prompt_tokens"]
        self.tokens["extract"]["cached"] += usage["prompt_tokens_details"]["cached_tokens"]
        self.tokens["extract"]["completion"] += usage["completion_tokens"]
        self.tokens["extract"]["count"] += 1
        return response["parsed"].dict()

    async def encode_batch(self, queries: List[str]) -> List[List[float]]:
        """Encode multiple strings in a single API call. Returns embeddings in input order."""
        async with self._sem:
            response = await self.openai_client.embeddings.create(input=queries, model=self.embedding_model)
        return [e.embedding for e in sorted(response.data, key=lambda e: e.index)]

    async def extract_all(self, xs: List[str]) -> List[dict]:
        """Extract structured fields for all descriptions concurrently. Returns one dict per input."""
        raw = await asyncio.gather(*[self.extract(q) for q in xs], return_exceptions=True)
        return [r if not isinstance(r, Exception) else {} for r in raw]

    async def _hnsw_add(self, documents: List[Document], flush: bool = True):
        """
        Embed documents and insert them into the per-attribute HNSW indexes.
        New values are written to catalog.db (hnsw_values table) on flush.
        """
        by_attr: Dict[str, List[Document]] = {}
        for doc in documents:
            attr = doc.metadata["attribute"]
            by_attr.setdefault(attr, []).append(doc)

        new_db_rows: List[tuple] = []
        for attr, docs in by_attr.items():
            texts = [doc.page_content for doc in docs]
            async with self._sem:
                resp = await self.openai_client.embeddings.create(input=texts, model=self.embedding_model)
            vecs = np.array([e.embedding for e in resp.data], dtype=np.float32)
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            vecs = vecs / np.maximum(norms, 1e-9)

            self.hnsw_indexes[attr].add(vecs)
            for doc in docs:
                val = doc.page_content.removeprefix(f"{attr} ")
                pos = len(self.hnsw_values[attr])
                self.hnsw_values[attr].append(val)
                new_db_rows.append((attr, pos, val))

        if flush:
            for attr in by_attr:
                faiss.write_index(
                    self.hnsw_indexes[attr],
                    os.path.join(self.persist_directory, f"hnsw_{attr}.index"),
                )
            if new_db_rows:
                self.catalog_conn.executemany(
                    "INSERT OR IGNORE INTO hnsw_values VALUES (?, ?, ?)", new_db_rows
                )
                self.catalog_conn.commit()

    @retry(
        retry=retry_if_exception_type(_RETRYABLE),
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(10),
    )
    async def _is_match(self, x: str, y: str, attr: str) -> bool:
        """Binary entity resolution: returns True if x and y refer to the same real-world entity."""
        key = (attr, min(x, y), max(x, y))
        if key in self._match_cache:
            return self._match_cache[key]
        async with self._sem:
            response = await self.matching_chain.ainvoke({"x": x, "y": y, "attribute": attr})
        usage = response["raw"].response_metadata.get("token_usage", {})
        self.tokens["match"]["prompt"] += usage.get("prompt_tokens", 0)
        self.tokens["match"]["cached"] += (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
        self.tokens["match"]["completion"] += usage.get("completion_tokens", 0)
        self.tokens["match"]["count"] += 1
        result = response["parsed"].is_match
        self._match_cache[key] = result
        return result

    async def match(self, y: str, D: List[Document], attr: str, extraction: dict) -> str:
        """Match extracted value y against candidates D using full vehicle context.

        Builds s_hat = Serialize(x_hat) and, for each candidate c_i,
        s_hat_{j->c_i} = Serialize(a_1, ..., c_i, ..., a_m) by swapping attr's value in the
        extraction dict and re-serializing. Returns the first matching canonical
        in similarity-ranked order, or ''.
        """
        D_vals = [doc.page_content.removeprefix(f"{attr} ") for doc in D]
        shat = serialize_row(pd.Series(extraction))
        swapped = [serialize_row(pd.Series({**extraction, attr: c})) for c in D_vals]
        results = await asyncio.gather(
            *[self._is_match(shat, ss, attr) for ss in swapped],
            return_exceptions=True,
        )
        for c, is_match in zip(D_vals, results):
            if is_match is True:
                return c
        return ""

    async def _intra_batch_dedup(self, attr: str, val_contexts: List[tuple]) -> Dict[str, str]:
        """
        Sequentially deduplicate new values within a batch for a single attribute.

        Processes values in order: the first value is always canonical. Each subsequent
        value is matched against all accumulated canonicals - if a match is found it is
        redirected to that canonical, otherwise it becomes a new canonical itself.

        val_contexts: list of (value, extraction) tuples where extraction is the full
        extracted vehicle dict used as context for ER comparisons.

        Returns a mapping from each value to its canonical (which may be itself).
        """
        result: Dict[str, str] = {}
        canonicals: List[str] = []
        for val, extraction in val_contexts:
            if canonicals:
                canonical_docs = [
                    Document(page_content=f"{attr} {cv}", metadata={"attribute": attr})
                    for cv in canonicals
                ]
                intra_match = await self.match(val, canonical_docs, attr, extraction=extraction)
                if intra_match:
                    result[val] = intra_match
                    continue
            canonicals.append(val)
            result[val] = val
        return result

    async def __call__(
        self,
        xs: List[str],
        extractions: List[dict] | None = None,
        verbose: bool = False,
    ) -> tuple[List[dict], dict]:
        """
        Normalize a list of open-text vehicle descriptions.

        Args:
            xs: input descriptions.
            extractions: optional pre-computed extraction dicts (skips step 1).
            verbose: if True, each result dict contains {query, vehicle, extraction, operations}.
                     if False (default), each result dict is the vehicle dict directly.

        Returns (results, metrics).

        Behaviour depends on ``self.match_mode``:

          "off"       - return the IE extractions directly. No catalog, no FAISS.
          "threshold" - IE -> IR -> accept top-1 canonical iff cos >= match_threshold,
                        else treat as novel and add. Intra-batch dedup uses the
                        same cosine threshold.
          "llm"       - IE -> IR -> full LLM entity-resolution with context-swap,
                        plus sequential LLM-based intra-batch dedup.
        """
        t0 = _time.perf_counter()

        # Step 1: extract all concurrently (skip if pre-extracted)
        if extractions is None:
            extractions = await self.extract_all(xs)
        t1 = _time.perf_counter()

        if self.match_mode == "off":
            if verbose:
                output = [{"query": q, "vehicle": y} for q, y in zip(xs, extractions)]
            else:
                output = list(extractions)
            metrics = {"match_mode": "off", "extract_s": t1 - t0, "retrieve_s": 0.0,
                       "match_s": 0.0, "dedup_s": 0.0, "total_s": t1 - t0, "match_calls": 0,
                       "cache_hits": len(xs), "catalog_brand": 0, "catalog_model": 0}
            return output, metrics

        # Step 2: encode e_j = Encode(a_j) per attribute and retrieve
        rows_needing_retrieval = [
            i for i, y in enumerate(extractions)
            if any(
                y.get(attr) and y[attr] not in self.catalog_sets[attr]
                for attr in self.attributes
            )
        ]
        cache_hits = len(xs) - len(rows_needing_retrieval)

        # Collect unique (attr, val) pairs to embed, tracking which rows reference each
        attr_val_rows: Dict[tuple, List[int]] = {}
        for i in rows_needing_retrieval:
            y = extractions[i]
            for attr in self.attributes:
                val = y.get(attr)
                if val and val not in self.catalog_sets[attr]:
                    attr_val_rows.setdefault((attr, val), []).append(i)

        retrieval_map: Dict[int, Dict[str, List[Document]]] = {}
        pair_top_cos: Dict[tuple, float] = {}
        pair_embedding: Dict[tuple, np.ndarray] = {}
        if attr_val_rows:
            unique_attr_vals = list(attr_val_rows.keys())
            embeddings = await self.encode_batch([f"{attr} {val}" for attr, val in unique_attr_vals])
            for (attr, val), emb in zip(unique_attr_vals, embeddings):
                q = np.array(emb, dtype=np.float32).reshape(1, -1)
                norm = np.linalg.norm(q)
                if norm > 1e-9:
                    q /= norm
                pair_embedding[(attr, val)] = q[0]
                idx = self.hnsw_indexes[attr]
                docs = []
                top_cos = -1.0
                if idx.ntotal > 0:
                    k = min(self.k, idx.ntotal)
                    distances, indices = idx.search(q, k)
                    # FAISS IndexHNSWFlat returns squared L2; for unit vectors
                    # cos = 1 - d^2/2.
                    if len(distances[0]) > 0:
                        top_cos = float(1.0 - distances[0][0] / 2.0)
                    docs = [
                        Document(
                            page_content=f"{attr} {self.hnsw_values[attr][j]}",
                            metadata={"attribute": attr},
                        )
                        for j in indices[0]
                        if 0 <= j < len(self.hnsw_values[attr])
                    ]
                pair_top_cos[(attr, val)] = top_cos
                for i in attr_val_rows[(attr, val)]:
                    retrieval_map.setdefault(i, {})[attr] = docs
        t2 = _time.perf_counter()

        # Step 3: match unique (attr, value) pairs concurrently
        unique_pairs: Dict[tuple, List[Document]] = {}
        unique_pair_extraction: Dict[tuple, dict] = {}
        unique_pair_brands: Dict[tuple, str] = {}
        for i, y in enumerate(extractions):
            if i not in retrieval_map:
                continue
            for attr in self.attributes:
                val = y.get(attr)
                if val and val not in self.catalog_sets[attr]:
                    key = (attr, val)
                    if key not in unique_pairs:
                        unique_pairs[key] = retrieval_map[i][attr]
                        unique_pair_extraction[key] = y
                        unique_pair_brands[key] = y.get("brand", "") if attr != "brand" else ""

        # Split pairs: mapping cache hit / needs match / no candidates (auto-insert)
        pairs_from_cache:    List[tuple] = []
        pairs_needing_match: List[tuple] = []
        pairs_auto_insert:   List[tuple] = []
        for attr, val in unique_pairs:
            key = (attr, val)
            brand_ctx = unique_pair_brands.get(key, "")
            if (attr, brand_ctx, val) in self._mapping_cache:
                pairs_from_cache.append(key)
            elif unique_pairs[key]:
                pairs_needing_match.append(key)
            else:
                pairs_auto_insert.append(key)

        match_map: Dict[tuple, str] = {}
        for key in pairs_from_cache:
            attr, val = key
            brand_ctx = unique_pair_brands.get(key, "")
            match_map[key] = self._mapping_cache[(attr, brand_ctx, val)]
        for key in pairs_auto_insert:
            match_map[key] = ""

        if self.match_mode == "threshold":
            # Top-1 cosine threshold - no LLM calls.
            for key in pairs_needing_match:
                attr, _ = key
                docs = unique_pairs[key]
                cos = pair_top_cos.get(key, -1.0)
                if docs and cos >= self.match_threshold:
                    match_map[key] = docs[0].page_content.removeprefix(f"{attr} ")
                else:
                    match_map[key] = ""
        else:
            raw_matches = await asyncio.gather(
                *[self.match(val, unique_pairs[(attr, val)], attr, unique_pair_extraction[(attr, val)])
                  for attr, val in pairs_needing_match],
                return_exceptions=True,
            )
            for k, r in zip(pairs_needing_match, raw_matches):
                match_map[k] = r if not isinstance(r, Exception) else ""

        pair_keys = list(unique_pairs.keys())
        t3 = _time.perf_counter()

        # Step 4: intra-batch dedup for new values
        canonical_map: Dict[tuple, str] = {}
        if self.match_mode == "threshold":
            new_by_attr_emb: Dict[str, List[tuple]] = {}
            for (attr, val), matched in match_map.items():
                if not matched:
                    new_by_attr_emb.setdefault(attr, []).append((val, pair_embedding[(attr, val)]))
            for attr, val_embs in new_by_attr_emb.items():
                canonicals: List[tuple] = []  # [(val, emb), ...]
                for val, emb in val_embs:
                    matched_canonical = ""
                    for cv, ce in canonicals:
                        if float(np.dot(emb, ce)) >= self.match_threshold:
                            matched_canonical = cv
                            break
                    if matched_canonical:
                        canonical_map[(attr, val)] = matched_canonical
                    else:
                        canonicals.append((val, emb))
                        canonical_map[(attr, val)] = val
        else:
            new_by_attr: Dict[str, List[tuple]] = {}
            for (attr, val), matched in match_map.items():
                if not matched:
                    new_by_attr.setdefault(attr, []).append((val, unique_pair_extraction[(attr, val)]))
            dedup_results = await asyncio.gather(
                *[self._intra_batch_dedup(attr, val_ctxs) for attr, val_ctxs in new_by_attr.items()],
                return_exceptions=True,
            )
            for attr, result in zip(new_by_attr.keys(), dedup_results):
                if isinstance(result, Exception):
                    for val, _ in new_by_attr[attr]:
                        canonical_map[(attr, val)] = val
                else:
                    for val, canonical in result.items():
                        canonical_map[(attr, val)] = canonical
        t4 = _time.perf_counter()

        # Persist new mappings to SQLite
        # Only for pairs resolved via LLM (not from cache, not auto-inserts).
        # - matched to existing catalog entry -> save raw -> canonical
        # - dedup redirect (canonical != val) -> save raw -> canonical
        # - new catalog entry (canonical == val) -> skip; will be in catalog_sets next run
        mapping_inserts: List[tuple] = []
        for key in pairs_needing_match:
            attr, val = key
            brand_ctx = unique_pair_brands.get(key, "")
            matched = match_map.get(key, "")
            if matched:
                canonical = matched
            else:
                canonical = canonical_map.get(key, val)
                if canonical == val:
                    continue  # new catalog entry, no need to cache
            self._mapping_cache[(attr, brand_ctx, val)] = canonical
            mapping_inserts.append((attr, brand_ctx, val, canonical))
        if mapping_inserts:
            self.catalog_conn.executemany(
                "INSERT OR IGNORE INTO mappings VALUES (?, ?, ?, ?)", mapping_inserts
            )
            self.catalog_conn.commit()

        # Step 5: assemble results, write catalog inserts to SQLite
        logs = []
        docs_to_insert: List[Document] = []
        catalog_inserts: List[tuple] = []

        for i, (q, y) in enumerate(zip(xs, extractions)):
            vehicle = y.copy()
            operations = []

            if i in retrieval_map:
                for attr in self.attributes:
                    val = y.get(attr)
                    if not val or val in self.catalog_sets[attr]:
                        continue

                    key = (attr, val)
                    matched = match_map.get(key, "")

                    if matched:
                        operations.append({"operation": "update", "attribute": attr, "value": matched})
                        vehicle[attr] = matched
                    else:
                        canonical = canonical_map.get(key, val)
                        vehicle[attr] = canonical
                        # Don't pollute model catalog with brand names
                        brand_val = y.get("brand", "")
                        is_brand_pollution = (
                            attr != "brand"
                            and brand_val
                            and canonical.lower() == brand_val.lower()
                        )
                        if canonical == val and canonical not in self.catalog_sets[attr] and not is_brand_pollution:
                            operations.append({"operation": "insert", "attribute": attr, "value": canonical})
                            docs_to_insert.append(Document(
                                page_content=f"{attr} {canonical}",
                                metadata={"attribute": attr},
                            ))
                            self.catalog[attr].append(canonical)
                            self.catalog_sets[attr].add(canonical)
                            catalog_inserts.append((attr, canonical))
                        elif canonical != val:
                            operations.append({"operation": "update", "attribute": attr, "value": canonical})

            logs.append({
                "query": q,
                "vehicle": vehicle,
                "extraction": y,
                "operations": operations,
            })

        # Persist catalog inserts to SQLite (row-level, no full rewrite)
        if catalog_inserts:
            self.catalog_conn.executemany(
                "INSERT OR IGNORE INTO catalog VALUES (?, ?)", catalog_inserts
            )
            self.catalog_conn.commit()

        # Step 6: single HNSW flush for all new catalog entries
        if docs_to_insert:
            await self._hnsw_add(docs_to_insert, flush=True)

        t5 = _time.perf_counter()

        metrics = {
            "match_mode":           self.match_mode,
            "extract_s":            t1 - t0,
            "retrieve_s":           t2 - t1,
            "match_s":              t3 - t2,
            "dedup_s":              t4 - t3,
            "total_s":              t5 - t0,
            "match_calls":          len(pair_keys),
            "cache_hits":           cache_hits,
            "new_catalog_inserts":  len(catalog_inserts),
            "catalog_brand":        len(self.catalog_sets["brand"]),
            "catalog_model":        len(self.catalog_sets["model"]),
        }

        if verbose:
            return logs, metrics
        return [log["vehicle"] for log in logs], metrics
