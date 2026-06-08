"""
normalize_all.py  —  normalize unique vehicles from autoscout24 and mucars.

FULL PIPELINE MODE: extract → retrieve → match → catalog update.
Each vehicle description is extracted into structured fields, then brand/model/submodel/
trim_level are matched against a shared catalog to avoid duplicates. New canonical values
are added to the catalog automatically.

FILES WRITTEN:
  db/catalog.db                  — SQLite catalog (canonical values + HNSW value index)
  db/hnsw_brand.index            — FAISS HNSW index for brand similarity search
  db/hnsw_model.index            — FAISS HNSW index for model similarity search
  db/hnsw_submodel.index         — FAISS HNSW index for submodel similarity search
  db/hnsw_trim_level.index       — FAISS HNSW index for trim_level similarity search
  data/autoscout24_mappings.db   — SQLite result cache for autoscout24
  data/mucars_mappings.db        — SQLite result cache for mucars

Progress is saved after every batch. Ctrl+C or crashes are safe to resume.
Re-run the same command to continue from where you left off.

Usage:
    uv run python scripts/normalize_all.py                          # normalize all datasets
    uv run python scripts/normalize_all.py --datasets autoscout24  # one dataset
    uv run python scripts/normalize_all.py --test --test_n 5       # dry-run sample
    uv run python scripts/normalize_all.py --fresh_start           # wipe everything and restart
    uv run python scripts/normalize_all.py --batch_size 8          # tune concurrency
"""

import argparse
import asyncio
import json
import os
import shutil
import sqlite3
import sys
import time
from datetime import timedelta

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, ".")
from src.normalizer import Normalizer, serialize_row  # noqa: E402

# ── Per-dataset config ────────────────────────────────────────────────────────

DATASETS = {
    "autoscout24": {
        "path": "data/autoscout24.csv",
        "cols_to_normalize": [
            "brand", "model", "color", "year",
            "transmission_type", "fuel_type",
            "offer_description",
        ],
    },
    "mucars": {
        "path": "data/mucars.csv",
        "cols_to_normalize": ["Brand", "Model", "Year", "Gearbox"],
    },
}

DB_DIR = "db"
ANALYTICS_DB = "data/analytics.db"

# ── Utilities ─────────────────────────────────────────────────────────────────


def _fmt(seconds: float) -> str:
    return str(timedelta(seconds=int(seconds)))


def _db_path(csv_path: str) -> str:
    return os.path.splitext(csv_path)[0] + "_mappings.db"


def _open_analytics_db() -> sqlite3.Connection:
    conn = sqlite3.connect(ANALYTICS_DB)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS batch_metrics (
            id                        INTEGER PRIMARY KEY AUTOINCREMENT,
            ts                        REAL,
            dataset                   TEXT,
            batch_index               INTEGER,
            batch_size                INTEGER,
            batch_failed              INTEGER,
            new_catalog_inserts       INTEGER,
            catalog_brand             INTEGER,
            catalog_model             INTEGER,
            catalog_submodel          INTEGER,
            catalog_trim_level        INTEGER,
            extract_s                 REAL,
            retrieve_s                REAL,
            match_s                   REAL,
            dedup_s                   REAL,
            total_s                   REAL,
            match_calls               INTEGER,
            cache_hits                INTEGER,
            extract_prompt_tokens     INTEGER,
            extract_cached_tokens     INTEGER,
            extract_completion_tokens INTEGER,
            match_prompt_tokens       INTEGER,
            match_cached_tokens       INTEGER,
            match_completion_tokens   INTEGER,
            batch_cost                REAL
        )
    """)
    # Migrate existing DBs that predate the new columns
    new_cols = [
        ("batch_index",               "INTEGER"),
        ("batch_failed",              "INTEGER"),
        ("new_catalog_inserts",       "INTEGER"),
        ("extract_prompt_tokens",     "INTEGER"),
        ("extract_cached_tokens",     "INTEGER"),
        ("extract_completion_tokens", "INTEGER"),
        ("match_prompt_tokens",       "INTEGER"),
        ("match_cached_tokens",       "INTEGER"),
        ("match_completion_tokens",   "INTEGER"),
        ("batch_cost",                "REAL"),
    ]
    for col, col_type in new_cols:
        try:
            conn.execute(f"ALTER TABLE batch_metrics ADD COLUMN {col} {col_type}")
        except sqlite3.OperationalError:
            pass  # column already exists
    conn.commit()
    return conn


def _save_analytics(
    conn: sqlite3.Connection,
    dataset: str,
    batch_index: int,
    batch_size: int,
    batch_failed: int,
    metrics: dict,
    token_deltas: dict,
) -> None:
    conn.execute("""
        INSERT INTO batch_metrics
            (ts, dataset, batch_index, batch_size, batch_failed, new_catalog_inserts,
             catalog_brand, catalog_model, catalog_submodel, catalog_trim_level,
             extract_s, retrieve_s, match_s, dedup_s, total_s,
             match_calls, cache_hits,
             extract_prompt_tokens, extract_cached_tokens, extract_completion_tokens,
             match_prompt_tokens, match_cached_tokens, match_completion_tokens,
             batch_cost)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        time.time(), dataset, batch_index, batch_size, batch_failed,
        metrics["new_catalog_inserts"],
        metrics["catalog_brand"], metrics["catalog_model"],
        metrics.get("catalog_submodel", 0), metrics.get("catalog_trim_level", 0),
        metrics["extract_s"], metrics["retrieve_s"],
        metrics["match_s"], metrics["dedup_s"], metrics["total_s"],
        metrics["match_calls"], metrics["cache_hits"],
        token_deltas["extract_prompt"], token_deltas["extract_cached"], token_deltas["extract_completion"],
        token_deltas["match_prompt"], token_deltas["match_cached"], token_deltas["match_completion"],
        token_deltas["cost"],
    ))
    conn.commit()


def _open_db(path: str) -> tuple[sqlite3.Connection, dict]:
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS mappings "
        "(description TEXT PRIMARY KEY, result TEXT)"
    )
    conn.commit()
    cursor = conn.execute("SELECT description, result FROM mappings")
    mappings = {row[0]: json.loads(row[1]) for row in cursor}
    return conn, mappings


def _save_batch(conn: sqlite3.Connection, pairs: list[tuple[str, dict]]) -> None:
    rows = [(desc, json.dumps(result)) for desc, result in pairs]
    conn.executemany(
        "INSERT OR REPLACE INTO mappings (description, result) VALUES (?, ?)", rows
    )
    conn.commit()


def _confirm_fresh_start(selected: dict) -> bool:
    db_paths = [_db_path(cfg["path"]) for cfg in selected.values()]
    dirs_to_delete = [DB_DIR]

    print("\n  Files/dirs that will be DELETED:")
    for p in db_paths:
        exists = "(exists)" if os.path.exists(p) else "(not found)"
        print(f"    {p}  {exists}")
    for d in dirs_to_delete:
        exists = "(exists)" if os.path.exists(d) else "(not found)"
        print(f"    {d}/  {exists}")

    print()
    answer = input("  ARE YOU SURE? Type 'yes' to confirm: ").strip()
    return answer == "yes"


def _do_fresh_start(selected: dict) -> None:
    for cfg in selected.values():
        p = _db_path(cfg["path"])
        if os.path.exists(p):
            os.remove(p)
            print(f"  Deleted: {p}")
    if os.path.exists(DB_DIR):
        shutil.rmtree(DB_DIR)
        print(f"  Deleted: {DB_DIR}/")
    print()


# ── Dataset normalization ─────────────────────────────────────────────────────


async def _normalize_dataset(
    name: str,
    cfg: dict,
    normalizer: Normalizer,
    batch_size: int,
    prefetch: int,
    test: bool,
    test_n: int,
    overall: dict,
    analytics_conn: sqlite3.Connection,
) -> None:
    csv_path = cfg["path"]
    cols_to_normalize = cfg["cols_to_normalize"]
    db = _db_path(csv_path)

    print(f"\n{'=' * 60}")
    print(f"  Dataset : {name}")
    print(f"  Cache   : {db}")
    print(f"{'=' * 60}")

    df = pd.read_csv(csv_path)
    present_cols = [c for c in cols_to_normalize if c in df.columns]
    missing_cols = [c for c in cols_to_normalize if c not in df.columns]
    if missing_cols:
        print(f"  [warn] Columns not in CSV (skipped): {missing_cols}")

    unique_df = df[present_cols].drop_duplicates().reset_index(drop=True)
    unique_descs = [d for d in unique_df.apply(serialize_row, axis=1).unique() if d]

    total_rows = len(df)
    print(f"  Total rows       : {total_rows:>8,}")
    print(f"  Unique vehicles  : {len(unique_descs):>8,}  ({100*(1 - len(unique_descs)/total_rows):.1f}% dedup reduction)")

    conn, mappings = _open_db(db)
    try:
        to_normalize = [d for d in unique_descs if d not in mappings]
        print(f"  Already cached   : {len(mappings):>8,}")
        print(f"  To normalize     : {len(to_normalize):>8,}")

        # ── Test mode ─────────────────────────────────────────────────────────
        if test:
            sample = to_normalize[:test_n] or unique_descs[:test_n]
            print(f"\n  [TEST MODE] Normalizing {len(sample)} descriptions:\n")
            results, _ = await normalizer(sample, verbose=True)
            for r in results:
                print(f"  Input     : {r['query']}")
                print(f"  Vehicle   : {json.dumps(r['vehicle'], indent=4, ensure_ascii=False)}")
                print(f"  Operations: {r['operations']}\n")
            return

        if not to_normalize:
            print("  Nothing to normalize — all cached.")
            return

        # ── Batch loop ─────────────────────────────────────────────────────────
        total = len(to_normalize)
        done = 0
        failed_total = 0
        start_time = time.time()
        token_snapshot = {
            "prompt": normalizer.tokens["extract"]["prompt"] + normalizer.tokens["match"]["prompt"],
            "completion": normalizer.tokens["extract"]["completion"] + normalizer.tokens["match"]["completion"],
            "calls": normalizer.tokens["extract"]["count"] + normalizer.tokens["match"]["count"],
        }

        # Resume batch_index from last recorded value so Ctrl+C + restart keeps order
        row = analytics_conn.execute(
            "SELECT MAX(batch_index) FROM batch_metrics WHERE dataset = ?", (name,)
        ).fetchone()
        batch_index = (row[0] + 1) if row[0] is not None else 0

        batches = [to_normalize[i : i + batch_size] for i in range(0, total, batch_size)]

        # Pre-start extraction for the first `prefetch` batches
        prefetch_tasks: dict[int, asyncio.Task] = {}
        for j in range(min(prefetch, len(batches))):
            prefetch_tasks[j] = asyncio.create_task(normalizer.extract_all(batches[j]))

        print()
        for idx, batch in enumerate(batches):
            # Kick off extraction for the batch `prefetch` steps ahead
            ahead = idx + prefetch
            if ahead < len(batches):
                prefetch_tasks[ahead] = asyncio.create_task(normalizer.extract_all(batches[ahead]))

            # Snapshot tokens + cost before the batch for per-batch delta
            _t = normalizer.tokens
            pre = {
                "e_prompt":     _t["extract"]["prompt"],
                "e_cached":     _t["extract"]["cached"],
                "e_completion": _t["extract"]["completion"],
                "m_prompt":     _t["match"]["prompt"],
                "m_cached":     _t["match"]["cached"],
                "m_completion": _t["match"]["completion"],
                "cost":         normalizer.get_cost(),
            }

            pre_extractions = await prefetch_tasks.pop(idx) if idx in prefetch_tasks else None

            try:
                results, metrics = await normalizer(batch, extractions=pre_extractions)
            except Exception as e:
                print(f"    [BATCH FAIL] {type(e).__name__}: {e}")
                done += len(batch)
                overall["done"] += len(batch)
                failed_total += len(batch)
                continue

            _t = normalizer.tokens
            token_deltas = {
                "extract_prompt":     _t["extract"]["prompt"]     - pre["e_prompt"],
                "extract_cached":     _t["extract"]["cached"]     - pre["e_cached"],
                "extract_completion": _t["extract"]["completion"] - pre["e_completion"],
                "match_prompt":       _t["match"]["prompt"]       - pre["m_prompt"],
                "match_cached":       _t["match"]["cached"]       - pre["m_cached"],
                "match_completion":   _t["match"]["completion"]   - pre["m_completion"],
                "cost":               normalizer.get_cost()       - pre["cost"],
            }

            # Save successes only — failures are retried on re-run
            batch_pairs: list[tuple[str, dict]] = []
            batch_failed = 0
            for desc, vehicle in zip(batch, results):
                if not vehicle:
                    batch_failed += 1
                    failed_total += 1
                else:
                    batch_pairs.append((desc, vehicle))

            _save_analytics(analytics_conn, name, batch_index, len(batch), batch_failed, metrics, token_deltas)
            batch_index += 1

            if batch_pairs:
                _save_batch(conn, batch_pairs)

            done += len(batch)
            overall["done"] += len(batch)

            elapsed = time.time() - start_time
            elapsed_min = elapsed / 60 if elapsed > 0 else 1e-9
            pct_ds = done / total * 100
            pct_all = overall["done"] / overall["total"] * 100 if overall["total"] else 0
            t = normalizer.tokens
            llm_calls = t["extract"]["count"] + t["match"]["count"] - token_snapshot["calls"]
            rpm = llm_calls / elapsed_min
            tokens_since_start = (
                t["extract"]["prompt"] + t["match"]["prompt"] - token_snapshot["prompt"]
                + t["extract"]["completion"] + t["match"]["completion"] - token_snapshot["completion"]
            )
            tpm = tokens_since_start / elapsed_min
            eta_ds = (total - done) / (done / elapsed) if done > 0 else 0
            cost = normalizer.get_cost()
            catalog_size = sum(len(v) for v in normalizer.catalog.values())

            dpm = done / elapsed_min
            print(
                f"  [{name}  {pct_ds:5.1f}%]  {done:>6,}/{total:,}"
                f"  |  overall: {pct_all:5.1f}%"
                f"  |  DPM: {dpm:5.0f}  RPM: {rpm:5.0f}  TPM: {tpm/1e6:.3f}M"
                f"  |  ETA: {_fmt(eta_ds)}"
                f"  |  catalog: {catalog_size:,}"
                f"  |  cost: ${cost:.4f}"
                f"  |  failed: {batch_failed}/{len(batch)} (total: {failed_total})"
            )

        elapsed_total = time.time() - start_time
        succeeded = done - failed_total
        print(f"\n  Done: {succeeded:,} saved, {failed_total:,} failed (will retry next run)")
        print(f"  Time: {_fmt(elapsed_total)}  |  Cache: {db}")

    finally:
        conn.close()


# ── Main ──────────────────────────────────────────────────────────────────────


async def run(args) -> None:
    selected = {k: v for k, v in DATASETS.items() if k in args.datasets}
    if not selected:
        print(f"[error] No valid datasets. Choose from: {list(DATASETS.keys())}")
        sys.exit(1)

    # ── --fresh_start ─────────────────────────────────────────────────────────
    if args.fresh_start:
        if not _confirm_fresh_start(selected):
            print("  Aborted.")
            sys.exit(0)
        _do_fresh_start(selected)
        if os.path.exists(ANALYTICS_DB):
            os.remove(ANALYTICS_DB)
            print(f"  Deleted: {ANALYTICS_DB}")

    # ── Startup banner ────────────────────────────────────────────────────────
    if not args.test:
        print("\nFiles that will be written:")
        print(f"  {DB_DIR}/catalog.db   (canonical catalog + HNSW values)")
        print(f"  {DB_DIR}/hnsw_*.index   (FAISS similarity indexes)")
        print(f"  {ANALYTICS_DB}   (per-batch latency + catalog analytics)")
        for name, cfg in selected.items():
            print(f"  {_db_path(cfg['path'])}   ({name} result cache)")
        print("\nProgress is saved after every batch. Ctrl+C is safe.\n")

    normalizer = Normalizer(
        match_mode="llm",
        model=args.model,
        persist_directory=DB_DIR,
    )
    analytics_conn = _open_analytics_db()

    # Pre-compute total to-normalize count for overall progress
    overall = {"done": 0, "total": 0}
    if not args.test:
        for cfg in selected.values():
            try:
                df = pd.read_csv(cfg["path"])
                present_cols = [c for c in cfg["cols_to_normalize"] if c in df.columns]
                unique_descs = df[present_cols].drop_duplicates().apply(serialize_row, axis=1).unique()
                _, mappings = _open_db(_db_path(cfg["path"]))
                overall["total"] += sum(1 for d in unique_descs if d and d not in mappings)
            except Exception:
                pass

    try:
        for name, cfg in selected.items():
            await _normalize_dataset(
                name=name,
                cfg=cfg,
                normalizer=normalizer,
                batch_size=args.batch_size,
                prefetch=args.prefetch,
                test=args.test,
                test_n=args.test_n,
                overall=overall,
                analytics_conn=analytics_conn,
            )
    except KeyboardInterrupt:
        print("\n\n[!] Interrupted — progress saved. Re-run to resume.")
    finally:
        analytics_conn.close()

    # ── Final summary ─────────────────────────────────────────────────────────
    if not args.test:
        t_e = normalizer.tokens["extract"]
        t_m = normalizer.tokens["match"]
        total_tokens = t_e["prompt"] + t_e["completion"] + t_m["prompt"] + t_m["completion"]
        catalog_size = sum(len(v) for v in normalizer.catalog.values()) if normalizer.catalog else 0
        print(f"\n{'=' * 60}")
        print(f"  TOTAL COST     : ${normalizer.get_cost():.4f}")
        print(f"  Total tokens   : {total_tokens:,}")
        print(f"    Extract      : prompt={t_e['prompt']:,}  cached={t_e['cached']:,}  completion={t_e['completion']:,}  calls={t_e['count']:,}")
        print(f"    Match        : prompt={t_m['prompt']:,}  cached={t_m['cached']:,}  completion={t_m['completion']:,}  calls={t_m['count']:,}")
        print(f"  Catalog entries: {catalog_size:,}")
        print(f"{'=' * 60}")


def main():
    parser = argparse.ArgumentParser(
        description="Normalize unique vehicles from autoscout24 and mucars.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DATASETS.keys()),
        choices=list(DATASETS.keys()),
        metavar="DATASET",
        help=f"Datasets to normalize (default: all). Choices: {list(DATASETS.keys())}",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=20,
        help="Concurrent LLM calls per batch (default: 20)",
    )
    parser.add_argument(
        "--prefetch",
        type=int,
        default=1,
        help="Batches to pre-extract ahead during retrieve+match (default: 1)",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Print normalized output for a few samples without saving",
    )
    parser.add_argument(
        "--test_n",
        type=int,
        default=3,
        help="Number of samples in test mode (default: 3)",
    )
    parser.add_argument(
        "--model",
        choices=["gpt-4.1-nano", "gpt-4.1-mini"],
        default="gpt-4.1-nano",
        help="OpenAI model used for the IE step (default: gpt-4.1-nano)",
    )
    parser.add_argument(
        "--fresh_start",
        action="store_true",
        help="Wipe catalog, FAISS indexes, and all caches, then normalize from scratch (asks for confirmation)",
    )

    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
