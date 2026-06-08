"""
estimate.py  —  sample N descriptions per dataset, run a real timed batch through
the full pipeline (extract → retrieve → match → catalog), and extrapolate cost + time.

Uses a temporary catalog and FAISS indexes that are deleted after the run,
so it never pollutes your real data.

Usage:
    uv run python scripts/estimate.py               # 50 samples per dataset
    uv run python scripts/estimate.py --n 100       # more samples for tighter estimate
    uv run python scripts/estimate.py --model gpt-4.1-nano
"""

import argparse
import asyncio
import shutil
import sys
import tempfile
import time
from datetime import timedelta

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, ".")
from src.normalizer import Normalizer, serialize_row  # noqa: E402

DATASETS = {
    "autoscout24": {
        "path": "data/autoscout24.csv",
        "cols_to_normalize": [
            "brand",
            "model",
            "color",
            "year",
            "power_kw",
            "power_ps",
            "transmission_type",
            "fuel_type",
            "offer_description",
        ],
    },
    # "mucars": {
    #     "path": "data/mucars.csv",
    #     "cols_to_normalize": [
    #         "Brand",
    #         "Model",
    #         "Year",
    #         "Gearbox",
    #         "Fuel",
    #         "Equipment",
    #     ],
    # },
}


def _fmt(seconds: float) -> str:
    return str(timedelta(seconds=int(seconds)))


async def run(n: int, model: str, batch_size: int = 25, prefetch: int = 1) -> None:
    print(f"Loading datasets and sampling {n} descriptions per dataset...\n")

    samples_by_dataset: dict[str, list[str]] = {}
    totals: dict[str, int] = {}

    for name, cfg in DATASETS.items():
        df = pd.read_csv(cfg["path"])
        cols = [c for c in cfg["cols_to_normalize"] if c in df.columns]
        unique_descs = [
            d
            for d in df[cols].drop_duplicates().apply(serialize_row, axis=1).unique()
            if d
        ]
        totals[name] = len(unique_descs)
        sample = unique_descs[:n]
        samples_by_dataset[name] = sample
        print(
            f"  {name}: {len(unique_descs):,} unique descriptions  →  sampling {len(sample)}"
        )

    total_unique = sum(totals.values())
    print(f"\n  Total unique descriptions to normalize: {total_unique:,}")
    print(f"  Model: {model}\n")

    # Use a temp dir so the estimate never touches the real catalog / FAISS indexes
    tmp_dir = tempfile.mkdtemp(prefix="vn_estimate_")
    tmp_db = f"{tmp_dir}/db"

    try:
        normalizer = Normalizer(
            match_mode="llm",
            model=model,
            persist_directory=tmp_db,
        )

        print("=" * 60)
        print(f"  Running full pipeline (extract + match) ...")
        print("=" * 60)

        dataset_results = {}

        for name, descs in samples_by_dataset.items():
            t0 = time.time()
            results = []
            batches = [descs[i : i + batch_size] for i in range(0, len(descs), batch_size)]

            # Pre-start extraction for the first `prefetch` batches so that
            # while batch i is in retrieve+match, batch i+prefetch is already extracting.
            prefetch_tasks: dict[int, asyncio.Task] = {}
            for j in range(min(prefetch, len(batches))):
                prefetch_tasks[j] = asyncio.create_task(normalizer.extract_all(batches[j]))

            for idx, batch in enumerate(batches):
                # Kick off extraction for the batch that is `prefetch` steps ahead
                ahead = idx + prefetch
                if ahead < len(batches):
                    prefetch_tasks[ahead] = asyncio.create_task(
                        normalizer.extract_all(batches[ahead])
                    )
                # Await pre-extracted results (or extract now if prefetch=0)
                pre_extractions = await prefetch_tasks.pop(idx) if idx in prefetch_tasks else None
                batch_vehicles, _ = await normalizer(batch, extractions=pre_extractions)
                results.extend(batch_vehicles)

            elapsed = time.time() - t0

            failed = sum(1 for v in results if not v)
            succeeded = len(descs) - failed
            catalog_size = sum(len(v) for v in normalizer.catalog.values())

            dataset_results[name] = {
                "n": len(descs),
                "succeeded": succeeded,
                "failed": failed,
                "elapsed_s": elapsed,
                "s_per_desc": elapsed / len(descs),
            }
            print(
                f"\n  [{name}]  {succeeded}/{len(descs)} ok"
                f"  |  {elapsed:.1f}s  ({elapsed / len(descs):.2f}s/desc)"
                f"  |  catalog: {catalog_size:,}"
            )

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    # ── Token / cost summary ───────────────────────────────────────────────────
    t_e = normalizer.tokens["extract"]
    t_m = normalizer.tokens["match"]
    total_sampled = sum(r["n"] for r in dataset_results.values())
    total_tokens = t_e["prompt"] + t_e["completion"] + t_m["prompt"] + t_m["completion"]
    total_cost = normalizer.get_cost()
    tokens_per_desc = total_tokens / total_sampled
    cost_per_desc = total_cost / total_sampled

    print("\n" + "=" * 60)
    print("  SAMPLE RESULTS")
    print("=" * 60)
    print(f"  Descriptions sampled : {total_sampled}")
    print(f"  Tokens / description : {tokens_per_desc:,.0f}")
    print(
        f"    Extract            : prompt={t_e['prompt'] / total_sampled:,.0f}  cached={t_e['cached'] / total_sampled:,.0f}  completion={t_e['completion'] / total_sampled:,.0f}  calls={t_e['count']:,}"
    )
    print(
        f"    Match              : prompt={t_m['prompt'] / total_sampled:,.0f}  cached={t_m['cached'] / total_sampled:,.0f}  completion={t_m['completion'] / total_sampled:,.0f}  calls={t_m['count']:,}"
    )
    print(f"  Cost / description   : ${cost_per_desc:.5f}")
    print(f"  Sample cost          : ${total_cost:.4f}")

    print("\n" + "=" * 60)
    print("  EXTRAPOLATION TO FULL RUN")
    print("=" * 60)

    for name, res in dataset_results.items():
        full_n = totals[name]
        est_cost = full_n * cost_per_desc
        est_time_s = full_n * res["s_per_desc"]
        print(f"\n  {name}:")
        print(f"    Unique descriptions : {full_n:>8,}")
        print(f"    Estimated cost      : ${est_cost:>8.2f}")
        print(
            f"    Estimated time      : {_fmt(est_time_s)}  ({res['s_per_desc']:.2f}s/desc × {full_n:,})"
        )

    full_cost = total_unique * cost_per_desc
    full_time_s = sum(
        totals[name] * dataset_results[name]["s_per_desc"] for name in DATASETS
    )
    print(f"\n  {'TOTAL':-<40}")
    print(f"    Unique descriptions : {total_unique:>8,}")
    print(f"    Estimated cost      : ${full_cost:>8.2f}")
    print(f"    Estimated time      : {_fmt(full_time_s)}")
    print("=" * 60)
    print("\n  Note: time estimate assumes same concurrency as this test run.")
    print("  Match call count grows as catalog fills — early batches cost more.")
    print("  Cached prompt tokens will reduce cost on repeated runs.")


def main():
    parser = argparse.ArgumentParser(
        description="Estimate full-pipeline normalization cost and time."
    )
    parser.add_argument(
        "--n",
        type=int,
        default=500,
        help="Descriptions to sample per dataset (default: 500)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=20,
        help="Batch size matching normalize.py (default: 20)",
    )
    parser.add_argument(
        "--model",
        default="gpt-4.1-nano",
        choices=["gpt-4.1-nano"],
        help="Model to use (default: gpt-4.1-nano)",
    )
    parser.add_argument(
        "--prefetch",
        type=int,
        default=1,
        help="Batches to pre-extract ahead during retrieve+match (default: 1)",
    )
    args = parser.parse_args()
    asyncio.run(run(args.n, args.model, args.batch_size, args.prefetch))


if __name__ == "__main__":
    main()
