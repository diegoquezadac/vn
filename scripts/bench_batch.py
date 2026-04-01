"""
bench_batch.py  —  find the batch size that maximizes descriptions/minute.

Runs a fixed sample through the full pipeline at different batch sizes,
measures DPM/RPM/TPM, and prints a comparison table.

Uses a temp catalog so it never touches real data.

Usage:
    uv run python scripts/bench_batch.py
    uv run python scripts/bench_batch.py --n 50 --sizes 25 50 100 200
"""

import argparse
import asyncio
import shutil
import sys
import tempfile
import time

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, ".")
from src.normalizer import Normalizer, serialize_row  # noqa: E402

DATASET = {
    "path": "data/autoscout24.csv",
    "cols_to_normalize": ["brand", "model", "color", "year", "transmission_type", "fuel_type", "offer_description"],
}


async def bench(descs: list[str], batch_size: int, tmp_dir: str, prefetch: int = 0) -> dict:
    normalizer = Normalizer(
        extract_only=False,
        persist_directory=f"{tmp_dir}/db",
        max_concurrency=500,
    )

    batches = [descs[i : i + batch_size] for i in range(0, len(descs), batch_size)]

    t0 = time.time()
    if prefetch > 0:
        prefetch_tasks: dict[int, asyncio.Task] = {}
        for j in range(min(prefetch, len(batches))):
            prefetch_tasks[j] = asyncio.create_task(normalizer.extract_all(batches[j]))

        for idx, batch in enumerate(batches):
            ahead = idx + prefetch
            if ahead < len(batches):
                prefetch_tasks[ahead] = asyncio.create_task(normalizer.extract_all(batches[ahead]))
            pre_extractions = await prefetch_tasks.pop(idx) if idx in prefetch_tasks else None
            await normalizer(batch, extractions=pre_extractions)
    else:
        for batch in batches:
            await normalizer(batch)
    elapsed = time.time() - t0

    elapsed_min = elapsed / 60
    t = normalizer.tokens
    llm_calls = t["extract"]["count"] + t["match"]["count"]
    total_tokens = (
        t["extract"]["prompt"] + t["extract"]["completion"]
        + t["match"]["prompt"] + t["match"]["completion"]
    )
    return {
        "batch_size": batch_size,
        "elapsed_s": elapsed,
        "dpm": len(descs) / elapsed_min,
        "rpm": llm_calls / elapsed_min,
        "tpm": total_tokens / elapsed_min,
        "cost": normalizer.get_cost(),
    }


async def run(n: int, sizes: list[int]) -> None:
    df = pd.read_csv(DATASET["path"])
    cols = [c for c in DATASET["cols_to_normalize"] if c in df.columns]
    unique_descs = [d for d in df[cols].drop_duplicates().apply(serialize_row, axis=1).unique() if d]
    descs = unique_descs[:n]

    print(f"Sample: {len(descs)} descriptions from autoscout24")
    print(f"Testing batch sizes: {sizes}\n")
    print(f"  {'batch':>6}  {'prefetch':>8}  {'DPM':>6}  {'RPM':>6}  {'TPM':>8}  {'time':>8}  {'cost':>8}")
    print("  " + "-" * 65)

    results = []
    configs = [(size, prefetch) for size in sizes for prefetch in [0, 1]]
    for i, (size, prefetch) in enumerate(configs):
        if i > 0:
            print(f"  Waiting 60s for rate-limit window to reset...\n")
            await asyncio.sleep(60)

        tmp_dir = tempfile.mkdtemp(prefix="vn_bench_")
        try:
            r = await bench(descs, size, tmp_dir, prefetch=prefetch)
            r["prefetch"] = prefetch
            results.append(r)
            print(
                f"  {size:>6}  {prefetch:>8}  {r['dpm']:>6.0f}  {r['rpm']:>6.0f}"
                f"  {r['tpm']/1e6:>7.3f}M  {r['elapsed_s']:>7.1f}s  ${r['cost']:>7.4f}"
            )
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

        if prefetch == 1:
            print()

    best = max(results, key=lambda r: r["dpm"])
    print(f"  Best: batch_size={best['batch_size']}  prefetch={best['prefetch']}  ({best['dpm']:.0f} DPM)")


def main():
    parser = argparse.ArgumentParser(description="Benchmark batch sizes for normalization throughput.")
    parser.add_argument("--n", type=int, default=500, help="Descriptions to test per batch size (default: 500)")
    parser.add_argument("--sizes", nargs="+", type=int, default=[20, 30, 40, 50],
                        help="Batch sizes to test (default: 20, 30, 40, 50)")
    args = parser.parse_args()
    asyncio.run(run(args.n, args.sizes))


if __name__ == "__main__":
    main()
