"""
audit.py  -  run a small sample through the full pipeline and print detailed
results for debugging: raw extraction, canonical vehicle, catalog operations.

Uses a temporary catalog and FAISS indexes deleted after the run,
so it never touches the real data.

Usage:
    uv run python scripts/audit.py                          # 5 samples from each dataset
    uv run python scripts/audit.py --n 10                   # more samples
    uv run python scripts/audit.py --datasets autoscout24   # one dataset
"""

import argparse
import asyncio
import json
import random
import shutil
import sys
import tempfile

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
            "transmission_type",
            "fuel_type",
            "offer_description",
        ],
    },
    "mucars": {
        "path": "data/mucars.csv",
        "cols_to_normalize": [
            "Brand",
            "Model",
            "Year",
            "Condition",
            "Gearbox",
            "Fiscal Power",
            "Fuel",
            "Equipment",
            "Number of Doors",
        ],
    },
}

W = 70  # separator width


def _sep(char="-"):
    print(char * W)


def _header(title: str):
    print(f"\n{'=' * W}")
    print(f"  {title}")
    print(f"{'=' * W}")


def _print_result(i: int, r: dict):
    _sep()
    print(f"  [{i + 1}] INPUT")
    _sep(".")
    print(f"  {r['query']}")

    print()
    _sep()
    print(f"  {'FIELD':<26}  {'EXTRACTED':<28}  CANONICAL")
    _sep(".")
    all_keys = [k for k, v in r["extraction"].items() if v is not None]
    for k in all_keys:
        raw = r["extraction"].get(k)
        canonical = r["vehicle"].get(k)
        changed = raw != canonical and canonical is not None
        marker = "  <-" if changed else ""
        print(f"  {k:<26}  {str(raw):<28}  {canonical}{marker}")

    print()
    _sep()
    print("  CATALOG OPERATIONS")
    _sep(".")
    if r["operations"]:
        for op in r["operations"]:
            icon = "+" if op["operation"] == "insert" else "~"
            print(
                f"  [{icon}] {op['operation']:<8}  {op['attribute']:<14}  {op['value']}"
            )
    else:
        print("  (none - all attributes already in catalog)")
    print()


async def run(args) -> None:
    selected = {k: v for k, v in DATASETS.items() if k in args.datasets}

    if args.seed is not None:
        random.seed(args.seed)

    tmp_dir = tempfile.mkdtemp(prefix="vn_audit_")
    try:
        normalizer = Normalizer(
            match_mode="llm",
            persist_directory=tmp_dir,
        )

        for name, cfg in selected.items():
            _header(f"DATASET: {name.upper()}  -  {args.n} samples")

            df = pd.read_csv(cfg["path"])
            present_cols = [c for c in cfg["cols_to_normalize"] if c in df.columns]
            unique_descs = [
                d
                for d in df[present_cols]
                .drop_duplicates()
                .apply(serialize_row, axis=1)
                .unique()
                if d
            ]
            sample = random.sample(unique_descs, min(args.n, len(unique_descs)))

            print(f"  Unique vehicles in dataset : {len(unique_descs):,}")
            print(f"  Sample size                : {len(sample)}")
            print(
                f"  Catalog size (before)      : {sum(len(v) for v in normalizer.catalog.values()):,}"
            )
            print()

            results, _ = await normalizer(sample, verbose=True)

            for i, r in enumerate(results):
                _print_result(i, r)

            inserts = sum(
                1 for r in results for op in r["operations"] if op["operation"] == "insert"
            )
            updates = sum(
                1 for r in results for op in r["operations"] if op["operation"] == "update"
            )
            catalog_size = sum(len(v) for v in normalizer.catalog.values())

            _sep("=")
            print(f"  SUMMARY - {name}")
            _sep(".")
            print(f"  Samples processed          : {len(results)}")
            print(f"  Catalog inserts            : {inserts}")
            print(f"  Catalog updates (matched)  : {updates}")
            print(f"  Catalog size (after)       : {catalog_size:,}")
            cost = normalizer.get_cost()
            print(f"  Cost so far                : ${cost:.4f}")
            print()

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(
        description="Audit full-pipeline normalization on a small sample.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DATASETS.keys()),
        choices=list(DATASETS.keys()),
        metavar="DATASET",
        help=f"Datasets to audit (default: all). Choices: {list(DATASETS.keys())}",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=5,
        help="Number of samples per dataset (default: 5)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible sampling (default: random)",
    )
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
