"""
sampling.py  —  draw 500 evaluation records from autoscout24 and mucars.

Proportional stratified sampling on brand. This preserves the population
brand distribution in expectation, so dataset-level metrics estimated on
the sample remain unbiased estimators of the population mean. We deliberately
stratify on brand alone: adding a second attribute (e.g. fuel type) produces
brand × fuel strata that are extremely sparse relative to n, so floor
allocation leaves most strata at 0 and the leftover-fill phase does almost
all the work — without meaningfully improving balance on either attribute.

As a diagnostic we also report, per dataset, the fraction of rows
whose brand is in 𝒦_brand and whose (brand, model) pair appears in
DVM-CAR, both after lowercasing and whitespace stripping. When the
dataset encodes the model column as "{brand} {model}" (common in
AutoScout24) the brand prefix is stripped before lookup. Catalog
membership is evaluated against the DVM-CAR pair index read directly
from data/dvm.csv, so sampling has no dependency on the embedding
pipeline. These coverage numbers are purely descriptive; the
ground-truth novelty flag used in downstream metrics is assigned by
human annotators.

Output: data/{dataset}_sample.csv  (all original columns preserved).

Usage:
    uv run python scripts/sampling.py
"""

import argparse
import os

import numpy as np
import pandas as pd

DATASETS = {
    "autoscout24": {
        "path": "data/autoscout24.csv",
        "brand": "brand",
        "model": "model",
    },
    "mucars": {
        "path": "data/mucars.csv",
        "brand": "Brand",
        "model": "Model",
    },
}


def preprocess(s) -> str | None:
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return None
    v = str(s).strip().lower()
    return v or None


def load_catalog(dvm_path: str) -> dict:
    """Load DVM-CAR as a (brand, model) pair index.

    Returns {"brand": set of brands, "pairs": set of (brand, model) tuples}.
    Model membership is evaluated at the pair level so name collisions
    across brands (e.g. a model named "corolla" under a non-Toyota brand)
    register as out-of-catalog rather than as spurious IC hits.
    """
    df = pd.read_csv(dvm_path)
    brands = {preprocess(x) for x in df["Automaker"]} - {None}
    pairs = {
        (preprocess(b), preprocess(m))
        for b, m in zip(df["Automaker"], df["Genmodel"])
    }
    pairs = {p for p in pairs if p[0] is not None and p[1] is not None}
    return {"brand": brands, "pairs": pairs}


def stratified_sample(
    df: pd.DataFrame,
    by: list[str],
    n: int,
    rng: np.random.RandomState,
) -> pd.DataFrame:
    """Proportional allocation across strata defined by `by`, with fill-in if strata are sparse."""
    if n <= 0 or len(df) == 0:
        return df.iloc[0:0]
    if len(df) <= n:
        return df.copy()

    groups = list(df.groupby(by, dropna=False))
    total = sum(len(g) for _, g in groups)

    # Floor allocation, then distribute leftover by largest fractional part
    alloc: dict = {}
    fracs: list[tuple] = []
    for key, g in groups:
        exact = len(g) * n / total
        alloc[key] = int(exact)
        fracs.append((key, exact - int(exact)))
    remaining = n - sum(alloc.values())
    fracs.sort(key=lambda t: t[1], reverse=True)
    for key, _ in fracs[:remaining]:
        alloc[key] += 1

    parts = []
    for key, g in groups:
        take = min(alloc.get(key, 0), len(g))
        if take > 0:
            parts.append(g.sample(n=take, random_state=rng.randint(0, 2**31 - 1)))
    out = pd.concat(parts) if parts else df.iloc[0:0]

    # If we still owe rows (rare: rounding + tiny strata), fill randomly from unused
    if len(out) < n:
        leftover = df.drop(out.index)
        if len(leftover) > 0:
            fill = leftover.sample(
                n=min(n - len(out), len(leftover)),
                random_state=rng.randint(0, 2**31 - 1),
            )
            out = pd.concat([out, fill])
    return out


def sample_one(
    name: str,
    cfg: dict,
    catalog: dict,
    n: int,
    seed: int,
    out_dir: str,
) -> dict:
    df = pd.read_csv(cfg["path"])
    brand_col, model_col = cfg["brand"], cfg["model"]

    df["_brand"] = df[brand_col].map(preprocess)
    df["_model"] = df[model_col].map(preprocess)

    # A usable row needs at least a brand (the minimum signal for stratification)
    usable = df[df["_brand"].notna()].copy()

    # Some datasets (e.g. autoscout24) encode the model column as "{brand} {model}".
    # For catalog membership we compare the model stem (with brand prefix stripped
    # when present) against 𝒦_model. This is still a heuristic, but matches what
    # the IE step would extract at inference time.
    def strip_brand(row):
        m, b = row["_model"], row["_brand"]
        if m and b and m.startswith(b + " "):
            return m[len(b) + 1 :]
        return m

    usable["_model_stem"] = usable.apply(strip_brand, axis=1)
    usable["_brand_in"] = usable["_brand"].isin(catalog["brand"])
    # Brand-scoped: (brand, model) pair must exist in DVM-CAR.
    usable["_model_in"] = [
        (b, m) in catalog["pairs"]
        for b, m in zip(usable["_brand"], usable["_model_stem"])
    ]

    pop_brand_ic = 100 * usable["_brand_in"].mean()
    pop_model_ic = 100 * usable["_model_in"].mean()

    print(f"\n=== {name} ===")
    print(f"  rows total           : {len(df):>8,}")
    print(f"  rows with brand set  : {len(usable):>8,}")

    rng = np.random.RandomState(seed)
    sample = stratified_sample(usable, ["_brand"], n, rng)
    sample = sample.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    original_cols = [c for c in df.columns if not c.startswith("_")]
    out_df = sample[original_cols]

    out_path = os.path.join(out_dir, f"{name}_sample.csv")
    out_df.to_csv(out_path, index=False)

    smp_brand_ic = 100 * sample["_brand_in"].mean() if len(sample) else 0.0
    smp_model_ic = 100 * sample["_model_in"].mean() if len(sample) else 0.0
    print(f"  wrote                : {out_path}  ({len(out_df)} rows)")

    return {
        "dataset": name,
        "pop_brand_ic": pop_brand_ic,
        "pop_model_ic": pop_model_ic,
        "smp_brand_ic": smp_brand_ic,
        "smp_model_ic": smp_model_ic,
    }


def _render(header: tuple, rows: list[tuple], title: str) -> None:
    widths = [max(len(c[i]) for c in [header, *rows]) for i in range(len(header))]
    fmt = "  " + "  ".join(f"{{:<{w}}}" for w in widths)
    print(f"\n{title}")
    print(fmt.format(*header))
    print(fmt.format(*["-" * w for w in widths]))
    for r in rows:
        print(fmt.format(*r))


def print_summary(stats: list[dict]) -> None:
    # Per-attribute catalog coverage: population vs realized sample, brand and model separate.
    header = (
        "dataset",
        "pop brand IC", "pop brand OOC",
        "smp brand IC", "smp brand OOC",
        "pop model IC", "pop model OOC",
        "smp model IC", "smp model OOC",
    )
    rows = [
        (
            s["dataset"],
            f"{s['pop_brand_ic']:.1f}%", f"{100 - s['pop_brand_ic']:.1f}%",
            f"{s['smp_brand_ic']:.1f}%", f"{100 - s['smp_brand_ic']:.1f}%",
            f"{s['pop_model_ic']:.1f}%", f"{100 - s['pop_model_ic']:.1f}%",
            f"{s['smp_model_ic']:.1f}%", f"{100 - s['smp_model_ic']:.1f}%",
        )
        for s in stats
    ]
    _render(header, rows, "Per-attribute catalog coverage (population vs sample)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Draw proportional-stratified evaluation samples from each dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--dvm", default="data/dvm.csv")
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", default="data")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DATASETS.keys()),
        choices=list(DATASETS.keys()),
    )
    args = parser.parse_args()

    if not os.path.exists(args.dvm):
        raise SystemExit(f"DVM-CAR index not found at {args.dvm}.")

    catalog = load_catalog(args.dvm)
    print(
        f"Catalog: {len(catalog['brand'])} brands, "
        f"{len(catalog['pairs'])} (brand, model) pairs"
    )

    os.makedirs(args.out_dir, exist_ok=True)
    stats = []
    for name in args.datasets:
        stats.append(
            sample_one(
                name=name,
                cfg=DATASETS[name],
                catalog=catalog,
                n=args.n,
                seed=args.seed,
                out_dir=args.out_dir,
            )
        )
    print_summary(stats)


if __name__ == "__main__":
    main()
