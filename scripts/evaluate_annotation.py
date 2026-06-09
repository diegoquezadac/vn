"""
evaluate_annotation.py - inter-annotator agreement + ground-truth build.

For each dataset, reads every data/{dataset}_*_labeled.csv and computes:

  - Fleiss' κ                     (categorical attributes)
  - Mean pairwise raw agreement   (categorical attributes)
  - Mean pairwise Jaccard         (set-valued equipment)
  - Majority-vote ground truth    (ABSTAIN when all annotators disagree)
  - Flag if κ < κ_min             (default κ_min = 0.60)

Outputs:
  - data/{dataset}_ground_truth.csv
  - data/{dataset}_annotation_report.json
  - Printed summary per dataset.

Usage:
    uv run python scripts/evaluate_annotation.py
    uv run python scripts/evaluate_annotation.py --datasets autoscout24
"""

import argparse
import glob
import json
import os
from collections import Counter

import numpy as np
import pandas as pd
from statsmodels.stats.inter_rater import fleiss_kappa

DATASETS = ["autoscout24", "mucars"]
KAPPA_THRESHOLD = 0.60
# Require ~50 non-null cells per annotator (10% of n=500) so Fleiss' κ is
# stable. Below this, κ swings on a handful of disagreements.
MIN_NONNULL_FRAC = 0.10
# Lower floor for `_unit` fields: they are structurally sparser than
# attribute-value fields (many listings omit the unit token), but κ on the
# non-null rows is still the signal we want to surface - provided there's
# enough of it. Below this threshold the non-null sample is too small to
# interpret κ meaningfully (e.g. 1-2 disagreeing cells out of 500).
UNIT_MIN_NONNULL_FRAC = 0.10
NULL_SENTINEL = "__NULL__"

# Fields where 0 is physically impossible and is observed only as a
# sentinel output from gpt-4o-mini (which fills optional numeric fields
# with 0 instead of null). Treat these as null for agreement computation.
ZERO_IS_NULL_FIELDS = {
    "engine_power", "engine_size", "cylinders", "doors", "fuel_consumption",
}


def _normalize_value(s: str) -> str:
    """Canonicalize numeric string representations so '200000.0' == '200000'.
    Non-numeric values pass through unchanged."""
    if s == NULL_SENTINEL:
        return s
    try:
        v = float(s)
    except (ValueError, TypeError):
        return s
    if np.isnan(v):
        return NULL_SENTINEL
    if v.is_integer():
        return str(int(v))
    return repr(v)


def _find_labeled(dataset: str, data_dir: str) -> list[str]:
    return sorted(glob.glob(os.path.join(data_dir, f"{dataset}_*_labeled.csv")))


def _annotator_id(path: str, dataset: str) -> str:
    base = os.path.basename(path)
    return base.removeprefix(f"{dataset}_").removesuffix("_labeled.csv")


def pairwise_raw_agreement(raters: list[list[str]]) -> float:
    k = len(raters)
    if k < 2:
        return float("nan")
    arr = np.array(raters, dtype=object)
    n = arr.shape[1]
    total, pairs = 0.0, 0
    for i in range(k):
        for j in range(i + 1, k):
            total += (arr[i] == arr[j]).sum() / n
            pairs += 1
    return total / pairs


def fleiss_kappa_categorical(raters: list[list[str]]) -> float:
    n = len(raters[0])
    categories = sorted({v for row in raters for v in row})
    if len(categories) <= 1:
        return float("nan")  # degenerate: all raters in one category
    idx = {c: i for i, c in enumerate(categories)}
    m = np.zeros((n, len(categories)), dtype=int)
    for r in raters:
        for i, v in enumerate(r):
            m[i, idx[v]] += 1
    try:
        return float(fleiss_kappa(m, method="fleiss"))
    except Exception:
        return float("nan")


def majority_vote(values: list[str]) -> str:
    counts = Counter(values)
    top = counts.most_common()
    if len(top) == len(values):  # all distinct -> ABSTAIN
        return "ABSTAIN"
    return top[0][0]


def pairwise_jaccard(sets: list[list[set]]) -> float:
    k = len(sets)
    if k < 2:
        return float("nan")
    n = len(sets[0])
    total, pairs = 0.0, 0
    for i in range(k):
        for j in range(i + 1, k):
            s = 0.0
            for a, b in zip(sets[i], sets[j]):
                u = a | b
                s += len(a & b) / len(u) if u else 1.0
            total += s / n
            pairs += 1
    return total / pairs


def _parse_equipment(s):
    if s == NULL_SENTINEL:
        return set()
    try:
        v = json.loads(s)
        return set(v) if isinstance(v, list) else set()
    except (json.JSONDecodeError, TypeError):
        return set()


def evaluate(dataset: str, data_dir: str, min_nonnull_frac: float) -> dict | None:
    paths = _find_labeled(dataset, data_dir)
    if len(paths) < 2:
        print(f"[{dataset}] need >= 2 annotators; found {len(paths)}. Skipping.")
        return None

    annotators = [_annotator_id(p, dataset) for p in paths]
    dfs = [pd.read_csv(p) for p in paths]
    n = min(len(df) for df in dfs)
    dfs = [df.head(n) for df in dfs]

    ann_cols = [
        c for c in dfs[0].columns
        if c.startswith("ann_") and not c.endswith("_is_novel")
    ]
    attributes = [c.removeprefix("ann_") for c in ann_cols]

    print(f"\n=== {dataset} - {len(dfs)}, n={n} ===")
    header = f"  {'attribute':32s}  {'% null':>9s}  {'κ':>7s}  {'P_A':>6s}  flag"
    print(header)
    print("  " + "-" * (len(header) - 2))

    report: dict = {
        "dataset": dataset,
        "annotators": annotators,
        "n": n,
        "min_nonnull_frac": min_nonnull_frac,
        "attributes": {},
        "skipped_sparse": [],
    }
    gt = pd.DataFrame(index=range(n))

    kappas = []
    for attr, col in zip(attributes, ann_cols):
        raters_raw = [
            df[col].fillna(NULL_SENTINEL).astype(str).tolist() for df in dfs
        ]
        if attr != "equipment":
            raters_raw = [
                [_normalize_value(v) for v in r] for r in raters_raw
            ]
            if attr in ZERO_IS_NULL_FIELDS:
                raters_raw = [
                    [NULL_SENTINEL if v == "0" else v for v in r]
                    for r in raters_raw
                ]

        total_cells = len(dfs) * n
        nonnull_cells = sum(
            1 for r in raters_raw for v in r
            if v != NULL_SENTINEL and not (attr == "equipment" and v in ("", "[]"))
        )
        nonnull_frac = nonnull_cells / total_cells

        if attr == "equipment":
            set_raters = [[_parse_equipment(s) for s in r] for r in raters_raw]
            threshold = len(dfs) // 2 + 1
            mv = []
            for i in range(n):
                counts = Counter()
                for sr in set_raters:
                    counts.update(sr[i])
                items = sorted(
                    [item for item, c in counts.items() if c >= threshold]
                )
                mv.append(json.dumps(items))
            gt[attr] = mv

            null_frac = 1.0 - nonnull_frac
            if nonnull_frac < min_nonnull_frac:
                report["skipped_sparse"].append({"attribute": attr, "null_frac": null_frac})
                print(f"  {attr:32s}  {null_frac:>9.1%}  {'-':>7s}  {'-':>6s}  [sparse, skipped]")
            else:
                jac = pairwise_jaccard(set_raters)
                report["attributes"][attr] = {
                    "null_frac": null_frac,
                    "pairwise_jaccard": jac,
                    "flag": False,
                }
                print(f"  {attr:32s}  {null_frac:>9.1%}  {'n/a':>7s}  {jac:>6.3f}  (jaccard)")
            continue

        mv = [
            majority_vote([raters_raw[r][i] for r in range(len(raters_raw))])
            for i in range(n)
        ]
        gt[attr] = [v if v != NULL_SENTINEL else "" for v in mv]

        null_frac = 1.0 - nonnull_frac
        # `_unit` fields get a lower floor: structurally high-null across our
        # sources, but κ on the non-null rows is the signal we want to
        # surface when the sample is large enough to interpret.
        threshold = UNIT_MIN_NONNULL_FRAC if attr.endswith("_unit") else min_nonnull_frac
        if nonnull_frac < threshold:
            report["skipped_sparse"].append({"attribute": attr, "null_frac": null_frac})
            print(f"  {attr:32s}  {null_frac:>9.1%}  {'-':>7s}  {'-':>6s}  [sparse, skipped]")
            continue

        k = fleiss_kappa_categorical(raters_raw)
        pa = pairwise_raw_agreement(raters_raw)
        flag = (not np.isnan(k)) and (k < KAPPA_THRESHOLD)
        report["attributes"][attr] = {
            "null_frac": null_frac,
            "kappa": None if np.isnan(k) else k,
            "raw_agreement": pa,
            "flag": flag,
        }
        if not np.isnan(k):
            kappas.append(k)
        k_str = "  nan  " if np.isnan(k) else f"{k:+.3f}"
        print(f"  {attr:32s}  {null_frac:>9.1%}  {k_str:>7s}  {pa:>6.3f}  {'[FLAG]' if flag else ''}")

    pooled = float(np.mean(kappas)) if kappas else float("nan")
    report["pooled_kappa"] = None if np.isnan(pooled) else pooled
    report["abstain_rate"] = float((gt == "ABSTAIN").sum().sum() / (n * len(attributes)))
    print(f"  pooled κ (over {len(kappas)} reported attributes): {pooled:+.3f}")
    print(f"  skipped (sparse, >{1 - min_nonnull_frac:.0%} null): {len(report['skipped_sparse'])} attributes")
    print(f"  ABSTAIN rate: {report['abstain_rate']:.3%}")

    gt_path = os.path.join(data_dir, f"{dataset}_ground_truth.csv")
    gt.to_csv(gt_path, index=False)
    print(f"  wrote {gt_path}")

    report_path = os.path.join(data_dir, f"{dataset}_annotation_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  wrote {report_path}")

    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute inter-annotator agreement and build ground truth.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--datasets", nargs="+", default=DATASETS, choices=DATASETS)
    parser.add_argument("--data_dir", default="data")
    parser.add_argument(
        "--min_nonnull_frac", type=float, default=MIN_NONNULL_FRAC,
        help="Skip κ/P_A for attributes where fewer than this fraction of "
             "(annotator, row) cells carry a non-null value.",
    )
    args = parser.parse_args()

    for ds in args.datasets:
        evaluate(ds, args.data_dir, args.min_nonnull_frac)


if __name__ == "__main__":
    main()
