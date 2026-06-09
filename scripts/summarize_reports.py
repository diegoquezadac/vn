"""
summarize_reports.py - print the consolidated VN evaluation tables from
the saved data/{dataset}_normalizer_report_{suffix}.json files.

Loads three reports per dataset:
  - {dataset}_normalizer_report_3way.json    -> ie + threshold(τ=0.90) + llm
  - {dataset}_normalizer_report_thr085.json  -> threshold(τ=0.85)
  - {dataset}_normalizer_report_thr095.json  -> threshold(τ=0.95)

and prints:
  (1) Intrinsic accuracy / macro-F1 per (attribute, variant, dataset)
  (2) Novelty detection: FPR and F1 per (variant, dataset)
  (3) Canonicalization compression on `model`

Usage:
    uv run python scripts/summarize_reports.py
    uv run python scripts/summarize_reports.py --datasets autoscout24 mucars
"""

import argparse
import json
import os
import sys

DATASETS_DEFAULT = ["autoscout24", "mucars"]
SUFFIXES = {"3way": "3way", "085": "thr085", "095": "thr095"}


def load(ds: str, suffix: str, data_dir: str) -> dict:
    path = os.path.join(data_dir, f"{ds}_normalizer_report_{suffix}.json")
    if not os.path.exists(path):
        print(f"[error] missing {path}", file=sys.stderr)
        sys.exit(1)
    with open(path) as f:
        return json.load(f)


def fpr(n: dict) -> float:
    denom = n["fp"] + n["tn"]
    return n["fp"] / denom if denom else float("nan")


def prf(n: dict) -> tuple[float, float, float]:
    return n["precision"], n["recall"], n["f1"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=DATASETS_DEFAULT)
    parser.add_argument("--data_dir", default="data")
    args = parser.parse_args()

    r3 = {ds: load(ds, SUFFIXES["3way"], args.data_dir) for ds in args.datasets}
    r085 = {ds: load(ds, SUFFIXES["085"], args.data_dir) for ds in args.datasets}
    r095 = {ds: load(ds, SUFFIXES["095"], args.data_dir) for ds in args.datasets}

    # (1) Intrinsic accuracy / macro-F1 (with 95% bootstrap CIs)
    print("=" * 140)
    print("Intrinsic accuracy / macro-F1  [95% bootstrap CI]")
    print("=" * 140)
    hdr = f"{'attr':<7s} {'variant':<22s} " + " ".join(
        [f"{ds + ' acc [CI]':>26s} {ds + ' F1 [CI]':>26s}" for ds in args.datasets]
    )
    print(hdr)
    print("-" * len(hdr))

    def fmt(v: float, ci: list | None) -> str:
        if ci is None or any(c != c for c in ci):  # any NaN
            return f"{v:.3f}"
        return f"{v:.3f} [{ci[0]:.3f},{ci[1]:.3f}]"

    def intrinsic_row(attr: str, label: str, getter):
        cells = []
        for ds in args.datasets:
            (acc, acc_ci), (f1, f1_ci) = getter(ds, attr)
            cells.append(f"{fmt(acc, acc_ci):>26s} {fmt(f1, f1_ci):>26s}")
        print(f"{attr:<7s} {label:<22s} " + " ".join(cells))

    def pack(node_intr, node_prf, a):
        return (
            (node_intr[a]["accuracy"], node_intr[a].get("accuracy_ci")),
            (node_prf[a]["f1"],        node_prf[a].get("f1_ci")),
        )

    for attr in ["brand", "model"]:
        intrinsic_row(attr, "IE", lambda ds, a: pack(
            r3[ds]["intrinsic"]["ie"], r3[ds]["macro_prf"]["ie"], a))
        intrinsic_row(attr, "IE+IR (tau=0.85)", lambda ds, a: pack(
            r085[ds]["intrinsic"]["threshold"], r085[ds]["macro_prf"]["threshold"], a))
        intrinsic_row(attr, "IE+IR (tau=0.90)", lambda ds, a: pack(
            r3[ds]["intrinsic"]["threshold"], r3[ds]["macro_prf"]["threshold"], a))
        intrinsic_row(attr, "IE+IR (tau=0.95)", lambda ds, a: pack(
            r095[ds]["intrinsic"]["threshold"], r095[ds]["macro_prf"]["threshold"], a))
        intrinsic_row(attr, "Full (IE+IR+ER)", lambda ds, a: pack(
            r3[ds]["intrinsic"]["llm"], r3[ds]["macro_prf"]["llm"], a))
        print()

    # (2) Novelty detection - Precision / Recall / F1 on the novel class  [95% bootstrap CI]
    print("=" * 160)
    print("Novelty detection on (brand, model) pairs - DVM-CAR as known set  [95% bootstrap CI]")
    print("=" * 160)
    hdr = f"{'variant':<22s} " + " ".join(
        [f"{ds + ' P [CI]':>24s} {ds + ' R [CI]':>24s} {ds + ' F1 [CI]':>24s}" for ds in args.datasets]
    )
    print(hdr)
    print("-" * len(hdr))

    def _nci(node: dict, key: str) -> str:
        v = node[key]
        ci = node.get(f"{key}_ci")
        if ci is None or any(c != c for c in ci):
            return f"{v:.3f}"
        return f"{v:.3f} [{ci[0]:.3f},{ci[1]:.3f}]"

    def nov_row(label: str, getter):
        cells = []
        for ds in args.datasets:
            node = getter(ds)
            cells.append(f"{_nci(node, 'precision'):>24s} {_nci(node, 'recall'):>24s} {_nci(node, 'f1'):>24s}")
        print(f"{label:<22s} " + " ".join(cells))

    nov_row("IE",               lambda ds: r085[ds]["novelty"]["ie"])
    nov_row("IE+IR (tau=0.85)", lambda ds: r085[ds]["novelty"]["threshold"])
    nov_row("IE+IR (tau=0.90)", lambda ds: r3[ds]["novelty"]["threshold"])
    nov_row("IE+IR (tau=0.95)", lambda ds: r095[ds]["novelty"]["threshold"])
    nov_row("Full (IE+IR+ER)",  lambda ds: r3[ds]["novelty"]["llm"])
    print()

    # (3) Canonicalization compression on `model`
    print("=" * 80)
    print("Canonicalization compression (model) - IE-unique -> VN-unique")
    print("=" * 80)
    hdr = f"{'variant':<22s} " + " ".join([f"{ds:>24s}" for ds in args.datasets])
    print(hdr)
    print("-" * len(hdr))

    def comp_cell(canon: dict) -> str:
        ext = canon["model"]["ie_unique"]
        can = canon["model"]["vn_unique"]
        pct = (1 - can / ext) * 100 if ext else 0.0
        return f"{ext}->{can} ({pct:.1f}%)"

    def comp_row(label: str, getter):
        cells = [f"{comp_cell(getter(ds)):>24s}" for ds in args.datasets]
        print(f"{label:<22s} " + " ".join(cells))

    comp_row("IE+IR (tau=0.85)", lambda ds: r085[ds]["canonicalization"]["threshold"])
    comp_row("IE+IR (tau=0.90)", lambda ds: r3[ds]["canonicalization"]["threshold"])
    comp_row("IE+IR (tau=0.95)", lambda ds: r095[ds]["canonicalization"]["threshold"])
    comp_row("Full (IE+IR+ER)",  lambda ds: r3[ds]["canonicalization"]["llm"])


if __name__ == "__main__":
    main()
