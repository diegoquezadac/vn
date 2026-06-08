"""
ablation_k.py — A2 ablation: sensitivity of IE+IR+ER to retrieval depth k.

For each dataset (default: autoscout24, mucars):
  (1) Run IE once.
  (2) For each k in the grid, run the full IE+IR+ER pipeline with
      a fresh catalog and the IE extractions reused, and measure:
        - model accuracy + macro F1 (with 95% bootstrap CI)
        - novelty precision / recall / F1 (with CIs)
        - actual LLM ER call count and USD cost
        - calls per record (cost normalized to dataset size)

The script writes:
  data/ablation_k.json
  manuscript/figures/ablation_k.pdf, .png

Usage:
    uv run python scripts/ablation_k.py
    uv run python scripts/ablation_k.py --plot_only
    uv run python scripts/ablation_k.py --k_grid 1 3 5 10 20
"""

import argparse
import asyncio
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.normalizer import Normalizer, serialize_row  # noqa: E402
from scripts.evaluate_normalizer import (  # noqa: E402
    _fresh_catalog,
    clip_row,
    load_dvm_pairs,
)
from scripts.ablation_threshold import (  # noqa: E402
    COLORS,
    _hline,
    _plot_curve,
    _setup_rc,
    compute_metrics,
)

load_dotenv()


DEFAULT_K_GRID = [1, 3, 5, 10, 20]
DEFAULT_DATASETS = ["autoscout24", "mucars"]
N_BOOTSTRAP = 500
CHOSEN_K = 5


# --- experiment driver ------------------------------------------------------

async def evaluate_dataset(
    dataset: str,
    seed_dir: str,
    dvm_pairs: set,
    data_dir: str,
    k_grid: List[int],
    extraction_prompt: str,
    extraction_samples: str,
    n_bootstrap: int = N_BOOTSTRAP,
) -> dict:
    print(f"\n=== {dataset} ===")
    sample_df = pd.read_csv(f"{data_dir}/{dataset}_sample.csv")
    gt_df = pd.read_csv(f"{data_dir}/{dataset}_ground_truth.csv").head(
        len(sample_df)
    )
    listings = [serialize_row(clip_row(r)) for _, r in sample_df.iterrows()]
    print(f"  n={len(listings)}, k_grid={k_grid}")

    print("  [IE] one-shot extraction pass...")
    ie_norm = Normalizer(
        match_mode="off",
        extraction_prompt=extraction_prompt,
        extraction_samples=extraction_samples,
    )
    ie_results, _ = await ie_norm(listings)

    ie_metrics = compute_metrics(
        ie_results, ie_results, gt_df, dvm_pairs, n_bootstrap=n_bootstrap
    )

    k_metrics: Dict[str, dict] = {}
    for i, k in enumerate(k_grid):
        print(f"  [{i + 1}/{len(k_grid)}] k={k} ...")
        tmp = _fresh_catalog(seed_dir, f"{dataset}_k{k}")
        try:
            norm = Normalizer(
                match_mode="llm",
                k=k,
                persist_directory=tmp,
                extraction_prompt=extraction_prompt,
                extraction_samples=extraction_samples,
            )
            results, run_metrics = await norm(listings, extractions=ie_results)
            llm_calls = norm.tokens["match"]["count"]
            llm_cost = norm.get_cost()
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

        m = compute_metrics(
            results, ie_results, gt_df, dvm_pairs, n_bootstrap=n_bootstrap
        )
        m["llm_calls"] = llm_calls
        m["llm_calls_per_record"] = llm_calls / max(len(listings), 1)
        m["llm_cost_usd"] = llm_cost
        m["match_calls"] = run_metrics.get("match_calls", 0)
        k_metrics[str(k)] = m
        print(f"      model F1={m['model_f1']:.3f}  "
              f"novelty F1={m['novelty_f1']:.3f}  "
              f"calls={llm_calls}  ${llm_cost:.4f}")

    return {
        "dataset": dataset,
        "n": len(listings),
        "k_grid": k_grid,
        "ie_metrics": ie_metrics,
        "k_metrics": k_metrics,
    }


# --- plotting ---------------------------------------------------------------

def _series(report_ds: dict, key: str, k_grid: List[int]):
    values, cis = [], []
    for k in k_grid:
        m = report_ds["k_metrics"][str(k)]
        values.append(m.get(key, float("nan")))
        ci_key = f"{key}_ci"
        cis.append(m.get(ci_key, [float("nan"), float("nan")]))
    return values, cis


def _model_ylim(ds: dict, k_grid: List[int]) -> tuple:
    vals: List[float] = []
    for key in ("model_acc", "model_f1"):
        v, cis = _series(ds, key, k_grid)
        vals += [x for x in v if not np.isnan(x)]
        for lo, hi in cis:
            if not np.isnan(lo): vals.append(lo)
            if not np.isnan(hi): vals.append(hi)
    if not vals:
        return 0.7, 1.005
    lo = max(0.0, min(vals) - 0.04)
    hi = min(1.005, max(vals) + 0.04)
    return lo, hi


def _vline_k(ax, k=CHOSEN_K):
    ax.axvline(k, color=COLORS["tau_marker"], ls=(0, (1, 2)),
               lw=0.8, alpha=0.55, zorder=1)


def plot_report(report: dict, out_pdf: str, out_png: str) -> None:
    _setup_rc()

    datasets = list(report["datasets"].keys())
    n_rows = len(datasets)
    k_grid = report["k_grid"]

    fig, axes = plt.subplots(
        n_rows, 3,
        figsize=(9.2, 1.10 * n_rows + 0.75),
        sharex=True,
    )
    if n_rows == 1:
        axes = np.array([axes])

    pretty = {"autoscout24": "AutoScout24", "mucars": "MuCars"}

    for row, dataset in enumerate(datasets):
        ds = report["datasets"][dataset]
        ie = ds["ie_metrics"]

        # ─── Col 0: model accuracy + macro F1 ──────────────────────────────
        ax = axes[row, 0]
        vals_acc, cis_acc = _series(ds, "model_acc", k_grid)
        vals_f1, cis_f1 = _series(ds, "model_f1", k_grid)
        _plot_curve(ax, k_grid, vals_acc, cis_acc,
                    COLORS["blue_dark"], "Accuracy", marker="o")
        _plot_curve(ax, k_grid, vals_f1, cis_f1,
                    COLORS["blue_mid"], "Macro F1", marker="s")
        _hline(ax, ie.get("model_acc"), COLORS["ie_ref"], "IE baseline")
        _vline_k(ax)
        if row == 0:
            ax.set_title("Model normalization", pad=6)
        ax.grid(True)
        ax.set_ylim(*_model_ylim(ds, k_grid))
        if row == n_rows - 1:
            ax.set_xlabel(r"Retrieval depth $k$")
        ax.set_ylabel("Metric value", fontsize=9.5, labelpad=3)

        # ─── Col 1: novelty P / R / F1 ─────────────────────────────────────
        ax = axes[row, 1]
        vals_p, cis_p = _series(ds, "novelty_p", k_grid)
        vals_r, cis_r = _series(ds, "novelty_r", k_grid)
        vals_f, cis_f = _series(ds, "novelty_f1", k_grid)
        _plot_curve(ax, k_grid, vals_p, cis_p,
                    COLORS["green_dark"], "Precision", marker="o")
        _plot_curve(ax, k_grid, vals_r, cis_r,
                    COLORS["green_mid"], "Recall", marker="^")
        _plot_curve(ax, k_grid, vals_f, cis_f,
                    COLORS["teal_dark"], "F1", marker="s")
        _hline(ax, ie.get("novelty_f1"), COLORS["ie_ref"], "IE F1")
        _vline_k(ax)
        if row == 0:
            ax.set_title("Novelty detection", pad=6)
        ax.grid(True)
        ax.set_ylim(0.0, 1.05)
        if row == n_rows - 1:
            ax.set_xlabel(r"Retrieval depth $k$")

        # ─── Col 2: catalog compression ────────────────────────────────────
        ax = axes[row, 2]
        vals_mc, _ = _series(ds, "model_compression", k_grid)
        vals_bc, _ = _series(ds, "brand_compression", k_grid)
        ax.plot(
            k_grid, [v * 100 for v in vals_mc],
            marker="o", markersize=4.5, lw=1.5,
            color=COLORS["gold_dark"], label="Model",
            markeredgecolor="white", markeredgewidth=0.5, zorder=3,
        )
        ax.plot(
            k_grid, [v * 100 for v in vals_bc],
            marker="s", markersize=4.5, lw=1.5,
            color=COLORS["gold_mid"], label="Brand",
            markeredgecolor="white", markeredgewidth=0.5, zorder=3,
        )
        _vline_k(ax)
        if row == 0:
            ax.set_title("Catalog compression", pad=6)
        ax.grid(True)
        all_vals = [v * 100 for v in vals_mc + vals_bc if not np.isnan(v)]
        ymax = max(all_vals + [1.0]) * 1.22
        ax.set_ylim(-0.5, ymax)
        if row == n_rows - 1:
            ax.set_xlabel(r"Retrieval depth $k$")
        ax.set_ylabel("Compr. (%)", fontsize=9.5, labelpad=3)

        # Explicit x-ticks at k values (uneven spacing)
        for c in range(3):
            axes[row, c].set_xticks(k_grid)
            axes[row, c].set_xticklabels([str(k) for k in k_grid])

    fig.tight_layout(h_pad=0.5, w_pad=1.4, rect=(0.035, 0.16, 1, 1))

    # Per-row dataset labels in the left margin.
    for row, dataset in enumerate(datasets):
        ax = axes[row, 0]
        bbox = ax.get_position()
        y_center = (bbox.y0 + bbox.y1) / 2.0
        fig.text(
            0.008, y_center, pretty.get(dataset, dataset),
            fontsize=11, fontweight="bold", color="#222222",
            ha="left", va="center", rotation=90,
        )

    legend_kwargs = dict(
        loc="upper center",
        frameon=True, framealpha=0.92, edgecolor="#bbbbbb",
        fontsize=8.5, handlelength=1.8, columnspacing=1.3,
        borderpad=0.35, handletextpad=0.5,
    )
    bottom_axes_bbox = axes[-1, 0].get_position()
    legend_y = bottom_axes_bbox.y0 - 0.105
    for col in range(3):
        bbox = axes[-1, col].get_position()
        x_center = (bbox.x0 + bbox.x1) / 2.0
        handles, labels = axes[-1, col].get_legend_handles_labels()
        fig.legend(
            handles, labels,
            bbox_to_anchor=(x_center, legend_y),
            ncol=len(handles),
            **legend_kwargs,
        )

    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=220)
    print(f"  wrote {out_pdf}")
    print(f"  wrote {out_png}")


# --- entrypoint -------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="A2 ablation: retrieval depth k for IE+IR+ER.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS,
                        choices=DEFAULT_DATASETS)
    parser.add_argument("--k_grid", nargs="+", type=int,
                        default=DEFAULT_K_GRID)
    parser.add_argument("--seed_dir", default="experiments/db")
    parser.add_argument("--dvm", default="data/dvm.csv")
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--out_json", default="data/ablation_k.json")
    parser.add_argument("--out_pdf",
                        default="manuscript/figures/ablation_k.pdf")
    parser.add_argument("--out_png",
                        default="manuscript/figures/ablation_k.png")
    parser.add_argument("--extraction_prompt", default="extraction.j2")
    parser.add_argument("--extraction_samples",
                        default="./samples/extraction.json")
    parser.add_argument("--n_bootstrap", type=int, default=N_BOOTSTRAP)
    parser.add_argument("--plot_only", action="store_true",
                        help="Skip the sweep; re-render figure from out_json.")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out_pdf) or ".", exist_ok=True)

    if args.plot_only:
        with open(args.out_json) as f:
            report = json.load(f)
        plot_report(report, args.out_pdf, args.out_png)
        return

    if not os.path.exists(os.path.join(args.seed_dir, "catalog.db")):
        print(f"[error] seed catalog not found at {args.seed_dir}/catalog.db")
        print(f"        run: uv run python scripts/catalog.py "
              f"--out_dir {args.seed_dir}")
        sys.exit(1)

    _, dvm_pairs = load_dvm_pairs(args.dvm)
    print(f"DVM-CAR: {len(dvm_pairs)} pairs | seed_dir: {args.seed_dir}")
    print(f"k grid: {args.k_grid}  (n_bootstrap={args.n_bootstrap})")

    async def run_all():
        per_ds: dict = {}
        for ds in args.datasets:
            per_ds[ds] = await evaluate_dataset(
                ds, args.seed_dir, dvm_pairs, args.data_dir,
                k_grid=args.k_grid,
                extraction_prompt=args.extraction_prompt,
                extraction_samples=args.extraction_samples,
                n_bootstrap=args.n_bootstrap,
            )
        return per_ds

    per_ds = asyncio.run(run_all())

    report = {
        "k_grid":   args.k_grid,
        "chosen_k": CHOSEN_K,
        "datasets": per_ds,
    }
    with open(args.out_json, "w") as f:
        json.dump(report, f, indent=2, default=lambda o: None)
    print(f"\nwrote {args.out_json}")

    plot_report(report, args.out_pdf, args.out_png)


if __name__ == "__main__":
    main()
