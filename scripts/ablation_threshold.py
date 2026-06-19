"""
ablation_threshold.py - A1 ablation: sensitivity of IE+IR to τ.

For each dataset (default: autoscout24, mucars):
  (1) Run IE once.
  (2) For each τ in the grid, run IE+IR (fresh catalog, reusing IE
      extractions) and compute:
        - model accuracy + macro F1 (with 95% bootstrap CI)
        - novelty precision / recall / F1 (with CIs)
        - model and brand compression
  (3) Also compute IE-baseline metrics (no normalization).

The script writes:
  data/ablation_threshold.json
  figures/ablation_threshold.pdf, .png

Usage:
    uv run python scripts/catalog.py --out_dir experiments/db   # one-off seed
    uv run python scripts/ablation_threshold.py
    uv run python scripts/ablation_threshold.py --plot_only     # re-render
    uv run python scripts/ablation_threshold.py --tau_grid 0.5 0.6 0.7 0.8 0.85 0.9 0.95
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
    attribute_accuracy,
    bootstrap_ci,
    canonicalization_ratio,
    clip_row,
    load_dvm_pairs,
    macro_prf,
    normalize_value,
    novelty_metrics,
)

load_dotenv()


DEFAULT_TAU_GRID = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
DEFAULT_DATASETS = ["autoscout24", "mucars"]
N_BOOTSTRAP = 500
CHOSEN_TAU = 0.80


# --- metric computation -----------------------------------------------------

def _gt_pairs(gt_df: pd.DataFrame) -> List[tuple]:
    def to_pair(b, m):
        bn, mn = normalize_value(b), normalize_value(m)
        if bn == "abstain" or mn == "abstain":
            return ("", "")
        return (bn, mn) if bn and mn else ("", "")
    return [to_pair(gt_df["brand"].iloc[i], gt_df["model"].iloc[i])
            for i in range(len(gt_df))]


def compute_metrics(
    results: list,
    ie_results: list,
    gt_df: pd.DataFrame,
    dvm_pairs: set,
    n_bootstrap: int = N_BOOTSTRAP,
) -> dict:
    """Compute model acc/F1, novelty P/R/F1, and compression with CIs."""
    out: dict = {}

    # Open-domain attribute metrics
    for attr in ["brand", "model"]:
        truths = gt_df[attr].tolist()
        preds = [r.get(attr) for r in results]

        out[f"{attr}_acc"] = attribute_accuracy(preds, truths, attr).get(
            "accuracy", float("nan")
        )
        prf = macro_prf(preds, truths)
        out[f"{attr}_f1"] = prf["f1"]

        lo, hi = bootstrap_ci(
            lambda p, t, a=attr: attribute_accuracy(p, t, a).get(
                "accuracy", float("nan")
            ),
            preds, truths, n_resamples=n_bootstrap,
        )
        out[f"{attr}_acc_ci"] = [lo, hi]

        lo, hi = bootstrap_ci(
            lambda p, t: macro_prf(p, t).get("f1", float("nan")),
            preds, truths, n_resamples=n_bootstrap,
        )
        out[f"{attr}_f1_ci"] = [lo, hi]

    # Novelty (pair-level)
    gt_pairs = _gt_pairs(gt_df)
    pred_pairs = []
    for r in results:
        bn = normalize_value(r.get("brand"))
        mn = normalize_value(r.get("model"))
        if bn == "abstain" or mn == "abstain":
            pred_pairs.append(("", ""))
        else:
            pred_pairs.append((bn, mn) if bn and mn else ("", ""))

    filt = [(p, t) for p, t in zip(pred_pairs, gt_pairs) if t != ("", "")]
    preds_v = [p for p, _ in filt]
    truths_v = [t for _, t in filt]
    nov = novelty_metrics(preds_v, truths_v, dvm_pairs)
    out["novelty_p"] = nov["precision"]
    out["novelty_r"] = nov["recall"]
    out["novelty_f1"] = nov["f1"]

    for metric in ("precision", "recall", "f1"):
        lo, hi = bootstrap_ci(
            lambda p, t, m=metric: novelty_metrics(p, t, dvm_pairs).get(
                m, float("nan")
            ),
            preds_v, truths_v, n_resamples=n_bootstrap,
        )
        key = "novelty_p_ci" if metric == "precision" else (
            "novelty_r_ci" if metric == "recall" else "novelty_f1_ci"
        )
        out[key] = [lo, hi]

    # Compression
    for attr in ["brand", "model"]:
        ie_vals = [r.get(attr) for r in ie_results]
        vn_vals = [r.get(attr) for r in results]
        c = canonicalization_ratio(ie_vals, vn_vals)
        if c["ie_unique"]:
            out[f"{attr}_compression"] = 1.0 - c["vn_unique"] / c["ie_unique"]
        else:
            out[f"{attr}_compression"] = 0.0
        out[f"{attr}_unique_ie"] = c["ie_unique"]
        out[f"{attr}_unique_vn"] = c["vn_unique"]

    return out


# --- experiment driver ------------------------------------------------------

async def evaluate_dataset(
    dataset: str,
    seed_dir: str,
    dvm_pairs: set,
    data_dir: str,
    tau_grid: List[float],
    extraction_prompt: str,
    extraction_samples: str,
    n_bootstrap: int = N_BOOTSTRAP,
) -> dict:
    print(f"\n=== {dataset} ===")
    sample_df = pd.read_csv(f"{data_dir}/X_{dataset}.csv")
    gt_df = pd.read_csv(f"{data_dir}/Y_{dataset}.csv").head(
        len(sample_df)
    )
    listings = [serialize_row(clip_row(r)) for _, r in sample_df.iterrows()]
    print(f"  n={len(listings)}, tau_grid={tau_grid}")

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

    tau_metrics: Dict[str, dict] = {}
    for i, tau in enumerate(tau_grid):
        print(f"  [{i + 1}/{len(tau_grid)}] tau={tau:.2f} ...")
        tmp = _fresh_catalog(seed_dir, f"{dataset}_t{tau:.2f}")
        try:
            norm = Normalizer(
                match_mode="threshold",
                match_threshold=tau,
                persist_directory=tmp,
                extraction_prompt=extraction_prompt,
                extraction_samples=extraction_samples,
            )
            results, _ = await norm(listings, extractions=ie_results)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

        m = compute_metrics(
            results, ie_results, gt_df, dvm_pairs, n_bootstrap=n_bootstrap
        )
        tau_metrics[f"{tau:.2f}"] = m
        print(f"      model F1={m['model_f1']:.3f}  "
              f"novelty F1={m['novelty_f1']:.3f}  "
              f"compression={m['model_compression'] * 100:.1f}%")

    return {
        "dataset": dataset,
        "n": len(listings),
        "tau_grid": tau_grid,
        "ie_metrics": ie_metrics,
        "tau_metrics": tau_metrics,
    }


# --- plotting ---------------------------------------------------------------

# Thesis palette (shared with the thesis document and manuscript/main.tex;
# see ../thesis/CLAUDE.md). One hue family per panel using graduated shades:
# sky = model normalization, violet = novelty detection, mint = compression.
# Ink supplies neutral grays for baselines, markers, frame and text.
COLORS = {
    # Model normalization panel -> sky (deep + base)
    "blue_dark":   "#0A5480",  # sky deep
    "blue_mid":    "#3DB8F5",  # sky base
    # Novelty detection panel -> violet (F1 deep, precision dark, recall base)
    "teal_dark":   "#1C1B6A",  # violet deep  (F1)
    "teal_mid":    "#B2B1F0",  # violet light (unused)
    "teal_light":  "#D0CFF7",  # violet soft  (unused)
    "green_dark":  "#4B4AC0",  # violet dark  (precision)
    "green_mid":   "#8E8DE8",  # violet base  (recall)
    # Catalog compression panel -> mint (deep + base)
    "gold_dark":   "#0A5C44",  # mint deep
    "gold_mid":    "#2EEDB5",  # mint base
    # Neutral reference lines -> ink
    "ie_ref":      "#55528A",  # ink light
    "tau_marker":  "#3A3768",  # ink mid
}


def _setup_rc():
    plt.rcParams.update({
        "font.family":     "serif",
        "font.serif":      ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size":       9,
        "axes.labelsize":  9,
        "axes.titlesize":  10,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.linewidth":  0.7,
        "axes.edgecolor":  "#3A3768",  # ink mid (frame)
        "axes.labelcolor": "#07060F",  # ink (text)
        "xtick.color":     "#07060F",  # ink (text)
        "ytick.color":     "#07060F",  # ink (text)
        "grid.color":      "#7B78A8",  # ink muted
        "grid.alpha":      0.25,
        "grid.linewidth":  0.4,
        "legend.frameon":  True,
        "legend.framealpha": 0.92,
        "legend.edgecolor": "#7B78A8",  # ink muted
        "savefig.bbox":    "tight",
    })


def _plot_curve(ax, taus, values, cis, color, label, marker="o"):
    values = np.asarray(values, dtype=float)
    ax.plot(
        taus, values, marker=marker, markersize=4.0, lw=1.4,
        color=color, label=label, markeredgecolor="white",
        markeredgewidth=0.5, zorder=3,
    )
    if cis is not None:
        cis = np.asarray(cis, dtype=float)
        valid = ~np.any(np.isnan(cis), axis=1)
        if valid.any():
            ax.fill_between(
                np.asarray(taus)[valid],
                cis[valid, 0], cis[valid, 1],
                color=color, alpha=0.13, lw=0, zorder=2,
            )


def _hline(ax, y, color, label):
    if y is None or (isinstance(y, float) and np.isnan(y)):
        return
    ax.axhline(y, color=color, ls=(0, (4, 3)), lw=0.9, alpha=0.75,
               label=label, zorder=1)


def _vline_tau(ax, tau=CHOSEN_TAU):
    ax.axvline(tau, color=COLORS["tau_marker"], ls=(0, (1, 2)),
               lw=0.8, alpha=0.55, zorder=1)


def _series(report_ds, key):
    taus = sorted(float(t) for t in report_ds["tau_metrics"].keys())
    values, cis = [], []
    for t in taus:
        m = report_ds["tau_metrics"][f"{t:.2f}"]
        values.append(m.get(key, float("nan")))
        ci_key = f"{key}_ci"
        cis.append(m.get(ci_key, [float("nan"), float("nan")]))
    return taus, values, cis


def _model_ylim(ds: dict) -> tuple:
    """Data-driven y-range for the model panel (acc + F1 + their CIs)."""
    vals: List[float] = []
    for key in ("model_acc", "model_f1"):
        _, v, cis = _series(ds, key)
        vals += [x for x in v if not np.isnan(x)]
        for lo, hi in cis:
            if not np.isnan(lo): vals.append(lo)
            if not np.isnan(hi): vals.append(hi)
    if not vals:
        return 0.7, 1.005
    lo = max(0.0, min(vals) - 0.04)
    hi = min(1.005, max(vals) + 0.04)
    return lo, hi


def plot_report(report: dict, out_pdf: str, out_png: str) -> None:
    _setup_rc()

    datasets = list(report["datasets"].keys())
    n_rows = len(datasets)

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

        # Col 0: model accuracy + macro F1
        ax = axes[row, 0]
        taus, vals_acc, cis_acc = _series(ds, "model_acc")
        _, vals_f1, cis_f1 = _series(ds, "model_f1")
        _plot_curve(ax, taus, vals_acc, cis_acc,
                    COLORS["blue_dark"], "Accuracy", marker="o")
        _plot_curve(ax, taus, vals_f1, cis_f1,
                    COLORS["blue_mid"], "Macro F1", marker="s")
        _hline(ax, ie.get("model_acc"), COLORS["ie_ref"], "IE baseline")
        _vline_tau(ax)
        if row == 0:
            ax.set_title("Model normalization", pad=6)
        ax.grid(True)
        ax.set_ylim(*_model_ylim(ds))
        if row == n_rows - 1:
            ax.set_xlabel(r"Cosine threshold $\tau$")
        ax.set_ylabel("Metric value", fontsize=9.5, labelpad=3)

        # Col 1: novelty P / R / F1
        ax = axes[row, 1]
        _, vals_p, cis_p = _series(ds, "novelty_p")
        _, vals_r, cis_r = _series(ds, "novelty_r")
        _, vals_f, cis_f = _series(ds, "novelty_f1")
        _plot_curve(ax, taus, vals_p, cis_p,
                    COLORS["green_dark"], "Precision", marker="o")
        _plot_curve(ax, taus, vals_r, cis_r,
                    COLORS["green_mid"], "Recall", marker="^")
        _plot_curve(ax, taus, vals_f, cis_f,
                    COLORS["teal_dark"], "F1", marker="s")
        _hline(ax, ie.get("novelty_f1"), COLORS["ie_ref"], "IE F1")
        _vline_tau(ax)
        if row == 0:
            ax.set_title("Novelty detection", pad=6)
        ax.grid(True)
        ax.set_ylim(0.0, 1.05)
        if row == n_rows - 1:
            ax.set_xlabel(r"Cosine threshold $\tau$")

        # Col 2: compression %
        ax = axes[row, 2]
        _, vals_mc, _ = _series(ds, "model_compression")
        _, vals_bc, _ = _series(ds, "brand_compression")
        ax.plot(
            taus, [v * 100 for v in vals_mc],
            marker="o", markersize=4.5, lw=1.5,
            color=COLORS["gold_dark"], label="Model",
            markeredgecolor="white", markeredgewidth=0.5, zorder=3,
        )
        ax.plot(
            taus, [v * 100 for v in vals_bc],
            marker="s", markersize=4.5, lw=1.5,
            color=COLORS["gold_mid"], label="Brand",
            markeredgecolor="white", markeredgewidth=0.5, zorder=3,
        )
        _vline_tau(ax)
        if row == 0:
            ax.set_title("Catalog compression", pad=6)
        ax.grid(True)
        all_vals = [v * 100 for v in vals_mc + vals_bc if not np.isnan(v)]
        ymax = max(all_vals + [1.0]) * 1.22
        ax.set_ylim(-0.5, ymax)
        if row == n_rows - 1:
            ax.set_xlabel(r"Cosine threshold $\tau$")
        ax.set_ylabel("Compr. (%)", fontsize=9.5, labelpad=3)

    # Reserve a tight bottom strip for shared column legends.
    fig.tight_layout(h_pad=0.5, w_pad=1.4, rect=(0.035, 0.16, 1, 1))

    # Per-row dataset labels in the left margin.
    for row, dataset in enumerate(datasets):
        ax = axes[row, 0]
        bbox = ax.get_position()
        y_center = (bbox.y0 + bbox.y1) / 2.0
        fig.text(
            0.008, y_center, pretty.get(dataset, dataset),
            fontsize=11, fontweight="bold", color="#07060F",
            ha="left", va="center", rotation=90,
        )

    # One horizontal legend per column, hugging the x-label baseline.
    legend_kwargs = dict(
        loc="upper center",
        frameon=True, framealpha=0.92, edgecolor="#7B78A8",
        fontsize=8.5, handlelength=1.8, columnspacing=1.3,
        borderpad=0.35, handletextpad=0.5,
    )
    # Anchor legends just under the x-axis label of the bottom row.
    bottom_axes_bbox = axes[-1, 0].get_position()
    legend_y = bottom_axes_bbox.y0 - 0.105  # below the x-label
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
        description="A1 ablation: τ sensitivity for IE+IR.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS,
                        choices=DEFAULT_DATASETS)
    parser.add_argument("--tau_grid", nargs="+", type=float,
                        default=DEFAULT_TAU_GRID)
    parser.add_argument("--seed_dir", default="experiments/db")
    parser.add_argument("--dvm", default="data/dvm.csv")
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--out_json", default="data/ablation_threshold.json")
    parser.add_argument("--out_pdf",
                        default="manuscript/figures/ablation_threshold.pdf")
    parser.add_argument("--out_png",
                        default="manuscript/figures/ablation_threshold.png")
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
    print(f"τ grid: {args.tau_grid}  (n_bootstrap={args.n_bootstrap})")

    async def run_all():
        per_ds: dict = {}
        for ds in args.datasets:
            per_ds[ds] = await evaluate_dataset(
                ds, args.seed_dir, dvm_pairs, args.data_dir,
                tau_grid=args.tau_grid,
                extraction_prompt=args.extraction_prompt,
                extraction_samples=args.extraction_samples,
                n_bootstrap=args.n_bootstrap,
            )
        return per_ds

    per_ds = asyncio.run(run_all())

    report = {
        "tau_grid":   args.tau_grid,
        "chosen_tau": CHOSEN_TAU,
        "datasets":   per_ds,
    }
    with open(args.out_json, "w") as f:
        json.dump(report, f, indent=2, default=lambda o: None)
    print(f"\nwrote {args.out_json}")

    plot_report(report, args.out_pdf, args.out_png)


if __name__ == "__main__":
    main()
