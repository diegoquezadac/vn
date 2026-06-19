"""EDA for the X_*.csv files in data/.

Produces, per dataset:
  - Shape / dtypes / describe printed to stdout
  - NaN counts bar chart
  - Histograms for numerical columns
  - Top-N frequency bars for low/medium-cardinality categorical columns

And a cross-dataset comparative figure under data/eda/comparative/ where each row
is an attribute from the Vehicle model and each column is a dataset.
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as scst

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUT_DIR = DATA_DIR / "eda"
DATASETS = ["autoscout24", "mucars"]

TOP_N = 15
MAX_UNIQUE_RATIO = 0.9  # skip categoricals that are ~identifiers
MAX_AVG_TEXT_LEN = 80   # skip freeform text columns
GRID_COLS = 3

# Vibrant palette inspired by OpenAI Research plots: saturated, distinct hues
# on a neutral backdrop. Each color has a matching translucent fill.
DATASET_COLOR = {
    "autoscout24": "#0EA5E9",  # electric sky blue
    "mucars":      "#10B981",  # emerald
}
DATASET_FILL_ALPHA = 0.35
FULL_COLOR = "#D4D4D8"   # neutral zinc-300 for the full-dataset backdrop
GRID_COLOR = "#E4E4E7"


def _grid_shape(n: int, ncols: int = GRID_COLS) -> tuple[int, int]:
    nrows = max(1, math.ceil(n / ncols))
    return nrows, min(ncols, n)


def _savefig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_nans(df: pd.DataFrame, out: Path, name: str) -> None:
    counts = df.isna().sum().sort_values(ascending=False)
    pct = (counts / len(df) * 100).round(1)
    fig, ax = plt.subplots(figsize=(max(8, len(counts) * 0.35), 4.5))
    bars = ax.bar(range(len(counts)), counts.values)
    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels(counts.index, rotation=60, ha="right")
    ax.set_ylabel("NaN count")
    ax.set_title(f"{name} — NaN per column (n={len(df)})")
    for bar, p in zip(bars, pct.values):
        if bar.get_height() > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{p}%", ha="center", va="bottom", fontsize=8)
    _savefig(fig, out / "nans.png")


def plot_numeric(df: pd.DataFrame, out: Path, name: str) -> None:
    num = df.select_dtypes(include="number")
    # drop index-like columns
    num = num.loc[:, [c for c in num.columns if c.lower() not in {"unnamed: 0", "id"}]]
    if num.empty:
        return
    nrows, ncols = _grid_shape(num.shape[1])
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.2, nrows * 3.0), squeeze=False)
    for i, col in enumerate(num.columns):
        ax = axes[i // ncols][i % ncols]
        s = num[col].dropna()
        if s.empty:
            ax.set_title(f"{col} (all NaN)")
            ax.axis("off")
            continue
        ax.hist(s, bins=30, color="#4c78a8", edgecolor="white")
        ax.set_title(f"{col}\nμ={s.mean():.1f}  med={s.median():.1f}  σ={s.std():.1f}")
        ax.tick_params(axis="x", labelsize=8)
    for j in range(num.shape[1], nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")
    fig.suptitle(f"{name} — numerical distributions", y=1.02)
    _savefig(fig, out / "numeric.png")


def _categorical_columns(df: pd.DataFrame) -> list[str]:
    cols = []
    n = len(df)
    for c in df.select_dtypes(include=["object", "category"]).columns:
        s = df[c].dropna()
        if s.empty:
            continue
        if s.nunique() / max(n, 1) > MAX_UNIQUE_RATIO:
            continue
        try:
            if s.astype(str).str.len().mean() > MAX_AVG_TEXT_LEN:
                continue
        except Exception:
            continue
        cols.append(c)
    return cols


def plot_categorical(df: pd.DataFrame, out: Path, name: str) -> None:
    cols = _categorical_columns(df)
    if not cols:
        return
    nrows, ncols = _grid_shape(len(cols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.2, nrows * 3.4), squeeze=False)
    for i, col in enumerate(cols):
        ax = axes[i // ncols][i % ncols]
        vc = df[col].astype(str).value_counts().head(TOP_N)[::-1]
        ax.barh(vc.index, vc.values, color="#59a14f")
        uniq = df[col].nunique(dropna=True)
        ax.set_title(f"{col}  (unique={uniq})")
        ax.tick_params(axis="y", labelsize=8)
    for j in range(len(cols), nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")
    fig.suptitle(f"{name} — top-{TOP_N} categorical values", y=1.02)
    _savefig(fig, out / "categorical.png")


def summarise(df: pd.DataFrame, name: str) -> None:
    print(f"\n=== {name} ===")
    print(f"shape: {df.shape}")
    print("\ndtypes:")
    print(df.dtypes.to_string())
    nans = df.isna().sum()
    nans = nans[nans > 0].sort_values(ascending=False)
    print("\nNaN counts (non-zero):")
    print(nans.to_string() if not nans.empty else "  (none)")
    num = df.select_dtypes(include="number")
    if not num.empty:
        print("\nnumerical describe:")
        print(num.describe().round(2).to_string())


# --- Cross-dataset comparative figure (Vehicle-model attributes) -------------

def _parse_mucars_mileage(s: pd.Series) -> pd.Series:
    """mucars Mileage is a km range like '200 000 - 249 999' -> use midpoint."""
    def midpoint(v: object) -> float | None:
        if not isinstance(v, str):
            return None
        nums = re.findall(r"\d[\d\s]*", v)
        if not nums:
            return None
        try:
            vals = [int(n.replace(" ", "")) for n in nums]
        except ValueError:
            return None
        return float(sum(vals) / len(vals))
    return s.map(midpoint)


def _lower(s: pd.Series) -> pd.Series:
    return s.dropna().astype(str).str.strip().str.lower()


# Series extractors: each returns a clean pandas Series in a common unit / casing.
def _series_brand(df: pd.DataFrame, name: str) -> pd.Series:
    col = {"autoscout24": "brand", "mucars": "Brand"}[name]
    return _lower(df[col])


def _series_model(df: pd.DataFrame, name: str) -> pd.Series:
    col = {"autoscout24": "model", "mucars": "Model"}[name]
    return _lower(df[col])


def _series_year(df: pd.DataFrame, name: str) -> pd.Series:
    col = {"autoscout24": "year", "mucars": "Year"}[name]
    return pd.to_numeric(df[col], errors="coerce").dropna()


def _series_mileage_km(df: pd.DataFrame, name: str) -> pd.Series:
    if name == "autoscout24":
        return pd.to_numeric(df["mileage_in_km"], errors="coerce").dropna()
    return _parse_mucars_mileage(df["Mileage"]).dropna()


def _series_fuel(df: pd.DataFrame, name: str) -> pd.Series:
    col = {"autoscout24": "fuel_type", "mucars": "Fuel"}[name]
    return _lower(df[col])


def _series_transmission(df: pd.DataFrame, name: str) -> pd.Series:
    col = {"autoscout24": "transmission_type", "mucars": "Gearbox"}[name]
    return _lower(df[col])


# kind: "hist" (numeric) or "bar" (categorical top-N)
# xrange: common x-range for numeric attrs (None = let matplotlib pick)
def _series_mileage_k_km(df: pd.DataFrame, name: str) -> pd.Series:
    return _series_mileage_km(df, name) / 1000.0


ATTR_SPECS = [
    {"label": "Brand",            "kind": "bar",  "top": 10, "extract": _series_brand},
    {"label": "Model",            "kind": "bar",  "top": 10, "extract": _series_model},
    {"label": "Year",             "kind": "hist", "bins": 30, "xrange": (1980, 2025), "extract": _series_year},
    {"label": "Mileage (10³ km)", "kind": "hist", "bins": 30, "xrange": (0, 400), "extract": _series_mileage_k_km},
    {"label": "Fuel type",        "kind": "bar",  "top": 8,  "extract": _series_fuel},
    {"label": "Transmission",     "kind": "bar",  "top": 6,  "extract": _series_transmission},
]


def _truncate(s: str, n: int = 20) -> str:
    return s if len(s) <= n else s[: n - 1] + "…"


def plot_comparative(frames: dict[str, pd.DataFrame], out: Path) -> None:
    datasets = [d for d in DATASETS if d in frames]
    if not datasets:
        return
    nrows = len(ATTR_SPECS)
    ncols = len(datasets)

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.6, nrows * 2.2),
                             squeeze=False)

    for i, spec in enumerate(ATTR_SPECS):
        for j, name in enumerate(datasets):
            ax = axes[i][j]
            color = DATASET_COLOR[name]
            try:
                s = spec["extract"](frames[name], name)
            except KeyError:
                s = pd.Series(dtype=float)

            if s.empty:
                ax.text(0.5, 0.5, "no data", ha="center", va="center",
                        transform=ax.transAxes, fontsize=10, color="#999")
                ax.set_xticks([]); ax.set_yticks([])
            elif spec["kind"] == "hist":
                xr = spec.get("xrange")
                data = s if xr is None else s[(s >= xr[0]) & (s <= xr[1])]
                ax.hist(data, bins=spec["bins"], range=xr, color=color, edgecolor="white")
                if xr is not None:
                    ax.set_xlim(*xr)
                med = float(np.median(s))
                ax.axvline(med, color="black", linestyle="--", linewidth=0.8, alpha=0.6)
                ax.text(0.97, 0.92, f"n={len(s)}\nmed={med:.0f}",
                        transform=ax.transAxes, ha="right", va="top",
                        fontsize=7, color="#333")
            else:  # bar
                vc = s.value_counts().head(spec["top"])[::-1]
                labels = [_truncate(str(x)) for x in vc.index]
                ax.barh(labels, vc.values, color=color)
                uniq = s.nunique()
                ax.text(0.97, 0.05, f"unique={uniq}", transform=ax.transAxes,
                        ha="right", va="bottom", fontsize=7, color="#333")

            if i == 0:
                ax.set_title(name, fontsize=11, pad=6)
            if i == nrows - 1:
                ax.set_xlabel("count" if spec["kind"] == "bar" else spec["label"])

    # Row labels: place once per row, to the left of the first column.
    for i, spec in enumerate(ATTR_SPECS):
        ax = axes[i][0]
        # Measure bbox in figure coords to position label outside any y-tick labels.
        fig.canvas.draw()
        bbox = ax.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.transFigure.inverted())
        y_center = (bbox.y0 + bbox.y1) / 2
        fig.text(max(bbox.x0 - 0.012, 0.005), y_center, spec["label"],
                 ha="right", va="center", rotation=90, fontweight="bold", fontsize=11)

    fig.suptitle(f"Cross-dataset attribute distributions (raw sample, n={len(next(iter(frames.values())))} each)",
                 y=1.00, fontsize=12)
    out.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0.03, 0, 1, 0.99))
    fig.savefig(out / "attributes.png", dpi=150, bbox_inches="tight")
    fig.savefig(out / "attributes.pdf", bbox_inches="tight")
    plt.close(fig)
    plt.rcdefaults()


# --- Sample vs. full-dataset comparison --------------------------------------

# Per-dataset columns needed by the ATTR_SPECS extractors (speeds up full-CSV load).
FULL_USECOLS = {
    "autoscout24": ["brand", "model", "year", "mileage_in_km", "fuel_type", "transmission_type"],
    "mucars":     ["Brand", "Model", "Year", "Mileage", "Fuel", "Gearbox"],
}


def _load_full(name: str) -> pd.DataFrame | None:
    csv = DATA_DIR / f"{name}.csv"
    if not csv.exists():
        return None
    return pd.read_csv(csv, usecols=FULL_USECOLS[name], low_memory=False)


def _tvd(p_sample: pd.Series, p_full: pd.Series) -> float:
    """Total variation distance over the union of categories (proportions)."""
    cats = p_sample.index.union(p_full.index)
    a = p_sample.reindex(cats, fill_value=0.0).astype(float)
    b = p_full.reindex(cats, fill_value=0.0).astype(float)
    return float(0.5 * (a - b).abs().sum())


ACADEMIC_RC = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 10,
    "axes.titlesize": 10.5,
    "axes.labelsize": 9.5,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.7,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 3.5,
    "ytick.major.size": 3.5,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "legend.frameon": False,
    "legend.fontsize": 9,
    "axes.axisbelow": True,  # grid behind data
}


def plot_sample_vs_full(frames_sample: dict[str, pd.DataFrame],
                        frames_full: dict[str, pd.DataFrame],
                        out: Path) -> None:
    datasets = [d for d in DATASETS if d in frames_sample and d in frames_full]
    if not datasets:
        return
    nrows = len(ATTR_SPECS)
    ncols = len(datasets)

    plt.rcParams.update(ACADEMIC_RC)

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.8, nrows * 2.3),
                             squeeze=False)

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=FULL_COLOR, edgecolor="none", label="Full"),
        plt.Line2D([0], [0], color="#333", lw=1.6, label="Sample"),
    ]

    drift: dict[str, dict[str, float]] = {}

    for i, spec in enumerate(ATTR_SPECS):
        drift[spec["label"]] = {}
        for j, name in enumerate(datasets):
            ax = axes[i][j]
            color = DATASET_COLOR[name]
            s_sample = spec["extract"](frames_sample[name], name)
            s_full = spec["extract"](frames_full[name], name)

            if s_sample.empty or s_full.empty:
                ax.text(0.5, 0.5, "no data", ha="center", va="center",
                        transform=ax.transAxes, fontsize=10, color="#999")
                ax.set_xticks([]); ax.set_yticks([])
            elif spec["kind"] == "hist":
                xr = spec.get("xrange")
                def _clip(s: pd.Series) -> pd.Series:
                    return s if xr is None else s[(s >= xr[0]) & (s <= xr[1])]
                ax.hist(_clip(s_full), bins=spec["bins"], range=xr, density=True,
                        color=FULL_COLOR, edgecolor="none", label="full", zorder=1)
                # Sample: translucent fill + solid outline on top (OpenAI-style)
                ax.hist(_clip(s_sample), bins=spec["bins"], range=xr, density=True,
                        color=color, alpha=DATASET_FILL_ALPHA, edgecolor="none",
                        zorder=2)
                ax.hist(_clip(s_sample), bins=spec["bins"], range=xr, density=True,
                        histtype="step", color=color, linewidth=1.8, label="sample",
                        zorder=3)
                if xr is not None:
                    ax.set_xlim(*xr)
                ax.grid(axis="y", color=GRID_COLOR, linestyle="--", linewidth=0.6, zorder=0)
                ks = float(scst.ks_2samp(s_sample.values, s_full.values).statistic)
                drift[spec["label"]][name] = ks
                ax.text(0.97, 0.93,
                        f"KS = {ks:.02f}\n$n_s$ = {len(s_sample)}   $n_f$ = {len(s_full):,}",
                        transform=ax.transAxes, ha="right", va="top",
                        fontsize=7.5, color="#222",
                        bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                                  edgecolor="#cccccc", linewidth=0.5, alpha=0.9))
            else:  # bar
                vc_full = s_full.value_counts()
                vc_sample = s_sample.value_counts()
                top = vc_full.head(spec["top"]).index.tolist()[::-1]
                p_full_all = vc_full / vc_full.sum()
                p_sample_all = vc_sample / max(vc_sample.sum(), 1)
                y = np.arange(len(top))
                h = 0.4
                full_prop = [p_full_all.get(k, 0.0) for k in top]
                samp_prop = [p_sample_all.get(k, 0.0) for k in top]
                ax.barh(y - h / 2, full_prop, height=h, color=FULL_COLOR,
                        edgecolor="none", label="full", zorder=2)
                ax.barh(y + h / 2, samp_prop, height=h, color=color,
                        edgecolor="none", label="sample", zorder=2)
                ax.set_yticks(y)
                ax.set_yticklabels([_truncate(str(k)) for k in top])
                ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x*100:.0f}%"))
                tvd = _tvd(p_sample_all, p_full_all)
                drift[spec["label"]][name] = tvd
                ax.text(0.97, 0.05, f"TVD = {tvd:.02f}", transform=ax.transAxes,
                        ha="right", va="bottom", fontsize=7.5, color="#222",
                        bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                                  edgecolor="#cccccc", linewidth=0.5, alpha=0.9))

            if i == 0:
                ax.set_title(name, fontsize=11, pad=6)
            if i == nrows - 1:
                ax.set_xlabel("proportion" if spec["kind"] == "bar" else spec["label"])

    for i, spec in enumerate(ATTR_SPECS):
        ax = axes[i][0]
        fig.canvas.draw()
        bbox = ax.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.transFigure.inverted())
        y_center = (bbox.y0 + bbox.y1) / 2
        fig.text(max(bbox.x0 - 0.012, 0.005), y_center, spec["label"],
                 ha="right", va="center", rotation=90, fontweight="bold", fontsize=11)

    out.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0.03, 0.03, 1, 1))
    fig.legend(handles=legend_handles, loc="lower center",
               bbox_to_anchor=(0.5, -0.005), ncol=2, handlelength=1.8)
    fig.savefig(out / "sample_vs_full.png", dpi=200, bbox_inches="tight")
    fig.savefig(out / "sample_vs_full.pdf", bbox_inches="tight")
    plt.close(fig)
    plt.rcdefaults()

    _write_caption(out / "caption.txt", drift, frames_sample, frames_full)


def _write_caption(path: Path,
                   drift: dict[str, dict[str, float]],
                   frames_sample: dict[str, pd.DataFrame],
                   frames_full: dict[str, pd.DataFrame]) -> None:
    datasets = list(frames_sample)
    sizes = ", ".join(f"{d} ($n_f$={len(frames_full[d]):,})" for d in datasets)
    n_sample = len(next(iter(frames_sample.values())))

    def _row(label: str) -> str:
        vals = drift.get(label, {})
        parts = [f"{d}: {vals[d]:.02f}" for d in datasets if d in vals]
        return "; ".join(parts)

    n_corpora = {1: "one", 2: "two", 3: "three", 4: "four"}.get(len(datasets), str(len(datasets)))
    caption = (
        f"Figure X. Sample versus full-dataset distributions across the {n_corpora} "
        "automotive corpora: " + sizes + f". Each row corresponds to one "
        "attribute of the Vehicle model (brand, model, year, mileage, fuel "
        "type and transmission) and each column corresponds to one dataset. "
        f"The colored outline (or darker bar) denotes the random sample of "
        f"$n_s$={n_sample} vehicles per dataset used throughout this paper; "
        "the grey fill denotes the full dataset. Numeric attributes are "
        "plotted as density-normalized histograms on shared support and "
        "annotated with the two-sample Kolmogorov--Smirnov statistic "
        "($\\mathrm{KS}\\in[0,1]$, lower indicates closer agreement). "
        "Categorical attributes display the proportions of the top-10 most "
        "frequent values in the full dataset, compared against their "
        "proportions in the sample; drift is quantified by the total "
        "variation distance (TVD) computed over the union of sample and full "
        "vocabularies. Across all datasets the sample reproduces the full "
        "distribution closely for brand (TVD " + _row("Brand") + "), year "
        "(KS " + _row("Year") + "), mileage (KS " + _row("Mileage (10³ km)") +
        "), fuel type (TVD " + _row("Fuel type") + ") and transmission "
        "(TVD " + _row("Transmission") + "). Higher TVD values for model "
        "(" + _row("Model") + ") reflect the long-tailed, unnormalized "
        "vocabulary of the raw corpora, which motivates the normalization "
        "procedure introduced in this work."
    )
    path.write_text(caption + "\n", encoding="utf-8")


def run(datasets: list[str], skip_full: bool = False) -> None:
    frames: dict[str, pd.DataFrame] = {}
    for name in datasets:
        csv = DATA_DIR / f"X_{name}.csv"
        if not csv.exists():
            print(f"[skip] {csv} not found")
            continue
        df = pd.read_csv(csv)
        frames[name] = df
        summarise(df, name)
        out = OUT_DIR / name
        plot_nans(df, out, name)
        plot_numeric(df, out, name)
        plot_categorical(df, out, name)
        print(f"figures -> {out}")

    if len(frames) >= 2:
        comp_out = OUT_DIR / "comparative"
        plot_comparative(frames, comp_out)
        print(f"comparative figure -> {comp_out}")

        if not skip_full:
            full_frames: dict[str, pd.DataFrame] = {}
            for name in frames:
                full = _load_full(name)
                if full is not None:
                    full_frames[name] = full
                    print(f"loaded full {name}: {full.shape[0]:,} rows")
                else:
                    print(f"[skip] {name}.csv not found — no sample vs full for {name}")
            if full_frames:
                plot_sample_vs_full(frames, full_frames, comp_out)
                print(f"sample vs full figure -> {comp_out}/sample_vs_full.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=DATASETS, choices=DATASETS)
    parser.add_argument("--skip-full", action="store_true",
                        help="Skip the sample-vs-full comparison (avoids loading full CSVs)")
    args = parser.parse_args()
    run(args.datasets, skip_full=args.skip_full)


if __name__ == "__main__":
    main()
