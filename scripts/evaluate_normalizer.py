"""
evaluate_normalizer.py — end-to-end evaluation of the VN pipeline against ground truth.

For each of data/{dataset}_sample.csv + data/{dataset}_ground_truth.csv we run:

  (1) Intrinsic accuracy per attribute:
        - IE-only baseline  vs ground truth
        - Full VN pipeline  vs ground truth
      Δ shows the canonicalization contribution.

  (2) Canonicalization ratio for brand and model:
        #distinct IE strings / #distinct VN canonicals.
      High ratio ↔ strong string-variance collapse.

  (3) Novelty detection (pair-level):
        gold: (GT.brand, GT.model) ∉ DVM-CAR pairs
        pred: (VN.brand, VN.model) ∉ DVM-CAR pairs
      Report precision, recall, F1.

Each dataset runs against a fresh copy of the seed catalog so results do
not depend on dataset order. The IE step runs once; the full-VN pass reuses
those extractions via Normalizer(..., extractions=...).

Outputs:
  data/{dataset}_normalizer_report.json
  printed summary per dataset.

Usage:
    uv run python scripts/catalog.py --out_dir experiments/db   # one-off seed
    uv run python scripts/evaluate_normalizer.py
    uv run python scripts/evaluate_normalizer.py --datasets autoscout24
"""

import argparse
import asyncio
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.normalizer import Normalizer, serialize_row  # noqa: E402

load_dotenv()

DATASETS = ["autoscout24", "mucars"]
MAX_VALUE_CHARS = 2000

NULL_STRINGS = {
    "", "nan", "none", "n/a", "na", "null", "unknown",
    "unspecified", "not specified", "missing",
}
ZERO_IS_NULL_FIELDS = {
    "engine_power", "engine_size", "cylinders", "doors", "fuel_consumption",
}


def clip_row(row: pd.Series, max_chars: int = MAX_VALUE_CHARS) -> pd.Series:
    clipped = row.copy()
    for col, v in row.items():
        if isinstance(v, str) and len(v) > max_chars:
            clipped[col] = v[:max_chars]
    return clipped


def normalize_value(v) -> str:
    """Null-like → ''; numeric strings → int-ish canonical form."""
    if v is None:
        return ""
    if isinstance(v, float) and np.isnan(v):
        return ""
    s = str(v).strip().lower()
    if s in NULL_STRINGS:
        return ""
    try:
        f = float(s)
    except (ValueError, TypeError):
        return s
    if np.isnan(f):
        return ""
    return str(int(f)) if f.is_integer() else repr(f)


def parse_equipment(v) -> set:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return set()
    if isinstance(v, list):
        return {str(x).strip().lower() for x in v}
    try:
        x = json.loads(v) if isinstance(v, str) else v
        return {str(i).strip().lower() for i in x} if isinstance(x, list) else set()
    except (json.JSONDecodeError, TypeError):
        return set()


def load_dvm_pairs(dvm_path: str) -> tuple[set[str], set[tuple[str, str]]]:
    df = pd.read_csv(dvm_path)
    brands: set[str] = set()
    pairs: set[tuple[str, str]] = set()
    for b, m in zip(df["Automaker"], df["Genmodel"]):
        if pd.isna(b) or pd.isna(m):
            continue
        bn, mn = str(b).strip().lower(), str(m).strip().lower()
        if bn and mn:
            brands.add(bn)
            pairs.add((bn, mn))
    return brands, pairs


def attribute_accuracy(preds: list, truths: list, attr: str) -> dict:
    if attr == "equipment":
        jac = []
        for p, t in zip(preds, truths):
            if isinstance(t, str) and t.strip().upper() == "ABSTAIN":
                continue
            ps, ts = parse_equipment(p), parse_equipment(t)
            u = ps | ts
            jac.append(len(ps & ts) / len(u) if u else 1.0)
        return {"mean_jaccard": float(np.mean(jac)) if jac else float("nan"), "n": len(jac)}

    correct = total = gt_nn = pred_nn = both_nn_correct = 0
    for p, t in zip(preds, truths):
        if isinstance(t, str) and t.strip().upper() == "ABSTAIN":
            continue
        ps, ts = normalize_value(p), normalize_value(t)
        if attr in ZERO_IS_NULL_FIELDS:
            ps = "" if ps == "0" else ps
            ts = "" if ts == "0" else ts
        total += 1
        if ps == ts:
            correct += 1
            if ts != "":
                both_nn_correct += 1
        if ts != "":
            gt_nn += 1
        if ps != "":
            pred_nn += 1
    return {
        "accuracy": correct / total if total else float("nan"),
        "n": total,
        "gt_non_null": gt_nn,
        "pred_non_null": pred_nn,
        "both_non_null_correct": both_nn_correct,
    }


def macro_prf(preds: list, truths: list) -> dict:
    """Macro-averaged precision / recall / F1 over classes seen in either
    preds or truths. Rows with null truth or ABSTAIN are skipped."""
    from collections import Counter
    pairs = []
    for p, t in zip(preds, truths):
        if isinstance(t, str) and t.strip().upper() == "ABSTAIN":
            continue
        ps, ts = normalize_value(p), normalize_value(t)
        if ts == "":
            continue
        pairs.append((ps, ts))
    if not pairs:
        return {"precision": float("nan"), "recall": float("nan"), "f1": float("nan"), "n_classes": 0}
    classes = {t for _, t in pairs} | {p for p, _ in pairs if p}
    ps_, rs_, fs_ = [], [], []
    for c in classes:
        tp = sum(1 for p, t in pairs if p == c and t == c)
        fp = sum(1 for p, t in pairs if p == c and t != c)
        fn = sum(1 for p, t in pairs if p != c and t == c)
        if tp + fn == 0:
            continue  # class never appears in truth
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn)
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        ps_.append(prec); rs_.append(rec); fs_.append(f1)
    return {
        "precision": float(np.mean(ps_)) if ps_ else float("nan"),
        "recall": float(np.mean(rs_)) if rs_ else float("nan"),
        "f1": float(np.mean(fs_)) if fs_ else float("nan"),
        "n_classes": len(ps_),
        "n": len(pairs),
    }


def error_taxonomy(ie_vals: list, vn_vals: list, truths: list) -> dict:
    """Decompose VN behaviour wrt IE and GT.

    For rows where GT is non-null:
      correct       : VN == GT
      under_merge   : VN == IE != GT            (VN failed to canonicalize)
      bad_merge     : VN != IE == GT            (VN broke a correct IE output)
      substitution  : VN != IE, VN != GT, IE != GT  (VN changed value, still wrong)
      missing       : VN is null but GT is not
      spurious      : VN non-null but IE was null (caused by VN step, rare)
    """
    buckets = {
        "correct": 0, "under_merge": 0, "bad_merge": 0,
        "substitution": 0, "missing": 0, "spurious": 0,
    }
    examples = {k: [] for k in ["under_merge", "bad_merge", "substitution"]}
    n = 0
    for ie, vn, gt in zip(ie_vals, vn_vals, truths):
        if isinstance(gt, str) and gt.strip().upper() == "ABSTAIN":
            continue
        ie_n, vn_n, gt_n = normalize_value(ie), normalize_value(vn), normalize_value(gt)
        if gt_n == "":
            continue
        n += 1
        if vn_n == gt_n:
            buckets["correct"] += 1
            continue
        if vn_n == "":
            buckets["missing"] += 1
            continue
        if ie_n == "" and vn_n != "":
            buckets["spurious"] += 1
            continue
        if vn_n == ie_n and ie_n != gt_n:
            buckets["under_merge"] += 1
            if len(examples["under_merge"]) < 10:
                examples["under_merge"].append({"ie": ie_n, "vn": vn_n, "gt": gt_n})
        elif vn_n != ie_n and ie_n == gt_n:
            buckets["bad_merge"] += 1
            if len(examples["bad_merge"]) < 10:
                examples["bad_merge"].append({"ie": ie_n, "vn": vn_n, "gt": gt_n})
        else:
            buckets["substitution"] += 1
            if len(examples["substitution"]) < 10:
                examples["substitution"].append({"ie": ie_n, "vn": vn_n, "gt": gt_n})
    return {"n": n, "counts": buckets, "examples": examples}


def canonicalization_ratio(ie_vals: list, vn_vals: list) -> dict:
    ie_clean = [normalize_value(v) for v in ie_vals if normalize_value(v)]
    vn_clean = [normalize_value(v) for v in vn_vals if normalize_value(v)]
    ie_unique, vn_unique = set(ie_clean), set(vn_clean)
    return {
        "n_listings_non_null": len(ie_clean),
        "ie_unique": len(ie_unique),
        "vn_unique": len(vn_unique),
        "ratio": (len(ie_unique) / len(vn_unique)) if vn_unique else float("nan"),
    }


def merge_examples(ie_vals: list, vn_vals: list, k: int = 10) -> list:
    buckets: dict[str, set] = {}
    for ie, vn in zip(ie_vals, vn_vals):
        ie_c, vn_c = normalize_value(ie), normalize_value(vn)
        if ie_c and vn_c:
            buckets.setdefault(vn_c, set()).add(ie_c)
    merges = [(vn, sorted(vs)) for vn, vs in buckets.items() if len(vs) > 1]
    merges.sort(key=lambda x: -len(x[1]))
    return merges[:k]


def bootstrap_ci(
    compute,
    preds: list,
    truths: list,
    n_resamples: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """Percentile bootstrap CI for a scalar metric computed over (preds, truths).

    `compute(preds, truths) -> float` is called on each resample; NaNs are
    dropped. Returns (lo, hi) as the `ci`-level percentile interval.
    """
    rng = np.random.RandomState(seed)
    n = len(preds)
    if n == 0:
        return float("nan"), float("nan")
    samples: list[float] = []
    preds_arr = list(preds)
    truths_arr = list(truths)
    for _ in range(n_resamples):
        idx = rng.randint(0, n, size=n)
        p = [preds_arr[i] for i in idx]
        t = [truths_arr[i] for i in idx]
        v = compute(p, t)
        if v is not None and not (isinstance(v, float) and np.isnan(v)):
            samples.append(float(v))
    if not samples:
        return float("nan"), float("nan")
    alpha = (1.0 - ci) / 2.0
    lo = float(np.percentile(samples, 100 * alpha))
    hi = float(np.percentile(samples, 100 * (1 - alpha)))
    return lo, hi


def novelty_metrics(preds: list, truths: list, dvm_pairs: set) -> dict:
    tp = fp = fn = tn = 0
    for p, t in zip(preds, truths):
        p_novel = p not in dvm_pairs
        t_novel = t not in dvm_pairs
        if t_novel and p_novel: tp += 1
        elif (not t_novel) and p_novel: fp += 1
        elif t_novel and (not p_novel): fn += 1
        else: tn += 1
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    rec = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else float("nan")
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": prec, "recall": rec, "f1": f1,
    }


def _fresh_catalog(seed_dir: str, dataset: str) -> str:
    tmp_dir = tempfile.mkdtemp(prefix=f"vn_eval_{dataset}_")
    for fname in os.listdir(seed_dir):
        src = os.path.join(seed_dir, fname)
        if os.path.isfile(src):
            shutil.copy(src, tmp_dir)
    return tmp_dir


async def _run_variant(
    mode: str,
    listings: list,
    ie_results: list,
    seed_dir: str,
    dataset: str,
    extraction_prompt: str,
    extraction_samples: str,
    match_threshold: float,
) -> list:
    tmp_dir = _fresh_catalog(seed_dir, f"{dataset}_{mode}")
    try:
        norm = Normalizer(
            match_mode=mode,
            match_threshold=match_threshold,
            persist_directory=tmp_dir,
            extraction_prompt=extraction_prompt,
            extraction_samples=extraction_samples,
        )
        results, _ = await norm(listings, extractions=ie_results)
        return results
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _deltas_vs_ie(acc_ie: dict, acc_v: dict, prf_ie: dict, prf_v: dict) -> dict:
    out: dict = {}
    for a in ["brand", "model"]:
        ai, av = acc_ie[a]["accuracy"], acc_v[a]["accuracy"]
        pi, pv = prf_ie[a], prf_v[a]
        out[a] = {
            "acc_ie": ai, "acc_vn": av, "delta_acc": av - ai,
            "precision_ie": pi["precision"], "precision_vn": pv["precision"],
            "delta_precision": pv["precision"] - pi["precision"],
            "recall_ie":    pi["recall"],    "recall_vn":    pv["recall"],
            "delta_recall":    pv["recall"]    - pi["recall"],
            "f1_ie":        pi["f1"],        "f1_vn":        pv["f1"],
            "delta_f1":        pv["f1"]        - pi["f1"],
        }
    return out


async def evaluate_dataset(
    dataset: str,
    seed_dir: str,
    dvm_pairs: set,
    data_dir: str,
    variants: list[str],
    match_threshold: float = 0.9,
    extraction_prompt: str = "extraction.j2",
    extraction_samples: str = "./samples/extraction.json",
) -> dict:
    print(f"\n=== {dataset} ===")
    sample_df = pd.read_csv(f"{data_dir}/{dataset}_sample.csv")
    gt_df = pd.read_csv(f"{data_dir}/{dataset}_ground_truth.csv").head(len(sample_df))

    listings = [serialize_row(clip_row(r)) for _, r in sample_df.iterrows()]
    print(f"  n={len(listings)} | variants={variants} | τ={match_threshold}")

    # IE once, reuse for all downstream variants.
    print("  [1] IE-only pass …")
    ie_norm = Normalizer(
        match_mode="off",
        extraction_prompt=extraction_prompt,
        extraction_samples=extraction_samples,
    )
    ie_results, _ = await ie_norm(listings)

    variant_results: dict[str, list] = {"ie": ie_results}
    for v in variants:
        if v == "ie":
            continue
        print(f"  [{v}] full pass (reusing IE extractions) …")
        variant_results[v] = await _run_variant(
            mode=v, listings=listings, ie_results=ie_results,
            seed_dir=seed_dir, dataset=dataset,
            extraction_prompt=extraction_prompt,
            extraction_samples=extraction_samples,
            match_threshold=match_threshold,
        )

    attrs = list(gt_df.columns)

    # (1) intrinsic accuracy per variant (+ bootstrap CI for brand, model)
    intrinsic: dict[str, dict] = {}
    for v, results in variant_results.items():
        intrinsic[v] = {}
        for a in attrs:
            truths = gt_df[a].tolist()
            preds = [r.get(a) for r in results]
            intrinsic[v][a] = attribute_accuracy(preds, truths, a)
            if a in ("brand", "model"):
                lo, hi = bootstrap_ci(
                    lambda p, t, attr=a: attribute_accuracy(p, t, attr).get(
                        "accuracy", float("nan")
                    ),
                    preds, truths,
                )
                intrinsic[v][a]["accuracy_ci"] = [lo, hi]

    key = lambda a, d: d[a].get("mean_jaccard" if a == "equipment" else "accuracy", float("nan"))
    print(f"\n  {'attribute':<26s}  " + "  ".join(f"{v.upper():>8s}" for v in variant_results))
    for a in attrs:
        cells = []
        for v in variant_results:
            cells.append(f"{key(a, intrinsic[v]):8.3f}")
        print(f"  {a:<26s}  " + "  ".join(cells))

    # (2) macro P/R/F1 and error taxonomy (brand, model) per variant
    prf: dict[str, dict] = {}
    taxonomy: dict[str, dict] = {}
    for v, results in variant_results.items():
        prf[v] = {}
        for a in ["brand", "model"]:
            truths = gt_df[a].tolist()
            vals = [r.get(a) for r in results]
            prf[v][a] = macro_prf(vals, truths)
            lo, hi = bootstrap_ci(
                lambda p, t: macro_prf(p, t).get("f1", float("nan")),
                vals, truths,
            )
            prf[v][a]["f1_ci"] = [lo, hi]
        if v != "ie":
            taxonomy[v] = {}
            for a in ["brand", "model"]:
                truths = gt_df[a].tolist()
                ie_vals = [r.get(a) for r in ie_results]
                vn_vals = [r.get(a) for r in results]
                taxonomy[v][a] = error_taxonomy(ie_vals, vn_vals, truths)

    print(f"\n  macro F1 (brand, model):")
    print(f"    {'attr':<6s}  " + "  ".join(f"{v.upper():>7s}" for v in variant_results))
    for a in ["brand", "model"]:
        cells = [f"{prf[v][a]['f1']:7.3f}" for v in variant_results]
        print(f"    {a:<6s}  " + "  ".join(cells))

    # (3) canonicalization ratio and merge examples per non-IE variant
    canon: dict[str, dict] = {}
    examples: dict[str, dict] = {}
    for v, results in variant_results.items():
        if v == "ie":
            continue
        canon[v], examples[v] = {}, {}
        for a in ["brand", "model"]:
            ie_vals = [r.get(a) for r in ie_results]
            vn_vals = [r.get(a) for r in results]
            canon[v][a] = canonicalization_ratio(ie_vals, vn_vals)
            examples[v][a] = merge_examples(ie_vals, vn_vals)

    print(f"\n  canonicalization ratio (ie_unique → vn_unique):")
    for v in canon:
        for a in ["brand", "model"]:
            r = canon[v][a]
            print(f"    {v:>9s} {a:6s}: {r['ie_unique']:4d} → {r['vn_unique']:4d}   (ratio {r['ratio']:.2f}x)")

    # (4) novelty (pair-level) per non-IE variant
    def to_pair(b, m):
        bn, mn = normalize_value(b), normalize_value(m)
        if bn == "abstain" or mn == "abstain":
            return ("", "")
        return (bn, mn) if bn and mn else ("", "")

    gt_pairs = [to_pair(gt_df["brand"].iloc[i], gt_df["model"].iloc[i]) for i in range(len(gt_df))]
    novelty: dict[str, dict] = {}
    novelty_pairs: dict[str, dict] = {}  # persisted so future bootstrap is post-hoc
    for v, results in variant_results.items():
        vn_pairs = [to_pair(r.get("brand"), r.get("model")) for r in results]
        filt = [(p, t) for p, t in zip(vn_pairs, gt_pairs) if t != ("", "")]
        preds_v = [p for p, _ in filt]
        truths_v = [t for _, t in filt]
        novelty[v] = novelty_metrics(preds_v, truths_v, dvm_pairs)
        for metric in ("precision", "recall", "f1"):
            lo, hi = bootstrap_ci(
                lambda p, t, m=metric: novelty_metrics(p, t, dvm_pairs).get(m, float("nan")),
                preds_v, truths_v,
            )
            novelty[v][f"{metric}_ci"] = [lo, hi]
        novelty_pairs[v] = {
            "preds": [list(p) for p in preds_v],
            "truths": [list(t) for t in truths_v],
        }

    print(f"\n  novelty detection (pair-level, IE vs DVM-CAR raw; threshold/llm after canonicalization):")
    for v in variant_results:
        if v not in novelty:
            continue
        n = novelty[v]
        print(f"    {v:>9s}: P={n['precision']:.3f}  R={n['recall']:.3f}  F1={n['f1']:.3f}  "
              f"TP={n['tp']} FP={n['fp']} FN={n['fn']} TN={n['tn']}")

    # (5) deltas vs IE per non-IE variant
    deltas: dict[str, dict] = {}
    for v in variant_results:
        if v == "ie":
            continue
        deltas[v] = _deltas_vs_ie(intrinsic["ie"], intrinsic[v], prf["ie"], prf[v])

    return {
        "dataset": dataset,
        "n_listings": len(listings),
        "match_threshold": match_threshold,
        "variants": list(variant_results.keys()),
        "intrinsic": intrinsic,
        "macro_prf": prf,
        "deltas": deltas,
        "error_taxonomy": taxonomy,
        "canonicalization": canon,
        "merge_examples": {
            v: {k: [(c, list(vs)) for c, vs in ex] for k, ex in examples[v].items()}
            for v in examples
        },
        "novelty": novelty,
        "novelty_pairs": novelty_pairs,
    }


def _clean(obj):
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(x) for x in obj]
    if isinstance(obj, float) and np.isnan(obj):
        return None
    return obj


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the VN pipeline against ground truth.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--datasets", nargs="+", default=DATASETS, choices=DATASETS)
    parser.add_argument("--seed_dir", default="experiments/db")
    parser.add_argument("--dvm", default="data/dvm.csv")
    parser.add_argument("--data_dir", default="data")
    parser.add_argument(
        "--extraction_prompt", default="extraction.j2",
        help="Jinja2 template name under ./prompts/ (e.g. extraction.j2, extraction_strict.j2)",
    )
    parser.add_argument(
        "--extraction_samples", default="./samples/extraction.json",
        help="Path to few-shot examples JSON to render into the prompt",
    )
    parser.add_argument(
        "--report_suffix", default="",
        help="Suffix appended to report filename: {dataset}_normalizer_report{suffix}.json",
    )
    parser.add_argument(
        "--variants", nargs="+", default=["ie", "threshold", "llm"],
        choices=["ie", "threshold", "llm"],
        help="Which variants to compare. 'ie' always runs; others reuse its extractions.",
    )
    parser.add_argument(
        "--match_threshold", type=float, default=0.9,
        help="Cosine-similarity cutoff for the 'threshold' variant (and its intra-batch dedup).",
    )
    args = parser.parse_args()
    if "ie" not in args.variants:
        args.variants = ["ie"] + list(args.variants)

    if not os.path.exists(os.path.join(args.seed_dir, "catalog.db")):
        print(f"[error] seed catalog not found at {args.seed_dir}/catalog.db")
        print(f"        run: uv run python scripts/catalog.py --out_dir {args.seed_dir}")
        sys.exit(1)

    _, dvm_pairs = load_dvm_pairs(args.dvm)
    print(f"DVM-CAR: {len(dvm_pairs)} pairs | seed_dir: {args.seed_dir}")

    reports: list = []

    async def run_all():
        for ds in args.datasets:
            rep = await evaluate_dataset(
                ds, args.seed_dir, dvm_pairs, args.data_dir,
                variants=args.variants,
                match_threshold=args.match_threshold,
                extraction_prompt=args.extraction_prompt,
                extraction_samples=args.extraction_samples,
            )
            out = os.path.join(
                args.data_dir,
                f"{ds}_normalizer_report{args.report_suffix}.json",
            )
            with open(out, "w") as f:
                json.dump(_clean(rep), f, indent=2)
            print(f"  wrote {out}")
            reports.append(rep)

    asyncio.run(run_all())

    if reports:
        non_ie = [v for v in reports[0]["variants"] if v != "ie"]
        print("\n" + "=" * 104)
        print(f"Consolidated summary (brand, model) — variants vs IE baseline")
        print("=" * 104)
        header = f"{'attr':<6s} {'dataset':<12s} {'variant':<10s} {'acc_IE':>7s} {'acc_V':>7s} {'Δacc':>6s}  {'F1_IE':>6s} {'F1_V':>6s} {'ΔF1':>6s}  {'ΔP':>6s} {'ΔR':>6s}"
        print(header)
        print("-" * len(header))
        for a in ["brand", "model"]:
            for r in reports:
                for v in non_ie:
                    d = r["deltas"][v][a]
                    print(
                        f"{a:<6s} {r['dataset']:<12s} {v:<10s} "
                        f"{d['acc_ie']:7.3f} {d['acc_vn']:7.3f} {d['delta_acc']*100:+6.2f}  "
                        f"{d['f1_ie']:6.3f} {d['f1_vn']:6.3f} {d['delta_f1']*100:+6.2f}  "
                        f"{d['delta_precision']*100:+6.2f} {d['delta_recall']*100:+6.2f}"
                    )
        print("\n(Δs shown in percentage points.)")

        print("\nNovelty F1 / error taxonomy (model):")
        print(f"{'dataset':<12s} {'variant':<10s} {'novelty F1':>10s}  {'under':>6s}  {'bad':>5s}  {'subst':>6s}  {'miss':>5s}")
        for r in reports:
            for v in non_ie:
                nov = r["novelty"][v]["f1"]
                t = r["error_taxonomy"][v]["model"]["counts"]
                print(f"{r['dataset']:<12s} {v:<10s} {nov:10.3f}  {t['under_merge']:>6d}  {t['bad_merge']:>5d}  {t['substitution']:>6d}  {t['missing']:>5d}")


if __name__ == "__main__":
    main()
