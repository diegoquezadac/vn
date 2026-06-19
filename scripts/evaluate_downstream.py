"""
evaluate_downstream.py - downstream price-prediction evaluation (paper Table V, MuCars).

Trains CatBoost / XGBoost / LightGBM on the raw listing features vs. the raw features
augmented with the normalized representation, over N random splits, and reports the
MAPE / MdAPE / MAE improvement on the full test set and on the canonicalized-models
slice (rows whose model string was redirected to a different canonical value).

Inputs (raw datasets are not redistributed - see CLEI2026.md):
  data/mucars.csv             - raw listings
  data/mucars_mappings.db     - normalized representation (built by scripts/normalize.py)

Regressor pipeline (applied identically to the raw and normalized feature sets):
  - high-cardinality categoricals (brand, model) -> sklearn TargetEncoder (seeded)
  - low-cardinality categoricals (the canonical attributes + condition/gearbox/fuel)
    -> one-hot, so the trees can split on them cleanly
  - depth-limited gradient boosting (XGBoost and LightGBM max_depth=6); CatBoost
    uses native categorical handling at library defaults
  - n_jobs=1 + seeded TargetEncoder + LightGBM determinism flags -> reproducible

Usage:
    KMP_DUPLICATE_LIB_OK=TRUE uv run python scripts/evaluate_downstream.py
    KMP_DUPLICATE_LIB_OK=TRUE uv run python scripts/evaluate_downstream.py --seeds 10
"""

import argparse
import ast
import json
import re
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.train import train_catboost, train_sklearn_model  # noqa: E402

MODEL_KINDS = ["catboost", "xgboost", "lightgbm"]

# Paper Table V MuCars MAPE deltas (norm vs raw, %): (full set, canonicalized slice).
PAPER_MAPE = {"catboost": (-2.7, -4.5), "xgboost": (-5.3, -5.1), "lightgbm": (-4.3, -4.9)}

EQUIPMENT_ITEMS = [
    "ABS", "Airbags", "ESP", "Rear Camera", "Parking Sensors", "Air Conditioning",
    "Leather Seats", "Sunroof", "Navigation System/GPS", "CD/MP3/Bluetooth",
    "Onboard Computer", "Cruise Control", "Speed Limiter", "Electric Windows",
    "Central Locking", "Alloy Wheels",
]

# Low-cardinality categoricals -> one-hot; brand/model stay target-encoded.
ONEHOT = ["condition", "gearbox", "fuel", "n_body_type", "n_transmission_type",
          "n_transmission_technology", "n_engine_aspiration", "n_injection_type",
          "n_energy_source", "n_fuel_type", "n_propulsion_system", "n_drive_type"]

# Tuned config for the tree models (CatBoost keeps its defaults). Applied to both the
# raw and normalized feature sets, so the comparison stays fair.
SKLEARN_HP = {
    "onehot_columns": ONEHOT,
    "early_stopping_rounds": 120,
    "xgb_params": {"n_estimators": 4000, "learning_rate": 0.02, "max_depth": 6,
                   "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 3,
                   "n_jobs": 1},
    "lgbm_params": {"n_estimators": 4000, "learning_rate": 0.02, "num_leaves": 63,
                    "max_depth": 6, "subsample": 0.8, "subsample_freq": 1,
                    "colsample_bytree": 0.8, "min_child_samples": 20,
                    "n_jobs": 1, "deterministic": True, "force_row_wise": True},
}


# ── MuCars preprocessing ─────────────────────────────────────────────────
def parse_mileage(s):
    if pd.isna(s):
        return np.nan
    s = str(s).strip(); low = s.lower()
    nums = [int(n.replace(" ", "")) for n in re.findall(r"\d[\d ]*\d|\d", s)]
    if len(nums) >= 2:
        return (nums[0] + nums[1]) / 2
    if len(nums) == 1:
        return nums[0] if "plus" in low else nums[0] / 2 if "moins" in low else nums[0]
    return np.nan


def parse_fiscal_power(s):
    if pd.isna(s):
        return np.nan
    m = re.search(r"(\d+)", str(s))
    return float(m.group(1)) if m else np.nan


def preprocess_mucars(df):
    df = df.copy()
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
    df = df[df["Price"].notna() & (df["Price"] >= 5_000) & (df["Price"] <= 5_000_000)]
    df["year"] = pd.to_numeric(df["Year"], errors="coerce")
    df.loc[(df["year"] < 1950) | (df["year"] > 2026), "year"] = np.nan
    df["mileage_mid"] = df["Mileage"].apply(parse_mileage)
    df["fiscal_power"] = df["Fiscal Power"].apply(parse_fiscal_power)
    df["doors"] = pd.to_numeric(df["Number of Doors"], errors="coerce")
    for col in ["Brand", "Model", "Condition", "Gearbox", "Fuel"]:
        df[col.lower()] = df[col].astype(str).str.lower().str.strip().fillna("unknown")
    df["price"] = df["Price"]
    return df.reset_index(drop=True)


_NULLISH = {"nan", "none", "other", "unknown", "n/a", "na", "", "unspecified",
            "not specified", "missing"}
_DESC_COLS = ["Brand", "Model", "Year", "Gearbox"]


def make_desc(row):
    parts = []
    for col in _DESC_COLS:
        v = row.get(col)
        if pd.isna(v):
            continue
        s = str(v).strip()
        if s.lower() in _NULLISH:
            continue
        if isinstance(v, float) and v == int(v):
            s = str(int(v))
        parts.append(f"{col}: {s}")
    return ", ".join(parts)


def parse_equipment_set(s):
    if pd.isna(s):
        return set()
    try:
        v = ast.literal_eval(s)
        if isinstance(v, list):
            return {str(x).strip() for x in v}
    except Exception:
        pass
    return set()


def slug(s):
    return "eq_" + s.lower().replace("/", "_").replace(" ", "_")


def load_data(data_dir: str, mappings_db: str):
    df_raw = pd.read_csv(f"{data_dir}/mucars.csv")
    df = preprocess_mucars(df_raw)

    price = pd.to_numeric(df_raw["Price"], errors="coerce")
    df_orig = df_raw[price.notna() & (price >= 5_000) & (price <= 5_000_000)].reset_index(drop=True)
    df["_desc"] = df_orig.apply(make_desc, axis=1)

    conn = sqlite3.connect(mappings_db)
    mappings = {r[0]: json.loads(r[1]) for r in conn.execute("SELECT description, result FROM mappings")}
    conn.close()

    for key in sorted({k for v in mappings.values() for k in v if k != "equipment"}):
        df[f"n_{key}"] = df["_desc"].map(lambda d, k=key: (mappings.get(d) or {}).get(k))

    equip = df_orig["Equipment"].apply(parse_equipment_set).values
    for item in EQUIPMENT_ITEMS:
        df[slug(item)] = [int(item in s) for s in equip]

    cov = df["_desc"].isin(mappings).mean()
    print(f"rows={len(df):,} | mappings={len(mappings):,} | coverage={cov*100:.1f}%")
    return df


# ── imputation (brand/model group medians/modes) ─────────────────────────
def impute_column(df_tr, df_te, col, is_numeric, group_by, min_group_size=5):
    g1, g2 = group_by

    def agg(x):
        if x.count() < min_group_size:
            return None
        return x.median() if is_numeric else (x.mode()[0] if len(x.mode()) else None)

    bm = df_tr.groupby([g1, g2])[col].agg(agg).dropna()
    b = df_tr.groupby(g1)[col].agg(agg).dropna()
    overall = (df_tr[col].median() if is_numeric
               else (df_tr[col].mode()[0] if not df_tr[col].isna().all() else None))
    iv = {(br, mo): v for (br, mo), v in bm.items()}
    for br, v in b.items():
        iv[(br, None)] = v
    iv[("__default__", None)] = overall

    def fill(row):
        if not pd.isna(row[col]):
            return row[col]
        return iv.get((row[g1], row[g2])) or iv.get((row[g1], None)) or iv[("__default__", None)]

    df_tr, df_te = df_tr.copy(), df_te.copy()
    if df_tr[col].isna().any():
        df_tr[col] = df_tr.apply(fill, axis=1)
    if df_te[col].isna().any():
        df_te[col] = df_te.apply(fill, axis=1)
    return df_tr, df_te


def impute_all(df_tr, df_te, num_cols, cat_cols, group_by):
    for c in num_cols:
        df_tr, df_te = impute_column(df_tr, df_te, c, True, group_by)
    for c in cat_cols:
        df_tr, df_te = impute_column(df_tr, df_te, c, False, group_by)
    return df_tr, df_te


def cast_cats(df, cat_cols):
    df = df.copy()
    df[cat_cols] = df[cat_cols].fillna("").astype(str).replace("nan", "")
    return df


def fit_predict(model_kind, df_tr, df_te, num_cols, cat_cols, base_cfg):
    cfg = dict(base_cfg, num_columns=num_cols, cat_columns=cat_cols)
    if model_kind == "catboost":
        model, _ = train_catboost(df_tr, cfg)                 # CatBoost defaults
        return np.exp(model.predict(df_te[num_cols + cat_cols]))
    cfg.update(SKLEARN_HP)
    pipe, _ = train_sklearn_model(model_kind, cast_cats(df_tr, cat_cols), cfg)
    return np.exp(pipe.predict(cast_cats(df_te, cat_cols)[num_cols + cat_cols]))


def evaluate(df, seeds: int):
    eq_cols = [slug(i) for i in EQUIPMENT_ITEMS]
    raw_num = ["year", "mileage_mid", "fiscal_power", "doors"] + eq_cols
    raw_cat = ["brand", "model", "condition", "gearbox", "fuel"]
    n_num = [c for c in ["n_engine_power", "n_engine_size", "n_cylinders", "n_doors",
                         "n_transmission_gears"] if c in df.columns]
    n_cat = [c for c in ["n_brand", "n_model", "n_body_type", "n_transmission_type",
                        "n_transmission_technology", "n_engine_aspiration", "n_injection_type",
                        "n_energy_source", "n_fuel_type", "n_propulsion_system", "n_drive_type"]
             if c in df.columns]
    comb_num, comb_cat = raw_num + n_num, raw_cat + n_cat
    base = {"target": "price", "validation": "holdout", "val_size": 0.2, "loss_function": "MAE"}

    # (model, raw/norm, slice) -> per-seed [mape, mdape, mae]
    acc = {(m, c, s): {"mape": [], "mdape": [], "mae": []}
           for m in MODEL_KINDS for c in ("raw", "norm") for s in ("all", "merged")}

    for rs in range(seeds):
        tr, te = train_test_split(df, test_size=0.2, random_state=rs)
        y = te["price"].values
        base_num = ["year", "mileage_mid", "fiscal_power", "doors"]
        base_cat = ["brand", "model", "condition", "gearbox", "fuel"]
        tr_r, te_r = impute_all(tr, te, base_num, base_cat, ("brand", "model"))
        tr_c, te_c = impute_all(tr, te, base_num, base_cat, ("brand", "model"))
        tr_c, te_c = impute_all(tr_c, te_c, n_num, n_cat, ("n_brand", "n_model"))
        for c in eq_cols:
            tr_r[c] = tr_r[c].fillna(0); te_r[c] = te_r[c].fillna(0)
            tr_c[c] = tr_c[c].fillna(0); te_c[c] = te_c[c].fillna(0)

        nm = te["n_model"]
        merged = nm.notna().values & (te["model"].astype(str).values
                                      != nm.fillna("").astype(str).str.lower().values)
        masks = {"all": np.ones(len(y), bool), "merged": merged}

        print(f"[seed {rs}] merged {int(merged.sum())}/{len(y)}")
        for mk in MODEL_KINDS:
            preds = {"raw": fit_predict(mk, tr_r, te_r, raw_num, raw_cat, base),
                     "norm": fit_predict(mk, tr_c, te_c, comb_num, comb_cat, base)}
            for c, yp in preds.items():
                for s, mask in masks.items():
                    yt, yhat = y[mask], yp[mask]
                    ape = np.abs((yhat - yt) / yt)
                    acc[(mk, c, s)]["mape"].append(float(np.mean(ape)))
                    acc[(mk, c, s)]["mdape"].append(float(np.median(ape)))
                    acc[(mk, c, s)]["mae"].append(float(np.mean(np.abs(yhat - yt))))
    return acc


def report(acc):
    def pct(n, r):
        return (n - r) / r * 100

    for s, label in [("all", "Full test set"), ("merged", "Canonicalized models")]:
        print(f"\n{'='*84}\n  {label}\n{'='*84}")
        print(f"  {'Model':<9} {'MAPE raw->norm (Δ)':<26} {'MdAPE (Δ)':<18} {'MAE (Δ)':<22} {'paperΔ':>7} ok")
        print(f"  {'-'*82}")
        for mk in MODEL_KINDS:
            r = {k: np.mean(acc[(mk, "raw", s)][k]) for k in ("mape", "mdape", "mae")}
            n = {k: np.mean(acc[(mk, "norm", s)][k]) for k in ("mape", "mdape", "mae")}
            dmape = pct(n["mape"], r["mape"])
            tgt = PAPER_MAPE[mk][0 if s == "all" else 1]
            ok = "PASS" if dmape <= tgt + 0.05 else "fail"
            print(f"  {mk:<9} "
                  f"MAPE {r['mape']:.3f}->{n['mape']:.3f} ({dmape:+.1f}%)  "
                  f"MdAPE {r['mdape']:.3f}->{n['mdape']:.3f} ({pct(n['mdape'], r['mdape']):+.1f}%)  "
                  f"MAE {r['mae']:,.0f}->{n['mae']:,.0f} ({pct(n['mae'], r['mae']):+.1f}%)  [paper {tgt:+.1f} {ok}]")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--mappings_db", default="data/mucars_mappings.db")
    ap.add_argument("--seeds", type=int, default=10)
    args = ap.parse_args()
    df = load_data(args.data_dir, args.mappings_db)
    report(evaluate(df, args.seeds))


if __name__ == "__main__":
    main()
