"""
pixel_level_soc_model.py
========================
SOC estimation — pixel-level approach.

Instead of reducing each plot to one row (spatial median), every pixel
becomes its own training sample and inherits its plot's SOC_pct label.

  40 plots x ~40 pixels avg  =  ~1,600 pixel-level rows

VALIDATION (critical — must be plot-level to avoid data leakage):
  Outer loop: hold out ALL pixels from plot i
  Inner loop: train on pixels from the other 39 plots
  Prediction: mean of pixel-level predictions for held-out plot = plot SOC pred
  Metrics  : computed at plot level (40 predictions vs 40 ground-truth SOC values)

FEATURES (pixel-level):
  A  S2 bands only                                  12
  B  S2 + AEE                                       76
  C  S2 + spectral indices + AEE                    81
  D  S2 + spectral indices + AEE + bare_pred (0/1)  82   <- bare is pixel-level here

MODELS (per variant):
  - PLSR   (same family as teammate, for apples-to-apples)
  - Ridge  (linear, handles high-dim well)
  - XGBoost (non-linear, likely best for this data size)

Outputs
-------
  pixel_soc_loo_predictions.csv   per-plot LOO predictions for every variant x model
  pixel_soc_best_model.joblib     best (variant, model) fitted on all pixels
"""

import os, glob, re, gc, warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE        = os.path.dirname(os.path.abspath(__file__))
AEE_FOLDER  = os.path.join(BASE, "s2_soc_aee_data")
BARE_FOLDER = os.path.join(BASE, "s2_bare")
OUT_CSV     = os.path.join(BASE, "pixel_soc_loo_predictions.csv")
OUT_MODEL   = os.path.join(BASE, "pixel_soc_best_model.joblib")

# ── Feature columns ───────────────────────────────────────────────────────────
S2_BANDS = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12"]
AEE_COLS = [f"A{i:02d}" for i in range(64)]
IDX_COLS = ["BSI", "NDVI", "NDWI", "SWIR_ratio", "brightness"]
EPS      = 1e-8


def compute_indices(df: pd.DataFrame) -> pd.DataFrame:
    df["BSI"]        = ((df["B11"] + df["B4"]) - (df["B8"] + df["B2"])) / \
                       ((df["B11"] + df["B4"]) + (df["B8"] + df["B2"]) + EPS)
    df["NDVI"]       = (df["B8"]  - df["B4"]) / (df["B8"]  + df["B4"]  + EPS)
    df["NDWI"]       = (df["B3"]  - df["B8"]) / (df["B3"]  + df["B8"]  + EPS)
    df["SWIR_ratio"] = df["B11"]              / (df["B12"]              + EPS)
    df["brightness"] = (df["B2"] + df["B3"] + df["B4"] + df["B8"]) / 4.0
    return df


# ── Model definitions ─────────────────────────────────────────────────────────
PLSR_PIPE  = Pipeline([("sc", StandardScaler()), ("pls", PLSRegression())])
RIDGE_PIPE = Pipeline([("sc", StandardScaler()), ("r",   Ridge())])

MODELS = {
    "PLSR" : (PLSR_PIPE,  {"pls__n_components": [1, 2, 3, 4, 5, 6, 8, 10]}),
    "Ridge": (RIDGE_PIPE, {"r__alpha": [0.01, 0.1, 1, 10, 100, 1000]}),
    "XGB"  : (XGBRegressor(
                  n_estimators=300, learning_rate=0.05,
                  max_depth=4, subsample=0.8, colsample_bytree=0.8,
                  random_state=42, verbosity=0, n_jobs=-1),
              {"max_depth": [3, 4, 5], "learning_rate": [0.05, 0.1],
               "n_estimators": [200, 400]}),
}


# ── 1. Load all pixels ────────────────────────────────────────────────────────
print("=" * 65)
print("Step 1: Loading all pixels from 40 plots...")
print("=" * 65)

aee_files = sorted(glob.glob(os.path.join(AEE_FOLDER, "SOC_Poly_combined_plot*_2025.csv")))
bare_lookup = {
    int(re.search(r"plot(\d+)", f).group(1)): f
    for f in glob.glob(os.path.join(BARE_FOLDER, "SOC_Poly_combined_plot*_bare.csv"))
    if re.search(r"plot(\d+)", f)
}

all_chunks = []
for fp in aee_files:
    pid = int(re.search(r"plot(\d+)", fp).group(1))
    df  = pd.read_csv(fp)
    df  = compute_indices(df)

    # Join bare prediction column (pixel-level 0/1) by row position
    if pid in bare_lookup:
        df_bare = pd.read_csv(bare_lookup[pid], usecols=["prediction"])
        # Rows are in identical order (verified) — safe direct assignment
        df["bare_pred"] = df_bare["prediction"].values
    else:
        print(f"  [WARN] No bare file for plot {pid} — bare_pred set to 0")
        df["bare_pred"] = 0

    df["plot_id"] = pid
    # SOC_pct is already in every row; keep it
    all_chunks.append(df)
    del df_bare; gc.collect()

data = pd.concat(all_chunks, ignore_index=True)
del all_chunks; gc.collect()

n_pixels = len(data)
n_plots  = data["plot_id"].nunique()
print(f"Total pixels : {n_pixels:,}  across {n_plots} plots")
print(f"SOC range    : {data['SOC_pct'].min():.3f} – {data['SOC_pct'].max():.3f} %")
print(f"Bare fraction: {data['bare_pred'].mean():.3f}")

# ── 2. Build feature matrices ─────────────────────────────────────────────────
y_px      = data["SOC_pct"].values.astype(np.float32)
plot_ids  = data["plot_id"].values
soc_truth = data.groupby("plot_id")["SOC_pct"].first().sort_index()   # (40,) plot-level truth

FEAT_SETS = {
    "A_s2only"       : S2_BANDS,
    "B_s2_aee"       : S2_BANDS + AEE_COLS,
    "C_s2_idx_aee"   : S2_BANDS + IDX_COLS + AEE_COLS,
    "D_s2_idx_aee_bare": S2_BANDS + IDX_COLS + AEE_COLS + ["bare_pred"],
}

X_all = {name: data[cols].values.astype(np.float32)
         for name, cols in FEAT_SETS.items()}

print(f"\nFeature matrix shapes:")
for name, X in X_all.items():
    print(f"  {name:<24}: {X.shape}   NaN={np.isnan(X).sum()}")


# ── 3. Plot-level LOO-CV helper ───────────────────────────────────────────────
unique_plots = np.sort(data["plot_id"].unique())   # 1 … 40

def loo_cv(X, y, plot_ids_arr, model_template, param_grid, model_name):
    """
    Plot-level LOO. Returns plot-level SOC predictions (40,) and
    the best n_components / alpha used per fold.
    """
    is_xgb   = isinstance(model_template, XGBRegressor)
    plot_preds = {}

    for pid in unique_plots:
        test_mask  = (plot_ids_arr == pid)
        train_mask = ~test_mask

        X_tr, y_tr = X[train_mask], y[train_mask]
        X_te       = X[test_mask]

        if is_xgb:
            # GridSearchCV with XGB
            search = GridSearchCV(
                XGBRegressor(random_state=42, verbosity=0, n_jobs=1),
                param_grid,
                cv=KFold(5, shuffle=True, random_state=42),
                scoring="neg_mean_squared_error",
                n_jobs=-1,
            )
            search.fit(X_tr, y_tr)
            preds_px = search.best_estimator_.predict(X_te)
        else:
            search = GridSearchCV(
                model_template,
                param_grid,
                cv=KFold(5, shuffle=True, random_state=42),
                scoring="neg_mean_squared_error",
                n_jobs=-1,
            )
            search.fit(X_tr, y_tr)
            preds_px = search.best_estimator_.predict(X_te).ravel()

        # Average pixel predictions → single plot SOC estimate
        plot_preds[pid] = float(np.mean(preds_px))

    return plot_preds


# ── 4. Run LOO for all variants x models ─────────────────────────────────────
print()
print("=" * 65)
print("Step 2: Plot-level LOO-CV  (40 folds × 3 models × 4 variants)...")
print("=" * 65)

results_table = []   # rows: {variant, model, R2, RMSE, MAE}
all_predictions = {pid: {"plot_id": pid, "soc_measured": float(soc_truth[pid])}
                   for pid in unique_plots}

for feat_name, X in X_all.items():
    for model_name, (model_tmpl, param_grid) in MODELS.items():
        label = f"{feat_name} / {model_name}"
        print(f"\n  Running: {label} ...")

        plot_preds = loo_cv(X, y_px, plot_ids, model_tmpl, param_grid, model_name)

        y_true = np.array([soc_truth[pid] for pid in unique_plots])
        y_pred = np.array([plot_preds[pid] for pid in unique_plots])

        r2   = float(1 - np.sum((y_true - y_pred)**2) / np.sum((y_true - y_true.mean())**2))
        rmse = float(np.sqrt(((y_true - y_pred)**2).mean()))
        mae  = float(np.abs(y_true - y_pred).mean())

        col_key = f"{feat_name}__{model_name}"
        for pid in unique_plots:
            all_predictions[pid][col_key] = round(plot_preds[pid], 4)

        results_table.append({
            "variant": feat_name, "model": model_name,
            "R2": round(r2, 4), "RMSE": round(rmse, 4), "MAE": round(mae, 4),
        })
        print(f"    R2={r2:+.3f}  RMSE={rmse:.4f}  MAE={mae:.4f}")

# ── 5. Results table ──────────────────────────────────────────────────────────
print()
print("=" * 65)
print("Step 3: Full results")
print("=" * 65)
results_df = pd.DataFrame(results_table).sort_values("R2", ascending=False)
print(f"{'Variant':<24} {'Model':<7} {'R2':>8} {'RMSE':>8} {'MAE':>8}")
print("-" * 60)
for _, row in results_df.iterrows():
    marker = "  <-- BEST" if row["R2"] == results_df["R2"].max() else ""
    print(f"{row['variant']:<24} {row['model']:<7} {row['R2']:>+8.3f} {row['RMSE']:>8.4f} {row['MAE']:>8.4f}{marker}")

# ── 6. Save predictions CSV ───────────────────────────────────────────────────
pred_df = pd.DataFrame(list(all_predictions.values())).sort_values("plot_id").reset_index(drop=True)
pred_df.to_csv(OUT_CSV, index=False)
print(f"\nPer-plot predictions saved to: {OUT_CSV}")

# ── 7. Refit best on all pixels ───────────────────────────────────────────────
best_row  = results_df.iloc[0]
best_feat = best_row["variant"]
best_mod  = best_row["model"]
print()
print("=" * 65)
print(f"Step 4: Refitting best ({best_feat} / {best_mod}) on all {n_pixels:,} pixels...")
print("=" * 65)

X_best  = X_all[best_feat]
_, param_grid = MODELS[best_mod]
is_xgb_best  = (best_mod == "XGB")

if is_xgb_best:
    final_search = GridSearchCV(
        XGBRegressor(random_state=42, verbosity=0, n_jobs=1),
        param_grid,
        cv=KFold(5, shuffle=True, random_state=42),
        scoring="neg_mean_squared_error", n_jobs=-1,
    )
else:
    model_tmpl, _ = MODELS[best_mod]
    final_search = GridSearchCV(
        model_tmpl, param_grid,
        cv=KFold(5, shuffle=True, random_state=42),
        scoring="neg_mean_squared_error", n_jobs=-1,
    )

final_search.fit(X_best, y_px)
final_model = final_search.best_estimator_
print(f"Best params: {final_search.best_params_}")

bundle = {
    "model"       : final_model,
    "variant"     : best_feat,
    "model_type"  : best_mod,
    "feature_cols": FEAT_SETS[best_feat],
    "loo_r2"      : best_row["R2"],
    "approach"    : "pixel_level",
    "results_all" : results_table,
}
joblib.dump(bundle, OUT_MODEL)
print(f"Model bundle saved to: {OUT_MODEL}")

# ── 8. Final summary ──────────────────────────────────────────────────────────
print()
print("=" * 65)
print("FINAL SUMMARY — Plot-level LOO R2 comparison")
print("=" * 65)
print()
print("  PLOT-LEVEL MODEL (previous, spatial median per plot):")
print("    A S2 only         : R2 = -0.086")
print("    B S2+AEE          : R2 = -0.113")
print("    C S2+Idx+AEE      : R2 = -0.054")
print("    D S2+Idx+AEE+bare : R2 = -0.055")
print()
print("  PIXEL-LEVEL MODEL (this file):")
for _, row in results_df.iterrows():
    print(f"    {row['variant']:<24} / {row['model']:<7}: R2 = {row['R2']:+.3f}")
print()
print(f"  Best overall: {best_feat} / {best_mod}  (R2 = {best_row['R2']:+.3f})")
