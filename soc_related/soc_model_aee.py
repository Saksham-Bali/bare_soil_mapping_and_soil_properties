"""
soc_model_aee.py
================
SOC estimation model — 4 variants compared via Leave-One-Out CV.

Data source: soc_related/s2_soc_aee_data/  (S2 bands + 64-dim AEE + SOC_pct)
             soc_related/s2_bare/           (bare soil 0/1 predictions per pixel)

Feature reduction: spatial MEDIAN across pixels per plot (same as teammate's model).
  - S2 bands   : 12 bands -> 12 scalars
  - Indices    : BSI, NDVI, NDWI, SWIR_ratio, brightness -> 5 scalars
  - AEE        : A00-A63 -> 64 scalars
  - Bare frac  : mean of prediction (0/1) across pixels -> 1 scalar (0.0-1.0)

Variants
--------
  A  - S2 bands only                             (12 features)  [teammate baseline]
  B  - S2 bands + AEE                            (76 features)
  C  - S2 bands + Indices + AEE                  (81 features)
  D  - S2 bands + Indices + AEE + bare_fraction  (82 features)

Pipeline: StandardScaler -> PLSRegression
Grid    : n_components in {1,2,3,4,5,6,8,10}
CV      : LOO outer / 5-fold inner GridSearchCV

Outputs
-------
  soc_aee_model.joblib              Final model (best variant, fitted on all 40 plots)
  soc_predictions_comparison.csv   Per-plot predictions from all 4 variants
"""

import os, glob, re, gc, warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE         = os.path.dirname(os.path.abspath(__file__))
AEE_FOLDER   = os.path.join(BASE, "s2_soc_aee_data")
BARE_FOLDER  = os.path.join(BASE, "s2_bare")
OUT_MODEL    = os.path.join(BASE, "soc_aee_model.joblib")
OUT_CSV      = os.path.join(BASE, "soc_predictions_comparison.csv")

# ── Feature column definitions ────────────────────────────────────────────────
S2_BANDS  = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12"]
AEE_COLS  = [f"A{i:02d}" for i in range(64)]
EPS       = 1e-8

GRID      = {"pls__n_components": [1, 2, 3, 4, 5, 6, 8, 10]}


# ── Helpers ───────────────────────────────────────────────────────────────────

def compute_indices(df):
    """Add derived spectral index columns in-place. Returns df."""
    df["BSI"]        = ((df["B11"] + df["B4"]) - (df["B8"] + df["B2"])) / \
                       ((df["B11"] + df["B4"]) + (df["B8"] + df["B2"]) + EPS)
    df["NDVI"]       = (df["B8"]  - df["B4"]) / (df["B8"]  + df["B4"]  + EPS)
    df["NDWI"]       = (df["B3"]  - df["B8"]) / (df["B3"]  + df["B8"]  + EPS)
    df["SWIR_ratio"] = df["B11"]              / (df["B12"]              + EPS)
    df["brightness"] = (df["B2"] + df["B3"] + df["B4"] + df["B8"]) / 4.0
    return df

INDEX_COLS = ["BSI", "NDVI", "NDWI", "SWIR_ratio", "brightness"]


def spatial_median(df, cols):
    """Return median across all pixels for the given columns. Shape: (len(cols),)"""
    return df[cols].median(axis=0).values


def spatial_mean(df, cols):
    """Return mean across all pixels for the given columns."""
    return df[cols].mean(axis=0).values


# ── 1. Load all 40 plots ──────────────────────────────────────────────────────
print("=" * 65)
print("Step 1: Loading data from all 40 plots...")
print("=" * 65)

aee_files = sorted(glob.glob(os.path.join(AEE_FOLDER, "SOC_Poly_combined_plot*_2025.csv")))
assert len(aee_files) == 40, f"Expected 40 AEE files, found {len(aee_files)}"

plot_ids    = []
y_soc       = []
feat_s2     = []     # (40, 12)
feat_idx    = []     # (40, 5)
feat_aee    = []     # (40, 64)
feat_bare   = []     # (40, 1)

# Build bare-file lookup: plot_id -> path
bare_lookup = {}
for bf in glob.glob(os.path.join(BARE_FOLDER, "SOC_Poly_combined_plot*_bare.csv")):
    m = re.search(r"plot(\d+)", bf)
    if m:
        bare_lookup[int(m.group(1))] = bf

for fp in aee_files:
    pid = int(re.search(r"plot(\d+)", fp).group(1))
    df  = pd.read_csv(fp)
    df  = compute_indices(df)

    # SOC: one value per plot (all pixels identical)
    soc = float(df["SOC_pct"].iloc[0])

    # Spatial medians
    s2_vec  = spatial_median(df, S2_BANDS)
    idx_vec = spatial_median(df, INDEX_COLS)
    aee_vec = spatial_median(df, AEE_COLS)

    # Bare fraction: mean of prediction column from s2_bare
    if pid in bare_lookup:
        df_bare = pd.read_csv(bare_lookup[pid])
        bare_frac = float(df_bare["prediction"].mean())
    else:
        # Fallback: compute bare fraction from scratch using a simple BSI threshold
        # (should not happen if s2_bare is populated)
        print(f"  [WARN] No bare file for plot {pid} — using 0.0 as bare_fraction")
        bare_frac = 0.0

    plot_ids.append(pid)
    y_soc.append(soc)
    feat_s2.append(s2_vec)
    feat_idx.append(idx_vec)
    feat_aee.append(aee_vec)
    feat_bare.append([bare_frac])

    del df; gc.collect()

plot_ids  = np.array(plot_ids)
y         = np.array(y_soc)
X_s2      = np.array(feat_s2)           # (40, 12)
X_idx     = np.array(feat_idx)          # (40, 5)
X_aee     = np.array(feat_aee)          # (40, 64)
X_bare    = np.array(feat_bare)         # (40, 1)

# Assemble feature matrices for each variant
X_variants = {
    "A_s2only"       : X_s2,
    "B_s2_aee"       : np.hstack([X_s2, X_aee]),
    "C_s2_idx_aee"   : np.hstack([X_s2, X_idx, X_aee]),
    "D_s2_idx_aee_bare": np.hstack([X_s2, X_idx, X_aee, X_bare]),
}

print(f"Loaded {len(plot_ids)} plots  |  SOC range: {y.min():.3f} – {y.max():.3f} %")
print(f"Feature shapes per variant:")
for name, X in X_variants.items():
    print(f"  {name:<22}: {X.shape}  (NaN count: {np.isnan(X).sum()})")

# ── 2. PLSR pipeline builder ──────────────────────────────────────────────────

def make_pipeline():
    return Pipeline([
        ("scale", StandardScaler()),
        ("pls",   PLSRegression()),
    ])


# ── 3. Leave-One-Out CV for all variants ─────────────────────────────────────
print()
print("=" * 65)
print("Step 2: Leave-One-Out CV for all 4 variants...")
print("=" * 65)

N = len(y)
loo_preds = {name: np.zeros(N) for name in X_variants}
best_n_comps = {name: [] for name in X_variants}

for i in range(N):
    train_idx = np.setdiff1d(np.arange(N), [i])

    for name, X in X_variants.items():
        search = GridSearchCV(
            make_pipeline(), GRID,
            cv=KFold(5, shuffle=True, random_state=42),
            scoring="neg_mean_squared_error",
            n_jobs=-1,
        )
        search.fit(X[train_idx], y[train_idx])
        loo_preds[name][i] = search.best_estimator_.predict(X[i:i+1]).ravel()[0]
        best_n_comps[name].append(search.best_params_["pls__n_components"])

    if (i + 1) % 10 == 0 or i == N - 1:
        print(f"  LOO fold {i+1}/{N} done")

# ── 4. Metrics ────────────────────────────────────────────────────────────────
print()
print("=" * 65)
print("Step 3: Results")
print("=" * 65)

results = []
best_r2, best_name, best_preds = -np.inf, None, None

print(f"{'Variant':<24} {'R2':>8} {'RMSE':>8} {'MAE':>8}  {'Avg n_comp':>10}")
print("-" * 65)
for name, preds in loo_preds.items():
    r2   = float(1 - np.sum((y - preds)**2) / np.sum((y - y.mean())**2))
    rmse = float(np.sqrt(((y - preds)**2).mean()))
    mae  = float(np.abs(y - preds).mean())
    avgc = float(np.mean(best_n_comps[name]))
    print(f"{name:<24} {r2:>+8.3f} {rmse:>8.4f} {mae:>8.4f}  {avgc:>10.1f}")
    results.append({"variant": name, "R2": round(r2,4), "RMSE": round(rmse,4), "MAE": round(mae,4)})
    if r2 > best_r2:
        best_r2, best_name, best_preds = r2, name, preds

print()
print(f"Best variant: {best_name}  (R2 = {best_r2:+.3f})")

# ── 5. Save predictions CSV ───────────────────────────────────────────────────
pred_df = pd.DataFrame({"plot_id": plot_ids, "soc_measured": y})
for name, preds in loo_preds.items():
    pred_df[f"pred_{name}"] = np.round(preds, 4)
pred_df = pred_df.sort_values("plot_id").reset_index(drop=True)
pred_df.to_csv(OUT_CSV, index=False)
print(f"\nPredictions saved to: {OUT_CSV}")

# ── 6. Final model: refit best variant on ALL 40 plots ───────────────────────
print()
print("=" * 65)
print(f"Step 4: Refitting best variant ({best_name}) on all 40 plots...")
print("=" * 65)

X_best = X_variants[best_name]
final_search = GridSearchCV(
    make_pipeline(), GRID,
    cv=KFold(5, shuffle=True, random_state=42),
    scoring="neg_mean_squared_error",
    n_jobs=-1,
)
final_search.fit(X_best, y)
final_model = final_search.best_estimator_
n_comp_final = final_model.named_steps["pls"].n_components
print(f"n_components selected: {n_comp_final}")

# Save with metadata
model_bundle = {
    "model"       : final_model,
    "variant"     : best_name,
    "n_components": n_comp_final,
    "feature_cols": {
        "s2_bands"   : S2_BANDS,
        "index_cols" : INDEX_COLS if "idx" in best_name else [],
        "aee_cols"   : AEE_COLS  if "aee" in best_name else [],
        "bare_frac"  : ["bare_fraction"] if "bare" in best_name else [],
    },
    "loo_r2"      : round(best_r2, 4),
    "results_all" : results,
}
joblib.dump(model_bundle, OUT_MODEL)
print(f"Model bundle saved to: {OUT_MODEL}")

# ── 7. Quick example prediction ───────────────────────────────────────────────
print()
print("=" * 65)
print("Step 5: Example predictions (final model on training data)")
print("=" * 65)
example_preds = final_model.predict(X_best).ravel()
print(f"{'Plot':>5}  {'Measured':>9}  {'Predicted':>10}  {'Error':>8}")
print("-" * 40)
for i in np.argsort(plot_ids):
    pid_    = plot_ids[i]
    meas_   = y[i]
    pred_   = example_preds[i]
    err_    = pred_ - meas_
    print(f"{pid_:>5}  {meas_:>9.3f}  {pred_:>10.3f}  {err_:>+8.3f}")

print()
print("=" * 65)
print("Done.")
print(f"  Best model : {best_name}")
print(f"  LOO R2     : {best_r2:+.3f}")
print(f"  Saved      : {OUT_MODEL}")
print(f"  Predictions: {OUT_CSV}")
print("=" * 65)
