"""
pca_soc_model.py
================
Tests whether PCA preprocessing improves SOC prediction for linear models.

Experiments
-----------
PLOT-LEVEL (40 rows — spatial median per plot):
  PL-1  S2 bands (12)       Scale -> PCA -> Ridge
  PL-2  S2 + AEE (76)       Scale -> PCA -> Ridge
  PL-3  S2 + AEE (76)       Scale -> PCA -> ElasticNet

PIXEL-LEVEL (~1619 rows — all pixels, plot-level LOO):
  PX-1  S2 + AEE (76)            Scale -> PCA -> Ridge
  PX-2  S2 + Idx + AEE (81)      Scale -> PCA -> Ridge
  PX-3  S2 + Idx + AEE + Bare (82)  Scale -> PCA -> Ridge

Reference baselines (from previous runs, printed for comparison):
  Plot-level  PLSR on S2 only         R2 = +0.154  RMSE = 0.179%
  Plot-level  ElasticNet on AEE only  R2 = +0.090  RMSE = 0.186%
  Pixel-level XGBoost D-variant       R2 = +0.143  RMSE = 0.181%
  Pixel-level Ridge on S2 only        R2 = +0.056  RMSE = 0.190%

Validation
----------
  Plot-level : sklearn LeaveOneOut (40 folds)
  Pixel-level: custom plot-level LOO — hold out ALL pixels from one plot,
               predict SOC as mean of per-pixel predictions.
"""

import os, glob, re, gc, warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.model_selection import GridSearchCV, KFold, LeaveOneOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE        = os.path.dirname(os.path.abspath(__file__))
AEE_FOLDER  = os.path.join(BASE, "s2_soc_aee_data")
BARE_FOLDER = os.path.join(BASE, "s2_bare")

# ── Feature columns ───────────────────────────────────────────────────────────
S2_BANDS = ["B1","B2","B3","B4","B5","B6","B7","B8","B8A","B9","B11","B12"]
AEE_COLS = [f"A{i:02d}" for i in range(64)]
IDX_COLS = ["BSI","NDVI","NDWI","SWIR_ratio","brightness"]
EPS      = 1e-8


def compute_indices(df):
    df["BSI"]        = ((df["B11"]+df["B4"])-(df["B8"]+df["B2"])) / \
                       ((df["B11"]+df["B4"])+(df["B8"]+df["B2"])+EPS)
    df["NDVI"]       = (df["B8"]-df["B4"]) / (df["B8"]+df["B4"]+EPS)
    df["NDWI"]       = (df["B3"]-df["B8"]) / (df["B3"]+df["B8"]+EPS)
    df["SWIR_ratio"] = df["B11"] / (df["B12"]+EPS)
    df["brightness"] = (df["B2"]+df["B3"]+df["B4"]+df["B8"]) / 4.0
    return df


# ── 1. Load all pixels ────────────────────────────────────────────────────────
print("="*65)
print("Loading data ...")
print("="*65)

aee_files = sorted(glob.glob(os.path.join(AEE_FOLDER, "SOC_Poly_combined_plot*_2025.csv")))
bare_lookup = {
    int(re.search(r"plot(\d+)", f).group(1)): f
    for f in glob.glob(os.path.join(BARE_FOLDER, "SOC_Poly_combined_plot*_bare.csv"))
    if re.search(r"plot(\d+)", f)
}

chunks = []
for fp in aee_files:
    pid = int(re.search(r"plot(\d+)", fp).group(1))
    df  = pd.read_csv(fp)
    df  = compute_indices(df)
    if pid in bare_lookup:
        df["bare_pred"] = pd.read_csv(bare_lookup[pid], usecols=["prediction"])["prediction"].values
    else:
        df["bare_pred"] = 0
    df["plot_id"] = pid
    chunks.append(df)
    gc.collect()

data     = pd.concat(chunks, ignore_index=True)
plot_ids = data["plot_id"].values
y_px     = data["SOC_pct"].values.astype(np.float32)

# Plot-level arrays (spatial median)
unique_plots = np.sort(data["plot_id"].unique())
soc_truth    = {pid: float(data.loc[data["plot_id"]==pid, "SOC_pct"].iloc[0])
                for pid in unique_plots}
y_plot = np.array([soc_truth[p] for p in unique_plots])

def plot_median(cols):
    rows = []
    for pid in unique_plots:
        rows.append(data.loc[data["plot_id"]==pid, cols].median(axis=0).values)
    return np.array(rows, dtype=np.float32)

X_plot = {
    "S2"     : plot_median(S2_BANDS),
    "S2_AEE" : plot_median(S2_BANDS + AEE_COLS),
}
X_px = {
    "S2_AEE"      : data[S2_BANDS + AEE_COLS].values.astype(np.float32),
    "S2_Idx_AEE"  : data[S2_BANDS + IDX_COLS + AEE_COLS].values.astype(np.float32),
    "S2_Idx_AEE_B": data[S2_BANDS + IDX_COLS + AEE_COLS + ["bare_pred"]].values.astype(np.float32),
}

print(f"Plot-level: {len(unique_plots)} rows")
for k,v in X_plot.items(): print(f"  {k}: {v.shape}")
print(f"Pixel-level: {len(data)} rows")
for k,v in X_px.items():   print(f"  {k}: {v.shape}")


# ── Metric helpers ────────────────────────────────────────────────────────────
def metrics(y_true, y_pred):
    r2   = float(1 - np.sum((y_true-y_pred)**2) / np.sum((y_true-y_true.mean())**2))
    rmse = float(np.sqrt(((y_true-y_pred)**2).mean()))
    mae  = float(np.abs(y_true-y_pred).mean())
    return r2, rmse, mae


def print_result(label, r2, rmse, mae, best_k=None):
    kstr = f"  best_K={best_k}" if best_k is not None else ""
    print(f"  {label:<40} R2={r2:+.3f}  RMSE={rmse:.4f}  MAE={mae:.4f}{kstr}")


# ── 2. PLOT-LEVEL experiments ─────────────────────────────────────────────────
print()
print("="*65)
print("PLOT-LEVEL  (LOO across 40 plots)")
print("="*65)

# Note: inside each LOO fold we train on 39 plots.
# In inner 5-fold CV on those 39 plots, each fold has ~31 training samples.
# PCA n_components must be <= min(n_samples_inner, n_features).
# For S2(12): max K = 12. For S2+AEE(76): max K = 31, we cap at 25 to be safe.

GRID_RIDGE = {"r__alpha": [0.01, 0.1, 1, 10, 100, 1000]}
GRID_EN    = {"en__alpha": [0.01, 0.1, 1, 10],
              "en__l1_ratio": [0.1, 0.5, 0.7, 0.9]}

LOO = LeaveOneOut()
INNER_CV = KFold(5, shuffle=True, random_state=42)


def run_plot_pca(X, y, regressor_name, regressor, reg_grid, K_grid, label):
    """LOO-CV at plot level with PCA inside the pipeline."""
    preds = np.zeros(len(y))
    best_Ks = []

    pipe = Pipeline([
        ("sc",  StandardScaler()),
        ("pca", PCA()),
        (regressor_name, regressor),
    ])
    full_grid = {"pca__n_components": K_grid}
    full_grid.update(reg_grid)

    for train_idx, test_idx in LOO.split(X):
        search = GridSearchCV(pipe, full_grid, cv=INNER_CV,
                              scoring="neg_mean_squared_error", n_jobs=-1)
        search.fit(X[train_idx], y[train_idx])
        preds[test_idx] = search.best_estimator_.predict(X[test_idx]).ravel()
        best_Ks.append(search.best_params_["pca__n_components"])

    r2, rmse, mae = metrics(y, preds)
    best_k = int(np.round(np.mean(best_Ks)))
    print_result(label, r2, rmse, mae, best_k)
    return r2, rmse, mae, preds


# PL-1: S2 (12) -> PCA -> Ridge
K_s2 = [2, 3, 4, 5, 6, 8, 10, 12]
r2, rmse, mae, _ = run_plot_pca(
    X_plot["S2"], y_plot, "r", Ridge(), GRID_RIDGE, K_s2,
    "PL-1  S2 -> PCA -> Ridge")

# PL-2: S2+AEE (76) -> PCA -> Ridge
K_aee = [3, 5, 8, 10, 12, 15, 20, 25]
r2, rmse, mae, _ = run_plot_pca(
    X_plot["S2_AEE"], y_plot, "r", Ridge(), GRID_RIDGE, K_aee,
    "PL-2  S2+AEE -> PCA -> Ridge")

# PL-3: S2+AEE (76) -> PCA -> ElasticNet
r2, rmse, mae, _ = run_plot_pca(
    X_plot["S2_AEE"], y_plot, "en", ElasticNet(max_iter=5000), GRID_EN, K_aee,
    "PL-3  S2+AEE -> PCA -> ElasticNet")

print()
print("  -- Reference baselines (previous runs) --")
print("  Teammate PLSR on S2 only             R2=+0.154  RMSE=0.1790  MAE=0.1460")
print("  Teammate ElasticNet on AEE only      R2=+0.090  RMSE=0.1860  MAE=0.1410")
print("  Our PLSR on S2 only (raw, no PCA)    R2=-0.086  RMSE=0.2034  MAE=0.1628")
print("  Our PLSR on S2+AEE  (raw, no PCA)    R2=-0.113  RMSE=0.2059  MAE=0.1677")


# ── 3. PIXEL-LEVEL experiments ────────────────────────────────────────────────
print()
print("="*65)
print("PIXEL-LEVEL  (plot-level LOO, ~1619 pixels)")
print("="*65)

# At pixel level, inner 5-fold on ~1580 training pixels -> ~1265 training samples.
# PCA can go up to min(1265, n_features). We allow up to 40 comps.
K_px = [5, 10, 15, 20, 30, 40]


def run_pixel_pca(X, y_px, plot_ids_arr, K_grid, label):
    """Plot-level LOO at pixel level with PCA+Ridge inside the pipeline."""
    pipe = Pipeline([
        ("sc",  StandardScaler()),
        ("pca", PCA()),
        ("r",   Ridge()),
    ])
    full_grid = {"pca__n_components": K_grid}
    full_grid.update(GRID_RIDGE)

    plot_preds = {}
    best_Ks    = []

    for pid in unique_plots:
        test_mask  = (plot_ids_arr == pid)
        train_mask = ~test_mask

        search = GridSearchCV(pipe, full_grid, cv=INNER_CV,
                              scoring="neg_mean_squared_error", n_jobs=-1)
        search.fit(X[train_mask], y_px[train_mask])
        preds_px = search.best_estimator_.predict(X[test_mask]).ravel()
        plot_preds[pid] = float(np.mean(preds_px))
        best_Ks.append(search.best_params_["pca__n_components"])

    y_true = np.array([soc_truth[p] for p in unique_plots])
    y_pred = np.array([plot_preds[p] for p in unique_plots])
    r2, rmse, mae = metrics(y_true, y_pred)
    best_k = int(np.round(np.mean(best_Ks)))
    print_result(label, r2, rmse, mae, best_k)
    return r2, rmse, mae


# PX-1: S2+AEE -> PCA -> Ridge
run_pixel_pca(X_px["S2_AEE"],       y_px, plot_ids, K_px, "PX-1  S2+AEE -> PCA -> Ridge")

# PX-2: S2+Idx+AEE -> PCA -> Ridge
run_pixel_pca(X_px["S2_Idx_AEE"],   y_px, plot_ids, K_px, "PX-2  S2+Idx+AEE -> PCA -> Ridge")

# PX-3: S2+Idx+AEE+Bare -> PCA -> Ridge
run_pixel_pca(X_px["S2_Idx_AEE_B"], y_px, plot_ids, K_px, "PX-3  S2+Idx+AEE+Bare -> PCA -> Ridge")

print()
print("  -- Reference baselines (previous runs) --")
print("  Pixel XGBoost D (S2+Idx+AEE+Bare)    R2=+0.143  RMSE=0.1807  MAE=0.1429")
print("  Pixel XGBoost C (S2+Idx+AEE)         R2=+0.142  RMSE=0.1808  MAE=0.1433")
print("  Pixel Ridge on raw S2+AEE             R2=-0.331  RMSE=0.2251  MAE=0.1781")
print("  Pixel Ridge on raw S2 only            R2=+0.056  RMSE=0.1896  MAE=0.1566")

print()
print("="*65)
print("Done.")
print("="*65)
