"""
The whole model, end to end, in one file.

  raster -> 12 band medians per plot -> PLSR -> SOC %
  validated with leave-one-out CV
"""
import numpy as np
import pandas as pd
import rasterio
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

TIF = "../soc_plots/Monthly_Median_Exports/SOC_Poly_{plot}_2025_10.tif"

# ---------------------------------------------------------------------------
# 1. AVERAGING: turn a plot's raster into 12 numbers
# ---------------------------------------------------------------------------
def band_medians(plot):
    with rasterio.open(TIF.format(plot=plot)) as src:
        a = src.read()                    # (12 bands, rows, cols), uint16
    a = a.astype(float) / 10000.0         # DN -> reflectance 0-1

    # A pixel is valid only if it is non-zero in ALL 12 bands. Zeros are the
    # nodata fill: pixels outside the field boundary, or cloud-masked.
    valid = ~(a == 0).all(axis=0)         # (rows, cols) boolean

    px = a[:, valid]                      # (12, n_valid_pixels)
    return np.median(px, axis=1)          # (12,) one median per band


# MEDIAN, not mean: with 6-190 pixels per plot, a single cloud edge or a road
# pixel would drag a mean noticeably. The median ignores them.
# Note the imagery is already a monthly median composite in time (Earth Engine
# reduced all of October's passes), so this is a second, spatial reduction.

X = np.array([band_medians(p) for p in range(1, 41)])   # (40 plots, 12 bands)
y = pd.read_csv("ground_truth.csv").SOC_pct.values       # (40,)

# ---------------------------------------------------------------------------
# 2. MODEL
# ---------------------------------------------------------------------------
model = Pipeline([
    ("scale", StandardScaler()),          # each band -> mean 0, sd 1
    ("pls", PLSRegression()),             # n_components tuned below
])
grid = {"pls__n_components": [1, 2, 3, 4, 5, 6]}

# ---------------------------------------------------------------------------
# 3. LEAVE-ONE-OUT VALIDATION
# ---------------------------------------------------------------------------
pred = np.zeros(40)
for i in range(40):
    train = np.setdiff1d(np.arange(40), [i])          # the other 39 plots

    # n_components chosen by 5-fold CV *inside* the 39 training plots,
    # so plot i has no influence on any decision.
    search = GridSearchCV(model, grid, cv=KFold(5, shuffle=True, random_state=0),
                          scoring="neg_mean_squared_error")
    search.fit(X[train], y[train])

    pred[i] = search.best_estimator_.predict(X[i:i + 1]).ravel()[0]

rmse = np.sqrt(((y - pred) ** 2).mean())
r2 = 1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)
print(f"R2   = {r2:+.3f}")
print(f"RMSE = {rmse:.3f} % SOC")
print(f"MAE  = {np.abs(y - pred).mean():.3f} % SOC")

# ---------------------------------------------------------------------------
# 4. FINAL MODEL: refit on all 40 plots, then predict a new plot
# ---------------------------------------------------------------------------
final = GridSearchCV(model, grid, cv=KFold(5, shuffle=True, random_state=0)
                     ).fit(X, y).best_estimator_
print(f"components used: {final.named_steps['pls'].n_components}")
print(f"example - plot 7 predicted {final.predict(X[6:7]).ravel()[0]:.2f} % "
      f"(measured {y[6]:.2f} %)")
