"""
old_oct_analysis.py
====================
1. Takes all 40 old October parquet files from s2_old/parquet_exports/
2. Applies dropna (equivalent to dropNulls:true in GEE) on band columns
3. Saves cleaned CSVs to s2_old_oct_cleaned/
4. Compares pixel counts: old_raw vs old_clean vs new_csv
5. Runs bare soil XGBoost predictions on the cleaned old files
6. Saves predictions to s2_old_oct_bare/
7. Prints a side-by-side summary of bare% across old vs new
"""

import os, glob, re, gc
import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings('ignore')

BASE          = os.path.dirname(os.path.abspath(__file__))
OLD_FOLDER    = os.path.join(BASE, 's2_old', 'parquet_exports')
NEW_FOLDER    = os.path.join(BASE, 's2')
CLEAN_FOLDER  = os.path.join(BASE, 's2_old_oct_cleaned')   # new folder: cleaned old CSVs
BARE_FOLDER   = os.path.join(BASE, 's2_old_oct_bare')       # new folder: predictions on old
MODEL_PATH    = os.path.join(BASE, '..', 'ml_bare_soil_pred', 'xgb_bare_soil.joblib')

BANDS         = ['B1','B2','B3','B4','B5','B6','B7','B8','B8A','B9','B11','B12']
BAND_COLS     = ['B11','B12','B2','B3','B4','B5','B6','B7','B8','B8A']
FEATURE_COLS  = BAND_COLS + ['BSI','NDVI','NDWI','SWIR_ratio','brightness']
THRESHOLD     = 0.5
EPS           = 1e-8

os.makedirs(CLEAN_FOLDER, exist_ok=True)
os.makedirs(BARE_FOLDER,  exist_ok=True)

# ── helpers ──────────────────────────────────────────────────────────────────

def compute_indices(df):
    df['BSI']        = ((df['B11']+df['B4'])-(df['B8']+df['B2'])) / \
                       ((df['B11']+df['B4'])+(df['B8']+df['B2'])+EPS)
    df['NDVI']       = (df['B8']-df['B4'])  / (df['B8']+df['B4']+EPS)
    df['NDWI']       = (df['B3']-df['B8'])  / (df['B3']+df['B8']+EPS)
    df['SWIR_ratio'] = df['B11']            / (df['B12']+EPS)
    df['brightness'] = (df['B2']+df['B3']+df['B4']+df['B8']) / 4.0
    return df

# ── 1. Load model ─────────────────────────────────────────────────────────────
print("Loading XGBoost model...")
model = joblib.load(MODEL_PATH)
print("Model loaded OK\n")

# ── 2. Gather old October files ───────────────────────────────────────────────
old_files = sorted(glob.glob(os.path.join(OLD_FOLDER, '*_2025_10.parquet')))
print(f"Found {len(old_files)} old October parquet files.\n")

# ── 3. Gather new CSV files for comparison ────────────────────────────────────
new_files = {int(re.search(r'plot(\d+)', f).group(1)): f
             for f in glob.glob(os.path.join(NEW_FOLDER, 'SOC_Poly_combined_plot*_2025.csv'))
             if re.search(r'plot(\d+)', f)}

# ── 4. Process each old October file ─────────────────────────────────────────
summary = []

print(f"{'Plot':>4} | {'Old raw':>8} | {'Old clean':>9} | {'Dropped':>7} | {'New csv':>7} | "
      f"{'Old bare':>8} | {'Old bare%':>9} | {'New bare%':>9}")
print("-" * 85)

for fp_path in old_files:
    fname  = os.path.basename(fp_path)
    m      = re.search(r'SOC_Poly_(\d+)_2025_10', fname)
    if not m:
        continue
    poly_id = int(m.group(1))

    # Load old parquet
    df_old = pd.read_parquet(fp_path)
    n_raw  = len(df_old)

    # Drop nulls on band columns (mirrors GEE dropNulls:true)
    df_clean = df_old.dropna(subset=BANDS).reset_index(drop=True)
    n_clean  = len(df_clean)
    n_drop   = n_raw - n_clean

    # Get new CSV pixel count for this plot
    n_new = len(pd.read_csv(new_files[poly_id])) if poly_id in new_files else -1

    # ── Save cleaned CSV ──────────────────────────────────────────────────────
    out_clean = os.path.join(CLEAN_FOLDER, f'SOC_Poly_{poly_id}_2025_10_clean.csv')
    df_clean.to_csv(out_clean, index=False)

    # ── Run bare soil prediction ──────────────────────────────────────────────
    df_pred = df_clean.copy()
    df_pred  = compute_indices(df_pred)

    miss = [c for c in FEATURE_COLS if c not in df_pred.columns]
    if miss:
        print(f"  [SKIP poly {poly_id}] missing: {miss}")
        continue

    X     = df_pred[FEATURE_COLS].values.astype(np.float32)
    probs = model.predict_proba(X)[:, 1]
    preds = (probs >= THRESHOLD).astype(np.int8)

    df_pred['bare_soil_prob'] = probs.astype(np.float32)
    df_pred['prediction']     = preds

    out_bare = os.path.join(BARE_FOLDER, f'SOC_Poly_{poly_id}_2025_10_bare.csv')
    df_pred.to_csv(out_bare, index=False)

    n_bare     = int(preds.sum())
    bare_pct   = 100.0 * n_bare / max(n_clean, 1)

    # Get new CSV bare% if available
    new_bare_f = os.path.join(BASE, 's2_bare', f'SOC_Poly_combined_plot{poly_id}_2025_bare.csv')
    if os.path.exists(new_bare_f):
        ndf        = pd.read_csv(new_bare_f)
        new_bare_n = int(ndf['prediction'].sum()) if 'prediction' in ndf.columns else -1
        new_bare_p = 100.0 * new_bare_n / max(len(ndf), 1)
    else:
        new_bare_n, new_bare_p = -1, -1.0

    print(f"{poly_id:>4} | {n_raw:>8,} | {n_clean:>9,} | {n_drop:>7,} | {n_new:>7,} | "
          f"{n_bare:>8,} | {bare_pct:>8.1f}% | {new_bare_p:>8.1f}%")

    summary.append({
        'poly_id'    : poly_id,
        'n_old_raw'  : n_raw,
        'n_old_clean': n_clean,
        'n_dropped'  : n_drop,
        'n_new'      : n_new,
        'old_bare'   : n_bare,
        'old_bare_pct': round(bare_pct, 1),
        'new_bare_pct': round(new_bare_p, 1),
    })

    del df_old, df_clean, df_pred, X, probs, preds
    gc.collect()

# ── 5. Overall summary ────────────────────────────────────────────────────────
print()
print("=" * 85)
sdf = pd.DataFrame(summary).set_index('poly_id')
total_old_raw   = sdf['n_old_raw'].sum()
total_old_clean = sdf['n_old_clean'].sum()
total_new       = sdf['n_new'].sum()
total_dropped   = sdf['n_dropped'].sum()

print(f"Total old raw pixels    : {total_old_raw:,}")
print(f"Total old clean pixels  : {total_old_clean:,}  ({100*total_old_clean/total_old_raw:.1f}% retained)")
print(f"Total nulls dropped     : {total_dropped:,}  ({100*total_dropped/total_old_raw:.1f}% dropped)")
print(f"Total new csv pixels    : {total_new:,}")
print()

if total_dropped == 0:
    print("[FINDING] Old parquet files have ZERO null values.")
    print("          dropNulls had no effect on old data.")
    print("          Pixel count difference is NOT due to cloud masking in the data,")
    print("          but due to a difference in GEE sampling grid/boundary between exports.")
else:
    pct_kept = 100.0 * total_old_clean / total_old_raw
    print(f"[FINDING] dropNulls reduced old pixel count to {pct_kept:.1f}%.")
    print(f"          New CSV has {total_new:,} pixels vs old clean {total_old_clean:,}.")

print()
print(f"Cleaned CSVs saved to : {os.path.abspath(CLEAN_FOLDER)}")
print(f"Predictions saved to  : {os.path.abspath(BARE_FOLDER)}")
print()
print("Bare% comparison (old clean vs new):")
print(sdf[['n_old_clean', 'n_new', 'old_bare_pct', 'new_bare_pct']].to_string())
