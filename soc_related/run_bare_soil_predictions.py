"""
Bare Soil Prediction Script
============================
Loads the trained XGBoost model and runs pixel-wise bare soil predictions
on all 40 Sentinel-2 CSV files in soc_related/s2/.
Saves results (original columns + bare_soil_prob + prediction) to soc_related/s2_bare/.

Prediction label:
  1 = Bare soil
  0 = Not bare soil (e.g. crop / vegetation)
"""

import os
import gc
import glob
import numpy as np
import pandas as pd
import joblib

# ── CONFIG ────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH    = os.path.join(BASE_DIR, '..', 'ml_bare_soil_pred', 'xgb_bare_soil.joblib')
INPUT_FOLDER  = os.path.join(BASE_DIR, 's2')
OUTPUT_FOLDER = os.path.join(BASE_DIR, 's2_bare')
THRESHOLD     = 0.5   # Probability cutoff for bare-soil label

# Feature columns — must match training exactly
BAND_COLS    = ['B11', 'B12', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A']
FEATURE_COLS = BAND_COLS + ['BSI', 'NDVI', 'NDWI', 'SWIR_ratio', 'brightness']
EPS          = 1e-8
# ──────────────────────────────────────────────────────────────


def compute_indices(df: pd.DataFrame) -> pd.DataFrame:
    """Compute derived spectral indices in-place (matches training logic)."""
    df['BSI']        = ((df['B11'] + df['B4']) - (df['B8'] + df['B2'])) / \
                       ((df['B11'] + df['B4']) + (df['B8'] + df['B2']) + EPS)
    df['NDVI']       = (df['B8'] - df['B4'])  / (df['B8']  + df['B4']  + EPS)
    df['NDWI']       = (df['B3'] - df['B8'])  / (df['B3']  + df['B8']  + EPS)
    df['SWIR_ratio'] = df['B11']              / (df['B12']              + EPS)
    df['brightness'] = (df['B2'] + df['B3'] + df['B4'] + df['B8']) / 4.0
    return df


def predict_file(model, input_path: str, output_path: str) -> None:
    """Run prediction on one CSV file and save result."""
    file_name = os.path.basename(input_path)
    print(f'  Processing: {file_name} ...', end='', flush=True)

    # Load CSV
    df = pd.read_csv(input_path)
    n_pixels = len(df)

    # Compute spectral indices
    df = compute_indices(df)

    # Check that all required feature columns are present
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        print(f'\n    [SKIP] Missing columns: {missing}')
        return

    # Prepare feature matrix
    X = df[FEATURE_COLS].values.astype(np.float32)

    # Inference
    probs = model.predict_proba(X)[:, 1]          # probability of bare soil
    preds = (probs >= THRESHOLD).astype(np.int8)  # 1=bare, 0=not bare

    df['bare_soil_prob'] = probs.astype(np.float32)
    df['prediction']     = preds                   # 1=Bare, 0=Not-bare

    # Drop the temporary index columns if they were added by compute_indices
    # (none were, but keeping this for safety)
    df.to_csv(output_path, index=False)

    n_bare = int(preds.sum())
    pct    = 100.0 * n_bare / max(n_pixels, 1)
    print(f' done  | pixels={n_pixels:,}  bare={n_bare:,} ({pct:.1f}%)')
    del df, X, probs, preds
    gc.collect()


def main():
    # Create output folder
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # Load model
    print(f'Loading model from:\n  {os.path.abspath(MODEL_PATH)}\n')
    model = joblib.load(MODEL_PATH)
    print('Model loaded OK\n')

    # Gather input files
    input_files = sorted(glob.glob(os.path.join(INPUT_FOLDER, '*.csv')))
    if not input_files:
        print(f'No CSV files found in: {INPUT_FOLDER}')
        return

    print(f'Found {len(input_files)} file(s) to process.')
    print(f'Output folder: {os.path.abspath(OUTPUT_FOLDER)}\n')

    success, failed = 0, []

    for input_path in input_files:
        # Build output filename:  SOC_Poly_combined_plot1_2025.csv
        #                    →    SOC_Poly_combined_plot1_2025_bare.csv
        stem     = os.path.splitext(os.path.basename(input_path))[0]  # e.g. SOC_Poly_combined_plot1_2025
        out_name = f'{stem}_bare.csv'
        out_path = os.path.join(OUTPUT_FOLDER, out_name)

        try:
            predict_file(model, input_path, out_path)
            success += 1
        except Exception as e:
            print(f'\n    [ERROR] {os.path.basename(input_path)}: {e}')
            failed.append(os.path.basename(input_path))

    print(f'\n{"-"*55}')
    print(f'[DONE] Completed: {success}/{len(input_files)} files')
    if failed:
        print(f'[FAIL] Failed   : {len(failed)} file(s)')
        for f in failed:
            print(f'     - {f}')
    print(f'Results saved to: {os.path.abspath(OUTPUT_FOLDER)}')


if __name__ == '__main__':
    main()
