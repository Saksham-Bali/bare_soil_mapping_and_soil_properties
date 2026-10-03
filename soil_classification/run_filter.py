import sys, os, glob, warnings, joblib, numpy as np, pandas as pd
warnings.filterwarnings('ignore')

INPUT     = r'D:\Bare_soil\Bare_Soil_Mapping\soil_classification\S2_Soil_CSV_PerPolygon'
MODEL     = r'D:\Bare_soil\Bare_Soil_Mapping\ml_bare_soil_pred\xgb_bare_soil.joblib'
OUTPUT    = r'D:\Bare_soil\Bare_Soil_Mapping\soil_classification\bare_soil_pixels.parquet'
THRESHOLD = 0.5

BAND_COLS    = ['B11','B12','B2','B3','B4','B5','B6','B7','B8','B8A']
INDEX_COLS   = ['BSI','NDVI','NDWI','SWIR_ratio','brightness']
FEATURE_COLS = BAND_COLS + INDEX_COLS

def compute_indices(df):
    eps = 1e-8
    df['BSI']        = ((df['B11']+df['B4'])-(df['B8']+df['B2'])) / ((df['B11']+df['B4'])+(df['B8']+df['B2'])+eps)
    df['NDVI']       = (df['B8']-df['B4'])  / (df['B8']+df['B4']+eps)
    df['NDWI']       = (df['B3']-df['B8'])  / (df['B3']+df['B8']+eps)
    df['SWIR_ratio'] = df['B11'] / (df['B12']+eps)
    df['brightness'] = (df['B2']+df['B3']+df['B4']+df['B8']) / 4.0
    return df

model = joblib.load(MODEL)
print("Model loaded  |  classes=%s  |  n_features=%d" % (model.classes_, model.n_features_in_))

csv_files = sorted(glob.glob(os.path.join(INPUT, '*.csv')))
print("Found %d CSV files\n" % len(csv_files))
print("%-50s %8s %8s %7s" % ("File", "Total", "Bare", "Pct"))
print("-" * 76)

bare_dfs   = []
total_px   = 0
total_bare = 0
zero_files = []

for path in csv_files:
    fname = os.path.basename(path)
    try:
        df = pd.read_csv(path)
        # Extract polygon ID (e.g., 'poly0' from 'Amritsar_s1_poly0.csv')
        poly_str = fname.replace('.csv', '').split('_')[-1]
        df['polygon_id'] = poly_str

    except Exception as e:
        print("  ERROR: %s: %s" % (fname, e))
        continue

    df.drop(columns=[c for c in df.columns if c.startswith('system:')], inplace=True, errors='ignore')
    df.dropna(subset=BAND_COLS, inplace=True)
    n_total = len(df)
    total_px += n_total

    if n_total == 0:
        zero_files.append((fname, 'empty after NaN drop'))
        print("  %-48s %8d  EMPTY" % (fname, 0))
        continue

    df = compute_indices(df)
    X = df[FEATURE_COLS].values.astype(np.float32)
    probs = model.predict_proba(X)[:, 1]
    df['bare_soil_prob'] = probs.astype('float32')
    df['bare_soil_pred'] = (probs >= THRESHOLD).astype('int8')

    bare_df = df[df['bare_soil_pred'] == 1].copy()
    bare_df.drop(columns=INDEX_COLS, inplace=True, errors='ignore')

    n_bare = len(bare_df)
    total_bare += n_bare
    pct = 100.0 * n_bare / n_total

    print("  %-48s %8d %8d %6.1f%%" % (fname, n_total, n_bare, pct))

    if n_bare == 0:
        zero_files.append((fname, "%d px in, 0 bare out" % n_total))
    else:
        bare_dfs.append(bare_df)

print()
print("=" * 76)
print("TOTAL pixels processed  : %d" % total_px)
print("TOTAL bare pixels kept  : %d" % total_bare)
dropped = total_px - total_bare
print("TOTAL dropped (non-bare): %d  (%.1f%%)" % (dropped, 100.0*dropped/max(total_px,1)))

if zero_files:
    print("\nPolygons with 0 bare pixels (%d):" % len(zero_files))
    for f, reason in zero_files:
        print("  - %s  [%s]" % (f, reason))
else:
    print("\nAll polygons have at least 1 bare pixel.")

if bare_dfs:
    merged = pd.concat(bare_dfs, ignore_index=True)
    
    # Merge s3 and s4 into a unified sandy loam class based on physical and spectral similarity
    mapping = {
        's1': 's1_loam',
        's2': 's2_loam_clay',
        's3': 's3_sandy_loam',
        's4': 's3_sandy_loam'
    }
    merged['soil_type'] = merged['soil_type'].map(mapping).fillna(merged['soil_type'])

    print("\nSoil type distribution (bare pixels only, after merge):")
    print(merged['soil_type'].value_counts().to_string())
    print("\nDistrict distribution (bare pixels only):")
    print(merged['district'].value_counts().to_string())
    merged.to_parquet(OUTPUT, index=False, compression='snappy')
    size_mb = os.path.getsize(OUTPUT) / 1e6
    print("\nSaved: %s  (%.2f MB)" % (OUTPUT, size_mb))
    print("Columns: %s" % list(merged.columns))
else:
    print("No bare pixels found -- parquet not written.")
