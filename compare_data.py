"""
compare_new_vs_old.py
=====================
Compares the new s2 CSV data (soc_related/s2/) vs the October files from
the old parquet data (soc_related/s2_old/parquet_exports/).

Strategy:
  - Extract October (_10) files from old data
  - For each, compute: pixel count, lat centroid, lon centroid, mean B4, mean B8
  - Do the same for each new CSV
  - Attempt to match new <-> old by (1) pixel count, (2) spatial centroid proximity
  - Report whether new plot N == old poly N, or if there is a mapping shift
"""

import os
import glob
import re
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
NEW_FOLDER = os.path.join(BASE, 'soc_related', 's2')
OLD_FOLDER = os.path.join(BASE, 'soc_related', 's2_old', 'parquet_exports')

BAND_COMPARE = ['B2', 'B3', 'B4', 'B8']  # bands present in both datasets

# ── helpers ──────────────────────────────────────────────────────────────────

def fingerprint_df(df, label):
    """Return a dict of key statistics for matching."""
    n = len(df)
    fp = {
        'label'      : label,
        'n_pixels'   : n,
        'lat_mean'   : round(df['latitude'].mean(), 5) if 'latitude' in df.columns else np.nan,
        'lon_mean'   : round(df['longitude'].mean(), 5) if 'longitude' in df.columns else np.nan,
        'lat_min'    : round(df['latitude'].min(), 5)  if 'latitude' in df.columns else np.nan,
        'lat_max'    : round(df['latitude'].max(), 5)  if 'latitude' in df.columns else np.nan,
        'lon_min'    : round(df['longitude'].min(), 5) if 'longitude' in df.columns else np.nan,
        'lon_max'    : round(df['longitude'].max(), 5) if 'longitude' in df.columns else np.nan,
    }
    for b in BAND_COMPARE:
        if b in df.columns:
            fp[f'{b}_mean'] = round(float(df[b].mean()), 2)
    return fp


def centroid_dist(fp1, fp2):
    """Euclidean distance in lat/lon degrees between two fingerprints."""
    dlat = fp1['lat_mean'] - fp2['lat_mean']
    dlon = fp1['lon_mean'] - fp2['lon_mean']
    return (dlat**2 + dlon**2) ** 0.5


# ── 1. Load OLD October parquet files ────────────────────────────────────────
print("=" * 60)
print("Loading OLD October parquet files...")
print("=" * 60)

old_oct_files = sorted(glob.glob(os.path.join(OLD_FOLDER, '*_2025_10.parquet')))
print(f"Found {len(old_oct_files)} October files in old data.\n")

old_fps = {}
for fp_path in old_oct_files:
    fname = os.path.basename(fp_path)
    # e.g. SOC_Poly_1_2025_10.parquet  -> poly_id = 1
    m = re.search(r'SOC_Poly_(\d+)_2025_10', fname)
    if not m:
        continue
    poly_id = int(m.group(1))
    try:
        df = pd.read_parquet(fp_path)
        old_fps[poly_id] = fingerprint_df(df, f'OLD_poly_{poly_id}')
        old_fps[poly_id]['file'] = fname
        del df
    except Exception as e:
        print(f"  [ERROR] reading {fname}: {e}")

print(f"Fingerprinted {len(old_fps)} old polygons.\n")

# ── 2. Load NEW CSV files ────────────────────────────────────────────────────
print("=" * 60)
print("Loading NEW CSV files (October download)...")
print("=" * 60)

new_csv_files = sorted(glob.glob(os.path.join(NEW_FOLDER, 'SOC_Poly_combined_plot*_2025.csv')))
print(f"Found {len(new_csv_files)} files in new data.\n")

new_fps = {}
for fp_path in new_csv_files:
    fname = os.path.basename(fp_path)
    # e.g. SOC_Poly_combined_plot1_2025.csv -> plot_id = 1
    m = re.search(r'plot(\d+)_2025', fname)
    if not m:
        continue
    plot_id = int(m.group(1))
    try:
        df = pd.read_csv(fp_path)
        new_fps[plot_id] = fingerprint_df(df, f'NEW_plot_{plot_id}')
        new_fps[plot_id]['file'] = fname
        del df
    except Exception as e:
        print(f"  [ERROR] reading {fname}: {e}")

print(f"Fingerprinted {len(new_fps)} new plots.\n")

# ── 3. Print fingerprint tables ───────────────────────────────────────────────
print("=" * 60)
print("OLD October data - spatial fingerprints:")
print("=" * 60)
old_rows = []
for pid in sorted(old_fps.keys()):
    fp = old_fps[pid]
    old_rows.append({
        'poly_id' : pid,
        'n_pixels': fp['n_pixels'],
        'lat_mean': fp['lat_mean'],
        'lon_mean': fp['lon_mean'],
        'B4_mean' : fp.get('B4_mean', np.nan),
        'B8_mean' : fp.get('B8_mean', np.nan),
    })
old_df = pd.DataFrame(old_rows).set_index('poly_id')
print(old_df.to_string())

print()
print("=" * 60)
print("NEW data - spatial fingerprints:")
print("=" * 60)
new_rows = []
for pid in sorted(new_fps.keys()):
    fp = new_fps[pid]
    new_rows.append({
        'plot_id' : pid,
        'n_pixels': fp['n_pixels'],
        'lat_mean': fp['lat_mean'],
        'lon_mean': fp['lon_mean'],
        'B4_mean' : fp.get('B4_mean', np.nan),
        'B8_mean' : fp.get('B8_mean', np.nan),
    })
new_df = pd.DataFrame(new_rows).set_index('plot_id')
print(new_df.to_string())

# ── 4. Match new -> old by nearest centroid ────────────────────────────────
print()
print("=" * 60)
print("Matching NEW plots -> OLD polys by nearest centroid:")
print("=" * 60)

old_ids = sorted(old_fps.keys())
new_ids = sorted(new_fps.keys())

match_rows = []
for nid in new_ids:
    nfp = new_fps[nid]
    best_oid, best_dist = None, float('inf')
    for oid in old_ids:
        ofp = old_fps[oid]
        d = centroid_dist(nfp, ofp)
        if d < best_dist:
            best_dist = d
            best_oid = oid

    pixel_match = (nfp['n_pixels'] == old_fps[best_oid]['n_pixels'])
    id_match    = (nid == best_oid)
    flag = ''
    if not id_match:
        flag = '  <-- ID MISMATCH: new_plot_{} matched old_poly_{}'.format(nid, best_oid)
    if not pixel_match:
        flag += '  [pixel count differs: new={} old={}]'.format(
            nfp['n_pixels'], old_fps[best_oid]['n_pixels'])

    match_rows.append({
        'new_plot'   : nid,
        'best_old_poly': best_oid,
        'dist_deg'   : round(best_dist, 6),
        'n_new'      : nfp['n_pixels'],
        'n_old'      : old_fps[best_oid]['n_pixels'],
        'id_matches' : id_match,
        'px_matches' : pixel_match,
        'flag'       : flag.strip(),
    })

match_df = pd.DataFrame(match_rows)
print(match_df.to_string(index=False))

# ── 5. Summary ────────────────────────────────────────────────────────────────
print()
print("=" * 60)
print("SUMMARY")
print("=" * 60)

id_ok  = match_df['id_matches'].sum()
id_bad = (~match_df['id_matches']).sum()
px_ok  = match_df['px_matches'].sum()
px_bad = (~match_df['px_matches']).sum()

print(f"Total new plots      : {len(new_ids)}")
print(f"ID correct (plot N == poly N) : {id_ok}")
print(f"ID mismatched        : {id_bad}")
print(f"Pixel count matches  : {px_ok}")
print(f"Pixel count differs  : {px_bad}")

if id_bad == 0 and px_bad == 0:
    print()
    print("[RESULT] Everything looks CORRECT. New plot IDs align with old poly IDs.")
    print("         The data appears to be consistently labeled.")
elif id_bad > 0 and all(match_df['dist_deg'] < 0.001):
    print()
    print("[RESULT] Naming MISMATCH detected but spatial locations match.")
    print("         The data is the same polygons but the plot numbers are shuffled.")
    print("         See 'flag' column above for the true mapping.")
else:
    print()
    print("[RESULT] Discrepancies found. Check flagged rows above.")
    print("         Some plots may have wrong spatial data (not just a rename).")

# ── 6. Band-level spot-check for matched pairs ────────────────────────────────
print()
print("=" * 60)
print("Band-level spot-check (mean B4, B8) for matched pairs:")
print("=" * 60)

spot_rows = []
for row in match_rows:
    nid = row['new_plot']
    oid = row['best_old_poly']
    nfp = new_fps[nid]
    ofp = old_fps[oid]
    b4_diff = round(abs(nfp.get('B4_mean', 0) - ofp.get('B4_mean', 0)), 1)
    b8_diff = round(abs(nfp.get('B8_mean', 0) - ofp.get('B8_mean', 0)), 1)
    spot_rows.append({
        'new_plot': nid, 'old_poly': oid,
        'new_B4': nfp.get('B4_mean'), 'old_B4': ofp.get('B4_mean'), 'dB4': b4_diff,
        'new_B8': nfp.get('B8_mean'), 'old_B8': ofp.get('B8_mean'), 'dB8': b8_diff,
        'ok': 'YES' if b4_diff < 200 and b8_diff < 200 else 'CHECK',
    })

spot_df = pd.DataFrame(spot_rows)
print(spot_df.to_string(index=False))

print()
print("Done. Output above contains all comparison results.")
