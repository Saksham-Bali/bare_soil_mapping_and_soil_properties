"""
After all GEE tasks complete and CSVs are downloaded from Drive,
run this script to merge all per-polygon CSVs into one ML-ready file.

Usage:
    python merge_csvs.py --input_folder path/to/downloaded/csvs --output merged_s2_soil.csv
"""

import pandas as pd
import glob
import os
import argparse

def merge_polygon_csvs(input_folder, output_path):
    csv_files = sorted(glob.glob(os.path.join(input_folder, '*.csv')))
    if not csv_files:
        raise ValueError(f"No CSVs found in {input_folder}")

    dfs = []
    for path in csv_files:
        fname = os.path.basename(path)
        try:
            df = pd.read_csv(path)
            if df.empty:
                print(f"  SKIP (empty): {fname}")
                continue
            # Extract district and soil_type from filename if not already columns
            if 'soil_type' not in df.columns or 'district' not in df.columns:
                parts     = fname.replace('.csv', '').split('_')
                soil_type = parts[-2]   # second-to-last (last is idx)
                district  = '_'.join(parts[:-2])
                df['soil_type'] = soil_type
                df['district']  = district
            dfs.append(df)
            print(f"  OK ({len(df)} pixels): {fname}")
        except Exception as e:
            print(f"  ERROR: {fname} -> {e}")

    merged = pd.concat(dfs, ignore_index=True)

    # Drop GEE system columns if present
    drop_cols = [c for c in merged.columns if c.startswith('system:')]
    merged.drop(columns=drop_cols, inplace=True, errors='ignore')

    merged.to_csv(output_path, index=False)
    print(f"\nMerged {len(dfs)} files -> {len(merged)} total pixels")
    print(f"Columns: {list(merged.columns)}")
    print(f"Soil type distribution:\n{merged['soil_type'].value_counts()}")
    print(f"\nSaved to: {output_path}")
    return merged

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_folder', required=True,
                        help='Folder containing downloaded per-polygon CSVs')
    parser.add_argument('--output', default='merged_s2_soil.csv',
                        help='Output merged CSV path')
    args = parser.parse_args()

    merge_polygon_csvs(args.input_folder, args.output)
