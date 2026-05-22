import pandas as pd
import numpy as np
from PIL import Image
import time
import json

def process_data():
    t0 = time.time()
    print("Loading parquet file (this may take a minute)...")
    # We load B4 (Red), B3 (Green), B2 (Blue) and target
    df = pd.read_parquet('jalandhar_202403_march.parquet', columns=['latitude', 'longitude', 'B4', 'B3', 'B2', 'target'])
    print(f"Loaded {len(df)} rows in {time.time() - t0:.2f} seconds.")

    print("Mapping coordinates to grid...")
    lats_unique, y_indices_asc = np.unique(df['latitude'].values, return_inverse=True)
    lons_unique, x_indices = np.unique(df['longitude'].values, return_inverse=True)

    # Image Y axis goes from top to bottom, so we invert the Y indices
    # (latitude max is top, latitude min is bottom)
    y_indices = (len(lats_unique) - 1) - y_indices_asc

    h = len(lats_unique)
    w = len(lons_unique)
    print(f"Grid size: {h} (H) x {w} (W) = {h * w} pixels")

    # Create empty arrays
    rgb = np.zeros((h, w, 4), dtype=np.uint8) # RGBA
    mask = np.zeros((h, w, 4), dtype=np.uint8) # RGBA

    # --- Process RGB ---
    print("Processing True Color (RGB) image...")
    def normalize_band(b):
        p2, p98 = np.nanpercentile(b, (2, 98))
        b_norm = np.clip((b - p2) / (p98 - p2), 0, 1) * 255
        return np.nan_to_num(b_norm, nan=0).astype(np.uint8)

    rgb[y_indices, x_indices, 0] = normalize_band(df['B4'].values) # Red
    rgb[y_indices, x_indices, 1] = normalize_band(df['B3'].values) # Green
    rgb[y_indices, x_indices, 2] = normalize_band(df['B2'].values) # Blue
    rgb[y_indices, x_indices, 3] = 255 # Alpha channel fully opaque where data exists

    # --- Process Mask ---
    print("Processing Target Mask image...")
    target_vals = df['target'].fillna(-1).values

    # Swapped class colors: target=1.0 -> Green (Cropland), target=2.0 -> Brown (Bare Soil)
    # In the Bare Soil overlay, only brown (bare soil) pixels are made visible.
    r_mask = np.zeros(len(df), dtype=np.uint8)
    g_mask = np.zeros(len(df), dtype=np.uint8)
    b_mask = np.zeros(len(df), dtype=np.uint8)
    a_mask = np.zeros(len(df), dtype=np.uint8)

    cropland = (target_vals == 1.0)
    bare_soil = (target_vals == 2.0)

    # Green for cropland (swapped)
    r_mask[cropland] = 34
    g_mask[cropland] = 139
    b_mask[cropland] = 34

    # Brown for bare soil (swapped)
    r_mask[bare_soil] = 139  # SaddleBrown
    g_mask[bare_soil] = 69
    b_mask[bare_soil] = 19

    # Keep only brown bare-soil pixels visible in the bare mask layer.
    a_mask[bare_soil] = 255

    mask[y_indices, x_indices, 0] = r_mask
    mask[y_indices, x_indices, 1] = g_mask
    mask[y_indices, x_indices, 2] = b_mask
    mask[y_indices, x_indices, 3] = a_mask

    # Save bounds for Streamlit to know where to place the images
    bounds = {
        "min_lat": float(lats_unique.min()),
        "max_lat": float(lats_unique.max()),
        "min_lon": float(lons_unique.min()),
        "max_lon": float(lons_unique.max())
    }
    with open('bounds.json', 'w') as f:
        json.dump(bounds, f)

    print("Saving PNG files...")
    Image.fromarray(rgb).save('rgb_overlay.png')
    Image.fromarray(mask).save('mask_overlay.png')

    print(f"Done in {time.time() - t0:.2f} seconds!")

if __name__ == "__main__":
    process_data()
