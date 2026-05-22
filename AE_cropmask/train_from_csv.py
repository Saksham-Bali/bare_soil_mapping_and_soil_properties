import rasterio
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from rasterio.windows import Window
import joblib
import sys

# Paths
csv_path = '/Users/aaravsinghluthra/Desktop/EarthEngine_Exports 2/final.csv'
mosaic_path = '/Users/aaravsinghluthra/Desktop/EarthEngine_Exports 2/merged_mosaic.tif'
output_path = '/Users/aaravsinghluthra/Desktop/EarthEngine_Exports 2/jalandhar_full_mask_from_csv.tif'
model_path = '/Users/aaravsinghluthra/Desktop/EarthEngine_Exports 2/rf_model_from_csv.pkl'

# Band Mapping (A00..A63 -> Band Index 0..63)
# Deduced from analysis: Bands are sorted by string index ("0", "1", "10", "11"...)
BAND_INDICES = [
    0, 1, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 2, 20, 21, 22, 
    23, 24, 25, 26, 27, 28, 29, 3, 30, 31, 32, 33, 34, 35, 36, 37, 
    38, 39, 4, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 5, 50, 51, 
    52, 53, 54, 55, 56, 57, 58, 59, 6, 60, 61, 62, 63, 7, 8, 9
]

def train_model():
    print(f"Loading training data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Features: A00 to A63
    feature_cols = [f'A{i:02d}' for i in range(64)]
    
    X = df[feature_cols].values
    y = df['class'].values
    
    print(f"Training data shape: X={X.shape}, y={y.shape}")
    
    print("Training Random Forest...")
    clf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
    clf.fit(X, y)
    
    print("Model trained.")
    joblib.dump(clf, model_path)
    return clf

def predict_full_raster(clf):
    print("Starting full district prediction (Windowed)...")
    print("Applying band reordering to match CSV columns.")
    
    with rasterio.open(mosaic_path) as src:
        profile = src.profile.copy()
        profile.update(
            dtype=rasterio.uint8,
            count=1,
            nodata=255,
            compress='lzw'
        )
        
        width = src.width
        height = src.height
        block_size = 2048
        
        with rasterio.open(output_path, 'w', **profile) as dst:
            for j in range(0, height, block_size):
                for i in range(0, width, block_size):
                    w = min(block_size, width - i)
                    h = min(block_size, height - j)
                    window = Window(i, j, w, h)
                    
                    print(f"Processing window: col={i}, row={j}", end='\r')
                    sys.stdout.flush()
                    
                    # Read data (64, h, w)
                    img_data = src.read(window=window) 
                    
                    # Reorder bands
                    # img_data[k] should be the band corresponding to Ak
                    # We use BAND_INDICES to select the correct bands
                    ordered_data = img_data[BAND_INDICES, :, :]
                    
                    n_bands, nh, nw = ordered_data.shape
                    
                    # Reshape to (n_pixels, n_features)
                    features = ordered_data.transpose(1, 2, 0).reshape(-1, n_bands)
                    
                    # 1. Mask NaNs
                    nan_mask = np.isnan(features).any(axis=1)
                    
                    # 2. Mask Zeros
                    zero_mask = np.all(features == 0, axis=1)
                    
                    # Combined invalid mask
                    invalid_mask = nan_mask | zero_mask
                    
                    # Initialize prediction array with nodata (255)
                    preds_flat = np.full(features.shape[0], 255, dtype=np.uint8)
                    
                    # Predict only on valid pixels
                    valid_indices = ~invalid_mask
                    
                    if np.any(valid_indices):
                        valid_features = features[valid_indices]
                        # Predict
                        valid_preds = clf.predict(valid_features)
                        preds_flat[valid_indices] = valid_preds
                    
                    # Reshape back to image
                    preds_img = preds_flat.reshape(1, nh, nw)
                    
                    dst.write(preds_img, window=window)
                    
    print("\nPrediction complete.")
    print(f"Output saved to: {output_path}")

if __name__ == "__main__":
    print(f"Loading model from {model_path}...")
    model = joblib.load(model_path)
    predict_full_raster(model)
