import rasterio
import numpy as np
from PIL import Image

def convert_tif_to_png(tif_path="type2_export.tif", png_path="type2_export.png", max_val=3000.0):
    """
    Reads the 10-band GeoTIFF, extracts the True Color bands (Red, Green, Blue),
    scales them to 0-255, adds an alpha channel for transparency, and saves as PNG.
    """
    print(f"Reading {tif_path}...")
    with rasterio.open(tif_path) as src:
        # Based on typical Sentinel-2 band mappings:
        # Band 1: B2 (Blue)
        # Band 2: B3 (Green)
        # Band 3: B4 (Red)
        b = src.read(1)
        g = src.read(2)
        r = src.read(3)
        
    print("Scaling channels and applying transparency mask...")
    # Create valid mask (true where data is NOT NaN)
    valid_mask = ~np.isnan(r)
    
    # Replace NaN with 0 so math works cleanly
    r = np.nan_to_num(r, nan=0.0)
    g = np.nan_to_num(g, nan=0.0)
    b = np.nan_to_num(b, nan=0.0)
    
    # Scale raw reflectance values (usually 0-3000ish) to 8-bit RGB (0-255)
    r = np.clip((r / max_val) * 255, 0, 255).astype(np.uint8)
    g = np.clip((g / max_val) * 255, 0, 255).astype(np.uint8)
    b = np.clip((b / max_val) * 255, 0, 255).astype(np.uint8)
    
    # Alpha channel: 255 (opaque) where valid, 0 (transparent) where NaN
    a = (valid_mask * 255).astype(np.uint8)
    
    # Stack into a 3D array: (Height, Width, 4 Channels)
    rgba = np.stack([r, g, b, a], axis=-1)
    
    print(f"Saving to {png_path}...")
    img = Image.fromarray(rgba, 'RGBA')
    img.save(png_path)
    print("Done! Conversion complete.")

if __name__ == "__main__":
    convert_tif_to_png()
