import pandas as pd
import numpy as np
import rasterio
from rasterio.transform import from_bounds
import os

def generate_bare_soil_tiff(parquet_path="jalandhar_202403_march.parquet", output_path="bare_soil_export.tif"):
    """
    Reads the parquet file, extracts all 10 bands for bare soil pixels,
    and writes them to a multi-band, georeferenced GeoTIFF.
    """
    print(f"Loading {parquet_path}...")
    columns = ['latitude', 'longitude', 'target', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12']
    df = pd.read_parquet(parquet_path, columns=columns)
    
    # Filter for bare soil pixels only
    print("Filtering bare soil pixels...")
    df_bare = df[df['target'] == 1.0]
    
    if df_bare.empty:
        raise ValueError("No bare soil pixels found in the dataset.")
        
    print("Constructing spatial grid...")
    # Get all unique coordinates from the original dataframe to preserve the exact grid size
    lats = np.unique(df['latitude'].values)
    lons = np.unique(df['longitude'].values)
    
    # Sort lats descending (North to South) and lons ascending (West to East)
    lats_desc = lats[::-1]
    
    height = len(lats)
    width = len(lons)
    
    min_lat, max_lat = lats.min(), lats.max()
    min_lon, max_lon = lons.min(), lons.max()
    
    # Determine pixel resolution
    res_y = (max_lat - min_lat) / (height - 1)
    res_x = (max_lon - min_lon) / (width - 1)
    
    # from_bounds(west, south, east, north, width, height)
    # Expand bounds by half a pixel so the min/max coordinates represent pixel centers
    west = min_lon - (res_x / 2)
    east = max_lon + (res_x / 2)
    south = min_lat - (res_y / 2)
    north = max_lat + (res_y / 2)
    
    transform = from_bounds(west, south, east, north, width, height)
    print(f"Grid size: {width}x{height}")
    
    # Map lat/lon to indices using numpy searchsorted for extreme speed
    # lats is ascending, so we can search against it and then invert for top-to-bottom
    y_indices_asc = np.searchsorted(lats, df_bare['latitude'].values)
    y_indices = (height - 1) - y_indices_asc
    
    x_indices = np.searchsorted(lons, df_bare['longitude'].values)
    
    bands = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12']
    
    # Create the profile for the GeoTIFF
    profile = {
        'driver': 'GTiff',
        'dtype': 'float32',
        'nodata': np.nan,
        'width': width,
        'height': height,
        'count': len(bands),
        'crs': 'EPSG:4326',
        'transform': transform,
        'compress': 'lzw', # Excellent compression for data with lots of NaNs
        'tiled': True
    }
    
    print(f"Writing to {output_path}...")
    with rasterio.open(output_path, 'w', **profile) as dst:
        for idx, band in enumerate(bands, start=1):
            print(f"Writing band {band}...")
            # Create a 2D array filled with NaN
            band_data = np.full((height, width), np.nan, dtype=np.float32)
            
            # Populate with bare soil values
            band_data[y_indices, x_indices] = df_bare[band].values.astype(np.float32)
            
            # Write band
            dst.write(band_data, idx)
            dst.set_band_description(idx, band)
            
    print("Export complete!")
    return output_path

if __name__ == "__main__":
    generate_bare_soil_tiff()
