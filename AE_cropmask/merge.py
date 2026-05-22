"""
FIXED: Fast and reliable GeoTIFF mosaic merger
Uses rasterio's built-in merge for speed and correctness
"""

import os
import glob
import numpy as np
import rasterio
from rasterio.merge import merge
from rasterio.enums import Resampling
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


def get_tiff_info(tiff_paths):
    """Get information about all GeoTIFF files"""
    print("Analyzing input files...")
    print("="*60)
    
    file_info = []
    total_size_mb = 0
    
    for path in tiff_paths:
        with rasterio.open(path) as src:
            file_size_mb = os.path.getsize(path) / (1024*1024)
            total_size_mb += file_size_mb
            
            info = {
                'path': path,
                'width': src.width,
                'height': src.height,
                'crs': src.crs,
                'transform': src.transform,
                'bounds': src.bounds,
                'n_bands': src.count,
                'dtype': src.dtypes[0],
                'nodata': src.nodata,
                'resolution': src.res,
                'size_mb': file_size_mb
            }
            file_info.append(info)
            
            print(f"\n📄 {os.path.basename(path)}")
            print(f"   Size: {file_size_mb:.2f} MB")
            print(f"   Dimensions: {info['width']} x {info['height']}")
            print(f"   Bands: {info['n_bands']}")
            print(f"   Bounds: ({info['bounds'].left:.2f}, {info['bounds'].bottom:.2f}, "
                  f"{info['bounds'].right:.2f}, {info['bounds'].top:.2f})")
            print(f"   CRS: {info['crs']}")
            print(f"   Resolution: {info['resolution']}")
            print(f"   NoData: {info['nodata']}")
            
            # Check for valid data
            sample = src.read(1, window=rasterio.windows.Window(0, 0, 
                            min(100, src.width), min(100, src.height)))
            if src.nodata is not None:
                valid_count = np.sum(sample != src.nodata)
            else:
                valid_count = np.sum(~np.isnan(sample))
            print(f"   Sample valid pixels: {valid_count}/{sample.size}")
    
    print(f"\n{'='*60}")
    print(f"Total input size: {total_size_mb:.2f} MB")
    print(f"{'='*60}")
    
    return file_info


def check_compatibility(file_info):
    """Check if files can be merged"""
    print("\nChecking compatibility...")
    
    # Check CRS
    first_crs = file_info[0]['crs']
    crs_match = all(info['crs'] == first_crs for info in file_info)
    
    if not crs_match:
        print("⚠️  WARNING: Files have different CRS - will be reprojected")
        for info in file_info:
            if info['crs'] != first_crs:
                print(f"   {os.path.basename(info['path'])}: {info['crs']}")
    else:
        print("✅ All files have same CRS")
    
    # Check bands
    first_bands = file_info[0]['n_bands']
    bands_match = all(info['n_bands'] == first_bands for info in file_info)
    
    if not bands_match:
        print("⚠️  WARNING: Files have different band counts")
        for info in file_info:
            print(f"   {os.path.basename(info['path'])}: {info['n_bands']} bands")
    else:
        print(f"✅ All files have {first_bands} bands")
    
    # Check resolution
    first_res = file_info[0]['resolution']
    res_match = all(
        abs(info['resolution'][0] - first_res[0]) < 1e-6 and 
        abs(info['resolution'][1] - first_res[1]) < 1e-6 
        for info in file_info
    )
    
    if not res_match:
        print("⚠️  WARNING: Files have different resolutions - will be resampled")
        for info in file_info:
            print(f"   {os.path.basename(info['path'])}: {info['resolution']}")
    else:
        print(f"✅ All files have same resolution: {first_res}")
    
    return crs_match and bands_match and res_match


def merge_tiffs_fast(input_folder, output_path, pattern='*.tif',
                     method='first', compression='LZW', 
                     target_nodata=None, resampling=Resampling.nearest):
    """
    Fast and reliable mosaic merge using rasterio's built-in merge function
    
    Parameters:
    -----------
    input_folder : str
        Folder containing GeoTIFF files
    output_path : str
        Path for output merged GeoTIFF
    pattern : str
        File pattern to match (default: '*.tif')
    method : str
        Merge method: 'first', 'last', 'min', 'max', 'sum'
    compression : str
        Compression method: 'LZW', 'DEFLATE', 'ZSTD', or None
    target_nodata : float
        NoData value for output (default: use source nodata)
    resampling : Resampling
        Resampling method for reprojection
    """
    
    print(f"\n{'='*60}")
    print(f"FAST GEOTIFF MOSAIC MERGER")
    print(f"{'='*60}\n")
    
    # Find all TIFF files
    search_pattern = os.path.join(input_folder, pattern)
    tiff_files = sorted(glob.glob(search_pattern))
    
    if len(tiff_files) == 0:
        print(f"❌ No files found matching pattern: {search_pattern}")
        return None
    
    print(f"Found {len(tiff_files)} files to merge\n")
    
    # Get file information
    file_info = get_tiff_info(tiff_files)
    
    # Check compatibility
    is_compatible = check_compatibility(file_info)
    
    print(f"\n{'='*60}")
    print(f"Starting merge with method: '{method}'")
    print(f"{'='*60}\n")
    
    # Open all source files
    src_files_to_mosaic = []
    
    for tiff_file in tiff_files:
        src = rasterio.open(tiff_file)
        src_files_to_mosaic.append(src)
    
    # Perform merge - this is MUCH faster than manual windowing
    print("Merging files (this may take a minute)...")
    
    try:
        mosaic, out_trans = merge(
            src_files_to_mosaic,
            method=method,
            nodata=target_nodata,
            resampling=resampling,
            target_aligned_pixels=True  # Ensures clean pixel alignment
        )
        
        print("✓ Merge operation complete")
        
    except Exception as e:
        print(f"❌ Error during merge: {e}")
        for src in src_files_to_mosaic:
            src.close()
        return None
    
    # Get metadata from first file
    out_meta = src_files_to_mosaic[0].meta.copy()
    
    # Update metadata for output
    out_meta.update({
        "driver": "GTiff",
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "transform": out_trans,
        "compress": compression,
        "tiled": True,
        "blockxsize": 512,
        "blockysize": 512
    })
    
    if target_nodata is not None:
        out_meta['nodata'] = target_nodata
    
    # Check for valid data before writing
    print("\nValidating merged data...")
    nodata_val = out_meta.get('nodata')
    
    if nodata_val is not None:
        valid_pixels = np.sum(mosaic != nodata_val)
    else:
        valid_pixels = np.sum(~np.isnan(mosaic))
    
    total_pixels = mosaic.size
    valid_percent = (valid_pixels / total_pixels) * 100
    
    print(f"Valid pixels: {valid_pixels:,} / {total_pixels:,} ({valid_percent:.2f}%)")
    
    if valid_pixels == 0:
        print("\n❌ ERROR: Merged mosaic contains NO valid data!")
        print("   This usually means:")
        print("   1. All input files have no valid data (check your GEE exports)")
        print("   2. NoData values are not properly set")
        print("   3. Files don't actually overlap geographically")
        
        for src in src_files_to_mosaic:
            src.close()
        return None
    
    # Write output
    print(f"\nWriting output to: {output_path}")
    
    with rasterio.open(output_path, "w", **out_meta) as dest:
        dest.write(mosaic)
    
    # Close all source files
    for src in src_files_to_mosaic:
        src.close()
    
    output_size_mb = os.path.getsize(output_path) / (1024*1024)
    
    print(f"\n{'='*60}")
    print(f"✅ MERGE COMPLETE!")
    print(f"{'='*60}")
    print(f"Output file: {output_path}")
    print(f"Dimensions: {mosaic.shape[2]} x {mosaic.shape[1]} pixels")
    print(f"Bands: {mosaic.shape[0]}")
    print(f"File size: {output_size_mb:.2f} MB")
    print(f"Valid data: {valid_percent:.2f}%")
    
    return output_path


def create_vrt_alternative(tiff_files, vrt_path):
    """
    Create a VRT file as a fast alternative to merging
    VRT provides instant mosaic without creating large files
    """
    
    print(f"\n{'='*60}")
    print("Creating VRT (Virtual Mosaic) - INSTANT!")
    print(f"{'='*60}\n")
    
    # Use GDAL command if available
    try:
        import subprocess
        
        # Build VRT using gdalbuildvrt command
        cmd = ['gdalbuildvrt', vrt_path] + tiff_files
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            vrt_size_kb = os.path.getsize(vrt_path) / 1024
            print(f"✅ VRT created: {vrt_path}")
            print(f"   File size: {vrt_size_kb:.2f} KB")
            print(f"   This provides instant mosaic access!")
            print(f"   Use this with QGIS or rasterio instead of the large merged file")
            return vrt_path
        else:
            print(f"⚠️  gdalbuildvrt failed: {result.stderr}")
            
    except FileNotFoundError:
        print("⚠️  GDAL not found - cannot create VRT")
        print("   Install with: pip install gdal")
    except Exception as e:
        print(f"⚠️  Could not create VRT: {e}")
    
    return None


def verify_mosaic(output_path):
    """Quick verification of the merged file"""
    
    print(f"\n{'='*60}")
    print("VERIFYING OUTPUT")
    print(f"{'='*60}\n")
    
    with rasterio.open(output_path) as src:
        print(f"Dimensions: {src.width} x {src.height}")
        print(f"Bands: {src.count}")
        print(f"Bounds: {src.bounds}")
        print(f"CRS: {src.crs}")
        print(f"NoData: {src.nodata}")
        
        # Check each band
        for band_idx in range(1, min(src.count + 1, 4)):  # Check first 3 bands
            print(f"\nBand {band_idx}:")
            data = src.read(band_idx)
            
            if src.nodata is not None:
                valid_mask = data != src.nodata
            else:
                valid_mask = ~np.isnan(data)
            
            valid_count = np.sum(valid_mask)
            valid_percent = (valid_count / data.size) * 100
            
            print(f"  Valid pixels: {valid_count:,} ({valid_percent:.2f}%)")
            
            if valid_count > 0:
                valid_data = data[valid_mask]
                print(f"  Min: {np.min(valid_data):.4f}")
                print(f"  Max: {np.max(valid_data):.4f}")
                print(f"  Mean: {np.mean(valid_data):.4f}")
            else:
                print(f"  ⚠️  WARNING: No valid data in band {band_idx}!")


def diagnose_input_files(input_folder, pattern='*.tif'):
    """
    Diagnose why input files might have no data
    """
    
    print(f"\n{'='*60}")
    print("DIAGNOSING INPUT FILES")
    print(f"{'='*60}\n")
    
    tiff_files = sorted(glob.glob(os.path.join(input_folder, pattern)))
    
    for tiff_file in tiff_files:
        print(f"\n📄 {os.path.basename(tiff_file)}")
        
        with rasterio.open(tiff_file) as src:
            # Read full first band
            data = src.read(1)
            
            # Check for different types of "no data"
            nan_count = np.sum(np.isnan(data))
            if src.nodata is not None:
                nodata_count = np.sum(data == src.nodata)
            else:
                nodata_count = 0
            
            valid_count = data.size - nan_count - nodata_count
            
            print(f"   Total pixels: {data.size:,}")
            print(f"   Valid pixels: {valid_count:,} ({valid_count/data.size*100:.2f}%)")
            print(f"   NaN pixels: {nan_count:,}")
            print(f"   NoData pixels: {nodata_count:,}")
            
            if valid_count > 0:
                valid_data = data[~np.isnan(data)]
                if src.nodata is not None:
                    valid_data = valid_data[valid_data != src.nodata]
                
                if len(valid_data) > 0:
                    print(f"   Data range: [{np.min(valid_data):.4f}, {np.max(valid_data):.4f}]")
                    print(f"   ✅ File has valid data")
                else:
                    print(f"   ❌ File has NO valid data!")
            else:
                print(f"   ❌ File has NO valid data!")


# ============================================
# MAIN SCRIPT
# ============================================

if __name__ == "__main__":
    
    # ===== CONFIGURATION =====
    INPUT_FOLDER = '/Users/aaravsinghluthra/Desktop/EarthEngine_Exports'
    OUTPUT_FILE = '/Users/aaravsinghluthra/Desktop/EarthEngine_Exports/merged_mosaic.tif'
    VRT_FILE = '/content/drive/MyDrive/EarthEngine_Exports/merged_mosaic.vrt'
    FILE_PATTERN = '*.tif'
    
    # Merge settings
    MERGE_METHOD = 'first'  # 'first', 'last', 'min', 'max', 'sum'
    COMPRESSION = 'LZW'  # 'LZW', 'DEFLATE', 'ZSTD', or None
    TARGET_NODATA = None  # Set to a number if you want to override
    RESAMPLING = Resampling.nearest  # or bilinear, cubic, etc.
    
    # ===== STEP 1: DIAGNOSE INPUT FILES =====
    print("\n🔍 STEP 1: DIAGNOSING INPUT FILES")
    diagnose_input_files(INPUT_FOLDER, FILE_PATTERN)
    
    # ===== STEP 2: CREATE VRT (FAST ALTERNATIVE) =====
    print("\n🚀 STEP 2: CREATING VRT (INSTANT MOSAIC)")
    tiff_files = sorted(glob.glob(os.path.join(INPUT_FOLDER, FILE_PATTERN)))
    create_vrt_alternative(tiff_files, VRT_FILE)
    
    # ===== STEP 3: MERGE FILES =====
    print("\n⚡ STEP 3: MERGING FILES")
    output = merge_tiffs_fast(
        input_folder=INPUT_FOLDER,
        output_path=OUTPUT_FILE,
        pattern=FILE_PATTERN,
        method=MERGE_METHOD,
        compression=COMPRESSION,
        target_nodata=TARGET_NODATA,
        resampling=RESAMPLING
    )
    
    # ===== STEP 4: VERIFY OUTPUT =====
    if output and os.path.exists(output):
        verify_mosaic(output)
    
    print("\n" + "="*60)
    print("ALL OPERATIONS COMPLETE!")
    print("="*60)
    print("\n💡 TIPS:")
    print("   - If output has NaN values, check your GEE export code")
    print("   - Use the .vrt file for instant access (works in QGIS)")
    print("   - VRT is tiny and provides same functionality as merged file")
    print("   - If files don't overlap, you'll get a large file with gaps")