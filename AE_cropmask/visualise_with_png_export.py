"""
Quick script to check GeoTIFF data validity and create false color composite
"""

import numpy as np
import rasterio
from rasterio.plot import show
import matplotlib.pyplot as plt
import folium
from folium import raster_layers
import base64
from io import BytesIO
from PIL import Image

def check_geotiff_data(geotiff_path):
    """
    Check GeoTIFF for valid data and print statistics

    Parameters:
    -----------
    geotiff_path : str
        Path to GeoTIFF file
    """
    print("="*60)
    print(f"Checking: {geotiff_path}")
    print("="*60)

    with rasterio.open(geotiff_path) as src:
        # Basic info
        print(f"\n📊 IMAGE METADATA:")
        print(f"  Dimensions: {src.width} x {src.height} pixels")
        print(f"  Number of bands: {src.count}")
        print(f"  Data type: {src.dtypes[0]}")
        print(f"  CRS: {src.crs}")
        print(f"  NoData value: {src.nodata}")
        print(f"  Bounds: {src.bounds}")

        # Read all bands
        data = src.read()

        print(f"\n📈 DATA STATISTICS:")
        total_pixels = src.height * src.width

        # Check for valid data
        if src.nodata is not None:
            valid_mask = data != src.nodata
        else:
            valid_mask = ~np.isnan(data)

        valid_pixels = np.sum(valid_mask[0])  # Check first band
        valid_percentage = (valid_pixels / total_pixels) * 100

        print(f"  Total pixels: {total_pixels:,}")
        print(f"  Valid pixels: {valid_pixels:,} ({valid_percentage:.2f}%)")
        print(f"  Invalid/NoData pixels: {total_pixels - valid_pixels:,} ({100-valid_percentage:.2f}%)")

        if valid_pixels == 0:
            print("\n⚠️  WARNING: NO VALID DATA FOUND!")
            print("   The GeoTIFF contains only NoData/NaN values.")
            return None, None

        # Statistics per band
        print(f"\n📊 PER-BAND STATISTICS:")
        for i in range(src.count):
            band_data = data[i]
            valid_band = band_data[valid_mask[i]]

            if len(valid_band) > 0:
                print(f"  Band {i+1}:")
                print(f"    Min: {np.min(valid_band):.4f}")
                print(f"    Max: {np.max(valid_band):.4f}")
                print(f"    Mean: {np.mean(valid_band):.4f}")
                print(f"    Std: {np.std(valid_band):.4f}")

        if valid_percentage > 0:
            print(f"\n✅ GeoTIFF contains valid data!")

        return src, data


def create_false_color_composite(geotiff_path, bands=[1, 16, 9],
                                 output_path='false_color_composite.png',
                                 stretch_percentile=(2, 98)):
    """
    Create false color composite from GeoTIFF

    Parameters:
    -----------
    geotiff_path : str
        Path to GeoTIFF file
    bands : list
        List of 3 band numbers to use for RGB (default: [1, 16, 9])
    output_path : str
        Path to save the composite image
    stretch_percentile : tuple
        Percentiles for contrast stretching (default: (2, 98))
    """

    print(f"\n{'='*60}")
    print(f"Creating False Color Composite")
    print(f"{'='*60}")

    with rasterio.open(geotiff_path) as src:
        # Check if requested bands exist
        if max(bands) > src.count:
            print(f"⚠️  Warning: Requested band {max(bands)} but image only has {src.count} bands")
            print(f"   Using bands [1, 2, 3] instead")
            bands = [1, 2, 3]

        print(f"Using bands: R={bands[0]}, G={bands[1]}, B={bands[2]}")

        # Read the three bands
        r = src.read(bands[0])
        g = src.read(bands[1])
        b = src.read(bands[2])

        # Create RGB composite
        rgb = np.dstack((r, g, b))

        # Handle NoData
        if src.nodata is not None:
            mask = (r == src.nodata) | (g == src.nodata) | (b == src.nodata)
        else:
            mask = np.isnan(r) | np.isnan(g) | np.isnan(b)

        # Normalize each band using percentile stretch
        def normalize_band(band, mask):
            valid_data = band[~mask]
            if len(valid_data) == 0:
                return np.zeros_like(band)

            p_low, p_high = np.percentile(valid_data, stretch_percentile)
            band_norm = np.clip((band - p_low) / (p_high - p_low), 0, 1)
            band_norm[mask] = 0  # Set NoData to black
            return band_norm

        rgb_norm = np.dstack([
            normalize_band(r, mask),
            normalize_band(g, mask),
            normalize_band(b, mask)
        ])

        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

        # Plot false color composite
        ax1.imshow(rgb_norm)
        ax1.set_title(f'False Color Composite\n(R=Band{bands[0]}, G=Band{bands[1]}, B=Band{bands[2]})',
                     fontsize=12, fontweight='bold')
        ax1.axis('off')

        # Plot with geographic context
        show(src, ax=ax2, cmap='viridis')
        ax2.set_title('Band 1 with Coordinates', fontsize=12, fontweight='bold')

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Composite saved to: {output_path}")
        plt.show()

        return rgb_norm, src.bounds, src.crs


def create_interactive_map(geotiff_path, bands=[1, 16, 9],
                          output_html='interactive_map.html'):
    """
    Create interactive Folium map with GeoTIFF overlay

    Parameters:
    -----------
    geotiff_path : str
        Path to GeoTIFF file
    bands : list
        List of 3 band numbers for RGB composite
    output_html : str
        Path to save HTML map
    """

    print(f"\n{'='*60}")
    print(f"Creating Interactive Map")
    print(f"{'='*60}")

    with rasterio.open(geotiff_path) as src:
        # Get bounds
        bounds = src.bounds
        center_lat = (bounds.bottom + bounds.top) / 2
        center_lon = (bounds.left + bounds.right) / 2

        # Check if requested bands exist
        if max(bands) > src.count:
            bands = [1, min(2, src.count), min(3, src.count)]

        # Read bands
        r = src.read(bands[0])
        g = src.read(bands[1])
        b = src.read(bands[2])

        # Handle NoData
        if src.nodata is not None:
            mask = (r == src.nodata) | (g == src.nodata) | (b == src.nodata)
        else:
            mask = np.isnan(r) | np.isnan(g) | np.isnan(b)

        # Normalize bands
        def normalize(band, mask):
            valid = band[~mask]
            if len(valid) == 0:
                return np.zeros_like(band, dtype=np.uint8)
            p2, p98 = np.percentile(valid, [2, 98])
            norm = np.clip((band - p2) / (p98 - p2) * 255, 0, 255).astype(np.uint8)
            norm[mask] = 0
            return norm

        r_norm = normalize(r, mask)
        g_norm = normalize(g, mask)
        b_norm = normalize(b, mask)

        # Create RGB image
        rgb_array = np.dstack((r_norm, g_norm, b_norm))

        # Add alpha channel (transparency for NoData)
        alpha = (~mask * 255).astype(np.uint8)
        rgba_array = np.dstack((rgb_array, alpha))

        # Convert to PIL Image
        img = Image.fromarray(rgba_array, 'RGBA')

        # Create Folium map
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=11,
            tiles='OpenStreetMap'
        )

        # Add different base layers
        folium.TileLayer('cartodbpositron', name='CartoDB Positron').add_to(m)
        folium.TileLayer('cartodbdark_matter', name='CartoDB Dark').add_to(m)

        # Convert image to base64 for embedding
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()

        # Add image overlay
        folium.raster_layers.ImageOverlay(
            image=f'data:image/png;base64,{img_str}',
            bounds=[[bounds.bottom, bounds.left], [bounds.top, bounds.right]],
            opacity=0.7,
            name=f'AlphaEarth Composite (R={bands[0]}, G={bands[1]}, B={bands[2]})'
        ).add_to(m)

        # Add layer control
        folium.LayerControl().add_to(m)

        # Add boundary rectangle
        folium.Rectangle(
            bounds=[[bounds.bottom, bounds.left], [bounds.top, bounds.right]],
            color='red',
            fill=False,
            weight=2,
            popup='Study Area Boundary'
        ).add_to(m)

        # Add center marker
        folium.Marker(
            [center_lat, center_lon],
            popup='Center of Study Area',
            icon=folium.Icon(color='red', icon='info-sign')
        ).add_to(m)

        # Save map
        m.save(output_html)
        print(f"✅ Interactive map saved to: {output_html}")
        print(f"   Open this file in a web browser to view")

        return m


# ============================================
# MAIN SCRIPT
# ============================================

if __name__ == "__main__":

    # Path to your exported GeoTIFF
    GEOTIFF_PATH = '/Users/aaravsinghluthra/Desktop/EarthEngine_Exports/merged_mosaic.tif'

    # Bands to use for false color composite
    # For AlphaEarth embeddings, try different combinations:
    # [1, 16, 9], [5, 10, 15], [3, 12, 20], etc.
    BANDS = [1, 16, 9]

    print("\n🔍 GEOTIFF DATA VALIDATION & VISUALIZATION")
    print("="*60)

    # Step 1: Check data validity
    src, data = check_geotiff_data(GEOTIFF_PATH)

    if data is not None:
        # Step 2: Create false color composite
        rgb, bounds, crs = create_false_color_composite(
            GEOTIFF_PATH,
            bands=BANDS,
            output_path='false_color_composite.png'
        )

        # Step 3: Create interactive map
        interactive_map = create_interactive_map(
            GEOTIFF_PATH,
            bands=BANDS,
            output_html='interactive_map.html'
        )

        print("\n" + "="*60)
        print("✅ ALL CHECKS COMPLETE!")
        print("="*60)
        print("Generated files:")
        print("  📄 false_color_composite.png - Static false color image")
        print("  🗺️  interactive_map.html - Interactive web map")
        print("\nOpen 'interactive_map.html' in your browser to explore!")
    else:
        print("\n❌ Cannot create visualizations - no valid data found")
        print("\nPossible solutions:")
        print("  1. Check your GEE export - it may have failed")
        print("  2. Verify the date range has data for your region")
        print("  3. Try using .mosaic() or .median() instead of .first()")
        print("  4. Expand the date range in your GEE script")