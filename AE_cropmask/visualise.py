import rasterio
from rasterio.plot import show
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

# Path to the generated full mask
file_path = '/Users/aaravsinghluthra/Desktop/EarthEngine_Exports 2/jalandhar_full_mask_from_csv.tif'

print(f"Visualizing: {file_path}")

try:
    with rasterio.open(file_path) as src:
        data = src.read(1) # Read band 1
        
        # Check if data exists
        if np.count_nonzero(data) == 0:
            print("Warning: The raster is empty (all zeros).")
        else:
            print("Data found. Plotting...")
            
            plt.figure(figsize=(15, 15))
            
            # Create a custom colormap
            # 0 = Red (Non-Ag), 1 = Green (Ag), 255 = Nodata (Transparent/White)
            # We need to handle the values carefully.
            # Let's mask the nodata
            masked_data = np.ma.masked_where(data == 255, data)
            
            # Define colors: 0 -> Red, 1 -> Green
            cmap = mcolors.ListedColormap(['red', 'green'])
            bounds = [-0.5, 0.5, 1.5]
            norm = mcolors.BoundaryNorm(bounds, cmap.N)
            
            show(masked_data, transform=src.transform, cmap=cmap, norm=norm, title="Jalandhar District Agricultural Mask\nGreen: Agricultural | Red: Non-Agricultural")
            
            # Add legend
            # Create a proxy artist for the legend
            import matplotlib.patches as mpatches
            red_patch = mpatches.Patch(color='red', label='Non-Agricultural')
            green_patch = mpatches.Patch(color='green', label='Agricultural')
            plt.legend(handles=[red_patch, green_patch], loc='upper right')
            
            plt.show()

except Exception as e:
    print(f"Error: {e}")
