import os
import glob
import pandas as pd

# CONFIGURATION

INPUT_FOLDER = r"D:\Bare_soil\UP\UP_thresholds"
OUTPUT_FILE = "UP_thresholds.js"

# Find all CSV files ending in 'thresholds.csv'
csv_files = glob.glob(os.path.join(INPUT_FOLDER, "*_thresholds.csv"))

if not csv_files:
    print(f" No threshold CSV files found in: '{INPUT_FOLDER}'")
    exit()

print(f"🔍 Found {len(csv_files)} CSV files. Extracting thresholds...\n")

js_entries = []

for csv_file in csv_files:
    filename = os.path.basename(csv_file)
    raw_name = filename.replace('_thresholds.csv', '').replace('_plots.csv', '')
    
    dict_key = "_".join([word.capitalize() for word in raw_name.split('_')])
    gaul_name = " ".join([word.capitalize() for word in raw_name.split('_')])

    try:
        df = pd.read_csv(csv_file)
        
        # Extract mean NDVI and p10 BSI
        mean_ndvi = df.loc[df['stat'] == 'mean', 'NDVI'].values[0]
        p10_bsi = df.loc[df['stat'] == 'p10', 'BSI'].values[0]
        
        # Build the exact JavaScript format needed for Earth Engine
        entry = f"    '{dict_key}': {{\n        ndvi: {mean_ndvi},\n        bsi: {p10_bsi},\n        gaulName: '{gaul_name}'\n    }}"
        js_entries.append(entry)
        
        print(f"{gaul_name} -> NDVI (mean): {mean_ndvi:.4f} | BSI (p10): {p10_bsi:.4f}")
        
    except Exception as e:
        print(f"Error processing {filename}: {e}")

# SAVE TO A SINGLE GEE DICTIONARY
if js_entries:
    final_js = "var districtsConfig = {\n" + ",\n".join(js_entries) + "\n};\n"
    
    with open(OUTPUT_FILE, "w") as f:
        f.write(final_js)
        
    print(f"\nSUCCESS! Copy the contents of '{OUTPUT_FILE}' into your Earth Engine script.")
else:
    print("\nFailed: No valid data was extracted.")