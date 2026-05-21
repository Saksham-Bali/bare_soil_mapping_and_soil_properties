import json
import os
import glob
import pandas as pd

# CONFIGURATION
INPUT_FOLDER = r"D:\Bare_soil\Maharashtra\Ground_truth_SOC_Maharashtra"
OUTPUT_FILE = "maharashtra_plots.js"

# FIND ALL CSV FILES
csv_files = glob.glob(os.path.join(INPUT_FOLDER, "*.csv"))

if not csv_files:
    print(f"No CSV files found in the folder: '{INPUT_FOLDER}'")
    exit()

print(f"Found {len(csv_files)} CSV files. Starting conversion...\n")

all_js_arrays = []

for csv_full_path in csv_files:
    print(f"Processing: {csv_full_path}")
    df = pd.read_csv(csv_full_path)

    file_name_with_ext = os.path.basename(csv_full_path)
    file_name_base = os.path.splitext(file_name_with_ext)[0]
    file_name_base = file_name_base.replace('_plots', '').replace('_thresholds', '')

    geometry_definitions = []

    for i, row in df.iterrows():
        try:
            if '.geo' in df.columns:
                geo_json_str = row['.geo']
            elif 'geo' in df.columns:
                geo_json_str = row['geo']
            else:
                raise Exception("No geometry column found (.geo or geo)")

            geo_data = json.loads(geo_json_str)
            geo_type = geo_data.get('type', 'Polygon')
            coordinates = geo_data['coordinates']
            coords_str = json.dumps(coordinates)

            definition = f"    ee.Geometry.{geo_type}({coords_str})"
            geometry_definitions.append(definition)

        except Exception as e:
            pass

    # Build dictionary entry for this district
    if geometry_definitions:
        district_array_string = f"  '{file_name_base}': [\n" + ",\n".join(geometry_definitions) + "\n  ]"
        all_js_arrays.append(district_array_string)
        print(f"Extracted {len(geometry_definitions)} geometries for {file_name_base}")

# SAVE AS A SINGLE GEE DICTIONARY
if all_js_arrays:
    full_output = "var districtPlots = {\n" + ",\n\n".join(all_js_arrays) + "\n};\n"
    with open(OUTPUT_FILE, "w") as f:
        f.write(full_output)
    print(f"\nAll districts combined and saved to: {OUTPUT_FILE}")
else:
    print("\nFailed: No valid data was found.")