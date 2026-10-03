import pandas as pd
import json
import os
import glob

def build_feature_str(geom_json, soil_type, district):
    geom = json.loads(geom_json)
    coords = json.dumps(geom['coordinates'])
    return (f"  ee.Feature(ee.Geometry.Polygon({coords}), "
            f"{{'soil_type': '{soil_type}', 'district': '{district}'}})")

def generate_gee_js(folder_path, out_js_path="gee_features.js"):
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
    if not csv_files:
        raise ValueError(f"No CSVs found in {folder_path}")

    js_features = []
    skipped = 0

    for csv_path in csv_files:
        filename = os.path.basename(csv_path).replace(".csv", "")
        parts = filename.split("_")
        if len(parts) < 2:
            print(f"Skipping {filename}: can't parse district/soil_type")
            continue
        
        soil_type = parts[-1]           # last part → 's1', 's2', 's3', 's4'
        district = "_".join(parts[:-1]) # everything before last → handles multi-word districts

        df = pd.read_csv(csv_path)
        count = 0
        for _, row in df.iterrows():
            if pd.notna(row.get('.geo')):
                try:
                    js_features.append(build_feature_str(row['.geo'], soil_type, district))
                    count += 1
                except Exception as e:
                    skipped += 1
        print(f"{filename}: {count} polygons processed")

    js_collection_str = ("var roiFeatures = ee.FeatureCollection([\n" 
                         + ",\n".join(js_features) 
                         + "\n]);")

    with open(out_js_path, "w") as f:
        f.write(js_collection_str)

    print(f"\nTotal features: {len(js_features)}  |  Skipped (bad geometry): {skipped}")
    print(f"JS written to: {out_js_path}")

# ---- Run it ----
generate_gee_js(
    r"d:\Bare_soil\Bare_Soil_Mapping\soil_classification\Soil_TypePunjab",
    r"d:\Bare_soil\Bare_Soil_Mapping\soil_classification\gee_features.js"
)
