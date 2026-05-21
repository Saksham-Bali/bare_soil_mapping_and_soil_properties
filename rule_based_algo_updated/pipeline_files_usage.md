# Rule-Based Bare Soil Mapping Pipeline

This directory contains scripts for generating **district-level bare soil hybrid maps** using Sentinel-2 imagery and ground-truth polygon data collected on Google Earth Engine (GEE).

The pipeline is a **4-step process**:
1. Collect ground-truth polygons on GEE → export CSVs
2. Convert CSVs to a GEE-compatible JavaScript dictionary
3. Compute spectral descriptive statistics per district on GEE
4. Extract thresholds and generate monthly hybrid maps on GEE

---

## Pipeline Overview

```
GEE_data_collection_tool.js   (Step 0 — in manual_data_collection/)
          │
          │  exports per-district polygon CSVs to Google Drive
          ▼
   csv_polygon_to_js.py        (Step 1 — run locally)
          │
          │  prints a JS dictionary of geometries
          ▼
   descriptive_stats.js        (Step 2 — paste output into GEE)
          │
          │  exports one CSV per district with spectral stats
          ▼
  threshold_extraction.py      (Step 3 — run locally)
          │
          │  prints a JS dictionary of NDVI/BSI thresholds
          ▼
 s2_hybrid_download_final.js   (Step 4 — paste output into GEE)
          │
          │  exports monthly S2 composites + hybrid maps to Drive
          ▼
        Done — GeoTIFFs in Google Drive per district per month
```

---

## Prerequisites

- A Google Earth Engine account
- Google Drive access (all GEE exports go to Drive)

---

## Step 0 — Collect Ground-Truth Polygons on GEE

**Script:** `../manual_data_collection/GEE_data_collection_tool.js`

This is an **interactive GEE app**. Run it in the GEE Code Editor.

### What it does
- Lets you select a **state** and **district** from dropdowns
- Displays the Sentinel-2 median composite and ESA WorldCover crop mask for that district
- Lets you **draw polygons** over bare soil plots
- Exports the drawn polygons as a CSV to Google Drive

### How to use
1. Paste the script into the [GEE Code Editor](https://code.earthengine.google.com/) and click **Run**.
2. In the left panel:
   - Set your **date range** (e.g., `2024-04-01` to `2024-05-01` — a crop-free / post-harvest month works best).
   - Select a **State**, then a **District**.
3. Use the GEE drawing tools to **draw polygons** over bare soil areas visible in the Sentinel-2 image. Yellow areas are ESA-detected croplands — focus your polygons there.
4. Click **Save Drawn Polygons** to export the polygons as a CSV.
5. Repeat for each district in the state.

### Output
- A folder in Google Drive named `Ground_truth_SOC_<StateName>/`
- One CSV per district, named `<DistrictName>_plots.csv`
- Each CSV contains a `.geo` column with the polygon geometry in GeoJSON format.

> **Tip:** Download all the CSVs for a state into a single local folder before proceeding to Step 1.

---

## Step 1 — Convert CSVs to a JavaScript Geometry Dictionary

**Script:** [`csv_polygon_to_js.py`](./csv_polygon_to_js.py)

### What it does
Reads all `_plots.csv` files from a local folder and converts the polygon geometries into a single JavaScript dictionary (`var districtPlots = { ... }`) that GEE can understand.

### Configuration
Edit the top of the script before running:

```python
# Path to the folder containing all district CSVs for one state
INPUT_FOLDER = r"D:\Bare_soil\Maharashtra\Ground_truth_SOC_Maharashtra"

# Name of the output .js file
OUTPUT_FILE = "maharashtra_plots.js"
```

### How to run
```bash
python csv_polygon_to_js.py
```

### Output
A `.js` file (e.g., `maharashtra_plots.js`) containing a dictionary like:

```javascript
var districtPlots = {
  'Ahmednagar': [
    ee.Geometry.Polygon([...]),
    ee.Geometry.Polygon([...])
  ],
  'Pune': [
    ee.Geometry.Polygon([...])
  ]
};
```

> **Next:** Copy the **entire contents** of this file — you will paste it into the GEE script in Step 2.

---

## Step 2 — Compute Descriptive Statistics per District on GEE

**Script:** [`descriptive_stats.js`](./descriptive_stats.js)

### What it does
For each district's polygon set, it computes spectral statistics (min, max, mean, stdDev, p10, p25, p50, p75) across all Sentinel-2 bands and spectral indices (NDVI, SAVI, NDWI, BSI, NDBI). One CSV is exported per district.

### How to use

1. Open [`descriptive_stats.js`](./descriptive_stats.js) in the GEE Code Editor.

2. **Update the general parameters** at the top of the file:

   ```javascript
   var COUNTRY_NAME = 'India';
   var STATE_NAME   = 'Karnataka'; // Change for your state
   var startDate    = '2024-03-01';
   var endDate      = '2024-04-01';
   var cloudPct     = 10;
   var scale        = 10; // Sentinel-2 resolution in metres
   ```

   > Use the **same date range** you used in Step 0 when collecting the polygons.

3. **Paste the Python output** from Step 1 into the designated section:

   ```javascript
   // ============== 2. PASTE PYTHON OUTPUT HERE ==============
   // Paste the entire output from the Python script csv_polygon_to_js.py below this line
   var districtPlots = {
     // <-- paste here
   };
   ```

4. Click **Run**, then submit all export tasks from the **Tasks** tab.

### Output
- A Google Drive folder named `<StateName>_Districts_thresholds/`
- One CSV per district, named `<DistrictName>_thresholds.csv`
- Each CSV has one row per statistic (min, max, mean, stdDev, p10, p25, p50, p75) and one column per band/index.

> Download all these CSVs into a single local folder before proceeding to Step 3.

---

## Step 3 — Extract NDVI & BSI Thresholds

**Script:** [`threshold_extraction.py`](./threshold_extraction.py)

### What it does
Reads the `_thresholds.csv` files produced by Step 2 and extracts two key values per district:
- **NDVI threshold** — the `mean` NDVI across bare soil plots
- **BSI threshold** — the 10th percentile (`p10`) BSI across bare soil plots

These thresholds are formatted as a JavaScript dictionary ready for use in GEE Step 4.

### Configuration
Edit the top of the script before running:

```python
# Path to the folder containing all *_thresholds.csv files downloaded from Drive
INPUT_FOLDER = r"D:\Bare_soil\UP\UP_thresholds"

# Name of the output .js file
OUTPUT_FILE = "UP_thresholds.js"
```

### How to run
```bash
python threshold_extraction.py
```

### Output
A `.js` file (e.g., `UP_thresholds.js`) containing:

```javascript
var districtsConfig = {
    'Agra': {
        ndvi: 0.1523,
        bsi: -0.0412,
        gaulName: 'Agra'
    },
    'Mathura': {
        ndvi: 0.1801,
        bsi: -0.0355,
        gaulName: 'Mathura'
    }
    // ...
};
```

> **Next:** Copy the **entire contents** of this file — you will paste it into the GEE script in Step 4.

---

## Step 4 — Generate Monthly S2 Composites & Hybrid Maps on GEE

**Script:** [`s2_hybrid_download_final.js`](./s2_hybrid_download_final.js)

### What it does
For every district in `districtsConfig` and every month of the configured year, this script:
1. Builds a **Sentinel-2 monthly median composite** (all 12 bands)
2. Applies the district-specific NDVI/BSI thresholds and the ESA WorldCover crop mask to create a **hybrid classification map** with three classes:
   - `0` — Background (non-cropland)
   - `1` — Crop
   - `2` — Bare Soil

Both outputs are exported as GeoTIFFs to Google Drive.

### How to use

1. Open [`s2_hybrid_download_final.js`](./s2_hybrid_download_final.js) in the GEE Code Editor.

2. **Update the general settings** at the top:

   ```javascript
   var COUNTRY_NAME    = 'India';
   var STATE_NAME      = 'Uttar Pradesh'; // Must match the official GAUL state name
   var FOLDER_PREFIX   = 'Uttar_Pradesh'; // Prefix for Google Drive folder names
   var year            = 2024;
   var cloudThreshold  = 20;
   ```

3. **Paste the Python output** from Step 3 into the designated section (around line 19–24):

   ```javascript
   // PASTE THE OUTPUT FROM YOUR threshold_extraction.py SCRIPT HERE

   var districtsConfig = {
     // <-- paste here
   };
   ```

4. Click **Run**. The console will show all tasks being configured.

5. Go to the **Tasks** tab and click **Run All** to submit all export jobs.

### Output structure in Google Drive

```
<FOLDER_PREFIX>_<DistrictName>/
  January_2024/
    <District>_202401_January_S2_AllBands.tif   ← 12-band Sentinel-2 float
    <District>_202401_January_Hybrid.tif        ← Byte raster (0/1/2)
  February_2024/
    ...
  December_2024/
    ...
```

> **Scale:** All exports are at **10 m resolution**, CRS **EPSG:4326**.

---

## File Summary

| File | Type | Purpose |
|------|------|---------|
| [`csv_polygon_to_js.py`](./csv_polygon_to_js.py) | Python | Converts downloaded polygon CSVs to a GEE JS geometry dictionary |
| [`descriptive_stats.js`](./descriptive_stats.js) | GEE JS | Computes per-district spectral statistics from ground-truth plots |
| [`threshold_extraction.py`](./threshold_extraction.py) | Python | Extracts NDVI/BSI bare soil thresholds from stats CSVs |
| [`s2_hybrid_download_final.js`](./s2_hybrid_download_final.js) | GEE JS | Exports monthly S2 composites and hybrid bare soil maps |

---

