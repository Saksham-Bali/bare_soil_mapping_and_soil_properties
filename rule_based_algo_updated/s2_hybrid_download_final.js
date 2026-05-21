/**************************************************************
GENERALIZED — MONTHLY S2 COMPOSITE & HYBRID EXPORT (2024)
For each district, for each month (Jan-Dec 2024):
  - Export 1: Sentinel-2 median composite (all bands)
  - Export 2: Hybrid classification (0=Background, 1=Crop, 2=Bare Soil)
**************************************************************/

// 1. GENERAL SETTINGS (Update these when changing states!)
var COUNTRY_NAME = 'India';
var STATE_NAME = 'Uttar Pradesh';     // Must match the official GAUL State name (e.g. 'Uttar Pradesh')
var FOLDER_PREFIX = 'Uttar_Pradesh';     // Your Google Drive root folder prefix for these exports
var year = 2024;
var cloudThreshold = 20;

// Center Map generically over India (you can pan to your specific state)
Map.setCenter(78.9, 22.5, 5);

// 2. DISTRICT CONFIGURATION 
// PASTE THE OUTPUT FROM YOUR threshold_extraction.py SCRIPT HERE



//PASTE THE OUTPUT FROM YOUR PYTHON SCRIPT HERE 


// ============================================================
// 3. GLOBALS & INDEX FUNCTION
// ============================================================
var gaul2 = ee.FeatureCollection("FAO/GAUL/2015/level2");
var months = [
    { month: 1, name: 'January' }, { month: 2, name: 'February' },
    { month: 3, name: 'March' }, { month: 4, name: 'April' },
    { month: 5, name: 'May' }, { month: 6, name: 'June' },
    { month: 7, name: 'July' }, { month: 8, name: 'August' },
    { month: 9, name: 'September' }, { month: 10, name: 'October' },
    { month: 11, name: 'November' }, { month: 12, name: 'December' }
];

// List of exact bands we want to export
var bandNames = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12'];

function addIndices(img) {
    var ndvi = img.normalizedDifference(['B8', 'B4']).rename('NDVI');
    var bsi = img.expression(
        '( (SWIR + RED) - (NIR + BLUE) ) / ( (SWIR + RED) + (NIR + BLUE) )',
        {
            'SWIR': img.select('B11'),
            'RED': img.select('B4'),
            'NIR': img.select('B8'),
            'BLUE': img.select('B2')
        }
    ).rename('BSI');
    return img.addBands([ndvi, bsi]);
}

// ============================================================
// 4. MAIN PROCESSOR (S2 COMPOSITE & HYBRID MAP)
// ============================================================
function processDistrictMonth(districtKey, config, monthInfo) {

    // Dynamic date generation
    var start = ee.Date.fromYMD(year, monthInfo.month, 1);
    var end = start.advance(1, 'month');

    print('Processing: ' + districtKey + ' - ' + monthInfo.name + ' ' + year);

    var districtGeom = gaul2
        .filter(ee.Filter.eq('ADM0_NAME', COUNTRY_NAME))
        .filter(ee.Filter.eq('ADM1_NAME', STATE_NAME))
        .filter(ee.Filter.eq('ADM2_NAME', config.gaulName))
        .geometry();

    // Crop mask (ESA WorldCover class 40)
    var cropMask = ee.ImageCollection('ESA/WorldCover/v200')
        .first()
        .select('Map')
        .clip(districtGeom)
        .eq(40);

    // Monthly Sentinel-2 composite
    var s2col = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
        .filterDate(start, end)
        .filterBounds(districtGeom)
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloudThreshold))
        .select(bandNames); // Explicitly select bands

    // No fallback image. If the collection is empty, the task will naturally fail
    // on Earth Engine servers, saving download time/storage.
    var s2Median = s2col.median().clip(districtGeom);

    // Compute Indices
    var s2Idx = addIndices(s2Median);

    // Build Hybrid Logic
    var bare = s2Idx.select('NDVI').lte(config.ndvi)
        .and(s2Idx.select('BSI').gte(config.bsi))
        .and(cropMask);

    var crop = cropMask
        .and(bare.not())
        .and(s2Idx.select('NDVI').gt(config.ndvi + 0.05))
        .and(s2Idx.select('NDVI').gt(0.25));

    // DENSE RASTER: 0 = Background, 1 = Crop, 2 = Bare Soil
    var hybrid = ee.Image.constant(0)
        .where(crop, 1)
        .where(bare, 2)
        .unmask(0)
        .toByte()
        .clip(districtGeom);

    // Naming & Folder Structure setup
    var monthStr = (monthInfo.month < 10) ? '0' + monthInfo.month : '' + monthInfo.month;
    var folderName = FOLDER_PREFIX + '_' + districtKey + '/' + monthInfo.name + '_' + year;
    var filePrefix = districtKey + '_' + year + monthStr + '_' + monthInfo.name;

    // Export 1: Sentinel-2 all bands
    Export.image.toDrive({
        image: s2Median.toFloat(),
        description: districtKey + '_' + monthInfo.name + '_S2',
        folder: folderName,
        fileNamePrefix: filePrefix + '_S2_AllBands',
        region: districtGeom,
        scale: 10,
        maxPixels: 1e13,
        crs: 'EPSG:4326'
    });

    // Export 2: Hybrid Classification Map
    Export.image.toDrive({
        image: hybrid,
        description: districtKey + '_' + monthInfo.name + '_Hybrid',
        folder: folderName,
        fileNamePrefix: filePrefix + '_Hybrid',
        region: districtGeom,
        scale: 10,
        maxPixels: 1e13,
        crs: 'EPSG:4326'
    });
}

// ============================================================
// 5. BATCH EXECUTION LOOP
// ============================================================
var districtKeys = Object.keys(districtsConfig);
var totalExports = 0;

print('╔════════════════════════════════════════════════════════╗');
print('║  GENERALIZED MONTHLY EXPORTS: S2 & HYBRID MAPS         ║');
print('║  State: ' + STATE_NAME + '                                    ║');
print('║  ' + districtKeys.length + ' Districts × 12 Months × 2 Files = ' + (districtKeys.length * 24) + ' Tasks           ║');
print('╚════════════════════════════════════════════════════════╝\n');

for (var i = 0; i < districtKeys.length; i++) {
    var districtKey = districtKeys[i];
    var config = districtsConfig[districtKey];

    print('\n▶ Configuring District: ' + districtKey + ' (' + (i + 1) + '/' + districtKeys.length + ')');

    for (var m = 0; m < months.length; m++) {
        processDistrictMonth(districtKey, config, months[m]);
        totalExports += 2;
    }
}

print('\n╔════════════════════════════════════════════════════════╗');
print('║  CONFIGURATION COMPLETE                                ║');
print('║                                                        ║');
print('║  Total tasks configured: ' + totalExports + '                           ║');
print('║  Exports per month folder:                             ║');
print('║    1. _S2_AllBands.tif (12-band float)                 ║');
print('║    2. _Hybrid.tif (Value 0=Bg, 1=Crop, 2=Bare Soil)    ║');
print('╚════════════════════════════════════════════════════════╝');