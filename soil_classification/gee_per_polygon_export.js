// ==============================================================================
// GEE SCRIPT: S2 Pixel Extraction — One 25k-Pixel Task Per Polygon
// ==============================================================================

// ==============================================================================
// 1. PASTE roiFeatures FROM gee_features.js HERE
// ==============================================================================
// REPLACE the dummy feature collection below with your actual pasted Python output:



// ==============================================================================
// 2. PARAMETERS
// ==============================================================================
var startDate = '2024-05-01';
var endDate = '2024-06-01';
var cloudPct = 20;
var scale = 10;
var bands = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12'];
var driveFolder = 'S2_Soil_CSV_PerPolygon';
var pixelsPerPolygon = 30000;  // Increased to 30k as requested

// ==============================================================================
// 3. AGRICULTURAL MASK (ESA WorldCover — Cropland only, class 40)
// ==============================================================================
var agriMask = ee.ImageCollection('ESA/WorldCover/v200')
  .first()
  .select('Map')
  .eq(40);

// ==============================================================================
// 4. PREPROCESSING FUNCTION
// ==============================================================================
function preprocessS2(image) {
  var qa = image.select('QA60');
  var qaMask = qa.bitwiseAnd(1 << 10).eq(0)
    .and(qa.bitwiseAnd(1 << 11).eq(0));

  var scl = image.select('SCL');
  var sclMask = scl.neq(3)
    .and(scl.neq(8))
    .and(scl.neq(9))
    .and(scl.neq(10));

  // Pre-filter: Loose NDVI mask to remove obvious vegetation BEFORE sampling.
  // This ensures our 30k sample cap isn't wasted on green crops.
  var ndvi = image.normalizedDifference(['B8', 'B4']);
  var ndviMask = ndvi.lt(0.25);

  return image
    .updateMask(qaMask.and(sclMask).and(agriMask).and(ndviMask))
    .select(bands);
}

// ==============================================================================
// 5. LOOP: ONE EXPORT TASK PER POLYGON
// ==============================================================================
var featureList = roiFeatures.toList(roiFeatures.size());
var numFeatures = roiFeatures.size().getInfo();

print('Total polygons found:', numFeatures);
print('Pixels per polygon cap:', pixelsPerPolygon);
print('Date range:', startDate, '->', endDate);

for (var i = 0; i < numFeatures; i++) {

  var feature = ee.Feature(featureList.get(i));
  var geom = feature.geometry();

  // .getInfo() safely converts the server objects into normal JS text strings
  var soilType = ee.String(feature.get('soil_type')).getInfo();
  var district = ee.String(feature.get('district')).getInfo();

  var s2Median = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    .filterBounds(geom)
    .filterDate(startDate, endDate)
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloudPct))
    .map(preprocessS2)
    .median();

  var pixels = s2Median.sample({
    region: geom,
    scale: scale,
    numPixels: pixelsPerPolygon,
    tileScale: 8,
    geometries: false
  });

  var labeledPixels = pixels.map(function (p) {
    return p.set('soil_type', soilType)
      .set('district', district);
  });

  var taskName = district + '_' + soilType + '_poly' + i;

  Export.table.toDrive({
    collection: labeledPixels,
    description: taskName,
    folder: driveFolder,
    fileNamePrefix: taskName,
    fileFormat: 'CSV'
  });
}

print('All ' + numFeatures + ' export tasks queued!');
print('Go to Tasks panel -> click "Run all tasks"');
print('Output: Google Drive -> ' + driveFolder + '/');

// ==============================================================================
// 6. VISUAL SANITY CHECK
// ==============================================================================
Map.centerObject(roiFeatures, 8);
Map.addLayer(roiFeatures, { color: 'FF0000' }, 'ROI Polygons');
Map.addLayer(
  agriMask.selfMask(),
  { palette: ['00C853'] },
  'Agricultural Mask (cropland only)'
);