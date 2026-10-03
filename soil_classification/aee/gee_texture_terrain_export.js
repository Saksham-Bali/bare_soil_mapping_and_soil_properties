// ==============================================================================
// GEE SCRIPT: S2 Pixel Extraction + AEE + GLCM Texture + Terrain
// ==============================================================================

// ==============================================================================
// 1. PASTE roiFeatures FROM gee_features.js HERE
// ==============================================================================
// REPLACE the dummy feature collection below with your actual pasted Python output:



// ==============================================================================
// 2. PARAMETERS & BATCH CONFIG
// ==============================================================================
// Change these two lines each run to prevent Code Editor crashes
var BATCH_START = 0;   // inclusive
var BATCH_END   = 7;   // exclusive — e.g. run 1: 0-7, run 2: 7-14, etc.

var startDate = '2024-05-01';
var endDate = '2024-06-01';
var cloudPct = 20;
var scale = 10;
var bands = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12'];
var driveFolder = 'S2_Soil_Texture_Terrain_CSV'; // NEW OUTPUT FOLDER
var pixelsPerPolygon = 30000;

// ==============================================================================
// 3. MASKS & GLOBAL LAYERS
// ==============================================================================
// Agricultural Mask (ESA WorldCover — Cropland only, class 40)
var agriMask = ee.ImageCollection('ESA/WorldCover/v200')
  .first()
  .select('Map')
  .eq(40);

// Terrain
var srtm = ee.Image('USGS/SRTMGL1_003');  // 30m global DEM
var elevation = srtm.select('elevation');
var slope = ee.Terrain.slope(srtm);
var aspect = ee.Terrain.aspect(srtm);

var terrain_features = elevation.rename('elevation')
  .addBands(slope.rename('slope'))
  .addBands(aspect.rename('aspect'));

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

  // Only apply cloud masking here so GLCM sees a continuous unmasked neighborhood
  return image
    .updateMask(qaMask.and(sclMask))
    .select(bands);
}

// GLCM Helper Function
// Sentinel-2 SR is scaled 0-10000. GLCM requires integers, so we scale it to 0-255.
function getGlcm(image, bandName) {
  var bandInt = image.select(bandName).divide(10000).multiply(255).toInt();
  var glcm = bandInt.glcmTexture({size: 1}); // 3x3 window
  return glcm.select([
    bandName + '_contrast', 
    bandName + '_corr', 
    bandName + '_asm', 
    bandName + '_ent', 
    bandName + '_idm'
  ]);
}

// ==============================================================================
// 5. LOOP: BATCHED EXPORT TASKS
// ==============================================================================
var featureList = roiFeatures.toList(roiFeatures.size());
var numFeatures = roiFeatures.size().getInfo();
var batchEnd = Math.min(BATCH_END, numFeatures);

print('Running polygons ' + BATCH_START + ' to ' + (batchEnd - 1) + ' of ' + numFeatures);
print('Pixels per polygon cap:', pixelsPerPolygon);
print('Date range:', startDate, '->', endDate);

for (var i = BATCH_START; i < batchEnd; i++) {

  var feature = ee.Feature(featureList.get(i));
  var geom = feature.geometry();

  var soilType = ee.String(feature.get('soil_type')).getInfo();
  var district = ee.String(feature.get('district')).getInfo();

  // 1. Get cloud-free median composite (but NOT masked for agri/NDVI yet)
  var s2Median = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    .filterBounds(geom)
    .filterDate(startDate, endDate)
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloudPct))
    .map(preprocessS2)
    .median();

  // 2. Compute brightness
  var brightness = s2Median.select(['B2','B3','B4','B8']).reduce(ee.Reducer.mean()).rename('brightness');
  s2Median = s2Median.addBands(brightness);

  // 3. Compute GLCM Texture on the full continuous composite
  var glcm_b11 = getGlcm(s2Median, 'B11');
  var glcm_b8a = getGlcm(s2Median, 'B8A');
  var glcm_brightness = getGlcm(s2Median, 'brightness');

  s2Median = s2Median.addBands([glcm_b11, glcm_b8a, glcm_brightness]);

  // 4. NOW apply the strict bare-soil and agricultural masks to the final image
  var ndvi = s2Median.normalizedDifference(['B8', 'B4']);
  var ndviMask = ndvi.lt(0.25);
  s2Median = s2Median.updateMask(agriMask.and(ndviMask));

  // 5. Get Alpha Earth Embeddings
  var startYear = startDate.substring(0, 4);
  var alphaEmbeddings = ee.ImageCollection('GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL')
    .filterBounds(geom)
    .filterDate(startYear + '-01-01', (parseInt(startYear) + 1) + '-01-01')
    .first();

  // 6. Combine everything
  var combinedImage = s2Median
    .addBands(alphaEmbeddings)
    .addBands(terrain_features);

  var pixels = combinedImage.sample({
    region: geom,
    scale: scale,
    numPixels: pixelsPerPolygon,
    tileScale: 8,
    geometries: false
  });

  var labeledPixels = pixels.map(function (p) {
    return p.set('soil_type', soilType)
      .set('district', district)
      .set('polygon_id', 'poly' + i);
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

print('Batch queued: ' + BATCH_START + '-' + (batchEnd - 1));
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
