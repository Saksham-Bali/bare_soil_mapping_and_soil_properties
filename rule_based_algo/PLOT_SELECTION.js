// ==================== SENTINEL-2: ONE LAYER PER DISTRICT (Punjab) - FIXED ====================

// 1. Load Punjab districts (FAO/GAUL level2) for *India only*
var districts = ee.FeatureCollection('FAO/GAUL/2015/level2')
  .filter(ee.Filter.eq('ADM0_NAME', 'India'))   // ✅ Only India
  .filter(ee.Filter.eq('ADM1_NAME', 'Punjab')); // ✅ Only Punjab state

Map.centerObject(districts, 7);

// 2. Parameters - change as needed
var startDate = '2024-05-01';
var endDate   = '2024-06-10';
var cloudPct  = 10;                // max CLOUDY_PIXEL_PERCENTAGE
var visParams = {bands: ['B4','B3','B2'], min: 0, max: 3000, gamma: 1.2};

// 3. Prepare Sentinel-2 collection (filtered by date/cloud)
var s2Collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
  .filterDate(startDate, endDate)
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloudPct));

// 4. Convert to client-side list to iterate and create named layers
var districtList = districts.toList(districts.size());
var nDistricts = districtList.size().getInfo(); // small client-side call to loop

for (var i = 0; i < nDistricts; i++) {
  // get feature and name
  var feat = ee.Feature(districtList.get(i));
  var name = feat.get('ADM2_NAME').getInfo(); // district name as string

  // create median mosaic only over the district and clip to its geometry
  var img = s2Collection
    .filterBounds(feat.geometry())
    .median()
    .clip(feat.geometry());

  // add the image layer (named with district)
  Map.addLayer(img, visParams, name);

  // FIX: wrap the single feature into a FeatureCollection before styling
  var boundary = ee.FeatureCollection(feat).style({
    color: '000000',       // stroke color (black)
    fillColor: '00000000', // fully transparent fill
    width: 1
  });
  Map.addLayer(boundary, {}, name + ' boundary');
}

// 5. Helpful overview layer: blend of all districts (for reference)
var allRgb = s2Collection.median().clip(districts);
Map.addLayer(allRgb, visParams, 'Punjab - Sentinel-2 (All districts)');
Map.addLayer(districts.style({color:'FF0000', fillColor:'00000000', width:1}), {}, 'Punjab districts outline');