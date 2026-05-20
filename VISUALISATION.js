/**************************************************************
ALL PUNJAB DISTRICTS — IMPROVED Bare Soil Detection
Uses MEAN NDVI and p10 BSI for MORE SENSITIVE detection
** ESA WorldCover ONLY (Dynamic World Removed) **
**************************************************************/

var districtsConfig = {
  'Amritsar': {
    ndvi: 0.1481653568923403,
    bsi: 0.10878096009294193,
    startDate: '2024-05-01',
    endDate: '2024-06-10',
    gaulName: 'Amritsar'
  },
  'Bathinda': {
    ndvi: 0.11291772127151489,
    bsi: 0.11519411951303482,
    startDate: '2024-05-01',
    endDate: '2024-06-10',
    gaulName: 'Bathinda'
  },
  'Faridkot': {
    ndvi: 0.09783066064119339,
    bsi: 0.11690010130405426,
    startDate: '2024-05-01',
    endDate: '2024-06-10',
    gaulName: 'Faridkot'
  },
  'Fatehgarh_Sahib': {
    ndvi: 0.11249449849128723,
    bsi: 0.10798495821654797,
    startDate: '2024-05-01',
    endDate: '2024-06-10',
    gaulName: 'Fatehgarh Sahib'
  },
  'Firozpur': {
    ndvi: 0.1012588068842888,
    bsi: 0.10941299796104431,
    startDate: '2024-05-01',
    endDate: '2024-06-10',
    gaulName: 'Firozpur'
  },
  'Gurdaspur': {
    ndvi: 0.14481832087039948,
    bsi: 0.11162389069795609,
    startDate: '2024-05-01',
    endDate: '2024-06-10',
    gaulName: 'Gurdaspur'
  },
  'Hoshiarpur': {
    ndvi: 0.13302426040172577,
    bsi: 0.11243852972984314,
    startDate: '2024-05-01',
    endDate: '2024-06-10',
    gaulName: 'Hoshiarpur'
  },
  'Jalandhar': {
    ndvi: 0.15703558921813965,
    bsi: 0.1082776840776205,
    startDate: '2024-05-01',
    endDate: '2024-06-10',
    gaulName: 'Jalandhar'
  },
  'Kapurthala': {
    ndvi: 0.14510729908943176,
    bsi: 0.11014474928379059,
    startDate: '2024-05-01',
    endDate: '2024-06-10',
    gaulName: 'Kapurthala'
  },
  'Ludhiana': {
    ndvi: 0.14119543135166168,
    bsi: 0.10648281872272491,
    startDate: '2024-05-01',
    endDate: '2024-06-10',
    gaulName: 'Ludhiana'
  },
  'Mansa': {
    ndvi: 0.11649054288864136,
    bsi: 0.11928980797529221,
    startDate: '2024-05-01',
    endDate: '2024-06-10',
    gaulName: 'Mansa'
  },
  'Moga': {
    ndvi: 0.14073002338409424,
    bsi: 0.1133353989571333,
    startDate: '2024-05-01',
    endDate: '2024-06-10',
    gaulName: 'Moga'
  },
  'Muktsar': {
    ndvi: 0.1199018582701683,
    bsi: 0.11184468865394592,
    startDate: '2024-05-01',
    endDate: '2024-06-10',
    gaulName: 'Muktsar'
  },
  'Nawanshahr': {
    ndvi: 0.14197164773941040,
    bsi: 0.10528685152530670,
    startDate: '2024-05-01',
    endDate: '2024-06-10',
    gaulName: 'Nawanshahr'
  },
  'Patiala': {
    ndvi: 0.10680353721282682,
    bsi: 0.17679147831459713,
    startDate: '2024-05-01',
    endDate: '2024-06-10',
    gaulName: 'Patiala'
  },
  'Rupnagar': {
    ndvi: 0.12158370763063431,
    bsi: 0.11021906137466431,
    startDate: '2024-05-01',
    endDate: '2024-06-10',
    gaulName: 'Rupnagar'
  },
  'Sangrur': {
    ndvi: 0.12563292682170868,
    bsi: 0.11262594908475876,
    startDate: '2024-05-01',
    endDate: '2024-06-10',
    gaulName: 'Sangrur'
  }
};

var cloudPctThreshold = 20;
var year = 2024;
var gaul2 = ee.FeatureCollection("FAO/GAUL/2015/level2");

var s2VisParams = {bands: ['B4', 'B3', 'B2'], min: 0, max: 3000};
var cropMaskVis = {palette: ['#00FF00'], opacity: 0.6};
var bareSoilVis = {palette: ['#A0522D'], opacity: 0.8};
var boundaryVis = {color: '#0000FF', width: 2, fillColor: '00000000'};

Map.setCenter(75.5, 30.9, 8);

function addIndices(img) {
  var nir = img.select('B8');
  var red = img.select('B4');
  var blue = img.select('B2');
  var swir = img.select('B11');
  var ndvi = nir.subtract(red).divide(nir.add(red)).rename('NDVI');
  var bsi = swir.add(red).subtract(nir.add(blue))
              .divide(swir.add(red).add(nir.add(blue))).rename('BSI');
  return img.addBands([ndvi, bsi]);
}

function computeCountsAndArea(maskImg, bandLabel, geometry) {
  var pixelArea = ee.Image.pixelArea();
  var maskArea = maskImg.multiply(pixelArea).rename(bandLabel + '_m2');
  var combined = maskImg.addBands(maskArea);
  return combined.reduceRegion({
    reducer: ee.Reducer.sum().combine({
      reducer2: ee.Reducer.count(),
      sharedInputs: true
    }),
    geometry: geometry,
    scale: 10,
    maxPixels: 1e13
  });
}

function printStats(statsObj, label, districtName) {
  statsObj.evaluate(function(r) {
    print('=== ' + districtName + ' - ' + label + ' ===');
    if (!r) {
      print(label + ': No result returned.');
      return;
    }
    var sumKey = null, areaKey = null, countKey = null;
    for (var k in r) {
      if (!r.hasOwnProperty(k)) continue;
      var lname = String(k).toLowerCase();
      if (lname.indexOf('_m2_sum') !== -1 || lname.indexOf('_m2') !== -1) { 
        areaKey = k; 
      } else if (lname.indexOf('_sum') !== -1 && lname.indexOf('_m2') === -1) { 
        sumKey = k; 
      } else if (lname.indexOf('_count') !== -1 || lname.indexOf('_cnt') !== -1) { 
        countKey = k; 
      }
    }
    var pixels = sumKey ? (r[sumKey] || 0) : 0;
    var area_m2 = areaKey ? (r[areaKey] || 0) : 0;
    var totalPixelsConsidered = countKey ? (r[countKey] || 0) : 'unknown';
    var area_ha = (area_m2 / 10000.0);
    print('Pixels flagged:', pixels);
    print('Area (ha):', area_ha.toFixed(2));
    print('Total pixels:', totalPixelsConsidered);
  });
}

function processDistrict(districtKey, config) {
  print('\n========================================');
  print('PROCESSING: ' + districtKey);
  print('NDVI threshold (MEAN): ' + config.ndvi);
  print('BSI threshold (p10): ' + config.bsi);
  print('Date: ' + config.startDate + ' to ' + config.endDate);
  print('========================================\n');
  
  var districtFC = gaul2.filter(ee.Filter.eq('ADM0_NAME', 'India'))
                        .filter(ee.Filter.eq('ADM2_NAME', config.gaulName));
  var district = ee.Feature(ee.Algorithms.If(
    districtFC.size().gt(0), 
    districtFC.first(), 
    ee.Feature(null)
  ));
  var districtGeom = ee.Geometry(ee.Algorithms.If(
    district.geometry(), 
    district.geometry(),
    ee.Geometry.Point([75.5, 30.5]).buffer(30000)
  ));
  
  // --------------------------------------------------------------
  // MODIFIED: USE ESA WORLDCOVER ONLY (Dynamic World REMOVED)
  // --------------------------------------------------------------
  var esa = ee.ImageCollection('ESA/WorldCover/v200').first()
              .select('Map').clip(districtGeom);
  
  // Select only Class 40 (Cropland) from ESA
  var esaCrop = esa.eq(40).rename('esa_crop');
  
  // The crop mask is now JUST the ESA layer
  var cropMask = esaCrop.selfMask(); 
  // --------------------------------------------------------------
  
  var s2col = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    .filterDate(config.startDate, config.endDate)
    .filterBounds(districtGeom)
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloudPctThreshold));
  
  var medianS2 = ee.Image(ee.Algorithms.If(
    s2col.size().gt(0),
    s2col.median(),
    ee.Image.constant(0).rename(['B1','B2','B3','B4','B5','B6','B7','B8','B8A','B9','B11','B12'])
  )).clip(districtGeom);
  
  var medianWithIdx = addIndices(medianS2);
  
  var ndviCond = medianWithIdx.select('NDVI').lte(config.ndvi).and(cropMask);
  var bsiCond = medianWithIdx.select('BSI').gte(config.bsi).and(cropMask);
  
  var ndviMask = ndviCond.updateMask(ndviCond).unmask(0).rename('NDVI_mask');
  var bsiMask = bsiCond.updateMask(bsiCond).unmask(0).rename('BSI_mask');
  
  var intersectionBool = ndviCond.and(bsiCond);
  var intersectionMask = intersectionBool.updateMask(intersectionBool).unmask(0)
                                         .rename('BareSoil_mask');
  
  var ndviStats = computeCountsAndArea(ndviMask, 'NDVI', districtGeom);
  var bsiStats = computeCountsAndArea(bsiMask, 'BSI', districtGeom);
  var interStats = computeCountsAndArea(intersectionMask, 'BareSoil', districtGeom);
  
  printStats(ndviStats, 'NDVI condition (NDVI <= ' + config.ndvi + ')', districtKey);
  printStats(bsiStats, 'BSI condition (BSI >= ' + config.bsi + ')', districtKey);
  printStats(interStats, 'Bare Soil (NDVI ∧ BSI)', districtKey);
  
  Map.addLayer(districtGeom, boundaryVis, districtKey + ' - Boundary', false);
  Map.addLayer(medianS2, s2VisParams, districtKey + ' - Sentinel-2 RGB', false);
  Map.addLayer(cropMask, cropMaskVis, districtKey + ' - ESA Crop Mask Only', false);
  Map.addLayer(intersectionMask.selfMask(), bareSoilVis, districtKey + ' - Bare Soil Mask', false);
  Map.addLayer(medianS2.updateMask(intersectionMask), s2VisParams, districtKey + ' - S2 (Bare Soil Only)', false);
  
  print('✓ Visualization layers added for ' + districtKey);
  
  var exportPrefix = districtKey + '_' + config.startDate.replace(/-/g, '') + 
                     '_' + config.endDate.replace(/-/g, '');
  
  Export.image.toDrive({
    image: medianS2.toFloat(),
    description: districtKey + '_MedianS2_Export',
    folder: 'GEE_Punjab_Districts',
    fileNamePrefix: exportPrefix + '_MedianS2',
    region: districtGeom,
    scale: 10,
    maxPixels: 1e13,
    crs: 'EPSG:4326'
  });
  
  Export.image.toDrive({
    image: intersectionMask.toByte(),
    description: districtKey + '_BareSoilMask_Export',
    folder: 'GEE_Punjab_Districts',
    fileNamePrefix: exportPrefix + '_BareSoilMask',
    region: districtGeom,
    scale: 10,
    maxPixels: 1e13,
    crs: 'EPSG:4326'
  });
  
  print('✓ Exports configured for ' + districtKey);
}

print('╔════════════════════════════════════════════════════════╗');
print('║  PUNJAB DISTRICTS - IMPROVED BARE SOIL DETECTION       ║');
print('║  Using MEAN NDVI & p10 BSI (MORE SENSITIVE)            ║');
print('║  ** ESA WorldCover ONLY (No Dynamic World) ** ║');
print('║  Processing ' + Object.keys(districtsConfig).length + ' districts                            ║');
print('╚════════════════════════════════════════════════════════╝\n');

var districtKeys = Object.keys(districtsConfig);
for (var i = 0; i < districtKeys.length; i++) {
  var key = districtKeys[i];
  var config = districtsConfig[key];
  processDistrict(key, config);
}

print('\n╔════════════════════════════════════════════════════════╗');
print('║  PROCESSING COMPLETE                                   ║');
print('║  IMPROVED THRESHOLDS APPLIED:                          ║');
print('║    - NDVI: MEAN (more inclusive)                       ║');
print('║    - BSI: p10 (more sensitive)                         ║');
print('║  This will detect MORE bare land areas                 ║');
print('║                                                        ║');
print('║  Total districts: ' + districtKeys.length + '                        ║');
print('║  Total exports: ' + (districtKeys.length * 2) + '                          ║');
print('║  Total layers: ' + (districtKeys.length * 5) + '                           ║');
print('╚════════════════════════════════════════════════════════╝');