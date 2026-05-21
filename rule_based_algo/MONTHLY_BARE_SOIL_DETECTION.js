/**************************************************************
MONTHLY BARE SOIL DETECTION - ALL PUNJAB DISTRICTS
For each district, for each month (Jan-Dec 2024):
  - Export 1: Sentinel-2 median composite (all bands)
  - Export 2: Crop mask (ESA ONLY)
  - Export 3: Bare soil mask (NDVI ∧ BSI)

UPDATED: Using ONLY ESA WorldCover for crop mask (no Dynamic World)
Structure: 17 districts × 12 month folders × 3 files = 612 exports
Each district has 12 folders (one per month)
Each month folder contains 3 files
**************************************************************/

var districtsConfig = {
  'Amritsar': {ndvi: 0.1481653568923403, bsi: 0.10878096009294193, gaulName: 'Amritsar'},
  'Bathinda': {ndvi: 0.11291772127151489, bsi: 0.11519411951303482, gaulName: 'Bathinda'},
  'Faridkot': {ndvi: 0.09783066064119339, bsi: 0.11690010130405426, gaulName: 'Faridkot'},
  'Fatehgarh_Sahib': {ndvi: 0.11249449849128723, bsi: 0.10798495821654797, gaulName: 'Fatehgarh Sahib'},
  'Firozpur': {ndvi: 0.1012588068842888, bsi: 0.10941299796104431, gaulName: 'Firozpur'},
  'Gurdaspur': {ndvi: 0.14481832087039948, bsi: 0.11162389069795609, gaulName: 'Gurdaspur'},
  'Hoshiarpur': {ndvi: 0.13302426040172577, bsi: 0.11243852972984314, gaulName: 'Hoshiarpur'},
  'Jalandhar': {ndvi: 0.15703558921813965, bsi: 0.1082776840776205, gaulName: 'Jalandhar'},
  'Kapurthala': {ndvi: 0.14510729908943176, bsi: 0.11014474928379059, gaulName: 'Kapurthala'},
  'Ludhiana': {ndvi: 0.14119543135166168, bsi: 0.10648281872272491, gaulName: 'Ludhiana'},
  'Mansa': {ndvi: 0.11649054288864136, bsi: 0.11928980797529221, gaulName: 'Mansa'},
  'Moga': {ndvi: 0.14073002338409424, bsi: 0.1133353989571333, gaulName: 'Moga'},
  'Muktsar': {ndvi: 0.1199018582701683, bsi: 0.11184468865394592, gaulName: 'Muktsar'},
  'Nawan_Shehar': {ndvi: 0.14197164773941040, bsi: 0.10528685152530670, gaulName: 'Nawan Shehar'},
  'Patiala': {ndvi: 0.10680353721282682, bsi: 0.17679147831459713, gaulName: 'Patiala'},
  'Rupnagar': {ndvi: 0.12158370763063431, bsi: 0.11021906137466431, gaulName: 'Rupnagar'},
  'Sangrur': {ndvi: 0.12563292682170868, bsi: 0.11262594908475876, gaulName: 'Sangrur'}
};

var months = [
  {month: 1, name: 'January', days: 31},
  {month: 2, name: 'February', days: 29},  // 2024 is leap year
  {month: 3, name: 'March', days: 31},
  {month: 4, name: 'April', days: 30},
  {month: 5, name: 'May', days: 31},
  {month: 6, name: 'June', days: 30},
  {month: 7, name: 'July', days: 31},
  {month: 8, name: 'August', days: 31},
  {month: 9, name: 'September', days: 30},
  {month: 10, name: 'October', days: 31},
  {month: 11, name: 'November', days: 30},
  {month: 12, name: 'December', days: 31}
];

var cloudPctThreshold = 20;
var year = 2024;
var gaul2 = ee.FeatureCollection("FAO/GAUL/2015/level2");

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

function processDistrictMonth(districtKey, config, monthInfo) {
  var monthNum = monthInfo.month;
  var monthName = monthInfo.name;
  var lastDay = monthInfo.days;
  
  var startDate = ee.Date.fromYMD(year, monthNum, 1);
  var endDate = ee.Date.fromYMD(year, monthNum, lastDay);
  
  print('Processing: ' + districtKey + ' - ' + monthName + ' ' + year);
  
  // Get district geometry
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
  
  // Build crop mask using ONLY ESA WorldCover (class 40 = cropland)
  var esa = ee.ImageCollection('ESA/WorldCover/v200').first()
              .select('Map').clip(districtGeom);
  var cropMask = esa.eq(40).selfMask().rename('CropMask');
  
  // Build monthly Sentinel-2 median composite
  var s2col = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    .filterDate(startDate, endDate)
    .filterBounds(districtGeom)
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloudPctThreshold));
  
  var medianS2 = ee.Image(ee.Algorithms.If(
    s2col.size().gt(0),
    s2col.median(),
    ee.Image.constant(0).rename(['B1','B2','B3','B4','B5','B6','B7','B8','B8A','B9','B11','B12'])
  )).clip(districtGeom);
  
  // Compute indices and bare soil mask
  var medianWithIdx = addIndices(medianS2);
  var ndviCond = medianWithIdx.select('NDVI').lte(config.ndvi).and(cropMask);
  var bsiCond = medianWithIdx.select('BSI').gte(config.bsi).and(cropMask);
  var intersectionBool = ndviCond.and(bsiCond);
  var bareSoilMask = intersectionBool.updateMask(intersectionBool).unmask(0).rename('BareSoil');
  
  // Prepare crop mask for export (convert to byte)
  var cropMaskExport = cropMask.unmask(0).toByte();
  
  // FOLDER STRUCTURE: Punjab_District/Month_Year/files
  var monthStr = (monthNum < 10) ? '0' + monthNum : '' + monthNum;
  var folderName = 'Punjab_' + districtKey + '/' + monthName + '_2024';
  var filePrefix = districtKey + '_2024' + monthStr + '_' + monthName;
  
  // Export 1: Sentinel-2 all bands
  Export.image.toDrive({
    image: medianS2.toFloat(),
    description: districtKey + '_' + monthName + '_S2',
    folder: folderName,
    fileNamePrefix: filePrefix + '_S2_AllBands',
    region: districtGeom,
    scale: 10,
    maxPixels: 1e13,
    crs: 'EPSG:4326'
  });
  
  // Export 2: Crop mask (ESA only)
  Export.image.toDrive({
    image: cropMaskExport,
    description: districtKey + '_' + monthName + '_CropMask',
    folder: folderName,
    fileNamePrefix: filePrefix + '_CropMask',
    region: districtGeom,
    scale: 10,
    maxPixels: 1e13,
    crs: 'EPSG:4326'
  });
  
  // Export 3: Bare soil mask
  Export.image.toDrive({
    image: bareSoilMask.toByte(),
    description: districtKey + '_' + monthName + '_BareSoil',
    folder: folderName,
    fileNamePrefix: filePrefix + '_BareSoil',
    region: districtGeom,
    scale: 10,
    maxPixels: 1e13,
    crs: 'EPSG:4326'
  });
}

// Main processing
print('╔════════════════════════════════════════════════════════╗');
print('║  PUNJAB MONTHLY BARE SOIL DETECTION - 2024           ║');
print('║  17 Districts × 12 Months × 3 Files = 612 Exports    ║');
print('║  UPDATED: ESA WorldCover ONLY (no Dynamic World)     ║');
print('╚════════════════════════════════════════════════════════╝\n');

var districtKeys = Object.keys(districtsConfig);
var totalExports = 0;

for (var i = 0; i < districtKeys.length; i++) {
  var districtKey = districtKeys[i];
  var config = districtsConfig[districtKey];
  
  print('\n▶ District: ' + districtKey + ' (' + (i+1) + '/' + districtKeys.length + ')');
  
  for (var m = 0; m < months.length; m++) {
    processDistrictMonth(districtKey, config, months[m]);
    totalExports += 3;
  }
  
  print('  ✓ Configured 36 exports for ' + districtKey + ' (12 month folders)');
}

print('\n╔════════════════════════════════════════════════════════╗');
print('║  CONFIGURATION COMPLETE                               ║');
print('║                                                        ║');
print('║  Total exports configured: ' + totalExports + '                       ║');
print('║  Districts processed: ' + districtKeys.length + '                             ║');
print('║  Month folders per district: 12                       ║');
print('║  Files per month folder: 3                            ║');
print('║                                                        ║');
print('║  CROP MASK SOURCE:                                    ║');
print('║  ✓ ESA WorldCover v200 (2021)                         ║');
print('║  ✓ Class 40 = Cropland                                ║');
print('║  ✗ Dynamic World NOT used                             ║');
print('║                                                        ║');
print('║  FOLDER STRUCTURE:                                    ║');
print('║                                                        ║');
print('║  Google Drive/                                        ║');
print('║    Punjab_Amritsar/                                   ║');
print('║      January_2024/                                    ║');
print('║        Amritsar_202401_January_S2_AllBands.tif       ║');
print('║        Amritsar_202401_January_CropMask.tif          ║');
print('║        Amritsar_202401_January_BareSoil.tif          ║');
print('║      February_2024/                                   ║');
print('║        ... (3 files)                                  ║');
print('║      ... (12 month folders total)                     ║');
print('║    Punjab_Bathinda/                                   ║');
print('║      ... (12 month folders)                           ║');
print('║    ... (17 district folders total)                    ║');
print('║                                                        ║');
print('║  FILES PER MONTH FOLDER:                              ║');
print('║  1. Sentinel-2 (all 12 bands) - Float                 ║');
print('║  2. Crop Mask (ESA only, binary) - Byte               ║');
print('║  3. Bare Soil Mask (NDVI ∧ BSI, binary) - Byte        ║');
print('║                                                        ║');
print('║  NEXT STEPS:                                          ║');
print('║  1. Go to Tasks tab (top-right)                       ║');
print('║  2. Use bulk runner script to automate               ║');
print('║  3. Tasks will run in parallel (typically 15-20)      ║');
print('║  4. Each task takes 5-20 minutes                      ║');
print('║  5. Total time: ~8-12 hours for all 612 exports       ║');
print('╚════════════════════════════════════════════════════════╝');

print('\n💡 TIP: ESA crop mask will be identical across all months');
print('   (static 2021 land cover classification).');
print('\n💡 TIP: Use the improved bulk runner script to avoid');
print('   skipping tasks after 300 iterations.');