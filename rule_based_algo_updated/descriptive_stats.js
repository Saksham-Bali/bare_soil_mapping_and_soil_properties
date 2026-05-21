// Output: CSV to Drive with stats for each district

// ------------ 1. GENERAL PARAMETERS ------------
var COUNTRY_NAME = 'India';
var STATE_NAME = 'Karnataka'; // Change this for new states!
var startDate = '2024-03-01';
var endDate = '2024-04-01';
var cloudPct = 10;
var scale = 10;

var driveFolder = STATE_NAME.replace(/\s+/g, '_') + '_Districts_thresholds';

// ============== 2. PASTE PYTHON OUTPUT HERE ==============

// Paste the entire output from the Python script csv_polygon_to_js.py below this line
var districtPlots = {
};

// ==========================================================


// ============== 3. SMART DISTRICT REGISTRATION ==============
// Automatically extracts names, fixes underscores for GAUL, and checks arrays
var activeDistricts = Object.keys(districtPlots).map(function (key) {
    return {
        name: key.replace(/_/g, ' '), 
        safeName: key,              
        plots: districtPlots[key]
    };
}).filter(function (dist) {
    return dist.plots.length > 0;
});

print('Number of active districts to process:', activeDistricts.length);

if (activeDistricts.length === 0) {
    print('NO POLYGONS FOUND: Please paste your Python output into the districtPlots variable.');
}

// ============== 4. BUILD MEDIAN S2 & COMPUTE INDICES ==============
var bandList = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12'];

var stateFilter = ee.FeatureCollection('FAO/GAUL/2015/level1')
    .filter(ee.Filter.eq('ADM0_NAME', COUNTRY_NAME))
    .filter(ee.Filter.eq('ADM1_NAME', STATE_NAME));

var s2col = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    .filterDate(startDate, endDate)
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloudPct))
    .filterBounds(stateFilter.geometry());

var s2Median = s2col.median().select(bandList).toFloat();

// Compute indices
var ndvi = s2Median.normalizedDifference(['B8', 'B4']).rename('NDVI');
var savi = s2Median.expression(
    '((NIR - RED) * L) / (NIR + RED + (L * 0.5))',
    { 'NIR': s2Median.select('B8'), 'RED': s2Median.select('B4'), 'L': 0.5 }
).rename('SAVI');
var ndwi = s2Median.normalizedDifference(['B3', 'B11']).rename('NDWI');
var bsi = s2Median.expression(
    '((SWIR + RED) - (NIR + BLUE)) / ((SWIR + RED) + (NIR + BLUE))',
    {
        'SWIR': s2Median.select('B11'), 'RED': s2Median.select('B4'),
        'NIR': s2Median.select('B8'), 'BLUE': s2Median.select('B2')
    }
).rename('BSI');
var ndbi = s2Median.normalizedDifference(['B11', 'B8']).rename('NDBI');

var imageWithIndices = s2Median.addBands([ndvi, savi, ndwi, bsi, ndbi]).toFloat();
var allBands = imageWithIndices.bandNames();

// ============== 5. PROCESS EACH DISTRICT ==============
var statSuffixes = [
    { key: 'min', label: 'min' },
    { key: 'max', label: 'max' },
    { key: 'mean', label: 'mean' },
    { key: 'stdDev', label: 'stdDev' },
    { key: 'p10', label: 'p10' },
    { key: 'p25', label: 'p25' },
    { key: 'p50', label: 'p50' },
    { key: 'p75', label: 'p75' }
];

var percentileList = [10, 25, 50, 75];
var reducers = ee.Reducer.minMax()
    .combine(ee.Reducer.mean(), '', true)
    .combine(ee.Reducer.stdDev(), '', true)
    .combine(ee.Reducer.percentile(percentileList), '', true);

// Function to process a single district
var processDistrict = function (districtInfo) {
    var districtName = districtInfo.name;
    var safeName = districtInfo.safeName;
    var plots = districtInfo.plots;

    // Create feature collection from plots
    var plotFeatures = plots.map(function (geom, idx) {
        return ee.Feature(geom, { id: safeName + '_plot' + (idx + 1) });
    });
    var plotsFC = ee.FeatureCollection(plotFeatures);

    
    var unionGeom = plotsFC.geometry(1); 

    // Compute stats
    var stats = imageWithIndices.reduceRegion({
        reducer: reducers,
        geometry: unionGeom,
        scale: scale,
        maxPixels: 1e10,
        tileScale: 16,
        bestEffort: true
    });

    // Build features for each stat
    var makeFeatureForStat = function (statSuffix, label) {
        var statKey = statSuffix;
        var values = allBands.map(function (b) {
            b = ee.String(b);
            var key = b.cat('_').cat(statKey);
            return ee.Dictionary(stats).get(key);
        });
        var props = ee.Dictionary.fromLists(allBands, values);
        props = props.set('stat', label)
            .set('district', districtName)
            .set('startDate', startDate)
            .set('endDate', endDate);
        return ee.Feature(null, props);
    };

    return ee.FeatureCollection(
        ee.List(statSuffixes).map(function (s) {
            s = ee.Dictionary(s);
            return makeFeatureForStat(s.get('key'), s.get('label'));
        })
    );
};

// ============== 6. PROCESS AND EXPORT ==============
activeDistricts.forEach(function (dist) {
    var districtStats = processDistrict(dist);

    print('Processing district:', dist.name);
    print('  - Number of plots loaded:', dist.plots.length);
    print('  - Number of stat rows:', districtStats.size());

    Export.table.toDrive({
        collection: districtStats,
        description: dist.safeName + '_thresholds',
        folder: driveFolder,
        fileNamePrefix: dist.safeName + '_thresholds',
        fileFormat: 'CSV'
    });

    print('  - Export scheduled: ' + dist.safeName + '_thresholds.csv');
});

print('========================================');
print('All active exports scheduled');