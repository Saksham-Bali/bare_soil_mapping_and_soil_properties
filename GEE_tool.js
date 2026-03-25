// ===================================================================================
// 0. SETUP: Load data and define parameters
// ===================================================================================

// Load states (level 1) and districts (level 2) for India
var states = ee.FeatureCollection('FAO/GAUL/2015/level1')
    .filter(ee.Filter.eq('ADM0_NAME', 'India'));
var districts = ee.FeatureCollection('FAO/GAUL/2015/level2')
    .filter(ee.Filter.eq('ADM0_NAME', 'India'));

// S2 Parameters - default values
var cloudPct = 10; // Increased slightly since we are now masking pixels
var visParams = { bands: ['B4', 'B3', 'B2'], min: 0, max: 3000, gamma: 1.2 };

// ===================================================================================
// 1. DEFINE UI WIDGETS AND PANEL
// ===================================================================================

// --- Create the main panel ---
var panel = ui.Panel({
    style: { width: '320px', position: 'top-left' }
});
ui.root.add(panel);

// --- 1. Date Range Selection ---
panel.add(ui.Label('1. Select Date Range:', { fontWeight: 'bold' }));

// Start Date
var startDateBox = ui.Textbox({
    placeholder: 'YYYY-MM-DD',
    value: '2024-04-01',
    style: { stretch: 'horizontal' }
});
panel.add(ui.Label('Start Date:'));
panel.add(startDateBox);

// End Date
var endDateBox = ui.Textbox({
    placeholder: 'YYYY-MM-DD',
    value: '2024-05-01',
    style: { stretch: 'horizontal' }
});
panel.add(ui.Label('End Date:'));
panel.add(endDateBox);

// Apply Date Button
var applyDateButton = ui.Button({
    label: 'Apply Date Range',
    onClick: function () {
        var start = startDateBox.getValue();
        var end = endDateBox.getValue();

        // Validate date format (basic check)
        var datePattern = /^\d{4}-\d{2}-\d{2}$/;
        if (!datePattern.test(start) || !datePattern.test(end)) {
            print('Error: Please use YYYY-MM-DD format for dates.');
            return;
        }

        // Check if district is selected
        var districtName = currentDistrictLabel.getValue();
        if (districtName === 'No district loaded') {
            print('Please select a district first, then apply date range.');
            return;
        }

        print('Date range updated: ' + start + ' to ' + end);
        // Reload the district with new dates
        onDistrictChange(districtName);
    },
    style: { stretch: 'horizontal' }
});
panel.add(applyDateButton);

panel.add(ui.Label('─────────────────────────'));

// --- 2. State Selector ---
panel.add(ui.Label('2. Select a State:', { fontWeight: 'bold' }));
var stateSelect = ui.Select({
    placeholder: 'Loading states...',
    style: { stretch: 'horizontal' },
    onChange: onStateChange
});
panel.add(stateSelect);

// --- 3. District Selector ---
panel.add(ui.Label('3. Select a District:', { fontWeight: 'bold' }));
var districtSelect = ui.Select({
    placeholder: 'Select a state first...',
    style: { stretch: 'horizontal' },
    onChange: onDistrictChange
});
panel.add(districtSelect);

// --- 4. Label to store the current district name (hidden) ---
var currentDistrictLabel = ui.Label('No district loaded');
currentDistrictLabel.style().set('shown', false);
panel.add(currentDistrictLabel);

panel.add(ui.Label('─────────────────────────'));

// --- 5. Save Button ---
panel.add(ui.Label('4. Draw Polygons & Save:', { fontWeight: 'bold' }));
var saveButton = ui.Button({
    label: 'Save Drawn Polygons',
    onClick: savePolygons,
    style: { stretch: 'horizontal', color: 'green' }
});
panel.add(saveButton);

// --- 6. Clear Polygons Button ---
var clearButton = ui.Button({
    label: 'Clear All Polygons',
    onClick: clearAllPolygons,
    style: { stretch: 'horizontal', color: 'red' }
});
panel.add(clearButton);

// ===================================================================================
// 2. DEFINE CALLBACK FUNCTIONS & HELPERS
// ===================================================================================

// --- Helper: Cloud Masking Function ---
function maskS2clouds(image) {
    var qa = image.select('QA60');

    // Bits 10 and 11 are clouds and cirrus, respectively.
    var cloudBitMask = 1 << 10;
    var cirrusBitMask = 1 << 11;

    // Both flags should be set to zero, indicating clear conditions.
    var mask = qa.bitwiseAnd(cloudBitMask).eq(0)
        .and(qa.bitwiseAnd(cirrusBitMask).eq(0));

    // Return the masked image. 
    // Note: We do NOT divide by 10000 here so that we keep your original visParams (0-3000) working.
    return image.updateMask(mask);
}

// --- A. Runs when a state is selected ---
function onStateChange(stateName) {
    districtSelect.items().reset();
    districtSelect.setPlaceholder('Loading districts...');
    currentDistrictLabel.setValue('No district loaded');

    var districtsInState = districts
        .filter(ee.Filter.eq('ADM1_NAME', stateName))
        .aggregate_array('ADM2_NAME')
        .distinct()
        .sort();

    districtsInState.evaluate(function (districtList) {
        districtSelect.items().reset(districtList);
        districtSelect.setPlaceholder('Select a district');
    });
}

// --- B. Runs when a district is selected ---
function onDistrictChange(districtName) {
    if (!districtName) return;

    // 1. Update the hidden label with the current district name
    currentDistrictLabel.setValue(districtName);

    // 2. Clear the map
    Map.clear();
    Map.drawingTools().setShown(true);

    // 3. Get the date range from the textboxes
    var startDate = startDateBox.getValue();
    var endDate = endDateBox.getValue();

    // 4. Get the single district's geometry
    var stateName = stateSelect.getValue();
    var districtFeat = districts
        .filter(ee.Filter.and(
            ee.Filter.eq('ADM1_NAME', stateName),
            ee.Filter.eq('ADM2_NAME', districtName)
        ))
        .first();

    // 5. Center map on the district
    Map.centerObject(districtFeat, 10);

    // -------------------------------------------------------------
    // LAYER 1: ESA WorldCover 2021 (Binary Crop Mask)
    // -------------------------------------------------------------
    var worldCover = ee.ImageCollection('ESA/WorldCover/v200').first();

    // Select the classification map
    var landCover = worldCover.select('Map');

    // Create a binary image: 1 if Cropland (Class 40), 0 otherwise
    var binaryCropMask = landCover.eq(40);

    // Clip to district
    var binaryCropMaskClipped = binaryCropMask.clip(districtFeat.geometry());



    // -------------------------------------------------------------
    // LAYER 2: Sentinel-2 (Cloud Masked + Median)
    // -------------------------------------------------------------

    // Create Sentinel-2 collection with cloud masking
    var s2Collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
        .filterDate(startDate, endDate)
        // Filter scenes that are too cloudy overall
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloudPct))
        // Apply the cloud mask to every image in the collection
        .map(maskS2clouds);

    // We take the median. Because we masked clouds earlier, the median will ignore cloudy pixels.
    var img = s2Collection
        .filterBounds(districtFeat.geometry())
        .median()
        .clip(districtFeat.geometry());

    Map.addLayer(img, visParams, districtName + ' S2 Cloud-Masked');
    // Add to map
    // 0 (Non-Crop) will be black, 1 (Crop) will be Yellow
    Map.addLayer(binaryCropMaskClipped, { min: 0, max: 1, palette: ['000000', 'FFFF00'] }, districtName + ' Binary Crop Mask (Yellow=Crop)');


    // -------------------------------------------------------------
    // LAYER 3: District Boundary
    // -------------------------------------------------------------
    var boundary = ee.FeatureCollection([districtFeat]).style({
        color: '000000',
        fillColor: '00000000',
        width: 2
    });
    Map.addLayer(boundary, {}, districtName + ' boundary');

    print('Loaded ' + districtName + ' with date range: ' + startDate + ' to ' + endDate);
}

// --- C. Runs when the 'Save' button is clicked ---
function savePolygons() {
    var districtName = currentDistrictLabel.getValue();
    var stateName = stateSelect.getValue();

    if (districtName === 'No district loaded') {
        print('Error: Please select a district first.');
        return;
    }

    var drawnPolygons = Map.drawingTools().toFeatureCollection();

    if (drawnPolygons.size().getInfo() === 0) {
        print('Error: No polygons drawn. Draw at least one polygon first.');
        return;
    }

    // Clean the district name: replace spaces with underscores
    var safeDistrictName = districtName.replace(/\s+/g, '_');

    // Construct the filename using the safe name
    var fileName = safeDistrictName + '_plots';

    // Construct dynamic folder path: Ground_truth_SOC/StateName
    var dynamicFolder = 'Ground_truth_SOC_' + stateName;

    print('Exporting polygons for: ' + districtName + ' in ' + stateName);
    print('Filename will be: ' + fileName);
    print('Destination Folder: ' + dynamicFolder);

    Export.table.toDrive({
        collection: drawnPolygons,
        description: fileName,
        fileNamePrefix: fileName,
        fileFormat: 'CSV',
        folder: dynamicFolder
    });
}

// --- D. Runs when the 'Clear All Polygons' button is clicked ---
function clearAllPolygons() {
    var layers = Map.drawingTools().layers();

    if (layers.length() === 0) {
        print('No polygons to clear.');
        return;
    }

    // Remove all drawing layers
    while (layers.length() > 0) {
        var layer = layers.get(0);
        layers.remove(layer);
    }

    print('All polygons cleared. Ready to draw new polygons.');
}

// ===================================================================================
// 3. INITIALIZE THE SCRIPT
// ===================================================================================

states.aggregate_array('ADM1_NAME').distinct().sort().evaluate(function (stateList) {
    stateSelect.items().reset(stateList);
    stateSelect.setPlaceholder('Select a state');
});

Map.drawingTools().setShown(true);
Map.centerObject(districts, 5);