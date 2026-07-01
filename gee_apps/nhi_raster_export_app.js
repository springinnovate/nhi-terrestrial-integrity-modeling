var DEFAULTYEAR = "2005";
var DEFAULT_DRIVE_FOLDER = "gee_response_variables";
var DEFAULT_MAX_PIXELS = 1e13;
var EXPORT_CRS = "EPSG:4326";
var EXPORT_SCALE_METERS = 500;

var LANDSAT_NDVI_DATASET = "LANDSAT/COMPOSITES/C02/T1_L2_8DAY_NDVI";
var MODIS_PHENOLOGY_DATASET = "MODIS/061/MCD12Q2";
var SHORT_VEG_HEIGHT_DATASET =
    "projects/global-pasture-watch/assets/gsvh-30m/v1/short-veg-height_m";
var MODIS_VEGETATION_COVER_DATASET = "MODIS/061/MOD44B";
var MODIS_LAI_FPAR_DATASET = "MODIS/061/MOD15A2H";
var MODIS_PRODUCTIVITY_DATASET = "MODIS/061/MOD17A3HGF";
var ERA5_DAILY_DATASET = "ECMWF/ERA5/DAILY";
var ERA5_MONTHLY_DATASET = "ECMWF/ERA5/MONTHLY";
var GRIDMET_DROUGHT_DATASET = "GRIDMET/DROUGHT";
var VIIRS_BURNED_AREA_DATASET = "NASA/VIIRS/002/VNP64A1";
var JRC_MONTHLY_WATER_DATASET = "JRC/GSW1_4/MonthlyHistory";
var MERIT_HYDRO_DATASET = "MERIT/Hydro/v1_0_1";
var ISRIC_SOC_DATASET = "projects/soilgrids-isric/soc_mean";
var GLDAS_DATASET = "NASA/GLDAS/V021/NOAH/G025/T3H";
var SMAP_DATASET = "NASA/SMAP/SPL4SMGP/008";
var MODIS_ET_DATASET = "MODIS/061/MOD16A2GF";
var SRTM_LANDFORMS_DATASET = "CSP/ERGo/1_0/Global/SRTM_landforms";
var ALOS_TOPO_DIVERSITY_DATASET = "CSP/ERGo/1_0/Global/ALOS_topoDiversity";
var ERA5_START_YEAR = 1979;
var INTERANNUAL_RAINFALL_WINDOW_YEARS = 10;
var STREAM_UPSTREAM_AREA_THRESHOLD_KM2 = 25;

var REGION_DEFINITIONS = [
    {
        name: "Wyoming Basin",
        assetId: "projects/ecoshard-202922/assets/nhi_assets/wyoming_basin2"
    }
];

var GRASSLAND_PROB_IC = ee.ImageCollection(
    "projects/global-pasture-watch/assets/ggc-30m/v1/nat-semi-grassland_p"
);
var HMI_IMG = ee.Image(
    "projects/hm-30x30/assets/output/v20240801/HMv20240801_2022s_AA_300"
);
var HII_IC = ee
    .ImageCollection("projects/HII/v1/hii")
    .filterDate("2001-01-01", "2021-01-01");

var PROBABILITY_INTEGRITY_START_YEAR = 2001;
var PROBABILITY_INTEGRITY_END_YEAR = 2020;
var DEFAULT_GRASSLAND_PROB_THRESHOLD = 60;
var DEFAULT_HMI_THRESHOLD = 0.1;
var DEFAULT_HII_THRESHOLD = 0.08;

function yearRangePrompt(startYear, endYear) {
    return startYear + "-" + endYear;
}

var YEAR_PROMPT_GRASSLAND_REFERENCE = "reference period 2001-2020";
var YEAR_PROMPT_STATIC = "static";
var YEAR_PROMPT_LANDSAT = yearRangePrompt(1984, 2026);
var YEAR_PROMPT_MODIS_PHENOLOGY = yearRangePrompt(2001, 2024);
var YEAR_PROMPT_SHORT_VEG_HEIGHT = yearRangePrompt(2000, 2022);
var YEAR_PROMPT_MODIS_COVER = yearRangePrompt(2000, 2024);
var YEAR_PROMPT_MODIS_LAI_FPAR = yearRangePrompt(2000, 2026);
var YEAR_PROMPT_MODIS_PRODUCTIVITY = yearRangePrompt(2001, 2024);
var YEAR_PROMPT_ERA5_COMPLETE_YEARS = yearRangePrompt(1979, 2019);
var YEAR_PROMPT_GROWING_SEASON_CLIMATE = yearRangePrompt(2001, 2019);
var YEAR_PROMPT_GRIDMET_DROUGHT = yearRangePrompt(1980, 2025);
var YEAR_PROMPT_VIIRS_FIRE = yearRangePrompt(2012, 2026);
var YEAR_PROMPT_JRC_WATER = yearRangePrompt(1984, 2021);
var YEAR_PROMPT_GLDAS = yearRangePrompt(2000, 2026);
var YEAR_PROMPT_MODIS_ET = yearRangePrompt(2000, 2025);
var YEAR_PROMPT_SMAP = yearRangePrompt(2015, 2026);

function noTwoConsecutiveZerosFromAnnualBinary(buildAnnualBinary) {
    var years = ee.List.sequence(
        PROBABILITY_INTEGRITY_START_YEAR,
        PROBABILITY_INTEGRITY_END_YEAR
    );
    var annualBinaryIC = ee.ImageCollection.fromImages(
        years.map(function (year) {
            year = ee.Number(year);
            return ee
                .Image(buildAnnualBinary(year))
                .rename("g")
                .set("year", year);
        })
    );
    var list = annualBinaryIC.toList(annualBinaryIC.size());
    return ee.ImageCollection.fromImages(
        ee.List.sequence(0, ee.Number(list.size()).subtract(2)).map(
            function (i) {
                i = ee.Number(i);
                return ee.Image(list.get(i)).or(ee.Image(list.get(i.add(1))));
            }
        )
    )
        .reduce(ee.Reducer.min())
        .eq(1);
}

function defaultReferenceThresholds() {
    return {
        grasslandProbability: DEFAULT_GRASSLAND_PROB_THRESHOLD,
        hmi: DEFAULT_HMI_THRESHOLD,
        hii: DEFAULT_HII_THRESHOLD
    };
}

function probabilityIntegrityIndex(year, thresholds) {
    thresholds = thresholds || defaultReferenceThresholds();
    return noTwoConsecutiveZerosFromAnnualBinary(function (year) {
        return GRASSLAND_PROB_IC.filterDate(
            ee.Date.fromYMD(year, 1, 1),
            ee.Date.fromYMD(year.add(1), 1, 1)
        )
            .first()
            .select(0)
            .gte(thresholds.grasslandProbability);
    })
        .and(
            noTwoConsecutiveZerosFromAnnualBinary(function (year) {
                return HII_IC.filterDate(
                    ee.Date.fromYMD(year, 1, 1),
                    ee.Date.fromYMD(year.add(1), 1, 1)
                )
                    .mean()
                    .divide(7000)
                    .lt(thresholds.hii);
            })
        )
        .and(HMI_IMG.lte(thresholds.hmi))
        .selfMask()
        .toByte();
}

function yearStart(year) {
    return ee.Date.fromYMD(ee.Number(year).toInt(), 1, 1);
}

function annualCollection(dataset, year) {
    var start = yearStart(year);
    return ee
        .ImageCollection(dataset)
        .filterDate(start, start.advance(1, "year"));
}

function annualFirst(dataset, year) {
    return annualCollection(dataset, year).first();
}

function landsatNdviPercentile(percentile) {
    return function (year) {
        return annualCollection(LANDSAT_NDVI_DATASET, year)
            .select("NDVI")
            .reduce(ee.Reducer.percentile([percentile]));
    };
}

function modisPhenologyDayOfYear(year, bandName) {
    var epochStart = ee.Date("1970-01-01");
    var startDay = yearStart(year).difference(epochStart, "day");

    return ee
        .Image(annualFirst(MODIS_PHENOLOGY_DATASET, year))
        .select(bandName)
        .subtract(startDay);
}

function modisPhenologyBand(bandName) {
    return function (year) {
        return ee
            .Image(annualFirst(MODIS_PHENOLOGY_DATASET, year))
            .select(bandName);
    };
}

function modisGrowingSeasonLength(cycleNumber) {
    return function (year) {
        var phenology = ee.Image(annualFirst(MODIS_PHENOLOGY_DATASET, year));
        return phenology
            .select("Senescence_" + cycleNumber)
            .subtract(phenology.select("Greenup_" + cycleNumber));
    };
}

function modisGreenupTiming(cycleNumber) {
    return function (year) {
        return modisPhenologyDayOfYear(year, "Greenup_" + cycleNumber);
    };
}

function shortVegetationHeight(year) {
    return ee
        .Image(annualFirst(SHORT_VEG_HEIGHT_DATASET, year))
        .select("height")
        .multiply(0.1);
}

function modisVegetationCover(bandName) {
    return function (year) {
        return ee
            .Image(annualFirst(MODIS_VEGETATION_COVER_DATASET, year))
            .select(bandName);
    };
}

function scaledAnnualCollection(dataset, year, bandName, scale) {
    return annualCollection(dataset, year)
        .select(bandName)
        .map(function (image) {
            return image
                .multiply(scale)
                .copyProperties(image, ["system:time_start"]);
        });
}

function annualScaledSummary(dataset, bandName, scale, reducer) {
    return function (year) {
        return scaledAnnualCollection(dataset, year, bandName, scale).reduce(
            reducer
        );
    };
}

function annualScaledFirst(dataset, bandName, scale) {
    return function (year) {
        return ee
            .Image(annualFirst(dataset, year))
            .select(bandName)
            .multiply(scale);
    };
}

function toCelsius(image) {
    return image.subtract(273.15);
}

function toMillimeters(image) {
    return image.multiply(1000);
}

function era5DailyForYear(year) {
    return annualCollection(ERA5_DAILY_DATASET, year);
}

function era5MonthlyForYear(year) {
    return annualCollection(ERA5_MONTHLY_DATASET, year);
}

function annualMaxTemperatureForYear(year) {
    return toCelsius(
        era5DailyForYear(year).select("maximum_2m_air_temperature").max()
    );
}

function annualMeanTemperatureForYear(year) {
    return toCelsius(
        era5MonthlyForYear(year).select("mean_2m_air_temperature").mean()
    );
}

function annualMedianTemperatureForYear(year) {
    return toCelsius(
        era5MonthlyForYear(year).select("mean_2m_air_temperature").median()
    );
}

function annualMinTemperatureForYear(year) {
    return toCelsius(
        era5DailyForYear(year).select("minimum_2m_air_temperature").min()
    );
}

function annualPrecipForYear(year) {
    return toMillimeters(
        era5DailyForYear(year).select("total_precipitation").sum()
    );
}

function growingSeasonDailyForYear(year) {
    var epochStart = ee.Date("1970-01-01");
    var phenology = ee.Image(annualFirst(MODIS_PHENOLOGY_DATASET, year));
    var greenup = phenology.select("Greenup_1");
    var senescence = phenology.select("Senescence_1");

    return era5DailyForYear(year).map(function (image) {
        var imageDay = ee
            .Date(image.get("system:time_start"))
            .difference(epochStart, "day");
        var growingSeasonMask = ee.Image.constant(imageDay)
            .gte(greenup)
            .and(ee.Image.constant(imageDay).lte(senescence));

        return image.updateMask(growingSeasonMask);
    });
}

function growingSeasonAverageTemperatureForYear(year) {
    return toCelsius(
        growingSeasonDailyForYear(year).select("mean_2m_air_temperature").mean()
    );
}

function growingSeasonAveragePrecipitationForYear(year) {
    return toMillimeters(
        growingSeasonDailyForYear(year).select("total_precipitation").mean()
    );
}

function interannualRainfallVariability(endYear) {
    var startYear = ee
        .Number(endYear)
        .subtract(INTERANNUAL_RAINFALL_WINDOW_YEARS - 1)
        .max(ERA5_START_YEAR)
        .toInt();
    var years = ee.List.sequence(startYear, endYear);
    var annualTotals = ee.ImageCollection.fromImages(
        years.map(function (year) {
            return annualPrecipForYear(year);
        })
    );
    var mean = annualTotals.mean();
    var stdDev = annualTotals.reduce(ee.Reducer.stdDev());

    return stdDev.divide(mean).multiply(100).updateMask(mean.neq(0));
}

function gridmetDroughtMean(year) {
    return annualCollection(GRIDMET_DROUGHT_DATASET, year)
        .select("spi30d")
        .mean();
}

function gridmetDroughtFifthPercentile(year) {
    return annualCollection(GRIDMET_DROUGHT_DATASET, year)
        .select("spi30d")
        .reduce(ee.Reducer.percentile([5]));
}

function fireBurnedMonthCount(year) {
    return annualCollection(VIIRS_BURNED_AREA_DATASET, year)
        .select("Burn_Date")
        .map(function (image) {
            return image.gt(0).unmask(0);
        })
        .sum();
}

function waterPresenceAnnualVariation(year) {
    return annualCollection(JRC_MONTHLY_WATER_DATASET, year)
        .select("water")
        .map(function (image) {
            return image.eq(2).updateMask(image.neq(0));
        })
        .reduce(ee.Reducer.stdDev());
}

function distanceToStreams(year) {
    var streams = ee
        .Image(MERIT_HYDRO_DATASET)
        .select("upa")
        .gte(STREAM_UPSTREAM_AREA_THRESHOLD_KM2)
        .unmask(0);

    return streams
        .fastDistanceTransform()
        .sqrt()
        .multiply(ee.Image.pixelArea().sqrt());
}

function soilOrganicCarbon10cm(year) {
    return ee.Image(ISRIC_SOC_DATASET).select("soc_5-15cm_mean").divide(10);
}

function gldasAnnualSoilMoisture(year) {
    return annualCollection(GLDAS_DATASET, year)
        .select("SoilMoi10_40cm_inst")
        .mean();
}

function srtmLandformType(year) {
    return ee.Image(SRTM_LANDFORMS_DATASET).select("constant");
}

function alosTopographicDiversity(year) {
    return ee.Image(ALOS_TOPO_DIVERSITY_DATASET).select("constant");
}

function modisAnnualEvapotranspiration(year) {
    return scaledAnnualCollection(MODIS_ET_DATASET, year, "ET", 0.1).sum();
}

function positiveSnowDepthMean(dataset, bandName) {
    return function (year) {
        return annualCollection(dataset, year)
            .select(bandName)
            .map(function (image) {
                return image.updateMask(image.gt(0));
            })
            .mean();
    };
}

function makeLayerDefinition(
    name,
    build,
    yearRange,
    exportScale,
    exportOptions
) {
    exportOptions = exportOptions || {};
    return {
        name: name,
        build: function (year, thresholds) {
            return ee.Image(build(year, thresholds)).rename("B0");
        },
        yearRange: yearRange,
        exportScale: exportScale,
        isReferenceLayer: exportOptions.isReferenceLayer === true
    };
}

var LAYER_DEFINITIONS = [
    makeLayerDefinition(
        "Grassland Reference Sites",
        probabilityIntegrityIndex,
        YEAR_PROMPT_GRASSLAND_REFERENCE,
        30,
        { isReferenceLayer: true }
    ),
    makeLayerDefinition(
        "NDVI 95th percentile across the year",
        landsatNdviPercentile(95),
        YEAR_PROMPT_LANDSAT,
        30
    ),
    makeLayerDefinition(
        "NDVI 50th percentile across the year",
        landsatNdviPercentile(50),
        YEAR_PROMPT_LANDSAT,
        30
    ),
    makeLayerDefinition(
        "Length of growing season 1",
        modisGrowingSeasonLength(1),
        YEAR_PROMPT_MODIS_PHENOLOGY,
        500
    ),
    makeLayerDefinition(
        "Length of growing season 2",
        modisGrowingSeasonLength(2),
        YEAR_PROMPT_MODIS_PHENOLOGY,
        500
    ),
    makeLayerDefinition(
        "Timing of green up 1",
        modisGreenupTiming(1),
        YEAR_PROMPT_MODIS_PHENOLOGY,
        500
    ),
    makeLayerDefinition(
        "Timing of green up 2",
        modisGreenupTiming(2),
        YEAR_PROMPT_MODIS_PHENOLOGY,
        500
    ),
    makeLayerDefinition(
        "Short vegetation height",
        shortVegetationHeight,
        YEAR_PROMPT_SHORT_VEG_HEIGHT,
        30
    ),
    makeLayerDefinition(
        "Percent tree cover",
        modisVegetationCover("Percent_Tree_Cover"),
        YEAR_PROMPT_MODIS_COVER,
        250
    ),
    makeLayerDefinition(
        "Percent veg, but not tree cover",
        modisVegetationCover("Percent_NonTree_Vegetation"),
        YEAR_PROMPT_MODIS_COVER,
        250
    ),
    makeLayerDefinition(
        "Percent bare",
        modisVegetationCover("Percent_NonVegetated"),
        YEAR_PROMPT_MODIS_COVER,
        250
    ),
    makeLayerDefinition(
        "Leaf Area Index (LAI) annual max",
        annualScaledSummary(
            MODIS_LAI_FPAR_DATASET,
            "Lai_500m",
            0.1,
            ee.Reducer.max()
        ),
        YEAR_PROMPT_MODIS_LAI_FPAR,
        500
    ),
    makeLayerDefinition(
        "Leaf Area Index (LAI) annual SD",
        annualScaledSummary(
            MODIS_LAI_FPAR_DATASET,
            "Lai_500m",
            0.1,
            ee.Reducer.stdDev()
        ),
        YEAR_PROMPT_MODIS_LAI_FPAR,
        500
    ),
    makeLayerDefinition(
        "Fraction of Photosynthetically Active Radiation (FPAR) annual mean",
        annualScaledSummary(
            MODIS_LAI_FPAR_DATASET,
            "Fpar_500m",
            0.01,
            ee.Reducer.mean()
        ),
        YEAR_PROMPT_MODIS_LAI_FPAR,
        500
    ),
    makeLayerDefinition(
        "Fraction of Photosynthetically Active Radiation (FPAR) annual SD",
        annualScaledSummary(
            MODIS_LAI_FPAR_DATASET,
            "Fpar_500m",
            0.01,
            ee.Reducer.stdDev()
        ),
        YEAR_PROMPT_MODIS_LAI_FPAR,
        500
    ),
    makeLayerDefinition(
        "FPAR Variability max",
        annualScaledSummary(
            MODIS_LAI_FPAR_DATASET,
            "FparStdDev_500m",
            0.01,
            ee.Reducer.max()
        ),
        YEAR_PROMPT_MODIS_LAI_FPAR,
        500
    ),
    makeLayerDefinition(
        "Number of growing seasons",
        modisPhenologyBand("NumCycles"),
        YEAR_PROMPT_MODIS_PHENOLOGY,
        500
    ),
    makeLayerDefinition(
        "NPP",
        annualScaledFirst(MODIS_PRODUCTIVITY_DATASET, "Npp", 0.0001),
        YEAR_PROMPT_MODIS_PRODUCTIVITY,
        500
    ),
    makeLayerDefinition(
        "GPP",
        annualScaledFirst(MODIS_PRODUCTIVITY_DATASET, "Gpp", 0.0001),
        YEAR_PROMPT_MODIS_PRODUCTIVITY,
        500
    ),
    makeLayerDefinition(
        "Maximum annual temperature (C)",
        annualMaxTemperatureForYear,
        YEAR_PROMPT_ERA5_COMPLETE_YEARS,
        27830
    ),
    makeLayerDefinition(
        "Mean annual temperature (C)",
        annualMeanTemperatureForYear,
        YEAR_PROMPT_ERA5_COMPLETE_YEARS,
        27830
    ),
    makeLayerDefinition(
        "Median annual temperature (C)",
        annualMedianTemperatureForYear,
        YEAR_PROMPT_ERA5_COMPLETE_YEARS,
        27830
    ),
    makeLayerDefinition(
        "Minimum annual temperature (C)",
        annualMinTemperatureForYear,
        YEAR_PROMPT_ERA5_COMPLETE_YEARS,
        27830
    ),
    makeLayerDefinition(
        "Annual precipitation (mm)",
        annualPrecipForYear,
        YEAR_PROMPT_ERA5_COMPLETE_YEARS,
        27830
    ),
    makeLayerDefinition(
        "Growing season avg temp (C)",
        growingSeasonAverageTemperatureForYear,
        YEAR_PROMPT_GROWING_SEASON_CLIMATE,
        27830
    ),
    makeLayerDefinition(
        "Growing season avg precipitation (mm/day)",
        growingSeasonAveragePrecipitationForYear,
        YEAR_PROMPT_GROWING_SEASON_CLIMATE,
        27830
    ),
    makeLayerDefinition(
        "Interannual rainfall variability (CV%, 10-year)",
        interannualRainfallVariability,
        YEAR_PROMPT_ERA5_COMPLETE_YEARS,
        27830
    ),
    makeLayerDefinition(
        "Drought mean (SPI 30-day)",
        gridmetDroughtMean,
        YEAR_PROMPT_GRIDMET_DROUGHT,
        4638
    ),
    makeLayerDefinition(
        "Drought 5th percentile (SPI 30-day)",
        gridmetDroughtFifthPercentile,
        YEAR_PROMPT_GRIDMET_DROUGHT,
        4638
    ),
    makeLayerDefinition(
        "Fire frequency (burned months in selected year)",
        fireBurnedMonthCount,
        YEAR_PROMPT_VIIRS_FIRE,
        500
    ),
    makeLayerDefinition(
        "Annual variation in water presence",
        waterPresenceAnnualVariation,
        YEAR_PROMPT_JRC_WATER,
        30
    ),
    makeLayerDefinition(
        "Distance to streams (m)",
        distanceToStreams,
        YEAR_PROMPT_STATIC,
        90
    ),
    makeLayerDefinition(
        "Soil organic carbon (10 cm, g/kg)",
        soilOrganicCarbon10cm,
        YEAR_PROMPT_STATIC,
        250
    ),
    makeLayerDefinition(
        "Soil moisture annual mean (GLDAS 10-40 cm)",
        gldasAnnualSoilMoisture,
        YEAR_PROMPT_GLDAS,
        27830
    ),
    makeLayerDefinition(
        "Landform type (SRTM)",
        srtmLandformType,
        YEAR_PROMPT_STATIC,
        90
    ),
    makeLayerDefinition(
        "Topographic diversity (ALOS)",
        alosTopographicDiversity,
        YEAR_PROMPT_STATIC,
        90
    ),
    makeLayerDefinition(
        "Annual evapotranspiration (MODIS ET, mm)",
        modisAnnualEvapotranspiration,
        YEAR_PROMPT_MODIS_ET,
        500
    ),
    makeLayerDefinition(
        "Average snow depth when present (GLDAS, m)",
        positiveSnowDepthMean(GLDAS_DATASET, "SnowDepth_inst"),
        YEAR_PROMPT_GLDAS,
        27830
    ),
    makeLayerDefinition(
        "Average snow depth when present (SMAP, m)",
        positiveSnowDepthMean(SMAP_DATASET, "snow_depth"),
        YEAR_PROMPT_SMAP,
        11000
    )
];

function regionNames() {
    return REGION_DEFINITIONS.map(function (region) {
        return region.name;
    });
}

function regionDefinitionByName(name) {
    for (var i = 0; i < REGION_DEFINITIONS.length; i++) {
        if (REGION_DEFINITIONS[i].name === name) {
            return REGION_DEFINITIONS[i];
        }
    }
    return REGION_DEFINITIONS[0];
}

function regionGeometry(regionDefinition) {
    return ee.FeatureCollection(regionDefinition.assetId).geometry();
}

function exportRegion(region) {
    return region.bounds(1, ee.Projection(EXPORT_CRS));
}

function exportImage(image, region) {
    var exportGeometry = region.transform(ee.Projection(EXPORT_CRS), 1);
    return image
        .reproject({
            crs: EXPORT_CRS,
            scale: EXPORT_SCALE_METERS
        })
        .clip(exportGeometry)
        .toFloat();
}

function regionOutline(regionDefinition) {
    return ee.FeatureCollection(regionDefinition.assetId).style({
        color: "ffff00",
        fillColor: "00000000",
        width: 2
    });
}

function referenceSitesLayer(region, thresholds) {
    return probabilityIntegrityIndex(null, thresholds)
        .eq(1)
        .selfMask()
        .clip(region);
}

function slug(text) {
    return text
        .toLowerCase()
        .replace(/[^a-z0-9]+/g, "_")
        .replace(/^_+|_+$/g, "");
}

function thresholdNameParts(thresholds) {
    return [
        slug("grassland_prob_" + thresholds.grasslandProbability),
        slug("hmi_" + thresholds.hmi),
        slug("hii_" + thresholds.hii)
    ];
}

function exportName(layerDefinition, year, regionDefinition, thresholds) {
    var parts = [
        slug(layerDefinition.name),
        slug("year_" + year),
        slug(regionDefinition.name)
    ];
    if (layerDefinition.isReferenceLayer) {
        parts = parts.concat(thresholdNameParts(thresholds));
    }
    return parts.join("_");
}

var controlPanel = ui.Panel({
    layout: ui.Panel.Layout.flow("vertical"),
    style: { width: "520px", padding: "12px" }
});

var title = ui.Label({
    value: "Response Variable Raster Exports",
    style: { fontWeight: "bold", fontSize: "18px", margin: "0 0 8px 0" }
});

var grasslandProbabilityInput = ui.Textbox({
    value: String(DEFAULT_GRASSLAND_PROB_THRESHOLD),
    style: { width: "120px" },
    onChange: function () {
        updateMapLayers();
    }
});

var hmiThresholdInput = ui.Textbox({
    value: String(DEFAULT_HMI_THRESHOLD),
    style: { width: "120px" },
    onChange: function () {
        updateMapLayers();
    }
});

var hiiThresholdInput = ui.Textbox({
    value: String(DEFAULT_HII_THRESHOLD),
    style: { width: "120px" },
    onChange: function () {
        updateMapLayers();
    }
});

var yearInput = ui.Textbox({
    value: DEFAULTYEAR,
    style: { width: "120px" }
});

var driveFolderInput = ui.Textbox({
    value: DEFAULT_DRIVE_FOLDER,
    style: { width: "260px" }
});

var exportGridLabel = ui.Label({
    value: String(EXPORT_SCALE_METERS) + " m, " + EXPORT_CRS,
    style: { margin: "4px 0" }
});

var regionSelect = ui.Select({
    items: regionNames(),
    value: REGION_DEFINITIONS[0].name,
    onChange: function () {
        updateMapLayers();
    },
    style: { width: "260px" }
});

var statusLabel = ui.Label({
    value: "Exports will use the selected export year and be sent to your authorized Google Drive.",
    style: { margin: "8px 0 0 0" }
});

function fieldRow(label, widget) {
    return ui.Panel({
        widgets: [
            ui.Label({
                value: label,
                style: { width: "150px", margin: "4px 8px 4px 0" }
            }),
            widget
        ],
        layout: ui.Panel.Layout.flow("horizontal")
    });
}

function referenceThresholds() {
    return {
        grasslandProbability: parseFloat(grasslandProbabilityInput.getValue()),
        hmi: parseFloat(hmiThresholdInput.getValue()),
        hii: parseFloat(hiiThresholdInput.getValue())
    };
}

var layerRows = [];
var layerList = ui.Panel({
    layout: ui.Panel.Layout.flow("vertical"),
    style: { maxHeight: "480px", stretch: "horizontal" }
});

LAYER_DEFINITIONS.forEach(function (layerDefinition) {
    var checkbox = ui.Checkbox({
        label: layerDefinition.name,
        value: false,
        style: { width: "300px", margin: "0 8px 0 0" }
    });
    layerRows.push({
        layerDefinition: layerDefinition,
        checkbox: checkbox
    });
    layerList.add(
        ui.Panel({
            widgets: [
                checkbox,
                ui.Label({
                    value: layerDefinition.yearRange,
                    style: { width: "120px", margin: "0 8px 0 0" }
                }),
                ui.Label({
                    value: String(EXPORT_SCALE_METERS) + " m",
                    style: { width: "80px", margin: "0" }
                })
            ],
            layout: ui.Panel.Layout.flow("horizontal")
        })
    );
});

function setAllLayerCheckboxes(value) {
    layerRows.forEach(function (row) {
        row.checkbox.setValue(value, false);
    });
}

function selectedLayerDefinitions() {
    var selected = [];
    layerRows.forEach(function (row) {
        if (row.checkbox.getValue()) {
            selected.push(row.layerDefinition);
        }
    });
    return selected;
}

function queueDriveExports() {
    var year = parseInt(yearInput.getValue(), 10);
    var regionDefinition = regionDefinitionByName(regionSelect.getValue());
    var region = regionGeometry(regionDefinition);
    var regionBounds = exportRegion(region);
    var thresholds = referenceThresholds();
    var selectedLayers = selectedLayerDefinitions();

    selectedLayers.forEach(function (layerDefinition) {
        var name = exportName(
            layerDefinition,
            year,
            regionDefinition,
            thresholds
        );
        Export.image.toDrive({
            image: exportImage(layerDefinition.build(year, thresholds), region),
            description: name,
            folder: driveFolderInput.getValue(),
            fileNamePrefix: name,
            region: regionBounds,
            crs: EXPORT_CRS,
            scale: EXPORT_SCALE_METERS,
            maxPixels: DEFAULT_MAX_PIXELS
        });
    });

    statusLabel.setValue(
        selectedLayers.length +
            " Drive export task(s) queued for export year " +
            year +
            " in the Tasks tab."
    );
}

var map = ui.Map();
map.setOptions("HYBRID");
map.style().set("stretch", "both");
map.setControlVisibility({ mapTypeControl: true });

function updateMapLayers() {
    var regionDefinition = regionDefinitionByName(regionSelect.getValue());
    var region = regionGeometry(regionDefinition);
    var thresholds = referenceThresholds();
    map.layers().reset([
        ui.Map.Layer(
            referenceSitesLayer(region, thresholds),
            { palette: ["ff2db2"], min: 1, max: 1 },
            "Grassland Reference Sites"
        ),
        ui.Map.Layer(regionOutline(regionDefinition), {}, regionDefinition.name)
    ]);
    map.centerObject(region, 7);
}

controlPanel.add(title);
controlPanel.add(
    ui.Label({
        value: "Reference site thresholds",
        style: { fontWeight: "bold", margin: "8px 0 4px 0" }
    })
);
controlPanel.add(fieldRow("Grassland prob", grasslandProbabilityInput));
controlPanel.add(fieldRow("HMI", hmiThresholdInput));
controlPanel.add(fieldRow("HII", hiiThresholdInput));
controlPanel.add(fieldRow("Export year", yearInput));
controlPanel.add(fieldRow("Region", regionSelect));
controlPanel.add(fieldRow("Drive folder", driveFolderInput));
controlPanel.add(fieldRow("Export grid", exportGridLabel));
controlPanel.add(
    ui.Panel({
        widgets: [
            ui.Button("Select all", function () {
                setAllLayerCheckboxes(true);
            }),
            ui.Button("Clear all", function () {
                setAllLayerCheckboxes(false);
            }),
            ui.Button("Queue Drive exports", queueDriveExports)
        ],
        layout: ui.Panel.Layout.flow("horizontal"),
        style: { margin: "8px 0" }
    })
);
controlPanel.add(
    ui.Panel({
        widgets: [
            ui.Label({
                value: "Dataset",
                style: {
                    width: "300px",
                    fontWeight: "bold",
                    margin: "0 8px 4px 0"
                }
            }),
            ui.Label({
                value: "Years",
                style: {
                    width: "120px",
                    fontWeight: "bold",
                    margin: "0 8px 4px 0"
                }
            }),
            ui.Label({
                value: "Export",
                style: {
                    width: "80px",
                    fontWeight: "bold",
                    margin: "0 0 4px 0"
                }
            })
        ],
        layout: ui.Panel.Layout.flow("horizontal")
    })
);
controlPanel.add(layerList);
controlPanel.add(statusLabel);

ui.root.widgets().reset([
    ui.SplitPanel({
        firstPanel: controlPanel,
        secondPanel: map,
        orientation: "horizontal",
        style: { stretch: "both" }
    })
]);
updateMapLayers();
