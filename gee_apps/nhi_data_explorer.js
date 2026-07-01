var DEFAULTYEAR = "2005";
var SAMPLE_SCALE_METERS = 30;
var CLEAR_LABEL = "(*clear*)";

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
var GRASSLAND_PROB_THRESHOLD = 60;
var HMI_THRESHOLD = 0.1;
var HII_THRESHOLD = 0.08;
var DEFAULT_YEAR_PROMPT = "Select a layer to see year range";

function yearRangePrompt(startYear, endYear) {
    return "Select a year between " + startYear + "-" + endYear;
}

var YEAR_PROMPT_GRASSLAND_REFERENCE =
    "Reference period 2001-2020; year is ignored";
var YEAR_PROMPT_STATIC = "Static layer; year is ignored";
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

var PROBABILITY_INTEGRITY_INDEX = noTwoConsecutiveZerosFromAnnualBinary(
    function (year) {
        return GRASSLAND_PROB_IC.filterDate(
            ee.Date.fromYMD(year, 1, 1),
            ee.Date.fromYMD(year.add(1), 1, 1)
        )
            .first()
            .select(0)
            .gte(GRASSLAND_PROB_THRESHOLD);
    }
)
    .and(
        noTwoConsecutiveZerosFromAnnualBinary(function (year) {
            return HII_IC.filterDate(
                ee.Date.fromYMD(year, 1, 1),
                ee.Date.fromYMD(year.add(1), 1, 1)
            )
                .mean()
                .divide(7000)
                .lt(HII_THRESHOLD);
        })
    )
    .and(HMI_IMG.lte(HMI_THRESHOLD))
    .selfMask()
    .toByte();

function probabilityIntegrityIndex() {
    return PROBABILITY_INTEGRITY_INDEX;
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
    return ee.Image(ISRIC_SOC_DATASET).select("b10").divide(10);
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

function makeLayerDefinition(name, build, defaultRange, yearPrompt) {
    return {
        name: name,
        build: function (year) {
            return ee.Image(build(year)).rename("B0");
        },
        defaultRange: defaultRange,
        yearPrompt: yearPrompt
    };
}

var LAYER_DEFINITIONS = [
    makeLayerDefinition(
        "Grassland Reference Sites",
        probabilityIntegrityIndex,
        { min: 0, max: 1 },
        YEAR_PROMPT_GRASSLAND_REFERENCE
    ),
    makeLayerDefinition(
        "NDVI 95th percentile across the year",
        landsatNdviPercentile(95),
        { min: 0, max: 1 },
        YEAR_PROMPT_LANDSAT
    ),
    makeLayerDefinition(
        "NDVI 50th percentile across the year",
        landsatNdviPercentile(50),
        { min: 0, max: 1 },
        YEAR_PROMPT_LANDSAT
    ),
    makeLayerDefinition(
        "Length of growing season 1",
        modisGrowingSeasonLength(1),
        { min: 0, max: 250 },
        YEAR_PROMPT_MODIS_PHENOLOGY
    ),
    makeLayerDefinition(
        "Length of growing season 2",
        modisGrowingSeasonLength(2),
        { min: 0, max: 250 },
        YEAR_PROMPT_MODIS_PHENOLOGY
    ),
    makeLayerDefinition(
        "Timing of green up 1",
        modisGreenupTiming(1),
        { min: 1, max: 365 },
        YEAR_PROMPT_MODIS_PHENOLOGY
    ),
    makeLayerDefinition(
        "Timing of green up 2",
        modisGreenupTiming(2),
        { min: 1, max: 365 },
        YEAR_PROMPT_MODIS_PHENOLOGY
    ),
    makeLayerDefinition(
        "Short vegetation height",
        shortVegetationHeight,
        { min: 0, max: 3 },
        YEAR_PROMPT_SHORT_VEG_HEIGHT
    ),
    makeLayerDefinition(
        "Percent tree cover",
        modisVegetationCover("Percent_Tree_Cover"),
        { min: 0, max: 100 },
        YEAR_PROMPT_MODIS_COVER
    ),
    makeLayerDefinition(
        "Percent veg, but not tree cover",
        modisVegetationCover("Percent_NonTree_Vegetation"),
        { min: 0, max: 100 },
        YEAR_PROMPT_MODIS_COVER
    ),
    makeLayerDefinition(
        "Percent bare",
        modisVegetationCover("Percent_NonVegetated"),
        { min: 0, max: 100 },
        YEAR_PROMPT_MODIS_COVER
    ),
    makeLayerDefinition(
        "Leaf Area Index (LAI) annual max",
        annualScaledSummary(
            MODIS_LAI_FPAR_DATASET,
            "Lai_500m",
            0.1,
            ee.Reducer.max()
        ),
        { min: 0, max: 8 },
        YEAR_PROMPT_MODIS_LAI_FPAR
    ),
    makeLayerDefinition(
        "Leaf Area Index (LAI) annual SD",
        annualScaledSummary(
            MODIS_LAI_FPAR_DATASET,
            "Lai_500m",
            0.1,
            ee.Reducer.stdDev()
        ),
        { min: 0, max: 2 },
        YEAR_PROMPT_MODIS_LAI_FPAR
    ),
    makeLayerDefinition(
        "Fraction of Photosynthetically Active Radiation (FPAR) annual mean",
        annualScaledSummary(
            MODIS_LAI_FPAR_DATASET,
            "Fpar_500m",
            0.01,
            ee.Reducer.mean()
        ),
        { min: 0, max: 1 },
        YEAR_PROMPT_MODIS_LAI_FPAR
    ),
    makeLayerDefinition(
        "Fraction of Photosynthetically Active Radiation (FPAR) annual SD",
        annualScaledSummary(
            MODIS_LAI_FPAR_DATASET,
            "Fpar_500m",
            0.01,
            ee.Reducer.stdDev()
        ),
        { min: 0, max: 0.4 },
        YEAR_PROMPT_MODIS_LAI_FPAR
    ),
    makeLayerDefinition(
        "FPAR Variability max",
        annualScaledSummary(
            MODIS_LAI_FPAR_DATASET,
            "FparStdDev_500m",
            0.01,
            ee.Reducer.max()
        ),
        { min: 0, max: 0.4 },
        YEAR_PROMPT_MODIS_LAI_FPAR
    ),
    makeLayerDefinition(
        "Number of growing seasons",
        modisPhenologyBand("NumCycles"),
        { min: 0, max: 7 },
        YEAR_PROMPT_MODIS_PHENOLOGY
    ),
    makeLayerDefinition(
        "NPP",
        annualScaledFirst(MODIS_PRODUCTIVITY_DATASET, "Npp", 0.0001),
        { min: 0, max: 2 },
        YEAR_PROMPT_MODIS_PRODUCTIVITY
    ),
    makeLayerDefinition(
        "GPP",
        annualScaledFirst(MODIS_PRODUCTIVITY_DATASET, "Gpp", 0.0001),
        { min: 0, max: 4 },
        YEAR_PROMPT_MODIS_PRODUCTIVITY
    ),
    makeLayerDefinition(
        "Maximum annual temperature (C)",
        annualMaxTemperatureForYear,
        { min: 0, max: 45 },
        YEAR_PROMPT_ERA5_COMPLETE_YEARS
    ),
    makeLayerDefinition(
        "Mean annual temperature (C)",
        annualMeanTemperatureForYear,
        { min: -20, max: 30 },
        YEAR_PROMPT_ERA5_COMPLETE_YEARS
    ),
    makeLayerDefinition(
        "Median annual temperature (C)",
        annualMedianTemperatureForYear,
        { min: -20, max: 30 },
        YEAR_PROMPT_ERA5_COMPLETE_YEARS
    ),
    makeLayerDefinition(
        "Minimum annual temperature (C)",
        annualMinTemperatureForYear,
        { min: -40, max: 20 },
        YEAR_PROMPT_ERA5_COMPLETE_YEARS
    ),
    makeLayerDefinition(
        "Annual precipitation (mm)",
        annualPrecipForYear,
        { min: 0, max: 3000 },
        YEAR_PROMPT_ERA5_COMPLETE_YEARS
    ),
    makeLayerDefinition(
        "Growing season avg temp (C)",
        growingSeasonAverageTemperatureForYear,
        { min: 0, max: 30 },
        YEAR_PROMPT_GROWING_SEASON_CLIMATE
    ),
    makeLayerDefinition(
        "Growing season avg precipitation (mm/day)",
        growingSeasonAveragePrecipitationForYear,
        { min: 0, max: 10 },
        YEAR_PROMPT_GROWING_SEASON_CLIMATE
    ),
    makeLayerDefinition(
        "Interannual rainfall variability (CV%, 10-year)",
        interannualRainfallVariability,
        { min: 0, max: 50 },
        YEAR_PROMPT_ERA5_COMPLETE_YEARS
    ),
    makeLayerDefinition(
        "Drought mean (SPI 30-day)",
        gridmetDroughtMean,
        { min: -2, max: 2 },
        YEAR_PROMPT_GRIDMET_DROUGHT
    ),
    makeLayerDefinition(
        "Drought 5th percentile (SPI 30-day)",
        gridmetDroughtFifthPercentile,
        { min: -3, max: 1 },
        YEAR_PROMPT_GRIDMET_DROUGHT
    ),
    makeLayerDefinition(
        "Fire frequency (burned months in selected year)",
        fireBurnedMonthCount,
        { min: 0, max: 12 },
        YEAR_PROMPT_VIIRS_FIRE
    ),
    makeLayerDefinition(
        "Annual variation in water presence",
        waterPresenceAnnualVariation,
        { min: 0, max: 0.5 },
        YEAR_PROMPT_JRC_WATER
    ),
    makeLayerDefinition(
        "Distance to streams (m)",
        distanceToStreams,
        { min: 1, max: 5000 },
        YEAR_PROMPT_STATIC
    ),
    makeLayerDefinition(
        "Soil organic carbon (10 cm, g/kg)",
        soilOrganicCarbon10cm,
        { min: 0, max: 25 },
        YEAR_PROMPT_STATIC
    ),
    makeLayerDefinition(
        "Soil moisture annual mean (GLDAS 10-40 cm)",
        gldasAnnualSoilMoisture,
        { min: 0, max: 150 },
        YEAR_PROMPT_GLDAS
    ),
    makeLayerDefinition(
        "Landform type (SRTM)",
        srtmLandformType,
        { min: 11, max: 42 },
        YEAR_PROMPT_STATIC
    ),
    makeLayerDefinition(
        "Topographic diversity (ALOS)",
        alosTopographicDiversity,
        { min: 0, max: 1 },
        YEAR_PROMPT_STATIC
    ),
    makeLayerDefinition(
        "Annual evapotranspiration (MODIS ET, mm)",
        modisAnnualEvapotranspiration,
        { min: 0, max: 1500 },
        YEAR_PROMPT_MODIS_ET
    ),
    makeLayerDefinition(
        "Average snow depth when present (GLDAS, m)",
        positiveSnowDepthMean(GLDAS_DATASET, "SnowDepth_inst"),
        { min: 0, max: 2 },
        YEAR_PROMPT_GLDAS
    ),
    makeLayerDefinition(
        "Average snow depth when present (SMAP, m)",
        positiveSnowDepthMean(SMAP_DATASET, "snow_depth"),
        { min: 0, max: 2 },
        YEAR_PROMPT_SMAP
    )
];

var legend_styles = {
    black_to_red: ["000000", "005aff", "43c8c8", "fff700", "ff0000"],
    blue_to_green: ["440154", "414287", "218e8d", "5ac864", "fde725"],
    cividis: ["00204d", "414d6b", "7c7b78", "b9ac70", "ffea46"],
    viridis: ["440154", "355e8d", "20928c", "70cf57", "fde725"],
    blues: ["f7fbff", "c6dbef", "6baed6", "2171b5", "08306b"],
    reds: ["fff5f0", "fcbba1", "fb6a4a", "cb181d", "67000d"],
    turbo: ["321543", "2eb4f2", "affa37", "f66c19", "7a0403"]
};
var default_legend_style = "blue_to_green";

function changeColorScheme(key, active_context) {
    active_context.visParams.palette = legend_styles[key];
    active_context.build_legend_panel();
    active_context.updateVisParams();
}

function detectRange(image, geometry, fallbackRange, callback) {
    var dictionary = image.reduceRegion({
        reducer: ee.Reducer.percentile([10, 90], ["p10", "p90"]),
        geometry: geometry,
        scale: SAMPLE_SCALE_METERS * 100,
        bestEffort: true,
        maxPixels: 1e8,
        tileScale: 4
    });

    ee.data.computeValue(dictionary, function (val) {
        var min = val && val.B0_p10;
        var max = val && val.B0_p90;

        if (
            min === null ||
            min === undefined ||
            max === null ||
            max === undefined
        ) {
            callback(fallbackRange);
            return;
        }

        if (min === max) {
            callback(fallbackRange);
            return;
        }

        callback({ min: min, max: max });
    });
}

function makeLegendPanel(active_context) {
    function makeRow(color, name) {
        var colorBox = ui.Label({
            style: {
                backgroundColor: "#" + color,
                padding: "4px 25px 4px 25px",
                margin: "0 0 0px 0",
                position: "bottom-center"
            }
        });

        var description = ui.Label({
            value: name,
            style: {
                margin: "0 0 0px 0px",
                position: "top-center",
                fontSize: "10px",
                padding: 0,
                border: 0,
                textAlign: "center",
                backgroundColor: "rgba(0, 0, 0, 0)"
            }
        });

        return ui.Panel({
            widgets: [colorBox, description],
            layout: ui.Panel.Layout.Flow("vertical"),
            style: { backgroundColor: "rgba(0, 0, 0, 0)" }
        });
    }

    var names = ["Low", "", "", "", "High"];

    if (active_context.legend_panel === null) {
        active_context.legend_panel = ui.Panel({
            layout: ui.Panel.Layout.Flow("horizontal"),
            style: {
                position: "top-center",
                padding: "0px",
                backgroundColor: "rgba(255, 255, 255, 0.4)"
            }
        });

        active_context.legend_select = ui.Select({
            items: Object.keys(legend_styles),
            value: default_legend_style,
            onChange: function (key) {
                changeColorScheme(key, active_context);
            }
        });

        active_context.map.add(active_context.legend_panel);
    } else {
        active_context.legend_panel.clear();
    }

    active_context.legend_panel.add(active_context.legend_select);
    for (var i = 0; i < 5; i++) {
        active_context.legend_panel.add(
            makeRow(active_context.visParams.palette[i], names[i])
        );
    }
}

function buildLayerNames() {
    return [CLEAR_LABEL].concat(
        LAYER_DEFINITIONS.map(function (def) {
            return def.name;
        })
    );
}

function layerDefinitionByName(name) {
    for (var i = 0; i < LAYER_DEFINITIONS.length; i++) {
        if (LAYER_DEFINITIONS[i].name === name) {
            return LAYER_DEFINITIONS[i];
        }
    }
    return null;
}

function getCachedLayer(active_context, layerDefinition, year) {
    var key = year + "|" + layerDefinition.name;
    if (!active_context.layerCache[key]) {
        active_context.layerCache[key] = layerDefinition.build(year);
    }
    return active_context.layerCache[key];
}

function formatPointValue(value) {
    return typeof value === "number"
        ? String(Math.round(value * 100) / 100)
        : String(value);
}

var leftMap = ui.root.widgets().get(0);
var rightMap = ui.Map();
var linker = ui.Map.Linker([leftMap, rightMap]);
var splitPanel = ui.SplitPanel({
    firstPanel: linker.get(0),
    secondPanel: linker.get(1),
    orientation: "horizontal",
    wipe: true,
    style: { stretch: "both" }
});
ui.root.widgets().reset([splitPanel]);

var panel_list = [];
[
    [leftMap, "left"],
    [rightMap, "right"]
].forEach(function (mapside) {
    var active_context = {
        layerCache: {},
        currentYear: parseInt(DEFAULTYEAR, 10),
        last_layer: null,
        raster: null,
        datasetName: null,
        activeLayerDefinition: null,
        point_val: null,
        last_point_layer: null,
        map: mapside[0],
        legend_panel: null,
        legend_select: null,
        renderId: 0,
        visParams: {
            min: 0,
            max: 100,
            palette: legend_styles[default_legend_style]
        }
    };

    function updateVisParams() {
        if (active_context.last_layer !== null) {
            active_context.last_layer.setVisParams(active_context.visParams);
        }
    }

    active_context.updateVisParams = updateVisParams;
    active_context.build_legend_panel = function () {
        makeLegendPanel(active_context);
    };

    function clearLayer() {
        if (active_context.last_layer !== null) {
            active_context.map.remove(active_context.last_layer);
            active_context.last_layer = null;
        }
        if (active_context.last_point_layer !== null) {
            active_context.map.remove(active_context.last_point_layer);
            active_context.last_point_layer = null;
        }
        active_context.raster = null;
        active_context.datasetName = null;
        active_context.activeLayerDefinition = null;
        year_label.setValue(DEFAULT_YEAR_PROMPT);
        min_val.setValue("n/a", false);
        max_val.setValue("n/a", false);
        min_val.setDisabled(true);
        max_val.setDisabled(true);
        active_context.point_val.setValue("nothing clicked");
    }

    function loadLayer(layerName, done) {
        done = done || function () {};

        if (layerName === CLEAR_LABEL) {
            clearLayer();
            done();
            return;
        }

        var layerDefinition = layerDefinitionByName(layerName);
        year_label.setValue(layerDefinition.yearPrompt);
        var image = getCachedLayer(
            active_context,
            layerDefinition,
            active_context.currentYear
        );
        var renderId = ++active_context.renderId;

        if (active_context.last_layer !== null) {
            active_context.map.remove(active_context.last_layer);
            active_context.last_layer = null;
        }

        active_context.raster = image;
        active_context.datasetName = layerDefinition.name;
        active_context.activeLayerDefinition = layerDefinition;
        active_context.visParams.palette =
            legend_styles[
                active_context.legend_select.getValue() || default_legend_style
            ];
        active_context.build_legend_panel();

        detectRange(
            image,
            active_context.map.getBounds(true),
            layerDefinition.defaultRange,
            function (range) {
                if (renderId !== active_context.renderId) {
                    done();
                    return;
                }

                active_context.visParams = {
                    min: range.min,
                    max: range.max,
                    palette: active_context.visParams.palette
                };
                active_context.last_layer = active_context.map.addLayer(
                    image,
                    active_context.visParams,
                    layerDefinition.name
                );
                min_val.setValue(String(active_context.visParams.min), false);
                max_val.setValue(String(active_context.visParams.max), false);
                min_val.setDisabled(false);
                max_val.setDisabled(false);
                done();
            }
        );
    }

    active_context.map.style().set("cursor", "crosshair");

    var panel = ui.Panel({
        layout: ui.Panel.Layout.flow("vertical"),
        style: {
            position: "middle-" + mapside[1],
            backgroundColor: "rgba(255, 255, 255, 0.4)"
        }
    });

    var controls_label = ui.Label({
        value: mapside[1] + " controls",
        style: { backgroundColor: "rgba(0, 0, 0, 0)" }
    });

    var select = ui.Select({
        items: buildLayerNames(),
        placeholder: "Choose a dataset...",
        onChange: function (layerName, self) {
            self.setDisabled(true);
            loadLayer(layerName, function () {
                self.setDisabled(false);
            });
        }
    });

    var active_year = ui.Textbox({
        value: DEFAULTYEAR,
        style: { width: "200px" },
        onChange: function (value) {
            active_context.currentYear = parseInt(value, 10);
            var selected = select.getValue();
            if (selected && selected !== CLEAR_LABEL) {
                loadLayer(selected);
            }
        }
    });

    var year_label = ui.Label({
        value: DEFAULT_YEAR_PROMPT,
        style: { backgroundColor: "rgba(0, 0, 0, 0)" }
    });

    var min_val = ui.Textbox({
        value: "n/a",
        onChange: function (value) {
            active_context.visParams.min = +value;
            updateVisParams();
        }
    });
    min_val.setDisabled(true);

    var max_val = ui.Textbox({
        value: "n/a",
        onChange: function (value) {
            active_context.visParams.max = +value;
            updateVisParams();
        }
    });
    max_val.setDisabled(true);

    active_context.point_val = ui.Textbox({ value: "nothing clicked" });

    var range_button = ui.Button("Detect Range", function (self) {
        if (
            active_context.raster === null ||
            active_context.activeLayerDefinition === null
        ) {
            return;
        }

        self.setDisabled(true);
        var label = self.getLabel();
        self.setLabel("Detecting...");

        detectRange(
            active_context.raster,
            active_context.map.getBounds(true),
            active_context.activeLayerDefinition.defaultRange,
            function (range) {
                min_val.setValue(String(range.min), false);
                max_val.setValue(String(range.max), true);
                self.setLabel(label);
                self.setDisabled(false);
            }
        );
    });

    panel.add(year_label);
    panel.add(active_year);
    panel.add(controls_label);
    panel.add(select);
    panel.add(
        ui.Label({
            value: "min",
            style: { backgroundColor: "rgba(0, 0, 0, 0)" }
        })
    );
    panel.add(min_val);
    panel.add(
        ui.Label({
            value: "max",
            style: { backgroundColor: "rgba(0, 0, 0, 0)" }
        })
    );
    panel.add(max_val);
    panel.add(range_button);
    panel.add(
        ui.Label({
            value: "picked point",
            style: { backgroundColor: "rgba(0, 0, 0, 0)" }
        })
    );
    panel.add(active_context.point_val);

    panel_list.push([panel, min_val, max_val, active_context]);
    active_context.map.add(panel);
    active_context.map.setControlVisibility(false);
    active_context.map.setControlVisibility({ mapTypeControl: true });
    active_context.build_legend_panel();
});

var clone_to_right = ui.Button("Use this range in both windows", function () {
    panel_list[1][1].setValue(panel_list[0][1].getValue(), false);
    panel_list[1][2].setValue(panel_list[0][2].getValue(), true);
});

var clone_to_left = ui.Button("Use this range in both windows", function () {
    panel_list[0][1].setValue(panel_list[1][1].getValue(), false);
    panel_list[0][2].setValue(panel_list[1][2].getValue(), true);
});

panel_list.forEach(function (panel_array) {
    var map = panel_array[3].map;
    map.onClick(function (obj) {
        var point = ee.Geometry.Point([obj.lon, obj.lat]);

        [panel_list[0][3], panel_list[1][3]].forEach(function (active_context) {
            if (active_context.raster === null) {
                return;
            }

            active_context.point_val.setValue("sampling...");
            var point_sample = active_context.raster.sampleRegions({
                collection: ee.FeatureCollection(point),
                geometries: true,
                scale: SAMPLE_SCALE_METERS
            });

            ee.data.computeValue(point_sample, function (val) {
                if (val.features.length > 0) {
                    var feature = val.features[0];
                    active_context.point_val.setValue(
                        formatPointValue(feature.properties.B0)
                    );

                    if (active_context.last_point_layer !== null) {
                        active_context.map.remove(
                            active_context.last_point_layer
                        );
                    }

                    active_context.last_point_layer =
                        active_context.map.addLayer(point, {
                            color: "#FF00FF"
                        });
                } else {
                    active_context.point_val.setValue("nodata");
                }
            });
        });
    });
});

panel_list[0][0].add(clone_to_right);
panel_list[1][0].add(clone_to_left);
