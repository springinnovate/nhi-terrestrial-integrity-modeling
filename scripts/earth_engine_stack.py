"""Build the configured Habitat raster bands as an Earth Engine image."""

from __future__ import annotations

from typing import Any, Callable

import ee

from .analysis_config import RasterStackConfiguration


def build_earth_engine_stack(
    configuration: RasterStackConfiguration,
) -> ee.Image:
    """Build the Earth Engine image declared by the stack configuration.

    Args:
        configuration: Dataset aliases, algorithms, year, thresholds,
            and ordered bands for the analysis.

    Returns:
        Computed Earth Engine image with configured stable export names.
    """

    analysis_year = configuration.year
    reference_settings = configuration.reference
    dataset_ids = configuration.datasets

    def annual_collection(
        dataset_id: str,
        selected_year: Any,
    ) -> ee.ImageCollection:
        start_date = ee.Date.fromYMD(ee.Number(selected_year).toInt(), 1, 1)
        return ee.ImageCollection(dataset_id).filterDate(
            start_date,
            start_date.advance(1, "year"),
        )

    def meets_threshold_without_consecutive_failures(
        annual_threshold_image_builder: Callable[[Any], ee.Image],
    ) -> ee.Image:
        reference_years = ee.List.sequence(
            configuration.reference_start_year,
            configuration.reference_end_year,
        )
        annual_binary_images = ee.ImageCollection.fromImages(
            reference_years.map(
                lambda reference_year: annual_threshold_image_builder(
                    reference_year
                )
                .rename("g")
                .set("year", reference_year)
            )
        )
        annual_binary_list = annual_binary_images.toList(annual_binary_images.size())
        adjacent_year_pairs = ee.ImageCollection.fromImages(
            ee.List.sequence(
                0,
                ee.Number(annual_binary_list.size()).subtract(2),
            ).map(
                lambda list_index: ee.Image(annual_binary_list.get(list_index)).Or(
                    ee.Image(
                        annual_binary_list.get(ee.Number(list_index).add(1))
                    )
                )
            )
        )
        return adjacent_year_pairs.reduce(ee.Reducer.min()).eq(1)

    grassland_probability_collection = ee.ImageCollection(
        dataset_ids["grassland_probability"]
    )
    human_influence_collection = ee.ImageCollection(
        dataset_ids["human_influence"]
    ).filterDate(
        f"{configuration.reference_start_year}-01-01",
        f"{configuration.reference_end_year + 1}-01-01",
    )
    grassland_probability_integrity = meets_threshold_without_consecutive_failures(
        lambda reference_year: ee.Image(
            grassland_probability_collection.filterDate(
                ee.Date.fromYMD(ee.Number(reference_year).toInt(), 1, 1),
                ee.Date.fromYMD(ee.Number(reference_year).add(1).toInt(), 1, 1),
            ).first()
        )
        .select(0)
        .gte(reference_settings.grassland_probability)
    )
    human_influence_integrity = meets_threshold_without_consecutive_failures(
        lambda reference_year: human_influence_collection.filterDate(
            ee.Date.fromYMD(ee.Number(reference_year).toInt(), 1, 1),
            ee.Date.fromYMD(ee.Number(reference_year).add(1).toInt(), 1, 1),
        )
        .mean()
        .divide(7000)
        .lt(reference_settings.human_influence)
    )
    reference_sites = (
        grassland_probability_integrity.And(human_influence_integrity)
        .And(
            ee.Image(dataset_ids["human_modification"]).lte(
                reference_settings.human_modification
            )
        )
        .selfMask()
        .toByte()
    )

    year_start = ee.Date.fromYMD(analysis_year, 1, 1)
    epoch_start = ee.Date("1970-01-01")
    phenology = ee.Image(
        annual_collection(dataset_ids["modis_phenology"], analysis_year).first()
    )
    year_start_day = year_start.difference(epoch_start, "day")
    landsat_ndvi = annual_collection(
        dataset_ids["landsat_ndvi"],
        analysis_year,
    ).select("NDVI")
    vegetation_height = (
        ee.Image(
            annual_collection(
                dataset_ids["short_vegetation_height"],
                analysis_year,
            ).first()
        )
        .select("height")
        .multiply(0.1)
    )
    vegetation_cover = ee.Image(
        annual_collection(
            dataset_ids["modis_vegetation_cover"],
            analysis_year,
        ).first()
    )
    lai_fpar_collection = annual_collection(
        dataset_ids["modis_lai_fpar"],
        analysis_year,
    )
    productivity = ee.Image(
        annual_collection(
            dataset_ids["modis_productivity"],
            analysis_year,
        ).first()
    )

    computed_images_by_name = {
        "grassland_reference_sites": reference_sites,
        "ndvi_95th_percentile": landsat_ndvi.reduce(
            ee.Reducer.percentile([95])
        ),
        "ndvi_median": landsat_ndvi.reduce(ee.Reducer.percentile([50])),
        "growing_season_length_1": phenology.select("Senescence_1").subtract(
            phenology.select("Greenup_1")
        ),
        "growing_season_length_2": phenology.select("Senescence_2").subtract(
            phenology.select("Greenup_2")
        ),
        "greenup_timing_1": phenology.select("Greenup_1").subtract(
            year_start_day
        ),
        "greenup_timing_2": phenology.select("Greenup_2").subtract(
            year_start_day
        ),
        "short_vegetation_height": vegetation_height,
        "tree_cover": vegetation_cover.select("Percent_Tree_Cover"),
        "non_tree_vegetation_cover": vegetation_cover.select(
            "Percent_NonTree_Vegetation"
        ),
        "bare_ground": vegetation_cover.select("Percent_NonVegetated"),
        "maximum_lai": lai_fpar_collection.select("Lai_500m")
        .map(lambda image: image.multiply(0.1))
        .max(),
        "lai_variability": lai_fpar_collection.select("Lai_500m")
        .map(lambda image: image.multiply(0.1))
        .reduce(ee.Reducer.stdDev()),
        "mean_fpar": lai_fpar_collection.select("Fpar_500m")
        .map(lambda image: image.multiply(0.01))
        .mean(),
        "fpar_variability": lai_fpar_collection.select("Fpar_500m")
        .map(lambda image: image.multiply(0.01))
        .reduce(ee.Reducer.stdDev()),
        "maximum_fpar_variability": lai_fpar_collection.select(
            "FparStdDev_500m"
        )
        .map(lambda image: image.multiply(0.01))
        .max(),
        "growing_season_count": phenology.select("NumCycles"),
        "npp": productivity.select("Npp").multiply(0.0001),
        "gpp": productivity.select("Gpp").multiply(0.0001),
    }

    era5_daily = annual_collection(dataset_ids["era5_daily"], analysis_year)
    era5_monthly = annual_collection(dataset_ids["era5_monthly"], analysis_year)
    annual_precipitation = era5_daily.select("total_precipitation").sum().multiply(
        1000
    )
    greenup = phenology.select("Greenup_1")
    senescence = phenology.select("Senescence_1")

    def apply_growing_season_mask(image: ee.Image) -> ee.Image:
        image_day = ee.Date(image.get("system:time_start")).difference(
            epoch_start,
            "day",
        )
        image_day_constant = ee.Image.constant(image_day)
        growing_season_mask = image_day_constant.gte(greenup).And(
            image_day_constant.lte(senescence)
        )
        return image.updateMask(growing_season_mask)

    growing_season_daily = era5_daily.map(apply_growing_season_mask)
    rainfall_start_year = max(
        analysis_year
        - configuration.interannual_rainfall_window_years
        + 1,
        configuration.era5_start_year,
    )
    annual_rainfall_totals = ee.ImageCollection.fromImages(
        ee.List.sequence(rainfall_start_year, analysis_year).map(
            lambda rainfall_year: annual_collection(
                dataset_ids["era5_daily"],
                rainfall_year,
            )
            .select("total_precipitation")
            .sum()
            .multiply(1000)
        )
    )
    mean_annual_rainfall = annual_rainfall_totals.mean()
    rainfall_variability = (
        annual_rainfall_totals.reduce(ee.Reducer.stdDev())
        .divide(mean_annual_rainfall)
        .multiply(100)
        .updateMask(mean_annual_rainfall.neq(0))
    )
    gridmet_drought = annual_collection(
        dataset_ids["gridmet_drought"],
        analysis_year,
    ).select("spi30d")
    burned_month_count = (
        annual_collection(dataset_ids["viirs_burned_area"], analysis_year)
        .select("Burn_Date")
        .map(lambda image: image.gt(0).unmask(0))
        .sum()
    )
    water_presence_variation = (
        annual_collection(dataset_ids["jrc_monthly_water"], analysis_year)
        .select("water")
        .map(lambda image: image.eq(2).updateMask(image.neq(0)))
        .reduce(ee.Reducer.stdDev())
    )
    streams = (
        ee.Image(dataset_ids["merit_hydro"])
        .select("upa")
        .gte(configuration.stream_upstream_area_threshold_km2)
        .unmask(0)
    )
    distance_to_streams = (
        streams.fastDistanceTransform()
        .sqrt()
        .multiply(ee.Image.pixelArea().sqrt())
    )
    gldas_annual = annual_collection(dataset_ids["gldas"], analysis_year)
    modis_evapotranspiration = (
        annual_collection(
            dataset_ids["modis_evapotranspiration"],
            analysis_year,
        )
        .select("ET")
        .map(lambda image: image.multiply(0.1))
        .sum()
    )
    gldas_snow_depth = (
        gldas_annual.select("SnowDepth_inst")
        .map(lambda image: image.updateMask(image.gt(0)))
        .mean()
    )
    # SMAP contains thousands of three-hourly images per year. Monthly positive
    # sums and counts preserve the observation-weighted annual mean while keeping
    # each interactive Earth Engine reduction below its server-memory limit.
    def summarize_smap_month(month: Any) -> ee.Image:
        month_start = ee.Date.fromYMD(
            analysis_year,
            ee.Number(month).toInt(),
            1,
        )
        positive_monthly_snow = (
            ee.ImageCollection(dataset_ids["smap"])
            .filterDate(month_start, month_start.advance(1, "month"))
            .select("snow_depth")
            .map(lambda image: image.updateMask(image.gt(0)))
        )
        return positive_monthly_snow.sum().rename("snow_sum").addBands(
            positive_monthly_snow.count().rename("snow_count")
        )

    monthly_smap_snow = ee.ImageCollection.fromImages(
        ee.List.sequence(1, 12).map(summarize_smap_month)
    )
    annual_smap_snow_count = monthly_smap_snow.select("snow_count").sum()
    smap_snow_depth = (
        monthly_smap_snow.select("snow_sum")
        .sum()
        .divide(annual_smap_snow_count)
        .updateMask(annual_smap_snow_count.gt(0))
    )
    computed_images_by_name.update(
        {
            "maximum_annual_temperature": era5_daily.select(
                "maximum_2m_air_temperature"
            )
            .max()
            .subtract(273.15),
            "mean_annual_temperature": era5_monthly.select(
                "mean_2m_air_temperature"
            )
            .mean()
            .subtract(273.15),
            "median_annual_temperature": era5_monthly.select(
                "mean_2m_air_temperature"
            )
            .median()
            .subtract(273.15),
            "minimum_annual_temperature": era5_daily.select(
                "minimum_2m_air_temperature"
            )
            .min()
            .subtract(273.15),
            "annual_precipitation": annual_precipitation,
            "growing_season_temperature": growing_season_daily.select(
                "mean_2m_air_temperature"
            )
            .mean()
            .subtract(273.15),
            "growing_season_precipitation": growing_season_daily.select(
                "total_precipitation"
            )
            .mean()
            .multiply(1000),
            "interannual_rainfall_variability": rainfall_variability,
            "mean_drought": gridmet_drought.mean(),
            "drought_fifth_percentile": gridmet_drought.reduce(
                ee.Reducer.percentile([5])
            ),
            "burned_month_count": burned_month_count,
            "water_presence_variation": water_presence_variation,
            "distance_to_streams": distance_to_streams,
            "soil_organic_carbon": ee.Image(
                dataset_ids["isric_soil_organic_carbon"]
            )
            .select("soc_5-15cm_mean")
            .divide(10),
            "soil_moisture": gldas_annual.select("SoilMoi10_40cm_inst").mean(),
            "landform_type": ee.Image(dataset_ids["srtm_landforms"]).select(
                "constant"
            ),
            "topographic_diversity": ee.Image(
                dataset_ids["alos_topographic_diversity"]
            ).select("constant"),
            "annual_evapotranspiration": modis_evapotranspiration,
            "gldas_snow_depth": gldas_snow_depth,
            "smap_snow_depth": smap_snow_depth,
        }
    )

    configured_band_images = [
        ee.Image(computed_images_by_name[band_definition.computation]).rename(
            band_name
        )
        for band_definition, band_name in zip(
            configuration.bands,
            configuration.band_names(),
            strict=True,
        )
    ]
    return ee.Image.cat(*configured_band_images).toFloat()
