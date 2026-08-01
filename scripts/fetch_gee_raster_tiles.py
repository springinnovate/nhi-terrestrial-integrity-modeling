"""Fetch deterministic Earth Engine raster-stack tiles for an AOI."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Sequence

import ee
import httplib2
import rasterio
from pyproj import Transformer
from rasterio.crs import CRS
from rasterio.transform import Affine
from shapely.geometry import box, shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform, unary_union
from tqdm.auto import tqdm

from .analysis_config import (
    DEFAULT_ANALYSIS_CONFIG_PATH,
    AnalysisConfiguration,
    RasterCacheGrid as CacheGrid,
    load_analysis_configuration,
)


MANIFEST_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class CacheTile:
    """Identify one deterministic cache tile and its projected bounds.

    Attributes:
        column: Tile column relative to the global grid origin.
        row: Tile row relative to the global grid origin.
        tile_id: Stable file-safe identifier derived from the column and row.
        left: Minimum projected x coordinate.
        bottom: Minimum projected y coordinate.
        right: Maximum projected x coordinate.
        top: Maximum projected y coordinate.
    """

    column: int
    row: int
    tile_id: str
    left: float
    bottom: float
    right: float
    top: float


@dataclass(frozen=True)
class FetchSummary:
    """Summarize one AOI cache request.

    Attributes:
        requested_tiles: Number of cache tiles intersecting the AOI.
        reused_tiles: Number of valid existing tiles reused.
        downloaded_tiles: Number of tiles fetched during this request.
        failed_tiles: Number of tiles Earth Engine could not return.
        downloaded_bytes: Total bytes written for newly downloaded tiles.
    """

    requested_tiles: int
    reused_tiles: int
    downloaded_tiles: int
    failed_tiles: int
    downloaded_bytes: int


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Analysis definition and operational refresh and progress settings.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Fetch an analysis-defined Earth Engine raster stack into "
            "deterministic cache tiles intersecting its AOI."
        )
    )
    parser.add_argument(
        "analysis_configuration",
        type=Path,
        help=(
            "Complete TOML analysis definition containing the AOI, year, "
            "Earth Engine project, cache, reference thresholds, and stack. "
            f"Example: {DEFAULT_ANALYSIS_CONFIG_PATH}."
        ),
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Download every intersecting tile even when a valid cache entry exists.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Suppress tqdm progress output.",
    )
    return parser.parse_args()


def load_wgs84_aoi(aoi_path: Path) -> BaseGeometry:
    """Load one polygonal AOI from a GeoJSON file.

    Args:
        aoi_path: GeoJSON path containing WGS84 coordinates.

    Returns:
        Valid, nonempty Polygon or MultiPolygon geometry.

    Raises:
        FileNotFoundError: If the AOI file does not exist.
        ValueError: If the GeoJSON does not contain a valid polygonal AOI.
    """

    resolved_path = aoi_path.expanduser().resolve()
    geojson = json.loads(resolved_path.read_text(encoding="utf-8"))
    object_type = geojson.get("type")
    if object_type == "FeatureCollection":
        aoi_geometry = unary_union(
            [shape(feature["geometry"]) for feature in geojson["features"]]
        )
    elif object_type == "Feature":
        aoi_geometry = shape(geojson["geometry"])
    else:
        aoi_geometry = shape(geojson)

    if (
        aoi_geometry.is_empty
        or not aoi_geometry.is_valid
        or aoi_geometry.geom_type not in {"Polygon", "MultiPolygon"}
    ):
        raise ValueError("AOI must be a valid, nonempty Polygon or MultiPolygon.")
    return aoi_geometry


def select_intersecting_tiles(
    wgs84_aoi: BaseGeometry,
    cache_grid: CacheGrid,
) -> tuple[BaseGeometry, tuple[CacheTile, ...]]:
    """Project an AOI and select every positive-area intersecting cache tile.

    Args:
        wgs84_aoi: Polygonal AOI with longitude-latitude coordinates.
        cache_grid: Fixed projected cache grid.

    Returns:
        Projected AOI and tiles ordered north-to-south then west-to-east.
    """

    coordinate_transformer = Transformer.from_crs(
        "EPSG:4326",
        cache_grid.crs,
        always_xy=True,
    )
    projected_aoi = transform(coordinate_transformer.transform, wgs84_aoi)
    tile_span = cache_grid.pixel_size_meters * cache_grid.tile_size_pixels
    minimum_x, minimum_y, maximum_x, maximum_y = projected_aoi.bounds
    minimum_column = math.floor((minimum_x - cache_grid.origin_x) / tile_span)
    maximum_column = math.floor(
        (math.nextafter(maximum_x, -math.inf) - cache_grid.origin_x) / tile_span
    )
    minimum_row = math.floor((minimum_y - cache_grid.origin_y) / tile_span)
    maximum_row = math.floor(
        (math.nextafter(maximum_y, -math.inf) - cache_grid.origin_y) / tile_span
    )

    intersecting_tiles = []
    for row in range(maximum_row, minimum_row - 1, -1):
        for column in range(minimum_column, maximum_column + 1):
            tile_left = cache_grid.origin_x + column * tile_span
            tile_bottom = cache_grid.origin_y + row * tile_span
            tile_right = tile_left + tile_span
            tile_top = tile_bottom + tile_span
            if projected_aoi.intersection(
                box(tile_left, tile_bottom, tile_right, tile_top)
            ).area <= 0:
                continue
            intersecting_tiles.append(
                CacheTile(
                    column=column,
                    row=row,
                    tile_id=f"x{column:+07d}_y{row:+07d}",
                    left=tile_left,
                    bottom=tile_bottom,
                    right=tile_right,
                    top=tile_top,
                )
            )
    return projected_aoi, tuple(intersecting_tiles)


def build_stack_identifier(
    analysis_configuration: AnalysisConfiguration,
) -> str:
    """Build a stable cache identifier for one source configuration.

    Args:
        analysis_configuration: Complete analysis and raster-stack definition.

    Returns:
        File-safe stack identifier.
    """

    reference = analysis_configuration.reference
    human_modification_text = format(reference.human_modification, "g").replace(
        ".", "p"
    )
    human_influence_text = format(reference.human_influence, "g").replace(".", "p")
    return (
        f"{analysis_configuration.stack_name}_v"
        f"{analysis_configuration.stack_version}_"
        f"{analysis_configuration.raster_configuration_sha256[:12]}_"
        f"year_{analysis_configuration.year}_"
        f"gp_{reference.grassland_probability}_"
        f"hmi_{human_modification_text}_hii_{human_influence_text}"
    )


def load_cache_manifest(
    manifest_path: Path,
    cache_grid: CacheGrid,
) -> dict[str, Any]:
    """Load a cache manifest or initialize an empty compatible manifest.

    Args:
        manifest_path: JSON manifest path.
        cache_grid: Grid configuration required by this invocation.

    Returns:
        Mutable manifest dictionary.

    Raises:
        ValueError: If an existing manifest uses another schema or grid.
    """

    expected_grid_metadata = asdict(cache_grid)
    if not manifest_path.exists():
        return {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "grid": expected_grid_metadata,
            "stacks": {},
            "tiles": {},
            "requests": [],
        }

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            "Cache manifest schema does not match the current script version."
        )
    if manifest.get("grid") != expected_grid_metadata:
        raise ValueError(
            "Cache manifest grid differs from the current fixed cache grid."
        )
    return manifest


def write_cache_manifest(manifest_path: Path, manifest: dict[str, Any]) -> None:
    """Atomically write the complete cache manifest.

    Args:
        manifest_path: Destination JSON path.
        manifest: Complete serializable manifest.

    Returns:
        None: The manifest is written and atomically replaced.
    """

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = manifest_path.with_name(f"{manifest_path.name}.tmp")
    temporary_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_path, manifest_path)


def calculate_file_sha256(path: Path) -> str:
    """Calculate a file's SHA-256 checksum without loading it all at once.

    Args:
        path: File to hash.

    Returns:
        Lowercase hexadecimal SHA-256 digest.
    """

    checksum = hashlib.sha256()
    with path.open("rb") as file_handle:
        for file_block in iter(lambda: file_handle.read(1024 * 1024), b""):
            checksum.update(file_block)
    return checksum.hexdigest()


def validate_cached_tile(
    tile_path: Path,
    tile: CacheTile,
    cache_grid: CacheGrid,
    band_names: Sequence[str],
    expected_checksum: str | None = None,
    expected_file_size: int | None = None,
) -> tuple[int, str]:
    """Validate cached bytes and geospatial metadata for one tile.

    Args:
        tile_path: GeoTIFF path to validate.
        tile: Expected grid address and projected bounds.
        cache_grid: Expected pixel dimensions and CRS.
        band_names: Expected ordered band descriptions.
        expected_checksum: Previously recorded SHA-256 digest, when available.
        expected_file_size: Previously recorded byte size, when available.

    Returns:
        Validated file size and SHA-256 checksum.

    Raises:
        ValueError: If file size, checksum, or raster metadata is inconsistent.
    """

    actual_file_size = tile_path.stat().st_size
    if expected_file_size is not None and actual_file_size != expected_file_size:
        raise ValueError("Cached tile file size differs from its manifest record.")

    expected_transform = Affine(
        cache_grid.pixel_size_meters,
        0,
        tile.left,
        0,
        -cache_grid.pixel_size_meters,
        tile.top,
    )
    with rasterio.open(tile_path) as source:
        if source.width != cache_grid.tile_size_pixels:
            raise ValueError("Cached tile width does not match the cache grid.")
        if source.height != cache_grid.tile_size_pixels:
            raise ValueError("Cached tile height does not match the cache grid.")
        if source.count != len(band_names):
            raise ValueError("Cached tile band count does not match the stack.")
        if source.crs != CRS.from_string(cache_grid.crs):
            raise ValueError("Cached tile CRS does not match the cache grid.")
        if not source.transform.almost_equals(expected_transform):
            raise ValueError("Cached tile transform does not match its tile address.")
        if tuple(source.descriptions) != tuple(band_names):
            raise ValueError("Cached tile band names do not match the stack schema.")

    actual_checksum = calculate_file_sha256(tile_path)
    if expected_checksum is not None and actual_checksum != expected_checksum:
        raise ValueError("Cached tile checksum differs from its manifest record.")
    return actual_file_size, actual_checksum


def build_earth_engine_stack(
    analysis_configuration: AnalysisConfiguration,
) -> ee.Image:
    """Build the Earth Engine image declared by the stack configuration.

    Args:
        analysis_configuration: Dataset aliases, algorithms, year, thresholds,
            and ordered bands for the analysis.

    Returns:
        Computed Earth Engine image with configured stable export names.
    """

    year = analysis_configuration.year
    reference = analysis_configuration.reference
    datasets = analysis_configuration.datasets

    def annual_collection(dataset: str, selected_year: Any) -> ee.ImageCollection:
        start_date = ee.Date.fromYMD(ee.Number(selected_year).toInt(), 1, 1)
        return ee.ImageCollection(dataset).filterDate(
            start_date,
            start_date.advance(1, "year"),
        )

    def no_two_consecutive_zeros(
        annual_binary_builder: Callable[[Any], ee.Image],
    ) -> ee.Image:
        reference_years = ee.List.sequence(
            analysis_configuration.reference_start_year,
            analysis_configuration.reference_end_year,
        )
        annual_binary_images = ee.ImageCollection.fromImages(
            reference_years.map(
                lambda reference_year: annual_binary_builder(reference_year)
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
        datasets["grassland_probability"]
    )
    human_influence_collection = ee.ImageCollection(
        datasets["human_influence"]
    ).filterDate(
        f"{analysis_configuration.reference_start_year}-01-01",
        f"{analysis_configuration.reference_end_year + 1}-01-01",
    )
    grassland_probability_integrity = no_two_consecutive_zeros(
        lambda reference_year: ee.Image(
            grassland_probability_collection.filterDate(
                ee.Date.fromYMD(ee.Number(reference_year).toInt(), 1, 1),
                ee.Date.fromYMD(ee.Number(reference_year).add(1).toInt(), 1, 1),
            ).first()
        )
        .select(0)
        .gte(reference.grassland_probability)
    )
    human_influence_integrity = no_two_consecutive_zeros(
        lambda reference_year: human_influence_collection.filterDate(
            ee.Date.fromYMD(ee.Number(reference_year).toInt(), 1, 1),
            ee.Date.fromYMD(ee.Number(reference_year).add(1).toInt(), 1, 1),
        )
        .mean()
        .divide(7000)
        .lt(reference.human_influence)
    )
    reference_sites = (
        grassland_probability_integrity.And(human_influence_integrity)
        .And(
            ee.Image(datasets["human_modification"]).lte(
                reference.human_modification
            )
        )
        .selfMask()
        .toByte()
    )

    year_start = ee.Date.fromYMD(year, 1, 1)
    epoch_start = ee.Date("1970-01-01")
    phenology = ee.Image(
        annual_collection(datasets["modis_phenology"], year).first()
    )
    year_start_day = year_start.difference(epoch_start, "day")
    landsat_ndvi = annual_collection(datasets["landsat_ndvi"], year).select("NDVI")
    vegetation_height = (
        ee.Image(
            annual_collection(datasets["short_vegetation_height"], year).first()
        )
        .select("height")
        .multiply(0.1)
    )
    vegetation_cover = ee.Image(
        annual_collection(datasets["modis_vegetation_cover"], year).first()
    )
    lai_fpar_collection = annual_collection(datasets["modis_lai_fpar"], year)
    productivity = ee.Image(
        annual_collection(datasets["modis_productivity"], year).first()
    )

    computed_images = {
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

    era5_daily = annual_collection(datasets["era5_daily"], year)
    era5_monthly = annual_collection(datasets["era5_monthly"], year)
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
        year - analysis_configuration.interannual_rainfall_window_years + 1,
        analysis_configuration.era5_start_year,
    )
    annual_rainfall_totals = ee.ImageCollection.fromImages(
        ee.List.sequence(rainfall_start_year, year).map(
            lambda rainfall_year: annual_collection(
                datasets["era5_daily"],
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
    gridmet_drought = annual_collection(datasets["gridmet_drought"], year).select(
        "spi30d"
    )
    burned_month_count = (
        annual_collection(datasets["viirs_burned_area"], year)
        .select("Burn_Date")
        .map(lambda image: image.gt(0).unmask(0))
        .sum()
    )
    water_presence_variation = (
        annual_collection(datasets["jrc_monthly_water"], year)
        .select("water")
        .map(lambda image: image.eq(2).updateMask(image.neq(0)))
        .reduce(ee.Reducer.stdDev())
    )
    streams = (
        ee.Image(datasets["merit_hydro"])
        .select("upa")
        .gte(analysis_configuration.stream_upstream_area_threshold_km2)
        .unmask(0)
    )
    distance_to_streams = (
        streams.fastDistanceTransform()
        .sqrt()
        .multiply(ee.Image.pixelArea().sqrt())
    )
    gldas_annual = annual_collection(datasets["gldas"], year)
    modis_evapotranspiration = (
        annual_collection(datasets["modis_evapotranspiration"], year)
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
        month_start = ee.Date.fromYMD(year, ee.Number(month).toInt(), 1)
        positive_monthly_snow = (
            ee.ImageCollection(datasets["smap"])
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
    computed_images.update(
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
                datasets["isric_soil_organic_carbon"]
            )
            .select("soc_5-15cm_mean")
            .divide(10),
            "soil_moisture": gldas_annual.select("SoilMoi10_40cm_inst").mean(),
            "landform_type": ee.Image(datasets["srtm_landforms"]).select(
                "constant"
            ),
            "topographic_diversity": ee.Image(
                datasets["alos_topographic_diversity"]
            ).select("constant"),
            "annual_evapotranspiration": modis_evapotranspiration,
            "gldas_snow_depth": gldas_snow_depth,
            "smap_snow_depth": smap_snow_depth,
        }
    )

    configured_images = [
        computed_images[band.computation] for band in analysis_configuration.bands
    ]
    renamed_layers = [
        ee.Image(layer_image).rename(band_name)
        for layer_image, band_name in zip(
            configured_images,
            analysis_configuration.band_names(),
            strict=True,
        )
    ]
    raster_stack = renamed_layers[0]
    for renamed_layer in renamed_layers[1:]:
        raster_stack = raster_stack.addBands(renamed_layer)
    return raster_stack.toFloat()


def fetch_tile_bytes(
    raster_stack: ee.Image,
    tile: CacheTile,
    cache_grid: CacheGrid,
    band_names: Sequence[str],
) -> bytes:
    """Fetch one explicitly gridded GeoTIFF tile with ``computePixels``.

    Args:
        raster_stack: Computed Earth Engine d01-d39 image.
        tile: Cache tile to request.
        cache_grid: Fixed output grid.
        band_names: Ordered bands to include.

    Returns:
        Raw GeoTIFF bytes returned by Earth Engine.
    """

    request = {
        "expression": raster_stack,
        "fileFormat": "GEO_TIFF",
        "bandIds": list(band_names),
        "grid": {
            "dimensions": {
                "width": cache_grid.tile_size_pixels,
                "height": cache_grid.tile_size_pixels,
            },
            "affineTransform": {
                "scaleX": cache_grid.pixel_size_meters,
                "shearX": 0,
                "translateX": tile.left,
                "shearY": 0,
                "scaleY": -cache_grid.pixel_size_meters,
                "translateY": tile.top,
            },
            # Earth Engine does not recognize the EPSG:6933 authority code, but
            # its PixelGrid API accepts the same EASE-Grid definition as WKT1.
            "crsWkt": CRS.from_string(cache_grid.crs).to_wkt(
                version="WKT1_GDAL"
            ),
        },
        "workloadTag": "nhi-raster-tile-cache",
    }
    return ee.data.computePixels(request)


def cache_aoi_tiles(
    analysis_configuration: AnalysisConfiguration,
    refresh: bool,
    show_progress: bool,
    compute_tile: (
        Callable[[ee.Image | None, CacheTile, CacheGrid, Sequence[str]], bytes]
        | None
    ) = None,
) -> FetchSummary:
    """Populate reusable Earth Engine cache tiles intersecting one AOI.

    Args:
        analysis_configuration: Authoritative AOI, year, Earth Engine, cache,
            reference threshold, grid, and raster-stack settings.
        refresh: Whether valid existing tiles should be replaced.
        show_progress: Whether to display tqdm progress.
        compute_tile: Optional tile-fetch implementation used by offline tests.

    Returns:
        Counts and byte totals for requested, reused, downloaded, and failed tiles.

    Raises:
        RuntimeError: If Earth Engine initialization fails or a tile exhausts
            its configured request attempts.
    """

    resolved_cache_directory = analysis_configuration.earth_engine.cache_directory
    request_timeout_seconds = (
        analysis_configuration.earth_engine.request_timeout_seconds
    )
    request_retry_count = analysis_configuration.earth_engine.request_retry_count
    resolved_cache_directory.mkdir(parents=True, exist_ok=True)
    manifest_path = resolved_cache_directory / "manifest.json"
    cache_grid = CacheGrid(**asdict(analysis_configuration.grid))
    wgs84_aoi = load_wgs84_aoi(analysis_configuration.aoi_path)
    projected_aoi, requested_tiles = select_intersecting_tiles(
        wgs84_aoi,
        cache_grid,
    )
    band_names = analysis_configuration.band_names()
    stack_identifier = build_stack_identifier(analysis_configuration)
    manifest = load_cache_manifest(manifest_path, cache_grid)
    manifest["stacks"][stack_identifier] = {
        "analysis_name": analysis_configuration.analysis_name,
        "display_name": analysis_configuration.display_name,
        "aoi_path": str(analysis_configuration.aoi_path),
        "name": analysis_configuration.stack_name,
        "definition_version": analysis_configuration.stack_version,
        "configuration_path": str(analysis_configuration.path),
        "analysis_configuration_sha256": (
            analysis_configuration.configuration_sha256
        ),
        "raster_configuration_sha256": (
            analysis_configuration.raster_configuration_sha256
        ),
        "datasets": dict(analysis_configuration.datasets),
        "year": analysis_configuration.year,
        "reference_thresholds": asdict(analysis_configuration.reference),
        "bands": [
            {
                **asdict(definition),
                "name": band_name,
            }
            for definition, band_name in zip(
                analysis_configuration.bands,
                band_names,
                strict=True,
            )
        ],
    }
    write_cache_manifest(manifest_path, manifest)

    valid_cached_tiles: set[str] = set()
    tiles_requiring_download = []
    tile_directory = resolved_cache_directory / "tiles" / stack_identifier
    with tqdm(
        requested_tiles,
        total=len(requested_tiles),
        desc="Checking cached grids",
        unit="grid",
        dynamic_ncols=True,
        disable=not show_progress,
    ) as cache_progress:
        for tile in cache_progress:
            tile_record_key = f"{stack_identifier}/{tile.tile_id}"
            tile_record = manifest["tiles"].get(tile_record_key)
            tile_path = tile_directory / f"{tile.tile_id}.tif"
            if not refresh and tile_record is not None and tile_path.exists():
                try:
                    validate_cached_tile(
                        tile_path,
                        tile,
                        cache_grid,
                        band_names,
                        expected_checksum=tile_record["sha256"],
                        expected_file_size=tile_record["file_size_bytes"],
                    )
                    valid_cached_tiles.add(tile.tile_id)
                    cache_progress.set_postfix(
                        cached=len(valid_cached_tiles),
                        download=len(tiles_requiring_download),
                        refresh=refresh,
                    )
                    continue
                except (OSError, ValueError, rasterio.errors.RasterioError):
                    pass
            manifest["tiles"].pop(tile_record_key, None)
            tiles_requiring_download.append(tile)
            cache_progress.set_postfix(
                cached=len(valid_cached_tiles),
                download=len(tiles_requiring_download),
                refresh=refresh,
            )
    write_cache_manifest(manifest_path, manifest)

    print()
    print("Earth Engine grid plan")
    print(f"  Intersecting grids: {len(requested_tiles):,}")
    print(f"  Valid cached grids: {len(valid_cached_tiles):,}")
    print(f"  Grids to download: {len(tiles_requiring_download):,}")
    print(
        "  Request policy: "
        f"{request_timeout_seconds:g} second timeout, "
        f"{request_retry_count + 1} attempt(s) per grid"
    )

    raster_stack = None
    tile_fetcher = compute_tile or fetch_tile_bytes
    if tiles_requiring_download and compute_tile is None:
        try:
            ee.data.setMaxRetries(request_retry_count)
            ee.Initialize(
                project=analysis_configuration.earth_engine.project,
                http_transport=httplib2.Http(timeout=request_timeout_seconds),
            )
            ee.data.setDeadline(request_timeout_seconds * 1000)
        except Exception as error:
            raise RuntimeError(
                "Could not initialize or configure Earth Engine. Authenticate with "
                "`earthengine authenticate` and verify earth_engine.project "
                "in the analysis TOML."
            ) from error
        raster_stack = build_earth_engine_stack(analysis_configuration)

    downloaded_tile_count = 0
    downloaded_byte_count = 0
    failed_tile_ids = []
    failed_tile_error: Exception | None = None
    tile_directory.mkdir(parents=True, exist_ok=True)
    with tqdm(
        total=len(requested_tiles),
        initial=len(valid_cached_tiles),
        desc="Processing grids",
        unit="grid",
        dynamic_ncols=True,
        disable=not show_progress,
    ) as processing_progress:
        processing_progress.set_postfix(
            processed=len(valid_cached_tiles),
            cached=len(valid_cached_tiles),
            downloaded=downloaded_tile_count,
            failed=len(failed_tile_ids),
        )
        for tile in tiles_requiring_download:
            destination_path = tile_directory / f"{tile.tile_id}.tif"
            temporary_path = destination_path.with_suffix(".tif.partial")
            try:
                tile_bytes = tile_fetcher(
                    raster_stack,
                    tile,
                    cache_grid,
                    band_names,
                )
                temporary_path.write_bytes(tile_bytes)
                # computePixels preserves band order but does not populate GeoTIFF
                # descriptions. Add the stable schema before validation and hashing.
                with rasterio.open(temporary_path, "r+") as temporary_raster:
                    for band_index, band_name in enumerate(band_names, start=1):
                        temporary_raster.set_band_description(band_index, band_name)
                file_size, checksum = validate_cached_tile(
                    temporary_path,
                    tile,
                    cache_grid,
                    band_names,
                )
                os.replace(temporary_path, destination_path)
                tile_record_key = f"{stack_identifier}/{tile.tile_id}"
                manifest["tiles"][tile_record_key] = {
                    "tile_id": tile.tile_id,
                    "stack_id": stack_identifier,
                    "year": analysis_configuration.year,
                    "column": tile.column,
                    "row": tile.row,
                    "bounds": [tile.left, tile.bottom, tile.right, tile.top],
                    "crs": cache_grid.crs,
                    "pixel_size_meters": cache_grid.pixel_size_meters,
                    "width_pixels": cache_grid.tile_size_pixels,
                    "height_pixels": cache_grid.tile_size_pixels,
                    "transform": [
                        cache_grid.pixel_size_meters,
                        0,
                        tile.left,
                        0,
                        -cache_grid.pixel_size_meters,
                        tile.top,
                    ],
                    "relative_path": destination_path.relative_to(
                        resolved_cache_directory
                    ).as_posix(),
                    "fetched_at_utc": datetime.now(UTC).isoformat(),
                    "file_size_bytes": file_size,
                    "sha256": checksum,
                }
                write_cache_manifest(manifest_path, manifest)
                downloaded_tile_count += 1
                downloaded_byte_count += file_size
            except Exception as error:
                temporary_path.unlink(missing_ok=True)
                failed_tile_ids.append(tile.tile_id)
                failed_tile_error = error
                processing_progress.write(
                    f"Failed {tile.tile_id} after "
                    f"{request_retry_count + 1} attempt(s): "
                    f"{type(error).__name__}: {error}"
                )
            finally:
                processed_tile_count = (
                    len(valid_cached_tiles)
                    + downloaded_tile_count
                    + len(failed_tile_ids)
                )
                processing_progress.update(1)
                processing_progress.set_postfix(
                    processed=processed_tile_count,
                    cached=len(valid_cached_tiles),
                    downloaded=downloaded_tile_count,
                    failed=len(failed_tile_ids),
                )
            if failed_tile_error is not None:
                break

    request_timestamp = datetime.now(UTC).isoformat()
    request_identifier = hashlib.sha256(
        (
            f"{request_timestamp}|{analysis_configuration.aoi_path}|"
            f"{stack_identifier}"
        ).encode("utf-8")
    ).hexdigest()[:16]
    manifest["requests"].append(
        {
            "request_id": request_identifier,
            "requested_at_utc": request_timestamp,
            "analysis_name": analysis_configuration.analysis_name,
            "aoi_path": str(analysis_configuration.aoi_path),
            "aoi_bounds_wgs84": list(wgs84_aoi.bounds),
            "aoi_area_m2": projected_aoi.area,
            "stack_id": stack_identifier,
            "tile_ids": [tile.tile_id for tile in requested_tiles],
            "reused_tile_ids": sorted(valid_cached_tiles),
            "downloaded_tiles": downloaded_tile_count,
            "failed_tile_ids": failed_tile_ids,
            "request_timeout_seconds": request_timeout_seconds,
            "request_retry_count": request_retry_count,
            "aborted_after_failure": failed_tile_error is not None,
            "failure": (
                {
                    "type": type(failed_tile_error).__name__,
                    "message": str(failed_tile_error),
                }
                if failed_tile_error is not None
                else None
            ),
        }
    )
    write_cache_manifest(manifest_path, manifest)

    summary = FetchSummary(
        requested_tiles=len(requested_tiles),
        reused_tiles=len(valid_cached_tiles),
        downloaded_tiles=downloaded_tile_count,
        failed_tiles=len(failed_tile_ids),
        downloaded_bytes=downloaded_byte_count,
    )
    print()
    print("Earth Engine raster tile cache")
    print(f"  Analysis: {analysis_configuration.analysis_name}")
    print(f"  AOI: {analysis_configuration.aoi_path}")
    print(f"  AOI area: {projected_aoi.area / 1_000_000:,.2f} km2")
    print(f"  Data year: {analysis_configuration.year}")
    print(f"  Stack: {stack_identifier}")
    print(f"  Configuration: {analysis_configuration.path}")
    print(
        "  Grid: "
        f"{cache_grid.crs}, {cache_grid.pixel_size_meters} m pixels, "
        f"{cache_grid.tile_size_pixels} x {cache_grid.tile_size_pixels} pixels/tile"
    )
    print(f"  Requested tiles: {summary.requested_tiles:,}")
    print(f"  Reused from cache: {summary.reused_tiles:,}")
    print(f"  Downloaded: {summary.downloaded_tiles:,}")
    print(f"  Downloaded size: {summary.downloaded_bytes / 1_048_576:,.2f} MiB")
    print(f"  Failed: {summary.failed_tiles:,}")
    print(f"  Manifest: {manifest_path}")

    if failed_tile_ids:
        raise RuntimeError(
            f"Earth Engine failed to return tile {failed_tile_ids[0]} after "
            f"{request_retry_count + 1} attempt(s) with a "
            f"{request_timeout_seconds:g} second timeout per attempt. "
            "Completed tiles remain cached; rerun the same command to resume."
        ) from failed_tile_error
    return summary


def main() -> None:
    """Fetch and cache all Earth Engine raster tiles intersecting the AOI.

    Returns:
        None: Cache tiles, manifest metadata, and reports are written to disk.
    """

    args = parse_args()
    analysis_configuration = load_analysis_configuration(
        args.analysis_configuration
    )
    cache_aoi_tiles(
        analysis_configuration,
        refresh=args.refresh,
        show_progress=not args.no_progress,
    )


if __name__ == "__main__":
    main()
