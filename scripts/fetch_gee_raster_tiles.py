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


MANIFEST_SCHEMA_VERSION = 1
STACK_DEFINITION_VERSION = 1
DEFAULT_CACHE_DIRECTORY = Path("data/gee_raster_cache")
DEFAULT_CACHE_CRS = "EPSG:6933"
DEFAULT_PIXEL_SIZE_METERS = 500
DEFAULT_TILE_SIZE_PIXELS = 128
DEFAULT_GRASSLAND_PROBABILITY_THRESHOLD = 80
DEFAULT_HMI_THRESHOLD = 0.1
DEFAULT_HII_THRESHOLD = 0.08
# Earth Engine limits interactive computations to five minutes. The extra
# minute lets the server return its own timeout before the transport is closed.
DEFAULT_REQUEST_TIMEOUT_SECONDS = 360
DEFAULT_REQUEST_RETRY_COUNT = 1
MINIMUM_COMPLETE_STACK_YEAR = 2015
MAXIMUM_COMPLETE_STACK_YEAR = 2019
REFERENCE_START_YEAR = 2001
REFERENCE_END_YEAR = 2020
ERA5_START_YEAR = 1979
INTERANNUAL_RAINFALL_WINDOW_YEARS = 10
STREAM_UPSTREAM_AREA_THRESHOLD_KM2 = 25

MAYBE_GRASSLAND_ECOREGIONS = (
    "projects/ecoshard-202922/assets/nhi_assets/"
    "maybe_grassland_ecoregions_simplified_100m"
)
GRASSLAND_PROBABILITY_COLLECTION = (
    "projects/global-pasture-watch/assets/ggc-30m/v1/nat-semi-grassland_p"
)
HUMAN_MODIFICATION_IMAGE = (
    "projects/hm-30x30/assets/output/v20240801/HMv20240801_2022s_AA_300"
)
HUMAN_INFLUENCE_COLLECTION = "projects/HII/v1/hii"
LANDSAT_NDVI_COLLECTION = "LANDSAT/COMPOSITES/C02/T1_L2_8DAY_NDVI"
MODIS_PHENOLOGY_COLLECTION = "MODIS/061/MCD12Q2"
SHORT_VEGETATION_HEIGHT_COLLECTION = (
    "projects/global-pasture-watch/assets/gsvh-30m/v1/short-veg-height_m"
)
MODIS_VEGETATION_COVER_COLLECTION = "MODIS/061/MOD44B"
MODIS_LAI_FPAR_COLLECTION = "MODIS/061/MOD15A2H"
MODIS_PRODUCTIVITY_COLLECTION = "MODIS/061/MOD17A3HGF"
ERA5_DAILY_COLLECTION = "ECMWF/ERA5/DAILY"
ERA5_MONTHLY_COLLECTION = "ECMWF/ERA5/MONTHLY"
GRIDMET_DROUGHT_COLLECTION = "GRIDMET/DROUGHT"
VIIRS_BURNED_AREA_COLLECTION = "NASA/VIIRS/002/VNP64A1"
JRC_MONTHLY_WATER_COLLECTION = "JRC/GSW1_4/MonthlyHistory"
MERIT_HYDRO_IMAGE = "MERIT/Hydro/v1_0_1"
ISRIC_SOIL_ORGANIC_CARBON_IMAGE = "projects/soilgrids-isric/soc_mean"
GLDAS_COLLECTION = "NASA/GLDAS/V021/NOAH/G025/T3H"
SMAP_COLLECTION = "NASA/SMAP/SPL4SMGP/008"
MODIS_EVAPOTRANSPIRATION_COLLECTION = "MODIS/061/MOD16A2GF"
SRTM_LANDFORMS_IMAGE = "CSP/ERGo/1_0/Global/SRTM_landforms"
ALOS_TOPOGRAPHIC_DIVERSITY_IMAGE = "CSP/ERGo/1_0/Global/ALOS_topoDiversity"


@dataclass(frozen=True)
class RasterBandDefinition:
    """Describe one band in the cached Earth Engine raster stack.

    Attributes:
        number: Stable one-based d01-d39 band number.
        suffix: Stable export-name suffix used after the year and band number.
        display_name: Human-readable ecological or environmental variable name.
        source_dataset_ids: Earth Engine datasets used to construct the band.
    """

    number: int
    suffix: str
    display_name: str
    source_dataset_ids: tuple[str, ...]


BAND_DEFINITIONS = (
    RasterBandDefinition(
        1,
        "grassland_reference_sites",
        "Grassland reference sites",
        (
            GRASSLAND_PROBABILITY_COLLECTION,
            HUMAN_MODIFICATION_IMAGE,
            HUMAN_INFLUENCE_COLLECTION,
            MAYBE_GRASSLAND_ECOREGIONS,
        ),
    ),
    RasterBandDefinition(
        2,
        "ndvi_95th_percentile_across",
        "NDVI 95th percentile",
        (LANDSAT_NDVI_COLLECTION,),
    ),
    RasterBandDefinition(
        3,
        "ndvi_50th_percentile_across",
        "NDVI median",
        (LANDSAT_NDVI_COLLECTION,),
    ),
    RasterBandDefinition(
        4,
        "length_of_growing_season_1",
        "Growing-season length 1",
        (MODIS_PHENOLOGY_COLLECTION,),
    ),
    RasterBandDefinition(
        5,
        "length_of_growing_season_2",
        "Growing-season length 2",
        (MODIS_PHENOLOGY_COLLECTION,),
    ),
    RasterBandDefinition(
        6,
        "timing_of_green_up_1",
        "Green-up timing 1",
        (MODIS_PHENOLOGY_COLLECTION,),
    ),
    RasterBandDefinition(
        7,
        "timing_of_green_up_2",
        "Green-up timing 2",
        (MODIS_PHENOLOGY_COLLECTION,),
    ),
    RasterBandDefinition(
        8,
        "short_vegetation_height",
        "Short vegetation height",
        (SHORT_VEGETATION_HEIGHT_COLLECTION,),
    ),
    RasterBandDefinition(
        9,
        "percent_tree_cover",
        "Tree cover",
        (MODIS_VEGETATION_COVER_COLLECTION,),
    ),
    RasterBandDefinition(
        10,
        "percent_veg_but_not_tree_cov",
        "Non-tree vegetation cover",
        (MODIS_VEGETATION_COVER_COLLECTION,),
    ),
    RasterBandDefinition(
        11,
        "percent_bare",
        "Bare ground",
        (MODIS_VEGETATION_COVER_COLLECTION,),
    ),
    RasterBandDefinition(
        12,
        "leaf_area_index_lai_annual_m",
        "Maximum leaf area index",
        (MODIS_LAI_FPAR_COLLECTION,),
    ),
    RasterBandDefinition(
        13,
        "leaf_area_index_lai_annual_s",
        "Leaf area index variability",
        (MODIS_LAI_FPAR_COLLECTION,),
    ),
    RasterBandDefinition(
        14,
        "fraction_of_photosynthetical",
        "Mean FPAR",
        (MODIS_LAI_FPAR_COLLECTION,),
    ),
    RasterBandDefinition(
        15,
        "fraction_of_photosynthetical",
        "FPAR variability",
        (MODIS_LAI_FPAR_COLLECTION,),
    ),
    RasterBandDefinition(
        16,
        "fpar_variability_max",
        "Maximum FPAR variability",
        (MODIS_LAI_FPAR_COLLECTION,),
    ),
    RasterBandDefinition(
        17,
        "number_of_growing_seasons",
        "Number of growing seasons",
        (MODIS_PHENOLOGY_COLLECTION,),
    ),
    RasterBandDefinition(
        18,
        "npp",
        "Net primary productivity",
        (MODIS_PRODUCTIVITY_COLLECTION,),
    ),
    RasterBandDefinition(
        19,
        "gpp",
        "Gross primary productivity",
        (MODIS_PRODUCTIVITY_COLLECTION,),
    ),
    RasterBandDefinition(
        20,
        "maximum_annual_temperature_c",
        "Maximum annual temperature (C)",
        (ERA5_DAILY_COLLECTION,),
    ),
    RasterBandDefinition(
        21,
        "mean_annual_temperature_c",
        "Mean annual temperature (C)",
        (ERA5_MONTHLY_COLLECTION,),
    ),
    RasterBandDefinition(
        22,
        "median_annual_temperature_c",
        "Median annual temperature (C)",
        (ERA5_MONTHLY_COLLECTION,),
    ),
    RasterBandDefinition(
        23,
        "minimum_annual_temperature_c",
        "Minimum annual temperature (C)",
        (ERA5_DAILY_COLLECTION,),
    ),
    RasterBandDefinition(
        24,
        "annual_precipitation_mm",
        "Annual precipitation (mm)",
        (ERA5_DAILY_COLLECTION,),
    ),
    RasterBandDefinition(
        25,
        "growing_season_avg_temp_c",
        "Growing-season average temperature (C)",
        (ERA5_DAILY_COLLECTION, MODIS_PHENOLOGY_COLLECTION),
    ),
    RasterBandDefinition(
        26,
        "growing_season_avg_precipita",
        "Growing-season average precipitation (mm/day)",
        (ERA5_DAILY_COLLECTION, MODIS_PHENOLOGY_COLLECTION),
    ),
    RasterBandDefinition(
        27,
        "interannual_rainfall_variabi",
        "Interannual rainfall variability (CV%, 10-year)",
        (ERA5_DAILY_COLLECTION,),
    ),
    RasterBandDefinition(
        28,
        "drought_mean_spi_30_day",
        "Drought mean (SPI 30-day)",
        (GRIDMET_DROUGHT_COLLECTION,),
    ),
    RasterBandDefinition(
        29,
        "drought_5th_percentile_spi_3",
        "Drought 5th percentile (SPI 30-day)",
        (GRIDMET_DROUGHT_COLLECTION,),
    ),
    RasterBandDefinition(
        30,
        "fire_frequency_burned_months",
        "Fire frequency (burned months)",
        (VIIRS_BURNED_AREA_COLLECTION,),
    ),
    RasterBandDefinition(
        31,
        "annual_variation_in_water_pr",
        "Annual variation in water presence",
        (JRC_MONTHLY_WATER_COLLECTION,),
    ),
    RasterBandDefinition(
        32,
        "distance_to_streams_m",
        "Distance to streams (m)",
        (MERIT_HYDRO_IMAGE,),
    ),
    RasterBandDefinition(
        33,
        "soil_organic_carbon_10_cm_g",
        "Soil organic carbon (10 cm, g/kg)",
        (ISRIC_SOIL_ORGANIC_CARBON_IMAGE,),
    ),
    RasterBandDefinition(
        34,
        "soil_moisture_annual_mean_gl",
        "Soil moisture annual mean (GLDAS 10-40 cm)",
        (GLDAS_COLLECTION,),
    ),
    RasterBandDefinition(
        35,
        "landform_type_srtm",
        "Landform type (SRTM)",
        (SRTM_LANDFORMS_IMAGE,),
    ),
    RasterBandDefinition(
        36,
        "topographic_diversity_alos",
        "Topographic diversity (ALOS)",
        (ALOS_TOPOGRAPHIC_DIVERSITY_IMAGE,),
    ),
    RasterBandDefinition(
        37,
        "annual_evapotranspiration_mo",
        "Annual evapotranspiration (MODIS ET, mm)",
        (MODIS_EVAPOTRANSPIRATION_COLLECTION,),
    ),
    RasterBandDefinition(
        38,
        "average_snow_depth_when_pres",
        "Average snow depth when present (GLDAS, m)",
        (GLDAS_COLLECTION,),
    ),
    RasterBandDefinition(
        39,
        "average_snow_depth_when_pres",
        "Average snow depth when present (SMAP, m)",
        (SMAP_COLLECTION,),
    ),
)


@dataclass(frozen=True)
class CacheGrid:
    """Define the globally aligned pixel and tile grid.

    Attributes:
        crs: Projected coordinate reference system used by every tile.
        pixel_size_meters: Width and height of one square output pixel.
        tile_size_pixels: Width and height of one square cache tile.
        origin_x: X coordinate to which tile columns are anchored.
        origin_y: Y coordinate to which tile rows are anchored.
    """

    crs: str = DEFAULT_CACHE_CRS
    pixel_size_meters: int = DEFAULT_PIXEL_SIZE_METERS
    tile_size_pixels: int = DEFAULT_TILE_SIZE_PIXELS
    origin_x: int = 0
    origin_y: int = 0


@dataclass(frozen=True)
class ReferenceThresholds:
    """Thresholds used to identify grassland reference sites.

    Attributes:
        grassland_probability: Minimum annual grassland probability percentage.
        human_modification: Maximum human-modification index.
        human_influence: Maximum scaled human-influence index.
    """

    grassland_probability: int = DEFAULT_GRASSLAND_PROBABILITY_THRESHOLD
    human_modification: float = DEFAULT_HMI_THRESHOLD
    human_influence: float = DEFAULT_HII_THRESHOLD


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
        AOI, Earth Engine, cache, threshold, request, and refresh settings.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Fetch the d01-d39 Earth Engine raster stack into deterministic "
            "64 km cache tiles intersecting a GeoJSON AOI."
        )
    )
    parser.add_argument(
        "aoi",
        type=Path,
        help="GeoJSON Polygon, MultiPolygon, Feature, or FeatureCollection in WGS84.",
    )
    parser.add_argument(
        "year",
        type=int,
        choices=range(MINIMUM_COMPLETE_STACK_YEAR, MAXIMUM_COMPLETE_STACK_YEAR + 1),
        metavar=f"{{{MINIMUM_COMPLETE_STACK_YEAR}..{MAXIMUM_COMPLETE_STACK_YEAR}}}",
        help="Data year for the complete d01-d39 stack.",
    )
    parser.add_argument(
        "--project",
        required=True,
        help="Google Cloud project registered for Earth Engine use.",
    )
    parser.add_argument(
        "--cache-directory",
        type=Path,
        default=DEFAULT_CACHE_DIRECTORY,
        help=f"Tile cache directory. Default: {DEFAULT_CACHE_DIRECTORY}.",
    )
    parser.add_argument(
        "--grassland-probability-threshold",
        type=int,
        default=DEFAULT_GRASSLAND_PROBABILITY_THRESHOLD,
        help=(
            "Minimum reference-site grassland probability percentage. "
            f"Default: {DEFAULT_GRASSLAND_PROBABILITY_THRESHOLD}."
        ),
    )
    parser.add_argument(
        "--hmi-threshold",
        type=float,
        default=DEFAULT_HMI_THRESHOLD,
        help=f"Maximum reference-site HMI. Default: {DEFAULT_HMI_THRESHOLD}.",
    )
    parser.add_argument(
        "--hii-threshold",
        type=float,
        default=DEFAULT_HII_THRESHOLD,
        help=f"Maximum reference-site HII. Default: {DEFAULT_HII_THRESHOLD}.",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Download every intersecting tile even when a valid cache entry exists.",
    )
    parser.add_argument(
        "--request-timeout-seconds",
        type=float,
        default=DEFAULT_REQUEST_TIMEOUT_SECONDS,
        help=(
            "Maximum wait for one Earth Engine request attempt. Default: "
            f"{DEFAULT_REQUEST_TIMEOUT_SECONDS} seconds."
        ),
    )
    parser.add_argument(
        "--request-retries",
        type=int,
        default=DEFAULT_REQUEST_RETRY_COUNT,
        help=(
            "Retries after a failed Earth Engine request attempt. Default: "
            f"{DEFAULT_REQUEST_RETRY_COUNT}."
        ),
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


def expected_band_names(year: int) -> tuple[str, ...]:
    """Return exact d01-d39 band names for one year.

    Args:
        year: Four-digit source-data year.

    Returns:
        Ordered Earth Engine export band names.
    """

    return tuple(
        f"y{year}_d{definition.number:02d}_{definition.suffix}"
        for definition in BAND_DEFINITIONS
    )


def build_stack_identifier(year: int, thresholds: ReferenceThresholds) -> str:
    """Build a stable cache identifier for one source configuration.

    Args:
        year: Four-digit source-data year.
        thresholds: Reference-site thresholds affecting d01.

    Returns:
        File-safe stack identifier.
    """

    human_modification_text = format(thresholds.human_modification, "g").replace(
        ".", "p"
    )
    human_influence_text = format(thresholds.human_influence, "g").replace(".", "p")
    return (
        f"v{STACK_DEFINITION_VERSION}_year_{year}_"
        f"gp_{thresholds.grassland_probability}_"
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
    year: int,
    thresholds: ReferenceThresholds,
) -> ee.Image:
    """Build the same d01-d39 image used by the Earth Engine export app.

    Args:
        year: Source-data year for annual response and environmental variables.
        thresholds: Reference-site grassland, HMI, and HII thresholds.

    Returns:
        Computed 39-band Earth Engine image with stable export names.
    """

    def annual_collection(dataset: str, selected_year: Any) -> ee.ImageCollection:
        start_date = ee.Date.fromYMD(ee.Number(selected_year).toInt(), 1, 1)
        return ee.ImageCollection(dataset).filterDate(
            start_date,
            start_date.advance(1, "year"),
        )

    def no_two_consecutive_zeros(
        annual_binary_builder: Callable[[Any], ee.Image],
    ) -> ee.Image:
        reference_years = ee.List.sequence(REFERENCE_START_YEAR, REFERENCE_END_YEAR)
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

    maybe_grassland_mask = (
        ee.Image()
        .byte()
        .paint(ee.FeatureCollection(MAYBE_GRASSLAND_ECOREGIONS), 1)
        .rename("maybe_grassland_ecoregion")
        .selfMask()
    )
    grassland_probability_collection = ee.ImageCollection(
        GRASSLAND_PROBABILITY_COLLECTION
    )
    human_influence_collection = ee.ImageCollection(
        HUMAN_INFLUENCE_COLLECTION
    ).filterDate("2001-01-01", "2021-01-01")
    grassland_probability_integrity = no_two_consecutive_zeros(
        lambda reference_year: ee.Image(
            grassland_probability_collection.filterDate(
                ee.Date.fromYMD(ee.Number(reference_year).toInt(), 1, 1),
                ee.Date.fromYMD(ee.Number(reference_year).add(1).toInt(), 1, 1),
            ).first()
        )
        .select(0)
        .gte(thresholds.grassland_probability)
    )
    human_influence_integrity = no_two_consecutive_zeros(
        lambda reference_year: human_influence_collection.filterDate(
            ee.Date.fromYMD(ee.Number(reference_year).toInt(), 1, 1),
            ee.Date.fromYMD(ee.Number(reference_year).add(1).toInt(), 1, 1),
        )
        .mean()
        .divide(7000)
        .lt(thresholds.human_influence)
    )
    reference_sites = (
        grassland_probability_integrity.And(human_influence_integrity)
        .And(ee.Image(HUMAN_MODIFICATION_IMAGE).lte(thresholds.human_modification))
        .And(maybe_grassland_mask)
        .selfMask()
        .toByte()
    )

    year_start = ee.Date.fromYMD(year, 1, 1)
    epoch_start = ee.Date("1970-01-01")
    phenology = ee.Image(annual_collection(MODIS_PHENOLOGY_COLLECTION, year).first())
    year_start_day = year_start.difference(epoch_start, "day")
    landsat_ndvi = annual_collection(LANDSAT_NDVI_COLLECTION, year).select("NDVI")
    vegetation_height = (
        ee.Image(annual_collection(SHORT_VEGETATION_HEIGHT_COLLECTION, year).first())
        .select("height")
        .multiply(0.1)
    )
    vegetation_cover = ee.Image(
        annual_collection(MODIS_VEGETATION_COVER_COLLECTION, year).first()
    )
    lai_fpar_collection = annual_collection(MODIS_LAI_FPAR_COLLECTION, year)
    productivity = ee.Image(
        annual_collection(MODIS_PRODUCTIVITY_COLLECTION, year).first()
    )

    response_images = [
        landsat_ndvi.reduce(ee.Reducer.percentile([95])),
        landsat_ndvi.reduce(ee.Reducer.percentile([50])),
        phenology.select("Senescence_1").subtract(
            phenology.select("Greenup_1")
        ),
        phenology.select("Senescence_2").subtract(
            phenology.select("Greenup_2")
        ),
        phenology.select("Greenup_1").subtract(year_start_day),
        phenology.select("Greenup_2").subtract(year_start_day),
        vegetation_height,
        vegetation_cover.select("Percent_Tree_Cover"),
        vegetation_cover.select("Percent_NonTree_Vegetation"),
        vegetation_cover.select("Percent_NonVegetated"),
        lai_fpar_collection.select("Lai_500m")
        .map(lambda image: image.multiply(0.1))
        .max(),
        lai_fpar_collection.select("Lai_500m")
        .map(lambda image: image.multiply(0.1))
        .reduce(ee.Reducer.stdDev()),
        lai_fpar_collection.select("Fpar_500m")
        .map(lambda image: image.multiply(0.01))
        .mean(),
        lai_fpar_collection.select("Fpar_500m")
        .map(lambda image: image.multiply(0.01))
        .reduce(ee.Reducer.stdDev()),
        lai_fpar_collection.select("FparStdDev_500m")
        .map(lambda image: image.multiply(0.01))
        .max(),
        phenology.select("NumCycles"),
        productivity.select("Npp").multiply(0.0001),
        productivity.select("Gpp").multiply(0.0001),
    ]

    era5_daily = annual_collection(ERA5_DAILY_COLLECTION, year)
    era5_monthly = annual_collection(ERA5_MONTHLY_COLLECTION, year)
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
        year - INTERANNUAL_RAINFALL_WINDOW_YEARS + 1,
        ERA5_START_YEAR,
    )
    annual_rainfall_totals = ee.ImageCollection.fromImages(
        ee.List.sequence(rainfall_start_year, year).map(
            lambda rainfall_year: annual_collection(
                ERA5_DAILY_COLLECTION,
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
    gridmet_drought = annual_collection(GRIDMET_DROUGHT_COLLECTION, year).select(
        "spi30d"
    )
    burned_month_count = (
        annual_collection(VIIRS_BURNED_AREA_COLLECTION, year)
        .select("Burn_Date")
        .map(lambda image: image.gt(0).unmask(0))
        .sum()
    )
    water_presence_variation = (
        annual_collection(JRC_MONTHLY_WATER_COLLECTION, year)
        .select("water")
        .map(lambda image: image.eq(2).updateMask(image.neq(0)))
        .reduce(ee.Reducer.stdDev())
    )
    streams = (
        ee.Image(MERIT_HYDRO_IMAGE)
        .select("upa")
        .gte(STREAM_UPSTREAM_AREA_THRESHOLD_KM2)
        .unmask(0)
    )
    distance_to_streams = (
        streams.fastDistanceTransform()
        .sqrt()
        .multiply(ee.Image.pixelArea().sqrt())
    )
    gldas_annual = annual_collection(GLDAS_COLLECTION, year)
    modis_evapotranspiration = (
        annual_collection(MODIS_EVAPOTRANSPIRATION_COLLECTION, year)
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
            ee.ImageCollection(SMAP_COLLECTION)
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
    environmental_images = [
        era5_daily.select("maximum_2m_air_temperature").max().subtract(273.15),
        era5_monthly.select("mean_2m_air_temperature").mean().subtract(273.15),
        era5_monthly.select("mean_2m_air_temperature").median().subtract(273.15),
        era5_daily.select("minimum_2m_air_temperature").min().subtract(273.15),
        annual_precipitation,
        growing_season_daily.select("mean_2m_air_temperature")
        .mean()
        .subtract(273.15),
        growing_season_daily.select("total_precipitation").mean().multiply(1000),
        rainfall_variability,
        gridmet_drought.mean(),
        gridmet_drought.reduce(ee.Reducer.percentile([5])),
        burned_month_count,
        water_presence_variation,
        distance_to_streams,
        ee.Image(ISRIC_SOIL_ORGANIC_CARBON_IMAGE)
        .select("soc_5-15cm_mean")
        .divide(10),
        gldas_annual.select("SoilMoi10_40cm_inst").mean(),
        ee.Image(SRTM_LANDFORMS_IMAGE).select("constant"),
        ee.Image(ALOS_TOPOGRAPHIC_DIVERSITY_IMAGE).select("constant"),
        modis_evapotranspiration,
        gldas_snow_depth,
        smap_snow_depth,
    ]

    layer_images = [reference_sites, *response_images, *environmental_images]
    renamed_layers = [
        ee.Image(layer_image).rename(band_name)
        for layer_image, band_name in zip(
            layer_images,
            expected_band_names(year),
            strict=True,
        )
    ]
    raster_stack = renamed_layers[0]
    for renamed_layer in renamed_layers[1:]:
        raster_stack = raster_stack.addBands(renamed_layer)
    return raster_stack.updateMask(maybe_grassland_mask).toFloat()


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
    aoi_path: Path,
    year: int,
    earth_engine_project: str,
    cache_directory: Path,
    thresholds: ReferenceThresholds,
    refresh: bool,
    show_progress: bool,
    request_timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
    request_retry_count: int = DEFAULT_REQUEST_RETRY_COUNT,
    compute_tile: (
        Callable[[ee.Image | None, CacheTile, CacheGrid, Sequence[str]], bytes]
        | None
    ) = None,
) -> FetchSummary:
    """Populate reusable Earth Engine cache tiles intersecting one AOI.

    Args:
        aoi_path: WGS84 GeoJSON AOI.
        year: Complete-stack source-data year.
        earth_engine_project: Cloud project registered for Earth Engine.
        cache_directory: Root directory for tiles and manifest metadata.
        thresholds: Reference-site thresholds used to construct d01.
        refresh: Whether valid existing tiles should be replaced.
        show_progress: Whether to display tqdm progress.
        request_timeout_seconds: Maximum socket wait for one request attempt.
        request_retry_count: Retry attempts after the initial request fails.
        compute_tile: Optional tile-fetch implementation used by offline tests.

    Returns:
        Counts and byte totals for requested, reused, downloaded, and failed tiles.

    Raises:
        ValueError: If request timeout or retry settings are invalid.
        RuntimeError: If Earth Engine initialization fails or a tile exhausts
            its request attempts.
    """

    if request_timeout_seconds <= 0:
        raise ValueError("request_timeout_seconds must be positive.")
    if not 0 <= request_retry_count < 100:
        raise ValueError("request_retry_count must be between 0 and 99.")
    resolved_cache_directory = cache_directory.expanduser().resolve()
    resolved_cache_directory.mkdir(parents=True, exist_ok=True)
    manifest_path = resolved_cache_directory / "manifest.json"
    cache_grid = CacheGrid()
    wgs84_aoi = load_wgs84_aoi(aoi_path)
    projected_aoi, requested_tiles = select_intersecting_tiles(
        wgs84_aoi,
        cache_grid,
    )
    band_names = expected_band_names(year)
    stack_identifier = build_stack_identifier(year, thresholds)
    manifest = load_cache_manifest(manifest_path, cache_grid)
    manifest["stacks"][stack_identifier] = {
        "definition_version": STACK_DEFINITION_VERSION,
        "year": year,
        "reference_thresholds": asdict(thresholds),
        "bands": [
            {
                **asdict(definition),
                "name": band_name,
            }
            for definition, band_name in zip(
                BAND_DEFINITIONS,
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
                project=earth_engine_project,
                http_transport=httplib2.Http(timeout=request_timeout_seconds),
            )
            ee.data.setDeadline(request_timeout_seconds * 1000)
        except Exception as error:
            raise RuntimeError(
                "Could not initialize or configure Earth Engine. Authenticate "
                "with `earthengine authenticate` and verify --project."
            ) from error
        raster_stack = build_earth_engine_stack(year, thresholds)

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
                    "year": year,
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
            f"{request_timestamp}|{aoi_path.expanduser().resolve()}|"
            f"{stack_identifier}"
        ).encode("utf-8")
    ).hexdigest()[:16]
    manifest["requests"].append(
        {
            "request_id": request_identifier,
            "requested_at_utc": request_timestamp,
            "aoi_path": str(aoi_path.expanduser().resolve()),
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
    print(f"  AOI: {aoi_path.expanduser().resolve()}")
    print(f"  AOI area: {projected_aoi.area / 1_000_000:,.2f} km2")
    print(f"  Data year: {year}")
    print(f"  Stack: {stack_identifier}")
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
    cache_aoi_tiles(
        args.aoi,
        args.year,
        args.project,
        args.cache_directory,
        ReferenceThresholds(
            grassland_probability=args.grassland_probability_threshold,
            human_modification=args.hmi_threshold,
            human_influence=args.hii_threshold,
        ),
        refresh=args.refresh,
        show_progress=not args.no_progress,
        request_timeout_seconds=args.request_timeout_seconds,
        request_retry_count=args.request_retries,
    )


if __name__ == "__main__":
    main()
