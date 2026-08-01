"""Shared addressing and validation for analysis raster-cache tiles."""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import rasterio
from pyproj import Transformer
from rasterio.crs import CRS
from rasterio.transform import Affine
from shapely.geometry import box, shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform, unary_union
from tqdm.auto import tqdm

from .analysis_config import AnalysisConfiguration, RasterCacheGrid


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
class CachedRasterTile:
    """Pair one expected cache-grid tile with its validated local file.

    Attributes:
        tile: Deterministic global grid address and bounds.
        path: Absolute path to the validated multiband GeoTIFF.
        file_size_bytes: Validated on-disk file size.
        sha256: Validated file checksum recorded by the manifest.
    """

    tile: CacheTile
    path: Path
    file_size_bytes: int
    sha256: str


@dataclass(frozen=True)
class AnalysisCacheTiles:
    """Validated cache inputs required by one configured AOI.

    Attributes:
        stack_identifier: Raster-effective cache namespace.
        manifest_path: Manifest used to resolve and validate files.
        wgs84_aoi: Configured AOI in longitude-latitude coordinates.
        projected_aoi: Configured AOI transformed to the cache CRS.
        tiles: Required cache tiles ordered north-to-south, west-to-east.
        total_file_size_bytes: Combined size of the validated tile files.
    """

    stack_identifier: str
    manifest_path: Path
    wgs84_aoi: BaseGeometry
    projected_aoi: BaseGeometry
    tiles: tuple[CachedRasterTile, ...]
    total_file_size_bytes: int


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
    geojson_object = json.loads(resolved_path.read_text(encoding="utf-8"))
    geojson_type = geojson_object.get("type")
    if geojson_type == "FeatureCollection":
        aoi_geometry = unary_union(
            [
                shape(feature["geometry"])
                for feature in geojson_object["features"]
            ]
        )
    elif geojson_type == "Feature":
        aoi_geometry = shape(geojson_object["geometry"])
    else:
        aoi_geometry = shape(geojson_object)

    if (
        aoi_geometry.is_empty
        or not aoi_geometry.is_valid
        or aoi_geometry.geom_type not in {"Polygon", "MultiPolygon"}
    ):
        raise ValueError("AOI must be a valid, nonempty Polygon or MultiPolygon.")
    return aoi_geometry


def select_intersecting_tiles(
    wgs84_aoi: BaseGeometry,
    cache_grid: RasterCacheGrid,
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
    tile_span_meters = (
        cache_grid.pixel_size_meters * cache_grid.tile_size_pixels
    )
    minimum_x, minimum_y, maximum_x, maximum_y = projected_aoi.bounds
    minimum_column = math.floor(
        (minimum_x - cache_grid.origin_x) / tile_span_meters
    )
    maximum_column = math.floor(
        (math.nextafter(maximum_x, -math.inf) - cache_grid.origin_x)
        / tile_span_meters
    )
    minimum_row = math.floor(
        (minimum_y - cache_grid.origin_y) / tile_span_meters
    )
    maximum_row = math.floor(
        (math.nextafter(maximum_y, -math.inf) - cache_grid.origin_y)
        / tile_span_meters
    )

    intersecting_tiles = []
    for row in range(maximum_row, minimum_row - 1, -1):
        for column in range(minimum_column, maximum_column + 1):
            tile_left = cache_grid.origin_x + column * tile_span_meters
            tile_bottom = cache_grid.origin_y + row * tile_span_meters
            tile_right = tile_left + tile_span_meters
            tile_top = tile_bottom + tile_span_meters
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

    reference_settings = analysis_configuration.reference
    human_modification_text = format(
        reference_settings.human_modification,
        "g",
    ).replace(".", "p")
    human_influence_text = format(
        reference_settings.human_influence,
        "g",
    ).replace(".", "p")
    return (
        f"{analysis_configuration.stack_name}_v"
        f"{analysis_configuration.stack_version}_"
        f"{analysis_configuration.raster_configuration_sha256[:12]}_"
        f"year_{analysis_configuration.year}_"
        f"gp_{reference_settings.grassland_probability}_"
        f"hmi_{human_modification_text}_hii_{human_influence_text}"
    )


def load_cache_manifest(
    manifest_path: Path,
    cache_grid: RasterCacheGrid,
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

    cache_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if cache_manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            "Cache manifest schema does not match the current script version."
        )
    if cache_manifest.get("grid") != expected_grid_metadata:
        raise ValueError(
            "Cache manifest grid differs from the current fixed cache grid."
        )
    return cache_manifest


def write_cache_manifest(
    manifest_path: Path,
    cache_manifest: dict[str, Any],
) -> None:
    """Atomically write the complete cache manifest.

    Args:
        manifest_path: Destination JSON path.
        cache_manifest: Complete serializable manifest.

    Returns:
        None: The manifest is written and atomically replaced.
    """

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = manifest_path.with_name(f"{manifest_path.name}.tmp")
    temporary_path.write_text(
        json.dumps(cache_manifest, indent=2, sort_keys=True) + "\n",
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
    cache_grid: RasterCacheGrid,
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
    with rasterio.open(tile_path) as cached_raster:
        if cached_raster.width != cache_grid.tile_size_pixels:
            raise ValueError("Cached tile width does not match the cache grid.")
        if cached_raster.height != cache_grid.tile_size_pixels:
            raise ValueError("Cached tile height does not match the cache grid.")
        if cached_raster.count != len(band_names):
            raise ValueError("Cached tile band count does not match the stack.")
        if cached_raster.crs != CRS.from_string(cache_grid.crs):
            raise ValueError("Cached tile CRS does not match the cache grid.")
        if not cached_raster.transform.almost_equals(expected_transform):
            raise ValueError("Cached tile transform does not match its tile address.")
        if tuple(cached_raster.descriptions) != tuple(band_names):
            raise ValueError("Cached tile band names do not match the stack schema.")

    actual_checksum = calculate_file_sha256(tile_path)
    if expected_checksum is not None and actual_checksum != expected_checksum:
        raise ValueError("Cached tile checksum differs from its manifest record.")
    return actual_file_size, actual_checksum


def resolve_analysis_cache_tiles(
    analysis_configuration: AnalysisConfiguration,
    show_progress: bool,
) -> AnalysisCacheTiles:
    """Resolve and validate every cached tile required by an analysis AOI.

    Args:
        analysis_configuration: Complete AOI, stack, grid, and cache contract.
        show_progress: Whether to display tile-validation progress.

    Returns:
        Validated tile paths and AOI geometries for downstream processing.

    Raises:
        RuntimeError: If any required cache tile is absent or invalid.
    """

    cache_directory = analysis_configuration.earth_engine.cache_directory
    manifest_path = cache_directory / "manifest.json"
    cache_grid = analysis_configuration.grid
    wgs84_aoi = load_wgs84_aoi(analysis_configuration.aoi_path)
    projected_aoi, required_tiles = select_intersecting_tiles(
        wgs84_aoi,
        cache_grid,
    )
    stack_identifier = build_stack_identifier(analysis_configuration)
    cache_manifest = load_cache_manifest(manifest_path, cache_grid)
    expected_band_names = analysis_configuration.band_names()
    validation_failures = []
    validated_tiles = []

    if stack_identifier not in cache_manifest["stacks"]:
        validation_failures.append(
            f"stack namespace {stack_identifier} is absent from the manifest"
        )

    for cache_tile in tqdm(
        required_tiles,
        desc="Validating cached raster tiles",
        unit="tile",
        dynamic_ncols=True,
        disable=not show_progress,
    ):
        tile_record_key = f"{stack_identifier}/{cache_tile.tile_id}"
        tile_record = cache_manifest["tiles"].get(tile_record_key)
        if tile_record is None:
            validation_failures.append(
                f"{cache_tile.tile_id}: no manifest record"
            )
            continue
        tile_path = (
            cache_directory / str(tile_record["relative_path"])
        ).resolve()
        try:
            file_size_bytes, sha256 = validate_cached_tile(
                tile_path,
                cache_tile,
                cache_grid,
                expected_band_names,
                expected_checksum=str(tile_record["sha256"]),
                expected_file_size=int(tile_record["file_size_bytes"]),
            )
        except (OSError, ValueError, rasterio.errors.RasterioError) as error:
            validation_failures.append(f"{cache_tile.tile_id}: {error}")
            continue
        validated_tiles.append(
            CachedRasterTile(
                tile=cache_tile,
                path=tile_path,
                file_size_bytes=file_size_bytes,
                sha256=sha256,
            )
        )

    if validation_failures:
        displayed_failures = "; ".join(validation_failures[:5])
        omitted_failure_count = len(validation_failures) - 5
        omitted_failure_text = (
            f"; plus {omitted_failure_count:,} more"
            if omitted_failure_count > 0
            else ""
        )
        raise RuntimeError(
            f"{len(validation_failures):,} required cache tile(s) are missing or "
            f"invalid: {displayed_failures}{omitted_failure_text}. Run "
            "`python -m scripts.fetch_gee_raster_tiles "
            f"{analysis_configuration.path}` and rerun sampling."
        )

    return AnalysisCacheTiles(
        stack_identifier=stack_identifier,
        manifest_path=manifest_path,
        wgs84_aoi=wgs84_aoi,
        projected_aoi=projected_aoi,
        tiles=tuple(validated_tiles),
        total_file_size_bytes=sum(
            tile.file_size_bytes for tile in validated_tiles
        ),
    )
