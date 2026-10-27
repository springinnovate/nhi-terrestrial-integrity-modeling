"""Fetch deterministic Earth Engine raster-stack tiles for an AOI."""

from __future__ import annotations

import argparse
import hashlib
import os
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable, Sequence

import ee
import httplib2
import rasterio
from rasterio.crs import CRS
from tqdm.auto import tqdm

from .analysis_config import (
    DEFAULT_ANALYSIS_CONFIG_PATH,
    AnalysisConfiguration,
    RasterCacheGrid,
    load_analysis_configuration,
)
from .earth_engine_stack import build_earth_engine_stack
from .raster_cache_utils import (
    CacheTile,
    build_stack_identifier,
    load_cache_manifest,
    load_wgs84_aoi,
    select_intersecting_tiles,
    validate_cached_tile,
    write_cache_manifest,
)


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


def fetch_tile_bytes(
    raster_stack: ee.Image,
    tile: CacheTile,
    cache_grid: RasterCacheGrid,
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

    compute_pixels_request = {
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
    return ee.data.computePixels(compute_pixels_request)


def cache_aoi_tiles(
    analysis_configuration: AnalysisConfiguration,
    refresh: bool,
    show_progress: bool,
    tile_fetcher: (
        Callable[
            [ee.Image | None, CacheTile, RasterCacheGrid, Sequence[str]],
            bytes,
        ]
        | None
    ) = None,
) -> FetchSummary:
    """Populate reusable Earth Engine cache tiles intersecting one AOI.

    Args:
        analysis_configuration: Authoritative AOI, year, Earth Engine, cache,
            reference threshold, grid, and raster-stack settings.
        refresh: Whether valid existing tiles should be replaced.
        show_progress: Whether to display tqdm progress.
        tile_fetcher: Optional tile-fetch implementation used by offline tests.

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
    cache_grid = analysis_configuration.grid
    wgs84_aoi = load_wgs84_aoi(analysis_configuration.aoi_path)
    projected_aoi, requested_tiles = select_intersecting_tiles(
        wgs84_aoi,
        cache_grid,
    )
    band_names = analysis_configuration.band_names()
    stack_identifier = build_stack_identifier(analysis_configuration)
    cache_manifest = load_cache_manifest(manifest_path, cache_grid)
    cache_manifest["stacks"][stack_identifier] = {
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
    write_cache_manifest(manifest_path, cache_manifest)

    valid_cached_tile_ids: set[str] = set()
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
            tile_record = cache_manifest["tiles"].get(tile_record_key)
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
                    valid_cached_tile_ids.add(tile.tile_id)
                    cache_progress.set_postfix(
                        cached=len(valid_cached_tile_ids),
                        download=len(tiles_requiring_download),
                        refresh=refresh,
                    )
                    continue
                except (OSError, ValueError, rasterio.errors.RasterioError):
                    pass
            cache_manifest["tiles"].pop(tile_record_key, None)
            tiles_requiring_download.append(tile)
            cache_progress.set_postfix(
                cached=len(valid_cached_tile_ids),
                download=len(tiles_requiring_download),
                refresh=refresh,
            )
    write_cache_manifest(manifest_path, cache_manifest)

    print()
    print("Earth Engine grid plan")
    print(f"  Intersecting grids: {len(requested_tiles):,}")
    print(f"  Valid cached grids: {len(valid_cached_tile_ids):,}")
    print(f"  Grids to download: {len(tiles_requiring_download):,}")
    print(
        "  Request policy: "
        f"{request_timeout_seconds:g} second timeout, "
        f"{request_retry_count + 1} attempt(s) per grid"
    )

    raster_stack = None
    if tiles_requiring_download and tile_fetcher is None:
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
    if tile_fetcher is None:
        tile_fetcher = fetch_tile_bytes

    downloaded_tile_count = 0
    downloaded_byte_count = 0
    failed_tile_ids = []
    failed_tile_error: Exception | None = None
    tile_directory.mkdir(parents=True, exist_ok=True)
    with tqdm(
        total=len(requested_tiles),
        initial=len(valid_cached_tile_ids),
        desc="Processing grids",
        unit="grid",
        dynamic_ncols=True,
        disable=not show_progress,
    ) as processing_progress:
        processing_progress.set_postfix(
            processed=len(valid_cached_tile_ids),
            cached=len(valid_cached_tile_ids),
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
                cache_manifest["tiles"][tile_record_key] = {
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
                write_cache_manifest(manifest_path, cache_manifest)
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
                    len(valid_cached_tile_ids)
                    + downloaded_tile_count
                    + len(failed_tile_ids)
                )
                processing_progress.update(1)
                processing_progress.set_postfix(
                    processed=processed_tile_count,
                    cached=len(valid_cached_tile_ids),
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
    cache_manifest["requests"].append(
        {
            "request_id": request_identifier,
            "requested_at_utc": request_timestamp,
            "analysis_name": analysis_configuration.analysis_name,
            "aoi_path": str(analysis_configuration.aoi_path),
            "aoi_bounds_wgs84": list(wgs84_aoi.bounds),
            "aoi_area_m2": projected_aoi.area,
            "stack_id": stack_identifier,
            "tile_ids": [tile.tile_id for tile in requested_tiles],
            "reused_tile_ids": sorted(valid_cached_tile_ids),
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
    write_cache_manifest(manifest_path, cache_manifest)

    fetch_summary = FetchSummary(
        requested_tiles=len(requested_tiles),
        reused_tiles=len(valid_cached_tile_ids),
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
    print(f"  Requested tiles: {fetch_summary.requested_tiles:,}")
    print(f"  Reused from cache: {fetch_summary.reused_tiles:,}")
    print(f"  Downloaded: {fetch_summary.downloaded_tiles:,}")
    print(
        "  Downloaded size: "
        f"{fetch_summary.downloaded_bytes / 1_048_576:,.2f} MiB"
    )
    print(f"  Failed: {fetch_summary.failed_tiles:,}")
    print(f"  Manifest: {manifest_path}")

    if failed_tile_ids:
        raise RuntimeError(
            f"Earth Engine failed to return tile {failed_tile_ids[0]} after "
            f"{request_retry_count + 1} attempt(s) with a "
            f"{request_timeout_seconds:g} second timeout per attempt. "
            "Completed tiles remain cached; rerun the same command to resume."
        ) from failed_tile_error
    return fetch_summary


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
