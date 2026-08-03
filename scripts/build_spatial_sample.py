"""Build a spatially balanced sample from validated raster-cache tiles."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import textwrap
import time
from dataclasses import dataclass, field
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import rasterio
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from pyproj import Transformer
from rasterio.coords import BoundingBox
from rasterio.features import geometry_mask
from shapely.geometry import box, mapping
from shapely.geometry.base import BaseGeometry
from tqdm.auto import tqdm

from .analysis_config import (
    DEFAULT_ANALYSIS_CONFIG_PATH,
    AnalysisConfiguration,
    load_analysis_configuration,
)
from .raster_cache_utils import (
    AnalysisCacheTiles,
    CachedRasterTile,
    resolve_analysis_cache_tiles,
)
from .reference_condition_utils import EQUAL_AREA_CRS


MEBIBYTE = 1024**2
LOCATION_FIGURE_DPI = 300
SUPPORTED_FIGURE_SUFFIXES = {".pdf", ".png", ".svg"}
# Arrow schema metadata uses byte-string keys. This key stores the analysis,
# cache, and sampling provenance that is verified after the Parquet write.
PARQUET_PROVENANCE_KEY = b"nhi_spatial_sample_provenance"
# The d01 reference label is binary: background pixels are 0 and reference
# pixels are 1. Sampling and weight summaries treat those classes separately.
BACKGROUND_SITE_CLASS = 0
REFERENCE_SITE_CLASS = 1
REFERENCE_SITE_CLASSES = (BACKGROUND_SITE_CLASS, REFERENCE_SITE_CLASS)


@dataclass(frozen=True)
class SamplingCandidate:
    """Identify one retained source pixel before its band values are reread.

    Attributes:
        tile_sequence: Position of the source tile in deterministic AOI order.
        tile_id: Stable global cache-tile identifier.
        local_pixel_index: Row-major pixel position inside the cache tile.
        sampling_block_column: Global equal-area sampling-block column.
        sampling_block_row: Global equal-area sampling-block row.
        reference_site_class: Zero for background or one for reference.
        random_priority: Seeded priority used for sampling without replacement.
    """

    tile_sequence: int
    tile_id: str
    local_pixel_index: int
    sampling_block_column: int
    sampling_block_row: int
    reference_site_class: int
    random_priority: float


@dataclass
class SamplingStratumState:
    """Accumulate one global sampling-block and reference-class stratum.

    Attributes:
        available_pixel_count: Eligible source pixels encountered in the stratum.
        available_area_m2: Eligible source area encountered in the stratum.
        candidates: Lowest-priority pixels retained up to the configured cap.
    """

    available_pixel_count: int = 0
    available_area_m2: float = 0.0
    candidates: list[SamplingCandidate] = field(default_factory=list)


@dataclass(frozen=True)
class RasterBandScanSummary:
    """Summarize one configured band while cache tiles are streamed.

    Attributes:
        index: One-based physical band index.
        name: Configured stable raster-band name.
        defined_pixel_count: AOI pixels with finite, unmasked values.
        defined_percent: Percentage of AOI pixels with defined values.
        defined_area_km2: Approximate full-pixel represented area.
        minimum: Minimum defined value, or ``None`` for an empty band.
        mean: Mean defined value, or ``None`` for an empty band.
        maximum: Maximum defined value, or ``None`` for an empty band.
    """

    index: int
    name: str
    defined_pixel_count: int
    defined_percent: float
    defined_area_km2: float
    minimum: float | None
    mean: float | None
    maximum: float | None


@dataclass(frozen=True)
class RasterCacheScanSummary:
    """Describe source coverage accumulated without an AOI-sized array.

    Attributes:
        cache_tile_count: Number of validated tiles scanned.
        cache_file_size_bytes: Combined compressed tile size.
        aoi_pixel_count: Cache-grid pixel centers inside the AOI.
        any_band_defined_pixel_count: AOI pixels defined in at least one band.
        every_band_defined_pixel_count: AOI pixels defined in every band.
        eligible_pixel_count: AOI pixels defined in a non-reference band.
        excluded_reference_pixel_count: Reference pixels outside that footprint.
        pixel_area_m2: Constant configured equal-area grid-cell area.
        peak_tile_array_bytes: Largest value, validity, and AOI-mask allocation.
        band_summaries: Ordered per-band coverage and value summaries.
    """

    cache_tile_count: int
    cache_file_size_bytes: int
    aoi_pixel_count: int
    any_band_defined_pixel_count: int
    every_band_defined_pixel_count: int
    eligible_pixel_count: int
    excluded_reference_pixel_count: int
    pixel_area_m2: float
    peak_tile_array_bytes: int
    band_summaries: tuple[RasterBandScanSummary, ...]


@dataclass(frozen=True)
class SamplingClassSummary:
    """Summarize one binary reference-site class.

    Attributes:
        reference_site_class: Zero for background or one for reference.
        available_pixels: Eligible source pixels before sampling.
        sampled_pixels: Pixels retained in the Parquet table.
        available_area_m2: Eligible source area represented by the class.
        weighted_pixels: Source count reconstructed from sampling weights.
        weighted_area_m2: Source area reconstructed from area weights.
        blocks_with_class: Sampling blocks containing the class.
        minimum_sampling_weight: Smallest stratum weight in the class.
        maximum_sampling_weight: Largest stratum weight in the class.
    """

    reference_site_class: int
    available_pixels: int
    sampled_pixels: int
    available_area_m2: float
    weighted_pixels: float
    weighted_area_m2: float
    blocks_with_class: int
    minimum_sampling_weight: float
    maximum_sampling_weight: float


@dataclass(frozen=True)
class SpatialSample:
    """Model-ready rows and diagnostics from streamed cache tiles.

    Attributes:
        table: Selected pixels, weights, coordinates, and raster values.
        reference_band_name: Configured band used to identify reference sites.
        sampled_band_names: Non-reference bands written to the table.
        sampled_band_defined_pixel_counts: Defined sampled values by band.
        complete_sampled_band_row_count: Rows defined in every sampled band.
        block_size_meters: Configured global sampling-block width and height.
        samples_per_class_per_block: Configured cap for each block and class.
        random_seed: Seed used to assign deterministic pixel priorities.
        block_count: Global sampling blocks containing eligible AOI pixels.
        minimum_available_pixels_per_block: Smallest eligible block population.
        median_available_pixels_per_block: Median eligible block population.
        maximum_available_pixels_per_block: Largest eligible block population.
        class_summaries: Background and reference sampling diagnostics.
        elapsed_seconds: Time spent scanning and assembling sample rows.
    """

    table: pd.DataFrame
    reference_band_name: str
    sampled_band_names: tuple[str, ...]
    sampled_band_defined_pixel_counts: tuple[int, ...]
    complete_sampled_band_row_count: int
    block_size_meters: int
    samples_per_class_per_block: int
    random_seed: int
    block_count: int
    minimum_available_pixels_per_block: int
    median_available_pixels_per_block: float
    maximum_available_pixels_per_block: int
    class_summaries: tuple[SamplingClassSummary, SamplingClassSummary]
    elapsed_seconds: float


@dataclass(frozen=True)
class SamplingScan:
    """First-pass strata and coverage required to assemble sample rows.

    Attributes:
        strata: Sampling state keyed by block column, block row, and class.
        raster_summary: Incremental AOI and band statistics.
        reference_band_offset: Zero-based reference-band position.
        sampled_band_offsets: Zero-based non-reference band positions.
        sampled_band_names: Names corresponding to sampled-band positions.
    """

    strata: dict[tuple[int, int, int], SamplingStratumState]
    raster_summary: RasterCacheScanSummary
    reference_band_offset: int
    sampled_band_offsets: tuple[int, ...]
    sampled_band_names: tuple[str, ...]


@dataclass(frozen=True)
class ParquetWriteSummary:
    """Verified metadata for one compressed Parquet sample.

    Attributes:
        path: Absolute output path.
        rows: Verified Parquet row count.
        columns: Verified Parquet column count.
        row_groups: Number of written row groups.
        compression: Compression codec reported for the first column.
        file_size_bytes: Compressed file size.
        elapsed_seconds: Time spent writing and verifying the file.
    """

    path: Path
    rows: int
    columns: int
    row_groups: int
    compression: str
    file_size_bytes: int
    elapsed_seconds: float


@dataclass(frozen=True)
class LocationFigureSummary:
    """Describe a generated world locator figure.

    Attributes:
        path: Absolute saved figure path.
        analysis_name: Human-readable AOI label.
        bounds: AOI bounds in longitude and latitude.
        land_basemap_available: Whether Natural Earth land rendered successfully.
    """

    path: Path
    analysis_name: str
    bounds: BoundingBox
    land_basemap_available: bool


def parse_args() -> argparse.Namespace:
    """Parse the config-only spatial-sampling command.

    Returns:
        Analysis configuration and operational output/progress controls.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Build a spatially balanced Parquet sample by streaming the "
            "validated Earth Engine cache tiles defined by an analysis TOML."
        )
    )
    parser.add_argument(
        "analysis_configuration",
        type=Path,
        help=(
            "Complete TOML analysis definition containing the AOI, cache, "
            "raster schema, and sampling settings. Example: "
            f"{DEFAULT_ANALYSIS_CONFIG_PATH}."
        ),
    )
    parser.add_argument(
        "--sample-output",
        type=Path,
        help=(
            "Destination .parquet path. Defaults to "
            "outputs/samples/<analysis-name>_spatial_sample.parquet."
        ),
    )
    parser.add_argument(
        "--location-figure",
        type=Path,
        help=(
            "Destination .png, .pdf, or .svg locator map. Defaults to "
            "outputs/figures/<analysis-name>_world_location.png."
        ),
    )
    parser.add_argument(
        "--no-location-figure",
        action="store_true",
        help="Skip the world locator map.",
    )
    parser.add_argument(
        "--no-band-report",
        action="store_true",
        help="Skip the per-band source coverage and value table.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Suppress tqdm progress bars.",
    )
    return parser.parse_args()


def _read_tile_values_and_validity(
    cached_tile: CachedRasterTile,
) -> tuple[np.ndarray, np.ndarray, rasterio.Affine, int]:
    """Read one validated tile and preserve finite per-band validity.

    Args:
        cached_tile: Validated cache tile to read.

    Returns:
        Values, validity flags, affine transform, and allocated array bytes.
    """

    with rasterio.open(cached_tile.path) as source:
        masked_values = source.read(masked=True)
        tile_values = np.asarray(np.ma.getdata(masked_values))
        source_mask = np.ma.getmaskarray(masked_values)
        tile_validity = ~source_mask
        finite_value_mask = np.isfinite(tile_values)
        tile_validity &= finite_value_mask
        # During validity construction, the source mask, its inverse, and the
        # finite-value mask coexist with the tile values. Report that peak rather
        # than only the arrays returned to the caller.
        allocated_bytes = (
            tile_values.nbytes
            + source_mask.nbytes
            + tile_validity.nbytes
            + finite_value_mask.nbytes
        )
        return tile_values, tile_validity, source.transform, allocated_bytes


def scan_cached_tiles(
    analysis_configuration: AnalysisConfiguration,
    analysis_cache_tiles: AnalysisCacheTiles,
    show_progress: bool,
) -> SamplingScan:
    """Scan cache tiles and retain deterministic candidates per global stratum.

    Every source pixel receives a repeatable random priority based on the configured
    seed and its cache-tile address. Keeping the lowest priorities makes each
    block/class sample uniform without replacement and lets candidates from blocks
    spanning multiple cache tiles merge correctly.

    Args:
        analysis_configuration: Complete raster and sampling contract.
        analysis_cache_tiles: Validated AOI tile files and geometry.
        show_progress: Whether to display tile-processing progress.

    Returns:
        Sampling strata, retained candidate locations, and source statistics.

    Raises:
        RuntimeError: If no AOI pixel has a defined non-reference raster value.
    """

    configured_band_names = analysis_configuration.band_names()
    reference_band_offset = next(
        band_offset
        for band_offset, band_definition in enumerate(
            analysis_configuration.bands
        )
        if band_definition.role == "reference"
    )
    sampled_band_offsets = tuple(
        band_offset
        for band_offset in range(len(configured_band_names))
        if band_offset != reference_band_offset
    )
    sampled_band_names = tuple(
        configured_band_names[band_offset]
        for band_offset in sampled_band_offsets
    )
    sampling_settings = analysis_configuration.sampling
    cache_grid = analysis_configuration.grid
    cache_to_equal_area = Transformer.from_crs(
        cache_grid.crs,
        EQUAL_AREA_CRS,
        always_xy=True,
    )
    pixel_area_m2 = float(cache_grid.pixel_size_meters**2)

    band_count = len(configured_band_names)
    band_defined_pixel_counts = np.zeros(band_count, dtype=np.int64)
    band_value_sums = np.zeros(band_count, dtype=np.float64)
    band_minimum_values = np.full(band_count, np.inf, dtype=np.float64)
    band_maximum_values = np.full(band_count, -np.inf, dtype=np.float64)
    strata: dict[tuple[int, int, int], SamplingStratumState] = {}
    aoi_pixel_count = 0
    any_band_defined_pixel_count = 0
    every_band_defined_pixel_count = 0
    eligible_pixel_count = 0
    excluded_reference_pixel_count = 0
    retained_candidate_count = 0
    retained_background_candidate_count = 0
    retained_reference_candidate_count = 0
    peak_tile_array_bytes = 0

    with tqdm(
        enumerate(analysis_cache_tiles.tiles),
        total=len(analysis_cache_tiles.tiles),
        desc="Scanning cached raster tiles",
        unit="tile",
        dynamic_ncols=True,
        disable=not show_progress,
    ) as tile_progress:
        for tile_sequence, cached_tile in tile_progress:
            (
                tile_values,
                tile_validity,
                tile_transform,
                tile_array_bytes,
            ) = _read_tile_values_and_validity(cached_tile)
            tile_aoi_geometry = analysis_cache_tiles.projected_aoi.intersection(
                box(
                    cached_tile.tile.left,
                    cached_tile.tile.bottom,
                    cached_tile.tile.right,
                    cached_tile.tile.top,
                )
            )
            aoi_pixel_mask = geometry_mask(
                [mapping(tile_aoi_geometry)],
                out_shape=tile_validity.shape[1:],
                transform=tile_transform,
                invert=True,
            )
            peak_tile_array_bytes = max(
                peak_tile_array_bytes,
                tile_array_bytes + aoi_pixel_mask.nbytes,
            )
            tile_aoi_pixel_count = int(np.count_nonzero(aoi_pixel_mask))
            if tile_aoi_pixel_count == 0:
                continue
            aoi_pixel_count += tile_aoi_pixel_count
            any_band_defined_pixel_count += int(
                np.count_nonzero(aoi_pixel_mask & np.any(tile_validity, axis=0))
            )
            every_band_defined_pixel_count += int(
                np.count_nonzero(aoi_pixel_mask & np.all(tile_validity, axis=0))
            )

            for band_offset in range(band_count):
                defined_pixel_mask = (
                    aoi_pixel_mask & tile_validity[band_offset]
                )
                defined_values = tile_values[band_offset][defined_pixel_mask]
                defined_pixel_count = int(defined_values.size)
                if defined_pixel_count == 0:
                    continue
                band_defined_pixel_counts[band_offset] += defined_pixel_count
                band_value_sums[band_offset] += float(
                    np.sum(defined_values, dtype=np.float64)
                )
                band_minimum_values[band_offset] = min(
                    band_minimum_values[band_offset],
                    float(np.min(defined_values)),
                )
                band_maximum_values[band_offset] = max(
                    band_maximum_values[band_offset],
                    float(np.max(defined_values)),
                )

            sampled_data_footprint = np.zeros(
                aoi_pixel_mask.shape,
                dtype=np.bool_,
            )
            for sampled_band_offset in sampled_band_offsets:
                sampled_data_footprint |= tile_validity[
                    sampled_band_offset
                ]
            sampled_data_footprint &= aoi_pixel_mask
            reference_site_mask = (
                aoi_pixel_mask
                & tile_validity[reference_band_offset]
                & (tile_values[reference_band_offset] == 1)
            )
            excluded_reference_pixel_count += int(
                np.count_nonzero(reference_site_mask & ~sampled_data_footprint)
            )
            eligible_local_pixel_indices = np.flatnonzero(
                sampled_data_footprint.ravel()
            )
            tile_eligible_pixel_count = int(
                eligible_local_pixel_indices.size
            )
            eligible_pixel_count += tile_eligible_pixel_count
            if tile_eligible_pixel_count == 0:
                continue

            tile_height, tile_width = sampled_data_footprint.shape
            local_rows = eligible_local_pixel_indices // tile_width
            local_columns = eligible_local_pixel_indices % tile_width
            projected_x = (
                tile_transform.c
                + (local_columns.astype(np.float64) + 0.5) * tile_transform.a
                + (local_rows.astype(np.float64) + 0.5) * tile_transform.b
            )
            projected_y = (
                tile_transform.f
                + (local_columns.astype(np.float64) + 0.5) * tile_transform.d
                + (local_rows.astype(np.float64) + 0.5) * tile_transform.e
            )
            equal_area_x, equal_area_y = cache_to_equal_area.transform(
                projected_x,
                projected_y,
            )
            sampling_block_columns = np.floor(
                np.asarray(equal_area_x)
                / sampling_settings.block_size_meters
            ).astype(np.int64)
            sampling_block_rows = np.floor(
                np.asarray(equal_area_y)
                / sampling_settings.block_size_meters
            ).astype(np.int64)
            reference_site_classes = reference_site_mask.ravel()[
                eligible_local_pixel_indices
            ].astype(np.uint8)

            # SHA-256 converts the signed tile address and user seed into a
            # stable NumPy seed without relying on process-randomized hash().
            tile_seed_material = (
                f"{sampling_settings.random_seed}:{cached_tile.tile.tile_id}"
            ).encode("utf-8")
            tile_seed = int.from_bytes(
                hashlib.sha256(tile_seed_material).digest()[:8],
                byteorder="little",
                signed=False,
            )
            tile_random_generator = np.random.default_rng(tile_seed)
            all_tile_priorities = tile_random_generator.random(
                tile_height * tile_width
            )
            eligible_priorities = all_tile_priorities[
                eligible_local_pixel_indices
            ]

            stratum_records = np.empty(
                tile_eligible_pixel_count,
                dtype=[
                    ("block_column", np.int64),
                    ("block_row", np.int64),
                    ("reference_class", np.uint8),
                ],
            )
            stratum_records["block_column"] = sampling_block_columns
            stratum_records["block_row"] = sampling_block_rows
            stratum_records["reference_class"] = reference_site_classes
            unique_strata, stratum_offsets = np.unique(
                stratum_records,
                return_inverse=True,
            )
            positions_sorted_by_stratum = np.argsort(
                stratum_offsets,
                kind="stable",
            )
            pixels_per_tile_stratum = np.bincount(
                stratum_offsets,
                minlength=len(unique_strata),
            )
            stratum_start_offset = 0
            for stratum_offset, unique_stratum in enumerate(unique_strata):
                available_in_tile = int(
                    pixels_per_tile_stratum[stratum_offset]
                )
                positions_in_stratum = positions_sorted_by_stratum[
                    stratum_start_offset : (
                        stratum_start_offset + available_in_tile
                    )
                ]
                stratum_start_offset += available_in_tile
                stratum_key = (
                    int(unique_stratum["block_column"]),
                    int(unique_stratum["block_row"]),
                    int(unique_stratum["reference_class"]),
                )
                stratum_state = strata.setdefault(
                    stratum_key,
                    SamplingStratumState(),
                )
                stratum_state.available_pixel_count += available_in_tile
                stratum_state.available_area_m2 += (
                    available_in_tile * pixel_area_m2
                )
                retained_from_tile = min(
                    sampling_settings.samples_per_class_per_block,
                    available_in_tile,
                )
                if retained_from_tile < available_in_tile:
                    local_priority_offsets = np.argpartition(
                        eligible_priorities[positions_in_stratum],
                        retained_from_tile - 1,
                    )[:retained_from_tile]
                    retained_positions = positions_in_stratum[
                        local_priority_offsets
                    ]
                else:
                    retained_positions = positions_in_stratum
                new_candidates = [
                    SamplingCandidate(
                        tile_sequence=tile_sequence,
                        tile_id=cached_tile.tile.tile_id,
                        local_pixel_index=int(
                            eligible_local_pixel_indices[position]
                        ),
                        sampling_block_column=stratum_key[0],
                        sampling_block_row=stratum_key[1],
                        reference_site_class=stratum_key[2],
                        random_priority=float(eligible_priorities[position]),
                    )
                    for position in retained_positions
                ]
                previous_candidate_count = len(stratum_state.candidates)
                stratum_state.candidates = sorted(
                    [*stratum_state.candidates, *new_candidates],
                    key=lambda candidate: (
                        candidate.random_priority,
                        candidate.tile_sequence,
                        candidate.local_pixel_index,
                    ),
                )[: sampling_settings.samples_per_class_per_block]
                retained_candidate_delta = (
                    len(stratum_state.candidates) - previous_candidate_count
                )
                retained_candidate_count += retained_candidate_delta
                if stratum_key[2] == 1:
                    retained_reference_candidate_count += retained_candidate_delta
                else:
                    retained_background_candidate_count += retained_candidate_delta

            tile_progress.set_postfix(
                aoi=f"{aoi_pixel_count:,}",
                background=f"{retained_background_candidate_count:,}",
                eligible=f"{eligible_pixel_count:,}",
                reference=f"{retained_reference_candidate_count:,}",
                retained=f"{retained_candidate_count:,}",
                strata=f"{len(strata):,}",
            )

    if eligible_pixel_count == 0:
        raise RuntimeError(
            "No AOI pixels contain a defined non-reference raster value."
        )

    band_summaries = []
    for band_offset, band_name in enumerate(configured_band_names):
        defined_pixel_count = int(band_defined_pixel_counts[band_offset])
        band_summaries.append(
            RasterBandScanSummary(
                index=band_offset + 1,
                name=band_name,
                defined_pixel_count=defined_pixel_count,
                defined_percent=(
                    100.0 * defined_pixel_count / aoi_pixel_count
                ),
                defined_area_km2=(
                    defined_pixel_count * pixel_area_m2 / 1_000_000.0
                ),
                minimum=(
                    float(band_minimum_values[band_offset])
                    if defined_pixel_count
                    else None
                ),
                mean=(
                    float(
                        band_value_sums[band_offset]
                        / defined_pixel_count
                    )
                    if defined_pixel_count
                    else None
                ),
                maximum=(
                    float(band_maximum_values[band_offset])
                    if defined_pixel_count
                    else None
                ),
            )
        )

    return SamplingScan(
        strata=strata,
        raster_summary=RasterCacheScanSummary(
            cache_tile_count=len(analysis_cache_tiles.tiles),
            cache_file_size_bytes=(
                analysis_cache_tiles.total_file_size_bytes
            ),
            aoi_pixel_count=aoi_pixel_count,
            any_band_defined_pixel_count=any_band_defined_pixel_count,
            every_band_defined_pixel_count=every_band_defined_pixel_count,
            eligible_pixel_count=eligible_pixel_count,
            excluded_reference_pixel_count=excluded_reference_pixel_count,
            pixel_area_m2=pixel_area_m2,
            peak_tile_array_bytes=peak_tile_array_bytes,
            band_summaries=tuple(band_summaries),
        ),
        reference_band_offset=reference_band_offset,
        sampled_band_offsets=sampled_band_offsets,
        sampled_band_names=sampled_band_names,
    )


def assemble_spatial_sample(
    analysis_configuration: AnalysisConfiguration,
    analysis_cache_tiles: AnalysisCacheTiles,
    sampling_scan: SamplingScan,
    started_at: float,
    show_progress: bool,
) -> SpatialSample:
    """Reread selected tiles and assemble final model-ready sample columns.

    Args:
        analysis_configuration: Complete raster and sampling contract.
        analysis_cache_tiles: Validated AOI cache tiles.
        sampling_scan: First-pass strata and candidate locations.
        started_at: Performance-counter value from the start of sampling.
        show_progress: Whether to display row-assembly progress.

    Returns:
        Model-ready DataFrame and sampling diagnostics.
    """

    all_candidates = sorted(
        (
            candidate
            for stratum_state in sampling_scan.strata.values()
            for candidate in stratum_state.candidates
        ),
        key=lambda candidate: (
            candidate.tile_sequence,
            candidate.local_pixel_index,
        ),
    )
    sampled_row_count = len(all_candidates)
    candidates_by_tile: dict[int, list[SamplingCandidate]] = {}
    for candidate in all_candidates:
        candidates_by_tile.setdefault(candidate.tile_sequence, []).append(
            candidate
        )

    block_keys = sorted(
        {
            (block_column, block_row)
            for block_column, block_row, _ in sampling_scan.strata
        }
    )
    block_id_by_key = {
        block_key: block_offset + 1
        for block_offset, block_key in enumerate(block_keys)
    }
    available_pixels_by_block = {block_key: 0 for block_key in block_keys}
    for stratum_key, stratum_state in sampling_scan.strata.items():
        available_pixels_by_block[stratum_key[:2]] += (
            stratum_state.available_pixel_count
        )

    table_columns: dict[str, np.ndarray] = {
        "row": np.empty(sampled_row_count, dtype=np.int32),
        "column": np.empty(sampled_row_count, dtype=np.int32),
        "cache_tile_id": np.empty(sampled_row_count, dtype=object),
        "longitude": np.empty(sampled_row_count, dtype=np.float64),
        "latitude": np.empty(sampled_row_count, dtype=np.float64),
        "sampling_block_id": np.empty(sampled_row_count, dtype=np.int64),
        "sampling_block_column": np.empty(sampled_row_count, dtype=np.int64),
        "sampling_block_row": np.empty(sampled_row_count, dtype=np.int64),
        "reference_site": np.empty(sampled_row_count, dtype=np.uint8),
        "pixel_area_m2": np.empty(sampled_row_count, dtype=np.float64),
        "available_pixels_in_block_class": np.empty(
            sampled_row_count,
            dtype=np.int64,
        ),
        "sampled_pixels_in_block_class": np.empty(
            sampled_row_count,
            dtype=np.int32,
        ),
        "sampling_probability": np.empty(
            sampled_row_count,
            dtype=np.float64,
        ),
        "sampling_weight": np.empty(sampled_row_count, dtype=np.float64),
        "area_weight_m2": np.empty(sampled_row_count, dtype=np.float64),
    }
    for sampled_band_name in sampling_scan.sampled_band_names:
        table_columns[sampled_band_name] = np.full(
            sampled_row_count,
            np.nan,
            dtype=np.float64,
        )

    cache_grid = analysis_configuration.grid
    geographic_transformer = Transformer.from_crs(
        cache_grid.crs,
        "EPSG:4326",
        always_xy=True,
    )
    sampled_band_defined_pixel_counts = np.zeros(
        len(sampling_scan.sampled_band_names),
        dtype=np.int64,
    )
    complete_sampled_band_row_mask = np.ones(
        sampled_row_count,
        dtype=np.bool_,
    )
    write_offset = 0

    with tqdm(
        sorted(candidates_by_tile.items()),
        total=len(candidates_by_tile),
        desc="Reading selected raster pixels",
        unit="tile",
        dynamic_ncols=True,
        disable=not show_progress,
    ) as selected_tile_progress:
        for tile_sequence, tile_candidates in selected_tile_progress:
            cached_tile = analysis_cache_tiles.tiles[tile_sequence]
            (
                tile_values,
                tile_validity,
                tile_transform,
                _tile_array_bytes,
            ) = _read_tile_values_and_validity(cached_tile)
            tile_candidate_count = len(tile_candidates)
            destination_slice = slice(
                write_offset,
                write_offset + tile_candidate_count,
            )
            local_pixel_indices = np.asarray(
                [
                    candidate.local_pixel_index
                    for candidate in tile_candidates
                ],
                dtype=np.int64,
            )
            tile_width = tile_values.shape[2]
            local_rows = local_pixel_indices // tile_width
            local_columns = local_pixel_indices % tile_width
            # Candidate addresses are tile-local row and column indices. Apply
            # the tile's affine transform at each pixel center to recover cache-
            # CRS coordinates, which are then converted to the longitude and
            # latitude stored with each sampled row.
            projected_x = (
                tile_transform.c
                + (local_columns.astype(np.float64) + 0.5) * tile_transform.a
                + (local_rows.astype(np.float64) + 0.5) * tile_transform.b
            )
            projected_y = (
                tile_transform.f
                + (local_columns.astype(np.float64) + 0.5) * tile_transform.d
                + (local_rows.astype(np.float64) + 0.5) * tile_transform.e
            )
            longitudes, latitudes = geographic_transformer.transform(
                projected_x,
                projected_y,
            )
            table_columns["row"][destination_slice] = (
                -(cached_tile.tile.row + 1) * tile_width + local_rows
            ).astype(np.int32)
            table_columns["column"][destination_slice] = (
                cached_tile.tile.column * tile_width + local_columns
            ).astype(np.int32)
            table_columns["cache_tile_id"][destination_slice] = (
                cached_tile.tile.tile_id
            )
            table_columns["longitude"][destination_slice] = longitudes
            table_columns["latitude"][destination_slice] = latitudes

            for candidate_offset, candidate in enumerate(tile_candidates):
                destination_row = write_offset + candidate_offset
                stratum_key = (
                    candidate.sampling_block_column,
                    candidate.sampling_block_row,
                    candidate.reference_site_class,
                )
                stratum_state = sampling_scan.strata[stratum_key]
                sampled_in_stratum = len(stratum_state.candidates)
                sampling_weight = (
                    stratum_state.available_pixel_count / sampled_in_stratum
                )
                table_columns["sampling_block_id"][destination_row] = (
                    block_id_by_key[stratum_key[:2]]
                )
                table_columns["sampling_block_column"][destination_row] = (
                    candidate.sampling_block_column
                )
                table_columns["sampling_block_row"][destination_row] = (
                    candidate.sampling_block_row
                )
                table_columns["reference_site"][destination_row] = (
                    candidate.reference_site_class
                )
                table_columns["pixel_area_m2"][destination_row] = (
                    sampling_scan.raster_summary.pixel_area_m2
                )
                table_columns[
                    "available_pixels_in_block_class"
                ][destination_row] = stratum_state.available_pixel_count
                table_columns[
                    "sampled_pixels_in_block_class"
                ][destination_row] = sampled_in_stratum
                table_columns["sampling_probability"][destination_row] = (
                    sampled_in_stratum
                    / stratum_state.available_pixel_count
                )
                table_columns["sampling_weight"][destination_row] = (
                    sampling_weight
                )
                table_columns["area_weight_m2"][destination_row] = (
                    sampling_scan.raster_summary.pixel_area_m2
                    * sampling_weight
                )

            # np.newaxis reshapes the band offsets to (bands, 1) and the paired
            # row and column offsets to (1, pixels). NumPy then broadcasts these
            # indices to return one (bands, pixels) matrix without a Python loop.
            sampled_band_values = tile_values[
                np.asarray(sampling_scan.sampled_band_offsets)[:, np.newaxis],
                local_rows[np.newaxis, :],
                local_columns[np.newaxis, :],
            ].astype(np.float64)
            sampled_band_validity = tile_validity[
                np.asarray(sampling_scan.sampled_band_offsets)[:, np.newaxis],
                local_rows[np.newaxis, :],
                local_columns[np.newaxis, :],
            ]
            sampled_band_values[~sampled_band_validity] = np.nan
            for sampled_band_offset, sampled_band_name in enumerate(
                sampling_scan.sampled_band_names
            ):
                table_columns[sampled_band_name][destination_slice] = (
                    sampled_band_values[sampled_band_offset]
                )
                sampled_band_defined_pixel_counts[sampled_band_offset] += int(
                    np.count_nonzero(
                        sampled_band_validity[sampled_band_offset]
                    )
                )
            complete_sampled_band_row_mask[destination_slice] = np.all(
                sampled_band_validity,
                axis=0,
            )
            write_offset += tile_candidate_count
            selected_tile_progress.set_postfix(
                rows=f"{write_offset:,}/{sampled_row_count:,}"
            )

    class_summaries = []
    for reference_site_class in REFERENCE_SITE_CLASSES:
        class_strata = [
            stratum_state
            for (
                _block_column,
                _block_row,
                stratum_reference_class,
            ), stratum_state in sampling_scan.strata.items()
            if stratum_reference_class == reference_site_class
        ]
        class_sampling_weights = [
            stratum_state.available_pixel_count
            / len(stratum_state.candidates)
            for stratum_state in class_strata
        ]
        class_summaries.append(
            SamplingClassSummary(
                reference_site_class=reference_site_class,
                available_pixels=sum(
                    stratum.available_pixel_count for stratum in class_strata
                ),
                sampled_pixels=sum(
                    len(stratum.candidates) for stratum in class_strata
                ),
                available_area_m2=sum(
                    stratum.available_area_m2 for stratum in class_strata
                ),
                weighted_pixels=float(
                    sum(
                        len(stratum.candidates) * sampling_weight
                        for stratum, sampling_weight in zip(
                            class_strata,
                            class_sampling_weights,
                            strict=True,
                        )
                    )
                ),
                weighted_area_m2=float(
                    sum(
                        len(stratum.candidates)
                        * sampling_weight
                        * sampling_scan.raster_summary.pixel_area_m2
                        for stratum, sampling_weight in zip(
                            class_strata,
                            class_sampling_weights,
                            strict=True,
                        )
                    )
                ),
                blocks_with_class=len(class_strata),
                minimum_sampling_weight=(
                    min(class_sampling_weights)
                    if class_sampling_weights
                    else math.nan
                ),
                maximum_sampling_weight=(
                    max(class_sampling_weights)
                    if class_sampling_weights
                    else math.nan
                ),
            )
        )

    available_pixels_per_block = np.asarray(
        list(available_pixels_by_block.values()),
        dtype=np.int64,
    )
    return SpatialSample(
        table=pd.DataFrame(table_columns, copy=False),
        reference_band_name=analysis_configuration.band_names()[
            sampling_scan.reference_band_offset
        ],
        sampled_band_names=sampling_scan.sampled_band_names,
        sampled_band_defined_pixel_counts=tuple(
            int(count) for count in sampled_band_defined_pixel_counts
        ),
        complete_sampled_band_row_count=int(
            np.count_nonzero(complete_sampled_band_row_mask)
        ),
        block_size_meters=analysis_configuration.sampling.block_size_meters,
        samples_per_class_per_block=(
            analysis_configuration.sampling.samples_per_class_per_block
        ),
        random_seed=analysis_configuration.sampling.random_seed,
        block_count=len(block_keys),
        minimum_available_pixels_per_block=int(
            np.min(available_pixels_per_block)
        ),
        median_available_pixels_per_block=float(
            np.median(available_pixels_per_block)
        ),
        maximum_available_pixels_per_block=int(
            np.max(available_pixels_per_block)
        ),
        class_summaries=(class_summaries[0], class_summaries[1]),
        elapsed_seconds=time.perf_counter() - started_at,
    )


def build_spatial_sample(
    analysis_configuration: AnalysisConfiguration,
    analysis_cache_tiles: AnalysisCacheTiles,
    show_progress: bool,
) -> tuple[SpatialSample, RasterCacheScanSummary]:
    """Run both bounded-memory passes over validated cache tiles.

    Args:
        analysis_configuration: Complete raster and sampling contract.
        analysis_cache_tiles: Validated AOI cache inputs.
        show_progress: Whether to display scanning and assembly progress.

    Returns:
        Spatial sample and incremental source-raster summary.
    """

    started_at = time.perf_counter()
    sampling_scan = scan_cached_tiles(
        analysis_configuration,
        analysis_cache_tiles,
        show_progress,
    )
    spatial_sample = assemble_spatial_sample(
        analysis_configuration,
        analysis_cache_tiles,
        sampling_scan,
        started_at,
        show_progress,
    )
    return spatial_sample, sampling_scan.raster_summary


def write_spatial_sample_parquet(
    spatial_sample: SpatialSample,
    analysis_configuration: AnalysisConfiguration,
    analysis_cache_tiles: AnalysisCacheTiles,
    output_path: Path,
    show_progress: bool,
) -> ParquetWriteSummary:
    """Write a compressed sample with embedded analysis/cache provenance.

    Args:
        spatial_sample: Model-ready sample and diagnostics.
        analysis_configuration: Authoritative analysis contract.
        analysis_cache_tiles: Validated cache files represented by the sample.
        output_path: Destination path ending in ``.parquet``.
        show_progress: Whether to display write and verification progress.

    Returns:
        Verified Parquet metadata and write measurements.

    Raises:
        ValueError: If the destination does not use the Parquet suffix.
        RuntimeError: If written rows, columns, or provenance do not verify.
    """

    started_at = time.perf_counter()
    resolved_output_path = output_path.expanduser().resolve()
    if resolved_output_path.suffix.lower() != ".parquet":
        raise ValueError(
            "Sample output must end in .parquet: "
            f"{resolved_output_path}"
        )
    tile_set_sha256 = hashlib.sha256(
        "\n".join(
            f"{cached_tile.tile.tile_id}:{cached_tile.sha256}"
            for cached_tile in analysis_cache_tiles.tiles
        ).encode("utf-8")
    ).hexdigest()
    provenance = {
        "schema_version": 1,
        "analysis_name": analysis_configuration.analysis_name,
        "display_name": analysis_configuration.display_name,
        "analysis_configuration_path": str(analysis_configuration.path),
        "analysis_configuration_sha256": (
            analysis_configuration.configuration_sha256
        ),
        "raster_configuration_sha256": (
            analysis_configuration.raster_configuration_sha256
        ),
        "aoi_path": str(analysis_configuration.aoi_path),
        "stack_identifier": analysis_cache_tiles.stack_identifier,
        "cache_manifest_path": str(analysis_cache_tiles.manifest_path),
        "cache_tile_count": len(analysis_cache_tiles.tiles),
        "cache_tile_set_sha256": tile_set_sha256,
        "sampling": {
            "block_size_meters": spatial_sample.block_size_meters,
            "samples_per_class_per_block": (
                spatial_sample.samples_per_class_per_block
            ),
            "random_seed": spatial_sample.random_seed,
        },
    }

    progress = tqdm(
        total=3,
        desc="Writing Parquet sample",
        unit="step",
        disable=not show_progress,
    )
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    progress.update()
    arrow_table = pa.Table.from_pandas(
        spatial_sample.table,
        preserve_index=False,
    )
    schema_metadata = dict(arrow_table.schema.metadata or {})
    schema_metadata[PARQUET_PROVENANCE_KEY] = json.dumps(
        provenance,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    arrow_table = arrow_table.replace_schema_metadata(schema_metadata)
    pq.write_table(
        arrow_table,
        resolved_output_path,
        compression="zstd",
    )
    progress.update()

    parquet_file = pq.ParquetFile(resolved_output_path)
    parquet_metadata = parquet_file.metadata
    written_provenance = json.loads(
        parquet_file.schema_arrow.metadata[PARQUET_PROVENANCE_KEY]
    )
    if parquet_metadata.num_rows != len(spatial_sample.table):
        raise RuntimeError("Parquet row verification failed.")
    if parquet_metadata.num_columns != spatial_sample.table.shape[1]:
        raise RuntimeError("Parquet column verification failed.")
    if written_provenance != provenance:
        raise RuntimeError("Parquet analysis provenance verification failed.")
    progress.update()
    progress.close()
    return ParquetWriteSummary(
        path=resolved_output_path,
        rows=parquet_metadata.num_rows,
        columns=parquet_metadata.num_columns,
        row_groups=parquet_metadata.num_row_groups,
        compression=parquet_metadata.row_group(0).column(0).compression,
        file_size_bytes=resolved_output_path.stat().st_size,
        elapsed_seconds=time.perf_counter() - started_at,
    )


def _locator_bounds(
    geographic_bounds: BoundingBox,
    minimum_span_degrees: float,
) -> BoundingBox:
    """Expand small AOI bounds into a visible world-map locator box.

    Args:
        geographic_bounds: AOI longitude-latitude bounds.
        minimum_span_degrees: Minimum locator width and height.

    Returns:
        Geographic bounds enclosing the AOI with a visible minimum span.
    """

    center_longitude = (
        geographic_bounds.left + geographic_bounds.right
    ) / 2.0
    center_latitude = (
        geographic_bounds.bottom + geographic_bounds.top
    ) / 2.0
    longitude_span = max(
        geographic_bounds.right - geographic_bounds.left,
        minimum_span_degrees,
    )
    latitude_span = max(
        geographic_bounds.top - geographic_bounds.bottom,
        minimum_span_degrees,
    )
    return BoundingBox(
        left=max(-180.0, center_longitude - longitude_span / 2.0),
        bottom=max(-90.0, center_latitude - latitude_span / 2.0),
        right=min(180.0, center_longitude + longitude_span / 2.0),
        top=min(90.0, center_latitude + latitude_span / 2.0),
    )


def create_analysis_location_figure(
    wgs84_aoi: BaseGeometry,
    analysis_name: str,
    figure_path: Path,
    show_progress: bool,
) -> LocationFigureSummary:
    """Create a publication-resolution world locator map from the TOML AOI.

    Args:
        wgs84_aoi: Configured Polygon or MultiPolygon in EPSG:4326.
        analysis_name: Human-readable label for the map callout.
        figure_path: PNG, PDF, or SVG output path.
        show_progress: Whether to display figure-generation progress.

    Returns:
        Saved path, AOI bounds, and basemap availability.

    Raises:
        ValueError: If the name is empty or figure suffix is unsupported.
    """

    cleaned_analysis_name = analysis_name.strip()
    if not cleaned_analysis_name:
        raise ValueError("The analysis display name cannot be empty.")
    resolved_figure_path = figure_path.expanduser().resolve()
    if resolved_figure_path.suffix.lower() not in SUPPORTED_FIGURE_SUFFIXES:
        supported_suffixes = ", ".join(sorted(SUPPORTED_FIGURE_SUFFIXES))
        raise ValueError(
            f"Location figure must use one of: {supported_suffixes}."
        )

    minimum_x, minimum_y, maximum_x, maximum_y = wgs84_aoi.bounds
    geographic_bounds = BoundingBox(
        minimum_x,
        minimum_y,
        maximum_x,
        maximum_y,
    )
    locator_bounds = _locator_bounds(geographic_bounds, 5.0)
    center_longitude = (
        geographic_bounds.left + geographic_bounds.right
    ) / 2.0
    center_latitude = (
        geographic_bounds.bottom + geographic_bounds.top
    ) / 2.0
    progress = tqdm(
        total=3,
        desc="Generating location figure",
        unit="step",
        disable=not show_progress,
    )
    figure = None
    land_basemap_available = True
    try:
        with plt.rc_context(
            {
                "font.family": "DejaVu Sans",
                "font.size": 10,
                "axes.titleweight": "bold",
                "axes.titlesize": 16,
            }
        ):
            figure = plt.figure(figsize=(12.0, 6.4), facecolor="white")
            axis = figure.add_subplot(1, 1, 1, projection=ccrs.Robinson())
            axis.set_global()
            axis.set_facecolor("#DCEAF1")
            land_artist = axis.add_feature(
                cfeature.LAND.with_scale("110m"),
                facecolor="#EEEDE8",
                edgecolor="#586166",
                linewidth=0.45,
                zorder=1,
            )
            axis.gridlines(
                crs=ccrs.PlateCarree(),
                draw_labels=False,
                linewidth=0.35,
                color="#FFFFFF",
                alpha=0.9,
                linestyle="-",
                zorder=2,
            )
            axis.set_title("Global analysis location", pad=16)
            axis.add_geometries(
                [wgs84_aoi],
                crs=ccrs.PlateCarree(),
                facecolor="#D1493F",
                edgecolor="#8E2722",
                linewidth=0.7,
                alpha=0.88,
                zorder=4,
            )
            axis.add_patch(
                Rectangle(
                    (locator_bounds.left, locator_bounds.bottom),
                    locator_bounds.right - locator_bounds.left,
                    locator_bounds.top - locator_bounds.bottom,
                    fill=False,
                    edgecolor="#161A1D",
                    linewidth=1.25,
                    linestyle=(0, (4, 2)),
                    transform=ccrs.PlateCarree(),
                    zorder=5,
                )
            )
            label_x = 0.16 if center_longitude >= 0.0 else 0.84
            label_y = 0.20 if center_latitude >= 0.0 else 0.80
            axis.annotate(
                textwrap.fill(
                    cleaned_analysis_name,
                    width=28,
                    break_long_words=False,
                ),
                xy=(center_longitude, center_latitude),
                xycoords=ccrs.PlateCarree()._as_mpl_transform(axis),
                xytext=(label_x, label_y),
                textcoords="axes fraction",
                ha="center",
                va="center",
                fontsize=11,
                fontweight="bold",
                color="#161A1D",
                bbox={
                    "boxstyle": "square,pad=0.45",
                    "facecolor": "white",
                    "edgecolor": "#161A1D",
                    "linewidth": 0.7,
                    "alpha": 0.96,
                },
                arrowprops={
                    "arrowstyle": "-|>",
                    "color": "#161A1D",
                    "linewidth": 1.15,
                    "shrinkA": 5,
                    "shrinkB": 4,
                    "connectionstyle": "arc3,rad=0.08",
                },
                zorder=7,
            )
            axis.legend(
                handles=[
                    Patch(
                        facecolor="#D1493F",
                        edgecolor="none",
                        label="Configured AOI",
                    ),
                    Line2D(
                        [0],
                        [0],
                        color="#161A1D",
                        linewidth=1.25,
                        linestyle=(0, (4, 2)),
                        label="AOI locator box",
                    ),
                ],
                loc="lower left",
                bbox_to_anchor=(0.025, 0.025),
                ncol=2,
                frameon=True,
                facecolor="white",
                edgecolor="none",
                framealpha=0.88,
                fontsize=9,
                handlelength=2.4,
            )
            basemap_note = figure.text(
                0.99,
                0.015,
                "Base map: Natural Earth 1:110m",
                ha="right",
                va="bottom",
                fontsize=7.5,
                color="#596268",
            )
            figure.subplots_adjust(
                left=0.025,
                right=0.975,
                bottom=0.09,
                top=0.90,
            )
            progress.update(2)
            resolved_figure_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                figure.savefig(
                    resolved_figure_path,
                    dpi=LOCATION_FIGURE_DPI,
                    bbox_inches="tight",
                    facecolor=figure.get_facecolor(),
                )
            except (EOFError, OSError):
                land_artist.remove()
                basemap_note.set_text("Natural Earth land layer unavailable")
                figure.savefig(
                    resolved_figure_path,
                    dpi=LOCATION_FIGURE_DPI,
                    bbox_inches="tight",
                    facecolor=figure.get_facecolor(),
                )
                land_basemap_available = False
            progress.update()
    finally:
        if figure is not None:
            plt.close(figure)
        progress.close()

    return LocationFigureSummary(
        path=resolved_figure_path,
        analysis_name=cleaned_analysis_name,
        bounds=geographic_bounds,
        land_basemap_available=land_basemap_available,
    )


def print_cache_scan_report(
    analysis_configuration: AnalysisConfiguration,
    analysis_cache_tiles: AnalysisCacheTiles,
    raster_summary: RasterCacheScanSummary,
    include_band_report: bool,
) -> None:
    """Print cache provenance, bounded memory, AOI coverage, and band statistics.

    Args:
        analysis_configuration: Complete analysis identity and grid contract.
        analysis_cache_tiles: Validated cache files represented by the scan.
        raster_summary: Incremental source coverage and value measurements.
        include_band_report: Whether to print every configured band.

    Returns:
        None: The report is written to standard output.
    """

    aoi_area_km2 = (
        raster_summary.aoi_pixel_count
        * raster_summary.pixel_area_m2
        / 1_000_000.0
    )
    any_band_defined_percent = (
        100.0
        * raster_summary.any_band_defined_pixel_count
        / raster_summary.aoi_pixel_count
    )
    every_band_defined_percent = (
        100.0
        * raster_summary.every_band_defined_pixel_count
        / raster_summary.aoi_pixel_count
    )
    print()
    print("Raster cache scan")
    print(f"Analysis: {analysis_configuration.display_name}")
    print(f"Configuration: {analysis_configuration.path}")
    print(f"AOI: {analysis_configuration.aoi_path}")
    print(f"Stack: {analysis_cache_tiles.stack_identifier}")
    print(f"Manifest: {analysis_cache_tiles.manifest_path}")
    print(f"Validated cache tiles: {raster_summary.cache_tile_count:,}")
    print(
        "Compressed cache input: "
        f"{raster_summary.cache_file_size_bytes / MEBIBYTE:,.2f} MiB"
    )
    print(
        "Largest in-memory source tile: "
        f"{raster_summary.peak_tile_array_bytes / MEBIBYTE:,.2f} MiB"
    )
    print(
        "Grid: "
        f"{analysis_configuration.grid.crs}, "
        f"{analysis_configuration.grid.pixel_size_meters:,} m pixels, "
        f"{analysis_configuration.grid.tile_size_pixels} x "
        f"{analysis_configuration.grid.tile_size_pixels} pixels/tile"
    )
    print(
        f"AOI grid cells: {raster_summary.aoi_pixel_count:,} "
        f"({aoi_area_km2:,.2f} km2 full-pixel approximation)"
    )
    print(
        "Defined in at least one band: "
        f"{raster_summary.any_band_defined_pixel_count:,} "
        f"({any_band_defined_percent:.2f}%)"
    )
    print(
        "Defined in every band: "
        f"{raster_summary.every_band_defined_pixel_count:,} "
        f"({every_band_defined_percent:.2f}%)"
    )
    print(
        "Eligible non-reference raster footprint: "
        f"{raster_summary.eligible_pixel_count:,} pixels"
    )
    print(
        "Reference pixels excluded outside that footprint: "
        f"{raster_summary.excluded_reference_pixel_count:,}"
    )

    if include_band_report:
        print()
        print("Configured band coverage inside the AOI")
        print(
            "Band  Name                                            "
            "Defined       Cover       Area km2       Min       Mean       Max"
        )
        for band_summary in raster_summary.band_summaries:
            minimum_text = (
                "NA"
                if band_summary.minimum is None
                else f"{band_summary.minimum:.6g}"
            )
            mean_text = (
                "NA"
                if band_summary.mean is None
                else f"{band_summary.mean:.6g}"
            )
            maximum_text = (
                "NA"
                if band_summary.maximum is None
                else f"{band_summary.maximum:.6g}"
            )
            print(
                f"{band_summary.index:>4}  "
                f"{band_summary.name[:46]:<46}  "
                f"{band_summary.defined_pixel_count:>11,}  "
                f"{band_summary.defined_percent:>8.2f}%  "
                f"{band_summary.defined_area_km2:>13,.2f}  "
                f"{minimum_text:>8}  "
                f"{mean_text:>8}  "
                f"{maximum_text:>8}"
            )


def print_spatial_sampling_report(spatial_sample: SpatialSample) -> None:
    """Print class, block, weight, and sampled-band diagnostics.

    Args:
        spatial_sample: Completed sample and diagnostic measurements.

    Returns:
        None: The report is written to standard output.
    """

    print()
    print("Spatial sampling report")
    print(f"Reference band: {spatial_sample.reference_band_name}")
    print("Reference coding: 1 = reference site; 0 = background")
    print(
        "Sampling blocks: "
        f"{spatial_sample.block_count:,} at "
        f"{spatial_sample.block_size_meters:,} m square"
    )
    print(
        "Per-block/class cap: "
        f"{spatial_sample.samples_per_class_per_block:,} pixels"
    )
    print(f"Random seed: {spatial_sample.random_seed}")
    print(
        "Eligible pixels per block: "
        f"minimum {spatial_sample.minimum_available_pixels_per_block:,}, "
        f"median {spatial_sample.median_available_pixels_per_block:,.1f}, "
        f"maximum {spatial_sample.maximum_available_pixels_per_block:,}"
    )
    print()
    print("Class sampling and weight checks")
    for class_summary in spatial_sample.class_summaries:
        class_label = (
            "reference" if class_summary.reference_site_class else "background"
        )
        retention_percent = (
            100.0
            * class_summary.sampled_pixels
            / class_summary.available_pixels
            if class_summary.available_pixels
            else 0.0
        )
        print(
            f"  {class_summary.reference_site_class} {class_label}: "
            f"{class_summary.sampled_pixels:,} / "
            f"{class_summary.available_pixels:,} pixels retained "
            f"({retention_percent:.2f}%), "
            f"{class_summary.blocks_with_class:,} blocks"
        )
        print(
            "    reconstructed pixels: "
            f"{class_summary.weighted_pixels:,.1f}; "
            "reconstructed area: "
            f"{class_summary.weighted_area_m2 / 1_000_000:,.2f} km2; "
            "weight range: "
            f"{class_summary.minimum_sampling_weight:,.3f}-"
            f"{class_summary.maximum_sampling_weight:,.3f}"
        )

    sampled_row_count = len(spatial_sample.table)
    fully_defined_band_count = sum(
        defined_pixel_count == sampled_row_count
        for defined_pixel_count in (
            spatial_sample.sampled_band_defined_pixel_counts
        )
    )
    completely_missing_band_count = sum(
        defined_pixel_count == 0
        for defined_pixel_count in (
            spatial_sample.sampled_band_defined_pixel_counts
        )
    )
    partially_defined_band_count = (
        len(spatial_sample.sampled_band_names)
        - fully_defined_band_count
        - completely_missing_band_count
    )
    complete_sampled_band_row_percent = (
        100.0
        * spatial_sample.complete_sampled_band_row_count
        / sampled_row_count
    )
    print()
    print("Sampled raster-band coverage")
    print(f"Sample rows: {sampled_row_count:,}")
    print(f"Raster data columns: {len(spatial_sample.sampled_band_names):,}")
    print(f"Fully defined bands: {fully_defined_band_count:,}")
    print(f"Partially defined bands: {partially_defined_band_count:,}")
    print(f"Completely missing bands: {completely_missing_band_count:,}")
    print(
        "Rows complete across every raster data band: "
        f"{spatial_sample.complete_sampled_band_row_count:,} / "
        f"{sampled_row_count:,} "
        f"({complete_sampled_band_row_percent:.2f}%)"
    )
    lowest_coverage_bands = sorted(
        zip(
            spatial_sample.sampled_band_defined_pixel_counts,
            spatial_sample.sampled_band_names,
            strict=True,
        )
    )[: min(8, len(spatial_sample.sampled_band_names))]
    print("Lowest-coverage raster data bands in the sample:")
    for defined_pixel_count, band_name in lowest_coverage_bands:
        print(
            f"  {band_name}: {defined_pixel_count:,} / "
            f"{sampled_row_count:,} "
            f"({100.0 * defined_pixel_count / sampled_row_count:.2f}%)"
        )
    print(f"Sampling time: {spatial_sample.elapsed_seconds:,.2f} seconds")


def print_parquet_report(
    parquet_summary: ParquetWriteSummary,
    table_memory_bytes: int,
) -> None:
    """Print verified sample-table storage measurements.

    Args:
        parquet_summary: Verified Parquet metadata and timings.
        table_memory_bytes: In-memory pandas table size.

    Returns:
        None: The report is written to standard output.
    """

    print()
    print("Spatial sample Parquet")
    print(f"Path: {parquet_summary.path}")
    print(
        f"Shape: {parquet_summary.rows:,} rows x "
        f"{parquet_summary.columns:,} columns"
    )
    print(f"Row groups: {parquet_summary.row_groups:,}")
    print(f"Compression: {parquet_summary.compression}")
    print(f"In-memory table: {table_memory_bytes / MEBIBYTE:,.2f} MiB")
    print(
        "Compressed file: "
        f"{parquet_summary.file_size_bytes / MEBIBYTE:,.2f} MiB"
    )
    print(f"Write and verification: {parquet_summary.elapsed_seconds:,.2f} s")


def print_location_figure_report(
    figure_summary: LocationFigureSummary,
) -> None:
    """Print world-location figure output and mapped bounds.

    Args:
        figure_summary: Saved path and AOI map metadata.

    Returns:
        None: The report is written to standard output.
    """

    print()
    print("Analysis world-location figure")
    print(f"Path: {figure_summary.path}")
    print(f"Label: {figure_summary.analysis_name}")
    print(
        "AOI bounds: "
        f"{figure_summary.bounds.left:.4f}, "
        f"{figure_summary.bounds.bottom:.4f}, "
        f"{figure_summary.bounds.right:.4f}, "
        f"{figure_summary.bounds.top:.4f}"
    )
    print(
        "Natural Earth land basemap: "
        f"{'included' if figure_summary.land_basemap_available else 'unavailable'}"
    )


def main() -> None:
    """Resolve configured cache tiles and build the spatial sample."""

    arguments = parse_args()
    analysis_configuration = load_analysis_configuration(
        arguments.analysis_configuration
    )
    show_progress = not arguments.no_progress
    default_sample_path = (
        Path("outputs")
        / "samples"
        / f"{analysis_configuration.analysis_name}_spatial_sample.parquet"
    )
    sample_output_path = arguments.sample_output or default_sample_path
    default_figure_path = (
        Path("outputs")
        / "figures"
        / f"{analysis_configuration.analysis_name}_world_location.png"
    )
    figure_output_path = arguments.location_figure or default_figure_path

    analysis_cache_tiles = resolve_analysis_cache_tiles(
        analysis_configuration,
        show_progress,
    )
    spatial_sample, raster_summary = build_spatial_sample(
        analysis_configuration,
        analysis_cache_tiles,
        show_progress,
    )
    print_cache_scan_report(
        analysis_configuration,
        analysis_cache_tiles,
        raster_summary,
        not arguments.no_band_report,
    )
    print_spatial_sampling_report(spatial_sample)
    table_memory_bytes = int(
        spatial_sample.table.memory_usage(index=True, deep=True).sum()
    )
    parquet_summary = write_spatial_sample_parquet(
        spatial_sample,
        analysis_configuration,
        analysis_cache_tiles,
        sample_output_path,
        show_progress,
    )
    print_parquet_report(parquet_summary, table_memory_bytes)

    if not arguments.no_location_figure:
        figure_summary = create_analysis_location_figure(
            analysis_cache_tiles.wgs84_aoi,
            analysis_configuration.display_name,
            figure_output_path,
            show_progress,
        )
        print_location_figure_report(figure_summary)


if __name__ == "__main__":
    main()
