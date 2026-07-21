"""Load a multiband ecoregion GeoTIFF and report its in-memory coverage."""

from __future__ import annotations

import argparse
import math
import re
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import rasterio
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from pyproj import Transformer
from rasterio.coords import BoundingBox
from rasterio.crs import CRS
from rasterio.transform import Affine
from rasterio.warp import Resampling, calculate_default_transform, reproject
from tqdm.auto import tqdm


EARTH_RADIUS_METERS = 6_371_008.8
MEBIBYTE = 1024**2
MAX_FOOTPRINT_DIMENSION = 600
LOCATION_FIGURE_DPI = 300
SUPPORTED_FIGURE_SUFFIXES = {".pdf", ".png", ".svg"}
EQUAL_AREA_CRS = "EPSG:8857"
DEFAULT_SAMPLING_BLOCK_SIZE_METERS = 25_000.0
DEFAULT_SAMPLES_PER_CLASS_PER_BLOCK = 100
DEFAULT_RANDOM_SEED = 42


@dataclass(frozen=True)
class RasterPixelData:
    """All values and per-band validity masks from one raster.

    Attributes:
        path: Source GeoTIFF path.
        values: Pixel values with shape ``(bands, rows, columns)``. Invalid
            floating-point cells contain ``NaN`` and invalid integer cells
            contain zero; ``validity`` is authoritative in either case.
        validity: Boolean validity flags with the same shape as ``values``.
        band_names: Band descriptions, with generated names for undescribed
            bands.
        source_dtypes: Rasterio dtype name for each source band.
        nodata_values: Nodata value reported for each source band.
        transform: Affine transform from pixel coordinates to the raster CRS.
        crs: Raster coordinate reference system, if one is defined.
        bounds: Raster bounds in the raster coordinate reference system.
    """

    path: Path
    values: np.ndarray
    validity: np.ndarray
    band_names: tuple[str, ...]
    source_dtypes: tuple[str, ...]
    nodata_values: tuple[float | None, ...]
    transform: Affine
    crs: CRS | None
    bounds: BoundingBox

    @property
    def band_count(self) -> int:
        """Return the number of raster bands.

        Returns:
            Number of bands loaded into memory.
        """

        return self.values.shape[0]

    @property
    def height(self) -> int:
        """Return the raster height in pixels.

        Returns:
            Number of raster rows.
        """

        return self.values.shape[1]

    @property
    def width(self) -> int:
        """Return the raster width in pixels.

        Returns:
            Number of raster columns.
        """

        return self.values.shape[2]

    @property
    def pixel_count(self) -> int:
        """Return the number of grid cells in the raster rectangle.

        Returns:
            Product of raster width and height.
        """

        return self.height * self.width

    @property
    def memory_bytes(self) -> int:
        """Return memory occupied by the value and validity arrays.

        Returns:
            Combined array size in bytes.
        """

        return self.values.nbytes + self.validity.nbytes

    def pixel_values(self) -> np.ndarray:
        """Return a zero-copy pixel-by-band view of the value cube.

        Returns:
            Array with shape ``(pixels, bands)`` in row-major pixel order.
        """

        return np.moveaxis(self.values, 0, -1).reshape(
            self.pixel_count,
            self.band_count,
        )

    def pixel_validity(self) -> np.ndarray:
        """Return a zero-copy pixel-by-band view of validity flags.

        Returns:
            Boolean array with shape ``(pixels, bands)`` in row-major pixel
            order.
        """

        return np.moveaxis(self.validity, 0, -1).reshape(
            self.pixel_count,
            self.band_count,
        )

    def pixel_centers(self) -> tuple[np.ndarray, np.ndarray]:
        """Calculate center coordinates for every pixel.

        Coordinates are returned in the raster CRS and use the same row-major
        order as ``pixel_values``.

        Returns:
            Pair of one-dimensional ``(x, y)`` coordinate arrays.
        """

        rows, columns = np.indices((self.height, self.width), dtype=np.float64)
        column_centers = columns + 0.5
        row_centers = rows + 0.5
        x_coordinates = (
            self.transform.c
            + column_centers * self.transform.a
            + row_centers * self.transform.b
        )
        y_coordinates = (
            self.transform.f
            + column_centers * self.transform.d
            + row_centers * self.transform.e
        )
        return x_coordinates.ravel(), y_coordinates.ravel()


@dataclass(frozen=True)
class CoverageSummary:
    """Coverage measurements for one two-dimensional pixel mask.

    Attributes:
        defined_pixels: Number of pixels marked as defined.
        total_pixels: Number of pixels in the raster rectangle.
        defined_percent: Percentage of raster pixels marked as defined.
        area_square_kilometers: Approximate defined area, or ``None`` when the
            raster CRS cannot be converted to area.
    """

    defined_pixels: int
    total_pixels: int
    defined_percent: float
    area_square_kilometers: float | None


@dataclass(frozen=True)
class BandSummary:
    """Coverage and value measurements for one raster band.

    Attributes:
        index: One-based raster band index.
        name: Raster band description or generated fallback name.
        coverage: Defined-pixel coverage measurements.
        minimum: Minimum defined value, or ``None`` for an empty band.
        mean: Mean defined value, or ``None`` for an empty band.
        maximum: Maximum defined value, or ``None`` for an empty band.
    """

    index: int
    name: str
    coverage: CoverageSummary
    minimum: float | None
    mean: float | None
    maximum: float | None


@dataclass(frozen=True)
class GeographicFootprint:
    """Coarsened ecoregion footprint in geographic coordinates.

    Attributes:
        mask: Two-dimensional Boolean mask in EPSG:4326.
        transform: Affine transform from mask pixels to longitude and latitude.
        bounds: Bounds of defined mask cells in longitude and latitude.
        source_defined_pixels: Number of defined source-grid cells.
    """

    mask: np.ndarray
    transform: Affine
    bounds: BoundingBox
    source_defined_pixels: int


@dataclass(frozen=True)
class LocationFigureSummary:
    """Metadata describing a generated ecoregion locator figure.

    Attributes:
        path: Saved figure path.
        ecoregion_name: Label shown in the figure.
        bounds: Mapped footprint bounds in longitude and latitude.
        display_width: Number of columns in the coarsened display mask.
        display_height: Number of rows in the coarsened display mask.
        display_defined_pixels: Number of defined cells in the display mask.
    """

    path: Path
    ecoregion_name: str
    bounds: BoundingBox
    display_width: int
    display_height: int
    display_defined_pixels: int


@dataclass(frozen=True)
class SamplingClassSummary:
    """Sampling measurements for one binary target value.

    Attributes:
        target_value: Binary reference-site value represented by this class.
        available_pixels: Eligible source pixels before sampling.
        sampled_pixels: Pixels retained in the sample table.
        available_area_square_meters: Source area represented by the class.
        weighted_pixels: Source pixel count reconstructed from sampling weights.
        weighted_area_square_meters: Source area estimated from area weights.
        blocks_with_class: Number of sampling blocks containing the class.
        minimum_sampling_weight: Smallest sampling weight in this class.
        maximum_sampling_weight: Largest sampling weight in this class.
    """

    target_value: int
    available_pixels: int
    sampled_pixels: int
    available_area_square_meters: float
    weighted_pixels: float
    weighted_area_square_meters: float
    blocks_with_class: int
    minimum_sampling_weight: float
    maximum_sampling_weight: float


@dataclass(frozen=True)
class SpatialSample:
    """A spatially balanced sample and diagnostics from one ecoregion raster.

    Attributes:
        table: Model-ready table of selected pixels, weights, and raster bands.
        target_band_name: Reference-site band used to construct the target.
        ignored_reference_band_names: Additional reference-site bands excluded
            from the predictor table.
        predictor_band_names: Non-reference raster bands written as columns.
        predictor_defined_pixels: Defined sampled values for each predictor.
        complete_predictor_rows: Sampled rows defined in every predictor.
        block_size_meters: Width and height of each equal-area sampling block.
        samples_per_class_per_block: Per-block cap applied separately to zeroes
            and ones.
        random_seed: Seed used for reproducible random selection.
        block_count: Number of sampling blocks covering eligible pixels.
        reference_block_count: Blocks containing at least one reference pixel.
        nonreference_block_count: Blocks containing at least one non-reference
            pixel.
        minimum_available_pixels_per_block: Smallest eligible block population.
        median_available_pixels_per_block: Median eligible block population.
        maximum_available_pixels_per_block: Largest eligible block population.
        excluded_reference_pixels: Reference pixels lacking every predictor and
            therefore excluded from the modeling domain.
        class_summaries: Diagnostics for non-reference and reference pixels.
        elapsed_seconds: Time used to construct the sample in memory.
    """

    table: pd.DataFrame
    target_band_name: str
    ignored_reference_band_names: tuple[str, ...]
    predictor_band_names: tuple[str, ...]
    predictor_defined_pixels: tuple[int, ...]
    complete_predictor_rows: int
    block_size_meters: float
    samples_per_class_per_block: int
    random_seed: int
    block_count: int
    reference_block_count: int
    nonreference_block_count: int
    minimum_available_pixels_per_block: int
    median_available_pixels_per_block: float
    maximum_available_pixels_per_block: int
    excluded_reference_pixels: int
    class_summaries: tuple[SamplingClassSummary, SamplingClassSummary]
    elapsed_seconds: float


@dataclass(frozen=True)
class ParquetWriteSummary:
    """Verified metadata for a written Parquet sample table.

    Attributes:
        path: Absolute path to the generated Parquet file.
        rows: Rows reported by Parquet metadata.
        columns: Columns reported by Parquet metadata.
        row_groups: Number of Parquet row groups.
        compression: Compression codec reported for the first data column.
        file_bytes: On-disk file size.
        elapsed_seconds: Time used to write and verify the file.
    """

    path: Path
    rows: int
    columns: int
    row_groups: int
    compression: str
    file_bytes: int
    elapsed_seconds: float


def _positive_float(value: str) -> float:
    """Parse a positive floating-point command-line value.

    Args:
        value: User-provided argument text.

    Returns:
        Parsed positive floating-point value.

    Raises:
        argparse.ArgumentTypeError: If the value is not finite and positive.
    """

    try:
        parsed = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a number") from error
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def _positive_int(value: str) -> int:
    """Parse a positive integer command-line value.

    Args:
        value: User-provided argument text.

    Returns:
        Parsed positive integer.

    Raises:
        argparse.ArgumentTypeError: If the value is not a positive integer.
    """

    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer") from error
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed command-line namespace.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Load every band and pixel from one ecoregion GeoTIFF into memory "
            "and print coverage diagnostics."
        )
    )
    parser.add_argument("geotiff", type=Path, help="Multiband GeoTIFF to load.")
    parser.add_argument(
        "--no-band-report",
        action="store_true",
        help="Skip the per-band coverage and value table.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bars.",
    )
    parser.add_argument(
        "--location-figure",
        type=Path,
        help=(
            "Output path for the location map (.png, .pdf, or .svg). Defaults "
            "to outputs/figures/<ecoregion>_world_location.png."
        ),
    )
    parser.add_argument(
        "--no-location-figure",
        action="store_true",
        help="Skip generation of the world location map.",
    )
    parser.add_argument(
        "--sample-output",
        type=Path,
        help=(
            "Output path for the spatial sample (.parquet). Defaults to "
            "outputs/samples/<ecoregion>_spatial_sample.parquet."
        ),
    )
    parser.add_argument(
        "--sampling-block-size-m",
        type=_positive_float,
        default=DEFAULT_SAMPLING_BLOCK_SIZE_METERS,
        help=(
            "Square sampling-block size in meters "
            f"(default: {DEFAULT_SAMPLING_BLOCK_SIZE_METERS:g})."
        ),
    )
    parser.add_argument(
        "--samples-per-class-per-block",
        type=_positive_int,
        default=DEFAULT_SAMPLES_PER_CLASS_PER_BLOCK,
        help=(
            "Maximum sampled pixels for each binary target value in each "
            "block "
            f"(default: {DEFAULT_SAMPLES_PER_CLASS_PER_BLOCK})."
        ),
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help=f"Random sampling seed (default: {DEFAULT_RANDOM_SEED}).",
    )
    parser.add_argument(
        "--no-sampling",
        action="store_true",
        help="Skip spatial sampling and Parquet generation.",
    )
    return parser.parse_args()


def _band_names(descriptions: Sequence[str | None]) -> tuple[str, ...]:
    """Build stable names from optional raster band descriptions.

    Args:
        descriptions: Band descriptions in source order.

    Returns:
        Nonempty band name for every source band.
    """

    return tuple(
        description or f"band_{index:02d}"
        for index, description in enumerate(descriptions, start=1)
    )


def _common_dtype(source_dtypes: Sequence[str]) -> np.dtype:
    """Choose one NumPy dtype capable of holding every source band.

    Args:
        source_dtypes: Rasterio dtype names in source band order.

    Returns:
        Common NumPy dtype for the in-memory value cube.
    """

    return np.dtype(np.result_type(*(np.dtype(dtype) for dtype in source_dtypes)))


def load_raster_pixels(
    geotiff_path: Path,
    *,
    show_progress: bool,
) -> RasterPixelData:
    """Load every pixel value and validity flag from a GeoTIFF.

    Each source band is read separately so loading progress remains visible.
    Values retain their common source dtype, while a separate Boolean array
    preserves band-specific masks for later target/background stratification.

    Args:
        geotiff_path: GeoTIFF path to read.
        show_progress: Whether to display a tqdm band-loading progress bar.

    Returns:
        Fully loaded raster values, validity flags, and spatial metadata.

    Raises:
        FileNotFoundError: If ``geotiff_path`` does not exist.
        ValueError: If the raster has no bands.
        RuntimeError: If the arrays cannot be allocated in memory.
    """

    path = geotiff_path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"GeoTIFF does not exist: {path}")

    with rasterio.open(path) as source:
        if source.count < 1:
            raise ValueError(f"GeoTIFF has no raster bands: {path}")

        source_dtypes = tuple(source.dtypes)
        common_dtype = _common_dtype(source_dtypes)
        shape = (source.count, source.height, source.width)
        try:
            values = np.empty(shape, dtype=common_dtype)
            validity = np.empty(shape, dtype=np.bool_)
        except MemoryError as error:
            estimated_bytes = math.prod(shape) * (
                common_dtype.itemsize + np.dtype(np.bool_).itemsize
            )
            raise RuntimeError(
                "Could not allocate approximately "
                f"{estimated_bytes / MEBIBYTE:,.1f} MiB for {path}."
            ) from error

        floating_values = np.issubdtype(common_dtype, np.floating)
        band_indices = range(1, source.count + 1)
        for band_index in tqdm(
            band_indices,
            total=source.count,
            desc="Loading raster bands",
            unit="band",
            disable=not show_progress,
        ):
            source_band = source.read(band_index, masked=True)
            band_values = np.asarray(np.ma.getdata(source_band), dtype=common_dtype)
            band_validity = ~np.ma.getmaskarray(source_band)
            if floating_values:
                band_validity &= np.isfinite(band_values)
                values[band_index - 1] = np.where(
                    band_validity,
                    band_values,
                    np.nan,
                )
            else:
                values[band_index - 1] = np.where(
                    band_validity,
                    band_values,
                    0,
                )
            validity[band_index - 1] = band_validity

        return RasterPixelData(
            path=path,
            values=values,
            validity=validity,
            band_names=_band_names(source.descriptions),
            source_dtypes=source_dtypes,
            nodata_values=tuple(source.nodatavals),
            transform=source.transform,
            crs=source.crs,
            bounds=source.bounds,
        )


def pixel_area_by_row_square_meters(
    raster: RasterPixelData,
) -> np.ndarray | None:
    """Estimate one-pixel area for every raster row.

    Projected rasters use the affine transform determinant and the CRS linear
    unit conversion. North-up geographic rasters use a spherical Earth area
    calculation so longitude-width shrinkage with latitude is represented.

    Args:
        raster: Loaded raster and spatial metadata.

    Returns:
        One square-meter pixel area per raster row, or ``None`` when the CRS is
        missing or a geographic grid is rotated.
    """

    if raster.crs is None:
        return None

    if raster.crs.is_projected:
        pixel_area_crs_units = abs(
            raster.transform.a * raster.transform.e
            - raster.transform.b * raster.transform.d
        )
        try:
            _, meters_per_unit = raster.crs.linear_units_factor
        except (TypeError, ValueError):
            return None
        pixel_area_square_meters = pixel_area_crs_units * meters_per_unit**2
        return np.full(raster.height, pixel_area_square_meters, dtype=np.float64)

    if not raster.crs.is_geographic:
        return None
    if not (
        math.isclose(raster.transform.b, 0.0, abs_tol=1e-12)
        and math.isclose(raster.transform.d, 0.0, abs_tol=1e-12)
    ):
        return None

    latitude_edges = (
        raster.transform.f
        + np.arange(raster.height + 1, dtype=np.float64) * raster.transform.e
    )
    latitude_edges = np.clip(latitude_edges, -90.0, 90.0)
    longitude_width_radians = math.radians(abs(raster.transform.a))
    sine_latitudes = np.sin(np.radians(latitude_edges))
    return (
        EARTH_RADIUS_METERS**2
        * longitude_width_radians
        * np.abs(np.diff(sine_latitudes))
    )


def _reference_band_offsets(band_names: Sequence[str]) -> tuple[int, ...]:
    """Locate exported grassland reference-site bands.

    Args:
        band_names: Raster band descriptions in source order.

    Returns:
        Zero-based offsets of all reference-site bands.

    Raises:
        ValueError: If no reference-site band can be identified.
    """

    offsets = tuple(
        offset
        for offset, name in enumerate(band_names)
        if name.lower() == "reference_sites"
        or name.lower().endswith("_grassland_reference_sites")
    )
    if not offsets:
        raise ValueError(
            "Could not find a Grassland Reference Sites band. Available bands: "
            + ", ".join(band_names)
        )
    return offsets


def assign_sampling_blocks(
    x_meters: np.ndarray,
    y_meters: np.ndarray,
    block_size_meters: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Assign equal-area coordinates to stable square sampling blocks.

    Block boundaries are anchored at the projected coordinate-system origin.
    Returned block IDs are one-based and sorted by block-column and block-row
    indices, making the same coordinates produce the same IDs in repeated runs.

    Args:
        x_meters: One-dimensional projected x coordinates in meters.
        y_meters: One-dimensional projected y coordinates in meters.
        block_size_meters: Positive square block width and height.

    Returns:
        Tuple containing dense block IDs, global block-column indices, and
        global block-row indices for every coordinate.

    Raises:
        ValueError: If coordinate arrays are invalid or block size is not
            finite and positive.
    """

    x_values = np.asarray(x_meters, dtype=np.float64)
    y_values = np.asarray(y_meters, dtype=np.float64)
    if x_values.ndim != 1 or y_values.ndim != 1:
        raise ValueError("Sampling coordinates must be one-dimensional arrays.")
    if x_values.shape != y_values.shape:
        raise ValueError("Sampling coordinate arrays must have matching shapes.")
    if x_values.size == 0:
        raise ValueError("No eligible pixel coordinates were available for sampling.")
    if not np.all(np.isfinite(x_values)) or not np.all(np.isfinite(y_values)):
        raise ValueError("Sampling coordinates must all be finite.")
    if not math.isfinite(block_size_meters) or block_size_meters <= 0:
        raise ValueError("Sampling block size must be finite and greater than zero.")

    block_columns = np.floor(x_values / block_size_meters).astype(np.int64)
    block_rows = np.floor(y_values / block_size_meters).astype(np.int64)
    block_pairs = np.empty(
        x_values.size,
        dtype=[("column", np.int64), ("row", np.int64)],
    )
    block_pairs["column"] = block_columns
    block_pairs["row"] = block_rows
    _, inverse = np.unique(block_pairs, return_inverse=True)
    return inverse.astype(np.int64) + 1, block_columns, block_rows


def _transformed_coordinates(
    raster: RasterPixelData,
    flat_pixel_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Transform selected raster-cell centers to geographic and metric CRSs.

    Args:
        raster: Loaded raster and spatial metadata.
        flat_pixel_indices: Row-major source pixel indices to transform.

    Returns:
        Longitude, latitude, equal-area x, and equal-area y coordinate arrays.

    Raises:
        ValueError: If the raster CRS is missing or transformed coordinates are
            not finite.
    """

    if raster.crs is None:
        raise ValueError("Spatial sampling requires a defined raster CRS.")

    rows = flat_pixel_indices // raster.width
    columns = flat_pixel_indices % raster.width
    column_centers = columns.astype(np.float64) + 0.5
    row_centers = rows.astype(np.float64) + 0.5
    selected_x = (
        raster.transform.c
        + column_centers * raster.transform.a
        + row_centers * raster.transform.b
    )
    selected_y = (
        raster.transform.f
        + column_centers * raster.transform.d
        + row_centers * raster.transform.e
    )
    source_crs = raster.crs.to_wkt()
    geographic_transformer = Transformer.from_crs(
        source_crs,
        "EPSG:4326",
        always_xy=True,
    )
    equal_area_transformer = Transformer.from_crs(
        source_crs,
        EQUAL_AREA_CRS,
        always_xy=True,
    )
    longitudes, latitudes = geographic_transformer.transform(selected_x, selected_y)
    x_meters, y_meters = equal_area_transformer.transform(selected_x, selected_y)
    arrays = tuple(
        np.asarray(values, dtype=np.float64)
        for values in (longitudes, latitudes, x_meters, y_meters)
    )
    if any(not np.all(np.isfinite(values)) for values in arrays):
        raise ValueError(
            "Could not transform every eligible pixel to finite coordinates."
        )
    return arrays


def create_spatial_sample(
    raster: RasterPixelData,
    block_size_meters: float,
    samples_per_class_per_block: int,
    random_seed: int,
    show_progress: bool,
) -> SpatialSample:
    """Create a deterministic spatially balanced sample of raster pixels.

    Earth Engine masks zeroes from the exported reference-site bands. Within
    the usable predictor footprint, a defined reference value of one becomes
    target one and every other pixel becomes target zero. Reference and
    non-reference pixels are sampled separately inside each equal-area block.

    Args:
        raster: Fully loaded ecoregion raster.
        block_size_meters: Width and height of square sampling blocks.
        samples_per_class_per_block: Maximum pixels retained for each target
            value within each block.
        random_seed: Seed for reproducible sampling without replacement.
        show_progress: Whether to display tqdm progress bars.

    Returns:
        Sample table and diagnostics describing its source population.

    Raises:
        ValueError: If bands, target values, pixel areas, or coordinates cannot
            support sampling.
        RuntimeError: If no eligible pixels remain in the predictor footprint.
    """

    started = time.perf_counter()
    if samples_per_class_per_block <= 0:
        raise ValueError("Samples per class per block must be greater than zero.")

    reference_offsets = _reference_band_offsets(raster.band_names)
    target_offset = reference_offsets[0]
    predictor_offsets = tuple(
        offset for offset in range(raster.band_count) if offset not in reference_offsets
    )
    if not predictor_offsets:
        raise ValueError("Spatial sampling requires at least one non-reference band.")

    predictor_names = tuple(raster.band_names[offset] for offset in predictor_offsets)
    if len(set(predictor_names)) != len(predictor_names):
        raise ValueError(
            "Predictor band descriptions must be unique for Parquet output."
        )
    reserved_columns = {
        "row",
        "column",
        "longitude",
        "latitude",
        "sampling_block_id",
        "sampling_block_column",
        "sampling_block_row",
        "reference_site",
        "pixel_area_m2",
        "available_pixels_in_block_class",
        "sampled_pixels_in_block_class",
        "sampling_probability",
        "sampling_weight",
        "area_weight_m2",
    }
    conflicting_names = sorted(reserved_columns.intersection(predictor_names))
    if conflicting_names:
        raise ValueError(
            "Predictor band names conflict with sampling metadata columns: "
            + ", ".join(conflicting_names)
        )

    target_values = raster.values[target_offset]
    target_validity = raster.validity[target_offset]
    defined_target_values = target_values[target_validity]
    unexpected_values = np.unique(
        defined_target_values[
            (defined_target_values != 0) & (defined_target_values != 1)
        ]
    )
    if unexpected_values.size:
        rendered = ", ".join(str(value) for value in unexpected_values[:10])
        raise ValueError(
            "Grassland Reference Sites must contain only zero and one; found "
            f"{rendered}."
        )

    reference_mask = target_validity & (target_values == 1)
    predictor_domain = np.zeros((raster.height, raster.width), dtype=np.bool_)
    for offset in tqdm(
        predictor_offsets,
        total=len(predictor_offsets),
        desc="Building predictor footprint",
        unit="band",
        disable=not show_progress,
    ):
        np.logical_or(
            predictor_domain,
            raster.validity[offset],
            out=predictor_domain,
        )
    excluded_reference_pixels = int(
        np.count_nonzero(reference_mask & ~predictor_domain)
    )
    eligible_flat_indices = np.flatnonzero(predictor_domain.ravel())
    if eligible_flat_indices.size == 0:
        raise RuntimeError("No pixels contain a defined non-reference predictor value.")

    pixel_area_by_row = pixel_area_by_row_square_meters(raster)
    if pixel_area_by_row is None:
        raise ValueError("Could not calculate square-meter pixel areas for sampling.")

    preparation_progress = tqdm(
        total=3,
        desc="Preparing sampling grid",
        unit="step",
        disable=not show_progress,
    )
    rows = eligible_flat_indices // raster.width
    columns = eligible_flat_indices % raster.width
    pixel_areas = pixel_area_by_row[rows]
    targets = reference_mask.ravel()[eligible_flat_indices].astype(np.uint8)
    preparation_progress.update()

    longitudes, latitudes, x_meters, y_meters = _transformed_coordinates(
        raster,
        eligible_flat_indices,
    )
    preparation_progress.update()
    block_ids, block_columns, block_rows = assign_sampling_blocks(
        x_meters,
        y_meters,
        block_size_meters,
    )
    preparation_progress.update()
    preparation_progress.close()

    block_count = int(np.max(block_ids))
    group_keys = (block_ids - 1) * 2 + targets
    order = np.argsort(group_keys, kind="stable")
    sorted_group_keys = group_keys[order]
    unique_group_keys, starts, available_counts = np.unique(
        sorted_group_keys,
        return_index=True,
        return_counts=True,
    )
    sampled_counts = np.minimum(available_counts, samples_per_class_per_block)
    available_by_group = np.zeros(block_count * 2, dtype=np.int64)
    sampled_by_group = np.zeros(block_count * 2, dtype=np.int64)
    available_by_group[unique_group_keys] = available_counts
    sampled_by_group[unique_group_keys] = sampled_counts

    selected_positions = np.empty(int(np.sum(sampled_counts)), dtype=np.int64)
    random_generator = np.random.default_rng(random_seed)
    cursor = 0
    group_iterator = zip(starts, available_counts, sampled_counts, strict=True)
    for start, available_count, sampled_count in tqdm(
        group_iterator,
        total=unique_group_keys.size,
        desc="Sampling block classes",
        unit="stratum",
        disable=not show_progress,
    ):
        group_positions = order[start : start + available_count]
        if sampled_count < available_count:
            chosen = random_generator.choice(
                group_positions,
                size=sampled_count,
                replace=False,
            )
        else:
            chosen = group_positions
        selected_positions[cursor : cursor + sampled_count] = chosen
        cursor += sampled_count
    selected_positions.sort()

    selected_flat_indices = eligible_flat_indices[selected_positions]
    selected_targets = targets[selected_positions]
    selected_group_keys = group_keys[selected_positions]
    selected_available_counts = available_by_group[selected_group_keys]
    selected_sampled_counts = sampled_by_group[selected_group_keys]
    sampling_probabilities = selected_sampled_counts / selected_available_counts
    sampling_weights = selected_available_counts / selected_sampled_counts
    selected_pixel_areas = pixel_areas[selected_positions]
    area_weights = selected_pixel_areas * sampling_weights

    table_columns: dict[str, np.ndarray] = {
        "row": rows[selected_positions].astype(np.int32),
        "column": columns[selected_positions].astype(np.int32),
        "longitude": longitudes[selected_positions],
        "latitude": latitudes[selected_positions],
        "sampling_block_id": block_ids[selected_positions],
        "sampling_block_column": block_columns[selected_positions],
        "sampling_block_row": block_rows[selected_positions],
        "reference_site": selected_targets,
        "pixel_area_m2": selected_pixel_areas,
        "available_pixels_in_block_class": selected_available_counts,
        "sampled_pixels_in_block_class": selected_sampled_counts,
        "sampling_probability": sampling_probabilities,
        "sampling_weight": sampling_weights,
        "area_weight_m2": area_weights,
    }

    pixel_values = raster.pixel_values()
    pixel_validity = raster.pixel_validity()
    predictor_defined_pixels = []
    complete_predictor_mask = np.ones(selected_positions.size, dtype=np.bool_)
    for offset, name in tqdm(
        zip(predictor_offsets, predictor_names, strict=True),
        total=len(predictor_offsets),
        desc="Building predictor columns",
        unit="band",
        disable=not show_progress,
    ):
        valid_values = pixel_validity[selected_flat_indices, offset]
        predictor_values = np.asarray(
            pixel_values[selected_flat_indices, offset],
            dtype=np.float64,
        ).copy()
        predictor_values[~valid_values] = np.nan
        table_columns[name] = predictor_values
        predictor_defined_pixels.append(int(np.count_nonzero(valid_values)))
        complete_predictor_mask &= valid_values

    table = pd.DataFrame(table_columns, copy=False)
    class_summaries = []
    for target_value in (0, 1):
        available_mask = targets == target_value
        sampled_mask = selected_targets == target_value
        class_sampling_weights = sampling_weights[sampled_mask]
        class_summaries.append(
            SamplingClassSummary(
                target_value=target_value,
                available_pixels=int(np.count_nonzero(available_mask)),
                sampled_pixels=int(np.count_nonzero(sampled_mask)),
                available_area_square_meters=float(np.sum(pixel_areas[available_mask])),
                weighted_pixels=float(np.sum(class_sampling_weights)),
                weighted_area_square_meters=float(np.sum(area_weights[sampled_mask])),
                blocks_with_class=int(np.unique(block_ids[available_mask]).size),
                minimum_sampling_weight=(
                    float(np.min(class_sampling_weights))
                    if class_sampling_weights.size
                    else math.nan
                ),
                maximum_sampling_weight=(
                    float(np.max(class_sampling_weights))
                    if class_sampling_weights.size
                    else math.nan
                ),
            )
        )

    block_populations = np.bincount(block_ids)[1:]
    return SpatialSample(
        table=table,
        target_band_name=raster.band_names[target_offset],
        ignored_reference_band_names=tuple(
            raster.band_names[offset] for offset in reference_offsets[1:]
        ),
        predictor_band_names=predictor_names,
        predictor_defined_pixels=tuple(predictor_defined_pixels),
        complete_predictor_rows=int(np.count_nonzero(complete_predictor_mask)),
        block_size_meters=block_size_meters,
        samples_per_class_per_block=samples_per_class_per_block,
        random_seed=random_seed,
        block_count=block_count,
        reference_block_count=class_summaries[1].blocks_with_class,
        nonreference_block_count=class_summaries[0].blocks_with_class,
        minimum_available_pixels_per_block=int(np.min(block_populations)),
        median_available_pixels_per_block=float(np.median(block_populations)),
        maximum_available_pixels_per_block=int(np.max(block_populations)),
        excluded_reference_pixels=excluded_reference_pixels,
        class_summaries=(class_summaries[0], class_summaries[1]),
        elapsed_seconds=time.perf_counter() - started,
    )


def write_spatial_sample_parquet(
    sample: SpatialSample,
    output_path: Path,
    show_progress: bool,
) -> ParquetWriteSummary:
    """Write and metadata-verify a compressed Parquet sample table.

    Args:
        sample: Spatial sample to serialize.
        output_path: Destination path ending in ``.parquet``.
        show_progress: Whether to display a tqdm stage progress bar.

    Returns:
        Verified Parquet metadata and write measurements.

    Raises:
        ValueError: If the destination does not use the Parquet suffix.
        RuntimeError: If written row or column counts do not match the sample.
    """

    started = time.perf_counter()
    path = output_path.expanduser().resolve()
    if path.suffix.lower() != ".parquet":
        raise ValueError(f"Sample output must end in .parquet: {path}")

    progress = tqdm(
        total=3,
        desc="Writing Parquet sample",
        unit="step",
        disable=not show_progress,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    progress.update()
    sample.table.to_parquet(path, engine="pyarrow", compression="zstd", index=False)
    progress.update()

    parquet_file = pq.ParquetFile(path)
    metadata = parquet_file.metadata
    if metadata.num_rows != len(sample.table):
        raise RuntimeError(
            f"Parquet row verification failed: expected {len(sample.table):,}, "
            f"found {metadata.num_rows:,}."
        )
    if metadata.num_columns != sample.table.shape[1]:
        raise RuntimeError(
            "Parquet column verification failed: expected "
            f"{sample.table.shape[1]:,}, found {metadata.num_columns:,}."
        )
    if parquet_file.schema_arrow.names != list(sample.table.columns):
        raise RuntimeError("Parquet column-name verification failed.")
    compression = metadata.row_group(0).column(0).compression
    progress.update()
    progress.close()
    return ParquetWriteSummary(
        path=path,
        rows=metadata.num_rows,
        columns=metadata.num_columns,
        row_groups=metadata.num_row_groups,
        compression=compression,
        file_bytes=path.stat().st_size,
        elapsed_seconds=time.perf_counter() - started,
    )


def summarize_coverage(
    mask: np.ndarray,
    pixel_area_by_row: np.ndarray | None,
) -> CoverageSummary:
    """Summarize pixel count, proportion, and area for a validity mask.

    Args:
        mask: Two-dimensional Boolean pixel mask.
        pixel_area_by_row: Square-meter pixel area for each mask row, or
            ``None`` when area is unavailable.

    Returns:
        Coverage measurements for the supplied mask.

    Raises:
        ValueError: If ``mask`` is not two-dimensional or row-area dimensions
            do not match the mask.
    """

    if mask.ndim != 2:
        raise ValueError("Coverage masks must be two-dimensional.")
    if pixel_area_by_row is not None and len(pixel_area_by_row) != mask.shape[0]:
        raise ValueError("Pixel-area row count does not match the coverage mask.")

    defined_pixels = int(np.count_nonzero(mask))
    total_pixels = int(mask.size)
    defined_percent = 100.0 * defined_pixels / total_pixels if total_pixels else 0.0
    area_square_kilometers = None
    if pixel_area_by_row is not None:
        pixels_by_row = np.count_nonzero(mask, axis=1)
        area_square_kilometers = float(
            np.sum(pixels_by_row * pixel_area_by_row) / 1_000_000.0
        )
    return CoverageSummary(
        defined_pixels=defined_pixels,
        total_pixels=total_pixels,
        defined_percent=defined_percent,
        area_square_kilometers=area_square_kilometers,
    )


def summarize_bands(
    raster: RasterPixelData,
    pixel_area_by_row: np.ndarray | None,
    *,
    show_progress: bool,
) -> list[BandSummary]:
    """Calculate coverage and descriptive statistics for every band.

    Args:
        raster: Fully loaded raster values and masks.
        pixel_area_by_row: Square-meter pixel area for each raster row, or
            ``None`` when area is unavailable.
        show_progress: Whether to display a tqdm summarization progress bar.

    Returns:
        Band summaries in source order.
    """

    summaries: list[BandSummary] = []
    band_offsets = range(raster.band_count)
    for band_offset in tqdm(
        band_offsets,
        total=raster.band_count,
        desc="Summarizing raster bands",
        unit="band",
        disable=not show_progress,
    ):
        band_validity = raster.validity[band_offset]
        coverage = summarize_coverage(band_validity, pixel_area_by_row)
        if coverage.defined_pixels:
            defined_values = raster.values[band_offset][band_validity]
            minimum = float(np.min(defined_values))
            mean = float(np.mean(defined_values, dtype=np.float64))
            maximum = float(np.max(defined_values))
        else:
            minimum = None
            mean = None
            maximum = None
        summaries.append(
            BandSummary(
                index=band_offset + 1,
                name=raster.band_names[band_offset],
                coverage=coverage,
                minimum=minimum,
                mean=mean,
                maximum=maximum,
            )
        )
    return summaries


def infer_ecoregion_name(geotiff_path: Path) -> str:
    """Infer a readable ecoregion name from an Earth Engine export filename.

    Earth Engine exports in this project place a numeric ecoregion identifier
    and response-variable suffix after the ecoregion name. The source name can
    be truncated by export naming limits, so callers can override this inferred
    label through the command line.

    Args:
        geotiff_path: GeoTIFF path whose filename should be interpreted.

    Returns:
        Human-readable ecoregion label.
    """

    name_stem = geotiff_path.stem
    name_stem = re.sub(
        r"_e\d+(?:_response_variables.*)?$",
        "",
        name_stem,
        flags=re.IGNORECASE,
    )
    name_stem = re.sub(
        r"_response_variables.*$",
        "",
        name_stem,
        flags=re.IGNORECASE,
    )
    words = re.sub(r"[_-]+", " ", name_stem).strip()
    return words.title() or "Ecoregion"


def _geographic_footprint(raster: RasterPixelData) -> GeographicFootprint:
    """Coarsen and reproject the raster's defined footprint to EPSG:4326.

    Maximum-value resampling keeps a destination cell defined when any source
    cell contributing to it is defined. This preserves small and fragmented
    ecoregion parts while limiting figure-generation work.

    Args:
        raster: Fully loaded raster values, validity, and spatial metadata.

    Returns:
        Coarsened Boolean footprint and geographic bounds.

    Raises:
        ValueError: If the raster has no CRS or no defined pixels.
    """

    if raster.crs is None:
        raise ValueError("A CRS is required to generate the world location map.")

    source_mask = np.any(raster.validity, axis=0)
    source_defined_pixels = int(np.count_nonzero(source_mask))
    if source_defined_pixels == 0:
        raise ValueError("The raster has no defined pixels to map.")

    destination_crs = CRS.from_epsg(4326)
    default_transform, default_width, default_height = calculate_default_transform(
        raster.crs,
        destination_crs,
        raster.width,
        raster.height,
        *raster.bounds,
    )
    largest_dimension = max(default_width, default_height)
    scale = max(1.0, largest_dimension / MAX_FOOTPRINT_DIMENSION)
    destination_width = max(1, int(math.ceil(default_width / scale)))
    destination_height = max(1, int(math.ceil(default_height / scale)))
    destination_transform = default_transform * Affine.scale(
        default_width / destination_width,
        default_height / destination_height,
    )
    destination_mask = np.zeros(
        (destination_height, destination_width),
        dtype=np.uint8,
    )
    reproject(
        source=source_mask.astype(np.uint8),
        destination=destination_mask,
        src_transform=raster.transform,
        src_crs=raster.crs,
        dst_transform=destination_transform,
        dst_crs=destination_crs,
        src_nodata=0,
        dst_nodata=0,
        resampling=Resampling.max,
    )
    geographic_mask = destination_mask.astype(np.bool_)
    defined_rows, defined_columns = np.nonzero(geographic_mask)
    if len(defined_rows) == 0:
        raise ValueError("The raster footprint became empty during reprojection.")

    minimum_column = int(np.min(defined_columns))
    maximum_column = int(np.max(defined_columns)) + 1
    minimum_row = int(np.min(defined_rows))
    maximum_row = int(np.max(defined_rows)) + 1
    first_corner = destination_transform * (minimum_column, minimum_row)
    second_corner = destination_transform * (maximum_column, maximum_row)
    bounds = BoundingBox(
        left=min(first_corner[0], second_corner[0]),
        bottom=min(first_corner[1], second_corner[1]),
        right=max(first_corner[0], second_corner[0]),
        top=max(first_corner[1], second_corner[1]),
    )
    return GeographicFootprint(
        mask=geographic_mask,
        transform=destination_transform,
        bounds=bounds,
        source_defined_pixels=source_defined_pixels,
    )


def _callout_position(bounds: BoundingBox) -> tuple[float, float]:
    """Choose a map-relative label position opposite the footprint.

    Args:
        bounds: Geographic ecoregion bounds.

    Returns:
        Pair of x and y positions in axes-fraction coordinates.
    """

    center_longitude = (bounds.left + bounds.right) / 2.0
    center_latitude = (bounds.bottom + bounds.top) / 2.0
    label_x = 0.16 if center_longitude >= 0.0 else 0.84
    label_y = 0.20 if center_latitude >= 0.0 else 0.80
    return label_x, label_y


def _locator_bounds(bounds: BoundingBox, minimum_span_degrees: float) -> BoundingBox:
    """Expand small footprint bounds into a visible world-map locator box.

    Args:
        bounds: Geographic ecoregion bounds.
        minimum_span_degrees: Minimum displayed width and height in degrees.

    Returns:
        Geographic bounds enclosing the footprint with a visible minimum span.
    """

    center_longitude = (bounds.left + bounds.right) / 2.0
    center_latitude = (bounds.bottom + bounds.top) / 2.0
    longitude_span = max(bounds.right - bounds.left, minimum_span_degrees)
    latitude_span = max(bounds.top - bounds.bottom, minimum_span_degrees)
    return BoundingBox(
        left=max(-180.0, center_longitude - longitude_span / 2.0),
        bottom=max(-90.0, center_latitude - latitude_span / 2.0),
        right=min(180.0, center_longitude + longitude_span / 2.0),
        top=min(90.0, center_latitude + latitude_span / 2.0),
    )


def create_ecoregion_location_figure(
    raster: RasterPixelData,
    ecoregion_name: str,
    figure_path: Path,
    show_progress: bool,
) -> LocationFigureSummary:
    """Create a publication-quality world locator map for an ecoregion.

    The highlighted footprint is defined by cells valid in at least one raster
    band. The display mask is reprojected and coarsened before plotting, while
    maximum-value resampling retains small disconnected parts.

    Args:
        raster: Fully loaded raster values, validity, and spatial metadata.
        ecoregion_name: Label to show in the map callout.
        figure_path: PNG, PDF, or SVG output path.
        show_progress: Whether to display a tqdm figure-generation progress bar.

    Returns:
        Saved figure path and mapped-footprint metadata.

    Raises:
        ValueError: If the label is empty, the suffix is unsupported, or the
            raster footprint cannot be mapped.
    """

    cleaned_name = ecoregion_name.strip()
    if not cleaned_name:
        raise ValueError("The ecoregion name cannot be empty.")
    path = figure_path.expanduser().resolve()
    if path.suffix.lower() not in SUPPORTED_FIGURE_SUFFIXES:
        supported = ", ".join(sorted(SUPPORTED_FIGURE_SUFFIXES))
        raise ValueError(f"Location figure must use one of: {supported}.")

    progress = tqdm(
        total=4,
        desc="Generating location figure",
        unit="step",
        disable=not show_progress,
    )
    figure = None
    try:
        footprint = _geographic_footprint(raster)
        progress.update()

        style = {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titleweight": "bold",
            "axes.titlesize": 16,
        }
        with plt.rc_context(style):
            figure = plt.figure(figsize=(12.0, 6.4), facecolor="white")
            axis = figure.add_subplot(1, 1, 1, projection=ccrs.Robinson())
            axis.set_global()
            axis.set_facecolor("#DCEAF1")
            axis.add_feature(
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
            axis.set_title("Global ecoregion location", pad=16)
            progress.update()

            longitude_edges = (
                footprint.transform.c
                + np.arange(footprint.mask.shape[1] + 1) * footprint.transform.a
            )
            latitude_edges = (
                footprint.transform.f
                + np.arange(footprint.mask.shape[0] + 1) * footprint.transform.e
            )
            highlighted_mask = np.ma.masked_where(
                ~footprint.mask,
                np.ones(footprint.mask.shape, dtype=np.uint8),
            )
            axis.pcolormesh(
                longitude_edges,
                latitude_edges,
                highlighted_mask,
                cmap=ListedColormap(["#D1493F"]),
                vmin=0,
                vmax=1,
                shading="flat",
                transform=ccrs.PlateCarree(),
                alpha=0.88,
                zorder=4,
            )

            bounds = footprint.bounds
            locator_bounds = _locator_bounds(bounds, 5.0)
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
            center_longitude = (bounds.left + bounds.right) / 2.0
            center_latitude = (bounds.bottom + bounds.top) / 2.0
            axis.plot(
                center_longitude,
                center_latitude,
                marker="o",
                markersize=5,
                markerfacecolor="#D1493F",
                markeredgecolor="#161A1D",
                markeredgewidth=0.9,
                transform=ccrs.PlateCarree(),
                zorder=6,
            )
            label_x, label_y = _callout_position(bounds)
            axis.annotate(
                textwrap.fill(cleaned_name, width=28, break_long_words=False),
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
            legend_handles = [
                Patch(
                    facecolor="#D1493F",
                    edgecolor="none",
                    label="Defined ecoregion footprint",
                ),
                Line2D(
                    [0],
                    [0],
                    color="#161A1D",
                    linewidth=1.25,
                    linestyle=(0, (4, 2)),
                    label="Ecoregion locator box",
                ),
            ]
            axis.legend(
                handles=legend_handles,
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
            figure.text(
                0.99,
                0.015,
                "Base map: Natural Earth 1:110m",
                ha="right",
                va="bottom",
                fontsize=7.5,
                color="#596268",
            )
            figure.subplots_adjust(left=0.025, right=0.975, bottom=0.09, top=0.90)
            progress.update()

            path.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(
                path,
                dpi=LOCATION_FIGURE_DPI,
                bbox_inches="tight",
                facecolor=figure.get_facecolor(),
            )
            progress.update()
    finally:
        if figure is not None:
            plt.close(figure)
        progress.close()

    return LocationFigureSummary(
        path=path,
        ecoregion_name=cleaned_name,
        bounds=footprint.bounds,
        display_width=footprint.mask.shape[1],
        display_height=footprint.mask.shape[0],
        display_defined_pixels=int(np.count_nonzero(footprint.mask)),
    )


def print_location_figure_report(summary: LocationFigureSummary) -> None:
    """Print output and footprint details for a location figure.

    Args:
        summary: Generated figure metadata.
    """

    print()
    print("Location figure")
    print(f"Path: {summary.path}")
    print(f"Ecoregion label: {summary.ecoregion_name}")
    print(
        "Geographic bounds: "
        f"west={summary.bounds.left:.4f}, south={summary.bounds.bottom:.4f}, "
        f"east={summary.bounds.right:.4f}, north={summary.bounds.top:.4f}"
    )
    print(
        "Display footprint: "
        f"{summary.display_defined_pixels:,} defined cells on a "
        f"{summary.display_width:,} x {summary.display_height:,} coarsened grid"
    )


def _format_area(area_square_kilometers: float | None) -> str:
    """Format an optional area for report output.

    Args:
        area_square_kilometers: Area in square kilometers, or ``None``.

    Returns:
        Human-readable area value.
    """

    if area_square_kilometers is None:
        return "unavailable"
    return f"{area_square_kilometers:,.3f} km^2"


def _format_statistic(value: float | None) -> str:
    """Format an optional numeric band statistic.

    Args:
        value: Numeric statistic, or ``None``.

    Returns:
        Compact statistic value for tabular output.
    """

    if value is None:
        return "n/a"
    return f"{value:.6g}"


def _format_nodata_values(nodata_values: Sequence[float | None]) -> str:
    """Format the distinct declared nodata values.

    Args:
        nodata_values: Per-band nodata values from the source raster.

    Returns:
        Compact description of declared nodata values.
    """

    rendered_values = []
    for value in nodata_values:
        rendered = "none" if value is None else _format_statistic(float(value))
        if rendered not in rendered_values:
            rendered_values.append(rendered)
    return ", ".join(rendered_values)


def _print_coverage(label: str, summary: CoverageSummary) -> None:
    """Print one labeled coverage summary.

    Args:
        label: Human-readable coverage category.
        summary: Coverage measurements to print.
    """

    print(
        f"{label}: {summary.defined_pixels:,} / {summary.total_pixels:,} pixels "
        f"({summary.defined_percent:.2f}%), "
        f"approx. area {_format_area(summary.area_square_kilometers)}"
    )


def _format_weight(value: float) -> str:
    """Format a sampling weight or missing weight marker.

    Args:
        value: Sampling weight, potentially ``NaN`` for an empty class.

    Returns:
        Compact weight text.
    """

    if not math.isfinite(value):
        return "n/a"
    return f"{value:,.3f}"


def print_spatial_sampling_report(sample: SpatialSample) -> None:
    """Print detailed diagnostics for a completed spatial sample.

    Args:
        sample: In-memory sample and source-population measurements.
    """

    total_available = sum(
        summary.available_pixels for summary in sample.class_summaries
    )
    total_sampled = len(sample.table)
    table_memory = int(sample.table.memory_usage(index=True, deep=True).sum())

    print()
    print("Spatial sampling report")
    print(f"Target band: {sample.target_band_name}")
    if sample.ignored_reference_band_names:
        print(
            "Ignored duplicate reference band(s): "
            + ", ".join(sample.ignored_reference_band_names)
        )
    else:
        print("Ignored duplicate reference band(s): none")
    print("Target interpretation: 1 = reference site; 0 = non-reference site")
    print(f"Equal-area sampling CRS: {EQUAL_AREA_CRS}")
    print(
        f"Block size: {sample.block_size_meters:,.0f} m x "
        f"{sample.block_size_meters:,.0f} m"
    )
    print(
        "Sampling cap: "
        f"{sample.samples_per_class_per_block:,} pixels per target value per block"
    )
    print(f"Random seed: {sample.random_seed}")
    print(f"Eligible source pixels: {total_available:,}")
    print(
        f"Selected pixels: {total_sampled:,} "
        f"({100.0 * total_sampled / total_available:.2f}% retained)"
    )
    print(f"In-memory sample table: {table_memory / MEBIBYTE:,.2f} MiB")
    print(f"Sampling blocks: {sample.block_count:,}")
    print(f"Blocks containing reference sites: {sample.reference_block_count:,}")
    print(f"Blocks containing non-reference sites: {sample.nonreference_block_count:,}")
    print(
        "Eligible pixels per block: "
        f"min={sample.minimum_available_pixels_per_block:,}, "
        f"median={sample.median_available_pixels_per_block:,.1f}, "
        f"max={sample.maximum_available_pixels_per_block:,}"
    )
    print(
        "Reference pixels excluded because every predictor was missing: "
        f"{sample.excluded_reference_pixels:,}"
    )

    print()
    print("Class sampling and weight checks")
    print(
        f"{'Target':<15} {'Available':>12} {'Sampled':>12} {'Retained':>10} "
        f"{'Area km^2':>14} {'Weighted count':>16} {'Area error':>11} "
        f"{'Weight range':>21}"
    )
    for summary in sample.class_summaries:
        retained_percent = (
            100.0 * summary.sampled_pixels / summary.available_pixels
            if summary.available_pixels
            else 0.0
        )
        available_area_km2 = summary.available_area_square_meters / 1_000_000.0
        if summary.available_area_square_meters:
            area_error_percent = (
                100.0
                * (
                    summary.weighted_area_square_meters
                    - summary.available_area_square_meters
                )
                / summary.available_area_square_meters
            )
            area_error = f"{area_error_percent:+.4f}%"
        else:
            area_error = "n/a"
        label = "0 non-reference" if summary.target_value == 0 else "1 reference"
        weight_range = (
            f"{_format_weight(summary.minimum_sampling_weight)}-"
            f"{_format_weight(summary.maximum_sampling_weight)}"
        )
        print(
            f"{label:<15} {summary.available_pixels:>12,} "
            f"{summary.sampled_pixels:>12,} {retained_percent:>9.2f}% "
            f"{available_area_km2:>14,.3f} "
            f"{summary.weighted_pixels:>16,.3f} {area_error:>11} "
            f"{weight_range:>21}"
        )
    for summary in sample.class_summaries:
        count_error = summary.weighted_pixels - summary.available_pixels
        if not math.isclose(count_error, 0.0, abs_tol=1e-8):
            print(
                "WARNING: weighted pixel count differs from the source for "
                f"target {summary.target_value} by {count_error:,.6f}."
            )
    if sample.class_summaries[1].available_pixels == 0:
        print("WARNING: this ecoregion contains no reference-site pixels.")

    sampled_rows = len(sample.table)
    fully_defined = sum(
        count == sampled_rows for count in sample.predictor_defined_pixels
    )
    completely_missing = sum(count == 0 for count in sample.predictor_defined_pixels)
    partially_defined = (
        len(sample.predictor_band_names) - fully_defined - completely_missing
    )
    print()
    print("Sampled predictor coverage")
    print(f"Predictor columns: {len(sample.predictor_band_names):,}")
    print(f"Fully defined predictors: {fully_defined:,}")
    print(f"Partially defined predictors: {partially_defined:,}")
    print(f"Completely missing predictors: {completely_missing:,}")
    print(
        "Rows complete across every predictor: "
        f"{sample.complete_predictor_rows:,} / {sampled_rows:,} "
        f"({100.0 * sample.complete_predictor_rows / sampled_rows:.2f}%)"
    )
    lowest_coverage = sorted(
        zip(
            sample.predictor_defined_pixels,
            sample.predictor_band_names,
            strict=True,
        )
    )[: min(8, len(sample.predictor_band_names))]
    print("Lowest-coverage predictor bands in the sample:")
    for defined_pixels, name in lowest_coverage:
        print(
            f"  {name}: {defined_pixels:,} / {sampled_rows:,} "
            f"({100.0 * defined_pixels / sampled_rows:.2f}%)"
        )
    print(f"Sample construction time: {sample.elapsed_seconds:,.2f} seconds")


def print_parquet_report(summary: ParquetWriteSummary, table_memory_bytes: int) -> None:
    """Print verified metadata for a generated Parquet file.

    Args:
        summary: Verified Parquet metadata and file measurements.
        table_memory_bytes: In-memory pandas table size before serialization.
    """

    compression_ratio = (
        table_memory_bytes / summary.file_bytes if summary.file_bytes else math.nan
    )
    print()
    print("Parquet output report")
    print(f"Path: {summary.path}")
    print(f"Verified dimensions: {summary.rows:,} rows x {summary.columns:,} columns")
    print(f"Row groups: {summary.row_groups:,}")
    print(f"Compression: {summary.compression}")
    print(f"File size: {summary.file_bytes / MEBIBYTE:,.2f} MiB")
    print(f"In-memory-to-file size ratio: {compression_ratio:,.2f}x")
    print(f"Write and verification time: {summary.elapsed_seconds:,.2f} seconds")


def print_raster_report(
    raster: RasterPixelData,
    include_band_report: bool,
    show_progress: bool,
) -> None:
    """Print metadata, memory, coverage, and optional per-band diagnostics.

    Args:
        raster: Fully loaded raster values and validity flags.
        include_band_report: Whether to print coverage and statistics for every
            band.
        show_progress: Whether to display tqdm progress while summarizing bands.
    """

    pixel_areas = pixel_area_by_row_square_meters(raster)
    any_band_coverage = summarize_coverage(
        np.any(raster.validity, axis=0),
        pixel_areas,
    )
    every_band_coverage = summarize_coverage(
        np.all(raster.validity, axis=0),
        pixel_areas,
    )
    grid_area = None
    if pixel_areas is not None:
        grid_area = float(np.sum(pixel_areas) * raster.width / 1_000_000.0)
    defined_pixels_by_band = np.count_nonzero(raster.validity, axis=(1, 2))
    bands_with_data = int(np.count_nonzero(defined_pixels_by_band))

    print()
    print("Raster report")
    print(f"Path: {raster.path}")
    print(
        f"Dimensions: {raster.width:,} columns x {raster.height:,} rows x "
        f"{raster.band_count:,} bands"
    )
    print(f"Grid cells: {raster.pixel_count:,}")
    print(f"CRS: {raster.crs or 'undefined'}")
    print(
        f"Resolution: {abs(raster.transform.a):.12g} x {abs(raster.transform.e):.12g}"
    )
    print(
        "Bounds: "
        f"left={raster.bounds.left:.12g}, bottom={raster.bounds.bottom:.12g}, "
        f"right={raster.bounds.right:.12g}, top={raster.bounds.top:.12g}"
    )
    print(f"Source dtypes: {', '.join(sorted(set(raster.source_dtypes)))}")
    print(f"Declared nodata values: {_format_nodata_values(raster.nodata_values)}")
    print(f"Value array dtype: {raster.values.dtype}")
    print(
        f"Array memory: {raster.memory_bytes / MEBIBYTE:,.2f} MiB "
        f"({raster.values.nbytes / MEBIBYTE:,.2f} MiB values + "
        f"{raster.validity.nbytes / MEBIBYTE:,.2f} MiB validity)"
    )
    print(f"Approx. raster-grid area: {_format_area(grid_area)}")
    print(
        f"Bands with defined pixels: {bands_with_data:,} / {raster.band_count:,} "
        f"({raster.band_count - bands_with_data:,} completely undefined)"
    )
    _print_coverage("Defined in any band", any_band_coverage)
    _print_coverage("Defined in every band", every_band_coverage)

    if not include_band_report:
        return

    band_summaries = summarize_bands(
        raster,
        pixel_areas,
        show_progress=show_progress,
    )
    print()
    print("Per-band report")
    print(
        f"{'#':>3}  {'Band':<42} {'Defined':>12} {'Coverage':>9} "
        f"{'Area km^2':>13} {'Min':>12} {'Mean':>12} {'Max':>12}"
    )
    for summary in band_summaries:
        area = summary.coverage.area_square_kilometers
        area_text = "n/a" if area is None else f"{area:,.3f}"
        print(
            f"{summary.index:>3}  {summary.name[:42]:<42} "
            f"{summary.coverage.defined_pixels:>12,} "
            f"{summary.coverage.defined_percent:>8.2f}% "
            f"{area_text:>13} "
            f"{_format_statistic(summary.minimum):>12} "
            f"{_format_statistic(summary.mean):>12} "
            f"{_format_statistic(summary.maximum):>12}"
        )


def main() -> None:
    """Load, report, sample, serialize, and map one ecoregion GeoTIFF."""

    args = parse_args()
    try:
        raster = load_raster_pixels(
            args.geotiff,
            show_progress=not args.no_progress,
        )
    except (
        FileNotFoundError,
        ValueError,
        RuntimeError,
        rasterio.errors.RasterioError,
    ) as error:
        raise SystemExit(str(error)) from error
    print_raster_report(raster, not args.no_band_report, not args.no_progress)
    ecoregion_name = infer_ecoregion_name(raster.path)
    ecoregion_slug = re.sub(r"[^a-z0-9]+", "_", ecoregion_name.lower()).strip("_")

    if not args.no_sampling:
        sample_path = args.sample_output or (
            Path("outputs")
            / "samples"
            / f"{ecoregion_slug or 'ecoregion'}_spatial_sample.parquet"
        )
        try:
            sample = create_spatial_sample(
                raster,
                args.sampling_block_size_m,
                args.samples_per_class_per_block,
                args.random_seed,
                not args.no_progress,
            )
            print_spatial_sampling_report(sample)
            table_memory = int(sample.table.memory_usage(index=True, deep=True).sum())
            parquet_summary = write_spatial_sample_parquet(
                sample,
                sample_path,
                not args.no_progress,
            )
        except (ValueError, RuntimeError, OSError) as error:
            raise SystemExit(f"Could not create spatial sample: {error}") from error
        print_parquet_report(parquet_summary, table_memory)

    if not args.no_location_figure:
        figure_path = args.location_figure or (
            Path("outputs")
            / "figures"
            / f"{ecoregion_slug or 'ecoregion'}_world_location.png"
        )
        try:
            figure_summary = create_ecoregion_location_figure(
                raster,
                ecoregion_name,
                figure_path,
                not args.no_progress,
            )
        except (ValueError, OSError, rasterio.errors.RasterioError) as error:
            raise SystemExit(f"Could not generate location figure: {error}") from error
        print_location_figure_report(figure_summary)


if __name__ == "__main__":
    main()
