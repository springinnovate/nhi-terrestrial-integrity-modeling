"""Load a multiband ecoregion GeoTIFF and report its in-memory coverage."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import rasterio
from rasterio.coords import BoundingBox
from rasterio.crs import CRS
from rasterio.transform import Affine
from tqdm.auto import tqdm


EARTH_RADIUS_METERS = 6_371_008.8
MEBIBYTE = 1024**2


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
        "Resolution: "
        f"{abs(raster.transform.a):.12g} x {abs(raster.transform.e):.12g}"
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
    """Load the requested GeoTIFF and print its diagnostic report."""

    args = parse_args()
    try:
        raster = load_raster_pixels(
            args.geotiff,
            show_progress=not args.no_progress,
        )
    except (FileNotFoundError, ValueError, RuntimeError, rasterio.errors.RasterioError) as error:
        raise SystemExit(str(error)) from error
    print_raster_report(raster, not args.no_band_report, not args.no_progress)


if __name__ == "__main__":
    main()
