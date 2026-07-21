"""Load a multiband ecoregion GeoTIFF and report its in-memory coverage."""

from __future__ import annotations

import argparse
import math
import re
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
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
        "--ecoregion-name",
        help=(
            "Name shown on the location map. Defaults to a readable name "
            "inferred from the GeoTIFF filename."
        ),
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


def default_location_figure_path(ecoregion_name: str) -> Path:
    """Build the default location-figure path for an ecoregion.

    Args:
        ecoregion_name: Human-readable ecoregion label.

    Returns:
        Relative PNG output path beneath ``outputs/figures``.
    """

    slug = re.sub(r"[^a-z0-9]+", "_", ecoregion_name.lower()).strip("_")
    return Path("outputs") / "figures" / f"{slug or 'ecoregion'}_world_location.png"


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


def _world_land_feature() -> cfeature.Feature:
    """Return low-resolution Natural Earth land geometry.

    Returns:
        Cartopy feature backed by Natural Earth 1:110 million land polygons.
    """

    return cfeature.LAND.with_scale("110m")


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
                _world_land_feature(),
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
    """Load a GeoTIFF, report diagnostics, and create its location figure."""

    args = parse_args()
    try:
        raster = load_raster_pixels(
            args.geotiff,
            show_progress=not args.no_progress,
        )
    except (FileNotFoundError, ValueError, RuntimeError, rasterio.errors.RasterioError) as error:
        raise SystemExit(str(error)) from error
    print_raster_report(raster, not args.no_band_report, not args.no_progress)
    if args.no_location_figure:
        return

    ecoregion_name = args.ecoregion_name or infer_ecoregion_name(raster.path)
    figure_path = args.location_figure or default_location_figure_path(ecoregion_name)
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
