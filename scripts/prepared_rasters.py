"""Read aligned raster bands and model masks from prepared local files."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.io import DatasetReader
from rasterio.transform import Affine
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window, transform as window_transform


@dataclass(frozen=True)
class RasterBand:
    """One named band selected from a local raster file.

    Attributes:
        name: Name used by the caller, independent of the file's band description.
        path: Local raster path; relative paths use the working directory.
        index: One-based band number.
    """

    name: str
    path: Path
    index: int = 1


@dataclass(frozen=True)
class PreparedRasterWindow:
    """Raster values and masks covering the same requested pixel window.

    Attributes:
        values: Float64 (bands, rows, columns) array in the reader's band order.
            Source NoData, masked and nonfinite values are represented by NaN.
        reference_mask: Boolean (rows, columns) mask of defined reference values
            equal to one. AOI clipping belongs to the caller.
        assessment_mask: Optional boolean mask of defined assessment values
            equal to one, aligned to the input grid using nearest resampling.
        transform: Affine transform for this window's pixel coordinates.
    """

    values: np.ndarray
    reference_mask: np.ndarray
    assessment_mask: np.ndarray | None
    transform: Affine


@dataclass(frozen=True)
class PreparedRasterReader:
    """Open raster inputs sharing one grid, ready for explicit window reads.

    Obtain this reader from ``open_prepared_rasters`` and use it only inside that
    context. It holds the original files open without copying or changing them.

    Attributes:
        band_names: Caller-supplied names in output array order.
        crs: Shared EPSG:6933 coordinate system.
        transform: Shared full-raster affine transform.
        width: Number of columns in the shared grid.
        height: Number of rows in the shared grid.
    """

    band_names: tuple[str, ...]
    crs: CRS
    transform: Affine
    width: int
    height: int
    _bands: tuple[tuple[DatasetReader, int], ...]
    _reference_band: tuple[DatasetReader, int]
    _assessment_band: tuple[WarpedVRT, int] | None

    def read_window(self, window: Window) -> PreparedRasterWindow:
        """Read named bands and binary masks for one window on the shared grid.

        Args:
            window: Caller-selected window with integer offsets and positive
                integer dimensions entirely inside the shared raster. The caller
                owns traversal, AOI clipping and cancellation between reads.

        Returns:
            Ordered finite-or-NaN values, boolean masks and the window transform.

        Raises:
            RasterioError: If a source cannot be read.
        """

        values = np.stack([
            _read_band_values(source, index, window)
            for source, index in self._bands
        ])
        reference_mask = _read_band_values(*self._reference_band, window) == 1
        assessment_mask = (
            None if self._assessment_band is None
            else _read_band_values(*self._assessment_band, window) == 1
        )
        return PreparedRasterWindow(
            values=values,
            reference_mask=reference_mask,
            assessment_mask=assessment_mask,
            transform=window_transform(window, self.transform),
        )


def _read_band_values(
    source: DatasetReader | WarpedVRT,
    index: int,
    window: Window,
) -> np.ndarray:
    """Read one band and replace source NoData and nonfinite values with NaN.

    Args:
        source: Open dataset or aligned assessment-mask view.
        index: One-based band number checked when opening the source.
        window: In-bounds pixel window selected by the caller.

    Returns:
        Float64 (rows, columns) values, with NaN for missing data.

    Raises:
        RasterioError: If the dataset cannot supply the requested values.
    """

    masked = source.read(index, window=window, masked=True, out_dtype="float64")
    values = np.ma.getdata(masked)
    values[np.ma.getmaskarray(masked) | ~np.isfinite(values)] = np.nan
    return values


def _open_raster_band(
    band: RasterBand,
    sources: dict[Path, DatasetReader],
    stack: ExitStack,
) -> tuple[DatasetReader, int]:
    """Open a selected band, reusing a file already open in this context.

    Args:
        band: Caller-provided file and band selection.
        sources: Open files keyed by their supplied paths; updated in place.
        stack: Context that will close every opened file.

    Returns:
        The open source and its validated one-based band index.

    Raises:
        ValueError: If the selected band does not exist.
        RasterioError: If the file cannot be opened.
    """

    if band.path not in sources:
        sources[band.path] = stack.enter_context(rasterio.open(band.path))
    source = sources[band.path]
    if band.index not in source.indexes:
        raise ValueError(
            f"{band.name}: band {band.index} does not exist in {band.path}."
        )
    return source, band.index


def _require_same_grid(source: DatasetReader, grid: DatasetReader) -> None:
    """Check a newly opened file against the input stack's grid.

    Args:
        source: File being admitted as data or an aligned reference mask.
        grid: First data source defining the required grid.

    Raises:
        ValueError: If CRS, dimensions or affine transform differ.
    """

    if (
        source.crs != grid.crs
        or source.shape != grid.shape
        or source.transform != grid.transform
    ):
        raise ValueError(f"{source.name} does not share the grid of {grid.name}.")


@contextmanager
def open_prepared_rasters(
    bands: Sequence[RasterBand],
    reference_mask: RasterBand,
    assessment_mask: RasterBand | None = None,
) -> Iterator[PreparedRasterReader]:
    """Open aligned EPSG:6933 inputs for sampling and prediction window reads.

    Data and reference rasters must have identical dimensions and affine transforms.
    Only the optional assessment mask is aligned from another grid, using the
    existing nearest-neighbor mask convention. Check metadata once at opening;
    the caller keeps source files unchanged while using the reader.

    Args:
        bands: At least one input band, with nonempty unique caller-supplied names.
            Their order defines the returned value-array order.
        reference_mask: Binary raster band already aligned with the input grid.
        assessment_mask: Optional binary raster band with a defined CRS.

    Yields:
        A reader with grid metadata and open source handles. All handles close on
        context exit, including exceptions during opening or calculation.

    Raises:
        ValueError: If names, selected bands, CRS or required grids are invalid.
        RasterioError: If a file cannot be opened or an assessment view created.
    """

    names = tuple(band.name for band in bands)
    if not names or not all(names) or len(set(names)) != len(names):
        raise ValueError("Input raster band names must be nonempty and unique.")

    with ExitStack() as stack:
        sources: dict[Path, DatasetReader] = {}
        opened_bands = tuple(_open_raster_band(band, sources, stack) for band in bands)
        grid = opened_bands[0][0]
        if grid.crs != CRS.from_epsg(6933):
            raise ValueError(f"{grid.name} must use EPSG:6933.")
        for source in sources.values():
            if source is not grid:
                _require_same_grid(source, grid)
        aligned_sources = set(sources.values())
        reference_band = _open_raster_band(reference_mask, sources, stack)
        if reference_band[0] not in aligned_sources:
            _require_same_grid(reference_band[0], grid)
        assessment_band = None
        if assessment_mask is not None:
            source, index = _open_raster_band(assessment_mask, sources, stack)
            if source.crs is None:
                raise ValueError(f"{assessment_mask.path} must define its CRS.")
            aligned_mask = stack.enter_context(WarpedVRT(
                source, crs=grid.crs, transform=grid.transform,
                width=grid.width, height=grid.height, resampling=Resampling.nearest,
            ))
            assessment_band = aligned_mask, index
        yield PreparedRasterReader(
            band_names=names, crs=grid.crs, transform=grid.transform,
            width=grid.width, height=grid.height, _bands=opened_bands,
            _reference_band=reference_band, _assessment_band=assessment_band,
        )
