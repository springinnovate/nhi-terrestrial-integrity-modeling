"""Build weighted model samples directly from prepared raster bands."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from itertools import groupby
import math

import numpy as np
import pandas as pd
from rasterio.features import geometry_mask
from rasterio.transform import xy
from rasterio.windows import Window, transform as window_transform
from shapely.affinity import affine_transform
from shapely.geometry import MultiPolygon, Polygon, box, mapping
from shapely.geometry.base import BaseGeometry

from .prepared_rasters import PreparedRasterReader
from . import spatial_sampling


@dataclass(frozen=True)
class _SamplingWindow:
    """A full sampling tile and its readable intersection with the source.

    Attributes:
        tile_id: Spatial address used by the seeded pixel selector.
        tile_window: Full square tile in source pixel coordinates, possibly
            extending beyond the source. Its positions determine random draws.
        read_window: Integer intersection of that tile and the source raster.
    """

    tile_id: str
    tile_window: Window
    read_window: Window


def _sampling_windows(
    reader: PreparedRasterReader,
    pixel_aoi: BaseGeometry,
    tile_size: int,
    tile_origin: tuple[float, float],
) -> list[_SamplingWindow]:
    """Locate AOI-intersecting sampling tiles without creating cache files.

    Args:
        reader: Open, admitted raster grid.
        pixel_aoi: Nonempty AOI clipped to the source, in pixel coordinates.
        tile_size: Positive square tile width in pixels.
        tile_origin: Projected origin for the sampling tile addresses.

    Returns:
        Full and clipped windows in increasing pixel row, then column order.
        Grids matching the CLI use its tile addresses. Other affine grids follow
        their own pixel axes without shifting or resampling the data.
    """

    grid = reader.transform
    column_offset, row_offset = np.floor(np.linalg.solve(
        [[grid.a, grid.b], [grid.d, grid.e]],
        [grid.c - tile_origin[0], grid.f - tile_origin[1]],
    )).astype(np.int64)
    left, top, right, bottom = pixel_aoi.bounds
    source_window = Window(0, 0, reader.width, reader.height)
    windows = []
    for row in range(
        math.floor((top + row_offset) / tile_size),
        math.ceil((bottom + row_offset) / tile_size),
    ):
        for column in range(
            math.floor((left + column_offset) / tile_size),
            math.ceil((right + column_offset) / tile_size),
        ):
            tile = Window(
                int(column * tile_size - column_offset),
                int(row * tile_size - row_offset), tile_size, tile_size,
            )
            footprint = box(
                tile.col_off, tile.row_off,
                tile.col_off + tile.width, tile.row_off + tile.height,
            )
            if pixel_aoi.intersection(footprint).area > 0:
                windows.append(_SamplingWindow(
                    f"x{column:+07d}_y{-row - 1:+07d}", tile,
                    tile.intersection(source_window),
                ))
    return windows


def sample_prepared_rasters(
    reader: PreparedRasterReader,
    aoi: Polygon | MultiPolygon,
    *,
    block_size_meters: float,
    samples_per_class_per_block: int,
    random_seed: int,
    tile_size_pixels: int,
    tile_origin: tuple[float, float] = (0, 0),
    progress: Callable[[str, int, int], None] | None = None,
) -> pd.DataFrame:
    """Sample reference and background pixels inside an AOI for model fitting.

    Reuse the existing spatial selector and area weights. A pixel is eligible
    when it is inside the AOI and at least one named input band is defined.
    Preserve missing values in other bands. The assessment mask does not limit
    training. Read one tile at a time, then re-read only tiles with retained pixels
    to assemble their values. The caller owns the reader's context and keeps its
    source files unchanged throughout both passes.

    Args:
        reader: Reader from ``open_prepared_rasters`` with predictor/response bands
            and a reference mask. Band names must not collide with table metadata.
        aoi: Valid, nonempty polygon or multipolygon in EPSG:6933.
        block_size_meters: Positive width of global sampling blocks, anchored at
            projected (0, 0), matching the existing sampling method.
        samples_per_class_per_block: Positive sample cap for each block and
            reference/background class, shared across tile boundaries.
        random_seed: Seed used by the existing deterministic pixel selector.
        tile_size_pixels: Positive square tile width. This affects seeded sample
            selection; use the CLI's tile size when comparing identical inputs.
        tile_origin: Finite projected anchor for sampling tile addresses. If it
            falls between pixel edges, tile boundaries follow the raster's edges.
        progress: Optional callback receiving phase, completed windows and total
            windows, initially and after each read. Phases are "Selecting sample
            pixels" and "Reading sample values". It may raise to cancel; exceptions
            propagate and the reader remains owned by its calling context.

    Returns:
        Sample table in tile/pixel order: source-relative zero-based ``row`` and
        ``column``, the existing spatial sample weight columns (including
        ``reference_site`` and ``area_weight_m2``), and named input band values.
        This table can be passed directly to ``prepare_reference_condition_data``.

    Raises:
        ValueError: If sampling settings or AOI are invalid, band names collide
            with metadata, or no AOI pixel has any defined input value.
        RasterioError: If an input window cannot be read.
    """

    if (
        not math.isfinite(block_size_meters) or block_size_meters <= 0
        or samples_per_class_per_block <= 0 or tile_size_pixels <= 0
        or not all(math.isfinite(value) for value in tile_origin)
    ):
        raise ValueError("Sampling sizes must be positive and the tile origin finite.")
    if aoi.is_empty or not aoi.is_valid:
        raise ValueError("The sampling AOI must be a valid, nonempty polygon.")

    pixel_area = abs(reader.transform.determinant)
    # Derive the reserved names from their producer rather than repeating its schema.
    columns = spatial_sampling.build_sample_weight_columns([], {}, pixel_area)
    columns.update(row=np.empty(0, dtype=np.int64), column=np.empty(0, dtype=np.int64))
    collisions = set(reader.band_names).intersection(columns)
    if collisions:
        raise ValueError(
            f"Input band names conflict with sample metadata: {sorted(collisions)}"
        )

    inverse = ~reader.transform
    pixel_aoi = affine_transform(aoi, [
        inverse.a, inverse.b, inverse.d, inverse.e, inverse.c, inverse.f,
    ]).intersection(box(0, 0, reader.width, reader.height))
    windows = (
        [] if pixel_aoi.is_empty else
        _sampling_windows(reader, pixel_aoi, tile_size_pixels, tile_origin)
    )
    strata: dict[tuple[int, int, int], spatial_sampling.SamplingStratumState] = {}
    phase = "Selecting sample pixels"
    if progress is not None:
        progress(phase, 0, len(windows))
    for sequence, planned in enumerate(windows):
        window = planned.read_window
        data = reader.read_window(window)
        tile_transform = window_transform(planned.tile_window, reader.transform)
        tile_footprint = Polygon([
            tile_transform * point for point in
            ((0, 0), (tile_size_pixels, 0),
             (tile_size_pixels, tile_size_pixels), (0, tile_size_pixels))
        ])
        # Rasterize in the original full tile: moving polygon edges to a cropped
        # or inverse-transformed grid can change Rasterio's boundary pixels.
        tile_mask = geometry_mask(
            [mapping(aoi.intersection(tile_footprint))],
            out_shape=(tile_size_pixels, tile_size_pixels),
            transform=tile_transform, invert=True,
        )
        row_offset = int(window.row_off - planned.tile_window.row_off)
        col_offset = int(window.col_off - planned.tile_window.col_off)
        inside = tile_mask[
            row_offset:row_offset + int(window.height),
            col_offset:col_offset + int(window.width),
        ]
        rows, cols = np.nonzero(inside & np.any(np.isfinite(data.values), axis=0))
        if len(rows):
            x, y = xy(data.transform, rows, cols)
            full_indices = (
                (rows + row_offset) * tile_size_pixels + cols + col_offset
            )
            spatial_sampling.retain_tile_candidates(
                strata, tile_id=planned.tile_id, tile_sequence=sequence,
                tile_pixel_count=tile_size_pixels**2,
                eligible_pixel_indices=full_indices,
                sampling_block_columns=np.floor(
                    np.asarray(x) / block_size_meters,
                ).astype(np.int64),
                sampling_block_rows=np.floor(
                    np.asarray(y) / block_size_meters,
                ).astype(np.int64),
                reference_site_classes=data.reference_mask[rows, cols].astype(np.uint8),
                random_seed=random_seed,
                samples_per_class_per_block=samples_per_class_per_block,
            )
        if progress is not None:
            progress(phase, sequence + 1, len(windows))
    candidates = sorted(
        (candidate for stratum in strata.values() for candidate in stratum.candidates),
        key=lambda candidate: (candidate.tile_sequence, candidate.local_pixel_index),
    )
    if not candidates:
        raise ValueError("No AOI pixels contain a defined input raster value.")

    columns = spatial_sampling.build_sample_weight_columns(candidates, strata, pixel_area)
    for name in ("row", "column"):
        columns[name] = np.empty(len(candidates), dtype=np.int64)
    for name in reader.band_names:
        columns[name] = np.empty(len(candidates), dtype=np.float64)
    groups = [
        (sequence, list(group)) for sequence, group in
        groupby(candidates, key=lambda candidate: candidate.tile_sequence)
    ]
    phase = "Reading sample values"
    if progress is not None:
        progress(phase, 0, len(groups))
    offset = 0
    for completed, (sequence, retained) in enumerate(groups, start=1):
        planned = windows[sequence]
        indices = np.asarray([candidate.local_pixel_index for candidate in retained])
        rows = indices // tile_size_pixels + int(planned.tile_window.row_off)
        cols = indices % tile_size_pixels + int(planned.tile_window.col_off)
        destination = slice(offset, offset + len(retained))
        columns["row"][destination] = rows
        columns["column"][destination] = cols
        data = reader.read_window(planned.read_window)
        values = data.values[
            :, rows - int(planned.read_window.row_off),
            cols - int(planned.read_window.col_off),
        ]
        for name, band_values in zip(reader.band_names, values):
            columns[name][destination] = band_values
        offset += len(retained)
        if progress is not None:
            progress(phase, completed, len(groups))
    return pd.DataFrame(columns)
