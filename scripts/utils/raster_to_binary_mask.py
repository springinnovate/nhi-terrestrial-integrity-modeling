"""Convert a numeric raster band to a compressed binary mask."""

from __future__ import annotations

import argparse
import concurrent.futures
import math
import operator
import os
import re
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np
import rasterio
from rasterio.shutil import copy as copy_raster
from rasterio.windows import Window
from tqdm.auto import tqdm


DEFAULT_BAND = 1
DEFAULT_WINDOW_SIZE_PIXELS = 4096
OUTPUT_BLOCK_SIZE_PIXELS = 512
COMPRESSION = "ZSTD"
COMPARISON_PATTERN = re.compile(
    r"^\s*(>=|<=|==|>|<)\s*"
    r"([+-]?(?:(?:\d+(?:\.\d*)?)|(?:\.\d+))(?:[eE][+-]?\d+)?)\s*$"
)
COMPARISON_FUNCTIONS: dict[str, Callable[[np.ndarray, float], np.ndarray]] = {
    ">": operator.gt,
    "<": operator.lt,
    ">=": operator.ge,
    "<=": operator.le,
    "==": operator.eq,
}


@dataclass(frozen=True)
class RasterComparison:
    """Parsed numeric comparison used to classify source pixels.

    Attributes:
        operator: One of ``>``, ``<``, ``>=``, ``<=``, or ``==``.
        threshold: Finite numeric threshold on the source band's scale.
        expression: Normalized expression stored in output metadata.
    """

    operator: str
    threshold: float
    expression: str

    def evaluate(self, values: np.ndarray) -> np.ndarray:
        """Return the Boolean comparison result for numeric source values.

        Args:
            values: Numeric source values of any integer or floating dtype.

        Returns:
            Boolean array with the same shape as ``values``.
        """

        return COMPARISON_FUNCTIONS[self.operator](values, self.threshold)


@dataclass(frozen=True)
class MaskConversionSummary:
    """Metadata and pixel counts from one completed mask conversion.

    Attributes:
        input_path: Resolved source raster path.
        output_path: Resolved binary-mask path.
        expression: Normalized comparison expression.
        source_band: One-based source band number.
        width: Output columns, identical to the source.
        height: Output rows, identical to the source.
        true_pixels: Pixels written as one.
        false_pixels: Pixels written as zero, including invalid source pixels.
        invalid_source_pixels: Source nodata, masked, NaN, or infinite pixels.
        cog: Whether the output was converted with GDAL's COG driver.
        elapsed_seconds: End-to-end elapsed wall time.
    """

    input_path: Path
    output_path: Path
    expression: str
    source_band: int
    width: int
    height: int
    true_pixels: int
    false_pixels: int
    invalid_source_pixels: int
    cog: bool
    elapsed_seconds: float


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for raster mask conversion.

    Returns:
        Parsed command-line namespace.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Convert one numeric raster band into a ZSTD-compressed 0/1 mask. "
            "Quote comparison expressions in the shell, for example \">=80\"."
        )
    )
    parser.add_argument("input_raster", type=Path, help="Source numeric raster.")
    parser.add_argument("output_raster", type=Path, help="Output binary GeoTIFF.")
    parser.add_argument(
        "comparison",
        help="Required comparison in the form [>|<|>=|<=|==][number].",
    )
    parser.add_argument(
        "--band",
        type=int,
        default=DEFAULT_BAND,
        help=f"One-based source band to classify. Default: {DEFAULT_BAND}.",
    )
    parser.add_argument(
        "--window-size-pixels",
        type=int,
        default=DEFAULT_WINDOW_SIZE_PIXELS,
        help=(
            "Maximum source rows and columns processed per window. "
            f"Default: {DEFAULT_WINDOW_SIZE_PIXELS}."
        ),
    )
    parser.add_argument(
        "--cog",
        action="store_true",
        help=(
            "Convert the completed mask to a Cloud Optimized GeoTIFF with "
            "ZSTD compression and nearest-neighbor overviews."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output raster after conversion succeeds.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Suppress the conversion progress bar.",
    )
    return parser.parse_args()


def parse_comparison_expression(expression: str) -> RasterComparison:
    """Parse and validate a required raster comparison expression.

    Args:
        expression: Text in the form ``[>|<|>=|<=|==][number]``. Optional
            surrounding whitespace and whitespace between the operator and
            number are accepted. Signed decimals and scientific notation are
            supported.

    Returns:
        Parsed operator, finite threshold, and normalized expression.

    Raises:
        ValueError: If the operator or number does not match the contract.
    """

    match = COMPARISON_PATTERN.fullmatch(expression)
    if not match:
        raise ValueError(
            "comparison must have the form [>|<|>=|<=|==][number], "
            'for example ">=80", "<0.25", or "==1".'
        )
    comparison_operator, threshold_text = match.groups()
    threshold = float(threshold_text)
    if not math.isfinite(threshold):
        raise ValueError("comparison threshold must be finite.")
    return RasterComparison(
        operator=comparison_operator,
        threshold=threshold,
        expression=f"{comparison_operator}{threshold_text}",
    )


def iter_raster_windows(
    width: int,
    height: int,
    window_size_pixels: int,
) -> Iterator[Window]:
    """Yield bounded, row-major raster windows.

    Args:
        width: Raster columns.
        height: Raster rows.
        window_size_pixels: Maximum rows and columns per window.

    Yields:
        Windows covering every source pixel exactly once.
    """

    for row_offset in range(0, height, window_size_pixels):
        window_height = min(window_size_pixels, height - row_offset)
        for column_offset in range(0, width, window_size_pixels):
            window_width = min(window_size_pixels, width - column_offset)
            yield Window(
                col_off=column_offset,
                row_off=row_offset,
                width=window_width,
                height=window_height,
            )


def _temporary_gtiff_profile(source: rasterio.DatasetReader) -> dict[str, object]:
    """Build a portable one-bit, tiled, ZSTD GeoTIFF profile."""

    return {
        "driver": "GTiff",
        "width": source.width,
        "height": source.height,
        "count": 1,
        "dtype": "uint8",
        "crs": source.crs,
        "transform": source.transform,
        "nodata": None,
        "tiled": True,
        "blockxsize": OUTPUT_BLOCK_SIZE_PIXELS,
        "blockysize": OUTPUT_BLOCK_SIZE_PIXELS,
        "compress": COMPRESSION,
        "nbits": 1,
        "bigtiff": "IF_SAFER",
    }


def _write_mask_metadata(
    source: rasterio.DatasetReader,
    destination: rasterio.DatasetWriter,
    source_path: Path,
    source_band: int,
    comparison: RasterComparison,
) -> None:
    """Copy source context and record the binary-mask contract."""

    destination.update_tags(**source.tags())
    destination.update_tags(
        artifact_type="binary_raster_mask",
        source_raster=str(source_path),
        source_band=str(source_band),
        comparison_expression=comparison.expression,
        true_value="1",
        false_value="0",
        invalid_source_policy="source nodata, masked, NaN, and infinite pixels are 0",
    )
    destination.update_tags(1, **source.tags(source_band))
    destination.update_tags(
        1,
        source_band=str(source_band),
        comparison_expression=comparison.expression,
    )
    destination.set_band_description(1, f"binary mask: {comparison.expression}")


def _write_intermediate_mask(
    source: rasterio.DatasetReader,
    source_path: Path,
    output_path: Path,
    source_band: int,
    comparison: RasterComparison,
    window_size_pixels: int,
    show_progress: bool,
) -> tuple[int, int]:
    """Stream source values into a temporary tiled binary GeoTIFF."""

    total_windows = math.ceil(source.width / window_size_pixels) * math.ceil(
        source.height / window_size_pixels
    )
    true_pixels = 0
    invalid_source_pixels = 0
    windows = tqdm(
        iter_raster_windows(source.width, source.height, window_size_pixels),
        total=total_windows,
        desc="Classifying raster",
        unit="window",
        disable=not show_progress,
    )
    with rasterio.open(output_path, "w", **_temporary_gtiff_profile(source)) as output:
        _write_mask_metadata(
            source,
            output,
            source_path,
            source_band,
            comparison,
        )
        for window in windows:
            source_values = source.read(source_band, window=window, masked=True)
            numeric_values = np.asarray(source_values.data)
            valid_source = ~np.ma.getmaskarray(source_values)
            if np.issubdtype(numeric_values.dtype, np.floating):
                valid_source &= np.isfinite(numeric_values)
            output_values = np.zeros(numeric_values.shape, dtype=np.uint8)
            output_values[valid_source] = comparison.evaluate(
                numeric_values[valid_source]
            ).astype(np.uint8, copy=False)
            output.write(output_values, 1, window=window)
            true_pixels += int(np.count_nonzero(output_values))
            invalid_source_pixels += int(
                valid_source.size - np.count_nonzero(valid_source)
            )
    return true_pixels, invalid_source_pixels


def _copy_cog_with_rasterio(intermediate_path: Path, cog_path: Path) -> None:
    """Create a COG through Rasterio's portable GDAL wrapper."""

    copy_raster(
        intermediate_path,
        cog_path,
        driver="COG",
        strict=True,
        COMPRESS=COMPRESSION,
        BLOCKSIZE=OUTPUT_BLOCK_SIZE_PIXELS,
        BIGTIFF="IF_SAFER",
        OVERVIEW_RESAMPLING="NEAREST",
        NUM_THREADS="ALL_CPUS",
        NBITS=1,
    )


def _load_gdal() -> Any | None:
    """Return GDAL's optional Python module when it is installed."""

    try:
        from osgeo import gdal
    except ImportError:
        return None
    gdal.UseExceptions()
    return gdal


def _create_cog_with_gdal_progress(
    gdal: Any,
    intermediate_path: Path,
    cog_path: Path,
    show_progress: bool,
) -> None:
    """Create a COG with GDAL's genuine completion callback."""

    progress = tqdm(
        total=100.0,
        desc="Creating COG",
        unit="%",
        disable=not show_progress,
    )

    def report_progress(
        complete: float,
        _message: str,
        _callback_data: object,
    ) -> int:
        target = max(0.0, min(100.0, complete * 100.0))
        progress.update(max(0.0, target - progress.n))
        return 1

    try:
        options = gdal.TranslateOptions(
            format="COG",
            strict=True,
            creationOptions=[
                f"COMPRESS={COMPRESSION}",
                f"BLOCKSIZE={OUTPUT_BLOCK_SIZE_PIXELS}",
                "BIGTIFF=IF_SAFER",
                "OVERVIEW_RESAMPLING=NEAREST",
                "NUM_THREADS=ALL_CPUS",
                "NBITS=1",
            ],
            callback=report_progress,
        )
        completed_dataset = gdal.Translate(
            str(cog_path),
            str(intermediate_path),
            options=options,
        )
        if completed_dataset is None:
            raise RuntimeError("GDAL did not create the Cloud Optimized GeoTIFF.")
        completed_dataset.FlushCache()
        completed_dataset = None
        progress.update(max(0.0, 100.0 - progress.n))
    finally:
        progress.close()


def _create_cog_with_rasterio_progress(
    intermediate_path: Path,
    cog_path: Path,
    show_progress: bool,
) -> None:
    """Create a COG while reporting bytes written when callbacks are unavailable."""

    if not show_progress:
        _copy_cog_with_rasterio(intermediate_path, cog_path)
        return

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        copy_future = executor.submit(
            _copy_cog_with_rasterio,
            intermediate_path,
            cog_path,
        )
        with tqdm(
            desc="Creating COG (bytes written)",
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        ) as progress:
            while not copy_future.done():
                current_size = cog_path.stat().st_size if cog_path.exists() else 0
                progress.update(max(0, current_size - progress.n))
                time.sleep(0.25)
            copy_future.result()
            completed_size = cog_path.stat().st_size if cog_path.exists() else 0
            progress.update(max(0, completed_size - progress.n))


def _create_cog(
    intermediate_path: Path,
    cog_path: Path,
    show_progress: bool,
) -> None:
    """Create a final ZSTD COG after full-resolution classification finishes."""

    gdal = _load_gdal()
    if gdal is not None:
        _create_cog_with_gdal_progress(
            gdal,
            intermediate_path,
            cog_path,
            show_progress,
        )
        return
    _create_cog_with_rasterio_progress(
        intermediate_path,
        cog_path,
        show_progress,
    )


def _validate_completed_mask(
    output_path: Path,
    source_width: int,
    source_height: int,
    source_crs: rasterio.crs.CRS | None,
    source_transform: rasterio.Affine,
    cog: bool,
) -> None:
    """Verify output grid, dtype, compression, and optional COG layout."""

    with rasterio.open(output_path) as output:
        if output.width != source_width or output.height != source_height:
            raise RuntimeError("completed mask dimensions differ from the source.")
        if output.crs != source_crs or output.transform != source_transform:
            raise RuntimeError("completed mask georeferencing differs from the source.")
        if output.count != 1 or output.dtypes != ("uint8",):
            raise RuntimeError("completed mask is not a single-band uint8 raster.")
        if (
            output.compression is None
            or output.compression.value.upper() != COMPRESSION
        ):
            raise RuntimeError("completed mask does not use ZSTD compression.")
        if output.tags(1, ns="IMAGE_STRUCTURE").get("NBITS") != "1":
            raise RuntimeError("completed mask does not use one-bit pixel storage.")
        if cog and output.tags(ns="IMAGE_STRUCTURE").get("LAYOUT") != "COG":
            raise RuntimeError("completed mask is not marked with COG layout.")


def convert_raster_to_binary_mask(
    input_path: Path,
    output_path: Path,
    comparison_expression: str,
    *,
    source_band: int = DEFAULT_BAND,
    cog: bool = False,
    overwrite: bool = False,
    window_size_pixels: int = DEFAULT_WINDOW_SIZE_PIXELS,
    show_progress: bool = True,
) -> MaskConversionSummary:
    """Convert one numeric raster band to a compressed 0/1 mask.

    Source nodata, masked, NaN, and infinite pixels are written as zero. The
    output always preserves source dimensions, CRS, transform, and alignment.
    A regular output is a tiled ZSTD GeoTIFF. With ``cog=True``, the temporary
    GeoTIFF is copied through GDAL's COG driver using ZSTD and nearest-neighbor
    overviews.

    Args:
        input_path: Rasterio-readable source raster.
        output_path: Destination GeoTIFF path.
        comparison_expression: Required expression such as ``">=80"``.
        source_band: One-based source band to classify.
        cog: Whether to create a Cloud Optimized GeoTIFF.
        overwrite: Whether an existing destination may be atomically replaced.
        window_size_pixels: Maximum source rows and columns held per window.
        show_progress: Whether to display a tqdm progress bar.

    Returns:
        Completed paths, grid dimensions, classification counts, and timing.

    Raises:
        ValueError: If arguments or the selected source band are invalid.
        FileExistsError: If the output exists and overwrite is false.
        RuntimeError: If final output verification fails.
    """

    started = time.perf_counter()
    comparison = parse_comparison_expression(comparison_expression)
    resolved_input_path = input_path.expanduser().resolve()
    resolved_output_path = output_path.expanduser().resolve()
    if resolved_input_path == resolved_output_path:
        raise ValueError("input and output raster paths must differ.")
    if source_band < 1:
        raise ValueError("source_band must be at least one.")
    if window_size_pixels < 1:
        raise ValueError("window_size_pixels must be positive.")
    if resolved_output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {resolved_output_path}. "
            "Use --overwrite to replace it."
        )
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(resolved_input_path) as source:
        if source_band > source.count:
            raise ValueError(
                f"source band {source_band} exceeds raster band count {source.count}."
            )
        source_dtype = np.dtype(source.dtypes[source_band - 1])
        if source_dtype.kind not in "iuf":
            raise ValueError(
                f"source band dtype {source_dtype} is not byte, integer, or float."
            )
        source_width = source.width
        source_height = source.height
        source_crs = source.crs
        source_transform = source.transform
        with tempfile.TemporaryDirectory(
            prefix=f".{resolved_output_path.stem}-",
            dir=resolved_output_path.parent,
        ) as temporary_directory:
            temporary_path = Path(temporary_directory)
            intermediate_path = temporary_path / "binary_mask.tif"
            true_pixels, invalid_source_pixels = _write_intermediate_mask(
                source,
                resolved_input_path,
                intermediate_path,
                source_band,
                comparison,
                window_size_pixels,
                show_progress,
            )
            completed_path = intermediate_path
            if cog:
                completed_path = temporary_path / "binary_mask_cog.tif"
                _create_cog(intermediate_path, completed_path, show_progress)
            _validate_completed_mask(
                completed_path,
                source_width,
                source_height,
                source_crs,
                source_transform,
                cog,
            )
            os.replace(completed_path, resolved_output_path)

    total_pixels = source_width * source_height
    elapsed_seconds = time.perf_counter() - started
    return MaskConversionSummary(
        input_path=resolved_input_path,
        output_path=resolved_output_path,
        expression=comparison.expression,
        source_band=source_band,
        width=source_width,
        height=source_height,
        true_pixels=true_pixels,
        false_pixels=total_pixels - true_pixels,
        invalid_source_pixels=invalid_source_pixels,
        cog=cog,
        elapsed_seconds=elapsed_seconds,
    )


def print_conversion_summary(summary: MaskConversionSummary) -> None:
    """Print a concise, human-readable completed conversion report."""

    total_pixels = summary.width * summary.height
    print("Binary raster mask")
    print(f"Input: {summary.input_path}")
    print(f"Output: {summary.output_path}")
    print(f"Source band: {summary.source_band}")
    print(f"Comparison: {summary.expression}")
    print(f"Dimensions: {summary.width:,} columns x {summary.height:,} rows")
    print(
        f"True pixels: {summary.true_pixels:,} "
        f"({summary.true_pixels / total_pixels:.2%})"
    )
    print(
        f"False pixels: {summary.false_pixels:,} "
        f"({summary.false_pixels / total_pixels:.2%})"
    )
    print(f"Invalid source pixels written as false: {summary.invalid_source_pixels:,}")
    print(f"Compression: {COMPRESSION}")
    print(f"Cloud Optimized GeoTIFF: {'yes' if summary.cog else 'no'}")
    print(f"File size: {summary.output_path.stat().st_size / 1024**2:,.2f} MiB")
    print(f"Completed in {summary.elapsed_seconds:.2f} seconds")


def main() -> None:
    """Run binary-mask conversion from the command line."""

    args = parse_args()
    try:
        summary = convert_raster_to_binary_mask(
            args.input_raster,
            args.output_raster,
            args.comparison,
            source_band=args.band,
            cog=args.cog,
            overwrite=args.overwrite,
            window_size_pixels=args.window_size_pixels,
            show_progress=not args.no_progress,
        )
    except (OSError, RuntimeError, ValueError) as error:
        raise SystemExit(f"Could not create binary raster mask: {error}") from error
    print_conversion_summary(summary)


if __name__ == "__main__":
    main()
