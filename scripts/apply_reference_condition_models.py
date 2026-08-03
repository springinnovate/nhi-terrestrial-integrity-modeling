"""Apply fitted reference-condition response models to cached analysis tiles."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing
import os
import re
import time
from collections.abc import Iterator, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from contextlib import ExitStack
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from tempfile import TemporaryDirectory

import joblib
import numpy as np
import pandas as pd
import rasterio
from matplotlib import colormaps, rc_context
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from rasterio.coords import BoundingBox
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.features import geometry_mask
from rasterio.transform import from_origin
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window, transform as window_transform
from shapely.geometry import mapping
from shapely.geometry.base import BaseGeometry
from threadpoolctl import threadpool_limits
from tqdm.auto import tqdm

from .analysis_config import AnalysisConfiguration, load_analysis_configuration
from .fit_grassland_integrity_parameters import predict_expected_response
from .raster_cache_utils import (
    AnalysisCacheTiles,
    CachedRasterTile,
    calculate_file_sha256,
    resolve_analysis_cache_tiles,
)
from .reference_condition_utils import FIGURE_DPI


MAXIMUM_DISPLAY_DIMENSION = 700
DISPLAY_COLOR_MAXIMUM = 10.0
DISPLAY_YELLOW_GREEN_VALUE = 3.0
DISPLAY_COLOR_TICKS = (0.0, 1.0, 3.0, 5.0, 7.0, 10.0)
PERCENTILE_COLOR_TICKS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
REFERENCE_SITE_COLOR = "#0072B2"
REFERENCE_SITE_OUTLINE_COLOR = "#FFFFFF"
REFERENCE_SITE_OUTLINE_WIDTH = 0.4
FLOAT_NODATA = -9999.0
STATUS_NODATA = 255
STATUS_OUTSIDE_TARGET = 0
STATUS_INSUFFICIENT_PREDICTORS = 1
STATUS_PREDICTED = 2
INFERENCE_CHECKPOINT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ResponseModel:
    """One serialized ecological-response model prepared for raster inference.

    Attributes:
        path: Joblib artifact containing the fitted model bundle.
        response_name: Source raster band modeled as the ecological response.
        response_band: Short response identifier such as ``d02``.
        display_name: Human-readable ecological-response name.
        predictor_names: Ordered source raster bands required by the model.
        reference_rmse: Pooled out-of-fold reference RMSE used to standardize
            observed-minus-expected deviations.
        bundle: Deserialized model, preprocessing, and imputation objects.
    """

    path: Path
    response_name: str
    response_band: str
    display_name: str
    predictor_names: tuple[str, ...]
    reference_rmse: float
    bundle: dict[str, object]


@dataclass(frozen=True)
class ReferenceDepartureCalibration:
    """Reference distribution used to convert response vectors into percentiles.

    Attributes:
        prediction_table_path: Parquet table containing out-of-fold response
            predictions and standardized deviations.
        response_bands: Ordered response bands in every departure vector.
        reference_mean_vector: Area-weighted reference mean standardized-
            departure vector.
        reference_covariance_matrix: Area-weighted reference covariance before
            stabilization.
        stabilized_reference_covariance_matrix: Reference covariance after
            diagonal shrinkage.
        reference_precision_matrix: Inverse of the stabilized reference
            covariance matrix.
        sorted_reference_distances: Ascending Mahalanobis distances for complete
            reference rows.
        cumulative_reference_area_fractions: Cumulative represented-area
            fractions corresponding to the sorted distances.
        covariance_shrinkage: Fraction of covariance shrunk toward its diagonal.
        reference_row_count: Number of labeled reference rows before
            completeness filtering.
        complete_reference_row_count: Number of reference rows defining the
            matrix.
        reference_area_m2: Total represented reference area before filtering.
        complete_reference_area_m2: Represented area defining the matrix.
        covariance_condition_number: Condition number before stabilization.
        stabilized_covariance_condition_number: Condition number after
            stabilization.
    """

    prediction_table_path: Path
    response_bands: tuple[str, ...]
    reference_mean_vector: np.ndarray
    reference_covariance_matrix: np.ndarray
    stabilized_reference_covariance_matrix: np.ndarray
    reference_precision_matrix: np.ndarray
    sorted_reference_distances: np.ndarray
    cumulative_reference_area_fractions: np.ndarray
    covariance_shrinkage: float
    reference_row_count: int
    complete_reference_row_count: int
    reference_area_m2: float
    complete_reference_area_m2: float
    covariance_condition_number: float
    stabilized_covariance_condition_number: float

    def calculate_mahalanobis_distances(
        self,
        standardized_departures: np.ndarray,
    ) -> np.ndarray:
        """Calculate covariance-aware distance for complete response vectors.

        Args:
            standardized_departures: Matrix with one pixel per row and one
                standardized ecological-response departure per column.

        Returns:
            Mahalanobis distance for every input row.
        """

        centered_standardized_departures = (
            np.asarray(standardized_departures, dtype=np.float64)
            - self.reference_mean_vector
        )
        squared_distances = np.einsum(
            "ij,jk,ik->i",
            centered_standardized_departures,
            self.reference_precision_matrix,
            centered_standardized_departures,
        )
        return np.sqrt(np.maximum(squared_distances, 0.0))

    def calculate_reference_departure_percentiles(
        self,
        mahalanobis_distances: np.ndarray,
    ) -> np.ndarray:
        """Evaluate distances against the area-weighted reference distribution.

        Args:
            mahalanobis_distances: Mahalanobis distances to transform.

        Returns:
            Area-weighted empirical reference percentiles on the 0–1 scale.
        """

        mahalanobis_distance_array = np.asarray(
            mahalanobis_distances,
            dtype=np.float64,
        )
        insertion_offsets = np.searchsorted(
            self.sorted_reference_distances,
            mahalanobis_distance_array,
            side="right",
        )
        reference_departure_percentiles = np.zeros(
            mahalanobis_distance_array.shape,
            dtype=np.float64,
        )
        has_reference_at_or_below = insertion_offsets > 0
        reference_departure_percentiles[has_reference_at_or_below] = (
            self.cumulative_reference_area_fractions[
                insertion_offsets[has_reference_at_or_below] - 1
            ]
        )
        return reference_departure_percentiles


@dataclass
class ResponseStatistics:
    """Streaming pixel statistics for one inferred ecological response."""

    expected_pixels: int = 0
    deviation_pixels: int = 0
    missing_observed_pixels: int = 0
    standardized_sum: float = 0.0
    standardized_sum_of_squares: float = 0.0
    standardized_minimum: float = math.inf
    standardized_maximum: float = -math.inf
    absolute_standardized_above_one: int = 0
    absolute_standardized_above_two: int = 0
    absolute_standardized_above_three: int = 0

    def update(self, expected_pixels: int, standardized_values: np.ndarray) -> None:
        """Accumulate counts and standardized-deviation moments.

        Args:
            expected_pixels: Number of pixels receiving a model prediction in
                the current raster window.
            standardized_values: Finite standardized deviations for pixels
                whose observed response is also defined.

        Returns:
            None: Statistics are accumulated on this object.
        """

        values = np.asarray(standardized_values, dtype=np.float64)
        self.expected_pixels += expected_pixels
        self.deviation_pixels += len(values)
        self.missing_observed_pixels += expected_pixels - len(values)
        if len(values) == 0:
            return
        self.standardized_sum += float(values.sum())
        self.standardized_sum_of_squares += float(np.square(values).sum())
        self.standardized_minimum = min(
            self.standardized_minimum,
            float(values.min()),
        )
        self.standardized_maximum = max(
            self.standardized_maximum,
            float(values.max()),
        )
        absolute_values = np.abs(values)
        self.absolute_standardized_above_one += int(
            np.count_nonzero(absolute_values > 1.0)
        )
        self.absolute_standardized_above_two += int(
            np.count_nonzero(absolute_values > 2.0)
        )
        self.absolute_standardized_above_three += int(
            np.count_nonzero(absolute_values > 3.0)
        )

    def summarize(self) -> dict[str, float | int | None]:
        """Return JSON-ready counts and standardized-deviation summaries.

        Returns:
            Counts, moments, range, and threshold exceedance percentages.
        """

        if self.deviation_pixels == 0:
            return {
                "expected_pixels": self.expected_pixels,
                "deviation_pixels": 0,
                "missing_observed_pixels": self.missing_observed_pixels,
                "standardized_mean": None,
                "standardized_standard_deviation": None,
                "standardized_minimum": None,
                "standardized_maximum": None,
                "absolute_standardized_above_one_percent": None,
                "absolute_standardized_above_two_percent": None,
                "absolute_standardized_above_three_percent": None,
            }
        mean = self.standardized_sum / self.deviation_pixels
        variance = max(
            self.standardized_sum_of_squares / self.deviation_pixels - mean**2,
            0.0,
        )
        return {
            "expected_pixels": self.expected_pixels,
            "deviation_pixels": self.deviation_pixels,
            "missing_observed_pixels": self.missing_observed_pixels,
            "standardized_mean": mean,
            "standardized_standard_deviation": math.sqrt(variance),
            "standardized_minimum": self.standardized_minimum,
            "standardized_maximum": self.standardized_maximum,
            "absolute_standardized_above_one_percent": (
                100.0
                * self.absolute_standardized_above_one
                / self.deviation_pixels
            ),
            "absolute_standardized_above_two_percent": (
                100.0
                * self.absolute_standardized_above_two
                / self.deviation_pixels
            ),
            "absolute_standardized_above_three_percent": (
                100.0
                * self.absolute_standardized_above_three
                / self.deviation_pixels
            ),
        }


@dataclass
class DeparturePercentileStatistics:
    """Accumulate non-reference departure-percentile summary statistics.

    Attributes:
        pixel_count: Number of contributing non-reference pixels.
        percentile_sum: Sum of all contributing departure percentiles.
        percentile_sum_of_squares: Sum of squared departure percentiles.
        percentile_minimum: Smallest contributing departure percentile.
        percentile_maximum: Largest contributing departure percentile.
        pixels_at_or_above_90: Pixels at or above the 90th reference percentile.
        pixels_at_or_above_95: Pixels at or above the 95th reference percentile.
        pixels_at_or_above_99: Pixels at or above the 99th reference percentile.
    """

    pixel_count: int = 0
    percentile_sum: float = 0.0
    percentile_sum_of_squares: float = 0.0
    percentile_minimum: float = math.inf
    percentile_maximum: float = -math.inf
    pixels_at_or_above_90: int = 0
    pixels_at_or_above_95: int = 0
    pixels_at_or_above_99: int = 0

    def update(self, departure_percentiles: np.ndarray) -> None:
        """Accumulate one raster window of finite percentile values.

        Args:
            departure_percentiles: Reference-departure percentiles from one
                window.

        Returns:
            None: Statistics are accumulated on this object.
        """

        departure_percentile_array = np.asarray(
            departure_percentiles,
            dtype=np.float64,
        )
        if len(departure_percentile_array) == 0:
            return
        self.pixel_count += len(departure_percentile_array)
        self.percentile_sum += float(departure_percentile_array.sum())
        self.percentile_sum_of_squares += float(
            np.square(departure_percentile_array).sum()
        )
        self.percentile_minimum = min(
            self.percentile_minimum,
            float(departure_percentile_array.min()),
        )
        self.percentile_maximum = max(
            self.percentile_maximum,
            float(departure_percentile_array.max()),
        )
        self.pixels_at_or_above_90 += int(
            np.count_nonzero(departure_percentile_array >= 0.90)
        )
        self.pixels_at_or_above_95 += int(
            np.count_nonzero(departure_percentile_array >= 0.95)
        )
        self.pixels_at_or_above_99 += int(
            np.count_nonzero(departure_percentile_array >= 0.99)
        )

    def summarize(self) -> dict[str, float | int]:
        """Return JSON-ready coverage and distribution statistics.

        Returns:
            Pixel count, moments, range, and upper-percentile percentages.
        """

        mean = self.percentile_sum / self.pixel_count
        variance = max(
            self.percentile_sum_of_squares / self.pixel_count - mean**2,
            0.0,
        )
        return {
            "pixels": self.pixel_count,
            "mean": mean,
            "standard_deviation": math.sqrt(variance),
            "minimum": self.percentile_minimum,
            "maximum": self.percentile_maximum,
            "at_or_above_90_percent": (
                100.0 * self.pixels_at_or_above_90 / self.pixel_count
            ),
            "at_or_above_95_percent": (
                100.0 * self.pixels_at_or_above_95 / self.pixel_count
            ),
            "at_or_above_99_percent": (
                100.0 * self.pixels_at_or_above_99 / self.pixel_count
            ),
        }


@dataclass(frozen=True)
class InferenceRunSummary:
    """Principal outputs and pixel counts from one raster inference run."""

    output_directory: Path
    expected_reference_path: Path
    observed_minus_expected_path: Path
    standardized_deviation_path: Path
    departure_percentile_path: Path
    inference_status_path: Path
    aggregate_deviation_figure_path: Path
    departure_percentile_figure_path: Path
    report_path: Path
    metadata_path: Path
    response_count: int
    raster_pixels: int
    aoi_pixels: int
    target_pixels: int
    predicted_pixels: int
    departure_percentile_pixels: int
    insufficient_predictor_pixels: int
    imputed_pixels: int
    elapsed_seconds: float


@dataclass(frozen=True)
class InferenceOutputGrid:
    """Describe the pixel-aligned stitched grid and cache-tile placement.

    Attributes:
        crs: Cache-grid coordinate reference system.
        transform: Affine transform for the stitched output grid.
        bounds: Pixel-aligned output bounds around the configured AOI.
        width: Output columns.
        height: Output rows.
        pixel_size: Square output pixel size in projected units.
    """

    crs: CRS
    transform: rasterio.Affine
    bounds: BoundingBox
    width: int
    height: int
    pixel_size: float

    def overlapping_windows(
        self,
        cached_tile: CachedRasterTile,
    ) -> tuple[Window, Window]:
        """Return matching source-tile and stitched-output windows.

        Args:
            cached_tile: Cache tile to place on this output grid.

        Returns:
            Source and destination windows covering their shared bounds.
        """

        tile = cached_tile.tile
        shared_left = max(self.bounds.left, tile.left)
        shared_bottom = max(self.bounds.bottom, tile.bottom)
        shared_right = min(self.bounds.right, tile.right)
        shared_top = min(self.bounds.top, tile.top)
        shared_width = round((shared_right - shared_left) / self.pixel_size)
        shared_height = round((shared_top - shared_bottom) / self.pixel_size)
        source_window = Window(
            round((shared_left - tile.left) / self.pixel_size),
            round((tile.top - shared_top) / self.pixel_size),
            shared_width,
            shared_height,
        )
        destination_window = Window(
            round((shared_left - self.bounds.left) / self.pixel_size),
            round((self.bounds.top - shared_top) / self.pixel_size),
            shared_width,
            shared_height,
        )
        return source_window, destination_window


@dataclass(frozen=True)
class InferenceArtifactPaths:
    """Name every persistent raster, figure, report, and checkpoint artifact."""

    expected_reference: Path
    observed_minus_expected: Path
    standardized_deviation: Path
    departure_percentile: Path
    inference_status: Path
    aggregate_deviation_figure: Path
    departure_percentile_figure: Path
    report: Path
    metadata: Path
    checkpoint: Path


@dataclass(frozen=True)
class InferenceOutputStatistics:
    """Store the streaming summaries reconstructed from stitched outputs."""

    response_statistics: dict[str, ResponseStatistics]
    departure_percentile_statistics: DeparturePercentileStatistics
    aggregate_value_sums: np.ndarray
    aggregate_value_counts: np.ndarray
    percentile_value_sums: np.ndarray
    percentile_value_counts: np.ndarray
    reference_pixel_counts: np.ndarray
    aoi_pixels: int
    target_pixels: int
    predicted_pixels: int
    insufficient_predictor_pixels: int
    imputed_pixels: int


@dataclass(frozen=True)
class InferenceWindowResult:
    """Hold all output arrays calculated for one stitched raster window."""

    destination_window: Window
    expected_reference: np.ndarray
    observed_minus_expected: np.ndarray
    standardized_deviation: np.ndarray
    departure_percentile: np.ndarray
    inference_status: np.ndarray


@dataclass(frozen=True)
class ComputedInferenceTile:
    """Hold one cache tile's calculated windows until serialized output.

    Attributes:
        tile_id: Stable cache-grid tile identifier.
        windows: Calculated output arrays and destination windows.
        worker_process_id: Operating-system process that calculated the tile.
    """

    tile_id: str
    windows: tuple[InferenceWindowResult, ...]
    worker_process_id: int


@dataclass(frozen=True)
class InferenceWorkerContext:
    """Immutable model and raster context initialized once in each worker.

    Attributes:
        application_mask_path: Optional raster selecting inference pixels.
        window_size_pixels: Maximum processing-window width and height.
        projected_aoi: Analysis AOI transformed to the cache-grid CRS.
        output_grid: Stitched output dimensions and tile placement.
        required_band_indices: Ordered one-based source raster bands to read.
        predictor_names: Ordered predictor columns expected by the models.
        response_models: Fitted ecological-response model bundles.
        reference_calibration: Reference distribution for percentiles.
        maximum_predictor_missing_fraction: Allowed predictor missingness.
        reference_band_offset: Reference layer offset in each source read.
    """

    application_mask_path: Path | None
    window_size_pixels: int
    projected_aoi: BaseGeometry
    output_grid: InferenceOutputGrid
    required_band_indices: tuple[int, ...]
    predictor_names: tuple[str, ...]
    response_models: tuple[ResponseModel, ...]
    reference_calibration: ReferenceDepartureCalibration
    maximum_predictor_missing_fraction: float
    reference_band_offset: int


_INFERENCE_WORKER_CONTEXT: InferenceWorkerContext | None = None
_INFERENCE_NUMERICAL_THREAD_LIMIT = None


def build_inference_output_grid(
    analysis_configuration: AnalysisConfiguration,
    analysis_cache_tiles: AnalysisCacheTiles,
) -> InferenceOutputGrid:
    """Build the smallest cache-aligned rectangle containing the AOI.

    Args:
        analysis_configuration: Analysis grid definition.
        analysis_cache_tiles: Validated tiles and projected AOI geometry.

    Returns:
        Pixel-aligned stitched output grid in the cache CRS.
    """

    cache_grid = analysis_configuration.grid
    pixel_size = float(cache_grid.pixel_size_meters)
    minimum_x, minimum_y, maximum_x, maximum_y = (
        analysis_cache_tiles.projected_aoi.bounds
    )
    # CRS round trips can place a boundary a few floating-point units beyond an
    # exact grid line. The pixel-relative tolerance prevents that numerical
    # noise from adding a full row or column to the stitched rectangle.
    grid_line_tolerance = 1e-9
    output_left = cache_grid.origin_x + math.floor(
        (minimum_x - cache_grid.origin_x) / pixel_size
        + grid_line_tolerance
    ) * pixel_size
    output_bottom = cache_grid.origin_y + math.floor(
        (minimum_y - cache_grid.origin_y) / pixel_size
        + grid_line_tolerance
    ) * pixel_size
    output_right = cache_grid.origin_x + math.ceil(
        (maximum_x - cache_grid.origin_x) / pixel_size
        - grid_line_tolerance
    ) * pixel_size
    output_top = cache_grid.origin_y + math.ceil(
        (maximum_y - cache_grid.origin_y) / pixel_size
        - grid_line_tolerance
    ) * pixel_size
    output_width = round((output_right - output_left) / pixel_size)
    output_height = round((output_top - output_bottom) / pixel_size)
    return InferenceOutputGrid(
        crs=CRS.from_string(cache_grid.crs),
        transform=from_origin(
            output_left,
            output_top,
            pixel_size,
            pixel_size,
        ),
        bounds=BoundingBox(
            output_left,
            output_bottom,
            output_right,
            output_top,
        ),
        width=output_width,
        height=output_height,
        pixel_size=pixel_size,
    )


def build_inference_fingerprint(
    analysis_configuration: AnalysisConfiguration,
    analysis_cache_tiles: AnalysisCacheTiles,
    model_run_directory: Path,
    response_models: tuple[ResponseModel, ...],
    reference_prediction_table_path: Path,
) -> str:
    """Hash every input that can change completed inference pixels.

    Args:
        analysis_configuration: Complete effective analysis configuration.
        analysis_cache_tiles: Validated source tiles and checksums.
        model_run_directory: Model run containing metadata and fitted models.
        response_models: Ordered fitted response models.
        reference_prediction_table_path: Calibration table used for percentiles.

    Returns:
        Hexadecimal SHA-256 digest for checkpoint compatibility.
    """

    application_mask_path = analysis_configuration.inference.application_mask_path
    application_mask_identity = None
    if application_mask_path is not None:
        application_mask_statistics = application_mask_path.stat()
        application_mask_identity = {
            "path": str(application_mask_path),
            "file_size_bytes": application_mask_statistics.st_size,
            "modified_time_ns": application_mask_statistics.st_mtime_ns,
        }
    fingerprint_inputs = {
        "checkpoint_schema_version": INFERENCE_CHECKPOINT_SCHEMA_VERSION,
        # Operational worker and window counts do not change pixel values and
        # therefore must not invalidate otherwise reusable completed tiles.
        "raster_configuration_sha256": (
            analysis_configuration.raster_configuration_sha256
        ),
        "covariance_shrinkage": (
            analysis_configuration.inference.covariance_shrinkage
        ),
        "stack_identifier": analysis_cache_tiles.stack_identifier,
        "projected_aoi_sha256": hashlib.sha256(
            analysis_cache_tiles.projected_aoi.wkb
        ).hexdigest(),
        "application_mask": application_mask_identity,
        "source_tiles": {
            cached_tile.tile.tile_id: cached_tile.sha256
            for cached_tile in analysis_cache_tiles.tiles
        },
        "run_metadata_sha256": calculate_file_sha256(
            model_run_directory / "run_metadata.json"
        ),
        "reference_prediction_table_sha256": calculate_file_sha256(
            reference_prediction_table_path
        ),
        "response_models": {
            response_model.response_band: calculate_file_sha256(
                response_model.path
            )
            for response_model in response_models
        },
    }
    return hashlib.sha256(
        json.dumps(
            fingerprint_inputs,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed analysis, model, output, and progress arguments.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Apply final reference-condition models to the configured AOI "
            "using validated Earth Engine raster-cache tiles."
        )
    )
    parser.add_argument(
        "analysis_configuration",
        type=Path,
        help=(
            "Complete TOML analysis definition containing inference settings "
            "and the raster-band contract."
        ),
    )
    parser.add_argument(
        "model_run_directory",
        type=Path,
        help=(
            "Output directory from fit_grassland_integrity_parameters.py, "
            "containing run_metadata.json and models/."
        ),
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        help=(
            "Output directory. Defaults to "
            "outputs/reference_condition_inference/<analysis_name>."
        ),
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Suppress tqdm progress output.",
    )
    return parser.parse_args()


def load_response_models(
    model_run_directory: Path,
) -> tuple[dict[str, object], tuple[ResponseModel, ...], float]:
    """Load one compatible set of response models and its run configuration.

    Args:
        model_run_directory: Directory containing ``run_metadata.json`` and a
            ``models`` subdirectory created by the response-model workflow.

    Returns:
        Run metadata, response models sorted by response band, and the maximum
        predictor missingness fraction used during training.

    Raises:
        ValueError: If no models exist, model predictor signatures differ, or
            a model lacks a positive cross-validated reference RMSE.
    """

    metadata_path = model_run_directory / "run_metadata.json"
    run_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    model_paths = sorted(
        (model_run_directory / "models").glob("*_reference_condition_gam.joblib")
    )
    if not model_paths:
        raise ValueError(
            f"No reference-condition models found under {model_run_directory}."
        )

    response_models = []
    for model_path in model_paths:
        model_bundle = joblib.load(model_path)
        continuous_predictor_names = tuple(
            model_bundle["continuous_predictor_names"]
        )
        categorical_predictor_name = str(
            model_bundle["categorical_predictor_name"]
        )
        out_of_fold_reference_rmse = float(
            model_bundle["reference_residual_rmse_oof"]
        )
        if (
            not np.isfinite(out_of_fold_reference_rmse)
            or out_of_fold_reference_rmse <= 0
        ):
            raise ValueError(
                f"{model_path.name} has invalid cross-validated reference RMSE "
                f"{out_of_fold_reference_rmse}."
            )
        response_models.append(
            ResponseModel(
                path=model_path,
                response_name=str(model_bundle["response"]),
                response_band=str(model_bundle["response_band"]),
                display_name=str(model_bundle["display_name"]),
                predictor_names=(
                    *continuous_predictor_names,
                    categorical_predictor_name,
                ),
                reference_rmse=out_of_fold_reference_rmse,
                bundle=model_bundle,
            )
        )

    response_models.sort(key=lambda model: model.response_band)
    expected_predictor_names = response_models[0].predictor_names
    if any(
        model.predictor_names != expected_predictor_names
        for model in response_models[1:]
    ):
        raise ValueError(
            "Models in one inference run must use the same ordered predictor bands."
        )
    maximum_predictor_missing_fraction = float(
        run_metadata["configuration"]["maximum_row_missing_fraction"]
    )
    return (
        run_metadata,
        tuple(response_models),
        maximum_predictor_missing_fraction,
    )


def build_reference_departure_calibration(
    model_run_directory: Path,
    response_models: tuple[ResponseModel, ...],
    covariance_shrinkage: float,
) -> ReferenceDepartureCalibration:
    """Fit the multivariate reference distribution from out-of-fold residuals.

    Args:
        model_run_directory: Output directory from the response-model workflow.
        response_models: Ordered fitted responses included in each vector.
        covariance_shrinkage: Fraction of covariance shrunk toward its diagonal
            before inversion. The analysis configuration guarantees a value
            strictly between zero and one.

    Returns:
        Complete reference calibration for distance and percentile inference.

    Raises:
        ValueError: If too few complete reference vectors are available or a
            fitted response has no reference variance.
    """

    prediction_table_path = (
        model_run_directory / "ecological_response_predictions.parquet"
    )
    prediction_table = pd.read_parquet(prediction_table_path)
    response_bands = tuple(model.response_band for model in response_models)
    standardized_deviation_columns = [
        f"{response_band}_standardized_deviation_oof"
        for response_band in response_bands
    ]
    reference_row_mask = prediction_table["reference_site"].eq(1)
    area_weights = pd.to_numeric(
        prediction_table["area_weight_m2"], errors="coerce"
    )
    reference_area_weights = area_weights.loc[reference_row_mask]
    reference_area_m2 = float(reference_area_weights.sum())
    standardized_departure_table = prediction_table[
        standardized_deviation_columns
    ].apply(
        pd.to_numeric,
        errors="coerce",
    )
    complete_reference_row_mask = (
        reference_row_mask
        & np.isfinite(standardized_departure_table).all(axis=1)
        & np.isfinite(area_weights)
        & area_weights.gt(0)
    )
    complete_reference_departures = standardized_departure_table.loc[
        complete_reference_row_mask
    ].to_numpy(dtype=np.float64)
    complete_reference_area_weights = area_weights.loc[
        complete_reference_row_mask
    ].to_numpy(dtype=np.float64)
    if len(complete_reference_departures) <= len(response_models):
        raise ValueError(
            "Reference departure calibration requires more complete reference "
            "rows than fitted responses."
        )

    complete_reference_area_m2 = float(complete_reference_area_weights.sum())
    reference_mean_vector = np.average(
        complete_reference_departures,
        axis=0,
        weights=complete_reference_area_weights,
    )
    centered_reference_departures = (
        complete_reference_departures - reference_mean_vector
    )
    reference_covariance_matrix = (
        (
            centered_reference_departures
            * complete_reference_area_weights[:, np.newaxis]
        ).T
        @ centered_reference_departures
        / complete_reference_area_m2
    )
    reference_variances = np.diag(reference_covariance_matrix)
    if not np.isfinite(reference_variances).all() or np.any(
        reference_variances <= 0
    ):
        raise ValueError(
            "Every fitted response must have positive finite reference variance."
        )

    # Shrinking only off-diagonal covariance preserves each response's reference
    # variance while preventing near-duplicate responses from destabilizing inversion.
    stabilized_reference_covariance_matrix = (
        (1.0 - covariance_shrinkage) * reference_covariance_matrix
        + covariance_shrinkage * np.diag(reference_variances)
    )
    reference_precision_matrix = np.linalg.inv(
        stabilized_reference_covariance_matrix
    )
    squared_reference_distances = np.einsum(
        "ij,jk,ik->i",
        centered_reference_departures,
        reference_precision_matrix,
        centered_reference_departures,
    )
    reference_distances = np.sqrt(np.maximum(squared_reference_distances, 0.0))
    reference_distance_sort_order = np.argsort(reference_distances, kind="stable")
    sorted_reference_distances = reference_distances[reference_distance_sort_order]
    cumulative_reference_area_fractions = np.cumsum(
        complete_reference_area_weights[reference_distance_sort_order]
    ) / complete_reference_area_m2
    cumulative_reference_area_fractions[-1] = 1.0

    return ReferenceDepartureCalibration(
        prediction_table_path=prediction_table_path,
        response_bands=response_bands,
        reference_mean_vector=reference_mean_vector,
        reference_covariance_matrix=reference_covariance_matrix,
        stabilized_reference_covariance_matrix=(
            stabilized_reference_covariance_matrix
        ),
        reference_precision_matrix=reference_precision_matrix,
        sorted_reference_distances=sorted_reference_distances,
        cumulative_reference_area_fractions=cumulative_reference_area_fractions,
        covariance_shrinkage=covariance_shrinkage,
        reference_row_count=int(np.count_nonzero(reference_row_mask)),
        complete_reference_row_count=len(complete_reference_departures),
        reference_area_m2=reference_area_m2,
        complete_reference_area_m2=complete_reference_area_m2,
        covariance_condition_number=float(
            np.linalg.cond(reference_covariance_matrix)
        ),
        stabilized_covariance_condition_number=float(
            np.linalg.cond(stabilized_reference_covariance_matrix)
        ),
    )


def write_inference_report(
    output_path: Path,
    inference_metadata: dict[str, object],
) -> None:
    """Write a human-readable raster inference report.

    Args:
        output_path: Destination path for the Markdown report.
        inference_metadata: JSON-ready inference metadata and response
            statistics.

    Returns:
        None: The completed report is written to ``output_path``.
    """

    coverage_statistics = inference_metadata["coverage"]
    inference_configuration = inference_metadata["configuration"]
    aggregate_figure_metadata = inference_metadata[
        "aggregate_deviation_figure"
    ]
    reference_calibration_metadata = inference_metadata[
        "reference_departure_calibration"
    ]
    departure_percentile_metadata = inference_metadata[
        "reference_departure_percentile"
    ]
    percentile_statistics = departure_percentile_metadata["statistics"]
    color_scale_upper_value = aggregate_figure_metadata[
        "color_scale_upper_value"
    ]
    complete_reference_area_percent = reference_calibration_metadata[
        "complete_reference_area_percent"
    ]
    stabilized_covariance_condition_number = reference_calibration_metadata[
        "stabilized_covariance_condition_number"
    ]
    cells_at_or_above_color_maximum_percent = aggregate_figure_metadata[
        "cells_at_or_above_color_maximum_percent"
    ]
    application_mask_metadata = inference_metadata["application_mask"]
    application_mask_path_text = (
        str(application_mask_metadata["path"])
        if application_mask_metadata is not None
        else "not supplied"
    )
    report_lines = [
        "# Reference-condition raster inference: "
        f"{inference_metadata['ecoregion_name']}",
        "",
    ]
    if application_mask_metadata is None:
        report_lines.extend(
            [
                "> **Important:** No application mask was supplied. These outputs "
                "cover the usable AOI predictor footprint and must not be "
                "interpreted as grassland integrity maps.",
                "",
            ]
        )
    report_lines.extend(
        [
            "## Inputs",
            "",
            "- Raster cache manifest: "
            f"`{inference_metadata['input_raster_cache']['manifest']}`",
            "- Raster cache stack: "
            f"`{inference_metadata['input_raster_cache']['stack_identifier']}`",
            "- Validated source tiles: "
            f"{inference_metadata['input_raster_cache']['source_tiles']:,}",
            f"- Model run: `{inference_metadata['model_run_directory']}`",
            f"- Application mask: `{application_mask_path_text}`",
            "- Application mask selection: "
            f"{inference_metadata['mask_interpretation']}",
            f"- Responses: {inference_metadata['response_count']}",
            (
                "- Maximum predictor missingness: "
                f"{inference_configuration['maximum_predictor_missing_fraction']:.1%}"
            ),
            "- Processing window: "
            f"{inference_configuration['window_size_pixels']} pixels",
            "- Configured tile workers: "
            f"{inference_configuration['worker_count']}",
            (
                "- Covariance diagonal shrinkage: "
                f"{inference_configuration['covariance_shrinkage']:.1%}"
            ),
            "- Inference tiles processed in this run: "
            f"{inference_metadata['tile_processing']['processed_tiles']:,}",
            "- Inference tiles resumed from checkpoint: "
            f"{inference_metadata['tile_processing']['resumed_tiles']:,}",
            "",
            "## Pixel coverage",
            "",
            "- Stitched bounding-grid pixels: "
            f"{coverage_statistics['raster_pixels']:,}",
            f"- Pixels inside exact AOI: {coverage_statistics['aoi_pixels']:,}",
            f"- Target pixels: {coverage_statistics['target_pixels']:,}",
            f"- Predicted pixels: {coverage_statistics['predicted_pixels']:,}",
            (
                "- Insufficient-predictor pixels: "
                f"{coverage_statistics['insufficient_predictor_pixels']:,}"
            ),
            "- Predicted pixels using imputation: "
            f"{coverage_statistics['imputed_pixels']:,}",
            (
                "- Complete non-reference percentile pixels: "
                f"{coverage_statistics['departure_percentile_pixels']:,}"
            ),
            "",
            "Status raster codes: 0 is outside the target, 1 has too many missing "
            "predictors, and 2 received model predictions. Its second band records "
            "the number of missing predictors before imputation.",
            "",
            "## Multivariate reference-departure percentile",
            "",
            (
                "The calibration uses complete standardized out-of-fold residual "
                "vectors from "
                f"{reference_calibration_metadata['complete_reference_rows']:,} of "
                f"{reference_calibration_metadata['reference_rows']:,} reference "
                "rows, representing "
                f"{complete_reference_area_percent:.1f}% "
                "of sampled "
                "reference area. The area-weighted reference mean and covariance "
                "are calculated from those vectors."
            ),
            "",
            (
                "The covariance matrix is stabilized with "
                f"{reference_calibration_metadata['covariance_shrinkage']:.1%} "
                "diagonal shrinkage "
                "before inversion. Its condition number changes from "
                f"{reference_calibration_metadata['covariance_condition_number']:.3g} "
                "to "
                f"{stabilized_covariance_condition_number:.3g}."
            ),
            "",
            (
                "For each complete non-reference pixel, `D_i` is the Mahalanobis "
                "distance between its standardized-departure vector and the "
                "reference mean. `P_i` is the represented-area fraction of "
                "complete reference rows whose distance is less than or equal to "
                "`D_i`. A value of 0.95 therefore means the pixel is farther from "
                "the reference center than 95% of represented calibration area."
            ),
            "",
            (
                f"The percentile raster contains {percentile_statistics['pixels']:,} "
                "non-reference pixels. Its mean is "
                f"{percentile_statistics['mean']:.3f}, and "
                f"{percentile_statistics['at_or_above_95_percent']:.1f}% of defined "
                "pixels have `P_i >= 0.95`. Reference pixels and pixels missing any "
                "fitted response are nodata."
            ),
            "",
            (
                "The percentile PNG uses a fixed 0–1 scale, with 0 in green and 1 "
                "in red. Blue display cells contain reference sites and are "
                "excluded from the colored surface. `P_i` measures multivariate "
                "departure from the sampled reference distribution; it does not "
                "prove degradation or constitute an ecological integrity score."
            ),
            "",
            "## Aggregate standardized-deviation map",
            "",
            (
                "The PNG maps the mean pixel-level `sum(abs(z_j))` within each "
                "coarsened display cell, using every fitted ecological response. "
                "Green indicates lower total standardized departure from modeled "
                "reference condition and red indicates larger departure."
            ),
            "",
            (
                "Only non-reference pixels with defined standardized deviations "
                "for every response contribute to the colored surface. Black "
                "outlines identify display cells containing supplied reference-site "
                "pixels. A fixed linear scale maps 0 to green, "
                f"{DISPLAY_YELLOW_GREEN_VALUE:g} to yellow-green, and "
                f"{color_scale_upper_value:g} or more to red. "
                f"{cells_at_or_above_color_maximum_percent:.1f}% "
                "of colored display cells are at or above "
                f"{color_scale_upper_value:g}. This is a diagnostic total-departure "
                "map, not a grassland integrity score."
            ),
            "",
            "## Response outputs",
            "",
            (
                "| Band | Response | Cross-validated reference RMSE | Expected "
                "pixels | Deviation pixels | Mean z | SD z | Min z | Max z | "
                "Abs(z) > 2 |"
            ),
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for response_metadata in inference_metadata["responses"]:
        response_statistics = response_metadata["statistics"]
        if response_statistics["standardized_mean"] is None:
            standardized_mean_text = "NA"
            standardized_standard_deviation_text = "NA"
            standardized_minimum_text = "NA"
            standardized_maximum_text = "NA"
            above_two_percent_text = "NA"
        else:
            standardized_mean_text = (
                f"{response_statistics['standardized_mean']:.3f}"
            )
            standardized_standard_deviation_text = (
                f"{response_statistics['standardized_standard_deviation']:.3f}"
            )
            standardized_minimum_text = (
                f"{response_statistics['standardized_minimum']:.3f}"
            )
            standardized_maximum_text = (
                f"{response_statistics['standardized_maximum']:.3f}"
            )
            above_two_percent_text = (
                f"{response_statistics['absolute_standardized_above_two_percent']:.1f}%"
            )
        report_lines.append(
            f"| {response_metadata['response_band']} | "
            f"{response_metadata['display_name']} | "
            f"{response_metadata['reference_residual_rmse_oof']:.6g} | "
            f"{response_statistics['expected_pixels']:,} | "
            f"{response_statistics['deviation_pixels']:,} | "
            f"{standardized_mean_text} | "
            f"{standardized_standard_deviation_text} | "
            f"{standardized_minimum_text} | {standardized_maximum_text} | "
            f"{above_two_percent_text} |"
        )
    report_lines.extend(
        [
            "",
            "## Interpretation",
            "",
            (
                "For each response, expected reference condition is predicted by the "
                "final model fitted to all usable reference observations. Raw "
                "deviation is observed minus expected. Standardized deviation divides "
                "that value by the pooled out-of-fold reference RMSE."
            ),
            "",
            (
                "Positive standardized deviation means observed is above expected; it "
                "does not automatically mean higher integrity. The multivariate "
                "percentile combines covariance-aware distance, but it does not "
                "assign ecological directions or convert departure into an integrity "
                "score."
            ),
            "",
            "## Artifacts",
            "",
        ]
    )
    for artifact_name, artifact_path in inference_metadata["artifacts"].items():
        report_lines.append(f"- {artifact_name}: `{artifact_path}`")
    report_lines.append("")
    output_path.write_text("\n".join(report_lines), encoding="utf-8")


def create_aggregate_deviation_figure(
    value_sums: np.ndarray,
    value_counts: np.ndarray,
    reference_counts: np.ndarray,
    raster_bounds: BoundingBox,
    raster_crs: CRS | None,
    response_count: int,
    ecoregion_name: str,
    application_mask_supplied: bool,
    output_path: Path,
) -> dict[str, object]:
    """Map coarsened total standardized departure and reference-site locations.

    The source-pixel diagnostic is the sum of absolute standardized deviations
    across all fitted ecological responses. Each visible display cell contains
    the mean diagnostic among complete-response, non-reference source pixels.
    Reference pixels are excluded from the colored surface and shown as black
    outlines around display cells containing at least one reference pixel.

    Args:
        value_sums: Sum of source-pixel aggregate deviations per display cell.
        value_counts: Contributing non-reference source pixels per display cell.
        reference_counts: Reference-site source pixels per display cell.
        raster_bounds: Spatial bounds of the source raster.
        raster_crs: Source raster coordinate reference system, when defined.
        response_count: Number of standardized response deviations in each sum.
        ecoregion_name: Human-readable label included in the title.
        application_mask_supplied: Whether inference was limited by an external
            application mask.
        output_path: Destination path for the publication-resolution PNG.

    Returns:
        JSON-ready display dimensions, counts, aggregation, and color limits.

    Raises:
        RuntimeError: If no complete-response, non-reference pixels are available
            to display.
    """

    display_values = np.full(value_sums.shape, np.nan, dtype=np.float64)
    np.divide(
        value_sums,
        value_counts,
        out=display_values,
        where=value_counts > 0,
    )
    finite_values = display_values[np.isfinite(display_values)]
    if len(finite_values) == 0:
        raise RuntimeError(
            "No non-reference pixels have standardized deviations for every "
            "response; the aggregate deviation figure cannot be created."
        )

    cells_at_or_above_maximum = int(
        np.count_nonzero(finite_values >= DISPLAY_COLOR_MAXIMUM)
    )
    cells_at_or_above_maximum_percent = (
        100.0 * cells_at_or_above_maximum / len(finite_values)
    )

    color_map = colormaps["RdYlGn_r"].copy()
    color_map.set_bad("#ECEFF1")
    reference_display_mask = reference_counts > 0
    extent = (
        raster_bounds.left,
        raster_bounds.right,
        raster_bounds.bottom,
        raster_bounds.top,
    )
    with rc_context({"font.family": "DejaVu Sans", "font.size": 9}):
        figure = Figure(figsize=(10.0, 7.5), facecolor="white")
        FigureCanvasAgg(figure)
        axis = figure.subplots()
        image = axis.imshow(
            np.ma.masked_invalid(display_values),
            cmap=color_map,
            origin="upper",
            extent=extent,
            interpolation="nearest",
            vmin=0.0,
            vmax=DISPLAY_COLOR_MAXIMUM,
        )
        if np.any(reference_display_mask):
            x_cell_size = (raster_bounds.right - raster_bounds.left) / len(
                reference_display_mask[0]
            )
            y_cell_size = (raster_bounds.top - raster_bounds.bottom) / len(
                reference_display_mask
            )
            x_centers = np.linspace(
                raster_bounds.left + x_cell_size / 2.0,
                raster_bounds.right - x_cell_size / 2.0,
                reference_display_mask.shape[1],
            )
            y_centers = np.linspace(
                raster_bounds.top - y_cell_size / 2.0,
                raster_bounds.bottom + y_cell_size / 2.0,
                reference_display_mask.shape[0],
            )
            if np.all(reference_display_mask):
                axis.plot(
                    [
                        raster_bounds.left,
                        raster_bounds.right,
                        raster_bounds.right,
                        raster_bounds.left,
                        raster_bounds.left,
                    ],
                    [
                        raster_bounds.bottom,
                        raster_bounds.bottom,
                        raster_bounds.top,
                        raster_bounds.top,
                        raster_bounds.bottom,
                    ],
                    color="#111111",
                    linewidth=1.4,
                    zorder=3,
                )
            elif min(reference_display_mask.shape) > 1:
                axis.contour(
                    x_centers,
                    y_centers,
                    reference_display_mask.astype(np.uint8),
                    levels=[0.5],
                    colors=["#111111"],
                    linewidths=1.3,
                    zorder=3,
                )

        color_bar = figure.colorbar(
            image,
            ax=axis,
            pad=0.025,
            shrink=0.88,
            extend="max",
        )
        color_bar.set_label(
            "Mean pixel sum of |z| across all fitted responses",
            rotation=90,
            labelpad=12,
        )
        color_bar.set_ticks(DISPLAY_COLOR_TICKS)
        axis.set_aspect("equal", adjustable="box")
        if raster_crs is not None and raster_crs.is_geographic:
            axis.set_xlabel("Longitude")
            axis.set_ylabel("Latitude")
        else:
            axis.set_xlabel("Raster x coordinate")
            axis.set_ylabel("Raster y coordinate")
        axis.set_title(
            f"Total standardized departure from modeled reference condition\n"
            f"{ecoregion_name}",
            fontsize=15,
            weight="bold",
            pad=34,
            linespacing=1.25,
        )
        axis.text(
            0.0,
            1.015,
            (
                f"Fixed linear scale: 0 is green, "
                f"{DISPLAY_YELLOW_GREEN_VALUE:g} is yellow-green, and "
                f"{DISPLAY_COLOR_MAXIMUM:g} or more is red; black outlines "
                "contain reference sites"
            ),
            transform=axis.transAxes,
            ha="left",
            va="bottom",
            color="#4B5459",
        )
        axis.legend(
            handles=[
                Patch(
                    facecolor="white",
                    edgecolor="#111111",
                    linewidth=1.3,
                    label="Contains reference sites",
                )
            ],
            loc="best",
            frameon=True,
            facecolor="white",
            edgecolor="none",
            framealpha=0.94,
        )
        warning = (
            " No application mask was supplied, so the modeled surface includes "
            "the usable AOI predictor footprint."
            if not application_mask_supplied
            else ""
        )
        figure.text(
            0.5,
            0.01,
            (
                f"Each display cell is the mean of pixel-level sum(|z_j|) across "
                f"{response_count} responses. Reference pixels are outlined and "
                f"excluded from the color values. Diagnostic only, not an integrity "
                f"score.{warning}"
            ),
            ha="center",
            va="bottom",
            fontsize=8.5,
            color="#4B5459",
            wrap=True,
        )
        axis.spines[["top", "right"]].set_visible(False)
        figure.tight_layout(rect=(0.0, 0.06, 1.0, 1.0))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")

    return {
        "metric": "sum(abs(z_j)) across every fitted ecological response",
        "display_aggregation": (
            "mean among complete-response non-reference source pixels"
        ),
        "display_width": int(display_values.shape[1]),
        "display_height": int(display_values.shape[0]),
        "colored_display_cells": int(len(finite_values)),
        "reference_display_cells": int(np.count_nonzero(reference_display_mask)),
        "contributing_source_pixels": int(value_counts.sum()),
        "reference_source_pixels": int(reference_counts.sum()),
        "response_count": response_count,
        "color_normalization": (
            f"linear over the fixed 0 to {DISPLAY_COLOR_MAXIMUM:g} range"
        ),
        "color_scale_lower_value": 0.0,
        "color_scale_upper_value": DISPLAY_COLOR_MAXIMUM,
        "yellow_green_anchor_value": DISPLAY_YELLOW_GREEN_VALUE,
        "yellow_green_anchor_normalized_position": (
            DISPLAY_YELLOW_GREEN_VALUE / DISPLAY_COLOR_MAXIMUM
        ),
        "cells_at_or_above_color_maximum": cells_at_or_above_maximum,
        "cells_at_or_above_color_maximum_percent": (
            cells_at_or_above_maximum_percent
        ),
        "display_value_minimum": float(finite_values.min()),
        "display_value_median": float(np.median(finite_values)),
        "display_value_maximum": float(finite_values.max()),
    }


def create_departure_percentile_figure(
    departure_percentile_sums: np.ndarray,
    departure_percentile_counts: np.ndarray,
    reference_pixel_counts: np.ndarray,
    raster_bounds: BoundingBox,
    raster_crs: CRS | None,
    response_count: int,
    ecoregion_name: str,
    application_mask_supplied: bool,
    output_path: Path,
) -> dict[str, object]:
    """Map coarsened departure percentiles and reference-site locations.

    Each colored display cell contains the mean ``P_i`` among complete-response,
    non-reference source pixels. Display cells containing one or more reference
    pixels are drawn in deep violet with a white boundary over the percentile
    surface.

    Args:
        departure_percentile_sums: Sum of source-pixel departure percentiles per
            display cell.
        departure_percentile_counts: Contributing non-reference pixels per
            display cell.
        reference_pixel_counts: Reference-site source pixels per display cell.
        raster_bounds: Spatial bounds of the source raster.
        raster_crs: Source raster coordinate reference system, when defined.
        response_count: Number of responses in each multivariate distance.
        ecoregion_name: Human-readable label included in the title.
        application_mask_supplied: Whether inference used an external
            application mask.
        output_path: Destination path for the publication-resolution PNG.

    Returns:
        JSON-ready display dimensions, counts, aggregation, and color limits.

    Raises:
        RuntimeError: If no complete-response non-reference pixels are available.
    """

    mean_display_percentiles = np.full(
        departure_percentile_sums.shape,
        np.nan,
        dtype=np.float64,
    )
    np.divide(
        departure_percentile_sums,
        departure_percentile_counts,
        out=mean_display_percentiles,
        where=departure_percentile_counts > 0,
    )
    displayed_percentiles = mean_display_percentiles[
        np.isfinite(mean_display_percentiles)
    ]
    if len(displayed_percentiles) == 0:
        raise RuntimeError(
            "No non-reference pixels have complete standardized-departure "
            "vectors; the departure-percentile figure cannot be created."
        )

    reference_display_mask = reference_pixel_counts > 0
    raster_extent = (
        raster_bounds.left,
        raster_bounds.right,
        raster_bounds.bottom,
        raster_bounds.top,
    )
    departure_color_map = colormaps["RdYlGn_r"].copy()
    departure_color_map.set_bad("#ECEFF1")
    reference_color_map = ListedColormap([REFERENCE_SITE_COLOR])
    with rc_context({"font.family": "DejaVu Sans", "font.size": 9}):
        figure = Figure(figsize=(10.0, 7.5), facecolor="white")
        FigureCanvasAgg(figure)
        axis = figure.subplots()
        departure_percentile_image = axis.imshow(
            np.ma.masked_invalid(mean_display_percentiles),
            cmap=departure_color_map,
            origin="upper",
            extent=raster_extent,
            interpolation="nearest",
            vmin=0.0,
            vmax=1.0,
        )
        if np.any(reference_display_mask):
            axis.imshow(
                np.ma.masked_where(
                    ~reference_display_mask,
                    np.ones(reference_display_mask.shape, dtype=np.uint8),
                ),
                cmap=reference_color_map,
                origin="upper",
                extent=raster_extent,
                interpolation="nearest",
                vmin=0,
                vmax=1,
                zorder=3,
            )
            x_cell_size = (raster_bounds.right - raster_bounds.left) / len(
                reference_display_mask[0]
            )
            y_cell_size = (raster_bounds.top - raster_bounds.bottom) / len(
                reference_display_mask
            )
            padded_reference_mask = np.pad(reference_display_mask, 1)
            x_centers = np.linspace(
                raster_bounds.left - x_cell_size / 2.0,
                raster_bounds.right + x_cell_size / 2.0,
                padded_reference_mask.shape[1],
            )
            y_centers = np.linspace(
                raster_bounds.top + y_cell_size / 2.0,
                raster_bounds.bottom - y_cell_size / 2.0,
                padded_reference_mask.shape[0],
            )
            axis.contour(
                x_centers,
                y_centers,
                padded_reference_mask.astype(np.uint8),
                levels=[0.5],
                colors=[REFERENCE_SITE_OUTLINE_COLOR],
                linewidths=REFERENCE_SITE_OUTLINE_WIDTH,
                zorder=4,
            )

        color_bar = figure.colorbar(
            departure_percentile_image,
            ax=axis,
            pad=0.025,
            shrink=0.88,
        )
        color_bar.set_label(
            r"Reference-condition departure percentile ($P_i$)",
            rotation=90,
            labelpad=12,
        )
        color_bar.set_ticks(PERCENTILE_COLOR_TICKS)
        axis.set_aspect("equal", adjustable="box")
        if raster_crs is not None and raster_crs.is_geographic:
            axis.set_xlabel("Longitude")
            axis.set_ylabel("Latitude")
        else:
            axis.set_xlabel("Raster x coordinate")
            axis.set_ylabel("Raster y coordinate")
        axis.set_title(
            f"Multivariate departure from reference condition\n{ecoregion_name}",
            fontsize=15,
            weight="bold",
            pad=34,
            linespacing=1.25,
        )
        axis.text(
            0.0,
            1.015,
            (
                "Area-weighted reference percentile: 0 is green, 1 is red, "
                "and violet cells with white boundaries contain reference sites"
            ),
            transform=axis.transAxes,
            ha="left",
            va="bottom",
            color="#4B5459",
        )
        axis.legend(
            handles=[
                Patch(
                    facecolor=REFERENCE_SITE_COLOR,
                    edgecolor=REFERENCE_SITE_OUTLINE_COLOR,
                    linewidth=1.0,
                    label="Contains reference sites",
                )
            ],
            loc="best",
            frameon=True,
            facecolor="white",
            edgecolor="none",
            framealpha=0.94,
        )
        missing_application_mask_warning = (
            " No application mask was supplied, so the modeled surface includes "
            "the usable AOI predictor footprint."
            if not application_mask_supplied
            else ""
        )
        figure.text(
            0.5,
            0.01,
            (
                f"Each display cell is the mean $P_i$ across complete non-reference "
                f"pixels using {response_count} responses. Violet display cells "
                "contain reference pixels, which are excluded from colored values. "
                "This measures departure from reference, not ecological degradation "
                f"by itself.{missing_application_mask_warning}"
            ),
            ha="center",
            va="bottom",
            fontsize=8.5,
            color="#4B5459",
            wrap=True,
        )
        axis.spines[["top", "right"]].set_visible(False)
        figure.tight_layout(rect=(0.0, 0.06, 1.0, 1.0))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")

    return {
        "metric": "area-weighted empirical reference-distance percentile P_i",
        "display_aggregation": (
            "mean among complete-response non-reference source pixels"
        ),
        "display_width": int(mean_display_percentiles.shape[1]),
        "display_height": int(mean_display_percentiles.shape[0]),
        "colored_display_cells": int(len(displayed_percentiles)),
        "reference_display_cells": int(np.count_nonzero(reference_display_mask)),
        "contributing_source_pixels": int(departure_percentile_counts.sum()),
        "reference_source_pixels": int(reference_pixel_counts.sum()),
        "response_count": response_count,
        "color_normalization": "linear over the fixed 0 to 1 range",
        "color_scale_lower_value": 0.0,
        "color_scale_upper_value": 1.0,
        "reference_color": REFERENCE_SITE_COLOR,
        "reference_outline_color": REFERENCE_SITE_OUTLINE_COLOR,
        "reference_outline_width_points": REFERENCE_SITE_OUTLINE_WIDTH,
        "display_value_minimum": float(displayed_percentiles.min()),
        "display_value_median": float(np.median(displayed_percentiles)),
        "display_value_maximum": float(displayed_percentiles.max()),
    }


def iter_cached_tile_windows(
    cached_tile: CachedRasterTile,
    output_grid: InferenceOutputGrid,
    window_size_pixels: int,
):
    """Yield matching bounded source and stitched-output windows for one tile.

    Args:
        cached_tile: Validated source tile.
        output_grid: Stitched AOI output grid.
        window_size_pixels: Maximum processing-window width and height.

    Yields:
        Matching source and destination windows no larger than the configured
        processing size.
    """

    source_overlap, destination_overlap = output_grid.overlapping_windows(
        cached_tile
    )
    for local_row_offset in range(0, int(source_overlap.height), window_size_pixels):
        for local_column_offset in range(
            0,
            int(source_overlap.width),
            window_size_pixels,
        ):
            window_width = min(
                window_size_pixels,
                int(source_overlap.width) - local_column_offset,
            )
            window_height = min(
                window_size_pixels,
                int(source_overlap.height) - local_row_offset,
            )
            yield (
                Window(
                    int(source_overlap.col_off) + local_column_offset,
                    int(source_overlap.row_off) + local_row_offset,
                    window_width,
                    window_height,
                ),
                Window(
                    int(destination_overlap.col_off) + local_column_offset,
                    int(destination_overlap.row_off) + local_row_offset,
                    window_width,
                    window_height,
                ),
            )


def output_artifacts_are_compatible(
    artifact_paths: InferenceArtifactPaths,
    output_grid: InferenceOutputGrid,
    response_count: int,
) -> bool:
    """Check whether existing rasters can safely resume tile writes.

    Args:
        artifact_paths: Expected inference artifact paths.
        output_grid: Required stitched raster grid.
        response_count: Number of fitted ecological responses.

    Returns:
        Whether all five raster artifacts have compatible dimensions, CRS,
        transform, band count, and data type.
    """

    expected_layouts = (
        (artifact_paths.expected_reference, response_count, "float32"),
        (artifact_paths.observed_minus_expected, response_count, "float32"),
        (artifact_paths.standardized_deviation, response_count, "float32"),
        (artifact_paths.departure_percentile, 1, "float32"),
        (artifact_paths.inference_status, 2, "uint8"),
    )
    try:
        for raster_path, expected_band_count, expected_dtype in expected_layouts:
            with rasterio.open(raster_path) as source:
                if (
                    source.width != output_grid.width
                    or source.height != output_grid.height
                    or source.count != expected_band_count
                    or source.crs != output_grid.crs
                    or not source.transform.almost_equals(output_grid.transform)
                    or any(dtype != expected_dtype for dtype in source.dtypes)
                ):
                    return False
    except (OSError, rasterio.errors.RasterioError):
        return False
    return True


def calculate_inference_tile(
    cached_tile: CachedRasterTile,
    application_mask_path: Path | None,
    window_size_pixels: int,
    projected_aoi: BaseGeometry,
    output_grid: InferenceOutputGrid,
    required_band_indices: tuple[int, ...],
    predictor_names: tuple[str, ...],
    response_models: tuple[ResponseModel, ...],
    reference_calibration: ReferenceDepartureCalibration,
    maximum_predictor_missing_fraction: float,
    reference_band_offset: int,
) -> ComputedInferenceTile:
    """Calculate one cache tile without writing shared output rasters.

    Args:
        cached_tile: Validated multiband source tile.
        application_mask_path: Optional raster selecting inference pixels.
        window_size_pixels: Maximum width and height processed at once.
        projected_aoi: Analysis AOI transformed to the cache-grid CRS.
        output_grid: Stitched output placement.
        required_band_indices: Ordered source bands to read.
        predictor_names: Ordered model predictor names.
        response_models: Ordered fitted ecological-response models.
        reference_calibration: Reference distribution for percentiles.
        maximum_predictor_missing_fraction: Training missingness contract.
        reference_band_offset: Reference layer offset in the source read.

    Returns:
        Calculated arrays for every output window in the cache tile.

    Raises:
        RuntimeError: If a fitted model produces a nonfinite prediction.
    """

    predictor_count = len(predictor_names)
    calculated_windows = []
    with ExitStack() as stack:
        tile_source = stack.enter_context(rasterio.open(cached_tile.path))
        if application_mask_path is None:
            aligned_application_mask = None
        else:
            application_mask_source = stack.enter_context(
                rasterio.open(application_mask_path)
            )
            aligned_application_mask = stack.enter_context(
                WarpedVRT(
                    application_mask_source,
                    crs=output_grid.crs,
                    transform=output_grid.transform,
                    width=output_grid.width,
                    height=output_grid.height,
                    resampling=Resampling.nearest,
                )
            )

        for source_window, destination_window in iter_cached_tile_windows(
            cached_tile,
            output_grid,
            window_size_pixels,
        ):
            masked_window_values = tile_source.read(
                required_band_indices,
                window=source_window,
                masked=True,
            )
            window_values = np.asarray(
                np.ma.getdata(masked_window_values),
                dtype=np.float64,
            )
            window_value_validity = ~np.ma.getmaskarray(masked_window_values)
            window_value_validity &= np.isfinite(window_values)
            window_values[~window_value_validity] = np.nan

            window_shape = (
                int(destination_window.height),
                int(destination_window.width),
            )
            aoi_pixel_mask = geometry_mask(
                [mapping(projected_aoi)],
                out_shape=window_shape,
                transform=window_transform(
                    destination_window,
                    output_grid.transform,
                ),
                invert=True,
            )
            predictor_values = window_values[:predictor_count]
            predictor_value_validity = window_value_validity[:predictor_count]
            missing_predictor_counts = np.count_nonzero(
                ~predictor_value_validity,
                axis=0,
            ).astype(np.uint8)
            if aligned_application_mask is None:
                target_pixel_mask = aoi_pixel_mask & np.any(
                    predictor_value_validity,
                    axis=0,
                )
            else:
                masked_application_values = aligned_application_mask.read(
                    1,
                    window=destination_window,
                    masked=True,
                )
                application_mask_values = np.asarray(
                    np.ma.getdata(masked_application_values)
                )
                target_pixel_mask = (
                    aoi_pixel_mask
                    & ~np.ma.getmaskarray(masked_application_values)
                    & np.isfinite(application_mask_values)
                    & (application_mask_values == 1)
                )
            usable_pixel_mask = target_pixel_mask & (
                missing_predictor_counts / predictor_count
                <= maximum_predictor_missing_fraction
            )
            reference_pixel_mask = (
                aoi_pixel_mask
                & window_value_validity[reference_band_offset]
                & (window_values[reference_band_offset] != 0)
            )
            complete_response_pixel_mask = usable_pixel_mask.copy()
            for response_model_offset in range(len(response_models)):
                complete_response_pixel_mask &= window_value_validity[
                    predictor_count + response_model_offset
                ]

            expected_output = np.full(
                (len(response_models), *window_shape),
                FLOAT_NODATA,
                dtype=np.float32,
            )
            deviation_output = np.full_like(expected_output, FLOAT_NODATA)
            standardized_output = np.full_like(expected_output, FLOAT_NODATA)
            percentile_output = np.full(
                window_shape,
                FLOAT_NODATA,
                dtype=np.float32,
            )
            status_output = np.full(
                window_shape,
                STATUS_NODATA,
                dtype=np.uint8,
            )
            status_output[aoi_pixel_mask] = STATUS_OUTSIDE_TARGET
            status_output[target_pixel_mask] = STATUS_INSUFFICIENT_PREDICTORS
            status_output[usable_pixel_mask] = STATUS_PREDICTED
            imputation_output = np.full(
                window_shape,
                STATUS_NODATA,
                dtype=np.uint8,
            )
            imputation_output[target_pixel_mask] = missing_predictor_counts[
                target_pixel_mask
            ]

            usable_pixel_mask_flat = usable_pixel_mask.ravel()
            if np.any(usable_pixel_mask_flat):
                predictor_table = pd.DataFrame(
                    predictor_values.reshape(predictor_count, -1).T[
                        usable_pixel_mask_flat
                    ],
                    columns=predictor_names,
                )
                for response_model_offset, response_model in enumerate(
                    response_models
                ):
                    expected_reference_values = predict_expected_response(
                        response_model.bundle,
                        predictor_table,
                    )
                    if not np.isfinite(expected_reference_values).all():
                        raise RuntimeError(
                            f"{response_model.response_band} produced a "
                            "nonfinite prediction."
                        )
                    expected_output_flat = expected_output[
                        response_model_offset
                    ].ravel()
                    expected_output_flat[usable_pixel_mask_flat] = (
                        expected_reference_values.astype(np.float32)
                    )
                    response_offset = predictor_count + response_model_offset
                    observed_values = window_values[response_offset].ravel()
                    deviation_pixel_mask = (
                        usable_pixel_mask_flat
                        & window_value_validity[response_offset].ravel()
                    )
                    deviation_values = (
                        observed_values[deviation_pixel_mask]
                        - expected_output_flat[deviation_pixel_mask]
                    )
                    standardized_values = (
                        deviation_values / response_model.reference_rmse
                    )
                    deviation_output[response_model_offset].ravel()[
                        deviation_pixel_mask
                    ] = deviation_values.astype(np.float32)
                    standardized_output[response_model_offset].ravel()[
                        deviation_pixel_mask
                    ] = standardized_values.astype(np.float32)

            complete_non_reference_pixel_mask = (
                complete_response_pixel_mask & ~reference_pixel_mask
            )
            standardized_departure_vectors = standardized_output[
                :, complete_non_reference_pixel_mask
            ].T.astype(np.float64)
            mahalanobis_distances = (
                reference_calibration.calculate_mahalanobis_distances(
                    standardized_departure_vectors
                )
            )
            departure_percentiles = (
                reference_calibration.calculate_reference_departure_percentiles(
                    mahalanobis_distances
                )
            )
            percentile_output[complete_non_reference_pixel_mask] = (
                departure_percentiles.astype(np.float32)
            )
            calculated_windows.append(
                InferenceWindowResult(
                    destination_window=destination_window,
                    expected_reference=expected_output,
                    observed_minus_expected=deviation_output,
                    standardized_deviation=standardized_output,
                    departure_percentile=percentile_output,
                    inference_status=np.stack(
                        [status_output, imputation_output]
                    ),
                )
            )

    return ComputedInferenceTile(
        tile_id=cached_tile.tile.tile_id,
        windows=tuple(calculated_windows),
        worker_process_id=os.getpid(),
    )


def initialize_inference_worker(
    context_path: Path,
) -> None:
    """Initialize one process from the shared model-context artifact.

    Args:
        context_path: Joblib artifact containing models, calibration, AOI, and
            raster settings shared by every worker process.

    Returns:
        None: The context and numerical-library limit are process globals.
    """

    global _INFERENCE_WORKER_CONTEXT
    global _INFERENCE_NUMERICAL_THREAD_LIMIT
    _INFERENCE_WORKER_CONTEXT = joblib.load(context_path, mmap_mode="r")
    # Tile-level processes provide the parallelism. Allowing NumPy or
    # scikit-learn to create another thread pool inside every process would
    # oversubscribe the machine and can make inference substantially slower.
    _INFERENCE_NUMERICAL_THREAD_LIMIT = threadpool_limits(limits=1)


def calculate_inference_tile_in_worker(
    cached_tile: CachedRasterTile,
) -> ComputedInferenceTile:
    """Calculate one tile using the process-global initialized context.

    Args:
        cached_tile: Validated multiband source tile assigned to this process.

    Returns:
        Calculated tile arrays returned to the parent process for writing.
    """

    context = _INFERENCE_WORKER_CONTEXT
    if context is None:
        raise RuntimeError("Inference worker context was not initialized.")
    return calculate_inference_tile_from_context(cached_tile, context)


def calculate_inference_tile_from_context(
    cached_tile: CachedRasterTile,
    context: InferenceWorkerContext,
) -> ComputedInferenceTile:
    """Apply a reusable worker context to one source tile.

    Args:
        cached_tile: Validated multiband source tile.
        context: Models, calibration, AOI, and raster calculation settings.

    Returns:
        Calculated tile arrays ready for serialized output.
    """

    return calculate_inference_tile(
        cached_tile,
        application_mask_path=context.application_mask_path,
        window_size_pixels=context.window_size_pixels,
        projected_aoi=context.projected_aoi,
        output_grid=context.output_grid,
        required_band_indices=context.required_band_indices,
        predictor_names=context.predictor_names,
        response_models=context.response_models,
        reference_calibration=context.reference_calibration,
        maximum_predictor_missing_fraction=(
            context.maximum_predictor_missing_fraction
        ),
        reference_band_offset=context.reference_band_offset,
    )


def iter_calculated_inference_tiles(
    cached_tiles: tuple[CachedRasterTile, ...],
    worker_context: InferenceWorkerContext,
    worker_count: int,
) -> Iterator[ComputedInferenceTile]:
    """Yield calculated tiles from the parent or a bounded process pool.

    Args:
        cached_tiles: Source tiles requiring inference.
        worker_context: Reusable model and raster calculation context.
        worker_count: Number of tile calculations allowed concurrently.

    Yields:
        Completed tile results in completion order.
    """

    if worker_count == 1:
        for cached_tile in cached_tiles:
            yield calculate_inference_tile_from_context(
                cached_tile,
                worker_context,
            )
        return

    tile_iterator = iter(cached_tiles)
    process_context = multiprocessing.get_context("spawn")
    with TemporaryDirectory(prefix="nhi-inference-workers-") as temporary_directory:
        worker_context_path = Path(temporary_directory) / "worker_context.joblib"
        # Windows spawn otherwise pickles the complete model context serially
        # for every process. A shared, uncompressed artifact is written once;
        # workers load its arrays read-only through Joblib memory mapping.
        joblib.dump(worker_context, worker_context_path, compress=0)
        with ProcessPoolExecutor(
            max_workers=worker_count,
            mp_context=process_context,
            initializer=initialize_inference_worker,
            initargs=(worker_context_path,),
        ) as executor:
            pending_futures: set[Future[ComputedInferenceTile]] = set()
            for _ in range(worker_count):
                pending_futures.add(
                    executor.submit(
                        calculate_inference_tile_in_worker,
                        next(tile_iterator),
                    )
                )

            while pending_futures:
                completed_futures, _ = wait(
                    tuple(pending_futures),
                    return_when=FIRST_COMPLETED,
                )
                for completed_future in completed_futures:
                    pending_futures.remove(completed_future)
                    computed_tile = completed_future.result()
                    # Replenish this process before the parent writes its
                    # result so serialized output never holds a worker idle.
                    try:
                        cached_tile = next(tile_iterator)
                    except StopIteration:
                        pass
                    else:
                        pending_futures.add(
                            executor.submit(
                                calculate_inference_tile_in_worker,
                                cached_tile,
                            )
                        )
                    yield computed_tile


def write_computed_inference_tiles(
    computed_tiles: Sequence[ComputedInferenceTile],
    artifact_paths: InferenceArtifactPaths,
) -> None:
    """Persist one calculated tile batch and flush before checkpointing.

    Args:
        computed_tiles: Tile arrays returned by inference workers.
        artifact_paths: Shared stitched output rasters.

    Returns:
        None: Every tile window is written and all datasets are closed once.
    """

    with ExitStack() as stack:
        expected_destination = stack.enter_context(
            rasterio.open(artifact_paths.expected_reference, "r+")
        )
        deviation_destination = stack.enter_context(
            rasterio.open(artifact_paths.observed_minus_expected, "r+")
        )
        standardized_destination = stack.enter_context(
            rasterio.open(artifact_paths.standardized_deviation, "r+")
        )
        percentile_destination = stack.enter_context(
            rasterio.open(artifact_paths.departure_percentile, "r+")
        )
        status_destination = stack.enter_context(
            rasterio.open(artifact_paths.inference_status, "r+")
        )
        for computed_tile in computed_tiles:
            for window_result in computed_tile.windows:
                destination_window = window_result.destination_window
                expected_destination.write(
                    window_result.expected_reference,
                    window=destination_window,
                )
                deviation_destination.write(
                    window_result.observed_minus_expected,
                    window=destination_window,
                )
                standardized_destination.write(
                    window_result.standardized_deviation,
                    window=destination_window,
                )
                percentile_destination.write(
                    window_result.departure_percentile,
                    1,
                    window=destination_window,
                )
                status_destination.write(
                    window_result.inference_status,
                    window=destination_window,
                )


def write_inference_tiles(
    analysis_configuration: AnalysisConfiguration,
    analysis_cache_tiles: AnalysisCacheTiles,
    output_grid: InferenceOutputGrid,
    artifact_paths: InferenceArtifactPaths,
    response_models: tuple[ResponseModel, ...],
    reference_calibration: ReferenceDepartureCalibration,
    maximum_predictor_missing_fraction: float,
    inference_fingerprint: str,
    ecoregion_name: str,
    model_run_directory: Path,
    show_progress: bool,
) -> tuple[dict[str, object] | None, int, int]:
    """Write resumable model results from cache tiles into stitched rasters.

    A cache tile is recorded as complete only after all of its output windows
    have been written to all five destination rasters. A rerun with identical
    effective inputs skips those completed tiles and reconstructs reports and
    figures from the stitched outputs.

    Args:
        analysis_configuration: Complete analysis and inference contract.
        analysis_cache_tiles: Validated source tiles and projected AOI.
        output_grid: Stitched destination grid.
        artifact_paths: Raster and checkpoint destinations.
        response_models: Ordered fitted ecological-response models.
        reference_calibration: Reference distribution for departure percentiles.
        maximum_predictor_missing_fraction: Training-time missingness limit.
        inference_fingerprint: Hash of source, model, and configuration inputs.
        ecoregion_name: Model-run name stored in output tags.
        model_run_directory: Resolved model run directory.
        show_progress: Whether to render tqdm progress.

    Returns:
        Application-mask metadata, resumed tile count, and newly processed tile
        count.

    Raises:
        ValueError: If model bands are absent from the configured raster stack.
        RuntimeError: If a fitted model produces a nonfinite prediction.
    """

    configured_band_names = analysis_configuration.band_names()
    configured_band_indices = {
        band_name: band_index
        for band_index, band_name in enumerate(configured_band_names, start=1)
    }
    predictor_names = response_models[0].predictor_names
    response_names = tuple(model.response_name for model in response_models)
    required_band_names = (*predictor_names, *response_names)
    missing_band_names = [
        band_name
        for band_name in required_band_names
        if band_name not in configured_band_indices
    ]
    if missing_band_names:
        raise ValueError(
            "Configured raster stack is missing model bands: "
            + ", ".join(missing_band_names)
        )
    reference_band_name = next(
        band_name
        for band_name, band_definition in zip(
            configured_band_names,
            analysis_configuration.bands,
            strict=True,
        )
        if band_definition.role == "reference"
    )
    required_band_indices = [
        configured_band_indices[band_name] for band_name in required_band_names
    ]
    required_band_indices.append(configured_band_indices[reference_band_name])
    predictor_count = len(predictor_names)
    reference_band_offset = len(required_band_names)

    checkpoint_metadata: dict[str, object] = {}
    if artifact_paths.checkpoint.is_file():
        try:
            checkpoint_metadata = json.loads(
                artifact_paths.checkpoint.read_text(encoding="utf-8")
            )
        except (OSError, ValueError):
            checkpoint_metadata = {}
    can_resume = (
        checkpoint_metadata.get("schema_version")
        == INFERENCE_CHECKPOINT_SCHEMA_VERSION
        and checkpoint_metadata.get("inference_fingerprint")
        == inference_fingerprint
        and output_artifacts_are_compatible(
            artifact_paths,
            output_grid,
            len(response_models),
        )
    )
    completed_tile_ids = (
        set(checkpoint_metadata.get("completed_tile_ids", []))
        if can_resume
        else set()
    )
    expected_tile_ids = {
        cached_tile.tile.tile_id for cached_tile in analysis_cache_tiles.tiles
    }
    completed_tile_ids.intersection_update(expected_tile_ids)

    float_profile = {
        "driver": "GTiff",
        "width": output_grid.width,
        "height": output_grid.height,
        "count": len(response_models),
        "dtype": "float32",
        "crs": output_grid.crs,
        "transform": output_grid.transform,
        "nodata": FLOAT_NODATA,
        "compress": "deflate",
        "predictor": 3,
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
        "interleave": "band",
        "BIGTIFF": "IF_SAFER",
        "SPARSE_OK": True,
    }
    status_profile = {
        **float_profile,
        "count": 2,
        "dtype": "uint8",
        "nodata": STATUS_NODATA,
        "predictor": 2,
    }
    percentile_profile = {**float_profile, "count": 1}
    output_mode = "r+" if can_resume else "w"

    application_mask_path = analysis_configuration.inference.application_mask_path
    application_mask_metadata = None
    with ExitStack() as stack:
        if application_mask_path is not None:
            application_mask_source = stack.enter_context(
                rasterio.open(application_mask_path)
            )
            application_mask_metadata = {
                "path": str(application_mask_path),
                "selected_value": 1,
                "resampling": "nearest",
                "source_width": application_mask_source.width,
                "source_height": application_mask_source.height,
                "source_crs": (
                    str(application_mask_source.crs)
                    if application_mask_source.crs
                    else None
                ),
                "source_transform": list(application_mask_source.transform),
            }

        expected_destination = stack.enter_context(
            rasterio.open(
                artifact_paths.expected_reference,
                output_mode,
                **(float_profile if output_mode == "w" else {}),
            )
        )
        deviation_destination = stack.enter_context(
            rasterio.open(
                artifact_paths.observed_minus_expected,
                output_mode,
                **(float_profile if output_mode == "w" else {}),
            )
        )
        standardized_destination = stack.enter_context(
            rasterio.open(
                artifact_paths.standardized_deviation,
                output_mode,
                **(float_profile if output_mode == "w" else {}),
            )
        )
        percentile_destination = stack.enter_context(
            rasterio.open(
                artifact_paths.departure_percentile,
                output_mode,
                **(percentile_profile if output_mode == "w" else {}),
            )
        )
        status_destination = stack.enter_context(
            rasterio.open(
                artifact_paths.inference_status,
                output_mode,
                **(status_profile if output_mode == "w" else {}),
            )
        )

        if output_mode == "w":
            shared_output_tags = {
                "analysis_name": analysis_configuration.analysis_name,
                "ecoregion_name": ecoregion_name,
                "input_raster_cache": str(analysis_cache_tiles.manifest_path),
                "stack_identifier": analysis_cache_tiles.stack_identifier,
                "model_run_directory": str(model_run_directory),
                "application_mask": str(application_mask_path or "not_supplied"),
                "application_mask_selected_value": "1",
                "application_mask_resampling": "nearest",
            }
            expected_destination.update_tags(
                artifact_type="expected_reference_condition",
                **shared_output_tags,
            )
            deviation_destination.update_tags(
                artifact_type="observed_minus_expected_reference_condition",
                **shared_output_tags,
            )
            standardized_destination.update_tags(
                artifact_type="standardized_reference_condition_deviation",
                interpretation=(
                    "observed minus expected divided by pooled out-of-fold "
                    "reference RMSE"
                ),
                **shared_output_tags,
            )
            percentile_destination.update_tags(
                artifact_type="reference_condition_departure_percentile",
                interpretation=(
                    "area-weighted empirical percentile of covariance-aware "
                    "distance among complete out-of-fold reference vectors"
                ),
                response_bands=",".join(reference_calibration.response_bands),
                covariance_shrinkage=str(
                    analysis_configuration.inference.covariance_shrinkage
                ),
                value_minimum="0",
                value_maximum="1",
                reference_pixels="nodata",
                **shared_output_tags,
            )
            percentile_destination.set_band_description(
                1,
                "reference_departure_percentile",
            )
            status_destination.update_tags(
                artifact_type="reference_condition_inference_status",
                status_0="inside AOI but outside inference target",
                status_1="target pixel with excessive predictor missingness",
                status_2="reference-condition predictions written",
                **shared_output_tags,
            )
            status_destination.set_band_description(1, "inference_status")
            status_destination.set_band_description(2, "imputed_predictor_count")
            for output_band_index, response_model in enumerate(
                response_models,
                start=1,
            ):
                expected_destination.set_band_description(
                    output_band_index,
                    f"{response_model.response_band}_expected_reference",
                )
                deviation_destination.set_band_description(
                    output_band_index,
                    f"{response_model.response_band}_observed_minus_expected",
                )
                standardized_destination.set_band_description(
                    output_band_index,
                    f"{response_model.response_band}_standardized_deviation",
                )
                response_tags = {
                    "response_band": response_model.response_band,
                    "display_name": response_model.display_name,
                    "source_response_band": response_model.response_name,
                    "reference_residual_rmse_oof": str(
                        response_model.reference_rmse
                    ),
                    "model_path": str(response_model.path),
                }
                expected_destination.update_tags(
                    output_band_index,
                    **response_tags,
                )
                deviation_destination.update_tags(
                    output_band_index,
                    **response_tags,
                )
                standardized_destination.update_tags(
                    output_band_index,
                    **response_tags,
                )

        checkpoint_metadata = {
            "schema_version": INFERENCE_CHECKPOINT_SCHEMA_VERSION,
            "inference_fingerprint": inference_fingerprint,
            "analysis_name": analysis_configuration.analysis_name,
            "stack_identifier": analysis_cache_tiles.stack_identifier,
            "tile_count": len(analysis_cache_tiles.tiles),
            "completed_tile_ids": sorted(completed_tile_ids),
        }
        temporary_checkpoint_path = artifact_paths.checkpoint.with_suffix(
            ".json.tmp"
        )
        temporary_checkpoint_path.write_text(
            json.dumps(checkpoint_metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_checkpoint_path, artifact_paths.checkpoint)
        # Closing after initialization persists headers and band metadata. Each
        # completed batch below reopens and closes all outputs before its tile
        # IDs are checkpointed, so a recorded tile never depends on GDAL write
        # buffers that were still in memory when the process stopped.
        expected_destination.close()
        deviation_destination.close()
        standardized_destination.close()
        percentile_destination.close()
        status_destination.close()

        tiles_to_process = tuple(
            cached_tile
            for cached_tile in analysis_cache_tiles.tiles
            if cached_tile.tile.tile_id not in completed_tile_ids
        )
        configured_worker_count = (
            analysis_configuration.inference.worker_count
        )
        active_worker_count = min(
            configured_worker_count,
            len(tiles_to_process),
        )
        newly_processed_tile_count = 0
        failed_tile_count = 0
        utilized_worker_process_ids: set[int] = set()
        with tqdm(
            total=len(analysis_cache_tiles.tiles),
            initial=len(completed_tile_ids),
            desc="Applying response models",
            unit="tile",
            dynamic_ncols=True,
            disable=not show_progress,
        ) as tile_progress:
            tile_progress.set_postfix(
                workers=active_worker_count,
                processes=0,
                cached=len(completed_tile_ids),
                processed=0,
                failed=0,
            )
            if tiles_to_process:
                worker_context = InferenceWorkerContext(
                    application_mask_path=application_mask_path,
                    window_size_pixels=(
                        analysis_configuration.inference.window_size_pixels
                    ),
                    projected_aoi=analysis_cache_tiles.projected_aoi,
                    output_grid=output_grid,
                    required_band_indices=tuple(required_band_indices),
                    predictor_names=predictor_names,
                    response_models=response_models,
                    reference_calibration=reference_calibration,
                    maximum_predictor_missing_fraction=(
                        maximum_predictor_missing_fraction
                    ),
                    reference_band_offset=reference_band_offset,
                )
                try:
                    calculated_tiles = iter_calculated_inference_tiles(
                        tiles_to_process,
                        worker_context,
                        active_worker_count,
                    )
                    while computed_tile_batch := tuple(
                        islice(calculated_tiles, active_worker_count)
                    ):
                        utilized_worker_process_ids.update(
                            computed_tile.worker_process_id
                            for computed_tile in computed_tile_batch
                        )
                        write_computed_inference_tiles(
                            computed_tile_batch,
                            artifact_paths,
                        )
                        completed_tile_ids.update(
                            computed_tile.tile_id
                            for computed_tile in computed_tile_batch
                        )
                        newly_processed_tile_count += len(computed_tile_batch)
                        checkpoint_metadata["completed_tile_ids"] = sorted(
                            completed_tile_ids
                        )
                        temporary_checkpoint_path.write_text(
                            json.dumps(
                                checkpoint_metadata,
                                indent=2,
                                sort_keys=True,
                            )
                            + "\n",
                            encoding="utf-8",
                        )
                        os.replace(
                            temporary_checkpoint_path,
                            artifact_paths.checkpoint,
                        )
                        tile_progress.update(len(computed_tile_batch))
                        tile_progress.set_postfix(
                            workers=active_worker_count,
                            processes=len(utilized_worker_process_ids),
                            cached=(
                                len(completed_tile_ids)
                                - newly_processed_tile_count
                            ),
                            processed=newly_processed_tile_count,
                            failed=failed_tile_count,
                        )
                except Exception:
                    failed_tile_count += 1
                    tile_progress.set_postfix(
                        workers=active_worker_count,
                        processes=len(utilized_worker_process_ids),
                        cached=(
                            len(completed_tile_ids)
                            - newly_processed_tile_count
                        ),
                        processed=newly_processed_tile_count,
                        failed=failed_tile_count,
                    )
                    raise

    return (
        application_mask_metadata,
        len(completed_tile_ids) - newly_processed_tile_count,
        newly_processed_tile_count,
    )


def summarize_inference_outputs(
    analysis_configuration: AnalysisConfiguration,
    analysis_cache_tiles: AnalysisCacheTiles,
    output_grid: InferenceOutputGrid,
    artifact_paths: InferenceArtifactPaths,
    response_models: tuple[ResponseModel, ...],
    show_progress: bool,
) -> InferenceOutputStatistics:
    """Reconstruct reports and display arrays from completed stitched outputs.

    Re-reading outputs makes summaries deterministic after either a fresh run or
    a resumed run and avoids persisting fragile partial statistics in the tile
    checkpoint.

    Args:
        analysis_configuration: Complete raster-band and window contract.
        analysis_cache_tiles: Validated source tiles containing reference labels.
        output_grid: Stitched destination grid.
        artifact_paths: Completed raster artifacts.
        response_models: Ordered fitted ecological-response models.
        show_progress: Whether to render tqdm progress.

    Returns:
        Streaming coverage, response, percentile, and display summaries.
    """

    reference_band_index = next(
        band_index
        for band_index, band_definition in enumerate(
            analysis_configuration.bands,
            start=1,
        )
        if band_definition.role == "reference"
    )
    display_scale = min(
        1.0,
        MAXIMUM_DISPLAY_DIMENSION / max(output_grid.width, output_grid.height),
    )
    display_width = max(1, round(output_grid.width * display_scale))
    display_height = max(1, round(output_grid.height * display_scale))
    aggregate_value_sums = np.zeros(
        (display_height, display_width),
        dtype=np.float64,
    )
    aggregate_value_counts = np.zeros(
        (display_height, display_width),
        dtype=np.int64,
    )
    percentile_value_sums = np.zeros(
        (display_height, display_width),
        dtype=np.float64,
    )
    percentile_value_counts = np.zeros(
        (display_height, display_width),
        dtype=np.int64,
    )
    reference_pixel_counts = np.zeros(
        (display_height, display_width),
        dtype=np.int64,
    )
    response_statistics = {
        model.response_band: ResponseStatistics() for model in response_models
    }
    departure_percentile_statistics = DeparturePercentileStatistics()
    aoi_pixel_count = 0
    target_pixel_count = 0
    predicted_pixel_count = 0
    insufficient_predictor_pixel_count = 0
    imputed_pixel_count = 0

    with (
        rasterio.open(artifact_paths.expected_reference) as expected_source,
        rasterio.open(artifact_paths.standardized_deviation) as standardized_source,
        rasterio.open(artifact_paths.departure_percentile) as percentile_source,
        rasterio.open(artifact_paths.inference_status) as status_source,
    ):
        for cached_tile in tqdm(
            analysis_cache_tiles.tiles,
            total=len(analysis_cache_tiles.tiles),
            desc="Summarizing inference outputs",
            unit="tile",
            dynamic_ncols=True,
            disable=not show_progress,
        ):
            with rasterio.open(cached_tile.path) as tile_source:
                for source_window, destination_window in iter_cached_tile_windows(
                    cached_tile,
                    output_grid,
                    analysis_configuration.inference.window_size_pixels,
                ):
                    masked_status_values = status_source.read(
                        1,
                        window=destination_window,
                        masked=True,
                    )
                    status_values = np.asarray(
                        np.ma.getdata(masked_status_values)
                    )
                    aoi_pixel_mask = ~np.ma.getmaskarray(masked_status_values)
                    insufficient_pixel_mask = (
                        aoi_pixel_mask
                        & (status_values == STATUS_INSUFFICIENT_PREDICTORS)
                    )
                    predicted_pixel_mask = (
                        aoi_pixel_mask & (status_values == STATUS_PREDICTED)
                    )
                    aoi_pixel_count += int(np.count_nonzero(aoi_pixel_mask))
                    insufficient_predictor_pixel_count += int(
                        np.count_nonzero(insufficient_pixel_mask)
                    )
                    window_predicted_pixel_count = int(
                        np.count_nonzero(predicted_pixel_mask)
                    )
                    predicted_pixel_count += window_predicted_pixel_count
                    target_pixel_count += (
                        window_predicted_pixel_count
                        + int(np.count_nonzero(insufficient_pixel_mask))
                    )
                    masked_imputation_values = status_source.read(
                        2,
                        window=destination_window,
                        masked=True,
                    )
                    imputed_pixel_count += int(
                        np.count_nonzero(
                            predicted_pixel_mask
                            & ~np.ma.getmaskarray(masked_imputation_values)
                            & (np.ma.getdata(masked_imputation_values) > 0)
                        )
                    )

                    masked_expected_values = expected_source.read(
                        window=destination_window,
                        masked=True,
                    )
                    masked_standardized_values = standardized_source.read(
                        window=destination_window,
                        masked=True,
                    )
                    standardized_values = np.asarray(
                        np.ma.getdata(masked_standardized_values),
                        dtype=np.float64,
                    )
                    standardized_validity = ~np.ma.getmaskarray(
                        masked_standardized_values
                    )
                    for response_offset, response_model in enumerate(
                        response_models
                    ):
                        expected_pixel_count = int(
                            np.count_nonzero(
                                ~np.ma.getmaskarray(masked_expected_values)[
                                    response_offset
                                ]
                            )
                        )
                        response_statistics[
                            response_model.response_band
                        ].update(
                            expected_pixel_count,
                            standardized_values[response_offset][
                                standardized_validity[response_offset]
                            ],
                        )

                    masked_percentile_values = percentile_source.read(
                        1,
                        window=destination_window,
                        masked=True,
                    )
                    percentile_validity = ~np.ma.getmaskarray(
                        masked_percentile_values
                    )
                    departure_percentile_statistics.update(
                        np.asarray(np.ma.getdata(masked_percentile_values))[
                            percentile_validity
                        ]
                    )
                    masked_reference_values = tile_source.read(
                        reference_band_index,
                        window=source_window,
                        masked=True,
                    )
                    reference_values = np.asarray(
                        np.ma.getdata(masked_reference_values)
                    )
                    reference_pixel_mask = (
                        aoi_pixel_mask
                        & ~np.ma.getmaskarray(masked_reference_values)
                        & np.isfinite(reference_values)
                        & (reference_values != 0)
                    )
                    complete_non_reference_pixel_mask = (
                        np.all(standardized_validity, axis=0)
                        & ~reference_pixel_mask
                    )
                    total_absolute_standardized_departures = np.sum(
                        np.abs(standardized_values),
                        axis=0,
                    )
                    output_rows = np.arange(
                        int(destination_window.row_off),
                        int(
                            destination_window.row_off
                            + destination_window.height
                        ),
                    )
                    output_columns = np.arange(
                        int(destination_window.col_off),
                        int(
                            destination_window.col_off
                            + destination_window.width
                        ),
                    )
                    display_rows = np.minimum(
                        output_rows * display_height // output_grid.height,
                        display_height - 1,
                    )
                    display_columns = np.minimum(
                        output_columns * display_width // output_grid.width,
                        display_width - 1,
                    )
                    display_cell_indices = (
                        display_rows[:, np.newaxis] * display_width
                        + display_columns[np.newaxis, :]
                    )
                    np.add.at(
                        aggregate_value_sums.ravel(),
                        display_cell_indices[complete_non_reference_pixel_mask],
                        total_absolute_standardized_departures[
                            complete_non_reference_pixel_mask
                        ],
                    )
                    np.add.at(
                        aggregate_value_counts.ravel(),
                        display_cell_indices[complete_non_reference_pixel_mask],
                        1,
                    )
                    np.add.at(
                        percentile_value_sums.ravel(),
                        display_cell_indices[percentile_validity],
                        np.asarray(np.ma.getdata(masked_percentile_values))[
                            percentile_validity
                        ],
                    )
                    np.add.at(
                        percentile_value_counts.ravel(),
                        display_cell_indices[percentile_validity],
                        1,
                    )
                    np.add.at(
                        reference_pixel_counts.ravel(),
                        display_cell_indices[reference_pixel_mask],
                        1,
                    )

    return InferenceOutputStatistics(
        response_statistics=response_statistics,
        departure_percentile_statistics=departure_percentile_statistics,
        aggregate_value_sums=aggregate_value_sums,
        aggregate_value_counts=aggregate_value_counts,
        percentile_value_sums=percentile_value_sums,
        percentile_value_counts=percentile_value_counts,
        reference_pixel_counts=reference_pixel_counts,
        aoi_pixels=aoi_pixel_count,
        target_pixels=target_pixel_count,
        predicted_pixels=predicted_pixel_count,
        insufficient_predictor_pixels=insufficient_predictor_pixel_count,
        imputed_pixels=imputed_pixel_count,
    )


def run_reference_condition_inference(
    analysis_configuration: AnalysisConfiguration,
    model_run_directory: Path,
    output_directory: Path | None = None,
    show_progress: bool = True,
    analysis_cache_tiles: AnalysisCacheTiles | None = None,
) -> InferenceRunSummary:
    """Apply final response models to cached AOI tiles in bounded windows.

    Args:
        analysis_configuration: Authoritative analysis identity, mask, window,
            covariance, and raster-band settings.
        model_run_directory: Output directory from the response-model workflow.
        output_directory: Destination directory. ``None`` uses an ecoregion-
            specific directory under ``outputs/reference_condition_inference``.
        show_progress: Whether to display tqdm window progress.
        analysis_cache_tiles: Prevalidated cache resolution used by tests and
            callers that already resolved the configured cache. ``None`` loads
            and validates the analysis tiles from the cache manifest.

    Returns:
        Paths, counts, and elapsed time for the completed inference run.

    Raises:
        ValueError: If required raster bands are absent.
        RuntimeError: If a fitted model produces a nonfinite prediction.
    """

    started = time.perf_counter()
    window_size_pixels = analysis_configuration.inference.window_size_pixels
    covariance_shrinkage = analysis_configuration.inference.covariance_shrinkage
    resolved_model_run_directory = model_run_directory.expanduser().resolve()
    application_mask_path = analysis_configuration.inference.application_mask_path
    if analysis_cache_tiles is None:
        analysis_cache_tiles = resolve_analysis_cache_tiles(
            analysis_configuration,
            show_progress=show_progress,
        )
    output_grid = build_inference_output_grid(
        analysis_configuration,
        analysis_cache_tiles,
    )
    (
        run_metadata,
        response_models,
        maximum_predictor_missing_fraction,
    ) = load_response_models(
        resolved_model_run_directory,
    )
    reference_calibration = build_reference_departure_calibration(
        resolved_model_run_directory,
        response_models,
        covariance_shrinkage,
    )
    ecoregion_name = str(run_metadata["ecoregion_name"])
    analysis_slug = re.sub(
        r"[^a-z0-9]+",
        "_",
        analysis_configuration.analysis_name.lower(),
    ).strip("_")
    analysis_slug = analysis_slug or "analysis"
    resolved_output_directory = (
        output_directory.expanduser().resolve()
        if output_directory is not None
        else (
            Path("outputs")
            / "reference_condition_inference"
            / analysis_slug
        ).resolve()
    )
    resolved_output_directory.mkdir(parents=True, exist_ok=True)

    artifact_paths = InferenceArtifactPaths(
        expected_reference=(
            resolved_output_directory / f"{analysis_slug}_expected_reference.tif"
        ),
        observed_minus_expected=(
            resolved_output_directory
            / f"{analysis_slug}_observed_minus_expected.tif"
        ),
        standardized_deviation=(
            resolved_output_directory
            / f"{analysis_slug}_standardized_deviation.tif"
        ),
        departure_percentile=(
            resolved_output_directory
            / f"{analysis_slug}_reference_departure_percentile.tif"
        ),
        inference_status=(
            resolved_output_directory / f"{analysis_slug}_inference_status.tif"
        ),
        aggregate_deviation_figure=(
            resolved_output_directory
            / f"{analysis_slug}_aggregate_standardized_deviation.png"
        ),
        departure_percentile_figure=(
            resolved_output_directory
            / f"{analysis_slug}_reference_departure_percentile.png"
        ),
        report=(
            resolved_output_directory / f"{analysis_slug}_inference_report.md"
        ),
        metadata=(
            resolved_output_directory / f"{analysis_slug}_inference_metadata.json"
        ),
        checkpoint=(
            resolved_output_directory / f"{analysis_slug}_inference_checkpoint.json"
        ),
    )

    print("Reference-condition raster inference")
    print(f"Analysis: {analysis_configuration.analysis_name}")
    print(f"Raster cache: {analysis_cache_tiles.manifest_path}")
    print(f"Validated source tiles: {len(analysis_cache_tiles.tiles):,}")
    print(f"Model run: {resolved_model_run_directory}")
    print(f"Ecoregion: {ecoregion_name}")
    print(f"Responses: {len(response_models)}")
    print(
        "Reference calibration: "
        f"{reference_calibration.complete_reference_row_count:,} complete rows "
        f"of {reference_calibration.reference_row_count:,} reference rows"
    )
    print(
        "Reference calibration area: "
        f"{reference_calibration.complete_reference_area_m2 / 1_000_000:.2f} km^2 "
        f"of {reference_calibration.reference_area_m2 / 1_000_000:.2f} km^2 "
        "represented reference area"
    )
    print(
        "Covariance condition number: "
        f"{reference_calibration.covariance_condition_number:.3g} before, "
        f"{reference_calibration.stabilized_covariance_condition_number:.3g} "
        f"after {covariance_shrinkage:.1%} diagonal shrinkage"
    )
    print(f"Output directory: {resolved_output_directory}")
    print(
        "Stitched grid: "
        f"{output_grid.width:,} columns x {output_grid.height:,} rows, "
        f"{output_grid.pixel_size:g} m pixels, {output_grid.crs}"
    )
    print(
        "Concurrent tile workers: "
        f"{analysis_configuration.inference.worker_count}"
    )
    if application_mask_path is None:
        print(
            "Application mask: not supplied; inferring across the usable AOI "
            "predictor footprint"
        )
    else:
        print(f"Application mask: {application_mask_path}")
        print("Application mask target: defined first-band pixels equal to 1")

    inference_fingerprint = build_inference_fingerprint(
        analysis_configuration,
        analysis_cache_tiles,
        resolved_model_run_directory,
        response_models,
        reference_calibration.prediction_table_path,
    )
    (
        application_mask_metadata,
        resumed_tile_count,
        processed_tile_count,
    ) = write_inference_tiles(
        analysis_configuration,
        analysis_cache_tiles,
        output_grid,
        artifact_paths,
        response_models,
        reference_calibration,
        maximum_predictor_missing_fraction,
        inference_fingerprint,
        ecoregion_name,
        resolved_model_run_directory,
        show_progress,
    )
    print(
        "Inference tile results: "
        f"{processed_tile_count:,} processed, "
        f"{resumed_tile_count:,} resumed from checkpoint, 0 failed"
    )

    output_statistics = summarize_inference_outputs(
        analysis_configuration,
        analysis_cache_tiles,
        output_grid,
        artifact_paths,
        response_models,
        show_progress,
    )
    response_statistics = output_statistics.response_statistics
    departure_percentile_statistics = (
        output_statistics.departure_percentile_statistics
    )
    aggregate_value_sums = output_statistics.aggregate_value_sums
    aggregate_value_counts = output_statistics.aggregate_value_counts
    percentile_value_sums = output_statistics.percentile_value_sums
    percentile_value_counts = output_statistics.percentile_value_counts
    reference_pixel_counts = output_statistics.reference_pixel_counts
    raster_pixel_count = output_grid.width * output_grid.height
    aoi_pixel_count = output_statistics.aoi_pixels
    target_pixel_count = output_statistics.target_pixels
    predicted_pixel_count = output_statistics.predicted_pixels
    insufficient_predictor_pixel_count = (
        output_statistics.insufficient_predictor_pixels
    )
    imputed_pixel_count = output_statistics.imputed_pixels
    source_bounds = output_grid.bounds
    source_crs = output_grid.crs
    aggregate_figure_metadata = create_aggregate_deviation_figure(
        aggregate_value_sums,
        aggregate_value_counts,
        reference_pixel_counts,
        source_bounds,
        source_crs,
        len(response_models),
        analysis_configuration.display_name,
        application_mask_path is not None,
        artifact_paths.aggregate_deviation_figure,
    )
    departure_percentile_figure_metadata = create_departure_percentile_figure(
        percentile_value_sums,
        percentile_value_counts,
        reference_pixel_counts,
        source_bounds,
        source_crs,
        len(response_models),
        analysis_configuration.display_name,
        application_mask_path is not None,
        artifact_paths.departure_percentile_figure,
    )
    elapsed_seconds = time.perf_counter() - started
    departure_percentile_summary = departure_percentile_statistics.summarize()
    reference_distance_quantile_probabilities = np.array(
        [0.50, 0.90, 0.95, 0.99],
        dtype=np.float64,
    )
    reference_distance_quantile_offsets = np.searchsorted(
        reference_calibration.cumulative_reference_area_fractions,
        reference_distance_quantile_probabilities,
        side="left",
    )
    reference_distance_quantiles = {
        f"p{round(probability * 100):02d}": float(
            reference_calibration.sorted_reference_distances[offset]
        )
        for probability, offset in zip(
            reference_distance_quantile_probabilities,
            reference_distance_quantile_offsets,
            strict=True,
        )
    }
    response_summary_records = []
    for output_band_index, response_model in enumerate(response_models, start=1):
        response_summary_records.append(
            {
                "output_band_index": output_band_index,
                "response_band": response_model.response_band,
                "response_name": response_model.response_name,
                "display_name": response_model.display_name,
                "model_path": str(response_model.path),
                "reference_residual_rmse_oof": response_model.reference_rmse,
                "statistics": response_statistics[
                    response_model.response_band
                ].summarize(),
            }
        )
    artifact_path_records = {
        "expected_reference": str(artifact_paths.expected_reference),
        "observed_minus_expected": str(artifact_paths.observed_minus_expected),
        "standardized_deviation": str(artifact_paths.standardized_deviation),
        "reference_departure_percentile": str(
            artifact_paths.departure_percentile
        ),
        "inference_status": str(artifact_paths.inference_status),
        "aggregate_standardized_deviation_figure": str(
            artifact_paths.aggregate_deviation_figure
        ),
        "reference_departure_percentile_figure": str(
            artifact_paths.departure_percentile_figure
        ),
        "report": str(artifact_paths.report),
        "metadata": str(artifact_paths.metadata),
        "checkpoint": str(artifact_paths.checkpoint),
    }
    inference_metadata: dict[str, object] = {
        "artifact_type": "reference_condition_raster_inference",
        "format_version": 3,
        "ecoregion_name": ecoregion_name,
        "analysis_configuration": {
            "path": str(analysis_configuration.path),
            "analysis_name": analysis_configuration.analysis_name,
            "display_name": analysis_configuration.display_name,
            "aoi_path": str(analysis_configuration.aoi_path),
            "year": analysis_configuration.year,
            "sha256": analysis_configuration.configuration_sha256,
        },
        "input_raster_cache": {
            "manifest": str(analysis_cache_tiles.manifest_path),
            "stack_identifier": analysis_cache_tiles.stack_identifier,
            "source_tiles": len(analysis_cache_tiles.tiles),
            "source_bytes": analysis_cache_tiles.total_file_size_bytes,
        },
        "model_run_directory": str(resolved_model_run_directory),
        "tile_processing": {
            "checkpoint": str(artifact_paths.checkpoint),
            "inference_fingerprint": inference_fingerprint,
            "source_tiles": len(analysis_cache_tiles.tiles),
            "processed_tiles": processed_tile_count,
            "resumed_tiles": resumed_tile_count,
            "failed_tiles": 0,
        },
        "application_mask": application_mask_metadata,
        "mask_interpretation": (
            "defined first-band pixels equal to 1 after nearest-neighbor alignment"
            if application_mask_path
            else "unmasked usable AOI predictor footprint"
        ),
        "response_count": len(response_models),
        "configuration": {
            "maximum_predictor_missing_fraction": (
                maximum_predictor_missing_fraction
            ),
            "window_size_pixels": window_size_pixels,
            "worker_count": analysis_configuration.inference.worker_count,
            "covariance_shrinkage": covariance_shrinkage,
            "imputation": "final-reference-training values stored in each model",
        },
        "source_grid": {
            "width": output_grid.width,
            "height": output_grid.height,
            "crs": str(output_grid.crs),
            "transform": list(output_grid.transform),
            "bounds": list(output_grid.bounds),
            "pixel_size": output_grid.pixel_size,
        },
        "coverage": {
            "raster_pixels": raster_pixel_count,
            "aoi_pixels": aoi_pixel_count,
            "target_pixels": target_pixel_count,
            "predicted_pixels": predicted_pixel_count,
            "insufficient_predictor_pixels": (
                insufficient_predictor_pixel_count
            ),
            "imputed_pixels": imputed_pixel_count,
            "departure_percentile_pixels": departure_percentile_summary["pixels"],
        },
        "status_codes": {
            "0": "outside inference target",
            "1": "target pixel with excessive predictor missingness",
            "2": "reference-condition predictions written",
            "255": "nodata for imputed-predictor-count band",
        },
        "responses": response_summary_records,
        "reference_departure_calibration": {
            "prediction_table": str(reference_calibration.prediction_table_path),
            "response_bands": list(reference_calibration.response_bands),
            "reference_rows": reference_calibration.reference_row_count,
            "complete_reference_rows": (
                reference_calibration.complete_reference_row_count
            ),
            "reference_area_m2": reference_calibration.reference_area_m2,
            "complete_reference_area_m2": (
                reference_calibration.complete_reference_area_m2
            ),
            "complete_reference_area_percent": (
                100.0
                * reference_calibration.complete_reference_area_m2
                / reference_calibration.reference_area_m2
            ),
            "mean_vector": reference_calibration.reference_mean_vector.tolist(),
            "covariance_matrix": (
                reference_calibration.reference_covariance_matrix.tolist()
            ),
            "covariance_shrinkage": covariance_shrinkage,
            "stabilized_covariance_matrix": (
                reference_calibration.stabilized_reference_covariance_matrix.tolist()
            ),
            "covariance_condition_number": (
                reference_calibration.covariance_condition_number
            ),
            "stabilized_covariance_condition_number": (
                reference_calibration.stabilized_covariance_condition_number
            ),
            "reference_distance_quantiles": reference_distance_quantiles,
            "percentile_definition": (
                "represented-area fraction of complete reference rows with "
                "Mahalanobis distance less than or equal to the assessment "
                "pixel distance"
            ),
        },
        "reference_departure_percentile": {
            "statistics": departure_percentile_summary,
            "reference_pixels": "excluded and written as nodata",
            "required_responses": "every fitted response",
            "figure": departure_percentile_figure_metadata,
        },
        "aggregate_deviation_figure": aggregate_figure_metadata,
        "artifacts": artifact_path_records,
        "elapsed_seconds": elapsed_seconds,
    }
    artifact_paths.metadata.write_text(
        json.dumps(inference_metadata, indent=2),
        encoding="utf-8",
    )
    write_inference_report(artifact_paths.report, inference_metadata)

    print()
    print("Inference coverage")
    print(f"  Raster pixels: {raster_pixel_count:,}")
    print(f"  Pixels inside exact AOI: {aoi_pixel_count:,}")
    print(f"  Target pixels: {target_pixel_count:,}")
    print(f"  Predicted pixels: {predicted_pixel_count:,}")
    print(
        "  Insufficient-predictor pixels: "
        f"{insufficient_predictor_pixel_count:,}"
    )
    print(f"  Predicted pixels using imputation: {imputed_pixel_count:,}")
    print(
        "  Complete non-reference percentile pixels: "
        f"{departure_percentile_summary['pixels']:,}"
    )
    print()
    print("Response standardized deviations")
    for response_summary in response_summary_records:
        statistics = response_summary["statistics"]
        mean = statistics["standardized_mean"]
        mean_text = f"{mean:7.3f}" if mean is not None else "     NA"
        print(
            f"  {response_summary['response_band']} "
            f"{response_summary['display_name']:<32} "
            f"pixels={statistics['deviation_pixels']:>10,}  mean z={mean_text}"
        )
    print()
    print("Reference-condition departure percentiles")
    print(f"  Mean P_i: {departure_percentile_summary['mean']:.3f}")
    print(
        "  Pixels at or above P_i=0.95: "
        f"{departure_percentile_summary['at_or_above_95_percent']:.1f}%"
    )
    print()
    print(f"Inference report: {artifact_paths.report}")
    print(f"Metadata: {artifact_paths.metadata}")
    print(f"Aggregate deviation figure: {artifact_paths.aggregate_deviation_figure}")
    print(f"Departure percentile raster: {artifact_paths.departure_percentile}")
    print(f"Departure percentile figure: {artifact_paths.departure_percentile_figure}")
    print(f"Completed in {elapsed_seconds:.1f} seconds")

    return InferenceRunSummary(
        output_directory=resolved_output_directory,
        expected_reference_path=artifact_paths.expected_reference,
        observed_minus_expected_path=artifact_paths.observed_minus_expected,
        standardized_deviation_path=artifact_paths.standardized_deviation,
        departure_percentile_path=artifact_paths.departure_percentile,
        inference_status_path=artifact_paths.inference_status,
        aggregate_deviation_figure_path=(
            artifact_paths.aggregate_deviation_figure
        ),
        departure_percentile_figure_path=(
            artifact_paths.departure_percentile_figure
        ),
        report_path=artifact_paths.report,
        metadata_path=artifact_paths.metadata,
        response_count=len(response_models),
        raster_pixels=raster_pixel_count,
        aoi_pixels=aoi_pixel_count,
        target_pixels=target_pixel_count,
        predicted_pixels=predicted_pixel_count,
        departure_percentile_pixels=int(departure_percentile_summary["pixels"]),
        insufficient_predictor_pixels=insufficient_predictor_pixel_count,
        imputed_pixels=imputed_pixel_count,
        elapsed_seconds=elapsed_seconds,
    )


def main() -> None:
    """Run reference-condition raster inference from the command line.

    Returns:
        None: Outputs and reports are written by the inference workflow.
    """

    args = parse_args()
    analysis_configuration = load_analysis_configuration(
        args.analysis_configuration
    )
    run_reference_condition_inference(
        analysis_configuration,
        args.model_run_directory,
        output_directory=args.output_directory,
        show_progress=not args.no_progress,
    )


if __name__ == "__main__":
    main()
