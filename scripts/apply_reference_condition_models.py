"""Apply fitted reference-condition response models to an ecoregion raster."""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path

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
from rasterio.windows import Window
from tqdm.auto import tqdm

if __package__:
    from .fit_grassland_integrity_parameters import predict_expected_response
    from .reference_condition_utils import FIGURE_DPI
else:
    from fit_grassland_integrity_parameters import predict_expected_response
    from reference_condition_utils import FIGURE_DPI


DEFAULT_WINDOW_SIZE_PIXELS = 256
DEFAULT_COVARIANCE_SHRINKAGE = 0.10
MAXIMUM_DISPLAY_DIMENSION = 700
DISPLAY_COLOR_MAXIMUM = 10.0
DISPLAY_YELLOW_GREEN_VALUE = 3.0
DISPLAY_COLOR_TICKS = (0.0, 1.0, 3.0, 5.0, 7.0, 10.0)
PERCENTILE_COLOR_TICKS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
REFERENCE_SITE_COLOR = "#1565C0"
FLOAT_NODATA = -9999.0
STATUS_NODATA = 255
STATUS_OUTSIDE_TARGET = 0
STATUS_INSUFFICIENT_PREDICTORS = 1
STATUS_PREDICTED = 2


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
        mean_vector: Area-weighted reference mean standardized-departure vector.
        covariance_matrix: Area-weighted covariance before stabilization.
        stabilized_covariance_matrix: Covariance after diagonal shrinkage.
        precision_matrix: Inverse of the stabilized covariance matrix.
        sorted_reference_distances: Ascending Mahalanobis distances for complete
            reference rows.
        cumulative_reference_area_fractions: Cumulative represented-area
            fractions corresponding to the sorted distances.
        covariance_shrinkage: Fraction of covariance shrunk toward its diagonal.
        reference_rows: Number of labeled reference rows before completeness
            filtering.
        complete_reference_rows: Number of reference rows defining the matrix.
        reference_area_m2: Total represented reference area before filtering.
        complete_reference_area_m2: Represented area defining the matrix.
        covariance_condition_number: Condition number before stabilization.
        stabilized_covariance_condition_number: Condition number after
            stabilization.
    """

    prediction_table_path: Path
    response_bands: tuple[str, ...]
    mean_vector: np.ndarray
    covariance_matrix: np.ndarray
    stabilized_covariance_matrix: np.ndarray
    precision_matrix: np.ndarray
    sorted_reference_distances: np.ndarray
    cumulative_reference_area_fractions: np.ndarray
    covariance_shrinkage: float
    reference_rows: int
    complete_reference_rows: int
    reference_area_m2: float
    complete_reference_area_m2: float
    covariance_condition_number: float
    stabilized_covariance_condition_number: float

    def calculate_distances(
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

        centered = (
            np.asarray(standardized_departures, dtype=np.float64) - self.mean_vector
        )
        squared_distances = np.einsum(
            "ij,jk,ik->i",
            centered,
            self.precision_matrix,
            centered,
        )
        return np.sqrt(np.maximum(squared_distances, 0.0))

    def calculate_percentiles(self, distances: np.ndarray) -> np.ndarray:
        """Evaluate distances against the area-weighted reference distribution.

        Args:
            distances: Mahalanobis distances to transform.

        Returns:
            Area-weighted empirical reference percentiles on the 0–1 scale.
        """

        values = np.asarray(distances, dtype=np.float64)
        insertion_offsets = np.searchsorted(
            self.sorted_reference_distances,
            values,
            side="right",
        )
        percentiles = np.zeros(values.shape, dtype=np.float64)
        has_reference_at_or_below = insertion_offsets > 0
        percentiles[has_reference_at_or_below] = (
            self.cumulative_reference_area_fractions[
                insertion_offsets[has_reference_at_or_below] - 1
            ]
        )
        return percentiles


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
    """Streaming summaries for non-reference departure percentiles."""

    pixels: int = 0
    value_sum: float = 0.0
    value_sum_of_squares: float = 0.0
    minimum: float = math.inf
    maximum: float = -math.inf
    at_or_above_90: int = 0
    at_or_above_95: int = 0
    at_or_above_99: int = 0

    def update(self, percentiles: np.ndarray) -> None:
        """Accumulate one raster window of finite percentile values.

        Args:
            percentiles: Reference-departure percentiles from one window.

        Returns:
            None: Statistics are accumulated on this object.
        """

        values = np.asarray(percentiles, dtype=np.float64)
        if len(values) == 0:
            return
        self.pixels += len(values)
        self.value_sum += float(values.sum())
        self.value_sum_of_squares += float(np.square(values).sum())
        self.minimum = min(self.minimum, float(values.min()))
        self.maximum = max(self.maximum, float(values.max()))
        self.at_or_above_90 += int(np.count_nonzero(values >= 0.90))
        self.at_or_above_95 += int(np.count_nonzero(values >= 0.95))
        self.at_or_above_99 += int(np.count_nonzero(values >= 0.99))

    def summarize(self) -> dict[str, float | int | None]:
        """Return JSON-ready coverage and distribution statistics.

        Returns:
            Pixel count, moments, range, and upper-percentile percentages.
        """

        if self.pixels == 0:
            return {
                "pixels": 0,
                "mean": None,
                "standard_deviation": None,
                "minimum": None,
                "maximum": None,
                "at_or_above_90_percent": None,
                "at_or_above_95_percent": None,
                "at_or_above_99_percent": None,
            }
        mean = self.value_sum / self.pixels
        variance = max(self.value_sum_of_squares / self.pixels - mean**2, 0.0)
        return {
            "pixels": self.pixels,
            "mean": mean,
            "standard_deviation": math.sqrt(variance),
            "minimum": self.minimum,
            "maximum": self.maximum,
            "at_or_above_90_percent": 100.0 * self.at_or_above_90 / self.pixels,
            "at_or_above_95_percent": 100.0 * self.at_or_above_95 / self.pixels,
            "at_or_above_99_percent": 100.0 * self.at_or_above_99 / self.pixels,
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
    target_pixels: int
    predicted_pixels: int
    departure_percentile_pixels: int
    insufficient_predictor_pixels: int
    imputed_pixels: int
    elapsed_seconds: float


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed raster, model, output, mask, window, and progress arguments.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Apply final reference-condition models to an aligned ecoregion "
            "raster stack without constructing an integrity score."
        )
    )
    parser.add_argument("raster_stack", type=Path, help="Multiband GeoTIFF to score.")
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
            "outputs/reference_condition_inference/<ecoregion>."
        ),
    )
    parser.add_argument(
        "--grassland-mask",
        type=Path,
        help=(
            "Optional exactly aligned raster whose defined nonzero first-band "
            "pixels identify the inference target."
        ),
    )
    parser.add_argument(
        "--window-size-pixels",
        type=int,
        default=DEFAULT_WINDOW_SIZE_PIXELS,
        help=f"Square processing-window size. Default: {DEFAULT_WINDOW_SIZE_PIXELS}.",
    )
    parser.add_argument(
        "--covariance-shrinkage",
        type=float,
        default=DEFAULT_COVARIANCE_SHRINKAGE,
        help=(
            "Fraction of the reference covariance shrunk toward its diagonal "
            f"before inversion. Default: {DEFAULT_COVARIANCE_SHRINKAGE:.2f}."
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
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    model_paths = sorted(
        (model_run_directory / "models").glob("*_reference_condition_gam.joblib")
    )
    if not model_paths:
        raise ValueError(
            f"No reference-condition models found under {model_run_directory}."
        )

    response_models = []
    for model_path in model_paths:
        bundle = joblib.load(model_path)
        continuous_predictors = tuple(bundle["continuous_predictor_names"])
        categorical_predictor = str(bundle["categorical_predictor_name"])
        reference_rmse = float(bundle["reference_residual_rmse_oof"])
        if not np.isfinite(reference_rmse) or reference_rmse <= 0:
            raise ValueError(
                f"{model_path.name} has invalid cross-validated reference RMSE "
                f"{reference_rmse}."
            )
        response_models.append(
            ResponseModel(
                path=model_path,
                response_name=str(bundle["response"]),
                response_band=str(bundle["response_band"]),
                display_name=str(bundle["display_name"]),
                predictor_names=(*continuous_predictors, categorical_predictor),
                reference_rmse=reference_rmse,
                bundle=bundle,
            )
        )

    response_models.sort(key=lambda model: model.response_band)
    predictor_names = response_models[0].predictor_names
    if any(model.predictor_names != predictor_names for model in response_models[1:]):
        raise ValueError(
            "Models in one inference run must use the same ordered predictor bands."
        )
    maximum_missing_fraction = float(
        metadata["configuration"]["maximum_row_missing_fraction"]
    )
    return metadata, tuple(response_models), maximum_missing_fraction


def load_reference_departure_calibration(
    model_run_directory: Path,
    response_models: tuple[ResponseModel, ...],
    covariance_shrinkage: float,
) -> ReferenceDepartureCalibration:
    """Fit the multivariate reference distribution from out-of-fold residuals.

    Args:
        model_run_directory: Output directory from the response-model workflow.
        response_models: Ordered fitted responses included in each vector.
        covariance_shrinkage: Fraction of covariance shrunk toward its diagonal
            before inversion. Must be strictly between zero and one.

    Returns:
        Complete reference calibration for distance and percentile inference.

    Raises:
        ValueError: If shrinkage is invalid, required prediction columns are
            missing, or too few complete reference vectors are available.
    """

    if not 0.0 < covariance_shrinkage < 1.0:
        raise ValueError("covariance_shrinkage must be between zero and one.")
    prediction_table_path = (
        model_run_directory / "ecological_response_predictions.parquet"
    )
    prediction_table = pd.read_parquet(prediction_table_path)
    response_bands = tuple(model.response_band for model in response_models)
    standardized_columns = [
        f"{response_band}_standardized_deviation_oof"
        for response_band in response_bands
    ]
    required_columns = ["reference_site", "area_weight_m2", *standardized_columns]
    missing_columns = [
        column for column in required_columns if column not in prediction_table.columns
    ]
    if missing_columns:
        raise ValueError(
            "Reference prediction table is missing columns: "
            + ", ".join(missing_columns)
        )

    reference_rows = prediction_table["reference_site"].eq(1)
    area_weights = pd.to_numeric(
        prediction_table["area_weight_m2"], errors="coerce"
    )
    reference_weights = area_weights.loc[reference_rows]
    reference_area_m2 = float(reference_weights.sum())
    standardized_departures = prediction_table[standardized_columns].apply(
        pd.to_numeric,
        errors="coerce",
    )
    complete_reference_rows = (
        reference_rows
        & standardized_departures.notna().all(axis=1)
        & np.isfinite(standardized_departures).all(axis=1)
        & np.isfinite(area_weights)
        & area_weights.gt(0)
    )
    complete_departures = standardized_departures.loc[
        complete_reference_rows
    ].to_numpy(dtype=np.float64)
    complete_weights = area_weights.loc[complete_reference_rows].to_numpy(
        dtype=np.float64
    )
    if len(complete_departures) <= len(response_models):
        raise ValueError(
            "Reference departure calibration requires more complete reference "
            "rows than fitted responses."
        )

    complete_reference_area_m2 = float(complete_weights.sum())
    mean_vector = np.average(
        complete_departures,
        axis=0,
        weights=complete_weights,
    )
    centered_departures = complete_departures - mean_vector
    covariance_matrix = (
        (centered_departures * complete_weights[:, np.newaxis]).T
        @ centered_departures
        / complete_reference_area_m2
    )
    covariance_diagonal = np.diag(covariance_matrix)
    if not np.isfinite(covariance_diagonal).all() or np.any(
        covariance_diagonal <= 0
    ):
        raise ValueError(
            "Every fitted response must have positive finite reference variance."
        )

    # Shrinking only off-diagonal covariance preserves each response's reference
    # variance while preventing near-duplicate responses from destabilizing inversion.
    stabilized_covariance_matrix = (
        (1.0 - covariance_shrinkage) * covariance_matrix
        + covariance_shrinkage * np.diag(covariance_diagonal)
    )
    precision_matrix = np.linalg.inv(stabilized_covariance_matrix)
    squared_reference_distances = np.einsum(
        "ij,jk,ik->i",
        centered_departures,
        precision_matrix,
        centered_departures,
    )
    reference_distances = np.sqrt(np.maximum(squared_reference_distances, 0.0))
    distance_order = np.argsort(reference_distances, kind="stable")
    sorted_reference_distances = reference_distances[distance_order]
    cumulative_reference_area_fractions = np.cumsum(
        complete_weights[distance_order]
    ) / complete_reference_area_m2
    cumulative_reference_area_fractions[-1] = 1.0

    return ReferenceDepartureCalibration(
        prediction_table_path=prediction_table_path,
        response_bands=response_bands,
        mean_vector=mean_vector,
        covariance_matrix=covariance_matrix,
        stabilized_covariance_matrix=stabilized_covariance_matrix,
        precision_matrix=precision_matrix,
        sorted_reference_distances=sorted_reference_distances,
        cumulative_reference_area_fractions=cumulative_reference_area_fractions,
        covariance_shrinkage=covariance_shrinkage,
        reference_rows=int(np.count_nonzero(reference_rows)),
        complete_reference_rows=len(complete_departures),
        reference_area_m2=reference_area_m2,
        complete_reference_area_m2=complete_reference_area_m2,
        covariance_condition_number=float(np.linalg.cond(covariance_matrix)),
        stabilized_covariance_condition_number=float(
            np.linalg.cond(stabilized_covariance_matrix)
        ),
    )


def write_inference_report(
    output_path: Path,
    metadata: dict[str, object],
) -> None:
    """Write a human-readable raster inference report.

    Args:
        output_path: Destination path for the Markdown report.
        metadata: JSON-ready inference metadata and response statistics.

    Returns:
        None: The completed report is written to ``output_path``.
    """

    coverage = metadata["coverage"]
    configuration = metadata["configuration"]
    aggregate_figure = metadata["aggregate_deviation_figure"]
    calibration = metadata["reference_departure_calibration"]
    departure_percentile = metadata["reference_departure_percentile"]
    percentile_statistics = departure_percentile["statistics"]
    color_scale_upper_value = aggregate_figure["color_scale_upper_value"]
    lines = [
        f"# Reference-condition raster inference: {metadata['ecoregion_name']}",
        "",
    ]
    if metadata["grassland_mask"] is None:
        lines.extend(
            [
                "> **Important:** No grassland mask was supplied. These outputs cover "
                "the usable ecoregion predictor footprint and must not be interpreted "
                "as grassland integrity maps.",
                "",
            ]
        )
    lines.extend(
        [
            "## Inputs",
            "",
            f"- Raster stack: `{metadata['input_raster']}`",
            f"- Model run: `{metadata['model_run_directory']}`",
            f"- Grassland mask: `{metadata['grassland_mask'] or 'not supplied'}`",
            f"- Responses: {metadata['response_count']}",
            (
                "- Maximum predictor missingness: "
                f"{configuration['maximum_predictor_missing_fraction']:.1%}"
            ),
            f"- Processing window: {configuration['window_size_pixels']} pixels",
            (
                "- Covariance diagonal shrinkage: "
                f"{configuration['covariance_shrinkage']:.1%}"
            ),
            "",
            "## Pixel coverage",
            "",
            f"- Raster pixels: {coverage['raster_pixels']:,}",
            f"- Target pixels: {coverage['target_pixels']:,}",
            f"- Predicted pixels: {coverage['predicted_pixels']:,}",
            (
                "- Insufficient-predictor pixels: "
                f"{coverage['insufficient_predictor_pixels']:,}"
            ),
            f"- Predicted pixels using imputation: {coverage['imputed_pixels']:,}",
            (
                "- Complete non-reference percentile pixels: "
                f"{coverage['departure_percentile_pixels']:,}"
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
                f"vectors from {calibration['complete_reference_rows']:,} of "
                f"{calibration['reference_rows']:,} reference rows, representing "
                f"{calibration['complete_reference_area_percent']:.1f}% of sampled "
                "reference area. The area-weighted reference mean and covariance "
                "are calculated from those vectors."
            ),
            "",
            (
                "The covariance matrix is stabilized with "
                f"{calibration['covariance_shrinkage']:.1%} diagonal shrinkage "
                "before inversion. Its condition number changes from "
                f"{calibration['covariance_condition_number']:.3g} to "
                f"{calibration['stabilized_covariance_condition_number']:.3g}."
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
                f"{aggregate_figure['cells_at_or_above_color_maximum_percent']:.1f}% "
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
    for response in metadata["responses"]:
        statistics = response["statistics"]
        if statistics["standardized_mean"] is None:
            mean = standard_deviation = minimum = maximum = above_two = "NA"
        else:
            mean = f"{statistics['standardized_mean']:.3f}"
            standard_deviation = (
                f"{statistics['standardized_standard_deviation']:.3f}"
            )
            minimum = f"{statistics['standardized_minimum']:.3f}"
            maximum = f"{statistics['standardized_maximum']:.3f}"
            above_two = (
                f"{statistics['absolute_standardized_above_two_percent']:.1f}%"
            )
        lines.append(
            f"| {response['response_band']} | {response['display_name']} | "
            f"{response['reference_residual_rmse_oof']:.6g} | "
            f"{statistics['expected_pixels']:,} | "
            f"{statistics['deviation_pixels']:,} | {mean} | "
            f"{standard_deviation} | {minimum} | {maximum} | {above_two} |"
        )
    lines.extend(
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
    for artifact_name, artifact_path in metadata["artifacts"].items():
        lines.append(f"- {artifact_name}: `{artifact_path}`")
    lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def create_aggregate_deviation_figure(
    value_sums: np.ndarray,
    value_counts: np.ndarray,
    reference_counts: np.ndarray,
    raster_bounds: BoundingBox,
    raster_crs: CRS | None,
    response_count: int,
    ecoregion_name: str,
    grassland_mask_supplied: bool,
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
        grassland_mask_supplied: Whether inference was limited by an external
            grassland mask.
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
            " No grassland mask was supplied, so the modeled surface includes "
            "the usable ecoregion predictor footprint."
            if not grassland_mask_supplied
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
    percentile_sums: np.ndarray,
    percentile_counts: np.ndarray,
    reference_counts: np.ndarray,
    raster_bounds: BoundingBox,
    raster_crs: CRS | None,
    response_count: int,
    ecoregion_name: str,
    grassland_mask_supplied: bool,
    output_path: Path,
) -> dict[str, object]:
    """Map coarsened departure percentiles and reference-site locations.

    Each colored display cell contains the mean ``P_i`` among complete-response,
    non-reference source pixels. Display cells containing one or more reference
    pixels are drawn in blue over the percentile surface.

    Args:
        percentile_sums: Sum of source-pixel percentiles per display cell.
        percentile_counts: Contributing non-reference pixels per display cell.
        reference_counts: Reference-site source pixels per display cell.
        raster_bounds: Spatial bounds of the source raster.
        raster_crs: Source raster coordinate reference system, when defined.
        response_count: Number of responses in each multivariate distance.
        ecoregion_name: Human-readable label included in the title.
        grassland_mask_supplied: Whether inference used an external grassland
            mask.
        output_path: Destination path for the publication-resolution PNG.

    Returns:
        JSON-ready display dimensions, counts, aggregation, and color limits.

    Raises:
        RuntimeError: If no complete-response non-reference pixels are available.
    """

    display_values = np.full(percentile_sums.shape, np.nan, dtype=np.float64)
    np.divide(
        percentile_sums,
        percentile_counts,
        out=display_values,
        where=percentile_counts > 0,
    )
    finite_values = display_values[np.isfinite(display_values)]
    if len(finite_values) == 0:
        raise RuntimeError(
            "No non-reference pixels have complete standardized-departure "
            "vectors; the departure-percentile figure cannot be created."
        )

    reference_display_mask = reference_counts > 0
    extent = (
        raster_bounds.left,
        raster_bounds.right,
        raster_bounds.bottom,
        raster_bounds.top,
    )
    color_map = colormaps["RdYlGn_r"].copy()
    color_map.set_bad("#ECEFF1")
    reference_color_map = ListedColormap([REFERENCE_SITE_COLOR])
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
                extent=extent,
                interpolation="nearest",
                vmin=0,
                vmax=1,
                zorder=3,
            )

        color_bar = figure.colorbar(
            image,
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
                "and blue cells contain reference sites"
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
                    edgecolor=REFERENCE_SITE_COLOR,
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
            " No grassland mask was supplied, so the modeled surface includes "
            "the usable ecoregion predictor footprint."
            if not grassland_mask_supplied
            else ""
        )
        figure.text(
            0.5,
            0.01,
            (
                f"Each display cell is the mean $P_i$ across complete non-reference "
                f"pixels using {response_count} responses. Blue display cells "
                "contain reference pixels, which are excluded from colored values. "
                "This measures departure from reference, not ecological degradation "
                f"by itself.{warning}"
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
        "display_width": int(display_values.shape[1]),
        "display_height": int(display_values.shape[0]),
        "colored_display_cells": int(len(finite_values)),
        "reference_display_cells": int(np.count_nonzero(reference_display_mask)),
        "contributing_source_pixels": int(percentile_counts.sum()),
        "reference_source_pixels": int(reference_counts.sum()),
        "response_count": response_count,
        "color_normalization": "linear over the fixed 0 to 1 range",
        "color_scale_lower_value": 0.0,
        "color_scale_upper_value": 1.0,
        "reference_color": REFERENCE_SITE_COLOR,
        "display_value_minimum": float(finite_values.min()),
        "display_value_median": float(np.median(finite_values)),
        "display_value_maximum": float(finite_values.max()),
    }


def run_reference_condition_inference(
    raster_stack_path: Path,
    model_run_directory: Path,
    output_directory: Path | None = None,
    grassland_mask_path: Path | None = None,
    window_size_pixels: int = DEFAULT_WINDOW_SIZE_PIXELS,
    covariance_shrinkage: float = DEFAULT_COVARIANCE_SHRINKAGE,
    show_progress: bool = True,
) -> InferenceRunSummary:
    """Apply final response models to aligned raster pixels in bounded windows.

    Args:
        raster_stack_path: Multiband ecoregion GeoTIFF containing every model
            predictor and observed response band.
        model_run_directory: Output directory from the response-model workflow.
        output_directory: Destination directory. ``None`` uses an ecoregion-
            specific directory under ``outputs/reference_condition_inference``.
        grassland_mask_path: Optional exactly aligned raster. Defined nonzero
            values in its first band identify target pixels.
        window_size_pixels: Width and height of each processing window.
        covariance_shrinkage: Fraction of covariance shrunk toward its diagonal
            before calculating multivariate reference distances.
        show_progress: Whether to display tqdm window progress.

    Returns:
        Paths, counts, and elapsed time for the completed inference run.

    Raises:
        ValueError: If the window size is invalid, required bands are absent,
            or a supplied mask is not exactly aligned with the raster stack.
        RuntimeError: If a fitted model produces a nonfinite prediction.
    """

    if window_size_pixels <= 0:
        raise ValueError("window_size_pixels must be positive.")
    started = time.perf_counter()
    resolved_raster_path = raster_stack_path.expanduser().resolve()
    resolved_model_run_directory = model_run_directory.expanduser().resolve()
    resolved_mask_path = (
        grassland_mask_path.expanduser().resolve()
        if grassland_mask_path is not None
        else None
    )
    run_metadata, response_models, maximum_missing_fraction = load_response_models(
        resolved_model_run_directory
    )
    reference_calibration = load_reference_departure_calibration(
        resolved_model_run_directory,
        response_models,
        covariance_shrinkage,
    )
    ecoregion_name = str(run_metadata["ecoregion_name"])
    ecoregion_slug = re.sub(r"[^a-z0-9]+", "_", ecoregion_name.lower()).strip("_")
    ecoregion_slug = ecoregion_slug or "ecoregion"
    resolved_output_directory = (
        output_directory.expanduser().resolve()
        if output_directory is not None
        else (
            Path("outputs")
            / "reference_condition_inference"
            / ecoregion_slug
        ).resolve()
    )
    resolved_output_directory.mkdir(parents=True, exist_ok=True)

    expected_reference_path = (
        resolved_output_directory / f"{ecoregion_slug}_expected_reference.tif"
    )
    observed_minus_expected_path = (
        resolved_output_directory
        / f"{ecoregion_slug}_observed_minus_expected.tif"
    )
    standardized_deviation_path = (
        resolved_output_directory
        / f"{ecoregion_slug}_standardized_deviation.tif"
    )
    departure_percentile_path = (
        resolved_output_directory
        / f"{ecoregion_slug}_reference_departure_percentile.tif"
    )
    inference_status_path = (
        resolved_output_directory / f"{ecoregion_slug}_inference_status.tif"
    )
    aggregate_deviation_figure_path = (
        resolved_output_directory
        / f"{ecoregion_slug}_aggregate_standardized_deviation.png"
    )
    departure_percentile_figure_path = (
        resolved_output_directory
        / f"{ecoregion_slug}_reference_departure_percentile.png"
    )
    report_path = resolved_output_directory / f"{ecoregion_slug}_inference_report.md"
    metadata_path = (
        resolved_output_directory / f"{ecoregion_slug}_inference_metadata.json"
    )

    predictor_names = response_models[0].predictor_names
    response_names = tuple(model.response_name for model in response_models)
    required_band_names = (*predictor_names, *response_names)
    response_statistics = {
        model.response_band: ResponseStatistics() for model in response_models
    }
    departure_percentile_statistics = DeparturePercentileStatistics()
    raster_pixels = 0
    target_pixels = 0
    predicted_pixels = 0
    insufficient_predictor_pixels = 0
    imputed_pixels = 0

    print("Reference-condition raster inference")
    print(f"Raster stack: {resolved_raster_path}")
    print(f"Model run: {resolved_model_run_directory}")
    print(f"Ecoregion: {ecoregion_name}")
    print(f"Responses: {len(response_models)}")
    print(
        "Reference calibration: "
        f"{reference_calibration.complete_reference_rows:,} complete rows "
        f"of {reference_calibration.reference_rows:,} reference rows"
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
    if resolved_mask_path is None:
        print(
            "Grassland mask: not supplied; inferring across the usable ecoregion "
            "predictor footprint"
        )
    else:
        print(f"Grassland mask: {resolved_mask_path}")

    with ExitStack() as stack:
        source = stack.enter_context(rasterio.open(resolved_raster_path))
        grassland_mask = (
            stack.enter_context(rasterio.open(resolved_mask_path))
            if resolved_mask_path is not None
            else None
        )
        if grassland_mask is not None and (
            grassland_mask.width != source.width
            or grassland_mask.height != source.height
            or grassland_mask.crs != source.crs
            or grassland_mask.transform != source.transform
        ):
            raise ValueError(
                "Grassland mask width, height, CRS, and transform must exactly "
                "match the raster stack."
            )

        source_band_indices = {}
        for band_index, description in enumerate(source.descriptions, start=1):
            if description is not None:
                source_band_indices[description] = band_index
        missing_band_names = [
            band_name
            for band_name in required_band_names
            if band_name not in source_band_indices
        ]
        if missing_band_names:
            raise ValueError(
                "Raster stack is missing model bands: " + ", ".join(missing_band_names)
            )
        reference_band_candidates = [
            (band_name, band_index)
            for band_name, band_index in source_band_indices.items()
            if band_name.lower() == "reference_sites"
            or band_name.lower().endswith("_grassland_reference_sites")
        ]
        if not reference_band_candidates:
            raise ValueError(
                "Raster stack must contain a reference_sites or "
                "*_grassland_reference_sites band for the aggregate diagnostic."
            )
        reference_band_name, reference_band_index = min(
            reference_band_candidates,
            key=lambda candidate: (
                not candidate[0].lower().startswith("y2018_"),
                candidate[1],
            ),
        )
        required_band_indices = [
            source_band_indices[band_name] for band_name in required_band_names
        ]
        required_band_indices.append(reference_band_index)
        reference_band_offset = len(required_band_names)

        display_scale = min(
            1.0,
            MAXIMUM_DISPLAY_DIMENSION / max(source.width, source.height),
        )
        display_width = max(1, round(source.width * display_scale))
        display_height = max(1, round(source.height * display_scale))
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
        source_bounds = source.bounds
        source_crs = source.crs
        print(f"Reference-site band: {reference_band_name}")
        print(
            "Aggregate map display grid: "
            f"{display_width:,} columns x {display_height:,} rows"
        )

        float_profile = source.profile.copy()
        float_profile.update(
            driver="GTiff",
            count=len(response_models),
            dtype="float32",
            nodata=FLOAT_NODATA,
            compress="deflate",
            predictor=3,
            tiled=True,
            blockxsize=256,
            blockysize=256,
            interleave="band",
            BIGTIFF="IF_SAFER",
        )
        status_profile = float_profile.copy()
        status_profile.update(
            count=2,
            dtype="uint8",
            nodata=STATUS_NODATA,
            predictor=2,
        )
        percentile_profile = float_profile.copy()
        percentile_profile.update(count=1)
        expected_destination = stack.enter_context(
            rasterio.open(expected_reference_path, "w", **float_profile)
        )
        deviation_destination = stack.enter_context(
            rasterio.open(observed_minus_expected_path, "w", **float_profile)
        )
        standardized_destination = stack.enter_context(
            rasterio.open(standardized_deviation_path, "w", **float_profile)
        )
        percentile_destination = stack.enter_context(
            rasterio.open(departure_percentile_path, "w", **percentile_profile)
        )
        status_destination = stack.enter_context(
            rasterio.open(inference_status_path, "w", **status_profile)
        )

        common_tags = {
            "ecoregion_name": ecoregion_name,
            "input_raster": str(resolved_raster_path),
            "model_run_directory": str(resolved_model_run_directory),
            "grassland_mask": str(resolved_mask_path or "not_supplied"),
        }
        expected_destination.update_tags(
            artifact_type="expected_reference_condition",
            **common_tags,
        )
        deviation_destination.update_tags(
            artifact_type="observed_minus_expected_reference_condition",
            **common_tags,
        )
        standardized_destination.update_tags(
            artifact_type="standardized_reference_condition_deviation",
            interpretation=(
                "observed minus expected divided by pooled out-of-fold reference RMSE"
            ),
            **common_tags,
        )
        percentile_destination.update_tags(
            artifact_type="reference_condition_departure_percentile",
            interpretation=(
                "area-weighted empirical percentile of covariance-aware distance "
                "among complete out-of-fold reference vectors"
            ),
            response_bands=",".join(reference_calibration.response_bands),
            covariance_shrinkage=str(covariance_shrinkage),
            value_minimum="0",
            value_maximum="1",
            reference_pixels="nodata",
            **common_tags,
        )
        percentile_destination.set_band_description(
            1,
            "reference_departure_percentile",
        )
        status_destination.update_tags(
            artifact_type="reference_condition_inference_status",
            status_0="outside inference target",
            status_1="target pixel with excessive predictor missingness",
            status_2="reference-condition predictions written",
            **common_tags,
        )
        status_destination.set_band_description(1, "inference_status")
        status_destination.set_band_description(2, "imputed_predictor_count")
        for output_band_index, model in enumerate(response_models, start=1):
            expected_destination.set_band_description(
                output_band_index,
                f"{model.response_band}_expected_reference",
            )
            deviation_destination.set_band_description(
                output_band_index,
                f"{model.response_band}_observed_minus_expected",
            )
            standardized_destination.set_band_description(
                output_band_index,
                f"{model.response_band}_standardized_deviation",
            )
            response_tags = {
                "response_band": model.response_band,
                "display_name": model.display_name,
                "source_response_band": model.response_name,
                "reference_residual_rmse_oof": str(model.reference_rmse),
                "model_path": str(model.path),
            }
            expected_destination.update_tags(output_band_index, **response_tags)
            deviation_destination.update_tags(output_band_index, **response_tags)
            standardized_destination.update_tags(output_band_index, **response_tags)

        window_rows = math.ceil(source.height / window_size_pixels)
        window_columns = math.ceil(source.width / window_size_pixels)
        window_iterator = (
            Window(
                column_offset,
                row_offset,
                min(window_size_pixels, source.width - column_offset),
                min(window_size_pixels, source.height - row_offset),
            )
            for row_offset in range(0, source.height, window_size_pixels)
            for column_offset in range(0, source.width, window_size_pixels)
        )
        for window in tqdm(
            window_iterator,
            total=window_rows * window_columns,
            desc="Applying response models",
            unit="window",
            disable=not show_progress,
        ):
            masked_values = source.read(
                required_band_indices,
                window=window,
                masked=True,
            )
            window_values = np.asarray(
                np.ma.getdata(masked_values),
                dtype=np.float64,
            )
            window_validity = ~np.ma.getmaskarray(masked_values)
            window_validity &= np.isfinite(window_values)
            window_values[~window_validity] = np.nan

            predictor_count = len(predictor_names)
            predictor_values = window_values[:predictor_count]
            predictor_validity = window_validity[:predictor_count]
            missing_predictor_counts = np.count_nonzero(
                ~predictor_validity,
                axis=0,
            ).astype(np.uint8)
            if grassland_mask is None:
                target = np.any(predictor_validity, axis=0)
            else:
                masked_target = grassland_mask.read(1, window=window, masked=True)
                target_values = np.asarray(np.ma.getdata(masked_target))
                target = (
                    ~np.ma.getmaskarray(masked_target)
                    & np.isfinite(target_values)
                    & (target_values != 0)
                )
            missing_fraction = missing_predictor_counts / predictor_count
            usable = target & (missing_fraction <= maximum_missing_fraction)
            reference_pixels = (
                window_validity[reference_band_offset]
                & (window_values[reference_band_offset] != 0)
            )
            complete_response_validity = usable.copy()
            for model_offset in range(len(response_models)):
                response_offset = predictor_count + model_offset
                complete_response_validity &= window_validity[response_offset]

            window_height = int(window.height)
            window_width = int(window.width)
            window_shape = (window_height, window_width)
            expected_output = np.full(
                (len(response_models), *window_shape),
                FLOAT_NODATA,
                dtype=np.float32,
            )
            deviation_output = np.full_like(expected_output, FLOAT_NODATA)
            standardized_output = np.full_like(expected_output, FLOAT_NODATA)
            percentile_output = np.full(window_shape, FLOAT_NODATA, dtype=np.float32)
            status_output = np.zeros(window_shape, dtype=np.uint8)
            status_output[target] = STATUS_INSUFFICIENT_PREDICTORS
            status_output[usable] = STATUS_PREDICTED
            imputation_output = np.full(window_shape, STATUS_NODATA, dtype=np.uint8)
            imputation_output[target] = missing_predictor_counts[target]

            raster_pixels += target.size
            target_count = int(np.count_nonzero(target))
            usable_count = int(np.count_nonzero(usable))
            target_pixels += target_count
            predicted_pixels += usable_count
            insufficient_predictor_pixels += target_count - usable_count
            imputed_pixels += int(
                np.count_nonzero(usable & (missing_predictor_counts > 0))
            )

            usable_flat = usable.ravel()
            if usable_count > 0:
                predictor_matrix = predictor_values.reshape(
                    predictor_count,
                    -1,
                ).T[usable_flat]
                predictor_table = pd.DataFrame(
                    predictor_matrix,
                    columns=predictor_names,
                )
                for model_offset, model in enumerate(response_models):
                    expected_values = predict_expected_response(
                        model.bundle,
                        predictor_table,
                    )
                    if not np.isfinite(expected_values).all():
                        raise RuntimeError(
                            f"{model.response_band} produced a nonfinite prediction."
                        )
                    expected_flat = expected_output[model_offset].ravel()
                    expected_flat[usable_flat] = expected_values.astype(np.float32)

                    response_offset = predictor_count + model_offset
                    observed_values = window_values[response_offset].ravel()
                    observed_validity = window_validity[response_offset].ravel()
                    deviation_validity = usable_flat & observed_validity
                    deviation_values = (
                        observed_values[deviation_validity]
                        - expected_flat[deviation_validity]
                    )
                    standardized_values = deviation_values / model.reference_rmse
                    deviation_output[model_offset].ravel()[deviation_validity] = (
                        deviation_values.astype(np.float32)
                    )
                    standardized_output[model_offset].ravel()[deviation_validity] = (
                        standardized_values.astype(np.float32)
                    )
                    response_statistics[model.response_band].update(
                        usable_count,
                        standardized_values,
                    )

            aggregate_validity = complete_response_validity & ~reference_pixels
            standardized_vectors = standardized_output[
                :, aggregate_validity
            ].T.astype(np.float64)
            departure_distances = reference_calibration.calculate_distances(
                standardized_vectors
            )
            departure_percentiles = reference_calibration.calculate_percentiles(
                departure_distances
            )
            percentile_output[aggregate_validity] = departure_percentiles.astype(
                np.float32
            )
            departure_percentile_statistics.update(departure_percentiles)
            aggregate_values = np.sum(
                np.abs(standardized_output.astype(np.float64)),
                axis=0,
            )
            source_rows = np.arange(
                int(window.row_off),
                int(window.row_off + window.height),
            )
            source_columns = np.arange(
                int(window.col_off),
                int(window.col_off + window.width),
            )
            display_rows = np.minimum(
                source_rows * display_height // source.height,
                display_height - 1,
            )
            display_columns = np.minimum(
                source_columns * display_width // source.width,
                display_width - 1,
            )
            display_cell_indices = (
                display_rows[:, np.newaxis] * display_width
                + display_columns[np.newaxis, :]
            )
            np.add.at(
                aggregate_value_sums.ravel(),
                display_cell_indices[aggregate_validity],
                aggregate_values[aggregate_validity],
            )
            np.add.at(
                aggregate_value_counts.ravel(),
                display_cell_indices[aggregate_validity],
                1,
            )
            np.add.at(
                percentile_value_sums.ravel(),
                display_cell_indices[aggregate_validity],
                departure_percentiles,
            )
            np.add.at(
                percentile_value_counts.ravel(),
                display_cell_indices[aggregate_validity],
                1,
            )
            np.add.at(
                reference_pixel_counts.ravel(),
                display_cell_indices[reference_pixels],
                1,
            )

            expected_destination.write(expected_output, window=window)
            deviation_destination.write(deviation_output, window=window)
            standardized_destination.write(standardized_output, window=window)
            percentile_destination.write(percentile_output, 1, window=window)
            status_destination.write(
                np.stack([status_output, imputation_output]),
                window=window,
            )

    aggregate_figure_metadata = create_aggregate_deviation_figure(
        aggregate_value_sums,
        aggregate_value_counts,
        reference_pixel_counts,
        source_bounds,
        source_crs,
        len(response_models),
        ecoregion_name,
        resolved_mask_path is not None,
        aggregate_deviation_figure_path,
    )
    departure_percentile_figure_metadata = create_departure_percentile_figure(
        percentile_value_sums,
        percentile_value_counts,
        reference_pixel_counts,
        source_bounds,
        source_crs,
        len(response_models),
        ecoregion_name,
        resolved_mask_path is not None,
        departure_percentile_figure_path,
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
    response_summaries = []
    for output_band_index, model in enumerate(response_models, start=1):
        response_summaries.append(
            {
                "output_band_index": output_band_index,
                "response_band": model.response_band,
                "response_name": model.response_name,
                "display_name": model.display_name,
                "model_path": str(model.path),
                "reference_residual_rmse_oof": model.reference_rmse,
                "statistics": response_statistics[model.response_band].summarize(),
            }
        )
    artifacts = {
        "expected_reference": str(expected_reference_path),
        "observed_minus_expected": str(observed_minus_expected_path),
        "standardized_deviation": str(standardized_deviation_path),
        "reference_departure_percentile": str(departure_percentile_path),
        "inference_status": str(inference_status_path),
        "aggregate_standardized_deviation_figure": str(
            aggregate_deviation_figure_path
        ),
        "reference_departure_percentile_figure": str(
            departure_percentile_figure_path
        ),
        "report": str(report_path),
        "metadata": str(metadata_path),
    }
    inference_metadata: dict[str, object] = {
        "artifact_type": "grassland_reference_condition_raster_inference",
        "format_version": 2,
        "ecoregion_name": ecoregion_name,
        "input_raster": str(resolved_raster_path),
        "model_run_directory": str(resolved_model_run_directory),
        "grassland_mask": str(resolved_mask_path) if resolved_mask_path else None,
        "mask_interpretation": (
            "defined nonzero first-band pixels"
            if resolved_mask_path
            else "unmasked usable ecoregion predictor footprint"
        ),
        "response_count": len(response_models),
        "configuration": {
            "maximum_predictor_missing_fraction": maximum_missing_fraction,
            "window_size_pixels": window_size_pixels,
            "covariance_shrinkage": covariance_shrinkage,
            "imputation": "final-reference-training values stored in each model",
        },
        "source_grid": {
            "width": source.width,
            "height": source.height,
            "crs": str(source.crs) if source.crs else None,
            "transform": list(source.transform),
        },
        "coverage": {
            "raster_pixels": raster_pixels,
            "target_pixels": target_pixels,
            "predicted_pixels": predicted_pixels,
            "insufficient_predictor_pixels": insufficient_predictor_pixels,
            "imputed_pixels": imputed_pixels,
            "departure_percentile_pixels": departure_percentile_summary["pixels"],
        },
        "status_codes": {
            "0": "outside inference target",
            "1": "target pixel with excessive predictor missingness",
            "2": "reference-condition predictions written",
            "255": "nodata for imputed-predictor-count band",
        },
        "responses": response_summaries,
        "reference_departure_calibration": {
            "prediction_table": str(reference_calibration.prediction_table_path),
            "response_bands": list(reference_calibration.response_bands),
            "reference_rows": reference_calibration.reference_rows,
            "complete_reference_rows": (
                reference_calibration.complete_reference_rows
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
            "mean_vector": reference_calibration.mean_vector.tolist(),
            "covariance_matrix": (
                reference_calibration.covariance_matrix.tolist()
            ),
            "covariance_shrinkage": covariance_shrinkage,
            "stabilized_covariance_matrix": (
                reference_calibration.stabilized_covariance_matrix.tolist()
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
        "artifacts": artifacts,
        "elapsed_seconds": elapsed_seconds,
    }
    metadata_path.write_text(
        json.dumps(inference_metadata, indent=2),
        encoding="utf-8",
    )
    write_inference_report(report_path, inference_metadata)

    print()
    print("Inference coverage")
    print(f"  Raster pixels: {raster_pixels:,}")
    print(f"  Target pixels: {target_pixels:,}")
    print(f"  Predicted pixels: {predicted_pixels:,}")
    print(
        "  Insufficient-predictor pixels: "
        f"{insufficient_predictor_pixels:,}"
    )
    print(f"  Predicted pixels using imputation: {imputed_pixels:,}")
    print(
        "  Complete non-reference percentile pixels: "
        f"{departure_percentile_summary['pixels']:,}"
    )
    print()
    print("Response standardized deviations")
    for response in response_summaries:
        statistics = response["statistics"]
        mean = statistics["standardized_mean"]
        mean_text = f"{mean:7.3f}" if mean is not None else "     NA"
        print(
            f"  {response['response_band']} {response['display_name']:<32} "
            f"pixels={statistics['deviation_pixels']:>10,}  mean z={mean_text}"
        )
    print()
    print("Reference-condition departure percentiles")
    if departure_percentile_summary["pixels"]:
        print(f"  Mean P_i: {departure_percentile_summary['mean']:.3f}")
        print(
            "  Pixels at or above P_i=0.95: "
            f"{departure_percentile_summary['at_or_above_95_percent']:.1f}%"
        )
    print()
    print(f"Inference report: {report_path}")
    print(f"Metadata: {metadata_path}")
    print(f"Aggregate deviation figure: {aggregate_deviation_figure_path}")
    print(f"Departure percentile raster: {departure_percentile_path}")
    print(f"Departure percentile figure: {departure_percentile_figure_path}")
    print(f"Completed in {elapsed_seconds:.1f} seconds")

    return InferenceRunSummary(
        output_directory=resolved_output_directory,
        expected_reference_path=expected_reference_path,
        observed_minus_expected_path=observed_minus_expected_path,
        standardized_deviation_path=standardized_deviation_path,
        departure_percentile_path=departure_percentile_path,
        inference_status_path=inference_status_path,
        aggregate_deviation_figure_path=aggregate_deviation_figure_path,
        departure_percentile_figure_path=departure_percentile_figure_path,
        report_path=report_path,
        metadata_path=metadata_path,
        response_count=len(response_models),
        raster_pixels=raster_pixels,
        target_pixels=target_pixels,
        predicted_pixels=predicted_pixels,
        departure_percentile_pixels=int(departure_percentile_summary["pixels"]),
        insufficient_predictor_pixels=insufficient_predictor_pixels,
        imputed_pixels=imputed_pixels,
        elapsed_seconds=elapsed_seconds,
    )


def main() -> None:
    """Run reference-condition raster inference from the command line.

    Returns:
        None: Outputs and reports are written by the inference workflow.
    """

    args = parse_args()
    run_reference_condition_inference(
        args.raster_stack,
        args.model_run_directory,
        output_directory=args.output_directory,
        grassland_mask_path=args.grassland_mask,
        window_size_pixels=args.window_size_pixels,
        covariance_shrinkage=args.covariance_shrinkage,
        show_progress=not args.no_progress,
    )


if __name__ == "__main__":
    main()
