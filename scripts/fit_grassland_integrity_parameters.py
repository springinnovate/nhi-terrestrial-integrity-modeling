"""Fit spatially validated reference-condition GAMs for ecological responses."""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder, SplineTransformer
from tqdm.auto import tqdm

if __package__:
    from .reference_condition_utils import (
        ENVIRONMENTAL_BAND_PATTERN,
        FIGURE_DPI,
        PREDICTOR_DISPLAY_NAMES,
        ReferenceConditionConfiguration,
        calculate_imputation_values,
        create_fold_map,
        infer_ecoregion_name,
        prepare_reference_condition_data,
        weighted_quantiles,
    )
else:
    from reference_condition_utils import (
        ENVIRONMENTAL_BAND_PATTERN,
        FIGURE_DPI,
        PREDICTOR_DISPLAY_NAMES,
        ReferenceConditionConfiguration,
        calculate_imputation_values,
        create_fold_map,
        infer_ecoregion_name,
        prepare_reference_condition_data,
        weighted_quantiles,
    )


DEFAULT_MINIMUM_RESPONSE_COVERAGE = 0.50
DEFAULT_RIDGE_ALPHA = 1.0
RESPONSE_BAND_PATTERN = re.compile(r"^y2018_d(0[2-9]|1[0-9])_")
RESPONSE_DISPLAY_NAMES = {
    2: "NDVI 95th percentile",
    3: "NDVI median",
    4: "Growing-season length 1",
    5: "Growing-season length 2",
    6: "Green-up timing 1",
    7: "Green-up timing 2",
    8: "Short vegetation height",
    9: "Tree cover",
    10: "Non-tree vegetation cover",
    11: "Bare ground",
    12: "Maximum leaf area index",
    13: "Leaf area index variability",
    14: "Mean FPAR",
    15: "FPAR variability",
    16: "Maximum FPAR variability",
    17: "Number of growing seasons",
    18: "Net primary productivity",
    19: "Gross primary productivity",
}
REGRESSION_METRIC_NAMES = (
    "weighted_r2",
    "weighted_rmse",
    "weighted_mae",
    "weighted_spearman",
    "weighted_bias",
)


@dataclass(frozen=True)
class IntegrityConfiguration(ReferenceConditionConfiguration):
    """Settings for ecological-response screening and GAM fitting.

    Attributes:
        minimum_response_coverage: Minimum represented reference-site area
            coverage needed to fit an ecological response.
        ridge_alpha: L2 regularization strength used by each response model.
    """

    minimum_response_coverage: float = DEFAULT_MINIMUM_RESPONSE_COVERAGE
    ridge_alpha: float = DEFAULT_RIDGE_ALPHA


@dataclass(frozen=True)
class IntegrityRunSummary:
    """Principal artifacts and counts from an ecological-response GAM run."""

    output_directory: Path
    predictions_path: Path
    response_coverage_path: Path
    fold_metrics_path: Path
    response_metrics_path: Path
    deviation_correlation_path: Path
    report_path: Path
    metadata_path: Path
    model_paths: tuple[Path, ...]
    figure_paths: tuple[Path, ...]
    sampled_rows: int
    usable_rows: int
    fitted_responses: int
    elapsed_seconds: float


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Fit one spatially validated additive reference-condition model per "
            "ecological response in an ecoregion sample."
        )
    )
    parser.add_argument("sample_parquet", type=Path, help="Spatial sample Parquet.")
    parser.add_argument(
        "--output-directory",
        type=Path,
        help=(
            "Output directory. Defaults to outputs/integrity_parameters/<sample stem>."
        ),
    )
    parser.add_argument(
        "--responses",
        nargs="+",
        help=(
            "Response bands to fit, such as d02 d11 d18, or full column names. "
            "Defaults to every 2018 response band d02-d19."
        ),
    )
    parser.add_argument(
        "--ecoregion-name",
        help="Figure label. Defaults to a name inferred from the sample filename.",
    )
    parser.add_argument(
        "--no-partial-response-figures",
        action="store_true",
        help="Skip the one-predictor-at-a-time figure for each fitted response.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bars.",
    )
    return parser.parse_args()


def _response_columns_by_band(columns: Sequence[str]) -> dict[int, str]:
    """Map every available 2018 ecological-response band to its column.

    Args:
        columns (Sequence[str]): Column names from the ecoregion sample table.

    Returns:
        dict[int, str]: Response column name keyed by its d02-d19 band number.

    Raises:
        ValueError: If a response band is duplicated or any d02-d19 band is
            absent.
    """

    response_columns: dict[int, str] = {}
    for column_name in columns:
        match = RESPONSE_BAND_PATTERN.match(column_name)
        if not match:
            continue
        band_number = int(match.group(1))
        if band_number in response_columns:
            raise ValueError(
                f"Multiple columns were found for response d{band_number:02d}."
            )
        response_columns[band_number] = column_name
    missing_bands = sorted(set(range(2, 20)).difference(response_columns))
    if missing_bands:
        raise ValueError(
            "The sample must contain every 2018 ecological-response band d02-d19; "
            f"missing bands: {missing_bands}."
        )
    return response_columns


def resolve_response_names(
    columns: Sequence[str],
    requested_responses: Sequence[str] | None,
) -> tuple[str, ...]:
    """Resolve dNN aliases or full names to ordered response columns.

    Args:
        columns (Sequence[str]): Column names from the ecoregion sample table.
        requested_responses (Sequence[str] | None): Requested dNN aliases or
            complete response column names. ``None`` selects every response.

    Returns:
        tuple[str, ...]: Unique response column names in requested order.

    Raises:
        ValueError: If any requested response is not a d02-d19 alias or known
            response column.
    """

    response_columns = _response_columns_by_band(columns)
    if not requested_responses:
        return tuple(response_columns[band] for band in sorted(response_columns))

    column_lookup = {name.lower(): name for name in response_columns.values()}
    resolved_names = []
    for requested_response in requested_responses:
        normalized = requested_response.strip().lower()
        if normalized in column_lookup:
            response_name = column_lookup[normalized]
        else:
            band_match = re.fullmatch(r"d?(0?[2-9]|1[0-9])", normalized)
            if not band_match:
                raise ValueError(
                    f"Unknown response '{requested_response}'. Use d02-d19 or a "
                    "full 2018 response column name."
                )
            response_name = response_columns[int(band_match.group(1))]
        if response_name not in resolved_names:
            resolved_names.append(response_name)
    return tuple(resolved_names)


def weighted_standard_deviation(
    values: np.ndarray,
    weights: np.ndarray,
) -> float:
    """Calculate a represented-area weighted population standard deviation.

    Args:
        values (numpy.ndarray): Numeric response observations.
        weights (numpy.ndarray): Represented-area weight for each observation.

    Returns:
        float: Weighted population standard deviation, or ``NaN`` when no
        finite observation has positive weight.
    """

    numeric_values = np.asarray(values, dtype=np.float64)
    numeric_weights = np.asarray(weights, dtype=np.float64)
    valid = (
        np.isfinite(numeric_values)
        & np.isfinite(numeric_weights)
        & (numeric_weights > 0)
    )
    if not np.any(valid):
        return float("nan")
    values_valid = numeric_values[valid]
    weights_valid = numeric_weights[valid]
    mean_value = float(np.average(values_valid, weights=weights_valid))
    return float(
        np.sqrt(np.average((values_valid - mean_value) ** 2, weights=weights_valid))
    )


def weighted_correlation(
    first_values: np.ndarray,
    second_values: np.ndarray,
    weights: np.ndarray,
) -> float:
    """Calculate a represented-area weighted Pearson correlation.

    Args:
        first_values (numpy.ndarray): First numeric variable.
        second_values (numpy.ndarray): Second numeric variable.
        weights (numpy.ndarray): Represented-area weight for each paired
            observation.

    Returns:
        float: Weighted Pearson correlation, or ``NaN`` when fewer than two
        valid pairs remain or either variable has no weighted variation.
    """

    first = np.asarray(first_values, dtype=np.float64)
    second = np.asarray(second_values, dtype=np.float64)
    numeric_weights = np.asarray(weights, dtype=np.float64)
    valid = (
        np.isfinite(first)
        & np.isfinite(second)
        & np.isfinite(numeric_weights)
        & (numeric_weights > 0)
    )
    if np.count_nonzero(valid) < 2:
        return float("nan")
    first = first[valid]
    second = second[valid]
    numeric_weights = numeric_weights[valid]
    first_centered = first - np.average(first, weights=numeric_weights)
    second_centered = second - np.average(second, weights=numeric_weights)
    covariance = np.average(
        first_centered * second_centered,
        weights=numeric_weights,
    )
    first_variance = np.average(first_centered**2, weights=numeric_weights)
    second_variance = np.average(second_centered**2, weights=numeric_weights)
    if first_variance <= 0 or second_variance <= 0:
        return float("nan")
    return float(covariance / np.sqrt(first_variance * second_variance))


def calculate_regression_metrics(
    observed_values: np.ndarray,
    expected_values: np.ndarray,
    area_weights: np.ndarray,
) -> dict[str, float]:
    """Calculate area-weighted held-out regression diagnostics."""

    observed = np.asarray(observed_values, dtype=np.float64)
    expected = np.asarray(expected_values, dtype=np.float64)
    weights = np.asarray(area_weights, dtype=np.float64)
    valid = (
        np.isfinite(observed)
        & np.isfinite(expected)
        & np.isfinite(weights)
        & (weights > 0)
    )
    if np.count_nonzero(valid) < 2:
        return {metric_name: float("nan") for metric_name in REGRESSION_METRIC_NAMES}
    observed = observed[valid]
    expected = expected[valid]
    weights = weights[valid]
    residuals = observed - expected
    weighted_mean = float(np.average(observed, weights=weights))
    mean_squared_error = float(np.average(residuals**2, weights=weights))
    total_variation = float(np.sum(weights * (observed - weighted_mean) ** 2))
    residual_variation = float(np.sum(weights * residuals**2))
    weighted_r2 = (
        1.0 - residual_variation / total_variation
        if total_variation > 0
        else float("nan")
    )
    observed_ranks = pd.Series(observed).rank(method="average").to_numpy()
    expected_ranks = pd.Series(expected).rank(method="average").to_numpy()
    return {
        "weighted_r2": float(weighted_r2),
        "weighted_rmse": float(np.sqrt(mean_squared_error)),
        "weighted_mae": float(np.average(np.abs(residuals), weights=weights)),
        "weighted_spearman": weighted_correlation(
            observed_ranks,
            expected_ranks,
            weights,
        ),
        "weighted_bias": float(np.average(residuals, weights=weights)),
    }


def summarize_response_coverage(
    prepared_table: pd.DataFrame,
    response_names: Sequence[str],
    selected_response_names: Sequence[str],
    configuration: IntegrityConfiguration,
) -> pd.DataFrame:
    """Determine which response bands can support every spatial fold."""

    selected_names = set(selected_response_names)
    usable_reference = prepared_table["usable_for_gam"] & prepared_table[
        "reference_site"
    ].eq(1)
    usable_reference_area = float(
        prepared_table.loc[usable_reference, "area_weight_m2"].sum()
    )
    minimum_training_rows = max(2, configuration.spline_knot_count)
    records = []
    for response_name in response_names:
        response_match = RESPONSE_BAND_PATTERN.match(response_name)
        if response_match is None:
            raise ValueError(f"Not a 2018 ecological-response column: {response_name}")
        band_number = int(response_match.group(1))
        finite_response = np.isfinite(
            pd.to_numeric(prepared_table[response_name], errors="coerce")
        )
        defined_reference = usable_reference & finite_response
        defined_area = float(
            prepared_table.loc[defined_reference, "area_weight_m2"].sum()
        )
        coverage = (
            defined_area / usable_reference_area if usable_reference_area > 0 else 0.0
        )
        response_values = prepared_table.loc[defined_reference, response_name].to_numpy(
            dtype=np.float64
        )
        response_weights = prepared_table.loc[
            defined_reference, "area_weight_m2"
        ].to_numpy(dtype=np.float64)
        response_sd = weighted_standard_deviation(response_values, response_weights)
        unique_values = int(pd.Series(response_values).nunique())
        fold_training_rows = []
        fold_validation_rows = []
        for spatial_fold in range(1, configuration.fold_count + 1):
            fold_training_rows.append(
                int(
                    np.count_nonzero(
                        defined_reference
                        & prepared_table["spatial_fold"].ne(spatial_fold)
                    )
                )
            )
            fold_validation_rows.append(
                int(
                    np.count_nonzero(
                        defined_reference
                        & prepared_table["spatial_fold"].eq(spatial_fold)
                    )
                )
            )

        if response_name not in selected_names:
            status = "not_selected"
        elif not np.any(defined_reference):
            status = "no_reference_values"
        elif coverage < configuration.minimum_response_coverage:
            status = "insufficient_reference_coverage"
        elif unique_values < 2 or not np.isfinite(response_sd) or response_sd <= 0:
            status = "no_reference_variation"
        elif (
            min(fold_training_rows) < minimum_training_rows
            or min(fold_validation_rows) < 2
        ):
            status = "insufficient_spatial_fold_support"
        else:
            status = "fit"

        records.append(
            {
                "response": response_name,
                "response_band": f"d{band_number:02d}",
                "display_name": RESPONSE_DISPLAY_NAMES[band_number],
                "selected": response_name in selected_names,
                "status": status,
                "defined_reference_rows": int(np.count_nonzero(defined_reference)),
                "usable_reference_rows": int(np.count_nonzero(usable_reference)),
                "defined_reference_area_m2": defined_area,
                "usable_reference_area_m2": usable_reference_area,
                "reference_area_coverage": coverage,
                "unique_reference_values": unique_values,
                "reference_weighted_sd": response_sd,
                "minimum_fold_training_rows": min(fold_training_rows),
                "minimum_fold_validation_rows": min(fold_validation_rows),
            }
        )
    return pd.DataFrame.from_records(records)


def fit_response_gam(
    training_table: pd.DataFrame,
    response_name: str,
    continuous_predictor_names: tuple[str, ...],
    categorical_predictor_name: str,
    imputation_values: dict[str, float],
    configuration: IntegrityConfiguration,
) -> dict[str, object]:
    """Fit one regularized additive reference-condition regression."""

    predictor_names = (*continuous_predictor_names, categorical_predictor_name)
    imputed_predictors = training_table.loc[:, predictor_names].fillna(
        imputation_values
    )
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "continuous_splines",
                SplineTransformer(
                    n_knots=configuration.spline_knot_count,
                    degree=3,
                    knots="quantile",
                    extrapolation="linear",
                    include_bias=False,
                ),
                list(continuous_predictor_names),
            ),
            (
                "landform_categories",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                    dtype=np.float64,
                ),
                [categorical_predictor_name],
            ),
        ],
        sparse_threshold=0.0,
    )
    design_matrix = preprocessor.fit_transform(imputed_predictors)
    represented_areas = training_table["area_weight_m2"].to_numpy(dtype=np.float64)
    fitting_weights = represented_areas * len(training_table) / represented_areas.sum()
    regressor = Ridge(alpha=configuration.ridge_alpha)
    regressor.fit(
        design_matrix,
        training_table[response_name].to_numpy(dtype=np.float64),
        sample_weight=fitting_weights,
    )
    response_match = RESPONSE_BAND_PATTERN.match(response_name)
    if response_match is None:
        raise ValueError(f"Not a 2018 ecological-response column: {response_name}")
    band_number = int(response_match.group(1))
    return {
        "artifact_type": "grassland_reference_condition_additive_model",
        "format_version": 1,
        "response": response_name,
        "response_band": f"d{band_number:02d}",
        "display_name": RESPONSE_DISPLAY_NAMES[band_number],
        "continuous_predictor_names": continuous_predictor_names,
        "categorical_predictor_name": categorical_predictor_name,
        "imputation_values": imputation_values,
        "preprocessor": preprocessor,
        "regressor": regressor,
    }


def predict_expected_response(
    fitted_model: dict[str, object],
    predictor_table: pd.DataFrame,
) -> np.ndarray:
    """Predict expected reference condition with a serialized model bundle."""

    continuous_names = tuple(fitted_model["continuous_predictor_names"])
    categorical_name = str(fitted_model["categorical_predictor_name"])
    predictor_names = (*continuous_names, categorical_name)
    imputed_table = predictor_table.loc[:, predictor_names].fillna(
        fitted_model["imputation_values"]
    )
    design_matrix = fitted_model["preprocessor"].transform(imputed_table)
    return np.asarray(fitted_model["regressor"].predict(design_matrix))


def _response_output_columns(response_band: str) -> tuple[str, str, str]:
    """Return stable expected-condition output names for one response band."""

    return (
        f"{response_band}_expected_reference_oof",
        f"{response_band}_observed_minus_expected_oof",
        f"{response_band}_standardized_deviation_oof",
    )


def create_model_performance_figure(
    response_metrics: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    ecoregion_name: str,
    output_path: Path,
) -> None:
    """Compare held-out fit and rank stability for candidate responses."""

    ordered = response_metrics.sort_values("overall_weighted_r2")
    response_bands = ordered["response_band"].tolist()
    labels = [
        f"{row.response_band}  {row.display_name}"
        for row in ordered.itertuples(index=False)
    ]
    y_positions = np.arange(len(ordered))
    panels = (
        ("weighted_r2", "overall_weighted_r2", "Area-weighted held-out R2"),
        (
            "weighted_spearman",
            "overall_weighted_spearman",
            "Area-weighted held-out rank correlation",
        ),
    )
    figure_height = max(6.0, 0.48 * len(ordered) + 2.5)
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 9}):
        figure, axes = plt.subplots(
            1,
            2,
            figsize=(15.0, figure_height),
            sharey=True,
            facecolor="white",
        )
        for axis, (fold_column, overall_column, title) in zip(
            axes, panels, strict=True
        ):
            for y_position, response_band in zip(
                y_positions, response_bands, strict=True
            ):
                fold_values = fold_metrics.loc[
                    fold_metrics["response_band"].eq(response_band), fold_column
                ].to_numpy(dtype=np.float64)
                axis.scatter(
                    fold_values,
                    np.full(len(fold_values), y_position),
                    color="#9AA2A6",
                    s=25,
                    alpha=0.85,
                    zorder=2,
                )
            axis.scatter(
                ordered[overall_column],
                y_positions,
                color="#176B73",
                marker="D",
                s=42,
                label="All held-out reference rows",
                zorder=3,
            )
            axis.axvline(0.0, color="#6F777B", linewidth=0.8, linestyle="--")
            axis.set_title(title, fontsize=11, weight="bold")
            axis.set_xlabel("Metric value")
            axis.grid(axis="x", color="#E1E4E5", linewidth=0.6)
            axis.spines[["top", "right", "left"]].set_visible(False)
        axes[0].set_yticks(y_positions, labels)
        axes[1].tick_params(axis="y", left=False)
        axes[1].legend(
            handles=[
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    linestyle="none",
                    markerfacecolor="#9AA2A6",
                    markeredgecolor="none",
                    label="Individual spatial fold",
                ),
                plt.Line2D(
                    [0],
                    [0],
                    marker="D",
                    linestyle="none",
                    markerfacecolor="#176B73",
                    markeredgecolor="none",
                    label="All held-out reference rows",
                ),
            ],
            loc="lower right",
            frameon=False,
        )
        figure.suptitle(
            f"Ecological-response model performance and spatial variability\n"
            f"{ecoregion_name}",
            fontsize=17,
            weight="bold",
            y=0.995,
            linespacing=1.25,
        )
        figure.text(
            0.5,
            0.006,
            "Diamonds summarize all out-of-fold reference predictions; gray dots "
            "show how performance changes among spatial folds",
            ha="center",
            color="#4B5459",
        )
        figure.tight_layout(rect=(0.02, 0.03, 1.0, 0.94))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
        plt.close(figure)


def create_observed_expected_figure(
    scored_table: pd.DataFrame,
    response_metrics: pd.DataFrame,
    ecoregion_name: str,
    output_path: Path,
) -> None:
    """Plot observed against out-of-fold expected reference responses."""

    column_count = 4
    row_count = math.ceil(len(response_metrics) / column_count)
    figure, axes = plt.subplots(
        row_count,
        column_count,
        figsize=(16.0, 3.7 * row_count),
        facecolor="white",
        squeeze=False,
    )
    random_generator = np.random.default_rng(42)
    for axis, metric_row in zip(
        axes.flat, response_metrics.itertuples(index=False), strict=False
    ):
        expected_column, _, _ = _response_output_columns(metric_row.response_band)
        reference_rows = scored_table["reference_site"].eq(1)
        valid = (
            reference_rows
            & np.isfinite(scored_table[metric_row.response])
            & np.isfinite(scored_table[expected_column])
        )
        plot_table = scored_table.loc[
            valid, [metric_row.response, expected_column, "area_weight_m2"]
        ]
        if len(plot_table) > 2_500:
            probabilities = plot_table["area_weight_m2"].to_numpy(dtype=np.float64)
            probabilities /= probabilities.sum()
            selected_offsets = random_generator.choice(
                len(plot_table),
                size=2_500,
                replace=False,
                p=probabilities,
            )
            plot_table = plot_table.iloc[selected_offsets]
        observed = plot_table[metric_row.response].to_numpy(dtype=np.float64)
        expected = plot_table[expected_column].to_numpy(dtype=np.float64)
        combined = np.concatenate([observed, expected])
        lower_limit, upper_limit = np.quantile(combined, [0.01, 0.99])
        if math.isclose(lower_limit, upper_limit):
            lower_limit -= 0.5
            upper_limit += 0.5
        axis.scatter(
            observed,
            expected,
            s=8,
            color="#276678",
            alpha=0.24,
            linewidths=0,
            rasterized=True,
        )
        axis.plot(
            [lower_limit, upper_limit],
            [lower_limit, upper_limit],
            color="#7B3F3F",
            linewidth=1.2,
            linestyle="--",
        )
        axis.set_xlim(lower_limit, upper_limit)
        axis.set_ylim(lower_limit, upper_limit)
        axis.set_title(
            f"{metric_row.response_band}  {metric_row.display_name}",
            fontsize=9,
            weight="bold",
        )
        axis.text(
            0.04,
            0.94,
            (
                f"R2 {metric_row.overall_weighted_r2:.2f}\n"
                f"rank {metric_row.overall_weighted_spearman:.2f}"
            ),
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78},
        )
        axis.grid(color="#E1E4E5", linewidth=0.5)
        axis.spines[["top", "right"]].set_visible(False)
    for axis in axes.flat[len(response_metrics) :]:
        axis.set_visible(False)
    figure.suptitle(
        f"Held-out reference observations versus expected condition\n{ecoregion_name}",
        fontsize=17,
        weight="bold",
        y=0.995,
        linespacing=1.25,
    )
    figure.text(0.5, 0.006, "Observed response", ha="center", fontsize=11)
    figure.text(
        0.006,
        0.5,
        "Expected reference response",
        va="center",
        rotation=90,
        fontsize=11,
    )
    figure.tight_layout(rect=(0.025, 0.025, 1.0, 0.94))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(figure)


def create_residual_summary_figure(
    scored_table: pd.DataFrame,
    response_metrics: pd.DataFrame,
    ecoregion_name: str,
    output_path: Path,
) -> None:
    """Compare area-weighted held-out reference residual distributions."""

    ordered = response_metrics.sort_values("overall_weighted_r2")
    y_positions = np.arange(len(ordered))
    labels = []
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 9}):
        figure, axis = plt.subplots(
            figsize=(11.0, max(6.0, 0.48 * len(ordered) + 2.3)),
            facecolor="white",
        )
        for y_position, row in zip(
            y_positions, ordered.itertuples(index=False), strict=True
        ):
            _, _, standardized_column = _response_output_columns(row.response_band)
            valid = scored_table["reference_site"].eq(1) & np.isfinite(
                scored_table[standardized_column]
            )
            quantiles = weighted_quantiles(
                scored_table.loc[valid, standardized_column].to_numpy(dtype=np.float64),
                scored_table.loc[valid, "area_weight_m2"].to_numpy(dtype=np.float64),
                [0.05, 0.25, 0.50, 0.75, 0.95],
            )
            axis.plot(
                [quantiles[0], quantiles[4]],
                [y_position, y_position],
                color="#8B9498",
                linewidth=1.2,
            )
            axis.plot(
                [quantiles[1], quantiles[3]],
                [y_position, y_position],
                color="#2A6F73",
                linewidth=7,
                solid_capstyle="butt",
            )
            axis.scatter(
                [quantiles[2]],
                [y_position],
                color="#161A1D",
                s=24,
                zorder=3,
            )
            labels.append(f"{row.response_band}  {row.display_name}")
        axis.axvline(0.0, color="#7B3F3F", linewidth=1.0, linestyle="--")
        axis.set_yticks(y_positions, labels)
        axis.set_xlabel("Observed minus expected, divided by held-out reference RMSE")
        axis.set_title(
            f"Held-out reference deviation distributions\n{ecoregion_name}",
            fontsize=16,
            weight="bold",
            pad=16,
            linespacing=1.25,
        )
        axis.text(
            0.0,
            1.01,
            "Line: 5th-95th percentile; bar: 25th-75th; dot: median",
            transform=axis.transAxes,
            ha="left",
            va="bottom",
            color="#4B5459",
        )
        axis.grid(axis="x", color="#E1E4E5", linewidth=0.6)
        axis.spines[["top", "right", "left"]].set_visible(False)
        figure.tight_layout()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
        plt.close(figure)


def calculate_deviation_correlation(
    scored_table: pd.DataFrame,
    response_metrics: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate pairwise area-weighted correlations of response deviations."""

    response_bands = response_metrics["response_band"].tolist()
    correlation = pd.DataFrame(
        np.eye(len(response_bands), dtype=np.float64),
        index=response_bands,
        columns=response_bands,
    )
    weights = scored_table["area_weight_m2"].to_numpy(dtype=np.float64)
    for first_offset, first_band in enumerate(response_bands):
        first_column = _response_output_columns(first_band)[2]
        for second_offset in range(first_offset + 1, len(response_bands)):
            second_band = response_bands[second_offset]
            second_column = _response_output_columns(second_band)[2]
            value = weighted_correlation(
                scored_table[first_column].to_numpy(dtype=np.float64),
                scored_table[second_column].to_numpy(dtype=np.float64),
                weights,
            )
            correlation.loc[first_band, second_band] = value
            correlation.loc[second_band, first_band] = value
    return correlation


def create_deviation_correlation_figure(
    deviation_correlation: pd.DataFrame,
    response_metrics: pd.DataFrame,
    ecoregion_name: str,
    output_path: Path,
) -> None:
    """Plot correlation among standardized observed-minus-expected responses."""

    response_names = response_metrics.set_index("response_band")["display_name"]
    labels = [f"{band} {response_names[band]}" for band in deviation_correlation.index]
    figure_size = max(9.0, 0.64 * len(labels) + 4.0)
    figure, axis = plt.subplots(
        figsize=(figure_size, figure_size),
        facecolor="white",
    )
    image = axis.imshow(
        deviation_correlation.to_numpy(dtype=np.float64),
        vmin=-1.0,
        vmax=1.0,
        cmap="RdBu_r",
    )
    axis.set_xticks(np.arange(len(labels)), labels, rotation=45, ha="right", fontsize=8)
    axis.set_yticks(np.arange(len(labels)), labels, fontsize=8)
    for row_offset in range(len(labels)):
        for column_offset in range(len(labels)):
            value = deviation_correlation.iloc[row_offset, column_offset]
            if np.isfinite(value):
                axis.text(
                    column_offset,
                    row_offset,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if abs(value) > 0.55 else "#202427",
                )
    axis.set_title(
        f"Assessment-row ecological deviation correlation\n{ecoregion_name}",
        fontsize=16,
        weight="bold",
        pad=18,
        linespacing=1.25,
    )
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04, label="Correlation")
    figure.text(
        0.5,
        0.008,
        "Area-weighted correlations use all sampled rows with both observed "
        "responses; high absolute values flag potentially redundant parameters",
        ha="center",
        color="#4B5459",
        fontsize=9,
    )
    figure.tight_layout(rect=(0.02, 0.035, 1.0, 0.97))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(figure)


def create_partial_response_figure(
    fitted_model: dict[str, object],
    reference_training_table: pd.DataFrame,
    ecoregion_name: str,
    output_path: Path,
) -> None:
    """Plot one-predictor-at-a-time expected-condition model responses."""

    continuous_names = tuple(fitted_model["continuous_predictor_names"])
    categorical_name = str(fitted_model["categorical_predictor_name"])
    predictor_names = (*continuous_names, categorical_name)
    baseline = {
        predictor_name: fitted_model["imputation_values"][predictor_name]
        for predictor_name in predictor_names
    }
    column_count = 4
    row_count = math.ceil(len(predictor_names) / column_count)
    figure, axes = plt.subplots(
        row_count,
        column_count,
        figsize=(16.0, 3.4 * row_count),
        facecolor="white",
        squeeze=False,
    )
    for axis, predictor_name in zip(axes.flat, predictor_names, strict=False):
        band_match = ENVIRONMENTAL_BAND_PATTERN.match(predictor_name)
        display_name = PREDICTOR_DISPLAY_NAMES[int(band_match.group(1))]
        if predictor_name == categorical_name:
            values = np.sort(reference_training_table[predictor_name].dropna().unique())
            response_table = pd.DataFrame(
                [baseline] * len(values), columns=predictor_names
            )
            response_table[predictor_name] = values
            expected = predict_expected_response(fitted_model, response_table)
            axis.bar(np.arange(len(values)), expected, color="#667F4A", width=0.75)
            axis.set_xticks(
                np.arange(len(values)),
                [f"{value:g}" for value in values],
                rotation=45,
                ha="right",
            )
        else:
            lower, upper = weighted_quantiles(
                reference_training_table[predictor_name].to_numpy(dtype=np.float64),
                reference_training_table["area_weight_m2"].to_numpy(dtype=np.float64),
                [0.05, 0.95],
            )
            if math.isclose(lower, upper):
                predictor_values = np.array([lower])
            else:
                predictor_values = np.linspace(lower, upper, 100)
            response_table = pd.DataFrame(
                [baseline] * len(predictor_values), columns=predictor_names
            )
            response_table[predictor_name] = predictor_values
            expected = predict_expected_response(fitted_model, response_table)
            axis.plot(predictor_values, expected, color="#2A6F73", linewidth=2)
            axis.axvline(
                baseline[predictor_name],
                color="#8C9498",
                linewidth=0.8,
                linestyle="--",
            )
        axis.set_title(display_name, fontsize=9, weight="bold")
        axis.grid(axis="y", color="#E1E4E5", linewidth=0.5)
        axis.spines[["top", "right"]].set_visible(False)
    for axis in axes.flat[len(predictor_names) :]:
        axis.set_visible(False)
    figure.suptitle(
        f"Final additive model partial responses: {fitted_model['display_name']}\n"
        f"{ecoregion_name}",
        fontsize=17,
        weight="bold",
        y=0.995,
        linespacing=1.25,
    )
    figure.text(
        0.5,
        0.004,
        "One environmental predictor varies across its reference-site 5th-95th "
        "percentile; all others stay at the fitted baseline",
        ha="center",
        color="#4B5459",
    )
    figure.text(
        0.004,
        0.5,
        f"Expected {fitted_model['display_name']}",
        va="center",
        rotation=90,
        fontsize=11,
    )
    figure.tight_layout(rect=(0.025, 0.025, 1.0, 0.94))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(figure)


def _format_report_number(value: float, digits: int = 3) -> str:
    """Format finite report values while keeping unavailable metrics explicit."""

    return f"{value:.{digits}f}" if np.isfinite(value) else "NA"


def write_model_selection_report(
    output_path: Path,
    ecoregion_name: str,
    response_coverage: pd.DataFrame,
    response_metrics: pd.DataFrame,
    retained_predictor_count: int,
    excluded_predictor_count: int,
) -> None:
    """Write a standalone Markdown guide to response-model diagnostics."""

    metrics_by_band = response_metrics.set_index("response_band")
    lines = [
        f"# Ecological-response GAM report: {ecoregion_name}",
        "",
        "## What these models estimate",
        "",
        (
            "Each fitted GAM learns the expected ecological response at supplied "
            "reference sites from 2018 environmental predictors d20-d39. The models "
            "are trained only on reference rows. Out-of-fold expectations are then "
            "generated for every assessment row whose environmental predictors are "
            "usable."
        ),
        "",
        (
            "An observed-minus-expected value is a signed ecological deviation, not "
            "an integrity score. Positive is not automatically better: for example, "
            "positive bare-ground deviation and positive vegetation-cover deviation "
            "usually require different ecological interpretations."
        ),
        "",
        "## Candidate response summary",
        "",
        (
            f"Environmental predictors retained: {retained_predictor_count}; "
            f"excluded for low coverage: {excluded_predictor_count}."
        ),
        "",
        (
            "| Band | Ecological response | Status | Reference coverage | "
            "Held-out R2 | Held-out rank correlation | RMSE | Fold R2 range |"
        ),
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in response_coverage.itertuples(index=False):
        if row.response_band in metrics_by_band.index:
            metric = metrics_by_band.loc[row.response_band]
            r2_value = _format_report_number(metric["overall_weighted_r2"])
            rank_value = _format_report_number(metric["overall_weighted_spearman"])
            rmse_value = _format_report_number(metric["overall_weighted_rmse"])
            fold_range = (
                f"{_format_report_number(metric['fold_weighted_r2_minimum'])} to "
                f"{_format_report_number(metric['fold_weighted_r2_maximum'])}"
            )
        else:
            r2_value = rank_value = rmse_value = fold_range = "NA"
        lines.append(
            f"| {row.response_band} | {row.display_name} | {row.status} | "
            f"{row.reference_area_coverage:.1%} | {r2_value} | {rank_value} | "
            f"{rmse_value} | {fold_range} |"
        )
    lines.extend(
        [
            "",
            "## How to choose parameters",
            "",
            (
                "- Held-out R2 asks how much spatially held-out reference variation "
                "the environmental GAM explains. Zero means it does no better than "
                "the held-out reference mean; negative values are worse than that "
                "baseline."
            ),
            (
                "- Held-out rank correlation asks whether higher observed responses "
                "also tend to receive higher expected values, without requiring a "
                "perfect one-to-one scale."
            ),
            (
                "- Fold ranges show transfer stability. A strong overall value with "
                "one weak spatial fold needs ecological review before selection."
            ),
            (
                "- The deviation-correlation figure flags parameters that may repeat "
                "the same information. Correlation alone is not a reason to remove a "
                "response, but highly redundant responses should not receive "
                "independent full weight without justification."
            ),
            (
                "- Partial-response figures show the fitted additive relationships, "
                "with one environmental predictor varied at a time. They are model "
                "diagnostics, not causal effects."
            ),
            "",
            "## Important scope limit",
            "",
            (
                "The sampled zero class is background, not a verified set of current "
                "grasslands. These outputs therefore do not prove that high-scoring "
                "non-reference pixels are intact grassland. Apply the chosen response "
                "deviations to a defensible current-grassland mask before constructing "
                "or interpreting a present-day grassland integrity model."
            ),
            "",
            (
                "HMI and HII are not model predictors in this workflow. They may have "
                "contributed to how the supplied reference sites were defined, but the "
                "response GAMs themselves use only the retained d20-d39 environmental "
                "bands."
            ),
            "",
            "## Figures",
            "",
            "- `figures/spatial_folds.png`",
            "- `figures/response_model_performance.png`",
            "- `figures/observed_vs_expected.png`",
            "- `figures/reference_deviation_distributions.png`",
            "- `figures/response_deviation_correlation.png`",
            "- `figures/partial_responses/` (unless disabled)",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def run_integrity_parameter_gams(
    sample_path: Path,
    output_directory: Path,
    configuration: IntegrityConfiguration,
    requested_responses: Sequence[str] | None = None,
    show_progress: bool = True,
    ecoregion_name: str | None = None,
    create_partial_figures: bool = True,
) -> IntegrityRunSummary:
    """Fit, spatially validate, report, and serialize response GAMs."""

    if not 0 <= configuration.minimum_response_coverage <= 1:
        raise ValueError("minimum_response_coverage must be between zero and one.")
    if configuration.ridge_alpha < 0:
        raise ValueError("ridge_alpha must be non-negative.")

    started = time.perf_counter()
    resolved_sample_path = sample_path.expanduser().resolve()
    resolved_output_directory = output_directory.expanduser().resolve()
    resolved_ecoregion_name = ecoregion_name or infer_ecoregion_name(
        resolved_sample_path
    )
    print("Grassland ecological-response GAM validation")
    print(f"Input sample: {resolved_sample_path}")
    print(f"Output directory: {resolved_output_directory}")
    print(f"Ecoregion: {resolved_ecoregion_name}")
    sample_table = pd.read_parquet(resolved_sample_path)
    print(
        f"Loaded {len(sample_table):,} sampled rows x {sample_table.shape[1]:,} columns"
    )

    all_response_columns = _response_columns_by_band(sample_table.columns)
    selected_response_names = resolve_response_names(
        sample_table.columns, requested_responses
    )
    prepared = prepare_reference_condition_data(sample_table, configuration)
    response_coverage = summarize_response_coverage(
        prepared.table,
        tuple(all_response_columns[band] for band in sorted(all_response_columns)),
        selected_response_names,
        configuration,
    )
    fitted_response_names = tuple(
        response_coverage.loc[response_coverage["status"].eq("fit"), "response"]
    )
    if not fitted_response_names:
        statuses = response_coverage.loc[
            response_coverage["selected"], ["response_band", "status"]
        ]
        raise ValueError(
            "None of the selected responses can be fit: "
            + ", ".join(
                f"{row.response_band}={row.status}"
                for row in statuses.itertuples(index=False)
            )
        )

    print()
    print("Ecological-response screening")
    for row in response_coverage.loc[response_coverage["selected"]].itertuples(
        index=False
    ):
        print(
            f"  {row.response_band} {row.display_name:<32} "
            f"reference coverage={row.reference_area_coverage:6.1%}  {row.status}"
        )
    print(
        f"Fitting {len(fitted_response_names)} of "
        f"{len(selected_response_names)} selected responses"
    )

    resolved_output_directory.mkdir(parents=True, exist_ok=True)
    model_directory = resolved_output_directory / "models"
    figure_directory = resolved_output_directory / "figures"
    partial_figure_directory = figure_directory / "partial_responses"
    model_directory.mkdir(parents=True, exist_ok=True)
    figure_directory.mkdir(parents=True, exist_ok=True)

    usable_mask = prepared.table["usable_for_gam"]
    reference_mask = prepared.table["reference_site"].eq(1)
    scored_table = prepared.table.copy()
    fold_metric_records = []
    response_metric_records = []
    final_models: dict[str, dict[str, object]] = {}
    final_training_tables: dict[str, pd.DataFrame] = {}
    model_paths = []

    fit_progress = tqdm(
        total=len(fitted_response_names) * configuration.fold_count,
        desc="Fitting response folds",
        unit="model",
        disable=not show_progress,
    )
    for response_name in fitted_response_names:
        response_match = RESPONSE_BAND_PATTERN.match(response_name)
        if response_match is None:
            raise ValueError(f"Not a 2018 ecological-response column: {response_name}")
        band_number = int(response_match.group(1))
        response_band = f"d{band_number:02d}"
        finite_response = np.isfinite(
            pd.to_numeric(prepared.table[response_name], errors="coerce")
        )
        oof_expected = np.full(len(prepared.table), np.nan, dtype=np.float64)
        for spatial_fold in range(1, configuration.fold_count + 1):
            training_rows = (
                usable_mask
                & reference_mask
                & finite_response
                & prepared.table["spatial_fold"].ne(spatial_fold)
            )
            validation_reference_rows = (
                usable_mask
                & reference_mask
                & finite_response
                & prepared.table["spatial_fold"].eq(spatial_fold)
            )
            assessment_rows = usable_mask & prepared.table["spatial_fold"].eq(
                spatial_fold
            )
            training_table = prepared.table.loc[training_rows]
            imputation_values = calculate_imputation_values(
                training_table,
                prepared.continuous_predictor_names,
                prepared.categorical_predictor_name,
            )
            fitted_model = fit_response_gam(
                training_table,
                response_name,
                prepared.continuous_predictor_names,
                prepared.categorical_predictor_name,
                imputation_values,
                configuration,
            )
            fold_expected = predict_expected_response(
                fitted_model, prepared.table.loc[assessment_rows]
            )
            oof_expected[assessment_rows.to_numpy()] = fold_expected
            validation_table = prepared.table.loc[validation_reference_rows]
            validation_expected = oof_expected[validation_reference_rows.to_numpy()]
            fold_metrics = calculate_regression_metrics(
                validation_table[response_name].to_numpy(dtype=np.float64),
                validation_expected,
                validation_table["area_weight_m2"].to_numpy(dtype=np.float64),
            )
            fold_metric_records.append(
                {
                    "response": response_name,
                    "response_band": response_band,
                    "display_name": RESPONSE_DISPLAY_NAMES[band_number],
                    "spatial_fold": spatial_fold,
                    "training_reference_rows": int(np.count_nonzero(training_rows)),
                    "validation_reference_rows": int(
                        np.count_nonzero(validation_reference_rows)
                    ),
                    "validation_reference_area_m2": float(
                        validation_table["area_weight_m2"].sum()
                    ),
                    **fold_metrics,
                }
            )
            fit_progress.update()

        if not np.isfinite(oof_expected[usable_mask.to_numpy()]).all():
            raise RuntimeError(
                f"Not every usable row received an out-of-fold {response_band} "
                "expectation."
            )
        expected_column, deviation_column, standardized_column = (
            _response_output_columns(response_band)
        )
        scored_table[expected_column] = oof_expected
        observed = pd.to_numeric(scored_table[response_name], errors="coerce").to_numpy(
            dtype=np.float64
        )
        deviations = observed - oof_expected
        scored_table[deviation_column] = deviations
        reference_validation = (
            usable_mask.to_numpy() & reference_mask.to_numpy() & np.isfinite(observed)
        )
        overall_metrics = calculate_regression_metrics(
            observed[reference_validation],
            oof_expected[reference_validation],
            scored_table.loc[reference_validation, "area_weight_m2"].to_numpy(
                dtype=np.float64
            ),
        )
        residual_scale = overall_metrics["weighted_rmse"]
        scored_table[standardized_column] = (
            deviations / residual_scale if residual_scale > 0 else np.nan
        )

        response_fold_metrics = pd.DataFrame.from_records(fold_metric_records)
        response_fold_metrics = response_fold_metrics.loc[
            response_fold_metrics["response_band"].eq(response_band)
        ]
        coverage_row = response_coverage.loc[
            response_coverage["response"].eq(response_name)
        ].iloc[0]
        metric_record: dict[str, object] = {
            "response": response_name,
            "response_band": response_band,
            "display_name": RESPONSE_DISPLAY_NAMES[band_number],
            "reference_area_coverage": float(coverage_row["reference_area_coverage"]),
            "validation_reference_rows": int(np.count_nonzero(reference_validation)),
            "validation_reference_area_m2": float(
                scored_table.loc[reference_validation, "area_weight_m2"].sum()
            ),
            "reference_residual_rmse_oof": residual_scale,
        }
        metric_record.update(
            {f"overall_{name}": value for name, value in overall_metrics.items()}
        )
        for metric_name in REGRESSION_METRIC_NAMES:
            fold_values = response_fold_metrics[metric_name]
            metric_record.update(
                {
                    f"fold_{metric_name}_mean": float(fold_values.mean()),
                    f"fold_{metric_name}_standard_deviation": float(
                        fold_values.std(ddof=1)
                    ),
                    f"fold_{metric_name}_minimum": float(fold_values.min()),
                    f"fold_{metric_name}_maximum": float(fold_values.max()),
                }
            )
        response_metric_records.append(metric_record)

        final_training_rows = usable_mask & reference_mask & finite_response
        final_training_table = prepared.table.loc[final_training_rows]
        final_imputation_values = calculate_imputation_values(
            final_training_table,
            prepared.continuous_predictor_names,
            prepared.categorical_predictor_name,
        )
        final_model = fit_response_gam(
            final_training_table,
            response_name,
            prepared.continuous_predictor_names,
            prepared.categorical_predictor_name,
            final_imputation_values,
            configuration,
        )
        final_model["reference_residual_rmse_oof"] = residual_scale
        final_model["standardized_deviation_interpretation"] = (
            "(observed - expected reference response) divided by the area-weighted "
            "RMSE of out-of-fold reference residuals"
        )
        model_path = model_directory / f"{response_band}_reference_condition_gam.joblib"
        joblib.dump(final_model, model_path, compress=3)
        model_paths.append(model_path)
        final_models[response_band] = final_model
        final_training_tables[response_band] = final_training_table
        fit_progress.set_postfix(
            response=response_band,
            r2=f"{overall_metrics['weighted_r2']:.3f}",
        )
    fit_progress.close()

    fold_metrics = pd.DataFrame.from_records(fold_metric_records)
    response_metrics = pd.DataFrame.from_records(response_metric_records).sort_values(
        "response_band"
    )
    deviation_correlation = calculate_deviation_correlation(
        scored_table, response_metrics
    )

    predictions_path = (
        resolved_output_directory / "ecological_response_predictions.parquet"
    )
    predictor_coverage_path = resolved_output_directory / "predictor_coverage.csv"
    response_coverage_path = resolved_output_directory / "response_coverage.csv"
    fold_metrics_path = resolved_output_directory / "fold_metrics.csv"
    response_metrics_path = resolved_output_directory / "response_metrics.csv"
    deviation_correlation_path = (
        resolved_output_directory / "response_deviation_correlation.csv"
    )
    report_path = resolved_output_directory / "model_selection_report.md"
    metadata_path = resolved_output_directory / "run_metadata.json"

    scored_table.to_parquet(predictions_path, compression="zstd", index=False)
    prepared.predictor_coverage.to_csv(predictor_coverage_path, index=False)
    response_coverage.to_csv(response_coverage_path, index=False)
    fold_metrics.to_csv(fold_metrics_path, index=False)
    response_metrics.to_csv(response_metrics_path, index=False)
    deviation_correlation.to_csv(
        deviation_correlation_path, index_label="response_band"
    )

    base_figure_paths = (
        figure_directory / "spatial_folds.png",
        figure_directory / "response_model_performance.png",
        figure_directory / "observed_vs_expected.png",
        figure_directory / "reference_deviation_distributions.png",
        figure_directory / "response_deviation_correlation.png",
    )
    create_fold_map(
        prepared.block_summary,
        prepared.table,
        configuration,
        resolved_ecoregion_name,
        base_figure_paths[0],
    )
    create_model_performance_figure(
        response_metrics,
        fold_metrics,
        resolved_ecoregion_name,
        base_figure_paths[1],
    )
    create_observed_expected_figure(
        scored_table,
        response_metrics,
        resolved_ecoregion_name,
        base_figure_paths[2],
    )
    create_residual_summary_figure(
        scored_table,
        response_metrics,
        resolved_ecoregion_name,
        base_figure_paths[3],
    )
    create_deviation_correlation_figure(
        deviation_correlation,
        response_metrics,
        resolved_ecoregion_name,
        base_figure_paths[4],
    )
    partial_figure_paths = []
    if create_partial_figures:
        partial_progress = tqdm(
            response_metrics["response_band"],
            desc="Drawing partial responses",
            unit="figure",
            disable=not show_progress,
        )
        for response_band in partial_progress:
            partial_path = partial_figure_directory / f"{response_band}.png"
            create_partial_response_figure(
                final_models[response_band],
                final_training_tables[response_band],
                resolved_ecoregion_name,
                partial_path,
            )
            partial_figure_paths.append(partial_path)
    figure_paths = (*base_figure_paths, *partial_figure_paths)

    write_model_selection_report(
        report_path,
        resolved_ecoregion_name,
        response_coverage,
        response_metrics,
        len(prepared.retained_predictor_names),
        len(prepared.excluded_predictor_names),
    )
    metadata = {
        "input_sample": str(resolved_sample_path),
        "ecoregion_name": resolved_ecoregion_name,
        "configuration": asdict(configuration),
        "model": {
            "family": "one regularized additive ridge regression per response",
            "training_population": "usable supplied reference-site rows only",
            "continuous_terms": "independent cubic spline bases",
            "categorical_terms": "one-hot landform indicators",
            "interactions": False,
            "predictor_bands": "2018 d20-d39 environmental bands",
            "human_impact_predictors": False,
            "output_interpretation": (
                "expected reference condition and signed observed-minus-expected "
                "deviation; neither is a present-day integrity score"
            ),
        },
        "sampled_rows": int(len(prepared.table)),
        "usable_rows": int(usable_mask.sum()),
        "reference_rows": int(reference_mask.sum()),
        "usable_reference_rows": int(np.count_nonzero(usable_mask & reference_mask)),
        "validation_block_count": int(len(prepared.block_summary)),
        "selected_responses": list(selected_response_names),
        "fitted_responses": list(fitted_response_names),
        "retained_predictors": list(prepared.retained_predictor_names),
        "excluded_predictors": list(prepared.excluded_predictor_names),
        "artifacts": {
            "predictions": str(predictions_path),
            "predictor_coverage": str(predictor_coverage_path),
            "response_coverage": str(response_coverage_path),
            "fold_metrics": str(fold_metrics_path),
            "response_metrics": str(response_metrics_path),
            "deviation_correlation": str(deviation_correlation_path),
            "report": str(report_path),
            "models": [str(path) for path in model_paths],
            "figures": [str(path) for path in figure_paths],
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print()
    print("Spatially held-out reference-condition performance")
    print(
        "R2 measures explained held-out reference variation; rank correlation "
        "measures ordering. Metrics are represented-area weighted."
    )
    for row in response_metrics.sort_values(
        "overall_weighted_r2", ascending=False
    ).itertuples(index=False):
        print(
            f"  {row.response_band} {row.display_name:<32} "
            f"R2={row.overall_weighted_r2:7.3f}  "
            f"rank={row.overall_weighted_spearman:6.3f}  "
            f"RMSE={row.overall_weighted_rmse:g}"
        )
    elapsed_seconds = time.perf_counter() - started
    print()
    print("Artifacts")
    print(f"Model-selection report: {report_path}")
    print(f"Response metrics: {response_metrics_path}")
    print(f"Out-of-fold expectations and deviations: {predictions_path}")
    print(f"Models: {model_directory}")
    print(f"Figures: {figure_directory}")
    print(f"Completed in {elapsed_seconds:.2f} seconds")

    return IntegrityRunSummary(
        output_directory=resolved_output_directory,
        predictions_path=predictions_path,
        response_coverage_path=response_coverage_path,
        fold_metrics_path=fold_metrics_path,
        response_metrics_path=response_metrics_path,
        deviation_correlation_path=deviation_correlation_path,
        report_path=report_path,
        metadata_path=metadata_path,
        model_paths=tuple(model_paths),
        figure_paths=tuple(figure_paths),
        sampled_rows=int(len(prepared.table)),
        usable_rows=int(usable_mask.sum()),
        fitted_responses=len(fitted_response_names),
        elapsed_seconds=elapsed_seconds,
    )


def main() -> None:
    """Run the command-line ecological-response GAM workflow."""

    args = parse_args()
    configuration = IntegrityConfiguration()
    output_directory = args.output_directory or (
        Path("outputs") / "integrity_parameters" / args.sample_parquet.stem
    )
    try:
        run_integrity_parameter_gams(
            args.sample_parquet,
            output_directory,
            configuration,
            requested_responses=args.responses,
            show_progress=not args.no_progress,
            ecoregion_name=args.ecoregion_name,
            create_partial_figures=not args.no_partial_response_figures,
        )
    except (OSError, ValueError) as error:
        raise SystemExit(f"Could not fit ecological-response GAMs: {error}") from error


if __name__ == "__main__":
    main()
