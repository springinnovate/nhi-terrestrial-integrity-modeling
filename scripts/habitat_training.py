"""Prepare sampled rows and fit responses with spatial cross-validation."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

from . import habitat_condition


REGRESSION_METRIC_NAMES = (
    "weighted_r2",
    "weighted_rmse",
    "weighted_mae",
    "weighted_spearman",
    "weighted_bias",
)


@dataclass(frozen=True)
class ReferenceConditionConfiguration:
    """Settings shared by reference-condition preparation and model fitting.

    Attributes:
        fold_count: Number of spatial cross-validation folds.
        sampling_block_size_meters: Width of source sampling blocks.
        validation_block_size_meters: Width of grouped validation blocks.
        minimum_predictor_coverage: Minimum represented-area coverage needed
            to retain a predictor.
        maximum_row_missing_fraction: Largest retained-predictor missing
            fraction allowed for a modeled row.
        spline_knot_count: Number of knots for each continuous spline term.
    """

    fold_count: int
    sampling_block_size_meters: int
    validation_block_size_meters: int
    minimum_predictor_coverage: float
    maximum_row_missing_fraction: float
    spline_knot_count: int


@dataclass(frozen=True)
class IntegrityConfiguration(ReferenceConditionConfiguration):
    """Settings for ecological-response screening and GAM fitting.

    Attributes:
        minimum_response_coverage: Minimum represented reference-site area
            coverage needed to fit an ecological response.
        ridge_alpha: L2 regularization strength used by each response model.
    """

    minimum_response_coverage: float
    ridge_alpha: float


@dataclass(frozen=True)
class PreparedReferenceConditionData:
    """Sample table prepared for reference-condition model fitting.

    Attributes:
        table: Input rows with validation-block, fold, and missingness fields.
        block_summary: One row per grouped validation block.
        predictor_coverage: Coverage and retention diagnostics by predictor.
        retained_predictor_names: Predictor columns retained for modeling.
        excluded_predictor_names: Predictor columns removed for low coverage.
        continuous_predictor_names: Retained continuous predictor columns.
        categorical_predictor_name: Retained categorical predictor column.
    """

    table: pd.DataFrame
    block_summary: pd.DataFrame
    predictor_coverage: pd.DataFrame
    retained_predictor_names: tuple[str, ...]
    excluded_predictor_names: tuple[str, ...]
    continuous_predictor_names: tuple[str, ...]
    categorical_predictor_name: str


def weighted_quantiles(
    values: np.ndarray,
    weights: np.ndarray,
    quantiles: Sequence[float],
) -> np.ndarray:
    """Calculate quantiles of values representing unequal amounts of area.

    Args:
        values: One-dimensional numeric observations.
        weights: Positive represented-area weight for each observation.
        quantiles: Requested probabilities between zero and one.

    Returns:
        Weighted quantile values in requested order.
    """

    numeric_values = np.asarray(values, dtype=np.float64)
    numeric_weights = np.asarray(weights, dtype=np.float64)
    requested_quantiles = np.asarray(quantiles, dtype=np.float64)
    valid = (
        np.isfinite(numeric_values)
        & np.isfinite(numeric_weights)
        & (numeric_weights > 0)
    )
    sorted_offsets = np.argsort(numeric_values[valid], kind="stable")
    sorted_values = numeric_values[valid][sorted_offsets]
    sorted_weights = numeric_weights[valid][sorted_offsets]
    cumulative_probabilities = np.cumsum(sorted_weights) / np.sum(sorted_weights)
    # The empirical quantile is the first value whose cumulative represented
    # area reaches the requested share of the represented area.
    quantile_offsets = np.searchsorted(
        cumulative_probabilities,
        requested_quantiles,
        side="left",
    )
    quantile_offsets = np.clip(quantile_offsets, 0, len(sorted_values) - 1)
    return sorted_values[quantile_offsets]


def assign_spatial_folds(
    sample_table: pd.DataFrame,
    configuration: ReferenceConditionConfiguration,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Group sampling blocks and balance them across spatial folds.

    Whole validation blocks remain in one fold. Blocks containing reference
    sites are assigned first to balance represented reference area, after
    which other blocks are assigned to balance total represented area.

    Args:
        sample_table: Spatial sample with block indices, labels, and area.
        configuration: Spatial block sizes and requested fold count.

    Returns:
        Copy of the table with validation-block and fold fields, plus one row
        per block containing assignment diagnostics.
    """

    sampling_blocks_per_validation_side = (
        configuration.validation_block_size_meters
        // configuration.sampling_block_size_meters
    )
    assigned_table = sample_table.copy()
    assigned_table["validation_block_column"] = (
        assigned_table["sampling_block_column"].astype(np.int64)
        // sampling_blocks_per_validation_side
    )
    assigned_table["validation_block_row"] = (
        assigned_table["sampling_block_row"].astype(np.int64)
        // sampling_blocks_per_validation_side
    )

    block_pairs = (
        assigned_table[["validation_block_column", "validation_block_row"]]
        .drop_duplicates()
        .sort_values(["validation_block_column", "validation_block_row"])
        .reset_index(drop=True)
    )
    block_pairs["validation_block_id"] = np.arange(
        1,
        len(block_pairs) + 1,
        dtype=np.int64,
    )
    assigned_table = assigned_table.merge(
        block_pairs,
        on=["validation_block_column", "validation_block_row"],
        how="left",
        validate="many_to_one",
    )
    assigned_table["reference_area_m2"] = (
        assigned_table["area_weight_m2"] * assigned_table["reference_site"]
    )
    block_summary = assigned_table.groupby(
        [
            "validation_block_id",
            "validation_block_column",
            "validation_block_row",
        ],
        as_index=False,
        sort=True,
    ).agg(
        sampled_rows=("reference_site", "size"),
        sampled_reference_rows=("reference_site", "sum"),
        represented_area_m2=("area_weight_m2", "sum"),
        represented_reference_area_m2=("reference_area_m2", "sum"),
    )
    reference_blocks = block_summary[
        block_summary["represented_reference_area_m2"] > 0
    ].sort_values(
        ["represented_reference_area_m2", "represented_area_m2"],
        ascending=False,
    )
    fold_reference_areas = np.zeros(configuration.fold_count, dtype=np.float64)
    fold_total_areas = np.zeros(configuration.fold_count, dtype=np.float64)
    fold_block_counts = np.zeros(configuration.fold_count, dtype=np.int64)
    fold_by_block_id: dict[int, int] = {}
    for block in reference_blocks.itertuples(index=False):
        fold_offset = min(
            range(configuration.fold_count),
            key=lambda offset: (
                fold_reference_areas[offset],
                fold_total_areas[offset],
                fold_block_counts[offset],
                offset,
            ),
        )
        fold_by_block_id[int(block.validation_block_id)] = fold_offset + 1
        fold_reference_areas[fold_offset] += float(block.represented_reference_area_m2)
        fold_total_areas[fold_offset] += float(block.represented_area_m2)
        fold_block_counts[fold_offset] += 1

    nonreference_blocks = block_summary[
        block_summary["represented_reference_area_m2"] == 0
    ].sort_values("represented_area_m2", ascending=False)
    for block in nonreference_blocks.itertuples(index=False):
        fold_offset = min(
            range(configuration.fold_count),
            key=lambda offset: (
                fold_total_areas[offset],
                fold_block_counts[offset],
                offset,
            ),
        )
        fold_by_block_id[int(block.validation_block_id)] = fold_offset + 1
        fold_total_areas[fold_offset] += float(block.represented_area_m2)
        fold_block_counts[fold_offset] += 1

    block_summary["spatial_fold"] = block_summary["validation_block_id"].map(
        fold_by_block_id
    )
    assigned_table["spatial_fold"] = assigned_table["validation_block_id"].map(
        fold_by_block_id
    )
    assigned_table.drop(columns="reference_area_m2", inplace=True)
    assigned_table["spatial_fold"] = assigned_table["spatial_fold"].astype(np.int16)
    block_summary["spatial_fold"] = block_summary["spatial_fold"].astype(np.int16)
    return assigned_table, block_summary


def prepare_reference_condition_data(
    sample_table: pd.DataFrame,
    configuration: ReferenceConditionConfiguration,
    predictor_names: tuple[str, ...],
    categorical_predictor_name: str,
) -> PreparedReferenceConditionData:
    """Select environmental predictors and prepare spatial validation rows.

    Predictor retention is based on represented-area coverage across the
    area sample. Row usability is then based on the missing fraction among
    retained predictors. Reference-site labels are preserved without deriving
    or changing classes.

    Args:
        sample_table: Sampled rows with predictor values, reference labels,
            sampling-block indices and represented area in ``area_weight_m2``.
        configuration: Coverage, missingness, and spatial-fold settings.
        predictor_names: Ordered predictor columns in the sample table.
        categorical_predictor_name: Predictor column treated as categorical.

    Returns:
        Prepared rows, folds, coverage diagnostics, and predictor names.

    Raises:
        ValueError: If the categorical predictor does not meet the configured
            predictor coverage threshold.
    """

    total_represented_area = float(sample_table["area_weight_m2"].sum())
    coverage_records = []
    for predictor_name in predictor_names:
        defined_row_mask = sample_table[predictor_name].notna()
        defined_area_m2 = float(
            sample_table.loc[defined_row_mask, "area_weight_m2"].sum()
        )
        represented_area_coverage = defined_area_m2 / total_represented_area
        coverage_records.append(
            {
                "predictor": predictor_name,
                "predictor_type": (
                    "categorical"
                    if predictor_name == categorical_predictor_name
                    else "continuous"
                ),
                "defined_rows": int(defined_row_mask.sum()),
                "row_coverage": float(defined_row_mask.mean()),
                "defined_area_m2": defined_area_m2,
                "area_coverage": represented_area_coverage,
                "retained": (
                    represented_area_coverage
                    >= configuration.minimum_predictor_coverage
                ),
            }
        )
    predictor_coverage = pd.DataFrame.from_records(coverage_records)
    retained_predictor_names = tuple(
        predictor_coverage.loc[predictor_coverage["retained"], "predictor"]
    )
    excluded_predictor_names = tuple(
        predictor_coverage.loc[~predictor_coverage["retained"], "predictor"]
    )
    if categorical_predictor_name not in retained_predictor_names:
        raise ValueError(
            "The categorical predictor did not meet minimum coverage; it is "
            "required by the model contract."
        )

    prepared_table, block_summary = assign_spatial_folds(
        sample_table,
        configuration,
    )
    prepared_table["imputed_predictor_count"] = (
        prepared_table[list(retained_predictor_names)]
        .isna()
        .sum(axis=1)
        .astype(np.int16)
    )
    prepared_table["missing_predictor_fraction"] = prepared_table[
        "imputed_predictor_count"
    ] / len(retained_predictor_names)
    prepared_table["usable_for_gam"] = (
        prepared_table["missing_predictor_fraction"]
        <= configuration.maximum_row_missing_fraction
    )
    continuous_predictor_names = tuple(
        name for name in retained_predictor_names if name != categorical_predictor_name
    )
    return PreparedReferenceConditionData(
        table=prepared_table,
        block_summary=block_summary,
        predictor_coverage=predictor_coverage,
        retained_predictor_names=retained_predictor_names,
        excluded_predictor_names=excluded_predictor_names,
        continuous_predictor_names=continuous_predictor_names,
        categorical_predictor_name=categorical_predictor_name,
    )


def calculate_imputation_values(
    training_table: pd.DataFrame,
    continuous_predictor_names: Sequence[str],
    categorical_predictor_name: str,
) -> dict[str, float]:
    """Learn area-weighted replacements from training rows only.

    Continuous predictors use represented-area weighted medians. The categorical
    predictor uses the category representing the largest total training area.

    Args:
        training_table: Rows available to a fold's training model.
        continuous_predictor_names: Continuous columns needing medians.
        categorical_predictor_name: Categorical column needing a mode.

    Returns:
        Replacement value keyed by predictor column.
    """

    area_weights = training_table["area_weight_m2"].to_numpy(dtype=np.float64)
    imputation_values: dict[str, float] = {}
    for predictor_name in continuous_predictor_names:
        predictor_values = training_table[predictor_name].to_numpy(dtype=np.float64)
        imputation_values[predictor_name] = float(
            weighted_quantiles(predictor_values, area_weights, [0.5])[0]
        )

    defined_categories = training_table[categorical_predictor_name].notna()
    category_areas = (
        training_table.loc[defined_categories]
        .groupby(categorical_predictor_name, sort=True)["area_weight_m2"]
        .sum()
    )
    imputation_values[categorical_predictor_name] = float(category_areas.idxmax())
    return imputation_values


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
    """Calculate area-weighted held-out regression diagnostics.

    Args:
        observed_values (numpy.ndarray): Observed ecological response values.
        expected_values (numpy.ndarray): Model-predicted reference values.
        area_weights (numpy.ndarray): Represented-area weight for each pair.

    Returns:
        dict[str, float]: Weighted R2, RMSE, MAE, Spearman correlation, and
        signed bias. Every metric is ``NaN`` when fewer than two valid pairs
        have positive weight.
    """

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
    # R2 is undefined for a constant observed response. Scikit-learn otherwise
    # replaces that undefined value with a finite convenience score.
    if np.ptp(observed) > 0:
        weighted_r2 = float(
            r2_score(
                observed,
                expected,
                sample_weight=weights,
                force_finite=False,
            )
        )
    else:
        weighted_r2 = float("nan")
    weighted_rmse = root_mean_squared_error(
        observed,
        expected,
        sample_weight=weights,
    )
    weighted_mae = mean_absolute_error(
        observed,
        expected,
        sample_weight=weights,
    )
    observed_ranks = pd.Series(observed).rank(method="average").to_numpy()
    expected_ranks = pd.Series(expected).rank(method="average").to_numpy()
    return {
        "weighted_r2": weighted_r2,
        "weighted_rmse": float(weighted_rmse),
        "weighted_mae": float(weighted_mae),
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
    """Determine which response bands can support every spatial fold.

    Args:
        prepared_table (pandas.DataFrame): Spatially folded sample with
            reference labels, area weights, and response columns.
        response_names (Sequence[str]): Configured response columns to assess.
        selected_response_names (Sequence[str]): Response columns requested for
            model fitting.
        configuration (IntegrityConfiguration): Coverage, spline, and fold
            requirements used to screen responses.

    Returns:
        pandas.DataFrame: One screening record per response, including coverage,
        variation, fold support, selection state, and fit status.
    """

    selected_response_name_set = set(selected_response_names)
    usable_reference_row_mask = prepared_table["usable_for_gam"] & prepared_table[
        "reference_site"
    ].eq(1)
    usable_reference_area_m2 = float(
        prepared_table.loc[usable_reference_row_mask, "area_weight_m2"].sum()
    )
    minimum_training_rows = max(2, configuration.spline_knot_count)
    coverage_records = []
    for response_name in response_names:
        finite_response_row_mask = np.isfinite(
            pd.to_numeric(prepared_table[response_name], errors="coerce")
        )
        defined_reference_row_mask = (
            usable_reference_row_mask & finite_response_row_mask
        )
        defined_reference_area_m2 = float(
            prepared_table.loc[
                defined_reference_row_mask,
                "area_weight_m2",
            ].sum()
        )
        reference_area_coverage = (
            defined_reference_area_m2 / usable_reference_area_m2
            if usable_reference_area_m2 > 0
            else 0.0
        )
        response_values = prepared_table.loc[
            defined_reference_row_mask,
            response_name,
        ].to_numpy(dtype=np.float64)
        response_weights = prepared_table.loc[
            defined_reference_row_mask, "area_weight_m2"
        ].to_numpy(dtype=np.float64)
        reference_weighted_standard_deviation = weighted_standard_deviation(
            response_values,
            response_weights,
        )
        unique_reference_value_count = int(pd.Series(response_values).nunique())
        fold_training_rows = []
        fold_validation_rows = []
        for spatial_fold in range(1, configuration.fold_count + 1):
            fold_training_rows.append(
                int(
                    np.count_nonzero(
                        defined_reference_row_mask
                        & prepared_table["spatial_fold"].ne(spatial_fold)
                    )
                )
            )
            fold_validation_rows.append(
                int(
                    np.count_nonzero(
                        defined_reference_row_mask
                        & prepared_table["spatial_fold"].eq(spatial_fold)
                    )
                )
            )

        if response_name not in selected_response_name_set:
            status = "not_selected"
        elif not np.any(defined_reference_row_mask):
            status = "no_reference_values"
        elif reference_area_coverage < configuration.minimum_response_coverage:
            status = "insufficient_reference_coverage"
        elif (
            unique_reference_value_count < 2
            or not np.isfinite(reference_weighted_standard_deviation)
            or reference_weighted_standard_deviation <= 0
        ):
            status = "no_reference_variation"
        elif (
            min(fold_training_rows) < minimum_training_rows
            or min(fold_validation_rows) < 2
        ):
            status = "insufficient_spatial_fold_support"
        else:
            status = "fit"

        coverage_records.append(
            {
                "response": response_name,
                "selected": response_name in selected_response_name_set,
                "status": status,
                "defined_reference_rows": int(
                    np.count_nonzero(defined_reference_row_mask)
                ),
                "usable_reference_rows": int(
                    np.count_nonzero(usable_reference_row_mask)
                ),
                "defined_reference_area_m2": defined_reference_area_m2,
                "usable_reference_area_m2": usable_reference_area_m2,
                "reference_area_coverage": reference_area_coverage,
                "unique_reference_values": unique_reference_value_count,
                "reference_weighted_sd": (
                    reference_weighted_standard_deviation
                ),
                "minimum_fold_training_rows": min(fold_training_rows),
                "minimum_fold_validation_rows": min(fold_validation_rows),
            }
        )
    return pd.DataFrame.from_records(coverage_records)


@dataclass(frozen=True)
class ValidatedResponseModel:
    """A final response model with its spatially held-out predictions and diagnostics.

    Attributes:
        model: Model fitted to all usable reference rows for this response.
        training_table: Reference rows used for that final fit.
        out_of_fold_expected: Predictions made while each row's spatial block
            was held out; unusable rows remain NaN.
        observed_minus_expected: Observed response minus held-out prediction.
        standardized_departure: Those residuals divided by pooled reference RMSE.
        overall_metrics: Area-weighted diagnostics across held-out reference rows.
        fold_metrics: Reference counts, represented area and metrics for each fold.
    """

    model: habitat_condition.FittedResponseModel
    training_table: pd.DataFrame
    out_of_fold_expected: np.ndarray
    observed_minus_expected: np.ndarray
    standardized_departure: np.ndarray
    overall_metrics: dict[str, float]
    fold_metrics: pd.DataFrame


def fit_response_with_spatial_validation(
    prepared_data: PreparedReferenceConditionData,
    response_name: str,
    configuration: IntegrityConfiguration,
    on_fold_complete: Callable[[], None] | None = None,
) -> ValidatedResponseModel:
    """Validate one response across spatial folds, then fit all usable references.

    Imputation is learned from each fold's training rows. Held-out reference
    residuals determine the RMSE used to standardize response departures.

    Args:
        prepared_data: Rows and predictor definitions from
            ``prepare_reference_condition_data``.
        response_name: A response column that passed ``summarize_response_coverage``.
        configuration: Fold count and spline/ridge settings used during screening.
        on_fold_complete: Optional notification after each validation fold.
            Exceptions propagate, allowing a caller to cancel before further fits.

    Returns:
        Final model, its training rows and held-out predictions and diagnostics.

    Raises:
        ValueError: If the numerical library cannot fit the supplied data.
    """

    usable_row_mask = prepared_data.table["usable_for_gam"]
    reference_row_mask = prepared_data.table["reference_site"].eq(1)
    fold_metric_records = []
    finite_response_row_mask = np.isfinite(
        pd.to_numeric(
            prepared_data.table[response_name],
            errors="coerce",
        )
    )
    out_of_fold_expected_values = np.full(
        len(prepared_data.table),
        np.nan,
        dtype=np.float64,
    )
    for spatial_fold in range(1, configuration.fold_count + 1):
        training_row_mask = (
            usable_row_mask
            & reference_row_mask
            & finite_response_row_mask
            & prepared_data.table["spatial_fold"].ne(spatial_fold)
        )
        validation_reference_row_mask = (
            usable_row_mask
            & reference_row_mask
            & finite_response_row_mask
            & prepared_data.table["spatial_fold"].eq(spatial_fold)
        )
        assessment_row_mask = usable_row_mask & prepared_data.table[
            "spatial_fold"
        ].eq(
            spatial_fold,
        )
        training_table = prepared_data.table.loc[training_row_mask]
        imputation_values = calculate_imputation_values(
            training_table,
            prepared_data.continuous_predictor_names,
            prepared_data.categorical_predictor_name,
        )
        fitted_model = habitat_condition.fit_response_model(
            training_table,
            response_name,
            prepared_data.continuous_predictor_names,
            prepared_data.categorical_predictor_name,
            imputation_values,
            spline_knot_count=configuration.spline_knot_count,
            ridge_alpha=configuration.ridge_alpha,
        )
        fold_expected_values = habitat_condition.predict_expected_response(
            fitted_model,
            prepared_data.table.loc[assessment_row_mask],
        )
        out_of_fold_expected_values[assessment_row_mask.to_numpy()] = (
            fold_expected_values
        )
        validation_table = prepared_data.table.loc[
            validation_reference_row_mask
        ]
        validation_expected_values = out_of_fold_expected_values[
            validation_reference_row_mask.to_numpy()
        ]
        fold_metrics = calculate_regression_metrics(
            validation_table[response_name].to_numpy(dtype=np.float64),
            validation_expected_values,
            validation_table["area_weight_m2"].to_numpy(dtype=np.float64),
        )
        fold_metric_records.append(
            {
                "spatial_fold": spatial_fold,
                "training_reference_rows": int(
                    np.count_nonzero(training_row_mask)
                ),
                "validation_reference_rows": int(
                    np.count_nonzero(validation_reference_row_mask)
                ),
                "validation_reference_area_m2": float(
                    validation_table["area_weight_m2"].sum()
                ),
                **fold_metrics,
            }
        )
        if on_fold_complete is not None:
            on_fold_complete()

    observed_values = pd.to_numeric(
        prepared_data.table[response_name],
        errors="coerce",
    ).to_numpy(
        dtype=np.float64,
    )
    observed_minus_expected_values = (
        observed_values - out_of_fold_expected_values
    )
    reference_validation_row_mask = (
        usable_row_mask.to_numpy()
        & reference_row_mask.to_numpy()
        & np.isfinite(observed_values)
    )
    overall_metrics = calculate_regression_metrics(
        observed_values[reference_validation_row_mask],
        out_of_fold_expected_values[reference_validation_row_mask],
        prepared_data.table.loc[
            reference_validation_row_mask,
            "area_weight_m2",
        ].to_numpy(
            dtype=np.float64,
        ),
    )
    out_of_fold_reference_rmse = overall_metrics["weighted_rmse"]
    standardized_departure = (
        observed_minus_expected_values / out_of_fold_reference_rmse
        if out_of_fold_reference_rmse > 0
        else np.full_like(observed_minus_expected_values, np.nan)
    )

    final_training_row_mask = (
        usable_row_mask & reference_row_mask & finite_response_row_mask
    )
    final_training_table = prepared_data.table.loc[
        final_training_row_mask
    ]
    final_imputation_values = calculate_imputation_values(
        final_training_table,
        prepared_data.continuous_predictor_names,
        prepared_data.categorical_predictor_name,
    )
    final_model = habitat_condition.fit_response_model(
        final_training_table,
        response_name,
        prepared_data.continuous_predictor_names,
        prepared_data.categorical_predictor_name,
        final_imputation_values,
        spline_knot_count=configuration.spline_knot_count,
        ridge_alpha=configuration.ridge_alpha,
    )
    return ValidatedResponseModel(
        model=final_model,
        training_table=final_training_table,
        out_of_fold_expected=out_of_fold_expected_values,
        observed_minus_expected=observed_minus_expected_values,
        standardized_departure=standardized_departure,
        overall_metrics=overall_metrics,
        fold_metrics=pd.DataFrame.from_records(fold_metric_records),
    )
