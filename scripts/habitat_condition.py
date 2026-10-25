"""Fit expected responses and measure departures from reference-site conditions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder, SplineTransformer


FLOAT_NODATA = -9999.0
STATUS_NODATA = 255
STATUS_OUTSIDE_TARGET = 0
STATUS_INSUFFICIENT_PREDICTORS = 1
STATUS_PREDICTED = 2


class FittedResponseModel(TypedDict):
    """Predictor definitions and fitted estimators needed to predict one response.

    Attributes:
        continuous_predictor_names: Ordered columns represented by cubic splines.
        categorical_predictor_name: Column represented by one-hot categories.
        imputation_values: Training-derived replacements for missing predictors.
        preprocessor: Fitted spline and categorical transformations.
        regressor: Fitted area-weighted ridge regression.
    """

    continuous_predictor_names: tuple[str, ...]
    categorical_predictor_name: str
    imputation_values: dict[str, float]
    preprocessor: ColumnTransformer
    regressor: Ridge


def fit_response_model(
    training_table: pd.DataFrame,
    response_name: str,
    continuous_predictor_names: tuple[str, ...],
    categorical_predictor_name: str,
    imputation_values: dict[str, float],
    *,
    spline_knot_count: int,
    ridge_alpha: float,
) -> FittedResponseModel:
    """Fit an area-weighted spline and categorical regression for one response.

    Args:
        training_table: Selected reference rows containing predictor columns,
            a finite response, and positive represented areas in ``area_weight_m2``.
        response_name: Column to predict from the environmental predictors.
        continuous_predictor_names: Columns represented by independent cubic
            splines, using quantile knots and linear extrapolation.
        categorical_predictor_name: Column represented by one-hot categories.
        imputation_values: Training-derived replacement for each predictor.
        spline_knot_count: Number of knots for each continuous predictor.
        ridge_alpha: L2 regularization strength. Area weights are normalized
            to a mean of one before fitting.

    Returns:
        Predictor definitions and fitted estimators for ``predict_expected_response``.

    Raises:
        ValueError: If scikit-learn cannot fit the supplied data or settings.
    """

    predictor_names = (*continuous_predictor_names, categorical_predictor_name)
    imputed_predictors = training_table.loc[:, predictor_names].fillna(
        imputation_values
    )
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "continuous_splines",
                SplineTransformer(
                    n_knots=spline_knot_count,
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
    regressor = Ridge(alpha=ridge_alpha)
    regressor.fit(
        design_matrix,
        training_table[response_name].to_numpy(dtype=np.float64),
        sample_weight=fitting_weights,
    )
    return {
        "continuous_predictor_names": continuous_predictor_names,
        "categorical_predictor_name": categorical_predictor_name,
        "imputation_values": imputation_values,
        "preprocessor": preprocessor,
        "regressor": regressor,
    }


def predict_expected_response(
    fitted_model: FittedResponseModel,
    predictor_table: pd.DataFrame,
) -> np.ndarray:
    """Predict expected responses using the fitted preprocessing and regression.

    Args:
        fitted_model: Numerical model returned by ``fit_response_model``; CLI
            artifacts contain the same fields alongside their report metadata.
        predictor_table (pandas.DataFrame): Rows containing every predictor
            named by the fitted model.

    Returns:
        numpy.ndarray: Expected reference-condition response for each input row.
    """

    continuous_predictor_names = fitted_model["continuous_predictor_names"]
    categorical_predictor_name = fitted_model["categorical_predictor_name"]
    predictor_names = (*continuous_predictor_names, categorical_predictor_name)
    imputed_predictor_table = predictor_table.loc[:, predictor_names].fillna(
        fitted_model["imputation_values"]
    )
    design_matrix = fitted_model["preprocessor"].transform(
        imputed_predictor_table
    )
    return np.asarray(fitted_model["regressor"].predict(design_matrix))


@dataclass(frozen=True)
class ReferenceDepartureDistribution:
    """Reference distribution used to convert response vectors into percentiles.

    Attributes:
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
        covariance_condition_number: Condition number before stabilization.
        stabilized_covariance_condition_number: Condition number after
            stabilization.
    """

    reference_mean_vector: np.ndarray
    reference_covariance_matrix: np.ndarray
    stabilized_reference_covariance_matrix: np.ndarray
    reference_precision_matrix: np.ndarray
    sorted_reference_distances: np.ndarray
    cumulative_reference_area_fractions: np.ndarray
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
            Area-weighted empirical reference percentiles on the 0–1 scale,
            including all reference distances equal to each input distance.
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


def fit_reference_departure_distribution(
    complete_reference_departures: np.ndarray,
    complete_reference_area_weights: np.ndarray,
    covariance_shrinkage: float,
) -> ReferenceDepartureDistribution:
    """Fit weighted covariance and an empirical distribution of reference distances.

    Args:
        complete_reference_departures: Finite out-of-fold standardized residuals,
            with one selected reference row per row and one response per column.
        complete_reference_area_weights: Positive finite represented area for
            each selected reference row, in the same order.
        covariance_shrinkage: Fraction strictly between zero and one that shrinks
            the population covariance toward its diagonal before inversion.

    Returns:
        Weighted reference mean, covariance, precision and distance distribution.

    Raises:
        ValueError: If there are not more reference rows than responses, or a
            response has no positive finite reference variance.
        numpy.linalg.LinAlgError: If the stabilized covariance cannot be inverted.
    """

    if len(complete_reference_departures) <= complete_reference_departures.shape[1]:
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

    return ReferenceDepartureDistribution(
        reference_mean_vector=reference_mean_vector,
        reference_covariance_matrix=reference_covariance_matrix,
        stabilized_reference_covariance_matrix=(
            stabilized_reference_covariance_matrix
        ),
        reference_precision_matrix=reference_precision_matrix,
        sorted_reference_distances=sorted_reference_distances,
        cumulative_reference_area_fractions=cumulative_reference_area_fractions,
        covariance_condition_number=float(
            np.linalg.cond(reference_covariance_matrix)
        ),
        stabilized_covariance_condition_number=float(
            np.linalg.cond(stabilized_reference_covariance_matrix)
        ),
    )


@dataclass(frozen=True)
class RasterWindowPrediction:
    """Expected responses, departures and pixel status for one raster window.

    Attributes:
        expected_reference: Float32 expected values, ordered as the response
            models, with shape (responses, rows, columns).
        observed_minus_expected: Float32 observed-minus-expected values in the
            same order and shape. Missing observations remain FLOAT_NODATA.
        standardized_deviation: Float32 departures divided by each response's
            cross-validated reference RMSE, with the same order and shape.
        departure_percentile: Float32 (rows, columns) departure-distance
            percentiles against the reference distribution; higher means farther
            from reference conditions. Reference pixels and incomplete response
            vectors remain FLOAT_NODATA.
        inference_status: Uint8 (2, rows, columns) array: prediction status, then
            missing-predictor count. Status is STATUS_NODATA outside the AOI,
            STATUS_OUTSIDE_TARGET outside assessment, STATUS_INSUFFICIENT_PREDICTORS
            when missingness exceeds the allowed fraction, or STATUS_PREDICTED.
            Missing counts are STATUS_NODATA outside assessment. All other
            uncalculated float outputs use FLOAT_NODATA.
    """

    expected_reference: np.ndarray
    observed_minus_expected: np.ndarray
    standardized_deviation: np.ndarray
    departure_percentile: np.ndarray
    inference_status: np.ndarray


def predict_raster_window(
    predictor_values: np.ndarray,
    observed_responses: np.ndarray,
    *,
    predictor_names: tuple[str, ...],
    response_names: tuple[str, ...],
    response_models: tuple[FittedResponseModel, ...],
    reference_rmses: tuple[float, ...],
    reference_distribution: ReferenceDepartureDistribution,
    aoi_pixel_mask: np.ndarray,
    reference_pixel_mask: np.ndarray,
    assessment_pixel_mask: np.ndarray | None,
    maximum_predictor_missing_fraction: float,
) -> RasterWindowPrediction:
    """Predict reference conditions and score observed departures in one window.

    Use this after reading aligned inputs and preparing their boolean masks.
    Without an assessment mask, the target is AOI pixels with at least one valid
    predictor. Otherwise, the target is the AOI intersected with that mask.
    Allowed missing predictors are imputed by each fitted response model.
    Predictions require no observed response; departures require that response,
    and percentiles require all responses and exclude reference pixels.

    Args:
        predictor_values: Float64 (predictors, rows, columns) array containing
            finite values or NaN for missing data, ordered as predictor_names.
        observed_responses: Float64 (responses, rows, columns) array containing
            finite values or NaN for missing data, ordered as response_models.
        predictor_names: Ordered predictor columns shared by the fitted models.
        response_names: Response labels in model order, used in prediction errors.
        response_models: Fitted numerical models, in the same response order as
            the reference distribution.
        reference_rmses: Positive, finite cross-validated reference RMSEs in
            response order, established by the fitting or model-loading caller.
        reference_distribution: Fitted distance and empirical percentile model.
        aoi_pixel_mask: Boolean mask selecting the exact analysis area.
        reference_pixel_mask: Boolean mask identifying valid reference pixels
            inside that area. Their combined percentiles are left as NoData.
        assessment_pixel_mask: Optional boolean mask selecting assessment pixels.
            All input arrays and masks share the caller's window shape.
        maximum_predictor_missing_fraction: Training limit on the fraction of
            missing predictors allowed for prediction.

    Returns:
        Expected responses, departures, reference percentiles and status arrays.
        Expected values are rounded to float32 before subtraction; standardized
        departures are rounded to float32 before distance calculation.

    Raises:
        RuntimeError: If a fitted model produces a nonfinite prediction.
    """

    predictor_count = len(predictor_names)
    window_shape = aoi_pixel_mask.shape
    predictor_value_validity = ~np.isnan(predictor_values)
    response_value_validity = ~np.isnan(observed_responses)
    missing_predictor_counts = np.count_nonzero(
        ~predictor_value_validity,
        axis=0,
    ).astype(np.uint8)
    if assessment_pixel_mask is None:
        target_pixel_mask = aoi_pixel_mask & np.any(
            predictor_value_validity,
            axis=0,
        )
    else:
        target_pixel_mask = aoi_pixel_mask & assessment_pixel_mask
    usable_pixel_mask = target_pixel_mask & (
        missing_predictor_counts / predictor_count
        <= maximum_predictor_missing_fraction
    )
    complete_response_pixel_mask = usable_pixel_mask & np.all(
        response_value_validity,
        axis=0,
    )

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
            expected_reference_values = (
                predict_expected_response(
                    response_model,
                    predictor_table,
                )
            )
            if not np.isfinite(expected_reference_values).all():
                raise RuntimeError(
                    f"{response_names[response_model_offset]} produced a "
                    "nonfinite prediction."
                )
            expected_output_flat = expected_output[
                response_model_offset
            ].ravel()
            expected_output_flat[usable_pixel_mask_flat] = (
                expected_reference_values.astype(np.float32)
            )
            observed_values = observed_responses[response_model_offset].ravel()
            deviation_pixel_mask = (
                usable_pixel_mask_flat
                & response_value_validity[response_model_offset].ravel()
            )
            deviation_values = (
                observed_values[deviation_pixel_mask]
                - expected_output_flat[deviation_pixel_mask]
            )
            standardized_values = (
                deviation_values / reference_rmses[response_model_offset]
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
        reference_distribution.calculate_mahalanobis_distances(
            standardized_departure_vectors
        )
    )
    departure_percentiles = (
        reference_distribution.calculate_reference_departure_percentiles(
            mahalanobis_distances
        )
    )
    percentile_output[complete_non_reference_pixel_mask] = (
        departure_percentiles.astype(np.float32)
    )
    return RasterWindowPrediction(
        expected_reference=expected_output,
        observed_minus_expected=deviation_output,
        standardized_deviation=standardized_output,
        departure_percentile=percentile_output,
        inference_status=np.stack([status_output, imputation_output]),
    )
