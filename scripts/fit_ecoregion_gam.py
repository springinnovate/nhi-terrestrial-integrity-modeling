"""Fit and spatially validate an ecoregion reference-similarity GAM."""

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
from matplotlib.collections import PatchCollection
from matplotlib.patches import Patch, Rectangle
from pyproj import Transformer
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder, SplineTransformer
from tqdm.auto import tqdm


DEFAULT_SAMPLING_BLOCK_SIZE_METERS = 25_000
DEFAULT_VALIDATION_BLOCK_SIZE_METERS = 100_000
DEFAULT_FOLD_COUNT = 5
DEFAULT_MINIMUM_PREDICTOR_COVERAGE = 0.80
DEFAULT_MAXIMUM_ROW_MISSING_FRACTION = 0.20
DEFAULT_SPLINE_KNOT_COUNT = 6
DEFAULT_REGULARIZATION_C = 1.0
FIGURE_DPI = 300
TOP_AREA_FRACTIONS = (0.10, 0.20, 0.30)
EQUAL_AREA_CRS = "EPSG:8857"
REQUIRED_SAMPLE_COLUMNS = (
    "longitude",
    "latitude",
    "sampling_block_column",
    "sampling_block_row",
    "reference_site",
    "area_weight_m2",
)
ENVIRONMENTAL_BAND_PATTERN = re.compile(r"^y2018_d(2[0-9]|3[0-9])_")
LANDFORM_BAND_NUMBER = 35
PREDICTOR_DISPLAY_NAMES = {
    20: "Maximum annual temperature (C)",
    21: "Mean annual temperature (C)",
    22: "Median annual temperature (C)",
    23: "Minimum annual temperature (C)",
    24: "Annual precipitation (mm)",
    25: "Growing-season average temperature (C)",
    26: "Growing-season average precipitation (mm/day)",
    27: "Interannual rainfall variability (CV%, 10-year)",
    28: "Drought mean (SPI 30-day)",
    29: "Drought 5th percentile (SPI 30-day)",
    30: "Fire frequency (burned months)",
    31: "Annual variation in water presence",
    32: "Distance to streams (m)",
    33: "Soil organic carbon (10 cm, g/kg)",
    34: "Soil moisture annual mean (GLDAS 10-40 cm)",
    35: "Landform type (SRTM)",
    36: "Topographic diversity (ALOS)",
    37: "Annual evapotranspiration (MODIS ET, mm)",
    38: "Average snow depth when present (GLDAS, m)",
    39: "Average snow depth when present (SMAP, m)",
}


@dataclass(frozen=True)
class GamConfiguration:
    """Settings controlling spatial validation and additive model fitting.

    Attributes:
        fold_count: Number of spatial cross-validation folds.
        sampling_block_size_meters: Width of source sampling blocks.
        validation_block_size_meters: Width of grouped validation blocks.
        minimum_predictor_coverage: Minimum represented-area coverage needed
            to retain a predictor.
        maximum_row_missing_fraction: Largest retained-predictor missing
            fraction allowed for a modeled row.
        spline_knot_count: Number of knots for each continuous spline term.
        regularization_c: Inverse L2 regularization strength used by logistic
            regression.
    """

    fold_count: int = DEFAULT_FOLD_COUNT
    sampling_block_size_meters: int = DEFAULT_SAMPLING_BLOCK_SIZE_METERS
    validation_block_size_meters: int = DEFAULT_VALIDATION_BLOCK_SIZE_METERS
    minimum_predictor_coverage: float = DEFAULT_MINIMUM_PREDICTOR_COVERAGE
    maximum_row_missing_fraction: float = DEFAULT_MAXIMUM_ROW_MISSING_FRACTION
    spline_knot_count: int = DEFAULT_SPLINE_KNOT_COUNT
    regularization_c: float = DEFAULT_REGULARIZATION_C


@dataclass(frozen=True)
class PreparedGamData:
    """Sample table prepared for spatial model fitting.

    Attributes:
        table: Input rows with validation-block, fold, and missingness fields.
        block_summary: One row per 100 km validation block.
        predictor_coverage: Coverage and retention diagnostics by predictor.
        retained_predictor_names: Predictor columns retained for modeling.
        excluded_predictor_names: Predictor columns removed for low coverage.
        continuous_predictor_names: Retained continuous predictor columns.
        categorical_predictor_name: Retained categorical landform column.
    """

    table: pd.DataFrame
    block_summary: pd.DataFrame
    predictor_coverage: pd.DataFrame
    retained_predictor_names: tuple[str, ...]
    excluded_predictor_names: tuple[str, ...]
    continuous_predictor_names: tuple[str, ...]
    categorical_predictor_name: str


@dataclass
class FittedAdditiveGam:
    """Fitted preprocessing terms and logistic additive classifier.

    Attributes:
        continuous_predictor_names: Columns represented by spline bases.
        categorical_predictor_name: Landform column represented by indicators.
        imputation_values: Training-derived replacement for each predictor.
        preprocessor: Fitted spline and one-hot transformation.
        classifier: Fitted regularized logistic classifier.
    """

    continuous_predictor_names: tuple[str, ...]
    categorical_predictor_name: str
    imputation_values: dict[str, float]
    preprocessor: ColumnTransformer
    classifier: LogisticRegression

    def predict_reference_score(self, predictor_table: pd.DataFrame) -> np.ndarray:
        """Score rows after applying this model's training-derived replacements.

        Args:
            predictor_table: Table containing all model predictor columns.

        Returns:
            Relative reference-site similarity score for every row.
        """

        imputed_table = predictor_table.loc[
            :,
            (*self.continuous_predictor_names, self.categorical_predictor_name),
        ].fillna(self.imputation_values)
        design_matrix = self.preprocessor.transform(imputed_table)
        return self.classifier.predict_proba(design_matrix)[:, 1]


@dataclass(frozen=True)
class GamRunSummary:
    """Outputs and diagnostics from one complete GAM run.

    Attributes:
        output_directory: Directory containing every generated artifact.
        scored_sample_path: Parquet table containing folds and OOF scores.
        model_path: Serialized final model path.
        predictor_coverage_path: Predictor coverage CSV path.
        fold_metrics_path: Per-fold metric CSV path.
        aggregate_metrics_path: Aggregate metric JSON path.
        metadata_path: Run metadata JSON path.
        figure_paths: Publication figure paths.
        sampled_rows: Number of source sample rows.
        usable_rows: Number of rows included in modeling.
        validation_blocks: Number of grouped validation blocks.
        elapsed_seconds: End-to-end elapsed wall time.
    """

    output_directory: Path
    scored_sample_path: Path
    model_path: Path
    predictor_coverage_path: Path
    fold_metrics_path: Path
    aggregate_metrics_path: Path
    metadata_path: Path
    figure_paths: tuple[Path, ...]
    sampled_rows: int
    usable_rows: int
    validation_blocks: int
    elapsed_seconds: float


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed command-line namespace.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Fit and spatially validate an additive model of grassland "
            "reference-site similarity from one ecoregion sample Parquet."
        )
    )
    parser.add_argument("sample_parquet", type=Path, help="Spatial sample Parquet.")
    parser.add_argument(
        "--output-directory",
        type=Path,
        help="Output directory. Defaults to outputs/gam/<sample stem>.",
    )
    parser.add_argument(
        "--ecoregion-name",
        help="Figure label. Defaults to a name inferred from the sample filename.",
    )
    parser.add_argument(
        "--fold-count",
        type=int,
        default=DEFAULT_FOLD_COUNT,
        help=f"Spatial fold count (default: {DEFAULT_FOLD_COUNT}).",
    )
    parser.add_argument(
        "--sampling-block-size-m",
        type=int,
        default=DEFAULT_SAMPLING_BLOCK_SIZE_METERS,
        help=(
            "Source sampling-block width in meters "
            f"(default: {DEFAULT_SAMPLING_BLOCK_SIZE_METERS})."
        ),
    )
    parser.add_argument(
        "--validation-block-size-m",
        type=int,
        default=DEFAULT_VALIDATION_BLOCK_SIZE_METERS,
        help=(
            "Grouped validation-block width in meters "
            f"(default: {DEFAULT_VALIDATION_BLOCK_SIZE_METERS})."
        ),
    )
    parser.add_argument(
        "--minimum-predictor-coverage",
        type=float,
        default=DEFAULT_MINIMUM_PREDICTOR_COVERAGE,
        help=(
            "Minimum represented-area coverage for a predictor "
            f"(default: {DEFAULT_MINIMUM_PREDICTOR_COVERAGE:.2f})."
        ),
    )
    parser.add_argument(
        "--maximum-row-missing-fraction",
        type=float,
        default=DEFAULT_MAXIMUM_ROW_MISSING_FRACTION,
        help=(
            "Maximum retained-predictor missing fraction per modeled row "
            f"(default: {DEFAULT_MAXIMUM_ROW_MISSING_FRACTION:.2f})."
        ),
    )
    parser.add_argument(
        "--spline-knots",
        type=int,
        default=DEFAULT_SPLINE_KNOT_COUNT,
        help=(
            "Quantile knots in each continuous spline term "
            f"(default: {DEFAULT_SPLINE_KNOT_COUNT})."
        ),
    )
    parser.add_argument(
        "--regularization-c",
        type=float,
        default=DEFAULT_REGULARIZATION_C,
        help=(
            "Inverse L2 regularization strength "
            f"(default: {DEFAULT_REGULARIZATION_C:g})."
        ),
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bars.",
    )
    return parser.parse_args()


def infer_sample_ecoregion_name(sample_path: Path) -> str:
    """Infer a readable ecoregion label from a spatial-sample filename.

    Args:
        sample_path: Parquet path whose stem identifies the ecoregion.

    Returns:
        Human-readable ecoregion label.
    """

    name_stem = re.sub(
        r"_spatial_sample$",
        "",
        sample_path.stem,
        flags=re.IGNORECASE,
    )
    words = re.sub(r"[_-]+", " ", name_stem).strip()
    display_name = words.title()
    for conjunction in ("And", "Of", "The"):
        display_name = re.sub(
            rf"\b{conjunction}\b",
            conjunction.lower(),
            display_name,
        )
    return display_name or "Ecoregion"


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
    # A weighted empirical quantile is the first sampled value whose cumulative
    # represented area reaches the requested share of the ecoregion.
    quantile_offsets = np.searchsorted(
        cumulative_probabilities,
        requested_quantiles,
        side="left",
    )
    quantile_offsets = np.clip(quantile_offsets, 0, len(sorted_values) - 1)
    return sorted_values[quantile_offsets]


def assign_spatial_folds(
    sample_table: pd.DataFrame,
    configuration: GamConfiguration,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Group 25 km sampling blocks and balance them across spatial folds.

    Whole validation blocks remain in one fold. Blocks containing reference
    sites are assigned first to balance represented reference area, after
    which other blocks are assigned to balance total represented area.

    Args:
        sample_table: Spatial sample with block indices, labels, and area.
        configuration: Spatial block sizes and requested fold count.

    Returns:
        Copy of the table with validation-block and fold fields, plus one-row-
        per-block assignment diagnostics.

    Raises:
        ValueError: If block sizes are incompatible or there are too few
            reference-containing validation blocks for the requested folds.
    """

    if (
        configuration.validation_block_size_meters
        % configuration.sampling_block_size_meters
        != 0
    ):
        raise ValueError(
            "Validation-block size must be an integer multiple of sampling-block size."
        )
    sampling_blocks_per_validation_side = (
        configuration.validation_block_size_meters
        // configuration.sampling_block_size_meters
    )
    assigned_table = sample_table.copy()
    # Integer division by four combines a 4 x 4 group of 25 km cells, up to
    # 16 source sampling blocks, into one 100 km validation block.
    assigned_table["validation_block_column"] = (
        assigned_table["sampling_block_column"] // sampling_blocks_per_validation_side
    )
    assigned_table["validation_block_row"] = (
        assigned_table["sampling_block_row"] // sampling_blocks_per_validation_side
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
    if len(reference_blocks) < configuration.fold_count:
        raise ValueError(
            f"At least {configuration.fold_count} validation blocks containing "
            f"reference sites are required; found {len(reference_blocks)}."
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


def prepare_gam_data(
    sample_table: pd.DataFrame,
    configuration: GamConfiguration,
) -> PreparedGamData:
    """Select environmental predictors and prepare spatial validation rows.

    Predictor retention is based on represented-area coverage across the
    ecoregion sample. Row usability is then based on the missing fraction among
    retained predictors. Labels are kept exactly as supplied: one is a
    reference site and zero is sampled background.

    Args:
        sample_table: Table produced by ``load_ecoregion_geotiff.py``.
        configuration: Coverage, missingness, and spatial-fold settings.

    Returns:
        Prepared rows, folds, coverage diagnostics, and predictor names.

    Raises:
        ValueError: If the sample violates the expected table contract.
    """

    missing_required_columns = sorted(
        set(REQUIRED_SAMPLE_COLUMNS).difference(sample_table.columns)
    )
    if missing_required_columns:
        raise ValueError(
            "Sample is missing required columns: " + ", ".join(missing_required_columns)
        )
    if not sample_table["reference_site"].isin([0, 1]).all():
        raise ValueError("reference_site must contain only zero and one.")
    if not (sample_table["area_weight_m2"] > 0).all():
        raise ValueError("area_weight_m2 must be positive for every sample row.")

    predictor_band_numbers: dict[str, int] = {}
    for column_name in sample_table.columns:
        match = ENVIRONMENTAL_BAND_PATTERN.match(column_name)
        if match:
            predictor_band_numbers[column_name] = int(match.group(1))
    expected_band_numbers = set(range(20, 40))
    actual_band_numbers = set(predictor_band_numbers.values())
    if actual_band_numbers != expected_band_numbers:
        missing_band_numbers = sorted(expected_band_numbers - actual_band_numbers)
        raise ValueError(
            "The sample must contain every 2018 environmental band d20-d39; "
            f"missing bands: {missing_band_numbers}."
        )
    ordered_predictor_names = tuple(
        sorted(predictor_band_numbers, key=predictor_band_numbers.get)
    )
    categorical_predictor_name = next(
        name
        for name, band_number in predictor_band_numbers.items()
        if band_number == LANDFORM_BAND_NUMBER
    )

    total_represented_area = float(sample_table["area_weight_m2"].sum())
    coverage_records = []
    for predictor_name in ordered_predictor_names:
        defined = sample_table[predictor_name].notna()
        defined_area = float(sample_table.loc[defined, "area_weight_m2"].sum())
        area_coverage = defined_area / total_represented_area
        coverage_records.append(
            {
                "predictor": predictor_name,
                "band_number": predictor_band_numbers[predictor_name],
                "predictor_type": (
                    "categorical"
                    if predictor_name == categorical_predictor_name
                    else "continuous"
                ),
                "defined_rows": int(defined.sum()),
                "row_coverage": float(defined.mean()),
                "defined_area_m2": defined_area,
                "area_coverage": area_coverage,
                "retained": area_coverage >= configuration.minimum_predictor_coverage,
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
            "The landform predictor did not meet minimum coverage; categorical "
            "landform is required by the model contract."
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
    return PreparedGamData(
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

    Continuous predictors use represented-area weighted medians. Categorical
    landform uses the category representing the largest total training area.

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

    defined_landforms = training_table[categorical_predictor_name].notna()
    landform_areas = (
        training_table.loc[defined_landforms]
        .groupby(categorical_predictor_name, sort=True)["area_weight_m2"]
        .sum()
    )
    imputation_values[categorical_predictor_name] = float(landform_areas.idxmax())
    return imputation_values


def fit_additive_gam(
    training_table: pd.DataFrame,
    continuous_predictor_names: tuple[str, ...],
    categorical_predictor_name: str,
    imputation_values: dict[str, float],
    configuration: GamConfiguration,
) -> FittedAdditiveGam:
    """Fit a regularized additive logistic model to one training partition.

    Each continuous variable receives its own cubic spline basis and landform
    receives one-hot categorical terms. The transformed terms enter one
    logistic regression without interactions, making the model additive.
    Reference and background classes receive equal total fitting weight while
    represented-area differences remain intact within each class.

    Args:
        training_table: Usable rows not belonging to the held-out fold.
        continuous_predictor_names: Columns represented by spline terms.
        categorical_predictor_name: Column represented by indicator terms.
        imputation_values: Training-derived predictor replacements.
        configuration: Spline and regularization settings.

    Returns:
        Fitted preprocessor, classifier, and imputation values.
    """

    predictor_names = (*continuous_predictor_names, categorical_predictor_name)
    imputed_training_table = training_table.loc[:, predictor_names].fillna(
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
    design_matrix = preprocessor.fit_transform(imputed_training_table)

    reference_classes = training_table["reference_site"].to_numpy(dtype=np.uint8)
    represented_areas = training_table["area_weight_m2"].to_numpy(dtype=np.float64)
    fitting_weights = np.empty(len(training_table), dtype=np.float64)
    for reference_class in (0, 1):
        class_mask = reference_classes == reference_class
        represented_class_area = float(np.sum(represented_areas[class_mask]))
        # Each class sums to half the row count. This preserves within-class
        # area ratios while keeping regularization comparable across datasets.
        fitting_weights[class_mask] = (
            represented_areas[class_mask]
            * (len(training_table) / 2.0)
            / represented_class_area
        )

    classifier = LogisticRegression(
        C=configuration.regularization_c,
        solver="lbfgs",
        max_iter=2_000,
    )
    classifier.fit(
        design_matrix,
        reference_classes,
        sample_weight=fitting_weights,
    )
    return FittedAdditiveGam(
        continuous_predictor_names=continuous_predictor_names,
        categorical_predictor_name=categorical_predictor_name,
        imputation_values=imputation_values,
        preprocessor=preprocessor,
        classifier=classifier,
    )


def calculate_fold_metrics(
    validation_table: pd.DataFrame,
    reference_scores: np.ndarray,
) -> dict[str, float]:
    """Calculate represented-area ranking metrics for one held-out fold.

    These metrics assess whether held-out reference sites receive high scores
    relative to sampled background. They are not presence/absence accuracy
    estimates because zero labels represent unlabeled background.

    Args:
        validation_table: Usable rows from one held-out spatial fold.
        reference_scores: Out-of-fold score corresponding to each row.

    Returns:
        Weighted AUC, Boyce correlation, percentile, recovery, and separation
        measurements.
    """

    labels = validation_table["reference_site"].to_numpy(dtype=np.uint8)
    area_weights = validation_table["area_weight_m2"].to_numpy(dtype=np.float64)
    reference_mask = labels == 1
    reference_area = float(np.sum(area_weights[reference_mask]))
    total_area = float(np.sum(area_weights))
    metrics = {
        "weighted_reference_background_auc": float(
            roc_auc_score(labels, reference_scores, sample_weight=area_weights)
        )
    }

    score_area = (
        pd.DataFrame({"score": reference_scores, "area": area_weights})
        .groupby("score", sort=True, as_index=False)["area"]
        .sum()
    )
    score_area["percentile"] = score_area["area"].cumsum() / total_area
    percentile_by_score = dict(
        zip(score_area["score"], score_area["percentile"], strict=True)
    )
    reference_percentiles = np.array(
        [percentile_by_score[score] for score in reference_scores[reference_mask]],
        dtype=np.float64,
    )
    metrics["reference_score_percentile_mean"] = float(
        np.average(reference_percentiles, weights=area_weights[reference_mask])
    )
    metrics["reference_score_percentile_median"] = float(
        weighted_quantiles(
            reference_percentiles,
            area_weights[reference_mask],
            [0.5],
        )[0]
    )

    descending_offsets = np.argsort(-reference_scores, kind="stable")
    descending_areas = area_weights[descending_offsets]
    descending_reference = reference_mask[descending_offsets]
    cumulative_area_before_row = np.concatenate(
        ([0.0], np.cumsum(descending_areas[:-1]))
    )
    for area_fraction in TOP_AREA_FRACTIONS:
        area_budget = total_area * area_fraction
        included_area = np.clip(
            area_budget - cumulative_area_before_row,
            0.0,
            descending_areas,
        )
        recovered_reference_area = float(np.sum(included_area[descending_reference]))
        metrics[f"reference_recovery_top_{round(area_fraction * 100):02d}_pct"] = (
            recovered_reference_area / reference_area
        )

    minimum_score = float(np.min(reference_scores))
    maximum_score = float(np.max(reference_scores))
    score_range = maximum_score - minimum_score
    boyce_correlation = math.nan
    if score_range > 0:
        # Overlapping windows spanning 20% of the observed score range provide
        # a continuous predicted-to-expected curve instead of arbitrary bins.
        window_width = score_range * 0.20
        window_centers = np.linspace(minimum_score, maximum_score, 20)
        predicted_expected_ratios = []
        populated_centers = []
        for window_center in window_centers:
            in_window = np.abs(reference_scores - window_center) <= window_width / 2.0
            window_area = float(np.sum(area_weights[in_window]))
            if window_area == 0:
                continue
            window_reference_area = float(
                np.sum(area_weights[in_window & reference_mask])
            )
            expected_fraction = window_area / total_area
            observed_fraction = window_reference_area / reference_area
            populated_centers.append(window_center)
            predicted_expected_ratios.append(observed_fraction / expected_fraction)
        if len(populated_centers) >= 3 and np.ptp(predicted_expected_ratios) > 0:
            center_ranks = pd.Series(populated_centers).rank(method="average")
            ratio_ranks = pd.Series(predicted_expected_ratios).rank(method="average")
            boyce_correlation = float(center_ranks.corr(ratio_ranks))
    metrics["continuous_boyce_correlation"] = boyce_correlation

    reference_median = float(
        weighted_quantiles(
            reference_scores[reference_mask],
            area_weights[reference_mask],
            [0.5],
        )[0]
    )
    background_median = float(
        weighted_quantiles(
            reference_scores[~reference_mask],
            area_weights[~reference_mask],
            [0.5],
        )[0]
    )
    metrics["reference_score_median"] = reference_median
    metrics["background_score_median"] = background_median
    metrics["median_score_separation"] = reference_median - background_median
    return metrics


def _equal_area_sample_coordinates(
    sample_table: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Transform sampled pixel centers to equal-area kilometers.

    Args:
        sample_table: Sample rows containing longitude and latitude.

    Returns:
        Equal-area x and y coordinates in kilometers.
    """

    transformer = Transformer.from_crs(
        "EPSG:4326",
        EQUAL_AREA_CRS,
        always_xy=True,
    )
    x_meters, y_meters = transformer.transform(
        sample_table["longitude"].to_numpy(dtype=np.float64),
        sample_table["latitude"].to_numpy(dtype=np.float64),
    )
    return (
        np.asarray(x_meters, dtype=np.float64) / 1_000.0,
        np.asarray(y_meters, dtype=np.float64) / 1_000.0,
    )


def create_fold_map(
    block_summary: pd.DataFrame,
    sample_table: pd.DataFrame,
    configuration: GamConfiguration,
    ecoregion_name: str,
    output_path: Path,
) -> None:
    """Map grouped validation blocks over the sampled valid-pixel footprint.

    Args:
        block_summary: One row per assigned validation block.
        sample_table: Sampled valid pixels used to show the raster footprint.
        configuration: Validation block dimensions and fold count.
        ecoregion_name: Human-readable label included in the figure title.
        output_path: PNG path for the completed figure.
    """

    fold_colors = plt.get_cmap("Set2").colors[: configuration.fold_count]
    rectangles = []
    face_colors = []
    reference_rectangles = []
    block_size_kilometers = configuration.validation_block_size_meters / 1_000.0
    footprint_x, footprint_y = _equal_area_sample_coordinates(sample_table)
    for block in block_summary.itertuples(index=False):
        lower_left_x = block.validation_block_column * block_size_kilometers
        lower_left_y = block.validation_block_row * block_size_kilometers
        rectangle = Rectangle(
            (lower_left_x, lower_left_y),
            block_size_kilometers,
            block_size_kilometers,
        )
        rectangles.append(rectangle)
        face_colors.append(fold_colors[block.spatial_fold - 1])
        if block.represented_reference_area_m2 > 0:
            reference_rectangles.append(
                Rectangle(
                    (lower_left_x, lower_left_y),
                    block_size_kilometers,
                    block_size_kilometers,
                )
            )

    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 9}):
        figure, axis = plt.subplots(figsize=(9.0, 7.0), facecolor="white")
        axis.add_collection(
            PatchCollection(
                rectangles,
                facecolors=face_colors,
                edgecolors="white",
                linewidths=0.7,
            )
        )
        axis.scatter(
            footprint_x,
            footprint_y,
            color="#30383C",
            marker=".",
            s=1.2,
            alpha=0.28,
            linewidths=0,
            rasterized=True,
            zorder=2,
        )
        if reference_rectangles:
            axis.add_collection(
                PatchCollection(
                    reference_rectangles,
                    facecolors="none",
                    edgecolors="#161A1D",
                    linewidths=1.5,
                    zorder=3,
                )
            )
        axis.autoscale_view()
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlabel("Equal-area x coordinate (km)")
        axis.set_ylabel("Equal-area y coordinate (km)")
        axis.set_title(
            f"Spatial cross-validation folds\n{ecoregion_name}",
            fontsize=15,
            weight="bold",
            pad=34,
            linespacing=1.25,
        )
        axis.text(
            0.0,
            1.015,
            (
                "Gray points show sampled valid pixels; outlined blocks contain "
                "reference-site area"
            ),
            transform=axis.transAxes,
            ha="left",
            va="bottom",
            color="#4B5459",
        )
        legend_handles = [
            Patch(facecolor=fold_colors[index], label=f"Fold {index + 1}")
            for index in range(configuration.fold_count)
        ]
        legend_handles.append(
            plt.Line2D(
                [0],
                [0],
                marker=".",
                color="none",
                markerfacecolor="#30383C",
                markeredgecolor="none",
                markersize=6,
                label="Sampled valid pixels",
            )
        )
        legend_handles.append(
            Patch(
                facecolor="white",
                edgecolor="#161A1D",
                linewidth=1.5,
                label="Contains reference sites",
            )
        )
        axis.legend(
            handles=legend_handles,
            loc="best",
            frameon=True,
            facecolor="white",
            edgecolor="none",
            framealpha=0.94,
            ncol=2,
        )
        axis.spines[["top", "right"]].set_visible(False)
        figure.tight_layout()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
        plt.close(figure)


def create_oof_score_distribution_figure(
    scored_table: pd.DataFrame,
    ecoregion_name: str,
    output_path: Path,
) -> None:
    """Plot represented-area distributions of out-of-fold scores by class.

    Args:
        scored_table: Sample table containing finite OOF scores.
        ecoregion_name: Human-readable label included in the figure title.
        output_path: PNG path for the completed figure.
    """

    usable_table = scored_table[scored_table["oof_reference_score"].notna()]
    score_bins = np.linspace(0.0, 1.0, 31)
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10}):
        figure, axis = plt.subplots(figsize=(9.0, 5.7), facecolor="white")
        class_styles = (
            (0, "Sampled background", "#2A6F73"),
            (1, "Reference sites", "#C84B3A"),
        )
        for reference_class, label, color in class_styles:
            class_rows = usable_table["reference_site"] == reference_class
            class_area = usable_table.loc[class_rows, "area_weight_m2"]
            histogram_weights = class_area / class_area.sum() * 100.0
            axis.hist(
                usable_table.loc[class_rows, "oof_reference_score"],
                bins=score_bins,
                weights=histogram_weights,
                histtype="step",
                linewidth=2.2,
                color=color,
                label=label,
            )
        axis.set_xlabel("Out-of-fold reference-similarity score")
        axis.set_ylabel("Represented area within class (%)")
        axis.set_title(
            (
                "Held-out reference-similarity score distributions\n"
                f"{ecoregion_name}"
            ),
            fontsize=15,
            weight="bold",
            linespacing=1.25,
        )
        axis.legend(frameon=False)
        axis.grid(axis="y", color="#D6DADD", linewidth=0.6)
        axis.spines[["top", "right"]].set_visible(False)
        figure.tight_layout()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
        plt.close(figure)


def create_metric_variability_figure(
    fold_metrics: pd.DataFrame,
    ecoregion_name: str,
    output_path: Path,
) -> None:
    """Plot held-out ranking metrics and their variation among folds.

    Args:
        fold_metrics: One row of evaluation measurements per spatial fold.
        ecoregion_name: Human-readable label included in the figure title.
        output_path: PNG path for the completed figure.
    """

    metric_labels = {
        "weighted_reference_background_auc": (
            "How often reference sites outrank background\n(weighted AUC)"
        ),
        "continuous_boyce_correlation": (
            "Does reference enrichment rise with score?\n"
            "(Continuous Boyce rank correlation)"
        ),
        "reference_score_percentile_median": "Median reference percentile",
        "reference_recovery_top_10_pct": "Reference recovery: top 10% area",
        "reference_recovery_top_20_pct": "Reference recovery: top 20% area",
        "reference_recovery_top_30_pct": "Reference recovery: top 30% area",
    }
    plot_records = []
    for metric_name, metric_label in metric_labels.items():
        for row in fold_metrics.itertuples(index=False):
            plot_records.append(
                {
                    "metric": metric_label,
                    "fold": row.spatial_fold,
                    "value": getattr(row, metric_name),
                }
            )
    plot_table = pd.DataFrame.from_records(plot_records)
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 9}):
        figure, axis = plt.subplots(figsize=(10.0, 6.3), facecolor="white")
        fold_colors = plt.get_cmap("Set2").colors[: len(fold_metrics)]
        for metric_offset, metric_label in enumerate(metric_labels.values()):
            metric_rows = plot_table[plot_table["metric"] == metric_label]
            for row in metric_rows.itertuples(index=False):
                axis.scatter(
                    row.value,
                    metric_offset,
                    color=fold_colors[row.fold - 1],
                    edgecolor="white",
                    linewidth=0.5,
                    s=46,
                    zorder=3,
                )
            finite_values = metric_rows["value"].dropna()
            if not finite_values.empty:
                axis.scatter(
                    finite_values.mean(),
                    metric_offset,
                    marker="D",
                    color="#161A1D",
                    s=34,
                    zorder=4,
                )
        axis.set_yticks(range(len(metric_labels)), metric_labels.values())
        axis.invert_yaxis()
        axis.set_xlim(-1.05, 1.05)
        axis.axvline(0.0, color="#9AA2A6", linewidth=0.7)
        axis.set_xlabel("Metric value (higher is better; 1 is the maximum)")
        axis.set_title(
            f"Spatial holdout performance and variability\n{ecoregion_name}",
            fontsize=15,
            weight="bold",
            linespacing=1.25,
        )
        legend_handles = [
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=fold_colors[index],
                markeredgecolor="white",
                label=f"Fold {index + 1}",
                markersize=7,
            )
            for index in range(len(fold_metrics))
        ]
        legend_handles.append(
            plt.Line2D(
                [0],
                [0],
                marker="D",
                color="none",
                markerfacecolor="#161A1D",
                label="Fold mean",
                markersize=6,
            )
        )
        axis.legend(
            handles=legend_handles,
            frameon=False,
            ncol=3,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.10),
        )
        axis.grid(axis="x", color="#E1E4E5", linewidth=0.6)
        axis.spines[["top", "right", "left"]].set_visible(False)
        figure.tight_layout()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
        plt.close(figure)


def create_partial_response_figure(
    fitted_model: FittedAdditiveGam,
    usable_table: pd.DataFrame,
    ecoregion_name: str,
    output_path: Path,
) -> None:
    """Plot final-model partial response curves for every retained predictor.

    Each panel varies one predictor while holding all others at the final
    model's imputation baseline. Curves describe this fitted model and should
    not be interpreted as causal effects.

    Args:
        fitted_model: Final model refit using all usable rows.
        usable_table: All rows used by the final fit.
        ecoregion_name: Human-readable label included in the figure title.
        output_path: PNG path for the completed figure.
    """

    predictor_names = (
        *fitted_model.continuous_predictor_names,
        fitted_model.categorical_predictor_name,
    )
    baseline = {
        predictor_name: fitted_model.imputation_values[predictor_name]
        for predictor_name in predictor_names
    }
    figure_column_count = 4
    figure_row_count = math.ceil(len(predictor_names) / figure_column_count)
    figure, axes = plt.subplots(
        figure_row_count,
        figure_column_count,
        figsize=(16.0, 3.4 * figure_row_count),
        facecolor="white",
        squeeze=False,
    )
    for axis, predictor_name in zip(axes.flat, predictor_names, strict=False):
        band_number_match = ENVIRONMENTAL_BAND_PATTERN.match(predictor_name)
        display_name = PREDICTOR_DISPLAY_NAMES[int(band_number_match.group(1))]
        if predictor_name == fitted_model.categorical_predictor_name:
            category_values = np.sort(usable_table[predictor_name].dropna().unique())
            response_table = pd.DataFrame(
                [baseline] * len(category_values),
                columns=predictor_names,
            )
            response_table[predictor_name] = category_values
            reference_scores = fitted_model.predict_reference_score(response_table)
            axis.bar(
                np.arange(len(category_values)),
                reference_scores,
                color="#667F4A",
                width=0.75,
            )
            axis.set_xticks(
                np.arange(len(category_values)),
                [f"{value:g}" for value in category_values],
                rotation=45,
                ha="right",
            )
        else:
            predictor_values = usable_table[predictor_name].to_numpy(dtype=np.float64)
            area_weights = usable_table["area_weight_m2"].to_numpy(dtype=np.float64)
            lower_value, upper_value = weighted_quantiles(
                predictor_values,
                area_weights,
                [0.05, 0.95],
            )
            if math.isclose(lower_value, upper_value):
                response_table = pd.DataFrame([baseline], columns=predictor_names)
                reference_score = fitted_model.predict_reference_score(response_table)[
                    0
                ]
                axis.scatter(
                    [lower_value],
                    [reference_score],
                    color="#2A6F73",
                    s=32,
                    zorder=3,
                )
                axis.text(
                    0.5,
                    0.12,
                    "No variation in central 90%",
                    transform=axis.transAxes,
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    color="#4B5459",
                )
            else:
                response_values = np.linspace(lower_value, upper_value, 100)
                response_table = pd.DataFrame(
                    [baseline] * len(response_values),
                    columns=predictor_names,
                )
                response_table[predictor_name] = response_values
                reference_scores = fitted_model.predict_reference_score(response_table)
                axis.plot(
                    response_values,
                    reference_scores,
                    color="#2A6F73",
                    linewidth=2,
                )
                axis.axvline(
                    baseline[predictor_name],
                    color="#8C9498",
                    linewidth=0.8,
                    linestyle="--",
                )
        axis.set_title(display_name, fontsize=10, weight="bold")
        axis.set_ylim(0.0, 1.0)
        axis.grid(axis="y", color="#E1E4E5", linewidth=0.5)
        axis.spines[["top", "right"]].set_visible(False)
    for axis in axes.flat[len(predictor_names) :]:
        axis.set_visible(False)
    figure.suptitle(
        f"Final additive model partial responses\n{ecoregion_name}",
        fontsize=18,
        weight="bold",
        y=0.995,
        linespacing=1.25,
    )
    figure.text(
        0.5,
        0.004,
        "One predictor varied from its represented-area 5th to 95th percentile; "
        "others held at the fitted baseline",
        ha="center",
        va="bottom",
        fontsize=10,
        color="#4B5459",
    )
    figure.text(
        0.004,
        0.5,
        "Relative reference-site similarity",
        ha="left",
        va="center",
        rotation=90,
        fontsize=11,
    )
    figure.tight_layout(rect=(0.025, 0.025, 1.0, 0.945))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(figure)


def run_spatial_gam(
    sample_path: Path,
    output_directory: Path,
    configuration: GamConfiguration,
    show_progress: bool,
    ecoregion_name: str | None = None,
) -> GamRunSummary:
    """Run spatial cross-validation, final fitting, reporting, and outputs.

    Args:
        sample_path: Spatial sample Parquet from the raster-loading script.
        output_directory: Directory for model, tables, metadata, and figures.
        configuration: Spatial validation and model settings.
        show_progress: Whether to show tqdm progress bars.
        ecoregion_name: Optional figure label overriding filename inference.

    Returns:
        Paths and principal counts for the completed run.
    """

    started = time.perf_counter()
    resolved_sample_path = sample_path.expanduser().resolve()
    resolved_output_directory = output_directory.expanduser().resolve()
    resolved_ecoregion_name = (
        ecoregion_name or infer_sample_ecoregion_name(resolved_sample_path)
    )
    print("Spatial GAM validation")
    print(f"Input sample: {resolved_sample_path}")
    print(f"Output directory: {resolved_output_directory}")
    print(f"Ecoregion: {resolved_ecoregion_name}")
    sample_table = pd.read_parquet(resolved_sample_path)
    print(
        f"Loaded {len(sample_table):,} sampled rows x {sample_table.shape[1]:,} columns"
    )

    prepared = prepare_gam_data(sample_table, configuration)
    usable_mask = prepared.table["usable_for_gam"]
    usable_table = prepared.table.loc[usable_mask]
    represented_area = float(prepared.table["area_weight_m2"].sum())
    usable_area = float(usable_table["area_weight_m2"].sum())
    reference_area = float(
        prepared.table.loc[
            prepared.table["reference_site"] == 1,
            "area_weight_m2",
        ].sum()
    )
    usable_reference_area = float(
        usable_table.loc[
            usable_table["reference_site"] == 1,
            "area_weight_m2",
        ].sum()
    )
    usable_rows_requiring_imputation = usable_table["imputed_predictor_count"] > 0
    imputed_represented_area = float(
        usable_table.loc[
            usable_rows_requiring_imputation,
            "area_weight_m2",
        ].sum()
    )
    print()
    print("Predictor coverage and row usability")
    print(
        "Model predictors: 2018 environmental bands d20-d39 "
        f"({len(prepared.retained_predictor_names)} retained, "
        f"{len(prepared.excluded_predictor_names)} excluded)"
    )
    for row in prepared.predictor_coverage.itertuples(index=False):
        status = "retain" if row.retained else "exclude"
        print(
            f"  d{row.band_number:02d} {row.predictor_type:<11} "
            f"rows={row.row_coverage:6.1%} area={row.area_coverage:6.1%} "
            f"{status}"
        )
    print(
        f"Usable rows: {len(usable_table):,} / {len(prepared.table):,} "
        f"({len(usable_table) / len(prepared.table):.1%})"
    )
    print(
        f"Usable represented area: {usable_area / 1_000_000:,.2f} / "
        f"{represented_area / 1_000_000:,.2f} km^2 "
        f"({usable_area / represented_area:.1%})"
    )
    print(
        f"Usable reference area: {usable_reference_area / 1_000_000:,.2f} / "
        f"{reference_area / 1_000_000:,.2f} km^2 "
        f"({usable_reference_area / reference_area:.1%})"
    )
    print(
        "Usable rows requiring fold-specific imputation: "
        f"{int(usable_rows_requiring_imputation.sum()):,} / "
        f"{len(usable_table):,} ({usable_rows_requiring_imputation.mean():.1%}), "
        f"representing {imputed_represented_area / 1_000_000:,.2f} km^2"
    )

    fold_composition = (
        prepared.block_summary.groupby("spatial_fold", as_index=False)
        .agg(
            validation_blocks=("validation_block_id", "size"),
            reference_blocks=(
                "represented_reference_area_m2",
                lambda values: int(np.count_nonzero(values > 0)),
            ),
            represented_area_m2=("represented_area_m2", "sum"),
            represented_reference_area_m2=(
                "represented_reference_area_m2",
                "sum",
            ),
        )
        .sort_values("spatial_fold")
    )
    print()
    print("Spatial fold composition")
    print(
        f"Grouped {len(prepared.block_summary):,} validation blocks at "
        f"{configuration.validation_block_size_meters / 1_000:g} km; whole "
        "blocks are held out together."
    )
    for row in fold_composition.itertuples(index=False):
        print(
            f"  Fold {row.spatial_fold}: {row.validation_blocks} blocks, "
            f"{row.reference_blocks} reference blocks, "
            f"{row.represented_area_m2 / 1_000_000:,.1f} km^2 total, "
            f"{row.represented_reference_area_m2 / 1_000_000:,.1f} km^2 "
            "reference"
        )

    oof_scores = np.full(len(prepared.table), np.nan, dtype=np.float64)
    fold_metric_records = []
    fold_iterator = tqdm(
        range(1, configuration.fold_count + 1),
        desc="Fitting spatial folds",
        unit="fold",
        disable=not show_progress,
    )
    for spatial_fold in fold_iterator:
        training_rows = usable_mask & (prepared.table["spatial_fold"] != spatial_fold)
        validation_rows = usable_mask & (prepared.table["spatial_fold"] == spatial_fold)
        training_table = prepared.table.loc[training_rows]
        validation_table = prepared.table.loc[validation_rows]
        if training_table["reference_site"].nunique() != 2:
            raise ValueError(f"Fold {spatial_fold} training rows need both classes.")
        if validation_table["reference_site"].nunique() != 2:
            raise ValueError(f"Fold {spatial_fold} validation rows need both classes.")

        imputation_values = calculate_imputation_values(
            training_table,
            prepared.continuous_predictor_names,
            prepared.categorical_predictor_name,
        )
        fitted_model = fit_additive_gam(
            training_table,
            prepared.continuous_predictor_names,
            prepared.categorical_predictor_name,
            imputation_values,
            configuration,
        )
        validation_scores = fitted_model.predict_reference_score(validation_table)
        oof_scores[validation_rows.to_numpy()] = validation_scores
        fold_metrics = calculate_fold_metrics(validation_table, validation_scores)
        validation_reference_mask = validation_table["reference_site"] == 1
        imputed_rows = validation_table["imputed_predictor_count"] > 0
        fold_metric_records.append(
            {
                "spatial_fold": spatial_fold,
                "training_rows": int(len(training_table)),
                "validation_rows": int(len(validation_table)),
                "validation_blocks": int(
                    validation_table["validation_block_id"].nunique()
                ),
                "validation_reference_rows": int(validation_reference_mask.sum()),
                "validation_area_m2": float(validation_table["area_weight_m2"].sum()),
                "validation_reference_area_m2": float(
                    validation_table.loc[
                        validation_reference_mask,
                        "area_weight_m2",
                    ].sum()
                ),
                "imputed_validation_rows": int(imputed_rows.sum()),
                "imputed_validation_area_m2": float(
                    validation_table.loc[imputed_rows, "area_weight_m2"].sum()
                ),
                **fold_metrics,
            }
        )
        fold_iterator.set_postfix(
            auc=f"{fold_metrics['weighted_reference_background_auc']:.3f}",
            recovery20=(f"{fold_metrics['reference_recovery_top_20_pct']:.3f}"),
        )

    if not np.isfinite(oof_scores[usable_mask.to_numpy()]).all():
        raise RuntimeError("Not every usable row received one out-of-fold score.")
    scored_table = prepared.table.copy()
    scored_table["oof_reference_score"] = oof_scores
    fold_metrics = pd.DataFrame.from_records(fold_metric_records)

    final_imputation_values = calculate_imputation_values(
        usable_table,
        prepared.continuous_predictor_names,
        prepared.categorical_predictor_name,
    )
    final_model = fit_additive_gam(
        usable_table,
        prepared.continuous_predictor_names,
        prepared.categorical_predictor_name,
        final_imputation_values,
        configuration,
    )

    metric_columns = [
        "weighted_reference_background_auc",
        "continuous_boyce_correlation",
        "reference_score_percentile_mean",
        "reference_score_percentile_median",
        "reference_recovery_top_10_pct",
        "reference_recovery_top_20_pct",
        "reference_recovery_top_30_pct",
        "reference_score_median",
        "background_score_median",
        "median_score_separation",
    ]
    aggregate_metrics = {
        metric_name: {
            "mean": float(fold_metrics[metric_name].mean()),
            "standard_deviation": float(fold_metrics[metric_name].std(ddof=1)),
            "minimum": float(fold_metrics[metric_name].min()),
            "maximum": float(fold_metrics[metric_name].max()),
        }
        for metric_name in metric_columns
    }

    resolved_output_directory.mkdir(parents=True, exist_ok=True)
    figure_directory = resolved_output_directory / "figures"
    scored_sample_path = resolved_output_directory / "spatial_gam_scores.parquet"
    model_path = resolved_output_directory / "final_spatial_gam.joblib"
    predictor_coverage_path = resolved_output_directory / "predictor_coverage.csv"
    fold_metrics_path = resolved_output_directory / "fold_metrics.csv"
    aggregate_metrics_path = resolved_output_directory / "aggregate_metrics.json"
    metadata_path = resolved_output_directory / "run_metadata.json"
    figure_paths = (
        figure_directory / "spatial_folds.png",
        figure_directory / "oof_score_distributions.png",
        figure_directory / "fold_metric_variability.png",
        figure_directory / "partial_response_curves.png",
    )

    output_progress = tqdm(
        total=10,
        desc="Writing GAM outputs",
        unit="artifact",
        disable=not show_progress,
    )
    scored_table.to_parquet(scored_sample_path, compression="zstd", index=False)
    output_progress.update()
    joblib.dump(final_model, model_path, compress=3)
    output_progress.update()
    prepared.predictor_coverage.to_csv(predictor_coverage_path, index=False)
    output_progress.update()
    fold_metrics.to_csv(fold_metrics_path, index=False)
    output_progress.update()
    aggregate_metrics_path.write_text(
        json.dumps(aggregate_metrics, indent=2),
        encoding="utf-8",
    )
    output_progress.update()

    create_fold_map(
        prepared.block_summary,
        prepared.table,
        configuration,
        resolved_ecoregion_name,
        figure_paths[0],
    )
    output_progress.update()
    create_oof_score_distribution_figure(
        scored_table,
        resolved_ecoregion_name,
        figure_paths[1],
    )
    output_progress.update()
    create_metric_variability_figure(
        fold_metrics,
        resolved_ecoregion_name,
        figure_paths[2],
    )
    output_progress.update()
    create_partial_response_figure(
        final_model,
        usable_table,
        resolved_ecoregion_name,
        figure_paths[3],
    )
    output_progress.update()

    metadata = {
        "input_sample": str(resolved_sample_path),
        "ecoregion_name": resolved_ecoregion_name,
        "configuration": asdict(configuration),
        "model": {
            "family": "regularized logistic additive model",
            "continuous_terms": "independent cubic spline bases",
            "categorical_terms": "one-hot landform indicators",
            "interactions": False,
            "score_interpretation": (
                "relative similarity to supplied reference sites, not a "
                "calibrated probability of natural grassland presence"
            ),
        },
        "sampled_rows": int(len(prepared.table)),
        "usable_rows": int(usable_mask.sum()),
        "excluded_rows": int((~usable_mask).sum()),
        "represented_area_m2": represented_area,
        "usable_represented_area_m2": usable_area,
        "reference_area_m2": reference_area,
        "usable_reference_area_m2": usable_reference_area,
        "usable_rows_requiring_imputation": int(usable_rows_requiring_imputation.sum()),
        "imputed_represented_area_m2": imputed_represented_area,
        "validation_block_count": int(len(prepared.block_summary)),
        "retained_predictors": list(prepared.retained_predictor_names),
        "excluded_predictors": list(prepared.excluded_predictor_names),
        "final_imputation_values": final_imputation_values,
        "artifacts": {
            "scored_sample": str(scored_sample_path),
            "model": str(model_path),
            "predictor_coverage": str(predictor_coverage_path),
            "fold_metrics": str(fold_metrics_path),
            "aggregate_metrics": str(aggregate_metrics_path),
            "figures": [str(path) for path in figure_paths],
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    output_progress.update()
    output_progress.close()

    print()
    print("Out-of-fold performance")
    print(
        "Metrics compare supplied reference sites with sampled background; "
        "they are ranking diagnostics, not presence/absence accuracy."
    )
    for row in fold_metrics.itertuples(index=False):
        print(
            f"  Fold {row.spatial_fold}: "
            f"AUC={row.weighted_reference_background_auc:.3f}, "
            f"Boyce={row.continuous_boyce_correlation:.3f}, "
            f"top-20% recovery={row.reference_recovery_top_20_pct:.1%}, "
            f"median percentile={row.reference_score_percentile_median:.1%}"
        )
    print("Fold means and variability")
    for metric_name in (
        "weighted_reference_background_auc",
        "continuous_boyce_correlation",
        "reference_score_percentile_median",
        "reference_recovery_top_20_pct",
        "median_score_separation",
    ):
        values = aggregate_metrics[metric_name]
        print(
            f"  {metric_name}: {values['mean']:.3f} +/- "
            f"{values['standard_deviation']:.3f} "
            f"(range {values['minimum']:.3f}-{values['maximum']:.3f})"
        )
    elapsed_seconds = time.perf_counter() - started
    print()
    print("Artifacts")
    print(f"Scored sample: {scored_sample_path}")
    print(f"Final model: {model_path}")
    print(f"Fold metrics: {fold_metrics_path}")
    print(f"Metadata: {metadata_path}")
    for figure_path in figure_paths:
        print(f"Figure: {figure_path}")
    print(f"Completed in {elapsed_seconds:.2f} seconds")

    return GamRunSummary(
        output_directory=resolved_output_directory,
        scored_sample_path=scored_sample_path,
        model_path=model_path,
        predictor_coverage_path=predictor_coverage_path,
        fold_metrics_path=fold_metrics_path,
        aggregate_metrics_path=aggregate_metrics_path,
        metadata_path=metadata_path,
        figure_paths=figure_paths,
        sampled_rows=int(len(prepared.table)),
        usable_rows=int(usable_mask.sum()),
        validation_blocks=int(len(prepared.block_summary)),
        elapsed_seconds=elapsed_seconds,
    )


def main() -> None:
    """Run the command-line GAM workflow."""

    args = parse_args()
    configuration = GamConfiguration(
        fold_count=args.fold_count,
        sampling_block_size_meters=args.sampling_block_size_m,
        validation_block_size_meters=args.validation_block_size_m,
        minimum_predictor_coverage=args.minimum_predictor_coverage,
        maximum_row_missing_fraction=args.maximum_row_missing_fraction,
        spline_knot_count=args.spline_knots,
        regularization_c=args.regularization_c,
    )
    output_directory = args.output_directory or (
        Path("outputs") / "gam" / args.sample_parquet.stem
    )
    try:
        run_spatial_gam(
            args.sample_parquet,
            output_directory,
            configuration,
            not args.no_progress,
            ecoregion_name=args.ecoregion_name,
        )
    except (FileNotFoundError, OSError, ValueError, RuntimeError) as error:
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    main()
