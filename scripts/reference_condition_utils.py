"""Shared data preparation utilities for reference-condition models."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import PatchCollection
from matplotlib.patches import Patch, Rectangle
from pyproj import Transformer


DEFAULT_SAMPLING_BLOCK_SIZE_METERS = 25_000
DEFAULT_VALIDATION_BLOCK_SIZE_METERS = 100_000
DEFAULT_FOLD_COUNT = 5
DEFAULT_MINIMUM_PREDICTOR_COVERAGE = 0.80
DEFAULT_MAXIMUM_ROW_MISSING_FRACTION = 0.20
DEFAULT_SPLINE_KNOT_COUNT = 6
FIGURE_DPI = 300
EQUAL_AREA_CRS = "EPSG:8857"
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

    fold_count: int = DEFAULT_FOLD_COUNT
    sampling_block_size_meters: int = DEFAULT_SAMPLING_BLOCK_SIZE_METERS
    validation_block_size_meters: int = DEFAULT_VALIDATION_BLOCK_SIZE_METERS
    minimum_predictor_coverage: float = DEFAULT_MINIMUM_PREDICTOR_COVERAGE
    maximum_row_missing_fraction: float = DEFAULT_MAXIMUM_ROW_MISSING_FRACTION
    spline_knot_count: int = DEFAULT_SPLINE_KNOT_COUNT


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
        categorical_predictor_name: Retained categorical landform column.
    """

    table: pd.DataFrame
    block_summary: pd.DataFrame
    predictor_coverage: pd.DataFrame
    retained_predictor_names: tuple[str, ...]
    excluded_predictor_names: tuple[str, ...]
    continuous_predictor_names: tuple[str, ...]
    categorical_predictor_name: str


def infer_ecoregion_name(source_path: Path) -> str:
    """Infer a readable ecoregion label from a pipeline filename.

    Earth Engine GeoTIFF exports append an ecoregion identifier and response
    suffix. Spatial sample Parquet files append ``_spatial_sample``. Both
    conventions are removed before the remaining slug is formatted for
    reports and figures.

    Args:
        source_path: GeoTIFF or Parquet path whose stem identifies an ecoregion.

    Returns:
        Human-readable ecoregion label.
    """

    name_stem = re.sub(
        r"_e\d+(?:_response_variables.*)?$",
        "",
        source_path.stem,
        flags=re.IGNORECASE,
    )
    name_stem = re.sub(
        r"_response_variables.*$|_spatial_sample$",
        "",
        name_stem,
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
    # The empirical quantile is the first value whose cumulative represented
    # area reaches the requested share of the ecoregion.
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
) -> PreparedReferenceConditionData:
    """Select environmental predictors and prepare spatial validation rows.

    Predictor retention is based on represented-area coverage across the
    ecoregion sample. Row usability is then based on the missing fraction among
    retained predictors. Reference-site labels are preserved without deriving
    or changing classes.

    Args:
        sample_table: Table produced by ``load_ecoregion_geotiff.py``.
        configuration: Coverage, missingness, and spatial-fold settings.

    Returns:
        Prepared rows, folds, coverage diagnostics, and predictor names.

    Raises:
        ValueError: If categorical landform does not meet the configured
            predictor coverage threshold.
    """

    predictor_band_numbers: dict[str, int] = {}
    for column_name in sample_table.columns:
        match = ENVIRONMENTAL_BAND_PATTERN.match(column_name)
        if match:
            predictor_band_numbers[column_name] = int(match.group(1))
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
                "retained": (area_coverage >= configuration.minimum_predictor_coverage),
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
    configuration: ReferenceConditionConfiguration,
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
