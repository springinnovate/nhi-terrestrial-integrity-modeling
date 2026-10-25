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
from tqdm.auto import tqdm

from . import habitat_condition, habitat_training
from .analysis_config import AnalysisConfiguration, load_analysis_configuration
from .reference_condition_utils import (
    FIGURE_DPI,
    SPATIAL_FOLD_COLORS,
    create_fold_map,
    prepare_reference_condition_data,
)


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
    parser.add_argument(
        "analysis_configuration",
        type=Path,
        help=(
            "Complete TOML analysis definition containing model settings, "
            "selected responses, and the raster-band contract."
        ),
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


def resolve_response_names(
    columns: Sequence[str],
    requested_responses: Sequence[str] | None,
    analysis_configuration: AnalysisConfiguration,
) -> tuple[str, ...]:
    """Resolve dNN aliases or full names to ordered response columns.

    Args:
        columns (Sequence[str]): Column names from the ecoregion sample table.
        requested_responses (Sequence[str] | None): Requested dNN aliases or
            complete response column names. ``None`` selects every response.
        analysis_configuration (AnalysisConfiguration): Analysis band-role
            contract.

    Returns:
        tuple[str, ...]: Unique response column names in requested order.

    Raises:
        ValueError: If any requested response is not a configured dNN alias or known
            response column.
    """

    response_columns = analysis_configuration.columns_with_role(columns, "response")
    if not requested_responses:
        return tuple(response_columns.values())

    column_lookup = {name.lower(): name for name in response_columns.values()}
    resolved_names = []
    for requested_response in requested_responses:
        normalized = requested_response.strip().lower()
        if normalized in column_lookup:
            response_name = column_lookup[normalized]
        else:
            band_match = re.fullmatch(r"d?(\d+)", normalized)
            if not band_match:
                raise ValueError(
                    f"Unknown response '{requested_response}'. Use a configured "
                    "dNN alias or complete response column name."
                )
            response_identifier = f"d{int(band_match.group(1)):02d}"
            if response_identifier not in response_columns:
                raise ValueError(
                    f"Response '{requested_response}' is not available in the "
                    "configured sample columns."
                )
            response_name = response_columns[response_identifier]
        if response_name not in resolved_names:
            resolved_names.append(response_name)
    return tuple(resolved_names)


def summarize_response_coverage(
    prepared_table: pd.DataFrame,
    response_names: Sequence[str],
    selected_response_names: Sequence[str],
    configuration: habitat_training.IntegrityConfiguration,
    analysis_configuration: AnalysisConfiguration,
) -> pd.DataFrame:
    """Screen configured responses and attach their CLI band labels.

    Args:
        prepared_table (pandas.DataFrame): Spatially folded sample with
            reference labels, area weights, and response columns.
        response_names (Sequence[str]): Configured response columns to assess.
        selected_response_names (Sequence[str]): Response columns requested for
            model fitting.
        configuration (habitat_training.IntegrityConfiguration): Coverage, spline, and fold
            requirements used to screen responses.
        analysis_configuration (AnalysisConfiguration): Response metadata.

    Returns:
        pandas.DataFrame: One screening record per response, including coverage,
        variation, fold support, selection state, and fit status.
    """

    coverage = habitat_training.summarize_response_coverage(
        prepared_table,
        response_names,
        selected_response_names,
        configuration,
    )
    definitions = [
        analysis_configuration.band_for_column(name)
        for name in coverage["response"]
    ]
    coverage.insert(
        1, "response_band", [definition.identifier for definition in definitions]
    )
    coverage.insert(
        2, "display_name", [definition.display_name for definition in definitions]
    )
    return coverage


def build_response_model_artifact(
    fitted_model: habitat_condition.FittedResponseModel,
    response_name: str,
    analysis_configuration: AnalysisConfiguration,
) -> dict[str, object]:
    """Attach configured response and stack metadata to a fitted numerical model.

    Args:
        fitted_model: Predictors and fitted estimators from the numerical core.
        response_name: Response column used to fit the model.
        analysis_configuration: CLI band labels and stack identity.

    Returns:
        The existing Joblib artifact fields, including fitted numerical estimators.
    """

    predictor_names = (
        *fitted_model["continuous_predictor_names"],
        fitted_model["categorical_predictor_name"],
    )
    response_definition = analysis_configuration.band_for_column(response_name)
    predictor_display_names = {
        predictor_name: analysis_configuration.band_for_column(
            predictor_name
        ).display_name
        for predictor_name in predictor_names
    }
    return {
        "artifact_type": "grassland_reference_condition_additive_model",
        "format_version": 1,
        "response": response_name,
        "response_band": response_definition.identifier,
        "display_name": response_definition.display_name,
        "predictor_display_names": predictor_display_names,
        "raster_stack_name": analysis_configuration.stack_name,
        "raster_stack_version": analysis_configuration.stack_version,
        "analysis_configuration_sha256": (
            analysis_configuration.configuration_sha256
        ),
        **fitted_model,
    }


def _response_output_columns(response_band: str) -> tuple[str, str, str]:
    """Return stable expected-condition output names for one response band.

    Args:
        response_band (str): Short response identifier such as ``d02``.

    Returns:
        tuple[str, str, str]: Expected-reference, observed-minus-expected, and
        standardized-deviation output column names, in that order.
    """

    return (
        f"{response_band}_expected_reference_oof",
        f"{response_band}_observed_minus_expected_oof",
        f"{response_band}_standardized_deviation_oof",
    )


def create_model_performance_figure(
    response_metrics: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    configuration: habitat_training.IntegrityConfiguration,
    ecoregion_name: str,
    output_path: Path,
) -> None:
    """Compare held-out fit and rank stability for candidate responses.

    Args:
        response_metrics (pandas.DataFrame): Aggregate held-out metrics for each
            fitted ecological response.
        fold_metrics (pandas.DataFrame): Per-fold held-out metrics for each
            fitted ecological response.
        configuration (habitat_training.IntegrityConfiguration): Spatial-fold settings used to
            label and color per-fold metrics.
        ecoregion_name (str): Human-readable ecoregion figure label.
        output_path (pathlib.Path): Destination path for the PNG figure.

    Returns:
        None: The completed figure is written to ``output_path``.
    """

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
    fold_colors = SPATIAL_FOLD_COLORS[: configuration.fold_count]
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
                response_fold_metrics = fold_metrics.loc[
                    fold_metrics["response_band"].eq(response_band)
                ].sort_values("spatial_fold")
                fold_values = response_fold_metrics[fold_column].to_numpy(
                    dtype=np.float64
                )
                fold_point_colors = [
                    fold_colors[int(spatial_fold) - 1]
                    for spatial_fold in response_fold_metrics["spatial_fold"]
                ]
                axis.scatter(
                    fold_values,
                    np.full(len(fold_values), y_position),
                    color=fold_point_colors,
                    s=25,
                    alpha=0.85,
                    zorder=2,
                )
            axis.scatter(
                ordered[overall_column],
                y_positions,
                color="#273238",
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
        figure.legend(
            handles=[
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    linestyle="none",
                    markerfacecolor=fold_colors[index],
                    markeredgecolor="none",
                    label=f"Fold {index + 1}",
                )
                for index in range(configuration.fold_count)
            ]
            + [
                plt.Line2D(
                    [0],
                    [0],
                    marker="D",
                    linestyle="none",
                    markerfacecolor="#273238",
                    markeredgecolor="none",
                    label="All held-out reference rows",
                ),
            ],
            title="Spatial fold excluded from training",
            loc="upper center",
            bbox_to_anchor=(0.5, 0.93),
            ncol=configuration.fold_count + 1,
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
            "Diamonds summarize all out-of-fold reference predictions; colored "
            "dots show how performance changes among spatial folds",
            ha="center",
            color="#4B5459",
        )
        figure.tight_layout(rect=(0.02, 0.03, 1.0, 0.87))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
        plt.close(figure)


def create_observed_expected_figure(
    scored_table: pd.DataFrame,
    response_metrics: pd.DataFrame,
    configuration: habitat_training.IntegrityConfiguration,
    ecoregion_name: str,
    output_path: Path,
) -> None:
    """Plot observed against out-of-fold expected reference responses.

    Args:
        scored_table (pandas.DataFrame): Sample rows containing observed and
            out-of-fold expected responses.
        response_metrics (pandas.DataFrame): Aggregate metrics and display
            metadata for each fitted response.
        configuration (habitat_training.IntegrityConfiguration): Spatial-fold settings used to
            label and color the cross-validation predictions.
        ecoregion_name (str): Human-readable ecoregion figure label.
        output_path (pathlib.Path): Destination path for the PNG figure.

    Returns:
        None: The completed figure is written to ``output_path``.
    """

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
    fold_colors = SPATIAL_FOLD_COLORS[: configuration.fold_count]
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
            valid,
            [
                metric_row.response,
                expected_column,
                "area_weight_m2",
                "spatial_fold",
            ],
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
        point_colors = [
            fold_colors[int(spatial_fold) - 1]
            for spatial_fold in plot_table["spatial_fold"]
        ]
        combined = np.concatenate([observed, expected])
        lower_limit, upper_limit = np.quantile(combined, [0.01, 0.99])
        if math.isclose(lower_limit, upper_limit):
            lower_limit -= 0.5
            upper_limit += 0.5
        axis.scatter(
            observed,
            expected,
            s=8,
            color=point_colors,
            alpha=0.55,
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
        f"Cross-validated predictions for reference observations\n"
        f"{ecoregion_name}",
        fontsize=17,
        weight="bold",
        y=0.995,
        linespacing=1.25,
    )
    figure.text(0.5, 0.006, "Observed response", ha="center", fontsize=11)
    figure.text(
        0.006,
        0.5,
        "Cross-validated expected reference response",
        va="center",
        rotation=90,
        fontsize=11,
    )
    figure.legend(
        handles=[
            plt.Line2D(
                [0],
                [0],
                marker="o",
                linestyle="none",
                markerfacecolor=fold_colors[index],
                markeredgecolor="none",
                markersize=6,
                label=f"Fold {index + 1}",
            )
            for index in range(configuration.fold_count)
        ],
        title="Spatial fold excluded from training",
        loc="upper center",
        bbox_to_anchor=(0.5, 0.93),
        ncol=configuration.fold_count,
        frameon=False,
    )
    figure.text(
        0.5,
        0.025,
        "Each point is predicted by a model trained without that observation's "
        "spatial fold.",
        ha="center",
        color="#4B5459",
    )
    figure.tight_layout(rect=(0.025, 0.055, 1.0, 0.90))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(figure)


def create_residual_summary_figure(
    scored_table: pd.DataFrame,
    response_metrics: pd.DataFrame,
    ecoregion_name: str,
    output_path: Path,
) -> None:
    """Compare area-weighted held-out reference residual distributions.

    Args:
        scored_table (pandas.DataFrame): Sample rows containing standardized
            out-of-fold response deviations and area weights.
        response_metrics (pandas.DataFrame): Aggregate metrics and display
            metadata for each fitted response.
        ecoregion_name (str): Human-readable ecoregion figure label.
        output_path (pathlib.Path): Destination path for the PNG figure.

    Returns:
        None: The completed figure is written to ``output_path``.
    """

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
            quantiles = habitat_training.weighted_quantiles(
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
    """Calculate pairwise area-weighted correlations of response deviations.

    Args:
        scored_table (pandas.DataFrame): Sample rows containing standardized
            out-of-fold deviations and represented-area weights.
        response_metrics (pandas.DataFrame): Fitted response bands that define
            the matrix rows and columns.

    Returns:
        pandas.DataFrame: Symmetric weighted-correlation matrix indexed by
            response band.
    """

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
            value = habitat_training.weighted_correlation(
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
    """Plot correlation among standardized observed-minus-expected responses.

    Args:
        deviation_correlation (pandas.DataFrame): Pairwise weighted-correlation
            matrix indexed and columned by response band.
        response_metrics (pandas.DataFrame): Response display names keyed by
            response band.
        ecoregion_name (str): Human-readable ecoregion figure label.
        output_path (pathlib.Path): Destination path for the PNG figure.

    Returns:
        None: The completed figure is written to ``output_path``.
    """

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
    """Plot one-predictor-at-a-time expected-condition model responses.

    Args:
        fitted_model (dict[str, object]): Model bundle returned by
            ``build_response_model_artifact``.
        reference_training_table (pandas.DataFrame): Reference rows used to
            establish predictor ranges and represented-area quantiles.
        ecoregion_name (str): Human-readable ecoregion figure label.
        output_path (pathlib.Path): Destination path for the PNG figure.

    Returns:
        None: The completed figure is written to ``output_path``.
    """

    continuous_predictor_names = tuple(
        fitted_model["continuous_predictor_names"]
    )
    categorical_predictor_name = str(fitted_model["categorical_predictor_name"])
    predictor_names = (*continuous_predictor_names, categorical_predictor_name)
    baseline_predictor_values = {
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
        display_name = fitted_model["predictor_display_names"][predictor_name]
        if predictor_name == categorical_predictor_name:
            category_values = np.sort(
                reference_training_table[predictor_name].dropna().unique()
            )
            response_table = pd.DataFrame(
                [baseline_predictor_values] * len(category_values),
                columns=predictor_names,
            )
            response_table[predictor_name] = category_values
            expected_response_values = habitat_condition.predict_expected_response(
                fitted_model,
                response_table,
            )
            axis.bar(
                np.arange(len(category_values)),
                expected_response_values,
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
            lower, upper = habitat_training.weighted_quantiles(
                reference_training_table[predictor_name].to_numpy(dtype=np.float64),
                reference_training_table["area_weight_m2"].to_numpy(dtype=np.float64),
                [0.05, 0.95],
            )
            if math.isclose(lower, upper):
                predictor_values = np.array([lower])
            else:
                predictor_values = np.linspace(lower, upper, 100)
            response_table = pd.DataFrame(
                [baseline_predictor_values] * len(predictor_values),
                columns=predictor_names,
            )
            response_table[predictor_name] = predictor_values
            expected_response_values = habitat_condition.predict_expected_response(
                fitted_model,
                response_table,
            )
            axis.plot(
                predictor_values,
                expected_response_values,
                color="#2A6F73",
                linewidth=2,
            )
            axis.axvline(
                baseline_predictor_values[predictor_name],
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


def write_model_selection_report(
    output_path: Path,
    ecoregion_name: str,
    response_coverage: pd.DataFrame,
    response_metrics: pd.DataFrame,
    retained_predictor_count: int,
    excluded_predictor_count: int,
    diagnostic_figure_names: Sequence[str],
) -> None:
    """Write a standalone Markdown guide to response-model diagnostics.

    Args:
        output_path (pathlib.Path): Destination path for the Markdown report.
        ecoregion_name (str): Human-readable ecoregion report label.
        response_coverage (pandas.DataFrame): Screening status and reference
            coverage for every ecological response.
        response_metrics (pandas.DataFrame): Aggregate held-out metrics for
            fitted responses.
        retained_predictor_count (int): Number of environmental predictors
            retained after coverage screening.
        excluded_predictor_count (int): Number of environmental predictors
            excluded for insufficient coverage.
        diagnostic_figure_names (Sequence[str]): Ecoregion-specific filenames
            for the top-level diagnostic figures.

    Returns:
        None: The completed Markdown report is written to ``output_path``.
    """

    metrics_by_band = response_metrics.set_index("response_band")
    lines = [
        f"# Ecological-response GAM report: {ecoregion_name}",
        "",
        "## What these models estimate",
        "",
        (
            "Each fitted GAM learns the expected ecological response at supplied "
            "reference sites from the configured environmental predictors. The models "
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
            r2_metric = float(metric["overall_weighted_r2"])
            rank_metric = float(metric["overall_weighted_spearman"])
            rmse_metric = float(metric["overall_weighted_rmse"])
            fold_minimum = float(metric["fold_weighted_r2_minimum"])
            fold_maximum = float(metric["fold_weighted_r2_maximum"])
            r2_value = f"{r2_metric:.3f}" if np.isfinite(r2_metric) else "NA"
            rank_value = f"{rank_metric:.3f}" if np.isfinite(rank_metric) else "NA"
            rmse_value = f"{rmse_metric:.3f}" if np.isfinite(rmse_metric) else "NA"
            fold_minimum_value = (
                f"{fold_minimum:.3f}" if np.isfinite(fold_minimum) else "NA"
            )
            fold_maximum_value = (
                f"{fold_maximum:.3f}" if np.isfinite(fold_maximum) else "NA"
            )
            fold_range = f"{fold_minimum_value} to {fold_maximum_value}"
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
                "response GAMs themselves use only predictors assigned that role in "
                "the raster-stack TOML configuration."
            ),
            "",
            "## Figures",
            "",
            *[
                f"- `figures/{figure_name}`"
                for figure_name in diagnostic_figure_names
            ],
            "- `figures/partial_responses/` (unless disabled)",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def run_integrity_parameter_gams(
    sample_path: Path,
    output_directory: Path,
    configuration: habitat_training.IntegrityConfiguration,
    analysis_configuration: AnalysisConfiguration,
    requested_responses: Sequence[str] | None = None,
    show_progress: bool = True,
    create_partial_figures: bool = True,
) -> IntegrityRunSummary:
    """Fit, spatially validate, report, and serialize response GAMs.

    Args:
        sample_path (pathlib.Path): Spatial sample Parquet produced by the
            ecoregion GeoTIFF loader.
        output_directory (pathlib.Path): Directory for predictions, metrics,
            models, reports, and figures.
        configuration (habitat_training.IntegrityConfiguration): Predictor screening, spatial
            validation, response screening, spline, and ridge settings.
        analysis_configuration (AnalysisConfiguration): Configured response and
            predictor roles and display metadata.
        requested_responses (Sequence[str] | None): Optional dNN aliases or
            complete response column names. ``None`` fits every response that
            passes screening.
        show_progress (bool): Whether to display tqdm fitting and figure
            progress bars.
        create_partial_figures (bool): Whether to draw one partial-response
            figure for each fitted response.

    Returns:
        IntegrityRunSummary: Artifact paths, row and response counts, and total
        elapsed time for the completed run.

    Raises:
        ValueError: If none of the selected responses can be fit.
    """

    started = time.perf_counter()
    resolved_sample_path = sample_path.expanduser().resolve()
    resolved_output_directory = output_directory.expanduser().resolve()
    resolved_ecoregion_name = analysis_configuration.display_name
    print("Grassland ecological-response GAM validation")
    print(f"Input sample: {resolved_sample_path}")
    print(f"Output directory: {resolved_output_directory}")
    print(f"Ecoregion: {resolved_ecoregion_name}")
    print(f"Analysis configuration: {analysis_configuration.path}")
    sample_table = pd.read_parquet(resolved_sample_path)
    print(
        f"Loaded {len(sample_table):,} sampled rows x {sample_table.shape[1]:,} columns"
    )

    all_response_columns = analysis_configuration.columns_with_role(
        sample_table.columns,
        "response",
    )
    selected_response_names = resolve_response_names(
        sample_table.columns,
        requested_responses,
        analysis_configuration,
    )
    prepared_reference_data = prepare_reference_condition_data(
        sample_table,
        configuration,
        analysis_configuration,
    )
    response_coverage = summarize_response_coverage(
        prepared_reference_data.table,
        tuple(all_response_columns.values()),
        selected_response_names,
        configuration,
        analysis_configuration,
    )
    fitted_response_names = tuple(
        response_coverage.loc[response_coverage["status"].eq("fit"), "response"]
    )
    if not fitted_response_names:
        selected_response_statuses = response_coverage.loc[
            response_coverage["selected"], ["response_band", "status"]
        ]
        raise ValueError(
            "None of the selected responses can be fit: "
            + ", ".join(
                f"{row.response_band}={row.status}"
                for row in selected_response_statuses.itertuples(index=False)
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

    usable_row_mask = prepared_reference_data.table["usable_for_gam"]
    reference_row_mask = prepared_reference_data.table["reference_site"].eq(1)
    scored_table = prepared_reference_data.table.copy()
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
        response_definition = analysis_configuration.band_for_column(response_name)
        response_band = response_definition.identifier
        validated_response = habitat_training.fit_response_with_spatial_validation(
            prepared_reference_data,
            response_name,
            configuration,
            on_fold_complete=fit_progress.update,
        )
        fold_metric_records.extend(
            {
                "response": response_name,
                "response_band": response_band,
                "display_name": response_definition.display_name,
                **record,
            }
            for record in validated_response.fold_metrics.to_dict("records")
        )
        expected_column, deviation_column, standardized_column = (
            _response_output_columns(response_band)
        )
        scored_table[expected_column] = validated_response.out_of_fold_expected
        scored_table[deviation_column] = validated_response.observed_minus_expected
        scored_table[standardized_column] = validated_response.standardized_departure
        overall_metrics = validated_response.overall_metrics
        out_of_fold_reference_rmse = overall_metrics["weighted_rmse"]

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
            "display_name": response_definition.display_name,
            "reference_area_coverage": float(coverage_row["reference_area_coverage"]),
            "validation_reference_rows": len(validated_response.training_table),
            "validation_reference_area_m2": float(
                validated_response.training_table["area_weight_m2"].sum()
            ),
            "reference_residual_rmse_oof": out_of_fold_reference_rmse,
        }
        metric_record.update(
            {f"overall_{name}": value for name, value in overall_metrics.items()}
        )
        for metric_name in habitat_training.REGRESSION_METRIC_NAMES:
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

        final_training_table = validated_response.training_table
        final_model = build_response_model_artifact(
            validated_response.model,
            response_name,
            analysis_configuration,
        )
        final_model["reference_residual_rmse_oof"] = out_of_fold_reference_rmse
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
    prepared_reference_data.predictor_coverage.to_csv(
        predictor_coverage_path,
        index=False,
    )
    response_coverage.to_csv(response_coverage_path, index=False)
    fold_metrics.to_csv(fold_metrics_path, index=False)
    response_metrics.to_csv(response_metrics_path, index=False)
    deviation_correlation.to_csv(
        deviation_correlation_path, index_label="response_band"
    )

    ecoregion_slug = re.sub(
        r"[^a-z0-9]+", "_", resolved_ecoregion_name.lower()
    ).strip("_") or "ecoregion"
    base_figure_paths = (
        figure_directory / f"{ecoregion_slug}_spatial_folds.png",
        figure_directory / f"{ecoregion_slug}_response_model_performance.png",
        figure_directory / f"{ecoregion_slug}_observed_vs_expected.png",
        figure_directory
        / f"{ecoregion_slug}_reference_deviation_distributions.png",
        figure_directory
        / f"{ecoregion_slug}_response_deviation_correlation.png",
    )
    create_fold_map(
        prepared_reference_data.block_summary,
        prepared_reference_data.table,
        configuration,
        resolved_ecoregion_name,
        base_figure_paths[0],
    )
    create_model_performance_figure(
        response_metrics,
        fold_metrics,
        configuration,
        resolved_ecoregion_name,
        base_figure_paths[1],
    )
    create_observed_expected_figure(
        scored_table,
        response_metrics,
        configuration,
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
        len(prepared_reference_data.retained_predictor_names),
        len(prepared_reference_data.excluded_predictor_names),
        tuple(path.name for path in base_figure_paths),
    )
    run_metadata = {
        "input_sample": str(resolved_sample_path),
        "ecoregion_name": resolved_ecoregion_name,
        "configuration": asdict(configuration),
        "analysis_configuration": {
            "path": str(analysis_configuration.path),
            "analysis_name": analysis_configuration.analysis_name,
            "display_name": analysis_configuration.display_name,
            "aoi_path": str(analysis_configuration.aoi_path),
            "year": analysis_configuration.year,
            "stack_name": analysis_configuration.stack_name,
            "stack_version": analysis_configuration.stack_version,
            "sha256": analysis_configuration.configuration_sha256,
            "datasets": dict(analysis_configuration.datasets),
            "bands": [asdict(band) for band in analysis_configuration.bands],
        },
        "model": {
            "family": "one regularized additive ridge regression per response",
            "training_population": "usable supplied reference-site rows only",
            "continuous_terms": "independent cubic spline bases",
            "categorical_terms": "one-hot landform indicators",
            "interactions": False,
            "predictor_bands": "bands assigned the predictor role in TOML",
            "human_impact_predictors": False,
            "output_interpretation": (
                "expected reference condition and signed observed-minus-expected "
                "deviation; neither is a present-day integrity score"
            ),
        },
        "sampled_rows": int(len(prepared_reference_data.table)),
        "usable_rows": int(usable_row_mask.sum()),
        "reference_rows": int(reference_row_mask.sum()),
        "usable_reference_rows": int(
            np.count_nonzero(usable_row_mask & reference_row_mask)
        ),
        "validation_block_count": int(
            len(prepared_reference_data.block_summary)
        ),
        "selected_responses": list(selected_response_names),
        "fitted_responses": list(fitted_response_names),
        "retained_predictors": list(
            prepared_reference_data.retained_predictor_names
        ),
        "excluded_predictors": list(
            prepared_reference_data.excluded_predictor_names
        ),
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
    metadata_path.write_text(
        json.dumps(run_metadata, indent=2),
        encoding="utf-8",
    )

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
        sampled_rows=int(len(prepared_reference_data.table)),
        usable_rows=int(usable_row_mask.sum()),
        fitted_responses=len(fitted_response_names),
        elapsed_seconds=elapsed_seconds,
    )


def main() -> None:
    """Run the command-line ecological-response GAM workflow."""

    args = parse_args()
    analysis_configuration = load_analysis_configuration(
        args.analysis_configuration
    )
    integrity_configuration = habitat_training.IntegrityConfiguration(
        fold_count=analysis_configuration.model.fold_count,
        sampling_block_size_meters=(
            analysis_configuration.sampling.block_size_meters
        ),
        validation_block_size_meters=(
            analysis_configuration.model.validation_block_size_meters
        ),
        minimum_predictor_coverage=(
            analysis_configuration.model.minimum_predictor_coverage
        ),
        maximum_row_missing_fraction=(
            analysis_configuration.model.maximum_row_missing_fraction
        ),
        spline_knot_count=analysis_configuration.model.spline_knot_count,
        minimum_response_coverage=(
            analysis_configuration.model.minimum_response_coverage
        ),
        ridge_alpha=analysis_configuration.model.ridge_alpha,
    )
    output_directory = args.output_directory or (
        Path("outputs") / "integrity_parameters" / args.sample_parquet.stem
    )
    run_integrity_parameter_gams(
        args.sample_parquet,
        output_directory,
        integrity_configuration,
        analysis_configuration,
        requested_responses=analysis_configuration.model.responses or None,
        show_progress=not args.no_progress,
        create_partial_figures=not args.no_partial_response_figures,
    )


if __name__ == "__main__":
    main()
