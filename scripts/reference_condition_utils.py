"""Prepare CLI sample metadata and plot reference-condition sampling diagnostics."""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import PatchCollection
from matplotlib.patches import Patch, Rectangle
from pyproj import Transformer

from . import habitat_training
from .analysis_config import AnalysisConfiguration


FIGURE_DPI = 300
EQUAL_AREA_CRS = "EPSG:8857"
SPATIAL_FOLD_COLORS = tuple(plt.get_cmap("Set2").colors)


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


def prepare_reference_condition_data(
    sample_table: pd.DataFrame,
    configuration: habitat_training.ReferenceConditionConfiguration,
    analysis_configuration: AnalysisConfiguration,
) -> habitat_training.PreparedReferenceConditionData:
    """Prepare configured predictors and attach their CLI band-number metadata.

    Predictor retention is based on represented-area coverage across the
    ecoregion sample. Row usability is then based on the missing fraction among
    retained predictors. Reference-site labels are preserved without deriving
    or changing classes.

    Args:
        sample_table: Table produced by ``build_spatial_sample.py``.
        configuration: Coverage, missingness, and spatial-fold settings.
        analysis_configuration: Configured predictor roles and data types.

    Returns:
        Prepared rows, folds, coverage diagnostics, and predictor names.

    Raises:
        ValueError: If categorical landform does not meet the configured
            predictor coverage threshold.
    """

    predictor_columns_by_identifier = analysis_configuration.columns_with_role(
        sample_table.columns,
        "predictor",
    )
    predictor_names = tuple(predictor_columns_by_identifier.values())
    categorical_predictor_identifier = next(
        band.identifier
        for band in analysis_configuration.bands
        if band.role == "predictor" and band.data_type == "categorical"
    )
    categorical_predictor_name = predictor_columns_by_identifier[
        categorical_predictor_identifier
    ]
    predictor_band_numbers_by_name = {
        column_name: int(identifier[1:])
        for identifier, column_name in predictor_columns_by_identifier.items()
    }

    prepared = habitat_training.prepare_reference_condition_data(
        sample_table,
        configuration,
        predictor_names,
        categorical_predictor_name,
    )
    prepared.predictor_coverage.insert(
        1,
        "band_number",
        prepared.predictor_coverage["predictor"].map(predictor_band_numbers_by_name),
    )
    return prepared


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
    configuration: habitat_training.ReferenceConditionConfiguration,
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

    fold_colors = SPATIAL_FOLD_COLORS[: configuration.fold_count]
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
