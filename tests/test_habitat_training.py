"""Verify reusable reference-model training without acquisition or reporting."""

from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from scripts import habitat_training as training


def _prepare_sample() -> tuple[
    training.PreparedReferenceConditionData, training.IntegrityConfiguration
]:
    """Prepare six spatial blocks with named predictors and missing values.

    Returns:
        Prepared reference/background rows and settings for three validation
        folds. Two rows need one imputed predictor; row 2 is unusable because
        two of its three predictors are missing.
    """

    block = np.repeat(np.arange(6), 8)
    row = np.tile(np.arange(8), 6)
    elevation = block * 0.5 + row * 0.1
    wetness = block * 0.3 - row * 0.2
    land_cover = (block + row) % 3
    reference = row < 4
    table = pd.DataFrame(
        {
            "sampling_block_column": block * 2 + row % 2,
            "sampling_block_row": row % 2,
            "reference_site": reference.astype(np.int8),
            "area_weight_m2": 100.0 + row * 20.0,
            "elevation_m": elevation,
            "wetness": wetness,
            "land_cover": land_cover.astype(np.float64),
            "canopy_fraction": (
                0.2 * elevation + 0.3 * wetness + 0.15 * land_cover
                + 0.02 * np.cos(np.arange(len(row))) + 2.0 * ~reference
            ),
            "constant_response": 1.0,
        }
    )
    table.loc[0, "elevation_m"] = np.nan
    table.loc[1, "land_cover"] = np.nan
    table.loc[2, ["elevation_m", "wetness"]] = np.nan
    configuration = training.IntegrityConfiguration(
        fold_count=3,
        sampling_block_size_meters=1_000,
        validation_block_size_meters=2_000,
        minimum_predictor_coverage=0.8,
        maximum_row_missing_fraction=0.34,
        minimum_response_coverage=0.5,
        spline_knot_count=4,
        ridge_alpha=1.0,
    )
    prepared = training.prepare_reference_condition_data(
        table,
        configuration,
        ("elevation_m", "wetness", "land_cover"),
        "land_cover",
    )
    return prepared, configuration


class HabitatTrainingTest(unittest.TestCase):
    """Exercise named layers, supervised fitting and dependency independence."""

    def test_fits_named_layers_and_reports_each_completed_fold(self) -> None:
        """Fit usable reference rows while predicting reference and background."""

        prepared, configuration = _prepare_sample()
        responses = ("canopy_fraction", "constant_response")
        coverage = training.summarize_response_coverage(
            prepared.table, responses, responses, configuration
        ).set_index("response")
        self.assertEqual("fit", coverage.loc["canopy_fraction", "status"])
        self.assertEqual(
            "no_reference_variation",
            coverage.loc["constant_response", "status"],
        )
        completed_folds: list[None] = []
        fitted = training.fit_response_with_spatial_validation(
            prepared,
            "canopy_fraction",
            configuration,
            on_fold_complete=lambda: completed_folds.append(None),
        )

        self.assertEqual(3, len(completed_folds))
        self.assertEqual(23, len(fitted.training_table))
        self.assertTrue(fitted.training_table["reference_site"].eq(1).all())
        self.assertEqual(
            ("elevation_m", "wetness"),
            fitted.model["continuous_predictor_names"],
        )
        self.assertEqual("land_cover", fitted.model["categorical_predictor_name"])
        usable_rows = prepared.table["usable_for_gam"].to_numpy()
        self.assertEqual(47, np.count_nonzero(usable_rows))
        for values in (
            fitted.out_of_fold_expected,
            fitted.observed_minus_expected,
            fitted.standardized_departure,
        ):
            np.testing.assert_array_equal(np.isfinite(values), usable_rows)

    def test_propagates_cancellation_from_the_progress_callback(self) -> None:
        """Stop fitting when the supervising caller cancels after a fold."""

        prepared, configuration = _prepare_sample()
        cancellation = InterruptedError("The caller cancelled this run")
        completed_folds: list[None] = []

        def cancel_after_fold() -> None:
            """Record the completed fold and interrupt the calculation.

            Raises:
                InterruptedError: The caller has cancelled the calculation.
            """

            completed_folds.append(None)
            raise cancellation

        with self.assertRaises(InterruptedError) as raised:
            training.fit_response_with_spatial_validation(
                prepared,
                "canopy_fraction",
                configuration,
                on_fold_complete=cancel_after_fold,
            )
        self.assertIs(cancellation, raised.exception)
        self.assertEqual(1, len(completed_folds))

    def test_calculates_without_workflow_or_raster_imports(self) -> None:
        """Prepare and fit in a fresh process without CLI or plotting modules."""

        script = f'''
import runpy
import sys
namespace = runpy.run_path({str(Path(__file__).resolve())!r})
prepared, configuration = namespace["_prepare_sample"]()
namespace["training"].fit_response_with_spatial_validation(
    prepared, "canopy_fraction", configuration)
for name in sys.modules:
    assert not (
        name.startswith("scripts.")
        and name not in {{"scripts.habitat_training", "scripts.habitat_condition"}}
    ), name
    assert name.split(".")[0] not in {{"matplotlib", "rasterio", "ee", "tomllib"}}, name
'''
        result = subprocess.run(
            [sys.executable, "-B", "-c", script],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(0, result.returncode, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
