"""Verify reusable reference-model training without acquisition or reporting."""

from __future__ import annotations

from contextlib import ExitStack, redirect_stdout
from dataclasses import fields, replace
import io
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

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


def _fitting_sample() -> tuple[pd.DataFrame, training.IntegrityConfiguration]:
    """Create two varying responses with different missing reference rows.

    Returns:
        Original sampled columns and scientific settings. The additional constant
        response cannot be fit and must remain in screening diagnostics only.
    """
    prepared, configuration = _prepare_sample()
    table = prepared.table[[
        "sampling_block_column", "sampling_block_row", "reference_site",
        "area_weight_m2", "elevation_m", "wetness", "land_cover",
        "canopy_fraction", "constant_response",
    ]].copy()
    table["productivity"] = (
        0.7 * table["canopy_fraction"] + 0.03 * np.sin(np.arange(len(table)))
    )
    table.loc[8, "canopy_fraction"] = np.nan
    table.loc[[16, 25], "productivity"] = np.nan
    return table, configuration


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

    def test_fits_named_responses_in_requested_order_with_progress(self) -> None:
        """Screen responses and calibrate only complete reference departures."""
        sample, configuration = _fitting_sample()
        original = sample.copy(deep=True)
        events: list[tuple[str, int, int]] = []

        def report(phase: str, completed: int, total: int) -> None:
            """Record progress while fitting the sample.

            Args:
                phase: Current preparation, fitting or calibration phase.
                completed: Completed steps in this phase.
                total: Total steps in this phase.
            """
            events.append((phase, completed, total))

        fitted = training.fit_habitat_condition(
            sample, configuration,
            predictor_names=("land_cover", "elevation_m", "wetness"),
            categorical_predictor_name="land_cover",
            response_names=("productivity", "constant_response", "canopy_fraction"),
            covariance_shrinkage=0.1, progress=report,
        )
        pd.testing.assert_frame_equal(sample, original)
        self.assertEqual(("elevation_m", "wetness", "land_cover"), fitted.predictor_names)
        self.assertEqual(("productivity", "canopy_fraction"), tuple(fitted.responses))
        self.assertEqual(
            "no_reference_variation",
            fitted.response_coverage.set_index("response").loc["constant_response", "status"],
        )
        for response in fitted.responses.values():
            self.assertTrue(response.training_table["reference_site"].eq(1).all())
        complete_reference = sample["reference_site"].eq(1).to_numpy() & np.all(
            np.isfinite(np.column_stack([
                response.standardized_departure for response in fitted.responses.values()
            ])), axis=1,
        )
        self.assertLess(complete_reference.sum(), sample["reference_site"].sum())
        self.assertEqual(
            complete_reference.sum(), len(fitted.reference_distribution.sorted_reference_distances),
        )
        total_fits = 2 * (configuration.fold_count + 1)
        self.assertEqual(events, [
            ("Preparing model samples", 0, 1), ("Preparing model samples", 1, 1),
            *[("Fitting response models", i, total_fits) for i in range(total_fits + 1)],
            ("Calibrating reference departures", 0, 1),
            ("Calibrating reference departures", 1, 1),
        ])

    def test_matches_cli_models_diagnostics_and_reference_calibration(self) -> None:
        """Fit the same table through the callable and the existing CLI workflow."""
        from scripts import fit_grassland_integrity_parameters as cli
        from scripts.analysis_config import load_analysis_configuration
        from scripts.apply_reference_condition_models import (
            build_reference_departure_calibration, load_response_models,
        )
        from scripts.habitat_condition import predict_expected_response

        sample, configuration = _fitting_sample()
        sample = sample.rename(columns={
            "elevation_m": "y2018_d20_predictor", "wetness": "y2018_d24_predictor",
            "land_cover": "y2018_d35_predictor", "canopy_fraction": "y2018_d02_response",
            "productivity": "y2018_d03_response", "constant_response": "y2018_d17_response",
        })
        predictors = ("y2018_d20_predictor", "y2018_d24_predictor", "y2018_d35_predictor")
        responses = ("y2018_d03_response", "y2018_d17_response", "y2018_d02_response")
        actual = training.fit_habitat_condition(
            sample, configuration, predictor_names=predictors,
            categorical_predictor_name=predictors[-1], response_names=responses,
            covariance_shrinkage=0.1,
        )
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            path = Path(directory)
            sample_path = path / "sample.parquet"
            sample.to_parquet(sample_path, index=False)
            for figure in (
                "create_fold_map", "create_model_performance_figure",
                "create_observed_expected_figure", "create_residual_summary_figure",
                "create_deviation_correlation_figure",
            ):
                stack.enter_context(patch.object(cli, figure))
            with redirect_stdout(io.StringIO()):
                run = cli.run_integrity_parameter_gams(
                    sample_path, path / "cli", configuration,
                    load_analysis_configuration(), requested_responses=responses,
                    show_progress=False, create_partial_figures=False,
                )
            predictions = pd.read_parquet(run.predictions_path)
            coverage = pd.read_csv(run.response_coverage_path).set_index("response")
            pd.testing.assert_frame_equal(
                actual.response_coverage.set_index("response").sort_index(),
                coverage[list(actual.response_coverage.columns[1:])].sort_index(),
            )
            pd.testing.assert_frame_equal(
                actual.prepared_data.table, predictions[list(actual.prepared_data.table.columns)],
            )
            metrics = pd.read_csv(run.response_metrics_path).set_index("response")
            folds = pd.read_csv(run.fold_metrics_path)
            _, models, _ = load_response_models(run.output_directory)
            by_name = {model.response_name: model for model in models}
            ordered_models = tuple(by_name[name] for name in actual.responses)
            expected_distribution = build_reference_departure_calibration(
                run.output_directory, ordered_models, covariance_shrinkage=0.1,
            ).distribution
            for name, fitted in actual.responses.items():
                band = by_name[name].response_band
                for suffix, values in (
                    ("expected_reference_oof", fitted.out_of_fold_expected),
                    ("observed_minus_expected_oof", fitted.observed_minus_expected),
                    ("standardized_deviation_oof", fitted.standardized_departure),
                ):
                    np.testing.assert_array_equal(values, predictions[f"{band}_{suffix}"])
                for metric, value in fitted.overall_metrics.items():
                    self.assertAlmostEqual(value, metrics.loc[name, f"overall_{metric}"])
                pd.testing.assert_frame_equal(
                    fitted.fold_metrics,
                    folds.loc[folds["response"].eq(name), list(fitted.fold_metrics.columns)].reset_index(drop=True),
                )
                np.testing.assert_allclose(
                    predict_expected_response(fitted.model, sample),
                    predict_expected_response(by_name[name].bundle, sample), rtol=0, atol=0,
                )
            for field in fields(expected_distribution):
                np.testing.assert_allclose(
                    getattr(actual.reference_distribution, field.name),
                    getattr(expected_distribution, field.name), rtol=1e-12, atol=1e-14,
                )

    def test_stops_after_a_cancelled_fold_and_reports_unfittable_responses(self) -> None:
        """Propagate caller cancellation and explain scientific screening failures."""
        sample, configuration = _fitting_sample()
        arguments = dict(
            predictor_names=("elevation_m", "wetness", "land_cover"),
            categorical_predictor_name="land_cover", covariance_shrinkage=0.1,
        )
        completed: list[int] = []

        def cancel(phase: str, count: int, total: int) -> None:
            """Cancel after the first validation fit, before later model work.

            Args:
                phase: Current preparation, fitting or calibration phase.
                count: Completed steps in this phase.
                total: Total steps in this phase.

            Raises:
                InterruptedError: After the first validation fit.
            """
            if phase == "Fitting response models" and count:
                completed.append(count)
                raise InterruptedError("cancel fitting")

        with self.assertRaisesRegex(InterruptedError, "cancel fitting"):
            training.fit_habitat_condition(
                sample, configuration, response_names=("canopy_fraction", "productivity"),
                progress=cancel, **arguments,
            )
        self.assertEqual([1], completed)
        with self.assertRaisesRegex(ValueError, "constant_response.*no_reference_variation"):
            training.fit_habitat_condition(
                sample, configuration, response_names=("constant_response",), **arguments,
            )

    def test_rejects_nonpositive_or_nonfinite_validation_rmse(self) -> None:
        """Prevent undefined standardized residuals from reaching calibration."""
        prepared, configuration = _prepare_sample()
        sample, _ = _fitting_sample()
        response = training.fit_response_with_spatial_validation(prepared, "canopy_fraction", configuration)
        for rmse in (0.0, np.nan):
            invalid = replace(response, overall_metrics={**response.overall_metrics, "weighted_rmse": rmse})
            with self.subTest(rmse=rmse), patch.object(
                training, "fit_response_with_spatial_validation", return_value=invalid,
            ), self.assertRaisesRegex(ValueError, "canopy_fraction.*RMSE"):
                training.fit_habitat_condition(
                    sample, configuration,
                    predictor_names=("elevation_m", "wetness", "land_cover"),
                    categorical_predictor_name="land_cover", response_names=("canopy_fraction",),
                    covariance_shrinkage=0.1,
                )

    def test_calculates_without_workflow_or_raster_imports(self) -> None:
        """Prepare and fit in a fresh process without CLI or plotting modules."""

        script = f'''
import runpy
import sys
namespace = runpy.run_path({str(Path(__file__).resolve())!r})
prepared, configuration = namespace["_prepare_sample"]()
namespace["training"].fit_response_with_spatial_validation(
    prepared, "canopy_fraction", configuration)
sample, configuration = namespace["_fitting_sample"]()
namespace["training"].fit_habitat_condition(
    sample, configuration,
    predictor_names=("elevation_m", "wetness", "land_cover"),
    categorical_predictor_name="land_cover",
    response_names=("canopy_fraction", "productivity"), covariance_shrinkage=0.1)
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
