"""Tests for spatially validated ecological-response additive models."""

from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from scripts.fit_grassland_integrity_parameters import (
    IntegrityConfiguration,
    calculate_regression_metrics,
    predict_expected_response,
    resolve_response_names,
    run_integrity_parameter_gams,
    summarize_response_coverage,
)
from scripts.reference_condition_utils import prepare_reference_condition_data


class FitGrasslandIntegrityParametersTest(unittest.TestCase):
    """Verify response screening, spatial fitting, and output artifacts."""

    def setUp(self) -> None:
        """Create an isolated output directory and compact test settings."""

        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.temporary_path = Path(self.temporary_directory.name)
        self.configuration = IntegrityConfiguration(
            fold_count=5,
            sampling_block_size_meters=25_000,
            validation_block_size_meters=100_000,
            minimum_predictor_coverage=0.80,
            maximum_row_missing_fraction=0.20,
            minimum_response_coverage=0.50,
            spline_knot_count=4,
            ridge_alpha=1.0,
        )

    def _create_sample_table(self) -> pd.DataFrame:
        """Create ten spatial blocks with reference and background responses."""

        records = []
        random_generator = np.random.default_rng(731)
        for validation_block_column in range(10):
            for row_in_block in range(12):
                reference_site = int(row_in_block < 5)
                record = {
                    "longitude": -115.0 + validation_block_column * 0.5,
                    "latitude": 42.0 + row_in_block * 0.04,
                    "sampling_block_column": (
                        validation_block_column * 4 + row_in_block % 4
                    ),
                    "sampling_block_row": row_in_block % 4,
                    "reference_site": reference_site,
                    "area_weight_m2": float(600_000 + row_in_block * 20_000),
                }
                for band_number in range(20, 40):
                    predictor_name = f"y2018_d{band_number:02d}_predictor"
                    if band_number == 35:
                        value = float((validation_block_column + row_in_block) % 4 + 11)
                    else:
                        value = (
                            band_number * 0.05
                            + validation_block_column * 0.07
                            + row_in_block * 0.025
                            + random_generator.normal(0.0, 0.01)
                        )
                    record[predictor_name] = value

                environmental_signal = (
                    0.8 * record["y2018_d20_predictor"]
                    - 0.3 * record["y2018_d24_predictor"]
                    + 0.05 * record["y2018_d35_predictor"]
                )
                background_shift = 4.0 if reference_site == 0 else 0.0
                for band_number in range(2, 20):
                    response_name = f"y2018_d{band_number:02d}_response"
                    if band_number in (5, 7):
                        value = np.nan
                    elif band_number == 17:
                        value = 1.0
                    else:
                        value = (
                            band_number
                            + environmental_signal * (0.3 + band_number / 100.0)
                            + background_shift
                            + random_generator.normal(0.0, 0.025)
                        )
                    record[response_name] = value
                records.append(record)
        return pd.DataFrame.from_records(records)

    def test_resolves_response_band_aliases_and_full_names(self) -> None:
        """Accept convenient dNN selectors while retaining raster column names."""

        sample_table = self._create_sample_table()

        resolved = resolve_response_names(
            sample_table.columns,
            ["d02", "11", "y2018_d18_response", "d02"],
        )

        self.assertEqual(
            (
                "y2018_d02_response",
                "y2018_d11_response",
                "y2018_d18_response",
            ),
            resolved,
        )

    def test_calculates_area_weighted_regression_metrics(self) -> None:
        """Give a high-area miss more influence than low-area observations."""

        metrics = calculate_regression_metrics(
            np.array([0.0, 1.0, 2.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([1.0, 1.0, 8.0]),
        )

        self.assertAlmostEqual(np.sqrt(3.2), metrics["weighted_rmse"])
        self.assertAlmostEqual(1.6, metrics["weighted_mae"])
        self.assertLess(metrics["weighted_r2"], 0.0)

    def test_marks_missing_and_constant_reference_responses_unfit(self) -> None:
        """Explain why unusable response bands are skipped."""

        sample_table = self._create_sample_table()
        prepared = prepare_reference_condition_data(sample_table, self.configuration)
        response_names = tuple(
            f"y2018_d{band_number:02d}_response" for band_number in range(2, 20)
        )

        coverage = summarize_response_coverage(
            prepared.table,
            response_names,
            response_names,
            self.configuration,
        ).set_index("response_band")

        self.assertEqual("no_reference_values", coverage.loc["d05", "status"])
        self.assertEqual("no_reference_values", coverage.loc["d07", "status"])
        self.assertEqual("no_reference_variation", coverage.loc["d17", "status"])
        self.assertEqual("fit", coverage.loc["d02", "status"])

    def test_runs_selected_response_models_and_writes_reports(self) -> None:
        """Persist reloadable models, held-out deviations, metrics, and figures."""

        sample_path = self.temporary_path / "example_spatial_sample.parquet"
        output_directory = self.temporary_path / "integrity_parameters"
        source_table = self._create_sample_table()
        source_table.to_parquet(sample_path, compression="zstd", index=False)

        report = io.StringIO()
        with contextlib.redirect_stdout(report):
            summary = run_integrity_parameter_gams(
                sample_path,
                output_directory,
                self.configuration,
                requested_responses=("d02", "d11", "d17"),
                show_progress=False,
                create_partial_figures=True,
            )

        predictions = pd.read_parquet(summary.predictions_path)
        response_coverage = pd.read_csv(summary.response_coverage_path).set_index(
            "response_band"
        )
        fold_metrics = pd.read_csv(summary.fold_metrics_path)
        response_metrics = pd.read_csv(summary.response_metrics_path)
        metadata = json.loads(summary.metadata_path.read_text(encoding="utf-8"))
        fitted_model = joblib.load(summary.model_paths[0])

        self.assertEqual(120, summary.sampled_rows)
        self.assertEqual(120, summary.usable_rows)
        self.assertEqual(2, summary.fitted_responses)
        self.assertEqual(10, len(fold_metrics))
        self.assertEqual({"d02", "d11"}, set(response_metrics["response_band"]))
        self.assertEqual(
            "no_reference_variation", response_coverage.loc["d17", "status"]
        )
        self.assertEqual(7, len(summary.figure_paths))
        self.assertTrue(
            all(
                path.exists() and path.stat().st_size > 1_000
                for path in summary.figure_paths
            )
        )
        self.assertTrue(
            predictions.loc[predictions["usable_for_gam"], "d02_expected_reference_oof"]
            .notna()
            .all()
        )
        self.assertGreater(
            predictions.loc[
                predictions["reference_site"].eq(0),
                "d02_observed_minus_expected_oof",
            ].median(),
            2.0,
        )
        reloaded_predictions = predict_expected_response(
            fitted_model,
            predictions.loc[:4],
        )
        self.assertEqual(5, len(reloaded_predictions))
        self.assertEqual(
            "one regularized additive ridge regression per response",
            metadata["model"]["family"],
        )
        self.assertFalse(metadata["model"]["human_impact_predictors"])
        self.assertIn("Grassland ecological-response GAM validation", report.getvalue())
        self.assertIn("Important scope limit", summary.report_path.read_text())


if __name__ == "__main__":
    unittest.main()
