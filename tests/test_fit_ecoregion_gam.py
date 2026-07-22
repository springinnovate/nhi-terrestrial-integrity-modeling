"""Tests for spatially validating an ecoregion additive model."""

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

from scripts.fit_ecoregion_gam import (
    GamConfiguration,
    _equal_area_sample_coordinates,
    assign_spatial_folds,
    calculate_imputation_values,
    infer_sample_ecoregion_name,
    prepare_gam_data,
    run_spatial_gam,
    weighted_quantiles,
)


class FitEcoregionGamTest(unittest.TestCase):
    """Verify fold assignment, missing values, fitting, and artifacts."""

    def setUp(self) -> None:
        """Create an isolated output directory and synthetic sample table."""

        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.temporary_path = Path(self.temporary_directory.name)
        self.configuration = GamConfiguration(
            fold_count=5,
            sampling_block_size_meters=25_000,
            validation_block_size_meters=100_000,
            minimum_predictor_coverage=0.80,
            maximum_row_missing_fraction=0.20,
            spline_knot_count=4,
            regularization_c=1.0,
        )

    def _create_sample_table(self) -> pd.DataFrame:
        """Create ten spatial blocks with both reference-site classes.

        Returns:
            Sample table with every 2018 environmental predictor d20-d39.
        """

        records = []
        random_generator = np.random.default_rng(19)
        for validation_block_column in range(10):
            for row_in_block in range(10):
                reference_site = int(row_in_block < 3)
                record = {
                    "longitude": -120.0 + validation_block_column,
                    "latitude": 35.0 + row_in_block / 10.0,
                    "sampling_block_column": (
                        validation_block_column * 4 + row_in_block % 4
                    ),
                    "sampling_block_row": row_in_block % 4,
                    "reference_site": reference_site,
                    "area_weight_m2": float(800_000 + 10_000 * row_in_block),
                }
                for band_number in range(20, 40):
                    predictor_name = f"y2018_d{band_number:02d}_predictor"
                    if band_number == 35:
                        predictor_value = float(
                            (validation_block_column + row_in_block) % 4
                        )
                    else:
                        predictor_value = (
                            band_number * 0.1
                            + validation_block_column * 0.2
                            + row_in_block * 0.04
                            + reference_site * 0.7
                            + random_generator.normal(0.0, 0.02)
                        )
                    record[predictor_name] = predictor_value
                records.append(record)
        sample_table = pd.DataFrame.from_records(records)
        # d39 has less than 80% represented-area coverage and should be removed.
        sample_table.loc[:29, "y2018_d39_predictor"] = np.nan
        # Four missing values among the 19 retained predictors exceed the 20%
        # row threshold, so this row must remain in output but not enter a fit.
        sample_table.loc[
            0, [f"y2018_d{band:02d}_predictor" for band in range(20, 24)]
        ] = np.nan
        sample_table.loc[1, "y2018_d20_predictor"] = np.nan
        return sample_table

    def test_calculates_empirical_area_weighted_quantiles(self) -> None:
        """Let a row representing most area determine the weighted median."""

        quantiles = weighted_quantiles(
            np.array([0.0, 10.0, 20.0]),
            np.array([8.0, 1.0, 1.0]),
            [0.5, 0.9, 0.95],
        )

        np.testing.assert_array_equal(quantiles, [0.0, 10.0, 20.0])

    def test_equal_area_footprint_keeps_north_above_south(self) -> None:
        """Keep geographic north upward in the spatial-fold figure."""

        sample_table = pd.DataFrame(
            {
                "longitude": [-111.0, -111.0, -110.0],
                "latitude": [44.0, 45.0, 44.0],
            }
        )

        x_kilometers, y_kilometers = _equal_area_sample_coordinates(sample_table)

        self.assertGreater(y_kilometers[1], y_kilometers[0])
        self.assertGreater(x_kilometers[2], x_kilometers[0])

    def test_infers_ecoregion_name_from_spatial_sample(self) -> None:
        """Turn the sample stem into a standalone figure label."""

        name = infer_sample_ecoregion_name(
            Path("montana_valley_and_foothill_spatial_sample.parquet")
        )

        self.assertEqual("Montana Valley and Foothill", name)

    def test_groups_four_by_four_sampling_blocks_into_validation_blocks(self) -> None:
        """Keep every grouped 100 km block wholly inside one fold."""

        sample_table = self._create_sample_table()
        assigned_table, block_summary = assign_spatial_folds(
            sample_table,
            self.configuration,
        )

        self.assertEqual(10, len(block_summary))
        self.assertEqual(set(range(10)), set(block_summary["validation_block_column"]))
        self.assertEqual(5, block_summary["spatial_fold"].nunique())
        folds_per_block = assigned_table.groupby("validation_block_id")[
            "spatial_fold"
        ].nunique()
        self.assertTrue((folds_per_block == 1).all())
        self.assertTrue(
            (
                block_summary.groupby("spatial_fold")[
                    "represented_reference_area_m2"
                ].sum()
                > 0
            ).all()
        )

    def test_selects_environmental_bands_and_tracks_missing_rows(self) -> None:
        """Remove low-coverage d39 and flag a row above the missing limit."""

        prepared = prepare_gam_data(
            self._create_sample_table(),
            self.configuration,
        )

        self.assertEqual(19, len(prepared.retained_predictor_names))
        self.assertEqual(("y2018_d39_predictor",), prepared.excluded_predictor_names)
        self.assertEqual("y2018_d35_predictor", prepared.categorical_predictor_name)
        self.assertEqual(4, prepared.table.loc[0, "imputed_predictor_count"])
        self.assertFalse(bool(prepared.table.loc[0, "usable_for_gam"]))
        self.assertTrue(bool(prepared.table.loc[1, "usable_for_gam"]))

    def test_learns_area_weighted_imputation_from_training_rows(self) -> None:
        """Use weighted continuous medians and weighted categorical modes."""

        training_table = pd.DataFrame(
            {
                "continuous": [1.0, 10.0, np.nan],
                "landform": [2.0, 1.0, np.nan],
                "area_weight_m2": [9.0, 1.0, 2.0],
            }
        )

        imputation_values = calculate_imputation_values(
            training_table,
            ("continuous",),
            "landform",
        )

        self.assertEqual(1.0, imputation_values["continuous"])
        self.assertEqual(2.0, imputation_values["landform"])

    def test_runs_cross_validation_and_writes_complete_artifacts(self) -> None:
        """Score each usable row once and persist model diagnostics and figures."""

        sample_path = self.temporary_path / "sample.parquet"
        output_directory = self.temporary_path / "gam"
        self._create_sample_table().to_parquet(
            sample_path,
            compression="zstd",
            index=False,
        )

        report = io.StringIO()
        with contextlib.redirect_stdout(report):
            summary = run_spatial_gam(
                sample_path,
                output_directory,
                self.configuration,
                show_progress=False,
            )

        scored_table = pd.read_parquet(summary.scored_sample_path)
        fold_metrics = pd.read_csv(summary.fold_metrics_path)
        metadata = json.loads(summary.metadata_path.read_text(encoding="utf-8"))
        fitted_model = joblib.load(summary.model_path)

        self.assertEqual(100, summary.sampled_rows)
        self.assertEqual(99, summary.usable_rows)
        self.assertEqual(10, summary.validation_blocks)
        self.assertEqual(5, len(fold_metrics))
        self.assertTrue(
            scored_table.loc[
                scored_table["usable_for_gam"],
                "oof_reference_score",
            ]
            .between(0.0, 1.0)
            .all()
        )
        self.assertTrue(
            scored_table.loc[
                ~scored_table["usable_for_gam"],
                "oof_reference_score",
            ]
            .isna()
            .all()
        )
        self.assertEqual(19, len(metadata["retained_predictors"]))
        self.assertEqual("Sample", metadata["ecoregion_name"])
        self.assertIn("relative similarity", metadata["model"]["score_interpretation"])
        self.assertEqual(4, len(summary.figure_paths))
        self.assertTrue(
            all(path.stat().st_size > 1_000 for path in summary.figure_paths)
        )
        self.assertEqual(18, len(fitted_model.continuous_predictor_names))
        self.assertIn("Out-of-fold performance", report.getvalue())
        self.assertIn("requiring fold-specific imputation", report.getvalue())


if __name__ == "__main__":
    unittest.main()
