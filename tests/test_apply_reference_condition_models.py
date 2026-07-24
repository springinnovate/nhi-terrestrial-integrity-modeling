"""Tests for windowed reference-condition raster inference."""

from __future__ import annotations

import contextlib
import io
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin

from scripts.apply_reference_condition_models import (
    FLOAT_NODATA,
    STATUS_INSUFFICIENT_PREDICTORS,
    STATUS_NODATA,
    STATUS_OUTSIDE_TARGET,
    STATUS_PREDICTED,
    load_reference_departure_calibration,
    load_response_models,
    run_reference_condition_inference,
)
from scripts.fit_grassland_integrity_parameters import (
    IntegrityConfiguration,
    fit_response_gam,
    predict_expected_response,
)


class ApplyReferenceConditionModelsTest(unittest.TestCase):
    """Verify aligned outputs, calculations, missingness, and masking."""

    def setUp(self) -> None:
        """Create fitted response artifacts and a compact source raster."""

        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.temporary_path = Path(self.temporary_directory.name)
        self.model_run_directory = self.temporary_path / "model_run"
        self.model_directory = self.model_run_directory / "models"
        self.model_directory.mkdir(parents=True)
        self.predictor_names = (
            "y2018_d20_temperature",
            "y2018_d21_precipitation",
            "y2018_d22_soil",
            "y2018_d23_topography",
            "y2018_d35_landform",
        )
        self.response_names = (
            "y2018_d02_response",
            "y2018_d11_response",
        )
        self.response_rmse = {"d02": 2.0, "d11": 4.0}
        self.models = self._create_models()
        self.raster_path = self.temporary_path / "synthetic_ecoregion.tif"
        self.transform = from_origin(-110.0, 45.0, 0.01, 0.01)
        self.source_values = self._create_raster_stack()
        (self.model_run_directory / "run_metadata.json").write_text(
            json.dumps(
                {
                    "ecoregion_name": "Synthetic Prairie",
                    "configuration": {"maximum_row_missing_fraction": 0.20},
                }
            ),
            encoding="utf-8",
        )
        self._create_reference_prediction_table()

    def _create_reference_prediction_table(self) -> None:
        """Write a calibration table with an exact weighted reference CDF."""

        pd.DataFrame(
            {
                "reference_site": [1, 1, 1, 1, 1, 0],
                "area_weight_m2": [1_000, 1_000, 1_000, 1_000, 2_000, 1_000_000],
                "d02_standardized_deviation_oof": [-1, -1, 1, 1, 0, 100],
                "d11_standardized_deviation_oof": [-1, 1, -1, 1, 0, 100],
            }
        ).to_parquet(
            self.model_run_directory / "ecological_response_predictions.parquet",
            compression="zstd",
            index=False,
        )

    def _create_models(self) -> dict[str, dict[str, object]]:
        """Fit two response models sharing five raster predictors.

        Returns:
            Model bundles keyed by short response band.
        """

        row_count = 60
        row_offsets = np.arange(row_count, dtype=np.float64)
        training_table = pd.DataFrame(
            {
                self.predictor_names[0]: 2.0 + row_offsets * 0.08,
                self.predictor_names[1]: 10.0 + row_offsets * 0.12,
                self.predictor_names[2]: 1.0 + np.sin(row_offsets / 7.0),
                self.predictor_names[3]: 3.0 + np.cos(row_offsets / 9.0),
                self.predictor_names[4]: (row_offsets.astype(np.int64) % 3) + 1,
                "area_weight_m2": 1_000.0 + row_offsets * 10.0,
            }
        )
        environmental_signal = (
            0.6 * training_table[self.predictor_names[0]]
            + 0.2 * training_table[self.predictor_names[1]]
            - 0.4 * training_table[self.predictor_names[2]]
            + 0.3 * training_table[self.predictor_names[3]]
            + 0.5 * training_table[self.predictor_names[4]]
        )
        training_table[self.response_names[0]] = 5.0 + environmental_signal
        training_table[self.response_names[1]] = 20.0 + 2.0 * environmental_signal
        configuration = IntegrityConfiguration(spline_knot_count=4, ridge_alpha=0.1)
        models = {}
        for response_name, response_band in zip(
            self.response_names,
            ("d02", "d11"),
            strict=True,
        ):
            model = fit_response_gam(
                training_table,
                response_name,
                self.predictor_names[:4],
                self.predictor_names[4],
                {
                    predictor_name: float(training_table[predictor_name].median())
                    for predictor_name in self.predictor_names
                },
                configuration,
            )
            model["reference_residual_rmse_oof"] = self.response_rmse[response_band]
            model["standardized_deviation_interpretation"] = (
                "observed minus expected divided by cross-validated reference RMSE"
            )
            joblib.dump(
                model,
                self.model_directory
                / f"{response_band}_reference_condition_gam.joblib",
            )
            models[response_band] = model
        return models

    def _create_raster_stack(self) -> np.ndarray:
        """Write observed responses and predictors with controlled gaps.

        Returns:
            Source values in raster-band, row, and column order.
        """

        height = 4
        width = 5
        row_grid, column_grid = np.indices((height, width), dtype=np.float32)
        predictor_values = np.stack(
            [
                2.5 + row_grid * 0.3 + column_grid * 0.1,
                11.0 + row_grid * 0.2 + column_grid * 0.15,
                1.2 + row_grid * 0.05,
                3.4 + column_grid * 0.07,
                ((row_grid + column_grid) % 3) + 1,
            ]
        ).astype(np.float32)
        predictor_table = pd.DataFrame(
            predictor_values.reshape(len(self.predictor_names), -1).T,
            columns=self.predictor_names,
        )
        expected_d02 = predict_expected_response(
            self.models["d02"],
            predictor_table,
        ).reshape(height, width)
        expected_d11 = predict_expected_response(
            self.models["d11"],
            predictor_table,
        ).reshape(height, width)
        response_values = np.stack(
            [
                expected_d02 + 1.0,
                expected_d11 - 2.0,
            ]
        ).astype(np.float32)
        reference_values = np.zeros((1, height, width), dtype=np.float32)
        reference_values[0, 1, 1] = 1.0
        reference_values[0, 2, 2] = 1.0
        source_values = np.concatenate(
            [response_values, predictor_values, reference_values]
        )
        source_values[0, 0, 2] = FLOAT_NODATA
        source_values[3, 0, 1] = FLOAT_NODATA
        source_values[3:7, 0, 0] = FLOAT_NODATA
        source_values[2, 0, 0] = predictor_values[0, 0, 0]

        profile = {
            "driver": "GTiff",
            "width": width,
            "height": height,
            "count": len(source_values),
            "dtype": "float32",
            "crs": "EPSG:4326",
            "transform": self.transform,
            "nodata": FLOAT_NODATA,
            "tiled": True,
            "blockxsize": 16,
            "blockysize": 16,
        }
        with rasterio.open(self.raster_path, "w", **profile) as destination:
            destination.write(source_values)
            for band_index, band_name in enumerate(
                (
                    *self.response_names,
                    *self.predictor_names,
                    "y2018_d01_grassland_reference_sites",
                ),
                start=1,
            ):
                destination.set_band_description(band_index, band_name)
        return source_values

    def test_calibrates_weighted_reference_distance_and_percentile(self) -> None:
        """Use only weighted reference rows for covariance and empirical CDF."""

        _, response_models, _ = load_response_models(self.model_run_directory)
        calibration = load_reference_departure_calibration(
            self.model_run_directory,
            response_models,
            covariance_shrinkage=0.10,
        )

        np.testing.assert_allclose(calibration.mean_vector, [0.0, 0.0])
        np.testing.assert_allclose(
            calibration.covariance_matrix,
            np.diag([2.0 / 3.0, 2.0 / 3.0]),
        )
        distances = calibration.calculate_distances(
            np.array([[0.5, -0.5]], dtype=np.float64)
        )
        np.testing.assert_allclose(distances, [math.sqrt(0.75)])
        np.testing.assert_allclose(
            calibration.calculate_percentiles(distances),
            [1.0 / 3.0],
        )
        self.assertEqual(5, calibration.reference_rows)
        self.assertEqual(5, calibration.complete_reference_rows)
        self.assertEqual(6_000.0, calibration.complete_reference_area_m2)

    def test_writes_aligned_response_stacks_and_streaming_report(self) -> None:
        """Calculate expected, raw, and standardized values in raster windows."""

        output_directory = self.temporary_path / "unmasked_output"
        standard_output = io.StringIO()
        with (
            patch(
                "scripts.apply_reference_condition_models."
                "MAXIMUM_DISPLAY_DIMENSION",
                3,
            ),
            contextlib.redirect_stdout(standard_output),
        ):
            summary = run_reference_condition_inference(
                self.raster_path,
                self.model_run_directory,
                output_directory=output_directory,
                window_size_pixels=2,
                show_progress=False,
            )

        self.assertEqual(2, summary.response_count)
        self.assertEqual(20, summary.raster_pixels)
        self.assertEqual(20, summary.target_pixels)
        self.assertEqual(19, summary.predicted_pixels)
        self.assertEqual(1, summary.insufficient_predictor_pixels)
        self.assertEqual(1, summary.imputed_pixels)
        self.assertEqual(
            "synthetic_prairie_expected_reference.tif",
            summary.expected_reference_path.name,
        )
        self.assertEqual(
            "synthetic_prairie_observed_minus_expected.tif",
            summary.observed_minus_expected_path.name,
        )
        self.assertEqual(
            "synthetic_prairie_standardized_deviation.tif",
            summary.standardized_deviation_path.name,
        )
        self.assertEqual(
            "synthetic_prairie_reference_departure_percentile.tif",
            summary.departure_percentile_path.name,
        )
        self.assertEqual(
            "synthetic_prairie_inference_status.tif",
            summary.inference_status_path.name,
        )
        self.assertEqual(
            "synthetic_prairie_aggregate_standardized_deviation.png",
            summary.aggregate_deviation_figure_path.name,
        )
        self.assertEqual(
            "synthetic_prairie_reference_departure_percentile.png",
            summary.departure_percentile_figure_path.name,
        )
        self.assertGreater(
            summary.aggregate_deviation_figure_path.stat().st_size,
            1_000,
        )
        self.assertGreater(
            summary.departure_percentile_figure_path.stat().st_size,
            1_000,
        )
        with rasterio.open(summary.expected_reference_path) as expected_source:
            expected = expected_source.read(masked=True)
            self.assertEqual(self.transform, expected_source.transform)
            self.assertEqual("EPSG:4326", str(expected_source.crs))
            self.assertEqual(
                ("d02_expected_reference", "d11_expected_reference"),
                expected_source.descriptions,
            )
            self.assertEqual("d02", expected_source.tags(1)["response_band"])
        with rasterio.open(summary.observed_minus_expected_path) as deviation_source:
            deviations = deviation_source.read(masked=True)
        with rasterio.open(summary.standardized_deviation_path) as standardized_source:
            standardized = standardized_source.read(masked=True)
        with rasterio.open(summary.departure_percentile_path) as percentile_source:
            percentiles = percentile_source.read(1, masked=True)
            self.assertEqual(self.transform, percentile_source.transform)
            self.assertEqual(
                "reference_departure_percentile",
                percentile_source.descriptions[0],
            )
            self.assertEqual(
                "reference_condition_departure_percentile",
                percentile_source.tags()["artifact_type"],
            )
        with rasterio.open(summary.inference_status_path) as status_source:
            status = status_source.read()

        row = 2
        column = 3
        predictor_table = pd.DataFrame(
            [
                self.source_values[
                    2 : 2 + len(self.predictor_names),
                    row,
                    column,
                ]
            ],
            columns=self.predictor_names,
        )
        expected_d02 = predict_expected_response(
            self.models["d02"],
            predictor_table,
        )[0]
        self.assertAlmostEqual(expected_d02, float(expected[0, row, column]), places=5)
        self.assertAlmostEqual(1.0, float(deviations[0, row, column]), places=5)
        self.assertAlmostEqual(0.5, float(standardized[0, row, column]), places=5)
        self.assertAlmostEqual(-2.0, float(deviations[1, row, column]), places=5)
        self.assertAlmostEqual(-0.5, float(standardized[1, row, column]), places=5)
        self.assertAlmostEqual(1.0 / 3.0, float(percentiles[row, column]), places=5)
        self.assertTrue(bool(percentiles.mask[1, 1]))
        self.assertTrue(bool(percentiles.mask[2, 2]))
        self.assertTrue(bool(percentiles.mask[0, 2]))

        self.assertEqual(STATUS_INSUFFICIENT_PREDICTORS, status[0, 0, 0])
        self.assertEqual(4, status[1, 0, 0])
        self.assertEqual(STATUS_PREDICTED, status[0, 0, 1])
        self.assertEqual(1, status[1, 0, 1])
        self.assertFalse(bool(expected.mask[0, 0, 2]))
        self.assertTrue(bool(deviations.mask[0, 0, 2]))

        report = summary.report_path.read_text(encoding="utf-8")
        metadata = json.loads(summary.metadata_path.read_text(encoding="utf-8"))
        self.assertIn("No grassland mask was supplied", report)
        self.assertIn("Synthetic Prairie", report)
        self.assertIn("mean pixel-level `sum(abs(z_j))`", report)
        self.assertIn("fixed linear scale", report)
        self.assertIn("Multivariate reference-departure percentile", report)
        self.assertIn("farther from the reference center than 95%", report)
        self.assertIsNone(metadata["grassland_mask"])
        self.assertEqual(18, metadata["responses"][0]["statistics"]["deviation_pixels"])
        self.assertEqual(16, summary.departure_percentile_pixels)
        self.assertEqual(16, metadata["coverage"]["departure_percentile_pixels"])
        self.assertEqual(
            ["d02", "d11"],
            metadata["reference_departure_calibration"]["response_bands"],
        )
        self.assertEqual(
            5,
            metadata["reference_departure_calibration"][
                "complete_reference_rows"
            ],
        )
        self.assertEqual(
            1.0 / 3.0,
            metadata["reference_departure_percentile"]["statistics"]["mean"],
        )
        self.assertEqual(
            1.0,
            metadata["reference_departure_percentile"]["figure"][
                "color_scale_upper_value"
            ],
        )
        self.assertEqual(
            "#5E2B97",
            metadata["reference_departure_percentile"]["figure"][
                "reference_color"
            ],
        )
        self.assertEqual(
            "#FFFFFF",
            metadata["reference_departure_percentile"]["figure"][
                "reference_outline_color"
            ],
        )
        self.assertEqual(
            0.4,
            metadata["reference_departure_percentile"]["figure"][
                "reference_outline_width_points"
            ],
        )
        self.assertEqual(
            16,
            metadata["aggregate_deviation_figure"]["contributing_source_pixels"],
        )
        self.assertEqual(
            2,
            metadata["aggregate_deviation_figure"]["reference_source_pixels"],
        )
        self.assertEqual(
            2,
            metadata["aggregate_deviation_figure"]["response_count"],
        )
        self.assertEqual(3, metadata["aggregate_deviation_figure"]["display_width"])
        self.assertEqual(2, metadata["aggregate_deviation_figure"]["display_height"])
        self.assertEqual(
            10.0,
            metadata["aggregate_deviation_figure"]["color_scale_upper_value"],
        )
        self.assertEqual(
            "linear over the fixed 0 to 10 range",
            metadata["aggregate_deviation_figure"]["color_normalization"],
        )
        self.assertEqual(
            3.0,
            metadata["aggregate_deviation_figure"]["yellow_green_anchor_value"],
        )
        self.assertEqual(
            0.3,
            metadata["aggregate_deviation_figure"][
                "yellow_green_anchor_normalized_position"
            ],
        )
        self.assertIn(
            "Reference-condition raster inference",
            standard_output.getvalue(),
        )

    def test_limits_inference_to_an_aligned_nonzero_mask(self) -> None:
        """Leave mask-zero pixels outside the inference target."""

        mask_path = self.temporary_path / "grassland_mask.tif"
        mask_values = np.ones((4, 5), dtype=np.uint8)
        mask_values[2, 3] = 0
        with rasterio.open(
            mask_path,
            "w",
            driver="GTiff",
            width=5,
            height=4,
            count=1,
            dtype="uint8",
            crs="EPSG:4326",
            transform=self.transform,
        ) as destination:
            destination.write(mask_values, 1)

        with contextlib.redirect_stdout(io.StringIO()):
            summary = run_reference_condition_inference(
                self.raster_path,
                self.model_run_directory,
                output_directory=self.temporary_path / "masked_output",
                grassland_mask_path=mask_path,
                window_size_pixels=3,
                show_progress=False,
            )

        self.assertEqual(19, summary.target_pixels)
        self.assertEqual(18, summary.predicted_pixels)
        with rasterio.open(summary.expected_reference_path) as expected_source:
            expected = expected_source.read(masked=True)
        with rasterio.open(summary.inference_status_path) as status_source:
            status = status_source.read()
        with rasterio.open(summary.departure_percentile_path) as percentile_source:
            percentiles = percentile_source.read(1, masked=True)
        self.assertTrue(bool(expected.mask[0, 2, 3]))
        self.assertTrue(bool(percentiles.mask[2, 3]))
        self.assertEqual(STATUS_OUTSIDE_TARGET, status[0, 2, 3])
        self.assertEqual(STATUS_NODATA, status[1, 2, 3])
        self.assertNotIn(
            "No grassland mask was supplied",
            summary.report_path.read_text(encoding="utf-8"),
        )


if __name__ == "__main__":
    unittest.main()
