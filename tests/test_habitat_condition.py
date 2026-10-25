"""Check extracted calculations against the original numerical results."""

from pathlib import Path
import subprocess
import sys
import unittest

import numpy as np
import pandas as pd

from scripts import habitat_condition


class HabitatConditionTests(unittest.TestCase):
    """Preserve fitting, reference-distance and import boundaries."""

    def test_predictions_match_original_weighted_fit(self) -> None:
        """Preserve imputation, unseen categories, weighting and extrapolation."""
        x = np.arange(12, dtype=np.float64)
        y = np.array([2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11], dtype=np.float64)
        landform = np.array([11, 12, 13] * 4, dtype=np.float64)
        training = pd.DataFrame({
            "x": x,
            "y": y,
            "landform": landform,
            "response": 3 + 0.4 * x - 0.2 * y + 0.05 * x * x + 0.3 * (landform - 11),
            "area_weight_m2": [1, 2, 1, 3, 2, 1, 1, 2, 3, 1, 2, 1],
        })
        training.loc[4, "y"] = np.nan
        training.loc[7, "landform"] = np.nan
        fitted = habitat_condition.fit_response_model(
            training,
            "response",
            ("x", "y"),
            "landform",
            {"x": 5.5, "y": 6.5, "landform": 12.0},
            spline_knot_count=4,
            ridge_alpha=0.7,
        )
        assessment = pd.DataFrame({
            "x": [0., 5.5, 12., -2.],
            "y": [2., np.nan, 13., 0.],
            "landform": [11., 12., 99., np.nan],
        })

        # Recorded from the original CLI implementation at 5c35e814c17e3aea.
        np.testing.assert_allclose(
            habitat_condition.predict_expected_response(fitted, assessment),
            [
                3.7874698763762575,
                5.802965942646322,
                9.105196741770687,
                4.56106219606302,
            ],
            rtol=1e-10,
            atol=1e-10,
        )

    def test_weighted_distances_match_original_calibration(self) -> None:
        """Preserve covariance shrinkage and the weighted CDF, including ties."""
        distribution = habitat_condition.fit_reference_departure_distribution(
            np.array([[-1., -1.], [0., 0.], [1., 2.], [2., 1.], [1., 1.]]),
            np.array([1., 2., 1., 1., 3.]),
            covariance_shrinkage=0.2,
        )
        np.testing.assert_allclose(distribution.reference_mean_vector, [.625, .625])
        np.testing.assert_allclose(
            distribution.reference_covariance_matrix,
            [[.734375, .609375], [.609375, .734375]],
        )
        np.testing.assert_allclose(
            distribution.stabilized_reference_covariance_matrix,
            [[.734375, .4875], [.4875, .734375]],
        )
        distances = distribution.calculate_mahalanobis_distances(
            np.array([[.625, .625], [1., 1.], [0., 0.], [3., 3.]]),
        )
        np.testing.assert_allclose(
            distances,
            [0., .47976976575540103, .7996162762590016, 3.0385418497842065],
            rtol=1e-10,
            atol=1e-10,
        )
        np.testing.assert_allclose(
            distribution.calculate_reference_departure_percentiles(distances),
            [0., .375, .625, 1.],
        )
        tied_distribution = habitat_condition.fit_reference_departure_distribution(
            np.array([[-1.], [0.], [1.]]),
            np.array([1., 2., 1.]),
            covariance_shrinkage=0.2,
        )
        np.testing.assert_allclose(
            tied_distribution.calculate_reference_departure_percentiles(
                tied_distribution.calculate_mahalanobis_distances(
                    np.array([[0.], [1.]]),
                ),
            ),
            [.5, 1.],
        )

    def test_rejects_insufficient_or_constant_reference_departures(self) -> None:
        """Keep numerical fit failures local to the reference distribution."""
        for departures, message in [
            (np.array([[0., 1.], [1., 0.]]), "more complete reference rows"),
            (np.zeros((3, 1)), "positive finite reference variance"),
        ]:
            with (
                self.subTest(message=message),
                self.assertRaisesRegex(ValueError, message),
            ):
                habitat_condition.fit_reference_departure_distribution(
                    departures, np.ones(len(departures)), covariance_shrinkage=0.2,
                )

    def test_core_runs_without_workflow_or_raster_imports(self) -> None:
        """Use the scientific module in a fresh process without CLI dependencies."""
        script = '''
import sys
import numpy as np
import pandas as pd
from scripts import habitat_condition as model
table = pd.DataFrame({"x": [0., 1., 2., 3.], "category": [0., 1., 0., 1.],
                      "y": [1., 2., 4., 3.], "area_weight_m2": [1.] * 4})
fitted = model.fit_response_model(table, "y", ("x",), "category",
                                 {"x": 1., "category": 0.},
                                 spline_knot_count=3, ridge_alpha=1.)
model.predict_expected_response(fitted, table)
distribution = model.fit_reference_departure_distribution(
    np.array([[-1.], [0.], [1.]]), np.ones(3), .1)
distribution.calculate_reference_departure_percentiles(
    distribution.calculate_mahalanobis_distances(np.array([[0.]])))
model.predict_raster_window(
    table[["x", "category"]].to_numpy().T.reshape(2, 2, 2),
    table[["y"]].to_numpy().T.reshape(1, 2, 2),
    predictor_names=("x", "category"), response_names=("y",),
    response_models=(fitted,), reference_rmses=(1.,),
    reference_distribution=distribution,
    aoi_pixel_mask=np.ones((2, 2), dtype=bool),
    reference_pixel_mask=np.zeros((2, 2), dtype=bool),
    assessment_pixel_mask=None, maximum_predictor_missing_fraction=.5)
for name in sys.modules:
    assert not (name.startswith("scripts.") and name != "scripts.habitat_condition"), name
    assert name.split(".")[0] not in {"matplotlib", "rasterio", "ee", "tomllib"}, name
'''
        result = subprocess.run(
            [sys.executable, "-B", "-c", script],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(0, result.returncode, result.stdout + result.stderr)


class RasterPredictionTests(unittest.TestCase):
    """Preserve window masks, output precision and prediction failure handling."""

    def setUp(self) -> None:
        """Fit a response model and prepare pixels with controlled missing data."""
        x = np.arange(12, dtype=np.float64)
        training = pd.DataFrame({
            "x": x, "category": x % 2,
            "response": 10000 + 0.03125 * x + 0.3 * (x % 2),
            "area_weight_m2": np.ones(12),
        })
        self.model = habitat_condition.fit_response_model(
            training, "response", ("x",), "category", {"x": 5.5, "category": 0.},
            spline_knot_count=3, ridge_alpha=1.,
        )
        self.distribution = habitat_condition.fit_reference_departure_distribution(
            np.array([[-1.], [0.], [1.]]), np.array([1., 2., 1.]), .1,
        )
        self.predictors = np.stack((x, x % 2)).reshape(2, 3, 4)
        self.predictors[1, 0, 1] = np.nan
        self.predictors[:, 0, 2] = np.nan
        self.observed = np.full((1, 3, 4), 10000.5)
        self.observed[0, 1, 1] = np.nan
        self.aoi = np.ones((3, 4), dtype=bool)
        self.aoi[0, 3] = False
        self.references = np.zeros((3, 4), dtype=bool)
        self.references[1, 0] = True

    def _predict_window(
        self, assessment_pixel_mask: np.ndarray | None,
    ) -> habitat_condition.RasterWindowPrediction:
        """Calculate this fixture's outputs with the requested assessment mask.

        Args:
            assessment_pixel_mask: Optional pixels selected for prediction.

        Returns:
            Predicted responses, departures, percentiles and status arrays.
        """
        return habitat_condition.predict_raster_window(
            self.predictors, self.observed,
            predictor_names=("x", "category"), response_names=("response",),
            response_models=(self.model,), reference_rmses=(.3,),
            reference_distribution=self.distribution,
            aoi_pixel_mask=self.aoi, reference_pixel_mask=self.references,
            assessment_pixel_mask=assessment_pixel_mask,
            maximum_predictor_missing_fraction=.5,
        )

    def test_window_outputs_match_original_masks_and_float32_calculation(self) -> None:
        """Keep expected values, missingness and reference exclusions unchanged."""
        # Captured from calculate_inference_tile at a88198a using this fitted model.
        expected = np.array([
            [10000.0966796875, 10000.1025390625, -9999., -9999.],
            [10000.15234375, 10000.4462890625, 10000.1982421875, 10000.4921875],
            [10000.2421875, 10000.529296875, 10000.26953125, 10000.544921875],
        ], dtype=np.float32)[None]
        deviation = np.array([
            [.4033203125, .3974609375, -9999., -9999.],
            [.34765625, -9999., .3017578125, .0078125],
            [.2578125, -.029296875, .23046875, -.044921875],
        ], dtype=np.float32)[None]
        standardized = np.array([
            [1.3444010019302368, 1.3248697519302368, -9999., -9999.],
            [1.1588541269302368, -9999., 1.005859375, .02604166604578495],
            [.859375, -.09765625, .7682291865348816, -.1497395783662796],
        ], dtype=np.float32)[None]
        percentile = np.array([
            [1., 1., -9999., -9999.], [-9999., -9999., 1., .5], [.5, .5, .5, .5],
        ], dtype=np.float32)
        status = np.array([
            [[2, 2, 0, 255], [2, 2, 2, 2], [2, 2, 2, 2]],
            [[0, 1, 255, 255], [0, 0, 0, 0], [0, 0, 0, 0]],
        ], dtype=np.uint8)
        assessment = np.ones((3, 4), dtype=bool)
        assessment[1, 2] = False
        for mask in (None, assessment):
            with self.subTest(explicit_assessment_mask=mask is not None):
                if mask is not None:
                    for values in (expected, deviation, standardized):
                        values[0, 1, 2] = -9999.
                    percentile[1, 2] = -9999.
                    status[:, 1, 2] = [0, 255]
                    status[:, 0, 2] = [1, 2]
                result = self._predict_window(mask)
                for actual, wanted in (
                    (result.expected_reference, expected),
                    (result.observed_minus_expected, deviation),
                    (result.standardized_deviation, standardized),
                    (result.departure_percentile, percentile),
                    (result.inference_status, status),
                ):
                    self.assertEqual(wanted.dtype, actual.dtype)
                    np.testing.assert_array_equal(actual, wanted)

    def test_rejects_nonfinite_model_predictions(self) -> None:
        """Report the affected response when a fitted estimator returns NaN."""
        self.model["regressor"].intercept_ = np.nan
        with self.assertRaisesRegex(
            RuntimeError, "response produced a nonfinite prediction"
        ):
            self._predict_window(None)


if __name__ == "__main__":
    unittest.main()
