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


if __name__ == "__main__":
    unittest.main()
