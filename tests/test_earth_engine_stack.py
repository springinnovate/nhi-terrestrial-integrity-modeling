"""Build the configured scientific stack through the real Earth Engine client."""

from dataclasses import replace
from importlib.resources import files
import io
import json
import os
from pathlib import Path
import unittest
from typing import Any
from unittest.mock import patch

import ee
from googleapiclient.http import HttpMockSequence

from scripts.analysis_config import (
    load_analysis_configuration,
    load_raster_stack_configuration,
)
from scripts.earth_engine_stack import build_earth_engine_stack


class EarthEngineStackTest(unittest.TestCase):
    """Keep scientific image graphs independent of CLI and local-cache settings."""

    def setUp(self) -> None:
        """Initialize the official client with bundled metadata and no network access."""
        metadata = files(ee).joinpath("tests")
        discovery = metadata.joinpath("cloud_api_discovery_document.json").read_text(
            encoding="utf-8"
        )
        algorithms = metadata.joinpath("algorithms.json").read_text(encoding="utf-8")
        transport = HttpMockSequence(
            [
                ({"status": "200"}, discovery),
                ({"status": "200"}, discovery),
                ({"status": "200"}, algorithms),
            ]
        )
        ee.Reset()
        self.addCleanup(ee.Reset)
        with patch("urllib.request.urlopen", return_value=io.BytesIO(b"{}")):
            ee.Initialize(
                credentials=None, project="offline-stack-test", http_transport=transport
            )

    def test_scientific_configuration_preserves_the_stack_for_each_year(self) -> None:
        """Build identical band graphs from full and scientific-only configurations."""
        path = (
            Path(__file__).resolve().parents[1]
            / "config"
            / "south_africa_reference_condition_analysis.toml"
        )
        full = load_analysis_configuration(path)
        graphs: dict[int, str] = {}
        for requested_year in (None, 2019):
            with self.subTest(year=requested_year):
                scientific = load_raster_stack_configuration(path, year=requested_year)
                expected = build_earth_engine_stack(
                    replace(full, year=scientific.year)
                ).serialize()
                actual = build_earth_engine_stack(scientific).serialize()
                self.assertEqual(expected, actual)
                for band_name in scientific.band_names():
                    self.assertIn(json.dumps(band_name), actual)
                for dataset_id in scientific.datasets.values():
                    self.assertIn(json.dumps(dataset_id), actual)
                graphs[scientific.year] = actual
        self.assertNotEqual(graphs[2018], graphs[2019])

@unittest.skipUnless(
    os.environ.get("HABITAT_EARTH_ENGINE_PROJECT")
    and os.environ.get("HABITAT_EARTH_ENGINE_CREDENTIALS_FILE"),
    "Set HABITAT_EARTH_ENGINE_PROJECT and HABITAT_EARTH_ENGINE_CREDENTIALS_FILE "
    "to run live Earth Engine snow checks.",
)
class LiveSnowCalculationTest(unittest.TestCase):
    """Evaluate the production snow band with real Earth Engine reductions."""

    def setUp(self) -> None:
        """Initialize the service-account client and select only the snow band."""
        from google.oauth2.service_account import Credentials

        credentials = Credentials.from_service_account_file(
            os.environ["HABITAT_EARTH_ENGINE_CREDENTIALS_FILE"],
            scopes=["https://www.googleapis.com/auth/earthengine"],
        )
        ee.Reset()
        self.addCleanup(ee.Reset)
        ee.Initialize(credentials, project=os.environ["HABITAT_EARTH_ENGINE_PROJECT"])
        ee.data.setDeadline(360000)
        configuration = load_raster_stack_configuration(
            Path(__file__).resolve().parents[1]
            / "config"
            / "south_africa_reference_condition_analysis.toml",
            year=2018,
        )
        self.configuration = replace(
            configuration,
            bands=tuple(band for band in configuration.bands if band.identifier == "d39"),
        )

    def read_pixel(self, image: ee.Image) -> dict[str, float]:
        """Read one explicitly gridded snow-bearing Alaska pixel.

        Args:
            image: Named bands to evaluate, with masked pixels reported as -9999.

        Returns:
            Band names mapped to their scalar values at the requested pixel.

        Raises:
            ee.EEException: If source access or Earth Engine evaluation fails.
        """
        values = ee.data.computePixels({
            "expression": image.unmask(-9999),
            "fileFormat": "NUMPY_NDARRAY",
            "grid": {
                "dimensions": {"width": 1, "height": 1},
                "affineTransform": {
                    "scaleX": 0.01, "shearX": 0, "translateX": -147.75,
                    "shearY": 0, "scaleY": -0.01, "translateY": 64.8,
                },
                "crsCode": "EPSG:4326",
            },
        })
        return {name: float(values[name][0, 0]) for name in values.dtype.names}

    def test_available_observations_keep_their_weights_and_masks(self) -> None:
        """Check missing months, unequal sampling, nonpositive snow and empty years."""
        cases = {
            "unequal_counts_and_gaps": (
                [("2018-03-01", 1, 1), ("2018-03-02", 3, 1), ("2018-12-01", 8, 1)], 4,
            ),
            "single_month": ([("2018-07-01", 7, 1)], 7),
            "complete_year": (
                [(f"2018-{month:02d}-01", month, 1) for month in range(1, 13)], 6.5,
            ),
            "nonpositive_and_masked": (
                [("2018-03-01", 2, 1), ("2018-03-02", 0, 1),
                 ("2018-03-03", -2, 1), ("2018-03-04", 100, 0),
                 ("2018-09-01", 6, 1)], 4,
            ),
            "empty_year": ([], -9999),
            "no_positive_snow": (
                [("2018-05-01", 0, 1), ("2018-05-02", -1, 1)], -9999,
            ),
            "all_masked": ([("2018-06-01", 5, 0)], -9999),
            "outside_year_excluded": (
                [("2017-12-31", 100, 1), ("2018-04-01", 3, 1),
                 ("2019-01-01", 200, 1)], 3,
            ),
        }
        initialize_collection = ee.ImageCollection.__init__
        bands: list[ee.Image] = []
        for name, (observations, _) in cases.items():
            images = [
                ee.Image.constant(value).toFloat().rename("snow_depth")
                .updateMask(ee.Image.constant(valid))
                .set("system:time_start", ee.Date(date).millis())
                for date, value, valid in observations
            ]

            def substitute_smap(collection: ee.ImageCollection, args: Any) -> None:
                """Replace only the configured SMAP source with dated test images.

                Args:
                    collection: Real client collection being initialized.
                    args: Original source ID, images or computed collection.
                """
                if isinstance(args, str) and args == self.configuration.datasets["smap"]:
                    args = images
                initialize_collection(collection, args)

            with patch.object(ee.ImageCollection, "__init__", substitute_smap):
                bands.append(build_earth_engine_stack(self.configuration).rename(name))
        actual = self.read_pixel(ee.Image.cat(*bands))
        for name, (_, expected) in cases.items():
            with self.subTest(case=name):
                self.assertAlmostEqual(expected, actual[name], places=6)

    def test_real_smap_partial_year_and_full_year_equivalence(self) -> None:
        """Evaluate available 2015 months and preserve the original complete-year mean."""
        fixed = build_earth_engine_stack(self.configuration).rename("fixed")
        # Removing only the masked seed reconstructs the original monthly graph.
        with patch.object(ee.ImageCollection, "merge", lambda collection, other: collection):
            baseline = build_earth_engine_stack(self.configuration).rename("baseline")
        values = self.read_pixel(ee.Image.cat(fixed, baseline))
        self.assertGreater(values["baseline"], 0)
        self.assertEqual(values["baseline"], values["fixed"])
        partial = build_earth_engine_stack(
            replace(self.configuration, year=2015)
        ).rename("partial_2015")
        self.assertGreater(self.read_pixel(partial)["partial_2015"], 0)

if __name__ == "__main__":
    unittest.main()
