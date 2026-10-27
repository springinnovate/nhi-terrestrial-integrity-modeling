"""Build the configured scientific stack through the real Earth Engine client."""

from dataclasses import replace
from importlib.resources import files
import io
import json
from pathlib import Path
import unittest
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


if __name__ == "__main__":
    unittest.main()
