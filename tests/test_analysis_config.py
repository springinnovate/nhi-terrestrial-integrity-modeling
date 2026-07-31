"""Tests for shared TOML analysis configuration."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.analysis_config import load_analysis_configuration


MINIMAL_ANALYSIS_TOML = """
[analysis]
name = "test_analysis_2019"
display_name = "Test analysis"
aoi_path = "aoi.geojson"
year = 2019

[earth_engine]
project = "example-project"
cache_directory = "cache"
request_timeout_seconds = 45.5
request_retry_count = 2

[stack]
name = "test_stack"
version = 3
minimum_year = 2017
maximum_year = 2020
reference_start_year = 2001
reference_end_year = 2020
era5_start_year = 1979
interannual_rainfall_window_years = 10
stream_upstream_area_threshold_km2 = 25

[grid]
crs = "EPSG:6933"
pixel_size_meters = 1000
tile_size_pixels = 64
origin_x = 0
origin_y = 0

[reference]
grassland_probability = 75
human_modification = 0.2
human_influence = 0.1

[sampling]
block_size_meters = 25000
samples_per_class_per_block = 100
random_seed = 42

[model]
fold_count = 5
validation_block_size_meters = 100000
minimum_predictor_coverage = 0.8
maximum_row_missing_fraction = 0.2
spline_knot_count = 6
minimum_response_coverage = 0.5
ridge_alpha = 1.0
responses = ["d02"]

[inference]
application_mask_path = "application_mask.tif"
window_size_pixels = 256
covariance_shrinkage = 0.1

[datasets]
reference = "projects/example/reference"
landform = "projects/example/landform"
ndvi = "projects/example/ndvi"
precipitation = "projects/example/precipitation"

[[bands]]
id = "d01"
computation = "grassland_reference_sites"
suffix = "reference"
display_name = "Reference"
role = "reference"
data_type = "binary"
sources = ["reference"]

[[bands]]
id = "d35"
computation = "landform_type"
suffix = "landform"
display_name = "Landform"
role = "predictor"
data_type = "categorical"
sources = ["landform"]

[[bands]]
id = "d02"
computation = "ndvi_median"
suffix = "ndvi"
display_name = "NDVI"
role = "response"
data_type = "continuous"
sources = ["ndvi"]

[[bands]]
id = "d24"
computation = "annual_precipitation"
suffix = "precipitation"
display_name = "Precipitation"
role = "predictor"
data_type = "continuous"
sources = ["precipitation"]
"""


class AnalysisConfigurationTest(unittest.TestCase):
    """Verify schema loading, role selection, and stable identity."""

    def setUp(self) -> None:
        """Create an isolated directory for modified TOML definitions."""

        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.temporary_path = Path(self.temporary_directory.name)
        (self.temporary_path / "aoi.geojson").write_text(
            '{"type":"Polygon","coordinates":[[[0,0],[1,0],[1,1],[0,0]]]}',
            encoding="utf-8",
        )

    def test_loads_default_39_band_pipeline_contract(self) -> None:
        """Expose the existing stack through configured names and roles."""

        configuration = load_analysis_configuration()

        self.assertEqual(
            "south_africa_reference_condition_2018",
            configuration.analysis_name,
        )
        self.assertEqual("nhi_reference_condition", configuration.stack_name)
        self.assertEqual(2018, configuration.year)
        self.assertEqual("South Africa", configuration.display_name)
        self.assertTrue(configuration.aoi_path.is_file())
        self.assertEqual(360, configuration.earth_engine.request_timeout_seconds)
        self.assertEqual(1, configuration.earth_engine.request_retry_count)
        self.assertEqual(
            "grassland_mask_2018.tif",
            configuration.inference.application_mask_path.name,
        )
        self.assertEqual(39, len(configuration.bands))
        role_counts = {
            role: sum(band.role == role for band in configuration.bands)
            for role in ("reference", "response", "predictor")
        }
        self.assertEqual(
            {"reference": 1, "response": 18, "predictor": 20},
            role_counts,
        )
        self.assertEqual(
            "y2018_d01_grassland_reference_sites",
            configuration.band_names(2018)[0],
        )
        self.assertEqual(
            "y2018_d39_average_snow_depth_when_pres",
            configuration.band_names(2018)[-1],
        )

    def test_uses_configured_order_and_earliest_available_year(self) -> None:
        """Honor a reduced order and select one column per configured role band."""

        configuration_path = self.temporary_path / "reduced.toml"
        configuration_path.write_text(MINIMAL_ANALYSIS_TOML, encoding="utf-8")
        configuration = load_analysis_configuration(configuration_path)

        self.assertEqual(
            self.temporary_path / "application_mask.tif",
            configuration.inference.application_mask_path,
        )

        self.assertEqual(
            (
                "y2019_d01_reference",
                "y2019_d35_landform",
                "y2019_d02_ndvi",
                "y2019_d24_precipitation",
            ),
            configuration.band_names(2019),
        )
        predictor_columns = configuration.columns_with_role(
            (
                "y2019_d24_precipitation",
                "y2018_d35_landform",
                "y2018_d24_precipitation",
                "y2019_d35_landform",
            ),
            "predictor",
        )
        self.assertEqual(
            {
                "d35": "y2018_d35_landform",
                "d24": "y2018_d24_precipitation",
            },
            predictor_columns,
        )

    def test_effective_configuration_change_changes_hash(self) -> None:
        """Change cache identity when effective TOML content changes."""

        first_path = self.temporary_path / "first.toml"
        second_path = self.temporary_path / "second.toml"
        first_path.write_text(MINIMAL_ANALYSIS_TOML, encoding="utf-8")
        second_path.write_text(
            MINIMAL_ANALYSIS_TOML.replace(
                'display_name = "Precipitation"',
                'display_name = "Annual precipitation"',
            ),
            encoding="utf-8",
        )

        first = load_analysis_configuration(first_path)
        second = load_analysis_configuration(second_path)

        self.assertNotEqual(first.configuration_sha256, second.configuration_sha256)
        self.assertNotEqual(
            first.raster_configuration_sha256,
            second.raster_configuration_sha256,
        )

    def test_model_change_preserves_raster_cache_identity(self) -> None:
        """Do not redownload pixels when only a model setting changes."""

        first_path = self.temporary_path / "first.toml"
        second_path = self.temporary_path / "second.toml"
        first_path.write_text(MINIMAL_ANALYSIS_TOML, encoding="utf-8")
        second_path.write_text(
            MINIMAL_ANALYSIS_TOML.replace("ridge_alpha = 1.0", "ridge_alpha = 2.0"),
            encoding="utf-8",
        )

        first = load_analysis_configuration(first_path)
        second = load_analysis_configuration(second_path)

        self.assertNotEqual(first.configuration_sha256, second.configuration_sha256)
        self.assertEqual(
            first.raster_configuration_sha256,
            second.raster_configuration_sha256,
        )

    def test_request_policy_change_preserves_raster_cache_identity(self) -> None:
        """Do not redownload pixels when only request transport policy changes."""

        first_path = self.temporary_path / "first.toml"
        second_path = self.temporary_path / "second.toml"
        first_path.write_text(MINIMAL_ANALYSIS_TOML, encoding="utf-8")
        second_path.write_text(
            MINIMAL_ANALYSIS_TOML.replace(
                "request_timeout_seconds = 45.5",
                "request_timeout_seconds = 90",
            ),
            encoding="utf-8",
        )

        first = load_analysis_configuration(first_path)
        second = load_analysis_configuration(second_path)

        self.assertNotEqual(first.configuration_sha256, second.configuration_sha256)
        self.assertEqual(
            first.raster_configuration_sha256,
            second.raster_configuration_sha256,
        )


if __name__ == "__main__":
    unittest.main()
