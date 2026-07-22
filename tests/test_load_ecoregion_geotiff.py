"""Tests for loading multiband ecoregion GeoTIFF pixels."""

from __future__ import annotations

import contextlib
import io
import ssl
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.figure
import numpy as np
import pandas as pd
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_origin

from scripts.load_ecoregion_geotiff import (
    assign_sampling_blocks,
    create_spatial_sample,
    create_ecoregion_location_figure,
    infer_ecoregion_name,
    load_raster_pixels,
    pixel_area_by_row_square_meters,
    print_spatial_sampling_report,
    print_raster_report,
    summarize_bands,
    summarize_coverage,
    write_spatial_sample_parquet,
)


class LoadEcoregionGeoTiffTest(unittest.TestCase):
    """Verify multiband values, masks, coordinates, and reporting."""

    def setUp(self) -> None:
        """Create a temporary two-band geographic GeoTIFF."""

        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.raster_path = Path(self.temporary_directory.name) / "ecoregion.tif"

        first_band = np.array(
            [[1.0, -9999.0, 3.0], [4.0, 5.0, 6.0]],
            dtype=np.float32,
        )
        second_band = np.array(
            [[10.0, 11.0, 12.0], [13.0, 14.0, -9999.0]],
            dtype=np.float32,
        )
        with rasterio.open(
            self.raster_path,
            "w",
            driver="GTiff",
            width=3,
            height=2,
            count=2,
            dtype="float32",
            crs="EPSG:4326",
            transform=from_origin(-110.0, 45.0, 0.5, 0.5),
            nodata=-9999.0,
        ) as destination:
            destination.write(first_band, 1)
            destination.write(second_band, 2)
            destination.set_band_description(1, "reference_sites")
            destination.set_band_description(2, "annual_precipitation")

    def test_loads_values_validity_and_pixel_views(self) -> None:
        """Load all cells and preserve each source band's validity mask."""

        raster = load_raster_pixels(self.raster_path, show_progress=False)

        self.assertEqual((2, 2, 3), raster.values.shape)
        self.assertEqual((2, 2, 3), raster.validity.shape)
        self.assertEqual(("reference_sites", "annual_precipitation"), raster.band_names)
        self.assertEqual(6, raster.pixel_count)
        self.assertTrue(np.isnan(raster.values[0, 0, 1]))
        self.assertTrue(np.isnan(raster.values[1, 1, 2]))
        self.assertFalse(raster.validity[0, 0, 1])
        self.assertFalse(raster.validity[1, 1, 2])
        self.assertEqual((6, 2), raster.pixel_values().shape)
        self.assertEqual((6, 2), raster.pixel_validity().shape)

    def test_calculates_coordinates_and_geographic_area(self) -> None:
        """Calculate row-major pixel centers and positive geographic areas."""

        raster = load_raster_pixels(self.raster_path, show_progress=False)
        x_coordinates, y_coordinates = raster.pixel_centers()
        pixel_areas = pixel_area_by_row_square_meters(raster)

        np.testing.assert_allclose(x_coordinates[:3], [-109.75, -109.25, -108.75])
        np.testing.assert_allclose(y_coordinates[:3], [44.75, 44.75, 44.75])
        self.assertIsNotNone(pixel_areas)
        self.assertEqual((2,), pixel_areas.shape)
        self.assertTrue(np.all(pixel_areas > 0))

    def test_calculates_projected_pixel_area(self) -> None:
        """Calculate constant pixel area for a projected raster grid."""

        raster = load_raster_pixels(self.raster_path, show_progress=False)
        projected_raster = replace(
            raster,
            crs=CRS.from_epsg(3857),
            transform=from_origin(0.0, 1_000.0, 500.0, 500.0),
        )

        pixel_areas = pixel_area_by_row_square_meters(projected_raster)

        np.testing.assert_allclose(pixel_areas, [250_000.0, 250_000.0])

    def test_summarizes_any_every_and_per_band_coverage(self) -> None:
        """Report masks without collapsing partially defined pixels."""

        raster = load_raster_pixels(self.raster_path, show_progress=False)
        pixel_areas = pixel_area_by_row_square_meters(raster)
        any_coverage = summarize_coverage(
            np.any(raster.validity, axis=0),
            pixel_areas,
        )
        every_coverage = summarize_coverage(
            np.all(raster.validity, axis=0),
            pixel_areas,
        )
        band_summaries = summarize_bands(
            raster,
            pixel_areas,
            show_progress=False,
        )

        self.assertEqual(6, any_coverage.defined_pixels)
        self.assertEqual(4, every_coverage.defined_pixels)
        self.assertEqual(5, band_summaries[0].coverage.defined_pixels)
        self.assertEqual(1.0, band_summaries[0].minimum)
        self.assertEqual(6.0, band_summaries[0].maximum)
        self.assertGreater(any_coverage.area_square_kilometers, 0)

    def test_prints_interesting_raster_report(self) -> None:
        """Print dimensions, memory, coverage, area, and band names."""

        raster = load_raster_pixels(self.raster_path, show_progress=False)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            print_raster_report(raster, True, False)

        report = output.getvalue()
        self.assertIn("2 rows x 2 bands", report)
        self.assertIn("Array memory", report)
        self.assertIn("Declared nodata values", report)
        self.assertIn("Bands with defined pixels", report)
        self.assertIn("Defined in any band", report)
        self.assertIn("Defined in every band", report)
        self.assertIn("approx. area", report)
        self.assertIn("reference_sites", report)
        self.assertIn("annual_precipitation", report)

    def test_infers_ecoregion_name(self) -> None:
        """Convert an Earth Engine export stem into a readable map label."""

        export_path = Path(
            "northern_shortgrass_prairie_e0042_response_variables_year_2019.tif"
        )

        name = infer_ecoregion_name(export_path)

        self.assertEqual("Northern Shortgrass Prairie", name)

    def _create_sampling_raster(self) -> Path:
        """Create a reference-mask export with two predictor bands.

        Returns:
            Path to the synthetic four-band GeoTIFF.
        """

        path = Path(self.temporary_directory.name) / "sampling_ecoregion.tif"
        reference = np.full((4, 6), -9999.0, dtype=np.float32)
        reference.flat[[0, 1, 6, 15, 22, 23]] = 1.0
        first_predictor = np.arange(24, dtype=np.float32).reshape(4, 6)
        first_predictor[0, 0] = -9999.0
        second_predictor = np.arange(100, 124, dtype=np.float32).reshape(4, 6)

        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            width=6,
            height=4,
            count=4,
            dtype="float32",
            crs="EPSG:3857",
            transform=from_origin(100_000.0, 104_000.0, 1_000.0, 1_000.0),
            nodata=-9999.0,
        ) as destination:
            destination.write(reference, 1)
            destination.write(first_predictor, 2)
            destination.write(reference, 3)
            destination.write(second_predictor, 4)
            destination.set_band_description(
                1,
                "y2018_d01_grassland_reference_sites",
            )
            destination.set_band_description(2, "y2018_d02_annual_precipitation")
            destination.set_band_description(
                3,
                "y2019_d01_grassland_reference_sites",
            )
            destination.set_band_description(4, "y2019_d02_annual_precipitation")
        return path

    def test_assigns_stable_equal_area_sampling_blocks(self) -> None:
        """Assign coordinates on either side of fixed block boundaries."""

        block_ids, block_columns, block_rows = assign_sampling_blocks(
            np.array([-1.0, 0.0, 24_999.0, 25_000.0, 25_000.0]),
            np.array([0.0, 0.0, 24_999.0, 25_000.0, 25_000.0]),
            25_000.0,
        )

        np.testing.assert_array_equal(block_columns, [-1, 0, 0, 1, 1])
        np.testing.assert_array_equal(block_rows, [0, 0, 0, 1, 1])
        self.assertEqual(block_ids[1], block_ids[2])
        self.assertEqual(block_ids[3], block_ids[4])
        self.assertNotEqual(block_ids[0], block_ids[1])

    def test_samples_classes_separately_and_calculates_weights(self) -> None:
        """Cap each reference-site class and reconstruct source populations."""

        raster = load_raster_pixels(
            self._create_sampling_raster(),
            show_progress=False,
        )
        first_sample = create_spatial_sample(
            raster,
            block_size_meters=1_000_000_000.0,
            samples_per_class_per_block=2,
            random_seed=7,
            show_progress=False,
        )
        second_sample = create_spatial_sample(
            raster,
            block_size_meters=1_000_000_000.0,
            samples_per_class_per_block=2,
            random_seed=7,
            show_progress=False,
        )

        self.assertEqual(4, len(first_sample.table))
        self.assertEqual(1, first_sample.block_count)
        self.assertEqual(
            (0, 0, 1, 1),
            tuple(sorted(first_sample.table["reference_site"])),
        )
        self.assertEqual(18, first_sample.class_summaries[0].available_pixels)
        self.assertEqual(6, first_sample.class_summaries[1].available_pixels)
        self.assertEqual(9.0, first_sample.class_summaries[0].maximum_sampling_weight)
        self.assertEqual(3.0, first_sample.class_summaries[1].maximum_sampling_weight)
        self.assertAlmostEqual(18.0, first_sample.class_summaries[0].weighted_pixels)
        self.assertAlmostEqual(6.0, first_sample.class_summaries[1].weighted_pixels)
        self.assertAlmostEqual(
            first_sample.class_summaries[0].available_area_square_meters,
            first_sample.class_summaries[0].weighted_area_square_meters,
        )
        self.assertAlmostEqual(
            first_sample.class_summaries[1].available_area_square_meters,
            first_sample.class_summaries[1].weighted_area_square_meters,
        )
        self.assertEqual(
            first_sample.table[["row", "column"]].values.tolist(),
            second_sample.table[["row", "column"]].values.tolist(),
        )
        self.assertEqual(
            ("y2019_d01_grassland_reference_sites",),
            first_sample.ignored_reference_band_names,
        )
        self.assertNotIn(
            "y2018_d01_grassland_reference_sites",
            first_sample.predictor_band_names,
        )

    def test_preserves_missing_predictors_and_writes_parquet(self) -> None:
        """Retain source missingness and verify a compressed Parquet round trip."""

        raster = load_raster_pixels(
            self._create_sampling_raster(),
            show_progress=False,
        )
        sample = create_spatial_sample(
            raster,
            block_size_meters=1_000_000_000.0,
            samples_per_class_per_block=100,
            random_seed=42,
            show_progress=False,
        )
        output_path = Path(self.temporary_directory.name) / "sample.parquet"
        write_summary = write_spatial_sample_parquet(
            sample,
            output_path,
            show_progress=False,
        )
        restored = pd.read_parquet(output_path)
        missing_row = restored[(restored["row"] == 0) & (restored["column"] == 0)]

        self.assertEqual(24, len(sample.table))
        self.assertTrue((sample.table["sampling_weight"] == 1.0).all())
        self.assertEqual(24, write_summary.rows)
        self.assertEqual(sample.table.shape[1], write_summary.columns)
        self.assertEqual("ZSTD", write_summary.compression)
        self.assertTrue(missing_row["y2018_d02_annual_precipitation"].isna().all())
        pd.testing.assert_frame_equal(sample.table, restored)

    def test_prints_detailed_spatial_sampling_report(self) -> None:
        """Report reference, block, class, weight, and predictor diagnostics."""

        raster = load_raster_pixels(
            self._create_sampling_raster(),
            show_progress=False,
        )
        sample = create_spatial_sample(
            raster,
            block_size_meters=1_000_000_000.0,
            samples_per_class_per_block=2,
            random_seed=42,
            show_progress=False,
        )
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            print_spatial_sampling_report(sample)

        report = output.getvalue()
        self.assertIn("Spatial sampling report", report)
        self.assertIn("1 = reference site; 0 = non-reference site", report)
        self.assertIn("Blocks containing reference sites", report)
        self.assertIn("Class sampling and weight checks", report)
        self.assertIn("0 non-reference", report)
        self.assertIn("1 reference", report)
        self.assertIn("Sampled predictor coverage", report)
        self.assertIn("Lowest-coverage predictor bands", report)

    @patch("scripts.load_ecoregion_geotiff.cfeature.LAND.with_scale")
    def test_creates_world_location_figure_without_network(
        self,
        land_feature_mock,
    ) -> None:
        """Render the footprint, bounds, and label without downloading map data."""

        land_feature_mock.return_value = cfeature.ShapelyFeature(
            [],
            ccrs.PlateCarree(),
        )
        raster = load_raster_pixels(self.raster_path, show_progress=False)
        figure_path = Path(self.temporary_directory.name) / "location.png"

        summary = create_ecoregion_location_figure(
            raster,
            "Test Prairie",
            figure_path,
            False,
        )

        self.assertEqual(figure_path.resolve(), summary.path)
        self.assertEqual("Test Prairie", summary.ecoregion_name)
        self.assertTrue(summary.land_basemap_available)
        self.assertTrue(figure_path.exists())
        self.assertGreater(figure_path.stat().st_size, 1_000)
        self.assertGreater(summary.display_defined_pixels, 0)
        self.assertLess(summary.bounds.left, summary.bounds.right)
        self.assertLess(summary.bounds.bottom, summary.bounds.top)

    @patch("scripts.load_ecoregion_geotiff.cfeature.LAND.with_scale")
    def test_saves_location_figure_when_land_basemap_download_fails(
        self,
        land_feature_mock,
    ) -> None:
        """Fall back to a footprint-only map when Cartopy land data fails."""

        land_feature_mock.return_value = cfeature.ShapelyFeature(
            [],
            ccrs.PlateCarree(),
        )
        original_savefig = matplotlib.figure.Figure.savefig
        save_attempts = 0

        def savefig_with_one_ssl_failure(figure, *args, **kwargs):
            nonlocal save_attempts
            save_attempts += 1
            if save_attempts == 1:
                raise ssl.SSLError("[ASN1: NOT_ENOUGH_DATA] not enough data")
            return original_savefig(figure, *args, **kwargs)

        raster = load_raster_pixels(self.raster_path, show_progress=False)
        figure_path = Path(self.temporary_directory.name) / "location_fallback.png"

        with patch(
            "matplotlib.figure.Figure.savefig",
            new=savefig_with_one_ssl_failure,
        ):
            summary = create_ecoregion_location_figure(
                raster,
                "Test Prairie",
                figure_path,
                False,
            )

        self.assertFalse(summary.land_basemap_available)
        self.assertEqual(2, save_attempts)
        self.assertTrue(figure_path.exists())
        self.assertGreater(figure_path.stat().st_size, 1_000)


if __name__ == "__main__":
    unittest.main()
