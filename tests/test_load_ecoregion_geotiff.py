"""Tests for loading multiband ecoregion GeoTIFF pixels."""

from __future__ import annotations

import contextlib
import io
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_origin

from scripts.load_ecoregion_geotiff import (
    load_raster_pixels,
    pixel_area_by_row_square_meters,
    print_raster_report,
    summarize_bands,
    summarize_coverage,
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
            print_raster_report(raster, show_progress=False)

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


if __name__ == "__main__":
    unittest.main()
