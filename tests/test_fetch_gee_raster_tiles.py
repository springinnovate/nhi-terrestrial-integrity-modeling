"""Tests for deterministic Earth Engine raster tile caching."""

from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
from pyproj import Transformer
from rasterio.io import MemoryFile
from rasterio.transform import Affine
from shapely.geometry import box, mapping
from shapely.ops import transform

from scripts import fetch_gee_raster_tiles
from scripts.fetch_gee_raster_tiles import (
    CacheGrid,
    CacheTile,
    DEFAULT_STACK_CONFIGURATION,
    ReferenceThresholds,
    build_stack_identifier,
    cache_aoi_tiles,
    fetch_tile_bytes,
    select_intersecting_tiles,
)


class FetchGeeRasterTilesTest(unittest.TestCase):
    """Verify cache-grid selection, tile reuse, and manifest metadata."""

    def setUp(self) -> None:
        """Create an isolated cache and a small production-shaped test grid."""

        self.temporary_directory = tempfile.TemporaryDirectory()
        self.workspace = Path(self.temporary_directory.name)
        self.aoi_path = self.workspace / "aoi.geojson"
        self.cache_directory = self.workspace / "cache"
        self.test_grid = CacheGrid(
            crs="EPSG:6933",
            pixel_size_meters=1_000,
            tile_size_pixels=4,
        )

    def tearDown(self) -> None:
        """Remove the isolated test workspace."""

        self.temporary_directory.cleanup()

    def write_projected_aoi(self, projected_geometry) -> None:
        """Write an EPSG:6933 test geometry as WGS84 GeoJSON.

        Args:
            projected_geometry: Polygon expressed in the test cache CRS.

        Returns:
            None: The transformed geometry is written to ``self.aoi_path``.
        """

        to_wgs84 = Transformer.from_crs(
            self.test_grid.crs,
            "EPSG:4326",
            always_xy=True,
        )
        wgs84_geometry = transform(to_wgs84.transform, projected_geometry)
        self.aoi_path.write_text(
            json.dumps(mapping(wgs84_geometry)),
            encoding="utf-8",
        )

    @staticmethod
    def create_tile_bytes(
        raster_stack,
        tile,
        cache_grid,
        band_names,
    ) -> bytes:
        """Create a valid in-memory GeoTIFF for a mocked tile request.

        Args:
            raster_stack: Unused Earth Engine expression placeholder.
            tile: Requested deterministic cache tile.
            cache_grid: Pixel dimensions and projected CRS.
            band_names: Ordered output band descriptions.

        Returns:
            Encoded multiband GeoTIFF bytes.
        """

        del raster_stack
        transform_matrix = Affine(
            cache_grid.pixel_size_meters,
            0,
            tile.left,
            0,
            -cache_grid.pixel_size_meters,
            tile.top,
        )
        with MemoryFile() as memory_file:
            with memory_file.open(
                driver="GTiff",
                width=cache_grid.tile_size_pixels,
                height=cache_grid.tile_size_pixels,
                count=len(band_names),
                dtype="float32",
                crs=cache_grid.crs,
                transform=transform_matrix,
            ) as destination:
                destination.write(
                    np.zeros(
                        (
                            len(band_names),
                            cache_grid.tile_size_pixels,
                            cache_grid.tile_size_pixels,
                        ),
                        dtype=np.float32,
                    )
                )
                for band_index, band_name in enumerate(band_names, start=1):
                    destination.set_band_description(band_index, band_name)
            return memory_file.read()

    def test_band_schema_has_stable_d01_through_d39_names(self) -> None:
        """Expose one unique, ordered export name for every modeled band."""

        band_names = DEFAULT_STACK_CONFIGURATION.band_names(2018)

        self.assertEqual(39, len(DEFAULT_STACK_CONFIGURATION.bands))
        self.assertEqual(39, len(band_names))
        self.assertEqual(39, len(set(band_names)))
        self.assertEqual(
            "y2018_d01_grassland_reference_sites",
            band_names[0],
        )
        self.assertEqual(
            "y2018_d39_average_snow_depth_when_pres",
            band_names[-1],
        )

    def test_selects_globally_aligned_positive_area_intersections(self) -> None:
        """Select four shared tiles for an AOI spanning two rows and columns."""

        projected_aoi = box(100, 100, 7_900, 7_900)
        to_wgs84 = Transformer.from_crs(
            self.test_grid.crs,
            "EPSG:4326",
            always_xy=True,
        )
        wgs84_aoi = transform(to_wgs84.transform, projected_aoi)

        _, selected_tiles = select_intersecting_tiles(wgs84_aoi, self.test_grid)

        self.assertEqual(
            [
                "x+000000_y+000001",
                "x+000001_y+000001",
                "x+000000_y+000000",
                "x+000001_y+000000",
            ],
            [tile.tile_id for tile in selected_tiles],
        )

    def test_compute_pixels_request_uses_equal_area_wkt_grid(self) -> None:
        """Send explicit meter-based dimensions and WKT accepted by Earth Engine."""

        cache_tile = CacheTile(
            column=0,
            row=0,
            tile_id="x+000000_y+000000",
            left=0,
            bottom=0,
            right=4_000,
            top=4_000,
        )
        with patch.object(
            fetch_gee_raster_tiles.ee.data,
            "computePixels",
            return_value=b"geotiff bytes",
        ) as compute_pixels:
            payload = fetch_tile_bytes(
                None,
                cache_tile,
                self.test_grid,
                ["example_band"],
            )

        request = compute_pixels.call_args.args[0]
        self.assertEqual(b"geotiff bytes", payload)
        self.assertEqual("GEO_TIFF", request["fileFormat"])
        self.assertEqual(
            {"width": 4, "height": 4},
            request["grid"]["dimensions"],
        )
        self.assertIn(
            "Cylindrical_Equal_Area",
            request["grid"]["crsWkt"],
        )
        self.assertEqual(1_000, request["grid"]["affineTransform"]["scaleX"])
        self.assertEqual(-1_000, request["grid"]["affineTransform"]["scaleY"])

    def test_reuses_valid_cached_tile_and_records_complete_metadata(self) -> None:
        """Download once, persist metadata, and skip a repeated AOI request."""

        self.write_projected_aoi(box(100, 100, 3_900, 3_900))
        thresholds = ReferenceThresholds()
        with patch.object(
            fetch_gee_raster_tiles,
            "CacheGrid",
            return_value=self.test_grid,
        ):
            first_summary = cache_aoi_tiles(
                self.aoi_path,
                2018,
                "offline-test-project",
                self.cache_directory,
                thresholds,
                refresh=False,
                show_progress=False,
                compute_tile=self.create_tile_bytes,
            )

            def unexpected_download(*args, **kwargs):
                self.fail("A valid cached tile should not be downloaded again.")

            second_summary = cache_aoi_tiles(
                self.aoi_path,
                2018,
                "offline-test-project",
                self.cache_directory,
                thresholds,
                refresh=False,
                show_progress=False,
                compute_tile=unexpected_download,
            )

        self.assertEqual(1, first_summary.downloaded_tiles)
        self.assertEqual(0, first_summary.reused_tiles)
        self.assertEqual(0, second_summary.downloaded_tiles)
        self.assertEqual(1, second_summary.reused_tiles)

        manifest = json.loads(
            (self.cache_directory / "manifest.json").read_text(encoding="utf-8")
        )
        stack_identifier = build_stack_identifier(2018, thresholds)
        tile_record = next(iter(manifest["tiles"].values()))
        self.assertEqual(stack_identifier, tile_record["stack_id"])
        self.assertEqual(2018, tile_record["year"])
        self.assertEqual("EPSG:6933", tile_record["crs"])
        self.assertEqual(1_000, tile_record["pixel_size_meters"])
        self.assertEqual(4, tile_record["width_pixels"])
        self.assertEqual(64, len(tile_record["sha256"]))
        self.assertEqual(2, len(manifest["requests"]))
        self.assertEqual(39, len(manifest["stacks"][stack_identifier]["bands"]))
        self.assertEqual(
            DEFAULT_STACK_CONFIGURATION.configuration_sha256,
            manifest["stacks"][stack_identifier]["configuration_sha256"],
        )
        self.assertEqual(
            DEFAULT_STACK_CONFIGURATION.datasets["landsat_ndvi"],
            manifest["stacks"][stack_identifier]["datasets"]["landsat_ndvi"],
        )

    def test_writes_reduced_bands_in_configured_order(self) -> None:
        """Use TOML-derived inclusion and ordering for downloaded GeoTIFFs."""

        self.write_projected_aoi(box(100, 100, 3_900, 3_900))
        bands_by_identifier = {
            band.identifier: band for band in DEFAULT_STACK_CONFIGURATION.bands
        }
        reduced_configuration = replace(
            DEFAULT_STACK_CONFIGURATION,
            configuration_sha256="1" * 64,
            bands=tuple(
                bands_by_identifier[identifier]
                for identifier in ("d01", "d35", "d02", "d24")
            ),
        )
        self.assertNotEqual(
            build_stack_identifier(2018, ReferenceThresholds()),
            build_stack_identifier(
                2018,
                ReferenceThresholds(),
                reduced_configuration,
            ),
        )
        with patch.object(
            fetch_gee_raster_tiles,
            "CacheGrid",
            return_value=self.test_grid,
        ):
            cache_aoi_tiles(
                self.aoi_path,
                2018,
                "offline-test-project",
                self.cache_directory,
                ReferenceThresholds(),
                refresh=False,
                show_progress=False,
                stack_configuration=reduced_configuration,
                compute_tile=self.create_tile_bytes,
            )

        cached_tile_path = next((self.cache_directory / "tiles").rglob("*.tif"))
        with fetch_gee_raster_tiles.rasterio.open(cached_tile_path) as source:
            self.assertEqual(4, source.count)
            self.assertEqual(
                reduced_configuration.band_names(2018),
                source.descriptions,
            )

    def test_progress_reports_cache_and_processing_counts(self) -> None:
        """Display both tqdm stages and every requested live grid counter."""

        self.write_projected_aoi(box(100, 100, 3_900, 3_900))
        progress_output = io.StringIO()
        report_output = io.StringIO()
        with (
            patch.object(
                fetch_gee_raster_tiles,
                "CacheGrid",
                return_value=self.test_grid,
            ),
            redirect_stderr(progress_output),
            redirect_stdout(report_output),
        ):
            cache_aoi_tiles(
                self.aoi_path,
                2018,
                "offline-test-project",
                self.cache_directory,
                ReferenceThresholds(),
                refresh=False,
                show_progress=True,
                compute_tile=self.create_tile_bytes,
            )

        rendered_progress = progress_output.getvalue()
        self.assertIn("Checking cached grids", rendered_progress)
        self.assertIn("Processing grids", rendered_progress)
        self.assertIn("processed=1", rendered_progress)
        self.assertIn("cached=0", rendered_progress)
        self.assertIn("downloaded=1", rendered_progress)
        self.assertIn("failed=0", rendered_progress)

    def test_corrupt_cached_tile_is_replaced(self) -> None:
        """Reject a cached file whose checksum no longer matches the manifest."""

        self.write_projected_aoi(box(100, 100, 3_900, 3_900))
        thresholds = ReferenceThresholds()
        with patch.object(
            fetch_gee_raster_tiles,
            "CacheGrid",
            return_value=self.test_grid,
        ):
            cache_aoi_tiles(
                self.aoi_path,
                2018,
                "offline-test-project",
                self.cache_directory,
                thresholds,
                refresh=False,
                show_progress=False,
                compute_tile=self.create_tile_bytes,
            )
            cached_tile_path = next((self.cache_directory / "tiles").rglob("*.tif"))
            cached_tile_path.write_bytes(b"not a geotiff")
            replacement_summary = cache_aoi_tiles(
                self.aoi_path,
                2018,
                "offline-test-project",
                self.cache_directory,
                thresholds,
                refresh=False,
                show_progress=False,
                compute_tile=self.create_tile_bytes,
            )

        self.assertEqual(1, replacement_summary.downloaded_tiles)
        self.assertEqual(0, replacement_summary.reused_tiles)
        self.assertGreater(cached_tile_path.stat().st_size, len(b"not a geotiff"))


if __name__ == "__main__":
    unittest.main()
