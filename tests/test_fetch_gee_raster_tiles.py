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
from scripts.analysis_config import RasterCacheGrid, load_analysis_configuration
from scripts.fetch_gee_raster_tiles import (
    cache_aoi_tiles,
    fetch_tile_bytes,
)
from scripts.raster_cache_utils import (
    CacheTile,
    build_stack_identifier,
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
        self.test_grid = RasterCacheGrid(
            crs="EPSG:6933",
            pixel_size_meters=1_000,
            tile_size_pixels=4,
        )
        self.default_analysis = load_analysis_configuration()

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

    def analysis_configuration(self, **changes):
        """Return the default analysis pointed at isolated test resources."""

        return replace(
            self.default_analysis,
            aoi_path=self.aoi_path,
            earth_engine=replace(
                self.default_analysis.earth_engine,
                project="offline-test-project",
                cache_directory=self.cache_directory,
            ),
            grid=self.test_grid,
            **changes,
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

        band_names = self.default_analysis.band_names()

        self.assertEqual(39, len(self.default_analysis.bands))
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

    def test_reference_sites_have_no_ecoregion_source(self) -> None:
        """Construct d01 without a hidden spatial ecoregion restriction."""

        reference_definition = next(
            definition
            for definition in self.default_analysis.bands
            if definition.role == "reference"
        )

        self.assertNotIn(
            "maybe_grassland_ecoregions",
            self.default_analysis.datasets,
        )
        self.assertEqual(
            (
                "grassland_probability",
                "human_modification",
                "human_influence",
            ),
            reference_definition.source_dataset_keys,
        )

    def test_stack_version_invalidates_ecoregion_restricted_reference_tiles(
        self,
    ) -> None:
        """Use a new namespace after removing the d01 ecoregion restriction."""

        stack_identifier = build_stack_identifier(self.default_analysis)

        self.assertEqual(3, self.default_analysis.stack_version)
        self.assertIn("_v3_", stack_identifier)

    def test_request_policy_comes_from_analysis_configuration(self) -> None:
        """Keep Earth Engine request policy in the TOML single source of truth."""

        self.assertEqual(
            360,
            self.default_analysis.earth_engine.request_timeout_seconds,
        )
        self.assertEqual(1, self.default_analysis.earth_engine.request_retry_count)

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
        analysis_configuration = self.analysis_configuration()
        first_summary = cache_aoi_tiles(
            analysis_configuration,
            refresh=False,
            show_progress=False,
            tile_fetcher=self.create_tile_bytes,
        )

        def unexpected_download(*args, **kwargs):
            self.fail("A valid cached tile should not be downloaded again.")

        second_summary = cache_aoi_tiles(
            analysis_configuration,
            refresh=False,
            show_progress=False,
            tile_fetcher=unexpected_download,
        )

        self.assertEqual(1, first_summary.downloaded_tiles)
        self.assertEqual(0, first_summary.reused_tiles)
        self.assertEqual(0, second_summary.downloaded_tiles)
        self.assertEqual(1, second_summary.reused_tiles)

        manifest = json.loads(
            (self.cache_directory / "manifest.json").read_text(encoding="utf-8")
        )
        stack_identifier = build_stack_identifier(analysis_configuration)
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
            analysis_configuration.configuration_sha256,
            manifest["stacks"][stack_identifier][
                "analysis_configuration_sha256"
            ],
        )
        self.assertEqual(
            analysis_configuration.raster_configuration_sha256,
            manifest["stacks"][stack_identifier][
                "raster_configuration_sha256"
            ],
        )
        self.assertEqual(
            analysis_configuration.datasets["landsat_ndvi"],
            manifest["stacks"][stack_identifier]["datasets"]["landsat_ndvi"],
        )

    def test_writes_reduced_bands_in_configured_order(self) -> None:
        """Use TOML-derived inclusion and ordering for downloaded GeoTIFFs."""

        self.write_projected_aoi(box(100, 100, 3_900, 3_900))
        bands_by_identifier = {
            band.identifier: band for band in self.default_analysis.bands
        }
        reduced_configuration = replace(
            self.analysis_configuration(),
            configuration_sha256="1" * 64,
            raster_configuration_sha256="2" * 64,
            bands=tuple(
                bands_by_identifier[identifier]
                for identifier in ("d01", "d35", "d02", "d24")
            ),
        )
        self.assertNotEqual(
            build_stack_identifier(self.analysis_configuration()),
            build_stack_identifier(reduced_configuration),
        )
        cache_aoi_tiles(
            reduced_configuration,
            refresh=False,
            show_progress=False,
            tile_fetcher=self.create_tile_bytes,
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
        with redirect_stderr(progress_output), redirect_stdout(report_output):
            cache_aoi_tiles(
                self.analysis_configuration(),
                refresh=False,
                show_progress=True,
                tile_fetcher=self.create_tile_bytes,
            )

        rendered_progress = progress_output.getvalue()
        self.assertIn("Checking cached grids", rendered_progress)
        self.assertIn("Processing grids", rendered_progress)
        self.assertIn("processed=1", rendered_progress)
        self.assertIn("cached=0", rendered_progress)
        self.assertIn("downloaded=1", rendered_progress)
        self.assertIn("failed=0", rendered_progress)

    def test_configures_bounded_earth_engine_requests(self) -> None:
        """Apply the requested transport timeout and retry count."""

        self.write_projected_aoi(box(100, 100, 3_900, 3_900))
        analysis_configuration = self.analysis_configuration()
        analysis_configuration = replace(
            analysis_configuration,
            earth_engine=replace(
                analysis_configuration.earth_engine,
                request_timeout_seconds=12.5,
                request_retry_count=3,
            ),
        )
        with (
            patch.object(fetch_gee_raster_tiles.ee, "Initialize") as initialize,
            patch.object(
                fetch_gee_raster_tiles.ee.data,
                "setDeadline",
            ) as set_deadline,
            patch.object(
                fetch_gee_raster_tiles.ee.data,
                "setMaxRetries",
            ) as set_max_retries,
            patch.object(
                fetch_gee_raster_tiles,
                "build_earth_engine_stack",
                return_value=object(),
            ),
            patch.object(
                fetch_gee_raster_tiles,
                "fetch_tile_bytes",
                side_effect=self.create_tile_bytes,
            ),
        ):
            summary = cache_aoi_tiles(
                analysis_configuration,
                refresh=False,
                show_progress=False,
            )

        self.assertEqual(1, summary.downloaded_tiles)
        initialize_arguments = initialize.call_args.kwargs
        self.assertEqual("offline-test-project", initialize_arguments["project"])
        self.assertEqual(12.5, initialize_arguments["http_transport"].timeout)
        set_deadline.assert_called_once_with(12_500)
        set_max_retries.assert_called_once_with(3)

    def test_stops_after_failure_and_rerun_reuses_completed_tiles(self) -> None:
        """Fail fast during an outage and resume from validated cache entries."""

        self.write_projected_aoi(box(100, 100, 7_900, 7_900))
        attempted_tiles = []

        def interrupted_fetch(raster_stack, tile, cache_grid, band_names):
            """Return one tile, then simulate a disconnected request."""

            attempted_tiles.append(tile.tile_id)
            if len(attempted_tiles) == 2:
                raise TimeoutError("network unavailable")
            return self.create_tile_bytes(
                raster_stack,
                tile,
                cache_grid,
                band_names,
            )

        with self.assertRaisesRegex(
            RuntimeError,
            "Completed tiles remain cached; rerun the same command to resume",
        ):
            cache_aoi_tiles(
                self.analysis_configuration(),
                refresh=False,
                show_progress=False,
                tile_fetcher=interrupted_fetch,
            )

        resumed_summary = cache_aoi_tiles(
            self.analysis_configuration(),
            refresh=False,
            show_progress=False,
            tile_fetcher=self.create_tile_bytes,
        )

        self.assertEqual(2, len(attempted_tiles))
        self.assertEqual(4, resumed_summary.requested_tiles)
        self.assertEqual(1, resumed_summary.reused_tiles)
        self.assertEqual(3, resumed_summary.downloaded_tiles)
        manifest = json.loads(
            (self.cache_directory / "manifest.json").read_text(encoding="utf-8")
        )
        failed_request = manifest["requests"][-2]
        self.assertTrue(failed_request["aborted_after_failure"])
        self.assertEqual([attempted_tiles[-1]], failed_request["failed_tile_ids"])
        self.assertEqual("TimeoutError", failed_request["failure"]["type"])
        self.assertEqual(
            "network unavailable",
            failed_request["failure"]["message"],
        )
        self.assertEqual(4, len(manifest["tiles"]))

    def test_corrupt_cached_tile_is_replaced(self) -> None:
        """Reject a cached file whose checksum no longer matches the manifest."""

        self.write_projected_aoi(box(100, 100, 3_900, 3_900))
        analysis_configuration = self.analysis_configuration()
        cache_aoi_tiles(
            analysis_configuration,
            refresh=False,
            show_progress=False,
            tile_fetcher=self.create_tile_bytes,
        )
        cached_tile_path = next((self.cache_directory / "tiles").rglob("*.tif"))
        cached_tile_path.write_bytes(b"not a geotiff")
        replacement_summary = cache_aoi_tiles(
            analysis_configuration,
            refresh=False,
            show_progress=False,
            tile_fetcher=self.create_tile_bytes,
        )

        self.assertEqual(1, replacement_summary.downloaded_tiles)
        self.assertEqual(0, replacement_summary.reused_tiles)
        self.assertGreater(cached_tile_path.stat().st_size, len(b"not a geotiff"))


if __name__ == "__main__":
    unittest.main()
