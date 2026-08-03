"""Tests for bounded-memory spatial sampling from raster-cache tiles."""

from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from pyproj import Transformer
from rasterio.io import MemoryFile
from rasterio.transform import Affine
from shapely.geometry import box, mapping
from shapely.ops import transform

from scripts.build_spatial_sample import (
    PARQUET_PROVENANCE_KEY,
    build_spatial_sample,
    create_analysis_location_figure,
    parse_args,
    print_cache_scan_report,
    print_spatial_sampling_report,
    write_spatial_sample_parquet,
)
from scripts.fetch_gee_raster_tiles import cache_aoi_tiles
from scripts.raster_cache_utils import resolve_analysis_cache_tiles
from scripts.analysis_config import (
    RasterCacheGrid,
    load_analysis_configuration,
)


class BuildSpatialSampleTest(unittest.TestCase):
    """Verify cache validation, global strata, determinism, and outputs."""

    def setUp(self) -> None:
        """Create a two-tile equal-area AOI and isolated cache."""

        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.workspace = Path(self.temporary_directory.name)
        self.aoi_path = self.workspace / "aoi.geojson"
        self.cache_directory = self.workspace / "cache"
        self.test_grid = RasterCacheGrid(
            crs="EPSG:6933",
            pixel_size_meters=1_000,
            tile_size_pixels=4,
        )
        self.default_analysis = load_analysis_configuration()
        projected_aoi = box(100, 100, 7_900, 3_900)
        to_wgs84 = Transformer.from_crs(
            self.test_grid.crs,
            "EPSG:4326",
            always_xy=True,
        )
        self.wgs84_aoi = transform(to_wgs84.transform, projected_aoi)
        self.aoi_path.write_text(
            json.dumps(mapping(self.wgs84_aoi)),
            encoding="utf-8",
        )
        self.analysis_configuration = replace(
            self.default_analysis,
            aoi_path=self.aoi_path,
            earth_engine=replace(
                self.default_analysis.earth_engine,
                project="offline-test-project",
                cache_directory=self.cache_directory,
            ),
            grid=self.test_grid,
            sampling=replace(
                self.default_analysis.sampling,
                block_size_meters=10_000,
                samples_per_class_per_block=3,
                random_seed=17,
            ),
        )

    @staticmethod
    def create_tile_bytes(
        raster_stack,
        tile,
        cache_grid,
        band_names,
    ) -> bytes:
        """Create a valid tile with repeatable reference and response values.

        Args:
            raster_stack: Unused Earth Engine expression placeholder.
            tile: Requested deterministic cache tile.
            cache_grid: Pixel dimensions and projected CRS.
            band_names: Ordered configured output names.

        Returns:
            Encoded multiband GeoTIFF bytes.
        """

        del raster_stack
        tile_size = cache_grid.tile_size_pixels
        local_rows, local_columns = np.indices((tile_size, tile_size))
        global_columns = tile.column * tile_size + local_columns
        values = np.empty(
            (len(band_names), tile_size, tile_size),
            dtype=np.float32,
        )
        values[0] = ((global_columns + local_rows) % 3 == 0).astype(
            np.float32
        )
        for band_offset in range(1, len(band_names)):
            values[band_offset] = (
                band_offset
                + global_columns * 0.1
                + local_rows * 0.01
            )
        values[-1, 0, 0] = np.nan
        tile_transform = Affine(
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
                width=tile_size,
                height=tile_size,
                count=len(band_names),
                dtype="float32",
                crs=cache_grid.crs,
                transform=tile_transform,
            ) as destination:
                destination.write(values)
                for band_index, band_name in enumerate(
                    band_names,
                    start=1,
                ):
                    destination.set_band_description(
                        band_index,
                        band_name,
                    )
            return memory_file.read()

    def populate_cache(self):
        """Fetch mocked cache tiles and return their validated resolution."""

        with contextlib.redirect_stdout(io.StringIO()):
            cache_aoi_tiles(
                self.analysis_configuration,
                refresh=False,
                show_progress=False,
                tile_fetcher=self.create_tile_bytes,
            )
        return resolve_analysis_cache_tiles(
            self.analysis_configuration,
            show_progress=False,
        )

    def test_enforces_one_global_cap_across_two_cache_tiles(self) -> None:
        """Merge candidates when one sampling block crosses a tile boundary."""

        analysis_cache_tiles = self.populate_cache()
        spatial_sample, raster_summary = build_spatial_sample(
            self.analysis_configuration,
            analysis_cache_tiles,
            show_progress=False,
        )

        self.assertEqual(2, len(analysis_cache_tiles.tiles))
        self.assertEqual(32, raster_summary.aoi_pixel_count)
        self.assertEqual(32, raster_summary.eligible_pixel_count)
        self.assertEqual(1, spatial_sample.block_count)
        self.assertEqual(6, len(spatial_sample.table))
        self.assertEqual(
            {0: 3, 1: 3},
            spatial_sample.table["reference_site"].value_counts().to_dict(),
        )
        self.assertEqual(
            32,
            sum(
                class_summary.available_pixels
                for class_summary in spatial_sample.class_summaries
            ),
        )
        for class_summary in spatial_sample.class_summaries:
            self.assertAlmostEqual(
                class_summary.available_pixels,
                class_summary.weighted_pixels,
            )
        self.assertLess(raster_summary.peak_tile_array_bytes, 10_000)

    def test_cli_rejects_a_positional_geotiff(self) -> None:
        """Prevent a mask or monolithic raster from entering the cache workflow."""

        with (
            patch.object(
                sys,
                "argv",
                [
                    "build_spatial_sample",
                    str(self.default_analysis.path),
                    "data/grassland_mask_2018.tif",
                ],
            ),
            contextlib.redirect_stderr(io.StringIO()),
            self.assertRaises(SystemExit),
        ):
            parse_args()

    def test_repeated_sampling_selects_identical_rows(self) -> None:
        """Use fixed tile-local priorities for repeatable selected pixels."""

        analysis_cache_tiles = self.populate_cache()
        first_sample, _ = build_spatial_sample(
            self.analysis_configuration,
            analysis_cache_tiles,
            show_progress=False,
        )
        second_sample, _ = build_spatial_sample(
            self.analysis_configuration,
            analysis_cache_tiles,
            show_progress=False,
        )

        pd.testing.assert_frame_equal(first_sample.table, second_sample.table)

    def test_clips_complete_edge_tiles_to_the_configured_aoi(self) -> None:
        """Exclude reusable edge-tile pixels whose centers fall outside the AOI."""

        small_aoi_path = self.workspace / "small_aoi.geojson"
        to_wgs84 = Transformer.from_crs(
            self.test_grid.crs,
            "EPSG:4326",
            always_xy=True,
        )
        small_wgs84_aoi = transform(
            to_wgs84.transform,
            box(100, 100, 1_900, 1_900),
        )
        small_aoi_path.write_text(
            json.dumps(mapping(small_wgs84_aoi)),
            encoding="utf-8",
        )
        small_analysis_configuration = replace(
            self.analysis_configuration,
            aoi_path=small_aoi_path,
        )
        with contextlib.redirect_stdout(io.StringIO()):
            cache_aoi_tiles(
                small_analysis_configuration,
                refresh=False,
                show_progress=False,
                tile_fetcher=self.create_tile_bytes,
            )
        analysis_cache_tiles = resolve_analysis_cache_tiles(
            small_analysis_configuration,
            show_progress=False,
        )

        _, raster_summary = build_spatial_sample(
            small_analysis_configuration,
            analysis_cache_tiles,
            show_progress=False,
        )

        self.assertEqual(1, raster_summary.cache_tile_count)
        self.assertEqual(4, raster_summary.aoi_pixel_count)
        self.assertEqual(4, raster_summary.eligible_pixel_count)

    def test_missing_cache_tile_fails_before_sampling(self) -> None:
        """Require a complete validated cache and provide a fetch instruction."""

        analysis_cache_tiles = self.populate_cache()
        analysis_cache_tiles.tiles[0].path.unlink()

        with self.assertRaisesRegex(
            RuntimeError,
            "python -m scripts.fetch_gee_raster_tiles",
        ):
            resolve_analysis_cache_tiles(
                self.analysis_configuration,
                show_progress=False,
            )

    def test_rejects_schema_mismatch_during_cache_resolution(self) -> None:
        """Reject an incompatible raster header before scanning pixel arrays."""

        analysis_cache_tiles = self.populate_cache()
        first_tile_path = analysis_cache_tiles.tiles[0].path
        with patch(
            "scripts.raster_cache_utils.calculate_file_sha256",
            return_value=analysis_cache_tiles.tiles[0].sha256,
        ):
            with first_tile_path.open("r+b") as tile_file:
                tile_file.seek(0)
                tile_file.write(b"not-a-geotiff")
            with self.assertRaisesRegex(RuntimeError, "missing or invalid"):
                resolve_analysis_cache_tiles(
                    self.analysis_configuration,
                    show_progress=False,
                )

    def test_writes_model_compatible_parquet_with_provenance(self) -> None:
        """Round-trip selected rows and embed the active cache identity."""

        analysis_cache_tiles = self.populate_cache()
        spatial_sample, _ = build_spatial_sample(
            self.analysis_configuration,
            analysis_cache_tiles,
            show_progress=False,
        )
        output_path = self.workspace / "sample.parquet"

        parquet_summary = write_spatial_sample_parquet(
            spatial_sample,
            self.analysis_configuration,
            analysis_cache_tiles,
            output_path,
            show_progress=False,
        )

        round_trip_table = pd.read_parquet(output_path)
        parquet_file = pq.ParquetFile(output_path)
        provenance = json.loads(
            parquet_file.schema_arrow.metadata[PARQUET_PROVENANCE_KEY]
        )
        pd.testing.assert_frame_equal(spatial_sample.table, round_trip_table)
        self.assertEqual(len(spatial_sample.table), parquet_summary.rows)
        self.assertEqual(
            np.dtype(np.int64),
            round_trip_table["sampling_block_column"].dtype,
        )
        self.assertEqual(
            np.dtype(np.int64),
            round_trip_table["sampling_block_row"].dtype,
        )
        self.assertEqual(2, provenance["cache_tile_count"])
        self.assertEqual(
            analysis_cache_tiles.stack_identifier,
            provenance["stack_identifier"],
        )
        self.assertEqual(17, provenance["sampling"]["random_seed"])

    def test_reports_tile_memory_progress_and_sampling_checks(self) -> None:
        """Give visible confidence without implying a monolithic allocation."""

        analysis_cache_tiles = self.populate_cache()
        progress_output = io.StringIO()
        with contextlib.redirect_stderr(progress_output):
            spatial_sample, raster_summary = build_spatial_sample(
                self.analysis_configuration,
                analysis_cache_tiles,
                show_progress=True,
            )
        report_output = io.StringIO()
        with contextlib.redirect_stdout(report_output):
            print_cache_scan_report(
                self.analysis_configuration,
                analysis_cache_tiles,
                raster_summary,
                include_band_report=True,
            )
            print_spatial_sampling_report(spatial_sample)

        rendered_progress = progress_output.getvalue()
        rendered_report = report_output.getvalue()
        self.assertIn("Scanning cached raster tiles", rendered_progress)
        self.assertIn("background=", rendered_progress)
        self.assertIn("reference=", rendered_progress)
        self.assertIn("Reading selected raster pixels", rendered_progress)
        self.assertIn("Largest in-memory source tile", rendered_report)
        self.assertIn("Configured band coverage inside the AOI", rendered_report)
        self.assertIn("Class sampling and weight checks", rendered_report)
        self.assertIn("Rows complete across every raster data band", rendered_report)

    @patch("scripts.build_spatial_sample.cfeature.LAND.with_scale")
    def test_creates_location_figure_from_configured_aoi(
        self,
        land_feature_mock,
    ) -> None:
        """Map the TOML AOI without deriving a full-raster validity footprint."""

        land_feature_mock.return_value = cfeature.ShapelyFeature(
            [],
            ccrs.PlateCarree(),
        )
        figure_path = self.workspace / "location.png"

        figure_summary = create_analysis_location_figure(
            self.wgs84_aoi,
            "Two Tile Test",
            figure_path,
            show_progress=False,
        )

        self.assertEqual(figure_path.resolve(), figure_summary.path)
        self.assertEqual("Two Tile Test", figure_summary.analysis_name)
        self.assertTrue(figure_path.exists())
        self.assertGreater(figure_path.stat().st_size, 1_000)
        self.assertLess(figure_summary.bounds.left, figure_summary.bounds.right)

    def test_rejects_non_png_location_figure(self) -> None:
        """Keep the locator-map output contract limited to PNG."""

        with self.assertRaisesRegex(ValueError, "must use the .png suffix"):
            create_analysis_location_figure(
                self.wgs84_aoi,
                "Two Tile Test",
                self.workspace / "location.svg",
                show_progress=False,
            )


if __name__ == "__main__":
    unittest.main()
