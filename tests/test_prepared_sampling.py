"""Compare prepared-raster samples with the existing CLI and training inputs."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
from pyproj import Transformer
import rasterio
from rasterio import Affine
from shapely.affinity import affine_transform
from shapely.geometry import Polygon, box
from shapely.ops import transform

from scripts.analysis_config import RasterCacheGrid, load_analysis_configuration
from scripts.build_spatial_sample import build_spatial_sample
from scripts.habitat_training import (
    ReferenceConditionConfiguration, prepare_reference_condition_data,
)
from scripts.prepared_rasters import RasterBand, open_prepared_rasters
from scripts.prepared_sampling import sample_prepared_rasters
from scripts.raster_cache_utils import (
    AnalysisCacheTiles, CachedRasterTile, select_intersecting_tiles,
)


class PreparedSamplingTests(unittest.TestCase):
    """Exercise real input files, scientific parity and the callable boundary."""

    def setUp(self) -> None:
        """Create isolated prepared files and shared sampling settings."""
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.settings = dict(
            block_size_meters=300, samples_per_class_per_block=3,
            random_seed=17, tile_size_pixels=4, tile_origin=(11, -17),
        )

    def write_raster(
        self, name: str, values: np.ndarray, grid: Affine,
    ) -> Path:
        """Write fixture bands on an equal-area grid, with -9999 as NoData.

        Args:
            name: Filename in this test's temporary directory.
            values: Band, row and column array.
            grid: Pixel-to-EPSG:6933 transform.

        Returns:
            Path to the written GeoTIFF.
        """
        path = self.directory / name
        count, height, width = values.shape
        with rasterio.open(
            path, "w", driver="GTiff", count=count, width=width, height=height,
            dtype=values.dtype, crs="EPSG:6933", transform=grid, nodata=-9999,
        ) as destination:
            destination.write(values)
        return path

    def test_matches_cli_with_cropped_edges_holes_and_missing_data(self) -> None:
        """Preserve capped source pixels and weights across virtual tile edges."""
        for pixel_aoi in (
            box(1.2, 1.2, 10.8, 6.8).difference(box(4.2, 2.2, 6.8, 4.8)),
            Polygon([(1.5, 1.5), (10.5, 6.5), (1.5, 6.5)]),
        ):
            with self.subTest(aoi=pixel_aoi.wkt):
                self.compare_with_cli(pixel_aoi)

    def compare_with_cli(self, pixel_aoi: Polygon) -> None:
        """Compare a cropped source against its original complete CLI tiles.

        Args:
            pixel_aoi: AOI in the uncropped fixture's pixel coordinates.
        """
        default = load_analysis_configuration()
        definitions = tuple(
            next(band for band in default.bands if band.role == role)
            for role in ("reference", "predictor", "response")
        )
        grid = RasterCacheGrid("EPSG:6933", 30, 4, origin_x=11, origin_y=-17)
        configuration = replace(
            default, grid=grid, bands=definitions,
            sampling=replace(default.sampling, block_size_meters=300,
                             samples_per_class_per_block=3, random_seed=17),
        )
        names = configuration.band_names()
        full_transform = Affine(30, 0, -109, 0, -30, 223)
        rows, columns = np.indices((8, 12))
        values = np.stack((
            ((rows + columns) % 3 == 0).astype(float),
            columns + rows * 0.1, 100 + columns * 0.5 + rows,
        ))
        values[1, 2:5, 2] = -9999
        values[1:, 5, 8] = -9999
        values[2, 2:5, 8] = np.nan
        values[0, 5, 6] = -9999
        aoi = affine_transform(pixel_aoi, [30, 0, 0, -30, -109, 223])
        to_wgs84 = Transformer.from_crs(6933, 4326, always_xy=True)
        wgs84_aoi = transform(to_wgs84.transform, aoi)
        aoi, tiles = select_intersecting_tiles(wgs84_aoi, grid)
        cached_tiles = []
        for tile in tiles:
            column = round((tile.left - full_transform.c) / 30)
            row = round((full_transform.f - tile.top) / 30)
            path = self.write_raster(
                f"{tile.tile_id}.tif", values[:, row:row + 4, column:column + 4],
                Affine(30, 0, tile.left, 0, -30, tile.top),
            )
            cached_tiles.append(CachedRasterTile(tile, path, path.stat().st_size, ""))
        cache_inputs = AnalysisCacheTiles(
            "fixture", self.directory / "unused.json", wgs84_aoi, aoi,
            tuple(cached_tiles), sum(tile.file_size_bytes for tile in cached_tiles),
        )
        expected, _ = build_spatial_sample(configuration, cache_inputs, False)
        cropped_transform = full_transform * Affine.translation(1, 1)
        path = self.write_raster("prepared.tif", values[:, 1:7, 1:11], cropped_transform)
        mask_path = self.write_raster("assessment.tif", np.zeros((1, 6, 10)), cropped_transform)
        bands = tuple(RasterBand(name, path, i) for i, name in enumerate(names[1:], 2))
        with open_prepared_rasters(
            bands, RasterBand("reference", path), RasterBand("assessment", mask_path),
        ) as reader:
            actual = sample_prepared_rasters(reader, aoi, **self.settings)
            repeated = sample_prepared_rasters(reader, aoi, **self.settings)
        expected_table = expected.table.copy()
        expected_table["row"] = (expected_table["row"] + 7).astype(np.int64)
        expected_table["column"] = (expected_table["column"] + 3).astype(np.int64)
        pd.testing.assert_frame_equal(actual, expected_table[list(actual.columns)])
        pd.testing.assert_frame_equal(actual, repeated)
        self.assertGreater(len(actual), 0)
        self.assertLess(len(actual), 40)
        self.assertGreater(actual["reference_site"].sum(), 0)
        training = prepare_reference_condition_data(
            actual, ReferenceConditionConfiguration(2, 300, 600, 0.0, 1.0, 3),
            (names[1],), names[1],
        )
        np.testing.assert_array_equal(training.table["reference_site"], actual["reference_site"])
        self.assertEqual(len(training.table), len(actual))
        self.assertTrue(training.table["spatial_fold"].isin([1, 2]).all())

    def test_affine_grid_uses_pixel_centers_and_equal_area_weights(self) -> None:
        """Sample sheared, rotated and offset grids without square-pixel assumptions."""
        grid = Affine(30, 7, -107.3, 5, -20, 137.1)
        rows, columns = np.indices((5, 7))
        values = np.stack((columns + rows * 0.1, (rows % 2).astype(float)))
        path = self.write_raster("affine.tif", values, grid)
        aoi = affine_transform(box(0.1, 1.1, 5.9, 4.9), [grid.a, grid.b, grid.d, grid.e, grid.c, grid.f])
        with open_prepared_rasters(
            [RasterBand("response", path)], RasterBand("reference", path, 2),
        ) as reader:
            actual = sample_prepared_rasters(
                reader, aoi, **{**self.settings, "samples_per_class_per_block": 100},
            ).sort_values(["row", "column"])
        selected_rows, selected_columns = np.indices((4, 6))
        selected_rows = selected_rows.ravel() + 1
        selected_columns = selected_columns.ravel()
        np.testing.assert_array_equal(actual["row"], selected_rows)
        np.testing.assert_array_equal(actual["column"], selected_columns)
        np.testing.assert_array_equal(actual["response"], values[0, selected_rows, selected_columns])
        np.testing.assert_array_equal(actual["reference_site"], selected_rows % 2)
        x, y = grid * (selected_columns + 0.5, selected_rows + 0.5)
        np.testing.assert_array_equal(actual["sampling_block_column"], np.floor(x / 300))
        np.testing.assert_array_equal(actual["sampling_block_row"], np.floor(y / 300))
        np.testing.assert_array_equal(actual["sampling_weight"], np.ones(24))
        np.testing.assert_array_equal(actual["area_weight_m2"], np.full(24, abs(grid.determinant)))

    def test_progress_and_cancellation_leave_reader_with_its_caller(self) -> None:
        """Report both passes and allow the caller to cancel between reads."""
        grid = Affine(30, 0, 11, 0, -30, 103)
        path = self.write_raster("progress.tif", np.ones((2, 4, 8)), grid)
        aoi = box(12, -16, 250, 102)
        events: list[tuple[str, int, int]] = []

        def report(stage: str, completed: int, total: int) -> None:
            """Record progress for assertion after sampling.

            Args:
                stage: Sampling pass label.
                completed: Windows finished in this pass.
                total: Windows in this pass.
            """
            events.append((stage, completed, total))

        def cancel(stage: str, completed: int, total: int) -> None:
            """Cancel after one window has been processed.

            Args:
                stage: Sampling pass label.
                completed: Windows finished in this pass.
                total: Windows in this pass.

            Raises:
                InterruptedError: After the first window.
            """
            if completed == 1:
                raise InterruptedError("cancel sample")

        with open_prepared_rasters(
            [RasterBand("response", path)], RasterBand("reference", path, 2),
        ) as reader:
            sample_prepared_rasters(reader, aoi, progress=report, **self.settings)
            with self.assertRaisesRegex(InterruptedError, "cancel sample"):
                sample_prepared_rasters(reader, aoi, progress=cancel, **self.settings)
            self.assertGreater(len(sample_prepared_rasters(reader, aoi, **self.settings)), 0)
        for stage in ("Selecting sample pixels", "Reading sample values"):
            completed = [event[1] for event in events if event[0] == stage]
            total = next(event[2] for event in events if event[0] == stage)
            self.assertEqual(completed, list(range(total + 1)))

    def test_empty_area_and_conflicting_band_names_fail_at_sampling_boundary(self) -> None:
        """Reject empty sampling work and names that would overwrite sample weights."""
        grid = Affine(30, 0, 11, 0, -30, 103)
        path = self.write_raster("boundary.tif", np.ones((2, 4, 4)), grid)
        with open_prepared_rasters(
            [RasterBand("response", path)], RasterBand("reference", path, 2),
        ) as reader:
            for aoi in (Polygon(), box(1000, 1000, 1100, 1100)):
                with self.subTest(aoi=aoi), self.assertRaises(ValueError):
                    sample_prepared_rasters(reader, aoi, **self.settings)
        with open_prepared_rasters(
            [RasterBand("area_weight_m2", path)], RasterBand("reference", path, 2),
        ) as reader:
            with self.assertRaises(ValueError):
                sample_prepared_rasters(reader, box(12, -16, 130, 102), **self.settings)

    def test_import_does_not_load_acquisition_or_cli_modules(self) -> None:
        """Keep the callable sampling path independent of cache and report code."""
        result = subprocess.run(
            [sys.executable, "-B", "-c", (
                "import sys; from scripts.prepared_sampling import sample_prepared_rasters; "
                "forbidden = {'scripts.analysis_config', 'scripts.raster_cache_utils', "
                "'scripts.build_spatial_sample', 'matplotlib', 'cartopy', 'ee'}; "
                "assert not forbidden.intersection(sys.modules)"
            )],
            cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
