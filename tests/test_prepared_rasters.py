"""Read prepared model inputs through real Rasterio datasets."""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
import tempfile
import subprocess
import sys
import unittest
from unittest.mock import patch

import numpy as np
import rasterio
from rasterio import Affine
from rasterio.io import DatasetReader
from rasterio.transform import from_bounds, from_origin
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform_bounds
from rasterio.windows import Window

from scripts.prepared_rasters import RasterBand, open_prepared_rasters


class PreparedRasterTests(unittest.TestCase):
    """Check admitted grids, band ordering, masks and file-handle cleanup."""

    def setUp(self) -> None:
        """Create aligned inputs with distinct bands and a binary reference mask."""
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.directory = Path(self.temporary_directory.name)
        self.transform = from_origin(0, 4000, 1000, 1000)
        self.base_values = np.arange(16, dtype=np.float64).reshape(4, 4)
        self.data_path = self._write_raster(
            "data.tif", np.stack((self.base_values, 100 + self.base_values)),
        )
        self.reference_values = np.array([
            [0, 1, 2, 255], [1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 0, 1],
        ], dtype=np.uint8)
        self.reference_path = self._write_raster(
            "reference.tif", self.reference_values[None], nodata=255,
        )
        self.bands = (RasterBand("first", self.data_path),)
        self.reference = RasterBand("reference", self.reference_path)

    def _write_raster(
        self,
        filename: str,
        values: np.ndarray,
        *,
        transform: Affine | None = None,
        crs: str | None = "EPSG:6933",
        nodata: float | int | None = None,
    ) -> Path:
        """Write fixture bands with the requested geospatial metadata.

        Args:
            filename: Name inside this test's temporary directory.
            values: Band, row and column array to write.
            transform: Pixel transform; omitted to use the aligned fixture grid.
            crs: Coordinate reference system, or None for missing CRS metadata.
            nodata: Value marked missing in every band.

        Returns:
            Path to the created GeoTIFF.
        """
        count, height, width = values.shape
        path = self.directory / filename
        with rasterio.open(
            path, "w", driver="GTiff", width=width, height=height, count=count,
            dtype=values.dtype, crs=crs,
            transform=self.transform if transform is None else transform,
            nodata=nodata,
        ) as destination:
            destination.write(values)
        return path

    def test_reads_named_bands_in_order_and_normalizes_missing_values(self) -> None:
        """Preserve selected band order and make unavailable pixel values NaN."""
        values = np.stack((self.base_values, 100 + self.base_values))
        values[0, 1, 1] = -9999
        values[1, 1, 2] = np.nan
        values[1, 2, 2] = np.inf
        values[1, 2, 3] = -np.inf
        data_path = self._write_raster("missing.tif", values, nodata=-9999)
        other_path = self._write_raster("other.tif", (1000 + self.base_values)[None])
        bands = (
            RasterBand("other", other_path),
            RasterBand("second", data_path, 2),
            RasterBand("first", data_path),
        )
        with open_prepared_rasters(bands, self.reference) as reader:
            self.assertEqual(("other", "second", "first"), reader.band_names)
            self.assertEqual((4, 4), (reader.width, reader.height))
            self.assertEqual(rasterio.crs.CRS.from_epsg(6933), reader.crs)
            self.assertEqual(self.transform, reader.transform)
            result = reader.read_window(Window(1, 1, 3, 2))
        self.assertEqual(np.dtype("float64"), result.values.dtype)
        np.testing.assert_array_equal(result.values, [
            [[1005., 1006., 1007.], [1009., 1010., 1011.]],
            [[105., np.nan, 107.], [109., np.nan, np.nan]],
            [[np.nan, 6., 7.], [9., 10., 11.]],
        ])
        self.assertEqual(from_origin(1000, 3000, 1000, 1000), result.transform)
        np.testing.assert_array_equal(result.reference_mask, [[True, False, False], [True, True, False]])
        self.assertEqual(np.dtype("bool"), result.reference_mask.dtype)
        self.assertIsNone(result.assessment_mask)

    def test_reference_mask_uses_selected_band_and_excludes_nodata_one(self) -> None:
        """Exclude missing pixels even when the NoData sentinel is one."""
        for nodata in (255, 1):
            with self.subTest(nodata=nodata):
                path = self._write_raster(
                    f"reference-{nodata}.tif",
                    np.stack((np.ones((4, 4), dtype=np.uint8), self.reference_values)),
                    nodata=nodata,
                )
                reference = RasterBand("reference", path, 2)
                with open_prepared_rasters(self.bands, reference) as reader:
                    result = reader.read_window(Window(0, 0, 4, 4))
                expected = (
                    self.reference_values == 1
                    if nodata == 255 else np.zeros((4, 4), dtype=bool)
                )
                np.testing.assert_array_equal(result.reference_mask, expected)

    def test_aligns_assessment_mask_with_nearest_neighbor_and_preserves_validity(self) -> None:
        """Read a coarse mask in another CRS using only its defined ones."""
        bounds = transform_bounds("EPSG:6933", "EPSG:3857", 0, 0, 4000, 4000)
        mask_transform = from_bounds(*bounds, 2, 2)
        mask_values = np.array([[[1, 0], [2, 255]]], dtype=np.uint8)
        for nodata in (255, 1):
            with self.subTest(nodata=nodata):
                path = self._write_raster(
                    f"assessment-{nodata}.tif", mask_values,
                    transform=mask_transform, crs="EPSG:3857", nodata=nodata,
                )
                with open_prepared_rasters(
                    self.bands, self.reference, RasterBand("assessment", path),
                ) as reader:
                    result = reader.read_window(Window(0, 0, 4, 4))
                expected = np.zeros((4, 4), dtype=bool)
                if nodata == 255:
                    expected[:2, :2] = True
                np.testing.assert_array_equal(result.assessment_mask, expected)
                self.assertEqual(np.dtype("bool"), result.assessment_mask.dtype)
                np.testing.assert_array_equal(result.reference_mask, self.reference_values == 1)

    def test_rejects_invalid_input_metadata_when_opened(self) -> None:
        """Reject incompatible grids, missing bands and ambiguous input names."""
        wrong_crs = self._write_raster("wrong-crs.tif", self.base_values[None], crs="EPSG:3857")
        missing_crs = self._write_raster("missing-crs.tif", self.base_values[None], crs=None)
        shifted = self._write_raster(
            "shifted.tif", self.base_values[None],
            transform=from_origin(1000, 4000, 1000, 1000),
        )
        smaller = self._write_raster("smaller.tif", self.base_values[None, :, :3])
        almost_aligned = self._write_raster(
            "almost-aligned.tif", self.base_values[None],
            transform=from_origin(0.000001, 4000, 1000, 1000),
        )
        cases = (
            ("wrong CRS", (RasterBand("bad", wrong_crs),), self.reference, None),
            ("shifted data", (*self.bands, RasterBand("bad", shifted)), self.reference, None),
            ("different dimensions", (*self.bands, RasterBand("bad", smaller)), self.reference, None),
            ("slightly shifted data", (*self.bands, RasterBand("bad", almost_aligned)), self.reference, None),
            ("shifted reference", self.bands, RasterBand("reference", shifted), None),
            ("missing data band", (RasterBand("bad", self.data_path, 3),), self.reference, None),
            ("missing reference band", self.bands, RasterBand("reference", self.reference_path, 2), None),
            ("missing mask CRS", self.bands, self.reference, RasterBand("assessment", missing_crs)),
            ("missing assessment band", self.bands, self.reference, RasterBand("assessment", self.reference_path, 2)),
            ("empty name", (RasterBand("", self.data_path),), self.reference, None),
            ("duplicate name", (*self.bands, RasterBand("first", self.data_path, 2)), self.reference, None),
        )
        for label, bands, reference, assessment in cases:
            with self.subTest(label=label), self.assertRaises(ValueError):
                with open_prepared_rasters(bands, reference, assessment):
                    self.fail("Invalid inputs were admitted.")

    def test_closes_open_files_after_exit_and_failures(self) -> None:
        """Close every acquired dataset when the caller or a later input fails."""
        shifted_path = self._write_raster(
            "shifted-cleanup.tif", self.base_values[None],
            transform=from_origin(1000, 4000, 1000, 1000),
        )
        original_open = rasterio.open
        opened: list[DatasetReader | WarpedVRT] = []

        def record_open(path: str | Path, *args: object, **kwargs: object) -> DatasetReader:
            """Open a real fixture file and retain its handle to check cleanup.

            Args:
                path: File requested by the reader.
                *args: Additional Rasterio arguments.
                **kwargs: Additional Rasterio keyword arguments.

            Returns:
                The open Rasterio dataset.
            """
            source = original_open(path, *args, **kwargs)
            opened.append(source)
            return source

        def record_vrt(source: DatasetReader, **kwargs: object) -> WarpedVRT:
            """Retain the real aligned mask view to verify it also closes.

            Args:
                source: Open assessment-mask source.
                **kwargs: Rasterio alignment arguments.

            Returns:
                The aligned Rasterio mask view.
            """
            aligned = WarpedVRT(source, **kwargs)
            opened.append(aligned)
            return aligned

        cases = (
            ("normal", self.bands, None),
            ("caller", self.bands, RuntimeError),
            ("missing file", (*self.bands, RasterBand("missing", self.directory / "absent.tif")), rasterio.errors.RasterioIOError),
            ("bad grid", (*self.bands, RasterBand("shifted", shifted_path)), ValueError),
        )
        for label, bands, expected_error in cases:
            opened.clear()
            expectation = self.assertRaises(expected_error) if expected_error else nullcontext()
            with self.subTest(label=label), patch(
                "scripts.prepared_rasters.rasterio.open", side_effect=record_open,
            ), patch("scripts.prepared_rasters.WarpedVRT", side_effect=record_vrt), expectation:
                with open_prepared_rasters(bands, self.reference, self.reference) as reader:
                    self.assertTrue(opened)
                    self.assertTrue(all(not source.closed for source in opened))
                    self.assertTrue(any(isinstance(source, WarpedVRT) for source in opened))
                    reader.read_window(Window(0, 0, 1, 1))
                    if label == "caller":
                        raise RuntimeError("Caller interrupted the calculation.")
            self.assertTrue(opened)
            self.assertTrue(all(source.closed for source in opened))

    def test_reads_without_cli_cache_or_plotting_imports(self) -> None:
        """Read prepared files in a fresh process without importing workflows."""
        script = f'''
from pathlib import Path
import sys
from rasterio.windows import Window
from scripts.prepared_rasters import RasterBand, open_prepared_rasters
with open_prepared_rasters(
    [RasterBand("values", Path({str(self.data_path)!r}))],
    RasterBand("reference", Path({str(self.reference_path)!r})),
) as reader:
    result = reader.read_window(Window(0, 0, 1, 1))
    assert result.values[0, 0, 0] == 0
for name in sys.modules:
    assert not (name.startswith("scripts.") and name != "scripts.prepared_rasters"), name
    assert name.split(".")[0] not in {{"matplotlib", "cartopy", "ee", "tomllib"}}, name
'''
        result = subprocess.run(
            [sys.executable, "-B", "-c", script],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True, text=True, check=False,
        )
        self.assertEqual(0, result.returncode, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
