"""Tests for converting numeric raster bands to compressed binary masks."""

from __future__ import annotations

import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import rasterio
from rasterio.transform import from_origin

from scripts.utils.raster_to_binary_mask import (
    convert_raster_to_binary_mask,
    main,
    parse_comparison_expression,
)


class RasterToBinaryMaskTest(unittest.TestCase):
    """Verify comparison parsing, mask values, grids, compression, and COGs."""

    def setUp(self) -> None:
        """Create an isolated directory and a stable synthetic raster grid."""

        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.temporary_path = Path(self.temporary_directory.name)
        self.transform = from_origin(-120.0, 50.0, 0.25, 0.25)
        self.crs = "EPSG:4326"

    def _write_raster(
        self,
        path: Path,
        values: np.ndarray,
        *,
        nodata: int | float | None = None,
    ) -> None:
        """Write one or more bands on the test grid."""

        band_values = values[np.newaxis, ...] if values.ndim == 2 else values
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            width=band_values.shape[2],
            height=band_values.shape[1],
            count=band_values.shape[0],
            dtype=band_values.dtype,
            crs=self.crs,
            transform=self.transform,
            nodata=nodata,
        ) as destination:
            destination.write(band_values)

    def test_parses_supported_numeric_comparisons(self) -> None:
        """Accept ordered operators, signs, decimals, and scientific notation."""

        cases = {
            ">80": (">", 80.0, ">80"),
            " < 0.25 ": ("<", 0.25, "<0.25"),
            ">=-2.5e-3": (">=", -0.0025, ">=-2.5e-3"),
            "<=+4.": ("<=", 4.0, "<=+4."),
            "==.5": ("==", 0.5, "==.5"),
        }

        for expression, expected in cases.items():
            with self.subTest(expression=expression):
                comparison = parse_comparison_expression(expression)
                self.assertEqual(expected[0], comparison.operator)
                self.assertEqual(expected[1], comparison.threshold)
                self.assertEqual(expected[2], comparison.expression)

    def test_rejects_malformed_or_nonfinite_comparisons(self) -> None:
        """Reject ambiguous operators, missing thresholds, and nonfinite values."""

        for expression in ("3", "=>3", "!=3", ">=", ">=nan", "<inf", ">1 2"):
            with self.subTest(expression=expression):
                with self.assertRaises(ValueError):
                    parse_comparison_expression(expression)

    def test_converts_integer_values_and_source_nodata_to_zero_or_one(self) -> None:
        """Write a one-bit ZSTD mask while preserving the complete source grid."""

        input_path = self.temporary_path / "integer.tif"
        output_path = self.temporary_path / "integer_mask.tif"
        values = np.array(
            [[-2, 0, 3, -9999], [4, 2, 1, 7]],
            dtype=np.int16,
        )
        self._write_raster(input_path, values, nodata=-9999)

        summary = convert_raster_to_binary_mask(
            input_path,
            output_path,
            ">=3",
            window_size_pixels=2,
            show_progress=False,
        )

        with rasterio.open(input_path) as source, rasterio.open(output_path) as mask:
            np.testing.assert_array_equal(
                mask.read(1),
                np.array([[0, 0, 1, 0], [1, 0, 0, 1]], dtype=np.uint8),
            )
            self.assertEqual((source.width, source.height), (mask.width, mask.height))
            self.assertEqual(source.crs, mask.crs)
            self.assertEqual(source.transform, mask.transform)
            self.assertEqual(("uint8",), mask.dtypes)
            self.assertIsNone(mask.nodata)
            self.assertEqual("ZSTD", mask.compression.value.upper())
            self.assertEqual("1", mask.tags(1, ns="IMAGE_STRUCTURE")["NBITS"])
            self.assertEqual(">=3", mask.tags()["comparison_expression"])
        self.assertEqual(3, summary.true_pixels)
        self.assertEqual(5, summary.false_pixels)
        self.assertEqual(1, summary.invalid_source_pixels)
        self.assertFalse(summary.cog)

    def test_treats_float_nan_and_infinity_as_false(self) -> None:
        """Keep every output value binary even when float input is nonfinite."""

        input_path = self.temporary_path / "float.tif"
        output_path = self.temporary_path / "float_mask.tif"
        values = np.array(
            [[np.nan, -np.inf, -0.5, 0.0], [0.25, 0.5, np.inf, -1.0]],
            dtype=np.float32,
        )
        self._write_raster(input_path, values)

        summary = convert_raster_to_binary_mask(
            input_path,
            output_path,
            "<0",
            window_size_pixels=3,
            show_progress=False,
        )

        with rasterio.open(output_path) as mask:
            np.testing.assert_array_equal(
                mask.read(1),
                np.array([[0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.uint8),
            )
        self.assertEqual(2, summary.true_pixels)
        self.assertEqual(3, summary.invalid_source_pixels)

    def test_creates_zstd_cog_from_selected_band(self) -> None:
        """Use the requested source band and verify GDAL COG layout metadata."""

        input_path = self.temporary_path / "multiband.tif"
        output_path = self.temporary_path / "selected_band_mask.tif"
        second_band = np.indices((1024, 1024)).sum(axis=0).astype(np.uint8) % 3
        values = np.stack(
            [np.zeros((1024, 1024), dtype=np.uint8), second_band]
        )
        self._write_raster(input_path, values)
        with rasterio.open(input_path, "r+") as source:
            source.update_tags(2, ecological_role="classification source")

        summary = convert_raster_to_binary_mask(
            input_path,
            output_path,
            "==2",
            source_band=2,
            cog=True,
            window_size_pixels=256,
            show_progress=False,
        )

        with rasterio.open(output_path) as mask:
            np.testing.assert_array_equal(mask.read(1), second_band == 2)
            image_structure = mask.tags(ns="IMAGE_STRUCTURE")
            self.assertEqual("COG", image_structure["LAYOUT"])
            self.assertEqual("ZSTD", mask.compression.value.upper())
            self.assertEqual("1", mask.tags(1, ns="IMAGE_STRUCTURE")["NBITS"])
            self.assertEqual("2", mask.tags(1)["source_band"])
            self.assertEqual(
                "classification source", mask.tags(1)["ecological_role"]
            )
        self.assertTrue(summary.cog)

    def test_cli_requires_overwrite_for_an_existing_output(self) -> None:
        """Exercise the command interface and protect an existing destination."""

        input_path = self.temporary_path / "cli_input.tif"
        output_path = self.temporary_path / "cli_mask.tif"
        self._write_raster(
            input_path,
            np.array([[0, 1], [2, 3]], dtype=np.uint8),
        )
        arguments = [
            "raster_to_binary_mask.py",
            str(input_path),
            str(output_path),
            ">=2",
            "--no-progress",
        ]

        standard_output = io.StringIO()
        with patch.object(sys, "argv", arguments), contextlib.redirect_stdout(
            standard_output
        ):
            main()

        self.assertTrue(output_path.exists())
        self.assertIn("Comparison: >=2", standard_output.getvalue())
        with self.assertRaises(FileExistsError):
            convert_raster_to_binary_mask(
                input_path,
                output_path,
                ">=2",
                show_progress=False,
            )

    def test_rejects_same_path_and_out_of_range_band(self) -> None:
        """Reject destructive path collisions and unavailable source bands."""

        input_path = self.temporary_path / "single_band.tif"
        self._write_raster(input_path, np.ones((2, 2), dtype=np.uint8))

        with self.assertRaises(ValueError):
            convert_raster_to_binary_mask(
                input_path,
                input_path,
                "==1",
                show_progress=False,
            )
        with self.assertRaises(ValueError):
            convert_raster_to_binary_mask(
                input_path,
                self.temporary_path / "mask.tif",
                "==1",
                source_band=2,
                show_progress=False,
            )


if __name__ == "__main__":
    unittest.main()
