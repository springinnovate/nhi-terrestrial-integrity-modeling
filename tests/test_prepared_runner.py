"""Run Habitat condition from real prepared rasters through written GeoTIFFs."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import fields
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
import rasterio
from rasterio import Affine
from rasterio.features import geometry_mask
from shapely.affinity import affine_transform
from shapely.geometry import box, mapping

from scripts import habitat_condition
from scripts.habitat_training import IntegrityConfiguration
from scripts.prepared_rasters import RasterBand
from scripts.prepared_runner import HabitatConditionRun, run_habitat_condition


class PreparedRunnerTests(unittest.TestCase):
    """Check model wiring, masks, output placement and run-owned cleanup."""

    def setUp(self) -> None:
        """Write a small reference/background stack with controlled data gaps."""
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.grid = Affine(1000, 0, -6000, 0, -1000, 12000)
        rows, columns = np.indices((24, 24))
        temperature = rows * 0.3 + columns * 0.07
        category = (rows + columns) % 3
        self.reference = (rows < 5).astype(float)
        canopy = 1 + 0.3 * temperature + 0.1 * category + 0.04 * np.sin(rows + columns)
        productivity = 2 + 0.2 * temperature - 0.15 * category + 0.03 * np.cos(rows * 2 + columns)
        canopy += 0.15 * (columns % 3) * (1 - self.reference)
        productivity += 0.1 * (rows % 3) * (1 - self.reference)
        self.values = np.stack((canopy, category, productivity, temperature, self.reference))
        self.values[1, 6, 2] = -9999
        self.values[[1, 3], 6, 3] = -9999
        self.values[0, 6, 8] = -9999
        self.data_path = self.write_raster("inputs.tif", self.values)
        self.reference_band = RasterBand("reference", self.data_path, 5)
        self.assessment = (rows >= 5).astype(float)
        self.assessment[6, 4:8] = [0, 2, np.nan, -9999]
        assessment_path = self.write_raster("assessment.tif", self.assessment[None])
        self.assessment_band = RasterBand("assessment", assessment_path)
        self.predictors = (
            RasterBand("category", self.data_path, 2),
            RasterBand("temperature", self.data_path, 4),
        )
        self.responses = (
            RasterBand("productivity", self.data_path, 3),
            RasterBand("canopy", self.data_path),
        )
        pixels = (box(1.1, 1.1, 22.9, 10.9)
                  .difference(box(1.1, 1.1, 3.9, 3.9))
                  .difference(box(10.1, 7.1, 12.9, 8.9)))
        self.aoi = affine_transform(pixels, [1000, 0, 0, -1000, -6000, 12000])
        self.configuration = IntegrityConfiguration(
            fold_count=3, sampling_block_size_meters=2000,
            validation_block_size_meters=4000, minimum_predictor_coverage=0.8,
            maximum_row_missing_fraction=0.5, minimum_response_coverage=0.5,
            spline_knot_count=3, ridge_alpha=1.0,
        )

    def write_raster(self, filename: str, values: np.ndarray) -> Path:
        """Write aligned fixture bands with -9999 as NoData.

        Args:
            filename: Filename within the test directory.
            values: Band, row and column values.

        Returns:
            Path to the GeoTIFF.
        """
        path = self.directory / filename
        with rasterio.open(
            path, "w", driver="GTiff", count=len(values), width=24, height=24,
            dtype="float64", crs="EPSG:6933", transform=self.grid, nodata=-9999,
        ) as destination:
            destination.write(values)
        return path

    def run_model(
        self, name: str, assessment: RasterBand | None = None,
        progress: Callable[[str, int, int], None] | None = None,
    ) -> HabitatConditionRun:
        """Calculate this fixture into a new directory.

        Args:
            name: Output directory name within the test directory.
            assessment: Optional mask limiting predictions.
            progress: Optional supervising callback.

        Returns:
            Completed run and its output paths.
        """
        return run_habitat_condition(
            self.predictors, self.responses, self.reference_band, self.aoi,
            self.directory / name, self.configuration,
            categorical_predictor_name="category", samples_per_class_per_block=100,
            random_seed=17, tile_size_pixels=8, window_size_pixels=5,
            covariance_shrinkage=0.1, assessment_mask=assessment, progress=progress,
        )

    def test_writes_similarity_and_diagnostics_with_both_mask_choices(self) -> None:
        """Match CLI window calculations while writing an AOI-aligned crop."""
        from rasterio.coords import BoundingBox
        from rasterio.crs import CRS
        from scripts.apply_reference_condition_models import (
            InferenceOutputGrid, ReferenceDepartureCalibration, ResponseModel,
            calculate_inference_tile,
        )
        from scripts.raster_cache_utils import CachedRasterTile, CacheTile

        previous_sample = None
        for assessment in (None, self.assessment_band):
            with self.subTest(assessment=assessment is not None):
                events: list[tuple[str, int, int]] = []

                def report(phase: str, completed: int, total: int) -> None:
                    """Record progress for this run.

                    Args:
                        phase: Current calculation phase.
                        completed: Completed steps in the phase.
                        total: Steps in the phase.
                    """
                    events.append((phase, completed, total))

                run = self.run_model(f"output-{assessment is not None}", assessment, report)
                fitted = run.fitted
                self.assertEqual(("temperature", "category"), fitted.predictor_names)
                self.assertEqual(("productivity", "canopy"), tuple(fitted.responses))
                if previous_sample is not None:
                    pd.testing.assert_frame_equal(fitted.prepared_data.table, previous_sample)
                previous_sample = fitted.prepared_data.table
                crop_transform = self.grid * Affine.translation(1, 1)
                inside = geometry_mask(
                    [mapping(self.aoi)], out_shape=(10, 22),
                    transform=crop_transform, invert=True,
                )
                response_names = tuple(fitted.responses)
                models = tuple(ResponseModel(
                    self.directory / "unused-model.joblib", name, name, name,
                    fitted.predictor_names, response.overall_metrics["weighted_rmse"], response.model,
                ) for name, response in fitted.responses.items())
                reference_rows = fitted.prepared_data.table["reference_site"].eq(1).to_numpy()
                complete = reference_rows & np.isfinite(np.column_stack([
                    response.standardized_departure for response in fitted.responses.values()
                ])).all(axis=1)
                weights = fitted.prepared_data.table["area_weight_m2"].to_numpy()
                calibration = ReferenceDepartureCalibration(
                    fitted.reference_distribution, self.directory / "unused-predictions.parquet",
                    response_names, 0.1, int(reference_rows.sum()), int(complete.sum()),
                    float(weights[reference_rows].sum()), float(weights[complete].sum()),
                )
                calculated = calculate_inference_tile(
                    CachedRasterTile(
                        CacheTile(0, 0, "x+000000_y+000000", -6000, -12000, 18000, 12000),
                        self.data_path, self.data_path.stat().st_size, "",
                    ), None if assessment is None else assessment.path, 5, self.aoi,
                    InferenceOutputGrid(
                        CRS.from_epsg(6933), crop_transform,
                        BoundingBox(-5000, 1000, 17000, 11000), 22, 10, 1000,
                    ), (4, 2, 3, 1, 5), fitted.predictor_names, models, calibration, 0.5, 4,
                )
                arrays = {}
                for field in fields(calculated.windows[0].prediction):
                    first = getattr(calculated.windows[0].prediction, field.name)
                    arrays[field.name] = np.empty((*first.shape[:-2], 10, 22), dtype=first.dtype)
                for computed in calculated.windows:
                    slices = computed.destination_window.toslices()
                    for name, values in arrays.items():
                        values[(..., *slices)] = getattr(computed.prediction, name)
                prediction = habitat_condition.RasterWindowPrediction(**arrays)
                similarity = np.where(
                    prediction.departure_percentile == habitat_condition.FLOAT_NODATA,
                    habitat_condition.FLOAT_NODATA, 1 - prediction.departure_percentile,
                )
                for path, expected, descriptions, nodata in (
                    (run.reference_similarity, similarity[None], ("Reference similarity",), -9999),
                    (run.expected_reference, prediction.expected_reference, response_names, -9999),
                    (run.observed_minus_expected, prediction.observed_minus_expected, response_names, -9999),
                    (run.standardized_deviation, prediction.standardized_deviation, response_names, -9999),
                    (run.inference_status, prediction.inference_status,
                     ("Prediction status", "Missing predictor count"), 255),
                ):
                    with rasterio.open(path) as source:
                        self.assertEqual(source.crs.to_epsg(), 6933)
                        self.assertEqual(source.transform, crop_transform)
                        self.assertEqual(source.shape, (10, 22))
                        self.assertEqual(source.descriptions, descriptions)
                        self.assertEqual(source.nodata, nodata)
                        actual = source.read()
                    self.assertEqual(actual.dtype, expected.dtype)
                    np.testing.assert_array_equal(actual, expected)
                valid = similarity != -9999
                self.assertGreater(np.count_nonzero(valid), 0)
                self.assertTrue(np.all((similarity[valid] >= 0) & (similarity[valid] <= 1)))
                self.assertTrue(np.all(similarity[self.reference[1:11, 1:23] == 1] == -9999))
                self.assertTrue(np.all(similarity[~inside] == -9999))
                self.assertEqual(similarity[5, 7], -9999)
                self.assertEqual(
                    [event for event in events if event[0] == "Predicting raster windows"],
                    [("Predicting raster windows", count, 10) for count in range(11)],
                )

    def test_cancellation_removes_partial_outputs_and_closes_files(self) -> None:
        """Cancel after writing, preserve unrelated files, and allow a fresh run."""
        marker = self.directory / "unrelated.txt"
        marker.write_text("keep", encoding="utf-8")

        def cancel(phase: str, completed: int, total: int) -> None:
            """Cancel after the first prediction window is written.

            Args:
                phase: Current calculation phase.
                completed: Completed steps in the phase.
                total: Steps in the phase.

            Raises:
                InterruptedError: After the first written window.
            """
            if phase == "Predicting raster windows" and completed == 1:
                raise InterruptedError("cancel this run")

        with self.assertRaisesRegex(InterruptedError, "cancel this run"):
            self.run_model("cancelled", progress=cancel)
        self.assertFalse((self.directory / "cancelled").exists())
        self.assertEqual(marker.read_text(encoding="utf-8"), "keep")
        with rasterio.open(self.data_path) as source:
            np.testing.assert_array_equal(source.read(), self.values)
        rerun = self.run_model("cancelled")
        with rasterio.open(rerun.reference_similarity) as source:
            self.assertGreater(source.read(1, masked=True).count(), 0)

    def test_refusing_existing_output_preserves_its_contents(self) -> None:
        """Keep pre-existing files outside the runner's cleanup ownership."""
        output = self.directory / "existing"
        output.mkdir()
        marker = output / "keep.txt"
        marker.write_text("keep", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            self.run_model("existing")
        self.assertEqual(marker.read_text(encoding="utf-8"), "keep")

    def test_import_does_not_load_cli_acquisition_or_plotting(self) -> None:
        """Keep the callable runner separate from CLI-specific workflows."""
        result = subprocess.run(
            [sys.executable, "-B", "-c", (
                "import sys; import scripts.prepared_runner; "
                "forbidden = {'scripts.analysis_config', 'scripts.raster_cache_utils', "
                "'scripts.apply_reference_condition_models', 'matplotlib', 'cartopy', 'ee'}; "
                "assert not forbidden.intersection(sys.modules)"
            )], cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
