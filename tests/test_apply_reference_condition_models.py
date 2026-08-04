"""Tests for tile-backed reference-condition raster inference."""

from __future__ import annotations

import contextlib
import io
import json
import math
import os
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from rasterio.io import MemoryFile
from rasterio.transform import from_bounds, from_origin
from rasterio.warp import transform_bounds
from shapely.geometry import box, mapping
from shapely.ops import transform

from scripts.apply_reference_condition_models import (
    FLOAT_NODATA,
    STATUS_INSUFFICIENT_PREDICTORS,
    STATUS_NODATA,
    STATUS_OUTSIDE_TARGET,
    STATUS_PREDICTED,
    build_reference_departure_calibration,
    load_response_models,
    parse_args,
    run_reference_condition_inference,
    write_computed_inference_tiles,
)
from scripts.fetch_gee_raster_tiles import cache_aoi_tiles
from scripts.fit_grassland_integrity_parameters import (
    IntegrityConfiguration,
    fit_response_gam,
    predict_expected_response,
)
from scripts.analysis_config import RasterCacheGrid, load_analysis_configuration
from scripts.raster_cache_utils import resolve_analysis_cache_tiles


class ApplyReferenceConditionModelsTest(unittest.TestCase):
    """Verify aligned outputs, calculations, missingness, and masking."""

    def setUp(self) -> None:
        """Create fitted response artifacts and a compact source raster."""

        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.temporary_path = Path(self.temporary_directory.name)
        self.model_run_directory = self.temporary_path / "model_run"
        self.model_directory = self.model_run_directory / "models"
        self.model_directory.mkdir(parents=True)
        base_configuration = load_analysis_configuration()
        configured_band_names = base_configuration.band_names()
        band_name_by_identifier = {
            band_definition.identifier: band_name
            for band_definition, band_name in zip(
                base_configuration.bands,
                configured_band_names,
                strict=True,
            )
        }
        self.predictor_names = tuple(
            band_name_by_identifier[identifier]
            for identifier in ("d20", "d21", "d22", "d23", "d35")
        )
        self.response_names = (
            band_name_by_identifier["d02"],
            band_name_by_identifier["d11"],
        )
        self.reference_name = band_name_by_identifier["d01"]
        self.response_rmse = {"d02": 2.0, "d11": 4.0}
        self.aoi_path = self.temporary_path / "synthetic_aoi.geojson"
        self.cache_directory = self.temporary_path / "raster_cache"
        self.grid = RasterCacheGrid(
            crs="EPSG:6933",
            pixel_size_meters=1_000,
            tile_size_pixels=4,
        )
        # Keep the vector just inside the outer pixel edges so projection round-
        # trip noise cannot select neighboring cache rows or columns. Every
        # output pixel center remains inside this AOI.
        projected_aoi = box(100, 100, 4_900, 3_900)
        to_wgs84 = Transformer.from_crs(
            self.grid.crs,
            "EPSG:4326",
            always_xy=True,
        )
        self.aoi_path.write_text(
            json.dumps(mapping(transform(to_wgs84.transform, projected_aoi))),
            encoding="utf-8",
        )
        self.analysis_configuration = replace(
            base_configuration,
            analysis_name="synthetic_prairie",
            display_name="Synthetic Prairie",
            aoi_path=self.aoi_path,
            earth_engine=replace(
                base_configuration.earth_engine,
                project="offline-test-project",
                cache_directory=self.cache_directory,
            ),
            grid=self.grid,
            inference=replace(
                base_configuration.inference,
                application_mask_path=None,
                window_size_pixels=2,
            ),
        )
        self.models = self._create_models()
        self.transform = from_origin(0, 4_000, 1_000, 1_000)
        self.source_values = self._create_source_values(width=5)
        with contextlib.redirect_stdout(io.StringIO()):
            cache_aoi_tiles(
                self.analysis_configuration,
                refresh=False,
                show_progress=False,
                tile_fetcher=self._create_tile_bytes,
            )
        self.analysis_cache_tiles = resolve_analysis_cache_tiles(
            self.analysis_configuration,
            show_progress=False,
        )
        (self.model_run_directory / "run_metadata.json").write_text(
            json.dumps(
                {
                    "ecoregion_name": "Synthetic Prairie",
                    "configuration": {"maximum_row_missing_fraction": 0.20},
                    "analysis_configuration": {
                        "sha256": self.analysis_configuration.configuration_sha256,
                    },
                }
            ),
            encoding="utf-8",
        )
        self._create_reference_prediction_table()

    def _create_reference_prediction_table(self) -> None:
        """Write a calibration table with an exact weighted reference CDF."""

        pd.DataFrame(
            {
                "reference_site": [1, 1, 1, 1, 1, 0],
                "area_weight_m2": [1_000, 1_000, 1_000, 1_000, 2_000, 1_000_000],
                "d02_standardized_deviation_oof": [-1, -1, 1, 1, 0, 100],
                "d11_standardized_deviation_oof": [-1, 1, -1, 1, 0, 100],
            }
        ).to_parquet(
            self.model_run_directory / "ecological_response_predictions.parquet",
            compression="zstd",
            index=False,
        )

    def _create_models(self) -> dict[str, dict[str, object]]:
        """Fit two response models sharing five raster predictors.

        Returns:
            Model bundles keyed by short response band.
        """

        row_count = 60
        row_offsets = np.arange(row_count, dtype=np.float64)
        training_table = pd.DataFrame(
            {
                self.predictor_names[0]: 2.0 + row_offsets * 0.08,
                self.predictor_names[1]: 10.0 + row_offsets * 0.12,
                self.predictor_names[2]: 1.0 + np.sin(row_offsets / 7.0),
                self.predictor_names[3]: 3.0 + np.cos(row_offsets / 9.0),
                self.predictor_names[4]: (row_offsets.astype(np.int64) % 3) + 1,
                "area_weight_m2": 1_000.0 + row_offsets * 10.0,
            }
        )
        environmental_signal = (
            0.6 * training_table[self.predictor_names[0]]
            + 0.2 * training_table[self.predictor_names[1]]
            - 0.4 * training_table[self.predictor_names[2]]
            + 0.3 * training_table[self.predictor_names[3]]
            + 0.5 * training_table[self.predictor_names[4]]
        )
        training_table[self.response_names[0]] = 5.0 + environmental_signal
        training_table[self.response_names[1]] = 20.0 + 2.0 * environmental_signal
        configuration = IntegrityConfiguration(
            fold_count=5,
            sampling_block_size_meters=25_000,
            validation_block_size_meters=100_000,
            minimum_predictor_coverage=0.80,
            maximum_row_missing_fraction=0.20,
            spline_knot_count=4,
            minimum_response_coverage=0.50,
            ridge_alpha=0.1,
        )
        models = {}
        for response_name, response_band in zip(
            self.response_names,
            ("d02", "d11"),
            strict=True,
        ):
            model = fit_response_gam(
                training_table,
                response_name,
                self.predictor_names[:4],
                self.predictor_names[4],
                {
                    predictor_name: float(training_table[predictor_name].median())
                    for predictor_name in self.predictor_names
                },
                configuration,
                self.analysis_configuration,
            )
            model["reference_residual_rmse_oof"] = self.response_rmse[response_band]
            model["standardized_deviation_interpretation"] = (
                "observed minus expected divided by cross-validated reference RMSE"
            )
            joblib.dump(
                model,
                self.model_directory
                / f"{response_band}_reference_condition_gam.joblib",
            )
            models[response_band] = model
        return models

    def _create_source_values(self, width: int) -> np.ndarray:
        """Create observed responses and predictors with controlled gaps.

        Args:
            width: Number of source columns to generate.

        Returns:
            Source values in raster-band, row, and column order.
        """

        height = 4
        row_grid, column_grid = np.indices((height, width), dtype=np.float32)
        predictor_values = np.stack(
            [
                2.5 + row_grid * 0.3 + column_grid * 0.1,
                11.0 + row_grid * 0.2 + column_grid * 0.15,
                1.2 + row_grid * 0.05,
                3.4 + column_grid * 0.07,
                ((row_grid + column_grid) % 3) + 1,
            ]
        ).astype(np.float32)
        predictor_table = pd.DataFrame(
            predictor_values.reshape(len(self.predictor_names), -1).T,
            columns=self.predictor_names,
        )
        expected_d02 = predict_expected_response(
            self.models["d02"],
            predictor_table,
        ).reshape(height, width)
        expected_d11 = predict_expected_response(
            self.models["d11"],
            predictor_table,
        ).reshape(height, width)
        response_values = np.stack(
            [
                expected_d02 + 1.0,
                expected_d11 - 2.0,
            ]
        ).astype(np.float32)
        reference_values = np.zeros((1, height, width), dtype=np.float32)
        reference_values[0, 1, 1] = 1.0
        reference_values[0, 2, 2] = 1.0
        source_values = np.concatenate(
            [response_values, predictor_values, reference_values]
        )
        source_values[0, 0, 2] = FLOAT_NODATA
        source_values[3, 0, 1] = FLOAT_NODATA
        source_values[3:7, 0, 0] = FLOAT_NODATA
        source_values[2, 0, 0] = predictor_values[0, 0, 0]

        source_values[source_values == FLOAT_NODATA] = np.nan
        return source_values

    def _create_tile_bytes(
        self,
        raster_stack,
        tile,
        cache_grid,
        band_names,
    ) -> bytes:
        """Encode one configured cache tile from deterministic source values."""

        del raster_stack
        full_source_values = self._create_source_values(width=8)
        configured_values = np.zeros(
            (
                len(band_names),
                cache_grid.tile_size_pixels,
                cache_grid.tile_size_pixels,
            ),
            dtype=np.float32,
        )
        source_by_name = {
            **{
                response_name: full_source_values[response_offset]
                for response_offset, response_name in enumerate(
                    self.response_names
                )
            },
            **{
                predictor_name: full_source_values[2 + predictor_offset]
                for predictor_offset, predictor_name in enumerate(
                    self.predictor_names
                )
            },
            self.reference_name: full_source_values[-1],
        }
        first_global_column = tile.column * cache_grid.tile_size_pixels
        last_global_column = (
            first_global_column + cache_grid.tile_size_pixels
        )
        for band_offset, band_name in enumerate(band_names):
            if band_name in source_by_name:
                configured_values[band_offset] = source_by_name[band_name][
                    :,
                    first_global_column:last_global_column,
                ]
            else:
                configured_values[band_offset] = band_offset + 1
        with MemoryFile() as memory_file:
            with memory_file.open(
                driver="GTiff",
                width=cache_grid.tile_size_pixels,
                height=cache_grid.tile_size_pixels,
                count=len(band_names),
                dtype="float32",
                crs=cache_grid.crs,
                transform=from_origin(
                    tile.left,
                    tile.top,
                    cache_grid.pixel_size_meters,
                    cache_grid.pixel_size_meters,
                ),
            ) as destination:
                destination.write(configured_values)
            return memory_file.read()

    def test_calibrates_weighted_reference_distance_and_percentile(self) -> None:
        """Use only weighted reference rows for covariance and empirical CDF."""

        _, response_models, _ = load_response_models(self.model_run_directory)
        calibration = build_reference_departure_calibration(
            self.model_run_directory,
            response_models,
            covariance_shrinkage=0.10,
        )

        np.testing.assert_allclose(calibration.reference_mean_vector, [0.0, 0.0])
        np.testing.assert_allclose(
            calibration.reference_covariance_matrix,
            np.diag([2.0 / 3.0, 2.0 / 3.0]),
        )
        mahalanobis_distances = calibration.calculate_mahalanobis_distances(
            np.array([[0.5, -0.5]], dtype=np.float64)
        )
        np.testing.assert_allclose(mahalanobis_distances, [math.sqrt(0.75)])
        np.testing.assert_allclose(
            calibration.calculate_reference_departure_percentiles(
                mahalanobis_distances
            ),
            [1.0 / 3.0],
        )
        self.assertEqual(5, calibration.reference_row_count)
        self.assertEqual(5, calibration.complete_reference_row_count)
        self.assertEqual(6_000.0, calibration.complete_reference_area_m2)

    def test_cli_rejects_legacy_positional_raster_stack(self) -> None:
        """Require TOML cache inputs instead of a monolithic GeoTIFF."""

        with (
            patch.object(
                sys,
                "argv",
                [
                    "apply_reference_condition_models",
                    str(self.analysis_configuration.path),
                    "data/raster_stacks/obsolete.tif",
                    str(self.model_run_directory),
                ],
            ),
            contextlib.redirect_stderr(io.StringIO()),
            self.assertRaises(SystemExit),
        ):
            parse_args()

    def test_writes_aligned_response_stacks_and_streaming_report(self) -> None:
        """Calculate expected, raw, and standardized values in raster windows."""

        output_directory = self.temporary_path / "unmasked_output"
        standard_output = io.StringIO()
        with (
            patch(
                "scripts.apply_reference_condition_models."
                "MAXIMUM_DISPLAY_DIMENSION",
                3,
            ),
            contextlib.redirect_stdout(standard_output),
        ):
            summary = run_reference_condition_inference(
                self.analysis_configuration,
                self.model_run_directory,
                output_directory=output_directory,
                show_progress=False,
                analysis_cache_tiles=self.analysis_cache_tiles,
            )

        self.assertEqual(2, summary.response_count)
        self.assertEqual(20, summary.raster_pixels)
        self.assertEqual(20, summary.target_pixels)
        self.assertEqual(19, summary.predicted_pixels)
        self.assertEqual(1, summary.insufficient_predictor_pixels)
        self.assertEqual(1, summary.imputed_pixels)
        self.assertEqual(
            "synthetic_prairie_expected_reference.tif",
            summary.expected_reference_path.name,
        )
        self.assertEqual(
            "synthetic_prairie_observed_minus_expected.tif",
            summary.observed_minus_expected_path.name,
        )
        self.assertEqual(
            "synthetic_prairie_standardized_deviation.tif",
            summary.standardized_deviation_path.name,
        )
        self.assertEqual(
            "synthetic_prairie_reference_departure_percentile.tif",
            summary.departure_percentile_path.name,
        )
        self.assertEqual(
            "synthetic_prairie_inference_status.tif",
            summary.inference_status_path.name,
        )
        self.assertEqual(
            "synthetic_prairie_aggregate_standardized_deviation.png",
            summary.aggregate_deviation_figure_path.name,
        )
        self.assertEqual(
            "synthetic_prairie_reference_departure_percentile.png",
            summary.departure_percentile_figure_path.name,
        )
        self.assertGreater(
            summary.aggregate_deviation_figure_path.stat().st_size,
            1_000,
        )
        self.assertGreater(
            summary.departure_percentile_figure_path.stat().st_size,
            1_000,
        )
        with rasterio.open(summary.expected_reference_path) as expected_source:
            expected = expected_source.read(masked=True)
            self.assertEqual(self.transform, expected_source.transform)
            self.assertEqual("EPSG:6933", str(expected_source.crs))
            self.assertEqual(
                ("d02_expected_reference", "d11_expected_reference"),
                expected_source.descriptions,
            )
            self.assertEqual("d02", expected_source.tags(1)["response_band"])
        with rasterio.open(summary.observed_minus_expected_path) as deviation_source:
            deviations = deviation_source.read(masked=True)
        with rasterio.open(summary.standardized_deviation_path) as standardized_source:
            standardized = standardized_source.read(masked=True)
        with rasterio.open(summary.departure_percentile_path) as percentile_source:
            percentiles = percentile_source.read(1, masked=True)
            self.assertEqual(self.transform, percentile_source.transform)
            self.assertEqual(
                "reference_departure_percentile",
                percentile_source.descriptions[0],
            )
            self.assertEqual(
                "reference_condition_departure_percentile",
                percentile_source.tags()["artifact_type"],
            )
        with rasterio.open(summary.inference_status_path) as status_source:
            status = status_source.read()

        row = 2
        column = 3
        predictor_table = pd.DataFrame(
            [
                self.source_values[
                    2 : 2 + len(self.predictor_names),
                    row,
                    column,
                ]
            ],
            columns=self.predictor_names,
        )
        expected_d02 = predict_expected_response(
            self.models["d02"],
            predictor_table,
        )[0]
        self.assertAlmostEqual(expected_d02, float(expected[0, row, column]), places=5)
        self.assertAlmostEqual(1.0, float(deviations[0, row, column]), places=5)
        self.assertAlmostEqual(0.5, float(standardized[0, row, column]), places=5)
        self.assertAlmostEqual(-2.0, float(deviations[1, row, column]), places=5)
        self.assertAlmostEqual(-0.5, float(standardized[1, row, column]), places=5)
        self.assertAlmostEqual(1.0 / 3.0, float(percentiles[row, column]), places=5)
        self.assertTrue(bool(percentiles.mask[1, 1]))
        self.assertTrue(bool(percentiles.mask[2, 2]))
        self.assertTrue(bool(percentiles.mask[0, 2]))

        self.assertEqual(STATUS_INSUFFICIENT_PREDICTORS, status[0, 0, 0])
        self.assertEqual(4, status[1, 0, 0])
        self.assertEqual(STATUS_PREDICTED, status[0, 0, 1])
        self.assertEqual(1, status[1, 0, 1])
        self.assertFalse(bool(expected.mask[0, 0, 2]))
        self.assertTrue(bool(deviations.mask[0, 0, 2]))

        report = summary.report_path.read_text(encoding="utf-8")
        metadata = json.loads(summary.metadata_path.read_text(encoding="utf-8"))
        self.assertIn("No application mask was supplied", report)
        self.assertIn("Synthetic Prairie", report)
        self.assertIn("mean pixel-level `sum(abs(z_j))`", report)
        self.assertIn("fixed linear scale", report)
        self.assertIn("Multivariate reference-departure percentile", report)
        self.assertIn("farther from the reference center than 95%", report)
        self.assertIsNone(metadata["application_mask"])
        self.assertEqual(18, metadata["responses"][0]["statistics"]["deviation_pixels"])
        self.assertEqual(16, summary.departure_percentile_pixels)
        self.assertEqual(16, metadata["coverage"]["departure_percentile_pixels"])
        self.assertEqual(
            ["d02", "d11"],
            metadata["reference_departure_calibration"]["response_bands"],
        )
        self.assertEqual(
            5,
            metadata["reference_departure_calibration"][
                "complete_reference_rows"
            ],
        )
        self.assertAlmostEqual(
            1.0 / 3.0,
            metadata["reference_departure_percentile"]["statistics"]["mean"],
            places=6,
        )
        self.assertEqual(
            1.0,
            metadata["reference_departure_percentile"]["figure"][
                "color_scale_upper_value"
            ],
        )
        self.assertEqual(
            "#0072B2",
            metadata["reference_departure_percentile"]["figure"][
                "reference_color"
            ],
        )
        self.assertEqual(
            "#FFFFFF",
            metadata["reference_departure_percentile"]["figure"][
                "reference_outline_color"
            ],
        )
        self.assertEqual(
            0.4,
            metadata["reference_departure_percentile"]["figure"][
                "reference_outline_width_points"
            ],
        )
        self.assertEqual(
            16,
            metadata["aggregate_deviation_figure"]["contributing_source_pixels"],
        )
        self.assertEqual(
            2,
            metadata["aggregate_deviation_figure"]["reference_source_pixels"],
        )
        self.assertEqual(
            2,
            metadata["aggregate_deviation_figure"]["response_count"],
        )
        self.assertEqual(3, metadata["aggregate_deviation_figure"]["display_width"])
        self.assertEqual(2, metadata["aggregate_deviation_figure"]["display_height"])
        self.assertEqual(
            10.0,
            metadata["aggregate_deviation_figure"]["color_scale_upper_value"],
        )
        self.assertEqual(
            "linear over the fixed 0 to 10 range",
            metadata["aggregate_deviation_figure"]["color_normalization"],
        )
        self.assertEqual(
            3.0,
            metadata["aggregate_deviation_figure"]["yellow_green_anchor_value"],
        )
        self.assertEqual(
            0.3,
            metadata["aggregate_deviation_figure"][
                "yellow_green_anchor_normalized_position"
            ],
        )
        self.assertIn(
            "Reference-condition raster inference",
            standard_output.getvalue(),
        )
        self.assertEqual(
            self.analysis_configuration.inference.worker_count,
            metadata["configuration"]["worker_count"],
        )

        resumed_output = io.StringIO()
        with (
            patch(
                "scripts.apply_reference_condition_models."
                "predict_expected_response",
                side_effect=AssertionError(
                    "A completed inference tile must not be modeled again."
                ),
            ),
            contextlib.redirect_stdout(resumed_output),
        ):
            resumed_summary = run_reference_condition_inference(
                replace(
                    self.analysis_configuration,
                    inference=replace(
                        self.analysis_configuration.inference,
                        worker_count=1,
                    ),
                ),
                self.model_run_directory,
                output_directory=output_directory,
                show_progress=False,
                analysis_cache_tiles=self.analysis_cache_tiles,
            )
        self.assertEqual(summary.predicted_pixels, resumed_summary.predicted_pixels)
        self.assertIn("2 resumed from checkpoint", resumed_output.getvalue())

    def test_resumes_after_a_tile_failure(self) -> None:
        """Checkpoint the first tile and recompute only the interrupted tile."""

        output_directory = self.temporary_path / "interrupted_output"
        single_worker_configuration = replace(
            self.analysis_configuration,
            inference=replace(
                self.analysis_configuration.inference,
                worker_count=1,
            ),
        )
        prediction_call_count = 0

        def interrupt_second_tile(model_bundle, predictor_table):
            nonlocal prediction_call_count
            prediction_call_count += 1
            if prediction_call_count == 9:
                raise ConnectionError("synthetic interruption")
            return predict_expected_response(model_bundle, predictor_table)

        with (
            patch(
                "scripts.apply_reference_condition_models."
                "predict_expected_response",
                side_effect=interrupt_second_tile,
            ),
            contextlib.redirect_stdout(io.StringIO()),
            self.assertRaises(ConnectionError),
        ):
            run_reference_condition_inference(
                single_worker_configuration,
                self.model_run_directory,
                output_directory=output_directory,
                show_progress=False,
                analysis_cache_tiles=self.analysis_cache_tiles,
            )

        checkpoint_path = (
            output_directory / "synthetic_prairie_inference_checkpoint.json"
        )
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        self.assertEqual(1, len(checkpoint["completed_tile_ids"]))
        resumed_output = io.StringIO()
        with contextlib.redirect_stdout(resumed_output):
            summary = run_reference_condition_inference(
                single_worker_configuration,
                self.model_run_directory,
                output_directory=output_directory,
                show_progress=False,
                analysis_cache_tiles=self.analysis_cache_tiles,
            )
        self.assertEqual(19, summary.predicted_pixels)
        self.assertIn("1 resumed from checkpoint", resumed_output.getvalue())

    def test_calculates_tiles_in_worker_processes(self) -> None:
        """Calculate source tiles outside the parent writing process."""

        worker_process_ids = set()

        def record_worker_processes(computed_tiles, artifact_paths):
            worker_process_ids.update(
                computed_tile.worker_process_id
                for computed_tile in computed_tiles
            )
            write_computed_inference_tiles(computed_tiles, artifact_paths)

        with (
            patch(
                "scripts.apply_reference_condition_models."
                "write_computed_inference_tiles",
                side_effect=record_worker_processes,
            ),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            summary = run_reference_condition_inference(
                replace(
                    self.analysis_configuration,
                    inference=replace(
                        self.analysis_configuration.inference,
                        worker_count=2,
                    ),
                ),
                self.model_run_directory,
                output_directory=self.temporary_path / "parallel_output",
                show_progress=False,
                analysis_cache_tiles=self.analysis_cache_tiles,
            )

        self.assertEqual(19, summary.predicted_pixels)
        self.assertEqual(2, len(worker_process_ids))
        self.assertNotIn(os.getpid(), worker_process_ids)

    def test_writes_nodata_outside_exact_aoi(self) -> None:
        """Mask polygon holes even when cached tiles cover those pixels."""

        analysis_cache_tiles = replace(
            self.analysis_cache_tiles,
            projected_aoi=(
                box(0, 0, 5_000, 4_000)
                .difference(box(2_000, 1_000, 3_000, 2_000))
            ),
        )
        with contextlib.redirect_stdout(io.StringIO()):
            summary = run_reference_condition_inference(
                self.analysis_configuration,
                self.model_run_directory,
                output_directory=self.temporary_path / "aoi_hole_output",
                show_progress=False,
                analysis_cache_tiles=analysis_cache_tiles,
            )

        self.assertEqual(19, summary.aoi_pixels)
        with rasterio.open(summary.inference_status_path) as status_source:
            masked_status = status_source.read(1, masked=True)
        self.assertTrue(bool(masked_status.mask[2, 2]))

    def test_limits_inference_to_application_mask_value_one(self) -> None:
        """Select only defined first-band mask pixels equal to one."""

        mask_path = self.temporary_path / "application_mask.tif"
        mask_values = np.ones((4, 5), dtype=np.uint8)
        mask_values[2, 3] = 0
        mask_values[3, 4] = 2
        with rasterio.open(
            mask_path,
            "w",
            driver="GTiff",
            width=5,
            height=4,
            count=1,
            dtype="uint8",
            crs="EPSG:6933",
            transform=self.transform,
        ) as destination:
            destination.write(mask_values, 1)

        with contextlib.redirect_stdout(io.StringIO()):
            summary = run_reference_condition_inference(
                replace(
                    self.analysis_configuration,
                    inference=replace(
                        self.analysis_configuration.inference,
                        application_mask_path=mask_path,
                        window_size_pixels=3,
                    ),
                ),
                self.model_run_directory,
                output_directory=self.temporary_path / "masked_output",
                show_progress=False,
                analysis_cache_tiles=self.analysis_cache_tiles,
            )

        self.assertEqual(18, summary.target_pixels)
        self.assertEqual(17, summary.predicted_pixels)
        with rasterio.open(summary.expected_reference_path) as expected_source:
            expected = expected_source.read(masked=True)
        with rasterio.open(summary.inference_status_path) as status_source:
            status = status_source.read()
        with rasterio.open(summary.departure_percentile_path) as percentile_source:
            percentiles = percentile_source.read(1, masked=True)
        self.assertTrue(bool(expected.mask[0, 2, 3]))
        self.assertTrue(bool(percentiles.mask[2, 3]))
        self.assertEqual(STATUS_OUTSIDE_TARGET, status[0, 2, 3])
        self.assertEqual(STATUS_NODATA, status[1, 2, 3])
        self.assertTrue(bool(expected.mask[0, 3, 4]))
        self.assertTrue(bool(percentiles.mask[3, 4]))
        self.assertEqual(STATUS_OUTSIDE_TARGET, status[0, 3, 4])
        metadata = json.loads(summary.metadata_path.read_text(encoding="utf-8"))
        self.assertEqual(str(mask_path), metadata["application_mask"]["path"])
        self.assertEqual(1, metadata["application_mask"]["selected_value"])
        self.assertEqual("nearest", metadata["application_mask"]["resampling"])
        self.assertNotIn(
            "No application mask was supplied",
            summary.report_path.read_text(encoding="utf-8"),
        )

    def test_aligns_application_mask_from_another_crs(self) -> None:
        """Reproject a coarse global-style mask during windowed inference."""

        mask_path = self.temporary_path / "projected_application_mask.tif"
        source_bounds = rasterio.transform.array_bounds(4, 5, self.transform)
        projected_bounds = transform_bounds(
            "EPSG:6933",
            "EPSG:3857",
            *source_bounds,
        )
        with rasterio.open(
            mask_path,
            "w",
            driver="GTiff",
            width=1,
            height=1,
            count=1,
            dtype="uint8",
            crs="EPSG:3857",
            transform=from_bounds(*projected_bounds, width=1, height=1),
        ) as destination:
            destination.write(np.ones((1, 1), dtype=np.uint8), 1)

        with contextlib.redirect_stdout(io.StringIO()):
            summary = run_reference_condition_inference(
                replace(
                    self.analysis_configuration,
                    inference=replace(
                        self.analysis_configuration.inference,
                        application_mask_path=mask_path,
                    ),
                ),
                self.model_run_directory,
                output_directory=self.temporary_path / "projected_mask_output",
                show_progress=False,
                analysis_cache_tiles=self.analysis_cache_tiles,
            )

        self.assertEqual(20, summary.target_pixels)
        self.assertEqual(19, summary.predicted_pixels)
        metadata = json.loads(summary.metadata_path.read_text(encoding="utf-8"))
        self.assertEqual("EPSG:3857", metadata["application_mask"]["source_crs"])


if __name__ == "__main__":
    unittest.main()
