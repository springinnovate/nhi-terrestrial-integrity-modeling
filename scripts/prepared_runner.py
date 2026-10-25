"""Run Habitat condition from aligned local rasters and write its result maps."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from contextlib import ExitStack
from dataclasses import dataclass
import math
from pathlib import Path
import shutil

import rasterio
from rasterio.features import geometry_mask, geometry_window
from rasterio.windows import Window, transform as window_transform
from shapely.geometry import MultiPolygon, Polygon, mapping

from .habitat_condition import FLOAT_NODATA, STATUS_NODATA, predict_raster_window
from .habitat_training import FittedHabitatCondition, IntegrityConfiguration, fit_habitat_condition
from .prepared_rasters import RasterBand, open_prepared_rasters
from .prepared_sampling import sample_prepared_rasters


@dataclass(frozen=True)
class HabitatConditionRun:
    """Completed Habitat condition maps and the fit used to calculate them.

    All rasters share the source-aligned AOI crop. Float rasters use float32/-9999;
    status uses uint8/255. Response bands follow ``fitted.responses``.

    Attributes:
        fitted: Existing sample, coverage, validation and calibration diagnostics,
            together with the fitted response models. No model files are written.
        reference_similarity: Single-band 0–1 GeoTIFF; higher is more reference-like.
            Reference pixels and incomplete response vectors remain NoData.
        expected_reference: Expected responses, with one band per fitted response.
        observed_minus_expected: Observed-minus-expected response values.
        standardized_deviation: Departures divided by held-out reference RMSE.
        inference_status: Two bands: prediction status and missing-predictor count.
            Status is 0 outside assessment, 1 for insufficient predictors, 2 for
            predicted pixels, and 255 outside the AOI. Missing counts are 255
            outside assessment.
    """

    fitted: FittedHabitatCondition
    reference_similarity: Path
    expected_reference: Path
    observed_minus_expected: Path
    standardized_deviation: Path
    inference_status: Path


def run_habitat_condition(
    predictor_bands: Sequence[RasterBand],
    response_bands: Sequence[RasterBand],
    reference_mask: RasterBand,
    aoi: Polygon | MultiPolygon,
    output_directory: Path,
    configuration: IntegrityConfiguration,
    *,
    categorical_predictor_name: str,
    samples_per_class_per_block: int,
    random_seed: int,
    tile_size_pixels: int,
    window_size_pixels: int,
    covariance_shrinkage: float,
    tile_origin: tuple[float, float] = (0, 0),
    assessment_mask: RasterBand | None = None,
    progress: Callable[[str, int, int], None] | None = None,
) -> HabitatConditionRun:
    """Fit reference-site responses and write Habitat condition maps for an AOI.

    Sample and fit afresh through the existing model functions, then predict and
    write one window at a time. Reference similarity is one minus the existing
    departure percentile only where that percentile is defined. Reference pixels
    keep their diagnostic predictions but receive no similarity score.

    Args:
        predictor_bands: Nonempty named predictor selection, including exactly one
            categorical predictor. Names must be unique across both band roles.
        response_bands: Nonempty named response selection in requested fit order.
            Responses rejected by scientific screening remain in fit diagnostics.
        reference_mask: Binary reference band aligned with the data; one marks
            reference sites. Only reference pixels inside the AOI train the model.
        aoi: Valid, nonempty polygon or multipolygon in EPSG:6933.
        output_directory: New directory whose parent exists. Existing directories
            are never overwritten. The runner owns these outputs until success.
        configuration: Valid model-preset fitting settings, including sampling-block
            width and the maximum predictor missing fraction used for prediction.
        categorical_predictor_name: Name of the selected categorical predictor.
        samples_per_class_per_block: Positive reference/background sampling cap.
        random_seed: Seed for the existing deterministic pixel selector.
        tile_size_pixels: Positive virtual sampling tile width; preserve the CLI's
            tile size and origin when comparing identical source grids.
        window_size_pixels: Positive maximum prediction window width and height.
        covariance_shrinkage: Preset fraction strictly between zero and one.
        tile_origin: Finite EPSG:6933 anchor for sampling tile addresses.
        assessment_mask: Optional binary target mask, aligned by the reader using
            nearest resampling. It limits prediction, never sample selection.
        progress: Optional phase/completed/total callback passed through sampling
            and fitting. Prediction reports 0 then each written window. Raising
            cancels further work; all handles close and this run's files are removed.

    Returns:
        Five completed GeoTIFF paths and the existing fitted result. The output grid
        is the AOI's pixel bounding rectangle clipped to the source. Exact polygon
        masking leaves pixels outside the AOI as NoData. Inputs must remain unchanged
        during the call; their grids and metadata are admitted once by the reader.

    Raises:
        ValueError: If band roles or window size are invalid, or the existing reader,
            sampler or fitter rejects the supplied inputs or numerical fit.
        FileExistsError: If the output directory already exists.
        OSError: If output creation or cleanup fails.
        RasterioError: If a source or output raster cannot be opened, read or written.
        RuntimeError: If a fitted model produces a nonfinite prediction.
        numpy.linalg.LinAlgError: If reference covariance cannot be inverted.
    """
    predictor_names = tuple(band.name for band in predictor_bands)
    response_names = tuple(band.name for band in response_bands)
    if not predictor_names or not response_names or categorical_predictor_name not in predictor_names:
        raise ValueError("Select predictors including the categorical band, and at least one response.")
    if window_size_pixels <= 0:
        raise ValueError("Prediction window size must be positive.")

    output_directory.mkdir()
    try:
        with open_prepared_rasters(
            (*predictor_bands, *response_bands), reference_mask, assessment_mask,
        ) as reader:
            samples = sample_prepared_rasters(
                reader, aoi, block_size_meters=configuration.sampling_block_size_meters,
                samples_per_class_per_block=samples_per_class_per_block,
                random_seed=random_seed, tile_size_pixels=tile_size_pixels,
                tile_origin=tile_origin, progress=progress,
            )
            fitted = fit_habitat_condition(
                samples, configuration, predictor_names=predictor_names,
                categorical_predictor_name=categorical_predictor_name,
                response_names=response_names, covariance_shrinkage=covariance_shrinkage,
                progress=progress,
            )
            result = HabitatConditionRun(
                fitted=fitted,
                reference_similarity=output_directory / "reference_similarity.tif",
                expected_reference=output_directory / "expected_reference.tif",
                observed_minus_expected=output_directory / "observed_minus_expected.tif",
                standardized_deviation=output_directory / "standardized_deviation.tif",
                inference_status=output_directory / "inference_status.tif",
            )
            response_names = tuple(fitted.responses)
            predictor_indices = [reader.band_names.index(name) for name in fitted.predictor_names]
            response_indices = [reader.band_names.index(name) for name in response_names]
            response_models = tuple(response.model for response in fitted.responses.values())
            reference_rmses = tuple(
                response.overall_metrics["weighted_rmse"] for response in fitted.responses.values()
            )
            aoi_geometry = mapping(aoi)
            crop = geometry_window(reader, [aoi_geometry])
            width, height = int(crop.width), int(crop.height)
            total = math.ceil(width / window_size_pixels) * math.ceil(height / window_size_pixels)
            phase = "Predicting raster windows"
            if progress is not None:
                progress(phase, 0, total)
            with ExitStack() as outputs:
                writers = []
                for path, descriptions, is_status in (
                    (result.reference_similarity, ("Reference similarity",), False),
                    (result.expected_reference, response_names, False),
                    (result.observed_minus_expected, response_names, False),
                    (result.standardized_deviation, response_names, False),
                    (result.inference_status, ("Prediction status", "Missing predictor count"), True),
                ):
                    destination = outputs.enter_context(rasterio.open(
                        path, "w", driver="GTiff", width=width, height=height,
                        count=len(descriptions), dtype="uint8" if is_status else "float32",
                        nodata=STATUS_NODATA if is_status else FLOAT_NODATA,
                        crs=reader.crs, transform=window_transform(crop, reader.transform),
                        compress="deflate", predictor=2 if is_status else 3,
                        tiled=True, blockxsize=256, blockysize=256, interleave="band",
                        BIGTIFF="IF_SAFER",
                    ))
                    destination.descriptions = descriptions
                    writers.append(destination)
                similarity, expected, departure, standardized, status = writers
                similarity.update_tags(
                    interpretation="1 - reference departure percentile; higher is more reference-like",
                    reference_pixels="nodata", value_minimum="0", value_maximum="1",
                )
                status.update_tags(
                    status_0="outside assessment", status_1="insufficient predictors",
                    status_2="predicted", status_255="outside AOI",
                )
                completed = 0
                for row in range(0, height, window_size_pixels):
                    for column in range(0, width, window_size_pixels):
                        destination_window = Window(
                            column, row, min(window_size_pixels, width - column),
                            min(window_size_pixels, height - row),
                        )
                        source_window = Window(
                            crop.col_off + column, crop.row_off + row,
                            destination_window.width, destination_window.height,
                        )
                        data = reader.read_window(source_window)
                        inside = geometry_mask(
                            [aoi_geometry], out_shape=data.reference_mask.shape,
                            transform=data.transform, invert=True,
                        )
                        prediction = predict_raster_window(
                            data.values[predictor_indices], data.values[response_indices],
                            predictor_names=fitted.predictor_names, response_names=response_names,
                            response_models=response_models, reference_rmses=reference_rmses,
                            reference_distribution=fitted.reference_distribution,
                            aoi_pixel_mask=inside, reference_pixel_mask=inside & data.reference_mask,
                            assessment_pixel_mask=data.assessment_mask,
                            maximum_predictor_missing_fraction=configuration.maximum_row_missing_fraction,
                        )
                        scores = prediction.departure_percentile.copy()
                        defined = scores != FLOAT_NODATA
                        scores[defined] = 1.0 - scores[defined]
                        similarity.write(scores, 1, window=destination_window)
                        expected.write(prediction.expected_reference, window=destination_window)
                        departure.write(prediction.observed_minus_expected, window=destination_window)
                        standardized.write(prediction.standardized_deviation, window=destination_window)
                        status.write(prediction.inference_status, window=destination_window)
                        completed += 1
                        if progress is not None:
                            progress(phase, completed, total)
            return result
    except BaseException:
        shutil.rmtree(output_directory)
        raise
