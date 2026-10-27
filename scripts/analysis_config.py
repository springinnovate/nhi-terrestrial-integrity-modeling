"""Load the reproducible analysis definition used throughout the pipeline."""

from __future__ import annotations

import hashlib
import json
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence


DEFAULT_ANALYSIS_CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "config"
    / "south_africa_reference_condition_analysis.toml"
)
BAND_COLUMN_PATTERN = re.compile(r"^y(?P<year>\d{4})_(?P<band_id>d\d+)_")
VALID_BAND_ROLES = {"reference", "response", "predictor"}
VALID_DATA_TYPES = {"binary", "continuous", "categorical"}


@dataclass(frozen=True)
class RasterBandDefinition:
    """Describe one configured raster-stack band.

    Attributes:
        identifier: Stable band identifier such as ``d02``.
        computation: Python Earth Engine computation registered for the band.
        suffix: Stable suffix appended to the year and identifier.
        display_name: Human-readable variable name for reports and figures.
        role: Pipeline role: reference, response, or predictor.
        data_type: Binary, continuous, or categorical interpretation.
        source_dataset_keys: Dataset aliases used by the computation.
    """

    identifier: str
    computation: str
    suffix: str
    display_name: str
    role: str
    data_type: str
    source_dataset_keys: tuple[str, ...]


@dataclass(frozen=True)
class RasterCacheGrid:
    """Define the globally aligned raster cache grid.

    Attributes:
        crs: Projected coordinate reference system used by every tile.
        pixel_size_meters: Width and height of one output pixel.
        tile_size_pixels: Width and height of one square cache tile.
        origin_x: X coordinate anchoring tile columns.
        origin_y: Y coordinate anchoring tile rows.
    """

    crs: str
    pixel_size_meters: int
    tile_size_pixels: int
    origin_x: int = 0
    origin_y: int = 0


@dataclass(frozen=True)
class EarthEngineSettings:
    """Store Earth Engine access and local cache settings.

    Attributes:
        project: Google Cloud project registered for Earth Engine.
        cache_directory: Resolved local directory for fetched raster tiles.
        request_timeout_seconds: Maximum socket wait for one request attempt.
        request_retry_count: Retry attempts after the initial request fails.
    """

    project: str
    cache_directory: Path
    request_timeout_seconds: float
    request_retry_count: int


@dataclass(frozen=True)
class ReferenceSettings:
    """Store thresholds used to construct reference sites.

    Attributes:
        grassland_probability: Minimum annual grassland probability percentage.
        human_modification: Maximum human-modification index.
        human_influence: Maximum scaled human-influence index.
    """

    grassland_probability: int
    human_modification: float
    human_influence: float


@dataclass(frozen=True)
class SamplingSettings:
    """Store spatial sample construction settings.

    Attributes:
        block_size_meters: Width of source sampling blocks.
        samples_per_class_per_block: Maximum rows sampled per class and block.
        random_seed: Seed used for reproducible within-block sampling.
    """

    block_size_meters: int
    samples_per_class_per_block: int
    random_seed: int


@dataclass(frozen=True)
class ModelSettings:
    """Store response-model fitting and validation settings.

    Attributes:
        fold_count: Number of spatial cross-validation folds.
        validation_block_size_meters: Width of grouped validation blocks.
        minimum_predictor_coverage: Minimum area coverage for predictors.
        maximum_row_missing_fraction: Maximum predictor missingness per row.
        spline_knot_count: Number of knots for continuous spline terms.
        minimum_response_coverage: Minimum area coverage for responses.
        ridge_alpha: Ridge penalty applied to fitted response models.
        responses: Selected response band identifiers; empty selects all.
    """

    fold_count: int
    validation_block_size_meters: int
    minimum_predictor_coverage: float
    maximum_row_missing_fraction: float
    spline_knot_count: int
    minimum_response_coverage: float
    ridge_alpha: float
    responses: tuple[str, ...]


@dataclass(frozen=True)
class InferenceSettings:
    """Store raster-inference settings.

    Attributes:
        application_mask_path: Optional resolved raster defining target pixels.
        application_mask_label: Human-readable name for the selected extent.
        worker_count: Maximum cache tiles calculated concurrently.
        window_size_pixels: Width and height of each processing window.
        covariance_shrinkage: Reference covariance diagonal shrinkage fraction.
    """

    application_mask_path: Path | None
    application_mask_label: str | None
    worker_count: int
    window_size_pixels: int
    covariance_shrinkage: float


@dataclass(frozen=True)
class RasterStackConfiguration:
    """Define the scientific variables and grid of an Earth Engine raster stack.

    This configuration contains no local AOI, credentials, cache paths or model
    execution settings. Use it to construct the same stack for different areas.

    Attributes:
        raster_configuration_sha256: Hash of settings affecting fetched pixels.
        year: Source-data year used to construct the raster bands.
        stack_name: File-safe raster-stack name.
        stack_version: Explicit stack-definition version.
        minimum_year: First supported complete-stack year.
        maximum_year: Last supported complete-stack year.
        reference_start_year: First year used in reference persistence tests.
        reference_end_year: Last year used in reference persistence tests.
        era5_start_year: First year available to rainfall-window calculations.
        interannual_rainfall_window_years: Rainfall variability window length.
        stream_upstream_area_threshold_km2: Stream extraction threshold.
        grid: Deterministic raster cache grid.
        reference: Reference-site thresholds.
        datasets: Earth Engine dataset IDs keyed by stable aliases.
        bands: Ordered raster bands included in this stack.
    """

    raster_configuration_sha256: str
    year: int
    stack_name: str
    stack_version: int
    minimum_year: int
    maximum_year: int
    reference_start_year: int
    reference_end_year: int
    era5_start_year: int
    interannual_rainfall_window_years: int
    stream_upstream_area_threshold_km2: float
    grid: RasterCacheGrid
    reference: ReferenceSettings
    datasets: Mapping[str, str]
    bands: tuple[RasterBandDefinition, ...]

    def band_names(self, year: int | None = None) -> tuple[str, ...]:
        """Return ordered export names for the configured bands.

        Args:
            year: Optional source-data year. The analysis year is authoritative
                when this argument is omitted.

        Returns:
            Stable Earth Engine export names in configured order.
        """

        band_year = self.year if year is None else year
        return tuple(
            f"y{band_year}_{band.identifier}_{band.suffix}" for band in self.bands
        )


@dataclass(frozen=True)
class AnalysisConfiguration(RasterStackConfiguration):
    """Define a raster stack and the local analysis that uses it.

    Inherited fields describe the scientific raster stack. These additional
    settings identify local inputs and configure acquisition, sampling, fitting
    and inference for the command-line workflow.

    Attributes:
        path: Resolved TOML source path.
        configuration_sha256: Hash of normalized effective TOML content.
        analysis_name: File-safe analysis identifier.
        display_name: Human-readable AOI or ecoregion name.
        aoi_path: Resolved local single-layer GDAL-compatible vector AOI.
        earth_engine: Earth Engine project and local cache settings.
        sampling: Spatial sample construction settings.
        model: Response-model fitting and validation settings.
        inference: Raster-inference settings.
    """

    path: Path
    configuration_sha256: str
    analysis_name: str
    display_name: str
    aoi_path: Path
    earth_engine: EarthEngineSettings
    sampling: SamplingSettings
    model: ModelSettings
    inference: InferenceSettings

    def columns_with_role(
        self,
        columns: Sequence[str],
        role: str,
    ) -> dict[str, str]:
        """Select one role's configured columns from its earliest available year.

        Args:
            columns: Raster descriptions or sample-table column names.
            role: Reference, response, or predictor role.

        Returns:
            Column names keyed by stable band identifier in configured order.

        Raises:
            ValueError: If no supplied column belongs to the requested role.
        """

        band_definitions_by_identifier = {
            band.identifier: band for band in self.bands if band.role == role
        }
        matching_column_records = []
        for column_name in columns:
            column_match = BAND_COLUMN_PATTERN.match(column_name)
            if (
                column_match
                and column_match.group("band_id")
                in band_definitions_by_identifier
            ):
                matching_column_records.append(
                    (
                        int(column_match.group("year")),
                        column_match.group("band_id"),
                        column_name,
                    )
                )
        if not matching_column_records:
            raise ValueError(
                f"No supplied columns match the configured '{role}' band role."
            )
        selected_year = min(
            year for year, _, _ in matching_column_records
        )
        columns_by_identifier = {
            identifier: column_name
            for year, identifier, column_name in matching_column_records
            if year == selected_year
        }
        return {
            identifier: columns_by_identifier[identifier]
            for identifier in band_definitions_by_identifier
            if identifier in columns_by_identifier
        }

    def band_for_column(self, column: str) -> RasterBandDefinition:
        """Return the configured definition identified by a raster column.

        Args:
            column: Year-prefixed raster band name.

        Returns:
            Matching configured band definition.
        """

        column_match = BAND_COLUMN_PATTERN.match(column)
        band_identifier = column_match.group("band_id")
        return next(
            band for band in self.bands if band.identifier == band_identifier
        )


def _parse_raster_stack_configuration(
    raw_configuration: Mapping[str, Any], *, year: int | None
) -> RasterStackConfiguration:
    """Read the scientific sections of a parsed TOML analysis definition.

    Both configuration loaders use this parser so stack validation and cache
    identity are independent of local analysis settings.

    Args:
        raw_configuration: Parsed TOML containing stack, grid, reference,
            datasets and bands, plus analysis.year when no year is supplied.
        year: Source-data year, or None to read analysis.year from the TOML.

    Returns:
        Validated scientific settings and a checksum of the settings that
        determine downloaded pixels.

    Raises:
        KeyError: If a required scientific setting is absent.
        ValueError: If a scientific setting is invalid or the year is unsupported.
    """
    stack_section = raw_configuration["stack"]
    grid_section = raw_configuration["grid"]
    reference_section = raw_configuration["reference"]
    dataset_ids = {
        str(key): str(value) for key, value in raw_configuration["datasets"].items()
    }
    band_definitions = tuple(
        RasterBandDefinition(
            identifier=str(raw_band_definition["id"]),
            computation=str(raw_band_definition["computation"]),
            suffix=str(raw_band_definition["suffix"]),
            display_name=str(raw_band_definition["display_name"]),
            role=str(raw_band_definition["role"]),
            data_type=str(raw_band_definition["data_type"]),
            source_dataset_keys=tuple(
                str(key) for key in raw_band_definition["sources"]
            ),
        )
        for raw_band_definition in raw_configuration["bands"]
    )
    stack_name = str(stack_section["name"])
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", stack_name):
        raise ValueError(
            "Raster-stack names must contain lowercase letters, numbers, hyphens, "
            "or underscores."
        )
    minimum_year = int(stack_section["minimum_year"])
    maximum_year = int(stack_section["maximum_year"])
    analysis_year = int(
        raw_configuration["analysis"]["year"] if year is None else year
    )
    if minimum_year > maximum_year:
        raise ValueError("Raster-stack minimum_year cannot exceed maximum_year.")
    if not minimum_year <= analysis_year <= maximum_year:
        raise ValueError(
            f"Analysis year must be between {minimum_year} and {maximum_year}."
        )
    if (
        int(grid_section["pixel_size_meters"]) <= 0
        or int(grid_section["tile_size_pixels"]) <= 0
    ):
        raise ValueError("Raster cache pixel and tile sizes must be positive.")
    band_identifiers = [band.identifier for band in band_definitions]
    if len(band_identifiers) != len(set(band_identifiers)):
        raise ValueError("Raster-stack band identifiers must be unique.")
    if any(
        not re.fullmatch(r"d\d+", identifier) for identifier in band_identifiers
    ):
        raise ValueError("Raster-stack band identifiers must use dNN notation.")
    if any(band.role not in VALID_BAND_ROLES for band in band_definitions):
        raise ValueError(
            "Raster-stack band roles must be reference, response, or predictor."
        )
    if any(band.data_type not in VALID_DATA_TYPES for band in band_definitions):
        raise ValueError(
            "Raster-stack data types must be binary, continuous, or categorical."
        )
    missing_dataset_keys = sorted(
        {
            source_key
            for band in band_definitions
            for source_key in band.source_dataset_keys
            if source_key not in dataset_ids
        }
    )
    if missing_dataset_keys:
        raise ValueError(
            "Raster bands reference undefined dataset aliases: "
            + ", ".join(missing_dataset_keys)
        )
    if sum(band.role == "reference" for band in band_definitions) != 1:
        raise ValueError("A raster stack must define exactly one reference band.")
    categorical_predictor_definitions = [
        band
        for band in band_definitions
        if band.role == "predictor" and band.data_type == "categorical"
    ]
    if len(categorical_predictor_definitions) != 1:
        raise ValueError(
            "Reference-condition modeling requires exactly one categorical predictor."
        )
    raster_configuration_bytes = json.dumps(
        {
            "year": analysis_year,
            "stack": stack_section,
            "grid": grid_section,
            "reference": reference_section,
            "datasets": dataset_ids,
            "bands": raw_configuration["bands"],
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return RasterStackConfiguration(
        raster_configuration_sha256=hashlib.sha256(
            raster_configuration_bytes
        ).hexdigest(),
        year=analysis_year,
        stack_name=stack_name,
        stack_version=int(stack_section["version"]),
        minimum_year=minimum_year,
        maximum_year=maximum_year,
        reference_start_year=int(stack_section["reference_start_year"]),
        reference_end_year=int(stack_section["reference_end_year"]),
        era5_start_year=int(stack_section["era5_start_year"]),
        interannual_rainfall_window_years=int(
            stack_section["interannual_rainfall_window_years"]
        ),
        stream_upstream_area_threshold_km2=float(
            stack_section["stream_upstream_area_threshold_km2"]
        ),
        grid=RasterCacheGrid(
            crs=str(grid_section["crs"]),
            pixel_size_meters=int(grid_section["pixel_size_meters"]),
            tile_size_pixels=int(grid_section["tile_size_pixels"]),
            origin_x=int(grid_section["origin_x"]),
            origin_y=int(grid_section["origin_y"]),
        ),
        reference=ReferenceSettings(
            grassland_probability=int(
                reference_section["grassland_probability"]
            ),
            human_modification=float(reference_section["human_modification"]),
            human_influence=float(reference_section["human_influence"]),
        ),
        datasets=MappingProxyType(dataset_ids),
        bands=band_definitions,
    )


def load_raster_stack_configuration(
    configuration_path: Path, *, year: int | None = None
) -> RasterStackConfiguration:
    """Load Earth Engine variables and their grid without a local analysis AOI.

    Only scientific settings are interpreted. A complete analysis TOML can be
    used unchanged, or operational sections can be omitted. No credentials,
    AOI files or output directories are resolved or accessed.

    Args:
        configuration_path: TOML containing stack, grid, reference, datasets
            and ordered bands. analysis.year is required when year is omitted.
        year: Optional source-data year overriding analysis.year in the file.

    Returns:
        Validated stack settings and ordered band definitions.

    Raises:
        OSError: If the configuration file cannot be read.
        tomllib.TOMLDecodeError: If the file is not valid TOML.
        KeyError: If a required scientific setting is absent.
        ValueError: If a scientific setting is invalid or the year is unsupported.
    """
    with configuration_path.expanduser().resolve().open("rb") as configuration_file:
        raw_configuration = tomllib.load(configuration_file)
    return _parse_raster_stack_configuration(raw_configuration, year=year)


def load_analysis_configuration(
    configuration_path: Path = DEFAULT_ANALYSIS_CONFIG_PATH,
) -> AnalysisConfiguration:
    """Load and validate one TOML analysis definition.

    Local paths are resolved relative to the TOML file. The analysis definition
    is the authoritative source for scientific settings used by every pipeline
    stage.

    Args:
        configuration_path: Complete TOML analysis definition.

    Returns:
        Immutable typed configuration with a normalized content hash.

    Raises:
        OSError: If the configuration file cannot be read.
        tomllib.TOMLDecodeError: If the file is not valid TOML.
        KeyError: If a required analysis setting is absent.
        ValueError: If the analysis contract is internally inconsistent or the
            configured AOI file does not exist.
    """

    resolved_configuration_path = configuration_path.expanduser().resolve()
    with resolved_configuration_path.open("rb") as configuration_file:
        raw_configuration = tomllib.load(configuration_file)

    raster_stack = _parse_raster_stack_configuration(raw_configuration, year=None)
    analysis_section = raw_configuration["analysis"]
    earth_engine_section = raw_configuration["earth_engine"]
    sampling_section = raw_configuration["sampling"]
    model_section = raw_configuration["model"]
    inference_section = raw_configuration["inference"]
    analysis_name = str(analysis_section["name"])
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", analysis_name):
        raise ValueError(
            "Analysis names must contain lowercase letters, numbers, hyphens, "
            "or underscores."
        )
    if float(earth_engine_section["request_timeout_seconds"]) <= 0:
        raise ValueError("Earth Engine request_timeout_seconds must be positive.")
    if not 0 <= int(earth_engine_section["request_retry_count"]) < 100:
        raise ValueError("Earth Engine request_retry_count must be between 0 and 99.")
    if (
        int(sampling_section["block_size_meters"]) <= 0
        or int(sampling_section["samples_per_class_per_block"]) <= 0
        or int(model_section["fold_count"]) < 2
        or int(model_section["validation_block_size_meters"]) <= 0
        or int(model_section["spline_knot_count"]) < 2
        or float(model_section["ridge_alpha"]) < 0
        or int(inference_section["worker_count"]) <= 0
        or int(inference_section["window_size_pixels"]) <= 0
    ):
        raise ValueError("Analysis size, count, and penalty settings are invalid.")
    bounded_fraction_values = (
        float(model_section["minimum_predictor_coverage"]),
        float(model_section["maximum_row_missing_fraction"]),
        float(model_section["minimum_response_coverage"]),
        float(inference_section["covariance_shrinkage"]),
    )
    if any(not 0.0 < value < 1.0 for value in bounded_fraction_values):
        raise ValueError("Analysis coverage, missingness, and shrinkage must be 0-1.")
    response_identifiers = {
        band.identifier for band in raster_stack.bands if band.role == "response"
    }
    configured_responses = tuple(
        str(value) for value in model_section["responses"]
    )
    unknown_responses = sorted(set(configured_responses) - response_identifiers)
    if unknown_responses:
        raise ValueError(
            "Model responses are not configured response bands: "
            + ", ".join(unknown_responses)
        )

    configuration_directory = resolved_configuration_path.parent
    resolved_aoi_path = (
        configuration_directory / str(analysis_section["aoi_path"])
    ).expanduser().resolve()
    if not resolved_aoi_path.is_file():
        raise ValueError(f"Analysis AOI does not exist: {resolved_aoi_path}")
    application_mask_path_value = inference_section.get("application_mask_path")
    application_mask_label_value = inference_section.get(
        "application_mask_label"
    )
    if (application_mask_path_value is None) != (
        application_mask_label_value is None
    ):
        raise ValueError(
            "inference.application_mask_path and application_mask_label must "
            "be configured together."
        )
    application_mask_label = (
        str(application_mask_label_value).strip()
        if application_mask_label_value is not None
        else None
    )
    if application_mask_label == "":
        raise ValueError("inference.application_mask_label cannot be empty.")
    normalized_configuration_bytes = json.dumps(
        raw_configuration,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return AnalysisConfiguration(
        path=resolved_configuration_path,
        configuration_sha256=hashlib.sha256(
            normalized_configuration_bytes
        ).hexdigest(),
        raster_configuration_sha256=raster_stack.raster_configuration_sha256,
        analysis_name=analysis_name,
        display_name=str(analysis_section["display_name"]),
        aoi_path=resolved_aoi_path,
        year=raster_stack.year,
        earth_engine=EarthEngineSettings(
            project=str(earth_engine_section["project"]),
            cache_directory=(
                configuration_directory
                / str(earth_engine_section["cache_directory"])
            ).expanduser().resolve(),
            request_timeout_seconds=float(
                earth_engine_section["request_timeout_seconds"]
            ),
            request_retry_count=int(earth_engine_section["request_retry_count"]),
        ),
        stack_name=raster_stack.stack_name,
        stack_version=raster_stack.stack_version,
        minimum_year=raster_stack.minimum_year,
        maximum_year=raster_stack.maximum_year,
        reference_start_year=raster_stack.reference_start_year,
        reference_end_year=raster_stack.reference_end_year,
        era5_start_year=raster_stack.era5_start_year,
        interannual_rainfall_window_years=(
            raster_stack.interannual_rainfall_window_years
        ),
        stream_upstream_area_threshold_km2=(
            raster_stack.stream_upstream_area_threshold_km2
        ),
        grid=raster_stack.grid,
        reference=raster_stack.reference,
        sampling=SamplingSettings(
            block_size_meters=int(sampling_section["block_size_meters"]),
            samples_per_class_per_block=int(
                sampling_section["samples_per_class_per_block"]
            ),
            random_seed=int(sampling_section["random_seed"]),
        ),
        model=ModelSettings(
            fold_count=int(model_section["fold_count"]),
            validation_block_size_meters=int(
                model_section["validation_block_size_meters"]
            ),
            minimum_predictor_coverage=float(
                model_section["minimum_predictor_coverage"]
            ),
            maximum_row_missing_fraction=float(
                model_section["maximum_row_missing_fraction"]
            ),
            spline_knot_count=int(model_section["spline_knot_count"]),
            minimum_response_coverage=float(
                model_section["minimum_response_coverage"]
            ),
            ridge_alpha=float(model_section["ridge_alpha"]),
            responses=configured_responses,
        ),
        inference=InferenceSettings(
            application_mask_path=(
                (
                    configuration_directory / str(application_mask_path_value)
                ).expanduser().resolve()
                if application_mask_path_value is not None
                else None
            ),
            application_mask_label=application_mask_label,
            worker_count=int(inference_section["worker_count"]),
            window_size_pixels=int(inference_section["window_size_pixels"]),
            covariance_shrinkage=float(
                inference_section["covariance_shrinkage"]
            ),
        ),
        datasets=raster_stack.datasets,
        bands=raster_stack.bands,
    )
