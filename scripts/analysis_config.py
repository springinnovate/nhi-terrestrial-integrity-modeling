"""Load the reproducible analysis definition used throughout the pipeline."""

from __future__ import annotations

import hashlib
import json
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Sequence


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
        grassland_mask_path: Optional resolved raster defining target pixels.
        window_size_pixels: Width and height of each processing window.
        covariance_shrinkage: Reference covariance diagonal shrinkage fraction.
    """

    grassland_mask_path: Path | None
    window_size_pixels: int
    covariance_shrinkage: float


@dataclass(frozen=True)
class AnalysisConfiguration:
    """Represent one complete, validated reference-condition analysis.

    Attributes:
        path: Resolved TOML source path.
        configuration_sha256: Hash of normalized effective TOML content.
        raster_configuration_sha256: Hash of settings affecting fetched pixels.
        analysis_name: File-safe analysis identifier.
        display_name: Human-readable AOI or ecoregion name.
        aoi_path: Resolved local WGS84 GeoJSON AOI.
        year: Source-data year used throughout the analysis.
        earth_engine: Earth Engine project and local cache settings.
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
        sampling: Spatial sample construction settings.
        model: Response-model fitting and validation settings.
        inference: Raster-inference settings.
        datasets: Earth Engine dataset IDs keyed by stable aliases.
        bands: Ordered raster bands included in this stack.
    """

    path: Path
    configuration_sha256: str
    raster_configuration_sha256: str
    analysis_name: str
    display_name: str
    aoi_path: Path
    year: int
    earth_engine: EarthEngineSettings
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
    sampling: SamplingSettings
    model: ModelSettings
    inference: InferenceSettings
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

        definitions_by_id = {
            band.identifier: band for band in self.bands if band.role == role
        }
        available_columns = []
        for column in columns:
            match = BAND_COLUMN_PATTERN.match(column)
            if match and match.group("band_id") in definitions_by_id:
                available_columns.append(
                    (int(match.group("year")), match.group("band_id"), column)
                )
        if not available_columns:
            raise ValueError(
                f"No sample columns match the configured '{role}' band role."
            )
        selected_year = min(year for year, _, _ in available_columns)
        columns_by_identifier = {
            identifier: column
            for year, identifier, column in available_columns
            if year == selected_year
        }
        return {
            identifier: columns_by_identifier[identifier]
            for identifier in definitions_by_id
            if identifier in columns_by_identifier
        }

    def band_for_column(self, column: str) -> RasterBandDefinition:
        """Return the configured definition identified by a raster column.

        Args:
            column: Year-prefixed raster band name.

        Returns:
            Matching configured band definition.
        """

        band_identifier = BAND_COLUMN_PATTERN.match(column).group("band_id")
        return next(
            band for band in self.bands if band.identifier == band_identifier
        )


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
        ValueError: If the analysis contract is internally inconsistent.
    """

    resolved_path = configuration_path.expanduser().resolve()
    with resolved_path.open("rb") as configuration_file:
        raw_configuration = tomllib.load(configuration_file)

    analysis = raw_configuration["analysis"]
    earth_engine = raw_configuration["earth_engine"]
    stack = raw_configuration["stack"]
    grid = raw_configuration["grid"]
    reference = raw_configuration["reference"]
    sampling = raw_configuration["sampling"]
    model = raw_configuration["model"]
    inference = raw_configuration["inference"]
    datasets = {
        str(key): str(value) for key, value in raw_configuration["datasets"].items()
    }
    bands = tuple(
        RasterBandDefinition(
            identifier=str(raw_band["id"]),
            computation=str(raw_band["computation"]),
            suffix=str(raw_band["suffix"]),
            display_name=str(raw_band["display_name"]),
            role=str(raw_band["role"]),
            data_type=str(raw_band["data_type"]),
            source_dataset_keys=tuple(str(key) for key in raw_band["sources"]),
        )
        for raw_band in raw_configuration["bands"]
    )
    analysis_name = str(analysis["name"])
    stack_name = str(stack["name"])
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", analysis_name):
        raise ValueError(
            "Analysis names must contain lowercase letters, numbers, hyphens, "
            "or underscores."
        )
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", stack_name):
        raise ValueError(
            "Raster-stack names must contain lowercase letters, numbers, hyphens, "
            "or underscores."
        )
    minimum_year = int(stack["minimum_year"])
    maximum_year = int(stack["maximum_year"])
    analysis_year = int(analysis["year"])
    if minimum_year > maximum_year:
        raise ValueError("Raster-stack minimum_year cannot exceed maximum_year.")
    if not minimum_year <= analysis_year <= maximum_year:
        raise ValueError(
            f"Analysis year must be between {minimum_year} and {maximum_year}."
        )
    if int(grid["pixel_size_meters"]) <= 0 or int(grid["tile_size_pixels"]) <= 0:
        raise ValueError("Raster cache pixel and tile sizes must be positive.")
    if float(earth_engine["request_timeout_seconds"]) <= 0:
        raise ValueError("Earth Engine request_timeout_seconds must be positive.")
    if not 0 <= int(earth_engine["request_retry_count"]) < 100:
        raise ValueError("Earth Engine request_retry_count must be between 0 and 99.")
    if (
        int(sampling["block_size_meters"]) <= 0
        or int(sampling["samples_per_class_per_block"]) <= 0
        or int(model["fold_count"]) < 2
        or int(model["validation_block_size_meters"]) <= 0
        or int(model["spline_knot_count"]) < 2
        or float(model["ridge_alpha"]) < 0
        or int(inference["window_size_pixels"]) <= 0
    ):
        raise ValueError("Analysis size, count, and penalty settings are invalid.")
    fractions = (
        float(model["minimum_predictor_coverage"]),
        float(model["maximum_row_missing_fraction"]),
        float(model["minimum_response_coverage"]),
        float(inference["covariance_shrinkage"]),
    )
    if any(not 0.0 < value < 1.0 for value in fractions):
        raise ValueError("Analysis coverage, missingness, and shrinkage must be 0-1.")
    identifiers = [band.identifier for band in bands]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Raster-stack band identifiers must be unique.")
    if any(not re.fullmatch(r"d\d+", identifier) for identifier in identifiers):
        raise ValueError("Raster-stack band identifiers must use dNN notation.")
    if any(band.role not in VALID_BAND_ROLES for band in bands):
        raise ValueError(
            "Raster-stack band roles must be reference, response, or predictor."
        )
    if any(band.data_type not in VALID_DATA_TYPES for band in bands):
        raise ValueError(
            "Raster-stack data types must be binary, continuous, or categorical."
        )
    missing_dataset_keys = sorted(
        {
            source_key
            for band in bands
            for source_key in band.source_dataset_keys
            if source_key not in datasets
        }
    )
    if missing_dataset_keys:
        raise ValueError(
            "Raster bands reference undefined dataset aliases: "
            + ", ".join(missing_dataset_keys)
        )
    if sum(band.role == "reference" for band in bands) != 1:
        raise ValueError("A raster stack must define exactly one reference band.")
    categorical_predictors = [
        band
        for band in bands
        if band.role == "predictor" and band.data_type == "categorical"
    ]
    if len(categorical_predictors) != 1:
        raise ValueError(
            "Reference-condition modeling requires exactly one categorical predictor."
        )
    response_identifiers = {
        band.identifier for band in bands if band.role == "response"
    }
    configured_responses = tuple(str(value) for value in model["responses"])
    unknown_responses = sorted(set(configured_responses) - response_identifiers)
    if unknown_responses:
        raise ValueError(
            "Model responses are not configured response bands: "
            + ", ".join(unknown_responses)
        )

    def resolved_local_path(value: str) -> Path:
        return (resolved_path.parent / value).expanduser().resolve()

    resolved_aoi_path = resolved_local_path(str(analysis["aoi_path"]))
    if not resolved_aoi_path.is_file():
        raise ValueError(f"Analysis AOI does not exist: {resolved_aoi_path}")
    grassland_mask_value = inference.get("grassland_mask_path")
    normalized_content = json.dumps(
        raw_configuration,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    raster_content = json.dumps(
        {
            "year": analysis_year,
            "stack": stack,
            "grid": grid,
            "reference": reference,
            "datasets": datasets,
            "bands": raw_configuration["bands"],
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return AnalysisConfiguration(
        path=resolved_path,
        configuration_sha256=hashlib.sha256(normalized_content).hexdigest(),
        raster_configuration_sha256=hashlib.sha256(raster_content).hexdigest(),
        analysis_name=analysis_name,
        display_name=str(analysis["display_name"]),
        aoi_path=resolved_aoi_path,
        year=analysis_year,
        earth_engine=EarthEngineSettings(
            project=str(earth_engine["project"]),
            cache_directory=resolved_local_path(
                str(earth_engine["cache_directory"])
            ),
            request_timeout_seconds=float(
                earth_engine["request_timeout_seconds"]
            ),
            request_retry_count=int(earth_engine["request_retry_count"]),
        ),
        stack_name=stack_name,
        stack_version=int(stack["version"]),
        minimum_year=minimum_year,
        maximum_year=maximum_year,
        reference_start_year=int(stack["reference_start_year"]),
        reference_end_year=int(stack["reference_end_year"]),
        era5_start_year=int(stack["era5_start_year"]),
        interannual_rainfall_window_years=int(
            stack["interannual_rainfall_window_years"]
        ),
        stream_upstream_area_threshold_km2=float(
            stack["stream_upstream_area_threshold_km2"]
        ),
        grid=RasterCacheGrid(
            crs=str(grid["crs"]),
            pixel_size_meters=int(grid["pixel_size_meters"]),
            tile_size_pixels=int(grid["tile_size_pixels"]),
            origin_x=int(grid["origin_x"]),
            origin_y=int(grid["origin_y"]),
        ),
        reference=ReferenceSettings(
            grassland_probability=int(reference["grassland_probability"]),
            human_modification=float(reference["human_modification"]),
            human_influence=float(reference["human_influence"]),
        ),
        sampling=SamplingSettings(
            block_size_meters=int(sampling["block_size_meters"]),
            samples_per_class_per_block=int(
                sampling["samples_per_class_per_block"]
            ),
            random_seed=int(sampling["random_seed"]),
        ),
        model=ModelSettings(
            fold_count=int(model["fold_count"]),
            validation_block_size_meters=int(
                model["validation_block_size_meters"]
            ),
            minimum_predictor_coverage=float(
                model["minimum_predictor_coverage"]
            ),
            maximum_row_missing_fraction=float(
                model["maximum_row_missing_fraction"]
            ),
            spline_knot_count=int(model["spline_knot_count"]),
            minimum_response_coverage=float(model["minimum_response_coverage"]),
            ridge_alpha=float(model["ridge_alpha"]),
            responses=configured_responses,
        ),
        inference=InferenceSettings(
            grassland_mask_path=(
                resolved_local_path(str(grassland_mask_value))
                if grassland_mask_value is not None
                else None
            ),
            window_size_pixels=int(inference["window_size_pixels"]),
            covariance_shrinkage=float(inference["covariance_shrinkage"]),
        ),
        datasets=MappingProxyType(datasets),
        bands=bands,
    )
