"""Load the shared raster-stack definition used throughout the pipeline."""

from __future__ import annotations

import hashlib
import json
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Sequence


DEFAULT_RASTER_STACK_CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "config"
    / "reference_condition_raster_stack.toml"
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
class ReferenceThresholdDefaults:
    """Store default thresholds used to construct reference sites.

    Attributes:
        grassland_probability: Minimum annual grassland probability percentage.
        human_modification: Maximum human-modification index.
        human_influence: Maximum scaled human-influence index.
    """

    grassland_probability: int
    human_modification: float
    human_influence: float


@dataclass(frozen=True)
class RasterStackConfiguration:
    """Represent one validated raster-stack and modeling contract.

    Attributes:
        path: Resolved TOML source path.
        configuration_sha256: Hash of normalized effective TOML content.
        name: File-safe stack name.
        version: Explicit stack-definition version.
        minimum_year: First supported complete-stack year.
        maximum_year: Last supported complete-stack year.
        reference_start_year: First year used in reference persistence tests.
        reference_end_year: Last year used in reference persistence tests.
        era5_start_year: First year available to rainfall-window calculations.
        interannual_rainfall_window_years: Rainfall variability window length.
        stream_upstream_area_threshold_km2: Stream extraction threshold.
        grid: Deterministic raster cache grid.
        reference_defaults: Default reference-site thresholds.
        datasets: Earth Engine dataset IDs keyed by stable aliases.
        bands: Ordered raster bands included in this stack.
    """

    path: Path
    configuration_sha256: str
    name: str
    version: int
    minimum_year: int
    maximum_year: int
    reference_start_year: int
    reference_end_year: int
    era5_start_year: int
    interannual_rainfall_window_years: int
    stream_upstream_area_threshold_km2: float
    grid: RasterCacheGrid
    reference_defaults: ReferenceThresholdDefaults
    datasets: Mapping[str, str]
    bands: tuple[RasterBandDefinition, ...]

    def band_names(self, year: int) -> tuple[str, ...]:
        """Return ordered export names for the configured bands.

        Args:
            year: Four-digit source-data year.

        Returns:
            Stable Earth Engine export names in configured order.
        """

        return tuple(
            f"y{year}_{band.identifier}_{band.suffix}" for band in self.bands
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
                    (
                        int(match.group("year")),
                        match.group("band_id"),
                        column,
                    )
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


def load_raster_stack_configuration(
    configuration_path: Path = DEFAULT_RASTER_STACK_CONFIG_PATH,
) -> RasterStackConfiguration:
    """Load and validate one TOML raster-stack definition.

    Configuration is an external contract, so malformed identifiers, duplicate
    bands, unknown roles, and missing source aliases are rejected before Earth
    Engine or model processing starts.

    Args:
        configuration_path: TOML raster-stack definition.

    Returns:
        Immutable typed configuration with a normalized content hash.

    Raises:
        ValueError: If the stack contract is internally inconsistent.
    """

    resolved_path = configuration_path.expanduser().resolve()
    with resolved_path.open("rb") as configuration_file:
        raw_configuration = tomllib.load(configuration_file)

    stack = raw_configuration["stack"]
    grid = raw_configuration["grid"]
    reference_defaults = raw_configuration["reference_defaults"]
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
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", str(stack["name"])):
        raise ValueError(
            "Raster-stack names must contain lowercase letters, numbers, hyphens, "
            "or underscores."
        )
    if int(stack["minimum_year"]) > int(stack["maximum_year"]):
        raise ValueError("Raster-stack minimum_year cannot exceed maximum_year.")
    if int(grid["pixel_size_meters"]) <= 0 or int(grid["tile_size_pixels"]) <= 0:
        raise ValueError("Raster cache pixel and tile sizes must be positive.")
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

    normalized_content = json.dumps(
        raw_configuration,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return RasterStackConfiguration(
        path=resolved_path,
        configuration_sha256=hashlib.sha256(normalized_content).hexdigest(),
        name=str(stack["name"]),
        version=int(stack["version"]),
        minimum_year=int(stack["minimum_year"]),
        maximum_year=int(stack["maximum_year"]),
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
        reference_defaults=ReferenceThresholdDefaults(
            grassland_probability=int(
                reference_defaults["grassland_probability"]
            ),
            human_modification=float(reference_defaults["human_modification"]),
            human_influence=float(reference_defaults["human_influence"]),
        ),
        datasets=MappingProxyType(datasets),
        bands=bands,
    )
