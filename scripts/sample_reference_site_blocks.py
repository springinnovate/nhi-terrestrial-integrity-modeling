"""Sample reference-site blocks from a Google Earth Engine AOI.

The script mirrors the grassland reference-site mask from the GEE raster export
app, samples random candidate points in an AOI, keeps points whose centers pass
the reference mask, and starts small Drive export tasks around those centers.
"""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import ee
except ImportError:  # pragma: no cover - exercised only without dependency.
    ee = None


DEFAULT_DRIVE_FOLDER = "gee_reference_site_blocks"
DEFAULT_MAX_PIXELS = 1e13
EXPORT_CRS = "EPSG:4326"

GRASSLAND_PROB_DATASET = (
    "projects/global-pasture-watch/assets/ggc-30m/v1/nat-semi-grassland_p"
)
HMI_DATASET = (
    "projects/hm-30x30/assets/output/v20240801/HMv20240801_2022s_AA_300"
)
HII_DATASET = "projects/HII/v1/hii"

PROBABILITY_INTEGRITY_START_YEAR = 2001
PROBABILITY_INTEGRITY_END_YEAR = 2020
DEFAULT_GRASSLAND_PROB_THRESHOLD = 60
DEFAULT_HMI_THRESHOLD = 0.1
DEFAULT_HII_THRESHOLD = 0.08


@dataclass(frozen=True)
class ReferenceThresholds:
    grassland_probability: float
    hmi: float
    hii: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Randomly sample reference-site centers from an AOI and start small "
            "Google Drive image exports around those centers."
        )
    )
    region = parser.add_mutually_exclusive_group()
    region.add_argument(
        "--region-asset",
        help=(
            "Earth Engine FeatureCollection asset used as the AOI. "
            "If omitted, --bounds or the global default is used."
        ),
    )
    region.add_argument(
        "--bounds",
        nargs=4,
        type=float,
        metavar=("WEST", "SOUTH", "EAST", "NORTH"),
        help="AOI bounds in EPSG:4326. Defaults to the full world.",
    )
    parser.add_argument(
        "--region-name",
        help="Short name used in export task descriptions. Defaults from the asset or bounds.",
    )
    parser.add_argument(
        "--blocks",
        type=int,
        default=15,
        help="Maximum number of reference-site blocks to export. Default: 15.",
    )
    parser.add_argument(
        "--candidate-points",
        type=int,
        default=5000,
        help=(
            "Random candidate points to test inside the AOI. Increase this if too "
            "few points pass the reference-site mask. Default: 5000."
        ),
    )
    parser.add_argument(
        "--block-size-meters",
        type=float,
        default=10_000,
        help="Width/height of each exported block, approximately. Default: 10000.",
    )
    parser.add_argument(
        "--sample-scale",
        type=float,
        default=30,
        help="Scale in meters used to test whether candidate centers are reference sites.",
    )
    parser.add_argument(
        "--export-scale",
        type=float,
        default=30,
        help="Drive export scale in meters for each block. Default: 30.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for candidate points and selecting accepted blocks. Default: 42.",
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=PROBABILITY_INTEGRITY_START_YEAR,
        help="First year in the reference period. Default: 2001.",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=PROBABILITY_INTEGRITY_END_YEAR,
        help="Last year in the reference period. Default: 2020.",
    )
    parser.add_argument(
        "--grassland-prob-threshold",
        type=float,
        default=DEFAULT_GRASSLAND_PROB_THRESHOLD,
        help="Minimum annual grassland probability used in the persistence test.",
    )
    parser.add_argument(
        "--hmi-threshold",
        type=float,
        default=DEFAULT_HMI_THRESHOLD,
        help="Maximum HMI value allowed for reference sites.",
    )
    parser.add_argument(
        "--hii-threshold",
        type=float,
        default=DEFAULT_HII_THRESHOLD,
        help="Maximum scaled HII value allowed in the persistence test.",
    )
    parser.add_argument(
        "--drive-folder",
        default=DEFAULT_DRIVE_FOLDER,
        help=f"Google Drive folder for block exports. Default: {DEFAULT_DRIVE_FOLDER}.",
    )
    parser.add_argument(
        "--description-prefix",
        default="reference_site_block",
        help="Prefix for Earth Engine task descriptions and Drive file names.",
    )
    parser.add_argument(
        "--manifest-csv",
        type=Path,
        help="Optional CSV path to write sampled block centers and task metadata.",
    )
    parser.add_argument(
        "--context-bands",
        action="store_true",
        help=(
            "Export reference_site plus grassland_probability_mean, hii_mean_scaled, "
            "and hmi bands. By default only reference_site is exported."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Sample points and print planned exports without starting Drive tasks.",
    )
    parser.add_argument(
        "--project",
        default="ecoshard",
        help="Optional Google Cloud project passed to ee.Initialize(project=...).",
    )
    parser.add_argument(
        "--tile-scale",
        type=float,
        default=4,
        help="Earth Engine tileScale used while sampling candidate points. Default: 4.",
    )
    parser.add_argument(
        "--max-error",
        type=float,
        default=100,
        help="Maximum geometry error in meters for random points and buffers.",
    )
    parser.add_argument(
        "--max-pixels",
        type=float,
        default=DEFAULT_MAX_PIXELS,
        help=f"maxPixels for each Drive export. Default: {DEFAULT_MAX_PIXELS:.0e}.",
    )
    return parser.parse_args()


def require_earth_engine() -> None:
    if ee is None:
        raise SystemExit(
            "Missing dependency: earthengine-api. Install requirements or run "
            "`pip install earthengine-api`."
        )


def initialize_earth_engine(project: str | None) -> None:
    require_earth_engine()
    if project:
        ee.Initialize(project=project)
    else:
        ee.Initialize()


def slug(text: str) -> str:
    return re.sub(r"(^_+|_+$)", "", re.sub(r"[^a-z0-9]+", "_", text.lower()))


def region_label(args: argparse.Namespace) -> str:
    if args.region_name:
        return slug(args.region_name)
    if args.region_asset:
        return slug(args.region_asset.rstrip("/").split("/")[-1])
    if args.bounds:
        return "bounds"
    return "global"


def region_geometry(args: argparse.Namespace) -> Any:
    if args.region_asset:
        return ee.FeatureCollection(args.region_asset).geometry()
    bounds = args.bounds or [-180, -90, 180, 90]
    return ee.Geometry.Rectangle(bounds, proj=EXPORT_CRS, geodesic=False)


def annual_collection(dataset: str, year: Any) -> Any:
    start = ee.Date.fromYMD(ee.Number(year).toInt(), 1, 1)
    return ee.ImageCollection(dataset).filterDate(
        start, start.advance(1, "year")
    )


def no_two_consecutive_zeros_from_annual_binary(
    build_annual_binary: Any,
    start_year: int,
    end_year: int,
) -> Any:
    years = ee.List.sequence(start_year, end_year)
    annual_binary = ee.ImageCollection.fromImages(
        years.map(
            lambda year: ee.Image(build_annual_binary(ee.Number(year)))
            .rename("g")
            .set("year", year)
        )
    )
    images = annual_binary.toList(annual_binary.size())
    adjacent_passes = ee.ImageCollection.fromImages(
        ee.List.sequence(0, ee.Number(images.size()).subtract(2)).map(
            lambda index: ee.Image(images.get(ee.Number(index))).Or(
                ee.Image(images.get(ee.Number(index).add(1)))
            )
        )
    )
    return adjacent_passes.reduce(ee.Reducer.min()).eq(1)


def grassland_probability_image(year: Any) -> Any:
    return annual_collection(GRASSLAND_PROB_DATASET, year).first().select(0)


def annual_hii_scaled(year: Any) -> Any:
    start = ee.Date.fromYMD(ee.Number(year).toInt(), 1, 1)
    return (
        ee.ImageCollection(HII_DATASET)
        .filterDate(start, start.advance(1, "year"))
        .mean()
        .divide(7000)
    )


def reference_site_mask(
    thresholds: ReferenceThresholds,
    start_year: int,
    end_year: int,
) -> Any:
    grassland_persistent = no_two_consecutive_zeros_from_annual_binary(
        lambda year: grassland_probability_image(year).gte(
            thresholds.grassland_probability
        ),
        start_year,
        end_year,
    )
    hii_persistent = no_two_consecutive_zeros_from_annual_binary(
        lambda year: annual_hii_scaled(year).lt(thresholds.hii),
        start_year,
        end_year,
    )
    hmi_low = ee.Image(HMI_DATASET).lte(thresholds.hmi)

    return (
        grassland_persistent.And(hii_persistent)
        .And(hmi_low)
        .rename("reference_site")
        .selfMask()
        .toByte()
    )


def reference_context_image(
    reference_mask: Any,
    thresholds: ReferenceThresholds,
    start_year: int,
    end_year: int,
) -> Any:
    years = ee.List.sequence(start_year, end_year)
    grassland_mean = ee.ImageCollection.fromImages(
        years.map(lambda year: grassland_probability_image(year))
    ).mean()
    hii_mean = ee.ImageCollection.fromImages(
        years.map(lambda year: annual_hii_scaled(year))
    ).mean()
    hmi = ee.Image(HMI_DATASET)

    return (
        reference_mask.rename("reference_site")
        .unmask(0)
        .addBands(grassland_mean.rename("grassland_probability_mean"))
        .addBands(hii_mean.rename("hii_mean_scaled"))
        .addBands(hmi.rename("hmi"))
        .toFloat()
    )


def sample_reference_points(
    region: Any,
    reference_mask: Any,
    args: argparse.Namespace,
) -> Any:
    candidate_points = ee.FeatureCollection.randomPoints(
        region=region,
        points=args.candidate_points,
        seed=args.seed,
        maxError=args.max_error,
    )
    sampled = reference_mask.sampleRegions(
        collection=candidate_points,
        scale=args.sample_scale,
        geometries=True,
        tileScale=args.tile_scale,
    )
    return (
        sampled.filter(ee.Filter.notNull(["reference_site"]))
        .filter(ee.Filter.eq("reference_site", 1))
        .randomColumn("selection_random", args.seed)
        .limit(args.blocks, "selection_random")
    )


def block_geometry(
    point_geometry: Any, block_size_meters: float, max_error: float
) -> Any:
    return point_geometry.buffer(block_size_meters / 2, max_error).bounds(
        max_error,
        ee.Projection(EXPORT_CRS),
    )


def add_coordinate_properties(feature: Any) -> Any:
    coordinates = feature.geometry().coordinates()
    return feature.set(
        {
            "longitude": coordinates.get(0),
            "latitude": coordinates.get(1),
        }
    )


def write_manifest(
    manifest_csv: Path,
    sampled_points: Any,
    task_rows: list[dict[str, str]],
) -> None:
    manifest_csv.parent.mkdir(parents=True, exist_ok=True)
    point_info = sampled_points.map(add_coordinate_properties).getInfo()
    fieldnames = [
        "block_index",
        "description",
        "task_id",
        "longitude",
        "latitude",
        "reference_site",
        "selection_random",
    ]
    with manifest_csv.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for task_row, feature in zip(task_rows, point_info["features"]):
            properties = feature["properties"]
            writer.writerow(
                {
                    **task_row,
                    "longitude": properties.get("longitude"),
                    "latitude": properties.get("latitude"),
                    "reference_site": properties.get("reference_site"),
                    "selection_random": properties.get("selection_random"),
                }
            )


def queue_block_exports(args: argparse.Namespace) -> None:
    if args.start_year >= args.end_year:
        raise SystemExit("--start-year must be less than --end-year.")
    if args.blocks < 1:
        raise SystemExit("--blocks must be at least 1.")
    if args.candidate_points < args.blocks:
        raise SystemExit(
            "--candidate-points must be greater than or equal to --blocks."
        )

    initialize_earth_engine(args.project)

    thresholds = ReferenceThresholds(
        grassland_probability=args.grassland_prob_threshold,
        hmi=args.hmi_threshold,
        hii=args.hii_threshold,
    )
    region = region_geometry(args)
    label = region_label(args)
    reference_mask = reference_site_mask(
        thresholds, args.start_year, args.end_year
    )
    sampled_points = sample_reference_points(region, reference_mask, args)
    accepted_count = sampled_points.size().getInfo()

    if accepted_count == 0:
        raise SystemExit(
            "No candidate points passed the reference-site mask. Increase "
            "--candidate-points, relax thresholds, or use a smaller/more targeted AOI."
        )
    if accepted_count < args.blocks:
        print(
            f"Only {accepted_count} reference-site point(s) were found from "
            f"{args.candidate_points} candidate point(s)."
        )

    export_base_image = (
        reference_context_image(
            reference_mask,
            thresholds,
            args.start_year,
            args.end_year,
        )
        if args.context_bands
        else reference_mask.unmask(0).rename("reference_site").toByte()
    )

    sampled_list = sampled_points.toList(accepted_count)
    task_rows: list[dict[str, str]] = []
    for index in range(accepted_count):
        point = ee.Feature(sampled_list.get(index))
        block = block_geometry(
            point.geometry(), args.block_size_meters, args.max_error
        )
        description = "_".join(
            [
                slug(args.description_prefix),
                label,
                f"years_{args.start_year}_{args.end_year}",
                f"block_{index + 1:03d}",
            ]
        )
        task_id = ""
        if not args.dry_run:
            task = ee.batch.Export.image.toDrive(
                image=export_base_image.clip(block).toFloat(),
                description=description,
                folder=args.drive_folder,
                fileNamePrefix=description,
                region=block,
                crs=EXPORT_CRS,
                scale=args.export_scale,
                maxPixels=args.max_pixels,
            )
            task.start()
            task_id = task.id
        task_rows.append(
            {
                "block_index": str(index + 1),
                "description": description,
                "task_id": task_id,
            }
        )
        action = "Planned" if args.dry_run else "Started"
        print(f"{action} export {index + 1}/{accepted_count}: {description}")

    if args.manifest_csv:
        write_manifest(args.manifest_csv, sampled_points, task_rows)
        print(f"Wrote manifest to {args.manifest_csv}")


def main() -> None:
    queue_block_exports(parse_args())


if __name__ == "__main__":
    main()
