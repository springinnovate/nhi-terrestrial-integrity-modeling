"""Build a point table from overlapping valid pixels in a GeoTIFF stack."""

from __future__ import annotations

import argparse
import csv
import json
import random
from contextlib import ExitStack
from pathlib import Path
from typing import Iterable

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window


RASTER_SUFFIXES = {".tif", ".tiff"}
GRID_COLUMNS = ["x", "y", "row", "col"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Walk a directory of GeoTIFF rasters, sample them onto a template grid, "
            "and write rows where every raster has a finite, non-nodata value."
        )
    )
    parser.add_argument("input_dir", type=Path, help="Directory containing GeoTIFF rasters.")
    parser.add_argument("output_csv", type=Path, help="CSV table to write.")
    parser.add_argument(
        "--template",
        type=Path,
        help="Template raster for output point locations. Defaults to first raster found.",
    )
    parser.add_argument(
        "--band",
        type=int,
        default=1,
        help="Band number to read from each raster. Default: 1.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        help="Maximum rows to write. With random sampling, the full stack is still scanned.",
    )
    parser.add_argument(
        "--sample-mode",
        choices=("first", "random"),
        default="random",
        help="How to apply --max-rows. Default: random.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=250_000,
        help="Approximate number of template pixels to process per chunk. Default: 250000.",
    )
    parser.add_argument(
        "--resampling",
        choices=("nearest", "bilinear", "cubic", "average"),
        default="nearest",
        help="Resampling used when rasters do not match the template grid. Default: nearest.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output files if they already exist.",
    )
    return parser.parse_args()


def find_rasters(input_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in RASTER_SUFFIXES
    )


def unique_stems(paths: Iterable[Path]) -> list[str]:
    counts: dict[str, int] = {}
    names: list[str] = []
    for path in paths:
        stem = path.stem
        counts[stem] = counts.get(stem, 0) + 1
        names.append(stem if counts[stem] == 1 else f"{stem}_{counts[stem]}")
    return names


def same_grid(dataset: rasterio.DatasetReader, template: rasterio.DatasetReader) -> bool:
    return (
        dataset.crs == template.crs
        and dataset.transform == template.transform
        and dataset.width == template.width
        and dataset.height == template.height
    )


def open_on_template_grid(
    stack: ExitStack,
    path: Path,
    template: rasterio.DatasetReader,
    resampling: Resampling,
) -> rasterio.io.DatasetReader:
    dataset = stack.enter_context(rasterio.open(path))
    if same_grid(dataset, template):
        return dataset
    return stack.enter_context(
        WarpedVRT(
            dataset,
            crs=template.crs,
            transform=template.transform,
            width=template.width,
            height=template.height,
            resampling=resampling,
        )
    )


def pixel_centers(
    transform: rasterio.Affine,
    row_start: int,
    rows: int,
    cols: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    row_grid, col_grid = np.indices((rows, cols))
    abs_rows = row_grid + row_start
    x = (
        transform.a * (col_grid + 0.5)
        + transform.b * (abs_rows + 0.5)
        + transform.c
    )
    y = (
        transform.d * (col_grid + 0.5)
        + transform.e * (abs_rows + 0.5)
        + transform.f
    )
    return x, y, abs_rows, col_grid


def read_masked(
    dataset: rasterio.DatasetReader,
    band: int,
    window: Window,
) -> np.ma.MaskedArray:
    data = dataset.read(band, window=window, masked=True)
    data = np.ma.masked_invalid(data)
    return data


def valid_data_mask(array: np.ma.MaskedArray) -> np.ndarray:
    data = np.asarray(array)
    valid = ~np.ma.getmaskarray(array)
    if np.issubdtype(data.dtype, np.floating):
        valid &= np.isfinite(data)
    return valid


def add_rows_to_reservoir(
    reservoir: list[list[object]],
    rows: Iterable[list[object]],
    max_rows: int,
    seen_count: int,
    rng: random.Random,
) -> int:
    for row in rows:
        seen_count += 1
        if len(reservoir) < max_rows:
            reservoir.append(row)
            continue
        replacement_index = rng.randrange(seen_count)
        if replacement_index < max_rows:
            reservoir[replacement_index] = row
    return seen_count


def chunk_rows(
    readers: list[rasterio.DatasetReader],
    column_names: list[str],
    template: rasterio.DatasetReader,
    band: int,
    chunk_size: int,
) -> Iterable[list[list[object]]]:
    width = template.width
    rows_per_chunk = max(1, chunk_size // max(width, 1))

    for row_start in range(0, template.height, rows_per_chunk):
        height = min(rows_per_chunk, template.height - row_start)
        window = Window(0, row_start, width, height)
        arrays = [read_masked(reader, band, window) for reader in readers]

        valid = np.ones((height, width), dtype=bool)
        for array in arrays:
            valid &= valid_data_mask(array)

        if not valid.any():
            yield []
            continue

        x, y, rows, cols = pixel_centers(template.transform, row_start, height, width)
        valid_indices = np.where(valid)
        values = [np.asarray(array)[valid_indices] for array in arrays]

        chunk: list[list[object]] = []
        for i in range(valid_indices[0].size):
            row = [
                float(x[valid_indices][i]),
                float(y[valid_indices][i]),
                int(rows[valid_indices][i]),
                int(cols[valid_indices][i]),
            ]
            row.extend(float(variable_values[i]) for variable_values in values)
            chunk.append(row)
        yield chunk


def write_metadata(
    metadata_path: Path,
    raster_paths: list[Path],
    column_names: list[str],
    template_path: Path,
    template: rasterio.DatasetReader,
    valid_pixel_count: int,
    written_row_count: int,
    args: argparse.Namespace,
) -> None:
    metadata = {
        "input_dir": str(args.input_dir),
        "output_csv": str(args.output_csv),
        "template": str(template_path),
        "template_crs": str(template.crs),
        "template_transform": list(template.transform),
        "template_width": template.width,
        "template_height": template.height,
        "band": args.band,
        "resampling": args.resampling,
        "max_rows": args.max_rows,
        "sample_mode": args.sample_mode,
        "valid_pixel_count": valid_pixel_count,
        "written_row_count": written_row_count,
        "rasters": [
            {"column": column, "path": str(path)}
            for column, path in zip(column_names, raster_paths)
        ],
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if not args.input_dir.exists():
        raise SystemExit(f"Input directory does not exist: {args.input_dir}")
    if args.output_csv.exists() and not args.overwrite:
        raise SystemExit(f"Output CSV already exists: {args.output_csv}")

    metadata_path = args.output_csv.with_suffix(args.output_csv.suffix + ".metadata.json")
    if metadata_path.exists() and not args.overwrite:
        raise SystemExit(f"Metadata JSON already exists: {metadata_path}")

    raster_paths = [path.resolve() for path in find_rasters(args.input_dir)]
    if not raster_paths:
        raise SystemExit(f"No GeoTIFF rasters found under: {args.input_dir}")

    if args.template:
        template_path = args.template.resolve()
        if not template_path.exists():
            raise SystemExit(f"Template raster does not exist: {template_path}")
    else:
        template_path = raster_paths[0]

    column_names = unique_stems(raster_paths)
    header = GRID_COLUMNS + column_names
    resampling = getattr(Resampling, args.resampling)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)

    valid_pixel_count = 0
    written_row_count = 0
    rng = random.Random(42)
    reservoir: list[list[object]] = []

    with ExitStack() as stack:
        template = stack.enter_context(rasterio.open(template_path))
        readers = [
            open_on_template_grid(stack, path, template, resampling)
            for path in raster_paths
        ]

        if args.max_rows and args.sample_mode == "random":
            for chunk in chunk_rows(readers, column_names, template, args.band, args.chunk_size):
                valid_pixel_count += len(chunk)
                valid_seen_before = written_row_count
                written_row_count = add_rows_to_reservoir(
                    reservoir,
                    chunk,
                    args.max_rows,
                    written_row_count,
                    rng,
                )
                if valid_seen_before == written_row_count and not chunk:
                    continue
            rows_to_write = reservoir
            written_row_count = len(rows_to_write)
            with args.output_csv.open("w", newline="", encoding="utf-8") as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(header)
                writer.writerows(rows_to_write)
        else:
            with args.output_csv.open("w", newline="", encoding="utf-8") as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(header)
                stop = False
                for chunk in chunk_rows(
                    readers, column_names, template, args.band, args.chunk_size
                ):
                    valid_pixel_count += len(chunk)
                    if args.max_rows:
                        remaining = args.max_rows - written_row_count
                        chunk = chunk[:remaining]
                    writer.writerows(chunk)
                    written_row_count += len(chunk)
                    if args.max_rows and written_row_count >= args.max_rows:
                        stop = True
                    if stop:
                        break

        write_metadata(
            metadata_path,
            raster_paths,
            column_names,
            template_path,
            template,
            valid_pixel_count,
            written_row_count,
            args,
        )

    print(f"Wrote {written_row_count} row(s) to {args.output_csv}")
    print(f"Found {valid_pixel_count} overlapping valid pixel(s)")
    print(f"Wrote metadata to {metadata_path}")


if __name__ == "__main__":
    main()
