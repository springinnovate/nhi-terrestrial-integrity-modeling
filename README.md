# NHI Terrestrial Integrity Modeling

First-pass local analysis scripts for raster stacks exported from Google Earth Engine.

## Load one ecoregion GeoTIFF

Load every band and pixel from one multiband ecoregion export into memory and print
raster metadata, memory use, defined-pixel coverage, approximate defined area, and
per-band descriptive statistics. The same run creates a 300 DPI world locator map in
`outputs/figures` with the defined ecoregion footprint, its bounding box, and a labeled
callout:

```powershell
python scripts/load_ecoregion_geotiff.py data\raster_stacks\example.tif
```

The importable `RasterPixelData` object retains a value cube and a separate per-band
validity cube with shape `(bands, rows, columns)`. Its `pixel_values()` and
`pixel_validity()` methods expose pixel-by-band views for later stratification without
copying the arrays. Use `--no-band-report` for only the dataset-level summary or
`--no-progress` to suppress tqdm output. The first map run may download Cartopy's
Natural Earth 1:110 million land geometry.

The map label is inferred from the GeoTIFF filename. Supply an explicit PNG, PDF, or
SVG path when vector output or a different destination is needed:

```powershell
python scripts/load_ecoregion_geotiff.py data\raster_stacks\example.tif `
  --location-figure outputs\figures\northern_shortgrass_prairie.svg
```

Use `--no-location-figure` when only the in-memory data and text report are needed.

## Raster stack table

Build a CSV of pixels where every GeoTIFF has a defined value. Rasters are sampled onto
the grid of a template raster, which defaults to the first GeoTIFF found by sorted path.

```powershell
python scripts/raster_stack_to_table.py path\to\rasters outputs\stack_points.csv --max-rows 100000
```

Useful options:

- `--template path\to\template.tif`: choose the raster grid used for sampling.
- `--max-rows 100000`: limit the output table.
- `--sample-mode random`: reservoir-sample valid pixels while walking the full stack.
- `--resampling bilinear`: resample continuous rasters onto the template grid.
- `--no-progress`: disable the live raster-block progress bar.

The output table includes `x`, `y`, `row`, `col`, followed by one column per raster stem.
A companion metadata JSON is written next to the CSV.

## PCA

Run PCA from the raster stack CSV and write plots plus diagnostics.

```powershell
python scripts/pca_from_table.py outputs\stack_points.csv outputs\pca
```

Filter rows before fitting PCA with `--filter COLUMN EXPR`. Expressions can be exact
values, inclusive ranges, or comparisons: `'=1'`, `'0.2-0.8'`, `'>0.5'`, or
`'<=10'`. Repeat `--filter` to combine multiple filters.
Use `--no-progress` to disable table and PCA workflow progress bars.

```powershell
python scripts/pca_from_table.py 2015_stack.csv outputs\pca_reference --filter grassland_reference_sites_year_2015_wyoming_basin_grassland_prob_90_hmi_0_1_hii_0_08 '=1'
```

Outputs include PCA scores, loadings, explained variance, a scree plot, PC1/PC2 score
scatter, PC1/PC2 loading vectors, a loading heatmap, and loading intensity bars.
