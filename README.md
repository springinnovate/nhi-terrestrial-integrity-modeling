# NHI Terrestrial Integrity Modeling

First-pass local analysis scripts for raster stacks exported from Google Earth Engine.

## Load one ecoregion GeoTIFF

Load every band and pixel from one multiband ecoregion export into memory and print
raster metadata, memory use, defined-pixel coverage, approximate defined area, and
per-band descriptive statistics:

```powershell
python scripts/load_ecoregion_geotiff.py data\raster_stacks\example.tif
```

The importable `RasterPixelData` object retains a value cube and a separate per-band
validity cube with shape `(bands, rows, columns)`. Its `pixel_values()` and
`pixel_validity()` methods expose pixel-by-band views for later stratification without
copying the arrays. Use `--no-band-report` for only the dataset-level summary or
`--no-progress` to suppress tqdm output.

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

## Reference-site block sampling

Queue small Google Earth Engine Drive exports around randomly sampled grassland
reference-site centers. This uses the same Global Pasture Watch, HMI, and HII
reference mask as `gee_apps/nhi_raster_export_app.js`, but starts tasks from the
Python API so you do not need to click each task in the Code Editor.

```powershell
python scripts/sample_reference_site_blocks.py `
  --region-asset projects/ecoshard-202922/assets/nhi_assets/wyoming_basin2 `
  --region-name "Wyoming Basin" `
  --blocks 15 `
  --candidate-points 5000 `
  --manifest-csv outputs\reference_blocks.csv
```

Useful options:

- `--bounds -180 -90 180 90`: sample a bounding-box AOI, including global bounds.
- `--block-size-meters 10000`: choose the approximate width/height of each block.
- `--candidate-points 20000`: test more random centers when reference sites are sparse.
- `--context-bands`: include reference mask, mean grassland probability, mean HII, and HMI.
- `--dry-run`: sample and print planned exports without starting Drive tasks.
