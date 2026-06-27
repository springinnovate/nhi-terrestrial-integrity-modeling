# NHI Terrestrial Integrity Modeling

First-pass local analysis scripts for raster stacks exported from Google Earth Engine.

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

The output table includes `x`, `y`, `row`, `col`, followed by one column per raster stem.
A companion metadata JSON is written next to the CSV.

## PCA

Run PCA from the raster stack CSV and write plots plus diagnostics.

```powershell
python scripts/pca_from_table.py outputs\stack_points.csv outputs\pca
```

Outputs include PCA scores, loadings, explained variance, a scree plot, PC1/PC2 score
scatter, PC1/PC2 loading vectors, a loading heatmap, and loading intensity bars.
