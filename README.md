# NHI Terrestrial Integrity Modeling

First-pass local analysis scripts for raster stacks exported from Google Earth Engine.

## Load one ecoregion GeoTIFF

Load every band and pixel from one multiband ecoregion export into memory and print
raster metadata, memory use, defined-pixel coverage, approximate defined area, and
per-band descriptive statistics. The same run creates a spatially balanced Parquet
sample in `outputs/samples` and a 300 DPI world locator map in `outputs/figures`:

```powershell
python scripts/load_ecoregion_geotiff.py data\raster_stacks\example.tif
```

The importable `RasterPixelData` object retains a value cube and a separate per-band
validity cube with shape `(bands, rows, columns)`. Its `pixel_values()` and
`pixel_validity()` methods expose pixel-by-band views for later stratification without
copying the arrays. Use `--no-band-report` for only the dataset-level summary or
`--no-progress` to suppress tqdm output. The first map run may download Cartopy's
Natural Earth 1:110 million land geometry.

The spatial sample uses the first Grassland Reference Sites band as a binary class,
with `1` representing a reference site and `0` representing a non-reference site.
Duplicate reference bands are excluded from the predictor table. Eligible pixels are
assigned to 25 km square blocks in an equal-area coordinate system, then up to 100
pixels of each reference-site class are selected independently from every block. The table
records source coordinates, block IDs, pixel area, sampling probabilities, sampling
weights, area weights, and every non-reference raster band. Missing predictor values
remain missing.

Sampling is reproducible with random seed 42. Override the defaults or output path as
needed:

```powershell
python scripts/load_ecoregion_geotiff.py data\raster_stacks\example.tif `
  --sampling-block-size-m 25000 `
  --samples-per-class-per-block 100 `
  --random-seed 42 `
  --sample-output outputs\samples\example.parquet
```

The command prints progress bars plus class counts and areas, block occupancy,
retention rates, weight reconstruction checks, predictor missingness, and verified
Parquet metadata. Use `--no-sampling` when only the raster report and location figure
are needed.

The map label is inferred from the GeoTIFF filename. Supply an explicit PNG, PDF, or
SVG path when vector output or a different destination is needed:

```powershell
python scripts/load_ecoregion_geotiff.py data\raster_stacks\example.tif `
  --location-figure outputs\figures\northern_shortgrass_prairie.svg
```

Use `--no-location-figure` when only the in-memory data and text report are needed.

## Fit and spatially validate an ecoregion GAM

Fit a regularized additive logistic model from one spatial sample Parquet. The model
uses the 2018 environmental bands d20-d39: continuous predictors enter as independent
cubic spline terms, landform enters as a categorical term, and no interactions are
included. The response is the supplied reference-site indicator. Model scores therefore
measure relative similarity to those reference sites; they are not calibrated
probabilities of natural grassland presence.

```powershell
python scripts/fit_ecoregion_gam.py `
  outputs\samples\example_spatial_sample.parquet
```

The validation design combines each 4 by 4 group of 25 km sampling blocks into a
100 km validation block. Whole validation blocks are assigned to one of five folds.
Each row receives one out-of-fold score from the model that did not train on its fold,
then a final model is refit with every usable row.

Predictors covering less than 80% of represented sample area are removed. Rows missing
more than 20% of retained predictors are flagged and excluded from fitting. For every
held-out fold, continuous missing values are replaced with area-weighted training
medians and missing landforms with the area-weighted training mode. These values are
learned from training rows only. Use `--minimum-predictor-coverage` and
`--maximum-row-missing-fraction` to change the defaults.

The command reports predictor coverage, excluded rows and area, fold composition,
imputation, and held-out ranking performance. Outputs under
`outputs/gam/<sample stem>` include:

- A ZSTD Parquet copy of the sample with validation blocks, folds, usability fields,
  and out-of-fold scores.
- Per-fold and aggregate metrics for weighted reference-versus-background AUC,
  continuous Boyce correlation, reference percentile rank, top-area reference
  recovery, and score separation.
- The final serialized additive model, predictor coverage, and run metadata.
- Publication-resolution figures for spatial folds over the sampled valid-pixel
  footprint, score distributions, fold metric variability, and final-model partial
  response curves.

Use `--no-progress` to suppress tqdm output. Block sizes, fold count, spline knots, and
regularization strength are also configurable. Figure titles infer the ecoregion name
from the sample filename; use `--ecoregion-name` to override a truncated or ambiguous
label. Run with `--help` for the complete option list.

## Fit ecological-response reference conditions

Fit separate additive models for the ecological responses in bands d02-d19. Unlike
`fit_ecoregion_gam.py`, which classifies reference versus background, this workflow
trains only on supplied reference rows. Each model estimates the response expected at a
reference site with the pixel's d20-d39 environmental conditions. HMI and HII are not
fitted predictors.

```powershell
python scripts/fit_grassland_integrity_parameters.py `
  outputs\samples\example_spatial_sample.parquet
```

The default run screens all 2018 response bands. Bands with no reference observations,
too little represented-area coverage, no reference-site variation, or inadequate
spatial-fold support are listed with a reason and skipped. Fit a smaller candidate set
with response-band aliases:

```powershell
python scripts/fit_grassland_integrity_parameters.py `
  outputs\samples\example_spatial_sample.parquet `
  --responses d02 d03 d11 d12 d18 d19
```

Each continuous response receives its own regularized additive ridge regression.
Continuous environmental predictors enter as independent cubic splines, landform is
categorical, and no interactions are included. Spatial folds match the reference-site
GAM: each model is trained on reference rows outside one grouped 100 km fold, then
predicts expected reference condition for every usable row inside that fold.

Outputs under `outputs/integrity_parameters/<sample stem>` include:

- A ZSTD Parquet table with out-of-fold expected responses, signed
  observed-minus-expected deviations, and standardized deviations. Standardization
  divides by the held-out reference RMSE for that response.
- Response coverage, fold metrics, response-level metrics, predictor coverage, and an
  area-weighted deviation-correlation table.
- One portable Joblib model bundle per fitted response.
- A standalone Markdown model-selection report and publication-resolution figures for
  spatial folds, held-out fit, observed versus expected values, reference residuals,
  response redundancy, and final-model partial responses.

Area-weighted held-out R2 measures how much spatially held-out reference variation the
environmental model explains. Rank correlation measures whether observed and expected
responses have similar ordering. Fold ranges expose geographic instability. These
diagnostics help choose response parameters; they do not turn a response into an
integrity score by themselves.

A positive standardized deviation means observed is above expected, not necessarily
that ecological integrity is higher. Bare ground, vegetation cover, phenology, and
productivity need explicit ecological direction and weighting before combination. The
sampled zero class is background rather than a verified current-grassland mask, so use
a defensible current-grassland layer before interpreting deviations as present-day
grassland condition. Use `--no-partial-response-figures` for a faster screening run and
`--help` for all coverage, fold, spline, and regularization settings.

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
