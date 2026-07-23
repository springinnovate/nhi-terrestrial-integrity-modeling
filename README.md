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

## Fit ecological-response reference conditions

Fit separate additive models for the ecological responses in bands d02-d19. The
workflow trains only on supplied reference rows. Each model estimates the response
expected at a reference site with the pixel's d20-d39 environmental conditions. HMI
and HII are not fitted predictors.

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
categorical, and no interactions are included. The validation design combines each 4
by 4 group of 25 km sampling blocks into a 100 km validation block. Whole validation
blocks are assigned to one of five folds. Each model is trained on reference rows
outside one grouped block fold, then predicts expected reference condition for every
usable row inside that fold.

Predictors covering less than 80% of represented sample area are removed. Rows missing
more than 20% of retained predictors are flagged and excluded from fitting. For every
held-out fold, continuous missing values are replaced with area-weighted training
medians and missing landforms with the area-weighted training mode. These values are
learned from training rows only.

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
`--help` for response selection, output, labeling, and progress controls. Model-tuning
defaults live in `IntegrityConfiguration` rather than being repeated as CLI options.

Shared reference-condition preparation lives in
`scripts/reference_condition_utils.py`. It is an imported library module rather than a
runnable command. The GeoTIFF loader and response-model script use it for consistent
ecoregion naming, equal-area spatial configuration, predictor screening, fold
assignment, training-only imputation, weighted quantiles, and spatial-fold figures.

## Apply reference-condition models to a raster

Apply every final ecological-response model from one completed model run to its
ecoregion raster stack:

```powershell
python scripts/apply_reference_condition_models.py `
  data\raster_stacks\example.tif `
  outputs\integrity_parameters\example_spatial_sample
```

The command processes fixed raster windows rather than loading all inference products
into memory. For each fitted response it writes the final model's expected reference
value, observed-minus-expected deviation, and standardized deviation. Standardized
deviation divides by the pooled out-of-fold reference RMSE stored with that response
model. No models are retrained during inference.

Outputs under `outputs/reference_condition_inference/<ecoregion>` include:

- `<ecoregion>_expected_reference.tif`
- `<ecoregion>_observed_minus_expected.tif`
- `<ecoregion>_standardized_deviation.tif`
- `<ecoregion>_inference_status.tif`
- `<ecoregion>_aggregate_standardized_deviation.png`
- `<ecoregion>_inference_report.md`
- `<ecoregion>_inference_metadata.json`

The three float GeoTIFFs contain one aligned band per fitted response. The status
GeoTIFF contains an inference-status band and a missing-predictor-count band. Status 0
is outside the inference target, status 1 exceeds the training missingness threshold,
and status 2 received model predictions. Pixels within the threshold use the final
reference-training imputation values stored in each model.

The aggregate PNG makes the raster result visible at publication resolution. For
each source pixel with every modeled response defined, it calculates
`sum(abs(z_j))` across all responses. It then enlarges the result to at most 700
display cells along the longest raster dimension by taking the mean among
non-reference pixels in each display cell. Green indicates lower total standardized
departure and red indicates larger departure; the red endpoint is capped at the 95th
percentile of displayed values. Black outlines show display cells containing pixels
from the raster stack's 2018 reference-site band. Reference pixels do not contribute
to the colored values. This aggregate is a diagnostic, not an integrity score, and
responses with similar ecological information can be counted more than once.

Supply an exactly aligned mask whose defined nonzero first-band pixels identify the
target when a current-grassland layer is available:

```powershell
python scripts/apply_reference_condition_models.py `
  data\raster_stacks\example.tif `
  outputs\integrity_parameters\example_spatial_sample `
  --grassland-mask data\masks\example_current_grassland.tif
```

Without `--grassland-mask`, the script infers across the usable ecoregion predictor
footprint and marks the report accordingly. Unmasked outputs are diagnostic
reference-condition deviations, not grassland integrity maps. Positive standardized
deviation means observed is above expected; ecological direction and response
combination remain separate modeling decisions.
