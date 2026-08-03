# NHI Terrestrial Integrity Modeling

First-pass local analysis scripts for raster stacks exported from Google Earth Engine.

## Cache Earth Engine raster tiles for an AOI

Fetch an analysis-defined raster stack directly from Earth Engine without retaining a
complete AOI-sized export. Authenticate the Earth Engine Python client once, then
pass the complete analysis definition:

```powershell
earthengine authenticate

python -m scripts.fetch_gee_raster_tiles `
  config\south_africa_reference_condition_analysis.toml
```

`config/south_africa_reference_condition_analysis.toml` defines the project's
`d01-d39` multiband stack. The calculations originated in
`gee_apps/nhi_raster_export_app.js`, while the TOML now selects and orders the stack
used by the Python pipeline. It covers the configured AOI with globally aligned 128
by 128 pixel tiles in the NSIDC EASE-Grid 2.0 Global equal-area projection
(`EPSG:6933`) at 500 m resolution. Each tile is a 64 km square. Earth Engine's
`computePixels` endpoint computes only missing tiles, and overlapping AOI requests
reuse the same cache files. The default complete stack is available for 2015 through
2019.

TOML is a configuration format similar in purpose to YAML, with explicit sections,
key-value pairs, and repeated `[[bands]]` tables. One analysis file controls its
identity and display name, local AOI, year, Earth Engine project and cache, cache grid,
reference thresholds, sampling design, model settings and selected responses,
inference settings, Earth Engine datasets, and ordered bands. Local paths are resolved
relative to the TOML. Each band declares a stable ID, Python computation key, output
suffix, display name, pipeline role (`reference`, `response`, or `predictor`), data
type, and source aliases. Python still implements calculations such as phenology and
growing-season aggregation; TOML selects and orders those implementations.

The TOML-defined AOI is a general-purpose WGS84 polygon or multipolygon and is the
only project boundary used to select cache tiles. Edge tiles are stored in full so
later overlapping AOIs can reuse them. The `d01` reference criteria are evaluated
without a possible-grassland ecoregion boundary, and no project boundary masks the
ecological-response or environmental bands in `d02-d39`. Exact AOI clipping can be
applied when cached tiles are assembled without changing the shared cache contents.

Cached GeoTIFFs are stored by year and reference-threshold configuration under
`data/gee_raster_cache/tiles`. `data/gee_raster_cache/manifest.json` records the
grid, source datasets, data year, exact band schema, thresholds, tile bounds and
transform, pixel size, fetch timestamp, file size, SHA-256 checksum, and AOI request
history, configuration path, full analysis SHA-256, and raster-effective SHA-256. A
tile is entered into the manifest only after its temporary download has passed CRS,
alignment,
dimensions, band-name, and checksum validation.

Repeating a request validates and reuses existing files. Use `--refresh` to replace
every intersecting tile. Reference thresholds are changed in the analysis TOML; each
distinct configuration receives a separate cache namespace.
The namespace includes the stack version and raster-effective hash, so changing a
source, grid, role, label, year, threshold, or band order cannot silently reuse older
tiles. Sampling, model, and inference changes update the full analysis hash without
redownloading identical source pixels.
The command reports AOI area, requested tiles, cache hits, downloads, transferred
bytes, failures, and manifest location. A cache-validation progress bar first checks
every intersecting grid. A second tqdm bar then reports processed, cached,
downloaded, and failed grid counts while Earth Engine requests run.

Each synchronous Earth Engine request uses the timeout and retry count in the
`[earth_engine]` TOML section. If all configured attempts fail, the command stops
immediately instead of trying every remaining tile during an outage. Every tile
validated before the error remains in the manifest and is reused when the same
analysis is run again. `computePixels` requests are interactive calls rather than
persistent Earth Engine batch tasks, so an in-flight request cannot be recovered
after the process restarts.

## Build a spatial sample from cached tiles

After fetching the analysis, build its model-ready sample directly from the validated
Earth Engine tile cache:

```powershell
python -m scripts.build_spatial_sample `
  config\south_africa_reference_condition_analysis.toml
```

The TOML is the only scientific input. It identifies the AOI, cache directory,
raster-effective stack, globally anchored grid, ordered band schema, reference band,
sampling-block size, per-class cap, and random seed. The sampler recomputes the AOI's
required tile addresses, verifies each manifest entry, checksum, grid, transform, and
band description, and stops with a fetch/resume command if the cache is incomplete.
An unrelated one-band mask therefore cannot be mistaken for the multiband stack.

Tiles are read one at a time. Edge tiles are clipped to pixel centers inside the
configured AOI, and no AOI-sized value or validity cube is allocated. A first pass
counts source populations and assigns every eligible pixel a deterministic seeded
priority. The lowest priorities are retained for each global sampling-block and
reference-class combination. This merge is global, so a 25 km sampling block crossing
one or more 64 km cache-tile boundaries still receives only one configured per-class
cap. A second pass rereads only tiles containing retained candidates and extracts
their raster values.

The sample uses the band assigned the TOML `reference` role as a binary class, with
`1` representing a reference site and every other eligible pixel representing
background. The table records globally stable grid coordinates, cache-tile and block
IDs, longitude, latitude, pixel area, sampling probabilities, sampling weights, area
weights, and every non-reference raster band. Missing values remain missing. The
compressed Parquet schema embeds the analysis hashes, stack identifier, cache
manifest, tile-set fingerprint, and sampling settings.

By default, the command writes
`outputs/samples/<analysis-name>_spatial_sample.parquet` and a 300 DPI AOI locator map
to `outputs/figures/<analysis-name>_world_location.png`. Output destinations remain
operational options:

```powershell
python -m scripts.build_spatial_sample `
  config\south_africa_reference_condition_analysis.toml `
  --sample-output outputs\samples\south_africa.parquet `
  --location-figure outputs\figures\south_africa.png
```

Validation, scanning, selected-pixel extraction, Parquet writing, and figure creation
have separate tqdm stages. Reports include cache size, the largest source-tile memory
allocation, AOI and band coverage, retained class counts, represented area, block
occupancy, weight reconstruction, missingness, and verified Parquet metadata. Use
`--no-band-report`, `--no-location-figure`, or `--no-progress` to suppress optional
output.

## Convert a numeric raster to a binary mask

Convert one byte, integer, or floating-point raster band into a single-band 0/1
GeoTIFF with the same width, height, CRS, transform, and pixel alignment as the
source:

```powershell
python scripts/utils/raster_to_binary_mask.py `
  data\continuous_value.tif `
  outputs\masks\value_at_least_80.tif `
  ">=80"
```

The comparison is required and must use one of `>`, `<`, `>=`, `<=`, or `==`
followed by a number. Signed values, decimals, and scientific notation are accepted,
for example `">-2.5"`, `"<=0.25"`, and `">=1e-3"`. Quote the expression so the
shell does not interpret `<` or `>` as redirection. Equality is exact, including for
floating-point inputs.

The output uses `uint8` values with one-bit storage, 512-pixel tiles, ZSTD
compression, and BigTIFF when needed. Source nodata, masked, NaN, and infinite pixels
become valid output zeros, so every output pixel is strictly `0` or `1`. Processing is
windowed and does not load the complete raster into memory.

Add `--cog` to copy the completed mask through GDAL's Cloud Optimized GeoTIFF driver
after classification finishes, also using ZSTD compression and nearest-neighbor mask
overviews:

```powershell
python scripts/utils/raster_to_binary_mask.py `
  D:\eolab_data\nat_semi_grassland_p\nat_semi_grassland_p_2018.tif `
  outputs\masks\nat_semi_grassland_p_2018_80.tif `
  ">=80" `
  --cog
```

COG creation uses a temporary tiled GeoTIFF in the output directory, so that volume
must have room for the intermediate and final compressed files during conversion.
The `Creating COG` stage reports GDAL's completion percentage when its Python bindings
are installed. The portable Rasterio fallback reports bytes written instead. Use
`--band` for a band other than one, `--window-size-pixels` to tune memory and I/O,
`--overwrite` to replace an existing output only after validation succeeds, and
`--no-progress` to suppress both progress stages.

## Fit ecological-response reference conditions

Fit separate additive models for bands assigned the TOML `response` role. The workflow
trains only on supplied reference rows. Each model estimates the response expected at
a reference site using bands assigned the `predictor` role. HMI and HII are not fitted
predictors.

```powershell
python -m scripts.fit_grassland_integrity_parameters `
  config\south_africa_reference_condition_analysis.toml `
  outputs\samples\south_africa_reference_condition_2018_spatial_sample.parquet
```

The model-run metadata and each serialized model record the analysis identity, year,
stack name and version, and configuration hash.

The default run screens all configured response bands, using the earliest available
year when a sample contains repeated years. Bands with no reference observations, too
little represented-area coverage, no reference-site variation, or inadequate
spatial-fold support are listed with a reason and skipped. Set `model.responses` to
response-band aliases such as `["d02", "d03", "d11"]` to fit a smaller candidate
set; an empty list selects every configured response.

Each continuous response receives its own regularized additive ridge regression.
Configured continuous predictors enter as independent cubic splines, the configured
categorical predictor enters as one-hot categories, and no interactions are included.
The validation design combines each 4 by 4 group of 25 km sampling blocks into a 100
km validation block. Whole validation blocks are assigned to one of five folds. Each
model is trained on reference rows outside one grouped block fold, then predicts
expected reference condition for every usable row inside that fold.

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
`--help` for output and progress controls. Model tuning, response selection, and the
display name live only in the analysis TOML rather than being repeated as CLI options.

Shared reference-condition preparation lives in
`scripts/reference_condition_utils.py`. It is an imported library module rather than a
runnable command. The cache-backed sampler and response-model script use it for
consistent analysis naming, equal-area spatial configuration, predictor screening, fold
assignment, training-only imputation, weighted quantiles, and spatial-fold figures.

## Apply reference-condition models to a raster

Apply every final ecological-response model from one completed model run to its
ecoregion raster stack:

```powershell
python -m scripts.apply_reference_condition_models `
  config\south_africa_reference_condition_analysis.toml `
  data\raster_stacks\example.tif `
  outputs\integrity_parameters\south_africa_reference_condition_2018_spatial_sample
```

The command processes fixed raster windows rather than loading all inference products
into memory. For each fitted response it writes the final model's expected reference
value, observed-minus-expected deviation, and standardized deviation. Standardized
deviation divides by the pooled out-of-fold reference RMSE stored with that response
model. It also converts each complete standardized-departure vector into a
covariance-aware Mahalanobis distance and an area-weighted reference percentile. No
models are retrained during inference.

Outputs under `outputs/reference_condition_inference/<ecoregion>` include:

- `<ecoregion>_expected_reference.tif`
- `<ecoregion>_observed_minus_expected.tif`
- `<ecoregion>_standardized_deviation.tif`
- `<ecoregion>_reference_departure_percentile.tif`
- `<ecoregion>_inference_status.tif`
- `<ecoregion>_aggregate_standardized_deviation.png`
- `<ecoregion>_reference_departure_percentile.png`
- `<ecoregion>_inference_report.md`
- `<ecoregion>_inference_metadata.json`

The expected, raw-deviation, and standardized-deviation GeoTIFFs contain one aligned
band per fitted response. The percentile GeoTIFF contains one aligned band. The
status GeoTIFF contains an inference-status band and a missing-predictor-count band.
Status 0 is outside the inference target, status 1 exceeds the training missingness
threshold, and status 2 received model predictions. Pixels within the threshold use
the final reference-training imputation values stored in each model.

The aggregate PNG makes the raster result visible at publication resolution. For
each source pixel with every modeled response defined, it calculates
`sum(abs(z_j))` across all responses. It then enlarges the result to at most 700
display cells along the longest raster dimension by taking the mean among
non-reference pixels in each display cell. Green indicates lower total standardized
departure and red indicates larger departure. A fixed linear scale maps 0 to green,
3 to yellow-green, and values of 10 or more to red. Black outlines show display
cells containing pixels from the analysis-year reference-site band. Reference
pixels do not contribute to the colored values. This aggregate is a diagnostic, not
an integrity score, and responses with similar ecological information can be counted
more than once.

The reference-departure percentile uses only complete reference rows from
`ecological_response_predictions.parquet`. It calculates their area-weighted mean
standardized-departure vector and covariance matrix from out-of-fold predictions,
then shrinks the configured fraction of covariance toward its diagonal before
inversion. The shrinkage retains each response's variance while reducing instability
from strongly correlated responses such as NPP and GPP. Change
`inference.covariance_shrinkage` in the analysis TOML to change the fraction.

For every complete non-reference raster pixel, the script calculates a Mahalanobis
distance from the reference center and writes `P_i`: the represented-area fraction of
complete reference rows with an equal or smaller distance. Thus, `P_i=0.95` means the
pixel is farther from the reference center than 95% of represented calibration area.
The aligned single-band GeoTIFF uses the fixed 0 to 1 scale. Reference pixels and pixels
missing any fitted response are nodata. The corresponding PNG shows reference-site
display cells in deep violet with white boundaries and mean non-reference `P_i` from
green at 0 through red at 1. The report records complete-reference coverage,
covariance conditioning, reference distance quantiles, raster coverage, and
upper-percentile frequencies.

`P_i` is a multivariate reference-condition departure percentile, not proof of
degradation or an ecological integrity score. Keep the individual standardized
response rasters to identify the variables and directions responsible for a large
departure.

`analysis.aoi_path` and `inference.application_mask_path` have separate purposes.
The vector AOI determines which Earth Engine data are fetched. The application mask
determines which pixels in the supplied raster stack receive predictions, so it
belongs to the inference section even though the complete TOML defines one analysis.
Changing the mask does not redefine reference sites or require refetching or
refitting.

Set `inference.application_mask_path` to a raster whose defined first-band pixels
equal to `1` identify the inference target. The mask may have a different CRS,
resolution, transform, or global extent; it is aligned to each raster window with
nearest-neighbor resampling without loading the complete mask into memory. Zeros,
other values, nodata, and pixels outside the mask extent are excluded:

```powershell
python -m scripts.apply_reference_condition_models `
  config\south_africa_reference_condition_analysis.toml `
  data\raster_stacks\example.tif `
  outputs\integrity_parameters\south_africa_reference_condition_2018_spatial_sample
```

Without `inference.application_mask_path`, the script infers across the usable
ecoregion predictor footprint and marks the report accordingly. Unmasked outputs are
diagnostic reference-condition deviations, not grassland integrity maps. Positive
standardized deviation means observed is above expected. The percentile combines
multivariate departure without assigning ecological direction, so integrity
interpretation remains a separate modeling decision.
