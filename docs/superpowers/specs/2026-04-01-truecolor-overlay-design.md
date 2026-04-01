# True Color Overlay Renderers & West Coast Smoke Analysis

**Date:** 2026-04-01
**Status:** Approved
**Branch:** TBD (feature branch off develop)

## Problem

PlumeSentinelAI contains standalone scripts for two high-impact visualizations:

1. **HMS smoke contours on GOES true color** — vector polygons overlaid on satellite imagery
2. **MODIS AOD on MODIS true color** — gridded aerosol optical depth overlaid on NASA GIBS tiles

These scripts work but live outside DAVINCI's pipeline, can't be driven by YAML config, and duplicate infrastructure (styling, grid binning) that DAVINCI already provides. We want these as first-class DAVINCI pipeline outputs.

## Design Decisions

1. **True color backgrounds are a renderer concern, not observation data.** They are visual context, not data being evaluated. Backgrounds are configured in the `plots:` section and rendered by a shared module.

2. **HMS smoke contours are overlay layers, not observations.** Shapefiles don't fit xarray. Overlays are configured in the `plots:` section alongside backgrounds.

3. **MODIS AOD goes through the existing pipeline path.** DAVINCI already has a MODIS L2 AOD reader and SwathGrid pairing strategy. Only the final rendering step is new.

4. **Approach B: shared background/overlay modules** rather than embedding all logic in renderers. Keeps renderers focused, makes backgrounds and overlays reusable across future renderers.

## Architecture

```
plots/
├── backgrounds.py          # NEW: true color background rendering
├── overlays.py             # NEW: vector overlay rendering
└── renderers/
    └── spatial/
        ├── truecolor_aod.py      # NEW: gridded data on true color
        └── truecolor_contour.py  # NEW: true color + vector overlays

config/
└── schema.py               # MODIFIED: new plot config fields

pipeline/
└── stages.py               # MODIFIED: handle plots without pairs

analyses/
└── west-coast-smoke/       # NEW: example analysis
    ├── configs/
    ├── scripts/
    ├── data/
    ├── output/
    └── logs/
```

### Data Flow

**Use Case A: HMS Smoke on GOES True Color**

```
YAML config (plots section)
  ├── background: {type: goes_truecolor, file: ...}
  └── overlays: [{type: hms_smoke, file: ...}]
       │
       ▼
  GeneratePlots stage (no paired data)
       │
       ▼
  truecolor_contour renderer
       ├── backgrounds.render_background(ax, config)  →  GOES RGB imshow
       ├── overlays.render_overlays(ax, config)        →  HMS polygon outlines
       └── map features (states, coastlines, legend)
       │
       ▼
  PNG/PDF output
```

**Use Case B: MODIS AOD on MODIS True Color**

```
YAML config
  ├── obs: {obs_type: sat_swath_clm, ...}     ← existing MODIS L2 reader
  ├── pairs: {strategy: swath_grid, ...}       ← existing SwathGrid pairing
  └── plots:
       ├── background: {type: gibs_wmts, layer: MODIS_Terra_..., date: ...}
       └── type: truecolor_aod
            │
            ▼
       LoadObservations → Pairing (SwathGrid) → GeneratePlots
            │
            ▼
       truecolor_aod renderer
            ├── backgrounds.render_background(ax, config)  →  GIBS WMTS tiles
            ├── AOD imshow with alpha masking
            └── colorbar + map features
            │
            ▼
       PNG/PDF output
```

## Module Specifications

### `plots/backgrounds.py`

Public interface:

```python
def render_background(ax: matplotlib.axes.Axes, config: dict) -> None:
    """Render a true color background on the given axes.

    Dispatches to provider based on config["type"]:
      - "gibs_wmts": NASA GIBS tile service
      - "goes_truecolor": GOES-16 ABI L2 MCMIP NetCDF
    """
```

**GIBS WMTS provider** (`_render_gibs_wmts`):
- Calls `ax.add_wmts(GIBS_URL, layer, wmts_kwargs={"time": date})`
- GIBS URL: `https://gibs.earthdata.nasa.gov/wmts/epsg4326/best/wmts.cgi`
- Config fields: `layer` (str), `date` (str, YYYY-MM-DD)

**GOES true color provider** (`_render_goes_truecolor`):
- Opens NetCDF with `xr.open_dataset()`
- Reads CMI_C01 (blue), CMI_C02 (red), CMI_C03 (veggie/NIR)
- Synthetic green: `0.45 * red + 0.10 * blue + 0.45 * veggie`
- RGB stack, clip to [0, 1], gamma correction: `rgb^(1/gamma)`, default gamma=1.8
- Extracts geostationary projection from `goes_imager_projection` attributes
- Renders via `ax.imshow(rgb, transform=ccrs.Geostationary(...), ...)`
- Config fields: `file` (str, path to NetCDF), `gamma` (float, default 1.8)

### `plots/overlays.py`

Public interface:

```python
def render_overlays(
    ax: matplotlib.axes.Axes,
    overlays: list[dict],
) -> list[matplotlib.patches.Patch]:
    """Render overlay layers, return legend handles for integration."""
```

**HMS smoke provider** (`_render_hms_smoke`):
- Loads shapefile via `geopandas.read_file()`
- Groups by `Density` attribute
- Renders each density class as polygon outlines (facecolor="none"):
  - Light: #FFDD31, linewidth 1.0
  - Medium: #FF8C00, linewidth 1.5
  - Heavy: #D62839, linewidth 2.5
- All at alpha=0.9, zorder=4
- Returns list of `matplotlib.patches.Patch` for legend
- Config fields: `file` (str, path to shapefile)

### `plots/renderers/spatial/truecolor_aod.py`

Registered as `"truecolor_aod"`. Extends `BaseSpatialPlotter`.

```python
@register_plotter("truecolor_aod")
class TrueColorAODPlotter(BaseSpatialPlotter):
    name: str = "truecolor_aod"
```

**`plot()` method:**
1. Create figure with Cartopy projection (Lambert Conformal, configurable)
2. Set map extent from config
3. Call `render_background(ax, config.background)`
4. Extract gridded AOD from paired dataset (obs variable)
5. Build RGBA array: colormap + alpha masking (0.7 where finite, 0.0 where NaN)
6. `ax.imshow(rgba, ...)` with PlateCarree transform
7. Add colorbar (AOD scale)
8. Optionally call `render_overlays()` if configured
9. Add map features (states, coastlines)
10. Return figure

**Config-driven parameters:**
- `background` (dict): background layer config
- `overlays` (list[dict], optional): overlay layers
- `extent` (list[float]): [west, east, south, north]
- `projection` (dict): `{type, central_longitude, ...}`
- `cmap` (str, default "YlOrRd"): colormap name
- `alpha` (float, default 0.7): AOD layer opacity
- Standard `vmin_plot`/`vmax_plot` from variable config for AOD range

### `plots/renderers/spatial/truecolor_contour.py`

Registered as `"truecolor_contour"`. Extends `BaseSpatialPlotter`.

```python
@register_plotter("truecolor_contour")
class TrueColorContourPlotter(BaseSpatialPlotter):
    name: str = "truecolor_contour"
```

**`plot()` method:**
1. Create figure with Cartopy projection
2. Set map extent from config
3. Call `render_background(ax, config.background)`
4. Call `render_overlays(ax, config.overlays)` — returns legend handles
5. Add map features (states, coastlines)
6. Add legend from overlay handles
7. Return figure

**Key difference from truecolor_aod:** No paired data. This renderer is invoked with no dataset — it is purely a visualization of background + overlay layers.

### Config Schema Changes (`config/schema.py`)

Add to `PlotGroupConfig`:

```python
background: dict | None = None      # Background layer config
overlays: list[dict] | None = None   # Overlay layer configs
extent: list[float] | None = None    # Map extent [W, E, S, N]
projection: dict | None = None       # Map projection config
```

Make `data` (pairs reference) optional — currently it is required. When `data` is absent, the plot stage passes no dataset to the renderer.

### Pipeline Stage Changes (`pipeline/stages.py`)

In `GeneratePlots`, add a conditional path:

```python
if plot_config.data:
    # Existing path: iterate over pairs, call renderer.plot(dataset, ...)
else:
    # New path: call renderer.plot(config_only=True, ...)
    # Renderer uses background/overlays from config, no dataset
```

This is a small branch, not a refactor of the stage.

## Example YAML Configs

### `goes-hms-smoke.example.yaml`

```yaml
analysis:
  start_time: "2020-09-09"
  end_time: "2020-09-09"
  output_dir: ${WEST_COAST_SMOKE}/output
  log_dir: ${WEST_COAST_SMOKE}/logs
  style:
    theme: ncar
    context: default

plots:
  goes_hms_smoke:
    type: truecolor_contour
    background:
      type: goes_truecolor
      file: ${WEST_COAST_SMOKE_DATA}/goes16/ABI-L2-MCMIPC_20200909_2001.nc
      gamma: 1.8
    overlays:
      - type: hms_smoke
        file: ${WEST_COAST_SMOKE_DATA}/hms/smoke/2020/hms_smoke20200909.shp
    extent: [-130, -110, 30, 52]
    projection:
      type: lambert_conformal
      central_longitude: -120
    title: "GOES-16 True Color with HMS Smoke Contours — Sep 9, 2020"
```

### `modis-aod-truecolor.example.yaml`

```yaml
analysis:
  start_time: "2020-09-09"
  end_time: "2020-09-09"
  output_dir: ${WEST_COAST_SMOKE}/output
  log_dir: ${WEST_COAST_SMOKE}/logs
  style:
    theme: ncar
    context: default

obs:
  modis_aod:
    obs_type: sat_swath_clm
    filename: ${WEST_COAST_SMOKE_DATA}/modis_aod/MOD04_L2.A2020253.*.061.*.hdf
    variables:
      aod_550:
        units: "1"
        obs_min: 0
        obs_max: 5.0

pairs:
  modis_grid:
    obs: modis_aod
    variable:
      obs_var: aod_550
    strategy: swath_grid
    grid:
      resolution: 0.25

plots:
  modis_aod_truecolor:
    type: truecolor_aod
    data: [modis_grid]
    background:
      type: gibs_wmts
      layer: MODIS_Terra_CorrectedReflectance_TrueColor
      date: "2020-09-09"
    extent: [-130, -110, 30, 52]
    projection:
      type: lambert_conformal
      central_longitude: -120
    cmap: YlOrRd
    alpha: 0.7
    title: "MODIS AOD (550 nm) on Terra True Color — Sep 9, 2020"
```

## Analysis Directory

```
analyses/west-coast-smoke/
├── README.md
├── configs/
│   ├── goes-hms-smoke.example.yaml
│   ├── modis-aod-truecolor.example.yaml
│   └── *-gemini.yaml                    # gitignored
├── scripts/
│   ├── fetch_goes.py                    # GOES-16 ABI L2 MCMIP from AWS S3
│   ├── fetch_hms.py                     # HMS shapefiles from NOAA OSPO
│   ├── fetch_modis_aod.py               # MOD04_L2 via earthaccess
│   └── run_evaluation.py                # Pipeline execution
├── data/                                # gitignored
├── output/                              # gitignored
└── logs/                                # gitignored
```

**Fetch scripts** are ported from PlumeSentinelAI's `scripts/fetch_*.py`, adapted to:
- Use DAVINCI analysis directory conventions
- Accept env var overrides for output paths
- Be idempotent (skip existing files)
- Target the September 9, 2020 West Coast event

**Dev workflow:** Machine-specific `*-gemini.yaml` configs point to `~/Data/PlumeSentinelAI/` directly.

## Dependencies

Add to `environment.yml`:
- `geopandas` — HMS shapefile I/O
- `earthaccess` — NASA Earthdata search/download (fetch scripts)

Add to `pyproject.toml` optional deps (`[dev]` or a new `[smoke]` group).

Both are pure Python with conda packages available. `geopandas` pulls in `fiona` and `shapely` (likely already present via cartopy's shapely dependency).

## Testing Strategy

**Unit tests:**
- `backgrounds.py`: test GOES band math (synthetic green, gamma) with small synthetic arrays; test GIBS dispatch (mock `ax.add_wmts`)
- `overlays.py`: test HMS shapefile rendering with a small synthetic GeoDataFrame; test density filtering and color assignment
- `truecolor_aod.py`: test alpha masking logic (NaN → transparent, finite → opaque)
- `truecolor_contour.py`: test that it calls background + overlay in correct order

**Integration tests:**
- Run `goes-hms-smoke.example.yaml` through `PipelineRunner.run_from_config()` — verify PNG output exists and figure has expected layers
- Run `modis-aod-truecolor.example.yaml` through `PipelineRunner.run_from_config()` — verify full pipeline (load obs → swath grid pairing → truecolor_aod render)

Integration tests require the September 2020 data files and will be marked with `@pytest.mark.slow` or similar skip markers for CI.

## Out of Scope

- PlumeSentinelAI agent framework (Google ADK) — separate concern
- Radiative transfer module — already on ceres branch
- VIIRS AOD overlay — same pattern, can be added later with no design changes
- HRRR visibility overlay — different data type, future work
- Fire detection points overlay — trivial addition to overlays.py later
- Download automation / scheduled fetching
