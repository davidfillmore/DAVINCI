# Swath Grid Pairing Strategy — Design Document

**Date**: 2026-03-08
**Status**: Smoke test complete — grid_binning.py implemented, multi-day multi-model analysis working
**Branch**: feature/modis

## Problem

The existing `SwathStrategy` does per-pixel xarray `.isel()` calls in a Python for loop — infeasible for MODIS-scale data (100M+ pixels/day). MELODIES-MONET solved this with a numba-accelerated histogram binning approach (`grid_util.update_data_grid`) that bins swath pixels onto a uniform grid in compiled code. We need to port this approach into DAVINCI-MONET as a new pairing strategy.

## First Use Case: MODIS AOD vs CAM6

- **MODIS L2 AOD**: Terra (MOD04_L2) + Aqua (MYD04_L2), Collection 6.1
  - `~/Data/MODIS/Terra/C61/2019/{355,356,357}/` (~155 granules/day)
  - `~/Data/MODIS/Aqua/C61/2019/{355,356,357}/` (~155 granules/day)
  - Combined: ~310 granules, ~1.1M valid pixels per day
- **CAM6 Base**: `~/Data/CAM6/FCnudged_f09.mam.BaseMar27.2019_2021.001_AODVIS.nc`
- **CAM6 New Dust**: `~/Data/CAM6/FCnudged_f09.mam.newdustMar282025.2019_2021.001_AODVIS.nc`
  - Both: uniform rectilinear grid 192 lat (~0.94°) x 288 lon (1.25°), daily, variable AODVIS
- **Analysis period**: Dec 21-23, 2019 (Australian bushfire event)

Since CAM6 is on a uniform grid, MODIS pixels can be binned directly onto the model grid — no model regridding needed.

## Architecture

### New Files

```
davinci_monet/pairing/grid_binning.py              — numba-accelerated binning functions
davinci_monet/pairing/strategies/swath_grid.py      — SwathGridStrategy class
davinci_monet/tests/test_swath_grid_strategy.py     — unit tests
```

### Modified Files

```
davinci_monet/pairing/strategies/__init__.py              — export SwathGridStrategy
davinci_monet/config/schema.py                            — add intermediate_grid config section
davinci_monet/plots/renderers/spatial/distribution.py     — add 1D lat/lon pcolormesh path
davinci_monet/plots/renderers/spatial/bias.py             — add plot_type param with pcolormesh support
```

## Core Algorithm: grid_binning.py

Port of MELODIES-MONET's `grid_util.py` with three functions:

### 1. bin_swath_to_grid (numba JIT)

```python
@numba.jit(nopython=True)
def bin_swath_to_grid(
    time_edges, lon_edges, lat_edges,
    time_obs, lon_obs, lat_obs, data_obs,
    count_grid, data_grid
):
    """Accumulate swath pixels into (time, lon, lat) grid cells.

    Operates on flat numpy arrays. Each valid (non-NaN) pixel is assigned
    to a grid cell via simple floor division on the bin edges. Running
    sums and counts are accumulated in-place.
    """
    dt = time_edges[1] - time_edges[0]
    dx = lon_edges[1] - lon_edges[0]
    dy = lat_edges[1] - lat_edges[0]
    nt, nx, ny = data_grid.shape
    for i in range(len(data_obs)):
        if not np.isnan(data_obs[i]):
            it = int((time_obs[i] - time_edges[0]) / dt)
            ix = int((lon_obs[i] - lon_edges[0]) / dx)
            iy = int((lat_obs[i] - lat_edges[0]) / dy)
            # clamp to valid range
            it = max(0, min(it, nt - 1))
            ix = max(0, min(ix, nx - 1))
            iy = max(0, min(iy, ny - 1))
            count_grid[it, ix, iy] += 1
            data_grid[it, ix, iy] += data_obs[i]
```

### 2. normalize_grid

```python
def normalize_grid(count_grid, data_grid):
    """Divide accumulated sums by counts; set empty cells to NaN."""
    mask = count_grid > 0
    data_grid[~mask] = np.nan
    data_grid[mask] /= count_grid[mask]
```

### 3. generate_grid_edges

```python
def generate_grid_edges(lat_centers, lon_centers, time_centers):
    """Derive bin edges from grid center coordinates.

    For uniform grids, edges are placed at midpoints between centers,
    with half-spacing extensions at boundaries.
    """
```

This is needed for the `match_model` mode — derive edges from CAM6's lat/lon center arrays.

## SwathGridStrategy

Inherits `BasePairingStrategy`, implements the standard `pair()` interface.

### pair() Flow

```
1. Determine target grid:
   - match_model: derive edges from model lat/lon centers
   - resolution: build uniform grid at specified spacing
   - explicit: use ntime/nlat/nlon from config

2. Initialize accumulation arrays:
   - count_grid = zeros(ntime, nlon, nlat, dtype=int32)
   - data_grid  = zeros(ntime, nlon, nlat, dtype=float32)

3. For each variable to pair:
   - Extract flat arrays from obs: lon[], lat[], data[]
   - Convert obs timestamps to epoch seconds
   - Handle lon convention (shift -180..180 → 0..360 if model uses 0..360)
   - Call bin_swath_to_grid() — numba fast path

4. Normalize: data_grid /= count_grid

5. Wrap in xr.Dataset with proper coords (time, lon, lat)

6. Align model time to obs grid (nearest-neighbor select)

7. Create paired output:
   - obs_aod: binned MODIS on target grid
   - model_aod: CAM6 AODVIS on same grid
   - obs_count: number of MODIS pixels per cell (QA diagnostic)
```

### Longitude Convention Handling

CAM6 uses 0–360, MODIS uses -180–180. Before binning:

```python
lon_obs = np.where(lon_obs < 0, lon_obs + 360, lon_obs)
```

This is done once on the flat array before passing to the numba function.

### Multi-Granule Accumulation

The strategy receives an already-loaded obs `xr.Dataset` from the MODIS reader. For multi-granule data, the reader concatenates granules. The strategy flattens all lat/lon/data and bins in one pass — the numba loop handles any number of pixels efficiently.

If memory is a concern for very large datasets (global, multi-day), the pipeline can iterate over time intervals and call the strategy per interval, accumulating on the same grid arrays. This matches MELODIES-MONET's `for time_interval in an.time_intervals` pattern.

## Config YAML

### Pair Section

```yaml
pairs:
  modis_vs_cam6:
    model: cam6
    obs: terra_modis
    strategy: swath_grid
    intermediate_grid:
      mode: match_model           # bin obs directly onto model grid
      time_resolution: "1D"       # daily bins (match CAM6 output frequency)
    min_obs_count: 1              # require at least 1 obs per grid cell
    variable:
      model_var: AODVIS
      obs_var: AOD_550_Dark_Target_Deep_Blue_Combined
```

### Grid Mode Options

| Mode | Description | Required Fields |
|------|-------------|----------------|
| `match_model` | Derive grid from model lat/lon — no regridding needed | `time_resolution` |
| `resolution` | Uniform grid at specified degree spacing | `resolution`, `time_resolution` |
| `explicit` | Fully specified grid dimensions | `ntime`, `nlat`, `nlon` |

### Full Example Config

```yaml
analysis:
  start_time: "2019-12-21"
  end_time: "2019-12-22"
  output_dir: /Users/fillmore/EarthSystem/DAVINCI-MONET/analyses/modis/output
  log_dir: /Users/fillmore/EarthSystem/DAVINCI-MONET/analyses/modis/logs
  style:
    theme: ncar

model:
  cam6:
    mod_type: cesm_fv
    files: /Users/fillmore/Data/CAM6/FCnudged_f09.mam.BaseMar27.2019_2021.001_AODVIS.nc
    variables:
      AODVIS:
        unit_scale: 1.0

obs:
  terra_modis:
    obs_type: modis_l2_aod
    filename: /Users/fillmore/Data/MODIS/Terra/C61/2019/355/MOD04_L2.*.hdf
    variables:
      AOD_550_Dark_Target_Deep_Blue_Combined:
        minimum: 0.0
        maximum: 10.0
        scale: 0.001

pairs:
  modis_vs_cam6:
    model: cam6
    obs: terra_modis
    strategy: swath_grid
    intermediate_grid:
      mode: match_model
      time_resolution: "1D"
    min_obs_count: 1
    variable:
      model_var: AODVIS
      obs_var: AOD_550_Dark_Target_Deep_Blue_Combined

plots:
  aod_spatial:
    type: spatial_bias
    pairs: [modis_vs_cam6]
    title: "MODIS vs CAM6 AOD"

stats:
  metrics: [N, MB, RMSE, R, NMB, NME]
```

## Output Dataset Structure

The paired output `xr.Dataset`:

```
Dimensions:    (time: ntime, lat: nlat, lon: nlon)
Coordinates:
  * time       (time)  datetime64
  * lat        (lat)   float64
  * lon        (lon)   float64
Data variables:
    obs_aod    (time, lat, lon)  float32  — binned MODIS AOD (NaN where no obs)
    model_aod  (time, lat, lon)  float32  — CAM6 AODVIS on same grid
    obs_count  (time, lat, lon)  int32    — pixel count per cell
```

The `obs_count` variable enables downstream QA filtering (e.g., require >= 3 obs per cell) and is useful for visualization (shows orbital coverage patterns).

## Testing Plan

### Unit Tests (synthetic data)

1. **Binning correctness**: Generate synthetic swath pixels at known locations, verify they land in correct grid cells
2. **Normalization**: Multiple pixels in one cell → verify mean is correct
3. **Empty cells**: Cells with no observations → NaN
4. **Longitude wrapping**: Pixels at -179° correctly map to 181° grid cell
5. **Edge cases**: Pixels exactly on grid edges, pixels outside grid bounds
6. **min_obs_count filtering**: Cells below threshold masked out

### Integration Tests

7. **match_model mode**: Use synthetic model grid, verify obs grid edges match
8. **Full pair() call**: Synthetic swath + synthetic model → paired dataset with correct structure
9. **Variable naming**: Output uses `obs_` and `model_` prefix convention

### Real Data Smoke Test (manual) — COMPLETE

10. **MODIS + CAM6**: `analyses/modis-aod/scripts/smoke_test.py`
    - Terra + Aqua combined, 3 days (Dec 21-23 2019), 2 CAM6 runs (base + new dust)
    - 2x3 panel figures: top row AOD (MODIS/Base/NewDust), bottom row diffs
    - contourf with turbo colormap, CERES-SARB non-uniform levels, rasterized base layer for PDFs
    - Output: PNG (300 DPI) + PDF per day, figure captions and text for paper

## Dependencies

- **numba**: Already in environment.yml (used by MELODIES-MONET). Verify it's present in davinci-monet env.
- **pyhdf**: Already listed in CLAUDE.md as available for HDF4/HDF-EOS MODIS data.

## Relationship to Existing Strategies

| Strategy | Input Geometry | Method | Use Case |
|----------|---------------|--------|----------|
| `PointStrategy` | Point (site) | Nearest grid cell | Surface stations |
| `TrackStrategy` | Track (1D path) | Interpolation along path | Aircraft |
| `ProfileStrategy` | Profile (vertical) | Vertical interpolation | Sondes |
| `SwathStrategy` | Swath (L2) | Per-pixel extraction | Small swaths (kept for testing) |
| **`SwathGridStrategy`** | **Swath (L2)** | **Numba binning → grid** | **Satellite L2 at scale** |
| `GridStrategy` | Grid (L3) | Regrid + align | Gridded satellite/reanalysis |

`SwathGridStrategy` is the recommended strategy for all L2 satellite products. The existing `SwathStrategy` remains available but is not suitable for production-scale data.

## Plotting: Gridded Spatial Maps

The paired output is fully gridded on (time, lat, lon) with 1D coordinate arrays. This requires `pcolormesh` rendering, not scatter. The existing spatial plotters need minor updates.

### Current State

| Renderer | Plot Method | Gridded Support |
|----------|------------|-----------------|
| `spatial_bias` | scatter only | No — flattens all data to points |
| `spatial_distribution` | scatter or pcolormesh | Partial — pcolormesh only triggers for 2D lat/lon (curvilinear grids) |

Both plotters flatten lat/lon and data, then scatter. For gridded data on a regular grid with 1D lat/lon coords, this produces tiny scatter points instead of filled grid cells.

### Required Changes

**`spatial_distribution`** (`plots/renderers/spatial/distribution.py`):
- The `_plot_data()` method checks `lats.ndim == 2` for pcolormesh (line 289). Add a branch for 1D lat/lon (regular grid):
  ```python
  if plot_type == "pcolormesh":
      if lats.ndim == 2:
          # curvilinear grid — existing path
          ax.pcolormesh(lons, lats, data, ...)
      elif lats.ndim == 1 and data.ndim == 2:
          # regular grid — new path
          ax.pcolormesh(lons, lats, data, ...)
  ```
  matplotlib `pcolormesh` natively handles 1D coordinate arrays for regular grids.

**`spatial_bias`** (`plots/renderers/spatial/bias.py`):
- Add `plot_type` parameter (like `spatial_distribution` already has)
- When `plot_type == "pcolormesh"` and data is 2D with 1D lat/lon coords, use `pcolormesh` instead of scatter
- The bias field (model - obs) is already computed as a 2D array on the grid — just needs to be rendered as filled cells

### Plot Types for MODIS-CAM6

The config should produce three map types:

```yaml
plots:
  # 1. Obs-only: binned MODIS AOD
  aod_obs:
    type: spatial_distribution
    pairs: [modis_vs_cam6]
    show_var: obs
    plot_type: pcolormesh
    cmap: "YlOrRd"
    title: "MODIS AOD (binned to model grid)"

  # 2. Model-only: CAM6 AODVIS
  aod_model:
    type: spatial_distribution
    pairs: [modis_vs_cam6]
    show_var: model
    plot_type: pcolormesh
    cmap: "YlOrRd"
    title: "CAM6 AODVIS"

  # 3. Bias: model - obs
  aod_bias:
    type: spatial_bias
    pairs: [modis_vs_cam6]
    plot_type: pcolormesh
    cmap: "RdBu_r"
    title: "AOD Bias (CAM6 - MODIS)"

  # 4. Obs count: pixel coverage diagnostic
  aod_coverage:
    type: spatial_distribution
    pairs: [modis_vs_cam6]
    show_var: obs
    variable: obs_count         # override to show count instead of AOD
    plot_type: pcolormesh
    cmap: "viridis"
    title: "MODIS Pixel Count per Grid Cell"
```

### Modified Files (Plotting)

```
davinci_monet/plots/renderers/spatial/distribution.py  — add 1D lat/lon pcolormesh path
davinci_monet/plots/renderers/spatial/bias.py          — add plot_type param with pcolormesh support
```

These are small, backward-compatible changes — existing scatter behavior is preserved as the default.

## Reference

- MELODIES-MONET implementation: `/Users/fillmore/EarthSystem/MELODIES-MONET/melodies_monet/util/grid_util.py`
- MELODIES-MONET MODIS workflow: `/Users/fillmore/EarthSystem/MELODIES-MONET/examples/process_swath_data/process_modis_l2.py`
- MELODIES-MONET MODIS config: `/Users/fillmore/EarthSystem/MELODIES-MONET/examples/process_swath_data/control_modis_l2.yaml`
- CAM6 base: `~/Data/CAM6/FCnudged_f09.mam.BaseMar27.2019_2021.001_AODVIS.nc`
- CAM6 new dust: `~/Data/CAM6/FCnudged_f09.mam.newdustMar282025.2019_2021.001_AODVIS.nc`
- MODIS Terra: `~/Data/MODIS/Terra/C61/2019/{355,356,357}/` (~155 HDF4 L2 granules/day)
- MODIS Aqua: `~/Data/MODIS/Aqua/C61/2019/{355,356,357}/` (~155 HDF4 L2 granules/day)
- Smoke test script: `analyses/modis-aod/scripts/smoke_test.py`
- Smoke test output: `analyses/modis-aod/output/`

## Implementation Progress

| Component | Status |
|-----------|--------|
| `pairing/grid_binning.py` — numba binning functions | COMPLETE |
| `analyses/modis-aod/scripts/smoke_test.py` — multi-day, multi-model analysis | COMPLETE |
| `pairing/strategies/swath_grid.py` — SwathGridStrategy class | TODO |
| `config/schema.py` — intermediate_grid config | TODO |
| `plots/renderers/spatial/` — pcolormesh support for 1D grids | TODO |
| Unit tests — synthetic data | TODO |
