# LMA Flash Density Map Renderer — Design Document

**Date**: 2026-03-02
**Branch**: develop
**Dev case**: DC3 May 29, 2012 (Oklahoma supercell)

## Purpose

Build an `obs_lma_density` plot renderer that produces cartopy maps of hourly
Lightning Mapping Array flash extent density, with an optional flight track
overlay. Two modes:

1. **Standalone** — pcolormesh of hourly accumulated flash density on a map
2. **With flight tracks** — same base map with DC-8 and/or GV tracks overlaid
   as solid colored lines

## Data Inputs

- **LMA grids**: OKLMA NetCDF files with dims `(ntimes, lon, lat)` and vars
  `flash_extent`, `longitude`, `latitude`, `time`. 12 five-minute time steps
  per hourly file. Renderer sums across time steps to produce hourly totals.
- **Flight tracks** (optional): ICARTT-loaded xarray Datasets with `latitude`,
  `longitude`, `time` coordinates. Filtered to the same hour window.

## Architecture

### New file

`davinci_monet/plots/renderers/obs/obs_lma_density.py`

### Extends

`BaseSpatialPlotter` from `davinci_monet/plots/renderers/spatial/base.py` for
cartopy map setup (projection, extent, features, gridlines).

### Registration

```python
@register_obs_plotter("obs_lma_density")
class ObsLMADensityPlotter(BaseSpatialPlotter):
    ...
```

### Key methods

- `plot(obs_dataset, config, obs_datasets=None)` — main entry point
  - `obs_dataset`: LMA grid data (xr.Dataset)
  - `config`: plot config dict from YAML
  - `obs_datasets`: dict of all loaded obs datasets (for flight tracks)
- `_aggregate_hourly(ds)` → list of `(hour_label, summed_2d_array)` tuples
- `_render_density(ax, lon, lat, data, cmap, vmin, vmax)` → pcolormesh artist
- `_overlay_tracks(ax, obs_datasets, flight_tracks_config, time_start, time_end)` → line artists + legend

### Config contract (from YAML)

```yaml
lma_density:
  type: obs_lma_density
  obs: oklma                    # references obs section
  variable: flash_extent        # which LMA grid variable
  time_agg: hourly              # aggregation window
  cmap: YlOrRd                  # matplotlib colormap
  title: "..."                  # plot title template
  flight_tracks:                # optional
    dc8: dc8                    # label: obs_key
    gv: gv
  map:
    projection: LambertConformal
    features: [states, counties]
```

### Output

One PNG per hour with nonzero flash activity. Filename pattern:
`lma_flash_extent_YYYYMMDD_HHMM.png` (standalone) or
`lma_flash_extent_tracks_YYYYMMDD_HHMM.png` (with overlay).

## Styling

- Colormap: `YlOrRd` (sequential warm — white→yellow→red for flash density)
- Background: light gray land, white ocean (from BaseSpatialPlotter)
- Map features: state borders (dark gray), county lines (light gray, thin)
- Flight track colors: first two entries from `NCAR_PALETTE` (distinct from
  warm density colormap)
- Track line width: 1.5pt, with aircraft name in legend
- Font: Poppins / NCAR style (applied via `style.theme: ncar` in config)
- Colorbar: horizontal below map, label "Flash extent density (flashes per grid cell)"

## Map Projection

LambertConformal centered on OKLMA domain (~98.5°W, 35°N). Auto-fit extent
from LMA grid coordinates with a small padding margin.

## Testing

- Unit test with synthetic LMA-like grid data (small 10×10 grid, 12 time steps)
- Test hourly aggregation produces correct sums
- Test that flight track overlay doesn't error with empty tracks
- Test output file naming
- Integration: run `dc3-may29-gemini.yaml` and visually verify

## Decisions & Rationale

- **New renderer vs. extending spatial_distribution**: The spatial_distribution
  renderer handles single-time snapshots and model/obs side-by-side. LMA
  density maps need hourly aggregation across time steps and flight track
  overlays — different enough to warrant a dedicated renderer rather than
  overloading the existing one.
- **Hourly aggregation in the renderer, not the reader**: The LMA reader
  returns raw 5-minute data. Aggregation belongs in the plotter because
  different analyses may want different windows. The reader stays general.
- **One PNG per hour, not multi-panel**: Keeps file sizes small, each hour
  standalone for presentations. A multi-panel summary can be added later.
- **Flight track colors from NCAR_PALETTE**: Warm colormaps (YlOrRd) use
  red/orange/yellow — the palette blues/greens contrast well for tracks.
