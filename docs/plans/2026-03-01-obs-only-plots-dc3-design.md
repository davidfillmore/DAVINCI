# Design: Obs-Only Plot Types + DC3 Data Integration

**Date**: 2026-03-01
**Branch**: develop
**Status**: Approved

## Context

DAVINCI's plotting system is entirely paired-data-centric — all 14 plot types require both `obs_var` and `model_var`. This blocks observation-first workflows like DC3, where we want to visualize aircraft data before model runs are available.

This design adds obs-only plotting infrastructure and integrates DC3 field campaign aircraft data.

## Part A: Obs-Only Plot Infrastructure

### New ObsPlotter Base Class

File: `davinci_monet/plots/obs_base.py`

A parallel base class alongside the existing `BasePlotter`:

```python
class ObsPlotter(ABC):
    """Base class for observation-only plotters."""

    @abstractmethod
    def plot(
        self,
        obs_data: xr.Dataset,
        variable: str,
        ax: Axes | None = None,
        **kwargs: Any,
    ) -> Figure:
        """Plot a single observation variable."""
```

Key differences from `BasePlotter`:
- Takes raw `obs_data` (not paired data) with a single `variable` name
- No `model_var` parameter
- Reuses helper methods (create_figure, save, apply_text_style) via composition

### 4 New Obs-Only Renderers

Location: `davinci_monet/plots/renderers/obs/`

| Renderer | Registry Name | Description |
|----------|--------------|-------------|
| `flight_track_map.py` | `obs_flight_track` | 2D Cartopy map with flight path colored by variable value. Auto-zooms to domain, optional altitude shading. |
| `vertical_profile.py` | `obs_vertical_profile` | Altitude vs. concentration. Scatter or binned-mean with std envelope. Pressure axis option. |
| `obs_timeseries.py` | `obs_timeseries` | Variable vs. time along flight. Multi-flight overlay, altitude on secondary y-axis. |
| `obs_histogram.py` | `obs_histogram` | Distribution histogram with summary stats annotation (N, mean, median, std, p10/p90). |

All registered via `@register_plotter("obs_...")` using the existing plotter registry.

### New ObsPlottingStage

File: `davinci_monet/pipeline/stages.py` (new class)

- **Auto-detection**: Activated when YAML config has no `model` section
- **Data source**: Operates on `context.observations` instead of `context.paired`
- **Plot config**: New obs-only plot schema referencing `obs` key + single `variable`

### Obs-Only Statistics

Simple descriptive stats per variable (no skill scores — those require model comparison):
- N, mean, median, std, min, max
- Percentiles: p10, p25, p75, p90

### Pipeline Auto-Detection

In `pipeline/runner.py`, when no `model` section exists in config:
- Skip `load_models` and `pairing` stages
- Run `load_observations` → `obs_plotting` → `obs_statistics` → `save_results`

## Part B: DC3 Data Acquisition

### Data Source

NASA ASDC dataset `DC3_Merge_Data_1` — pre-generated ICARTT merge files containing all instruments merged per aircraft per flight.

- DC-8: `dc3-mrg10-dc8_merge_*.ict` (10-second resolution)
- GV: `dc3-mrg10-gv_merge_*.ict` (10-second resolution)
- Falcon: `dc3-mrg10-falcon_merge_*.ict` (10-second resolution)

### Download Script

File: `analyses/dc3/scripts/download_dc3_aircraft.py`

Uses `earthaccess` library:
1. `earthaccess.login()` — authenticates via `.netrc` or interactive
2. Searches for `DC3_Merge_Data_1` granules
3. Downloads to `~/Data/DC3/aircraft/merge/`

### Directory Layout

```
~/Data/DC3/
└── aircraft/
    └── merge/
        ├── dc3-mrg10-dc8_merge_*.ict
        ├── dc3-mrg10-gv_merge_*.ict
        └── dc3-mrg10-falcon_merge_*.ict
```

### DC3 Variable Mapping

Applied via YAML config `variables.source_name`, not code changes:

| Standard Name | GV Instrument | DC-8 Instrument | Falcon Instrument |
|--------------|---------------|-----------------|-------------------|
| NO | NO_NOxyO3 | NO_ESRL | NO_DLR |
| NO2 | NO2_NOxyO3 | NO2_TDLIF | NO2_DLR |
| O3 | O3_NOxyO3 | O3_ESRL | O3_DLR |
| CO | CO_ACOMCO | CO_DACOM | CO_DLR |
| NOy | NOy_NOxyO3 | NOy_ESRL | NOy_DLR |

### Analysis Directory

```
analyses/dc3/
├── configs/
│   ├── dc3-obs-dc8.yaml
│   ├── dc3-obs-gv.yaml
│   ├── dc3-obs-falcon.yaml
│   └── dc3-obs-all-aircraft.yaml
├── scripts/
│   ├── download_dc3_aircraft.py
│   └── run_obs_analysis.py
├── data/                         # Symlink to ~/Data/DC3/
├── output/
└── logs/
```

### Obs-Only YAML Config Pattern

```yaml
analysis:
  start_time: "2012-05-18"
  end_time: "2012-06-30"
  output_dir: ${DC3_ANALYSIS}/output
  style:
    theme: ncar

obs:
  dc8:
    obs_type: icartt
    filename: ${DC3_DATA}/aircraft/merge/dc3-mrg10-dc8_merge_*.ict
    variables:
      NO:  { source_name: NO_ESRL }
      NO2: { source_name: NO2_TDLIF }
      O3:  { source_name: O3_ESRL }
      CO:  { source_name: CO_DACOM }

# No model section → triggers obs-only pipeline mode

plots:
  dc8_track_o3:
    type: obs_flight_track
    obs: dc8
    variable: O3
    title: "DC-8 Flight Tracks colored by O3"

  dc8_profile_no:
    type: obs_vertical_profile
    obs: dc8
    variable: NO
    title: "DC-8 NO Vertical Profile"

  dc8_timeseries_co:
    type: obs_timeseries
    obs: dc8
    variable: CO
    title: "DC-8 CO Time Series"

  dc8_hist_o3:
    type: obs_histogram
    obs: dc8
    variable: O3
    title: "DC-8 O3 Distribution"

stats:
  metrics: [N, mean, median, std, min, max, p10, p25, p75, p90]
```

## Decisions & Rationale

1. **Parallel ObsPlotter vs. modifying BasePlotter**: Chose parallel hierarchy to avoid touching 14 existing renderers and risking regressions in paired analysis.

2. **Registry reuse**: Obs-only plotters use the same `plotter_registry` with `obs_` prefix convention. This keeps one unified registry without special-casing.

3. **Variable mapping in config, not code**: DC3 instrument suffixes are handled via YAML `source_name` field. This avoids campaign-specific code in the reader and works for any future campaign.

4. **earthaccess for download**: Standard NASA data access library, handles Earthdata Login via `.netrc`. Avoids custom auth code.

5. **10-second merge files**: Best balance of temporal resolution vs. file size. 1-second data available if needed later.

## What This Does NOT Include

- Lightning (LMA), radiosonde (ARM), or radar data — deferred to Phase B+
- MPAS model reader — deferred until model runs available
- Interactive/web-based plots — static matplotlib/cartopy only
- Modification of existing paired plotters
