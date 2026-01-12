# DAVINCI-MONET Architecture

This document describes the internal architecture of DAVINCI-MONET, focusing on the pipeline execution system and the unified pairing engine.

## System Overview

DAVINCI-MONET is a modular toolkit for evaluating atmospheric chemistry models against observations. The architecture follows these design principles:

- **Plugin-based extensibility** via Protocol interfaces and registries
- **Geometry-aware pairing** using strategy pattern for different observation types
- **Pipeline-based execution** with composable stages
- **xarray-native data model** throughout the system

```
                           DAVINCI-MONET System

    CLI ──────▶ Config ──────▶ Pipeline ──────▶ Output
   (Typer)      (YAML)          Runner        (Plots,
              (Pydantic)                       Stats)
                                  │
                                  ▼
                          Pipeline Stages
                          ---------------
                          load_models
                              ▼
                          load_observations
                              ▼
                          pairing
                              ▼
                          statistics
                              ▼
                          plotting
                              ▼
                          save_results
                                  │
                                  ▼
                          Pairing Engine
                          --------------
                    Point - Track - Profile - Swath - Grid
```

## Pipeline Architecture

The pipeline is the central execution mechanism. It orchestrates data loading, pairing, analysis, and output generation through a sequence of stages.

### Pipeline Components

```
                             PipelineRunner
                             --------------

Responsibilities:
  - Execute stages in sequence
  - Manage PipelineContext (shared state)
  - Handle errors and recovery
  - Report progress with animated display
  - Generate Markdown execution logs

                           PipelineContext
                           ---------------
  config: dict          # YAML configuration
  models: dict          # Loaded ModelData objects
  observations: dict    # Loaded ObservationData objects
  paired: dict          # PairedData from pairing stage
  results: dict         # StageResult from each completed stage
  metadata: dict        # Runtime metadata (paths, timing)
```

### Stage Execution Flow

Each stage receives the `PipelineContext`, performs its work, and returns a `StageResult`. The context accumulates data as it flows through stages.

```
                           PipelineContext
                                 │
         ┌───────────────────────┴───────────────────────┐
         ▼                                               ▼
    load_models                                  load_observations
         │                                               │
         └──────────────┐           ┌────────────────────┘
                        ▼           ▼
                        pairing
                           │
                           ▼
                      statistics
                           │
                           ▼
                       plotting
                           │
                           ▼
                     save_results

  Data flow:
    load_models ──▶ context.models
    load_observations ──▶ context.observations
    pairing ──▶ context.paired (uses models + observations)
```

### Standard Pipeline Stages

| Stage | Input | Output | Description |
|-------|-------|--------|-------------|
| `load_models` | config.model | context.models | Load model files, apply unit conversions |
| `load_observations` | config.obs | context.observations | Load observation files, filter by time/space |
| `pairing` | models + observations | context.paired | Match model to observations by geometry |
| `statistics` | context.paired | StageResult.data | Compute N, MB, RMSE, R, NMB, NME, IOA |
| `plotting` | context.paired | PNG/PDF files | Generate scatter, timeseries, spatial plots |
| `save_results` | statistics | CSV files | Write statistics tables |

### Stage Protocol

All stages implement the `Stage` protocol:

```python
@runtime_checkable
class Stage(Protocol):
    @property
    def name(self) -> str:
        """Stage name."""
        ...

    def execute(self, context: PipelineContext) -> StageResult:
        """Execute the stage and return result."""
        ...

    def validate(self, context: PipelineContext) -> bool:
        """Validate prerequisites before execution."""
        ...
```

### StageResult Structure

```python
@dataclass
class StageResult:
    stage_name: str              # Name of the stage
    status: StageStatus          # PENDING | RUNNING | COMPLETED | FAILED | SKIPPED
    data: Any                    # Output data (varies by stage)
    metadata: dict[str, Any]     # Execution metadata
    error: str | None            # Error message if failed
    error_type: str | None       # Exception class name
    traceback_str: str | None    # Full traceback for debugging
    duration_seconds: float      # Execution time
```

---

## Pairing Engine Architecture

The pairing engine is the core component that matches model output with observations. It uses a **strategy pattern** to handle different observation geometries uniformly.

### Design Philosophy

Traditional approaches pair data based on **data source** (e.g., different code for AirNow vs AERONET). DAVINCI-MONET instead pairs based on **data geometry**:

- The same `PointStrategy` handles both AirNow (surface) and AERONET (surface)
- The same `TrackStrategy` handles any aircraft data (DC-8, G-III, etc.)
- Adding new data sources requires no new pairing code if geometry is supported

### Pairing Engine Components

```
                            PairingEngine
                            -------------

  pair(model, obs, obs_vars, model_vars, config)
      │
      ├─1─▶ _detect_geometry(obs) ──▶ DataGeometry enum
      │
      ├─2─▶ _check_temporal_overlap(model, obs)
      │
      ├─3─▶ get_strategy(geometry) ──▶ PairingStrategy
      │
      └─4─▶ strategy.pair(model, obs, ...) ──▶ PairedData


                        Strategy Registry
                        -----------------
  POINT   ──▶ PointStrategy
  TRACK   ──▶ TrackStrategy
  PROFILE ──▶ ProfileStrategy
  SWATH   ──▶ SwathStrategy
  GRID    ──▶ GridStrategy
```

### Data Geometry Types

The `DataGeometry` enum defines five observation geometries:

```
                         Data Geometry Types
                         -------------------

POINT                              TRACK
-----                              -----
dims: (time, site)                 dims: (time,)
coords: lat[site], lon[site]       coords: lat[time], lon[time], alt[time]

Examples:                          Examples:
  - AirNow surface stations          - DC-8 aircraft
  - AERONET ground sites             - Ship tracks
  - Pandora spectrometers            - Mobile platforms

      ●        ●                         ●────●────●────●
    site1    site2                         flight path


PROFILE                            SWATH
-------                            -----
dims: (time, level)                dims: (time, scanline, pixel)
coords: lat[time], lon[time],      coords: lat[scanline,pixel],
        z[level]                           lon[scanline,pixel]

Examples:                          Examples:
  - Ozonesondes                      - TROPOMI L2
  - Lidar profiles                   - MODIS L2
  - Radiosondes                      - GEMS L2

     z                               ● ● ● ●  scanline 1
     │  ●                            ● ● ● ●  scanline 2
     │  ●                              pixels ──▶
     │  ●
     │  ●
     └────


GRID
----
dims: (time, lat, lon)

Examples:
  - Satellite L3 products
  - Reanalysis data
  - Regridded model output

     lat
      │  ● ● ● ●
      │  ● ● ● ●
      │  ● ● ● ●
      └─────────▶ lon
```

### Geometry Detection

The engine auto-detects geometry from dataset structure:

```python
def _detect_geometry(self, obs: xr.Dataset) -> DataGeometry:
    # 1. Check explicit attribute
    if "geometry" in obs.attrs:
        return DataGeometry[obs.attrs["geometry"].upper()]

    # 2. Infer from dimensions
    dims = set(obs.dims)

    if "lat" in dims and "lon" in dims:
        return DataGeometry.GRID

    if "scanline" in dims or "pixel" in dims:
        return DataGeometry.SWATH

    if "time" in dims and "level" in dims:
        return DataGeometry.PROFILE

    if "site" in dims or "station" in dims:
        return DataGeometry.POINT

    if "time" in dims and "lat" in obs.coords:
        return DataGeometry.TRACK
```

### Strategy Implementation

Each strategy inherits from `BasePairingStrategy` and implements the `pair()` method:

```
                       BasePairingStrategy
                       -------------------

Common Methods:
  _get_model_coords()      # Extract lat/lon from model
  _get_obs_coords()        # Extract lat/lon from observations
  _find_nearest_indices()  # Spatial matching (1D or 2D grids)
  _haversine_distance()    # Great-circle distance calculation
  _interpolate_time()      # Temporal interpolation
  _interpolate_vertical()  # Vertical interpolation
  _extract_surface()       # Get surface level from 3D model

                               │
       ┌───────────────────────┼───────────────────────┐
       ▼                       ▼                       ▼

PointStrategy           TrackStrategy          ProfileStrategy
-------------           -------------          ---------------
  Extract surface         Interpolate            Interpolate
    level                   along track            vertical levels
  Match by site           4D matching            Match profiles
                            (x,y,z,t)

       ┌───────────────────────┴───────────────────────┐
       ▼                                               ▼

SwathStrategy                                   GridStrategy
-------------                                   ------------
  Handle 2D footprints                            Regrid or interpolate
  Apply averaging kernels                         Direct grid matching
```

### Pairing Configuration

The `PairingConfig` dataclass controls pairing behavior:

```python
@dataclass
class PairingConfig:
    radius_of_influence: float = 12000.0   # Spatial search radius (meters)
    time_tolerance: TimeDelta | None = None # Max time difference
    vertical_method: str = "nearest"        # 'nearest', 'linear', 'log'
    horizontal_method: str = "nearest"      # 'nearest', 'bilinear'
    apply_averaging_kernel: bool = False    # For satellite retrievals
    require_overlap: bool = True            # Require temporal overlap
```

### Pairing Data Flow (Point Example)

```
    Model Data                    Observation Data
  (time,z,lat,lon)                  (time, site)
         │                               │
         └───────────────┬───────────────┘
                         ▼
               PointStrategy.pair()
               --------------------

  1. Extract surface level (z=0)
     model[time, lat, lon]

  2. Find nearest grid cell for each site
     lat_idx, lon_idx = find_nearest()

  3. Interpolate model to obs times
     model.interp(time=obs_times)

  4. Extract model values at sites
     model_at_sites[time, site]

  5. Combine with observations
     paired_ds[obs_var, model_var]

                         │
                         ▼
                    PairedData
                    ----------
  data: xr.Dataset
    dims: (time, site)
    vars: obs_pm25, model_pm25
    coords: lat, lon, time

  model_label: "cesm_asiaq"
  obs_label: "airnow"
  geometry: DataGeometry.POINT
  pairing_info: {radius, method, ...}
```

---

## Data Flow Diagram

Complete data flow from YAML config to output files:

```
asia-aq.yaml
------------
model:
  cesm_asiaq:
    files: *.nc
    variables: ...

obs:
  airnow:
    filename: ...
  pandora:
    filename: ...
  dc8:
    filename: ...

pairs:
  cesm_airnow_pm25:
    model: cesm_asiaq
    obs: airnow
    variable: ...

plots:
  pm25_scatter:
    type: scatter
    pairs: [...]

stats:
  metrics: [...]

                         │
    ┌────────────────────┴────────────────────┐
    ▼                                         ▼
load_models                           load_observations
    │                                         │
    ▼                                         ▼
context.models                        context.observations
  cesm_asiaq: ds                        airnow: ds
                                        pandora: ds
                                        dc8: ds
    │                                         │
    └────────────────────┬────────────────────┘
                         ▼
                      pairing
              (PointStrategy, TrackStrategy)
                         │
                         ▼
                  context.paired
                    cesm_airnow: pd
                    cesm_pandora: pd
                    cesm_dc8: pd
                         │
         ┌───────────────┴───────────────┐
         ▼                               ▼
    statistics                       plotting
         │                               │
         ▼                               ▼
  statistics_summary.csv          *.png, *.pdf
                                  (scatter, timeseries,
                                   spatial, 3d tracks)
```

---

## Module Structure

```
davinci_monet/
├── core/                    # Foundation layer
│   ├── protocols.py         # Interface definitions (ModelReader, etc.)
│   ├── registry.py          # Plugin registration system
│   ├── exceptions.py        # Custom exception hierarchy
│   ├── types.py             # Type aliases and helpers
│   └── base.py              # Base classes (PairedData, etc.)
│
├── config/                  # Configuration handling
│   ├── schemas.py           # Pydantic models for YAML validation
│   └── loader.py            # YAML parsing with env var expansion
│
├── models/                  # Model readers
│   ├── base.py              # BaseModelReader
│   ├── cmaq.py              # CMAQ reader
│   ├── wrfchem.py           # WRF-Chem reader
│   ├── cesm.py              # CESM/CAM-chem reader
│   ├── ufs.py               # UFS reader
│   └── generic.py           # Generic NetCDF reader
│
├── observations/            # Observation readers
│   ├── base.py              # BaseObservationReader
│   ├── surface/             # Surface observations
│   │   ├── airnow.py        # AirNow (US Embassy monitors)
│   │   ├── aeronet.py       # AERONET AOD
│   │   ├── aqs.py           # EPA AQS
│   │   ├── openaq.py        # OpenAQ
│   │   └── pandora.py       # Pandora NO2 columns
│   ├── aircraft/            # Aircraft observations
│   │   └── icartt.py        # ICARTT format reader
│   └── satellite/           # Satellite observations
│       ├── tropomi.py       # TROPOMI L2
│       └── goes.py          # GOES-16/17 AOD
│
├── pairing/                 # Pairing engine
│   ├── engine.py            # PairingEngine orchestrator
│   └── strategies/          # Geometry-specific strategies
│       ├── base.py          # BasePairingStrategy
│       ├── point.py         # Fixed locations
│       ├── track.py         # Aircraft/mobile tracks
│       ├── profile.py       # Vertical profiles
│       ├── swath.py         # Satellite swaths
│       └── grid.py          # Gridded data
│
├── pipeline/                # Execution engine
│   ├── runner.py            # PipelineRunner + ProgressFormatter
│   ├── stages.py            # Stage protocol + implementations
│   └── parallel.py          # Parallel execution utilities
│
├── plots/                   # Plotting system
│   ├── base.py              # BasePlotter + label formatting
│   ├── registry.py          # Plot type registration
│   └── renderers/           # Plot implementations
│       ├── scatter.py       # Scatter plots
│       ├── timeseries.py    # Time series
│       ├── spatial_bias.py  # Spatial bias maps
│       ├── site_timeseries.py    # Multi-panel site plots
│       ├── flight_timeseries.py  # Multi-panel flight plots
│       └── track_map_3d.py       # 3D flight track visualization
│
├── stats/                   # Statistics
│   ├── calculator.py        # MetricsCalculator
│   ├── metrics.py           # Individual metric implementations
│   └── output.py            # CSV/table formatters
│
├── io/                      # File I/O
│   ├── readers.py           # Generic readers
│   └── writers.py           # NetCDF writers with compression
│
└── cli/                     # Command-line interface
    ├── app.py               # Main Typer application
    └── commands/            # Subcommands
        ├── run.py           # Pipeline execution
        ├── validate.py      # Config validation
        └── get_data.py      # Data download
```

---

## Extending the System

### Adding a New Observation Source

1. Create reader in `observations/` that returns xr.Dataset with geometry metadata
2. Register with observation registry
3. No changes needed to pairing engine (auto-detected by geometry)

### Adding a New Plot Type

1. Create renderer in `plots/renderers/` inheriting from `BasePlotter`
2. Register with `@plot_registry.register("type_name")`
3. Use in YAML config with `type: type_name`

### Adding a New Pairing Strategy

1. Create strategy in `pairing/strategies/` inheriting from `BasePairingStrategy`
2. Implement `geometry` property and `pair()` method
3. Register with `engine.register_strategy(MyStrategy())`

---

## Performance Considerations

- **Lazy loading**: xarray's lazy evaluation defers computation until needed
- **Chunked processing**: Large files processed in chunks via dask (when available)
- **Spatial indexing**: 1D grids use binary search; 2D grids use haversine distance
- **Memory efficiency**: Paired data only includes matched points, not full grids

---

## See Also

- [CLAUDE.md](CLAUDE.md) - Development guide and quick start
- [PLAN.md](PLAN.md) - Implementation plan and phase details
- [Wiki](https://github.com/NCAR/DAVINCI-MONET/wiki) - User documentation
