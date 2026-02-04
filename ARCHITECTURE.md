# DAVINCI-MONET Architecture

This document describes the internal architecture of DAVINCI-MONET, focusing on the pipeline execution system and the unified pairing engine.

## System Overview

DAVINCI-MONET is a modular toolkit for evaluating atmospheric chemistry models against observations. The architecture follows these design principles:

- **Plugin-based extensibility** via Protocol interfaces and registries
- **Geometry-aware pairing** using strategy pattern for different observation types
- **Pipeline-based execution** with composable stages
- **xarray-native data model** throughout the system

*The system uses **Pydantic** for configuration validation. Pydantic is a Python library that uses type annotations to validate data at runtime. When you load a YAML config file, Pydantic automatically checks that all required fields are present, values have the correct types (strings, numbers, lists), and constraints are satisfied (e.g., paths exist, values are in valid ranges). This catches configuration errors early with clear error messages, rather than failing mysteriously deep in the pipeline.*

*All data flows through the system as **xarray Datasets**. xarray extends NumPy arrays with labeled dimensions and coordinates, making it natural to work with geospatial data that has time, latitude, longitude, and vertical dimensions. Operations like slicing by time range, interpolating to new coordinates, and aligning datasets with different grids become simple one-liners. xarray also supports lazy evaluation via Dask, enabling processing of datasets larger than memory.*

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

*A **context object** is a common Python pattern for passing shared state between components. Rather than using global variables or passing many individual parameters, a context bundles related data into a single object that flows through the system. Each stage can read from and write to the context, allowing data to accumulate as it moves through the pipeline. This pattern promotes loose coupling—stages don't need to know about each other, only about the context they share.*

```
                             PipelineRunner
                             --------------

Responsibilities:
  - Execute stages in sequence
  - Manage PipelineContext (shared state)
  - Handle errors and recovery
  - Report progress with animated pulsing display
  - Generate Markdown execution logs with timing tables

Progress Display:
  - Animated "DAVINCI-MONET" text with left-to-right color sweep
  - Elapsed time counter during stage execution
  - Nested progress: stage › substep › item
  - Summary of loaded models/obs/pairs after each stage

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
| `load_models` | config.model | context.models | Load model files (glob expansion), apply unit conversions |
| `load_observations` | config.obs | context.observations | Load obs files with **time filtering** (file-level + data-level) |
| `pairing` | models + observations | context.paired | Match model to obs by geometry (Dask optional, configurable) |
| `statistics` | context.paired | StageResult.data | Compute N, MB, RMSE, R, NMB, NME, IOA |
| `plotting` | context.paired | PNG/PDF files | Generate scatter, timeseries, spatial, 3D track plots |
| `save_results` | statistics | CSV files | Write statistics tables |

**Key optimizations** (see Performance Optimizations section):
- `load_observations`: Time filtering reduces 5-month files to analysis period (1,630x faster)
- `pairing`: Dask-backed pairs default to serial execution; concurrency and Dask threads are configurable based on CPU/RAM

### Stage Protocol

*A **Protocol** in Python (from the `typing` module) defines an interface—a contract specifying what methods and properties an object must have. Unlike abstract base classes, protocols use "structural subtyping" (duck typing): any class with the required methods automatically satisfies the protocol, without explicit inheritance. This enables loose coupling—code can depend on the protocol interface rather than specific implementations.*

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

*A **dataclass** (from Python's `dataclasses` module) is a decorator that automatically generates boilerplate code for classes that primarily hold data. It creates `__init__`, `__repr__`, and `__eq__` methods based on the class attributes you define. This reduces code duplication and makes data structures self-documenting—the class definition clearly shows what fields exist and their types.*

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

*The **strategy pattern** is a design pattern that defines a family of interchangeable algorithms. Instead of using conditional logic (if/elif chains) to select behavior, the code delegates to a strategy object that encapsulates the algorithm. This makes it easy to add new strategies without modifying existing code—you simply register a new strategy class. The pairing engine uses this pattern to handle different observation geometries: each geometry type (point, track, profile, swath, grid) has its own strategy that knows how to match that geometry with model data.*

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

### Interpolation Methods

The pairing engine uses several interpolation techniques to match model grid points with observation locations:

**Spatial Interpolation (Horizontal)**

*Interpolation estimates values at locations between known data points. For spatial matching, we need to find model grid cells that correspond to observation locations.*

- **Nearest-neighbor**: Selects the closest model grid cell to each observation point. Fast and preserves original model values, but can introduce discontinuities at cell boundaries. Uses the haversine formula for great-circle distance on the sphere:

  `d = 2r × arcsin(√(sin²(Δφ/2) + cos(φ₁)cos(φ₂)sin²(Δλ/2)))`

  where φ is latitude, λ is longitude, and r is Earth's radius (~6371 km).

- **Bilinear**: Weighted average of the four surrounding grid cells. Produces smoother fields but can blur sharp gradients. The weight for each corner is proportional to the area of the opposite rectangle formed by the target point.

**Temporal Interpolation**

- **Nearest-neighbor**: Selects the model time step closest to each observation time. Used when model output has coarse temporal resolution (e.g., hourly) relative to observation frequency.

- **Linear**: Linearly interpolates between bracketing time steps. Appropriate for smoothly varying fields but may miss rapid changes. Implemented via xarray's `interp()` method.

**Vertical Interpolation**

*Atmospheric models use various vertical coordinate systems (pressure levels, sigma coordinates, hybrid levels) that rarely match observation altitudes exactly.*

- **Nearest-level**: Selects the model level closest to the observation altitude. Simple but may have large errors in regions with strong vertical gradients.

- **Linear-in-pressure**: Interpolates linearly in pressure coordinates. Appropriate for most atmospheric variables since many quantities vary approximately linearly with log-pressure.

- **Log-pressure**: Interpolates linearly in log(pressure). Better for quantities that vary exponentially with altitude, such as density or trace gas concentrations in the free troposphere.

**Radius of Influence**

The `radius_of_influence` parameter (default: 12 km) defines the maximum distance for spatial matching. Observations with no model grid cell within this radius are excluded from pairing. This prevents spurious matches when observation networks extend beyond the model domain.

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

  1. Extract surface level (auto-detected: lev=-1 for CESM, z=0 for others)
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
│   │   ├── aeronet.py       # AERONET AOD (CSV + NetCDF L1.5 format)
│   │   ├── aqs.py           # EPA AQS
│   │   ├── openaq.py        # OpenAQ
│   │   └── pandora.py       # Pandora NO2 columns
│   ├── aircraft/            # Aircraft observations
│   │   └── icartt.py        # ICARTT format reader (NASA merge files)
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

### Observation Reader: AERONET Example

The AERONET reader demonstrates handling multiple input formats:

```python
# aeronet.py - _standardize_dataset()
#
# Input formats:
#   - CSV via monetio API: (time, siteid) → converted to (time, site)
#   - NetCDF L1.5: (time, y=1, x=508) where y is dummy dimension
#
# Standardization:
#   1. Squeeze dummy y dimension: ds.squeeze("y", drop=True)
#   2. Rename x → site: ds.rename({"x": "site"})
#   3. Rename coords: latitude → lat, longitude → lon
#   4. Set geometry attribute: ds.attrs["geometry"] = "point"
#
# Result: (time, site) with lat[site], lon[site] coords
```

The `LoadObservationsStage` detects AERONET files by label or filename pattern and routes to the specialized reader for proper dimension handling.

### Adding a New Plot Type

1. Create renderer in `plots/renderers/` inheriting from `BasePlotter`
2. Register with `@plot_registry.register("type_name")`
3. Use in YAML config with `type: type_name`

### Plot Figure Sizes

Each plotter has an optimized default figure size based on its content type. These can be overridden via YAML config.

| Plot Type | Default Size | Ratio | Notes |
|-----------|-------------|-------|-------|
| `timeseries` | (14, 6) | 2.3:1 | Wide for temporal data |
| `flight_timeseries` | (14, 6) | 2.3:1 | Wide for temporal data |
| `site_timeseries` | (14, 6) | 2.3:1 | Wide for temporal data |
| `diurnal` | (14, 6) | 2.3:1 | Wide for temporal data |
| `track_map_3d` | (12, 10) | 1.2:1 | Near-square for 3D viewing |
| `scatter` | (10, 10) | 1:1 | Square for x vs y |
| `taylor` | (10, 10) | 1:1 | Square for polar diagram |
| `curtain` | (14, 8) | 1.75:1 | Wide for geographic extent |
| `spatial_bias` | (14, 8) | 1.75:1 | Wide for geographic extent |
| `spatial_distribution` | (14, 8) | 1.75:1 | Wide for geographic extent |
| `spatial_overlay` | (14, 8) | 1.75:1 | Wide for geographic extent |
| `boxplot` | (12, 8) | 1.5:1 | Balanced |
| `scorecard` | (12, 8) | 1.5:1 | Balanced |

Override in YAML:
```yaml
plots:
  my_plot:
    type: scatter
    figsize: [12, 12]  # Custom size
```

### Adding a New Pairing Strategy

1. Create strategy in `pairing/strategies/` inheriting from `BasePairingStrategy`
2. Implement `geometry` property and `pair()` method
3. Register with `engine.register_strategy(MyStrategy())`

---

## Performance Optimizations

### Time Filtering at Observation Load (1,630x speedup)

Large observation files (e.g., 5-month AERONET NetCDF) are filtered at load time rather than loading everything into memory:

```
LoadObservationsStage
---------------------
1. File-level filtering:
   - Extract YYYYMMDD from filenames (ICARTT convention)
   - Skip files outside analysis period

2. Data-level filtering:
   - After xr.open_dataset(), apply xr.sel(time=slice(start, end))
   - O(1) for sorted time coordinates

Impact: load_observations 163s → 0.1s for 3-day analysis from 5-month file
```

### Dask Pairing Concurrency (Configurable)

Dask-backed model datasets require `.compute()` during pairing. Because each pair can trigger its own compute (and file I/O), the pipeline runs Dask-backed pairs in a dedicated phase and defaults to serial execution for safety.

Controls in `pairing` config:
- `dask_pair_workers`: number of Dask-backed pairs to run concurrently (default: 1).
- `dask_num_workers`: threads for the Dask scheduler inside each pair. If unset, derived from CPU count and capped by RAM (<=16 GB → 4, <=32 GB → 6, else up to 32).
- `max_workers`: thread count for eager (non-Dask) pairs (default: ~CPU/2 with low-RAM cap).

Example:
```yaml
pairing:
  dask_pair_workers: 1
  dask_num_workers: 4
  max_workers: 4
```

Increasing `dask_pair_workers` can reduce wall time when model data fits comfortably in memory and I/O bandwidth is high; otherwise it can multiply file reads and slow overall pairing.

### CESM Vertical Coordinate Handling

CESM hybrid sigma-pressure coordinates have surface at the **last** level index, not the first:

```
lev=0  → Top of Atmosphere (stratosphere, ~3 hPa)
lev=-1 → Surface (highest pressure, ~1000 hPa)
```

The `_extract_surface()` method auto-detects this by checking if coordinate values increase with index:

```python
if vert_vals[-1] > vert_vals[0]:
    surface_idx = -1  # CESM convention
else:
    surface_idx = 0   # Other conventions
```

### Performance Benchmarks

These measurements are from specific ASIA-AQ runs and are sensitive to hardware, I/O bandwidth, and Dask concurrency settings.

**3-Day Test (72 model files, scratch storage)**:

| Stage | Before | After | Speedup |
|-------|--------|-------|---------|
| load_models | 190s | 6.8s | 28x |
| load_observations | 163s | 0.1s | **1,630x** |
| pairing | 10+ min | 2.3s | Dask-enabled (varies) |
| **Total** | ~175s | ~8s | **22x** |

**Full Month (696 hourly files)**:

| Stage | Time | Notes |
|-------|------|-------|
| load_models | 54s | 696 hourly files |
| load_observations | 0.1s | Time filtering applied |
| pairing | 2.4s | Dask-enabled pairing (configurable) |
| statistics | 0.1s | |
| plotting | 6.5s | |
| **Total** | ~63s | ~1 min |

---

## Performance Considerations

- **Lazy loading**: xarray's lazy evaluation defers computation until needed
- **Chunked processing**: Large files processed in chunks via Dask
- **Time filtering**: Observations filtered at load time using `xr.sel(time=slice())`
- **Parallel I/O**: Dask threaded scheduler with configurable workers for model extraction
- **Spatial indexing**: 1D grids use binary search; 2D grids use haversine distance
- **Memory efficiency**: Paired data only includes matched points, not full grids
- **Storage hierarchy**: On HPC, use scratch storage (parallel FS) over campaign (tape-backed)

---

## See Also

- [CLAUDE.md](CLAUDE.md) - Development guide and quick start
- [PLAN.md](PLAN.md) - Implementation plan and phase details
- [Wiki](https://github.com/NCAR/DAVINCI-MONET/wiki) - User documentation
