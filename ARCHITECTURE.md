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

<pre>
                           DAVINCI-MONET System

    CLI <span style="color: #06b6d4">──────▶</span> Config <span style="color: #06b6d4">──────▶</span> Pipeline <span style="color: #06b6d4">──────▶</span> Output
   (Typer)      (YAML)          Runner        (Plots,
              (Pydantic)                       Stats)
                                  <span style="color: #06b6d4">│</span>
                                  <span style="color: #06b6d4">▼</span>
                          Pipeline Stages
                          ---------------
                          load_models
                              <span style="color: #06b6d4">▼</span>
                          load_observations
                              <span style="color: #06b6d4">▼</span>
                          pairing
                              <span style="color: #06b6d4">▼</span>
                          statistics
                              <span style="color: #06b6d4">▼</span>
                          plotting
                              <span style="color: #06b6d4">▼</span>
                          save_results
                                  <span style="color: #06b6d4">│</span>
                                  <span style="color: #06b6d4">▼</span>
                          Pairing Engine
                          --------------
                    Point - Track - Profile - Swath - Grid
</pre>

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

<pre>
                           PipelineContext
                                 <span style="color: #06b6d4">│</span>
         <span style="color: #06b6d4">┌───────────────────────┴───────────────────────┐</span>
         <span style="color: #06b6d4">▼</span>                                               <span style="color: #06b6d4">▼</span>
    load_models                                  load_observations
         <span style="color: #06b6d4">│</span>                                               <span style="color: #06b6d4">│</span>
         <span style="color: #06b6d4">└──────────────┐           ┌────────────────────┘</span>
                        <span style="color: #06b6d4">▼           ▼</span>
                        pairing
                           <span style="color: #06b6d4">│</span>
                           <span style="color: #06b6d4">▼</span>
                      statistics
                           <span style="color: #06b6d4">│</span>
                           <span style="color: #06b6d4">▼</span>
                       plotting
                           <span style="color: #06b6d4">│</span>
                           <span style="color: #06b6d4">▼</span>
                     save_results

  Data flow:
    load_models <span style="color: #06b6d4">──▶</span> context.models
    load_observations <span style="color: #06b6d4">──▶</span> context.observations
    pairing <span style="color: #06b6d4">──▶</span> context.paired (uses models + observations)
</pre>

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

<pre>
                            PairingEngine
                            -------------

  pair(model, obs, obs_vars, model_vars, config)
      <span style="color: #06b6d4">│</span>
      <span style="color: #06b6d4">├─1─▶</span> _detect_geometry(obs) <span style="color: #06b6d4">──▶</span> DataGeometry enum
      <span style="color: #06b6d4">│</span>
      <span style="color: #06b6d4">├─2─▶</span> _check_temporal_overlap(model, obs)
      <span style="color: #06b6d4">│</span>
      <span style="color: #06b6d4">├─3─▶</span> get_strategy(geometry) <span style="color: #06b6d4">──▶</span> PairingStrategy
      <span style="color: #06b6d4">│</span>
      <span style="color: #06b6d4">└─4─▶</span> strategy.pair(model, obs, ...) <span style="color: #06b6d4">──▶</span> PairedData


                        Strategy Registry
                        -----------------
  POINT   <span style="color: #06b6d4">──▶</span> PointStrategy
  TRACK   <span style="color: #06b6d4">──▶</span> TrackStrategy
  PROFILE <span style="color: #06b6d4">──▶</span> ProfileStrategy
  SWATH   <span style="color: #06b6d4">──▶</span> SwathStrategy
  GRID    <span style="color: #06b6d4">──▶</span> GridStrategy
</pre>

### Data Geometry Types

The `DataGeometry` enum defines five observation geometries:

<pre>
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

      ●        ●                         <span style="color: #06b6d4">●────●────●────●</span>
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
     <span style="color: #06b6d4">│</span>  ●                            ● ● ● ●  scanline 2
     <span style="color: #06b6d4">│</span>  ●                              pixels <span style="color: #06b6d4">──▶</span>
     <span style="color: #06b6d4">│</span>  ●
     <span style="color: #06b6d4">│</span>  ●
     <span style="color: #06b6d4">└────</span>


GRID
----
dims: (time, lat, lon)

Examples:
  - Satellite L3 products
  - Reanalysis data
  - Regridded model output

     lat
      <span style="color: #06b6d4">│</span>  ● ● ● ●
      <span style="color: #06b6d4">│</span>  ● ● ● ●
      <span style="color: #06b6d4">│</span>  ● ● ● ●
      <span style="color: #06b6d4">└─────────▶</span> lon
</pre>

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

<pre>
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

                               <span style="color: #06b6d4">│</span>
       <span style="color: #06b6d4">┌───────────────────────┼───────────────────────┐</span>
       <span style="color: #06b6d4">▼</span>                       <span style="color: #06b6d4">▼</span>                       <span style="color: #06b6d4">▼</span>

PointStrategy           TrackStrategy          ProfileStrategy
-------------           -------------          ---------------
  Extract surface         Interpolate            Interpolate
    level                   along track            vertical levels
  Match by site           4D matching            Match profiles
                            (x,y,z,t)

       <span style="color: #06b6d4">┌───────────────────────┴───────────────────────┐</span>
       <span style="color: #06b6d4">▼</span>                                               <span style="color: #06b6d4">▼</span>

SwathStrategy                                   GridStrategy
-------------                                   ------------
  Handle 2D footprints                            Regrid or interpolate
  Apply averaging kernels                         Direct grid matching
</pre>

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

<pre>
    Model Data                    Observation Data
  (time,z,lat,lon)                  (time, site)
         <span style="color: #06b6d4">│</span>                               <span style="color: #06b6d4">│</span>
         <span style="color: #06b6d4">└───────────────┬───────────────┘</span>
                         <span style="color: #06b6d4">▼</span>
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

                         <span style="color: #06b6d4">│</span>
                         <span style="color: #06b6d4">▼</span>
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
</pre>

---

## Data Flow Diagram

Complete data flow from YAML config to output files:

<pre>
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

                         <span style="color: #06b6d4">│</span>
    <span style="color: #06b6d4">┌────────────────────┴────────────────────┐</span>
    <span style="color: #06b6d4">▼</span>                                         <span style="color: #06b6d4">▼</span>
load_models                           load_observations
    <span style="color: #06b6d4">│</span>                                         <span style="color: #06b6d4">│</span>
    <span style="color: #06b6d4">▼</span>                                         <span style="color: #06b6d4">▼</span>
context.models                        context.observations
  cesm_asiaq: ds                        airnow: ds
                                        pandora: ds
                                        dc8: ds
    <span style="color: #06b6d4">│</span>                                         <span style="color: #06b6d4">│</span>
    <span style="color: #06b6d4">└────────────────────┬────────────────────┘</span>
                         <span style="color: #06b6d4">▼</span>
                      pairing
              (PointStrategy, TrackStrategy)
                         <span style="color: #06b6d4">│</span>
                         <span style="color: #06b6d4">▼</span>
                  context.paired
                    cesm_airnow: pd
                    cesm_pandora: pd
                    cesm_dc8: pd
                         <span style="color: #06b6d4">│</span>
         <span style="color: #06b6d4">┌───────────────┴───────────────┐</span>
         <span style="color: #06b6d4">▼</span>                               <span style="color: #06b6d4">▼</span>
    statistics                       plotting
         <span style="color: #06b6d4">│</span>                               <span style="color: #06b6d4">│</span>
         <span style="color: #06b6d4">▼</span>                               <span style="color: #06b6d4">▼</span>
  statistics_summary.csv          *.png, *.pdf
                                  (scatter, timeseries,
                                   spatial, 3d tracks)
</pre>

---

## Module Structure

<pre>
davinci_monet/
<span style="color: #06b6d4">├──</span> core/                    # Foundation layer
<span style="color: #06b6d4">│   ├──</span> protocols.py         # Interface definitions (ModelReader, etc.)
<span style="color: #06b6d4">│   ├──</span> registry.py          # Plugin registration system
<span style="color: #06b6d4">│   ├──</span> exceptions.py        # Custom exception hierarchy
<span style="color: #06b6d4">│   ├──</span> types.py             # Type aliases and helpers
<span style="color: #06b6d4">│   └──</span> base.py              # Base classes (PairedData, etc.)
<span style="color: #06b6d4">│</span>
<span style="color: #06b6d4">├──</span> config/                  # Configuration handling
<span style="color: #06b6d4">│   ├──</span> schemas.py           # Pydantic models for YAML validation
<span style="color: #06b6d4">│   └──</span> loader.py            # YAML parsing with env var expansion
<span style="color: #06b6d4">│</span>
<span style="color: #06b6d4">├──</span> models/                  # Model readers
<span style="color: #06b6d4">│   ├──</span> base.py              # BaseModelReader
<span style="color: #06b6d4">│   ├──</span> cmaq.py              # CMAQ reader
<span style="color: #06b6d4">│   ├──</span> wrfchem.py           # WRF-Chem reader
<span style="color: #06b6d4">│   ├──</span> cesm.py              # CESM/CAM-chem reader
<span style="color: #06b6d4">│   ├──</span> ufs.py               # UFS reader
<span style="color: #06b6d4">│   └──</span> generic.py           # Generic NetCDF reader
<span style="color: #06b6d4">│</span>
<span style="color: #06b6d4">├──</span> observations/            # Observation readers
<span style="color: #06b6d4">│   ├──</span> base.py              # BaseObservationReader
<span style="color: #06b6d4">│   ├──</span> surface/             # Surface observations
<span style="color: #06b6d4">│   │   ├──</span> airnow.py        # AirNow (US Embassy monitors)
<span style="color: #06b6d4">│   │   ├──</span> aeronet.py       # AERONET AOD
<span style="color: #06b6d4">│   │   ├──</span> aqs.py           # EPA AQS
<span style="color: #06b6d4">│   │   ├──</span> openaq.py        # OpenAQ
<span style="color: #06b6d4">│   │   └──</span> pandora.py       # Pandora NO2 columns
<span style="color: #06b6d4">│   ├──</span> aircraft/            # Aircraft observations
<span style="color: #06b6d4">│   │   └──</span> icartt.py        # ICARTT format reader
<span style="color: #06b6d4">│   └──</span> satellite/           # Satellite observations
<span style="color: #06b6d4">│       ├──</span> tropomi.py       # TROPOMI L2
<span style="color: #06b6d4">│       └──</span> goes.py          # GOES-16/17 AOD
<span style="color: #06b6d4">│</span>
<span style="color: #06b6d4">├──</span> pairing/                 # Pairing engine
<span style="color: #06b6d4">│   ├──</span> engine.py            # PairingEngine orchestrator
<span style="color: #06b6d4">│   └──</span> strategies/          # Geometry-specific strategies
<span style="color: #06b6d4">│       ├──</span> base.py          # BasePairingStrategy
<span style="color: #06b6d4">│       ├──</span> point.py         # Fixed locations
<span style="color: #06b6d4">│       ├──</span> track.py         # Aircraft/mobile tracks
<span style="color: #06b6d4">│       ├──</span> profile.py       # Vertical profiles
<span style="color: #06b6d4">│       ├──</span> swath.py         # Satellite swaths
<span style="color: #06b6d4">│       └──</span> grid.py          # Gridded data
<span style="color: #06b6d4">│</span>
<span style="color: #06b6d4">├──</span> pipeline/                # Execution engine
<span style="color: #06b6d4">│   ├──</span> runner.py            # PipelineRunner + ProgressFormatter
<span style="color: #06b6d4">│   ├──</span> stages.py            # Stage protocol + implementations
<span style="color: #06b6d4">│   └──</span> parallel.py          # Parallel execution utilities
<span style="color: #06b6d4">│</span>
<span style="color: #06b6d4">├──</span> plots/                   # Plotting system
<span style="color: #06b6d4">│   ├──</span> base.py              # BasePlotter + label formatting
<span style="color: #06b6d4">│   ├──</span> registry.py          # Plot type registration
<span style="color: #06b6d4">│   └──</span> renderers/           # Plot implementations
<span style="color: #06b6d4">│       ├──</span> scatter.py       # Scatter plots
<span style="color: #06b6d4">│       ├──</span> timeseries.py    # Time series
<span style="color: #06b6d4">│       ├──</span> spatial_bias.py  # Spatial bias maps
<span style="color: #06b6d4">│       ├──</span> site_timeseries.py    # Multi-panel site plots
<span style="color: #06b6d4">│       ├──</span> flight_timeseries.py  # Multi-panel flight plots
<span style="color: #06b6d4">│       └──</span> track_map_3d.py       # 3D flight track visualization
<span style="color: #06b6d4">│</span>
<span style="color: #06b6d4">├──</span> stats/                   # Statistics
<span style="color: #06b6d4">│   ├──</span> calculator.py        # MetricsCalculator
<span style="color: #06b6d4">│   ├──</span> metrics.py           # Individual metric implementations
<span style="color: #06b6d4">│   └──</span> output.py            # CSV/table formatters
<span style="color: #06b6d4">│</span>
<span style="color: #06b6d4">├──</span> io/                      # File I/O
<span style="color: #06b6d4">│   ├──</span> readers.py           # Generic readers
<span style="color: #06b6d4">│   └──</span> writers.py           # NetCDF writers with compression
<span style="color: #06b6d4">│</span>
<span style="color: #06b6d4">└──</span> cli/                     # Command-line interface
    <span style="color: #06b6d4">├──</span> app.py               # Main Typer application
    <span style="color: #06b6d4">└──</span> commands/            # Subcommands
        <span style="color: #06b6d4">├──</span> run.py           # Pipeline execution
        <span style="color: #06b6d4">├──</span> validate.py      # Config validation
        <span style="color: #06b6d4">└──</span> get_data.py      # Data download
</pre>

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
