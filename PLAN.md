# DAVINCI-MONET Implementation Plan

## Overview

Complete refactor of MELODIES-MONET addressing:
- `driver.py`: 3,116 lines → ~50 modules (<500 lines each)
- `plotting()`: 1,321-line method → individual plotter classes
- No type hints → Full type hints + mypy strict
- 211 print() → Structured logging
- 4 test files → Comprehensive test suite (>80% coverage)

### Key Design Principles

1. **Uniform Pairing Logic**: A single, consistent pairing engine that works across all observation types (surface, aircraft, satellite, sonde). The pairing strategy is determined by data geometry (point, profile, swath, grid) rather than data source.

2. **Synthetic Data for Testing**: Generate test data programmatically - no dependency on external datasets. Tests are self-contained and reproducible.

3. **xarray-Only Data Model**: All data flows through the system as `xr.Dataset`. No pandas in core pairing/analysis logic.

   | Component | Data Structure |
   |-----------|----------------|
   | Model data | `xr.Dataset` with dims (time, level, lat, lon) |
   | Point obs | `xr.Dataset` with dims (time, site) + lat/lon coords |
   | Track obs | `xr.Dataset` with dims (time,) + lat/lon/alt coords |
   | Profile obs | `xr.Dataset` with dims (time, level) + lat/lon coords |
   | Swath obs | `xr.Dataset` with dims (time, scanline, pixel) |
   | Grid obs | `xr.Dataset` with dims (time, lat, lon) |
   | Paired output | `xr.Dataset` |

   **pandas limited to:**
   - I/O adapters: Read CSV/tabular → convert to xarray
   - Statistics output: Final tables for export

---

## Phase 1: Foundation
**Status: COMPLETE**

- [x] Create package structure and `py.typed`
- [x] Create `pyproject.toml`
- [x] Implement `core/protocols.py` - Protocol definitions (15 protocols)
- [x] Implement `core/registry.py` - Plugin registry system
- [x] Implement `core/exceptions.py` - Custom exception hierarchy (20 exceptions)
- [x] Implement `core/types.py` - Type aliases
- [x] Implement `logging/config.py` - Structured logging

---

## Phase 2: Synthetic Data & Testing Infrastructure
**Status: COMPLETE**

Build test data generators early so all subsequent phases can be tested immediately.

- [x] Implement `tests/synthetic/generators.py` - Base data generators
  - Domain, TimeConfig, VariableSpec dataclasses
  - Coordinate grid, time axis, level axis creation
  - Random field generation with spatial correlation
  - Diurnal cycle and noise functions
- [x] Implement `tests/synthetic/models.py` - Synthetic model output
  - Gridded 3D/4D fields (lat, lon, time, level)
  - Configurable domain, resolution, variables
  - Realistic value ranges for common species (O3, PM2.5, NO2, etc.)
- [x] Implement `tests/synthetic/observations.py` - Synthetic observations
  - Point surface observations (station locations, time series)
  - Aircraft tracks (3D trajectories with measurements)
  - Satellite swaths (scan patterns, footprints)
  - Vertical profiles (sondes)
  - Gridded observations (L3)
- [x] Implement `tests/synthetic/scenarios.py` - Pre-built test scenarios
  - Perfect match (model = obs)
  - Known bias (model = obs + offset)
  - Spatial/temporal mismatch cases
- [x] Implement `tests/conftest.py` - Pytest fixtures using generators
- [x] Comprehensive tests (94 tests for synthetic module)

---

## Phase 3: Configuration
**Status: COMPLETE**

- [x] Implement `config/schema.py` - Pydantic models for YAML validation
  - FlexibleModel base class (extra="allow") for backward compatibility
  - StrictModel base class (extra="forbid") for strict validation
  - AnalysisConfig, ModelConfig, ObservationConfig, PlotGroupConfig, StatsConfig
  - VariableConfig, DataProcConfig
  - MonetConfig root configuration with get_model_obs_pairs() method
  - Field validators for datetime parsing (multiple formats)
- [x] Implement `config/parser.py` - YAML parsing with backward compat
  - load_yaml, load_config, validate_config, dump_config, config_to_yaml
  - expand_env_vars for ${VAR} expansion
  - preprocess_config for legacy format handling
  - merge_configs for deep dictionary merging
  - ConfigBuilder class for programmatic configuration
- [x] Implement `config/migration.py` - Config version migrations
  - detect_config_version for auto-detection
  - ConfigMigration class with migration registry
  - Built-in migrations: 0.0.0→1.0.0, 1.0.0→2.0.0
  - check_deprecated_fields for deprecation warnings
  - validate_version_compatibility for version checks
- [x] Write tests (85 tests for config module, 311 total)

---

## Phase 4: Core Data Classes
**Status: COMPLETE**

- [x] Implement `core/base.py` - Core data containers
  - DataContainer: Abstract base class for data wrappers
  - PairedData: Container for paired model-observation data
  - create_paired_dataset(): Factory for paired datasets
  - validate_dataset_geometry(): Geometry validation
- [x] Implement `models/base.py` - Model data handling
  - ModelData: Container for model output (CMAQ, WRF-Chem, etc.)
  - create_model_data(): Factory function
  - File resolution, variable scaling, masking, renaming
  - Vertical/horizontal interpolation, regridding
- [x] Implement `observations/base.py` - Observation data handling
  - ObservationData: Base container with geometry-based typing
  - Geometry-specific classes: PointObservation, TrackObservation,
    ProfileObservation, SwathObservation, GriddedObservation
  - create_observation_data(): Factory function
  - QA filtering, resampling, coordinate handling
- [x] Write tests (106 new tests, 417 total)

---

## Phase 5: Unified Pairing Engine
**Status: COMPLETE**

A single pairing system based on data geometry, not data source.

### Data Geometries
| Geometry | Model Side | Observation Side | Examples |
|----------|------------|------------------|----------|
| Point-to-Grid | 3D/4D grid | Point locations | Surface stations, ground sites |
| Profile-to-Grid | 3D/4D grid | Vertical profiles | Sondes, aircraft profiles |
| Track-to-Grid | 3D/4D grid | 3D trajectory | Aircraft, mobile platforms |
| Swath-to-Grid | 3D/4D grid | 2D swath pixels | Satellite L2 products |
| Grid-to-Grid | 3D/4D grid | Gridded product | Satellite L3, reanalysis |

### Implementation
- [x] Implement `pairing/engine.py` - Unified pairing orchestrator
  - PairingConfig: Configuration dataclass with radius, tolerance, methods
  - PairingEngine: Orchestrates pairing with strategy dispatch
  - Automatic geometry detection from observation datasets
  - Temporal overlap validation
- [x] Implement `pairing/strategies/base.py` - Base strategy class
  - BasePairingStrategy: Abstract base with common utilities
  - Haversine distance calculation
  - Nearest neighbor search (1D and 2D grids)
  - Time and vertical interpolation methods
  - Surface extraction from 3D models
- [x] Implement `pairing/strategies/point.py` - Point-to-grid matching
  - PointStrategy: Pairs surface stations with model grid
  - Site-based extraction at nearest grid cells
  - Temporal interpolation to observation times
- [x] Implement `pairing/strategies/track.py` - Track-to-grid matching
  - TrackStrategy: Pairs aircraft/mobile tracks with model
  - 3D interpolation along trajectory
  - Altitude-based vertical interpolation
- [x] Implement `pairing/strategies/profile.py` - Profile-to-grid matching
  - ProfileStrategy: Pairs vertical profiles (sondes) with model columns
  - Vertical level interpolation
- [x] Implement `pairing/strategies/swath.py` - Swath-to-grid matching
  - SwathStrategy: Pairs satellite swaths with model
  - Pixel-by-pixel extraction
  - Optional averaging kernel support (placeholder)
- [x] Implement `pairing/strategies/grid.py` - Grid-to-grid regridding
  - GridStrategy: Pairs gridded observations with model
  - Regrid to obs grid or model grid
  - Support for curvilinear grids
- [x] Write tests (27 tests for pairing module, 444 total)

### Common Pairing Parameters
All strategies share:
- `radius_of_influence` - Spatial search radius
- `time_tolerance` - Temporal matching window
- `vertical_method` - Interpolation method for vertical
- `horizontal_method` - Nearest, bilinear, etc.

---

## Phase 6: Model Implementations
**Status: COMPLETE**

Each model reader produces standardized output that feeds into the unified pairing engine.

- [x] Implement `models/cmaq.py`
  - CMAQReader with monetio integration (cmaq_mm fallback to xarray)
  - Dimension standardization (TSTEP→time, LAY→z, ROW→y, COL→x)
  - Variable mapping (O3, PM25_TOT, NO2, CO, SO2, etc.)
  - `open_cmaq()` convenience function
- [x] Implement `models/wrfchem.py`
  - WRFChemReader with monetio integration (_wrfchem_mm)
  - Dimension standardization (Time→time, bottom_top→z, south_north→y, west_east→x)
  - XLAT/XLONG coordinate handling
  - `open_wrfchem()` convenience function
- [x] Implement `models/ufs.py`
  - UFSReader for UFS-AQM output (grib2 and NetCDF)
  - RRFSReader alias for backward compatibility
  - Dimension standardization for grib2 and NetCDF formats
  - `open_ufs()` convenience function
- [x] Implement `models/cesm.py`
  - CESMFVReader for finite volume grid (regular lat-lon)
  - CESMSEReader for spectral element grid (unstructured, SCRIP file support)
  - Dimension standardization (lev→z, ilev→z_interface)
  - `open_cesm()` convenience function with grid_type parameter
- [x] Implement `models/generic.py` - Fallback handler
  - GenericReader with automatic engine detection
  - Common coordinate alias standardization
  - `open_model()` universal function with registry dispatch
- [x] Write tests (39 tests for model readers, 483 total)

---

## Phase 7: Observation Implementations
**Status: COMPLETE**

Each observation reader tags its data with geometry type for the pairing engine.

### Surface Observations (POINT geometry)
- [x] Implement `observations/surface/aqs.py` - EPA AQS data with monetio integration
  - AQSReader with file and API query support
  - `open_aqs()` convenience function
- [x] Implement `observations/surface/airnow.py` - AirNow real-time data
  - AirNowReader with monetio integration
  - `open_airnow()` convenience function
- [x] Implement `observations/surface/aeronet.py` - AERONET AOD data
  - AERONETReader with product selection (AOD10, AOD15, etc.)
  - `open_aeronet()` convenience function
- [x] Implement `observations/surface/openaq.py` - OpenAQ global data
  - OpenAQReader with API v2/v3 support
  - `open_openaq()` convenience function

### Aircraft Observations (TRACK geometry)
- [x] Implement `observations/aircraft/icartt.py` - ICARTT format
  - ICARTTReader with monetio fallback to basic parser
  - `open_icartt()` convenience function

### Satellite Observations (SWATH/GRID geometry)
- [x] Implement `observations/satellite/tropomi.py` - TROPOMI L2
  - TROPOMIReader with QA filtering
  - `open_tropomi()` convenience function
- [x] Implement `observations/satellite/goes.py` - GOES-ABI AOD
  - GOESReader with DQF filtering
  - `open_goes()` convenience function

### Profile Observations (PROFILE geometry)
- [x] Implement `observations/sonde/ozonesonde.py` - Ozonesonde profiles
  - OzonesondeReader with WOUDC/SHADOZ/NetCDF format support
  - `open_ozonesonde()` convenience function

- [x] Write tests (36 tests for observation readers, 519 total)

---

## Phase 8: Pipeline
**Status: COMPLETE**

- [x] Implement `pipeline/stages.py` - Stage definitions
  - Stage protocol and BaseStage abstract class
  - StageStatus enum, StageResult dataclass
  - PipelineContext for data flow between stages
  - Concrete stages: LoadModelsStage, LoadObservationsStage, PairingStage
  - Concrete stages: StatisticsStage, PlottingStage, SaveResultsStage
  - create_standard_pipeline() factory function
- [x] Implement `pipeline/runner.py` - PipelineRunner (replaces analysis class)
  - PipelineRunner: Orchestrates stage execution with fail-fast option
  - PipelineResult: Tracks success, stage results, timing
  - PipelineBuilder: Fluent API for pipeline construction
  - run_analysis(): Convenience function for config-based runs
  - Hook support: on_start, on_stage_start, on_stage_end, on_end
- [x] Implement `pipeline/parallel.py` - Parallel execution
  - ParallelExecutor: Thread/process pool-based parallel execution
  - ParallelResult: Generic result container with errors tracking
  - ParallelPairingExecutor: Parallel model-observation pairing
  - parallel_process_files(): Convenience for file processing
- [x] Implement `io/readers.py` - File readers
  - read_dataset(): Auto-format detection for NetCDF, pickle, grib
  - read_mfdataset(): Multi-file dataset reading with glob support
  - read_pickle(), read_csv(), read_csv_to_xarray()
  - read_icartt(): ICARTT format with basic parser fallback
  - read_saved_analysis(): Load saved analysis results
- [x] Implement `io/writers.py` - File writers (NetCDF, pickle, CSV)
  - write_dataset(): Auto-format detection for NetCDF, pickle, zarr
  - write_pickle(), write_csv()
  - write_paired_data(): Write paired datasets with prefix support
  - write_statistics(): CSV, JSON, and pickle output
  - write_analysis_results(): Complete analysis output
- [x] Write tests (103 tests for pipeline/io, 621 total)

---

## Phase 9: Plotting
**Status: COMPLETE**

**PRIMARY OBJECTIVE: Publication-quality figures.** All plots must be suitable for direct inclusion in peer-reviewed journal articles. This means:
- Clean, professional styling (NCAR branding with Poppins font)
- Appropriate axis labels with units
- Legends that don't obscure data
- Consistent color schemes (gray for obs, NCAR blue for model)
- High DPI output (300 for PNG, vector PDF)
- Density coloring for busy scatter plots (`show_density: true`)

### Implemented Plot Types

- [x] Implement `plots/base.py` - BasePlotter, PlotConfig, utility functions
- [x] Implement `plots/registry.py` - Plot registry and factory functions
- [x] Implement `plots/renderers/timeseries.py` - Time series comparisons
- [x] Implement `plots/renderers/diurnal.py` - Diurnal cycle plots
- [x] Implement `plots/renderers/taylor.py` - Taylor diagrams
- [x] Implement `plots/renderers/boxplot.py` - Box plot comparisons
- [x] Implement `plots/renderers/scatter.py` - Scatter plots with regression/density
- [x] Implement `plots/renderers/spatial/base.py` - BaseSpatialPlotter, MapConfig
- [x] Implement `plots/renderers/spatial/bias.py` - Spatial bias maps
- [x] Implement `plots/renderers/spatial/overlay.py` - Model contour + obs overlays
- [x] Implement `plots/renderers/spatial/distribution.py` - Value distribution maps
- [x] Implement `plots/renderers/curtain.py` - Vertical cross-sections
- [x] Implement `plots/renderers/scorecard.py` - Multi-metric heatmaps
- [x] Write tests (39 tests for plotting, 660 total)

---

## Phase 10: Statistics
**Status: COMPLETE**

- [x] Implement `stats/metrics.py` - 27 metric classes (bias, error, correlation, etc.)
- [x] Implement `stats/calculator.py` - StatisticsCalculator with groupby support
- [x] Implement `stats/output.py` - CSV, JSON, table image output formatters
- [x] Write tests (39 tests for statistics, 699 total)

---

## Phase 11: CLI
**Status: COMPLETE**

- [x] Implement `cli/app.py` - Main Typer application
  - Version callback, header display, timer context manager
  - Global commands: run, validate
  - Subcommand group: get (for data download)
- [x] Implement `cli/commands/run.py` - Run command
  - Execute full analysis pipeline from control file
  - Stage-by-stage execution with timing
- [x] Implement `cli/commands/get_data.py` - Data download commands
  - get aeronet, get airnow, get aqs, get openaq
  - Common options: dates, output file, compression, workers
- [x] Implement `cli/commands/validate.py` - Config validation
  - Version detection, deprecation warnings
  - Configuration summary display
- [x] Write tests (33 tests for CLI, 732 total)

---

## Phase 12: Documentation & Polish
**Status: PENDING**

- [ ] Integration tests with full synthetic scenarios
- [ ] API documentation
- [ ] Migration guide from MELODIES-MONET
- [ ] Example notebooks

---

## Technical Decisions

| Aspect | Decision |
|--------|----------|
| Python version | 3.10+ |
| Type checking | mypy strict mode |
| Validation | Pydantic v2 |
| Logging | Python `logging` module |
| CLI | Typer |
| Testing | pytest + pytest-cov + synthetic data |
| Formatting | Black + isort |
| Data model | xarray-only (pandas for I/O adapters and stats tables only) |
| Dependencies | Continue using monet/monetio |

---

## Unified Pairing Architecture

**All data as xarray.Dataset throughout the pipeline.**

```
┌─────────────────┐     ┌─────────────────┐
│  Model Reader   │     │   Obs Reader    │
│  (NetCDF, etc)  │     │ (NetCDF, CSV→xr)│
└────────┬────────┘     └────────┬────────┘
         │                       │
         ▼                       ▼
┌─────────────────┐     ┌─────────────────┐
│ xr.Dataset      │     │ xr.Dataset      │
│ dims: (time,    │     │ dims: varies by │
│   level, lat,   │     │   geometry type │
│   lon)          │     │ attrs: geometry │
└────────┬────────┘     └────────┬────────┘
         │                       │
         └───────────┬───────────┘
                     ▼
         ┌───────────────────────┐
         │   Pairing Engine      │
         │   (xarray operations) │
         │   ─────────────────   │
         │   Selects strategy    │
         │   based on geometry   │
         └───────────┬───────────┘
                     │
         ┌───────────┴────────────┐
         │   Strategy dispatch    │
         ├────────────────────────┤
         │ point   → PointPairer  │
         │ track   → TrackPairer  │
         │ profile → ProfilePairer│
         │ swath   → SwathPairer  │
         │ grid    → GridPairer   │
         └───────────┬────────────┘
                     │
                     ▼
         ┌───────────────────────┐
         │   Paired xr.Dataset   │
         │   (model + obs vars   │
         │    aligned on coords) │
         └───────────────────────┘
```

**Observation Dimension Conventions:**
```
Point:   (time, site)         - coords: lat(site), lon(site)
Track:   (time,)              - coords: lat(time), lon(time), alt(time)
Profile: (time, level)        - coords: lat(time), lon(time)
Swath:   (time, scanline, pixel) - coords: lat(...), lon(...)
Grid:    (time, lat, lon)     - regular grid
```

---

## Files to Reference (MELODIES-MONET)

| File | Purpose |
|------|---------|
| `melodies_monet/driver.py` | Main logic to decompose (3,116 lines) |
| `melodies_monet/_cli.py` | CLI implementation (1,524 lines) |
| `melodies_monet/plots/` | Plotting modules |
| `melodies_monet/stats/` | Statistics modules |
| `examples/yaml/` | 31 example YAML configs for testing |
