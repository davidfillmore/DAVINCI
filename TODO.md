# TODO - DAVINCI-MONET

Priority items for future development sessions.

## Priority 0: AERONET NetCDF Reader (Current Session)

### Objective
Get ASIA-AQ analysis working with the new AERONET NetCDF format.

### AERONET Data Structure
The new NetCDF file (`~/Data/ASIA-AQ/AERONET/AERONET_L15_20240101_20240501.nc`):
- Dimensions: `(time=2904, y=1, x=508)` where `y=1` is dummy, `x` is sites
- Coordinates: `latitude(508)`, `longitude(508)`, `siteid(508)` - 1D arrays
- Data: `aod_500nm(time, y, x)` - needs `y` squeezed and `x` renamed to `site`

### Tasks (ALL COMPLETE)
- [x] Copy `asia-aq.yaml` to `asia-aq-gemini.yaml`
- [x] Debug AERONET reader for NetCDF format
- [x] Test pipeline with 3-day period (Feb 1-3)
- [x] Verify pairing produces non-empty results
  - **731 paired points** for 3-day test
  - R = 0.615, NMB = -54% (model underpredicts)
- [x] Check time filtering is working
  - load_observations: 0.0s (filtered 5-month file to 3 days)
- [x] Generate both scatter and timeseries plots

### Fixes Applied

1. **AERONET reader** (`davinci_monet/observations/surface/aeronet.py`):
   - `_standardize_dataset()`: Squeeze `y` dimension (y=1 dummy dim)
   - `_standardize_dataset()`: Rename `x` to `site` for point geometry consistency

2. **LoadObservationsStage** (`davinci_monet/pipeline/stages.py`):
   - Added AERONET reader detection by label or filename
   - Uses proper reader instead of raw `xr.open_dataset`

3. **Config** (`asia-aq-gemini.yaml`):
   - Use `mod_type: generic` for 2D regridded CESM files
   - Use `aggregate_dim: site` for timeseries

### Config File
`analyses/asia-aq/configs/asia-aq-gemini.yaml` - Mac testing config

---

## Priority 1: Performance Optimizations

### Completed
- [x] Dask parallel scheduler for PointStrategy pairing (260x speedup)
- [x] Dask parallel scheduler for TrackStrategy pairing (aircraft data)
- [x] Scratch storage config for Derecho
- [x] Progress output fixes (Pairing/Paired pattern, duplicate removal)
- [x] Time filtering at observation load:
  - File-level: filters by YYYYMMDD in filename (e.g., ICARTT files)
  - Data-level: filters by time dimension after loading (e.g., NetCDF)
  - Result: load_observations 163s -> 0.1s (1,630x faster)

### Remaining
- [ ] Pre-chunked model data (daily/weekly concatenated files)
- [ ] **Optional: Pre-load Dask model before pairing** (potential 60s → 25s)

  **Problem**: Each Dask-backed pair independently calls `.compute()`, reloading/processing
  the 696 model files. With 3 pairs using `cesm_asiaq`, this happens 3 times.

  **Key insight - "Democracy, not monarchy"**: The nearly equal per-pair times (~22-25s each)
  show that labor is distributed equally - each pair redundantly does the same work. There's
  no "first thread does heavy lifting, others benefit" pattern. This is actually worse than
  if one thread warmed the cache for others.

  **ASIA-AQ Pair Configuration**:
  | Pair | Model | Dask? | Why |
  |------|-------|-------|-----|
  | cesm_asiaq_airnow | cesm_asiaq | Yes | 696 files, lazy loading |
  | cesm_asiaq_aeronet | cesm_asiaq | Yes | 696 files, lazy loading |
  | cesm_asiaq_dc8 | cesm_asiaq | Yes | 696 files, lazy loading |
  | cesm_no2_column_pandora | cesm_no2_column | No | 1 file, eager loading |

  **Current timing** (from profiling 2026-01-23):
  ```
  cesm_asiaq_airnow:   24.5s  (loads 696 files)
  cesm_asiaq_aeronet:  24.8s  (loads 696 files again)
  cesm_asiaq_dc8:      22.3s  (loads 696 files again)
  cesm_no2_column_pandora: 0.0s  (already in memory)
  Total: ~72s sequential, ~60s parallel (only 17% savings from incidental caching)
  ```

  **Proposed feature**: Optional `preload: true` in model config to call `.compute()` once
  before pairing, converting to in-memory NumPy arrays. All pairings would then be <1s each.

  ```yaml
  model:
    cesm_asiaq:
      mod_type: generic
      files: ${DATA}/*.nc
      preload: true  # Optional: load into memory before pairing
  ```

  **Trade-offs**:
  - Pro: 60s → ~25s for pairing stage
  - Con: Memory usage increases (~20 GB for 1-month CESM at 1°)
  - Con: Not feasible for 3+ month analyses or high-resolution models

  See `PERFORMANCE.md` for detailed analysis.

- [x] **Fix Pandora pairing bottleneck** (60.8s reported, actual pairing ~1s) - FIXED

  **Root Cause**: ThreadPoolExecutor + Dask GIL contention caused inflated timing.

  **Fix Applied**: Two-phase sequential pairing in `PairingStage`:
  1. Phase 1: Process Dask-backed model pairs (cesm_asiaq_*)
  2. Phase 2: Process eager model pairs (cesm_no2_column_pandora)

  **Result**: Per-pair timing now accurate:
  - cesm_asiaq_* pairs: ~24s each (actual Dask compute time)
  - cesm_no2_column_pandora: <1s (was falsely reported as 60s)

---

## Priority 2: Feature Additions

- [x] Pandora NO2 column preprocessing and reader (14 sites, 8.8k obs)
- [x] AirNow data download and integration (36 sites, PM2.5/O3)
- [x] **Per-flight statistics**: Compute and save statistics for each flight separately
  - Output: `statistics_per_flight.csv` with one row per flight per variable
  - Metrics: N, MO, MP, MB, RMSE, R, NMB_%, NME_%
  - Enable via `per_flight: true` in stats config section
  - Warning added when pairing produces no data (transient HDF5/Dask issue)
- [ ] MOPITT CO profile evaluation
- [ ] MODIS AOD comparison

---

## Notes

### Derecho Storage Hierarchy (fastest to slowest)
1. `/glade/derecho/scratch` - Parallel FS, purged after 60 days
2. `/glade/work` - Parallel FS (GPFS), persistent
3. `/glade/campaign` - Tape-backed, archival

### Performance Benchmarks

**3-day test (72 model files, scratch storage)**:

| Stage | Before | After | Speedup |
|-------|--------|-------|---------|
| load_models | 190s | 6.8s | 28x |
| load_observations | 163s | 0.1s | 1,630x |
| pairing | 10+ min | 2.3s | 260x |
| **Total** | ~175s | ~8s | **22x** |

**Full month (696 files, scratch, all optimizations)**:

| Stage | Time | Notes |
|-------|------|-------|
| load_models | 54.2s | 696 hourly files |
| load_observations | 163.2s | Time filtering may not have applied? |
| pairing | 210.5s | DC8 + AERONET combined |
| statistics | 0.0s | |
| plotting | 6.5s | |
| **Total** | ~435s | ~7.2 min |

See `DERECHO.md` for full environment setup and data paths.

---

## Priority 3: Testing Configuration

### 1-Week Test Config
Create a 1-week test configuration for faster iteration on issues:
- [ ] Create `asia-aq-1week.yaml` with Feb 1-7, 2024
- [ ] Use for debugging time filtering and pairing issues
- [ ] Target: full pipeline under 2 minutes

---

## Session Summary (2026-01-19)

### Key Accomplishments
1. Fixed pairing performance bottleneck (Dask parallel scheduler)
2. Added time filtering to observation loading (1,630x speedup)
3. Extended optimization to TrackStrategy for aircraft data
4. Set up full month analysis on scratch storage
5. Created comprehensive documentation in DERECHO.md and TODO.md

### Files Modified
- `davinci_monet/pipeline/stages.py` - Time filtering at load
- `davinci_monet/pipeline/runner.py` - Progress output fixes
- `davinci_monet/pairing/strategies/point.py` - Dask parallel scheduler
- `davinci_monet/pairing/strategies/track.py` - Dask parallel scheduler
- `davinci_monet/cli/commands/run.py` - Remove duplicate config display
- `analyses/asia-aq/configs/asia-aq-scratch.yaml` - Full month config

### Current Run
Full month (Feb 2024) analysis running with AERONET + DC8 on scratch storage.

### Issues (Resolved)
- [x] **AERONET pairing suspiciously fast** - RESOLVED: Pairing is actually fast because:
  - Time filtering properly reduces 5-month file to analysis period
  - 731 valid pairs for 3-day test confirms data is being paired correctly

- [x] **Time filtering not applied?** - RESOLVED: Time filtering IS working on Mac
  - load_observations: 0.0s with time filter
  - Derecho issue may have been transient or different config

---

## Session Summary (2026-01-23)

### Key Accomplishments
1. Fixed AERONET NetCDF reader for new data format:
   - Squeeze `y=1` dummy dimension
   - Rename `x` to `site` for consistency
2. Updated LoadObservationsStage to use AERONET reader
3. Created `asia-aq-gemini.yaml` Mac testing config
4. Verified pipeline produces correct results:
   - 731 paired points, R=0.615, NMB=-54%
   - Both scatter and timeseries plots working
5. **Fixed CESM vertical level extraction (RECURRING ISSUE - 4th time!)**:
   - Bug: `_extract_surface()` was extracting `lev=0` (stratosphere) instead of `lev=-1` (surface)
   - Symptom: O3 model values of 7253 ppb instead of ~50 ppb
   - Fix: Auto-detect vertical dimension and use correct surface index based on coordinate values
   - Updated CLAUDE.md, VALIDATION.md, ARCHITECTURE.md with consistent documentation
   - Added prominent "CRITICAL" warning section to CLAUDE.md to prevent future rediscovery

### Files Modified
- `davinci_monet/observations/surface/aeronet.py` - Standardize NetCDF dimensions
- `davinci_monet/pipeline/stages.py` - Use AERONET reader in observation loading
- `davinci_monet/pairing/strategies/base.py` - Fixed `_extract_surface()` for CESM coordinates
- `davinci_monet/pairing/strategies/track.py` - Use fixed base class method
- `analyses/asia-aq/configs/asia-aq-gemini.yaml` - Full month config with DC8 + AERONET
- `CLAUDE.md` - Added CRITICAL warning about CESM vertical coordinates
- `VALIDATION.md` - Corrected vertical coordinate documentation
- `ARCHITECTURE.md` - Corrected surface level extraction documentation

---

## Session Summary (2026-01-23 continued)

### Key Accomplishments
1. **Improved pairing progress display**:
   - Shows loading context during [0/N] phase: `loading cesm_asiaq → airnow, aeronet, dc8`
   - Makes it clear Dask lazy loading is happening, not one slow pair

2. **Suppressed ugly cleanup tracebacks**:
   - NetCDF file open errors no longer show `CachingFileManager.__del__` tracebacks
   - Added `_cleanup_with_suppressed_errors()` in `generic.py`

3. **Updated wiki documentation**:
   - Home.md: Test count (790+), performance feature
   - ASIA-AQ-Analysis.md: Performance section, CESM vertical troubleshooting
   - Configuration.md: `generic` vs `cesm_fv` guidance
   - API-Reference.md: Two-phase pairing execution

### Files Modified
- `davinci_monet/pipeline/runner.py` - Added `_parallel_loading_msg` for context display
- `davinci_monet/pipeline/stages.py` - Build loading message from Dask pairs
- `davinci_monet/models/generic.py` - Suppress cleanup errors on file open failure

### Environment Setup Note
`ASIA_AQ_DATA` environment variable must be set for pipeline to find model files:
```bash
export ASIA_AQ_DATA=~/Data/ASIA-AQ
```
Add to `~/.zshrc` for persistence across sessions.

---

## Session Summary (2026-01-23 Derecho)

### Key Accomplishments
1. **ASIA-AQ analysis working on Derecho**:
   - Full month (Feb 2024) pipeline completed successfully
   - 14 plots generated, statistics verified
   - Output: `/glade/derecho/scratch/fillmore/ASIA-AQ/output/`

2. **Consolidated config files**:
   - Updated `asia-aq-derecho.yaml` to use scratch storage
   - Deleted redundant `asia-aq-scratch.yaml`

3. **Updated performance benchmarks** (peak hours):
   - load_models: 96s (696 files)
   - load_observations: 165s (AERONET + DC8 ICARTT)
   - pairing: 600s (4 pairs, cluster load impact)
   - Total: ~15 min

4. **Cleaned up documentation**:
   - DERECHO.md: Updated paths, performance stats, removed stale references
   - TODO.md: Removed stale branch references

### Statistics Results
| Variable | N | R | NMB |
|----------|-------|------|------|
| O3 (DC8) | 19,391 | 0.27 | +35% |
| NO2 (DC8) | 19,509 | 0.41 | +533% |
| CO (DC8) | 19,271 | 0.31 | +10% |
| AOD (AERONET) | 8,559 | 0.52 | -45% |

### Notes
- Pairing is the main bottleneck (600s) due to Dask `.compute()` calls during peak hours
- Scratch storage essential - campaign storage would be 3-4x slower
- Working branch: `develop`

---

## Session Summary (2026-01-23 Derecho continued)

### Key Accomplishments
1. **Added AirNow surface observations**:
   - Downloaded via `scripts/download_airnow.py` (no API key needed)
   - 36 sites across Asia (Bangkok network, Beijing, Guangzhou, Shenyang, Hanoi, Manila, Singapore, etc.)
   - Variables: PM2.5 (1008 values), O3 (152 values)
   - Data saved to scratch: `airnow_asiaq_2024-02-01_2024-02-29.nc`

2. **Created single-obs config workflow**:
   - Faster iteration by processing one obs dataset at a time
   - Avoids "democracy not monarchy" Dask problem (repeated model loading)
   - Each run loads model once, pairs quickly

3. **Reorganized config files**:
   - `asia-aq-derecho.yaml` - Full config (all obs: AirNow, AERONET, DC8)
   - `asia-aq-airnow-derecho.yaml` - AirNow only (PM2.5, O3)
   - `asia-aq-aeronet-derecho.yaml` - AERONET only (AOD)
   - `asia-aq-dc8-derecho.yaml` - DC8 only (O3, NO2, CO)
   - `asia-aq-gemini.yaml` - Mac testing
   - Deleted old `asia-aq.yaml`

### Files Modified
- `analyses/asia-aq/scripts/download_airnow.py` - Updated for Feb 29 (leap year)
- `analyses/asia-aq/configs/asia-aq-derecho.yaml` - Added AirNow, full config
- `analyses/asia-aq/configs/asia-aq-airnow-derecho.yaml` - New single-obs config
- `analyses/asia-aq/configs/asia-aq-aeronet-derecho.yaml` - New single-obs config
- `analyses/asia-aq/configs/asia-aq-dc8-derecho.yaml` - New single-obs config

### Workflow Recommendation
For faster iteration, run single-obs configs instead of full config:
```bash
davinci-monet run analyses/asia-aq/configs/asia-aq-airnow-derecho.yaml
davinci-monet run analyses/asia-aq/configs/asia-aq-aeronet-derecho.yaml
davinci-monet run analyses/asia-aq/configs/asia-aq-dc8-derecho.yaml
```
All output goes to same directory, plots accumulate.

---

## Session Summary (2026-01-23 Pandora)

### Key Accomplishments
1. **Added Pandora NO2 column observations**:
   - Created `preprocess_pandora.py` to read L2 txt files
   - 14 sites: Korea (Seoul, Busan, Incheon, etc.), Bangkok, Singapore, Philippines, Malaysia, Japan
   - 9,847 quality-filtered observations in Feb 2024
   - Output: `pandora_no2_column_20240201_20240229.nc`

2. **Computed CESM NO2 tropospheric column**:
   - Created `compute_no2_column.py` to integrate 3D NO2 vertically
   - Uses hybrid pressure coordinates with 200 hPa tropopause threshold
   - Output: `cesm_no2_column_20240201_20240229.nc` (1.25 GB)

3. **Created Pandora config**:
   - `asia-aq-pandora-derecho.yaml` for NO2 column evaluation
   - Uses separate model file (NO2 column, not 3D surface extraction)

### Statistics Results
| Variable | N | R | NMB |
|----------|-------|------|------|
| NO2 column (Pandora) | 8,886 | 0.57 | +60% |

Model overpredicts NO2 column, consistent with DC8 aircraft NO2 (+533% at surface).

### Files Created
- `analyses/asia-aq/scripts/preprocess_pandora.py`
- `analyses/asia-aq/scripts/compute_no2_column.py`
- `analyses/asia-aq/configs/asia-aq-pandora-derecho.yaml`

### Preprocessed Data (on scratch)
- `/glade/derecho/scratch/fillmore/ASIA-AQ/obs/pandora_no2_column_20240201_20240229.nc`
- `/glade/derecho/scratch/fillmore/ASIA-AQ/obs/cesm_no2_column_20240201_20240229.nc`

---

## Session Summary (2026-01-24) - Plot Refinements

### Status
**"Last 10% takes 90% of the time"** - In the visual refinement phase where each iteration requires human feedback on generated plots. Slower going but necessary for publication-quality output.

### Key Accomplishments
1. **Font size standardization**:
   - Increased base TextConfig: fontsize 14→16, title 16→18, tick 12→14
   - Flight timeseries uses 0.9x scaling for cleaner appearance
   - 3D track plots use 0.8x scaling (adjusted down from 1.5x after figure size reduction)

2. **Figure size reduction** (for better Preview.app fit):
   - Timeseries/Diurnal/Curtain: (14, 6) → (9, 4)
   - 3D Track: (12, 10) → (7, 6)
   - Scatter/Taylor: (10, 10) → (6, 6)
   - Spatial: (14, 8) → (8, 5)
   - Default/Boxplot/Scorecard: (12, 8) → (8, 5)
   - PDFs are vector graphics - quality preserved when zooming

3. **3D track plot fixes**:
   - Added labelpad to axis labels to prevent tick label overlap
   - Increased colorbar padding to prevent label clipping
   - Added tight_layout rect for right margin

4. **Pipeline improvements**:
   - Added time of day (HH:MM) to header display
   - Enabled slideshow preview in run_evaluation.py

5. **Git workflow documented** in CLAUDE.md:
   - Auto commit/push on develop
   - Wait for user verification before merge to main
   - Return to develop after merge

### Remaining Plot Refinements
- [ ] Review all plot types at new sizes for visual balance
- [ ] Check font readability across plot types
- [ ] Verify colorbar positioning on spatial plots
- [ ] Test multi-panel layouts (site_timeseries, flight_timeseries grid)

### Files Modified
- `davinci_monet/plots/base.py` - Font sizes, default figsize
- `davinci_monet/plots/renderers/*.py` - Figure sizes, font scaling
- `davinci_monet/plots/renderers/track_map_3d.py` - Label padding, font scale
- `davinci_monet/pipeline/runner.py` - Time display
- `analyses/asia-aq/scripts/run_evaluation.py` - Enable slideshow
- `CLAUDE.md` - Git workflow guidelines

---

## Session Summary (2026-01-24) - Transient NetCDF/Dask Error Fix

### Issue
Python would sometimes exit/crash right after plotting completes but before the preview countdown,
likely due to stale NetCDF file handles in Dask-backed datasets being garbage collected.

### Fix Applied
Added `_cleanup_context_datasets()` method to `PipelineRunner`:
- Called after all stages complete and log data is extracted, but **before** preview/exit
- Explicitly closes all model and observation datasets in `context.models` and `context.observations`
- Forces garbage collection and clears xarray's `FILE_CACHE`
- Does NOT clear the dictionaries (other code may still reference them)

This complements the existing `_cleanup_hdf5_state()` which runs at pipeline **start**.

### Files Modified
- `davinci_monet/pipeline/runner.py` - Added `_cleanup_context_datasets()` method and call in finally block
- `PERFORMANCE.md` - Updated "NetCDF File Handle Cleanup Errors" section with mitigation details

### If Errors Recur
Taking a wait-and-see approach. If transient crashes still occur, potential next steps:
- Add multiple `gc.collect()` passes (sometimes needed for cyclic references)
- Add a brief `time.sleep(0.1)` before preview to let file handles fully release
- Check if paired datasets in `context.pairs` also need explicit cleanup
- Profile to identify specific failure points

---

## Session Summary (2026-01-24) - File I/O Error Handling

### Issue
Rare Python tracebacks on file I/O errors could crash the pipeline. Needed comprehensive
try/except/raise blocks around file operations.

### Analysis
Scanned codebase for unprotected file I/O operations. Found 17 unprotected locations,
15 high-risk (user-facing or fallback paths).

### Fixes Applied

**CLI Commands** (`cli/commands/get_data.py`):
- Added `_write_dataset_safe()` helper with PermissionError/OSError handling
- Updated all 4 download commands (aeronet, airnow, aqs, openaq) to use safe writer
- User-friendly error messages on write failures

**Pipeline Stages** (`pipeline/stages.py`):
- Wrapped `xr.open_mfdataset()` and `xr.open_dataset()` in observation loading
- Raises `DataFormatError` with descriptive message on failure

**Model Readers** (`models/*.py`):
- cesm.py: Wrapped FV and SE `_open_with_xarray()` methods, plus SCRIP file loading
- cmaq.py: Wrapped `_open_with_xarray()` method
- wrfchem.py: Wrapped `_open_with_xarray()` method
- ufs.py: Wrapped `_open_with_xarray()` method

**Base Classes** (`models/base.py`):
- Wrapped file list reading in `resolve_files()` for .txt pattern files

**I/O Module** (`io/readers.py`):
- Wrapped file open in `_parse_icartt_basic()` ICARTT fallback parser

### Files Modified
- `davinci_monet/core/exceptions.py` - Added `write_error_log()` utility function
- `davinci_monet/cli/commands/get_data.py` - Safe write helper with error logging, updated 4 commands
- `davinci_monet/pipeline/stages.py` - Wrapped observation loading with error logging
- `davinci_monet/models/base.py` - Wrapped file list reading with error logging
- `davinci_monet/models/cesm.py` - Wrapped xarray calls (FV, SE, SCRIP) with error logging
- `davinci_monet/models/cmaq.py` - Wrapped xarray calls with error logging
- `davinci_monet/models/wrfchem.py` - Wrapped xarray calls with error logging
- `davinci_monet/models/ufs.py` - Wrapped xarray calls with error logging
- `davinci_monet/io/readers.py` - Wrapped ICARTT parser file open with error logging

### Error Log Format
On I/O errors, a timestamped log file is created in the `logs/` directory containing:
- Timestamp
- Context (what operation failed)
- Error type and message
- Full traceback

User-facing messages remain clean, with a reference to the error log file for debugging.

### Test Results
All 846 tests pass.

---

## Investigate: HDF5 Thread Safety Segfaults

### Issue
Intermittent HDF5 errors with segfaults when loading model data:
```
HDF5-DIAG: Error detected in HDF5 (1.14.6) thread 1:
  #000: H5A.c line 1866 in H5Aiterate2(): invalid location identifier
HDF5-DIAG: Error detected in HDF5 (1.14.6) thread 2:
  #000: H5G.c line 511 in H5Gget_create_plist(): not a group ID
zsh: segmentation fault
```

### Workaround
```bash
HDF5_USE_FILE_LOCKING=FALSE davinci-monet run config.yaml
```

If it persists:
```bash
DASK_NUM_WORKERS=1 HDF5_USE_FILE_LOCKING=FALSE davinci-monet run config.yaml
```

### Root Cause
HDF5 thread safety issue at C library level - crashes before Python exception handling.
The "thread 1/thread 2" messages indicate concurrent access problems.

### If Recurs Frequently
- [ ] Add `HDF5_USE_FILE_LOCKING=FALSE` to pipeline startup code
- [ ] Consider setting `h5py.File` with `locking=False` at module import
- [ ] Investigate Dask scheduler settings for single-threaded file I/O
- [ ] Document in DERECHO.md if specific to HPC environment

---

## Session Summary (2026-01-27) - Per-Site Timeseries Plots

### Key Accomplishments
1. **Added `per_site_timeseries` plot type**:
   - New `PerSiteTimeSeriesPlotter` generates one detailed figure per monitoring site
   - `plot_per_site()` generator yields `(site_id, figure)` tuples (same pattern as `plot_per_flight()`)
   - Each figure: obs scatter (black) + model line (NCAR blue), stats box (N, MB, RMSE, NMB, R)
   - Title with site name and coordinates, smart date formatting, y=0 baseline
   - Scale factor support for column data (e.g., Pandora NO2)
   - `sanitize_site_id()` utility for filename-safe site names

2. **Generalized pipeline per-entity splitting**:
   - Added `split_by_site` support in `PlottingStage` alongside existing `split_by_flight`
   - Output files: `site_{SiteName}_{index}_{plotname}.png`

3. **Registered in plotting system**:
   - Added to `TEMPORAL_PLOTS` frozenset in registry
   - Exported from `renderers/__init__.py` and `plots/__init__.py`

### YAML Configuration
```yaml
plots:
  o3_per_site:
    type: per_site_timeseries
    pairs: [cesm_airnow_o3]
    title: "O3: Model vs AirNow"
    split_by_site: true
    min_points: 50
    show_stats: true
```

### Test Results
- 21 new tests in `test_per_site_timeseries.py`
- 867 total tests passing (21 new + 846 existing)
- mypy: no new errors (only pre-existing matplotlib stub warnings)

### Files Created
- `davinci_monet/plots/renderers/per_site_timeseries.py` - New plotter
- `davinci_monet/tests/test_per_site_timeseries.py` - Tests

### Files Modified
- `davinci_monet/pipeline/stages.py` - `split_by_site` support
- `davinci_monet/plots/__init__.py` - Export new plotter
- `davinci_monet/plots/registry.py` - Add to `TEMPORAL_PLOTS`
- `davinci_monet/plots/renderers/__init__.py` - Export new plotter

---

## Session Summary (2026-02-03)

### Key Accomplishments

1. **Added altitude display to flight timeseries plots**:
   - Right y-axis shows aircraft altitude in km (or m)
   - Auto-detects altitude variables from ICARTT data
   - Converts feet to meters for DC-8 `Pressure_Altitude_BENNETT`, `GPS_Altitude_BENNETT`
   - Light gray line for altitude, distinct from obs/model data
   - Configurable: `show_altitude`, `altitude_var`, `altitude_units` parameters

2. **Fixed ICARTT altitude coordinate to use geometric altitude**:
   - Bug: ICARTT reader was putting `Static_Pressure_BENNETT` (hPa) into `altitude` coordinate
   - Symptom: Flight plots showed narrow altitude range (0.7-1.0 "km" was actually pressure!)
   - Fix: Reordered alias list to prefer geometric altitude over pressure
   - Priority: GPS altitude (m) > pressure altitude (ft, converted) > static pressure (last resort)

3. **Implemented 3D vertical interpolation for track pairing**:
   - Bug: Track pairing always extracted surface model values regardless of aircraft altitude
   - Symptom: No correlation between altitude and species concentrations in model-obs comparison
   - Fix: Interpolate 3D model field to aircraft altitude at each track point
   - New function `altitude_to_pressure()` converts aircraft altitude (m) to pressure (hPa)
   - Supports both nearest-neighbor and linear (log-pressure space) interpolation
   - Linear interpolation uses log-pressure for better atmospheric profile representation

4. **Added unit tests for vertical interpolation**:
   - `test_vertical_interpolation()` verifies O3 increases with altitude in model
   - Creates model with known vertical profile, verifies interpolation captures altitude dependence
   - 28 pairing tests now pass

### Statistics After Vertical Interpolation Fix

| Variable | N | R | NMB |
|----------|-------|------|------|
| O3 (DC8) | 3,248 | 0.42 | +44% |
| NO2 (DC8) | 3,255 | 0.67 | +28% |
| CO (DC8) | 3,244 | 0.77 | -27% |

### Files Modified
- `davinci_monet/plots/renderers/flight_timeseries.py` - Altitude display on right y-axis
- `davinci_monet/observations/aircraft/icartt.py` - Fixed altitude coordinate priority
- `davinci_monet/pairing/strategies/track.py` - 3D vertical interpolation
- `davinci_monet/tests/test_pairing.py` - Vertical interpolation test
- `davinci_monet/tests/test_plots.py` - Altitude display tests
- `CLAUDE.md` - Updated Git Workflow to never auto-commit

---

## Session Summary (2026-02-03 continued) - Per-Flight Statistics

### Key Accomplishments

1. **Implemented per-flight statistics output**:
   - New `_calculate_per_flight_stats()` method in `StatisticsStage`
   - Computes N, MO, MP, MB, RMSE, R, NMB_%, NME_% for each flight
   - Outputs `statistics_per_flight.csv` with one row per flight per variable
   - Enable via `per_flight: true` in stats config section

2. **Added warning for empty pairing results**:
   - Detects when pairing produces no data (transient HDF5/Dask issue)
   - Logs warning with suggested workaround: `DASK_NUM_WORKERS=1 HDF5_USE_FILE_LOCKING=FALSE`
   - Makes it visible why statistics/plotting stages are skipped

### Per-Flight Statistics Results (DC-8)

| Variable | Flight | N | MO | MP | MB | R |
|----------|--------|---|-----|-----|-----|-----|
| O3 | Feb 06 | 498 | 35 ppb | 49 ppb | +14 | 0.51 |
| O3 | Feb 13 | 492 | 35 ppb | 51 ppb | +17 | 0.73 |
| NO2 | Feb 17 | 441 | 2.0 ppb | 3.6 ppb | +1.6 | 0.75 |
| CO | Feb 26 | 448 | 262 ppb | 157 ppb | -104 | 0.87 |

Key patterns:
- **CO**: Model consistently low (-20 to -40% NMB), good correlations (R=0.53-0.87)
- **NO2**: Mixed bias, high variability between flights
- **O3**: Model consistently high (+27 to +54% NMB), moderate correlations

### Files Modified
- `davinci_monet/pipeline/stages.py` - Per-flight stats calculation and empty pairing warning
- `analyses/asia-aq/configs/asia-aq-dc8-gemini.yaml` - Added `per_flight: true`
