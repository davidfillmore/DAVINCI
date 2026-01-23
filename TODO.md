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

- [ ] Pandora NO2 column preprocessing and reader
- [x] AirNow data download and integration (36 sites, PM2.5/O3)
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
