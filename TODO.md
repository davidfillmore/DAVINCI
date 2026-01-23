# TODO - DAVINCI-MONET

Priority items for future development sessions.

## Priority 0: AERONET NetCDF Reader (Current Session)

### Objective
Get `asia-aq-derecho` branch working on Mac with the new AERONET NetCDF format.

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

## Priority 1: Branch Reconciliation (ASIA-AQ)

### Branch Status
- `asia-aq`: Mac development branch (at `33b3fa3`)
- `asia-aq-derecho`: Derecho HPC branch (7 commits ahead)
- **No divergence** - `asia-aq-derecho` is strictly ahead, can fast-forward merge

### AERONET Data Format Differences

| Environment | Format | Source | File |
|-------------|--------|--------|------|
| Mac | CSV (date-filtered) | monetio API download | Per-analysis date range |
| Derecho | NetCDF (5 months) | Pre-processed file | `AERONET_L15_20240101_20240501.nc` (~1GB) |

**Issue**: Loading full 5-month file takes ~163s even for 3-day analysis.

**SOLVED**: Time filtering implemented at `LoadObservationsStage`:
- File-level: filters ICARTT files by YYYYMMDD in filename
- Data-level: filters NetCDF by time dimension using `xr.sel(time=slice(...))`
- Result: 163s → 0.1s (1,630x faster)

### CESM Reader vs Generic Reader

| Environment | Reader | Config `mod_type` |
|-------------|--------|-------------------|
| Mac | `cesm_fv` (via monetio) | `cesm_fv` |
| Derecho | `generic` (xarray only) | `generic` |

**Root cause**: The regridded CESM files have mixed dimensionality:
- `AODVISdn`: 2D variable (time, lat, lon) - column-integrated AOD
- `O3`, `NO2`, `CO`: 4D variables (time, lev, lat, lon)

The `cesm_fv` reader via monetio expects consistent vertical structure and fails
or produces unexpected results with mixed 2D/3D variables in the same file.

**Current workaround** (Derecho): Use `generic` reader which handles mixed dims.

**Proposed solutions**:
1. Update `cesm_fv` reader to handle 2D column variables gracefully
2. Add `skip_vertical_standardization` option for 2D variables
3. Document when to use `generic` vs `cesm_fv` based on file contents

### Files to Reconcile

```
# Configs using different approaches:
analyses/asia-aq/configs/asia-aq.yaml           # Mac: cesm_fv, env vars
analyses/asia-aq/configs/asia-aq-derecho.yaml   # Derecho: generic, hardcoded paths
analyses/asia-aq/configs/asia-aq-scratch.yaml   # Derecho: generic, scratch storage
```

### Action Items

- [ ] Merge `asia-aq-derecho` performance fixes back to `asia-aq`
- [ ] Test `generic` reader on Mac with local CESM files
- [x] Add time filtering to observation loading stage
- [ ] Document AERONET data preparation for both environments
- [ ] Consider unified config with environment-based path switching

---

## Priority 2: Performance Optimizations

### Completed (on `asia-aq-derecho`)
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

---

## Priority 3: Feature Additions

- [ ] Pandora NO2 column preprocessing and reader
- [ ] AirNow data download script for Derecho
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

## Priority 4: Testing Configuration

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

### Files Modified
- `davinci_monet/observations/surface/aeronet.py` - Standardize NetCDF dimensions
- `davinci_monet/pipeline/stages.py` - Use AERONET reader in observation loading
- `analyses/asia-aq/configs/asia-aq-gemini.yaml` - New Mac config
