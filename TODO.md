# TODO - DAVINCI-MONET

Priority items for future development sessions.

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

### Issues to Investigate
- [ ] **AERONET pairing suspiciously fast** - Nearly instantaneous pairing may indicate:
  - No valid time overlap between model and filtered observations
  - Time filtering too aggressive (check `_filter_by_time` slice bounds)
  - Empty paired dataset being generated
  - Verify output plots/stats have actual data points

- [ ] **Time filtering not applied?** - load_observations took 163.2s (expected 0.1s with filtering)
  - Check if time filtering code path is being reached
  - Verify AERONET NetCDF has time dimension with expected name
  - May need to add debug logging to `_filter_by_time()`
