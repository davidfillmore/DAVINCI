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

**Current workaround**: None - full file is loaded, then filtered at pairing.

**Proposed solutions**:
1. Add time filtering at `LoadObservationsStage` based on `analysis.start_time`/`end_time`
2. Create date-specific AERONET NetCDF subsets on Derecho
3. Use monetio API on Derecho (requires network access from compute nodes)

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
- [ ] Add time filtering to observation loading stage
- [ ] Document AERONET data preparation for both environments
- [ ] Consider unified config with environment-based path switching

---

## Priority 2: Performance Optimizations

### Completed (on `asia-aq-derecho`)
- [x] Dask parallel scheduler for pairing (260x speedup)
- [x] Scratch storage config for Derecho
- [x] Progress output fixes

### Remaining
- [x] Time filtering at observation load (avoid loading 5-month file for 3-day analysis)
  - File-level: filters by YYYYMMDD in filename (e.g., ICARTT files)
  - Data-level: filters by time dimension after loading (e.g., NetCDF)
  - Result: load_observations 163s -> 0.1s (1,630x faster)
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

### Performance Benchmarks (3-day test, 72 model files)

| Stage | Campaign | Scratch | Speedup |
|-------|----------|---------|---------|
| load_models | 190s | 6.8s | 28x |
| load_observations | 172s | 163s | ~1x |
| pairing | 10+ min | 2.3s | 260x |

See `DERECHO.md` for full environment setup and data paths.
