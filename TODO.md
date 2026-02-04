# TODO - DAVINCI-MONET

Updated: 2026-02-04

This file tracks active work items. For deep performance details, see `PERFORMANCE.md`.

## Priority 1: Performance and Reliability

- Pre-chunk model data (daily/weekly concatenated files) to reduce repeated Dask `.compute()` I/O.
- Optional model precompute / preload (e.g., `preload: true`) to load Dask-backed models once before pairing.
  Trade-off: high memory usage (~20 GB for 1 month CESM at 1°).
- Investigate HDF5 thread-safety segfaults and decide whether to set
  `HDF5_USE_FILE_LOCKING=FALSE` automatically or document per-environment guidance.

## Priority 2: Feature Additions

- MOPITT CO profile evaluation.
- MODIS AOD comparison.

## Priority 3: Testing and Workflow

- Create `asia-aq-1week.yaml` for fast end-to-end runs (target <2 minutes).
- Add a minimal regression config for CI (optional).

## Notes

- `ASIA_AQ_DATA` must be set for ASIA-AQ configs:
  `export ASIA_AQ_DATA=~/Data/ASIA-AQ`
