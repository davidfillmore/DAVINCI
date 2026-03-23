# Performance Analysis: DAVINCI Pipeline

This document captures performance insights discovered during development, particularly around Dask lazy loading and pairing bottlenecks.

## Dask Lazy Loading and Pairing Performance

**Date**: 2026-01-23
**Updated**: 2026-02-04 (pairing concurrency defaults and tuning guidance)
**Analysis**: ASIA-AQ evaluation with CESM/CAM-chem model

### The Problem

Pairing stage takes ~60s with parallel Dask pairs enabled (~72s serial) even though actual pairing computations are fast (<1s each). The bottleneck is **Dask lazy evaluation** - model data isn't loaded until `.compute()` is called during pairing.

### Model Loading Behavior

| Model | Files | Load Time | What Actually Happens |
|-------|-------|-----------|----------------------|
| `cesm_asiaq` | 696 NetCDF | 10.9s | Creates Dask task graph only (lazy) |
| `cesm_no2_column` | 1 NetCDF | 0.3s | Loads data into memory (eager) |

The 10.9s for `cesm_asiaq` is just building the task graph - no actual file I/O occurs yet.

### Pairing Timing Breakdown

```
Pair                        Time     Points   Notes
─────────────────────────────────────────────────────
cesm_asiaq_airnow          24.5s        29   Triggers .compute() → loads 696 files
cesm_asiaq_aeronet         24.8s     2,904   .compute() again → reprocesses files
cesm_asiaq_dc8             22.3s     3,269   .compute() again → reprocesses files
cesm_no2_column_pandora     0.0s     9,844   Data already in memory
```

**Key insight**: The ~24s per Dask pair is almost entirely file I/O, not pairing computation. The number of paired points (29 vs 3,269) has minimal impact on timing.

### Why Each Pair Triggers Separate Compute

1. Each pair extracts different spatial subsets of the model
2. The pairing engine calls `.compute()` to get actual values for interpolation
3. Dask doesn't cache the full computed result between calls
4. Result: 696 files processed 3 times for 3 pairs

### Current Concurrency Controls (Default = Safe/Serial for Dask)

The pipeline separates Dask-backed pairs from eager pairs and runs them in two phases. Defaults prioritize stability because each Dask-backed pair can trigger a full `.compute()` and re-read files.

Config keys in `pairing`:
- `dask_pair_workers`: number of Dask-backed pairs to run concurrently. Default `1`.
- `dask_num_workers`: threads for the Dask scheduler inside each pair. If unset, derived from CPU count and capped by RAM (<=16 GB → 4, <=32 GB → 6, else up to 32).
- `max_workers`: thread count for eager (non-Dask) pairs. Default ~CPU/2 with a low-RAM cap.

Example safe config for 16 GB laptops:
```yaml
pairing:
  dask_pair_workers: 1
  dask_num_workers: 4
  max_workers: 4
```

Increase `dask_pair_workers` only if the model fits comfortably in memory and storage bandwidth is high; otherwise it multiplies file reads and can slow overall wall time.

### Common Misconception: "Parallel Dask Pairs Share Data"

This is the main reason the default is **serial** for Dask-backed pairs.

**What you might expect:**
```
1. Load model data once → shared in memory
2. All 3 pairs use the shared data in parallel → each pairs in <1s
```

**What actually happens (if you enable parallel Dask pairs):**
```
1. Multiple Dask pairs start in parallel (`dask_pair_workers > 1`)
2. Each pair independently calls .compute() on the Dask model
3. Each .compute() triggers loading/processing of 696 files
4. Dask's scheduler coordinates somewhat, but computed data isn't shared
```

**Evidence from timing:**
```
Hypothetical (if computed data were shared):
  Pair 1: ~24s (loads and computes data into memory)
  Pair 2:  <1s (reuses in-memory arrays)
  Pair 3:  <1s (reuses in-memory arrays)
  Total:  ~25s

Actual observed timing (parallel Dask pairs enabled):
  cesm_asiaq_airnow:   24.5s
  cesm_asiaq_aeronet:  24.8s
  cesm_asiaq_dc8:      22.3s
  Total: ~60s (parallel), ~72s (if sequential)
```

The ~60s total (not ~25s) proves that pairs do **not** share computed NumPy arrays. Each pair does its own full `.compute()`.

With the default `dask_pair_workers=1`, you avoid redundant *simultaneous* I/O, but each pair still performs its own compute.

The 60s vs 72s difference (~17% savings) comes from lower-level caching:
- **OS file cache**: Files read by first thread are cached in RAM for others
- **Dask chunk cache**: Overlapping spatial regions may be computed once
- **I/O overlap**: Threads can read files while others compute

This is incidental caching, not intentional data sharing.

### Democracy, Not Monarchy

With parallel Dask pairs enabled, the nearly equal per-pair times (~22-25s each) show that labor is distributed equally - each pair does roughly the same amount of work independently. If one thread were doing the heavy lifting for others, we'd expect:

```
Pair 1: 40s  (does most loading, others benefit from cache)
Pair 2: 15s  (benefits from warm cache)
Pair 3:  8s  (benefits most from cache)
```

But we see equal times instead. This is actually **worse** than a "monarchy" model - at least then we'd get ~25s total. Instead, we have 3 equal workers redundantly doing the same job, with only minor (~17%) incidental cache benefits spread evenly across all three.

**This is why pre-computing would help**: forcing `.compute()` once before pairing would put the data in memory as NumPy arrays, which all pairs could then share.

### Current Architecture

By default, Dask-backed pairs run one at a time, but each pair still triggers its own `.compute()`.

```
load_models:
  cesm_asiaq → Dask Dataset (lazy, task graph only)

pairing:
  pair1: cesm_asiaq[subset1].compute() → loads files → pair → result1
  pair2: cesm_asiaq[subset2].compute() → loads files → pair → result2
  pair3: cesm_asiaq[subset3].compute() → loads files → pair → result3
```

### Potential Optimization: Pre-compute Dask Models (Not Implemented)

```
load_models:
  cesm_asiaq → Dask Dataset (lazy)
  cesm_asiaq.compute() → NumPy Dataset (in memory)  ← NEW STEP

pairing:
  pair1: cesm_asiaq_computed[subset1] → pair → result1  (~0.5s)
  pair2: cesm_asiaq_computed[subset2] → pair → result2  (~0.5s)
  pair3: cesm_asiaq_computed[subset3] → pair → result3  (~0.5s)
```

**Estimated improvement**: 60s → ~25s (load once) + ~1.5s (all pairings) = ~27s total

### Trade-offs

| Approach | Time | Memory | Pros | Cons |
|----------|------|--------|------|------|
| Current (lazy, serial Dask pairs) | ~72s | Low during load | Memory efficient, stable defaults | Slow for multiple pairs |
| Pre-compute | ~27s | High (full model) | Fast pairing | Needs RAM for full model |
| Chunked files | ~30s | Medium | Balance | Requires preprocessing |

### Memory Estimates (ASIA-AQ)

From pipeline logs:
- `cesm_asiaq`: 5 vars × 696 times × ~29M grid points = **~20 GB** if fully loaded
- `cesm_no2_column`: 1 var × 696 times × ~225K grid points = **~0.6 GB**

Pre-computing `cesm_asiaq` requires sufficient RAM. On HPC (Derecho) this is fine; on laptops it may cause swapping.

### Potential Enhancements (Not Implemented)

1. **Automatic pre-compute**: Detect Dask-backed models with multiple pairs, compute once
2. **Config flag**: Add `precompute: true` option in model config
3. **Chunked intermediate files**: Pre-process model to daily/weekly chunks

### Related Issues

- **GIL contention**: Fixed via two-phase pairing (Dask pairs first, then eager pairs)
- **Progress display**: Fixed via completion-based tracking with 1.0s visibility delay per pair
- **Time filtering**: Implemented at observation load (1,630x speedup)

## Other Performance Findings

### Observation Loading

| Optimization | Before | After | Speedup |
|--------------|--------|-------|---------|
| ICARTT file filtering by date | 163s | 0.1s | 1,630x |
| AERONET time slice | 163s | 0.1s | 1,630x |

### Pairing Strategies

| Strategy | Use Case | Performance Notes |
|----------|----------|-------------------|
| Point | Surface sites | Fast, O(sites × times) |
| Track | Aircraft | 3D interp, O(track_points × model_times) |
| Profile | Vertical profiles | Similar to track |
| Swath | Satellite pixels | Can be large, benefits from chunking |
| Grid | Gridded obs | Direct alignment, very fast |

## Profiling Commands

```python
# Time individual pairing operations
import time
from davinci_monet.pairing import PairingEngine, PairingConfig

engine = PairingEngine()
t0 = time.time()
result = engine.pair(model_ds, obs_ds, obs_vars=['var'], model_vars=['var'])
print(f"Pairing took {time.time() - t0:.1f}s")

# Check if dataset is Dask-backed
def is_dask_backed(ds):
    return any(ds[v].chunks is not None for v in ds.data_vars)
```

## Transient NetCDF Cleanup Warnings (Rare)

Occasionally you may still see `NetCDF: Not a valid ID` during cleanup after a failure. This should be rare now because the pipeline clears HDF5/NetCDF state on startup and closes datasets on shutdown. Treat these as cleanup noise and focus on the stage error that caused the failure.

## References

- Dask documentation: https://docs.dask.org/en/stable/array-best-practices.html
- xarray + Dask: https://docs.xarray.dev/en/stable/user-guide/dask.html
