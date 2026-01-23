# Performance Analysis: DAVINCI-MONET Pipeline

This document captures performance insights discovered during development, particularly around Dask lazy loading and pairing bottlenecks.

## Dask Lazy Loading and Pairing Performance

**Date**: 2026-01-23
**Analysis**: ASIA-AQ evaluation with CESM/CAM-chem model

### The Problem

Pairing stage takes ~60s even though actual pairing computations are fast (<1s each). The bottleneck is **Dask lazy evaluation** - model data isn't loaded until `.compute()` is called during pairing.

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

### Common Misconception: "Parallel Dask Pairs Share Data"

**What you might expect:**
```
1. Load model data once → shared in memory
2. All 3 pairs use the shared data in parallel → each pairs in <1s
```

**What actually happens:**
```
1. All 3 Dask pairs start in parallel (via ThreadPoolExecutor)
2. Each pair independently calls .compute() on the Dask model
3. Each .compute() triggers loading/processing of 696 files
4. Dask's scheduler coordinates somewhat, but computed data isn't shared
```

**Evidence from timing:**
```
If data were shared after first compute:
  Pair 1: ~24s (loads and computes data)
  Pair 2:  <1s (reuses computed data)
  Pair 3:  <1s (reuses computed data)
  Total:  ~25s

Actual observed timing:
  Pair 1: 24.5s
  Pair 2: 24.8s
  Pair 3: 22.3s
  Total: ~60s (parallel), ~72s (sequential)
```

The ~60s total (not ~25s) proves that pairs do **not** share computed data. Each pair does its own full `.compute()`. The 60s vs 72s difference comes from Dask's internal scheduler overlapping some I/O operations, not from sharing computed results.

**This is why pre-computing would help**: forcing `.compute()` once before pairing would put the data in memory as NumPy arrays, which all pairs could then share.

### Current Architecture

```
load_models:
  cesm_asiaq → Dask Dataset (lazy, task graph only)

pairing:
  pair1: cesm_asiaq[subset1].compute() → loads files → pair → result1
  pair2: cesm_asiaq[subset2].compute() → loads files → pair → result2
  pair3: cesm_asiaq[subset3].compute() → loads files → pair → result3
```

### Proposed Optimization: Pre-compute Dask Models

```
load_models:
  cesm_asiaq → Dask Dataset (lazy)
  cesm_asiaq.compute() → NumPy Dataset (in memory)  ← NEW STEP

pairing:
  pair1: cesm_asiaq_computed[subset1] → pair → result1  (~0.5s)
  pair2: cesm_asiaq_computed[subset2] → pair → result2  (~0.5s)
  pair3: cesm_asiaq_computed[subset3] → pair → result3  (~0.5s)
```

**Expected improvement**: 60s → ~25s (load once) + ~1.5s (all pairings) = ~27s total

### Trade-offs

| Approach | Time | Memory | Pros | Cons |
|----------|------|--------|------|------|
| Current (lazy) | ~60s | Low during load | Memory efficient | Slow for multiple pairs |
| Pre-compute | ~27s | High (full model) | Fast pairing | Needs RAM for full model |
| Chunked files | ~30s | Medium | Balance | Requires preprocessing |

### Memory Estimates (ASIA-AQ)

From pipeline logs:
- `cesm_asiaq`: 5 vars × 696 times × ~29M grid points = **~20 GB** if fully loaded
- `cesm_no2_column`: 1 var × 696 times × ~225K grid points = **~0.6 GB**

Pre-computing `cesm_asiaq` requires sufficient RAM. On HPC (Derecho) this is fine; on laptops it may cause swapping.

### Implementation Options

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

## References

- Dask documentation: https://docs.dask.org/en/stable/array-best-practices.html
- xarray + Dask: https://docs.xarray.dev/en/stable/user-guide/dask.html
