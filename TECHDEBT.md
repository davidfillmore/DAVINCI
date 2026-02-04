# Technical Debt Tracker

This document tracks technical debt items in the DAVINCI-MONET codebase.

## Priority Legend
- **P0**: Critical - Security or correctness issue
- **P1**: High - Should address soon
- **P2**: Medium - Address when convenient
- **P3**: Low - Nice to have

---

## P0: Critical

### 1. Replace `eval()` with safe expression evaluator
**Location**: `davinci_monet/models/base.py:407`

```python
result = eval(expression, {"__builtins__": {}}, local_vars)
```

The `apply_expression()` method uses `eval()` for user-provided expressions. While `__builtins__` is restricted, sophisticated attacks are still possible.

**Solution**: Replace with `numexpr`, `asteval`, or native xarray computation methods.

---

## P1: High

*No P1 items currently open.*

---

## P2: Medium

### 1. Silent exception handling masks errors
**Locations**:
- `davinci_monet/stats/calculator.py:417-418, 555` - Catches all exceptions, returns NaN
- `davinci_monet/observations/satellite/generic_l3.py:130` - Silent file open failures
- `davinci_monet/pairing/engine.py:282` - Bare `except KeyError: pass`

**Solution**: Use specific exception types, log warnings with details.

### 2. Global warning suppression
**Location**: `analyses/asia-aq/scripts/download_observations.py:23`

```python
warnings.filterwarnings("ignore")
```

**Solution**: Use context managers for targeted suppression.

### 3. Type ignore comments (38 instances)
**Locations**: Various files with `# type: ignore[arg-type]`

**Solution**: Investigate and fix underlying type mismatches where possible.

---

## P3: Low

### 1. Document deprecated feature removal timeline
**Locations**:
- `davinci_monet/observations/satellite/goes_l3_aod.py:309-315` - `open_goes` deprecated
- `davinci_monet/models/ufs.py:154-160` - Deprecated aliases

**Solution**: Add version number for planned removal to deprecation warnings.

### 2. Add progress bars for file I/O
**Location**: `davinci_monet/observations/satellite/generic_l3.py:125-140`

When loading many files, no indication of progress.

**Solution**: Add `tqdm` progress bars for file loading loops.

### 3. Analysis script documentation
**Location**: `analyses/asia-aq/scripts/`

Scripts lack clear documentation of:
- Expected data locations
- Failure modes and recovery
- Performance characteristics

### 4. Profile array operations for large datasets
**Observation**: 50 instances of `.flatten()` or `.ravel()` calls (as of 2026-02-04)

These create copies and may be inefficient for large satellite datasets.

**Solution**: Profile with real-world datasets, optimize hot paths.

### 5. Improve debug logging in data loaders
Model and observation readers should log:
- Files being loaded
- Variable mappings applied
- Unit conversions performed

---

## Completed

### Use absolute paths for data directories
**Location**: `analyses/asia-aq/scripts/`, `analyses/asia-aq/configs/`

- Added `ASIA_AQ_DATA` environment variable support with fallback to `~/Data/ASIA-AQ`
- Updated YAML config to use `${ASIA_AQ_DATA}` for model paths (config parser expands env vars)
- Updated scripts: `run_evaluation.py`, `explore_model.py`, `download_observations.py`

**Completed**: 2026-01-11

### Add STDOUT logging for pipeline progress
**Location**: `davinci_monet/pipeline/runner.py`

- Added `tqdm` dependency to `environment.yml`
- Added `show_progress` parameter to `PipelineRunner` (default: True)
- Pipeline now displays progress bar and stage timing to stdout
- Added `progress_callback` to `PipelineContext` for sub-stage progress (each model/obs/pair)
- Updated `PipelineBuilder` and `run_analysis()` to support `show_progress`
- Added `log_dir` config option with timestamped log files (`pipeline_YYYYMMDD_HHMMSS.log`)
- Log files capture all progress output even if stdout is interrupted (BrokenPipeError handling)

**Completed**: 2026-01-11

### Hardcoded user paths in analysis scripts
**Location**: `analyses/asia-aq/scripts/`, `analyses/asia-aq/configs/`

Resolved as part of P1 #2 above - all paths now use `ASIA_AQ_DATA` env var.

**Completed**: 2026-01-11

### Update synthetic data plot examples
**Location**: `examples/all_plot_types.py`

Added all 13 plot types with synthetic data:
- [x] `site_timeseries` - Site-by-site time series panels
- [x] `flight_timeseries` - Flight-by-flight time series panels
- [x] `track_map_3d` - 3D flight track visualization

**Completed**: 2026-01-11

---

## Notes

- Test count: 875 (last full run: 2026-02-04)
- Last updated: 2026-02-04
