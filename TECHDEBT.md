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

### 2. Use absolute paths for data directories
**Location**: `analyses/asia-aq/scripts/run_evaluation.py`

Scripts use paths relative to working directory, so they fail if run from a different directory.

**Solution**:
- Use `Path(__file__).parent` to resolve paths relative to script location
- Or use environment variables (`$ASIA_AQ_DATA`, `$DAVINCI_DATA`)

### 3. Add STDOUT logging for pipeline progress
**Location**: `davinci_monet/pipeline/runner.py`

Users have no visibility into pipeline progress during long runs. Should show:
- Current stage being executed
- Progress within stages (e.g., "Loading file 5/28...")
- Elapsed time per stage

**Solution**: Add `tqdm` progress bars and structured logging to stdout.

### 4. Hardcoded user paths in analysis scripts
**Locations**:
- `analyses/asia-aq/scripts/run_evaluation.py:86` - `/Users/fillmore/Data/ASIA-AQ/...`
- `analyses/asia-aq/misc/explore_model.py:15` - `Path.home() / "Data" / "ASIA-AQ"`

**Solution**: Use config file or environment variables for data paths.

---

## P2: Medium

### 5. Update synthetic data plot examples
**Location**: `examples/`

Example scripts need updating to include new plot types:
- [ ] `site_timeseries` - Site-by-site time series panels
- [ ] `flight_timeseries` - Flight-by-flight time series panels
- [ ] `track_map_3d` - 3D flight track visualization

### 6. Create generic `run_analysis` CLI command
**Location**: `davinci_monet/cli/`

Currently `davinci-monet run` exists, but analyses have custom `run_evaluation.py` scripts. Consider:
- Single entry point that takes YAML config path
- Handles data path resolution
- Reports progress to stdout

### 7. Silent exception handling masks errors
**Locations**:
- `davinci_monet/stats/calculator.py:417-418, 555` - Catches all exceptions, returns NaN
- `davinci_monet/observations/satellite/generic_l3.py:130` - Silent file open failures
- `davinci_monet/pairing/engine.py:251` - Bare `except KeyError: pass`

**Solution**: Use specific exception types, log warnings with details.

### 8. Global warning suppression
**Location**: `analyses/asia-aq/scripts/download_observations.py:21`

```python
warnings.filterwarnings("ignore")
```

**Solution**: Use context managers for targeted suppression.

### 9. Type ignore comments (11+ instances)
**Locations**: Various files with `# type: ignore[arg-type]`

**Solution**: Investigate and fix underlying type mismatches where possible.

---

## P3: Low

### 10. Document deprecated feature removal timeline
**Locations**:
- `davinci_monet/observations/satellite/goes_l3_aod.py:309-315` - `open_goes` deprecated
- `davinci_monet/models/ufs.py:154-160` - Deprecated aliases

**Solution**: Add version number for planned removal to deprecation warnings.

### 11. Add progress bars for file I/O
**Location**: `davinci_monet/observations/satellite/generic_l3.py:125-140`

When loading many files, no indication of progress.

**Solution**: Add `tqdm` progress bars for file loading loops.

### 12. Analysis script documentation
**Location**: `analyses/asia-aq/scripts/`

Scripts lack clear documentation of:
- Expected data locations
- Failure modes and recovery
- Performance characteristics

### 13. Profile array operations for large datasets
**Observation**: 46 instances of `.flatten()` or `.ravel()` calls

These create copies and may be inefficient for large satellite datasets.

**Solution**: Profile with real-world datasets, optimize hot paths.

### 14. Improve debug logging in data loaders
Model and observation readers should log:
- Files being loaded
- Variable mappings applied
- Unit conversions performed

---

## Completed

_Move items here when resolved._

---

## Notes

- Test count: 792+ (all passing)
- Last updated: 2024-02-29
