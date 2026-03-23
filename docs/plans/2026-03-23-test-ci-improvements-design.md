# Test & CI Improvements — Codex Review Response

**Date**: 2026-03-23
**Branch**: TBD (from develop)
**Trigger**: Codex code review of CI workflow and test suite

## Context

Codex reviewed the CI workflow, test suite, and CLI code. Five findings plus open
questions. This spec addresses all of them plus warning cleanup, organized as seven
workstreams.

Current state: 963 tests passing, 67% coverage, 694 warnings, CI has no coverage gate.

## Workstream 1: CI Coverage Enforcement

**Problem**: CI runs pytest without `--cov` or a threshold. The coverage config in
`pyproject.toml` exists but is unused. CI reports green regardless of coverage.

**Changes**:

`.github/workflows/ci.yml` — update the test step:
```yaml
- name: Run tests
  env:
    CI_ARTIFACTS_DIR: ${{ github.workspace }}/ci-artifacts
  run: |
    mkdir -p "$CI_ARTIFACTS_DIR"
    pytest davinci_monet/tests/ \
      -v --tb=short \
      --junitxml=test-results.xml \
      --cov=davinci_monet \
      --cov-report=term-missing \
      --cov-fail-under=70
```

`pyproject.toml` — drop 3.10 classifier (only 3.11+ is supported):
- Remove `Programming Language :: Python :: 3.10` from classifiers
- Update `requires-python` if it currently allows 3.10

## Workstream 2: CLI End-to-End Tests

**Problem**: No test exercises the real CLI→YAML→`load_config()`→pipeline path.
The integration tests pass Python dicts to `PipelineRunner.run_from_config()`,
which skips the YAML loading branch (`runner.py:1701-1704`).

**New file**: `davinci_monet/tests/test_cli_e2e.py`

Uses Typer's `CliRunner` to invoke the `app` with actual YAML config files
and synthetic NetCDF data.

**Tests**:

1. `test_cli_run_happy_path` — Write synthetic model + obs NetCDF to tmp_path.
   Write a YAML config referencing them. Invoke `["run", config_path]`. Assert:
   - exit code 0
   - "Analysis complete" in stdout
   - Output PNG and CSV files exist
   - Pipeline log exists

2. `test_cli_run_file_not_found` — Invoke with nonexistent YAML path. Assert:
   - exit code 2
   - "does not exist" in stdout

3. `test_cli_run_config_error` — Write a YAML with invalid dates. Invoke `run`.
   Assert:
   - exit code 1
   - "Configuration Error" or "Validation" in stdout

**Also**: Fix the `test_integration.py` docstring (lines 1-5) to say "pipeline
integration" rather than "the same path a user takes with `davinci-monet run`".

**Synthetic data**: Reuse `synthetic.generators.Domain`, `TimeConfig`, and
`PerfectMatchScenario` from the existing integration test infrastructure.

## Workstream 3: Error Config Parametrized Tests

**Problem**: 15 curated error configs in `tests/error_configs/` are documented
but never run by CI.

**Location**: `davinci_monet/tests/test_cli_e2e.py` (same new file)

**Approach**: Parametrize over error config files using the CLI `validate` command.

```python
ERROR_CONFIGS = sorted(Path("tests/error_configs").glob("*.yaml"))

@pytest.mark.parametrize("config_path", ERROR_CONFIGS, ids=lambda p: p.stem)
def test_error_config_rejected(config_path):
    runner = CliRunner()
    result = runner.invoke(app, ["validate", str(config_path)])
    assert result.exit_code != 0 or "error" in result.stdout.lower() or "failed" in result.stdout.lower()
```

For configs that are YAML-malformed (01, 02), `validate` may fail at parse time.
For others, it should fail at schema validation. Both produce non-zero exit or
error text — the assertion covers both.

## Workstream 4: Fix Weak CLI Assertions

**Problem**: Several CLI tests have no assertions or overly permissive `or` patterns
that pass on any failure.

**File**: `davinci_monet/tests/test_cli.py`

**Fixes**:

1. `test_validate_strict_mode` (line 249-272) — Add assertion:
   ```python
   assert "Validation passed" in result.stdout
   ```
   Extra fields are ignored by Pydantic's default mode, so this should pass.

2. `test_run_valid_config_parses` (line 175-185) — The config points to
   nonexistent data files, so the pipeline fails. Tighten to:
   ```python
   assert result.exit_code != 0
   # Should fail on missing data, not on config parsing
   assert "Error" in result.stdout or "not found" in result.stdout.lower()
   ```

3. `test_validate_invalid_config` (line 285-292) — Remove `or` fallback:
   ```python
   assert "Validation failed" in result.stdout or "error" in result.stdout.lower()
   ```

4. `test_validate_complete_config` (line 540-548) — Remove `or` fallback:
   ```python
   assert "Validation passed" in result.stdout
   ```

## Workstream 5: SpatialOverlayPlotter Behavioral Test

**Problem**: `SpatialOverlayPlotter` is registered but has no behavioral test.
15% coverage.

**File**: `davinci_monet/tests/test_plots.py` (add to existing spatial test section)

**Test**: Create a synthetic paired dataset with:
- 2D model field (lat × lon grid with values)
- Observation points with lat/lon/values

Call the plotter's `plot()` method. Assert:
- Returns a matplotlib Figure
- Figure has at least one Axes (the map)
- The axes contain both a contour/pcolormesh (model) and scatter (obs) layer

Pattern matches existing spatial plotter tests in the same file.

## Workstream 6: Stats and Plotter Coverage

**Problem**: Several modules have medium coverage gaps from untested branches.

### Stats (`davinci_monet/stats/`)

**New file**: `davinci_monet/tests/test_stats_coverage.py`

Parametrized tests for edge cases in `metrics.py` and `calculator.py`:
- Empty arrays → should return NaN or 0 for count
- Single-point arrays → R should be NaN, MB/RMSE should work
- All-NaN arrays → all metrics return NaN
- All-identical values → standard deviation 0, R undefined

### Plotters

**File**: `davinci_monet/tests/test_plots.py` (add to existing sections)

- `timeseries.py` (48%): Test aggregate mode (`aggregate_dim` parameter),
  multi-pair overlay, and individual-sites mode
- `scorecard.py` (61%): Test multi-variable scorecard generation
- `track_map_3d.py` (53%): Test with multi-flight track data, color-by-altitude
  mode

All use existing synthetic data infrastructure — no external files.

## Workstream 7: Warning Cleanup

**Problem**: 694 warnings bury new signal. Breakdown:
- ~670: matplotlib/pyparsing DeprecationWarnings (third-party, unfixable)
- 8: xarray `Dataset.dims` deprecation (our code)
- ~10: numpy RuntimeWarnings (expected stats edge cases)
- 1: numpy ndarray binary compat (conda artifact)
- 1: unclosed file handle (our code)

### Fix in our code

**xarray `.dims` mapping access** — change to `.sizes`:
- `davinci_monet/observations/base.py:474`: `int(self.data.dims[dim])` → `int(self.data.sizes[dim])`
- `davinci_monet/observations/base.py:482`: `int(self.data.dims["time"])` → `int(self.data.sizes["time"])`
- `davinci_monet/plots/renderers/timeseries.py:130`: `len(obs_data.dims)` → `len(obs_data.sizes)`
- `davinci_monet/plots/renderers/timeseries.py:132`: `d for d in obs_data.dims` → `d for d in obs_data.sizes`

Note: `in ds.dims` membership checks are fine — both mapping and set support `in`.

**Unclosed file handle** — investigate during implementation. Likely a log file
opened without `with` statement in the pipeline logger.

### Suppress third-party noise in `pyproject.toml`

```toml
[tool.pytest.ini_options]
filterwarnings = [
    "error::UserWarning",
    "ignore::DeprecationWarning:matplotlib",
    "ignore::DeprecationWarning:pyparsing",
    "ignore:numpy.ndarray size changed:RuntimeWarning",
    "ignore:Polyfit may be poorly conditioned:numpy.RankWarning",
    "ignore::RuntimeWarning:numpy.lib",
    "ignore:The return type of `Dataset.dims`:FutureWarning:xarray",
]
```

The `error::UserWarning` line promotes our own warnings to errors so new warnings
from our code are caught immediately. Third-party DeprecationWarnings and expected
numpy RuntimeWarnings are suppressed. The xarray FutureWarning filter is a safety
net until all `.dims` usages are migrated.

## Out of Scope

- Observation/model reader I/O tests (HDF4, ICARTT, etc.) — requires large format-specific fixtures
- `get_data.py` command body tests — requires mocked monetio downloads
- Python 3.10 support — confirmed not a target

## Files Summary

**New files**:
- `davinci_monet/tests/test_cli_e2e.py` — CLI end-to-end + error config tests
- `davinci_monet/tests/test_stats_coverage.py` — stats edge case tests

**Modified files**:
- `.github/workflows/ci.yml` — add coverage flags and threshold
- `pyproject.toml` — drop 3.10 classifier, add filterwarnings
- `davinci_monet/tests/test_cli.py` — fix weak assertions
- `davinci_monet/tests/test_integration.py` — fix docstring
- `davinci_monet/tests/test_plots.py` — add overlay, timeseries, scorecard, track_map_3d tests
- `davinci_monet/observations/base.py` — `.dims[x]` → `.sizes[x]`
- `davinci_monet/plots/renderers/timeseries.py` — `.dims` → `.sizes` for len/iteration

## Success Criteria

- All existing 963 tests still pass
- New tests pass
- CI enforces `--cov-fail-under=70`
- 0 warnings in test output
- Error configs are exercised in CI
- CLI YAML→pipeline path is tested end-to-end
