# DAVINCI-MONET End-to-End Code Review

## Codex Review (2026-02-16)

Reviewer: Codex (GPT-5)

### Scope and Method
- Reviewed repository structure and key runtime paths end-to-end (`config`, `pipeline`, `pairing`, `stats`, `io`, `cli`, plus examples/docs and tests).
- Performed static review with line-level inspection and cross-checked schema/config/examples against runtime stage behavior.
- Attempted full test run, but execution is currently blocked in this environment by missing runtime dependencies (`xarray` not installed).

### Verification Limits
- `pytest -q` fails immediately with `ModuleNotFoundError: No module named 'xarray'` from `davinci_monet/tests/conftest.py`.
- Findings below are based on code inspection and consistency analysis, not full runtime execution in this environment.

---

## Claude Review Pass (2026-02-15)

Reviewer: Claude Opus 4.6

### Scope and Method
- Verified all 11 Codex findings against source code with line-level inspection.
- Performed independent review of pipeline, pairing, config, models, stats, and I/O modules.
- Identified additional issues not covered by original review.

### Verdict on Codex Findings
- **11/11 confirmed valid** - Codex's analysis was thorough and accurate.
- Severity adjustments noted below where warranted.

---

## Findings (Ordered by Severity)

### 1) Critical: Plot pipeline is disconnected from the validated config schema

**Status: CONFIRMED**

- Evidence:
  - Plot schema defines `data: list[str]` as pair references: `config/schema.py:478`.
  - Plot runtime stage reads `plot_spec.get("pairs", [])` instead: `pipeline/stages.py:1257`.
  - Legacy example (`examples/configs/cmaq_airnow.yaml:57`) uses `data`; working production configs (`analyses/asia-aq/configs/asia-aq-airnow-derecho.yaml:101`) use `pairs`.
  - No adaptation or migration logic maps `data` -> `pairs` anywhere in the codebase.
- Impact:
  - Any config following the Pydantic schema and using `data` silently produces zero plots. Configs only work if they bypass the schema and use the undocumented `pairs` key.
- Recommendation:
  - `PlottingStage` should read `plot_spec.get("data") or plot_spec.get("pairs", [])` with `data` taking precedence. Add a deprecation warning for `pairs`.

---

### 2) Critical: Requested statistics metrics can be silently ignored

**Status: CONFIRMED - upgraded from High to Critical**

- Evidence:
  - `StatsConfig.stat_list` defaults to `["MB", "NMB", "R2", "RMSE"]`: `config/schema.py:566`.
  - Stage uses `config.get("stat_list") or config.get("metrics")`: `pipeline/stages.py:1077`.
  - Since `StatsConfig` inherits `FlexibleModel` (`extra="allow"`), user-supplied `metrics` lands in `model_extra` while `stat_list` always gets its default.
  - After `model_dump()` in `runner.py:1694`, both keys are present, and `stat_list` wins the OR.
  - **All example configs** and **all production configs** use `metrics`, meaning they all silently get the wrong metric set.
- Impact:
  - Users requesting 9 metrics receive 4 default metrics with no warning. This affects every real config in the project.
- Recommendation:
  - Reverse precedence: `config.get("metrics") or config.get("stat_list")`. Better: make `stat_list` default to `None` and deprecate it.

---

### 3) High: `ModelData.extract_surface()` ignores CESM vertical convention

**Status: NEW - identified by Claude**

- Evidence:
  - `models/base.py:248-269`: `extract_surface()` always takes `isel({level_dim: 0})`.
  - The **correct** auto-detection logic exists in `pairing/strategies/base.py:417-422` (checks if pressure increases with index, uses `-1` for CESM).
  - CLAUDE.md documents this as a recurring bug (rediscovered 4+ times).
- Impact:
  - Any code path calling `model_data.extract_surface()` directly (bypassing pairing strategies) gets stratospheric data for CESM - O3 values of 5000-10000 ppb instead of 30-80 ppb.
- Recommendation:
  - Copy the auto-detection logic from `pairing/strategies/base.py:_extract_surface()` into `models/base.py:extract_surface()`.

---

### 4) High: Relative file paths are resolved against process CWD, not config file directory

**Status: CONFIRMED - downgraded from High to Medium (Low practical risk)**

- Evidence:
  - `config_path` stored in metadata (`runner.py:1708`) but never used for resolution.
  - Observation loading uses `Path.cwd()` explicitly: `pipeline/stages.py:397-434`.
  - Model loading passes paths directly to `glob()` with no normalization: `pipeline/stages.py:285-310`.
- Impact:
  - Relative paths in configs break when running from a different directory.
- Mitigating factors:
  - All production configs use absolute paths or `${ENV_VAR}` expansion.
  - Only one relative path found in the entire project (`analyses/asia-aq/misc/cesm_surface.yaml`), in a non-production "misc" directory.
  - The env-var expansion pattern is well-documented and preferred.
- Recommendation:
  - Low priority. If addressed, resolve relative paths against `Path(context.metadata["config_path"]).parent` when available.

---

### 5) High: Skipped stages are not finalized in runner UI/log lifecycle

**Status: CONFIRMED**

- Evidence:
  - Runner only calls `stage_end` for `FAILED` and `COMPLETED`: `runner.py:1602-1620`.
  - No `elif` clause for `StageStatus.SKIPPED`.
  - `stage_end` stops the live animation thread: `runner.py:836-839`.
- Impact:
  - `SKIPPED` stages leave the live animation running, producing garbled terminal output and inconsistent log entries.
- Recommendation:
  - Add `SKIPPED` handling with `formatter.stage_end(...)` and `log_collector.end_stage(...)`.

---

### 6) High: Time filtering has off-by-one bug for ISO format timestamps

**Status: NEW - identified by Claude**

- Evidence:
  - `pipeline/stages.py:709-715`: Regex `r"\d{2}:\d{2}"` detects time components.
  - `"2024-01-01 00:00:00"` matches (space separator) - correctly treated as timestamp.
  - `"2024-01-01T00:00:00"` does NOT match (`T` separator) - incorrectly treated as date-only, extended to end of day.
- Impact:
  - ISO format timestamps (`T` separator) get an extra ~24 hours added to the end time.
- Recommendation:
  - Fix regex to `r"[T ]\d{2}:\d{2}"`.

---

### 7) Medium: `time_tolerance` is largely not enforced during pairing

**Status: CONFIRMED**

- Evidence:
  - All 5 strategies accept `time_tolerance` in their signatures but none enforce it:
    - Point (`point.py:52`): uses `method="nearest"` without distance check.
    - Track (`track.py:83`): parameter accepted, never referenced.
    - Profile (`profile.py:49`): parameter accepted, never referenced.
    - Swath (`swath.py:49`): parameter accepted, never referenced.
    - Grid (`grid.py:262-264`): checks presence but doesn't validate match distance.
- Impact:
  - Matches with 6+ hour time differences accepted silently when user requests 1-hour tolerance.
- Recommendation:
  - Use xarray's `tolerance=` parameter in `.sel(method="nearest")`, or post-filter to reject matches exceeding tolerance.

---

### 8) Medium: `NME_%` fallback formula in CSV summary is incorrect

**Status: CONFIRMED**

- Evidence:
  - Fallback uses `RMSE / |Mean_Obs| * 100`: `pipeline/stages.py:1489-1490`.
  - Correct NME formula is `mean(|mod - obs|) / mean(obs) * 100` (from `stats/metrics.py:439-453`).
  - RMSE >= MAE for any dataset, so the fallback systematically overestimates NME.
- Impact:
  - Inflated NME values when the NME metric was not explicitly computed.
- Recommendation:
  - Use `ME / |Mean_Obs| * 100` if ME is available, otherwise leave as NaN.

---

### 9) Medium: CLI `--strict` validation mode is not implemented

**Status: CONFIRMED**

- Evidence:
  - CLI exposes `--strict` flag: `cli/app.py:366`.
  - `validate_config_command` prints "Mode: strict" but calls `load_config(p)` without passing `strict`: `cli/commands/validate.py:45-71`.
  - `StrictModel` with `extra="forbid"` exists at `config/schema.py:27-34` but is unused.
- Impact:
  - `davinci-monet validate config.yaml --strict` tells users it's running strict validation when it isn't.
- Recommendation:
  - Wire `strict` through to `load_config()`, using `StrictModel`-based schema variants when strict is requested.

---

### 10) Medium: Swath pairing collapses residual dimensions by averaging

**Status: CONFIRMED**

- Evidence:
  - Pixel extraction allocates flat `np.full([n_pixels], np.nan)` array: `pairing/strategies/swath.py:214`.
  - Residual dimensions collapsed via `float(point_val.mean().values)`: `swath.py:228-229`.
  - Comment says "for vertical" but it applies to any residual dimension (time, vertical, etc.).
- Impact:
  - Vertical profiles averaged to a single value instead of being interpolated to satellite retrieval levels.
- Recommendation:
  - Implement explicit vertical interpolation and time matching instead of unconditional averaging.

---

### 11) Medium: Statistics stage silently swallows per-pair computation errors

**Status: NEW - identified by Claude**

- Evidence:
  - `pipeline/stages.py:1038-1058`: Exceptions caught and appended to `context.metadata["stats_errors"]` but never logged to console or progress output.
  - Compare to plotting stage which has the same pattern (also silent).
- Impact:
  - "Statistics completed" shown to user while some pairs produced zero metrics. Must manually inspect metadata to discover failures.
- Recommendation:
  - Add `context.log_progress(f"Warning: Stats failed for {pair_key}: {e}")` in the except block.

---

### 12) Medium: Variable name collision risk in paired dataset assembly

**Status: NEW - identified by Claude**

- Evidence:
  - `pairing/engine.py:239-249`: `_select_var()` searches for both prefixed (`obs_O3`, `model_O3`) and unprefixed (`O3`) names.
  - If a strategy returns unprefixed variables, `obs_key` and `model_key` can both resolve to `"O3"`, causing both `obs_O3` and `model_O3` to contain the same data.
- Impact:
  - Silent data corruption in paired datasets when strategies don't prefix their outputs. Statistics would show perfect correlation (R=1.0).
- Recommendation:
  - Add validation: if `obs_key == model_key`, raise an error.

---

### 13) Low: Pairing stage can report success even when all pairings fail

**Status: CONFIRMED**

- Evidence:
  - When `total_pairs > 0` and `paired_count == 0`, logs warning but returns `StageStatus.COMPLETED`: `pipeline/stages.py:978-991`.
- Impact:
  - Pipeline exits successfully with empty analysis outputs. Downstream stages skip silently.
- Recommendation:
  - Return `FAILED` when `paired_count == 0` and `total_pairs > 0`.

---

### 14) Low: Statistics summary CSV loses pair identity

**Status: CONFIRMED**

- Evidence:
  - Rows created without `pair_key`: `pipeline/stages.py:1461-1466`.
  - Index is only `Variable`: `pipeline/stages.py:1498`.
- Impact:
  - Multi-pair runs produce ambiguous CSV rows. Two pairs evaluating PM2.5 are indistinguishable.
- Recommendation:
  - Add `Pair` column and index by `(Pair, Variable)`.

---

### 15) Low: README example passes wrong config type to `PipelineContext`

**Status: CONFIRMED**

- Evidence:
  - `README.md:90-92`: `load_config()` returns `MonetConfig`, passed directly to `PipelineContext`.
  - Stages use `context.config.get(...)` throughout, which Pydantic v2 `BaseModel` doesn't support.
  - Correct usage in `runner.py:1694`: `load_config(config).model_dump()`.
- Impact:
  - Following the README example produces `AttributeError` at runtime.
- Recommendation:
  - Update README to use `run_from_config("config.yaml")` or `load_config(...).model_dump()`.

---

### 16) Low: Inconsistent Dask worker configuration between stages

**Status: NEW - identified by Claude**

- Evidence:
  - Pairing stage has sophisticated auto-config based on CPU/RAM: `pipeline/stages.py:820-834`.
  - Model/obs loading uses `xr.open_mfdataset(..., parallel=True)` with no worker limits: `pipeline/stages.py:471`.
- Impact:
  - On low-RAM systems, unbounded loading threads can cause memory exhaustion before pairing's careful limits take effect.
- Recommendation:
  - Low priority. Apply similar worker limits to loading stages, or at minimum set `parallel=False` when `DASK_NUM_WORKERS=1`.

---

### 17) Low: `compute_tropospheric_column()` defined but never used

**Status: NEW - identified by Claude**

- Evidence:
  - `models/cesm.py:36-128`: Full implementation for satellite column comparisons.
  - Never called anywhere in the codebase (grep confirms).
- Impact:
  - Dead code. Users needing column comparisons don't know it exists.
- Recommendation:
  - Either expose via model API or document its existence. Low priority.

---

## Test Coverage Gaps Observed

- No end-to-end tests for plotting using schema-native `plots.data` references.
- No tests asserting `--strict` validation behavior beyond CLI argument parsing.
- No tests for `time_tolerance` enforcement across any pairing strategy.
- No tests for relative-path resolution from non-CWD locations.
- No tests for `ModelData.extract_surface()` with CESM-convention vertical coordinates.
- No tests for ISO-format (`T` separator) time filtering behavior.

## Overall Assessment

The codebase has strong structural organization and substantial test coverage (792 tests passing) around core primitives. The architecture is clean and well-decomposed.

The most impactful issues are **integration mismatches** between the config schema and runtime pipeline stages:
- Finding #1 (plot `data` vs `pairs`) and Finding #2 (stats `metrics` vs `stat_list`) mean that configs following the documented schema silently produce wrong results.
- Finding #3 (CESM surface extraction in `ModelData`) is a correctness bug that has been rediscovered 4+ times.

**Recommended fix priority:**
1. **Immediate** (silent wrong results): Findings #1, #2, #3 (CESM surface)
2. **Soon** (UI/UX bugs, data quality): Findings #5, #6, #8, #11 (silent errors)
3. **Next iteration** (robustness): Findings #7, #9, #10, #12, #13, #14, #15
4. **Backlog** (polish): Findings #4, #16, #17

Codex's original review was thorough and well-targeted - all 11 findings verified. The additional findings from the Claude pass primarily surface a second category of risk: silent error swallowing and edge-case data corruption that unit tests don't yet cover.

---

## MODIS Review (2026-03-09)

Reviewer: Codex (GPT-5)

### Scope and Method
- Reviewed [`PLAN_MODIS.md`](PLAN_MODIS.md) against the committed MODIS-related implementation on branch `feature/modis`.
- Inspected the current reader, pipeline stages, pairing engine, synthetic tests, and the smoke-test analysis under `analyses/modis-aod/`.
- Verified the isolated new primitive with `pytest -q davinci_monet/tests/test_swath_grid_strategy.py` (`19 passed`) and the existing MODIS reader tests with `pytest -q davinci_monet/tests/test_observation_readers.py -k MODISL2AODReader` (`3 passed`).
- Also cross-checked the installed `monetio.sat._modis_l2_mm` API during review because the plan depends on its exact return type.

### Findings (Ordered by Severity)

### M1) Critical: `MODISL2AODReader` does not match the real `monetio` MODIS API

**Status: CONFIRMED**

- Evidence:
  - The plan correctly documents that `monetio.sat._modis_l2_mm.read_mfdataset()` returns an `OrderedDict` of per-granule datasets: `PLAN_MODIS.md:13-18`.
  - The working smoke test uses `read_mfdataset(file_pattern, variable_dict)` and iterates `granules.items()`: `analyses/modis-aod/scripts/smoke_test.py:71-95`.
  - The committed reader treats `read_dataset()` / `read_mfdataset()` as returning a single `xr.Dataset`, and it passes a `variables` sequence instead of the required `variable_dict`: `davinci_monet/observations/satellite/modis_l2_aod.py:67-142`.
  - I verified during review that the installed `monetio` signatures are `read_dataset(fname, variable_dict)` and `read_mfdataset(fnames, variable_dict)`, with the multi-file path returning an `OrderedDict`.
- Impact:
  - The current MODIS reader is not usable for real HDF4 MODIS multi-file ingestion, so it cannot serve as the pipeline entry point described in the plan.
- Recommendation:
  - Replace this reader with the planned read-and-bin loader, or at minimum rework it around `variable_dict` plus the per-granule `OrderedDict` contract before any pipeline wiring is attempted.

### M2) High: The standard pipeline still has no MODIS-specific load-and-bin path

**Status: CONFIRMED**

- Evidence:
  - The plan's central implementation step is to add a `modis_l2` branch in `LoadObservationsStage` that bins granules onto a model grid during load: `PLAN_MODIS.md:61-108`.
  - The current stage only special-cases ICARTT, LMA, and AERONET; every other observation path goes through generic `xr.open_mfdataset()` / `xr.open_dataset()`: `davinci_monet/pipeline/stages.py:418-510`.
  - The smoke test still performs the real MODIS read + bin logic manually outside the pipeline: `analyses/modis-aod/scripts/smoke_test.py:71-194`.
- Impact:
  - `davinci-monet run` still cannot reproduce the working MODIS smoke test from YAML. The implementation is a reusable primitive plus a standalone script, not full pipeline support.
- Recommendation:
  - Add the explicit `modis_l2` observation loader branch first, sourcing grid coordinates from `context.models` as planned, and keep the smoke-test logic as the reference implementation to lift into production code.

### M3) High: `SwathGridStrategy` is implemented but unreachable from the pipeline

**Status: CONFIRMED**

- Evidence:
  - `SwathGridStrategy` exists and is tested directly, but it advertises the same geometry as the regular swath strategy: `davinci_monet/pairing/strategies/swath_grid.py:30-64`.
  - `PairingEngine` only registers `SwathStrategy()` for `DataGeometry.SWATH`; it never registers or selects `SwathGridStrategy()`: `davinci_monet/pairing/engine.py:79-95`.
  - `PairingStage` always instantiates the default engine and calls `engine.pair(...)` with no MODIS-specific override: `davinci_monet/pipeline/stages.py:785-799`, `davinci_monet/pipeline/stages.py:891-916`.
  - The plan itself identifies this as a required follow-up and proposes a `pairing_strategy` hint / direct strategy dispatch: `PLAN_MODIS.md:152-176`.
- Impact:
  - Even if MODIS swaths were loaded successfully, the normal run path would still use nearest-neighbor swath pairing rather than the new swath-to-grid binning algorithm.
- Recommendation:
  - Wire an explicit selection path for `SwathGridStrategy` in `PairingStage`, or introduce a distinct geometry/dispatch hint exactly as the plan describes.

### M4) Medium: The proposed MODIS YAML contract is not compatible with the current runtime contract

**Status: CONFIRMED**

- Evidence:
  - The plan's example config introduces `obs_type: modis_l2`, `platforms`, `grid_source`, `time_resolution`, `save_binned`, `binned_file`, `load_binned`, and a top-level `pairs:` block: `PLAN_MODIS.md:88-150`, `PLAN_MODIS.md:178-258`.
  - `ObservationConfig` does not define any of those MODIS-specific fields, and the runtime stage only reads `filename`, `obs_type`, `variables`, `resample`, `min_obs_count`, and `track_obs_count`: `davinci_monet/config/schema.py:329-367`, `davinci_monet/pipeline/stages.py:418-423`, `davinci_monet/pipeline/stages.py:544-567`.
  - Pair discovery currently ignores a top-level `pairs:` section and only reads `model.<label>.mapping`: `davinci_monet/pipeline/stages.py:785-799`.
  - `ObservationData.geometry_from_obs_type()` also does not recognize `modis_l2`, so `create_observation_data(..., obs_type="modis_l2")` currently defaults the container geometry to `POINT`: `davinci_monet/observations/base.py:128-169`, `davinci_monet/observations/base.py:501-547`.
- Impact:
  - Copying the plan's YAML into the repo today would not yield a working MODIS run, even before the reader/pairing issues above.
- Recommendation:
  - Finalize the runtime contract first: either adopt the plan's new fields end-to-end, or revise the plan so the example YAML matches what the runner actually consumes.

### M5) Medium: Test coverage is good for the new primitive, but it misses the real MODIS integration path

**Status: CONFIRMED**

- Evidence:
  - The new swath-grid tests are solid, but they instantiate `SwathGridStrategy` directly with synthetic xarray swaths: `davinci_monet/tests/test_swath_grid_strategy.py:195-345`.
  - The existing MODIS reader tests only cover name, mapping, and `_standardize_dataset()`; they never exercise `open()`, `monetio`, or `LoadObservationsStage`: `davinci_monet/tests/test_observation_readers.py:511-534`.
  - There is no committed YAML config under `analyses/modis-aod/configs/`; the analysis directory currently contains the smoke test and generated outputs only.
- Impact:
  - The current green test signal does not protect the actual failure modes that matter for end-to-end MODIS support: `monetio` API mismatch, HDF4 loading, stage integration, caching, and pairing dispatch.
- Recommendation:
  - Add one pipeline-level integration test that monkeypatches `monetio.sat._modis_l2_mm.read_mfdataset()` to return the same `OrderedDict` shape used by the smoke test, then assert that `LoadObservationsStage` + `PairingStage` produce gridded paired output.

### Current State Summary
- The repository now has a credible swath-to-grid binning primitive (`grid_binning.py` + `SwathGridStrategy`) and plotting support for gridded spatial output.
- The working MODIS example is still the standalone smoke test, not the standard YAML-driven pipeline promised by `PLAN_MODIS.md`.
- The highest-value next step is to implement the loader/dispatch contract, not more smoke-test refinement.
