# Unified Data-Source Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the small correctness and ergonomics gaps left after the model/obs unification refactor before starting the next major feature set.

**Architecture:** Keep the existing hybrid architecture: `LoadSourcesStage` remains the standard loader, legacy model/obs stages remain available as compatibility shims, and paired outputs keep source-label variable names. The cleanup focuses on making unified `sources:` runs behave consistently in plotting, migration, obs-only workflows, and pair-configuration error handling without deleting legacy paths.

**Tech Stack:** Python 3.11, xarray, pytest, Pydantic config models, DAVINCI pipeline stages and source registry.

---

## Repo Rules For This Plan

- Do not commit or push during execution unless the user explicitly asks.
- Use the `davinci` conda environment for validation:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate davinci
```

- Stop after each task if the user wants review checkpoints.
- Read every target file before editing it.

## File Structure

| File | Responsibility | Planned Change |
|------|----------------|----------------|
| `davinci_monet/pipeline/stages.py` | Source loading, pair job construction, statistics, plotting | Resolve unified pair specs in `PlottingStage`; infer source role when omitted; fail invalid configured pairs; update stale docstrings |
| `davinci_monet/config/migration.py` | Legacy config migration helpers | Normalize legacy observation types to registered source types; reject unsupported MODIS L2 gridded migration with a clear error |
| `davinci_monet/tests/test_unified_sources_runtime.py` | End-to-end source-schema runtime regressions | Add pipeline tests for unified pair plotting, role-less obs-only stats, and invalid configured pairs |
| `davinci_monet/tests/unit/config/test_migrate_to_sources.py` | Migration unit tests | Add tests for source-type normalization and unsupported legacy satellite migration |
| `CLAUDE.md` | Project context and current conventions | Clarify hybrid cleanup state and supported `sources:` behavior |

---

## Task 1: Make Unified `sources:` Pair Plots Generate Output

**Files:**
- Modify: `davinci_monet/pipeline/stages.py`
- Test: `davinci_monet/tests/test_unified_sources_runtime.py`

- [ ] **Step 1: Write the failing pipeline test**

Append this test to `davinci_monet/tests/test_unified_sources_runtime.py`:

```python
def test_sources_config_plotting_generates_outputs_for_unified_pair(tmp_path: Path) -> None:
    from davinci_monet.pipeline.runner import PipelineRunner

    model_path = tmp_path / "model.nc"
    obs_path = tmp_path / "obs.nc"
    output_dir = tmp_path / "out"
    _write_grid_source(model_path)
    _write_point_source(obs_path)

    config = {
        "analysis": {"output_dir": str(output_dir)},
        "sources": {
            "cam": {
                "type": "generic",
                "role": "model",
                "files": str(model_path),
                "radius_of_influence": 200000,
                "variables": {"O3": {"units": "ppb"}},
            },
            "airnow": {
                "type": "pt_sfc",
                "role": "obs",
                "filename": str(obs_path),
                "variables": {"o3": {"units": "ppb"}},
            },
        },
        "pairs": {
            "cam_airnow_o3": {
                "sources": ["cam", "airnow"],
                "reference": "airnow",
                "variables": {"cam": "O3", "airnow": "o3"},
            }
        },
        "plots": {"scatter_o3": {"type": "scatter", "pairs": ["cam_airnow_o3"]}},
        "stats": {"metrics": ["N", "MB"]},
    }

    result = PipelineRunner(show_progress=False).run_from_config(config)

    assert result.success
    assert result.context is not None
    plot_result = result.context.results["plotting"]
    assert sorted(Path(p).suffix for p in plot_result.data["plots_generated"]) == [".pdf", ".png"]
    assert (output_dir / "airnow" / "00_scatter_o3.png").exists()
    assert (output_dir / "airnow" / "00_scatter_o3.pdf").exists()
```

- [ ] **Step 2: Run the test and verify it fails**

Run:

```bash
pytest davinci_monet/tests/test_unified_sources_runtime.py::test_sources_config_plotting_generates_outputs_for_unified_pair -v
```

Expected: FAIL because `plot_result.data["plots_generated"]` is empty.

- [ ] **Step 3: Add helper methods to `PlottingStage`**

In `davinci_monet/pipeline/stages.py`, add these methods inside `class PlottingStage`, before `execute()`:

```python
    @staticmethod
    def _as_dataset(obj: Any) -> xr.Dataset | None:
        data = obj.data if hasattr(obj, "data") else obj
        return data if isinstance(data, xr.Dataset) else None

    @staticmethod
    def _source_var_config(
        config: dict[str, Any],
        source_label: str,
        variable_name: str,
    ) -> dict[str, Any]:
        sources_cfg = config.get("sources", {})
        source_cfg = sources_cfg.get(source_label, {}) if isinstance(sources_cfg, dict) else {}
        variables = source_cfg.get("variables", {}) if isinstance(source_cfg, dict) else {}
        if isinstance(variables, dict):
            if variable_name in variables:
                return dict(variables[variable_name])
            for cfg in variables.values():
                if isinstance(cfg, dict) and cfg.get("rename") == variable_name:
                    return dict(cfg)

        model_cfg = config.get("model", {})
        legacy_source_cfg = model_cfg.get(source_label, {}) if isinstance(model_cfg, dict) else {}
        legacy_variables = (
            legacy_source_cfg.get("variables", {}) if isinstance(legacy_source_cfg, dict) else {}
        )
        if isinstance(legacy_variables, dict) and variable_name in legacy_variables:
            return dict(legacy_variables[variable_name])
        return {}

    @staticmethod
    def _select_pair_vars(
        paired_data: xr.Dataset,
        pair_spec: dict[str, Any],
    ) -> tuple[str, str, str] | None:
        pair_vars = iter_paired_variable_pairs(paired_data)
        if not pair_vars:
            return None

        variables = pair_spec.get("variables") or {}
        wanted = {str(v) for v in variables.values() if v}
        if wanted:
            for reference_var, comparand_var, canonical in pair_vars:
                if (
                    canonical in wanted
                    or reference_var in wanted
                    or comparand_var in wanted
                    or paired_data[reference_var].attrs.get("source_label") in wanted
                    or paired_data[comparand_var].attrs.get("source_label") in wanted
                ):
                    return reference_var, comparand_var, canonical

        return pair_vars[0]

    def _resolve_plot_pair(
        self,
        context: PipelineContext,
        pair_name: str,
        pair_spec: dict[str, Any],
    ) -> tuple[str, xr.Dataset, str, str, str, str, str, dict[str, str]] | None:
        if "sources" in pair_spec:
            pair_key = pair_name
            if pair_key not in context.paired:
                return None
            paired_obj = context.paired[pair_key]
            paired_data = self._as_dataset(paired_obj)
            if paired_data is None:
                return None
            selected = self._select_pair_vars(paired_data, pair_spec)
            if selected is None:
                return None
            reference_var_name, comparand_var_name, canonical = selected
            reference_label = str(
                paired_data[reference_var_name].attrs.get("source_label", "reference")
            )
            comparand_label = str(
                paired_data[comparand_var_name].attrs.get("source_label", "comparand")
            )
            var_spec = {"obs_var": canonical, "model_var": canonical}
            return (
                pair_key,
                paired_data,
                reference_var_name,
                comparand_var_name,
                reference_label,
                comparand_label,
                canonical,
                var_spec,
            )

        model_label = str(pair_spec.get("model", ""))
        obs_label = str(pair_spec.get("obs", ""))
        var_spec = pair_spec.get("variable", {}) if isinstance(pair_spec, dict) else {}
        obs_var = str(var_spec.get("obs_var", ""))
        pair_key = f"{model_label}_{obs_label}"
        if not pair_spec and pair_name in context.paired:
            pair_key = pair_name
        if pair_key not in context.paired:
            return None
        paired_obj = context.paired[pair_key]
        paired_data = self._as_dataset(paired_obj)
        if paired_data is None:
            return None

        if not pair_spec:
            pair_vars = iter_paired_variable_pairs(paired_data)
            if not pair_vars:
                return None
            obs_var_name, model_var_name, canonical = pair_vars[0]
            obs_label = str(paired_data[obs_var_name].attrs.get("source_label", "reference"))
            model_label = str(paired_data[model_var_name].attrs.get("source_label", "comparand"))
            var_spec = {"obs_var": canonical, "model_var": canonical}
            return (
                pair_key,
                paired_data,
                obs_var_name,
                model_var_name,
                obs_label,
                model_label,
                canonical,
                var_spec,
            )

        obs_var_name, model_var_name = resolve_paired_var_names(
            paired_data, obs_var, obs_label, model_label
        )
        return (
            pair_key,
            paired_data,
            obs_var_name,
            model_var_name,
            obs_label,
            model_label,
            obs_var,
            dict(var_spec),
        )
```

- [ ] **Step 4: Replace the pair-resolution block in `PlottingStage.execute()`**

In `davinci_monet/pipeline/stages.py`, replace the block from `# Get pair configuration` through the legacy variable resolution with:

```python
                    pair_spec = pairs_config.get(pair_name, {})
                    resolved = self._resolve_plot_pair(context, str(pair_name), pair_spec)
                    if resolved is None:
                        continue
                    (
                        pair_key,
                        paired_data,
                        obs_var_name,
                        model_var_name,
                        obs_label,
                        model_label,
                        obs_var,
                        var_spec,
                    ) = resolved

                    if obs_var_name not in paired_data or model_var_name not in paired_data:
                        continue
```

Keep the existing domain filter immediately after this block.

- [ ] **Step 5: Use source-aware variable config for plot limits**

In `PlottingStage.execute()`, replace:

```python
                    model_var = var_spec.get("model_var", "")
                    var_config = (
                        model_config.get(model_label, {}).get("variables", {}).get(model_var, {})
                    )
```

with:

```python
                    model_var = var_spec.get("model_var", "")
                    var_config = self._source_var_config(context.config, model_label, model_var)
```

- [ ] **Step 6: Run the focused test**

Run:

```bash
pytest davinci_monet/tests/test_unified_sources_runtime.py::test_sources_config_plotting_generates_outputs_for_unified_pair -v
```

Expected: PASS.

- [ ] **Step 7: Run existing unified runtime tests**

Run:

```bash
pytest davinci_monet/tests/test_unified_sources_runtime.py -q
```

Expected: PASS.

- [ ] **Step 8: Manual checkpoint**

Show the diff for review:

```bash
git diff -- davinci_monet/pipeline/stages.py davinci_monet/tests/test_unified_sources_runtime.py
```

Do not commit unless the user explicitly asks.

---

## Task 2: Normalize Migration Source Types And Reject Unsupported MODIS L2 Gridding

**Files:**
- Modify: `davinci_monet/config/migration.py`
- Test: `davinci_monet/tests/unit/config/test_migrate_to_sources.py`

- [ ] **Step 1: Add failing migration tests**

Append these tests to `davinci_monet/tests/unit/config/test_migrate_to_sources.py`:

```python
def test_migrate_surface_obs_type_to_registered_source_type() -> None:
    out = migrate_to_sources(
        {
            "obs": {
                "airnow": {
                    "obs_type": "airnow",
                    "filename": "/data/airnow.nc",
                    "variables": {"o3": {}},
                }
            }
        }
    )

    assert out["sources"]["airnow"]["type"] == "airnow"
    assert out["sources"]["airnow"]["role"] == "obs"


def test_migrate_satellite_l2_generic_to_registered_source_type() -> None:
    out = migrate_to_sources(
        {
            "obs": {
                "sat": {
                    "obs_type": "sat_swath_clm",
                    "filename": "/data/sat/*.nc",
                    "variables": {"no2": {}},
                }
            }
        }
    )

    assert out["sources"]["sat"]["type"] == "satellite_l2"


def test_migrate_modis_l2_gridded_workflow_fails_loudly() -> None:
    import pytest

    from davinci_monet.core.exceptions import ConfigurationError

    with pytest.raises(ConfigurationError, match="MODIS L2 gridded"):
        migrate_to_sources(
            {
                "obs": {
                    "terra_modis": {
                        "obs_type": "sat_swath_clm",
                        "sat_type": "modis_l2",
                        "filename": "/data/MOD04_L2.*.hdf",
                        "grid_source": "cam",
                        "variables": {"AOD_550": {}},
                    }
                }
            }
        )
```

- [ ] **Step 2: Run the migration tests and verify failure**

Run:

```bash
pytest davinci_monet/tests/unit/config/test_migrate_to_sources.py -q
```

Expected: FAIL because `airnow` currently maps through `obs_type` inconsistently, `sat_swath_clm` is not normalized to `satellite_l2`, and `modis_l2` is not rejected.

- [ ] **Step 3: Add source-type normalization helpers**

In `davinci_monet/config/migration.py`, add this block above `migrate_to_sources()`:

```python
OBS_TYPE_TO_SOURCE_TYPE = {
    "pt_sfc": "pt_sfc",
    "surface": "pt_sfc",
    "point": "pt_sfc",
    "airnow": "airnow",
    "aeronet": "aeronet",
    "aqs": "aqs",
    "openaq": "openaq",
    "pandora": "pandora",
    "aircraft": "icartt",
    "track": "icartt",
    "mobile": "icartt",
    "sonde": "ozonesonde",
    "profile": "ozonesonde",
    "ozonesonde": "ozonesonde",
    "lma": "lma",
    "sat_swath_clm": "satellite_l2",
    "satellite": "satellite_l2",
    "swath": "satellite_l2",
    "l2": "satellite_l2",
    "sat_grid_clm": "satellite_l3",
    "gridded": "satellite_l3",
    "grid": "satellite_l3",
    "l3": "satellite_l3",
}

SAT_TYPE_TO_SOURCE_TYPE = {
    "goes_l3_aod": "goes_l3_aod",
    "mopitt_l3_co": "mopitt_l3_co",
    "omps_l3_o3": "omps_l3_o3",
    "tempo_l2_no2": "tempo_l2_no2",
    "tropomi": "tropomi",
    "modis_l2_aod": "modis_l2_aod",
}


def _legacy_obs_source_type(label: str, entry: dict[str, Any]) -> str | None:
    sat_type = entry.get("sat_type")
    if sat_type == "modis_l2":
        raise ConfigurationError(
            "Cannot automatically migrate MODIS L2 gridded observation "
            f"'{label}' to sources:. The legacy sat_type='modis_l2' path bins "
            "swath granules onto a model grid using grid_source, which does not "
            "yet have a role-neutral source reader. Keep this config in legacy "
            "model:/obs: form for now or create a pre-binned NetCDF source."
        )
    if sat_type:
        return SAT_TYPE_TO_SOURCE_TYPE.get(str(sat_type).lower(), str(sat_type))

    obs_type = entry.get("obs_type")
    if obs_type is None:
        return None
    obs_type_key = str(obs_type).lower()
    return OBS_TYPE_TO_SOURCE_TYPE.get(obs_type_key, str(obs_type))
```

- [ ] **Step 4: Use the helper in `migrate_to_sources()`**

In the observation migration loop, replace:

```python
        if "obs_type" in entry:
            new_entry["type"] = entry.pop("obs_type")
```

with:

```python
        source_type = _legacy_obs_source_type(str(label), entry)
        entry.pop("obs_type", None)
        if source_type is not None:
            new_entry["type"] = source_type
```

Do not remove `sat_type` from `entry`; it remains useful reader-specific metadata for source types that accept it.

- [ ] **Step 5: Run the migration tests**

Run:

```bash
pytest davinci_monet/tests/unit/config/test_migrate_to_sources.py -q
```

Expected: PASS.

- [ ] **Step 6: Run config unit tests**

Run:

```bash
pytest davinci_monet/tests/unit/config/test_migrate_to_sources.py davinci_monet/tests/unit/config/test_expand_sources.py davinci_monet/tests/unit/config/test_source_config.py -q
```

Expected: PASS.

- [ ] **Step 7: Manual checkpoint**

Show the diff:

```bash
git diff -- davinci_monet/config/migration.py davinci_monet/tests/unit/config/test_migrate_to_sources.py
```

Do not commit unless the user explicitly asks.

---

## Task 3: Infer Role For Role-Less Unified Sources

**Files:**
- Modify: `davinci_monet/pipeline/stages.py`
- Test: `davinci_monet/tests/test_unified_sources_runtime.py`

- [ ] **Step 1: Add failing role-less obs-only test**

Append this test to `davinci_monet/tests/test_unified_sources_runtime.py`:

```python
def test_roleless_point_source_runs_obs_only_statistics(tmp_path: Path) -> None:
    from davinci_monet.pipeline.runner import PipelineRunner

    obs_path = tmp_path / "obs.nc"
    _write_point_source(obs_path)

    config = {
        "analysis": {"output_dir": str(tmp_path / "out")},
        "sources": {
            "airnow": {
                "type": "pt_sfc",
                "filename": str(obs_path),
                "variables": {"o3": {"units": "ppb"}},
            }
        },
    }

    result = PipelineRunner(show_progress=False).run_from_config(config)

    assert result.success
    assert result.context is not None
    assert "airnow" in result.context.sources
    assert "airnow" in result.context.observations
    assert result.context.sources["airnow"].role == "obs"
    assert result.context.results["obs_statistics"].status.name == "COMPLETED"
    assert result.context.results["obs_statistics"].data["airnow"]["o3"]["N"] == 4
```

- [ ] **Step 2: Run the test and verify failure**

Run:

```bash
pytest davinci_monet/tests/test_unified_sources_runtime.py::test_roleless_point_source_runs_obs_only_statistics -v
```

Expected: FAIL because `airnow` is in `context.sources` but not `context.observations`, and `obs_statistics` is skipped.

- [ ] **Step 3: Add role inference helper**

In `davinci_monet/pipeline/stages.py`, add this method inside `class LoadSourcesStage` near `_data_geometry()`:

```python
    @staticmethod
    def _infer_role(source_type: str, geometry: DataGeometry) -> str | None:
        model_source_types = {
            "cmaq",
            "wrfchem",
            "ufs",
            "rrfs",
            "cesm_fv",
            "cesm_se",
            "generic",
            "raqms",
        }
        obs_source_types = {
            "pt_sfc",
            "airnow",
            "aeronet",
            "aqs",
            "openaq",
            "pandora",
            "icartt",
            "ozonesonde",
            "lma",
            "satellite_l2",
            "satellite_l3",
            "goes_l3_aod",
            "modis_l2_aod",
            "mopitt_l3_co",
            "omps_l3_o3",
            "tempo_l2_no2",
            "tropomi",
        }
        if source_type in model_source_types:
            return "model"
        if source_type in obs_source_types:
            return "obs"
        if geometry is not DataGeometry.GRID:
            return "obs"
        return None
```

- [ ] **Step 4: Apply inference in `_load_unified_source()`**

In `_load_unified_source()`, move the geometry lookup before constructing `SourceData`, then set role from config or inference:

```python
        geometry = self._data_geometry(getattr(reader, "geometry"))
        inferred_role = role or self._infer_role(source_type, geometry)
        source = SourceData(
            data=data,
            label=label,
            source_type=source_type,
            geometry=geometry,
            role=inferred_role,
            variables=variables,
            config=cfg,
        )
```

Remove the old `geometry = ...` line if it now appears twice.

- [ ] **Step 5: Run the focused role inference test**

Run:

```bash
pytest davinci_monet/tests/test_unified_sources_runtime.py::test_roleless_point_source_runs_obs_only_statistics -v
```

Expected: PASS.

- [ ] **Step 6: Run unified runtime tests**

Run:

```bash
pytest davinci_monet/tests/test_unified_sources_runtime.py -q
```

Expected: PASS.

- [ ] **Step 7: Manual checkpoint**

Show the diff:

```bash
git diff -- davinci_monet/pipeline/stages.py davinci_monet/tests/test_unified_sources_runtime.py
```

Do not commit unless the user explicitly asks.

---

## Task 4: Fail Invalid Configured Pair Specs Instead Of Succeeding As No-Ops

**Files:**
- Modify: `davinci_monet/pipeline/stages.py`
- Test: `davinci_monet/tests/test_unified_sources_runtime.py`

- [ ] **Step 1: Add failing invalid-pair test**

Append this test to `davinci_monet/tests/test_unified_sources_runtime.py`:

```python
def test_configured_pair_with_missing_variable_fails_pipeline(tmp_path: Path) -> None:
    from davinci_monet.pipeline.runner import PipelineRunner

    model_path = tmp_path / "model.nc"
    obs_path = tmp_path / "obs.nc"
    _write_grid_source(model_path)
    _write_point_source(obs_path)

    config = {
        "analysis": {"output_dir": str(tmp_path / "out")},
        "sources": {
            "cam": {"type": "generic", "role": "model", "files": str(model_path)},
            "airnow": {"type": "pt_sfc", "role": "obs", "filename": str(obs_path)},
        },
        "pairs": {
            "bad_pair": {
                "sources": ["cam", "airnow"],
                "reference": "airnow",
                "variables": {"cam": "O3"},
            }
        },
    }

    result = PipelineRunner(show_progress=False).run_from_config(config)

    assert not result.success
    assert result.failed_stages[0].stage_name == "pairing"
    assert "bad_pair" in (result.failed_stages[0].error or "")
    assert "missing variable mapping" in (result.failed_stages[0].error or "")
```

- [ ] **Step 2: Run the test and verify failure**

Run:

```bash
pytest davinci_monet/tests/test_unified_sources_runtime.py::test_configured_pair_with_missing_variable_fails_pipeline -v
```

Expected: FAIL because the pipeline currently reports success with no paired data.

- [ ] **Step 3: Change `_build_source_pair_jobs()` to return errors**

In `class PairingStage`, change the signature:

```python
    def _build_source_pair_jobs(self, context: PipelineContext) -> tuple[list[SourcePairJob], list[str]]:
```

At the start of the method, add:

```python
        errors: list[str] = []
```

Replace the early return for non-dict pairs config:

```python
        if not isinstance(pairs_config, dict):
            return [], []
```

For each invalid configured pair, append a specific error and continue. Use these replacements:

```python
                if len(srcs) != 2:
                    errors.append(
                        f"Pair '{pair_name}' must list exactly two sources; got {srcs!r}."
                    )
                    continue
```

```python
                missing_sources = [s for s in (a_label, b_label) if s not in context.sources]
                if missing_sources:
                    errors.append(
                        f"Pair '{pair_name}' references unloaded source(s): {missing_sources}."
                    )
                    continue
```

```python
                if not reference_var or not comparand_var:
                    errors.append(
                        f"Pair '{pair_name}' has missing variable mapping for "
                        f"reference '{reference_label}' or comparand '{comparand_label}'."
                    )
                    continue
```

For legacy pair specs, replace missing source and variable `continue` paths with:

```python
                missing_sources = [
                    label
                    for label in (model_label, obs_label)
                    if label not in context.sources
                ]
                if missing_sources:
                    errors.append(
                        f"Pair '{pair_name}' references unloaded source(s): {missing_sources}."
                    )
                    continue
```

```python
                if not model_var or not obs_var:
                    errors.append(
                        f"Pair '{pair_name}' has missing variable mapping for "
                        f"model '{model_label}' or obs '{obs_label}'."
                    )
                    continue
```

Return both values at the end:

```python
        return jobs, errors
```

- [ ] **Step 4: Fail the pairing stage on pair config errors**

In `PairingStage.execute()`, replace:

```python
        source_jobs = self._build_source_pair_jobs(context)
        if source_jobs:
            return self._execute_source_pair_jobs(
```

with:

```python
        source_jobs, source_pair_errors = self._build_source_pair_jobs(context)
        if source_pair_errors:
            message = "Invalid pair configuration: " + " ".join(source_pair_errors)
            context.metadata.setdefault("pairing_config_errors", []).extend(source_pair_errors)
            return self._create_result(
                StageStatus.FAILED,
                error=message,
                duration=time.time() - start,
            )
        if source_jobs:
            return self._execute_source_pair_jobs(
```

- [ ] **Step 5: Fail if all configured source pairs fail at execution time**

At the end of `_execute_source_pair_jobs()`, before returning the completed result, add:

```python
        if jobs and paired_count == 0:
            errors = context.metadata.get("pairing_errors", [])
            message = "All configured source pairs failed."
            if errors:
                message += " " + " ".join(str(e) for e in errors)
            return self._create_result(
                StageStatus.FAILED,
                error=message,
                duration=time.time() - start,
            )
```

Keep partial success behavior unchanged: if at least one configured pair succeeds, the stage returns completed and records failed pairs in metadata.

- [ ] **Step 6: Run the focused invalid-pair test**

Run:

```bash
pytest davinci_monet/tests/test_unified_sources_runtime.py::test_configured_pair_with_missing_variable_fails_pipeline -v
```

Expected: PASS.

- [ ] **Step 7: Run runtime pairing tests**

Run:

```bash
pytest davinci_monet/tests/test_unified_sources_runtime.py davinci_monet/tests/test_pair_direction.py davinci_monet/tests/test_pairing.py -q
```

Expected: PASS.

- [ ] **Step 8: Manual checkpoint**

Show the diff:

```bash
git diff -- davinci_monet/pipeline/stages.py davinci_monet/tests/test_unified_sources_runtime.py
```

Do not commit unless the user explicitly asks.

---

## Task 5: Update Current-State Documentation And Stale Comments

**Files:**
- Modify: `davinci_monet/pipeline/stages.py`
- Modify: `CLAUDE.md`

- [ ] **Step 1: Update `LoadSourcesStage` docstring**

In `davinci_monet/pipeline/stages.py`, replace the `LoadSourcesStage` class docstring with:

```python
    """Unified data-source loading stage.

    Loads the going-forward ``sources:`` schema directly through
    ``source_registry``. Legacy ``model:`` / ``obs:`` configs still run by
    delegating to ``LoadModelsStage`` and ``LoadObservationsStage``, then exposing
    the loaded containers through ``context.sources``.

    Every loaded dataset is tagged with ``source_label``, ``geometry``, and an
    inferred or explicit ``role`` where DAVINCI can determine one. The
    ``context.models`` and ``context.observations`` dictionaries remain populated
    as compatibility views for legacy consumers and obs-only stages.
    """
```

- [ ] **Step 2: Update `PairingStage` docstring**

In `davinci_monet/pipeline/stages.py`, replace the `PairingStage` class docstring with:

```python
    """Stage for pairing configured binary data sources.

    Unified ``pairs:`` entries use ``sources`` / ``reference`` / ``variables`` and
    are dispatched by reference/comparand geometry. Legacy ``model`` / ``obs``
    entries are still accepted and mapped to reference=obs, comparand=model.
    Invalid configured pairs fail the stage instead of silently producing no
    paired output.
    """
```

- [ ] **Step 3: Update `PlottingStage` docstring**

In `davinci_monet/pipeline/stages.py`, replace the `PlottingStage` class docstring with:

```python
    """Stage for generating plots from paired data.

    Plot inputs can reference legacy pair keys or unified source pair names.
    Paired variables are resolved through role/source-label attrs first and
    legacy ``obs_`` / ``model_`` names second.
    """
```

- [ ] **Step 4: Update `CLAUDE.md` unified source section**

In `CLAUDE.md`, under `## Unified Data-Source Config (sources:)`, add this paragraph after the YAML example:

```markdown
Current implementation note: the runtime is intentionally hybrid. `sources:` is
the preferred schema and the standard pipeline loads it directly, but legacy
`model:`/`obs:` configs and compatibility stages remain available. Plotting and
statistics resolve source-label variables first and legacy `obs_`/`model_`
variables second. Some specialized legacy observation workflows that depend on
model-grid pre-binning, especially MODIS L2 `sat_type: modis_l2`, should stay in
legacy config form until a dedicated role-neutral source reader exists.
```

- [ ] **Step 5: Run docs-sensitive checks**

Run:

```bash
python -m py_compile davinci_monet/pipeline/stages.py
```

Expected: no output and exit status 0.

- [ ] **Step 6: Manual checkpoint**

Show the diff:

```bash
git diff -- davinci_monet/pipeline/stages.py CLAUDE.md
```

Do not commit unless the user explicitly asks.

---

## Task 6: Run The Cleanup Validation Matrix

**Files:**
- No new edits expected.

- [ ] **Step 1: Run focused cleanup tests**

Run:

```bash
pytest \
  davinci_monet/tests/test_unified_sources_runtime.py \
  davinci_monet/tests/unit/config/test_migrate_to_sources.py \
  davinci_monet/tests/unit/config/test_expand_sources.py \
  davinci_monet/tests/unit/config/test_source_config.py \
  davinci_monet/tests/test_pair_direction.py \
  -q
```

Expected: PASS.

- [ ] **Step 2: Run broader pipeline and plotting tests**

Run:

```bash
pytest \
  davinci_monet/tests/test_pipeline.py \
  davinci_monet/tests/test_obs_pipeline.py \
  davinci_monet/tests/test_plots.py \
  davinci_monet/tests/test_plotting_var_resolution.py \
  davinci_monet/tests/test_renderer_role_styling.py \
  -q
```

Expected: PASS.

- [ ] **Step 3: Run type checking for touched modules**

Run:

```bash
mypy davinci_monet/pipeline/stages.py davinci_monet/config/migration.py
```

Expected: no new errors from the touched modules. If existing repo-wide mypy errors appear, record them with exact output before changing code.

- [ ] **Step 4: Run formatting checks for touched files**

Run:

```bash
black --check davinci_monet/pipeline/stages.py davinci_monet/config/migration.py davinci_monet/tests/test_unified_sources_runtime.py davinci_monet/tests/unit/config/test_migrate_to_sources.py
isort --check davinci_monet/pipeline/stages.py davinci_monet/config/migration.py davinci_monet/tests/test_unified_sources_runtime.py davinci_monet/tests/unit/config/test_migrate_to_sources.py
```

Expected: PASS. If either command fails, run `black` or `isort` on only the listed touched files, then rerun the check.

- [ ] **Step 5: Optional full suite**

Run when time allows:

```bash
HDF5_USE_FILE_LOCKING=FALSE pytest
```

Expected: PASS. If this is too slow for the cleanup session, record that the focused matrix passed and the full suite was not run.

- [ ] **Step 6: Final review diff**

Run:

```bash
git diff --stat
git diff -- davinci_monet/pipeline/stages.py davinci_monet/config/migration.py davinci_monet/tests/test_unified_sources_runtime.py davinci_monet/tests/unit/config/test_migrate_to_sources.py CLAUDE.md
```

Expected: changes are limited to the planned files and no generated artifacts are present.

---

## Self-Review Notes

- Review finding coverage:
  - Unified `sources:` plotting skip: Task 1.
  - Migrated config source-type problem and MODIS L2 gridded limitation: Task 2.
  - Role-less source obs-only skip: Task 3.
  - Silent success for invalid configured pairs: Task 4.
  - Stale comments/docs around hybrid state: Task 5.
  - Focused validation matrix: Task 6.
- No task deletes legacy loaders or obs-only stages; that remains out of scope for this cleanup.
- No automatic commit step is included because repository instructions require explicit user confirmation before commits.
