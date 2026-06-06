# Complete Source Unification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove remaining runtime and user-facing model/obs assumptions so DAVINCI's standard path treats every dataset as a role-neutral source with optional role metadata used only for styling and compatibility labels.

**Architecture:** Make `sources` the only internal runtime collection used by standard stages. Keep legacy `model:`/`obs:` config, `context.models`, `context.observations`, and `get_model()`/`get_observation()` as compatibility shims at the boundary, but no standard stage should require those dictionaries to work. Pairing, statistics, plotting, summaries, output naming, and validation should use `reference`/`comparand` and source labels.

**Tech Stack:** Python 3.11, xarray, pytest, Pydantic config models, DAVINCI pipeline stages, source registry, existing pairing strategies and plot renderers.

## Program context (added 2026-06-06)

This plan is **WS1** of the DAVINCI remediation program — see
`docs/superpowers/specs/2026-06-06-davinci-remediation-program-design.md`.
WS1's job: make the unified `sources:` path do **everything** the legacy
`model:`/`obs:` path does, so the legacy side can be deleted in **WS3 (hard break)**.

**Sequencing note (changed since first draft):** the program ratified a **hard
break**. Legacy `model:`/`obs:` config, `ModelData`/`ObservationData`, and the
legacy load/pair/plot paths are **deleted in WS3**, not kept as permanent shims.
The compatibility-preservation work in Tasks 7, 13, and 15 is therefore
**intentionally temporary** — keep legacy working through WS1 so each step stays
shippable, knowing WS3 removes it. Do not expand the compat surface beyond what
those tasks specify.

**Gap closers (both confirmed in `REVIEW.md`):**
- **Task 17 (new): apply `resample`/`min_obs_count`/`track_obs_count` in
  `LoadSourcesStage`** — today the unified path silently drops these (they run
  only in the legacy `LoadObservationsStage`), a behavioral regression.
- **MODIS-L2 reachability via unified `sources:`** — required for WS1
  "feature-complete"; it needs its own focused sub-plan because the approach must
  reconcile load-time binning (Task 10) with the existing `SwathGridStrategy`.
  See the expanded **Open Design Decision** at the end.

The original "Task 17: Final Residual Audit" is renumbered **Task 18** and must
run last.

---

## Repo Rules For This Plan

- Do not commit or push unless the user explicitly asks.
- Read every target file before editing it.
- Use the `davinci` conda environment for validation:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate davinci
```

- Use `apply_patch` for manual edits.
- Stop after planning until the user explicitly approves implementation.

## File Structure

| File | Responsibility | Planned Change |
|------|----------------|----------------|
| `davinci_monet/pipeline/stages.py` | Runtime source loading, pairing, stats, plotting, saves | Add source helpers; make single-source stats/plots source-neutral; make pair failures fail; use source store for overlays and MODIS grid targets |
| `davinci_monet/pipeline/runner.py` | Progress/log output and cleanup | Report sources instead of model/observation buckets; close datasets via `context.sources`; keep legacy bucket output only when legacy config is used |
| `davinci_monet/core/base.py` | Data containers and paired data helpers | Promote reference/comparand accessors on `PairedData`; keep model/obs properties as deprecated aliases |
| `davinci_monet/core/protocols.py` | Public protocol contracts | Update docstrings to identify neutral contracts as canonical and old contracts as compatibility-only |
| `davinci_monet/config/schema.py` | Pydantic config model | Add typed `pairs`; validate `sources` pair shape; add source-oriented plot/stat docs; keep legacy blocks supported |
| `davinci_monet/config/parser.py` | YAML preprocessing and builder | Add `sources` default handling and source-oriented builder methods |
| `davinci_monet/config/migration.py` | Legacy config migration helpers | Keep one-way migration; mark `expand_sources_to_legacy` as test/support-only compatibility |
| `davinci_monet/cli/commands/validate.py` | CLI config validation summary | Print `sources` and unified pairs; retain model/obs summary only for legacy input |
| `davinci_monet/plots/base.py` | Series, labels, color helpers | Add reference/comparand labels and source-neutral plot adapter helpers |
| `davinci_monet/plots/renderers/*.py` | Plot rendering | Move comparison renderer internals from obs/model variable naming to reference/comparand where behavior depends on semantics |
| `davinci_monet/stats/calculator.py` | Metric calculation interface | Add `reference_var`/`comparand_var` interface and keep old `obs_var`/`model_var` wrapper |
| `davinci_monet/stats/metrics.py` | Metric names and formulas | Keep formulas unchanged; document reference/comparand interpretation for bias metrics |
| `davinci_monet/io/writers.py` | Output writer docs/API | Update paired-output docstrings and any model-observation text |
| `davinci_monet/ai/payload.py` | Summary payload | Use source and reference/comparand vocabulary |
| `CLAUDE.md` | Project context | Update architecture status and residual compatibility rules |
| Tests listed below | Regression coverage | Add source-only, source-pair, unsupported-combo, output-label, CLI/schema tests |

---

## Definition Of Done

- Standard pipeline stages can execute from `sources:` without relying on `context.models` or `context.observations`.
- Legacy `model:`/`obs:` configs still run through `LoadSourcesStage`, emit `LegacyConfigWarning`, and populate compatibility dictionaries.
- A single source with `role: model`, `role: obs`, or no role can produce descriptive stats and supported single-source plots.
- Pairing errors for configured source pairs fail the `pairing` stage instead of silently returning success with zero outputs.
- Pairing error messages are based on geometry and source labels, not model/obs language.
- Comparison outputs use canonical reference/comparand names. Legacy model/obs public aliases remain only as explicit compatibility shims.
- Plotting and statistics work for grid-grid model-model, grid-grid obs-obs, grid-point model-obs, grid-point obs-model, and role-less source pairs where the geometry combination is supported.
- Unsupported geometry combinations fail early with a clear error listing supported combinations.
- High-frequency sources configured under `sources:` are resampled (`resample`/`min_obs_count`/`track_obs_count`) during loading, matching legacy `obs:` behavior (Task 17).
- MODIS-L2 swath data is reachable through a `sources:` entry (no legacy `obs:`/`sat_type` branch required) — tracked as a dedicated WS1 sub-plan (see Open Design Decision).
- `pytest`, `mypy davinci_monet`, `black --check davinci_monet`, and `isort --check davinci_monet` pass in the `davinci` env.

---

## Task 1: Add Source-Neutral Test Coverage For Current Residuals

**Files:**
- Modify: `davinci_monet/tests/test_unified_sources_runtime.py`
- Modify: `davinci_monet/tests/test_pair_direction.py`
- Modify: `davinci_monet/tests/test_obs_pipeline.py`
- Modify: `davinci_monet/tests/test_plots.py`
- Modify: `davinci_monet/tests/unit/config/test_schema.py`

- [ ] **Step 1: Add a source-only descriptive stats regression**

Append to `davinci_monet/tests/test_unified_sources_runtime.py`:

```python
def test_single_model_source_gets_descriptive_stats(tmp_path: Path) -> None:
    from davinci_monet.pipeline.runner import PipelineRunner

    source_path = tmp_path / "cam.nc"
    _write_grid_source(source_path)

    config = {
        "analysis": {"output_dir": str(tmp_path / "out")},
        "sources": {
            "cam": {
                "type": "generic",
                "role": "model",
                "files": str(source_path),
                "variables": {"O3": {"units": "ppb"}},
            }
        },
        "stats": {"metrics": ["N"]},
    }

    result = PipelineRunner(show_progress=False).run_from_config(config)

    assert result.success
    assert result.context is not None
    stats = result.context.results["statistics"].data
    assert "cam" in stats
    assert "O3" in stats["cam"]
    assert stats["cam"]["O3"]["N"] == 8
```

- [ ] **Step 2: Add a source-only plot regression using `source:`**

Append to `davinci_monet/tests/test_unified_sources_runtime.py`:

```python
def test_single_source_plot_uses_source_key_not_obs_key(tmp_path: Path) -> None:
    from davinci_monet.pipeline.runner import PipelineRunner

    source_path = tmp_path / "cam.nc"
    _write_grid_source(source_path)

    config = {
        "analysis": {"output_dir": str(tmp_path / "out")},
        "sources": {
            "cam": {
                "type": "generic",
                "role": "model",
                "files": str(source_path),
                "variables": {"O3": {"units": "ppb"}},
            }
        },
        "plots": {
            "hist_o3": {
                "type": "histogram",
                "source": "cam",
                "variable": "O3",
            }
        },
    }

    result = PipelineRunner(show_progress=False).run_from_config(config)

    assert result.success
    assert result.context is not None
    plots = result.context.results["plotting"].data["plots_generated"]
    assert len([p for p in plots if p.endswith(".png")]) == 1
```

- [ ] **Step 3: Add a configured source-pair failure regression**

Append to `davinci_monet/tests/test_unified_sources_runtime.py`:

```python
def test_unsupported_source_pair_fails_pairing_stage() -> None:
    from davinci_monet.core.protocols import DataGeometry
    from davinci_monet.pipeline.stages import (
        PairingStage,
        PipelineContext,
        SourceData,
        StageStatus,
    )

    point_a = xr.Dataset(
        {"o3": ("site", np.array([1.0]))},
        coords={
            "site": [0],
            "latitude": ("site", [40.0]),
            "longitude": ("site", [-105.0]),
        },
        attrs={"geometry": "point"},
    )
    track_b = xr.Dataset(
        {"o3": ("time", np.array([1.2]))},
        coords={
            "time": np.array(["2024-01-01T00:00"], dtype="datetime64[m]"),
            "latitude": ("time", [40.0]),
            "longitude": ("time", [-105.0]),
        },
        attrs={"geometry": "track"},
    )
    ctx = PipelineContext(
        config={
            "pairs": {
                "a_b_o3": {
                    "sources": ["a", "b"],
                    "reference": "a",
                    "variables": {"a": "o3", "b": "o3"},
                }
            }
        },
        sources={
            "a": SourceData(point_a, "a", "pt_sfc", DataGeometry.POINT, role="obs"),
            "b": SourceData(track_b, "b", "icartt", DataGeometry.TRACK, role="obs"),
        },
    )

    result = PairingStage().execute(ctx)

    assert result.status is StageStatus.FAILED
    assert "a_b_o3" in str(result.error)
    assert "Unsupported pairing combination" in str(result.error)
```

- [ ] **Step 4: Add a grid-grid obs-obs pair regression**

Append to `davinci_monet/tests/test_unified_sources_runtime.py`:

```python
def test_sources_config_supports_obs_obs_grid_pair(tmp_path: Path) -> None:
    from davinci_monet.pipeline.runner import PipelineRunner

    ref_path = tmp_path / "sat_ref.nc"
    comp_path = tmp_path / "sat_cmp.nc"
    _write_grid_source(ref_path, offset=0.0)
    _write_grid_source(comp_path, offset=1.0)

    config = {
        "analysis": {"output_dir": str(tmp_path / "out")},
        "sources": {
            "modis": {"type": "generic", "role": "obs", "files": str(ref_path)},
            "viirs": {"type": "generic", "role": "obs", "files": str(comp_path)},
        },
        "pairs": {
            "modis_viirs_o3": {
                "sources": ["modis", "viirs"],
                "reference": "modis",
                "variables": {"modis": "O3", "viirs": "O3"},
            }
        },
        "stats": {"metrics": ["N", "MB"]},
    }

    result = PipelineRunner(show_progress=False).run_from_config(config)

    assert result.success
    assert result.context is not None
    paired = result.context.paired["modis_viirs_o3"].data
    assert set(paired.data_vars) == {"modis_O3", "viirs_O3"}
    assert paired["modis_O3"].attrs["pair_role"] == "reference"
    assert paired["viirs_O3"].attrs["pair_role"] == "comparand"
    assert paired["modis_O3"].attrs["role"] == "obs"
    assert paired["viirs_O3"].attrs["role"] == "obs"
```

- [ ] **Step 5: Add schema tests for typed unified pairs**

Append to `davinci_monet/tests/unit/config/test_schema.py`:

```python
def test_monet_config_parses_unified_pairs() -> None:
    from davinci_monet.config.schema import MonetConfig

    cfg = MonetConfig.model_validate(
        {
            "sources": {
                "a": {"type": "generic", "files": "/tmp/a.nc"},
                "b": {"type": "generic", "files": "/tmp/b.nc"},
            },
            "pairs": {
                "a_b": {
                    "sources": ["a", "b"],
                    "reference": "a",
                    "variables": {"a": "O3", "b": "O3"},
                }
            },
        }
    )

    assert cfg.pairs["a_b"].sources == ["a", "b"]
    assert cfg.pairs["a_b"].reference == "a"
```

- [ ] **Step 6: Run the focused tests and confirm current failures**

Run:

```bash
pytest \
  davinci_monet/tests/test_unified_sources_runtime.py::test_single_model_source_gets_descriptive_stats \
  davinci_monet/tests/test_unified_sources_runtime.py::test_single_source_plot_uses_source_key_not_obs_key \
  davinci_monet/tests/test_unified_sources_runtime.py::test_unsupported_source_pair_fails_pairing_stage \
  davinci_monet/tests/test_unified_sources_runtime.py::test_sources_config_supports_obs_obs_grid_pair \
  davinci_monet/tests/unit/config/test_schema.py::test_monet_config_parses_unified_pairs \
  -v
```

Expected: at least the single-source model stats/plot, unsupported pair failure, and typed pair schema tests fail before implementation.

---

## Task 2: Type And Validate Unified Pair Config

**Files:**
- Modify: `davinci_monet/config/schema.py`
- Modify: `davinci_monet/config/parser.py`
- Test: `davinci_monet/tests/unit/config/test_schema.py`

- [ ] **Step 1: Read the target files**

Run:

```bash
sed -n '430,880p' davinci_monet/config/schema.py
sed -n '120,230p' davinci_monet/config/parser.py
```

- [ ] **Step 2: Add a flexible pair config union**

In `davinci_monet/config/schema.py`, replace `SourcePairConfig` with:

```python
class PairConfig(FlexibleModel):
    """Binary pair definition.

    Unified pairs use ``sources``/``reference``/``variables``. Legacy pairs use
    ``model``/``obs``/``variable`` and are retained for compatibility.
    """

    sources: list[str] = Field(default_factory=list)
    reference: str | None = None
    variables: dict[str, str] = Field(default_factory=dict)

    model: str | None = None
    obs: str | None = None
    variable: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_pair_shape(self) -> "PairConfig":
        has_sources = bool(self.sources)
        has_legacy = bool(self.model or self.obs or self.variable)
        if has_sources:
            if len(self.sources) != 2:
                raise ValueError("unified pair 'sources' must contain exactly two labels")
            if self.reference is not None and self.reference not in self.sources:
                raise ValueError("'reference' must be one of the pair sources")
            missing = [label for label in self.sources if label not in self.variables]
            if missing:
                raise ValueError(
                    "unified pair 'variables' missing source label(s): "
                    + ", ".join(missing)
                )
        elif has_legacy:
            if not self.model or not self.obs:
                raise ValueError("legacy pair must include both 'model' and 'obs'")
            if "model_var" not in self.variable or "obs_var" not in self.variable:
                raise ValueError("legacy pair 'variable' must include model_var and obs_var")
        return self
```

- [ ] **Step 3: Add `pairs` to `MonetConfig`**

In `MonetConfig`, add:

```python
    pairs: dict[str, PairConfig] = Field(default_factory=dict)
```

Place it after `sources`.

- [ ] **Step 4: Add a `parse_pairs` validator**

In `MonetConfig`, add this before `parse_plots`:

```python
    @field_validator("pairs", mode="before")
    @classmethod
    def parse_pairs(cls, v: Any) -> dict[str, PairConfig]:
        """Parse binary pair configurations."""
        if v is None:
            return {}
        if isinstance(v, dict):
            return {
                str(name): PairConfig(**cfg) if isinstance(cfg, dict) else cfg
                for name, cfg in v.items()
            }
        return dict(v)
```

- [ ] **Step 5: Ensure parser defaults include `sources` and `pairs`**

In `davinci_monet/config/parser.py`, change:

```python
    for section in ["model", "obs", "plots"]:
```

to:

```python
    for section in ["model", "obs", "sources", "pairs", "plots"]:
```

- [ ] **Step 6: Run schema tests**

Run:

```bash
pytest davinci_monet/tests/unit/config/test_schema.py::test_monet_config_parses_unified_pairs -v
pytest davinci_monet/tests/unit/config/test_schema.py -q
```

Expected: PASS.

---

## Task 3: Add Source Runtime Helpers And Stop Standard Stages From Reading Role Buckets

**Files:**
- Modify: `davinci_monet/pipeline/stages.py`
- Test: `davinci_monet/tests/test_load_sources_stage.py`
- Test: `davinci_monet/tests/test_unified_sources_runtime.py`

- [ ] **Step 1: Add helpers to `PipelineContext`**

In `davinci_monet/pipeline/stages.py`, inside `PipelineContext`, after `get_source()` add:

```python
    def iter_sources(self) -> list[tuple[str, Any]]:
        """Return loaded sources in insertion order."""
        return list(self.sources.items())

    def get_source_dataset(self, label: str) -> xr.Dataset:
        """Return the xarray Dataset for a source label."""
        source = self.get_source(label)
        data = source.data if hasattr(source, "data") else source
        if not isinstance(data, xr.Dataset):
            raise KeyError(f"Source '{label}' does not contain an xarray Dataset")
        return data

    def get_source_role(self, label: str) -> str | None:
        """Return optional source role metadata for a source label."""
        source = self.get_source(label)
        role = getattr(source, "role", None)
        if role:
            return str(role)
        data = source.data if hasattr(source, "data") else source
        if isinstance(data, xr.Dataset):
            raw = data.attrs.get("role")
            return str(raw) if raw else None
        return None
```

- [ ] **Step 2: Add a helper for plot/stat eligible single sources**

In `davinci_monet/pipeline/stages.py`, after `resolve_paired_var_names()` add:

```python
def iter_single_source_datasets(context: PipelineContext) -> list[tuple[str, xr.Dataset, str | None]]:
    """Return all loaded single sources as ``(label, dataset, role)`` triples."""
    sources = context.sources or {**context.models, **context.observations}
    out: list[tuple[str, xr.Dataset, str | None]] = []
    for label, obj in sources.items():
        data = obj.data if hasattr(obj, "data") else obj
        if not isinstance(data, xr.Dataset):
            continue
        role = getattr(obj, "role", None) or data.attrs.get("role")
        out.append((str(label), data, str(role) if role else None))
    return out
```

- [ ] **Step 3: Update `LoadSourcesStage._register_source` comments only**

Keep `context.models` and `context.observations` mirroring for compatibility, but update the docstring:

```python
    @staticmethod
    def _register_source(context: PipelineContext, label: str, obj: SourceData) -> None:
        """Register a loaded source.

        ``context.sources`` is canonical. Role-specific dictionaries are
        compatibility mirrors for legacy callers and must not be required by
        standard stages.
        """
```

- [ ] **Step 4: Run context/load tests**

Run:

```bash
pytest davinci_monet/tests/test_load_sources_stage.py davinci_monet/tests/test_pipeline.py::TestPipelineContext -q
```

Expected: PASS.

---

## Task 4: Make Single-Source Statistics Source-Neutral

**Files:**
- Modify: `davinci_monet/pipeline/stages.py`
- Test: `davinci_monet/tests/test_unified_sources_runtime.py`
- Test: `davinci_monet/tests/test_obs_pipeline.py`

- [ ] **Step 1: Change statistics validation**

In `StatisticsStage.validate`, replace:

```python
        return bool(context.paired) or bool(context.observations)
```

with:

```python
        return bool(context.paired) or bool(iter_single_source_datasets(context))
```

- [ ] **Step 2: Change descriptive stats iteration**

In `StatisticsStage._execute_descriptive`, replace:

```python
        for obs_label, obs_data in context.observations.items():
            ds = obs_data.data if hasattr(obs_data, "data") else obs_data
            obs_stats: dict[str, dict[str, float]] = {}
```

with:

```python
        for source_label, ds, _role in iter_single_source_datasets(context):
            source_stats: dict[str, dict[str, float]] = {}
```

Then replace all `obs_stats` references in that method with `source_stats`, and replace:

```python
            all_stats[obs_label] = obs_stats
```

with:

```python
            all_stats[source_label] = source_stats
```

- [ ] **Step 3: Change execute descriptive branch**

In `StatisticsStage.execute`, replace:

```python
        if not context.paired and context.observations:
            return self._execute_descriptive(context)
```

with:

```python
        if not context.paired and iter_single_source_datasets(context):
            return self._execute_descriptive(context)
```

- [ ] **Step 4: Run focused stats tests**

Run:

```bash
pytest \
  davinci_monet/tests/test_unified_sources_runtime.py::test_single_model_source_gets_descriptive_stats \
  davinci_monet/tests/test_obs_pipeline.py \
  -q
```

Expected: PASS.

---

## Task 5: Make Single-Source Plotting Source-Neutral

**Files:**
- Modify: `davinci_monet/pipeline/stages.py`
- Test: `davinci_monet/tests/test_unified_sources_runtime.py`
- Test: `davinci_monet/tests/test_obs_pipeline.py`

- [ ] **Step 1: Rename `_execute_obs` to `_execute_single_source`**

In `PlottingStage`, rename:

```python
    def _execute_obs(self, context: PipelineContext) -> StageResult:
```

to:

```python
    def _execute_single_source(self, context: PipelineContext) -> StageResult:
```

Update the docstring to:

```python
        """Single-source plotting.

        Renders plot specs against one source selected by ``source:``. The
        legacy ``obs:`` key remains accepted as an alias for old configs.
        """
```

- [ ] **Step 2: Change plot schema key set**

In that method, change `_SCHEMA_KEYS` from:

```python
            "obs",
```

to:

```python
            "source",
            "obs",
```

- [ ] **Step 3: Resolve source labels from `source:` first**

Replace:

```python
            if not plot_type or "obs" not in plot_spec:
                continue

            obs_label = plot_spec.get("obs", "")
            variable = plot_spec.get("variable", "")

            if obs_label not in context.observations:
                errors.append(f"Observation '{obs_label}' not found for plot '{plot_name}'")
                continue

            obs_data = context.observations[obs_label]
            ds = obs_data.data if hasattr(obs_data, "data") else obs_data
```

with:

```python
            source_label = str(plot_spec.get("source") or plot_spec.get("obs") or "")
            if not plot_type or not source_label:
                continue

            variable = plot_spec.get("variable", "")
            source_map = {
                label: (ds, role)
                for label, ds, role in iter_single_source_datasets(context)
            }
            source_entry = source_map.get(source_label)
            if source_entry is None:
                errors.append(f"Source '{source_label}' not found for plot '{plot_name}'")
                continue

            ds, source_role = source_entry
```

- [ ] **Step 4: Replace remaining `obs_label` references**

Within `_execute_single_source`, replace:

```python
obs_label
```

with:

```python
source_label
```

except in comments that explicitly mention the legacy `obs:` key.

- [ ] **Step 5: Preserve the actual role when tagging**

Replace:

```python
                        tag_source_roles(subset, role="obs", source_label=obs_label)
```

with:

```python
                        tag_source_roles(subset, role=source_role, source_label=source_label)
```

- [ ] **Step 6: Update `PlottingStage.validate` and execute branch**

Replace:

```python
        return bool(context.paired) or bool(context.observations)
```

with:

```python
        return bool(context.paired) or bool(iter_single_source_datasets(context))
```

Replace:

```python
        if not context.paired and context.observations:
            return self._execute_obs(context)
```

with:

```python
        if not context.paired and iter_single_source_datasets(context):
            return self._execute_single_source(context)
```

- [ ] **Step 7: Run focused plot tests**

Run:

```bash
pytest \
  davinci_monet/tests/test_unified_sources_runtime.py::test_single_source_plot_uses_source_key_not_obs_key \
  davinci_monet/tests/test_obs_pipeline.py \
  -q
```

Expected: PASS.

---

## Task 6: Make Configured Source Pair Failures Fatal

**Files:**
- Modify: `davinci_monet/pipeline/stages.py`
- Test: `davinci_monet/tests/test_unified_sources_runtime.py`
- Test: `davinci_monet/tests/test_pair_direction.py`

- [ ] **Step 1: Track execution errors in `_execute_source_pair_jobs`**

Inside `_execute_source_pair_jobs`, after `paired_count = 0`, add:

```python
        execution_errors: list[str] = []
```

- [ ] **Step 2: Append errors for missing datasets**

Replace:

```python
                context.metadata.setdefault("pairing_errors", []).append(
                    f"{job.pair_key}: reference or comparand data is None"
                )
```

with:

```python
                message = f"{job.pair_key}: reference or comparand data is None"
                execution_errors.append(message)
                context.metadata.setdefault("pairing_errors", []).append(message)
```

- [ ] **Step 3: Append errors for exceptions**

Replace:

```python
            except Exception as e:
                context.metadata.setdefault("pairing_errors", []).append(f"{job.pair_key}: {e}")
                context.log_progress(f"    parallel_completed: {job.pair_key} - FAILED")
```

with:

```python
            except Exception as e:
                message = f"{job.pair_key}: {e}"
                execution_errors.append(message)
                context.metadata.setdefault("pairing_errors", []).append(message)
                context.log_progress(f"    parallel_completed: {job.pair_key} - FAILED")
```

- [ ] **Step 4: Return failed status when configured jobs fail**

Before the final `return self._create_result(...)`, add:

```python
        if execution_errors:
            return self._create_result(
                StageStatus.FAILED,
                data={"paired_keys": list(context.paired.keys())},
                error="Source pair execution failed: " + "; ".join(execution_errors),
                duration=time.time() - start,
                count=paired_count,
            )
```

- [ ] **Step 5: Run focused failure tests**

Run:

```bash
pytest \
  davinci_monet/tests/test_unified_sources_runtime.py::test_unsupported_source_pair_fails_pairing_stage \
  davinci_monet/tests/test_unified_sources_runtime.py::test_invalid_sources_pair_unknown_source_fails \
  davinci_monet/tests/test_unified_sources_runtime.py::test_invalid_sources_pair_missing_variable_fails \
  -q
```

Expected: PASS.

---

## Task 7: Promote Reference/Comparand On `PairedData`

**Files:**
- Modify: `davinci_monet/core/base.py`
- Modify: `davinci_monet/pairing/engine.py`
- Test: `davinci_monet/tests/unit/core/test_base.py`
- Test: `davinci_monet/tests/test_unified_sources_runtime.py`

- [ ] **Step 1: Add neutral fields to `PairedData`**

In `davinci_monet/core/base.py`, update `PairedData` fields to:

```python
    data: xr.Dataset
    reference_label: str = "reference"
    comparand_label: str = "comparand"
    geometry: DataGeometry = DataGeometry.GRID
    pairing_info: dict[str, Any] = field(default_factory=dict)
```

- [ ] **Step 2: Add compatibility properties**

Inside `PairedData`, add:

```python
    @property
    def obs_label(self) -> str:
        """Deprecated alias for ``reference_label``."""
        return self.reference_label

    @property
    def model_label(self) -> str:
        """Deprecated alias for ``comparand_label``."""
        return self.comparand_label
```

- [ ] **Step 3: Add neutral variable properties**

Replace `model_variables` and `obs_variables` internals with neutral properties:

```python
    @property
    def reference_variables(self) -> list[str]:
        """List of reference-role variables in the paired data."""
        return [
            str(v)
            for v in self.data.data_vars
            if paired_variable_pair_role(self.data, str(v)) == "reference"
        ]

    @property
    def comparand_variables(self) -> list[str]:
        """List of comparand-role variables in the paired data."""
        return [
            str(v)
            for v in self.data.data_vars
            if paired_variable_pair_role(self.data, str(v)) == "comparand"
        ]

    @property
    def obs_variables(self) -> list[str]:
        """Deprecated alias for ``reference_variables``."""
        return self.reference_variables

    @property
    def model_variables(self) -> list[str]:
        """Deprecated alias for ``comparand_variables``."""
        return self.comparand_variables
```

- [ ] **Step 4: Add neutral getters**

Add:

```python
    def get_reference(self, variable: str) -> xr.DataArray:
        """Get reference variable by exact, source-label, legacy, or canonical name."""
        name = self._resolve_pair_role_var(variable, "reference")
        if name is None:
            raise VariableNotFoundError(
                f"Reference variable '{variable}' not found. "
                f"Available: {self.reference_variables}"
            )
        return self.data[name]

    def get_comparand(self, variable: str) -> xr.DataArray:
        """Get comparand variable by exact, source-label, legacy, or canonical name."""
        name = self._resolve_pair_role_var(variable, "comparand")
        if name is None:
            raise VariableNotFoundError(
                f"Comparand variable '{variable}' not found. "
                f"Available: {self.comparand_variables}"
            )
        return self.data[name]
```

Rename `_resolve_role_var` to `_resolve_pair_role_var` and make it accept `"reference"`/`"comparand"` directly. Keep `get_obs()` and `get_model()` as wrappers:

```python
    def get_obs(self, variable: str) -> xr.DataArray:
        """Deprecated alias for ``get_reference``."""
        return self.get_reference(variable)

    def get_model(self, variable: str) -> xr.DataArray:
        """Deprecated alias for ``get_comparand``."""
        return self.get_comparand(variable)
```

- [ ] **Step 5: Update `PairingEngine` constructors**

In `davinci_monet/pairing/engine.py`, change `PairedData(...)` calls:

```python
        return PairedData(
            data=result_ds,
            reference_label=obs_label,
            comparand_label=model_label,
            geometry=geometry,
            pairing_info={...},
        )
```

and:

```python
        return PairedData(
            data=result_ds,
            reference_label=reference_label,
            comparand_label=comparand_label,
            geometry=reference_geometry,
            pairing_info={...},
        )
```

- [ ] **Step 6: Update `subset_time` constructor**

In `PairedData.subset_time`, replace `model_label=` and `obs_label=` with:

```python
            reference_label=self.reference_label,
            comparand_label=self.comparand_label,
```

- [ ] **Step 7: Run core tests**

Run:

```bash
pytest davinci_monet/tests/unit/core/test_base.py davinci_monet/tests/test_unified_sources_runtime.py -q
```

Expected: PASS.

---

## Task 8: Make Statistics Interfaces And Output Labels Neutral

**Files:**
- Modify: `davinci_monet/stats/calculator.py`
- Modify: `davinci_monet/pipeline/stages.py`
- Modify: `davinci_monet/stats/__init__.py`
- Test: `davinci_monet/tests/test_stats.py`
- Test: `davinci_monet/tests/test_unified_sources_runtime.py`

- [ ] **Step 1: Add neutral parameters to `StatisticsCalculator.compute`**

In `davinci_monet/stats/calculator.py`, change the signature to:

```python
    def compute(
        self,
        paired_data: xr.Dataset,
        reference_var: str | None = None,
        comparand_var: str | None = None,
        metrics: Sequence[str] | None = None,
        groupby: str | Sequence[str] | None = None,
        *,
        obs_var: str | None = None,
        model_var: str | None = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
```

At the top of the method add:

```python
        if reference_var is None:
            reference_var = obs_var
        if comparand_var is None:
            comparand_var = model_var
        if reference_var is None or comparand_var is None:
            raise ValueError("reference_var and comparand_var are required")
```

Then replace:

```python
        obs_data = paired_data[obs_var]
        model_data = paired_data[model_var]
```

with:

```python
        reference_data = paired_data[reference_var]
        comparand_data = paired_data[comparand_var]
```

and pass those variables to grouped/overall helpers.

- [ ] **Step 2: Rename helper parameters internally**

In `StatisticsCalculator`, rename internal local variables from `obs_data`/`model_data` to `reference_data`/`comparand_data`. Keep DataFrame temporary column names `obs`/`mod` only if changing them would alter metric behavior; otherwise use `reference`/`comparand`.

- [ ] **Step 3: Update `StatisticsStage._calculate_stats` call**

Replace:

```python
            df = calculator.compute(
                paired_data,
                obs_var=obs_var,
                model_var=model_var,
                metrics=list(metrics) if metrics else None,
            )
```

with:

```python
            df = calculator.compute(
                paired_data,
                reference_var=obs_var,
                comparand_var=model_var,
                metrics=list(metrics) if metrics else None,
            )
```

- [ ] **Step 4: Write neutral CSV columns with legacy aliases**

In `SaveResultsStage.execute`, replace:

```python
                    row["Mean_Obs"] = _get_metric(var_stats, "MO", "obs_mean")
                    row["Mean_Model"] = _get_metric(var_stats, "MP", "model_mean")
```

with:

```python
                    row["Mean_Reference"] = _get_metric(var_stats, "MO", "obs_mean")
                    row["Mean_Comparand"] = _get_metric(var_stats, "MP", "model_mean")
                    row["Mean_Obs"] = row["Mean_Reference"]
                    row["Mean_Model"] = row["Mean_Comparand"]
```

This keeps existing downstream CSV consumers alive while making the canonical columns neutral.

- [ ] **Step 5: Update NMB fallback to neutral variable**

Replace:

```python
                    obs_mean = row["Mean_Obs"]
```

with:

```python
                    reference_mean = row["Mean_Reference"]
```

and update the fallback expression to use `reference_mean`.

- [ ] **Step 6: Run stats tests**

Run:

```bash
pytest davinci_monet/tests/test_stats.py davinci_monet/tests/test_stats_coverage.py davinci_monet/tests/test_unified_sources_runtime.py -q
```

Expected: PASS.

---

## Task 9: Make Comparison Plot Dispatch Source-Neutral

**Files:**
- Modify: `davinci_monet/pipeline/stages.py`
- Modify: `davinci_monet/plots/base.py`
- Test: `davinci_monet/tests/test_unified_sources_runtime.py`
- Test: `davinci_monet/tests/test_plots.py`
- Test: `davinci_monet/tests/test_renderer_role_styling.py`

- [ ] **Step 1: Rename local variables in `PlottingStage.execute`**

In the paired-data branch of `PlottingStage.execute`, rename local variables:

| Old local | New local |
|-----------|-----------|
| `obs_label` | `reference_label` |
| `model_label` | `comparand_label` |
| `obs_var` | `reference_var` |
| `model_var` | `comparand_var` |
| `obs_var_name` | `reference_var_name` |
| `model_var_name` | `comparand_var_name` |
| `obs_output_dir` | `reference_output_dir` |

Do not change public plotter method calls yet; call them with the neutral variables in positional order:

```python
fig = plotter.plot(
    paired_data, reference_var_name, comparand_var_name, **plot_options
)
```

- [ ] **Step 2: Resolve unified source-pair labels by pair role**

In the `"sources" in pair_spec` branch, use `iter_paired_variable_pairs(paired_data)` to select:

```python
reference_var_name, comparand_var_name, canonical_var = pair_vars[0]
reference_label = str(
    paired_data[reference_var_name].attrs.get("source_label", "reference")
)
comparand_label = str(
    paired_data[comparand_var_name].attrs.get("source_label", "comparand")
)
reference_var = canonical_var
comparand_var = canonical_var
var_spec = {"reference_var": reference_var, "comparand_var": comparand_var}
```

Keep legacy `var_spec["obs_var"]` and `var_spec["model_var"]` fallback when processing legacy pair specs.

- [ ] **Step 3: Make variable plot config source-aware**

Replace model-only variable config lookup with this helper:

```python
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
                value = variables[variable_name]
                return value.model_dump(exclude_none=True) if hasattr(value, "model_dump") else dict(value)

        for legacy_block in ("model", "obs"):
            legacy_cfg = config.get(legacy_block, {})
            source_cfg = legacy_cfg.get(source_label, {}) if isinstance(legacy_cfg, dict) else {}
            variables = source_cfg.get("variables", {}) if isinstance(source_cfg, dict) else {}
            if isinstance(variables, dict) and variable_name in variables:
                value = variables[variable_name]
                return value.model_dump(exclude_none=True) if hasattr(value, "model_dump") else dict(value)
        return {}
```

Use it for the comparand first, then reference as fallback:

```python
var_config = self._source_var_config(context.config, comparand_label, comparand_var)
if not var_config:
    var_config = self._source_var_config(context.config, reference_label, reference_var)
```

- [ ] **Step 4: Make `spatial_overlay` use source datasets**

Replace:

```python
model_obj = context.models.get(model_label)
```

with:

```python
source_obj = context.sources.get(comparand_label) or context.sources.get(reference_label)
```

Then use:

```python
source_ds = source_obj.data if hasattr(source_obj, "data") else source_obj
field_var = comparand_var if comparand_var in getattr(source_ds, "data_vars", {}) else reference_var
if source_ds is not None and field_var in getattr(source_ds, "data_vars", {}):
    plot_options["model_field"] = source_ds[field_var]
```

Keep the `model_field` option name for renderer compatibility in this task.

- [ ] **Step 5: Use reference label for output directory**

Replace output directory creation with:

```python
reference_output_dir = output_dir / reference_label
reference_output_dir.mkdir(parents=True, exist_ok=True)
```

Use `reference_output_dir` for saved paths.

- [ ] **Step 6: Allow neutral plot label overrides**

When forwarding label overrides, support both new and legacy keys:

```python
label_aliases = {
    "reference_label": "obs_label",
    "comparand_label": "model_label",
    "obs_label": "obs_label",
    "model_label": "model_label",
}
for input_key, plotter_key in label_aliases.items():
    if input_key in plot_spec:
        plotter_config[plotter_key] = plot_spec[input_key]
```

- [ ] **Step 7: Run plot tests**

Run:

```bash
pytest \
  davinci_monet/tests/test_unified_sources_runtime.py \
  davinci_monet/tests/test_plots.py \
  davinci_monet/tests/test_renderer_role_styling.py \
  -q
```

Expected: PASS.

---

## Task 10: Make Legacy MODIS Grid Target Source-Neutral

**Files:**
- Modify: `davinci_monet/pipeline/stages.py`
- Modify: `davinci_monet/config/schema.py`
- Test: `davinci_monet/tests/test_unified_sources_runtime.py`
- Test: `davinci_monet/tests/test_observation_readers.py`

- [ ] **Step 1: Update config docstring language**

In `ObservationConfig` docstring for `grid_source`, replace:

```text
Model label whose grid is used as the binning target for swath observations
```

with:

```text
Source label whose grid is used as the binning target for swath observations
```

- [ ] **Step 2: Update `_load_modis_l2` target lookup**

In `LoadObservationsStage._load_modis_l2`, replace:

```python
        model_obj = context.models.get(grid_source)
        if model_obj is None:
            context.log_progress(f"done: grid_source model '{grid_source}' not loaded, skipping")
            return None

        model_ds = model_obj.data if hasattr(model_obj, "data") else model_obj
```

with:

```python
        grid_obj = (
            context.sources.get(grid_source)
            or context.models.get(grid_source)
            or context.observations.get(grid_source)
        )
        if grid_obj is None:
            context.log_progress(f"done: grid_source '{grid_source}' not loaded, skipping")
            return None

        grid_ds = grid_obj.data if hasattr(grid_obj, "data") else grid_obj
```

Rename the subsequent local uses in this method from `model_ds` to `grid_ds`, including the latitude and longitude extraction.

- [ ] **Step 3: Run observation tests**

Run:

```bash
pytest davinci_monet/tests/test_observation_readers.py davinci_monet/tests/test_unified_sources_runtime.py -q
```

Expected: PASS.

---

## Task 11: Make Runner Logs And Cleanup Source-Neutral

**Files:**
- Modify: `davinci_monet/pipeline/runner.py`
- Modify: `davinci_monet/pipeline/stages.py`
- Test: `davinci_monet/tests/test_pipeline.py`
- Test: `davinci_monet/tests/test_cli_e2e.py`

- [ ] **Step 1: Teach progress callback about `Loading source:`**

In `LoadSourcesStage._load_unified_source`, add a progress message before reader open:

```python
        context.log_progress(f"    Loading source: {label}")
```

In `ProgressFormatter` and `LogCollector`, add parsing branches for `Loading source:` that categorize the item as `"source"`.

- [ ] **Step 2: Extract source details from `context.sources`**

In `LogCollector.extract_context_data`, add:

```python
        self.source_details: dict[str, dict[str, Any]] = {}
        for label, source_data in context.sources.items():
            details = {}
            ds = source_data.data if hasattr(source_data, "data") else source_data
            if ds is not None:
                details["variables"] = len(ds.data_vars)
                details["time_steps"] = ds.sizes.get("time", "-")
                details["role"] = getattr(source_data, "role", None) or ds.attrs.get("role", "-")
                total_size = sum(ds[v].size for v in ds.data_vars)
                details["data_points"] = total_size
            self.source_details[label] = details
```

Initialize `self.source_details` in `LogCollector.__init__`.

- [ ] **Step 3: Render a `Sources Loaded` table**

In `LogCollector.to_markdown`, add a `Sources Loaded` table before legacy model/observation tables. Use columns:

```text
Source | Role | Variables | Time Steps | Data Points
```

Keep existing model/observation tables only when those entries exist and `sources` is empty in the run config.

- [ ] **Step 4: Close datasets through `context.sources`**

In `_cleanup_context_datasets`, replace the separate model and observation loops with:

```python
        seen: set[int] = set()
        for _label, source_data in list(context.sources.items()):
            try:
                obj_id = id(source_data)
                if obj_id in seen:
                    continue
                seen.add(obj_id)
                if hasattr(source_data, "data") and hasattr(source_data.data, "close"):
                    source_data.data.close()
                elif hasattr(source_data, "close"):
                    source_data.close()
            except Exception:
                pass
```

Then keep fallback loops over `context.models` and `context.observations` only for contexts built manually without `sources`.

- [ ] **Step 5: Run runner tests**

Run:

```bash
pytest davinci_monet/tests/test_pipeline.py davinci_monet/tests/test_cli_e2e.py -q
```

Expected: PASS.

---

## Task 12: Update Config Builder And Validate CLI

**Files:**
- Modify: `davinci_monet/config/parser.py`
- Modify: `davinci_monet/cli/commands/validate.py`
- Test: `davinci_monet/tests/unit/config/test_parser.py`
- Test: `davinci_monet/tests/test_cli.py`

- [ ] **Step 1: Add source builder methods**

In `ConfigBuilder.__init__`, include:

```python
            "sources": {},
            "pairs": {},
```

Add methods:

```python
    def add_source(self, name: str, **kwargs: Any) -> "ConfigBuilder":
        """Add a unified source configuration."""
        self._data["sources"][name] = kwargs
        return self

    def add_pair(
        self,
        name: str,
        sources: list[str],
        variables: dict[str, str],
        reference: str | None = None,
        **kwargs: Any,
    ) -> "ConfigBuilder":
        """Add a unified binary pair configuration."""
        pair: dict[str, Any] = {"sources": sources, "variables": variables}
        if reference is not None:
            pair["reference"] = reference
        pair.update(kwargs)
        self._data["pairs"][name] = pair
        return self
```

Keep `add_model()` and `add_observation()` as compatibility methods.

- [ ] **Step 2: Print sources in `validate` CLI**

In `davinci_monet/cli/commands/validate.py`, after analysis summary add:

```python
        if config.sources:
            typer.echo(f"  Sources: {len(config.sources)} defined")
            for name, source_cfg in config.sources.items():
                role = f" ({source_cfg.role})" if source_cfg.role else ""
                typer.echo(f"    - {name}: {source_cfg.type}{role}")
```

Change the model/observation sections to run only when `config.sources` is empty:

```python
        if not config.sources and config.model:
            ...
        if not config.sources and config.obs:
            ...
```

- [ ] **Step 3: Print pairs in `validate` CLI**

Add:

```python
        if config.pairs:
            typer.echo(f"  Pairs: {len(config.pairs)} defined")
            for name, pair_cfg in config.pairs.items():
                if pair_cfg.sources:
                    typer.echo(f"    - {name}: {', '.join(pair_cfg.sources)}")
                else:
                    typer.echo(f"    - {name}: {pair_cfg.model}, {pair_cfg.obs}")
```

- [ ] **Step 4: Run parser and CLI tests**

Run:

```bash
pytest davinci_monet/tests/unit/config/test_parser.py davinci_monet/tests/test_cli.py -q
```

Expected: PASS.

---

## Task 13: Update Protocols And Public Docs Without Breaking Imports

**Files:**
- Modify: `davinci_monet/core/protocols.py`
- Modify: `davinci_monet/io/writers.py`
- Modify: `davinci_monet/plots/__init__.py`
- Modify: `davinci_monet/stats/__init__.py`
- Test: `davinci_monet/tests/unit/core/test_source_abstraction.py`
- Test: `davinci_monet/tests/unit/core/test_registry.py`

- [ ] **Step 1: Mark old protocols as compatibility**

In `core/protocols.py`, update comments above `ModelReader`, `ModelProcessor`, `ObservationReader`, and `ObservationProcessor` to:

```python
# =============================================================================
# Legacy Compatibility Protocols
# =============================================================================
#
# New source readers should implement SourceReader. These protocols are retained
# for downstream type imports and legacy adapters only.
```

- [ ] **Step 2: Update `PairingStrategy` docs**

Change `PairingStrategy` class docstring to describe `pair_sources()` as canonical and `pair()` as a legacy model/obs adapter. Do not remove the `pair()` method yet.

- [ ] **Step 3: Update package examples**

Replace examples like:

```python
plotter.plot(paired_data, "obs_o3", "model_o3")
```

with:

```python
plotter.plot(paired_data, "reference_o3", "comparand_o3")
```

When actual test fixtures still use `obs_`/`model_`, update docstrings only; do not alter tests unnecessarily.

- [ ] **Step 4: Run protocol/import tests**

Run:

```bash
pytest davinci_monet/tests/unit/core/test_source_abstraction.py davinci_monet/tests/unit/core/test_registry.py -q
```

Expected: PASS.

---

## Task 14: Sweep Renderer Internals For Semantic Model/Obs Dependencies

**Files:**
- Modify after the Step 2 classification identifies behavior-sensitive residuals:
  - `davinci_monet/plots/renderers/timeseries.py`
  - `davinci_monet/plots/renderers/scatter.py`
  - `davinci_monet/plots/renderers/diurnal.py`
  - `davinci_monet/plots/renderers/flight_timeseries.py`
  - `davinci_monet/plots/renderers/spatial/bias.py`
  - `davinci_monet/plots/renderers/spatial/overlay.py`
  - `davinci_monet/plots/renderers/spatial/distribution.py`
  - `davinci_monet/plots/renderers/boxplot.py`
  - `davinci_monet/plots/renderers/taylor.py`
  - `davinci_monet/plots/renderers/site_timeseries.py`
  - `davinci_monet/plots/renderers/per_site_timeseries.py`
- Test:
  - `davinci_monet/tests/test_plots.py`
  - `davinci_monet/tests/unit/plots/`

- [ ] **Step 1: Search renderer residuals**

Run:

```bash
rg -n "obs_var|model_var|obs_label|model_label|Mean Obs|Mean Model|Observed|Modeled|model_field|show_var" davinci_monet/plots/renderers davinci_monet/plots/base.py
```

- [ ] **Step 2: Classify residuals**

For each match, classify it as one of:

```text
compatibility API: keep, but docstring says deprecated alias
visual role styling: keep only if role metadata is intentionally used
pair semantics: rename to reference/comparand
hard model/obs behavior: replace with source/reference/comparand logic
```

Record the classification in the task notes before editing.

- [ ] **Step 3: Update behavior-sensitive renderers**

For renderers that compute differences, regression, axis labels, or source-specific styling, use local variable names:

```python
reference_data = paired_data[reference_var]
comparand_data = paired_data[comparand_var]
```

Keep the public `plot(paired_data, obs_var, model_var, ...)` signature until a later major API break, but immediately assign:

```python
reference_var = obs_var
comparand_var = model_var
```

This removes internal semantic coupling without breaking callers.

- [ ] **Step 4: Update labels to source labels**

Where a label fallback says `"Observations"` or `"Model"`, replace it with:

```python
get_series_label(paired_data, reference_var, self.config.obs_label) or "Reference"
get_series_label(paired_data, comparand_var, self.config.model_label) or "Comparand"
```

Keep `self.config.obs_label` and `self.config.model_label` as compatibility config fields until plot config is versioned.

- [ ] **Step 5: Run plot suite**

Run:

```bash
pytest davinci_monet/tests/test_plots.py davinci_monet/tests/unit/plots -q
```

Expected: PASS.

---

## Task 15: Clean Up Migration Helpers And Legacy Expansion

**Files:**
- Modify: `davinci_monet/config/migration.py`
- Modify: `davinci_monet/tests/unit/config/test_expand_sources.py`
- Modify: `davinci_monet/tests/unit/config/test_migrate_to_sources.py`

- [ ] **Step 1: Confirm `expand_sources_to_legacy` is unused by runtime**

Run:

```bash
rg -n "expand_sources_to_legacy" davinci_monet tests docs
```

Expected: only config exports, unit tests, and docs/plans reference it.

- [ ] **Step 2: Mark helper as test/support compatibility**

Update `expand_sources_to_legacy` docstring first paragraph to:

```python
    """Expand a unified ``sources:`` config back to legacy form for tests and
    downstream compatibility helpers.

    The standard runtime no longer uses this path. It is intentionally kept as
    a boundary shim for consumers that still need old ``model:``/``obs:``
    dictionaries.
    """
```

- [ ] **Step 3: Remove model/obs inference from standard docs**

Keep `MODEL_SOURCE_TYPES` only for `expand_sources_to_legacy`; update its comment to say it is not used by runtime source loading.

- [ ] **Step 4: Run migration tests**

Run:

```bash
pytest davinci_monet/tests/unit/config/test_expand_sources.py davinci_monet/tests/unit/config/test_migrate_to_sources.py -q
```

Expected: PASS.

---

## Task 16: Update AI Payload, Log Output, And Project Docs

**Files:**
- Modify: `davinci_monet/ai/payload.py`
- Modify: `CLAUDE.md`
- Modify: `README.md`
- Test: `davinci_monet/tests/integration/test_ai_summary_pipeline.py`
- Test: `davinci_monet/tests/integration/test_ai_summary_openrouter_pipeline.py`

- [ ] **Step 1: Update AI payload text**

In `davinci_monet/ai/payload.py`, keep `sources_summary` logic but change stats row key from `"pair"` to `"comparison"` while preserving `"pair"` as alias:

```python
stats_rows.append(
    {
        "comparison": pair_key,
        "pair": pair_key,
        "variable": var_name,
        "metrics": metrics,
    }
)
```

- [ ] **Step 2: Update `CLAUDE.md` architecture status**

Replace the current hybrid language with:

```markdown
## Unified Data-Source Runtime

The standard runtime uses `sources:` internally. `model:` and `obs:` remain
accepted legacy config blocks and are converted into source containers by
`LoadSourcesStage`; role-specific context dictionaries are compatibility mirrors
only. Pairing uses `reference` and `comparand`; `role: model|obs` is optional
metadata for labels and default colors, not pairing direction.
```

- [ ] **Step 3: Update variable naming docs**

Document:

```markdown
Paired datasets use `<source_label>_<canonical_var>` variables. Each variable
has `pair_role: reference|comparand`, optional `role`, and `source_label`.
Legacy `obs_`/`model_` names are accepted only by low-level compatibility APIs.
```

- [ ] **Step 4: Run AI summary tests**

Run:

```bash
pytest \
  davinci_monet/tests/integration/test_ai_summary_pipeline.py \
  davinci_monet/tests/integration/test_ai_summary_openrouter_pipeline.py \
  -q
```

Expected: PASS, with tests skipped if API keys are not configured.

---

## Task 17: Apply Source Processing (Resample) In `LoadSourcesStage`

**Files:**
- Modify: `davinci_monet/observations/base.py` (add module-level `resample_dataset`; refactor `ObservationData.resample_data` to call it)
- Modify: `davinci_monet/config/schema.py` (`SourceConfig`)
- Modify: `davinci_monet/pipeline/stages.py` (`LoadSourcesStage._load_unified_source`)
- Test: `davinci_monet/tests/test_unified_sources_runtime.py`

**Why:** `resample`/`min_obs_count`/`track_obs_count` are applied only in the legacy `LoadObservationsStage`; the unified `sources:` path (`_load_unified_source`) never calls them, so a high-frequency source migrated to `sources:` silently skips averaging. This closes that regression before WS3 deletes the legacy stage.

- [ ] **Step 1: Write the failing regression test**

Append to `davinci_monet/tests/test_unified_sources_runtime.py`:

```python
def test_unified_source_applies_resample(tmp_path: Path) -> None:
    """A `sources:` obs with `resample` is averaged to the target frequency at load."""
    import numpy as np
    import pandas as pd
    import xarray as xr

    from davinci_monet.pipeline.runner import PipelineRunner

    src = tmp_path / "hf.nc"
    times = pd.date_range("2024-01-01T00:00", periods=4, freq="15min")
    ds = xr.Dataset(
        {"o3": (("time", "site"), np.array([[10.0], [20.0], [30.0], [40.0]]))},
        coords={
            "time": times,
            "site": [0],
            "latitude": ("site", [40.0]),
            "longitude": ("site", [-105.0]),
        },
        attrs={"geometry": "point"},
    )
    ds.to_netcdf(src)

    config = {
        "analysis": {"output_dir": str(tmp_path / "out")},
        "sources": {
            "hf": {
                "type": "generic",
                "role": "obs",
                "files": str(src),
                "resample": "h",
                "track_obs_count": True,
                "variables": {"o3": {"units": "ppb"}},
            }
        },
    }

    result = PipelineRunner(show_progress=False).run_from_config(config)

    assert result.success
    assert result.context is not None
    loaded = result.context.sources["hf"].data
    assert loaded.sizes["time"] == 1
    assert float(loaded["o3"].isel(time=0, site=0)) == 25.0
    assert "obs_count" in loaded
    assert int(loaded["obs_count"].isel(time=0, site=0)) == 4
```

- [ ] **Step 2: Run it and confirm it fails**

Run:

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate davinci
HDF5_USE_FILE_LOCKING=FALSE pytest \
  davinci_monet/tests/test_unified_sources_runtime.py::test_unified_source_applies_resample -v
```

Expected: FAIL — `loaded.sizes["time"]` is 4 (no resampling) and `obs_count` is absent.

- [ ] **Step 3: Add a reusable `resample_dataset` function**

In `davinci_monet/observations/base.py`, add at module level (e.g. just above the `ObservationData` class):

```python
def resample_dataset(
    data: "xr.Dataset",
    freq: str,
    min_count: int | None = None,
    track_count: bool = False,
) -> "xr.Dataset":
    """Resample a dataset along ``time``, masking sparse bins and optionally counting.

    Pure-function form of :meth:`ObservationData.resample_data` so the unified
    source loader can resample bare datasets without an ``ObservationData`` wrapper.
    """
    if "time" not in data.dims:
        return data
    resampler = data.resample(time=freq)
    result = resampler.mean()
    if track_count or min_count is not None:
        data_vars = [
            v for v in data.data_vars if v not in ("latitude", "longitude", "altitude")
        ]
        if data_vars:
            counts = resampler.count()[data_vars[0]]
            if track_count:
                result["obs_count"] = counts
            if min_count is not None:
                mask = counts >= min_count
                for var in data_vars:
                    if var in result:
                        result[var] = result[var].where(mask)
    return result
```

- [ ] **Step 4: Refactor `ObservationData.resample_data` to delegate (DRY)**

In `davinci_monet/observations/base.py`, replace the body of `resample_data` (the block from `if self.data is None:` through `self.data = result`) with:

```python
        if self.data is None:
            return
        if freq is None:
            freq = self.resample
        if freq is None:
            return
        self.data = resample_dataset(
            self.data, freq, min_count=min_count, track_count=track_count
        )
```

- [ ] **Step 5: Add typed fields to `SourceConfig`**

In `davinci_monet/config/schema.py`, in `SourceConfig`, after the `display_name` field add:

```python
    resample: str | None = None
    min_obs_count: int | None = None
    track_obs_count: bool = False
```

- [ ] **Step 6: Apply resample in the unified loader**

In `davinci_monet/pipeline/stages.py`, in `LoadSourcesStage._load_unified_source`:

(a) add the three keys to `passthrough_keys` so they are consumed by the stage and never forwarded to `reader.open`:

```python
        passthrough_keys = {
            "type",
            "role",
            "files",
            "filename",
            "variables",
            "radius_of_influence",
            "mapping",
            "display_name",
            "resample",
            "min_obs_count",
            "track_obs_count",
        }
```

(b) after the `if variables:` variable-config block and before `geometry = self._data_geometry(getattr(reader, "geometry"))`, add:

```python
        resample_freq = cfg.get("resample")
        if resample_freq:
            from davinci_monet.observations.base import resample_dataset

            data = resample_dataset(
                data,
                str(resample_freq),
                min_count=cfg.get("min_obs_count"),
                track_count=bool(cfg.get("track_obs_count")),
            )
```

- [ ] **Step 7: Run the test and confirm it passes**

Run:

```bash
HDF5_USE_FILE_LOCKING=FALSE pytest \
  davinci_monet/tests/test_unified_sources_runtime.py::test_unified_source_applies_resample -v
```

Expected: PASS.

- [ ] **Step 8: Run the focused suite + quality gates**

Run:

```bash
HDF5_USE_FILE_LOCKING=FALSE pytest davinci_monet/tests/test_unified_sources_runtime.py -q
mypy davinci_monet/observations/base.py davinci_monet/config/schema.py davinci_monet/pipeline/stages.py
black --check davinci_monet && isort --check davinci_monet
```

Expected: all pass.

- [ ] **Step 9: Commit** (only if the user has explicitly approved committing — see Repo Rules)

```bash
git add davinci_monet/observations/base.py davinci_monet/config/schema.py \
  davinci_monet/pipeline/stages.py davinci_monet/tests/test_unified_sources_runtime.py
git commit -m "feat(sources): apply resample/min_obs_count/track_obs_count in unified loader"
```

---

## Task 18: Final Residual Audit

**Files:**
- Modify only files where this audit finds live runtime issues.
- Test: full suite.

- [ ] **Step 1: Search for role-bucket runtime dependencies**

Run:

```bash
rg -n "context\\.models|context\\.observations|get_model\\(|get_observation\\(" davinci_monet -g '*.py'
```

Expected allowed matches:

```text
davinci_monet/pipeline/stages.py compatibility mirroring and legacy fallback only
davinci_monet/pipeline/runner.py fallback cleanup only
tests for legacy compatibility
```

Any standard pairing/stats/plot/save/summary logic that requires those dictionaries must be fixed before proceeding.

- [ ] **Step 2: Search for model/obs vocabulary in runtime behavior**

Run:

```bash
rg -n "Mean_Obs|Mean_Model|model_field|obs_output_dir|Loading obs|Loading model|Observation .*not found|Model .*not found|obs-only|model-observation" davinci_monet -g '*.py'
```

Expected allowed matches:

```text
explicit compatibility aliases
deprecated config warnings
legacy tests
comments documenting old behavior
```

If a match controls standard source runtime behavior, replace it with source/reference/comparand vocabulary.

- [ ] **Step 3: Search for old config-only assumptions**

Run:

```bash
rg -n "model_obs|obs_model|model label first|obs label|model label|obs:" davinci_monet/config davinci_monet/cli docs README.md CLAUDE.md
```

Expected: old vocabulary appears only in legacy compatibility sections.

- [ ] **Step 4: Run focused unification tests**

Run:

```bash
pytest \
  davinci_monet/tests/test_unified_sources_runtime.py \
  davinci_monet/tests/test_pair_direction.py \
  davinci_monet/tests/test_paired_role_tags.py \
  davinci_monet/tests/test_paired_source_labels.py \
  davinci_monet/tests/test_obs_source_labels.py \
  davinci_monet/tests/test_renderer_role_styling.py \
  -q
```

Expected: PASS.

- [ ] **Step 5: Run full validation**

Run:

```bash
pytest
mypy davinci_monet
black --check davinci_monet
isort --check davinci_monet
```

Expected: all pass.

- [ ] **Step 6: Show final diff for review**

Run:

```bash
git status --short
git diff --stat
git diff -- davinci_monet docs README.md CLAUDE.md
```

Expected: only source-unification files changed, no generated artifacts, no commits.

---

## Implementation Order

1. Typed config and tests first, so invalid unified pair specs fail early.
2. Source helpers next, so later stages share one way to access data.
3. Single-source stats/plots, because these are the most obvious non-agnostic runtime components.
4. Pair failure semantics, because silent zero-output success hides real unsupported geometry gaps.
5. `PairedData`, stats, and plotting vocabulary, because they affect many callers and need compatibility aliases.
6. MODIS/grid-target and spatial-overlay source lookups, because they are concrete model-biased workflows.
7. Runner/CLI/docs cleanup, then source processing (Task 17 resample), then the final audit (Task 18).

## Open Design Decision

This plan treats unsupported geometry combinations as geometry limitations, not model/obs residuals. It keeps the current supported matrix of GRID comparand onto POINT/TRACK/PROFILE/SWATH/GRID references, plus existing GRID/GRID same-geometry behavior. It does not add POINT/POINT, TRACK/TRACK, PROFILE/PROFILE, SWATH/SWATH, or mixed irregular strategies in this pass. The required behavior is explicit failure with a clear geometry error, not silent success.

If full arbitrary geometry pairing is required now, add a follow-up plan focused only on pairing algorithms. That work should specify scientific semantics for nearest-site matching, track-track time/space matching, profile vertical alignment, and swath-swath footprint matching before implementation.

**MODIS-L2 unified reachability (required WS1 sub-plan).** Today `MODISL2Reader` is unregistered and reachable only through the legacy `obs:` + `sat_type: modis_l2` branch (`stages.py` `_load_modis_l2`), so it cannot be selected via `sources: type:`. WS3 deletes that legacy branch, so MODIS-L2 must gain a unified path first. Two candidate approaches must be reconciled before coding: (a) keep load-time swath→grid binning and special-case `type: modis_l2` in `_load_unified_source`, sourcing the grid target from `context.sources` (extends Task 10); or (b) register a SWATH-producing MODIS-L2 reader and bin at pairing time via the existing `SwathGridStrategy` (the documented "recommended strategy for all L2 products"), the cleaner unified model. Because this has real design depth and overlaps Task 10, it gets its own brainstorm + plan. Acceptance: a `sources:`-only config selecting MODIS-L2 by `type:` plus a grid reference produces a gridded paired result through `PipelineRunner.run_from_config`, with no `obs:`/`sat_type` keys.
