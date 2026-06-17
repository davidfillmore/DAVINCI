# Derived-Analysis Foundation (Plan A) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the reusable "derived-analysis" layer — a registry, a base class, config schema, a pipeline stage that runs analyses in dependency order and registers their outputs as pseudo-sources — with no new third-party dependencies and no concrete analyses yet.

**Architecture:** A new `davinci_monet/analysis/` package mirrors the statistics registry pattern. `AnalysesStage` runs after `LoadSourcesStage`, topologically orders the `analyses:` config entries, runs each via `analysis_registry`, and inserts each result back into `context.sources` as a `SourceData` marked `derived`. Pairing/stats references to a derived source are rejected at config-validation time (fail fast).

**Tech Stack:** Python 3.11/3.12, Pydantic v2, xarray, the existing DAVINCI pipeline/registry framework. Tests run through `PipelineRunner.run_from_config` in the `davinci` conda env.

**Spec:** `docs/superpowers/specs/2026-06-17-eof-and-wavelet-analysis-design.md` (§2, §3, §10, §11 Plan A).

**Conventions (from a codebase audit — do not deviate):**
- Registries store **classes**, keyed by a string; registration only enforces name-uniqueness.
- Config base classes set Pydantic model config via **class keyword args** (`StrictSchema(BaseModel, extra="forbid", ...)`), not `model_config = ConfigDict(...)`. The repo uses **no discriminated unions** — polymorphic config is built by `@field_validator(..., mode="before")`.
- Stages: `import time` + `start = time.time()` at the top of `execute`; heavy imports are **lazy** (inside methods); soft errors accumulate in `context.metadata.setdefault("<x>_errors", []).append(...)`.
- Tests live under `davinci_monet/tests/` (NOT top-level `tests/`). Integration tests carry `@pytest.mark.integration` and call `PipelineRunner(show_progress=False).run_from_config(<dict>)`. `filterwarnings = ["error::UserWarning", ...]` — a stray `UserWarning` fails the suite.
- Run gates in the `davinci` env: `HDF5_USE_FILE_LOCKING=FALSE python -m pytest`, `mypy davinci_monet`, `black davinci_monet && isort davinci_monet`.

---

## File Structure

- Create `davinci_monet/analysis/__init__.py` — package exports; importing it triggers registration of concrete analyses (added in Plans B/C). Empty of concrete imports in Plan A.
- Create `davinci_monet/analysis/base.py` — `DerivedAnalysis` ABC.
- Create `davinci_monet/pipeline/stages/analyses.py` — `AnalysesStage` + `_topological_order` helper.
- Modify `davinci_monet/core/registry.py` — add `analysis_registry`.
- Modify `davinci_monet/core/protocols.py` — add `DataGeometry.SPECTRUM`.
- Modify `davinci_monet/config/schema.py` — `PointReduce`, `EOFSpec`, `WaveletSpec`, `AnalysisSpec`, `build_analysis_spec`, `MonetConfig.analyses` + validators.
- Modify `davinci_monet/pipeline/stages/base.py` — `PipelineContext.analyses_config()`.
- Modify `davinci_monet/pipeline/stages/factory.py` + `__init__.py` — wire/export `AnalysesStage`.
- Create tests under `davinci_monet/tests/unit/` and `davinci_monet/tests/integration/`.

---

### Task 1: `analysis_registry`

**Files:**
- Modify: `davinci_monet/core/registry.py` (append next to `statistic_registry`)
- Test: `davinci_monet/tests/unit/test_analysis_registry.py`

- [ ] **Step 1: Write the failing test**

```python
"""The analysis registry registers and retrieves analysis classes."""

from __future__ import annotations

from davinci_monet.core.registry import analysis_registry


def test_analysis_registry_register_and_get() -> None:
    @analysis_registry.register("dummy_t1")
    class Dummy:
        name = "dummy_t1"

    assert analysis_registry.get("dummy_t1") is Dummy
    assert "dummy_t1" in analysis_registry.list()
    analysis_registry.unregister("dummy_t1")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/unit/test_analysis_registry.py -v`
Expected: FAIL — `ImportError: cannot import name 'analysis_registry'`.

- [ ] **Step 3: Add the registry**

Append at the end of `davinci_monet/core/registry.py`, immediately after the `statistic_registry` block:

```python
analysis_registry: Registry[type] = Registry("analysis")
"""Registry for derived-analysis classes (eof, wavelet, ...).

Like the other registries it stores component *classes* keyed by a unique
``type`` id. An analysis consumes one source dataset and emits a derived
dataset (see ``davinci_monet.analysis.base.DerivedAnalysis``)."""
```

- [ ] **Step 4: Run test to verify it passes**

Run: `HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/unit/test_analysis_registry.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add davinci_monet/core/registry.py davinci_monet/tests/unit/test_analysis_registry.py
git commit -m "feat(analysis): add analysis_registry"
```

---

### Task 2: `DataGeometry.SPECTRUM`

**Files:**
- Modify: `davinci_monet/core/protocols.py` (`DataGeometry` enum)
- Test: `davinci_monet/tests/unit/test_spectrum_geometry.py`

- [ ] **Step 1: Write the failing test**

```python
"""SPECTRUM geometry exists for wavelet (time, period) outputs."""

from __future__ import annotations

from davinci_monet.core.protocols import DataGeometry


def test_spectrum_geometry_member() -> None:
    assert DataGeometry.SPECTRUM.name == "SPECTRUM"
    assert DataGeometry.SPECTRUM not in {
        DataGeometry.POINT,
        DataGeometry.TRACK,
        DataGeometry.PROFILE,
        DataGeometry.SWATH,
        DataGeometry.GRID,
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/unit/test_spectrum_geometry.py -v`
Expected: FAIL — `AttributeError: SPECTRUM`.

- [ ] **Step 3: Add the member**

In `davinci_monet/core/protocols.py`, append inside `class DataGeometry(Enum)` after the `GRID = auto()` block:

```python
    SPECTRUM = auto()
    """Time-frequency spectrum (time, period) - wavelet power. Not pairable."""
```

- [ ] **Step 4: Run test to verify it passes**

Run: `HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/unit/test_spectrum_geometry.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add davinci_monet/core/protocols.py davinci_monet/tests/unit/test_spectrum_geometry.py
git commit -m "feat(analysis): add DataGeometry.SPECTRUM"
```

---

### Task 3: `DerivedAnalysis` base class + package

**Files:**
- Create: `davinci_monet/analysis/base.py`
- Create: `davinci_monet/analysis/__init__.py`
- Test: `davinci_monet/tests/unit/test_derived_analysis_base.py`

- [ ] **Step 1: Write the failing test**

```python
"""DerivedAnalysis is an ABC: a concrete subclass implements analyze()."""

from __future__ import annotations

import numpy as np
import xarray as xr

from davinci_monet.analysis import DerivedAnalysis
from davinci_monet.core.protocols import DataGeometry
from davinci_monet.core.registry import analysis_registry


def test_concrete_analysis_runs_and_registers() -> None:
    @analysis_registry.register("identity_t3")
    class Identity(DerivedAnalysis):
        name = "identity_t3"
        long_name = "Identity"
        output_geometry = DataGeometry.GRID

        def analyze(self, data: xr.Dataset, spec: object) -> xr.Dataset:
            return data

    ds = xr.Dataset({"x": ("t", np.arange(3.0))})
    out = analysis_registry.get("identity_t3")().analyze(ds, None)
    assert out is ds
    analysis_registry.unregister("identity_t3")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/unit/test_derived_analysis_base.py -v`
Expected: FAIL — `ModuleNotFoundError: davinci_monet.analysis`.

- [ ] **Step 3: Create the base class**

`davinci_monet/analysis/base.py`:

```python
"""Base class for derived analyses (field/series-producing, not scalar)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import xarray as xr

    from davinci_monet.core.protocols import DataGeometry


class DerivedAnalysis(ABC):
    """An analysis that consumes ONE source dataset and emits a derived dataset.

    Concrete analyses register via ``@analysis_registry.register("<type>")`` and
    set ``output_geometry`` to the geometry of their principal output field.
    """

    name: str = "base"
    long_name: str = "Base Derived Analysis"
    output_geometry: "DataGeometry"

    @abstractmethod
    def analyze(self, data: "xr.Dataset", spec: Any) -> "xr.Dataset":
        """Return a derived dataset.

        ``data`` is the fully-built input dataset (a raw source or an
        already-built derived source). ``spec`` is the validated Pydantic
        params for this analysis entry.
        """
        ...
```

`davinci_monet/analysis/__init__.py`:

```python
"""Derived-analysis package: EOF, wavelet, and the shared base/registry.

Importing this package registers all concrete analyses as an import
side-effect (added in later plans). The registry itself lives in
``davinci_monet.core.registry`` to avoid circular imports.
"""

from __future__ import annotations

from davinci_monet.analysis.base import DerivedAnalysis

__all__ = ["DerivedAnalysis"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/unit/test_derived_analysis_base.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add davinci_monet/analysis/ davinci_monet/tests/unit/test_derived_analysis_base.py
git commit -m "feat(analysis): add DerivedAnalysis base + package"
```

---

### Task 4: Config specs — `PointReduce`, `EOFSpec`, `WaveletSpec`, `build_analysis_spec`

**Files:**
- Modify: `davinci_monet/config/schema.py` (add models near the other section configs; add `build_analysis_spec` helper)
- Test: `davinci_monet/tests/unit/test_analysis_specs.py`

- [ ] **Step 1: Write the failing test**

```python
"""Analysis spec models parse and dispatch by type."""

from __future__ import annotations

import pytest

from davinci_monet.config.schema import (
    EOFSpec,
    PointReduce,
    WaveletSpec,
    build_analysis_spec,
)


def test_build_eof_spec() -> None:
    spec = build_analysis_spec({"type": "eof", "source": "cam", "variable": "O3", "n_modes": 6})
    assert isinstance(spec, EOFSpec)
    assert spec.n_modes == 6
    assert spec.standardize is False
    assert spec.rotation == "none"


def test_build_wavelet_spec_with_point_reduce() -> None:
    spec = build_analysis_spec(
        {"type": "wavelet", "source": "cam", "variable": "O3", "reduce": {"point": [40.0, -105.0]}}
    )
    assert isinstance(spec, WaveletSpec)
    assert isinstance(spec.reduce, PointReduce)
    assert spec.reduce.point == (40.0, -105.0)


def test_wavelet_default_reduce_is_area_mean() -> None:
    spec = build_analysis_spec({"type": "wavelet", "source": "cam", "variable": "O3"})
    assert spec.reduce == "area_mean"


def test_unknown_analysis_type_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown analysis type"):
        build_analysis_spec({"type": "bogus", "source": "cam", "variable": "O3"})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/unit/test_analysis_specs.py -v`
Expected: FAIL — `ImportError: cannot import name 'EOFSpec'`.

- [ ] **Step 3: Add the models + builder**

In `davinci_monet/config/schema.py`, add these models (place them after `AxisRef` / before `MonetConfig`). They extend the existing `StrictSchema`:

```python
class PointReduce(StrictSchema):
    """Reduce a gridded field to a series at a single (lat, lon) point."""

    point: tuple[float, float]


class EOFSpec(StrictSchema):
    """EOF decomposition of one gridded source variable."""

    type: Literal["eof"]
    source: str
    variable: str
    n_modes: int = 10
    standardize: bool = False
    remove_seasonal_cycle: bool = False
    rotation: Literal["none", "varimax"] = "none"
    level: int | None = None


class WaveletSpec(StrictSchema):
    """Continuous wavelet transform of one source variable (a 1-D series)."""

    type: Literal["wavelet"]
    source: str
    variable: str
    mode: int | None = None
    reduce: Literal["area_mean"] | PointReduce | None = "area_mean"
    omega0: float = 6.0
    significance_level: float = 0.95
    dj: float = 0.25
    s0: float | None = None
    j: int | None = None

    @field_validator("reduce", mode="before")
    @classmethod
    def _parse_reduce(cls, v: Any) -> Any:
        if isinstance(v, dict):
            return PointReduce(**v)
        return v


AnalysisSpec = EOFSpec | WaveletSpec


def build_analysis_spec(cfg: Any) -> AnalysisSpec:
    """Build the right AnalysisSpec submodel from a dict, dispatching on type."""
    if isinstance(cfg, (EOFSpec, WaveletSpec)):
        return cfg
    if not isinstance(cfg, dict):
        raise ValueError(f"analysis entry must be a mapping, got {type(cfg).__name__}")
    analysis_type = cfg.get("type")
    if analysis_type == "eof":
        return EOFSpec(**cfg)
    if analysis_type == "wavelet":
        return WaveletSpec(**cfg)
    raise ValueError(
        f"Unknown analysis type '{analysis_type}'. Available analysis types: eof, wavelet"
    )
```

`Literal` and `Any` are already imported at the top of `schema.py` (`from typing import Any, Literal`); `field_validator` is already imported.

- [ ] **Step 4: Run test to verify it passes**

Run: `HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/unit/test_analysis_specs.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add davinci_monet/config/schema.py davinci_monet/tests/unit/test_analysis_specs.py
git commit -m "feat(analysis): add EOF/Wavelet config specs + builder"
```

---

### Task 5: `MonetConfig.analyses` field + `parse_analyses` before-validator

**Files:**
- Modify: `davinci_monet/config/schema.py` (`MonetConfig`)
- Test: `davinci_monet/tests/unit/test_monetconfig_analyses.py`

- [ ] **Step 1: Write the failing test**

```python
"""MonetConfig parses the analyses: block into typed specs."""

from __future__ import annotations

from davinci_monet.config.schema import EOFSpec, MonetConfig, WaveletSpec


def test_analyses_block_parsed() -> None:
    cfg = MonetConfig(
        sources={"cam": {"type": "generic", "files": "x.nc", "variables": {"O3": {"units": "ppb"}}}},
        analyses={
            "cam_O3_eof": {"type": "eof", "source": "cam", "variable": "O3", "n_modes": 4},
            "pc1_wav": {"type": "wavelet", "source": "cam_O3_eof", "variable": "pc", "mode": 1},
        },
    )
    assert isinstance(cfg.analyses["cam_O3_eof"], EOFSpec)
    assert isinstance(cfg.analyses["pc1_wav"], WaveletSpec)
    assert cfg.analyses["cam_O3_eof"].n_modes == 4


def test_analyses_defaults_empty() -> None:
    cfg = MonetConfig()
    assert cfg.analyses == {}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/unit/test_monetconfig_analyses.py -v`
Expected: FAIL — `MonetConfig` has no field `analyses` (extra=forbid raises).

- [ ] **Step 3: Add the field + before-validator**

In `class MonetConfig(StrictSchema)`, add the field after `plots`:

```python
    analyses: dict[str, AnalysisSpec] = Field(default_factory=dict)
```

And add the before-validator alongside `parse_plots` (note: plural `analyses`, distinct from the existing singular `analysis: AnalysisConfig`):

```python
    @field_validator("analyses", mode="before")
    @classmethod
    def parse_analyses(cls, v: Any) -> dict[str, AnalysisSpec]:
        """Parse derived-analysis configurations (dispatch on type)."""
        if v is None:
            return {}
        if isinstance(v, dict):
            return {
                str(name): build_analysis_spec(cfg) if isinstance(cfg, dict) else cfg
                for name, cfg in v.items()
            }
        return dict(v)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/unit/test_monetconfig_analyses.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add davinci_monet/config/schema.py davinci_monet/tests/unit/test_monetconfig_analyses.py
git commit -m "feat(analysis): add MonetConfig.analyses block"
```

---

### Task 6: Reference + dependency validation (refs resolve, no cycles, derived not pairable)

**Files:**
- Modify: `davinci_monet/config/schema.py` (`MonetConfig.validate_data_names`)
- Test: `davinci_monet/tests/unit/test_analyses_validation.py`

- [ ] **Step 1: Write the failing test**

```python
"""Cross-reference and dependency rules for the analyses: block."""

from __future__ import annotations

import pytest

from davinci_monet.config.schema import MonetConfig

_SOURCES = {"cam": {"type": "generic", "files": "x.nc", "variables": {"O3": {"units": "ppb"}}}}


def test_analysis_unknown_source_rejected() -> None:
    with pytest.raises(ValueError, match="references unknown source"):
        MonetConfig(
            sources=_SOURCES,
            analyses={"a": {"type": "eof", "source": "nope", "variable": "O3"}},
        )


def test_analysis_cycle_rejected() -> None:
    with pytest.raises(ValueError, match="cycle"):
        MonetConfig(
            sources=_SOURCES,
            analyses={
                "a": {"type": "wavelet", "source": "b", "variable": "pc"},
                "b": {"type": "wavelet", "source": "a", "variable": "pc"},
            },
        )


def test_analysis_key_collides_with_source_rejected() -> None:
    with pytest.raises(ValueError, match="collides"):
        MonetConfig(
            sources=_SOURCES,
            analyses={"cam": {"type": "eof", "source": "cam", "variable": "O3"}},
        )


def test_plot_may_reference_derived_source() -> None:
    cfg = MonetConfig(
        sources=_SOURCES,
        analyses={"cam_O3_eof": {"type": "eof", "source": "cam", "variable": "O3"}},
        plots={"m": {"type": "eof_pattern", "source": "cam_O3_eof", "variable": "mode"}},
    )
    assert "m" in cfg.plots
```

> Note: `test_plot_may_reference_derived_source` also requires `eof_pattern` to be a registered plot type. In Plan A it is not yet registered, so **mark this one test** `@pytest.mark.skip(reason="eof_pattern registered in Plan B")` and remove the skip in Plan B. The other three tests are the Plan A deliverable.

- [ ] **Step 2: Run test to verify it fails**

Run: `HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/unit/test_analyses_validation.py -v`
Expected: FAIL — no validation yet (configs construct without error).

- [ ] **Step 3: Extend `validate_data_names`**

Inside `MonetConfig.validate_data_names` (the `@model_validator(mode="after")`), before the final `if errors:` block, add:

```python
        analysis_names = set(self.analyses)
        # Derived analyses become pseudo-sources; their keys must be unique.
        for name in analysis_names & source_names:
            errors.append(f"analyses.{name} collides with a source of the same name")

        # A valid source reference is a real source OR another analysis output.
        resolvable = source_names | analysis_names
        for a_name, a_spec in self.analyses.items():
            if a_spec.source not in resolvable:
                errors.append(f"analyses.{a_name}.source references unknown source '{a_spec.source}'")

        # Pairs may NOT reference a derived (analysis) source — not pairable.
        for pair_name, pair in self.pairs.items():
            for axis in ("x", "y"):
                ref = getattr(pair, axis).source
                if ref in analysis_names:
                    errors.append(
                        f"pairs.{pair_name}.{axis}.source '{ref}' is a derived analysis "
                        "output; derived sources are not pairable"
                    )

        # Detect cycles in the analysis dependency graph (topological sort).
        state: dict[str, int] = {}  # 0 unvisited, 1 visiting, 2 done

        def _visit(node: str) -> None:
            if state.get(node, 0) == 2:
                return
            if state.get(node, 0) == 1:
                errors.append(f"analyses dependency cycle detected at '{node}'")
                return
            state[node] = 1
            dep = self.analyses[node].source
            if dep in analysis_names:
                _visit(dep)
            state[node] = 2

        for a_name in analysis_names:
            _visit(a_name)
```

Also relax the existing plot-source check so a plot may reference a derived source. Find the line:

```python
            source_ref = plot.source
            if source_ref is not None and str(source_ref) not in source_names:
```

and change `source_names` there to `source_names | set(self.analyses)`.

- [ ] **Step 4: Run test to verify it passes**

Run: `HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/unit/test_analyses_validation.py -v`
Expected: PASS (3 pass, 1 skipped).

- [ ] **Step 5: Commit**

```bash
git add davinci_monet/config/schema.py davinci_monet/tests/unit/test_analyses_validation.py
git commit -m "feat(analysis): validate analysis refs, cycles, derived-not-pairable"
```

---

### Task 7: `PipelineContext.analyses_config()` accessor

**Files:**
- Modify: `davinci_monet/pipeline/stages/base.py` (`PipelineContext`)
- Test: `davinci_monet/tests/unit/pipeline/test_analyses_accessor.py`

- [ ] **Step 1: Write the failing test**

```python
"""PipelineContext.analyses_config() returns typed specs for typed and dict configs."""

from __future__ import annotations

from davinci_monet.config.schema import EOFSpec, MonetConfig
from davinci_monet.pipeline.stages.base import PipelineContext

_ANALYSES = {"cam_O3_eof": {"type": "eof", "source": "cam", "variable": "O3"}}
_SOURCES = {"cam": {"type": "generic", "files": "x.nc", "variables": {"O3": {"units": "ppb"}}}}


def test_accessor_from_typed_config() -> None:
    cfg = MonetConfig(sources=_SOURCES, analyses=_ANALYSES)
    ctx = PipelineContext(config=cfg)
    out = ctx.analyses_config()
    assert isinstance(out["cam_O3_eof"], EOFSpec)


def test_accessor_from_dict_config() -> None:
    ctx = PipelineContext(config={"sources": _SOURCES, "analyses": _ANALYSES})
    out = ctx.analyses_config()
    assert isinstance(out["cam_O3_eof"], EOFSpec)


def test_accessor_empty() -> None:
    assert PipelineContext(config={}).analyses_config() == {}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/unit/pipeline/test_analyses_accessor.py -v`
Expected: FAIL — `PipelineContext` has no `analyses_config`.

- [ ] **Step 3: Add the accessor**

In `davinci_monet/pipeline/stages/base.py`, next to `plots_config`, add:

```python
    def analyses_config(self) -> dict[str, "AnalysisSpec"]:
        """Typed ``analyses:`` mapping (the derived-analysis block)."""
        from davinci_monet.config.schema import build_analysis_spec

        typed = self._typed_config()
        if typed is not None:
            return typed.analyses
        section = self._config_section("analyses") or {}
        return {str(k): build_analysis_spec(v) for k, v in section.items()}
```

Add `AnalysisSpec` to the existing `from davinci_monet.config.schema import (...)` block at the top of the file (or under `TYPE_CHECKING` if that's where the other config types are imported — match the file's existing import style for `StatsConfig`/`PlotGroupConfig`).

- [ ] **Step 4: Run test to verify it passes**

Run: `HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/unit/pipeline/test_analyses_accessor.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add davinci_monet/pipeline/stages/base.py davinci_monet/tests/unit/pipeline/test_analyses_accessor.py
git commit -m "feat(analysis): add PipelineContext.analyses_config()"
```

---

### Task 8: `AnalysesStage` (topo-order + run + pseudo-source registration)

**Files:**
- Create: `davinci_monet/pipeline/stages/analyses.py`
- Test: `davinci_monet/tests/unit/pipeline/test_analyses_stage.py`

- [ ] **Step 1: Write the failing test**

```python
"""AnalysesStage runs analyses in dependency order and registers pseudo-sources."""

from __future__ import annotations

import numpy as np
import xarray as xr
import pytest

from davinci_monet.analysis import DerivedAnalysis
from davinci_monet.core.protocols import DataGeometry
from davinci_monet.core.registry import analysis_registry
from davinci_monet.pipeline.stages.analyses import AnalysesStage
from davinci_monet.pipeline.stages.base import PipelineContext, SourceData, StageStatus


@pytest.fixture
def _fake_eof_registered():
    @analysis_registry.register("eof", replace=True)
    class _FakeEOF(DerivedAnalysis):
        name = "eof"
        output_geometry = DataGeometry.GRID

        def analyze(self, data, spec):
            return xr.Dataset({"pc": ("time", np.arange(3.0))}, coords={"time": np.arange(3)})

    @analysis_registry.register("wavelet", replace=True)
    class _FakeWavelet(DerivedAnalysis):
        name = "wavelet"
        output_geometry = DataGeometry.SPECTRUM

        def analyze(self, data, spec):
            # Depends on the EOF output's `pc` being present.
            assert "pc" in data.data_vars
            return xr.Dataset({"power": (("time", "period"), np.ones((3, 2)))})

    yield
    analysis_registry.unregister("eof")
    analysis_registry.unregister("wavelet")


def _ctx() -> PipelineContext:
    cam = SourceData(
        data=xr.Dataset({"O3": ("time", np.arange(3.0))}, coords={"time": np.arange(3)}),
        label="cam",
        source_type="generic",
        geometry=DataGeometry.GRID,
    )
    return PipelineContext(
        config={
            "sources": {"cam": {"type": "generic", "files": "x.nc", "variables": {"O3": {"units": "ppb"}}}},
            "analyses": {
                "pc1_wav": {"type": "wavelet", "source": "cam_O3_eof", "variable": "pc", "mode": 1},
                "cam_O3_eof": {"type": "eof", "source": "cam", "variable": "O3"},
            },
        },
        sources={"cam": cam},
    )


def test_stage_registers_derived_sources_in_order(_fake_eof_registered) -> None:
    ctx = _ctx()
    stage = AnalysesStage()
    assert stage.validate(ctx) is True
    result = stage.execute(ctx)

    assert result.status is StageStatus.COMPLETED
    # Both derived sources are now registered.
    assert "cam_O3_eof" in ctx.sources
    assert "pc1_wav" in ctx.sources
    eof_src = ctx.sources["cam_O3_eof"]
    assert isinstance(eof_src, SourceData)
    assert eof_src.source_type == "eof"
    assert eof_src.geometry is DataGeometry.GRID
    assert eof_src.data.attrs["derived"] is True
    assert eof_src.data.attrs["geometry"] == "grid"
    # SPECTRUM geometry for the wavelet output.
    assert ctx.sources["pc1_wav"].geometry is DataGeometry.SPECTRUM


def test_stage_validate_false_when_no_analyses() -> None:
    ctx = PipelineContext(config={"sources": {}})
    assert AnalysesStage().validate(ctx) is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/unit/pipeline/test_analyses_stage.py -v`
Expected: FAIL — `ModuleNotFoundError: davinci_monet.pipeline.stages.analyses`.

- [ ] **Step 3: Implement the stage**

`davinci_monet/pipeline/stages/analyses.py`:

```python
"""Pipeline stage that runs derived analyses and registers their outputs.

Each analysis consumes one source dataset (a raw source or a prior analysis
output) and emits a derived dataset, which is wrapped in a ``SourceData`` and
inserted back into ``context.sources`` so the rest of the pipeline treats it
like any other source.
"""

from __future__ import annotations

from typing import Any

from davinci_monet.core.registry import analysis_registry
from davinci_monet.pipeline.stages.base import (
    BaseStage,
    PipelineContext,
    SourceData,
    StageResult,
    StageStatus,
)


def _topological_order(specs: dict[str, Any]) -> list[str]:
    """Order analysis keys so each runs after the analysis it depends on."""
    keys = set(specs)
    state: dict[str, int] = {}
    order: list[str] = []

    def visit(node: str) -> None:
        if state.get(node, 0) == 2:
            return
        if state.get(node, 0) == 1:
            raise ValueError(f"analyses dependency cycle detected at '{node}'")
        state[node] = 1
        dep = specs[node].source
        if dep in keys:
            visit(dep)
        state[node] = 2
        order.append(node)

    for key in specs:
        visit(key)
    return order


class AnalysesStage(BaseStage):
    """Run derived analyses (EOF, wavelet, ...) and register pseudo-sources."""

    def __init__(self) -> None:
        super().__init__("analyses")

    def validate(self, context: PipelineContext) -> bool:
        return bool(context.analyses_config())

    def execute(self, context: PipelineContext) -> StageResult:
        # Importing the package registers concrete analyses as a side-effect.
        import davinci_monet.analysis  # noqa: F401
        import time

        start = time.time()
        specs = context.analyses_config()
        summary: dict[str, Any] = {}

        try:
            order = _topological_order(specs)
        except ValueError as exc:
            context.metadata.setdefault("analysis_errors", []).append(str(exc))
            return self._create_result(
                StageStatus.FAILED, data=summary, error=str(exc), duration=time.time() - start
            )

        for key in order:
            spec = specs[key]
            try:
                context.log_progress(f"    Analysis: {key} ({spec.type})")
                src_obj = context.sources.get(spec.source)
                if src_obj is None:
                    raise ValueError(f"analysis '{key}' references unknown source '{spec.source}'")
                in_ds = src_obj.data if hasattr(src_obj, "data") else src_obj

                analysis = analysis_registry.get(spec.type)()
                out_ds = analysis.analyze(in_ds, spec)

                geometry = analysis.output_geometry
                out_ds.attrs["geometry"] = geometry.name.lower()
                out_ds.attrs["derived"] = True
                out_ds.attrs.setdefault("source_label", key)

                context.sources[key] = SourceData(
                    data=out_ds,
                    label=key,
                    source_type=spec.type,
                    geometry=geometry,
                    variables={},
                    config=spec.model_dump(),
                )
                summary[key] = {
                    "type": spec.type,
                    "geometry": geometry.name.lower(),
                    "variables": list(out_ds.data_vars),
                }
            except Exception as exc:  # noqa: BLE001 - soft per-analysis failure
                context.metadata.setdefault("analysis_errors", []).append(f"{key}: {exc}")
                context.log_progress(f"warning: analysis failed for {key}: {exc}")

        errors = context.metadata.get("analysis_errors") or []
        if errors:
            return self._create_result(
                StageStatus.FAILED,
                data=summary,
                error="Analyses failed: " + "; ".join(str(e) for e in errors),
                duration=time.time() - start,
            )
        return self._create_result(StageStatus.COMPLETED, data=summary, duration=time.time() - start)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/unit/pipeline/test_analyses_stage.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add davinci_monet/pipeline/stages/analyses.py davinci_monet/tests/unit/pipeline/test_analyses_stage.py
git commit -m "feat(analysis): add AnalysesStage (topo-order + pseudo-sources)"
```

---

### Task 9: Wire `AnalysesStage` into the pipeline factory + exports

**Files:**
- Modify: `davinci_monet/pipeline/stages/factory.py`
- Modify: `davinci_monet/pipeline/stages/__init__.py`
- Test: `davinci_monet/tests/unit/pipeline/test_factory_has_analyses.py`

- [ ] **Step 1: Write the failing test**

```python
"""AnalysesStage runs after LoadSources and before Pairing in the standard pipeline."""

from __future__ import annotations

from davinci_monet.pipeline.stages import AnalysesStage, create_standard_pipeline
from davinci_monet.pipeline.stages import create_geometry_pipeline


def _names(stages) -> list[str]:
    return [s.name for s in stages]


def test_analyses_in_standard_pipeline_order() -> None:
    names = _names(create_standard_pipeline())
    assert "analyses" in names
    assert names.index("analyses") == names.index("load_sources") + 1
    assert names.index("analyses") < names.index("pairing")


def test_analyses_in_geometry_pipeline() -> None:
    assert "analyses" in _names(create_geometry_pipeline())


def test_analyses_stage_exported() -> None:
    assert AnalysesStage().name == "analyses"
```

> The `load_sources` stage name: confirm `LoadSourcesStage` sets `super().__init__("load_sources")`. If its name differs, adjust the assertion to the actual name (check `pipeline/stages/load.py`).

- [ ] **Step 2: Run test to verify it fails**

Run: `HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/unit/pipeline/test_factory_has_analyses.py -v`
Expected: FAIL — `ImportError: cannot import name 'AnalysesStage'`.

- [ ] **Step 3: Wire factory + exports**

In `davinci_monet/pipeline/stages/factory.py`, add the import and insert the stage:

```python
from davinci_monet.pipeline.stages.analyses import AnalysesStage
```

`create_standard_pipeline()` becomes:

```python
def create_standard_pipeline() -> list[BaseStage]:
    return [
        LoadSourcesStage(),
        AnalysesStage(),
        PairingStage(),
        StatisticsStage(),
        PlottingStage(),
        SaveResultsStage(),
        SummaryStage(),
    ]
```

`create_geometry_pipeline()` becomes:

```python
def create_geometry_pipeline() -> list[BaseStage]:
    """Create a single-source pipeline (no pairing stage)."""
    return [
        LoadSourcesStage(),
        AnalysesStage(),
        StatisticsStage(),
        PlottingStage(),
        SaveResultsStage(),
        SummaryStage(),
    ]
```

In `davinci_monet/pipeline/stages/__init__.py`, add the import and `__all__` entry:

```python
from davinci_monet.pipeline.stages.analyses import AnalysesStage
```

and add `"AnalysesStage"` to `__all__` in the "Stage classes" section.

- [ ] **Step 4: Run test to verify it passes**

Run: `HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/unit/pipeline/test_factory_has_analyses.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add davinci_monet/pipeline/stages/factory.py davinci_monet/pipeline/stages/__init__.py davinci_monet/tests/unit/pipeline/test_factory_has_analyses.py
git commit -m "feat(analysis): wire AnalysesStage into pipeline factory"
```

---

### Task 10: End-to-end integration test through `PipelineRunner`

**Files:**
- Test: `davinci_monet/tests/integration/test_analyses_foundation_pipeline.py`

- [ ] **Step 1: Write the failing test**

```python
"""Integration: a derived analysis runs through the full pipeline and its output
is registered as a pseudo-source (proves the foundation end-to-end)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from davinci_monet.analysis import DerivedAnalysis
from davinci_monet.core.protocols import DataGeometry
from davinci_monet.core.registry import analysis_registry
from davinci_monet.pipeline.runner import PipelineRunner


@pytest.fixture
def _passthrough_eof():
    """Register a trivial 'eof' that emits a (time, mode) pc + (mode) variance."""

    @analysis_registry.register("eof", replace=True)
    class _PassEOF(DerivedAnalysis):
        name = "eof"
        output_geometry = DataGeometry.GRID

        def analyze(self, data, spec):  # noqa: ANN001
            nt = data.sizes["time"]
            return xr.Dataset(
                {
                    "pc": (("time", "mode"), np.zeros((nt, 2)), {"kind": "pc", "units": "1"}),
                    "explained_variance": ("mode", np.array([0.7, 0.3]), {"kind": "scalar"}),
                },
                coords={"time": data["time"].values, "mode": [1, 2]},
            )

    yield
    analysis_registry.unregister("eof")


def _grid_nc(path: Path) -> None:
    times = pd.date_range("2024-01-01", periods=6, freq="D")
    lat = np.linspace(20, 50, 4)
    lon = np.linspace(-120, -90, 5)
    rng = np.random.default_rng(0)
    data = rng.normal(size=(len(times), len(lat), len(lon)))
    xr.Dataset(
        {"O3": (("time", "lat", "lon"), data, {"units": "ppb"})},
        coords={
            "time": times,
            "lat": ("lat", lat),
            "lon": ("lon", lon),
            "latitude": ("lat", lat),
            "longitude": ("lon", lon),
        },
    ).to_netcdf(path)


@pytest.mark.integration
def test_analysis_runs_through_pipeline(tmp_path: Path, _passthrough_eof) -> None:
    src = tmp_path / "grid.nc"
    _grid_nc(src)
    config = {
        "analysis": {"output_dir": str(tmp_path / "out")},
        "sources": {"cam": {"type": "generic", "files": str(src), "variables": {"O3": {"units": "ppb"}}}},
        "analyses": {"cam_O3_eof": {"type": "eof", "source": "cam", "variable": "O3", "n_modes": 2}},
    }

    result = PipelineRunner(show_progress=False).run_from_config(config)

    assert result.success, getattr(result, "error", None)
    ctx = result.context
    assert "cam_O3_eof" in ctx.sources
    derived = ctx.sources["cam_O3_eof"]
    assert derived.source_type == "eof"
    assert derived.geometry is DataGeometry.GRID
    assert derived.data.attrs["derived"] is True
    assert set(derived.data.data_vars) == {"pc", "explained_variance"}
    # The analyses stage recorded a summary.
    assert "cam_O3_eof" in ctx.results["analyses"].data
```

- [ ] **Step 2: Run test to verify it fails**

Run: `HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/integration/test_analyses_foundation_pipeline.py -v`
Expected: FAIL initially only if anything above is incomplete; with Tasks 1-9 done it should PASS. Run it to confirm the full wiring.

- [ ] **Step 3: (No new implementation)**

This test exercises Tasks 1-9. If it fails, debug the wiring (most likely: `analyses_config()` not reachable from the dict-config path, or the stage not inserted). Do not add shortcuts — fix the real path.

- [ ] **Step 4: Run the full gate**

```bash
HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/unit/test_analysis_registry.py \
  davinci_monet/tests/unit/test_spectrum_geometry.py \
  davinci_monet/tests/unit/test_derived_analysis_base.py \
  davinci_monet/tests/unit/test_analysis_specs.py \
  davinci_monet/tests/unit/test_monetconfig_analyses.py \
  davinci_monet/tests/unit/test_analyses_validation.py \
  davinci_monet/tests/unit/pipeline/test_analyses_accessor.py \
  davinci_monet/tests/unit/pipeline/test_analyses_stage.py \
  davinci_monet/tests/unit/pipeline/test_factory_has_analyses.py \
  davinci_monet/tests/integration/test_analyses_foundation_pipeline.py -v
mypy davinci_monet
black davinci_monet && isort davinci_monet
```
Expected: all PASS, mypy clean, formatting clean.

- [ ] **Step 5: Commit**

```bash
git add davinci_monet/tests/integration/test_analyses_foundation_pipeline.py
git commit -m "test(analysis): end-to-end derived-analysis foundation through pipeline"
```

---

## Self-Review (completed by plan author)

- **Spec coverage:** §2.1 registry+base (T1,T3); §2.2 pseudo-source construction with exact SourceData fields + `derived` marker (T8); §2.3 SPECTRUM (T2); §2.4 stage placement + topo-order (T8,T9); §3.1-3.4 schema + accessor + validators (T4-T7); §10 derived-not-pairable enforced at config validation (T6). Concrete EOF/Wavelet *implementations*, renderers, and the `mode`/`display_level` plot wiring are intentionally deferred to Plans B/C.
- **Deviation from spec:** the spec described a Pydantic discriminated union; this plan uses the repo's idiomatic `build_analysis_spec` + before-validator dispatch (no discriminated union exists anywhere in the codebase). Same external behavior. Also, the §10 "PairingStage/StatisticsStage raise" guard is implemented one layer earlier — at config validation (T6) — which fails fast and is the only way a derived source could reach a pair (by its analyses key).
- **Type consistency:** `DerivedAnalysis.analyze(data, spec) -> xr.Dataset`, `output_geometry: DataGeometry`, `analysis_registry.get(type)() ` → instance, `SourceData(data,label,source_type,geometry,variables,config)` — consistent across T3/T8/T10.
- **Placeholders:** none. The one `@pytest.mark.skip` (T6 `test_plot_may_reference_derived_source`) is explicit and removed in Plan B.
