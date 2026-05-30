# Unified Data-Source — Phase 1 (Core Abstraction) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Introduce the unified `SourceReader` / `SourceProcessor` protocols and a single `source_registry`, additively, so the model/obs distinction can be removed in later phases without breaking the existing 961-test suite.

**Architecture:** Phase 1 of the 6-phase refactor described in `docs/superpowers/specs/2026-05-30-unified-data-source-design.md`. This phase is **purely additive**: it adds new abstractions alongside the existing `ModelReader`/`ObservationReader` protocols and `model_registry`/`observation_registry`. Nothing is renamed, removed, or rewired yet, so every existing test stays green. The new `SourceReader` requires a `geometry` property on every reader; existing observation readers already satisfy it, model readers will be migrated in Phase 2.

**Tech Stack:** Python 3.11+, `typing.Protocol` (`runtime_checkable`), pytest, mypy, xarray.

---

## Phase Roadmap (context only — not implemented here)

1. **Core abstraction** ← this plan
2. Readers/registry — migrate all registrations to `source_registry`; add `geometry` to model readers; audit type-id collisions
3. Pipeline — `LoadSourcesStage` + `context.sources`
4. Pairing — role-neutral `pair(reference, comparand)` + `(ref, comp)` dispatch
5. Plots/stats + obs-only consolidation
6. Config + migration CLI; delete legacy schema, shims, and aliases

Each subsequent phase gets its own plan written when we reach it.

## File Structure

| File | Responsibility | Change |
|------|---------------|--------|
| `davinci_monet/core/protocols.py` | Protocol definitions | Add `SourceReader`, `SourceProcessor` |
| `davinci_monet/core/registry.py` | Pre-configured registries | Add `source_registry` |
| `davinci_monet/core/__init__.py` | Public core exports | Export the three new names |
| `davinci_monet/tests/unit/core/test_source_abstraction.py` | Tests for the new abstractions | Create |

No existing symbols are modified or removed in this phase.

---

## Task 1: Add `SourceReader` and `SourceProcessor` protocols

**Files:**
- Modify: `davinci_monet/core/protocols.py` (insert a new section between the "Observation Protocols" block ending at line ~208 and the "Pairing Protocols" block beginning at line ~211)
- Test: `davinci_monet/tests/unit/core/test_source_abstraction.py`

- [ ] **Step 1: Write the failing test**

Create `davinci_monet/tests/unit/core/test_source_abstraction.py`:

```python
"""Tests for the unified data-source abstraction (Phase 1).

These verify the new SourceReader / SourceProcessor protocols and the
source_registry exist and behave correctly. They are additive: the legacy
ModelReader / ObservationReader protocols and registries are untouched.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import xarray as xr

from davinci_monet.core.protocols import (
    DataGeometry,
    SourceProcessor,
    SourceReader,
)


class _FullSourceReader:
    """A reader with name, geometry, open, and get_variable_mapping."""

    @property
    def name(self) -> str:
        return "mock_source"

    @property
    def geometry(self) -> DataGeometry:
        return DataGeometry.GRID

    def open(
        self,
        file_paths: Sequence[str | Path],
        variables: Sequence[str] | None = None,
        time_range: tuple[Any, Any] | None = None,
        **kwargs: Any,
    ) -> xr.Dataset:
        return xr.Dataset(attrs={"geometry": self.geometry.name})

    def get_variable_mapping(self) -> Mapping[str, str]:
        return {"ozone": "O3"}


class _NoGeometryReader:
    """A reader missing the required geometry property (former ModelReader shape)."""

    @property
    def name(self) -> str:
        return "no_geometry"

    def open(
        self,
        file_paths: Sequence[str | Path],
        variables: Sequence[str] | None = None,
        **kwargs: Any,
    ) -> xr.Dataset:
        return xr.Dataset()

    def get_variable_mapping(self) -> Mapping[str, str]:
        return {}


class TestSourceReaderProtocol:
    def test_runtime_checkable_accepts_full_reader(self) -> None:
        assert isinstance(_FullSourceReader(), SourceReader)

    def test_runtime_checkable_rejects_reader_without_geometry(self) -> None:
        # The key contract change: every source reader MUST declare geometry.
        assert not isinstance(_NoGeometryReader(), SourceReader)

    def test_geometry_is_data_geometry(self) -> None:
        assert _FullSourceReader().geometry is DataGeometry.GRID


class _MockProcessor:
    def process(self, dataset: xr.Dataset, **kwargs: Any) -> xr.Dataset:
        return dataset


class TestSourceProcessorProtocol:
    def test_runtime_checkable_accepts_processor(self) -> None:
        assert isinstance(_MockProcessor(), SourceProcessor)

    def test_processor_returns_dataset(self) -> None:
        ds = xr.Dataset()
        assert _MockProcessor().process(ds) is ds
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest davinci_monet/tests/unit/core/test_source_abstraction.py -v`
Expected: FAIL at import — `ImportError: cannot import name 'SourceReader' from 'davinci_monet.core.protocols'`

- [ ] **Step 3: Add the protocols**

In `davinci_monet/core/protocols.py`, insert the following block immediately before the `# Pairing Protocols` section header (currently around line 211). `xr` is already imported under `TYPE_CHECKING` at the top of the file, so no new import is needed.

```python
# =============================================================================
# Unified Source Protocols
# =============================================================================
#
# A data source is just data of a given geometry (point, track, profile,
# swath, grid). Models and observations are both data sources; the only thing
# that distinguishes them is topology, not origin. These protocols unify the
# legacy ModelReader/ModelProcessor and ObservationReader/ObservationProcessor
# pairs. A model/obs "role" may travel as metadata for labeling and styling,
# but it never appears in these contracts.


@runtime_checkable
class SourceReader(Protocol):
    """Protocol for data source readers (models and observations alike).

    Every source reader declares the geometry it produces and loads files into
    a standardized xarray Dataset whose ``attrs['geometry']`` is set. This is
    the unified replacement for ``ModelReader`` and ``ObservationReader``.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this source type (e.g. 'cesm_fv', 'pt_sfc')."""
        ...

    @property
    @abstractmethod
    def geometry(self) -> DataGeometry:
        """The data geometry this reader produces."""
        ...

    @abstractmethod
    def open(
        self,
        file_paths: Sequence[str | Path],
        variables: Sequence[str] | None = None,
        time_range: tuple[Any, Any] | None = None,
        **kwargs: Any,
    ) -> xr.Dataset:
        """Open source files and return a standardized Dataset.

        Parameters
        ----------
        file_paths
            Paths to source files (can include glob patterns).
        variables
            Optional list of variables to load. If None, load all.
        time_range
            Optional (start, end) time range to subset.
        **kwargs
            Additional reader-specific options.

        Returns
        -------
        xr.Dataset
            Source data with geometry-appropriate dimensions and the
            ``geometry`` attribute set.
        """
        ...

    @abstractmethod
    def get_variable_mapping(self) -> Mapping[str, str]:
        """Return mapping from standard variable names to source-specific names."""
        ...


@runtime_checkable
class SourceProcessor(Protocol):
    """Protocol for data source post-processing operations.

    Unifies ``ModelProcessor`` and ``ObservationProcessor``. Processors handle
    unit conversion, vertical-coordinate handling, resampling, QA/QC,
    subsetting, and aggregation, composed into one chain regardless of origin.
    """

    @abstractmethod
    def process(self, dataset: xr.Dataset, **kwargs: Any) -> xr.Dataset:
        """Apply processing to a source Dataset and return the result."""
        ...
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest davinci_monet/tests/unit/core/test_source_abstraction.py -v`
Expected: 5 passed. The test imports directly from `davinci_monet.core.protocols`, which now defines `SourceReader` and `SourceProcessor`, so it passes without the `core/__init__.py` exports (those are added in Task 3).

- [ ] **Step 5: Run mypy on the changed module**

Run: `mypy davinci_monet/core/protocols.py`
Expected: no new errors introduced by the added protocols.

- [ ] **Step 6: Commit**

```bash
git add davinci_monet/core/protocols.py davinci_monet/tests/unit/core/test_source_abstraction.py
git commit -m "feat(core): add unified SourceReader/SourceProcessor protocols

Phase 1 of model/obs unification. Additive only; legacy protocols untouched.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: Add `source_registry`

**Files:**
- Modify: `davinci_monet/core/registry.py` (after `observation_registry`, line ~270)
- Test: `davinci_monet/tests/unit/core/test_source_abstraction.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `davinci_monet/tests/unit/core/test_source_abstraction.py`:

```python
from davinci_monet.core.registry import Registry, source_registry


class TestSourceRegistry:
    def test_source_registry_exists(self) -> None:
        assert source_registry.name == "source"
        assert isinstance(source_registry, Registry)

    def test_register_and_get(self) -> None:
        local: Registry[type] = Registry("source")

        @local.register("cesm_fv")
        class _Reader:
            pass

        assert local.get("cesm_fv") is _Reader
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest davinci_monet/tests/unit/core/test_source_abstraction.py::TestSourceRegistry -v`
Expected: FAIL — `ImportError: cannot import name 'source_registry' from 'davinci_monet.core.registry'`

- [ ] **Step 3: Add the registry**

In `davinci_monet/core/registry.py`, insert immediately after the `observation_registry` definition (line ~270):

```python
source_registry: Registry[type] = Registry("source")
"""Unified registry for data source reader classes.

Models and observations both register here, keyed by a single ``type`` id and
distinguished only by the geometry their reader declares. Replaces the separate
model_registry and observation_registry in later phases of the unification."""
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest davinci_monet/tests/unit/core/test_source_abstraction.py::TestSourceRegistry -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add davinci_monet/core/registry.py davinci_monet/tests/unit/core/test_source_abstraction.py
git commit -m "feat(core): add unified source_registry

Phase 1 of model/obs unification. Additive only.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: Export the new symbols from `core/__init__.py`

**Files:**
- Modify: `davinci_monet/core/__init__.py`
- Test: `davinci_monet/tests/unit/core/test_source_abstraction.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `davinci_monet/tests/unit/core/test_source_abstraction.py`:

```python
class TestCorePackageExports:
    def test_protocols_exported_from_core(self) -> None:
        from davinci_monet.core import SourceProcessor, SourceReader

        assert SourceReader is not None
        assert SourceProcessor is not None

    def test_registry_exported_from_core(self) -> None:
        from davinci_monet.core import source_registry

        assert source_registry.name == "source"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest davinci_monet/tests/unit/core/test_source_abstraction.py::TestCorePackageExports -v`
Expected: FAIL — `ImportError: cannot import name 'SourceReader' from 'davinci_monet.core'`

- [ ] **Step 3: Add the exports**

In `davinci_monet/core/__init__.py`:

a) In the `from davinci_monet.core.protocols import (...)` block (lines 40-57), add `SourceProcessor,` and `SourceReader,` in alphabetical position (after `SpatialPlotter,` is fine; keep the existing ordering style):

```python
    SpatialPlotter,
    SourceProcessor,
    SourceReader,
    StatisticMetric,
    StatisticsCalculator,
)
```

b) In the `from davinci_monet.core.registry import (...)` block (lines 58-70), add `source_registry,`:

```python
    reader_registry,
    source_registry,
    statistic_registry,
    writer_registry,
)
```

c) In `__all__`, add the three names. Under the protocols section add (near "Pairing protocols"):

```python
    # Unified source protocols
    "SourceReader",
    "SourceProcessor",
```

and under "Pre-configured registries" add:

```python
    "source_registry",
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest davinci_monet/tests/unit/core/test_source_abstraction.py::TestCorePackageExports -v`
Expected: 2 passed.

- [ ] **Step 5: Run the full new test file**

Run: `pytest davinci_monet/tests/unit/core/test_source_abstraction.py -v`
Expected: 9 passed.

- [ ] **Step 6: Commit**

```bash
git add davinci_monet/core/__init__.py davinci_monet/tests/unit/core/test_source_abstraction.py
git commit -m "feat(core): export SourceReader, SourceProcessor, source_registry

Phase 1 of model/obs unification.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: Verify the full suite stays green

**Files:** none (verification only)

- [ ] **Step 1: Run the full test suite**

Run: `pytest`
Expected: all previously-passing tests still pass, plus 9 new tests. No failures, no errors. (If HDF5 thread-safety segfaults appear, re-run with `HDF5_USE_FILE_LOCKING=FALSE pytest` per CLAUDE.md gotcha #8.)

- [ ] **Step 2: Run mypy and formatters**

Run: `mypy davinci_monet/core && black --check davinci_monet/core davinci_monet/tests/unit/core/test_source_abstraction.py && isort --check davinci_monet/core davinci_monet/tests/unit/core/test_source_abstraction.py`
Expected: mypy clean on `core`; black/isort report no changes needed. If black/isort report diffs, run them without `--check` and amend the last commit.

- [ ] **Step 3: Confirm legacy abstractions are untouched**

Run: `pytest davinci_monet/tests/unit/core/test_protocols.py davinci_monet/tests/unit/core/test_registry.py -v`
Expected: all pass — `ModelReader`, `ObservationReader`, `model_registry`, `observation_registry`, `reader_registry` are all unchanged and still present.

---

## Self-Review Notes

- **Spec coverage:** Phase 1 implements spec §1 (the `SourceReader`/`SourceProcessor` protocols) and the registry half of §2 (`source_registry`). Registration migration and the type-id collision audit are explicitly deferred to Phase 2 per the spec's phasing. Adding `geometry` to model readers is Phase 2 (they don't satisfy `SourceReader` yet — locked by `test_runtime_checkable_rejects_reader_without_geometry`).
- **No placeholders:** every step has literal code and exact commands.
- **Type consistency:** `SourceReader.geometry -> DataGeometry`, `open(..., time_range=None, ...)`, `get_variable_mapping() -> Mapping[str, str]`, and `SourceProcessor.process(dataset, **kwargs) -> xr.Dataset` match the test mocks and the existing `ObservationReader` signature exactly.
- **Green-keeping:** all changes are additive; Task 4 verifies legacy protocols/registries and the full suite are intact.
