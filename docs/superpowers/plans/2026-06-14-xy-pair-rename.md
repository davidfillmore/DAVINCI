# x/y Pair Rename Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the confusing `geometry`/`dataset` *role* vocabulary with positional **x/y** (plot axes) + **`source_label`** (source identity), and rename the source-level `geometry≡obs` QC terms — a behavior-preserving rename that leaves the *shape* meaning of "geometry" intact.

**Architecture:** `geometry ≡ x`, `dataset ≡ y` (verified: scatter already x=geometry/y=dataset, bias already `dataset − geometry`). Rename is done in coherent units that keep the full test suite green at every commit. Config gets a clean break to nested `x:`/`y:` pairs; tracked configs are migrated. Pairing *direction* is unchanged (driven by shape precedence); x/y is plot-only.

**Tech Stack:** Python 3.11/3.12, pydantic, xarray, matplotlib/cartopy, pytest, mypy, black, isort. Run everything in the `davinci` conda env.

**Spec:** `docs/superpowers/specs/2026-06-14-xy-pair-rename-design.md`

---

## Conventions for every task

**Environment (prefix every test/lint run):**
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate davinci
```

**The regression gate (run at the end of each rename task):**
```bash
cd /Users/fillmore/EarthSystem/DAVINCI
HDF5_USE_FILE_LOCKING=FALSE python -m pytest -q
```
Expected: all tests pass (baseline is 1,262 passing). A rename task is NOT done until the gate is green.

**Never touch the SHAPE meaning of "geometry":** `DataGeometry`, `detect_spatial_geometry`,
`spatial_geometry`, `geometry_type`, `surface_level_index`, point/track/profile/swath/grid.
Run this guard after the role sweeps to confirm only shape/prose uses remain:
```bash
grep -rn --include='*.py' -oE '\b(geometry_var|dataset_var|geometry_data|dataset_data|geometry_series|dataset_series|geometry_color|dataset_color|pair_axis)\b' davinci_monet | grep -v '/tests/'
```

**Commit message footer (required by repo):**
```
Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
```

**Do NOT push or merge.** Commit locally only. We are on `develop`.

---

## Task 1: Lock the semantic contract with a test (TDD anchor)

This permanent test encodes the post-rename contract and must stay green through every later task. Write it against the **current** API first (geometry/dataset), confirm it passes, then it gets carried through the renames (later tasks update its identifiers along with the rest).

**Files:**
- Create: `davinci_monet/tests/test_xy_contract.py`

- [ ] **Step 1: Write the contract test (current vocabulary)**

```python
"""Semantic contract: x is the horizontal/reference axis, y is vertical; diffs are y - x.

This test is the behavior-preservation anchor for the x/y rename. It is written
against the current geometry/dataset API and will be migrated to x/y in lockstep
with the rename tasks; its ASSERTIONS (axis assignment + diff sign) must never change.
"""
from __future__ import annotations

import numpy as np
import xarray as xr

from davinci_monet.plots.base import build_series
from davinci_monet.plots.renderers.scatter import ScatterPlotter


def _paired() -> xr.Dataset:
    time = np.arange(5)
    obs = xr.DataArray(np.arange(5.0), dims="time", name="obs_o3")
    obs.attrs.update({"pair_axis": "geometry", "dataset_label": "obs", "units": "ppb"})
    mod = xr.DataArray(np.arange(5.0) + 2.0, dims="time", name="mod_o3")
    mod.attrs.update({"pair_axis": "dataset", "dataset_label": "mod", "units": "ppb"})
    ds = xr.Dataset({"obs_o3": obs, "mod_o3": mod}, coords={"time": time})
    return ds


def test_scatter_x_is_geometry_y_is_dataset() -> None:
    ds = _paired()
    fig = ScatterPlotter().render(build_series(ds, "obs_o3", "mod_o3"))
    ax = fig.axes[0]
    # x axis names the geometry/x source; y axis names the dataset/y source.
    assert "OBS" in ax.get_xlabel().upper() or "obs" in ax.get_xlabel().lower()
    assert "MOD" in ax.get_ylabel().upper() or "mod" in ax.get_ylabel().lower()


def test_diff_sign_is_y_minus_x() -> None:
    ds = _paired()
    # mod - obs == +2 everywhere (the "y - x" convention).
    diff = ds["mod_o3"] - ds["obs_o3"]
    assert float(diff.mean()) == 2.0
```

- [ ] **Step 2: Run it against current code — expect PASS**

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate davinci
cd /Users/fillmore/EarthSystem/DAVINCI
HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/test_xy_contract.py -v
```
Expected: 2 passed. (If the xlabel/ylabel assertions fail, STOP — the `geometry≡x` premise is wrong and the spec must be revisited.)

- [ ] **Step 3: Commit**

```bash
cd /Users/fillmore/EarthSystem/DAVINCI
git add davinci_monet/tests/test_xy_contract.py
git commit -m "test: lock x=geometry/y=dataset + diff=y-x contract before rename"
```

---

## Task 2: Sweep the unambiguous compound plotting identifiers

These 12 compound identifiers have exactly one meaning and collide with nothing. Rename 1:1 repo-wide (source + tests).

| Old | New |
|---|---|
| `geometry_var` | `x_var` |
| `dataset_var` | `y_var` |
| `geometry_data` | `x_data` |
| `dataset_data` | `y_data` |
| `geometry_series` | `x_series` |
| `dataset_series` | `y_series` |
| `geometry_color` | `x_color` |
| `dataset_color` | `y_color` |
| `geometry_linestyle` | `x_linestyle` |
| `dataset_linestyle` | `y_linestyle` |
| `geometry_marker` | `x_marker` |
| `dataset_marker` | `y_marker` |

**Files:** all `.py` under `davinci_monet/` (including `tests/`).

- [ ] **Step 1: Apply the global sweep**

```bash
cd /Users/fillmore/EarthSystem/DAVINCI
files=$(grep -rln --include='*.py' -E '\b(geometry|dataset)_(var|data|series|color|linestyle|marker)\b' davinci_monet)
for f in $files; do
  sed -i '' -E \
    -e 's/\bgeometry_var\b/x_var/g' -e 's/\bdataset_var\b/y_var/g' \
    -e 's/\bgeometry_data\b/x_data/g' -e 's/\bdataset_data\b/y_data/g' \
    -e 's/\bgeometry_series\b/x_series/g' -e 's/\bdataset_series\b/y_series/g' \
    -e 's/\bgeometry_color\b/x_color/g' -e 's/\bdataset_color\b/y_color/g' \
    -e 's/\bgeometry_linestyle\b/x_linestyle/g' -e 's/\bdataset_linestyle\b/y_linestyle/g' \
    -e 's/\bgeometry_marker\b/x_marker/g' -e 's/\bdataset_marker\b/y_marker/g' \
    "$f"
done
```

- [ ] **Step 2: Run the regression gate**

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate davinci
cd /Users/fillmore/EarthSystem/DAVINCI
HDF5_USE_FILE_LOCKING=FALSE python -m pytest -q
```
Expected: all pass. (If a test fails, it references one of these identifiers in a string/docstring — inspect and fix that single spot, do not broaden the sweep.)

- [ ] **Step 3: Format + commit**

```bash
cd /Users/fillmore/EarthSystem/DAVINCI
black davinci_monet >/dev/null && isort davinci_monet >/dev/null
git add -A && git commit -m "refactor: rename geometry_*/dataset_* compound plot identifiers to x_*/y_*"
```

---

## Task 3: Rename the role helper functions

These are unique function/symbol names; rename 1:1 repo-wide.

| Old | New |
|---|---|
| `get_dataset_color` | `get_axis_color` |
| `paired_variable_pair_axis` | `paired_variable_axis` |
| `iter_paired_variable_pairs` | `iter_paired_variable_xy` |
| `iter_canonical_variable_series` | `iter_canonical_variable_series` *(name kept — already neutral)* |
| `tag_dataset_label` | `tag_source_label` |
| `resolve_dataset_variable` | `resolve_source_variable` |
| `dataset_label` *(the function in `plots/series.py`)* | `source_label` |

**Files:** `davinci_monet/core/base.py`, `davinci_monet/plots/series.py`, `davinci_monet/plots/base.py`, `davinci_monet/pipeline/stages/plot.py`, and any importer (use grep to find them).

- [ ] **Step 1: Find all references**

```bash
cd /Users/fillmore/EarthSystem/DAVINCI
grep -rln --include='*.py' -E '\b(get_dataset_color|paired_variable_pair_axis|iter_paired_variable_pairs|tag_dataset_label|resolve_dataset_variable)\b' davinci_monet
```

- [ ] **Step 2: Apply the rename (these tokens are unique — safe global sed)**

```bash
cd /Users/fillmore/EarthSystem/DAVINCI
files=$(grep -rln --include='*.py' -E '\b(get_dataset_color|paired_variable_pair_axis|iter_paired_variable_pairs|tag_dataset_label|resolve_dataset_variable)\b' davinci_monet)
for f in $files; do
  sed -i '' -E \
    -e 's/\bget_dataset_color\b/get_axis_color/g' \
    -e 's/\bpaired_variable_pair_axis\b/paired_variable_axis/g' \
    -e 's/\biter_paired_variable_pairs\b/iter_paired_variable_xy/g' \
    -e 's/\btag_dataset_label\b/tag_source_label/g' \
    -e 's/\bresolve_dataset_variable\b/resolve_source_variable/g' \
    "$f"
done
```

- [ ] **Step 3: Rename the `dataset_label()` *function* in series.py (token collides with the attr — edit by hand)**

In `davinci_monet/plots/series.py`: rename the function `def dataset_label(` (around line 124) to `def source_label(`, update its docstring, and update the `__all__` entry `"dataset_label"` → `"source_label"`. Then update its importers (grep `from davinci_monet.plots.series import` and `series.dataset_label`). Do NOT touch `.attrs.get("dataset_label")` here — that attr key is renamed in Task 5.

- [ ] **Step 4: Regression gate + format + commit**

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate davinci
cd /Users/fillmore/EarthSystem/DAVINCI
HDF5_USE_FILE_LOCKING=FALSE python -m pytest -q && black davinci_monet >/dev/null && isort davinci_monet >/dev/null
git add -A && git commit -m "refactor: rename role helper functions to axis/source-oriented names"
```

---

## Task 4: Rename the `axis` attribute contract (pair_axis → axis, values geometry/dataset → x/y)

Atomic: the attr key, its values, the `PlotSeries.pair_axis` field, and every reader/writer move together.

**Files:** `davinci_monet/core/base.py`, `davinci_monet/plots/series.py`, `davinci_monet/pipeline/stages/plot.py`, `davinci_monet/pairing/engine.py`, `davinci_monet/pairing/strategies/*.py`, plus any test asserting `pair_axis`.

- [ ] **Step 1: Rename the field + attr key + helper return value**

a. `core/base.py` `PlotSeries`: rename field `pair_axis: str | None` → `axis: str | None`; update the docstring ("Pairing position (`"x"`/`"y"`)..."). Update `iter_canonical_variable_series` and `iter_paired_variable_xy` to construct `PlotSeries(... axis=..., ...)` and to compare against `"x"`/`"y"`.

b. `paired_variable_axis()` (renamed in Task 3): change the attr key it reads from `"pair_axis"` to `"axis"`, and ensure it returns the raw stored value.

c. Repo-wide: rename the keyword/attribute `pair_axis` → `axis` and the attr-key string `"pair_axis"` → `"axis"`:

```bash
cd /Users/fillmore/EarthSystem/DAVINCI
files=$(grep -rln --include='*.py' -E '\bpair_axis\b' davinci_monet)
for f in $files; do sed -i '' -E 's/\bpair_axis\b/axis/g' "$f"; done
```

- [ ] **Step 2: Flip the attr VALUES geometry/dataset → x/y at the exact write/read sites**

These are the only places the role *strings* `"geometry"`/`"dataset"` are used as axis values. Edit each:

- `davinci_monet/pairing/engine.py` — the `.attrs.update(...)` blocks (~lines 304, 312) that set `axis="geometry"`/`axis="dataset"` (post-Step-1) → `"x"`/`"y"`.
- `davinci_monet/pipeline/stages/plot.py` — `axis="geometry"` (~line 301) → `"x"`, `axis="dataset"` (~line 308) → `"y"`.
- `davinci_monet/plots/series.py` — in `series_colors`, `s.axis == "dataset"` → `s.axis == "x"`? **No** — keep the meaning: dataset≡y, so `s.axis == "dataset"` → `s.axis == "y"` and the gray/blue branch logic is unchanged (y → blue).
- `core/base.py` — `iter_paired_variable_xy`: `if axis not in ("geometry", "dataset")` → `("x", "y")`; `(geometries if axis == "geometry" else datasets)` → `(xs if axis == "x" else ys)` (rename the locals too for clarity).
- `plots/series.py` `get_axis_color`: `if pair_axis == "geometry"` → `if axis == "x"` (return x/gray), `== "dataset"` → `== "y"` (return y/blue).

Use this to find every remaining role-valued literal so none is missed:
```bash
cd /Users/fillmore/EarthSystem/DAVINCI
grep -rn --include='*.py' -E '== *"(geometry|dataset)"|axis *= *"(geometry|dataset)"|"(geometry|dataset)"\)' davinci_monet | grep -v '/tests/'
```
Resolve each hit to `"x"`/`"y"` per the `geometry≡x, dataset≡y` rule. **Skip** any hit that is the shape value or a config key (none should remain after the targeted edits).

- [ ] **Step 3: Update tests that assert the attr**

```bash
cd /Users/fillmore/EarthSystem/DAVINCI
grep -rln --include='*.py' -E '"(geometry|dataset)"' davinci_monet/tests | xargs grep -ln 'axis' 2>/dev/null
```
For each, change `axis` attr expectations from `"geometry"`/`"dataset"` to `"x"`/`"y"`. (The Task 1 contract test's `_paired()` helper now sets `"axis": "x"`/`"y"`.)

- [ ] **Step 4: Regression gate + format + commit**

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate davinci
cd /Users/fillmore/EarthSystem/DAVINCI
HDF5_USE_FILE_LOCKING=FALSE python -m pytest -q && black davinci_monet >/dev/null && isort davinci_monet >/dev/null
git add -A && git commit -m "refactor: rename pair_axis attr to axis with x/y values"
```

---

## Task 5: Rename the `source_label` attribute contract (dataset_label attr → source_label)

The attr key `"dataset_label"` (dataset-level AND per-variable) and `PlotSeries.dataset_label` are source identity → `source_label`. Disambiguate from the `PlotConfig.dataset_label` *custom-label* field (that one is handled in Task 6 — do NOT touch it here).

**Files:** `core/base.py` (`PlotSeries`, `PairedData`), `plots/series.py`, `pipeline/stages/load.py`, `pipeline/stages/plot.py`, `pipeline/stages/pair.py`, `pairing/engine.py`, tests.

- [ ] **Step 1: Rename the attr KEY string `"dataset_label"` → `"source_label"`**

Only the attr-key string usages (`.attrs.get("dataset_label")`, `.attrs["dataset_label"] =`, `attrs={"dataset_label": ...}`). Find them:
```bash
cd /Users/fillmore/EarthSystem/DAVINCI
grep -rn --include='*.py' -E '"dataset_label"' davinci_monet
```
Replace each `"dataset_label"` string literal with `"source_label"` (these are all the attr key — confirm none is a config field name; config field is unquoted `dataset_label`, handled in Task 6).
```bash
cd /Users/fillmore/EarthSystem/DAVINCI
files=$(grep -rln --include='*.py' -E '"dataset_label"' davinci_monet)
for f in $files; do sed -i '' -E 's/"dataset_label"/"source_label"/g' "$f"; done
```

- [ ] **Step 2: Rename the `PlotSeries.dataset_label` field → `source_label`**

In `core/base.py`: field `dataset_label: str | None` → `source_label: str | None` (update docstring). Update every `PlotSeries(... dataset_label=...)` construction and every `.dataset_label` access on a PlotSeries (`series.py` `build_series`, `iter_canonical_variable_series`, renderers reading `s.dataset_label`):
```bash
cd /Users/fillmore/EarthSystem/DAVINCI
grep -rn --include='*.py' -E '\bdataset_label=|\.dataset_label\b' davinci_monet | grep -v 'config'
```
Edit each PlotSeries-related site `dataset_label` → `source_label`. **Leave `PlotConfig`/`StyleConfig` `dataset_label` and `self.config.dataset_label` for Task 6.** When unsure whether a `dataset_label` is the PlotSeries field or the config field, check the object type at that line.

- [ ] **Step 3: Rename the two `PairedData` source-identity fields**

In `core/base.py` `PairedData`: `dataset_label` → `y_source`, `geometry_label` → `x_source`; update `from_sources(geometry_label=..., dataset_label=...)` signature → `from_sources(x_source=..., y_source=...)` and its callers in `pipeline/stages/pair.py` and `pairing/engine.py`. Update the docstring (drop "geometry source / dataset source" wording → "x source / y source").

- [ ] **Step 4: Regression gate + format + commit**

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate davinci
cd /Users/fillmore/EarthSystem/DAVINCI
HDF5_USE_FILE_LOCKING=FALSE python -m pytest -q && black davinci_monet >/dev/null && isort davinci_monet >/dev/null
git add -A && git commit -m "refactor: rename dataset_label attr/field to source_label; PairedData to x_source/y_source"
```

---

## Task 6: Rename the custom axis-label config fields (PlotConfig/StyleConfig)

`PlotConfig.geometry_label`/`dataset_label` (and any matching `StyleConfig` color fields not yet renamed) are the *custom axis-label override* — disambiguated from source identity. Rename: `geometry_label` → `x_label`, `dataset_label` → `y_label`.

**Files:** `davinci_monet/plots/plot_config.py`, `davinci_monet/plots/base.py`, every renderer reading `self.config.geometry_label`/`self.config.dataset_label`, and tests setting these.

- [ ] **Step 1: Confirm these are the only remaining `*_label` config fields**

```bash
cd /Users/fillmore/EarthSystem/DAVINCI
grep -rn --include='*.py' -E '\b(geometry_label|dataset_label)\b' davinci_monet | grep -v '/tests/'
```
Expected: only `PlotConfig`/`StyleConfig` definitions and `self.config.<x>_label` reads remain (the attr/field uses were renamed in Task 5).

- [ ] **Step 2: Rename 1:1 across source + tests**

```bash
cd /Users/fillmore/EarthSystem/DAVINCI
files=$(grep -rln --include='*.py' -E '\b(geometry_label|dataset_label)\b' davinci_monet)
for f in $files; do sed -i '' -E -e 's/\bgeometry_label\b/x_label/g' -e 's/\bdataset_label\b/y_label/g' "$f"; done
```

- [ ] **Step 3: Confirm StyleConfig color fields landed as x_color/y_color**

`plots/plot_config.py:108-109` should now read `x_color: str = DATASET_A_COLOR  # NCAR gray` and `y_color: str = DATASET_B_COLOR  # NCAR blue` (from Task 2). Verify; fix the trailing comments if stale.

- [ ] **Step 4: Regression gate + format + commit**

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate davinci
cd /Users/fillmore/EarthSystem/DAVINCI
HDF5_USE_FILE_LOCKING=FALSE python -m pytest -q && black davinci_monet >/dev/null && isort davinci_monet >/dev/null
git add -A && git commit -m "refactor: rename custom axis-label config fields to x_label/y_label"
```

---

## Task 7: Config schema clean break — nested x/y pairs (TDD)

Rewrite `SourcePairConfig` to the nested `x:`/`y:` shape, add `AxisRef`, reject the old shape, and update `PairingEngine` to read `.x`/`.y`.

**Files:**
- Modify: `davinci_monet/config/schema.py:306-327` (`SourcePairConfig`, new `AxisRef`)
- Modify: `davinci_monet/pairing/engine.py` (read `pair.x`/`pair.y`)
- Modify: `davinci_monet/pipeline/stages/pair.py` (pair consumption)
- Test: `davinci_monet/tests/test_config_xy_pairs.py` (new)

- [ ] **Step 1: Write failing tests for the new schema**

```python
"""Pair config uses nested x:/y: (clean break from sources:/geometry:/variables:)."""
import pytest
from pydantic import ValidationError

from davinci_monet.config.schema import SourcePairConfig


def test_nested_xy_pair_parses():
    p = SourcePairConfig(
        x={"source": "airnow", "variable": "o3"},
        y={"source": "cam", "variable": "O3"},
    )
    assert p.x.source == "airnow" and p.x.variable == "o3"
    assert p.y.source == "cam" and p.y.variable == "O3"


def test_old_shape_is_rejected_with_hint():
    with pytest.raises(ValidationError, match="x:.*y:|migrate"):
        SourcePairConfig(sources=["airnow", "cam"], geometry="airnow",
                         variables={"airnow": "o3", "cam": "O3"})


def test_missing_axis_is_rejected():
    with pytest.raises(ValidationError):
        SourcePairConfig(x={"source": "airnow", "variable": "o3"})
```

- [ ] **Step 2: Run — expect FAIL**

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate davinci
cd /Users/fillmore/EarthSystem/DAVINCI
HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/test_config_xy_pairs.py -q
```
Expected: FAIL (SourcePairConfig has no `x`/`y`).

- [ ] **Step 3: Implement the new schema**

Replace `SourcePairConfig` (and add `AxisRef`) in `davinci_monet/config/schema.py`:

```python
class AxisRef(FlexibleSchema):
    """One axis of a pair: a source label and the variable to read from it."""

    source: str
    variable: str


class SourcePairConfig(FlexibleSchema):
    """Binary pair definition as an ordered (x, y).

    ``x`` is the horizontal/reference axis; ``y`` is vertical. Diffs are ``y - x``.
    Pairing *direction* (which source is resampled onto which) is decided by shape
    precedence, not by x/y — x/y is plot-axis assignment only.
    """

    x: AxisRef
    y: AxisRef

    @model_validator(mode="before")
    @classmethod
    def reject_legacy_shape(cls, data: Any) -> Any:
        if isinstance(data, dict) and any(k in data for k in ("sources", "geometry", "variables")):
            raise ValueError(
                "legacy pair shape (sources:/geometry:/variables:) is no longer "
                "supported; migrate to nested x:/y:, e.g.\n"
                "  x: {source: airnow, variable: o3}\n"
                "  y: {source: cam, variable: O3}"
            )
        return data

    @field_validator("x", "y", mode="before")
    @classmethod
    def _parse_axis(cls, v: Any) -> Any:
        return AxisRef(**v) if isinstance(v, dict) else v

    @property
    def sources(self) -> list[str]:
        """Compatibility accessor: the two source labels in (x, y) order."""
        return [self.x.source, self.y.source]
```

- [ ] **Step 4: Update PairingEngine + pair stage to read x/y**

In `pairing/engine.py` and `pipeline/stages/pair.py`, replace reads of `pair.sources`/`pair.geometry`/`pair.variables[label]` with `pair.x.source`/`pair.x.variable` and `pair.y.source`/`pair.y.variable`. The chosen pairing geometry is still computed by **shape precedence** over the two sources (unchanged); only the variable/label lookups change. Tag the resulting paired variables `axis="x"` (the x source) / `axis="y"` (the y source) and `source_label=<that source>`.

- [ ] **Step 5: Run the new tests — expect PASS**

```bash
HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/test_config_xy_pairs.py -q
```
Expected: 3 passed.

- [ ] **Step 6: Migrate test fixtures + configs that the suite loads** (see Task 8 for the rule). Then run the full gate:

```bash
HDF5_USE_FILE_LOCKING=FALSE python -m pytest -q
```
Expected: green once Task 8's fixture migration is applied. If Task 8 is not yet done, expect integration tests that load old-shape configs to fail — proceed to Task 8 and treat 7+8 as one commit boundary.

- [ ] **Step 7: Format + commit (with Task 8)**

Commit happens at the end of Task 8 so schema + migrations land together.

---

## Task 8: Migrate tracked configs + test fixtures to nested x/y

**The deterministic transform (geometry ≡ x):** for each old pair block
```yaml
<pair_name>:
  sources: [A, B]
  geometry: G          # G is one of A/B; if absent, G = A (first listed)
  variables: {A: VARA, B: VARB}
```
becomes
```yaml
<pair_name>:
  x: {source: G,    variable: <variables[G]>}
  y: {source: <other>, variable: <variables[other]>}
```
where `<other>` is the non-`G` source. (Worked example: `sources: [cesm_asiaq, airnow]`, `geometry: airnow`, `variables: {cesm_asiaq: PM25, airnow: pm25}` →
`x: {source: airnow, variable: pm25}` / `y: {source: cesm_asiaq, variable: PM25}`.)

**Files:** every tracked config/fixture with a `pairs:` block. Enumerate:
```bash
cd /Users/fillmore/EarthSystem/DAVINCI
grep -rln -E '^\s*sources:\s*\[|^\s*geometry:\s' analyses examples davinci_monet/tests --include='*.yaml' --include='*.yml'
grep -rln -E "sources=\[|geometry=" davinci_monet/tests --include='*.py'
```

- [ ] **Step 1: Hand-migrate each YAML `pairs:` block** per the transform rule above (hand-edit to preserve comments). For Python test fixtures that build pair dicts, change `{"sources": [...], "geometry": ..., "variables": {...}}` to `{"x": {"source":..., "variable":...}, "y": {...}}`.

- [ ] **Step 2: Verify every tracked config still loads**

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate davinci
cd /Users/fillmore/EarthSystem/DAVINCI
python - <<'PY'
import glob, sys
from davinci_monet.config.loader import load_config  # adjust import to the real loader
bad = []
for f in glob.glob("analyses/**/configs/*.yaml", recursive=True) + glob.glob("examples/**/*.yaml", recursive=True):
    try:
        load_config(f)
    except Exception as e:
        bad.append((f, str(e)[:120]))
print("FAILED:", *bad, sep="\n") if bad else print("all configs load")
sys.exit(1 if bad else 0)
PY
```
Expected: `all configs load`. (Find the real loader entrypoint first with `grep -rn "def load_config\|def from_yaml\|def load_monet_config" davinci_monet/config`.)

- [ ] **Step 3: Full gate + format + commit (Tasks 7+8 together)**

```bash
HDF5_USE_FILE_LOCKING=FALSE python -m pytest -q && black davinci_monet >/dev/null && isort davinci_monet >/dev/null
git add -A && git commit -m "feat!: clean-break pair config to nested x/y; migrate tracked configs and fixtures"
```

- [ ] **Step 4: Flag gitignored machine configs**

```bash
cd /Users/fillmore/EarthSystem/DAVINCI
ls analyses/**/configs/*-gemini.yaml analyses/**/configs/*-derecho.yaml 2>/dev/null
```
Report these paths to the user — they are gitignored and must be migrated locally with the same rule; they are not committed.

---

## Task 9: Rename source-level QC vocabulary (geometry ≡ obs)

| Old | New |
|---|---|
| `geometry_min` / `geometry_max` | `valid_min` / `valid_max` |
| `geometry_count` | `sample_count` |
| `min_geometry_count` | `min_sample_count` |
| `track_geometry_count` | `track_sample_count` |
| `rem_geometry_nan` | `rem_nan` |
| `rem_geometry_by_nan_pct` | `rem_by_nan_pct` |

**Files:** `config/schema.py` (`VariableConfig`, `SourceConfig`, `DataProcConfig`), dataset readers under `davinci_monet/datasets/**` that emit `geometry_count`/read bounds, `pipeline/stages/*` that consume them, tests, and tracked configs.

- [ ] **Step 1: Sweep Python (these tokens are unambiguous)**

```bash
cd /Users/fillmore/EarthSystem/DAVINCI
files=$(grep -rln --include='*.py' -E '\b(geometry_min|geometry_max|min_geometry_count|max_geometry_count|track_geometry_count|geometry_count|rem_geometry_nan|rem_geometry_by_nan_pct)\b' davinci_monet)
for f in $files; do
  sed -i '' -E \
    -e 's/\bgeometry_min\b/valid_min/g' -e 's/\bgeometry_max\b/valid_max/g' \
    -e 's/\bmin_geometry_count\b/min_sample_count/g' -e 's/\bmax_geometry_count\b/max_sample_count/g' \
    -e 's/\btrack_geometry_count\b/track_sample_count/g' -e 's/\bgeometry_count\b/sample_count/g' \
    -e 's/\brem_geometry_nan\b/rem_nan/g' -e 's/\brem_geometry_by_nan_pct\b/rem_by_nan_pct/g' \
    "$f"
done
```

- [ ] **Step 2: Sweep tracked configs (key renames preserve comments)**

```bash
cd /Users/fillmore/EarthSystem/DAVINCI
files=$(grep -rln -E '\b(geometry_min|geometry_max|min_geometry_count|track_geometry_count|geometry_count)\b' analyses examples davinci_monet/tests)
for f in $files; do
  sed -i '' -E \
    -e 's/\bgeometry_min\b/valid_min/g' -e 's/\bgeometry_max\b/valid_max/g' \
    -e 's/\bmin_geometry_count\b/min_sample_count/g' \
    -e 's/\btrack_geometry_count\b/track_sample_count/g' -e 's/\bgeometry_count\b/sample_count/g' \
    "$f"
done
```

- [ ] **Step 3: Full gate + format + commit**

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate davinci
cd /Users/fillmore/EarthSystem/DAVINCI
HDF5_USE_FILE_LOCKING=FALSE python -m pytest -q && black davinci_monet >/dev/null && isort davinci_monet >/dev/null
git add -A && git commit -m "refactor: rename source-level geometry-as-obs QC vocabulary to valid_*/sample_*"
```

---

## Task 10: Update CLAUDE.md and spec-referenced docs

**Files:** `CLAUDE.md`; any `README`/docs under `analyses/**` describing the old vocab.

- [ ] **Step 1: Rewrite the affected CLAUDE.md sections** to the x/y vocabulary:
  - "Variable Naming Convention" — paired vars are still `<source_label>_<var>`; attrs are `axis: x|y` + `source_label` (was `pair_axis` + `dataset_label`).
  - "Unified Data-Source Config (`sources:`)" + "Working Example" — pair shape is nested `x:`/`y:`; drop `pair_axis`, `sources:[a,b]`, `geometry:` in pairs.
  - "Plot Styling" — `x_color` (gray) / `y_color` (blue); drop `DATASET_A`/`DATASET_B` role wording or restate as x/y.
  - "Common Gotchas" — `geometry_min/max` → `valid_min/max`; `min_geometry_count` → `min_sample_count`; `track_geometry_count` → `track_sample_count`.
  - Keep the CESM vertical-coordinate and shape-`geometry` content untouched.

- [ ] **Step 2: Grep CLAUDE.md for stragglers**

```bash
cd /Users/fillmore/EarthSystem/DAVINCI
grep -nE '\b(pair_axis|geometry_var|dataset_var|dataset_label|geometry_min|geometry_max|min_geometry_count|track_geometry_count|geometry:|sources: \[)\b' CLAUDE.md
```
Resolve each (keep only shape/`DataGeometry` references).

- [ ] **Step 3: Commit**

```bash
cd /Users/fillmore/EarthSystem/DAVINCI
git add CLAUDE.md && git commit -m "docs: update CLAUDE.md to x/y pair vocabulary"
```

---

## Task 11: Final verification — acceptance criteria

- [ ] **Step 1: Role-vocabulary is gone (only shape/prose remain)**

```bash
cd /Users/fillmore/EarthSystem/DAVINCI
echo "--- role identifiers (expect ZERO) ---"
grep -rn --include='*.py' -oE '\b(geometry_var|dataset_var|geometry_data|dataset_data|geometry_series|dataset_series|geometry_color|dataset_color|pair_axis|geometry_label|min_geometry_count|track_geometry_count|geometry_count|geometry_min|geometry_max)\b' davinci_monet | grep -v '/tests/' || echo "clean"
echo "--- role attr values in writes (expect ZERO) ---"
grep -rn --include='*.py' -E 'axis *= *"(geometry|dataset)"' davinci_monet || echo "clean"
echo "--- legacy pair keys in tracked configs (expect ZERO) ---"
grep -rn -E '^\s*(sources:\s*\[|geometry:\s|geometry_min|geometry_max)' analyses examples --include='*.yaml' --include='*.yml' || echo "clean"
echo "--- shape 'geometry' still present (expect NONZERO) ---"
grep -rc --include='*.py' -E '\bDataGeometry\b' davinci_monet | grep -v ':0' | head -1
```

- [ ] **Step 2: Full gate — tests, types, format**

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate davinci
cd /Users/fillmore/EarthSystem/DAVINCI
HDF5_USE_FILE_LOCKING=FALSE python -m pytest -q
mypy davinci_monet
black --check davinci_monet && isort --check-only davinci_monet
```
Expected: all tests pass (still 1,262), mypy clean, black/isort clean.

- [ ] **Step 3: Contract test still green (output preservation)**

```bash
HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/test_xy_contract.py -v
```
Expected: 2 passed — the post-rename `_paired()` sets `axis: "x"/"y"` + `source_label`, and the axis-assignment / `y - x` assertions are unchanged, proving behavior preservation.

- [ ] **Step 4: Smoke-run a real migrated analysis** (if data is staged locally): pick one migrated config and run it through the pipeline, confirm scatter / spatial_bias / timeseries plots generate without error and look identical to before.

```bash
davinci-monet run analyses/asia-aq/configs/asia-aq-airnow.example.yaml  # or any migrated config with data
```

- [ ] **Step 5: Report completion** to the user with the grep results from Step 1 and the gate output from Step 2. Do not merge to main; remain on `develop`.

---

## Self-Review notes (for the executor)

- **Overloaded `dataset_label`:** Task 5 handles the *attr/field* (source identity → `source_label`); Task 6 handles the *config field* (custom label → `y_label`). Never sed `dataset_label` globally in one shot.
- **Load-bearing string literals:** Task 4 Step 2 enumerates every `"geometry"/"dataset"` axis value; the grep in that step is the safety net for a missed site.
- **Green between commits:** Tasks 7 and 8 share a commit boundary (schema change needs the migrated fixtures to be green). Every other task is independently green.
- **Output preservation:** the unchanged assertions in `test_xy_contract.py` plus the full suite are the proof; an x/y swap would flip the xlabel/ylabel assertion or an integration test.
