# Intermediate-Gridding Pairing — Phase 1 (2-D) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make intermediate gridding (bin sources onto a common uniform grid, pair cell-to-cell) available for **all geometries** via an opt-in per-pair `method: grid`, by generalizing the swath-only `SwathGridStrategy` into one geometry-agnostic `IntermediateGridStrategy` that bins **both** sources symmetrically onto a `(time, lon, lat)` grid.

**Architecture:** Rename `swath_grid.py` → `intermediate_grid.py` (`IntermediateGridStrategy`), preserving today's swath default path verbatim (the "match a source's grid" / bin-x-align-y behavior). Add a new **symmetric** path that flattens both sources to `(time, lon, lat, value)` points (via `xarray.broadcast_like`, which uniformly handles point/track/swath/grid), bins each with the existing numba core (`grid_binning.py`), and emits `x_<var>`/`y_<var>`/`x_sample_count`/`y_sample_count`. A per-pair `method: grid` + `grid:` config block routes the engine to the symmetric path, bypassing geometry auto-selection.

**Tech Stack:** Python 3.11/3.12, xarray, numpy, numba, pandas, pydantic, pytest, mypy, black, isort. Run in the `davinci` conda env.

**Spec:** `docs/superpowers/specs/2026-06-14-intermediate-grid-pairing-design.md`

---

## Conventions (every task)

**Environment (prefix test runs):**
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate davinci
```
**Gate after each task:**
```bash
cd /Users/fillmore/EarthSystem/DAVINCI
HDF5_USE_FILE_LOCKING=FALSE python -m pytest -q && mypy davinci_monet && black --check davinci_monet && isort --check-only davinci_monet
```
**Commit footer (required):**
```
Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
```
Do NOT push/merge — local commits on `develop` only.

---

## Task 1: Rename `swath_grid` → `intermediate_grid` (behavior-preserving move)

**Files:**
- Rename: `davinci_monet/pairing/strategies/swath_grid.py` → `davinci_monet/pairing/strategies/intermediate_grid.py`
- Modify: `davinci_monet/pairing/strategies/__init__.py`, `davinci_monet/pairing/engine.py:87-101`

- [ ] **Step 1: Move the file and rename the class, keeping a back-compat alias**

```bash
cd /Users/fillmore/EarthSystem/DAVINCI
git mv davinci_monet/pairing/strategies/swath_grid.py davinci_monet/pairing/strategies/intermediate_grid.py
```
In `intermediate_grid.py`: rename `class SwathGridStrategy` → `class IntermediateGridStrategy`. At the bottom of the file add:
```python
# Back-compat alias (the strategy generalized beyond swath in 2026-06).
SwathGridStrategy = IntermediateGridStrategy
```
Keep the module docstring but update its first line to: `"""Intermediate-gridding pairing strategy using numba-accelerated binning."""`

- [ ] **Step 2: Update the strategies package exports**

In `davinci_monet/pairing/strategies/__init__.py`, wherever `SwathGridStrategy` is imported/exported from `swath_grid`, change the import to:
```python
from davinci_monet.pairing.strategies.intermediate_grid import (
    IntermediateGridStrategy,
    SwathGridStrategy,
)
```
and add `"IntermediateGridStrategy"` to `__all__` (keep `"SwathGridStrategy"` too).

- [ ] **Step 3: Update the engine registration**

In `davinci_monet/pairing/engine.py:87-101`, change the import line `SwathGridStrategy,` (inside `_register_default_strategies`) to `IntermediateGridStrategy,` and the registration `self.register_strategy(SwathGridStrategy())` → `self.register_strategy(IntermediateGridStrategy())`. Update the adjacent comment to say "IntermediateGridStrategy (numba binning) is the production SWATH handler."

- [ ] **Step 4: Fix any other importers**

```bash
cd /Users/fillmore/EarthSystem/DAVINCI
grep -rln --include='*.py' 'swath_grid\|SwathGridStrategy' davinci_monet
```
For each non-test hit that imports from `...strategies.swath_grid`, repoint to `...strategies.intermediate_grid`. Tests importing `SwathGridStrategy` keep working via the alias.

- [ ] **Step 5: Run the gate (behavior-preserving — expect green)**

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate davinci
cd /Users/fillmore/EarthSystem/DAVINCI
HDF5_USE_FILE_LOCKING=FALSE python -m pytest -q && mypy davinci_monet
```
Expected: all pass (the swath tests now exercise `IntermediateGridStrategy` via the alias).

- [ ] **Step 6: Commit**

```bash
black davinci_monet >/dev/null && isort davinci_monet >/dev/null
git add -A && git commit -m "refactor: rename SwathGridStrategy -> IntermediateGridStrategy (behavior-preserving)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: Add the symmetric flatten-and-bin core to `IntermediateGridStrategy`

Add the new symmetric path: flatten both sources to points, bin each, assemble the paired grid. Dispatch on the presence of `horizontal_res` in kwargs (the `method: grid` signal); without it, the existing swath path runs unchanged.

**Files:**
- Modify: `davinci_monet/pairing/strategies/intermediate_grid.py`
- Test: `davinci_monet/tests/test_intermediate_grid.py` (new)

- [ ] **Step 1: Write the failing unit test (point + point symmetric binning)**

```python
"""IntermediateGridStrategy symmetric binning (method: grid, 2-D)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from davinci_monet.pairing.strategies.intermediate_grid import IntermediateGridStrategy


def _point_ds(lats, lons, vals, var, t="2024-02-01"):
    n = len(lats)
    time = pd.to_datetime([t] * n)
    return xr.Dataset(
        {var: (["site"], np.asarray(vals, float), {"units": "1"})},
        coords={
            "site": np.arange(n),
            "time": ("site", time),
            "latitude": ("site", np.asarray(lats, float)),
            "longitude": ("site", np.asarray(lons, float)),
        },
    )


def test_symmetric_bins_both_point_sources_cell_means():
    # Two points fall in the SAME 1-degree cell -> their mean; one alone elsewhere.
    x = _point_ds([10.2, 10.7, 40.5], [20.2, 20.6, 50.5], [1.0, 3.0, 9.0], "aod")
    y = _point_ds([10.4, 40.4], [20.4, 50.4], [2.0, 8.0], "AOD")
    paired = IntermediateGridStrategy().pair_sources(
        x_data=x, y_data=y,
        x_var="aod", y_var="AOD",
        x_source="obs", y_source="mod",
        horizontal_res=1.0, time_resolution="1D", min_sample_count=1,
    )
    assert list(paired["x_aod"].dims) == ["time", "lon", "lat"]
    assert "y_AOD" in paired and "x_sample_count" in paired and "y_sample_count" in paired
    xa = paired["x_aod"].squeeze("time", drop=True)
    ya = paired["y_AOD"].squeeze("time", drop=True)
    # the cell containing (~10.x, ~20.x): x mean = (1+3)/2 = 2.0, y = 2.0, counts 2 and 1
    cell_x = xa.sel(lat=10.5, lon=20.5, method="nearest").item()
    cell_y = ya.sel(lat=10.5, lon=20.5, method="nearest").item()
    assert cell_x == pytest.approx(2.0)
    assert cell_y == pytest.approx(2.0)
    assert paired["x_sample_count"].sel(lat=10.5, lon=20.5, method="nearest").max().item() == 2
    # tagged for downstream stats/plots
    assert paired["x_aod"].attrs["axis"] == "x" and paired["x_aod"].attrs["source_label"] == "obs"
    assert paired["y_AOD"].attrs["axis"] == "y" and paired["y_AOD"].attrs["source_label"] == "mod"


def test_min_sample_count_masks_sparse_cells():
    x = _point_ds([10.2, 10.7], [20.2, 20.6], [1.0, 3.0], "aod")  # 2 in one cell
    y = _point_ds([10.4], [20.4], [2.0], "AOD")                    # 1 in that cell
    paired = IntermediateGridStrategy().pair_sources(
        x_data=x, y_data=y, x_var="aod", y_var="AOD",
        x_source="obs", y_source="mod",
        horizontal_res=1.0, time_resolution="1D", min_sample_count=2,
    )
    # y has only 1 sample in the cell -> masked to NaN under min_sample_count=2
    ya = paired["y_AOD"].squeeze("time", drop=True)
    assert np.isnan(ya.sel(lat=10.5, lon=20.5, method="nearest").item())
```

- [ ] **Step 2: Run — expect FAIL** (no `horizontal_res` path yet; current pair_sources ignores it and runs the swath path, which needs a y time dim / errors)

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate davinci
cd /Users/fillmore/EarthSystem/DAVINCI
HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/test_intermediate_grid.py -q
```
Expected: FAIL.

- [ ] **Step 3: Implement the symmetric path**

In `intermediate_grid.py`, add `import math` is not needed; ensure `numpy as np`, `pandas as pd`, `xarray as xr` imported (already are). At the **top of `pair_sources`**, add the dispatch before the existing swath logic:
```python
        if kwargs.get("horizontal_res") is not None:
            return self._pair_symmetric(
                x_data, x_data_var=kwargs.get("x_var"),
                y_data=y_data, y_data_var=kwargs.get("y_var"),
                x_source=kwargs.get("x_source"), y_source=kwargs.get("y_source"),
                horizontal_res=float(kwargs["horizontal_res"]),
                extent=kwargs.get("extent"),
                time_resolution=kwargs.get("time_resolution", "1D"),
                min_sample_count=int(kwargs.get("min_sample_count", 1)),
            )
```
Then add these methods to the class:
```python
    def _pair_symmetric(
        self,
        x_data: xr.Dataset,
        *,
        x_data_var: str | None,
        y_data: xr.Dataset,
        y_data_var: str | None,
        x_source: str | None,
        y_source: str | None,
        horizontal_res: float,
        extent: tuple[float, float, float, float] | None,
        time_resolution: str,
        min_sample_count: int,
    ) -> xr.Dataset:
        """Bin BOTH sources onto a common uniform (time, lon, lat) grid and pair."""
        x_var = x_data_var or str(list(x_data.data_vars)[0])
        y_var = y_data_var or str(list(y_data.data_vars)[0])
        # Phase 1 is 2-D: reduce any vertical dim to the surface for both sources.
        x_proc = self._reduce_to_surface(x_data)
        y_proc = self._reduce_to_surface(y_data)

        lon_centers, lat_centers, lon_edges, lat_edges = self._uniform_horizontal_grid(
            [x_proc, y_proc], horizontal_res, extent
        )
        time_centers_epoch, time_edges, time_coords = self._uniform_time_grid(
            [x_proc, y_proc], time_resolution
        )

        x_grid, x_count = self._bin_one_source(
            x_proc, x_var, time_edges, lon_edges, lat_edges,
            len(time_centers_epoch), len(lon_centers), len(lat_centers), min_sample_count,
        )
        y_grid, y_count = self._bin_one_source(
            y_proc, y_var, time_edges, lon_edges, lat_edges,
            len(time_centers_epoch), len(lon_centers), len(lat_centers), min_sample_count,
        )

        paired = xr.Dataset(
            {
                f"x_{x_var}": (["time", "lon", "lat"], x_grid.astype(np.float32)),
                f"y_{y_var}": (["time", "lon", "lat"], y_grid.astype(np.float32)),
                "x_sample_count": (["time", "lon", "lat"], x_count),
                "y_sample_count": (["time", "lon", "lat"], y_count),
            },
            coords={"time": time_coords, "lon": lon_centers, "lat": lat_centers},
        )
        paired[f"x_{x_var}"].attrs.update({"axis": "x", "source_label": x_source or ""})
        paired[f"y_{y_var}"].attrs.update({"axis": "y", "source_label": y_source or ""})
        paired.attrs.update({"created_by": "davinci_monet", "paired": True})
        return paired

    def _reduce_to_surface(self, ds: xr.Dataset) -> xr.Dataset:
        for dim_name in ("lev", "z", "level"):
            if dim_name in ds.dims:
                return self._extract_surface(ds, dim_name)
        return ds

    def _flatten_to_points(
        self, ds: xr.Dataset, var: str
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Flatten a source variable to (time_epoch, lon, lat, value) flat arrays.

        Uses ``broadcast_like`` so point/track/swath/grid all reduce uniformly:
        lat/lon (and time) are broadcast against the data variable's dims, then
        flattened in the variable's dim order (consistent C-order across arrays).
        """
        da = ds[var]
        lat, lon = self._get_x_coords(ds)
        order = da.dims
        data_flat = da.transpose(*order).values.astype(np.float64).flatten()
        lat_flat = lat.broadcast_like(da).transpose(*order).values.astype(np.float64).flatten()
        lon_flat = lon.broadcast_like(da).transpose(*order).values.astype(np.float64).flatten()
        if "time" in ds.coords or "time" in ds.dims:
            t = ds["time"]
            tvals = t.values
            if np.issubdtype(tvals.dtype, np.datetime64):
                epoch = tvals.astype("datetime64[s]").astype(np.float64)
            else:
                epoch = np.asarray(tvals, dtype=np.float64)
            epoch_da = xr.DataArray(epoch, dims=t.dims)
            time_flat = (
                epoch_da.broadcast_like(da).transpose(*order).values.astype(np.float64).flatten()
            )
        else:
            time_flat = np.zeros_like(data_flat)
        return time_flat, lon_flat, lat_flat, data_flat

    def _bin_one_source(
        self, ds, var, time_edges, lon_edges, lat_edges, ntime, nlon, nlat, min_sample_count
    ) -> tuple[np.ndarray, np.ndarray]:
        from davinci_monet.pairing.grid_binning import bin_swath_to_grid, normalize_grid

        time_flat, lon_flat, lat_flat, data_flat = self._flatten_to_points(ds, var)
        if lon_edges[0] >= 0 and np.any(lon_flat < 0):
            lon_flat = np.where(lon_flat < 0, lon_flat + 360.0, lon_flat)
        count = np.zeros((ntime, nlon, nlat), dtype=np.int32)
        acc = np.zeros((ntime, nlon, nlat), dtype=np.float64)
        bin_swath_to_grid(
            time_edges, lon_edges, lat_edges, time_flat, lon_flat, lat_flat, data_flat, count, acc
        )
        normalize_grid(count, acc)
        if min_sample_count > 1:
            acc[count < min_sample_count] = np.nan
        return acc, count

    def _uniform_horizontal_grid(self, datasets, res, extent):
        from davinci_monet.pairing.grid_binning import edges_from_centers

        if extent is not None:
            lon0, lon1, lat0, lat1 = (float(v) for v in extent)
        else:
            lons, lats = [], []
            for ds in datasets:
                lat, lon = self._get_x_coords(ds)
                lons.append(float(np.nanmin(lon.values))); lons.append(float(np.nanmax(lon.values)))
                lats.append(float(np.nanmin(lat.values))); lats.append(float(np.nanmax(lat.values)))
            lon0, lon1, lat0, lat1 = min(lons), max(lons), min(lats), max(lats)
        lat_centers = np.arange(lat0 + res / 2, lat1 + res / 2, res, dtype=np.float64)
        lon_centers = np.arange(lon0 + res / 2, lon1 + res / 2, res, dtype=np.float64)
        if len(lat_centers) == 0:
            lat_centers = np.array([(lat0 + lat1) / 2.0])
        if len(lon_centers) == 0:
            lon_centers = np.array([(lon0 + lon1) / 2.0])
        return lon_centers, lat_centers, edges_from_centers(lon_centers), edges_from_centers(lat_centers)

    def _uniform_time_grid(self, datasets, time_resolution):
        from davinci_monet.pairing.grid_binning import edges_from_centers

        starts, ends = [], []
        for ds in datasets:
            if "time" in ds.coords or "time" in ds.dims:
                ti = pd.DatetimeIndex(np.atleast_1d(ds["time"].values).ravel())
                starts.append(ti.min()); ends.append(ti.max())
        if not starts:
            t0 = pd.Timestamp("1970-01-01"); rng = pd.DatetimeIndex([t0])
        else:
            rng = pd.date_range(min(starts), max(ends), freq=time_resolution)
            if len(rng) < 1:
                rng = pd.DatetimeIndex([min(starts)])
        centers = rng.values.astype("datetime64[s]").astype(np.float64)
        return centers, edges_from_centers(centers), pd.to_datetime(centers, unit="s")
```

- [ ] **Step 4: Run the unit tests — expect PASS**

```bash
HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/test_intermediate_grid.py -q
```
Expected: 2 passed.

- [ ] **Step 5: Run the gate + commit**

```bash
HDF5_USE_FILE_LOCKING=FALSE python -m pytest -q && mypy davinci_monet
black davinci_monet >/dev/null && isort davinci_monet >/dev/null
git add -A && git commit -m "feat: symmetric bin-both intermediate gridding path (method: grid, 2-D)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: Config schema — `method` + `GridConfig`

**Files:**
- Modify: `davinci_monet/config/schema.py` (near `SourcePairConfig` / `AxisRef`)
- Test: `davinci_monet/tests/test_config_grid_method.py` (new)

- [ ] **Step 1: Write failing tests**

```python
import pytest
from pydantic import ValidationError
from davinci_monet.config.schema import SourcePairConfig


def test_method_grid_parses_with_grid_block():
    p = SourcePairConfig(
        x={"source": "a", "variable": "v"},
        y={"source": "b", "variable": "V"},
        method="grid",
        grid={"horizontal_res": 0.5, "time_resolution": "1D", "min_sample_count": 1},
    )
    assert p.method == "grid"
    assert p.grid is not None and p.grid.horizontal_res == 0.5


def test_method_defaults_to_auto():
    p = SourcePairConfig(x={"source": "a", "variable": "v"}, y={"source": "b", "variable": "V"})
    assert p.method == "auto" and p.grid is None


def test_method_grid_requires_grid_block():
    with pytest.raises(ValidationError, match="grid"):
        SourcePairConfig(x={"source": "a", "variable": "v"},
                         y={"source": "b", "variable": "V"}, method="grid")


def test_auto_with_grid_block_is_rejected():
    with pytest.raises(ValidationError, match="auto"):
        SourcePairConfig(x={"source": "a", "variable": "v"},
                         y={"source": "b", "variable": "V"},
                         method="auto", grid={"horizontal_res": 0.5})
```

- [ ] **Step 2: Run — expect FAIL**

```bash
HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/test_config_grid_method.py -q
```

- [ ] **Step 3: Implement the schema**

In `davinci_monet/config/schema.py`, add `GridConfig` ABOVE `SourcePairConfig`, ensure `Literal` is imported (it is), and add the fields + validator to `SourcePairConfig`:
```python
class GridConfig(FlexibleSchema):
    """Intermediate-grid settings for a pair using ``method: grid`` (2-D, Phase 1)."""

    horizontal_res: float
    extent: tuple[float, float, float, float] | None = None
    time_resolution: str = "1D"
    min_sample_count: int = 1
```
In `SourcePairConfig`, add after `y: AxisRef`:
```python
    method: Literal["auto", "grid"] = "auto"
    grid: GridConfig | None = None

    @field_validator("grid", mode="before")
    @classmethod
    def _parse_grid(cls, v: Any) -> Any:
        return GridConfig(**v) if isinstance(v, dict) else v

    @model_validator(mode="after")
    def _validate_method_grid(self) -> "SourcePairConfig":
        if self.method == "grid" and self.grid is None:
            raise ValueError("method: grid requires a 'grid:' block with horizontal_res")
        if self.method == "auto" and self.grid is not None:
            raise ValueError("'grid:' is only valid with method: grid (got method: auto)")
        return self
```

- [ ] **Step 4: Run — expect PASS**, then gate + commit

```bash
HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/test_config_grid_method.py -q
HDF5_USE_FILE_LOCKING=FALSE python -m pytest -q && mypy davinci_monet
black davinci_monet >/dev/null && isort davinci_monet >/dev/null
git add -A && git commit -m "feat: pair config method: grid + GridConfig (intermediate gridding opt-in)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: Engine routing for `method: grid`

When a pair specifies `method: grid`, the engine must use `IntermediateGridStrategy`'s symmetric path regardless of geometry (bypass `get_strategy_for`).

**Files:**
- Modify: `davinci_monet/pairing/engine.py` (`pair_sources`, ~line 192 onward)
- Test: `davinci_monet/tests/test_intermediate_grid.py` (add an engine-level test)

- [ ] **Step 1: Add the failing engine test** (append to `test_intermediate_grid.py`)

```python
def test_engine_routes_method_grid_to_symmetric():
    from davinci_monet.pairing.engine import PairingEngine
    x = _point_ds([10.2, 10.7], [20.2, 20.6], [1.0, 3.0], "aod")
    y = _point_ds([10.4], [20.4], [2.0], "AOD")
    paired = PairingEngine().pair_sources(
        x_data=x, y_data=y, x_vars=["aod"], y_vars=["AOD"],
        x_source="obs", y_source="mod",
        method="grid", horizontal_res=1.0, time_resolution="1D", min_sample_count=1,
    )
    data = paired.data if hasattr(paired, "data") else paired
    assert "x_aod" in data and "y_AOD" in data
    assert list(data["x_aod"].dims) == ["time", "lon", "lat"]
```

- [ ] **Step 2: Run — expect FAIL** (engine has no `method` param / doesn't route)

- [ ] **Step 3: Implement routing in `engine.pair_sources`**

Read the current `pair_sources` body (`engine.py:192`+). Add a `method: str = "auto"` keyword parameter to its signature, and near the top — after extracting `x_var`/`y_var` and BEFORE the geometry detection / `get_strategy_for` call — insert:
```python
        if method == "grid":
            from davinci_monet.pairing.strategies.intermediate_grid import (
                IntermediateGridStrategy,
            )
            strategy = IntermediateGridStrategy()
            paired_ds = strategy.pair_sources(
                x_data=x_data, y_data=y_data,
                x_var=x_vars[0], y_var=y_vars[0],
                x_source=kwargs.get("x_source"), y_source=kwargs.get("y_source"),
                horizontal_res=kwargs.get("horizontal_res"),
                extent=kwargs.get("extent"),
                time_resolution=kwargs.get("time_resolution", "1D"),
                min_sample_count=kwargs.get("min_sample_count", 1),
            )
            return self._wrap_paired(paired_ds, x_data, y_data, kwargs)  # see Step 4
```
Match the EXISTING return convention of `pair_sources`. Read how the current method packages its result (it returns either an `xr.Dataset` or a `PairedData` — inspect the existing `return` at the end of `pair_sources`). If it returns a bare `xr.Dataset`, `return paired_ds` directly. If it builds a `PairedData`, reuse that construction (same `x_source`/`y_source` labels, geometry=`DataGeometry.GRID`). Mirror exactly what the non-grid path returns so downstream stages treat it identically.

- [ ] **Step 4: Match the return type**

Inspect the tail of `pair_sources` (`grep -n "return" davinci_monet/pairing/engine.py`). If it wraps in `PairedData.from_sources(...)`, wrap the grid result the same way with `geometry=DataGeometry.GRID`; otherwise return the `xr.Dataset`. Remove the placeholder `self._wrap_paired(...)` and inline the matching construction. (No new helper unless the existing code already has one.)

- [ ] **Step 5: Run the engine test + gate + commit**

```bash
HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/test_intermediate_grid.py -q
HDF5_USE_FILE_LOCKING=FALSE python -m pytest -q && mypy davinci_monet
black davinci_monet >/dev/null && isort davinci_monet >/dev/null
git add -A && git commit -m "feat: engine routes method: grid to symmetric intermediate gridding

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: Thread `method`/`grid` from pair config through the pair stage

**Files:**
- Modify: `davinci_monet/pipeline/stages/pair.py` (`_build_source_pair_jobs` ~line 150, `_strategy_options` ~line 207, `_run_pair_job` ~line 293)

- [ ] **Step 1: Make `method`/`grid` flow as strategy options**

In `pair.py` `_strategy_options` (line 207-225), the `grid:` block is a nested dict that must be FLATTENED into kwargs, and `method` must pass through. Replace the body with:
```python
        control_keys = {
            "x", "y", "radius_of_influence", "time_tolerance", "time_method",
            "max_pair_workers", "dask_pair_workers",
        }
        options = {k: v for k, v in pairing_config_dict.items() if k not in control_keys}
        options.update({k: v for k, v in pair_spec.items() if k not in control_keys})
        # Flatten a nested ``grid:`` block (method: grid) into top-level kwargs.
        grid_block = options.pop("grid", None)
        if isinstance(grid_block, dict):
            options.update(grid_block)
        return {k: v for k, v in options.items() if v is not None}
```
This makes `method`, `horizontal_res`, `extent`, `time_resolution`, `min_sample_count` all top-level entries of `job.strategy_options`.

- [ ] **Step 2: Pass `method` to the engine in `_run_pair_job`**

In `_run_pair_job` (line 294-305), the call already does `**job.strategy_options`, so `method`/`horizontal_res`/etc. already arrive as kwargs. Confirm `engine.pair_sources` accepts `method` as a keyword (Task 4 added it) and that `method` is in `strategy_options` (Step 1 keeps it — it's not a control key). No further change needed unless `method` collides; verify with:
```bash
grep -n "def pair_sources" davinci_monet/pairing/engine.py
```
Ensure `method` is an explicit param so it's not swallowed ambiguously by `**kwargs` alongside the explicit pass. (It is, from Task 4.)

- [ ] **Step 3: Gate + commit**

```bash
HDF5_USE_FILE_LOCKING=FALSE python -m pytest -q && mypy davinci_monet
black davinci_monet >/dev/null && isort davinci_monet >/dev/null
git add -A && git commit -m "feat: thread method/grid from pair config through the pair stage

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: Integration test through the pipeline (`method: grid`)

**Files:**
- Test: `davinci_monet/tests/test_intermediate_grid.py` (add the integration test)

- [ ] **Step 1: Add the failing integration test**

```python
@pytest.mark.integration
def test_method_grid_runs_through_pipeline(tmp_path):
    from davinci_monet.pipeline.runner import PipelineRunner

    x = _point_ds([10.2, 10.7, 40.5], [20.2, 20.6, 50.5], [1.0, 3.0, 9.0], "aod")
    y = _point_ds([10.4, 40.4], [20.4, 50.4], [2.0, 8.0], "AOD")
    xp, yp = tmp_path / "x.nc", tmp_path / "y.nc"
    x.to_netcdf(xp); y.to_netcdf(yp)

    config = {
        "analysis": {"output_dir": str(tmp_path / "out")},
        "sources": {
            "obs": {"type": "generic", "files": str(xp), "variables": {"aod": {"units": "1"}}},
            "mod": {"type": "generic", "files": str(yp), "variables": {"AOD": {"units": "1"}}},
        },
        "pairs": {
            "obs_vs_mod": {
                "x": {"source": "obs", "variable": "aod"},
                "y": {"source": "mod", "variable": "AOD"},
                "method": "grid",
                "grid": {"horizontal_res": 1.0, "time_resolution": "1D", "min_sample_count": 1},
            }
        },
        "plots": {"sc": {"type": "scatter", "data": ["obs_vs_mod"]}},
    }
    result = PipelineRunner(show_progress=False).run_from_config(config)
    assert result.success, getattr(result, "error", None)
    ctx = result.context
    assert ctx is not None
    assert "obs_vs_mod" in ctx.paired
```

- [ ] **Step 2: Run it — expect FAIL or surface integration gaps**

```bash
HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/test_intermediate_grid.py::test_method_grid_runs_through_pipeline -q
```
If it fails, investigate the REAL pipeline path (do not stub it): the failure points to a remaining wiring gap (e.g. `method` not reaching the engine, the paired object not stored in `context.paired`, or the generic reader not setting a geometry the direction-resolver accepts). Fix the wiring (likely in `pair.py`'s job execution / storage), not the test. The two generic point sources both have POINT geometry; `method: grid` must bypass the geometry-precedence direction logic — verify `resolve_pair_direction` isn't rejecting a point+point pair when `method: grid` (if it is, skip the direction validation when `method == "grid"` in `_build_source_pair_jobs`).

- [ ] **Step 3: Make it pass**, then gate + commit

```bash
HDF5_USE_FILE_LOCKING=FALSE python -m pytest -q && mypy davinci_monet
black davinci_monet >/dev/null && isort davinci_monet >/dev/null
git add -A && git commit -m "test: method: grid intermediate gridding runs end-to-end through the pipeline

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 7: Docs + final verification

**Files:**
- Modify: `CLAUDE.md` (document `method: grid`), `docs/superpowers/specs/...` (mark Phase 1 done)

- [ ] **Step 1: Document `method: grid` in CLAUDE.md** near the pairing/config sections — a short block:
```yaml
pairs:
  a_vs_b:
    x: {source: aeronet, variable: aod_500nm}
    y: {source: cam,     variable: AODVISdn}
    method: grid          # intermediate gridding: bin BOTH sources onto a common grid
    grid: { horizontal_res: 0.5, time_resolution: 1D, min_sample_count: 1 }
```
with one sentence: "`method: grid` bins both sources onto a uniform `(time, lon, lat)` grid and pairs cell-to-cell (symmetric); default `method: auto` keeps geometry-based pairing. 3-D altitude grids are Phase 2."

- [ ] **Step 2: Final gate**

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate davinci
cd /Users/fillmore/EarthSystem/DAVINCI
HDF5_USE_FILE_LOCKING=FALSE python -m pytest -q
mypy davinci_monet
black --check davinci_monet && isort --check-only davinci_monet
```
Expected: all green, mypy clean, format clean.

- [ ] **Step 3: Commit + report**

```bash
git add CLAUDE.md docs/superpowers && git commit -m "docs: document method: grid intermediate gridding (Phase 1)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```
Report the suite/mypy results. Do not push/merge.

---

## Self-Review notes

- **Spec coverage:** Decision 1 (generalize/replace) → Task 1; Decision 2 (symmetric bin both) → Task 2; Decision 3 (grid def: res/extent/time/min) → Task 2 `_uniform_*`; Decision 4 (config) → Task 3; Decision 5 (engine routing) → Task 4 + Task 5; Decision 6 (output `x_/y_/x_sample_count/y_sample_count` tagged axis/source_label) → Task 2 Step 3. Error handling (missing lat/lon via `_get_x_coords`; method:grid without grid → Task 3 validator; min_sample_count mask → Task 2) covered. Acceptance criteria 1-5 → Tasks 2/3/6 + gate.
- **Swath regression:** Task 1 preserves the swath path verbatim (alias + unchanged code); the symmetric path only runs when `horizontal_res` is present. Swath tests are the regression gate.
- **Return-type match (Task 4):** the one place needing the engineer to read existing code — the plan explicitly says to mirror `pair_sources`' existing return (PairedData vs Dataset). Verify before finalizing.
- **point+point direction (Task 6):** `resolve_pair_direction` may reject two same-geometry irregular sources; `method: grid` must bypass that — handled in Task 6 Step 2.
