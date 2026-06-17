# EOF Analysis (Plan B) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the `eof` derived analysis (decompose a 2-D/3-D gridded field into spatial modes + unit-variance PCs + explained variance) and its plots (`eof_pattern` maps, `eof_scree`), and wire PC time series through the existing `timeseries` renderer with a plot-time `mode:` selector.

**Architecture:** `EOFAnalysis` (in the `davinci_monet/analysis/` package from Plan A) uses `xeofs` for the decomposition but derives the **physical spatial modes by regressing the (de-weighted) anomaly field onto unit-variance PCs** — making the mode/PC scaling robust to xeofs's internal normalization. 3-D fields are decomposed as a full coupled state vector with area×layer-mass weighting. Outputs register as a pseudo-source (Plan A), so plots use the normal single-source path.

**Tech Stack:** xeofs (new dep), xarray, numpy, matplotlib/cartopy, the DAVINCI plotting framework.

**Prerequisite:** Plan A (`2026-06-17-derived-analysis-foundation.md`) complete and merged.

**Spec:** `docs/superpowers/specs/2026-06-17-eof-and-wavelet-analysis-design.md` (§4, §6.1-6.3, §11 Plan B).

**Conventions:** Same as Plan A. CRITICAL extra: the suite runs with `filterwarnings = ["error::UserWarning"]` — advisory messages (mass-weight fallback, standardize+3-D) MUST use the module **logger** (`logging.getLogger(__name__).warning(...)`), NOT `warnings.warn(...)`, or the suite fails. Always `plt.close(fig)` in tests.

---

## File Structure

- Create `davinci_monet/analysis/eof.py` — `EOFAnalysis` + private preprocessing/weighting helpers.
- Modify `davinci_monet/analysis/__init__.py` — import `eof` so it registers.
- Create `davinci_monet/plots/renderers/eof_pattern.py` — `EOFPatternPlotter` (extends `BaseSpatialPlotter`).
- Create `davinci_monet/plots/renderers/eof_scree.py` — `EOFScreePlotter` (extends `BasePlotter`).
- Modify `davinci_monet/plots/contracts.py` — register types in arity + category sets.
- Modify `davinci_monet/plots/renderers/__init__.py` — export new renderers.
- Modify `davinci_monet/config/schema.py` — add `mode`/`display_level` to `PlotGroupConfig`.
- Modify `davinci_monet/pipeline/stages/plot.py` — mode-selection before `build_series`.
- Modify `davinci_monet/pipeline/stages/plot_options.py` — forward `display_level` to render kwargs.
- Modify `pyproject.toml`, `environment.yml` — add pinned `xeofs`.
- Create tests under `davinci_monet/tests/unit/{plots,analysis}/` and `davinci_monet/tests/integration/`.

---

### Task 1: Add `xeofs` dependency + API probe

**Files:**
- Modify: `pyproject.toml` (dependencies), `environment.yml`
- Test: `davinci_monet/tests/unit/analysis/test_xeofs_api.py`

- [ ] **Step 1: Add the dependency and install**

In `pyproject.toml` add to the runtime dependencies array: `"xeofs>=2.3,<4"`. In `environment.yml` add `- xeofs>=2.3` under pip (or conda-forge if available). Then in the `davinci` env:

```bash
conda activate davinci
pip install "xeofs>=2.3,<4"
```

- [ ] **Step 2: Write an API-probe test (verifies the exact names this plan depends on)**

```python
"""Pin the xeofs API surface EOFAnalysis relies on (single.EOF + scores/ev)."""

from __future__ import annotations

import numpy as np
import xarray as xr


def test_xeofs_minimal_api() -> None:
    from xeofs.single import EOF  # import path used by EOFAnalysis

    rng = np.random.default_rng(0)
    da = xr.DataArray(
        rng.normal(size=(40, 6, 5)),
        dims=("time", "lat", "lon"),
        coords={"time": np.arange(40), "lat": np.linspace(0, 10, 6), "lon": np.linspace(0, 8, 5)},
    )
    model = EOF(n_modes=3, use_coslat=False, standardize=False)
    model.fit(da, dim="time")
    scores = model.scores()
    ev = model.explained_variance_ratio()
    assert "time" in scores.dims and "mode" in scores.dims
    assert "mode" in ev.dims
    assert int(ev.sizes["mode"]) == 3
```

- [ ] **Step 3: Run the probe**

Run: `HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/unit/analysis/test_xeofs_api.py -v`
Expected: PASS. **If it fails on import path or method names**, the installed xeofs differs — record the correct `EOF`/`scores`/`explained_variance_ratio` names and adjust this test AND `EOFAnalysis` (Task 3) to match before proceeding. Do not work around a real API mismatch.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml environment.yml davinci_monet/tests/unit/analysis/test_xeofs_api.py
git commit -m "build(analysis): add xeofs dependency + API probe"
```

---

### Task 2: EOF preprocessing & weighting helpers

**Files:**
- Create: `davinci_monet/analysis/eof.py` (helpers only in this task)
- Test: `davinci_monet/tests/unit/analysis/test_eof_helpers.py`

- [ ] **Step 1: Write the failing test**

```python
"""EOF preprocessing/weighting helpers behave correctly (and log, not warn)."""

from __future__ import annotations

import logging

import numpy as np
import xarray as xr

from davinci_monet.analysis.eof import (
    _area_weight,
    _fix_sign,
    _lat_coord,
    _layer_mass_weight,
    _vertical_dim,
)


def _grid(nt=10, nlat=4, nlon=5, nlev=0) -> xr.DataArray:
    lat = np.linspace(10, 40, nlat)
    lon = np.linspace(-120, -90, nlon)
    if nlev:
        dims = ("time", "lev", "lat", "lon")
        shape = (nt, nlev, nlat, nlon)
        coords = {"time": np.arange(nt), "lev": np.arange(nlev), "lat": lat, "lon": lon,
                  "latitude": ("lat", lat), "longitude": ("lon", lon)}
    else:
        dims = ("time", "lat", "lon")
        shape = (nt, nlat, nlon)
        coords = {"time": np.arange(nt), "lat": lat, "lon": lon,
                  "latitude": ("lat", lat), "longitude": ("lon", lon)}
    return xr.DataArray(np.ones(shape), dims=dims, coords=coords, name="O3")


def test_lat_and_area_weight() -> None:
    da = _grid()
    lat = _lat_coord(da)
    w = _area_weight(da, lat)
    # weight = sqrt(cos(lat)); decreasing with latitude
    assert float(w.isel(lat=0)) > float(w.isel(lat=-1))


def test_vertical_dim_detection() -> None:
    assert _vertical_dim(_grid(nlev=0), _lat_coord(_grid()), _grid()["longitude"]) is None
    da3 = _grid(nlev=3)
    assert _vertical_dim(da3, da3["latitude"], da3["longitude"]) == "lev"


def test_layer_mass_weight_fallback_logs_not_warns(caplog) -> None:
    da3 = _grid(nlev=3)
    with caplog.at_level(logging.WARNING):
        mw = _layer_mass_weight(da3.to_dataset(), "lev")
    assert mw is None  # no hybrid/ilev info -> None (caller logs + falls back)


def test_fix_sign_makes_max_loading_positive() -> None:
    mode = xr.DataArray(
        np.array([[-3.0, 1.0], [2.0, -0.5]]),
        dims=("mode", "lat"),
        coords={"mode": [1, 2], "lat": [0, 1]},
    )
    pc = xr.DataArray(np.ones((2, 4)), dims=("mode", "time"), coords={"mode": [1, 2], "time": np.arange(4)})
    m2, p2 = _fix_sign(mode, pc)
    # mode 1's largest |loading| (-3) becomes +3; its PC is flipped too
    assert float(m2.sel(mode=1).max()) == 3.0
    assert float(p2.sel(mode=1).isel(time=0)) == -1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/unit/analysis/test_eof_helpers.py -v`
Expected: FAIL — `davinci_monet.analysis.eof` does not exist.

- [ ] **Step 3: Create `eof.py` with the helpers**

```python
"""EOF (Empirical Orthogonal Function) decomposition of a gridded field."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

if TYPE_CHECKING:
    from davinci_monet.config.schema import EOFSpec

logger = logging.getLogger(__name__)

_LAT_NAMES = ("latitude", "lat", "LAT", "Latitude")
_LON_NAMES = ("longitude", "lon", "LON", "Longitude")


def _named_coord(da: xr.DataArray, names: tuple[str, ...], kind: str) -> xr.DataArray:
    for name in names:
        if name in da.coords:
            return da.coords[name]
    raise ValueError(f"EOF requires a {kind} coordinate (one of {names})")


def _lat_coord(da: xr.DataArray) -> xr.DataArray:
    return _named_coord(da, _LAT_NAMES, "latitude")


def _lon_coord(da: xr.DataArray) -> xr.DataArray:
    return _named_coord(da, _LON_NAMES, "longitude")


def _vertical_dim(da: xr.DataArray, lat: xr.DataArray, lon: xr.DataArray) -> str | None:
    horiz = set(lat.dims) | set(lon.dims)
    verts = [d for d in da.dims if d != "time" and d not in horiz]
    if len(verts) > 1:
        raise ValueError(f"EOF: ambiguous vertical dims {verts}; expected one")
    return verts[0] if verts else None


def _area_weight(da: xr.DataArray, lat: xr.DataArray) -> xr.DataArray:
    """sqrt(cos(lat)) broadcast over the latitude dimension."""
    coslat = np.cos(np.deg2rad(lat)).clip(min=0.0)
    return np.sqrt(coslat)


def _layer_mass_weight(data: xr.Dataset, vdim: str) -> xr.DataArray | None:
    """sqrt(normalized layer pressure thickness) over the vertical dim, or None.

    Uses ``ilev`` pressure edges if present, else CESM hybrid coefficients
    (hyai/hybi + PS or P0). Returns None when no vertical thickness info exists;
    the caller then falls back to equal layer weight (logged, not warned).
    """
    nlev = int(data.sizes[vdim])
    dp: np.ndarray | None = None
    if "ilev" in data.coords and int(data.sizes.get("ilev", 0)) == nlev + 1:
        dp = np.abs(np.diff(np.asarray(data["ilev"].values, dtype=float)))
    elif {"hyai", "hybi"} <= set(data.variables):
        p0 = float(data["P0"]) if "P0" in data.variables else 1.0e5
        ps = float(np.asarray(data["PS"].values).mean()) if "PS" in data.variables else p0
        edges = np.asarray(data["hyai"].values, float) * p0 + np.asarray(data["hybi"].values, float) * ps
        if edges.size == nlev + 1:
            dp = np.abs(np.diff(edges))
    if dp is None:
        return None
    dpn = dp / dp.sum()
    return xr.DataArray(np.sqrt(dpn), dims=[vdim])


def _fix_sign(mode: xr.DataArray, pc: xr.DataArray) -> tuple[xr.DataArray, xr.DataArray]:
    """Flip each mode so its largest-|loading| spatial point is positive.

    Deterministic and robust for dipole modes (a domain-mean rule is not).
    """
    spatial = [d for d in mode.dims if d != "mode"]
    flat = mode.stack(_pt=spatial)
    idx = np.abs(flat).argmax("_pt")
    peak = flat.isel(_pt=idx)
    signs = xr.where(peak >= 0, 1.0, -1.0)
    return mode * signs, pc * signs


def _effective_n(anom: xr.DataArray, lat: xr.DataArray) -> float:
    """Effective independent sample count from the area-mean series lag-1 autocorr."""
    coslat = np.cos(np.deg2rad(lat)).clip(min=0.0)
    spatial = [d for d in anom.dims if d != "time"]
    am = anom.weighted(coslat).mean(dim=spatial)
    x = np.asarray(am.values, dtype=float)
    x = x[np.isfinite(x)]
    n = int(len(x))
    if n < 3:
        return float(max(n, 1))
    r1 = float(np.corrcoef(x[:-1], x[1:])[0, 1])
    r1 = float(np.clip(r1, -0.99, 0.99))
    return n * (1.0 - r1) / (1.0 + r1)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/unit/analysis/test_eof_helpers.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add davinci_monet/analysis/eof.py davinci_monet/tests/unit/analysis/test_eof_helpers.py
git commit -m "feat(eof): preprocessing and weighting helpers"
```

---

### Task 3: `EOFAnalysis` core (2-D) — register + planted-pattern recovery

**Files:**
- Modify: `davinci_monet/analysis/eof.py` (add the `EOFAnalysis` class)
- Modify: `davinci_monet/analysis/__init__.py` (import to register)
- Test: `davinci_monet/tests/unit/analysis/test_eof_analysis.py`

- [ ] **Step 1: Write the failing test (planted patterns must be recovered)**

```python
"""EOFAnalysis recovers two planted orthogonal patterns from a 2-D field."""

from __future__ import annotations

import numpy as np
import xarray as xr

from davinci_monet.analysis.eof import EOFAnalysis
from davinci_monet.config.schema import EOFSpec


def _planted(nt=200, nlat=6, nlon=8, seed=0) -> tuple[xr.Dataset, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    lat = np.linspace(-5, 5, nlat)      # near equator: cos-lat ~ uniform
    lon = np.linspace(0, 30, nlon)
    x = np.linspace(0, np.pi, nlon)
    p1 = np.cos(x)[None, :] * np.ones((nlat, 1))          # zonal pattern 1
    p2 = np.cos(2 * x)[None, :] * np.ones((nlat, 1))      # orthogonal pattern 2
    pc1 = rng.normal(size=nt)
    pc2 = rng.normal(size=nt)
    field = (3.0 * pc1[:, None, None] * p1[None] + 1.0 * pc2[:, None, None] * p2[None]
             + 0.05 * rng.normal(size=(nt, nlat, nlon)))
    ds = xr.Dataset(
        {"O3": (("time", "lat", "lon"), field, {"units": "ppb"})},
        coords={"time": np.arange(nt), "lat": lat, "lon": lon,
                "latitude": ("lat", lat), "longitude": ("lon", lon)},
    )
    return ds, p1.ravel(), pc1


def _corr(a, b) -> float:
    a = np.asarray(a, float).ravel(); b = np.asarray(b, float).ravel()
    return abs(float(np.corrcoef(a, b)[0, 1]))


def test_eof_recovers_patterns_and_pcs() -> None:
    ds, p1, pc1 = _planted()
    spec = EOFSpec(type="eof", source="cam", variable="O3", n_modes=3)
    out = EOFAnalysis().analyze(ds, spec)

    assert set(out.data_vars) >= {"mode", "pc", "explained_variance", "explained_variance_error"}
    assert out["mode"].dims == ("mode", "lat", "lon")
    assert out["pc"].dims == ("time", "mode")
    # mode 1 ~ planted pattern 1; pc 1 ~ planted pc 1
    assert _corr(out["mode"].sel(mode=1).values, p1) > 0.95
    assert _corr(out["pc"].sel(mode=1).values, pc1) > 0.9
    # variance ordering
    ev = out["explained_variance"].values
    assert ev[0] > ev[1] > ev[2]
    # PCs are unit variance
    assert float(out["pc"].sel(mode=1).std()) == __import__("pytest").approx(1.0, abs=0.05)


def test_sign_is_deterministic() -> None:
    ds, _, _ = _planted(seed=1)
    spec = EOFSpec(type="eof", source="cam", variable="O3", n_modes=2)
    a = EOFAnalysis().analyze(ds, spec)
    b = EOFAnalysis().analyze(ds, spec)
    assert np.allclose(a["mode"].values, b["mode"].values)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/unit/analysis/test_eof_analysis.py -v`
Expected: FAIL — `cannot import name 'EOFAnalysis'`.

- [ ] **Step 3: Implement `EOFAnalysis`**

Append to `davinci_monet/analysis/eof.py`:

```python
from davinci_monet.analysis.base import DerivedAnalysis  # noqa: E402
from davinci_monet.core.protocols import DataGeometry  # noqa: E402
from davinci_monet.core.registry import analysis_registry  # noqa: E402


@analysis_registry.register("eof")
class EOFAnalysis(DerivedAnalysis):
    """Empirical Orthogonal Function decomposition of a gridded field."""

    name = "eof"
    long_name = "Empirical Orthogonal Function Decomposition"
    output_geometry = DataGeometry.GRID

    def analyze(self, data: xr.Dataset, spec: "EOFSpec") -> xr.Dataset:
        from xeofs.single import EOF, EOFRotator

        da = data[spec.variable]
        lat = _lat_coord(da)
        lon = _lon_coord(da)
        vdim = _vertical_dim(da, lat, lon)
        if spec.level is not None and vdim is not None:
            da = da.isel({vdim: spec.level})
            vdim = None

        # Anomalies (always remove time mean).
        anom = da - da.mean("time")
        if spec.remove_seasonal_cycle:
            clim = anom.groupby("time.month").mean("time")
            anom = anom.groupby("time.month") - clim
        if spec.standardize:
            std = anom.std("time")
            anom = anom / std.where(std > 0)

        # Weights: area (sqrt cos lat), plus layer-mass for a 3-D covariance EOF.
        weight = _area_weight(anom, lat)
        if vdim is not None and not spec.standardize:
            mw = _layer_mass_weight(data, vdim)
            if mw is None:
                logger.warning(
                    "EOF 3-D mass weighting unavailable for '%s'; using equal layer weight",
                    spec.variable,
                )
            else:
                weight = weight * mw
        elif vdim is not None and spec.standardize:
            logger.warning(
                "EOF standardize=True with a 3-D field: vertical mass weighting disabled "
                "(per-cell standardization already equalizes variance)"
            )
        weight = weight.fillna(0.0)

        # Decompose the weighted anomalies; xeofs gives PCs + variance ratios.
        weighted = (anom * weight).fillna(0.0)
        model: object = EOF(n_modes=spec.n_modes, use_coslat=False, standardize=False)
        model.fit(weighted, dim="time")
        if spec.rotation == "varimax":
            model = EOFRotator(n_modes=spec.n_modes).fit(model)

        scores = model.scores()                       # (mode, time)
        ev_ratio = model.explained_variance_ratio()   # (mode,)

        # Unit-variance PCs; physical modes by regression of de-weighted anomalies.
        pc = scores / scores.std("time")
        mode = (anom * pc).mean("time")               # (mode, <spatial>)
        mode, pc = _fix_sign(mode, pc)

        n_modes = int(ev_ratio.sizes["mode"])
        out_vars: dict[str, xr.DataArray] = {
            "mode": mode.assign_attrs(
                units=str(da.attrs.get("units", "")),
                long_name=f"EOF spatial mode of {spec.variable}",
                kind="mode",
            ),
            "pc": pc.transpose("time", "mode").assign_attrs(
                units="1", long_name=f"Principal component of {spec.variable}", kind="pc"
            ),
            "explained_variance": ev_ratio.assign_attrs(kind="scalar", percent=True),
        }
        if spec.rotation == "none":
            n_eff = _effective_n(anom, lat)
            out_vars["explained_variance_error"] = (ev_ratio * np.sqrt(2.0 / n_eff)).assign_attrs(
                kind="scalar"
            )

        ds = xr.Dataset(out_vars)
        ds = ds.assign_coords(mode=np.arange(1, n_modes + 1))
        ds.attrs["eof_quantity"] = spec.variable
        return ds
```

In `davinci_monet/analysis/__init__.py`, add the registration import:

```python
from davinci_monet.analysis import eof as _eof  # noqa: F401  (registers "eof")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/unit/analysis/test_eof_analysis.py -v`
Expected: PASS. If pattern correlation is just under threshold, confirm `scores`/`ev_ratio` dim names from the Task 1 probe; the regression-based mode is independent of xeofs scaling, so a failure here points to dim handling.

- [ ] **Step 5: Commit**

```bash
git add davinci_monet/analysis/eof.py davinci_monet/analysis/__init__.py davinci_monet/tests/unit/analysis/test_eof_analysis.py
git commit -m "feat(eof): EOFAnalysis core with regression-based physical modes"
```

---

### Task 4: 3-D state-vector EOF + mass-weight fallback

**Files:**
- Test: `davinci_monet/tests/unit/analysis/test_eof_3d.py`

- [ ] **Step 1: Write the failing test**

```python
"""3-D EOF produces coupled (mode, lev, lat, lon) modes and logs the mass-weight fallback."""

from __future__ import annotations

import logging

import numpy as np
import xarray as xr

from davinci_monet.analysis.eof import EOFAnalysis
from davinci_monet.config.schema import EOFSpec


def _planted_3d(nt=150, nlev=3, nlat=5, nlon=6, seed=2) -> xr.Dataset:
    rng = np.random.default_rng(seed)
    x = np.linspace(0, np.pi, nlon)
    vstruct = np.array([1.0, 0.6, 0.2])[:nlev]                 # vertical structure
    p1 = vstruct[:, None, None] * np.cos(x)[None, None, :]      # (lev, 1, lon)
    p1 = np.broadcast_to(p1, (nlev, nlat, nlon))
    pc1 = rng.normal(size=nt)
    field = 3.0 * pc1[:, None, None, None] * p1[None] + 0.05 * rng.normal(size=(nt, nlev, nlat, nlon))
    lat = np.linspace(-5, 5, nlat); lon = np.linspace(0, 30, nlon)
    return xr.Dataset(
        {"O3": (("time", "lev", "lat", "lon"), field, {"units": "ppb"})},
        coords={"time": np.arange(nt), "lev": np.arange(nlev), "lat": lat, "lon": lon,
                "latitude": ("lat", lat), "longitude": ("lon", lon)},
    )


def test_3d_eof_shapes_and_fallback_logs(caplog) -> None:
    ds = _planted_3d()
    spec = EOFSpec(type="eof", source="cam", variable="O3", n_modes=3)
    with caplog.at_level(logging.WARNING):
        out = EOFAnalysis().analyze(ds, spec)
    assert out["mode"].dims == ("mode", "lev", "lat", "lon")
    assert out["explained_variance"].values[0] > 0.8  # one dominant coupled mode
    assert any("mass weighting unavailable" in r.message for r in caplog.records)


def test_level_select_reduces_to_2d() -> None:
    ds = _planted_3d()
    spec = EOFSpec(type="eof", source="cam", variable="O3", n_modes=2, level=-1)
    out = EOFAnalysis().analyze(ds, spec)
    assert out["mode"].dims == ("mode", "lat", "lon")
```

- [ ] **Step 2: Run test to verify it fails or passes**

Run: `HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/unit/analysis/test_eof_3d.py -v`
Expected: PASS with the Task 3 implementation (it already handles 3-D + fallback + `level`). If `caplog` does not capture the message, confirm `logging.getLogger(__name__)` is used (NOT `warnings.warn`).

- [ ] **Step 3: (No new implementation unless the test surfaces a gap)**

If `mode.dims` ordering differs (xeofs may return feature dims in a different order), add a `.transpose("mode", vdim, lat_dim, lon_dim)` on `mode` in `analyze`, deriving the dim names from the input. Implement only if the test requires it.

- [ ] **Step 4: Commit**

```bash
git add davinci_monet/tests/unit/analysis/test_eof_3d.py
git commit -m "test(eof): 3-D coupled state vector + mass-weight fallback"
```

---

### Task 5: `eof_pattern` renderer

**Files:**
- Create: `davinci_monet/plots/renderers/eof_pattern.py`
- Modify: `davinci_monet/plots/contracts.py`, `davinci_monet/plots/renderers/__init__.py`
- Test: `davinci_monet/tests/unit/plots/test_eof_pattern.py`

- [ ] **Step 1: Write the failing test**

```python
"""eof_pattern renders one signed QuadMesh map per mode and slices the surface."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import xarray as xr  # noqa: E402
from matplotlib.collections import QuadMesh  # noqa: E402

from davinci_monet.plots.base import build_series  # noqa: E402
from davinci_monet.plots.renderers.eof_pattern import EOFPatternPlotter  # noqa: E402


def _eof_ds(nlev=0) -> xr.Dataset:
    lat = np.linspace(10, 40, 4); lon = np.linspace(-120, -90, 5)
    if nlev:
        mode = np.zeros((2, nlev, 4, 5))
        mode[:, -1] = 1.0  # surface level distinct (=1), others 0
        da = xr.DataArray(mode, dims=("mode", "lev", "lat", "lon"),
                          coords={"mode": [1, 2], "lev": np.array([100.0, 500.0, 1000.0])[:nlev],
                                  "lat": lat, "lon": lon, "latitude": ("lat", lat), "longitude": ("lon", lon)})
    else:
        da = xr.DataArray(np.random.default_rng(0).normal(size=(2, 4, 5)),
                          dims=("mode", "lat", "lon"),
                          coords={"mode": [1, 2], "lat": lat, "lon": lon,
                                  "latitude": ("lat", lat), "longitude": ("lon", lon)})
    ds = xr.Dataset({"mode": da, "explained_variance": ("mode", np.array([0.7, 0.3]))})
    ds.attrs["eof_quantity"] = "O3"
    return ds


def test_eof_pattern_one_quadmesh_per_mode() -> None:
    figs = EOFPatternPlotter().render(build_series(_eof_ds(), "mode"))
    assert isinstance(figs, list) and len(figs) == 2
    labels = [lbl for lbl, _ in figs]
    assert labels == ["mode1", "mode2"]
    ax = figs[0][1].axes[0]
    assert any(isinstance(c, QuadMesh) for c in ax.collections)
    for _, f in figs:
        plt.close(f)


def test_eof_pattern_3d_defaults_to_surface() -> None:
    figs = EOFPatternPlotter().render(build_series(_eof_ds(nlev=3), "mode"))
    qm = next(c for c in figs[0][1].axes[0].collections if isinstance(c, QuadMesh))
    arr = np.asarray(qm.get_array(), dtype=float)
    # Surface level (index -1) == 1.0 everywhere; TOA == 0.
    assert np.nanmax(arr) == __import__("pytest").approx(1.0)
    assert np.nanmin(arr) == __import__("pytest").approx(1.0)
    for _, f in figs:
        plt.close(f)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/unit/plots/test_eof_pattern.py -v`
Expected: FAIL — module does not exist.

- [ ] **Step 3: Implement the renderer + register**

`davinci_monet/plots/renderers/eof_pattern.py`:

```python
"""EOF spatial-pattern map renderer (one signed map per mode)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from matplotlib.colors import TwoSlopeNorm

from davinci_monet.plots import labeling
from davinci_monet.plots.base import calculate_symmetric_limits, get_variable_units
from davinci_monet.plots.registry import register_plotter
from davinci_monet.plots.renderers.spatial.base import (
    BaseSpatialPlotter,
    draw_spatial_field,
    resolve_spatial_coords,
    surface_level_index,
)
from davinci_monet.plots.style import get_bias_cmap

if TYPE_CHECKING:
    import matplotlib.axes
    import matplotlib.figure

    from davinci_monet.core.base import PlotSeries


@register_plotter("eof_pattern")
class EOFPatternPlotter(BaseSpatialPlotter):
    """Render each EOF mode as a signed spatial map (diverging, centered at 0)."""

    name: str = "eof_pattern"
    default_figsize: tuple[float, float] = (8, 6)

    def render(
        self,
        series: list["PlotSeries"],
        ax: "matplotlib.axes.Axes | None" = None,
        *,
        display_level: int | None = None,
        **kwargs: Any,
    ) -> list[tuple[str, "matplotlib.figure.Figure"]]:
        if len(series) != 1:
            raise NotImplementedError(
                f"EOFPatternPlotter.render requires exactly 1 series; got {len(series)}."
            )
        s = series[0]
        ds = s.dataset
        field = ds[s.var_name]
        if "mode" not in field.dims:
            raise NotImplementedError("eof_pattern requires a 'mode' dimension on the variable.")

        lat_name, lon_name, lats, lons = resolve_spatial_coords(ds)
        horiz = set(ds[lat_name].dims) | set(ds[lon_name].dims)
        units = get_variable_units(ds, s.var_name)
        quantity = str(ds.attrs.get("eof_quantity", ""))

        figures: list[tuple[str, "matplotlib.figure.Figure"]] = []
        for m in [int(v) for v in field["mode"].values]:
            fld = field.sel(mode=m)
            vdims = [d for d in fld.dims if d not in horiz]
            if vdims:
                lev = vdims[0]
                idx = display_level if display_level is not None else surface_level_index(fld, lev)
                fld = fld.isel({lev: idx})

            vmin, vmax = calculate_symmetric_limits(fld.values)
            fig, axx = self.create_map_figure()
            self.add_map_features(axx)
            mappable = draw_spatial_field(
                axx,
                fld.values,
                lats,
                lons,
                plot_type="pcolormesh",
                cmap=get_bias_cmap(),
                vmin=vmin,
                vmax=vmax,
                marker_size=self.config.style.markersize * 2,
                alpha=1.0,
            )
            if vmin < 0 < vmax:
                mappable.set_norm(TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax))
            self.add_colorbar(fig, mappable, axx, label=labeling.format_units(units))

            ev_pct = None
            if "explained_variance" in ds:
                ev_pct = float(ds["explained_variance"].sel(mode=m).item()) * 100.0
            self.set_title(
                axx,
                labeling.title_text(quantity, operation=f"EOF Mode {m}"),
                subtitle=(f"{ev_pct:.1f}% variance" if ev_pct is not None else None),
            )
            figures.append((f"mode{m}", fig))
        return figures
```

In `davinci_monet/plots/contracts.py`: add `"eof_pattern"` to `SINGLE_SOURCE_PLOTS` and to `SPATIAL_PLOTS`.

In `davinci_monet/plots/renderers/__init__.py`: add `from davinci_monet.plots.renderers.eof_pattern import EOFPatternPlotter` and add `"EOFPatternPlotter"` to `__all__`.

- [ ] **Step 4: Run test to verify it passes**

Run: `HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/unit/plots/test_eof_pattern.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add davinci_monet/plots/renderers/eof_pattern.py davinci_monet/plots/contracts.py davinci_monet/plots/renderers/__init__.py davinci_monet/tests/unit/plots/test_eof_pattern.py
git commit -m "feat(eof): eof_pattern map renderer"
```

---

### Task 6: `eof_scree` renderer

**Files:**
- Create: `davinci_monet/plots/renderers/eof_scree.py`
- Modify: `davinci_monet/plots/contracts.py`, `davinci_monet/plots/renderers/__init__.py`
- Test: `davinci_monet/tests/unit/plots/test_eof_scree.py`

- [ ] **Step 1: Write the failing test**

```python
"""eof_scree renders explained-variance bars with North error bars."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import xarray as xr  # noqa: E402
from matplotlib.container import BarContainer  # noqa: E402

from davinci_monet.plots.base import build_series  # noqa: E402
from davinci_monet.plots.renderers.eof_scree import EOFScreePlotter  # noqa: E402


def _ds() -> xr.Dataset:
    ds = xr.Dataset(
        {
            "explained_variance": ("mode", np.array([0.6, 0.25, 0.1])),
            "explained_variance_error": ("mode", np.array([0.05, 0.03, 0.02])),
        },
        coords={"mode": [1, 2, 3]},
    )
    ds.attrs["eof_quantity"] = "O3"
    return ds


def test_eof_scree_bars_and_errorbars() -> None:
    fig = EOFScreePlotter().render(build_series(_ds(), "explained_variance"))
    ax = fig.axes[0]
    assert any(isinstance(c, BarContainer) for c in ax.containers)
    assert ax.get_ylabel().startswith("Explained variance")
    # 3 bars at modes 1,2,3 with heights in percent
    heights = sorted(rect.get_height() for rect in ax.patches)
    assert heights[-1] == __import__("pytest").approx(60.0, abs=0.1)
    plt.close(fig)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/unit/plots/test_eof_scree.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement the renderer + register**

`davinci_monet/plots/renderers/eof_scree.py`:

```python
"""EOF scree plot: explained variance (%) per mode with North error bars."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from davinci_monet.plots import labeling
from davinci_monet.plots.base import BasePlotter
from davinci_monet.plots.registry import register_plotter

if TYPE_CHECKING:
    import matplotlib.axes
    import matplotlib.figure

    from davinci_monet.core.base import PlotSeries


@register_plotter("eof_scree")
class EOFScreePlotter(BasePlotter):
    """Bar chart of explained variance (%) by EOF mode."""

    name: str = "eof_scree"
    default_figsize: tuple[float, float] = (8, 5)

    def render(
        self,
        series: list["PlotSeries"],
        ax: "matplotlib.axes.Axes | None" = None,
        *,
        title: str | None = None,
        **kwargs: Any,
    ) -> "matplotlib.figure.Figure":
        if len(series) != 1:
            raise NotImplementedError(
                f"EOFScreePlotter.render requires exactly 1 series; got {len(series)}."
            )
        s = series[0]
        ds = s.dataset
        ev = ds[s.var_name]
        modes = [int(v) for v in ev["mode"].values]
        heights = np.asarray(ev.values, dtype=float) * 100.0
        err = None
        if "explained_variance_error" in ds:
            err = np.asarray(ds["explained_variance_error"].values, dtype=float) * 100.0

        if ax is None:
            fig, ax = self.create_figure()
        else:
            fig = ax.get_figure()  # type: ignore[assignment]

        ax.bar(modes, heights, color=self.config.style.y_color, yerr=err, capsize=3)
        ax.set_xticks(modes)
        ax.set_xlabel("Mode", fontsize=self.config.text.fontsize)
        ax.set_ylabel("Explained variance (%)", fontsize=self.config.text.fontsize)
        quantity = str(ds.attrs.get("eof_quantity", ""))
        self.set_title(ax, title or labeling.title_text(quantity, operation="EOF Explained Variance"))
        ax.grid(True, alpha=0.3, axis="y")
        return fig
```

In `contracts.py`: add `"eof_scree"` to `SINGLE_SOURCE_PLOTS` and to `STATISTICAL_PLOTS`.
In `renderers/__init__.py`: import `EOFScreePlotter` and add to `__all__`.

- [ ] **Step 4: Run test to verify it passes**

Run: `HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/unit/plots/test_eof_scree.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add davinci_monet/plots/renderers/eof_scree.py davinci_monet/plots/contracts.py davinci_monet/plots/renderers/__init__.py davinci_monet/tests/unit/plots/test_eof_scree.py
git commit -m "feat(eof): eof_scree renderer"
```

---

### Task 7: PC time-series wiring — plot-time `mode:` selector

**Files:**
- Modify: `davinci_monet/config/schema.py` (`PlotGroupConfig`: add `mode`, `display_level`)
- Modify: `davinci_monet/pipeline/stages/plot.py` (select `mode` before `build_series`)
- Modify: `davinci_monet/pipeline/stages/plot_options.py` (forward `display_level`)
- Modify: `davinci_monet/tests/unit/test_analyses_validation.py` (remove the Plan A skip)
- Test: `davinci_monet/tests/unit/plots/test_pc_mode_selection.py`

- [ ] **Step 1: Write the failing test**

```python
"""A plot-time mode: selector picks one PC for the timeseries renderer."""

from __future__ import annotations

import numpy as np
import xarray as xr

from davinci_monet.config.schema import PlotGroupConfig
from davinci_monet.pipeline.stages.plot_options import single_source_plot_kwargs


def test_plotgroup_accepts_mode_and_display_level() -> None:
    cfg = PlotGroupConfig(type="timeseries", source="cam_O3_eof", variable="pc", mode=1)
    assert cfg.mode == 1
    cfg2 = PlotGroupConfig(type="eof_pattern", source="cam_O3_eof", variable="mode", display_level=-1)
    assert cfg2.display_level == -1


def test_display_level_forwarded_to_render_kwargs() -> None:
    spec = {"type": "eof_pattern", "source": "s", "variable": "mode", "display_level": -1}
    kwargs = single_source_plot_kwargs(spec, analysis_config=None)
    assert kwargs.get("display_level") == -1


def test_mode_selection_picks_single_pc() -> None:
    # Direct check of the selection idiom the plot stage applies.
    pc = xr.Dataset(
        {"pc": (("time", "mode"), np.array([[1.0, 9.0], [2.0, 9.0], [3.0, 9.0]]))},
        coords={"time": np.arange(3), "mode": [1, 2]},
    )
    selected = pc.sel(mode=1)["pc"].values
    assert list(selected) == [1.0, 2.0, 3.0]  # PC1, not the cross-mode mean (5.0,...)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/unit/plots/test_pc_mode_selection.py -v`
Expected: FAIL — `PlotGroupConfig` rejects `mode`/`display_level` (well, FlexibleSchema accepts them silently, so the attribute access `cfg.mode` fails) and `single_source_plot_kwargs` does not forward `display_level`.

- [ ] **Step 3: Implement**

In `davinci_monet/config/schema.py`, add to `class PlotGroupConfig(FlexibleSchema)` (alongside `source`/`variable`):

```python
    mode: int | None = None
    display_level: int | None = None
```

In `davinci_monet/pipeline/stages/plot.py`, inside `_render_single_source_plot`'s per-flight loop, immediately **before** `tag_source_label(subset, source_label=source_label)`, insert:

```python
            sel_mode = plot_spec.get("mode")
            if sel_mode is not None and "mode" in subset[variable].dims:
                subset = subset.sel(mode=sel_mode)
```

In `davinci_monet/pipeline/stages/plot_options.py`, open `single_source_plot_kwargs` and ensure `display_level` is copied from `plot_spec` into the returned render-kwargs dict when present (follow the existing pattern that forwards keys like `title`). For example, if the function builds a `kwargs` dict, add:

```python
    if plot_spec.get("display_level") is not None:
        kwargs["display_level"] = plot_spec["display_level"]
```

In `davinci_monet/tests/unit/test_analyses_validation.py`, remove the `@pytest.mark.skip(...)` decorator on `test_plot_may_reference_derived_source` (now that `eof_pattern` is registered).

- [ ] **Step 4: Run tests to verify they pass**

Run: `HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/unit/plots/test_pc_mode_selection.py davinci_monet/tests/unit/test_analyses_validation.py -v`
Expected: PASS (including the previously-skipped test).

- [ ] **Step 5: Commit**

```bash
git add davinci_monet/config/schema.py davinci_monet/pipeline/stages/plot.py davinci_monet/pipeline/stages/plot_options.py davinci_monet/tests/unit/plots/test_pc_mode_selection.py davinci_monet/tests/unit/test_analyses_validation.py
git commit -m "feat(eof): plot-time mode selector + display_level passthrough"
```

---

### Task 8: End-to-end integration — EOF analysis + all three plots

**Files:**
- Test: `davinci_monet/tests/integration/test_eof_pipeline.py`

- [ ] **Step 1: Write the failing test**

```python
"""Integration: an eof analysis runs through the pipeline and produces its plots."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from davinci_monet.pipeline.runner import PipelineRunner


def _grid_nc(path: Path) -> None:
    times = pd.date_range("2024-01-01", periods=120, freq="D")
    lat = np.linspace(-5, 5, 6); lon = np.linspace(0, 30, 8)
    x = np.linspace(0, np.pi, len(lon))
    rng = np.random.default_rng(0)
    p1 = np.cos(x)[None, :] * np.ones((len(lat), 1))
    pc1 = rng.normal(size=len(times))
    field = 3.0 * pc1[:, None, None] * p1[None] + 0.1 * rng.normal(size=(len(times), len(lat), len(lon)))
    xr.Dataset(
        {"O3": (("time", "lat", "lon"), field, {"units": "ppb"})},
        coords={"time": times, "lat": ("lat", lat), "lon": ("lon", lon),
                "latitude": ("lat", lat), "longitude": ("lon", lon)},
    ).to_netcdf(path)


@pytest.mark.integration
def test_eof_plots_through_pipeline(tmp_path: Path) -> None:
    src = tmp_path / "grid.nc"
    _grid_nc(src)
    config = {
        "analysis": {"output_dir": str(tmp_path / "out")},
        "sources": {"cam": {"type": "generic", "files": str(src), "variables": {"O3": {"units": "ppb"}}}},
        "analyses": {"cam_O3_eof": {"type": "eof", "source": "cam", "variable": "O3", "n_modes": 3}},
        "plots": {
            "eof_maps": {"type": "eof_pattern", "source": "cam_O3_eof", "variable": "mode"},
            "eof_var": {"type": "eof_scree", "source": "cam_O3_eof", "variable": "explained_variance"},
            "pc1": {"type": "timeseries", "source": "cam_O3_eof", "variable": "pc", "mode": 1},
        },
    }

    result = PipelineRunner(show_progress=False).run_from_config(config)

    assert result.success, getattr(result, "error", None)
    plots = result.context.results["plotting"].data["plots_generated"]
    pngs = [p for p in plots if p.endswith(".png")]
    # 3 mode maps + 1 scree + 1 pc timeseries = 5 PNGs
    assert sum("eof_maps" in p for p in pngs) == 3
    assert any("eof_var" in p for p in pngs)
    assert any("pc1" in p for p in pngs)
```

- [ ] **Step 2: Run test**

Run: `HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/integration/test_eof_pipeline.py -v`
Expected: PASS. Debug the real path if not (no shortcuts).

- [ ] **Step 3: Run the EOF gate**

```bash
HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/unit/analysis davinci_monet/tests/unit/plots/test_eof_pattern.py davinci_monet/tests/unit/plots/test_eof_scree.py davinci_monet/tests/unit/plots/test_pc_mode_selection.py davinci_monet/tests/integration/test_eof_pipeline.py -v
mypy davinci_monet
black davinci_monet && isort davinci_monet
```
Expected: all PASS, mypy clean, formatting clean.

- [ ] **Step 4: Commit**

```bash
git add davinci_monet/tests/integration/test_eof_pipeline.py
git commit -m "test(eof): end-to-end EOF analysis + plots through pipeline"
```

---

### Task 9: Docs, example config, gallery

**Files:**
- Modify: `CLAUDE.md` (document the `analyses:` block + EOF)
- Create: `analyses/_gallery/` entry or example config (follow existing gallery/example conventions)

- [ ] **Step 1: Document `analyses:` in CLAUDE.md**

Add a short subsection near the YAML config docs describing the `analyses:` block, the `eof` type and its fields (`n_modes`, `standardize`, `remove_seasonal_cycle`, `rotation`, `level`), that outputs become pseudo-sources (`mode`, `pc`, `explained_variance`), and the plot types `eof_pattern` / `eof_scree` / `timeseries` (with `mode:`). Keep it terse and consistent with the existing config docs.

- [ ] **Step 2: Add an example config**

Create `analyses/_gallery/` or an `*.example.yaml` mirroring §3.5 of the spec. Confirm the existing gallery harness (`analyses/_gallery/make_gallery.py`, per CLAUDE.md) renders the new plot types through the pipeline; add the three EOF plot types to it if that's how the gallery is structured.

- [ ] **Step 3: Run the gallery (if applicable) and verify a figure per new type**

Run the gallery generator and confirm `eof_pattern`/`eof_scree` figures are produced. Copy PDFs to the iCloud Claude folder per CLAUDE.md.

- [ ] **Step 4: Commit**

```bash
git add CLAUDE.md analyses/_gallery/
git commit -m "docs(eof): document analyses block + EOF; gallery + example config"
```

---

## Self-Review (completed by plan author)

- **Spec coverage:** §4.1 preprocessing order (T3); §4.2 weighting incl. Δp + fallback + standardize-exclusion via logging (T2,T3,T4); §4.3 xeofs (T1,T3); §4.4 scaling split via regression onto unit-variance PCs (T3); §4.5 mode-intrinsic sign (T2,T3); §4.6 North error + effective-N, unrotated only (T3); §4.7 outputs (T3); §6.1 eof_pattern + surface slice + display_level (T5,T7); §6.2 eof_scree (T6); §6.3 PC timeseries reuse + mode selector (T7).
- **Deferred/assumption:** xeofs API names are pinned in T1's probe; T3/T4 transpose-fix is conditional on the probe. `single_source_plot_kwargs` internals are unknown — T7 drives the `display_level` passthrough via a direct test of that function.
- **Type consistency:** EOF output vars `mode`/`pc`/`explained_variance`/`explained_variance_error`, attrs `kind`, `eof_quantity` — consistent across T3 (producer) and T5/T6 (consumers). `mode` is 1-indexed everywhere.
- **Placeholders:** none.
