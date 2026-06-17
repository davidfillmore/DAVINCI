# Wavelet Analysis (Plan C) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the `wavelet` derived analysis (Torrence & Compo continuous wavelet transform of a 1-D series — a station series, an area-mean of a gridded field, or an EOF PC) and the `wavelet_scalogram` renderer (time×period power with cone of influence, significance, and a global-spectrum side panel).

**Architecture:** `WaveletAnalysis` (in the `davinci_monet/analysis/` package) reduces its input to a 1-D series via shared helpers in `reductions.py`, then uses `pycwt` for the Morlet CWT, AR(1) red-noise significance (local + global), and cone of influence. The output registers as a `SPECTRUM`-geometry pseudo-source (Plan A), so the scalogram uses the normal single-source plot path. Running a wavelet on an EOF PC is just `source: <eof>, variable: pc, mode: N`.

**Tech Stack:** pycwt (new dep), xarray, numpy, matplotlib, the DAVINCI plotting framework.

**Prerequisites:** Plan A (foundation) complete. Task 5 (integration) also depends on Plan B (EOF) for the wavelet-of-PC test.

**Spec:** `docs/superpowers/specs/2026-06-17-eof-and-wavelet-analysis-design.md` (§5, §6.4, §11 Plan C).

**Conventions:** Same as Plans A/B. Advisory messages use the module **logger**, never `warnings.warn` (suite has `filterwarnings = ["error::UserWarning"]`). Always `plt.close(fig)` in tests.

---

## File Structure

- Create `davinci_monet/analysis/reductions.py` — series selection/reduction + time-axis regularize/detrend/ar1/normalize helpers.
- Create `davinci_monet/analysis/wavelet.py` — `WaveletAnalysis`.
- Modify `davinci_monet/analysis/__init__.py` — import `wavelet` so it registers.
- Create `davinci_monet/plots/renderers/wavelet_scalogram.py` — `WaveletScalogramPlotter`.
- Modify `davinci_monet/plots/contracts.py`, `davinci_monet/plots/renderers/__init__.py`.
- Modify `pyproject.toml`, `environment.yml` — add pinned `pycwt`.
- Create tests under `davinci_monet/tests/unit/{analysis,plots}/` and `davinci_monet/tests/integration/`.

---

### Task 1: Add `pycwt` dependency + API probe

**Files:**
- Modify: `pyproject.toml`, `environment.yml`
- Test: `davinci_monet/tests/unit/analysis/test_pycwt_api.py`

- [ ] **Step 1: Add the dependency and install**

In `pyproject.toml` add `"pycwt>=0.4"` to runtime deps; in `environment.yml` add `- pycwt>=0.4` under pip. Then:

```bash
conda activate davinci
pip install "pycwt>=0.4"
```

- [ ] **Step 2: Write the API-probe test**

```python
"""Pin the pycwt API surface WaveletAnalysis relies on."""

from __future__ import annotations

import numpy as np


def test_pycwt_minimal_api() -> None:
    import pycwt

    nt, dt = 256, 1.0
    t = np.arange(nt)
    sig = np.sin(2 * np.pi * t / 16.0)
    sig = (sig - sig.mean()) / sig.std()

    alpha = float(pycwt.ar1(sig)[0])
    mother = pycwt.Morlet(6)
    wave, scales, freqs, coi, _, _ = pycwt.cwt(sig, dt, 0.25, 2 * dt, -1, mother)
    power = np.abs(wave) ** 2
    assert power.shape == (scales.size, nt)
    assert coi.shape == (nt,)

    # Local significance: variance=1, sigma_test=0.
    signif, _ = pycwt.significance(1.0, dt, scales, 0, alpha, significance_level=0.95, wavelet=mother)
    assert signif.shape == (scales.size,)

    # Global significance: sigma_test=1 needs a per-scale dof vector.
    dof = nt - scales
    glbl, _ = pycwt.significance(1.0, dt, scales, 1, alpha, significance_level=0.95, dof=dof, wavelet=mother)
    assert glbl.shape == (scales.size,)
```

- [ ] **Step 3: Run the probe**

Run: `HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/unit/analysis/test_pycwt_api.py -v`
Expected: PASS. **If signatures differ** (e.g. `pycwt.ar1` arity, `significance` return tuple), record the real shapes and adjust this test AND `WaveletAnalysis` (Task 3) before proceeding.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml environment.yml davinci_monet/tests/unit/analysis/test_pycwt_api.py
git commit -m "build(wavelet): add pycwt dependency + API probe"
```

---

### Task 2: `reductions.py` — series extraction & preprocessing helpers

**Files:**
- Create: `davinci_monet/analysis/reductions.py`
- Test: `davinci_monet/tests/unit/analysis/test_reductions.py`

- [ ] **Step 1: Write the failing test**

```python
"""Series selection/reduction + preprocessing helpers for wavelet input."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from davinci_monet.analysis.reductions import (
    ar1_alpha,
    detrend_series,
    normalize_series,
    regularize,
    select_series,
)
from davinci_monet.config.schema import PointReduce, WaveletSpec


def _grid(nt=20, nlat=3, nlon=4) -> xr.Dataset:
    lat = np.linspace(-5, 5, nlat); lon = np.linspace(0, 9, nlon)
    times = pd.date_range("2024-01-01", periods=nt, freq="D")
    data = np.random.default_rng(0).normal(size=(nt, nlat, nlon))
    return xr.Dataset(
        {"O3": (("time", "lat", "lon"), data, {"units": "ppb"})},
        coords={"time": times, "lat": lat, "lon": lon,
                "latitude": ("lat", lat), "longitude": ("lon", lon)},
    )


def _pc(nt=20) -> xr.Dataset:
    times = pd.date_range("2024-01-01", periods=nt, freq="D")
    pc = np.stack([np.arange(nt, dtype=float), np.full(nt, 9.0)], axis=1)
    return xr.Dataset({"pc": (("time", "mode"), pc)}, coords={"time": times, "mode": [1, 2]})


def test_select_area_mean_reduces_to_1d() -> None:
    spec = WaveletSpec(type="wavelet", source="cam", variable="O3")  # default reduce=area_mean
    s = select_series(_grid(), spec)
    assert s.dims == ("time",)


def test_select_point() -> None:
    spec = WaveletSpec(type="wavelet", source="cam", variable="O3", reduce=PointReduce(point=(0.0, 3.0)))
    s = select_series(_grid(), spec)
    assert s.dims == ("time",)


def test_select_pc_mode_is_already_1d() -> None:
    spec = WaveletSpec(type="wavelet", source="eof", variable="pc", mode=1)
    s = select_series(_pc(), spec)
    assert s.dims == ("time",)
    assert list(s.values[:3]) == [0.0, 1.0, 2.0]  # PC1, not the cross-mode mean


def test_pc_without_mode_errors() -> None:
    spec = WaveletSpec(type="wavelet", source="eof", variable="pc")
    with pytest.raises(ValueError, match="requires mode"):
        select_series(_pc(), spec)


def test_point_reduce_on_1d_series_errors() -> None:
    spec = WaveletSpec(type="wavelet", source="eof", variable="pc", mode=1,
                       reduce=PointReduce(point=(0.0, 0.0)))
    with pytest.raises(ValueError, match="point.*1-D"):
        select_series(_pc(), spec)


def test_regularize_regular_series() -> None:
    s = select_series(_grid(), WaveletSpec(type="wavelet", source="c", variable="O3"))
    reg, dt, unit, frac = regularize(s)
    assert dt == pytest.approx(1.0)
    assert unit == "days"
    assert frac == 0.0


def test_detrend_and_normalize() -> None:
    y = np.arange(50, dtype=float) + 5.0
    d = detrend_series(y)
    assert abs(float(np.mean(d))) < 1e-9
    n, std, mean = normalize_series(d)
    assert float(np.std(n)) == pytest.approx(1.0, abs=1e-6)


def test_ar1_alpha_on_white_noise_is_small() -> None:
    y = np.random.default_rng(1).normal(size=500)
    assert abs(ar1_alpha(y)) < 0.2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/unit/analysis/test_reductions.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement `reductions.py`**

```python
"""Reduce a source variable to a 1-D time series and prepare it for the CWT."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import xarray as xr

if TYPE_CHECKING:
    from davinci_monet.config.schema import WaveletSpec

_LAT_NAMES = ("latitude", "lat", "LAT", "Latitude")
_LON_NAMES = ("longitude", "lon", "LON", "Longitude")


def _coord(da: xr.DataArray, names: tuple[str, ...], kind: str) -> xr.DataArray:
    for name in names:
        if name in da.coords:
            return da.coords[name]
    raise ValueError(f"wavelet reduction requires a {kind} coordinate (one of {names})")


def select_series(data: xr.Dataset, spec: "WaveletSpec") -> xr.DataArray:
    """Resolve spec.variable (+ mode + reduce) to a 1-D (time,) series."""
    from davinci_monet.config.schema import PointReduce

    da = data[spec.variable]
    if "mode" in da.dims:
        if spec.mode is None:
            raise ValueError(f"wavelet on '{spec.variable}' with a 'mode' dim requires mode: N")
        da = da.sel(mode=spec.mode)

    spatial = [d for d in da.dims if d != "time"]
    if not spatial:
        if isinstance(spec.reduce, PointReduce):
            raise ValueError("reduce: point is invalid for an already-1-D series")
        return da  # already 1-D: reduce is a no-op

    reduce = spec.reduce
    if reduce is None or reduce == "area_mean":
        lat = _coord(da, _LAT_NAMES, "latitude")
        w = np.cos(np.deg2rad(lat)).clip(min=0.0)
        return da.weighted(w).mean(dim=spatial)
    if isinstance(reduce, PointReduce):
        lat = _coord(da, _LAT_NAMES, "latitude")
        lon = _coord(da, _LON_NAMES, "longitude")
        i = int(np.abs(np.asarray(lat.values) - reduce.point[0]).argmin())
        j = int(np.abs(np.asarray(lon.values) - reduce.point[1]).argmin())
        da = da.isel({lat.dims[0]: i, lon.dims[0]: j})
        rem = [d for d in da.dims if d != "time"]
        return da.mean(rem) if rem else da
    raise ValueError(f"unknown reduce: {reduce!r}")


def _step_and_unit(time_values: np.ndarray) -> tuple[float, str, np.ndarray]:
    arr = np.asarray(time_values)
    if np.issubdtype(arr.dtype, np.datetime64):
        deltas = np.diff(arr).astype("timedelta64[s]").astype(float)
        med = float(np.median(deltas)) if deltas.size else 86400.0
        if med >= 86400.0:
            return med / 86400.0, "days", deltas
        return med / 3600.0, "hours", deltas
    deltas = np.diff(arr.astype(float))
    return (float(np.median(deltas)) if deltas.size else 1.0), "steps", deltas


def regularize(series: xr.DataArray) -> tuple[xr.DataArray, float, str, float]:
    """Return (regular series, dt, period-unit, fraction of synthesized samples)."""
    dt, unit, deltas = _step_and_unit(series["time"].values)
    if deltas.size == 0:
        return series, dt, unit, 0.0
    med = float(np.median(deltas))
    irregular = bool(np.any(np.abs(deltas - med) > 0.05 * med))
    if not irregular or unit == "steps":
        return series, dt, unit, 0.0
    n_before = int(series.sizes["time"])
    freq = pd.Timedelta(seconds=med)
    regular = series.resample(time=freq).mean()
    n_after = int(regular.sizes["time"])
    frac = max(0.0, (n_after - n_before) / max(n_after, 1))
    return regular, dt, unit, frac


def detrend_series(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    x = np.arange(y.size)
    coef = np.polyfit(x, y, 1)
    return y - np.polyval(coef, x)


def ar1_alpha(y: np.ndarray) -> float:
    """Lag-1 autocorrelation (red-noise parameter) of the (detrended) series."""
    try:
        import pycwt

        return float(pycwt.ar1(np.asarray(y, dtype=float))[0])
    except Exception:  # noqa: BLE001 - robust fallback
        y = np.asarray(y, dtype=float)
        if y.size < 3:
            return 0.0
        return float(np.clip(np.corrcoef(y[:-1], y[1:])[0, 1], -0.99, 0.99))


def normalize_series(y: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Return (unit-variance series, std, mean)."""
    y = np.asarray(y, dtype=float)
    mean = float(np.mean(y))
    std = float(np.std(y))
    std = std if std > 0 else 1.0
    return (y - mean) / std, std, mean
```

- [ ] **Step 4: Run test to verify it passes**

Run: `HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/unit/analysis/test_reductions.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add davinci_monet/analysis/reductions.py davinci_monet/tests/unit/analysis/test_reductions.py
git commit -m "feat(wavelet): series reduction + preprocessing helpers"
```

---

### Task 3: `WaveletAnalysis` core — register + injected-period recovery

**Files:**
- Create: `davinci_monet/analysis/wavelet.py`
- Modify: `davinci_monet/analysis/__init__.py`
- Test: `davinci_monet/tests/unit/analysis/test_wavelet_analysis.py`

- [ ] **Step 1: Write the failing test**

```python
"""WaveletAnalysis recovers an injected period and flags it significant."""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from davinci_monet.analysis.wavelet import WaveletAnalysis
from davinci_monet.config.schema import WaveletSpec


def _injected(nt=256, period=16.0, seed=0) -> xr.Dataset:
    rng = np.random.default_rng(seed)
    t = np.arange(nt)
    y = np.sin(2 * np.pi * t / period) + 0.3 * rng.normal(size=nt)
    times = pd.date_range("2020-01-01", periods=nt, freq="D")
    field = np.broadcast_to(y[:, None, None], (nt, 2, 2)).copy()
    lat = np.array([-1.0, 1.0]); lon = np.array([0.0, 1.0])
    return xr.Dataset(
        {"O3": (("time", "lat", "lon"), field, {"units": "ppb"})},
        coords={"time": times, "lat": lat, "lon": lon,
                "latitude": ("lat", lat), "longitude": ("lon", lon)},
    )


def test_wavelet_recovers_injected_period() -> None:
    spec = WaveletSpec(type="wavelet", source="cam", variable="O3")  # area_mean
    out = WaveletAnalysis().analyze(_injected(period=16.0), spec)

    assert out["power"].dims == ("time", "period")
    assert set(out.data_vars) >= {
        "power", "power_significance", "coi", "global_power", "global_significance",
    }
    assert out["period"].attrs.get("units") == "days"

    period = out["period"].values
    gp = out["global_power"].values
    peak = period[int(np.argmax(gp))]
    assert 12.0 < peak < 22.0  # near 16-day injected period (log-scale tolerance)
    # The peak exceeds the red-noise global significance there.
    gsig = out["global_significance"].values
    i = int(np.argmax(gp))
    assert gp[i] > gsig[i]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/unit/analysis/test_wavelet_analysis.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement `WaveletAnalysis`**

```python
"""Continuous wavelet transform (Torrence & Compo) of a 1-D series."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

from davinci_monet.analysis.base import DerivedAnalysis
from davinci_monet.analysis.reductions import (
    ar1_alpha,
    detrend_series,
    normalize_series,
    regularize,
    select_series,
)
from davinci_monet.core.protocols import DataGeometry
from davinci_monet.core.registry import analysis_registry

if TYPE_CHECKING:
    from davinci_monet.config.schema import WaveletSpec

logger = logging.getLogger(__name__)


@analysis_registry.register("wavelet")
class WaveletAnalysis(DerivedAnalysis):
    """Morlet CWT with AR(1) red-noise significance and cone of influence."""

    name = "wavelet"
    long_name = "Continuous Wavelet Transform"
    output_geometry = DataGeometry.SPECTRUM

    def analyze(self, data: xr.Dataset, spec: "WaveletSpec") -> xr.Dataset:
        import pycwt

        series = select_series(data, spec)
        regular, dt, unit, frac = regularize(series)
        if frac > 0.5:
            logger.warning(
                "wavelet input for '%s' was %.0f%% synthesized by time-axis "
                "regularization; AR(1) and power may be unreliable",
                spec.variable,
                100.0 * frac,
            )

        y = np.asarray(regular.values, dtype=float)
        y = detrend_series(y)
        alpha = ar1_alpha(y)            # estimate red noise BEFORE normalization
        y_norm, _std, _mean = normalize_series(y)

        mother = pycwt.Morlet(spec.omega0)
        s0 = spec.s0 if spec.s0 is not None else 2.0 * dt
        big_j = spec.j if spec.j is not None else -1
        wave, scales, freqs, coi, _, _ = pycwt.cwt(y_norm, dt, spec.dj, s0, big_j, mother)

        power = np.abs(wave) ** 2          # (scale, time)
        period = 1.0 / freqs               # (scale,)
        n = y_norm.size

        local_signif, _ = pycwt.significance(
            1.0, dt, scales, 0, alpha,
            significance_level=spec.significance_level, wavelet=mother,
        )
        power_sig = power / local_signif[:, None]

        global_power = power.mean(axis=1)  # (scale,)
        dof = n - scales
        global_signif, _ = pycwt.significance(
            1.0, dt, scales, 1, alpha,
            significance_level=spec.significance_level, dof=dof, wavelet=mother,
        )

        ds = xr.Dataset(
            {
                "power": (("time", "period"), power.T, {"kind": "power", "long_name": "Wavelet power"}),
                "power_significance": (("time", "period"), power_sig.T, {"kind": "power"}),
                "coi": (("time",), np.asarray(coi, dtype=float), {"kind": "coi", "units": unit}),
                "global_power": (("period",), global_power, {"kind": "global"}),
                "global_significance": (("period",), np.asarray(global_signif, dtype=float), {"kind": "global"}),
            },
            coords={
                "time": regular["time"].values,
                "period": ("period", period, {"units": unit, "long_name": "Period"}),
            },
        )
        ds.attrs["wavelet_quantity"] = spec.variable
        ds.attrs["dt"] = float(dt)
        ds.attrs["dt_units"] = unit
        return ds
```

In `davinci_monet/analysis/__init__.py`, add:

```python
from davinci_monet.analysis import wavelet as _wavelet  # noqa: F401  (registers "wavelet")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/unit/analysis/test_wavelet_analysis.py -v`
Expected: PASS. If significance shapes mismatch, recheck the Task 1 probe (especially the global `dof` vector).

- [ ] **Step 5: Commit**

```bash
git add davinci_monet/analysis/wavelet.py davinci_monet/analysis/__init__.py davinci_monet/tests/unit/analysis/test_wavelet_analysis.py
git commit -m "feat(wavelet): WaveletAnalysis core (Morlet CWT + significance + COI)"
```

---

### Task 4: `wavelet_scalogram` renderer

**Files:**
- Create: `davinci_monet/plots/renderers/wavelet_scalogram.py`
- Modify: `davinci_monet/plots/contracts.py`, `davinci_monet/plots/renderers/__init__.py`
- Test: `davinci_monet/tests/unit/plots/test_wavelet_scalogram.py`

- [ ] **Step 1: Write the failing test (fabricated wavelet dataset — no pycwt needed)**

```python
"""wavelet_scalogram draws a QuadMesh + a global-spectrum side panel."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xarray as xr  # noqa: E402
from matplotlib.collections import QuadMesh  # noqa: E402

from davinci_monet.plots.base import build_series  # noqa: E402
from davinci_monet.plots.renderers.wavelet_scalogram import WaveletScalogramPlotter  # noqa: E402


def _spectrum() -> xr.Dataset:
    nt, npd = 60, 6
    time = pd.date_range("2024-01-01", periods=nt, freq="D")
    period = np.array([2.0, 4.0, 8.0, 16.0, 32.0, 64.0])
    rng = np.random.default_rng(0)
    power = rng.random((nt, npd))
    ds = xr.Dataset(
        {
            "power": (("time", "period"), power, {"kind": "power"}),
            "power_significance": (("time", "period"), power / power.mean(0), {"kind": "power"}),
            "coi": (("time",), np.full(nt, 16.0), {"kind": "coi", "units": "days"}),
            "global_power": (("period",), power.mean(0), {"kind": "global"}),
            "global_significance": (("period",), np.ones(npd), {"kind": "global"}),
        },
        coords={"time": time, "period": ("period", period, {"units": "days"})},
    )
    ds.attrs["wavelet_quantity"] = "O3"
    return ds


def test_scalogram_quadmesh_and_global_panel() -> None:
    fig = WaveletScalogramPlotter().render(build_series(_spectrum(), "power"))
    # main pcolormesh axes carries a QuadMesh
    assert any(isinstance(c, QuadMesh) for ax in fig.axes for c in ax.collections)
    # at least a main axes + a global-spectrum axes (colorbar may add a third)
    assert len(fig.axes) >= 2
    plt.close(fig)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/unit/plots/test_wavelet_scalogram.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement the renderer + register**

```python
"""Wavelet scalogram: time x period power with COI, significance, global panel."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt

from davinci_monet.plots import labeling
from davinci_monet.plots.base import BasePlotter
from davinci_monet.plots.registry import register_plotter
from davinci_monet.plots.style import get_sequential_cmap

if TYPE_CHECKING:
    import matplotlib.axes
    import matplotlib.figure

    from davinci_monet.core.base import PlotSeries


@register_plotter("wavelet_scalogram")
class WaveletScalogramPlotter(BasePlotter):
    """Torrence & Compo style scalogram with a global-spectrum side panel."""

    name: str = "wavelet_scalogram"
    default_figsize: tuple[float, float] = (10, 6)

    def render(
        self,
        series: list["PlotSeries"],
        ax: "matplotlib.axes.Axes | None" = None,
        **kwargs: Any,
    ) -> "matplotlib.figure.Figure":
        if len(series) != 1:
            raise NotImplementedError(
                f"WaveletScalogramPlotter.render requires exactly 1 series; got {len(series)}."
            )
        s = series[0]
        ds = s.dataset
        power = ds[s.var_name]
        time = power["time"].values
        period = power["period"].values

        fig = plt.figure(
            figsize=self.config.figure.figsize,
            dpi=self.config.figure.dpi,
            facecolor=self.config.figure.facecolor,
        )
        gs = fig.add_gridspec(1, 2, width_ratios=[3, 1], wspace=0.06)
        ax_main = fig.add_subplot(gs[0, 0])
        ax_glob = fig.add_subplot(gs[0, 1], sharey=ax_main)

        mesh = ax_main.pcolormesh(
            time, period, power.transpose("period", "time").values,
            cmap=get_sequential_cmap(), shading="auto",
        )
        ax_main.set_yscale("log")
        ax_main.set_ylim(float(period.max()), float(period.min()))  # short periods on top

        if "power_significance" in ds:
            sig = ds["power_significance"].transpose("period", "time").values
            ax_main.contour(time, period, sig, levels=[1.0], colors="black", linewidths=1.0)

        if "coi" in ds:
            coi = ds["coi"].values
            ax_main.plot(time, coi, color="white", linestyle="--", linewidth=1.2)
            # Periods ABOVE the COI are edge-contaminated -> hatch them.
            ax_main.fill_between(time, coi, float(period.max()), color="white", alpha=0.3, hatch="xx")

        unit = str(ds["period"].attrs.get("units", ""))
        unit_label = labeling.format_units(unit)
        ax_main.set_ylabel(f"Period ({unit_label})" if unit_label else "Period",
                           fontsize=self.config.text.fontsize)
        ax_main.set_xlabel("Time", fontsize=self.config.text.fontsize)
        self.set_title(
            ax_main,
            labeling.title_text(str(ds.attrs.get("wavelet_quantity", "")), operation="Wavelet Power"),
        )
        fig.colorbar(mesh, ax=ax_main, label="Power", shrink=0.85, pad=0.02)

        if "global_power" in ds:
            ax_glob.plot(ds["global_power"].values, period, color=self.config.style.y_color)
            if "global_significance" in ds:
                ax_glob.plot(ds["global_significance"].values, period,
                             color="black", linestyle="--", linewidth=1.0)
        ax_glob.set_xlabel("Power", fontsize=self.config.text.fontsize)
        plt.setp(ax_glob.get_yticklabels(), visible=False)
        return fig
```

In `contracts.py`: add `"wavelet_scalogram"` to `SINGLE_SOURCE_PLOTS` and to `SPECIALIZED_PLOTS`.
In `renderers/__init__.py`: import `WaveletScalogramPlotter` and add to `__all__`.

- [ ] **Step 4: Run test to verify it passes**

Run: `HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/unit/plots/test_wavelet_scalogram.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add davinci_monet/plots/renderers/wavelet_scalogram.py davinci_monet/plots/contracts.py davinci_monet/plots/renderers/__init__.py davinci_monet/tests/unit/plots/test_wavelet_scalogram.py
git commit -m "feat(wavelet): wavelet_scalogram renderer"
```

---

### Task 5: End-to-end integration — wavelet of an EOF PC + area-mean wavelet

**Files:**
- Test: `davinci_monet/tests/integration/test_wavelet_pipeline.py`

> Depends on Plan B (EOF) for the wavelet-of-PC chain. If Plan B is not yet merged, mark `test_wavelet_of_eof_pc` with `@pytest.mark.skip(reason="needs Plan B EOF")` and keep `test_areamean_wavelet`.

- [ ] **Step 1: Write the failing test**

```python
"""Integration: wavelet runs through the pipeline, including on an EOF PC."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from davinci_monet.pipeline.runner import PipelineRunner


def _grid_nc(path: Path) -> None:
    times = pd.date_range("2020-01-01", periods=256, freq="D")
    lat = np.linspace(-5, 5, 6); lon = np.linspace(0, 30, 8)
    t = np.arange(len(times))
    x = np.linspace(0, np.pi, len(lon))
    rng = np.random.default_rng(0)
    pc1 = np.sin(2 * np.pi * t / 16.0) + 0.3 * rng.normal(size=len(times))
    p1 = np.cos(x)[None, :] * np.ones((len(lat), 1))
    field = 3.0 * pc1[:, None, None] * p1[None] + 0.1 * rng.normal(size=(len(times), len(lat), len(lon)))
    xr.Dataset(
        {"O3": (("time", "lat", "lon"), field, {"units": "ppb"})},
        coords={"time": times, "lat": ("lat", lat), "lon": ("lon", lon),
                "latitude": ("lat", lat), "longitude": ("lon", lon)},
    ).to_netcdf(path)


@pytest.mark.integration
def test_areamean_wavelet(tmp_path: Path) -> None:
    src = tmp_path / "grid.nc"
    _grid_nc(src)
    config = {
        "analysis": {"output_dir": str(tmp_path / "out")},
        "sources": {"cam": {"type": "generic", "files": str(src), "variables": {"O3": {"units": "ppb"}}}},
        "analyses": {"cam_wav": {"type": "wavelet", "source": "cam", "variable": "O3", "reduce": "area_mean"}},
        "plots": {"scal": {"type": "wavelet_scalogram", "source": "cam_wav", "variable": "power"}},
    }
    result = PipelineRunner(show_progress=False).run_from_config(config)
    assert result.success, getattr(result, "error", None)
    assert "cam_wav" in result.context.sources
    pngs = [p for p in result.context.results["plotting"].data["plots_generated"] if p.endswith(".png")]
    assert any("scal" in p for p in pngs)


@pytest.mark.integration
def test_wavelet_of_eof_pc(tmp_path: Path) -> None:
    src = tmp_path / "grid.nc"
    _grid_nc(src)
    config = {
        "analysis": {"output_dir": str(tmp_path / "out")},
        "sources": {"cam": {"type": "generic", "files": str(src), "variables": {"O3": {"units": "ppb"}}}},
        "analyses": {
            "cam_O3_eof": {"type": "eof", "source": "cam", "variable": "O3", "n_modes": 3},
            "pc1_wav": {"type": "wavelet", "source": "cam_O3_eof", "variable": "pc", "mode": 1},
        },
        "plots": {"scal": {"type": "wavelet_scalogram", "source": "pc1_wav", "variable": "power"}},
    }
    result = PipelineRunner(show_progress=False).run_from_config(config)
    assert result.success, getattr(result, "error", None)
    # pc1_wav was built after cam_O3_eof (dependency order) and registered.
    assert "pc1_wav" in result.context.sources
    pngs = [p for p in result.context.results["plotting"].data["plots_generated"] if p.endswith(".png")]
    assert any("scal" in p for p in pngs)
```

- [ ] **Step 2: Run test**

Run: `HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/integration/test_wavelet_pipeline.py -v`
Expected: PASS (both, if Plan B merged). Debug the real path if not.

- [ ] **Step 3: Run the wavelet gate**

```bash
HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/unit/analysis/test_reductions.py davinci_monet/tests/unit/analysis/test_wavelet_analysis.py davinci_monet/tests/unit/plots/test_wavelet_scalogram.py davinci_monet/tests/integration/test_wavelet_pipeline.py -v
mypy davinci_monet
black davinci_monet && isort davinci_monet
```
Expected: all PASS, mypy clean, formatting clean.

- [ ] **Step 4: Commit**

```bash
git add davinci_monet/tests/integration/test_wavelet_pipeline.py
git commit -m "test(wavelet): end-to-end wavelet (area-mean + of EOF PC) through pipeline"
```

---

### Task 6: Docs, example config, gallery

**Files:**
- Modify: `CLAUDE.md` (document the `wavelet` analysis type + `wavelet_scalogram`)
- Modify/Create: gallery entry / example config

- [ ] **Step 1: Document `wavelet` in CLAUDE.md**

Extend the `analyses:` subsection (added in Plan B) with the `wavelet` type: fields (`source`, `variable`, `mode`, `reduce` = `area_mean`/`{point: [lat,lon]}`/null, `omega0`, `significance_level`, `dj`, `s0`, `j`), the `SPECTRUM` output (`power`, `power_significance`, `coi`, `global_power`, `global_significance`), and the `wavelet_scalogram` plot type. Note the "wavelet of an EOF PC" pattern (`source: <eof>, variable: pc, mode: N`). Terse.

- [ ] **Step 2: Add to gallery / example config**

Add a `wavelet_scalogram` entry to `analyses/_gallery/make_gallery.py` (per CLAUDE.md, the gallery renders one figure per plot type through the pipeline) and extend the example config from Plan B with a wavelet analysis + scalogram.

- [ ] **Step 3: Run the gallery and verify**

Generate the gallery; confirm a `wavelet_scalogram` figure is produced through the pipeline. Copy PDFs to the iCloud Claude folder per CLAUDE.md.

- [ ] **Step 4: Commit**

```bash
git add CLAUDE.md analyses/_gallery/
git commit -m "docs(wavelet): document wavelet analysis + scalogram; gallery + example"
```

---

## Self-Review (completed by plan author)

- **Spec coverage:** §5.1 extraction + preprocessing order (reduce → regularize → detrend → ar1 **before** normalize → normalize) (T2,T3); §5.2 pinned pycwt calls — local `sigma_test=0`/var=1, global `sigma_test=1`/`dof=N−scales`, Morlet Fourier factor via `freqs` (T1,T3); §5.3 period units from dt (T2,T3); §5.4 outputs incl. COI semantics (T3); §6.4 scalogram + COI hatch (periods above COI) + significance contour + global panel (T4); wavelet-of-EOF-PC (T5).
- **Deferred/assumption:** pycwt signatures pinned by T1's probe; T3 conditional on it. The "explicit reduce on an already-1-D series" rule is implemented as: `point` reduce on a 1-D series errors; the defaulted `area_mean` is a silent no-op (the default cannot be distinguished from a user-set `area_mean`, so erroring on it would break the canonical PC example — documented divergence from the spec's stricter wording).
- **Type consistency:** output vars `power`/`power_significance`/`coi`/`global_power`/`global_significance`, `period` coord with `units`, attrs `kind`/`wavelet_quantity`/`dt`/`dt_units` — consistent across T3 (producer) and T4 (consumer). `reductions` helper names match between T2 and T3.
- **Placeholders:** none. The one conditional `@pytest.mark.skip` (T5 `test_wavelet_of_eof_pc`) is explicit and removed once Plan B is merged.
```
