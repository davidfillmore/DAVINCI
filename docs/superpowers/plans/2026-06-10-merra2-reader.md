# MERRA-2 Reader (`type: merra2`) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a dedicated `type: merra2` gridded source reader that opens MERRA-2 NetCDF4 (2D and 3D, model-level or pressure-level), standardizes coordinates to `(time, z, lat, lon)`, tags GRID geometry, and is validated through the pipeline.

**Architecture:** A standalone `MERRA2Reader` in `davinci_monet/models/merra2.py`, mirroring `models/cesm.py` and reusing the shared `io/reader_utils.py` helpers (`validate_file_list`, `retry_transient_open`, `select_variables`, `standardize_dims`, `set_geometry_attr`). Vertical *surface extraction* stays single-sourced downstream (`pairing/strategies/base._extract_surface`, `plots/renderers/spatial/base.surface_level_index`); the reader only preserves `z` ordering so that auto-detection works for both conventions.

**Tech Stack:** Python 3.11+, xarray (netCDF4 engine), pytest. No monetio (monetio has no MERRA-2 module). No new dependency.

**Scope:** Phase 2 of the reanalysis spec (`docs/superpowers/specs/2026-06-10-reanalysis-sources-design.md`). ERA5/CAMS readers and the MERRA-2 downloader are separate plans (downloader already implemented).

---

## Audit Findings (why this design)

1. **`generic` already reads MERRA-2.** `GenericReader` standardizes `lev`→`z` via `COMMON_COORDINATE_ALIASES` and selects the netcdf4 engine for `.nc4`. The existing `tests/integration/test_merra2_modis_aod_pipeline.py` reads MERRA-2-like data through `type: generic`. So `generic` remains the zero-config fallback; `merra2` must *add value*, not duplicate gridded read.
2. **Value a dedicated reader adds:** an explicit, discoverable `type: merra2` (matching the `cesm_fv`/`cmaq`/`ufs`/`wrfchem` convention of named readers); a deterministic `lev`→`z` rename; a documented guarantee about the Np vs Nv vertical convention; and **macOS resource-fork (`._*`) filtering** — a real gotcha because the staged MERRA-2 files on the external Io drive have `._*.nc4` sidecars that break `xr.open_mfdataset` when globbed.
3. **Convention = standalone, not subclassing.** `cesm.py`/`cmaq.py` are standalone readers using `reader_utils` helpers; none subclass `GenericReader`. This plan follows that.
4. **Vertical convention (verified against staged April 2026 files):** `*_Np` pressure-level products have `lev` running 1000 hPa (index 0) → 0.1 hPa (last), so **surface = index 0**; `*_Nv` model-level products run surface = last index. `surface_level_index()` auto-detects via `vert_vals[-1] > vert_vals[0]`, so both work with no special-casing — but this plan *tests* both so the guarantee is enforced.

**Alternative considered:** subclass `GenericReader` and override `name`. Rejected — it introduces a subclassing pattern used nowhere else and couples the reader to `generic`'s progress-callback/engine internals. (If you prefer it at plan review, say so.)

---

## File Structure

- Create: `davinci_monet/models/merra2.py` — `MERRA2Reader`, registered `type: merra2`.
- Modify: `davinci_monet/models/__init__.py` — import the module so registration runs (match how cesm is imported).
- Test (unit): `davinci_monet/tests/test_merra2_reader.py` — synthetic 2D/3D, multi-file, resource-fork, registration.
- Test (integration): `davinci_monet/tests/integration/test_merra2_reader_pipeline.py` — `type: merra2` through `PipelineRunner.run_from_config()` + a real-data skipif smoke against Io.
- Create: `analyses/reanalysis/configs/merra2-aod-modis.example.yaml` — portable template.

---

### Task 1: Reader skeleton + registration

**Files:**
- Create: `davinci_monet/models/merra2.py`
- Modify: `davinci_monet/models/__init__.py`
- Test: `davinci_monet/tests/test_merra2_reader.py`

- [ ] **Step 1: Write the failing test**

Create `davinci_monet/tests/test_merra2_reader.py`:

```python
"""Unit tests for the MERRA-2 gridded reader."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from davinci_monet.core.protocols import DataGeometry
from davinci_monet.core.registry import source_registry
from davinci_monet.models.merra2 import MERRA2Reader


def test_reader_registered_and_grid_geometry() -> None:
    assert "merra2" in source_registry
    reader = MERRA2Reader()
    assert reader.name == "merra2"
    assert reader.geometry is DataGeometry.GRID
```

- [ ] **Step 2: Run test to verify it fails**

Run: `HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/test_merra2_reader.py -q`
Expected: FAIL — `ModuleNotFoundError: davinci_monet.models.merra2`

- [ ] **Step 3: Write minimal implementation**

Create `davinci_monet/models/merra2.py`:

```python
"""MERRA-2 reader.

Reads MERRA-2 (NASA GMAO) gridded NetCDF4 output as a GRID source. Handles
2D collections (e.g. ``tavg1_2d_slv_Nx``, ``tavgM_2d_aer_Nx``) and 3D
collections on model levels (``*_Nv``) or pressure levels (``*_Np``).

Vertical convention
-------------------
The reader renames the vertical dim ``lev`` to the canonical ``z`` and
preserves its ordering. It does NOT slice a surface level — surface
extraction is single-sourced downstream (``_extract_surface`` /
``surface_level_index``), which auto-detects the surface by whether the
vertical coordinate increases with index. For MERRA-2 that means:

* ``*_Nv`` (model levels): values increase with index, surface = last.
* ``*_Np`` (pressure levels): values decrease with index (1000 hPa first),
  surface = first.

MERRA-2 files on external drives carry macOS resource-fork sidecars
(``._*.nc4``); these are filtered before opening.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import xarray as xr

from davinci_monet.core.protocols import DataGeometry
from davinci_monet.core.registry import source_registry
from davinci_monet.io.reader_utils import (
    retry_transient_open,
    select_variables,
    set_geometry_attr,
    standardize_dims,
    validate_file_list,
)


@source_registry.register("merra2")
class MERRA2Reader:
    """Reader for MERRA-2 gridded NetCDF4 output."""

    @property
    def name(self) -> str:
        """Return reader name."""
        return "merra2"

    @property
    def geometry(self) -> DataGeometry:
        """MERRA-2 output is gridded."""
        return DataGeometry.GRID
```

Modify `davinci_monet/models/__init__.py` to import the reader (registration runs on package import) and export it, matching the existing explicit style. Add the import alphabetically after the `generic` line:

```python
from davinci_monet.models.merra2 import MERRA2Reader
```

and add `"MERRA2Reader",` to the `__all__` list (after `"GenericReader",`).

- [ ] **Step 4: Run test to verify it passes**

Run: `HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/test_merra2_reader.py -q`
Expected: PASS (1 passed)

- [ ] **Step 5: Commit**

```bash
git add davinci_monet/models/merra2.py davinci_monet/models/__init__.py davinci_monet/tests/test_merra2_reader.py
git commit -m "feat(models): MERRA2Reader skeleton + registration (type: merra2)"
```

---

### Task 2: `open()` + 2D standardization

**Files:**
- Modify: `davinci_monet/models/merra2.py`
- Test: `davinci_monet/tests/test_merra2_reader.py`

- [ ] **Step 1: Write the failing test**

Append to `davinci_monet/tests/test_merra2_reader.py`:

```python
def _make_2d(varname: str, nt: int = 3) -> xr.Dataset:
    """A MERRA-2-like 2D field on (time, lat, lon)."""
    times = np.array(
        ["2026-04-01", "2026-04-02", "2026-04-03"], dtype="datetime64[ns]"
    )[:nt]
    lat = np.linspace(-90.0, 90.0, 6)
    lon = np.linspace(-180.0, 179.375, 8)
    rng = np.random.default_rng(0)
    data = rng.uniform(0.05, 0.8, size=(nt, 6, 8)).astype(np.float32)
    return xr.Dataset(
        {varname: (("time", "lat", "lon"), data)},
        coords={"time": times, "lat": lat, "lon": lon},
    )


def test_open_2d_standardizes_and_tags_geometry(tmp_path: Path) -> None:
    f = tmp_path / "MERRA2_400.tavgM_2d_aer_Nx.202604.nc4"
    _make_2d("TOTEXTTAU").to_netcdf(f)

    ds = MERRA2Reader().open([f])

    assert set(ds["TOTEXTTAU"].dims) == {"time", "lat", "lon"}
    assert "z" not in ds.dims  # 2D: no vertical
    assert ds.attrs["geometry"] == "grid"


def test_open_subsets_requested_variables(tmp_path: Path) -> None:
    ds_in = _make_2d("TOTEXTTAU")
    ds_in["DUEXTTAU"] = ds_in["TOTEXTTAU"] * 0.3
    f = tmp_path / "MERRA2_400.tavgM_2d_aer_Nx.202604.nc4"
    ds_in.to_netcdf(f)

    ds = MERRA2Reader().open([f], variables=["TOTEXTTAU"])

    assert "TOTEXTTAU" in ds.data_vars
    assert "DUEXTTAU" not in ds.data_vars
```

- [ ] **Step 2: Run test to verify it fails**

Run: `HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/test_merra2_reader.py -q`
Expected: FAIL — `AttributeError: 'MERRA2Reader' object has no attribute 'open'`

- [ ] **Step 3: Write minimal implementation**

Append to `MERRA2Reader` in `davinci_monet/models/merra2.py`:

```python
    def open(
        self,
        file_paths: Sequence[str | Path],
        variables: Sequence[str] | None = None,
        **kwargs: Any,
    ) -> xr.Dataset:
        """Open MERRA-2 NetCDF4 files and standardize to (time, z, lat, lon).

        Parameters
        ----------
        file_paths
            Paths to MERRA-2 ``.nc4`` files (resource-fork ``._*`` sidecars
            are ignored).
        variables
            Variables to load. If None, loads all.
        **kwargs
            Passed through to xarray's open functions.

        Returns
        -------
        xr.Dataset
            Standardized dataset with GRID geometry tagged.
        """
        # Filter macOS resource-fork sidecars before validation so the count
        # reflects real data files (external drives carry ``._*.nc4``).
        real = [Path(f) for f in file_paths if not Path(f).name.startswith("._")]
        file_list = validate_file_list(real, source_label="MERRA-2")

        def _open() -> xr.Dataset:
            if len(file_list) > 1:
                ds = xr.open_mfdataset(
                    [str(f) for f in file_list],
                    combine="by_coords",
                    parallel=True,
                    **kwargs,
                )
            else:
                ds = xr.open_dataset(str(file_list[0]), **kwargs)
            return select_variables(ds, variables)

        ds = retry_transient_open(_open, context="Opening MERRA-2 files")
        return self._standardize_dataset(ds)

    def _standardize_dataset(self, ds: xr.Dataset) -> xr.Dataset:
        """Rename ``lev``->``z`` (when present) and tag GRID geometry."""
        ds = standardize_dims(ds, {"lev": "z"})
        return set_geometry_attr(ds, DataGeometry.GRID)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/test_merra2_reader.py -q`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add davinci_monet/models/merra2.py davinci_monet/tests/test_merra2_reader.py
git commit -m "feat(models): MERRA2Reader.open + 2D standardization (lev->z, geometry, resource-fork filter)"
```

---

### Task 3: 3D vertical (`lev`->`z`) + surface-index guarantee (Np and Nv)

**Files:**
- Test: `davinci_monet/tests/test_merra2_reader.py`

(No implementation change — `_standardize_dataset` already renames `lev`->`z`. This task *proves* the documented Np/Nv surface guarantee end-to-end at the downstream helper boundary.)

- [ ] **Step 1: Write the failing test**

Append to `davinci_monet/tests/test_merra2_reader.py`:

```python
from davinci_monet.plots.renderers.spatial.base import surface_level_index


def _make_3d(lev_values: np.ndarray, nt: int = 2) -> xr.Dataset:
    """A MERRA-2-like 3D field on (time, lev, lat, lon)."""
    times = np.array(["2026-04-01", "2026-04-02"], dtype="datetime64[ns]")[:nt]
    lat = np.linspace(-90.0, 90.0, 5)
    lon = np.linspace(-180.0, 179.0, 6)
    nz = len(lev_values)
    rng = np.random.default_rng(1)
    data = rng.uniform(0.0, 1.0, size=(nt, nz, 5, 6)).astype(np.float32)
    return xr.Dataset(
        {"CLOUD": (("time", "lev", "lat", "lon"), data)},
        coords={"time": times, "lev": lev_values, "lat": lat, "lon": lon},
    )


def test_open_3d_renames_lev_to_z(tmp_path: Path) -> None:
    # Np pressure levels: 1000 hPa first -> 0.1 hPa last.
    f = tmp_path / "MERRA2_400.tavg3_3d_cld_Np.20260401.nc4"
    _make_3d(np.array([1000.0, 850.0, 500.0, 100.0, 0.1])).to_netcdf(f)

    ds = MERRA2Reader().open([f])

    assert "z" in ds.dims
    assert "lev" not in ds.dims
    assert set(ds["CLOUD"].dims) == {"time", "z", "lat", "lon"}


def test_surface_index_np_is_first(tmp_path: Path) -> None:
    # Pressure decreasing with index -> surface is index 0.
    f = tmp_path / "MERRA2_400.tavg3_3d_cld_Np.20260401.nc4"
    _make_3d(np.array([1000.0, 850.0, 500.0, 100.0, 0.1])).to_netcdf(f)
    ds = MERRA2Reader().open([f])
    assert surface_level_index(ds["CLOUD"], "z") == 0


def test_surface_index_nv_is_last(tmp_path: Path) -> None:
    # Model level index increasing -> surface is last index.
    f = tmp_path / "MERRA2_400.inst3_3d_aer_Nv.20260401.nc4"
    _make_3d(np.array([1.0, 2.0, 3.0, 71.0, 72.0])).to_netcdf(f)
    ds = MERRA2Reader().open([f])
    assert surface_level_index(ds["CLOUD"], "z") == -1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/test_merra2_reader.py -q`
Expected: the three new tests FAIL only if standardization is wrong; given Task 2 they should PASS. If any FAIL, fix `_standardize_dataset` before proceeding. (This task is a regression guard for the vertical guarantee.)

- [ ] **Step 3: No new implementation** — `_standardize_dataset` already covers it.

- [ ] **Step 4: Run test to verify it passes**

Run: `HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/test_merra2_reader.py -q`
Expected: PASS (6 passed)

- [ ] **Step 5: Commit**

```bash
git add davinci_monet/tests/test_merra2_reader.py
git commit -m "test(models): MERRA-2 3D lev->z + Np/Nv surface-index guarantee"
```

---

### Task 4: Multi-file concat + resource-fork filtering

**Files:**
- Test: `davinci_monet/tests/test_merra2_reader.py`

- [ ] **Step 1: Write the failing test**

Append to `davinci_monet/tests/test_merra2_reader.py`:

```python
def test_open_multifile_concats_time(tmp_path: Path) -> None:
    for day in ("01", "02"):
        ds = _make_2d("TOTEXTTAU", nt=1)
        ds = ds.assign_coords(
            time=np.array([f"2026-04-{day}"], dtype="datetime64[ns]")
        )
        ds.to_netcdf(tmp_path / f"MERRA2_400.tavgM_2d_aer_Nx.2026{day}.nc4")

    files = sorted(tmp_path.glob("MERRA2_400.*.nc4"))
    ds = MERRA2Reader().open(files)
    assert ds.sizes["time"] == 2


def test_open_ignores_resource_fork_sidecars(tmp_path: Path) -> None:
    real = tmp_path / "MERRA2_400.tavgM_2d_aer_Nx.202604.nc4"
    _make_2d("TOTEXTTAU").to_netcdf(real)
    # macOS resource-fork sidecar: not a valid NetCDF; would break open if read.
    (tmp_path / "._MERRA2_400.tavgM_2d_aer_Nx.202604.nc4").write_bytes(b"\x00\x05")

    files = sorted(tmp_path.glob("*.nc4"))  # includes the ._ sidecar
    ds = MERRA2Reader().open(files)
    assert "TOTEXTTAU" in ds.data_vars
```

- [ ] **Step 2: Run test to verify it fails or passes**

Run: `HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/test_merra2_reader.py -q`
Expected: both PASS given Task 2's filter + `open_mfdataset`. If `test_open_ignores_resource_fork_sidecars` FAILS, the filter in `open()` is wrong — fix it.

- [ ] **Step 3: No new implementation** if green; otherwise fix the `._` filter.

- [ ] **Step 4: Confirm pass** — `8 passed`.

- [ ] **Step 5: Commit**

```bash
git add davinci_monet/tests/test_merra2_reader.py
git commit -m "test(models): MERRA-2 multi-file concat + resource-fork sidecar filtering"
```

---

### Task 5: Pipeline integration (`type: merra2` through `run_from_config`)

**Files:**
- Create: `davinci_monet/tests/integration/test_merra2_reader_pipeline.py`

- [ ] **Step 1: Write the failing test**

Create `davinci_monet/tests/integration/test_merra2_reader_pipeline.py`:

```python
"""Integration: MERRA-2 reader through the full pipeline.

Exercises PipelineRunner.run_from_config() with a ``type: merra2`` GRID source
paired against a synthetic GRID obs source, mirroring MERRA-2 AOD vs a gridded
AOD reference. This is the pipeline path a user takes with ``davinci-monet run``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import xarray as xr
import yaml

from davinci_monet.pipeline.runner import PipelineRunner

pytestmark = pytest.mark.integration


def _grid(varname: str, seed: int) -> xr.Dataset:
    times = np.array(
        ["2026-04-01", "2026-04-02", "2026-04-03"], dtype="datetime64[ns]"
    )
    lat = np.linspace(-87.5, 87.5, 6)
    lon = np.linspace(-175.0, 175.0, 8)
    rng = np.random.default_rng(seed)
    data = rng.uniform(0.05, 0.8, size=(3, 6, 8)).astype(np.float32)
    return xr.Dataset(
        {varname: (("time", "lat", "lon"), data)},
        coords={"time": times, "lat": lat, "lon": lon},
    )


def test_merra2_reader_pipeline(tmp_path: Path) -> None:
    m_dir = tmp_path / "merra2"
    o_dir = tmp_path / "obs"
    m_dir.mkdir()
    o_dir.mkdir()
    _grid("TOTEXTTAU", seed=1).to_netcdf(
        m_dir / "MERRA2_400.tavgM_2d_aer_Nx.202604.nc4"
    )
    _grid("aod_550nm", seed=2).to_netcdf(o_dir / "obs.nc")

    out_dir = tmp_path / "output"
    config = {
        "analysis": {
            "start_time": "2026-04-01",
            "end_time": "2026-04-03",
            "output_dir": str(out_dir),
            "log_dir": str(tmp_path / "logs"),
        },
        "sources": {
            "merra2": {
                "type": "merra2",
                "role": "model",
                "files": str(m_dir / "*.nc4"),
                "variables": {"TOTEXTTAU": {"units": "1"}},
            },
            "ref": {
                "type": "generic",
                "role": "obs",
                "files": str(o_dir / "*.nc"),
                "variables": {"aod_550nm": {"units": "1"}},
            },
        },
        "pairs": {
            "merra2_vs_ref": {
                "sources": ["merra2", "ref"],
                "reference": "ref",
                "variables": {"merra2": "TOTEXTTAU", "ref": "aod_550nm"},
            }
        },
        "plots": {
            "bias": {"type": "spatial_bias", "pairs": ["merra2_vs_ref"], "title": "AOD Bias"},
            "sc": {"type": "scatter", "pairs": ["merra2_vs_ref"], "title": "AOD Scatter"},
        },
        "stats": {"output_table": True, "metrics": ["N", "MB", "RMSE", "R"]},
    }
    cfg = tmp_path / "config.yaml"
    cfg.write_text(yaml.safe_dump(config))

    result = PipelineRunner(show_progress=False).run_from_config(str(cfg))

    failed = [
        f"{s.stage_name}: {s.error}"
        for s in result.stage_results
        if s.status.name == "FAILED"
    ]
    assert result.success, f"Pipeline failed: {failed}"
    assert sorted(out_dir.rglob("*.png")), "expected plots"
    assert list(out_dir.rglob("*.csv")), "expected a stats CSV"
```

- [ ] **Step 2: Run test to verify it fails (or passes)**

Run: `HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/integration/test_merra2_reader_pipeline.py -q`
Expected: PASS once the reader is registered. If it FAILS with an unknown-source-type error, confirm `models/__init__.py` imports `merra2` (Task 1). If a pairing/plot error appears, read the stage error in the assertion message and fix — do NOT switch the source to `generic` to make it pass (that would bypass the code under test, violating the testing rules).

- [ ] **Step 3: No new implementation** unless Step 2 surfaces a real reader bug.

- [ ] **Step 4: Confirm pass.**

- [ ] **Step 5: Commit**

```bash
git add davinci_monet/tests/integration/test_merra2_reader_pipeline.py
git commit -m "test(integration): MERRA-2 reader through the full pipeline (type: merra2)"
```

---

### Task 6: Example config + real-data skipif smoke

**Files:**
- Create: `analyses/reanalysis/configs/merra2-aod-modis.example.yaml`
- Modify: `davinci_monet/tests/integration/test_merra2_reader_pipeline.py`

- [ ] **Step 1: Write the example config**

Create `analyses/reanalysis/configs/merra2-aod-modis.example.yaml`:

```yaml
# MERRA-2 (GOCART) total AOD vs a gridded AOD reference. Portable template:
# set REANALYSIS_DATA and REANALYSIS_OUT, or replace with absolute paths.
analysis:
  start_time: "2026-04-01"
  end_time: "2026-04-30"
  output_dir: ${REANALYSIS_OUT}/output
  log_dir: ${REANALYSIS_OUT}/logs
  style:
    theme: ncar
    context: default

sources:
  merra2:
    type: merra2
    role: model
    files: /Volumes/Io/MERRA2_tavgM/aer_Nx/MERRA2_*.tavgM_2d_aer_Nx.202604.nc4
    variables:
      TOTEXTTAU:
        units: "1"          # AOD is dimensionless; no unit_scale needed
  modis:
    type: generic
    role: obs
    files: ${REANALYSIS_DATA}/modis/*.nc
    variables:
      aod_550nm:
        obs_min: 0
        obs_max: 5

pairs:
  merra2_vs_modis:
    sources: [merra2, modis]
    reference: modis
    variables:
      merra2: TOTEXTTAU
      modis: aod_550nm

plots:
  aod_bias:
    type: spatial_bias
    pairs: [merra2_vs_modis]
    title: "MERRA-2 vs MODIS Total AOD (550 nm)"
  aod_scatter:
    type: scatter
    pairs: [merra2_vs_modis]
    title: "AOD Scatter"

stats:
  output_table: true
  metrics: [N, MB, RMSE, R, NMB, NME, IOA]
```

- [ ] **Step 2: Write the real-data skipif smoke test**

Append to `davinci_monet/tests/integration/test_merra2_reader_pipeline.py`:

```python
import os

_IO_AER = Path("/Volumes/Io/MERRA2_tavgM/aer_Nx")


@pytest.mark.skipif(
    not _IO_AER.is_dir(),
    reason="MERRA-2 monthly aerosol data not staged on /Volumes/Io",
)
def test_real_merra2_file_opens() -> None:
    """Smoke: open one staged monthly aerosol file via the MERRA-2 reader."""
    from davinci_monet.models.merra2 import MERRA2Reader

    files = sorted(
        f for f in _IO_AER.glob("MERRA2_*.tavgM_2d_aer_Nx.*.nc4")
        if not f.name.startswith("._")
    )
    if not files:
        pytest.skip("no monthly aerosol .nc4 files present")

    ds = MERRA2Reader().open([files[0]], variables=["TOTEXTTAU"])
    assert "TOTEXTTAU" in ds.data_vars
    assert {"time", "lat", "lon"} <= set(ds["TOTEXTTAU"].dims)
    assert ds.attrs["geometry"] == "grid"
    da = ds["TOTEXTTAU"]
    assert 0.0 <= float(da.min()) and float(da.max()) < 5.0  # physical AOD range
```

- [ ] **Step 3: Run tests**

Run: `HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/integration/test_merra2_reader_pipeline.py -q`
Expected: synthetic test PASS; real-data test PASS if Io is mounted with staged April 2026 data, else SKIP.

- [ ] **Step 4: Commit**

```bash
git add analyses/reanalysis/configs/merra2-aod-modis.example.yaml davinci_monet/tests/integration/test_merra2_reader_pipeline.py
git commit -m "test(integration)+docs: MERRA-2 example config + real-data skipif smoke"
```

---

### Task 7: Gate sweep + full suite

**Files:** (verification only)

- [ ] **Step 1: Targeted gates**

Run (in the `davinci` conda env):
```bash
HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/test_merra2_reader.py davinci_monet/tests/integration/test_merra2_reader_pipeline.py -q
mypy davinci_monet/models/merra2.py
black --check davinci_monet/models/merra2.py davinci_monet/tests/test_merra2_reader.py davinci_monet/tests/integration/test_merra2_reader_pipeline.py
isort --check davinci_monet/models/merra2.py davinci_monet/tests/test_merra2_reader.py davinci_monet/tests/integration/test_merra2_reader_pipeline.py
```
Expected: tests pass; mypy clean; black/isort report no changes (run without `--check` to auto-fix, then re-gate and amend the relevant commit).

- [ ] **Step 2: Full suite (no regressions)**

Run: `HDF5_USE_FILE_LOCKING=FALSE python -m pytest -q`
Expected: all pass (≥ prior 1270 + new tests), 1 pre-existing skip (+ 1 conditional skip if Io absent).

- [ ] **Step 3: No commit** — verification only (unless formatting fixes were needed in Step 1).

---

## Self-Review

- **Spec coverage:** Implements the spec's MERRA-2 reader (Phase 2): `type: merra2`, GRID, 2D + 3D, the AOD validation case (example config + pipeline test). Surface-extraction reuse and the Np/Nv convention are honored and tested.
- **Placeholder scan:** No TBD/TODO; every code step is complete.
- **Type/name consistency:** `MERRA2Reader` (`name`="merra2", `geometry`=GRID), `open(file_paths, variables, **kwargs)`, `_standardize_dataset` used consistently across tasks. `surface_level_index(field_da, level_dim)` signature matches `plots/renderers/spatial/base.py`. Config keys (`sources`/`pairs`/`reference`/`variables`) match the existing integration test.
- **Anti-reinvention:** reuses `reader_utils` helpers; `generic` stays the untyped fallback; standalone pattern matches `cesm.py`. Resource-fork filtering and the tested Np/Nv guarantee are the concrete value over `generic`.
- **Testing-rule compliance:** the integration test runs through `PipelineRunner.run_from_config()` (not the reader API directly); Step 2 explicitly forbids falling back to `generic` to force a green run.
```
