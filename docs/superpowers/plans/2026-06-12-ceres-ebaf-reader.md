# CERES EBAF Reader (Phase 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A `type: ceres_ebaf` GRID reader for the CERES EBAF monthly netCDF, validated through the pipeline and against the staged Io sample.

**Architecture:** One new module `observations/satellite/ceres_l3.py` containing `CERESEBAFReader` (the Phase 2 `CERESSYN1degReader` lands in the same file later). It mirrors `models/merra2.py` exactly — `@source_registry.register`, `name`/`geometry` properties, `open()` built from `reader_utils` helpers — plus two EBAF-specific standardizations: drop climatology dims (`ctime`/`sc`) when no selected variable uses them, and normalize longitude from 0–360 to the repo's sorted [-180, 180) convention. Registration requires importing the module in the satellite package `__init__`.

**Tech Stack:** xarray/netCDF4 (no pyhdf in this phase), pytest, mypy/black/isort in the `davinci` conda env.

**Spec:** `docs/superpowers/specs/2026-06-12-ceres-readers-design.md` (Phase 1 scope)

**Repo rules:**
- All commands in the `davinci` conda env: `source ~/miniconda3/etc/profile.d/conda.sh && conda activate davinci`
- pytest prefixed with `HDF5_USE_FILE_LOCKING=FALSE`
- Commit steps execute only with the user's standing approval for this plan.

---

### Task 1: `CERESEBAFReader` + unit tests

**Files:**
- Create: `davinci_monet/observations/satellite/ceres_l3.py`
- Modify: `davinci_monet/observations/satellite/__init__.py` (add import + `__all__` entry)
- Modify: `davinci_monet/observations/__init__.py` (add import, mirroring `modis_viirs` at line ~36)
- Test: `davinci_monet/tests/test_ceres_l3_readers.py`

- [ ] **Step 1.1: Write the failing tests**

Create `davinci_monet/tests/test_ceres_l3_readers.py`:

```python
"""Unit tests for the CERES L3 readers (EBAF in Phase 1)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from davinci_monet.core.protocols import DataGeometry
from davinci_monet.core.registry import source_registry
from davinci_monet.observations.satellite.ceres_l3 import CERESEBAFReader


def _ebaf_like(nt: int = 3) -> xr.Dataset:
    """An EBAF-shaped dataset: monthly vars on (time, lat, lon) with 0-360
    longitudes, plus a ctime-dimensioned climatology variable."""
    times = np.array(
        ["2025-10-01", "2025-11-01", "2025-12-01"], dtype="datetime64[ns]"
    )[:nt]
    lat = np.linspace(-89.5, 89.5, 4)
    lon = np.array([0.5, 90.5, 180.5, 270.5])  # EBAF convention: 0-360
    rng = np.random.default_rng(0)
    monthly = rng.uniform(150.0, 300.0, size=(nt, 4, 4)).astype(np.float32)
    clim = rng.uniform(150.0, 300.0, size=(2, 4, 4)).astype(np.float32)
    return xr.Dataset(
        {
            "toa_lw_all_mon": (("time", "lat", "lon"), monthly),
            "toa_sw_all_mon": (("time", "lat", "lon"), monthly[::-1] * 0.5),
            "toa_lw_all_clim": (("ctime", "lat", "lon"), clim),
        },
        coords={"time": times, "lat": lat, "lon": lon, "ctime": [1, 2]},
    )


def _write(ds: xr.Dataset, path: Path) -> Path:
    ds.to_netcdf(path)
    return path


def test_reader_registered_and_grid_geometry() -> None:
    reader_cls = source_registry.get("ceres_ebaf")
    reader = reader_cls()
    assert reader.name == "ceres_ebaf"
    assert reader.geometry is DataGeometry.GRID


def test_open_selects_variables_and_drops_climatology_dims(tmp_path: Path) -> None:
    path = _write(_ebaf_like(), tmp_path / "CERES_EBAF_Edition4.2.1_200003-202512.nc")

    ds = CERESEBAFReader().open([path], variables=["toa_lw_all_mon"])

    assert set(ds.data_vars) == {"toa_lw_all_mon"}
    assert "ctime" not in ds.dims  # climatology dim dropped when unused
    assert ds.attrs["geometry"] == "grid"


def test_open_without_selection_keeps_climatology(tmp_path: Path) -> None:
    path = _write(_ebaf_like(), tmp_path / "ebaf.nc")

    ds = CERESEBAFReader().open([path])

    assert "toa_lw_all_clim" in ds.data_vars
    assert "ctime" in ds.dims


def test_longitude_normalized_to_pm180_and_sorted(tmp_path: Path) -> None:
    src = _ebaf_like()
    # Plant a recognizable value at lon=270.5 (-> -89.5 after wrap), t=0, lat=0
    marked = src["toa_lw_all_mon"].values.copy()
    marked[0, 0, 3] = 222.25
    src["toa_lw_all_mon"] = (("time", "lat", "lon"), marked)
    path = _write(src, tmp_path / "ebaf.nc")

    ds = CERESEBAFReader().open([path], variables=["toa_lw_all_mon"])

    lon = ds["lon"].values
    assert lon.min() >= -180.0 and lon.max() < 180.0
    assert np.all(np.diff(lon) > 0)  # sorted ascending
    np.testing.assert_allclose(lon, [-179.5, -89.5, 0.5, 90.5])
    # Data moved with its coordinate: the marked value now sits at lon=-89.5
    got = float(ds["toa_lw_all_mon"].sel(lon=-89.5).isel(time=0, lat=0).values)
    assert got == pytest.approx(222.25)


def test_open_multifile_concats_time(tmp_path: Path) -> None:
    full = _ebaf_like()
    _write(full.isel(time=slice(0, 2)), tmp_path / "ebaf_a.nc")
    _write(full.isel(time=slice(2, 3)), tmp_path / "ebaf_b.nc")

    ds = CERESEBAFReader().open(
        sorted(tmp_path.glob("ebaf_*.nc")), variables=["toa_lw_all_mon"]
    )

    assert ds.sizes["time"] == 3


def test_open_ignores_resource_fork_sidecars(tmp_path: Path) -> None:
    path = _write(_ebaf_like(), tmp_path / "ebaf.nc")
    (tmp_path / "._ebaf.nc").write_bytes(b"\x00\x05\x16\x07junk")

    ds = CERESEBAFReader().open(
        sorted(tmp_path.glob("*ebaf.nc")), variables=["toa_lw_all_mon"]
    )

    assert ds.sizes["time"] == 3
```

- [ ] **Step 1.2: Run tests to verify they fail**

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate davinci
HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/test_ceres_l3_readers.py -v
```

Expected: collection error — `ImportError: cannot import name 'CERESEBAFReader'` (module does not exist).

- [ ] **Step 1.3: Implement the reader**

Create `davinci_monet/observations/satellite/ceres_l3.py`:

```python
"""CERES L3 gridded readers (EBAF; SYN1deg arrives in Phase 2).

EBAF (Energy Balanced and Filled) ships as a single whole-record monthly
netCDF on a 1-degree grid with CF-standard ``(time, lat, lon)`` dims plus
``ctime``/``sc``-dimensioned climatology variables. This reader:

* selects requested variables (native EBAF names, e.g. ``toa_lw_all_mon``);
* drops climatology dims when no selected variable uses them;
* normalizes longitude from EBAF's 0-360 to the repo convention of
  sorted [-180, 180);
* tags GRID geometry for the pairing engine.
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
    validate_file_list,
)


def _drop_unused_dims(ds: xr.Dataset) -> xr.Dataset:
    """Drop dims (e.g. EBAF's ``ctime``/``sc``) used by no data variable."""
    used: set[Any] = set()
    for var in ds.data_vars.values():
        used.update(var.dims)
    unused = [d for d in ds.dims if d not in used]
    return ds.drop_dims(unused) if unused else ds


def _normalize_longitude(ds: xr.Dataset) -> xr.Dataset:
    """Wrap a 0-360 ``lon`` coord to [-180, 180) and sort ascending."""
    if "lon" not in ds.coords:
        return ds
    lon = ds["lon"].values
    if lon.size and float(lon.max()) > 180.0:
        ds = ds.assign_coords(lon=(((lon + 180.0) % 360.0) - 180.0))
        ds = ds.sortby("lon")
    return ds


@source_registry.register("ceres_ebaf")
class CERESEBAFReader:
    """Reader for CERES EBAF monthly gridded netCDF (TOA + surface fluxes)."""

    @property
    def name(self) -> str:
        """Return reader name."""
        return "ceres_ebaf"

    @property
    def geometry(self) -> DataGeometry:
        """EBAF is gridded."""
        return DataGeometry.GRID

    def open(
        self,
        file_paths: Sequence[str | Path],
        variables: Sequence[str] | None = None,
        **kwargs: Any,
    ) -> xr.Dataset:
        """Open EBAF netCDF file(s) and standardize to (time, lat, lon).

        Parameters
        ----------
        file_paths
            Paths to EBAF ``.nc`` files (resource-fork ``._*`` sidecars are
            ignored). EBAF normally ships as one whole-record file.
        variables
            Native EBAF variable names to load (e.g. ``toa_lw_all_mon``).
            If None, loads all, including climatology variables.
        **kwargs
            Passed through to xarray's open functions.

        Returns
        -------
        xr.Dataset
            Standardized dataset with GRID geometry tagged.
        """
        real = [Path(f) for f in file_paths if not Path(f).name.startswith("._")]
        file_list = validate_file_list(real, source_label="CERES EBAF")

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

        ds = retry_transient_open(_open, context="Opening CERES EBAF files")
        return self._standardize_dataset(ds)

    def _standardize_dataset(self, ds: xr.Dataset) -> xr.Dataset:
        """Drop unused climatology dims, fix longitudes, tag GRID geometry."""
        ds = _drop_unused_dims(ds)
        ds = _normalize_longitude(ds)
        return set_geometry_attr(ds, DataGeometry.GRID)
```

Register the module. In `davinci_monet/observations/satellite/__init__.py`, add after the `modis_viirs` import block (keep isort grouping; isort will fix ordering if needed):

```python
from davinci_monet.observations.satellite.ceres_l3 import (  # noqa: F401  (registers "ceres_ebaf")
    CERESEBAFReader,
)
```

and add `"CERESEBAFReader",` to `__all__` (alphabetical position, near the top under a `# CERES L3` comment, matching the existing grouped style).

In `davinci_monet/observations/__init__.py`, add alongside the other satellite imports (~line 36):

```python
from davinci_monet.observations.satellite.ceres_l3 import CERESEBAFReader
```

and add `"CERESEBAFReader",` to that module's `__all__` (check how `MODISVIIRSReader` is listed there and mirror it — if it is not in that `__all__`, skip the `__all__` edit and keep only the import for registration parity).

- [ ] **Step 1.4: Run tests to verify they pass**

```bash
HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/test_ceres_l3_readers.py -v
```

Expected: 6 passed.

- [ ] **Step 1.5: Gates on touched files**

```bash
mypy davinci_monet/observations/satellite/ceres_l3.py
black --check davinci_monet/observations/satellite/ceres_l3.py davinci_monet/tests/test_ceres_l3_readers.py \
              davinci_monet/observations/satellite/__init__.py davinci_monet/observations/__init__.py
isort --check davinci_monet/observations/satellite/ceres_l3.py davinci_monet/tests/test_ceres_l3_readers.py \
              davinci_monet/observations/satellite/__init__.py davinci_monet/observations/__init__.py
```

Apply black/isort if they want changes, re-run the tests.

- [ ] **Step 1.6: Commit**

```bash
git add davinci_monet/observations/satellite/ceres_l3.py davinci_monet/tests/test_ceres_l3_readers.py \
        davinci_monet/observations/satellite/__init__.py davinci_monet/observations/__init__.py
git commit -m "feat(obs): CERES EBAF L3 gridded reader (type: ceres_ebaf)"
```

---

### Task 2: Pipeline integration test (synthetic)

**Files:**
- Test: `davinci_monet/tests/integration/test_ceres_readers_pipeline.py` (new)

- [ ] **Step 2.1: Write the failing-by-construction test, then run it**

Create `davinci_monet/tests/integration/test_ceres_readers_pipeline.py`:

```python
"""Integration: CERES readers through the full pipeline.

Exercises PipelineRunner.run_from_config() with a ``type: ceres_ebaf`` GRID
source as the pairing reference against a synthetic gridded model — the same
path a user takes with ``davinci-monet run``. SSF/SYN1deg pipeline tests
arrive with their phases.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import xarray as xr
import yaml

from davinci_monet.pipeline.runner import PipelineRunner

pytestmark = pytest.mark.integration


def _monthly_grid(varname: str, seed: int, lon0360: bool = False) -> xr.Dataset:
    times = np.array(
        ["2025-10-01", "2025-11-01", "2025-12-01"], dtype="datetime64[ns]"
    )
    lat = np.linspace(-87.5, 87.5, 6)
    lon = np.linspace(2.5, 357.5, 8) if lon0360 else np.linspace(-175.0, 175.0, 8)
    rng = np.random.default_rng(seed)
    data = rng.uniform(150.0, 300.0, size=(3, 6, 8)).astype(np.float32)
    return xr.Dataset(
        {varname: (("time", "lat", "lon"), data)},
        coords={"time": times, "lat": lat, "lon": lon},
    )


def test_ceres_ebaf_pipeline(tmp_path: Path) -> None:
    e_dir = tmp_path / "ebaf"
    m_dir = tmp_path / "model"
    e_dir.mkdir()
    m_dir.mkdir()
    # EBAF side uses 0-360 longitudes — the reader must normalize them so
    # GRID-GRID pairing aligns with the model's -180..180 grid.
    _monthly_grid("toa_lw_all_mon", seed=1, lon0360=True).to_netcdf(
        e_dir / "CERES_EBAF_Edition4.2.1_202510-202512.nc"
    )
    _monthly_grid("OLR", seed=2).to_netcdf(m_dir / "model.nc")

    out_dir = tmp_path / "output"
    config = {
        "analysis": {
            "start_time": "2025-10-01",
            "end_time": "2025-12-31",
            "output_dir": str(out_dir),
            "log_dir": str(tmp_path / "logs"),
        },
        "sources": {
            "ceres": {
                "type": "ceres_ebaf",
                "role": "obs",
                "files": str(e_dir / "*.nc"),
                "variables": {"toa_lw_all_mon": {"units": "W m-2"}},
            },
            "model": {
                "type": "generic",
                "role": "model",
                "files": str(m_dir / "*.nc"),
                "variables": {"OLR": {"units": "W m-2"}},
            },
        },
        "pairs": {
            "model_vs_ceres_olr": {
                "sources": ["model", "ceres"],
                "reference": "ceres",
                "variables": {"model": "OLR", "ceres": "toa_lw_all_mon"},
            }
        },
        "plots": {
            "bias": {
                "type": "spatial_bias",
                "pairs": ["model_vs_ceres_olr"],
                "title": "OLR Bias",
            },
            "sc": {
                "type": "scatter",
                "pairs": ["model_vs_ceres_olr"],
                "title": "OLR Scatter",
            },
        },
        "stats": {"output_table": True, "metrics": ["N", "MB", "RMSE", "R"]},
    }
    cfg = tmp_path / "config.yaml"
    cfg.write_text(yaml.safe_dump(config))

    result = PipelineRunner(show_progress=False).run_from_config(str(cfg))

    failed = [
        f"{s.stage_name}: {s.error}" for s in result.stage_results if s.status.name == "FAILED"
    ]
    assert result.success, f"Pipeline failed: {failed}"
    assert sorted(out_dir.rglob("*.png")), "expected plots"
    assert list(out_dir.rglob("*.csv")), "expected a stats CSV"
```

Run it:

```bash
HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/integration/test_ceres_readers_pipeline.py -v
```

Expected: PASS (Task 1 already implemented the reader). If it fails, the failure is in pipeline wiring (registration import, lon alignment, GRID-GRID reference handling) — fix the reader/registration, never relax the test to bypass the pipeline.

- [ ] **Step 2.2: Commit**

```bash
git add davinci_monet/tests/integration/test_ceres_readers_pipeline.py
git commit -m "test(integration): CERES EBAF reader through the full pipeline"
```

---

### Task 3: Real-data smoke + full gates

**Files:**
- Modify: `davinci_monet/tests/integration/test_ceres_readers_pipeline.py` (append)

- [ ] **Step 3.1: Append the skipif-gated smoke test**

Append to `davinci_monet/tests/integration/test_ceres_readers_pipeline.py`:

```python
_IO_EBAF = Path("/Volumes/Io/CERES/EBAF")


@pytest.mark.skipif(
    not _IO_EBAF.is_dir(),
    reason="CERES EBAF data not staged on /Volumes/Io",
)
def test_real_ebaf_file_opens() -> None:
    """Smoke: open the staged EBAF record via the reader, check physics."""
    from davinci_monet.observations.satellite.ceres_l3 import CERESEBAFReader

    files = sorted(
        f for f in _IO_EBAF.glob("CERES_EBAF_*.nc") if not f.name.startswith("._")
    )
    if not files:
        pytest.skip("no EBAF .nc files present")

    ds = CERESEBAFReader().open([files[0]], variables=["toa_lw_all_mon"])

    assert set(ds.data_vars) == {"toa_lw_all_mon"}
    assert ds.attrs["geometry"] == "grid"
    assert "ctime" not in ds.dims
    lon = ds["lon"].values
    assert lon.min() >= -180.0 and lon.max() < 180.0
    # Area-weighted global-mean OLR for one month must be physical.
    da = ds["toa_lw_all_mon"].isel(time=-1)
    weights = np.cos(np.deg2rad(ds["lat"]))
    gmean = float(da.weighted(weights).mean())
    assert 220.0 <= gmean <= 260.0, f"global-mean OLR {gmean:.1f} W m-2 unphysical"
```

Run (with the Io drive mounted):

```bash
HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/integration/test_ceres_readers_pipeline.py -v
```

Expected: 2 passed (or 1 passed + 1 skipped without the drive).

- [ ] **Step 3.2: Full gates**

```bash
HDF5_USE_FILE_LOCKING=FALSE python -m pytest
mypy davinci_monet
black --check davinci_monet && isort --check davinci_monet
```

Expected: full suite passes (1,313 pre-existing + 8 new = 1,321 ± skips), mypy/black/isort clean. The 2 GB EBAF open in the smoke test is lazy — it must not blow up runtime; if the suite slows noticeably, confirm no eager `.load()` crept in.

- [ ] **Step 3.3: Commit**

```bash
git add davinci_monet/tests/integration/test_ceres_readers_pipeline.py
git commit -m "test(integration): EBAF real-data smoke (skipif Io absent)"
```
