# CERES SYN1deg Reader (Phase 2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A `type: ceres_syn1deg` GRID reader assembling CERES SYN1deg HDF4 files (monthly / daily / hourly) into `(time, lat, lon)`, validated through the pipeline and against the staged Io samples.

**Architecture:** `CERESSYN1degReader` joins `CERESEBAFReader` in `observations/satellite/ceres_l3.py`. Per file: parse the date from the filename tail (`.YYYYMM` month / `.YYYYMMDD` day-or-hour), read the explicit `latitude`/`longitude` coordinate SDS (verified present in all three cadences; lat descending 89.5→−89.5, lon already ±180), read requested SDS via lazy-imported pyhdf, apply fill/valid_range/scale masking, infer hourly cadence from a `gmt_hr_index` dim (3-D SDS → 24 hourly timestamps), concat on time, flip lat ascending, tag GRID. The HDF4 masking helper moves from `MODISVIIRSReader._apply_hdf4_scale` into `io/reader_utils.py` as shared `apply_hdf4_scale` (modis_viirs delegates to it; its existing tests are the regression gate) so Phase 3 SSF can reuse it.

**Tech Stack:** pyhdf (lazy import, modis_viirs guard pattern), xarray, numpy, pytest (synthetic HDF4 written via pyhdf), mypy/black/isort in the `davinci` conda env.

**Spec:** `docs/superpowers/specs/2026-06-12-ceres-readers-design.md` (Phase 2 scope)

**Repo rules:** `davinci` conda env (`source ~/miniconda3/etc/profile.d/conda.sh && conda activate davinci`); pytest with `HDF5_USE_FILE_LOCKING=FALSE`; commits per task under the user's standing approval; real-data smokes are **opt-in via env var** (mirror the gating in `test_ceres_readers_pipeline.py` after the Phase 1 amendment — auto-on-mount smokes destabilize the suite).

**File-structure facts (CORRECTED during Task 2 review — the original
"all-2-D / all-3-D" claim was wrong; full inventories below verified against
staged samples 2026-06-12):**
- Month file (`*.202512`, 422 SDS): 145 regional 2-D `(latitude=180,
  longitude=360)` + 51 non-hourly 3-D (e.g. `(cloud_layer=5, 180, 360)`)
  + 224 1-D (zonal means like `obs_all_toa_lw_zon`).
- Day file (`*.20251201`): 94 regional 2-D + 51 non-hourly 3-D + 4 1-D.
- Hour file (`*.20251229`): 94 hourly 3-D `(gmt_hr_index=24, latitude,
  longitude)` + 51 4-D `(5, 24, 180, 360)` + 5 1-D.
- All carry 1-D `latitude` (89.5 → −89.5) and `longitude` (−179.5 → 179.5) SDS.
- Data SDS attrs include `_FillValue` (3.4028235e38), `valid_range` ([0, 500] style), `units`, `long_name`.
- Consequence: variable classification must use the SDS **dimension names**
  (`("latitude","longitude")` regional; `("gmt_hr_index","latitude","longitude")`
  hourly); everything else is skipped on a `variables=None` scan and rejected
  with a clear error when explicitly requested.

---

### Task 1: Shared `apply_hdf4_scale` in reader_utils

**Files:**
- Modify: `davinci_monet/io/reader_utils.py` (add helper + `__all__` entry)
- Modify: `davinci_monet/observations/satellite/modis_viirs.py` (`_apply_hdf4_scale` staticmethod delegates)
- Test: `davinci_monet/tests/test_reader_utils_hdf4.py` (new)

- [ ] **Step 1.1: Write the failing tests**

Create `davinci_monet/tests/test_reader_utils_hdf4.py`:

```python
"""Unit tests for the shared HDF4 scale/mask helper."""

from __future__ import annotations

import numpy as np

from davinci_monet.io.reader_utils import apply_hdf4_scale


def test_scale_and_offset_applied() -> None:
    raw = np.array([0, 10, 20], dtype=np.int16)
    out = apply_hdf4_scale(raw, {"scale_factor": 0.5, "add_offset": 1.0})
    np.testing.assert_allclose(out, [1.0, 6.0, 11.0])
    assert out.dtype == np.float64


def test_fill_value_masked_to_nan() -> None:
    fill = 3.4028234663852886e38
    raw = np.array([100.0, fill, 200.0], dtype=np.float32)
    out = apply_hdf4_scale(raw, {"_FillValue": fill})
    assert np.isnan(out[1])
    np.testing.assert_allclose(out[[0, 2]], [100.0, 200.0])


def test_valid_range_masked_to_nan() -> None:
    raw = np.array([-5.0, 250.0, 600.0])
    out = apply_hdf4_scale(raw, {"valid_range": [0.0, 500.0]})
    assert np.isnan(out[0]) and np.isnan(out[2])
    assert out[1] == 250.0


def test_no_attrs_is_identity_in_float64() -> None:
    raw = np.array([1, 2, 3], dtype=np.int32)
    out = apply_hdf4_scale(raw, {})
    np.testing.assert_allclose(out, [1.0, 2.0, 3.0])
    assert out.dtype == np.float64


def test_modis_viirs_staticmethod_delegates() -> None:
    from davinci_monet.observations.satellite.modis_viirs import MODISVIIRSReader

    raw = np.array([10], dtype=np.int16)
    out = MODISVIIRSReader._apply_hdf4_scale(raw, {"scale_factor": 2.0})
    np.testing.assert_allclose(out, [20.0])
```

- [ ] **Step 1.2: Run tests to verify they fail**

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate davinci
HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/test_reader_utils_hdf4.py -v
```

Expected: ImportError — `apply_hdf4_scale` does not exist.

- [ ] **Step 1.3: Implement**

In `davinci_monet/io/reader_utils.py`, add (matching the module's docstring/typing style; `np` import may need adding):

```python
def apply_hdf4_scale(raw: "np.ndarray[Any, np.dtype[Any]]", attrs: dict[str, Any]) -> "np.ndarray[Any, np.dtype[Any]]":
    """Apply HDF4/CF scale_factor, add_offset, _FillValue, and valid_range.

    CF convention: ``physical = raw * scale_factor + add_offset``. Values
    equal to ``_FillValue`` or outside ``valid_range`` are set to ``NaN``.
    The result is always ``float64``. Masking happens on the RAW values
    (HDF4 convention: fill/valid_range describe stored values), then the
    scale/offset is applied.
    """
    scale = float(attrs.get("scale_factor", 1.0))
    offset = float(attrs.get("add_offset", 0.0))
    fill_val = attrs.get("_FillValue")
    valid_range = attrs.get("valid_range")

    data = raw.astype("float64")
    if fill_val is not None:
        data[data == float(fill_val)] = np.nan
    if valid_range is not None and len(valid_range) == 2:
        lo, hi = float(valid_range[0]), float(valid_range[1])
        data[(data < lo) | (data > hi)] = np.nan
    if scale != 1.0 or offset != 0.0:
        data = data * scale + offset
    return data
```

IMPORTANT: before writing this, READ `MODISVIIRSReader._apply_hdf4_scale` in `davinci_monet/observations/satellite/modis_viirs.py` (~line 226) and copy ITS exact masking/scale ordering semantics into the shared helper (the snippet above reflects the expected semantics — raw-space masking, then scale; if the original differs, the original wins, since its tests pin behavior). Add `"apply_hdf4_scale"` to `__all__`. Then replace the body of `MODISVIIRSReader._apply_hdf4_scale` with a delegation:

```python
    @staticmethod
    def _apply_hdf4_scale(raw: np.ndarray, attrs: dict[str, Any]) -> np.ndarray:
        """Delegate to the shared helper in ``io.reader_utils``."""
        return apply_hdf4_scale(raw, attrs)
```

(adding the import at the top of modis_viirs.py).

- [ ] **Step 1.4: Verify**

```bash
HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/test_reader_utils_hdf4.py -v   # 5 passed
HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/ -k "modis or viirs or reader_utils" -q  # no regressions
mypy davinci_monet/io/reader_utils.py davinci_monet/observations/satellite/modis_viirs.py
black --check davinci_monet/io/reader_utils.py davinci_monet/observations/satellite/modis_viirs.py davinci_monet/tests/test_reader_utils_hdf4.py
isort --check davinci_monet/io/reader_utils.py davinci_monet/observations/satellite/modis_viirs.py davinci_monet/tests/test_reader_utils_hdf4.py
```

- [ ] **Step 1.5: Commit**

```bash
git add davinci_monet/io/reader_utils.py davinci_monet/observations/satellite/modis_viirs.py davinci_monet/tests/test_reader_utils_hdf4.py
git commit -m "refactor(io): shared apply_hdf4_scale helper (modis_viirs delegates)"
```

---

### Task 2: `CERESSYN1degReader` + unit tests

**Files:**
- Modify: `davinci_monet/observations/satellite/ceres_l3.py` (append reader)
- Modify: `davinci_monet/observations/satellite/__init__.py`, `davinci_monet/observations/__init__.py` (add `CERESSYN1degReader` import/`__all__` alongside the existing CERES entries)
- Test: `davinci_monet/tests/test_ceres_l3_readers.py` (append; includes a pyhdf synthetic-file writer)

- [ ] **Step 2.1: Write the failing tests**

Append to `davinci_monet/tests/test_ceres_l3_readers.py`:

```python
# ---------------------------------------------------------------------------
# SYN1deg (Phase 2)
# ---------------------------------------------------------------------------

pyhdf_SD = pytest.importorskip("pyhdf.SD", reason="pyhdf required for SYN1deg tests")

from davinci_monet.observations.satellite.ceres_l3 import CERESSYN1degReader  # noqa: E402

_SYN_FILL = 3.4028234663852886e38


def _write_syn_hdf4(
    path: Path,
    varname: str = "obs_all_toa_lw_reg",
    nlat: int = 4,
    nlon: int = 4,
    hourly: bool = False,
    fill_first_cell: bool = False,
) -> Path:
    """Write a minimal SYN1deg-like HDF4 file via pyhdf.

    Mirrors the real layout: 1-D ``latitude`` (descending 89.5->-89.5 style)
    and ``longitude`` (ascending +-180) SDS plus 2-D (lat, lon) or 3-D
    (gmt_hr_index, lat, lon) data SDS with fill/valid_range attrs.
    """
    SD, SDC = pyhdf_SD.SD, pyhdf_SD.SDC
    lat = np.linspace(89.5, -89.5, nlat).astype(np.float32)  # descending like real files
    lon = np.linspace(-179.5, 179.5, nlon).astype(np.float32)
    f = SD(str(path), SDC.WRITE | SDC.CREATE)

    s = f.create("latitude", SDC.FLOAT32, nlat)
    s.dim(0).setname("latitude")
    s[:] = lat
    s.endaccess()
    s = f.create("longitude", SDC.FLOAT32, nlon)
    s.dim(0).setname("longitude")
    s[:] = lon
    s.endaccess()

    rng = np.random.default_rng(0)
    if hourly:
        data = rng.uniform(150.0, 300.0, size=(24, nlat, nlon)).astype(np.float32)
        if fill_first_cell:
            data[0, 0, 0] = _SYN_FILL
        s = f.create(varname, SDC.FLOAT32, (24, nlat, nlon))
        s.dim(0).setname("gmt_hr_index")
        s.dim(1).setname("latitude")
        s.dim(2).setname("longitude")
    else:
        data = rng.uniform(150.0, 300.0, size=(nlat, nlon)).astype(np.float32)
        if fill_first_cell:
            data[0, 0] = _SYN_FILL
        s = f.create(varname, SDC.FLOAT32, (nlat, nlon))
        s.dim(0).setname("latitude")
        s.dim(1).setname("longitude")
    s[:] = data
    s.attr("_FillValue").set(SDC.FLOAT32, _SYN_FILL)
    s.attr("units").set(SDC.CHAR, "W m-2")
    s.endaccess()
    f.end()
    return path


def test_syn_reader_registered_and_grid_geometry() -> None:
    reader_cls = source_registry.get("ceres_syn1deg")
    reader = reader_cls()
    assert reader.name == "ceres_syn1deg"
    assert reader.geometry is DataGeometry.GRID


def test_syn_month_file_time_from_filename(tmp_path: Path) -> None:
    p = _write_syn_hdf4(tmp_path / "CER_SYN1deg-Month_Terra-Aqua-NOAA20_Edition4B_415412.202512")

    ds = CERESSYN1degReader().open([p], variables=["obs_all_toa_lw_reg"])

    assert ds.sizes["time"] == 1
    assert np.datetime64("2025-12-01") == ds["time"].values[0]
    assert set(ds["obs_all_toa_lw_reg"].dims) == {"time", "lat", "lon"}
    assert ds.attrs["geometry"] == "grid"


def test_syn_day_files_concat_and_sort(tmp_path: Path) -> None:
    # Written out of order; reader must sort by time.
    _write_syn_hdf4(tmp_path / "CER_SYN1deg-Day_Terra-Aqua-NOAA20_Edition4B_415412.20251202")
    _write_syn_hdf4(tmp_path / "CER_SYN1deg-Day_Terra-Aqua-NOAA20_Edition4B_415412.20251201")

    ds = CERESSYN1degReader().open(
        sorted(tmp_path.glob("*.2025120*")), variables=["obs_all_toa_lw_reg"]
    )

    assert ds.sizes["time"] == 2
    times = ds["time"].values
    assert times[0] == np.datetime64("2025-12-01") and times[1] == np.datetime64("2025-12-02")


def test_syn_hourly_file_expands_24_steps(tmp_path: Path) -> None:
    p = _write_syn_hdf4(
        tmp_path / "CER_SYN1deg-1Hour_Terra-Aqua-NOAA20_Edition4B_415412.20251229",
        hourly=True,
    )

    ds = CERESSYN1degReader().open([p], variables=["obs_all_toa_lw_reg"])

    assert ds.sizes["time"] == 24
    assert ds["time"].values[0] == np.datetime64("2025-12-29T00:00")
    assert ds["time"].values[-1] == np.datetime64("2025-12-29T23:00")


def test_syn_latitude_flipped_ascending_with_data(tmp_path: Path) -> None:
    p = _write_syn_hdf4(
        tmp_path / "CER_SYN1deg-Month_Terra-Aqua-NOAA20_Edition4B_415412.202512"
    )
    # Read the raw row that sits at descending-lat index 0 (lat=+89.5).
    SD, SDC = pyhdf_SD.SD, pyhdf_SD.SDC
    f = SD(str(p), SDC.READ)
    north_row = np.array(f.select("obs_all_toa_lw_reg").get()[0, :], dtype=np.float64)
    f.end()

    ds = CERESSYN1degReader().open([p], variables=["obs_all_toa_lw_reg"])

    lat = ds["lat"].values
    assert np.all(np.diff(lat) > 0)  # ascending after standardization
    got = ds["obs_all_toa_lw_reg"].sel(lat=89.5).isel(time=0).values
    np.testing.assert_allclose(got, north_row)  # data moved with its coord


def test_syn_fill_value_masked(tmp_path: Path) -> None:
    p = _write_syn_hdf4(
        tmp_path / "CER_SYN1deg-Month_Terra-Aqua-NOAA20_Edition4B_415412.202511",
        fill_first_cell=True,
    )

    ds = CERESSYN1degReader().open([p], variables=["obs_all_toa_lw_reg"])

    da = ds["obs_all_toa_lw_reg"].isel(time=0)
    assert bool(np.isnan(da.sel(lat=89.5, lon=-179.5)))  # filled cell -> NaN
    assert int(np.isnan(da).sum()) == 1


def test_syn_unparseable_filename_raises(tmp_path: Path) -> None:
    p = _write_syn_hdf4(tmp_path / "not_a_ceres_name.hdf")

    with pytest.raises(ValueError, match="filename"):
        CERESSYN1degReader().open([p], variables=["obs_all_toa_lw_reg"])
```

- [ ] **Step 2.2: Run tests to verify they fail**

```bash
HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/test_ceres_l3_readers.py -v
```

Expected: ImportError on `CERESSYN1degReader`; the 8 existing EBAF tests still pass if you temporarily comment the import — don't; just confirm the failure is the missing class.

- [ ] **Step 2.3: Implement the reader**

Append to `davinci_monet/observations/satellite/ceres_l3.py` (extend the module docstring's first line mention of Phase 2 accordingly — change "(EBAF; SYN1deg arrives in Phase 2)" to "(EBAF and SYN1deg)" and add a SYN1deg paragraph):

Module docstring addition (append after the EBAF paragraph):

```
SYN1deg ships as HDF4 — one file per month, day, or day-of-hours, with 2-D
``(latitude, longitude)`` SDS (3-D ``(gmt_hr_index, latitude, longitude)``
for hourly) and explicit 1-D coordinate SDS (lat descending). Timestamps
come from the filename tail (``.YYYYMM`` or ``.YYYYMMDD``); hourly files
expand to 24 steps. Fill/valid_range/scale handling uses the shared
``apply_hdf4_scale``.
```

Code (new imports at top: `re`, `datetime`, `numpy as np`, `apply_hdf4_scale` added to the reader_utils import list):

```python
_SYN_DATE_RE = re.compile(r"\.(\d{6}|\d{8})$")


def _syn_time_from_filename(path: Path) -> "np.datetime64":
    """Parse the date stamp from a SYN1deg filename tail.

    ``...Edition4B_415412.202512`` -> 2025-12-01;
    ``...Edition4B_415412.20251229`` -> 2025-12-29.
    """
    m = _SYN_DATE_RE.search(path.name)
    if m is None:
        raise ValueError(
            f"Cannot parse a SYN1deg date from filename {path.name!r}: expected a "
            "'.YYYYMM' or '.YYYYMMDD' tail"
        )
    stamp = m.group(1)
    if len(stamp) == 6:
        return np.datetime64(f"{stamp[:4]}-{stamp[4:6]}-01")
    return np.datetime64(f"{stamp[:4]}-{stamp[4:6]}-{stamp[6:8]}")


@source_registry.register("ceres_syn1deg")
class CERESSYN1degReader:
    """Reader for CERES SYN1deg HDF4 files (monthly, daily, or hourly)."""

    @property
    def name(self) -> str:
        """Return reader name."""
        return "ceres_syn1deg"

    @property
    def geometry(self) -> DataGeometry:
        """SYN1deg is gridded."""
        return DataGeometry.GRID

    def open(
        self,
        file_paths: Sequence[str | Path],
        variables: Sequence[str] | None = None,
        **kwargs: Any,
    ) -> xr.Dataset:
        """Open SYN1deg HDF4 file(s) and assemble (time, lat, lon).

        Parameters
        ----------
        file_paths
            Paths to SYN1deg HDF4 files (resource-fork ``._*`` sidecars are
            ignored). Monthly/daily/hourly cadences may not be mixed in one
            open.
        variables
            Native SYN1deg SDS names (e.g. ``obs_all_toa_lw_reg``). If None,
            loads all non-coordinate SDS.
        **kwargs
            Unused; accepted for protocol compatibility.

        Returns
        -------
        xr.Dataset
            Standardized dataset with GRID geometry tagged.
        """
        real = [Path(f) for f in file_paths if not Path(f).name.startswith("._")]
        file_list = validate_file_list(real, source_label="CERES SYN1deg")

        per_file = [self._open_one(path, variables) for path in file_list]
        ds = per_file[0] if len(per_file) == 1 else xr.concat(per_file, dim="time")
        ds = ds.sortby("time")
        if "lat" in ds.coords and ds["lat"].values[0] > ds["lat"].values[-1]:
            ds = ds.sortby("lat")
        return set_geometry_attr(ds, DataGeometry.GRID)

    def _open_one(self, path: Path, variables: Sequence[str] | None) -> xr.Dataset:
        """Read one SYN1deg HDF4 file into an in-memory (time, lat, lon) Dataset."""
        try:
            from pyhdf.SD import SD, SDC
        except ImportError as exc:  # pragma: no cover - environment-dependent
            raise ImportError(
                "pyhdf is required to read CERES SYN1deg HDF4 files "
                "(conda install pyhdf)"
            ) from exc

        base_time = _syn_time_from_filename(path)
        hdf = SD(str(path), SDC.READ)
        try:
            available = hdf.datasets()
            names = [v for v in (variables or available) if v in available]
            names = [n for n in names if n not in ("latitude", "longitude")]
            lat = np.asarray(hdf.select("latitude").get(), dtype=np.float64)
            lon = np.asarray(hdf.select("longitude").get(), dtype=np.float64)

            data_vars: dict[str, Any] = {}
            hourly = False
            for name in names:
                sds = hdf.select(name)
                attrs = sds.attributes()
                values = apply_hdf4_scale(np.asarray(sds.get()), attrs)
                sds.endaccess()
                keep_attrs = {k: attrs[k] for k in ("units", "long_name") if k in attrs}
                if values.ndim == 3:
                    hourly = True
                    data_vars[name] = (("time", "lat", "lon"), values, keep_attrs)
                else:
                    data_vars[name] = (("time", "lat", "lon"), values[None, ...], keep_attrs)
        finally:
            hdf.end()

        if hourly:
            times = base_time + np.arange(24) * np.timedelta64(1, "h")
        else:
            times = np.array([base_time])
        return xr.Dataset(
            data_vars, coords={"time": times, "lat": lat, "lon": lon}
        )
```

CORRECTION (Task 2 review): real files MIX ranks — month/day files carry 1-D zonal and non-hourly 3-D cloud-layer SDS alongside the regional 2-D fields, and hour files carry 4-D SDS. The implemented reader therefore classifies by SDS dimension names — `("latitude","longitude")` → regional, `("gmt_hr_index","latitude","longitude")` → hourly — skipping other signatures on a `variables=None` scan and raising a clear ValueError when such a variable is explicitly requested. Requested-but-missing variables also raise. Time values for hourly files are hour starts (00:00..23:00 GMT).

Registration: add `CERESSYN1degReader` to the existing ceres_l3 import lines and `__all__` entries in BOTH `davinci_monet/observations/satellite/__init__.py` and `davinci_monet/observations/__init__.py` (one-line extensions of the imports added in Phase 1).

- [ ] **Step 2.4: Verify**

```bash
HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/test_ceres_l3_readers.py -v
```

Expected: 15 passed (8 EBAF + 7 SYN1deg).

```bash
mypy davinci_monet/observations/satellite/ceres_l3.py
black --check davinci_monet/observations/satellite/ceres_l3.py davinci_monet/tests/test_ceres_l3_readers.py \
              davinci_monet/observations/satellite/__init__.py davinci_monet/observations/__init__.py
isort --check davinci_monet/observations/satellite/ceres_l3.py davinci_monet/tests/test_ceres_l3_readers.py \
              davinci_monet/observations/satellite/__init__.py davinci_monet/observations/__init__.py
```

- [ ] **Step 2.5: Commit**

```bash
git add davinci_monet/observations/satellite/ceres_l3.py davinci_monet/tests/test_ceres_l3_readers.py \
        davinci_monet/observations/satellite/__init__.py davinci_monet/observations/__init__.py
git commit -m "feat(obs): CERES SYN1deg L3 HDF4 reader (type: ceres_syn1deg)"
```

---

### Task 3: Pipeline integration + opt-in real smokes

**Files:**
- Modify: `davinci_monet/tests/integration/test_ceres_readers_pipeline.py` (append)

- [ ] **Step 3.1: Append the pipeline test**

Append (reuse the module's existing `_monthly_grid` helper and env-gate constants; the synthetic-HDF4 writer is imported from the unit test module):

```python
def test_ceres_syn1deg_pipeline(tmp_path: Path) -> None:
    from davinci_monet.tests.test_ceres_l3_readers import _write_syn_hdf4

    s_dir = tmp_path / "syn"
    m_dir = tmp_path / "model"
    s_dir.mkdir()
    m_dir.mkdir()
    for stamp in ("202510", "202511", "202512"):
        _write_syn_hdf4(
            s_dir / f"CER_SYN1deg-Month_Terra-Aqua-NOAA20_Edition4B_415412.{stamp}",
            nlat=6,
            nlon=8,
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
                "type": "ceres_syn1deg",
                "role": "obs",
                "files": str(s_dir / "*.2025*"),
                "variables": {"obs_all_toa_lw_reg": {"units": "W m-2"}},
            },
            "model": {
                "type": "generic",
                "role": "model",
                "files": str(m_dir / "*.nc"),
                "variables": {"OLR": {"units": "W m-2"}},
            },
        },
        "pairs": {
            "model_vs_syn_olr": {
                "sources": ["model", "ceres"],
                "reference": "ceres",
                "variables": {"model": "OLR", "ceres": "obs_all_toa_lw_reg"},
            }
        },
        "plots": {
            "sc": {"type": "scatter", "pairs": ["model_vs_syn_olr"], "title": "OLR"},
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
    csvs = list(out_dir.rglob("statistics_summary.csv"))
    assert csvs, "expected a stats CSV"
    stats = pd.read_csv(csvs[0])
    n_col = next(c for c in stats.columns if c.strip().upper() == "N")
    # 3 months x 6 lats x 8 lons, minus nothing (no fill cells in synthetic)
    assert int(stats[n_col].iloc[0]) == 144, f"expected N=144, got\n{stats}"
```

Note: the SYN synthetic writer produces lat 89.5→−89.5 over `nlat` points and lon −179.5→179.5 over `nlon` — different centers than the model's `_monthly_grid` (−87.5..87.5 / −175..175). GRID-GRID nearest interpolation onto the ceres (reference) grid handles non-coincident centers; out-of-bounds fills would shrink N below 144, so the N assertion also pins that the model grid covers the obs grid. If N comes out short because `interp` extrapolation NaNs edge rows (lat ±89.5 outside model's ±87.5), adjust the SYNTHETIC MODEL to cover the obs domain — change `_monthly_grid`'s lat to `np.linspace(-89.5, 89.5, 6)` and lon to `np.linspace(-179.5, 179.5, 8)` via new keyword args (keep the EBAF test's call sites passing the old values or update both tests consistently and re-verify the EBAF N=144 still holds). Do NOT weaken the assertion.

- [ ] **Step 3.2: Append the opt-in real smokes**

Append (mirroring the file's existing env-gate from the Phase 1 amendment — reuse its `_RUN_REAL`-style constant and reason wording):

```python
_IO_SYN_MONTH = Path("/Volumes/Io/CERES/SYN1deg/month")


@pytest.mark.skipif(
    not (_RUN_REAL and _IO_SYN_MONTH.is_dir() and _IO_EBAF.is_dir()),
    reason="real-data smoke is opt-in (env var) and needs /Volumes/Io",
)
def test_real_syn1deg_zonal_means_correlate_with_ebaf() -> None:
    """Smoke: SYN1deg 2025-12 zonal-mean OLR must track EBAF's (lat axis check)."""
    from davinci_monet.observations.satellite.ceres_l3 import (
        CERESEBAFReader,
        CERESSYN1degReader,
    )

    syn_files = sorted(_IO_SYN_MONTH.glob("CER_SYN1deg-Month_*.202512"))
    ebaf_files = sorted(
        f for f in _IO_EBAF.glob("CERES_EBAF_*.nc") if not f.name.startswith("._")
    )
    if not syn_files or not ebaf_files:
        pytest.skip("staged SYN1deg/EBAF samples not found")

    syn = CERESSYN1degReader().open([syn_files[0]], variables=["obs_all_toa_lw_reg"])
    ebaf = CERESEBAFReader().open([ebaf_files[0]], variables=["toa_lw_all_mon"])

    syn_zonal = syn["obs_all_toa_lw_reg"].isel(time=0).mean("lon")
    ebaf_zonal = ebaf["toa_lw_all_mon"].sel(time="2025-12").squeeze().mean("lon")

    r = float(np.corrcoef(syn_zonal.values, ebaf_zonal.values)[0, 1])
    assert r > 0.9, f"SYN vs EBAF zonal correlation {r:.3f} — latitude axis suspect"
    assert syn.sizes["time"] == 1 and set(syn["obs_all_toa_lw_reg"].dims) == {
        "time",
        "lat",
        "lon",
    }
```

- [ ] **Step 3.3: Verify**

```bash
HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/integration/test_ceres_readers_pipeline.py -v
```
Expected (env var unset): pipeline tests pass, smokes skipped.

With the env-gate variable set (one-off):
```bash
<ENVVAR>=1 HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/integration/test_ceres_readers_pipeline.py -v
```
Expected: all pass including both smokes (run this in isolation, NOT the full suite).

- [ ] **Step 3.4: Full gates**

```bash
HDF5_USE_FILE_LOCKING=FALSE python -m pytest
mypy davinci_monet
black --check davinci_monet && isort --check davinci_monet
```
Expected: full suite green (env var unset → smokes skip), gates clean.

- [ ] **Step 3.5: Commit**

```bash
git add davinci_monet/tests/integration/test_ceres_readers_pipeline.py
git commit -m "test(integration): SYN1deg pipeline + opt-in SYN-vs-EBAF zonal smoke"
```
