# CERES SSF Reader (Phase 3) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A `type: ceres_ssf` SWATH reader for CERES L2 footprint files — HDF4 (Terra/Aqua Edition4A) and netCDF (NOAA-20 Edition1C) — emitting a 1-D footprint stream that pairs through the existing `SwathGridStrategy`, validated end-to-end and against the staged Io samples.

**Architecture:** New module `observations/satellite/ceres_ssf.py`. A canonical variable catalog (`SSF_CATALOG`) maps config names (e.g. `toa_lw_up`) to the per-format sources (HDF4 SDS name with spaces / netCDF group+var). Names not in the catalog are treated as raw source names (HDF4 SDS, or `"Group/var"` for netCDF) — the escape hatch. Per file, format is sniffed by magic bytes (HDF4 `\x0e\x03\x13\x01`; HDF5/netCDF-4 `\x89HDF`); mixing formats in one open is rejected. Both paths produce dims `(time,)` with `lat`/`lon` coords on the same dim (footprints with invalid time/lat/lon are dropped), data vars fill-masked to NaN via the shared `apply_hdf4_scale` (HDF4) or xarray's native decoding (netCDF). Multi-granule opens concat and sort by time. Geometry attr `swath` routes pairing to `SwathGridStrategy`, which flattens and bins by lat/lon/time values — verified to need no 2-D dims.

**Tech Stack:** pyhdf (lazy import), xarray/netCDF4 (grouped reads), numpy, pytest, mypy/black/isort in the `davinci` conda env.

**Spec:** `docs/superpowers/specs/2026-06-12-ceres-readers-design.md` (Phase 3 scope)

**Repo rules:** `davinci` env; pytest with `HDF5_USE_FILE_LOCKING=FALSE`; commits per task (standing approval); real-data smokes opt-in via `CERES_DATA` env var (never auto-on-mount).

**File facts (verified against staged samples 2026-06-12):**
- HDF4 Ed4A (`CER_SSF_Terra-FM1-MODIS_Edition4A_*.YYYYMMDDHH`, no suffix): 221 1-D SDS on dim `Footprints` (~67k/hr). `Time of observation` is float64 **Julian Date** (units `day`, fill 1.798e308, valid_range [2440000, 2480000]); `Colatitude of CERES FOV at surface` 0–180 (fill 3.4028e38); `Longitude of CERES FOV at surface` 0–360. Flux SDS (e.g. `CERES LW TOA flux - upwards`) carry fill 3.4028e38 **present in the data** and valid_range [0, 500].
- netCDF Ed1C (`CER_SSF_NOAA20-FM6-VIIRS_Edition1C_*.nc`): ~15 groups on dim `Footprints` (~105k/hr). `Time_and_Position/time` (days since 1970-01-01, auto-decoded), `Time_and_Position/instrument_fov_latitude` / `instrument_fov_longitude`; fluxes in `TOA_and_Surface_Fluxes/`.
- Julian↔epoch: `np.datetime64("1970-01-01")` is JD 2440587.5; `2461131.5` = 2026-04-01T00:00.

---

### Task 1: Catalog + HDF4 path + unit tests

**Files:**
- Create: `davinci_monet/observations/satellite/ceres_ssf.py`
- Modify: `davinci_monet/observations/satellite/__init__.py`, `davinci_monet/observations/__init__.py` (registration imports + `__all__`, alongside the CERES L3 entries)
- Test: `davinci_monet/tests/test_ceres_ssf_reader.py` (new)

- [ ] **Step 1.1: Write the failing tests**

Create `davinci_monet/tests/test_ceres_ssf_reader.py`:

```python
"""Unit tests for the CERES SSF (L2 footprint) reader."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from davinci_monet.core.protocols import DataGeometry
from davinci_monet.core.registry import source_registry

pyhdf_SD = pytest.importorskip("pyhdf.SD", reason="pyhdf required for SSF HDF4 tests")

from davinci_monet.observations.satellite.ceres_ssf import (  # noqa: E402
    SSF_CATALOG,
    CERESSSFReader,
)

_FILL32 = 3.4028234663852886e38
_JD_EPOCH = 2440587.5  # Julian Date of 1970-01-01T00


def _jd(iso: str) -> float:
    """ISO timestamp -> Julian Date (float days)."""
    ns = (np.datetime64(iso) - np.datetime64("1970-01-01T00:00:00")) / np.timedelta64(1, "s")
    return _JD_EPOCH + float(ns) / 86400.0


def _write_ssf_hdf4(
    path: Path,
    n: int = 6,
    base_iso: str = "2026-04-01T00:30:00",
    flux_name: str = "CERES LW TOA flux - upwards",
    fill_flux_idx: int | None = None,
    fill_coord_idx: int | None = None,
) -> Path:
    """Write a minimal SSF-like HDF4 file: 1-D footprint SDS with real names."""
    SD, SDC = pyhdf_SD.SD, pyhdf_SD.SDC
    times = np.array([_jd(base_iso) + i * (10.0 / 86400.0) for i in range(n)])  # 10 s apart
    colat = np.linspace(60.0, 120.0, n).astype(np.float32)  # lat +30 .. -30
    lon = np.linspace(10.0, 350.0, n).astype(np.float32)  # crosses the 180 wrap
    flux = np.linspace(150.0, 300.0, n).astype(np.float32)
    if fill_flux_idx is not None:
        flux[fill_flux_idx] = _FILL32
    if fill_coord_idx is not None:
        colat[fill_coord_idx] = _FILL32

    f = SD(str(path), SDC.WRITE | SDC.CREATE)

    def _sds(name: str, data: np.ndarray, typ: int, fill: float, vr: list[float]) -> None:
        s = f.create(name, typ, n)
        s.dim(0).setname("Footprints")
        s[:] = data
        s.attr("_FillValue").set(typ, fill)
        s.attr("valid_range").set(typ, vr)
        s.endaccess()

    _sds("Time of observation", times, SDC.FLOAT64, 1.7976931348623157e308, [2440000.0, 2480000.0])
    _sds("Colatitude of CERES FOV at surface", colat, SDC.FLOAT32, _FILL32, [0.0, 180.0])
    _sds("Longitude of CERES FOV at surface", lon, SDC.FLOAT32, _FILL32, [0.0, 360.0])
    _sds(flux_name, flux, SDC.FLOAT32, _FILL32, [0.0, 500.0])
    f.end()
    return path


def test_reader_registered_and_swath_geometry() -> None:
    reader_cls = source_registry.get("ceres_ssf")
    reader = reader_cls()
    assert reader.name == "ceres_ssf"
    assert reader.geometry is DataGeometry.SWATH


def test_catalog_covers_spec_canonical_set() -> None:
    assert {
        "toa_sw_up",
        "toa_lw_up",
        "toa_solar_in",
        "sfc_sw_down",
        "sfc_sw_down_clr",
        "sfc_lw_down",
        "sfc_lw_down_clr",
        "sfc_sw_net",
        "sfc_lw_net",
    } == set(SSF_CATALOG)


def test_hdf4_canonical_open_and_coords(tmp_path: Path) -> None:
    p = _write_ssf_hdf4(tmp_path / "CER_SSF_Terra-FM1-MODIS_Edition4A_410406.2026040100")

    ds = CERESSSFReader().open([p], variables=["toa_lw_up"])

    assert ds.attrs["geometry"] == "swath"
    assert set(ds.data_vars) == {"toa_lw_up"}
    assert ds["toa_lw_up"].dims == ("time",)
    assert ds.sizes["time"] == 6
    # colat 60..120 -> lat +30..-30
    np.testing.assert_allclose(ds["lat"].values[[0, -1]], [30.0, -30.0])
    # lon 0-360 -> wrapped to [-180, 180): 350 -> -10
    assert float(ds["lon"].values[-1]) == pytest.approx(-10.0)
    assert float(ds["lon"].values[0]) == pytest.approx(10.0)
    # Julian time decoded: first footprint at base time (10 s spacing after)
    assert ds["time"].values[0] == np.datetime64("2026-04-01T00:30:00")
    assert ds["time"].values[1] - ds["time"].values[0] == np.timedelta64(10, "s")


def test_hdf4_flux_fill_masked_but_footprint_kept(tmp_path: Path) -> None:
    p = _write_ssf_hdf4(
        tmp_path / "CER_SSF_Terra-FM1-MODIS_Edition4A_410406.2026040100",
        fill_flux_idx=2,
    )

    ds = CERESSSFReader().open([p], variables=["toa_lw_up"])

    assert ds.sizes["time"] == 6  # footprint kept; value masked
    assert bool(np.isnan(ds["toa_lw_up"].values[2]))
    assert int(np.isnan(ds["toa_lw_up"].values).sum()) == 1


def test_hdf4_invalid_coord_footprint_dropped(tmp_path: Path) -> None:
    p = _write_ssf_hdf4(
        tmp_path / "CER_SSF_Terra-FM1-MODIS_Edition4A_410406.2026040100",
        fill_coord_idx=1,
    )

    ds = CERESSSFReader().open([p], variables=["toa_lw_up"])

    assert ds.sizes["time"] == 5  # footprint without a valid position is unusable
    assert not np.isnan(ds["lat"].values).any()


def test_hdf4_multigranule_concat_sorted(tmp_path: Path) -> None:
    _write_ssf_hdf4(
        tmp_path / "CER_SSF_Terra-FM1-MODIS_Edition4A_410406.2026040101",
        base_iso="2026-04-01T01:30:00",
    )
    _write_ssf_hdf4(
        tmp_path / "CER_SSF_Terra-FM1-MODIS_Edition4A_410406.2026040100",
        base_iso="2026-04-01T00:30:00",
    )

    ds = CERESSSFReader().open(sorted(tmp_path.glob("CER_SSF_*")), variables=["toa_lw_up"])

    assert ds.sizes["time"] == 12
    t = ds["time"].values
    assert (np.diff(t) > np.timedelta64(0, "s")).all()


def test_hdf4_raw_source_name_escape(tmp_path: Path) -> None:
    p = _write_ssf_hdf4(
        tmp_path / "CER_SSF_Terra-FM1-MODIS_Edition4A_410406.2026040100",
        flux_name="CERES SW TOA flux - upwards",
    )

    ds = CERESSSFReader().open([p], variables=["CERES SW TOA flux - upwards"])

    assert "CERES SW TOA flux - upwards" in ds.data_vars


def test_hdf4_missing_variable_raises(tmp_path: Path) -> None:
    p = _write_ssf_hdf4(tmp_path / "CER_SSF_Terra-FM1-MODIS_Edition4A_410406.2026040100")

    with pytest.raises(ValueError, match="not found"):
        CERESSSFReader().open([p], variables=["toa_solar_in"])
```

- [ ] **Step 1.2: Run tests to verify they fail**

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate davinci
HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/test_ceres_ssf_reader.py -v
```
Expected: ImportError — `ceres_ssf` module missing.

- [ ] **Step 1.3: Implement**

Create `davinci_monet/observations/satellite/ceres_ssf.py`:

```python
"""CERES SSF (Single Scanner Footprint) L2 reader.

SSF files are flat 1-D footprint streams (~10^5 footprints/hour) in two
editions: HDF4 (Terra/Aqua Edition4A — SDS names with spaces, colatitude,
0-360 longitude, Julian-Date time) and netCDF-4 (NOAA-20 Edition1C —
grouped variables, true latitude, epoch time). This reader standardizes
both to dims ``(time,)`` with ``lat``/``lon`` coords, canonical variable
names via ``SSF_CATALOG`` (raw source names pass through as an escape
hatch; for netCDF use ``"Group/var"``), fill-masked values, and SWATH
geometry. Pairing flows through ``SwathGridStrategy``, which flattens and
bins footprints by lat/lon/time values — no 2-D swath dims required.

Footprints with invalid time/lat/lon are dropped (they cannot be paired);
invalid data values become NaN but keep their footprint.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import xarray as xr

from davinci_monet.core.protocols import DataGeometry
from davinci_monet.core.registry import source_registry
from davinci_monet.io.reader_utils import (
    apply_hdf4_scale,
    set_geometry_attr,
    validate_file_list,
)

_JD_EPOCH = 2440587.5  # Julian Date of 1970-01-01T00

# HDF4 coordinate SDS names (Edition4A).
_H4_TIME = "Time of observation"
_H4_COLAT = "Colatitude of CERES FOV at surface"
_H4_LON = "Longitude of CERES FOV at surface"

# netCDF coordinate variables (Edition1C), all in Time_and_Position.
_NC_POSITION_GROUP = "Time_and_Position"
_NC_TIME = "time"
_NC_LAT = "instrument_fov_latitude"
_NC_LON = "instrument_fov_longitude"


@dataclass(frozen=True)
class SSFVariable:
    """Per-format source names for one canonical SSF variable."""

    hdf4_sds: str
    nc_group: str
    nc_var: str


# Canonical config name -> per-edition source. Surface fluxes standardize on
# Model B (the only parameterization with all-sky + clear-sky in both
# editions). Names outside this catalog are treated as raw source names.
SSF_CATALOG: dict[str, SSFVariable] = {
    "toa_sw_up": SSFVariable(
        "CERES SW TOA flux - upwards", "TOA_and_Surface_Fluxes", "toa_shortwave_flux"
    ),
    "toa_lw_up": SSFVariable(
        "CERES LW TOA flux - upwards", "TOA_and_Surface_Fluxes", "toa_longwave_flux"
    ),
    "toa_solar_in": SSFVariable(
        "TOA Incoming Solar Radiation",
        "TOA_and_Surface_Fluxes",
        "toa_incoming_solar_radiation",
    ),
    "sfc_sw_down": SSFVariable(
        "CERES downward SW surface flux - Model B",
        "TOA_and_Surface_Fluxes",
        "model_b_surface_shortwave_downward_flux",
    ),
    "sfc_sw_down_clr": SSFVariable(
        "CERES downward SW surface flux - Model B, clearsky",
        "TOA_and_Surface_Fluxes",
        "model_b_clearsky_surface_shortwave_downward_flux",
    ),
    "sfc_lw_down": SSFVariable(
        "CERES downward LW surface flux - Model B",
        "TOA_and_Surface_Fluxes",
        "model_b_surface_longwave_downward_flux",
    ),
    "sfc_lw_down_clr": SSFVariable(
        "CERES downward LW surface flux - Model B, clearsky",
        "TOA_and_Surface_Fluxes",
        "model_b_clearsky_surface_longwave_downward_flux",
    ),
    "sfc_sw_net": SSFVariable(
        "CERES net SW surface flux - Model B",
        "TOA_and_Surface_Fluxes",
        "model_b_surface_shortwave_net_flux",
    ),
    "sfc_lw_net": SSFVariable(
        "CERES net LW surface flux - Model B",
        "TOA_and_Surface_Fluxes",
        "model_b_surface_longwave_net_flux",
    ),
}

_HDF4_MAGIC = b"\x0e\x03\x13\x01"
_HDF5_MAGIC = b"\x89HDF"


def _sniff_format(path: Path) -> str:
    """Return ``"hdf4"`` or ``"netcdf"`` from the file magic bytes."""
    with open(path, "rb") as fh:
        magic = fh.read(4)
    if magic == _HDF4_MAGIC:
        return "hdf4"
    if magic == _HDF5_MAGIC or magic[:3] == b"CDF":
        return "netcdf"
    raise ValueError(f"Unrecognized SSF file format for {path.name!r} (magic {magic!r})")


def _jd_to_datetime64(jd: "np.ndarray[Any, np.dtype[Any]]") -> "np.ndarray[Any, np.dtype[Any]]":
    """Julian Date (float days) -> datetime64[ns] (millisecond precision)."""
    ms = np.round((jd - _JD_EPOCH) * 86400.0 * 1e3)
    return np.datetime64("1970-01-01T00:00:00").astype("datetime64[ms]") + ms.astype(
        "timedelta64[ms]"
    )


def _wrap_lon(lon: "np.ndarray[Any, np.dtype[Any]]") -> "np.ndarray[Any, np.dtype[Any]]":
    """Wrap 0-360 longitudes to [-180, 180)."""
    return ((lon + 180.0) % 360.0) - 180.0


@source_registry.register("ceres_ssf")
class CERESSSFReader:
    """Reader for CERES SSF L2 footprints (HDF4 Ed4A and netCDF Ed1C)."""

    @property
    def name(self) -> str:
        """Return reader name."""
        return "ceres_ssf"

    @property
    def geometry(self) -> DataGeometry:
        """SSF footprints are swath data (binned to a grid for pairing)."""
        return DataGeometry.SWATH

    def open(
        self,
        file_paths: Sequence[str | Path],
        variables: Sequence[str] | None = None,
        **kwargs: Any,
    ) -> xr.Dataset:
        """Open SSF granule(s) and standardize to a (time,) footprint stream.

        Parameters
        ----------
        file_paths
            SSF granule paths (resource-fork ``._*`` sidecars are ignored).
            All granules in one open must share a format (HDF4 or netCDF).
        variables
            Canonical names from ``SSF_CATALOG`` (e.g. ``toa_lw_up``) or raw
            source names (HDF4 SDS name; ``"Group/var"`` for netCDF). If
            None, loads every catalog variable present in the files.
        **kwargs
            Unused; accepted for protocol compatibility.

        Returns
        -------
        xr.Dataset
            Footprint stream with ``lat``/``lon`` coords on the ``time`` dim
            and SWATH geometry tagged.
        """
        real = [Path(f) for f in file_paths if not Path(f).name.startswith("._")]
        file_list = validate_file_list(real, source_label="CERES SSF")

        formats = {_sniff_format(p) for p in file_list}
        if len(formats) > 1:
            raise ValueError(
                "CERES SSF open received mixed file formats (HDF4 and netCDF); "
                "open each edition separately"
            )
        fmt = formats.pop()

        opener = self._open_one_hdf4 if fmt == "hdf4" else self._open_one_netcdf
        per_file = [opener(path, variables) for path in file_list]
        ds = per_file[0] if len(per_file) == 1 else xr.concat(per_file, dim="time")
        ds = ds.sortby("time")
        return set_geometry_attr(ds, DataGeometry.SWATH)

    # -- HDF4 (Edition4A) ---------------------------------------------------

    def _open_one_hdf4(self, path: Path, variables: Sequence[str] | None) -> xr.Dataset:
        """Read one Edition4A HDF4 granule into a footprint Dataset."""
        try:
            from pyhdf.SD import SD, SDC
        except ImportError as exc:  # pragma: no cover - environment-dependent
            raise ImportError(
                "pyhdf is required to read CERES SSF HDF4 files (conda install pyhdf)"
            ) from exc

        hdf = SD(str(path), SDC.READ)
        try:
            available = hdf.datasets()

            def _read(sds_name: str) -> "np.ndarray[Any, np.dtype[Any]]":
                sds = hdf.select(sds_name)
                values = apply_hdf4_scale(np.asarray(sds.get()), sds.attributes())
                sds.endaccess()
                return values

            time = _jd_to_datetime64(_read(_H4_TIME))
            lat = 90.0 - _read(_H4_COLAT)
            lon = _wrap_lon(_read(_H4_LON))

            requested = self._resolve(variables, available, fmt="hdf4", where=path.name)
            data_vars = {name: ("time", _read(src)) for name, src in requested.items()}
        finally:
            hdf.end()

        return self._assemble(time, lat, lon, data_vars)

    # -- netCDF (Edition1C) ---------------------------------------------------

    def _open_one_netcdf(self, path: Path, variables: Sequence[str] | None) -> xr.Dataset:
        """Read one Edition1C netCDF granule into a footprint Dataset."""
        pos = xr.open_dataset(str(path), group=_NC_POSITION_GROUP)
        try:
            time = pos[_NC_TIME].values
            lat = np.asarray(pos[_NC_LAT].values, dtype=np.float64)
            lon = _wrap_lon(np.asarray(pos[_NC_LON].values, dtype=np.float64))
        finally:
            pos.close()

        # Group requested variables by their netCDF group to minimize opens.
        requested = self._resolve(variables, None, fmt="netcdf", where=path.name)
        by_group: dict[str, dict[str, str]] = {}
        for name, src in requested.items():
            group, _, var = src.partition("/")
            by_group.setdefault(group, {})[name] = var

        scan = variables is None  # catalog scan skips absentees; explicit requests raise
        data_vars: dict[str, Any] = {}
        for group, mapping in by_group.items():
            grp = xr.open_dataset(str(path), group=group)
            try:
                missing = [v for v in mapping.values() if v not in grp.data_vars]
                if missing and not scan:
                    raise ValueError(
                        f"SSF variable(s) not found in {path.name!r} group "
                        f"{group!r}: {missing!r}"
                    )
                for name, var in mapping.items():
                    if var in grp.data_vars:
                        data_vars[name] = (
                            "time",
                            np.asarray(grp[var].values, dtype=np.float64),
                        )
            finally:
                grp.close()
        if scan and not data_vars:
            raise ValueError(
                f"No catalog SSF variables found in {path.name!r}; "
                "request explicit source names"
            )

        return self._assemble(time, lat, lon, data_vars)

    # -- shared ----------------------------------------------------------------

    def _resolve(
        self,
        variables: Sequence[str] | None,
        available: dict[str, Any] | None,
        *,
        fmt: str,
        where: str,
    ) -> dict[str, str]:
        """Map requested names to per-format source names.

        For HDF4, ``available`` is the SDS inventory and resolution is
        validated against it here. For netCDF, ``available`` is None and
        membership is validated group-by-group by the caller.
        """
        if variables is None:
            names = list(SSF_CATALOG)
        else:
            names = list(variables)

        resolved: dict[str, str] = {}
        for name in names:
            entry = SSF_CATALOG.get(name)
            if fmt == "hdf4":
                src = entry.hdf4_sds if entry is not None else name
            else:
                src = f"{entry.nc_group}/{entry.nc_var}" if entry is not None else name
            resolved[name] = src

        if available is not None:
            missing = [n for n, s in resolved.items() if s not in available]
            if variables is None:
                resolved = {n: s for n, s in resolved.items() if s in available}
                if not resolved:
                    raise ValueError(
                        f"No catalog SSF variables found in {where!r}; "
                        "request explicit source names"
                    )
            elif missing:
                raise ValueError(f"SSF variable(s) not found in {where!r}: {missing!r}")
        elif variables is None:
            # netCDF catalog-scan: all catalog entries share one group; the
            # caller validates membership and raises if a name is absent.
            pass
        return resolved

    @staticmethod
    def _assemble(
        time: "np.ndarray[Any, np.dtype[Any]]",
        lat: "np.ndarray[Any, np.dtype[Any]]",
        lon: "np.ndarray[Any, np.dtype[Any]]",
        data_vars: dict[str, Any],
    ) -> xr.Dataset:
        """Build the footprint Dataset, dropping footprints without position."""
        time_ns = np.asarray(time, dtype="datetime64[ns]")
        valid = (
            ~np.isnat(time_ns)
            & ~np.isnan(np.asarray(lat, dtype=np.float64))
            & ~np.isnan(np.asarray(lon, dtype=np.float64))
        )
        ds = xr.Dataset(
            {name: (dims, vals[valid]) for name, (dims, vals) in data_vars.items()},
            coords={
                "time": time_ns[valid],
                "lat": ("time", np.asarray(lat, dtype=np.float64)[valid]),
                "lon": ("time", np.asarray(lon, dtype=np.float64)[valid]),
            },
        )
        return ds
```

Registration: extend the CERES imports in `davinci_monet/observations/satellite/__init__.py` (import `CERESSSFReader` from `ceres_ssf` with the `# noqa: F401  (registers "ceres_ssf")` comment, `__all__` entry) and `davinci_monet/observations/__init__.py` (import + `__all__`), matching the ceres_l3 entries added in Phases 1-2.

Note on `_assemble`: NaT time entries — `apply_hdf4_scale` turns time fills into NaN floats, and `_jd_to_datetime64` propagates NaN→NaT via the rounded float cast; verify with the synthetic fill test, and if NaN does not propagate to NaT cleanly, mask explicitly (`np.isnan(jd)` before conversion) — adjust minimally, keeping the dropped-footprint behavior.

- [ ] **Step 1.4: Verify**

```bash
HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/test_ceres_ssf_reader.py -v   # 8 passed
mypy davinci_monet/observations/satellite/ceres_ssf.py
black --check davinci_monet/observations/satellite/ceres_ssf.py davinci_monet/tests/test_ceres_ssf_reader.py \
              davinci_monet/observations/satellite/__init__.py davinci_monet/observations/__init__.py
isort --check davinci_monet/observations/satellite/ceres_ssf.py davinci_monet/tests/test_ceres_ssf_reader.py \
              davinci_monet/observations/satellite/__init__.py davinci_monet/observations/__init__.py
```

- [ ] **Step 1.5: Commit**

```bash
git add davinci_monet/observations/satellite/ceres_ssf.py davinci_monet/tests/test_ceres_ssf_reader.py \
        davinci_monet/observations/satellite/__init__.py davinci_monet/observations/__init__.py
git commit -m "feat(obs): CERES SSF L2 footprint reader, HDF4 path (type: ceres_ssf)"
```

---

### Task 2: netCDF (Edition1C) path parity tests

**Files:**
- Test: `davinci_monet/tests/test_ceres_ssf_reader.py` (append)
- Modify (only if tests expose gaps): `davinci_monet/observations/satellite/ceres_ssf.py`

- [ ] **Step 2.1: Write the tests**

Append to `davinci_monet/tests/test_ceres_ssf_reader.py`:

```python
# ---------------------------------------------------------------------------
# netCDF (Edition1C) path
# ---------------------------------------------------------------------------


def _write_ssf_netcdf(
    path: Path,
    n: int = 6,
    base_iso: str = "2026-04-01T00:30:00",
    nc_var: str = "toa_longwave_flux",
    fill_flux_idx: int | None = None,
    fill_coord_idx: int | None = None,
) -> Path:
    """Write a minimal Edition1C-like grouped netCDF granule."""
    base = np.datetime64(base_iso)
    times = base + np.arange(n) * np.timedelta64(10, "s")
    lat = np.linspace(30.0, -30.0, n)
    lon = np.linspace(10.0, 350.0, n)  # 0-360 in-file; reader wraps
    flux = np.linspace(150.0, 300.0, n)
    if fill_coord_idx is not None:
        lat[fill_coord_idx] = np.nan  # xarray-decoded fill arrives as NaN

    pos = xr.Dataset(
        {
            "time": ("Footprints", times),
            "instrument_fov_latitude": ("Footprints", lat),
            "instrument_fov_longitude": ("Footprints", lon),
        }
    )
    flux_da = xr.DataArray(flux, dims=("Footprints",))
    if fill_flux_idx is not None:
        flux_vals = flux.copy()
        flux_vals[fill_flux_idx] = np.nan
        flux_da = xr.DataArray(flux_vals, dims=("Footprints",))
    fluxes = xr.Dataset({nc_var: flux_da})

    pos.to_netcdf(path, group="Time_and_Position", mode="w")
    fluxes.to_netcdf(path, group="TOA_and_Surface_Fluxes", mode="a")
    return path


def test_netcdf_canonical_open_matches_hdf4_semantics(tmp_path: Path) -> None:
    p = _write_ssf_netcdf(tmp_path / "CER_SSF_NOAA20-FM6-VIIRS_Edition1C_103103.2026040100.nc")

    ds = CERESSSFReader().open([p], variables=["toa_lw_up"])

    assert ds.attrs["geometry"] == "swath"
    assert set(ds.data_vars) == {"toa_lw_up"}
    assert ds["toa_lw_up"].dims == ("time",)
    assert ds.sizes["time"] == 6
    np.testing.assert_allclose(ds["lat"].values[[0, -1]], [30.0, -30.0])
    assert float(ds["lon"].values[-1]) == pytest.approx(-10.0)
    assert ds["time"].values[0] == np.datetime64("2026-04-01T00:30:00")


def test_netcdf_invalid_coord_footprint_dropped(tmp_path: Path) -> None:
    p = _write_ssf_netcdf(
        tmp_path / "CER_SSF_NOAA20-FM6-VIIRS_Edition1C_103103.2026040100.nc",
        fill_coord_idx=1,
    )

    ds = CERESSSFReader().open([p], variables=["toa_lw_up"])

    assert ds.sizes["time"] == 5
    assert not np.isnan(ds["lat"].values).any()


def test_netcdf_group_path_escape(tmp_path: Path) -> None:
    p = _write_ssf_netcdf(
        tmp_path / "CER_SSF_NOAA20-FM6-VIIRS_Edition1C_103103.2026040100.nc",
        nc_var="toa_longwave_channel_flux",
    )

    ds = CERESSSFReader().open(
        [p], variables=["TOA_and_Surface_Fluxes/toa_longwave_channel_flux"]
    )

    assert "TOA_and_Surface_Fluxes/toa_longwave_channel_flux" in ds.data_vars


def test_netcdf_missing_variable_raises(tmp_path: Path) -> None:
    p = _write_ssf_netcdf(tmp_path / "CER_SSF_NOAA20-FM6-VIIRS_Edition1C_103103.2026040100.nc")

    with pytest.raises(ValueError, match="not found"):
        CERESSSFReader().open([p], variables=["toa_solar_in"])


def test_mixed_formats_rejected(tmp_path: Path) -> None:
    h4 = _write_ssf_hdf4(tmp_path / "CER_SSF_Terra-FM1-MODIS_Edition4A_410406.2026040100")
    nc = _write_ssf_netcdf(tmp_path / "CER_SSF_NOAA20-FM6-VIIRS_Edition1C_103103.2026040100.nc")

    with pytest.raises(ValueError, match="mixed"):
        CERESSSFReader().open([h4, nc], variables=["toa_lw_up"])
```

- [ ] **Step 2.2: Run and fix**

```bash
HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/test_ceres_ssf_reader.py -v
```
Expected: 13 passed. The netCDF path was implemented in Task 1; these tests prove parity. If a test exposes a gap (e.g. variables=None handling in netCDF, NaT propagation), fix `ceres_ssf.py` minimally and note it.

- [ ] **Step 2.3: Gates + commit**

```bash
mypy davinci_monet/observations/satellite/ceres_ssf.py
black --check davinci_monet/observations/satellite/ceres_ssf.py davinci_monet/tests/test_ceres_ssf_reader.py
isort --check davinci_monet/observations/satellite/ceres_ssf.py davinci_monet/tests/test_ceres_ssf_reader.py
git add davinci_monet/tests/test_ceres_ssf_reader.py davinci_monet/observations/satellite/ceres_ssf.py
git commit -m "test(obs): SSF netCDF Edition1C parity + mixed-format rejection"
```

---

### Task 3: Pipeline integration (SwathGridStrategy) + opt-in real smokes + full gates

**Files:**
- Modify: `davinci_monet/tests/integration/test_ceres_readers_pipeline.py` (append)

- [ ] **Step 3.1: Append the pipeline test**

The synthetic SSF granules place exactly one footprint per model grid cell per day so the binned N is deterministic. Append:

```python
def test_ceres_ssf_pipeline(tmp_path: Path) -> None:
    """SSF footprints -> SwathGridStrategy binning -> stats, via the pipeline."""
    from davinci_monet.tests.test_ceres_ssf_reader import _write_ssf_hdf4_grid

    s_dir = tmp_path / "ssf"
    m_dir = tmp_path / "model"
    s_dir.mkdir()
    m_dir.mkdir()
    lat_centers = np.linspace(-87.5, 87.5, 6)
    lon_centers = np.linspace(-175.0, 175.0, 8)
    for i, day in enumerate(("2025-10-01", "2025-11-01", "2025-12-01")):
        _write_ssf_hdf4_grid(
            s_dir / f"CER_SSF_Terra-FM1-MODIS_Edition4A_410406.20251{i}0100",
            lat_centers,
            lon_centers,
            base_iso=f"{day}T12:00:00",
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
                "type": "ceres_ssf",
                "role": "obs",
                "files": str(s_dir / "CER_SSF_*"),
                "variables": {"toa_lw_up": {"units": "W m-2"}},
            },
            "model": {
                "type": "generic",
                "role": "model",
                "files": str(m_dir / "*.nc"),
                "variables": {"OLR": {"units": "W m-2"}},
            },
        },
        "pairs": {
            "model_vs_ssf_olr": {
                "sources": ["model", "ceres"],
                "reference": "ceres",
                "variables": {"model": "OLR", "ceres": "toa_lw_up"},
            }
        },
        "plots": {
            "sc": {"type": "scatter", "pairs": ["model_vs_ssf_olr"], "title": "OLR"},
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
    # One footprint per model cell per granule-day: 6x8 cells x 3 days = 144
    # bins with exactly one obs each. Fewer means binning lost footprints.
    assert int(stats[n_col].iloc[0]) == 144, f"expected N=144, got\n{stats}"
```

And add the grid-writer helper to `davinci_monet/tests/test_ceres_ssf_reader.py` (next to `_write_ssf_hdf4`):

```python
def _write_ssf_hdf4_grid(
    path: Path,
    lat_centers: np.ndarray,
    lon_centers: np.ndarray,
    base_iso: str,
) -> Path:
    """Write an SSF-like HDF4 granule with one footprint per (lat, lon) center."""
    SD, SDC = pyhdf_SD.SD, pyhdf_SD.SDC
    lat2d, lon2d = np.meshgrid(lat_centers, lon_centers, indexing="ij")
    lat = lat2d.ravel().astype(np.float32)
    lon = lon2d.ravel().astype(np.float32)
    n = lat.size
    times = np.array([_jd(base_iso) + i * (1.0 / 86400.0) for i in range(n)])
    flux = np.linspace(150.0, 300.0, n).astype(np.float32)

    f = SD(str(path), SDC.WRITE | SDC.CREATE)

    def _sds(name: str, data: np.ndarray, typ: int, fill: float, vr: list[float]) -> None:
        s = f.create(name, typ, n)
        s.dim(0).setname("Footprints")
        s[:] = data
        s.attr("_FillValue").set(typ, fill)
        s.attr("valid_range").set(typ, vr)
        s.endaccess()

    _sds("Time of observation", times, SDC.FLOAT64, 1.7976931348623157e308, [2440000.0, 2480000.0])
    _sds(
        "Colatitude of CERES FOV at surface",
        (90.0 - lat).astype(np.float32),
        SDC.FLOAT32,
        _FILL32,
        [0.0, 180.0],
    )
    _sds(
        "Longitude of CERES FOV at surface",
        np.where(lon < 0, lon + 360.0, lon).astype(np.float32),
        SDC.FLOAT32,
        _FILL32,
        [0.0, 360.0],
    )
    _sds("CERES LW TOA flux - upwards", flux, SDC.FLOAT32, _FILL32, [0.0, 500.0])
    f.end()
    return path
```

Run:
```bash
HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/integration/test_ceres_readers_pipeline.py::test_ceres_ssf_pipeline -v
```

Debug notes if N != 144 (investigate, don't weaken):
- SwathGridStrategy bins with `grid_mode="match_model"` and `time_resolution="1D"` defaults; footprints sit at exact model cell centers and distinct days, so each (day, cell) bin gets exactly one footprint.
- If the binner's edge arithmetic assigns a center-sitting footprint to a neighbor cell, check `normalize_grid`/`edges_from_centers` in `davinci_monet/pairing/grid_binning.py` and nudge the synthetic footprints off-center by +0.1° in both axes instead.
- If the strategy needs kwargs plumbed from the pair config, check how the pairing stage forwards pair-level options before changing anything.

- [ ] **Step 3.2: Append the opt-in real smokes**

```python
_IO_SSF_TERRA = Path("/Volumes/Io/CERES/SSF/Terra-FM1")
_IO_SSF_N20 = Path("/Volumes/Io/CERES/SSF/NOAA20-FM6")


@pytest.mark.skipif(
    not (_RUN_REAL and _IO_SSF_TERRA.is_dir() and _IO_SSF_N20.is_dir()),
    reason="real-data smoke is opt-in (set CERES_DATA) and needs /Volumes/Io",
)
def test_real_ssf_granules_open_in_both_editions() -> None:
    """Smoke: one HDF4 (Terra) and one netCDF (NOAA-20) granule via the reader."""
    from davinci_monet.observations.satellite.ceres_ssf import CERESSSFReader

    h4 = sorted(f for f in _IO_SSF_TERRA.glob("CER_SSF_*") if not f.name.startswith("._"))
    nc = sorted(f for f in _IO_SSF_N20.glob("CER_SSF_*.nc") if not f.name.startswith("._"))
    if not h4 or not nc:
        pytest.skip("staged SSF granules not found")

    for path, edition in ((h4[0], "Edition4A"), (nc[0], "Edition1C")):
        ds = CERESSSFReader().open([path], variables=["toa_lw_up"])
        assert ds.attrs["geometry"] == "swath", edition
        assert ds.sizes["time"] > 10_000, edition  # a real granule has ~1e5 footprints
        lat = ds["lat"].values
        assert lat.min() >= -90.0 and lat.max() <= 90.0, edition
        lon = ds["lon"].values
        assert lon.min() >= -180.0 and lon.max() < 180.0, edition
        olr = ds["toa_lw_up"].values
        finite = olr[~np.isnan(olr)]
        assert finite.size > 0 and 50.0 < float(np.median(finite)) < 400.0, edition
        # Times must fall within the granule hour stamped in the filename
        # (filename ends .YYYYMMDDHH or .YYYYMMDDHH.nc).
        digits = "".join(ch for ch in path.name.split(".")[-2 if path.suffix == ".nc" else -1] if ch.isdigit())
        t0 = np.datetime64(f"{digits[:4]}-{digits[4:6]}-{digits[6:8]}T{digits[8:10]}:00")
        t = ds["time"].values
        assert (t >= t0).all() and (t <= t0 + np.timedelta64(1, "h")).all(), edition
```

(If the filename-digit extraction proves awkward in practice, simplify to asserting all times fall within 2026-04-01..2026-04-04 — the staged sample window — rather than the exact hour; note the change.)

- [ ] **Step 3.3: Verify**

```bash
HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/integration/test_ceres_readers_pipeline.py -v   # env unset: 3 passed, 3 skipped
CERES_DATA=/Volumes/Io/CERES HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/integration/test_ceres_readers_pipeline.py -v   # 6 passed (isolation only)
```

- [ ] **Step 3.4: Full gates**

```bash
HDF5_USE_FILE_LOCKING=FALSE python -m pytest
mypy davinci_monet
black --check davinci_monet && isort --check davinci_monet
```
Expected: full suite green (smokes skip), gates clean. Paste the summary line.

- [ ] **Step 3.5: Commit**

```bash
git add davinci_monet/tests/integration/test_ceres_readers_pipeline.py davinci_monet/tests/test_ceres_ssf_reader.py
git commit -m "test(integration): SSF pipeline via SwathGridStrategy + opt-in dual-edition smoke"
```
