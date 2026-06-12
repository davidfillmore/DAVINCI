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
    times = np.array(["2025-10-01", "2025-11-01", "2025-12-01"], dtype="datetime64[ns]")[:nt]
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

    ds = CERESEBAFReader().open(sorted(tmp_path.glob("ebaf_*.nc")), variables=["toa_lw_all_mon"])

    assert ds.sizes["time"] == 3


def test_open_ignores_resource_fork_sidecars(tmp_path: Path) -> None:
    path = _write(_ebaf_like(), tmp_path / "ebaf.nc")
    (tmp_path / "._ebaf.nc").write_bytes(b"\x00\x05\x16\x07junk")

    ds = CERESEBAFReader().open(sorted(tmp_path.glob("*ebaf.nc")), variables=["toa_lw_all_mon"])

    assert ds.sizes["time"] == 3


def test_longitude_already_normalized_passes_through(tmp_path: Path) -> None:
    src = _ebaf_like()
    src = src.assign_coords(lon=np.array([-179.5, -89.5, 0.5, 90.5]))
    path = _write(src, tmp_path / "ebaf.nc")

    ds = CERESEBAFReader().open([path], variables=["toa_lw_all_mon"])

    np.testing.assert_allclose(ds["lon"].values, [-179.5, -89.5, 0.5, 90.5])


def test_drop_unused_dims_handles_dim_without_data_vars(tmp_path: Path) -> None:
    # A raw file can carry a dim used by no data variable at all (select_
    # variables already covers the selection path; this pins the helper's own
    # case: an orphan dim surviving into the opened dataset).
    src = _ebaf_like()
    src = src.assign_coords(orphan=("orphan", [1, 2, 3]))
    path = _write(src, tmp_path / "ebaf.nc")

    ds = CERESEBAFReader().open([path])

    assert "orphan" not in ds.dims
