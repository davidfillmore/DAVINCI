"""Unit tests for the CERES L3 readers (EBAF in Phase 1)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from davinci_monet.core.exceptions import DataFormatError
from davinci_monet.core.protocols import DataGeometry
from davinci_monet.core.registry import source_registry
from davinci_monet.datasets.satellite.ceres_l3 import CERESEBAFReader


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


def test_ebaf_missing_requested_variable_raises(tmp_path: Path) -> None:
    path = _write(_ebaf_like(), tmp_path / "CERES_EBAF_Edition4.2.1_200003-202512.nc")

    with pytest.raises(DataFormatError, match="EBAF variable"):
        CERESEBAFReader().open([path], variables=["not_a_ceres_var"])


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


# ---------------------------------------------------------------------------
# SYN1deg (Phase 2)
# ---------------------------------------------------------------------------

from davinci_monet.datasets.satellite.ceres_l3 import CERESSYN1degReader  # noqa: E402

_SYN_FILL = 3.4028234663852886e38


def _pyhdf_sd():
    return pytest.importorskip("pyhdf.SD", reason="pyhdf required for SYN1deg HDF4 tests")


def _write_syn_hdf4(
    path: Path,
    varname: str = "geometry_all_toa_lw_reg",
    nlat: int = 4,
    nlon: int = 4,
    hourly: bool = False,
    fill_first_cell: bool = False,
    extra_sds: str | None = None,
) -> Path:
    """Write a minimal SYN1deg-like HDF4 file via pyhdf.

    Mirrors the real layout: 1-D ``latitude`` (descending 89.5->-89.5 style)
    and ``longitude`` (ascending +-180) SDS plus 2-D (lat, lon) or 3-D
    (gmt_hr_index, lat, lon) data SDS with fill attrs. ``extra_sds`` adds a
    non-regional SDS like real files carry: ``"zonal"`` adds a 1-D
    ``geometry_all_toa_lw_zon`` on ("latitude",); ``"layered"`` adds a 3-D
    ``geometry_cld_amount_reg`` on ("cloud_layer", "latitude", "longitude").
    """
    pyhdf_SD = _pyhdf_sd()
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

    if extra_sds == "zonal":
        s = f.create("geometry_all_toa_lw_zon", SDC.FLOAT32, nlat)
        s.dim(0).setname("latitude")
        s[:] = np.linspace(150.0, 300.0, nlat).astype(np.float32)
        s.endaccess()
    elif extra_sds == "layered":
        s = f.create("geometry_cld_amount_reg", SDC.FLOAT32, (5, nlat, nlon))
        s.dim(0).setname("cloud_layer")
        s.dim(1).setname("latitude")
        s.dim(2).setname("longitude")
        s[:] = rng.uniform(0.0, 100.0, size=(5, nlat, nlon)).astype(np.float32)
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

    ds = CERESSYN1degReader().open([p], variables=["geometry_all_toa_lw_reg"])

    assert ds.sizes["time"] == 1
    assert np.datetime64("2025-12-01") == ds["time"].values[0]
    assert set(ds["geometry_all_toa_lw_reg"].dims) == {"time", "lat", "lon"}
    assert ds.attrs["geometry"] == "grid"


def test_syn_day_files_concat_and_sort(tmp_path: Path) -> None:
    # Written out of order; reader must sort by time.
    _write_syn_hdf4(tmp_path / "CER_SYN1deg-Day_Terra-Aqua-NOAA20_Edition4B_415412.20251202")
    _write_syn_hdf4(tmp_path / "CER_SYN1deg-Day_Terra-Aqua-NOAA20_Edition4B_415412.20251201")

    ds = CERESSYN1degReader().open(
        sorted(tmp_path.glob("*.2025120*")), variables=["geometry_all_toa_lw_reg"]
    )

    assert ds.sizes["time"] == 2
    times = ds["time"].values
    assert times[0] == np.datetime64("2025-12-01") and times[1] == np.datetime64("2025-12-02")


def test_syn_rejects_mixed_cadences(tmp_path: Path) -> None:
    month = _write_syn_hdf4(
        tmp_path / "CER_SYN1deg-Month_Terra-Aqua-NOAA20_Edition4B_415412.202512"
    )
    day = _write_syn_hdf4(tmp_path / "CER_SYN1deg-Day_Terra-Aqua-NOAA20_Edition4B_415412.20251202")

    with pytest.raises(ValueError, match="mixed SYN1deg cadences"):
        CERESSYN1degReader().open([month, day], variables=["geometry_all_toa_lw_reg"])


def test_syn_hourly_file_expands_24_steps(tmp_path: Path) -> None:
    p = _write_syn_hdf4(
        tmp_path / "CER_SYN1deg-1Hour_Terra-Aqua-NOAA20_Edition4B_415412.20251229",
        hourly=True,
    )

    ds = CERESSYN1degReader().open([p], variables=["geometry_all_toa_lw_reg"])

    assert ds.sizes["time"] == 24
    assert ds["time"].values[0] == np.datetime64("2025-12-29T00:00")
    assert ds["time"].values[-1] == np.datetime64("2025-12-29T23:00")


def test_syn_latitude_flipped_ascending_with_data(tmp_path: Path) -> None:
    p = _write_syn_hdf4(tmp_path / "CER_SYN1deg-Month_Terra-Aqua-NOAA20_Edition4B_415412.202512")
    # Read the raw row that sits at descending-lat index 0 (lat=+89.5).
    pyhdf_SD = _pyhdf_sd()
    SD, SDC = pyhdf_SD.SD, pyhdf_SD.SDC
    f = SD(str(p), SDC.READ)
    north_row = np.array(f.select("geometry_all_toa_lw_reg").get()[0, :], dtype=np.float64)
    f.end()

    ds = CERESSYN1degReader().open([p], variables=["geometry_all_toa_lw_reg"])

    lat = ds["lat"].values
    assert np.all(np.diff(lat) > 0)  # ascending after standardization
    got = ds["geometry_all_toa_lw_reg"].sel(lat=89.5).isel(time=0).values
    np.testing.assert_allclose(got, north_row)  # data moved with its coord


def test_syn_fill_value_masked(tmp_path: Path) -> None:
    p = _write_syn_hdf4(
        tmp_path / "CER_SYN1deg-Month_Terra-Aqua-NOAA20_Edition4B_415412.202511",
        fill_first_cell=True,
    )

    ds = CERESSYN1degReader().open([p], variables=["geometry_all_toa_lw_reg"])

    da = ds["geometry_all_toa_lw_reg"].isel(time=0)
    assert bool(np.isnan(da.sel(lat=89.5, lon=-179.5)))  # filled cell -> NaN
    assert int(np.isnan(da).sum()) == 1


def test_syn_unparseable_filename_raises(tmp_path: Path) -> None:
    p = _write_syn_hdf4(tmp_path / "not_a_ceres_name.hdf")

    with pytest.raises(ValueError, match="filename"):
        CERESSYN1degReader().open([p], variables=["geometry_all_toa_lw_reg"])


def test_syn_scan_skips_non_regional_sds(tmp_path: Path) -> None:
    p = _write_syn_hdf4(
        tmp_path / "CER_SYN1deg-Month_Terra-Aqua-NOAA20_Edition4B_415412.202512",
        extra_sds="zonal",
    )

    ds = CERESSYN1degReader().open([p])  # variables=None scan

    assert "geometry_all_toa_lw_reg" in ds.data_vars
    assert "geometry_all_toa_lw_zon" not in ds.data_vars  # 1-D zonal skipped


def test_syn_explicit_request_for_layered_var_raises(tmp_path: Path) -> None:
    p = _write_syn_hdf4(
        tmp_path / "CER_SYN1deg-Month_Terra-Aqua-NOAA20_Edition4B_415412.202512",
        extra_sds="layered",
    )

    with pytest.raises(ValueError, match="unsupported dims"):
        CERESSYN1degReader().open([p], variables=["geometry_cld_amount_reg"])


def test_syn_missing_requested_variable_raises(tmp_path: Path) -> None:
    p = _write_syn_hdf4(tmp_path / "CER_SYN1deg-Month_Terra-Aqua-NOAA20_Edition4B_415412.202512")

    with pytest.raises(ValueError, match="not found"):
        CERESSYN1degReader().open([p], variables=["no_such_var"])
