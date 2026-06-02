import numpy as np
import pytest
import xarray as xr

from davinci_monet.core.protocols import DataGeometry
from davinci_monet.observations.satellite.catalog import UnknownProductError
from davinci_monet.observations.satellite.modis_viirs import MODISVIIRSReader


def _write_mod08_like(path, fname):
    # 4x8 grid mirroring MOD08_M3 dim/coord layout
    lat = np.linspace(89.5, -89.5, 4)
    lon = np.linspace(-179.5, 179.5, 8)
    aod = np.random.default_rng(0).uniform(0.0, 1.0, size=(4, 8)).astype("float32")
    ds = xr.Dataset(
        {"Aerosol_Optical_Depth_Land_Ocean_Mean_Mean": (("YDim:mod08", "XDim:mod08"), aod)},
        coords={"YDim": ("YDim:mod08", lat), "XDim": ("XDim:mod08", lon)},
    )
    fpath = path / fname
    ds.to_netcdf(fpath)
    return str(fpath)


def test_reader_returns_grid_with_time_and_display_name(tmp_path):
    # 2024-032 = day-of-year 32 = 2024-02-01
    f = _write_mod08_like(tmp_path, "MOD08_M3.A2024032.061.0000.nc")
    reader = MODISVIIRSReader()
    ds = reader.open([f], variables=["aod_550nm"], product="MOD08_M3")

    assert reader.geometry == DataGeometry.GRID
    assert "aod_550nm" in ds.data_vars  # SDS renamed to display name
    assert {"lat", "lon"}.issubset(set(ds.coords))  # XDim/YDim -> lon/lat
    assert "time" in ds.coords
    assert str(ds["time"].values[0])[:7] == "2024-02"  # filename month parsed
    assert ds["aod_550nm"].attrs.get("wavelength_nm") == 550


def test_reader_concatenates_months_sorted(tmp_path):
    f2 = _write_mod08_like(
        tmp_path, "MOD08_M3.A2024061.061.0000.nc"
    )  # Mar 1 (day 61 in leap year 2024)
    f1 = _write_mod08_like(tmp_path, "MOD08_M3.A2024032.061.0000.nc")  # Feb
    reader = MODISVIIRSReader()
    ds = reader.open([f2, f1], variables=["aod_550nm"], product="MOD08_M3")
    months = [str(t)[:7] for t in ds["time"].values]
    assert months == ["2024-02", "2024-03"]  # sorted ascending


def test_reader_unknown_product_raises(tmp_path):
    f = _write_mod08_like(tmp_path, "MOD08_M3.A2024032.061.0000.nc")
    reader = MODISVIIRSReader()
    with pytest.raises(UnknownProductError):
        reader.open([f], variables=["aod_550nm"], product="NOPE_M3")
