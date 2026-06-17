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
    assert float(w.isel(lat=0)) > float(w.isel(lat=-1))


def test_vertical_dim_detection() -> None:
    assert _vertical_dim(_grid(nlev=0), _lat_coord(_grid()), _grid()["longitude"]) is None
    da3 = _grid(nlev=3)
    assert _vertical_dim(da3, da3["latitude"], da3["longitude"]) == "lev"


def test_layer_mass_weight_fallback_logs_not_warns(caplog) -> None:
    da3 = _grid(nlev=3)
    with caplog.at_level(logging.WARNING):
        mw = _layer_mass_weight(da3.to_dataset(), "lev")
    assert mw is None


def test_fix_sign_makes_max_loading_positive() -> None:
    mode = xr.DataArray(
        np.array([[-3.0, 1.0], [2.0, -0.5]]),
        dims=("mode", "lat"),
        coords={"mode": [1, 2], "lat": [0, 1]},
    )
    pc = xr.DataArray(np.ones((2, 4)), dims=("mode", "time"), coords={"mode": [1, 2], "time": np.arange(4)})
    m2, p2 = _fix_sign(mode, pc)
    assert float(m2.sel(mode=1).max()) == 3.0
    assert float(p2.sel(mode=1).isel(time=0)) == -1.0
