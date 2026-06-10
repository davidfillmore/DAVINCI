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
