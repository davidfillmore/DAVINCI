# davinci_monet/tests/unit/datasets/test_generic_grid_geometry.py
import numpy as np
import xarray as xr

import davinci_monet.datasets  # noqa: F401  (registers "generic")
from davinci_monet.core.protocols import DataGeometry
from davinci_monet.core.registry import source_registry


def test_generic_reader_tags_2d_field_as_grid(tmp_path):
    lat = np.linspace(-90, 90, 5)
    lon = np.linspace(-180, 175, 8)
    t = np.array(["2024-02-01"], dtype="datetime64[ns]")
    ds = xr.Dataset(
        {"TOTEXTTAU": (("time", "lat", "lon"), np.random.default_rng(0).uniform(0, 1, (1, 5, 8)))},
        coords={"time": t, "lat": lat, "lon": lon},
    )
    f = tmp_path / "merra2_like.nc"
    ds.to_netcdf(f)

    reader = source_registry.get("generic")()
    out = reader.open([str(f)], variables=["TOTEXTTAU"])
    assert "TOTEXTTAU" in out
    assert "lev" not in out.dims
    assert reader.geometry is DataGeometry.GRID
