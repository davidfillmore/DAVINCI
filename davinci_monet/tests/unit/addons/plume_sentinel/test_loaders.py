from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from davinci_monet.addons.plume_sentinel.loaders import load_input


class TestGoesLoader:
    def test_loads_goes_netcdf(self, tmp_path):
        ds = xr.Dataset(
            {
                "CMI_C01": (["y", "x"], np.random.rand(10, 10).astype(np.float32)),
                "CMI_C02": (["y", "x"], np.random.rand(10, 10).astype(np.float32)),
                "CMI_C03": (["y", "x"], np.random.rand(10, 10).astype(np.float32)),
                "goes_imager_projection": (
                    [],
                    0,
                    {
                        "perspective_point_height": 35786023.0,
                        "longitude_of_projection_origin": -75.0,
                        "sweep_angle_axis": "x",
                        "semi_major_axis": 6378137.0,
                        "semi_minor_axis": 6356752.31414,
                    },
                ),
            },
            coords={"x": np.linspace(-0.1, 0.1, 10), "y": np.linspace(-0.1, 0.1, 10)},
        )
        nc_path = tmp_path / "goes_test.nc"
        ds.to_netcdf(nc_path)
        spec = {"type": "goes_truecolor", "file": str(nc_path), "gamma": 1.8}
        result = load_input(spec)
        assert isinstance(result, xr.Dataset)
        assert "CMI_C01" in result
        assert "CMI_C02" in result
        assert "CMI_C03" in result

    def test_goes_missing_file_raises(self):
        spec = {"type": "goes_truecolor", "file": "/nonexistent/goes.nc"}
        with pytest.raises(FileNotFoundError):
            load_input(spec)


class TestHmsLoader:
    def test_hms_missing_file_raises(self):
        spec = {"type": "hms_smoke", "file": "/nonexistent/hms.shp"}
        with pytest.raises(Exception):
            load_input(spec)


class TestModisLoader:
    def test_modis_missing_files_raises(self):
        spec = {
            "type": "modis_l2_aod",
            "files": ["/nonexistent/MOD04.hdf"],
            "variable": "AOD_550_Dark_Target_Deep_Blue_Combined",
            "valid_range": [0.0, 5.0],
        }
        with pytest.raises(FileNotFoundError):
            load_input(spec)


class TestUnknownType:
    def test_unknown_type_raises(self):
        spec = {"type": "unknown_sensor"}
        with pytest.raises(ValueError, match="Unknown input type"):
            load_input(spec)
