"""Tests for radiative data loaders (CERES, MERRA-2, AERONET)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

# ---------------------------------------------------------------------------
# Fixtures — synthetic data
# ---------------------------------------------------------------------------


@pytest.fixture()
def synthetic_ceres_dir(tmp_path):
    """Create 5 CERES-like NetCDF files with descending latitudes."""
    lats = np.linspace(89.5, -89.5, 180)  # descending
    lons = np.linspace(-179.5, 179.5, 360)
    rng = np.random.default_rng(42)

    variables = [
        "obs_all_toa_sw",
        "obs_clr_toa_sw",
        "obs_all_toa_net",
        "init_match_aod55",
        "obs_cld_amount",
        "toa_sw_insol",
    ]

    for day in range(5, 10):
        ds = xr.Dataset(
            {
                name: (["lat", "lon"], rng.uniform(0, 400, (180, 360)).astype("f4"))
                for name in variables
            },
            coords={"lat": lats, "lon": lons},
        )
        fname = f"CERES_SYN1deg-Day_2020090{day}.nc"
        ds.to_netcdf(tmp_path / fname)

    return tmp_path


@pytest.fixture()
def synthetic_merra2_dir(tmp_path):
    """Create 5 MERRA-2-like NetCDF4 files with ascending latitudes."""
    lats = np.arange(-90, 90.5, 0.5)
    lons = np.arange(-180, 180.625, 0.625)
    rng = np.random.default_rng(99)

    for day in range(5, 10):
        times = pd.date_range(f"2020-09-{day:02d}", periods=8, freq="3h")
        ds = xr.Dataset(
            {
                "TOTEXTTAU": (
                    ["time", "lat", "lon"],
                    rng.uniform(0, 1, (8, len(lats), len(lons))).astype("f4"),
                ),
                "OCEXTTAU": (
                    ["time", "lat", "lon"],
                    rng.uniform(0, 0.5, (8, len(lats), len(lons))).astype("f4"),
                ),
                "BCEXTTAU": (
                    ["time", "lat", "lon"],
                    rng.uniform(0, 0.1, (8, len(lats), len(lons))).astype("f4"),
                ),
            },
            coords={"time": times, "lat": lats, "lon": lons},
        )
        fname = f"MERRA2_400.tavg1_2d_aer_Nx.2020090{day}.nc4"
        ds.to_netcdf(tmp_path / fname)

    return tmp_path


@pytest.fixture()
def synthetic_aeronet_csv(tmp_path):
    """Create a CSV with 3 sites, 5 days, 3 readings per day."""
    rows = []
    sites = [
        ("Site_A", 40.0, -105.0),
        ("Site_B", 35.0, -100.0),
        ("Site_C", 45.0, -110.0),
    ]
    rng = np.random.default_rng(7)
    for day in range(5, 10):
        for name, lat, lon in sites:
            for hour in [8, 12, 16]:
                rows.append(
                    {
                        "time": f"2020-09-{day:02d} {hour:02d}:00:00",
                        "site": name,
                        "AOD_500nm": float(rng.uniform(0.05, 1.5)),
                        "latitude": lat,
                        "longitude": lon,
                    }
                )
    df = pd.DataFrame(rows)
    csv_path = tmp_path / "aeronet_data.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


# ---------------------------------------------------------------------------
# CERES tests
# ---------------------------------------------------------------------------


class TestCeresLoader:
    def test_load_local_syn1deg(self, synthetic_ceres_dir):
        from davinci_monet.radiative.loaders.ceres import load_ceres_local

        ds = load_ceres_local(
            str(synthetic_ceres_dir / "CERES_SYN1deg-Day_*.nc"),
            domain=(-120, -90, 30, 50),
        )
        assert "time" in ds.dims
        assert ds.sizes["time"] == 5
        # Domain subset — lats should be within [30, 50]
        assert float(ds.lat.min()) >= 30.0
        assert float(ds.lat.max()) <= 50.0
        # Lons within [-120, -90]
        assert float(ds.lon.min()) >= -120.0
        assert float(ds.lon.max()) <= -90.0

    def test_load_with_variable_filter(self, synthetic_ceres_dir):
        from davinci_monet.radiative.loaders.ceres import load_ceres_local

        keep = ["obs_all_toa_sw", "toa_sw_insol"]
        ds = load_ceres_local(
            str(synthetic_ceres_dir / "CERES_SYN1deg-Day_*.nc"),
            domain=(-120, -90, 30, 50),
            variables=keep,
        )
        assert set(ds.data_vars) == set(keep)


# ---------------------------------------------------------------------------
# MERRA-2 tests
# ---------------------------------------------------------------------------


class TestMerra2Loader:
    def test_load_and_daily_mean(self, synthetic_merra2_dir):
        from davinci_monet.radiative.loaders.merra2 import load_merra2

        ds = load_merra2(
            str(synthetic_merra2_dir / "MERRA2_*.nc4"),
            domain=(-120, -90, 30, 50),
            smoke_species=["OCEXTTAU", "BCEXTTAU"],
        )
        assert ds.sizes["time"] == 5
        assert "SMOKEAOD" in ds.data_vars


# ---------------------------------------------------------------------------
# AERONET tests
# ---------------------------------------------------------------------------


class TestAeronetLoader:
    def test_load_all_sites(self, synthetic_aeronet_csv):
        from davinci_monet.radiative.loaders.aeronet import load_aeronet

        df = load_aeronet(
            str(synthetic_aeronet_csv),
            domain=(-120, -90, 30, 50),
        )
        assert "time" in df.columns
        assert "site" in df.columns
        assert "aod" in df.columns
        assert len(df) > 0

    def test_filter_by_sites(self, synthetic_aeronet_csv):
        from davinci_monet.radiative.loaders.aeronet import load_aeronet

        df = load_aeronet(
            str(synthetic_aeronet_csv),
            domain=(-120, -90, 30, 50),
            sites=["Site_A"],
        )
        assert set(df["site"].unique()) == {"Site_A"}
