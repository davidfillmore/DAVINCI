"""Tests for the radiative processing module."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from davinci_monet.radiative.processing import (
    compute_anomalies,
    compute_background,
    derive_smoke_aod,
    regrid_nearest,
    semi_empirical_surface_dimming,
)


@pytest.fixture
def daily_dataset() -> xr.Dataset:
    """Synthetic dataset: 5 time steps, 10x10 lat/lon grid."""
    np.random.seed(42)
    time = np.arange(5)
    lat = np.linspace(-45, 45, 10)
    lon = np.linspace(0, 180, 10)
    ds = xr.Dataset(
        {
            "toa_sw": (["time", "lat", "lon"], np.random.rand(5, 10, 10) * 400),
            "aod": (["time", "lat", "lon"], np.random.rand(5, 10, 10) * 2),
        },
        coords={"time": time, "lat": lat, "lon": lon},
    )
    return ds


# ---------------------------------------------------------------------------
# compute_background
# ---------------------------------------------------------------------------
class TestComputeBackground:
    def test_mean_of_first_n_days(self, daily_dataset: xr.Dataset) -> None:
        bg = compute_background(daily_dataset, window=3)
        expected = daily_dataset.isel(time=slice(0, 3)).mean(dim="time")
        xr.testing.assert_allclose(bg, expected)

    def test_window_larger_than_data_uses_all(self, daily_dataset: xr.Dataset) -> None:
        bg = compute_background(daily_dataset, window=100)
        expected = daily_dataset.mean(dim="time")
        xr.testing.assert_allclose(bg, expected)


# ---------------------------------------------------------------------------
# compute_anomalies
# ---------------------------------------------------------------------------
class TestComputeAnomalies:
    def test_anomaly_shape(self, daily_dataset: xr.Dataset) -> None:
        bg = compute_background(daily_dataset, window=3)
        anom = compute_anomalies(daily_dataset, bg)
        assert anom.sizes == daily_dataset.sizes

    def test_anomaly_values(self, daily_dataset: xr.Dataset) -> None:
        bg = compute_background(daily_dataset, window=3)
        anom = compute_anomalies(daily_dataset, bg)
        # Mean anomaly over the first 3 days should be near zero
        mean_anom = anom.isel(time=slice(0, 3)).mean(dim="time")
        assert float(np.abs(mean_anom["toa_sw"]).max()) < 200  # loose check
        assert float(np.abs(mean_anom["aod"]).max()) < 1.0


# ---------------------------------------------------------------------------
# derive_smoke_aod
# ---------------------------------------------------------------------------
class TestDeriveSmokeAod:
    def test_sum_of_species(self) -> None:
        ds = xr.Dataset(
            {
                "oc_aod": (["time"], [0.3]),
                "bc_aod": (["time"], [0.1]),
            }
        )
        result = derive_smoke_aod(ds, ["oc_aod", "bc_aod"])
        np.testing.assert_allclose(result.values, [0.4])

    def test_single_species(self) -> None:
        ds = xr.Dataset({"oc_aod": (["time"], [0.5, 0.6])})
        result = derive_smoke_aod(ds, ["oc_aod"])
        np.testing.assert_allclose(result.values, [0.5, 0.6])


# ---------------------------------------------------------------------------
# regrid_nearest
# ---------------------------------------------------------------------------
class TestRegridNearest:
    def test_coarsen(self) -> None:
        """An all-ones field should stay all ones after regridding."""
        lat_src = np.linspace(-90, 90, 20)
        lon_src = np.linspace(0, 360, 40)
        data = xr.DataArray(
            np.ones((20, 40)),
            dims=["lat", "lon"],
            coords={"lat": lat_src, "lon": lon_src},
        )
        target_lat = np.linspace(-80, 80, 5)
        target_lon = np.linspace(10, 350, 10)
        result = regrid_nearest(data, target_lat, target_lon)
        np.testing.assert_allclose(result.values, 1.0)
        assert result.sizes == {"lat": 5, "lon": 10}

    def test_preserves_pattern(self) -> None:
        """A latitude gradient should survive regridding."""
        lat_src = np.linspace(-90, 90, 37)
        lon_src = np.linspace(0, 360, 73)
        lat_grid, _ = np.meshgrid(lat_src, lon_src, indexing="ij")
        data = xr.DataArray(
            lat_grid,
            dims=["lat", "lon"],
            coords={"lat": lat_src, "lon": lon_src},
        )
        target_lat = np.array([-60.0, 0.0, 60.0])
        target_lon = np.array([90.0, 270.0])
        result = regrid_nearest(data, target_lat, target_lon)
        # Southern value < equator < northern value for every lon
        assert (result.sel(lat=-60.0) < result.sel(lat=0.0)).all()
        assert (result.sel(lat=0.0) < result.sel(lat=60.0)).all()


# ---------------------------------------------------------------------------
# semi_empirical_surface_dimming
# ---------------------------------------------------------------------------
class TestSemiEmpiricalDimming:
    def test_zero_aod_no_dimming(self) -> None:
        result = semi_empirical_surface_dimming(
            smoke_aod=np.array([0.0]),
            toa_insol=np.array([1361.0]),
        )
        np.testing.assert_allclose(result, 0.0, atol=1e-10)

    def test_dimming_increases_with_aod(self) -> None:
        aod = np.array([0.1, 0.5, 1.0, 3.0])
        toa = np.full_like(aod, 1361.0)
        result = semi_empirical_surface_dimming(aod, toa)
        # Dimming values should be negative and monotonically decreasing
        assert (result < 0).all()
        assert np.all(np.diff(result) < 0)

    def test_custom_ssa(self) -> None:
        aod = np.array([1.0])
        toa = np.array([1361.0])
        low_ssa = semi_empirical_surface_dimming(aod, toa, ssa=0.85)
        high_ssa = semi_empirical_surface_dimming(aod, toa, ssa=0.99)
        # Higher SSA means more scattering (less absorption) = less dimming (closer to 0)
        assert high_ssa > low_ssa  # both negative, high_ssa closer to 0
