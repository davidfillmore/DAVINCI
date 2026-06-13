"""Tests for SwathGridStrategy and grid_binning module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from davinci_monet.pairing.grid_binning import bin_swath_to_grid, edges_from_centers, normalize_grid
from davinci_monet.pairing.strategies.swath_grid import SwathGridStrategy

# =============================================================================
# grid_binning unit tests
# =============================================================================


class TestEdgesFromCenters:
    """Test edge derivation from uniform center arrays."""

    def test_basic(self):
        centers = np.array([1.0, 2.0, 3.0])
        edges = edges_from_centers(centers)
        assert len(edges) == 4
        np.testing.assert_allclose(edges, [0.5, 1.5, 2.5, 3.5])

    def test_single_center(self):
        centers = np.array([5.0, 6.0])
        edges = edges_from_centers(centers)
        np.testing.assert_allclose(edges, [4.5, 5.5, 6.5])

    def test_non_unit_spacing(self):
        centers = np.array([0.0, 1.25, 2.5])
        edges = edges_from_centers(centers)
        np.testing.assert_allclose(edges, [-0.625, 0.625, 1.875, 3.125])


class TestBinSwathToGrid:
    """Test numba-accelerated binning function."""

    def _make_grid(self, nt=1, nx=4, ny=4):
        """Create simple grid arrays for testing."""
        time_edges = np.array([0.0, 86400.0])  # one day
        lon_edges = np.linspace(0, 360, nx + 1)
        lat_edges = np.linspace(-90, 90, ny + 1)
        count = np.zeros((nt, nx, ny), dtype=np.int32)
        data = np.zeros((nt, nx, ny), dtype=np.float64)
        return time_edges, lon_edges, lat_edges, count, data

    def test_single_pixel_correct_cell(self):
        time_edges, lon_edges, lat_edges, count, data = self._make_grid()
        # Place one pixel at lon=45, lat=0 → should land in cell (0, 0, 2)
        bin_swath_to_grid(
            time_edges,
            lon_edges,
            lat_edges,
            np.array([100.0]),  # time (within first bin)
            np.array([45.0]),  # lon
            np.array([0.0]),  # lat
            np.array([1.5]),  # data value
            count,
            data,
        )
        assert count.sum() == 1
        assert data.sum() == 1.5
        # lon=45 in [0, 90) → ix=0; lat=0 in [-45, 45) → iy=2
        assert count[0, 0, 2] == 1
        assert data[0, 0, 2] == 1.5

    def test_multiple_pixels_same_cell_accumulate(self):
        time_edges, lon_edges, lat_edges, count, data = self._make_grid()
        n = 5
        bin_swath_to_grid(
            time_edges,
            lon_edges,
            lat_edges,
            np.full(n, 100.0),
            np.full(n, 45.0),
            np.full(n, 0.0),
            np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
            count,
            data,
        )
        assert count[0, 0, 2] == 5
        assert data[0, 0, 2] == 15.0  # sum, not mean yet

    def test_nan_pixels_skipped(self):
        time_edges, lon_edges, lat_edges, count, data = self._make_grid()
        bin_swath_to_grid(
            time_edges,
            lon_edges,
            lat_edges,
            np.array([100.0, 100.0, 100.0]),
            np.array([45.0, 45.0, 45.0]),
            np.array([0.0, 0.0, 0.0]),
            np.array([1.0, np.nan, 3.0]),
            count,
            data,
        )
        assert count[0, 0, 2] == 2
        assert data[0, 0, 2] == 4.0

    def test_pixels_in_different_cells(self):
        time_edges, lon_edges, lat_edges, count, data = self._make_grid()
        # Two pixels in different lon bins
        bin_swath_to_grid(
            time_edges,
            lon_edges,
            lat_edges,
            np.array([100.0, 100.0]),
            np.array([45.0, 135.0]),  # different lon bins
            np.array([0.0, 0.0]),
            np.array([10.0, 20.0]),
            count,
            data,
        )
        assert count[0, 0, 2] == 1  # lon=45
        assert count[0, 1, 2] == 1  # lon=135
        assert data[0, 0, 2] == 10.0
        assert data[0, 1, 2] == 20.0

    def test_clamping_at_boundaries(self):
        time_edges, lon_edges, lat_edges, count, data = self._make_grid()
        # Pixel at extreme edge
        bin_swath_to_grid(
            time_edges,
            lon_edges,
            lat_edges,
            np.array([100.0]),
            np.array([359.99]),  # near lon=360 edge
            np.array([89.99]),  # near lat=90 edge
            np.array([7.0]),
            count,
            data,
        )
        assert count.sum() == 1
        assert data.sum() == 7.0

    def test_out_of_domain_pixels_are_skipped(self):
        time_edges, lon_edges, lat_edges, count, data = self._make_grid(nt=1, nx=1, ny=1)
        bin_swath_to_grid(
            time_edges,
            lon_edges,
            lat_edges,
            np.array([100.0, 100.0]),
            np.array([180.0, 999.0]),
            np.array([0.0, 999.0]),
            np.array([10.0, 1000.0]),
            count,
            data,
        )

        assert count[0, 0, 0] == 1
        assert data[0, 0, 0] == 10.0


class TestNormalizeGrid:
    """Test grid normalization."""

    def test_basic_normalization(self):
        count = np.array([[[3, 0], [1, 2]]], dtype=np.int32)
        data = np.array([[[9.0, 0.0], [5.0, 8.0]]], dtype=np.float64)
        normalize_grid(count, data)
        assert data[0, 0, 0] == 3.0  # 9/3
        assert np.isnan(data[0, 0, 1])  # count=0 → NaN
        assert data[0, 1, 0] == 5.0  # 5/1
        assert data[0, 1, 1] == 4.0  # 8/2

    def test_all_empty(self):
        count = np.zeros((1, 2, 2), dtype=np.int32)
        data = np.zeros((1, 2, 2), dtype=np.float64)
        normalize_grid(count, data)
        assert np.all(np.isnan(data))


# =============================================================================
# SwathGridStrategy tests
# =============================================================================


def _make_synthetic_model(
    nlat: int = 10,
    nlon: int = 20,
    ntime: int = 3,
) -> xr.Dataset:
    """Create a synthetic model dataset on a regular grid."""
    lat = np.linspace(-90, 90, nlat)
    lon = np.linspace(0, 360, nlon, endpoint=False)
    time = pd.date_range("2019-12-20", periods=ntime, freq="1D")
    aod = np.random.default_rng(42).uniform(0.05, 0.5, (ntime, nlat, nlon))
    return xr.Dataset(
        {"AODVIS": (["time", "lat", "lon"], aod.astype(np.float32))},
        coords={"time": time, "lat": lat, "lon": lon},
    )


def _make_synthetic_swath(
    n_scanlines: int = 20,
    n_pixels: int = 30,
    lon_range: tuple[float, float] = (-180, 180),
    lat_range: tuple[float, float] = (-60, 60),
    aod_range: tuple[float, float] = (0.05, 0.8),
    time_val: str = "2019-12-21",
) -> xr.Dataset:
    """Create a synthetic satellite swath dataset."""
    rng = np.random.default_rng(123)
    lat = rng.uniform(lat_range[0], lat_range[1], (n_scanlines, n_pixels))
    lon = rng.uniform(lon_range[0], lon_range[1], (n_scanlines, n_pixels))
    aod = rng.uniform(aod_range[0], aod_range[1], (n_scanlines, n_pixels)).astype(np.float32)
    # Sprinkle some NaNs
    aod[rng.random((n_scanlines, n_pixels)) < 0.1] = np.nan
    return xr.Dataset(
        {"AOD_550": (["scanline", "pixel"], aod)},
        coords={
            "latitude": (["scanline", "pixel"], lat),
            "longitude": (["scanline", "pixel"], lon),
            "time": pd.Timestamp(time_val),
        },
    )


class TestSwathGridStrategy:
    """Integration tests for SwathGridStrategy."""

    def test_match_model_mode(self):
        model = _make_synthetic_model()
        obs = _make_synthetic_swath()
        strategy = SwathGridStrategy()
        paired = strategy.pair(
            model,
            obs,
            grid_mode="match_model",
            time_resolution="1D",
            obs_var="AOD_550",
            model_var="AODVIS",
        )
        # Check output structure
        assert "obs_AOD_550" in paired.data_vars
        assert "model_AODVIS" in paired.data_vars
        assert "obs_count" in paired.data_vars
        assert "time" in paired.dims
        assert "lat" in paired.dims
        assert "lon" in paired.dims

    def test_output_dims_match_model(self):
        model = _make_synthetic_model(nlat=10, nlon=20, ntime=3)
        obs = _make_synthetic_swath()
        strategy = SwathGridStrategy()
        paired = strategy.pair(
            model,
            obs,
            grid_mode="match_model",
            time_resolution="1D",
            obs_var="AOD_550",
            model_var="AODVIS",
        )
        # Lon and lat should match model grid
        assert paired.dims["lat"] == 10
        assert paired.dims["lon"] == 20

    def test_obs_count_positive_where_data(self):
        model = _make_synthetic_model()
        obs = _make_synthetic_swath()
        strategy = SwathGridStrategy()
        paired = strategy.pair(
            model,
            obs,
            grid_mode="match_model",
            time_resolution="1D",
            obs_var="AOD_550",
            model_var="AODVIS",
        )
        # Where obs_count > 0, obs data should be finite
        count = paired["obs_count"].values
        obs_data = paired["obs_AOD_550"].values
        assert np.all(np.isfinite(obs_data[count > 0]))

    def test_obs_nan_where_no_data(self):
        model = _make_synthetic_model()
        obs = _make_synthetic_swath()
        strategy = SwathGridStrategy()
        paired = strategy.pair(
            model,
            obs,
            grid_mode="match_model",
            time_resolution="1D",
            obs_var="AOD_550",
            model_var="AODVIS",
        )
        count = paired["obs_count"].values
        obs_data = paired["obs_AOD_550"].values
        # Where count == 0, data should be NaN
        assert np.all(np.isnan(obs_data[count == 0]))

    def test_min_obs_count_filtering(self):
        model = _make_synthetic_model()
        obs = _make_synthetic_swath()
        strategy = SwathGridStrategy()
        paired = strategy.pair(
            model,
            obs,
            grid_mode="match_model",
            time_resolution="1D",
            obs_var="AOD_550",
            model_var="AODVIS",
            min_obs_count=3,
        )
        count = paired["obs_count"].values
        obs_data = paired["obs_AOD_550"].values
        # Cells with count < 3 should be NaN
        assert np.all(np.isnan(obs_data[(count > 0) & (count < 3)]))

    def test_longitude_wrapping(self):
        """Obs with -180..180 lon should pair correctly with 0..360 model."""
        model = _make_synthetic_model(nlon=4)
        # Swath with negative longitudes
        obs = _make_synthetic_swath(lon_range=(-180, 0))
        strategy = SwathGridStrategy()
        paired = strategy.pair(
            model,
            obs,
            grid_mode="match_model",
            time_resolution="1D",
            obs_var="AOD_550",
            model_var="AODVIS",
        )
        # Should have data in the 180-360 range of the model grid
        count = paired["obs_count"].values
        assert count.sum() > 0

    def test_resolution_mode(self):
        model = _make_synthetic_model()
        obs = _make_synthetic_swath()
        strategy = SwathGridStrategy()
        paired = strategy.pair(
            model,
            obs,
            grid_mode="resolution",
            resolution=10.0,
            time_resolution="1D",
            obs_var="AOD_550",
            model_var="AODVIS",
        )
        # 10-degree grid: 18 lat bins, 36 lon bins
        assert paired.dims["lat"] == 18
        assert paired.dims["lon"] == 36

    def test_variable_prefix_convention(self):
        model = _make_synthetic_model()
        obs = _make_synthetic_swath()
        strategy = SwathGridStrategy()
        paired = strategy.pair(
            model,
            obs,
            grid_mode="match_model",
            time_resolution="1D",
            obs_var="AOD_550",
            model_var="AODVIS",
        )
        # Check naming convention: obs_ and model_ prefixes
        var_names = list(paired.data_vars)
        assert any(str(v).startswith("obs_") for v in var_names)
        assert any(str(v).startswith("model_") for v in var_names)

    def test_aod_values_reasonable(self):
        """Binned values should be within input range."""
        model = _make_synthetic_model()
        obs = _make_synthetic_swath(aod_range=(0.05, 0.8))
        strategy = SwathGridStrategy()
        paired = strategy.pair(
            model,
            obs,
            grid_mode="match_model",
            time_resolution="1D",
            obs_var="AOD_550",
            model_var="AODVIS",
        )
        obs_data = paired["obs_AOD_550"].values
        valid = obs_data[np.isfinite(obs_data)]
        assert np.all(valid >= 0.05)
        assert np.all(valid <= 0.8)


# =============================================================================
# Regression test: 2-D per-scanline multi-time-bin layout
# =============================================================================


def test_per_scanline_time_bins_across_multiple_days() -> None:
    """2-D swath with per-scanline times spanning 3 days must bin per day.

    Regression: DataArray.astype("datetime64[s]") silently kept ns units,
    collapsing every pixel into the final time bin (fixed in SSF Phase 3).

    The test uses a constant obs value per day (100 / 200 / 300) so the
    per-bin means are assertable without statistical uncertainty.
    """
    # -- observation dataset (scanline=6, pixel=4) ----------------------------
    n_scanlines = 6
    n_pixels = 4

    # Two scanlines per day across 2025-10-01 / 02 / 03
    days = pd.to_datetime(["2025-10-01", "2025-10-02", "2025-10-03"])
    # shape (6,): [day0, day0, day1, day1, day2, day2]
    scanline_times = np.array(
        [days[0], days[0], days[1], days[1], days[2], days[2]],
        dtype="datetime64[ns]",
    )

    # Known constant flux per day (assertable means after binning)
    day_values = np.array([100.0, 200.0, 300.0])
    # shape (6, 4): rows 0-1 → 100, rows 2-3 → 200, rows 4-5 → 300
    flux = np.repeat(day_values, 2).reshape(n_scanlines, 1) * np.ones((n_scanlines, n_pixels))

    # 2-D lat/lon spread across domain (-60..60 lat, -120..120 lon)
    lats = np.linspace(-60, 60, n_scanlines).reshape(-1, 1) * np.ones((n_scanlines, n_pixels))
    lons = np.linspace(-120, 120, n_pixels).reshape(1, -1) * np.ones((n_scanlines, n_pixels))

    obs = xr.Dataset(
        {"flux": (["scanline", "pixel"], flux.astype(np.float32))},
        coords={
            "latitude": (["scanline", "pixel"], lats),
            "longitude": (["scanline", "pixel"], lons),
            "time": ("scanline", scanline_times),
        },
    )

    # -- model dataset (time=3 days, lat=12, lon=24) --------------------------
    lat = np.linspace(-90, 90, 12)
    lon = np.linspace(-180, 180, 24, endpoint=False)
    model_time = days
    model_flux = np.ones((3, 12, 24), dtype=np.float32) * 999.0

    model = xr.Dataset(
        {"FLUX": (["time", "lat", "lon"], model_flux)},
        coords={"time": model_time, "lat": lat, "lon": lon},
    )

    # -- pair -----------------------------------------------------------------
    strategy = SwathGridStrategy()
    paired = strategy.pair(
        model,
        obs,
        grid_mode="match_model",
        time_resolution="1D",
        obs_var="flux",
        model_var="FLUX",
    )

    obs_binned = paired["obs_flux"].values  # (time, lon, lat)
    obs_count = paired["obs_count"].values

    # 3 distinct time bins must be present
    assert paired.dims["time"] == 3, f"Expected 3 time bins, got {paired.dims['time']}"

    # Each time bin must contain at least one finite obs pixel
    finite_per_bin = np.array([np.any(np.isfinite(obs_binned[t])) for t in range(3)])
    assert np.all(finite_per_bin), (
        f"Some time bins have no finite obs (pre-fix: all collapsed into last bin). "
        f"Finite per bin: {finite_per_bin}"
    )

    # Per-bin means of finite cells must match 100 / 200 / 300 respectively
    expected = [100.0, 200.0, 300.0]
    for t, exp in enumerate(expected):
        cells = obs_binned[t][np.isfinite(obs_binned[t])]
        assert len(cells) > 0, f"Time bin {t} is empty"
        np.testing.assert_allclose(
            cells,
            exp,
            rtol=1e-4,
            err_msg=f"Time bin {t}: expected mean {exp}, got {cells.mean():.1f}",
        )
