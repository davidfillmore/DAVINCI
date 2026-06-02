"""Tests for the unified pairing module."""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from davinci_monet.core.protocols import DataGeometry
from davinci_monet.pairing import (
    BasePairingStrategy,
    GridStrategy,
    PairingConfig,
    PairingEngine,
    PointStrategy,
    ProfileStrategy,
    SwathStrategy,
    TrackStrategy,
)

# =============================================================================
# Fixtures for synthetic test data
# =============================================================================


@pytest.fixture
def model_2d() -> xr.Dataset:
    """Create a simple 2D model dataset (time, lat, lon)."""
    times = pd.date_range("2024-01-01", periods=24, freq="h")
    lats = np.linspace(30, 50, 20)
    lons = np.linspace(-120, -80, 40)

    # Create temperature field with spatial gradient
    lat_grid, lon_grid = np.meshgrid(lats, lons, indexing="ij")
    temp_base = 280 + 0.5 * (lat_grid - 40) + 0.1 * (lon_grid + 100)

    # Add time variation
    temp = np.stack([temp_base + 5 * np.sin(2 * np.pi * i / 24) for i in range(24)])

    return xr.Dataset(
        {
            "temperature": (["time", "lat", "lon"], temp),
            "humidity": (["time", "lat", "lon"], 50 + 10 * np.random.randn(24, 20, 40)),
        },
        coords={
            "time": times,
            "lat": lats,
            "lon": lons,
        },
    )


@pytest.fixture
def model_3d() -> xr.Dataset:
    """Create a 3D model dataset (time, z, lat, lon)."""
    times = pd.date_range("2024-01-01", periods=24, freq="h")  # Match point obs freq
    z_levels = np.array([0, 100, 500, 1000, 2000, 5000])
    lats = np.linspace(30, 50, 10)
    lons = np.linspace(-120, -80, 20)

    temp = 290 - 0.006 * z_levels[:, None, None] + np.random.randn(24, 6, 10, 20)

    return xr.Dataset(
        {
            "temperature": (["time", "z", "lat", "lon"], temp),
            "ozone": (["time", "z", "lat", "lon"], 40 + 20 * np.random.randn(24, 6, 10, 20)),
        },
        coords={
            "time": times,
            "z": z_levels,
            "lat": lats,
            "lon": lons,
        },
    )


@pytest.fixture
def point_obs() -> xr.Dataset:
    """Create point observations (surface stations)."""
    times = pd.date_range("2024-01-01", periods=24, freq="h")
    n_sites = 5

    site_lats = np.array([35, 40, 45, 38, 42])
    site_lons = np.array([-100, -105, -95, -110, -90])

    return xr.Dataset(
        {
            "temperature": (["time", "site"], 285 + 5 * np.random.randn(24, n_sites)),
            "humidity": (["time", "site"], 55 + 10 * np.random.randn(24, n_sites)),
        },
        coords={
            "time": times,
            "site": np.arange(n_sites),
            "latitude": ("site", site_lats),
            "longitude": ("site", site_lons),
        },
    )


@pytest.fixture
def track_obs() -> xr.Dataset:
    """Create track observations (aircraft trajectory)."""
    times = pd.date_range("2024-01-01 06:00", periods=100, freq="2min")

    # Create a flight path
    lats = 35 + np.linspace(0, 10, 100)
    lons = -110 + np.linspace(0, 20, 100)
    alts = np.concatenate(
        [
            np.linspace(0, 5000, 30),
            np.ones(40) * 5000,
            np.linspace(5000, 0, 30),
        ]
    )

    return xr.Dataset(
        {
            "ozone": ("time", 50 + 10 * np.random.randn(100)),
            "temperature": ("time", 290 - 0.006 * alts + np.random.randn(100)),
        },
        coords={
            "time": times,
            "latitude": ("time", lats),
            "longitude": ("time", lons),
            "altitude": ("time", alts),
        },
    )


@pytest.fixture
def profile_obs() -> xr.Dataset:
    """Create profile observations (sonde)."""
    levels = np.array([1000, 925, 850, 700, 500, 300, 200, 100])
    times = pd.date_range("2024-01-01 12:00", periods=1, freq="h")

    return xr.Dataset(
        {
            "temperature": (
                ["time", "level"],
                (290 - 0.05 * (1000 - levels) + np.random.randn(8)).reshape(1, -1),
            ),
            "humidity": (
                ["time", "level"],
                (80 - 0.1 * (1000 - levels) + 5 * np.random.randn(8)).reshape(1, -1),
            ),
        },
        coords={
            "level": levels,
            "latitude": 40.0,
            "longitude": -105.0,
            "time": times,
        },
    )


@pytest.fixture
def swath_obs() -> xr.Dataset:
    """Create swath observations (satellite L2)."""
    n_scanlines = 50
    n_pixels = 30

    lats = 30 + np.linspace(0, 20, n_scanlines)[:, None] + np.zeros(n_pixels)
    lons = -120 + np.linspace(0, 40, n_pixels)[None, :] + np.zeros((n_scanlines, 1))

    return xr.Dataset(
        {
            "column_ozone": (
                ["scanline", "pixel"],
                300 + 50 * np.random.randn(n_scanlines, n_pixels),
            ),
        },
        coords={
            "scanline": np.arange(n_scanlines),
            "pixel": np.arange(n_pixels),
            "latitude": (["scanline", "pixel"], lats),
            "longitude": (["scanline", "pixel"], lons),
            "time": pd.Timestamp("2024-01-01 13:30"),
        },
    )


@pytest.fixture
def gridded_obs() -> xr.Dataset:
    """Create gridded observations (L3 product or reanalysis)."""
    times = pd.date_range("2024-01-01", periods=4, freq="6h")
    lats = np.linspace(32, 48, 15)
    lons = np.linspace(-118, -82, 30)

    return xr.Dataset(
        {
            "temperature": (["time", "lat", "lon"], 285 + 10 * np.random.randn(4, 15, 30)),
        },
        coords={
            "time": times,
            "lat": lats,
            "lon": lons,
        },
    )


# =============================================================================
# Tests for PairingConfig
# =============================================================================


class TestPairingConfig:
    """Tests for PairingConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = PairingConfig()
        assert config.radius_of_influence == 12000.0
        assert config.time_tolerance is None
        assert config.vertical_method == "nearest"
        assert config.horizontal_method == "nearest"
        assert config.apply_averaging_kernel is False
        assert config.require_overlap is True

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = PairingConfig(
            radius_of_influence=25000.0,
            time_tolerance=timedelta(hours=2),
            vertical_method="linear",
            require_overlap=False,
        )
        assert config.radius_of_influence == 25000.0
        assert config.time_tolerance == timedelta(hours=2)
        assert config.vertical_method == "linear"
        assert config.require_overlap is False


# =============================================================================
# Tests for PairingEngine
# =============================================================================


class TestPairingEngine:
    """Tests for PairingEngine class."""

    def test_initialization(self) -> None:
        """Test engine initializes with default strategies."""
        engine = PairingEngine()
        assert engine.get_strategy(DataGeometry.POINT) is not None
        assert engine.get_strategy(DataGeometry.TRACK) is not None
        assert engine.get_strategy(DataGeometry.PROFILE) is not None
        assert engine.get_strategy(DataGeometry.SWATH) is not None
        assert engine.get_strategy(DataGeometry.GRID) is not None

    def test_register_strategy(self) -> None:
        """Test registering a custom strategy."""
        engine = PairingEngine()
        strategy = PointStrategy()
        engine.register_strategy(strategy)
        assert engine.get_strategy(DataGeometry.POINT) is strategy

    def test_detect_geometry_point(self, point_obs: xr.Dataset) -> None:
        """Test geometry detection for point observations."""
        engine = PairingEngine()
        geometry = engine._detect_geometry(point_obs)
        assert geometry == DataGeometry.POINT

    def test_detect_geometry_track(self, track_obs: xr.Dataset) -> None:
        """Test geometry detection for track observations."""
        engine = PairingEngine()
        geometry = engine._detect_geometry(track_obs)
        assert geometry == DataGeometry.TRACK

    def test_detect_geometry_swath(self, swath_obs: xr.Dataset) -> None:
        """Test geometry detection for swath observations."""
        engine = PairingEngine()
        geometry = engine._detect_geometry(swath_obs)
        assert geometry == DataGeometry.SWATH

    def test_detect_geometry_grid(self, gridded_obs: xr.Dataset) -> None:
        """Test geometry detection for gridded observations."""
        engine = PairingEngine()
        geometry = engine._detect_geometry(gridded_obs)
        assert geometry == DataGeometry.GRID


# =============================================================================
# Tests for BasePairingStrategy
# =============================================================================


class TestBasePairingStrategy:
    """Tests for base strategy utility methods."""

    def test_haversine_distance(self) -> None:
        """Test haversine distance calculation."""
        strategy = PointStrategy()

        # Same point - distance should be 0
        dist = strategy._haversine_distance(40.0, -100.0, np.array([40.0]), np.array([-100.0]))
        assert dist[0] == pytest.approx(0.0, abs=1e-6)

        # Known distance: ~111 km per degree at equator
        dist = strategy._haversine_distance(0.0, 0.0, np.array([0.0]), np.array([1.0]))
        assert dist[0] == pytest.approx(111195, rel=0.01)

    def test_find_nearest_indices(self, model_2d: xr.Dataset) -> None:
        """Test finding nearest grid indices."""
        strategy = PointStrategy()
        model_lat = model_2d["lat"]
        model_lon = model_2d["lon"]

        obs_lat = xr.DataArray([40.0])
        obs_lon = xr.DataArray([-100.0])

        lat_idx, lon_idx = strategy._find_nearest_indices(model_lat, model_lon, obs_lat, obs_lon)

        # Check that indices are valid
        assert 0 <= lat_idx.values[0] < len(model_lat)
        assert 0 <= lon_idx.values[0] < len(model_lon)

    def test_find_nearest_with_radius_filter(self, model_2d: xr.Dataset) -> None:
        """Test that radius of influence filters distant points."""
        strategy = PointStrategy()
        model_lat = model_2d["lat"]
        model_lon = model_2d["lon"]

        # Point far outside the grid
        obs_lat = xr.DataArray([0.0])  # Very far south
        obs_lon = xr.DataArray([0.0])  # Very far east

        lat_idx, lon_idx = strategy._find_nearest_indices(
            model_lat,
            model_lon,
            obs_lat,
            obs_lon,
            radius_of_influence=100000.0,  # 100 km - too small
        )

        # Should be marked as invalid (-1)
        assert lat_idx.values[0] == -1
        assert lon_idx.values[0] == -1

    def test_extract_surface(self, model_3d: xr.Dataset) -> None:
        """Test surface extraction from 3D model."""
        strategy = PointStrategy()
        surface = strategy._extract_surface(model_3d)

        assert "z" not in surface.dims
        assert "temperature" in surface.data_vars


# =============================================================================
# Tests for PointStrategy
# =============================================================================


class TestPointStrategy:
    """Tests for point-to-grid pairing."""

    def test_geometry_property(self) -> None:
        """Test geometry property returns POINT."""
        strategy = PointStrategy()
        assert strategy.geometry == DataGeometry.POINT

    def test_pair_basic(self, model_2d: xr.Dataset, point_obs: xr.Dataset) -> None:
        """Test basic point pairing."""
        strategy = PointStrategy()
        paired = strategy.pair(model_2d, point_obs, radius_of_influence=100000.0)

        # Should have both obs and model variables (model vars are prefixed on collision)
        assert "temperature" in paired.data_vars  # Obs var
        assert "model_temperature" in paired.data_vars
        assert "humidity" in paired.data_vars  # Obs var
        assert "model_humidity" in paired.data_vars

        # Should have site dimension
        assert "site" in paired.dims

    def test_pair_with_3d_model(self, model_3d: xr.Dataset, point_obs: xr.Dataset) -> None:
        """Test pairing with 3D model (extracts surface)."""
        strategy = PointStrategy()
        paired = strategy.pair(model_3d, point_obs, radius_of_influence=200000.0)

        # Model variables keep original names (prefixing done by engine)
        assert "temperature" in paired.data_vars
        # Surface extraction removes z dimension
        assert "z" not in paired.dims

    def test_pair_time_method_linear_interpolates_between_sparse_snapshots(self) -> None:
        """time_method='linear' must interpolate model values between sparse
        snapshots, not hold them constant within nearest-neighbor bins.

        Background: when paired with hourly obs, a model with 6-hourly output
        (e.g. WRF-Chem AQ_WATCH) under the default 'nearest' time method
        produces a step function with discontinuities at the bin midpoints
        (03, 09, 15, 21 UTC). 'linear' smooths through the snapshots.
        """
        # 2 model timesteps 12 hours apart, with values 10.0 and 20.0
        # 13 obs timesteps spanning the same window (hourly)
        model_times = pd.date_range("2024-01-01 00:00", periods=2, freq="12h")
        obs_times = pd.date_range("2024-01-01 00:00", periods=13, freq="h")

        # Build a tiny rectangular-grid model around a single site
        lats = np.linspace(34, 36, 5)
        lons = np.linspace(-101, -99, 5)
        # Make the field vary in time but uniform in space, so site extraction
        # gives the temporal pattern cleanly.
        tvals = np.array([10.0, 20.0])[:, None, None]
        field = np.broadcast_to(tvals, (2, 5, 5)).copy()
        model = xr.Dataset(
            {"pm25": (["time", "lat", "lon"], field)},
            coords={"time": model_times, "lat": lats, "lon": lons},
        )

        # One obs site at the center of the model grid; obs values are 0
        # (irrelevant — we're checking the model interpolation, not stats)
        obs = xr.Dataset(
            {"pm25": (["time", "site"], np.zeros((13, 1)))},
            coords={
                "time": obs_times,
                "site": np.arange(1),
                "latitude": ("site", np.array([35.0])),
                "longitude": ("site", np.array([-100.0])),
            },
        )

        strategy = PointStrategy()
        paired = strategy.pair(model, obs, radius_of_influence=200000.0, time_method="linear")

        m = paired["model_pm25"].values.squeeze()  # shape (13,)
        # Endpoints exact
        assert abs(m[0] - 10.0) < 1e-6
        assert abs(m[12] - 20.0) < 1e-6
        # Midpoint linearly interpolated: 15.0 at hour 06
        assert abs(m[6] - 15.0) < 1e-6, f"Expected 15.0 at midpoint, got {m[6]}"
        # No step function: every hour should be strictly between neighbors
        diffs = np.diff(m)
        assert np.all(diffs > 0), "Linear interp must produce monotonic increase between 10 and 20"
        assert np.allclose(
            diffs, diffs[0]
        ), "Linear interp must produce equal increments, got " + str(diffs)

    def test_pair_time_method_nearest_still_steps(self) -> None:
        """Default time_method='nearest' must still produce step function.
        Regression guard so we don't accidentally flip the default."""
        model_times = pd.date_range("2024-01-01 00:00", periods=2, freq="12h")
        obs_times = pd.date_range("2024-01-01 00:00", periods=13, freq="h")

        lats = np.linspace(34, 36, 5)
        lons = np.linspace(-101, -99, 5)
        tvals = np.array([10.0, 20.0])[:, None, None]
        field = np.broadcast_to(tvals, (2, 5, 5)).copy()
        model = xr.Dataset(
            {"pm25": (["time", "lat", "lon"], field)},
            coords={"time": model_times, "lat": lats, "lon": lons},
        )
        obs = xr.Dataset(
            {"pm25": (["time", "site"], np.zeros((13, 1)))},
            coords={
                "time": obs_times,
                "site": np.arange(1),
                "latitude": ("site", np.array([35.0])),
                "longitude": ("site", np.array([-100.0])),
            },
        )

        paired = PointStrategy().pair(model, obs, radius_of_influence=200000.0)
        m = paired["model_pm25"].values.squeeze()

        # First 6 hours nearest to model[0]=10, last 7 nearest to model[1]=20.
        # (Ties at the midpoint resolve toward the later snapshot.)
        assert (m[:6] == 10.0).all()
        assert (m[7:] == 20.0).all()

    def test_pair_drops_sites_outside_radius(self, model_2d: xr.Dataset) -> None:
        """Sites beyond radius_of_influence must be removed from the paired output.

        Regression test for the WRF-Chem-vs-AirNow PM2.5 zig-zag: AirNow-International
        sites in Delhi/Chennai were thousands of km from the CONUS model grid, so the
        nearest-index lookup masked their model values to NaN — but the obs side kept
        the original values. Cross-site aggregates (e.g. timeseries domain-mean) were
        then polluted by sites with no model match.

        The paired dataset must contain only sites where *both* sides are valid.
        """
        # model_2d covers lat 30-50, lon -120 to -80. Build obs with 3 in-domain
        # sites and 2 far-out sites (analogous to Delhi/Chennai).
        times = pd.date_range("2024-01-01", periods=24, freq="h")
        site_lats = np.array([35.0, 40.0, 45.0, 28.6, 13.1])  # last two: Delhi, Chennai
        site_lons = np.array([-100.0, -105.0, -95.0, 77.2, 80.3])
        # Make the out-of-domain obs values dramatically different so any leak shows up
        temp = np.array([[285.0, 285.0, 285.0, 1000.0, 1000.0]] * 24)
        obs = xr.Dataset(
            {"temperature": (["time", "site"], temp)},
            coords={
                "time": times,
                "site": np.arange(5),
                "latitude": ("site", site_lats),
                "longitude": ("site", site_lons),
            },
        )

        strategy = PointStrategy()
        # 200 km radius matches model_2d's ~100 km grid spacing; Delhi/Chennai
        # are still ~10,000 km from any in-domain cell so they're dropped.
        paired = strategy.pair(model_2d, obs, radius_of_influence=200000.0)

        # Only the 3 in-domain sites should survive
        assert paired.sizes["site"] == 3, (
            f"Expected 3 paired sites, got {paired.sizes['site']}. "
            "Sites outside radius_of_influence must be dropped from the paired output."
        )

        # No NaN on the model side — every retained site has a valid model match
        assert not np.isnan(
            paired["model_temperature"].values
        ).any(), "Model values must be finite at all paired sites."

        # No leak of the 1000.0 sentinel values from the dropped Delhi/Chennai sites
        assert (
            paired["temperature"].values < 999.0
        ).all(), "Obs values from out-of-domain sites leaked into the paired dataset."

        # Retained sites must be exactly the in-domain ones
        np.testing.assert_array_equal(
            np.sort(paired["latitude"].values), np.array([35.0, 40.0, 45.0])
        )


# =============================================================================
# Tests for TrackStrategy
# =============================================================================


class TestTrackStrategy:
    """Tests for track-to-grid pairing."""

    def test_geometry_property(self) -> None:
        """Test geometry property returns TRACK."""
        strategy = TrackStrategy()
        assert strategy.geometry == DataGeometry.TRACK

    def test_pair_basic(self, model_3d: xr.Dataset, track_obs: xr.Dataset) -> None:
        """Test basic track pairing."""
        strategy = TrackStrategy()
        paired = strategy.pair(model_3d, track_obs, radius_of_influence=200000.0)

        # Should have both obs and model variables
        assert "ozone" in paired.data_vars
        assert "model_ozone" in paired.data_vars

        # Should have time dimension
        assert "time" in paired.dims

    def test_pair_drops_track_points_outside_radius(self, model_3d: xr.Dataset) -> None:
        """Track points beyond radius_of_influence must be removed from the paired output.

        Companion to test_pair_drops_sites_outside_radius. Same half-paired bug
        pattern: the model side was NaN-masked outside the radius, but the obs
        side along the track was retained, polluting any cross-time aggregate
        that includes obs-only points.
        """
        # model_3d covers lat 30-50, lon -120 to -80. Build a 10-point track
        # with 6 points inside the domain and 4 points far out over the Atlantic.
        times = pd.date_range("2024-01-01 06:00", periods=10, freq="h")
        in_lats = [35.0, 38.0, 42.0, 45.0, 40.0, 38.0]
        in_lons = [-110.0, -105.0, -95.0, -90.0, -100.0, -115.0]
        out_lats = [35.0, 35.0, 35.0, 35.0]
        out_lons = [0.0, 30.0, 60.0, 90.0]
        lats = np.array(in_lats + out_lats)
        lons = np.array(in_lons + out_lons)
        alts = np.full(10, 500.0)
        # Sentinel obs values for the out-of-domain points
        ozone = np.array([50.0] * 6 + [9999.0] * 4)

        obs = xr.Dataset(
            {"ozone": ("time", ozone)},
            coords={
                "time": times,
                "latitude": ("time", lats),
                "longitude": ("time", lons),
                "altitude": ("time", alts),
            },
        )

        strategy = TrackStrategy()
        paired = strategy.pair(model_3d, obs, radius_of_influence=200000.0)

        # Only the 6 in-domain track points should survive
        assert paired.sizes["time"] == 6, (
            f"Expected 6 paired track points, got {paired.sizes['time']}. "
            "Track points outside radius_of_influence must be dropped."
        )

        # No NaN on the model side at retained points
        assert not np.isnan(
            paired["model_ozone"].values
        ).any(), "Model values must be finite at all paired track points."

        # No leak of sentinel values from dropped points
        assert (
            paired["ozone"].values < 9000.0
        ).all(), "Obs values from out-of-domain track points leaked into the paired dataset."

        # Retained lat/lon coords match the in-domain track segment
        np.testing.assert_array_equal(paired["latitude"].values, np.array(in_lats))
        np.testing.assert_array_equal(paired["longitude"].values, np.array(in_lons))

    def test_vertical_interpolation(self) -> None:
        """Test vertical interpolation to aircraft altitude.

        Creates a model with pressure levels and known O3 profile,
        then verifies that track pairing interpolates to correct altitude.
        """
        # Create model with pressure levels (CESM-style, hPa)
        # Surface (~sea level) is at highest pressure (~1000 hPa)
        times = pd.date_range("2024-01-01", periods=4, freq="6h")
        lev_levels = np.array([100, 300, 500, 700, 850, 925, 1000])  # hPa
        lats = np.linspace(30, 50, 10)
        lons = np.linspace(-120, -80, 20)

        # Create O3 profile that increases with altitude (decreasing pressure)
        # Surface O3 ~ 40 ppb, tropopause O3 ~ 100 ppb
        lev_3d = lev_levels[:, np.newaxis, np.newaxis]
        o3_profile = 40 + 60 * (1 - lev_3d / 1000)  # Higher O3 at lower pressure
        o3_data = np.broadcast_to(o3_profile, (4, 7, 10, 20)).copy()

        model = xr.Dataset(
            {"O3": (["time", "lev", "lat", "lon"], o3_data)},
            coords={
                "time": times,
                "lev": lev_levels,
                "lat": lats,
                "lon": lons,
            },
        )

        # Create track observations at different altitudes
        track_times = pd.date_range("2024-01-01 03:00", periods=10, freq="30min")
        track_lats = np.linspace(35, 45, 10)
        track_lons = np.linspace(-100, -95, 10)
        # Altitudes from 0m (surface) to 9000m (near 300 hPa)
        track_alts = np.array([0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000])

        track = xr.Dataset(
            {"obs_o3": ("time", np.full(10, 50.0))},
            coords={
                "time": track_times,
                "latitude": ("time", track_lats),
                "longitude": ("time", track_lons),
                "altitude": ("time", track_alts),
            },
        )

        # Pair with vertical interpolation
        strategy = TrackStrategy()
        paired = strategy.pair(model, track, radius_of_influence=500000.0)

        # Verify model O3 values show altitude dependence
        model_o3 = paired["O3"].values

        # Surface (0m, ~1000 hPa) should have O3 near 40 ppb
        assert model_o3[0] < 50, f"Surface O3 should be ~40 ppb, got {model_o3[0]:.1f}"

        # High altitude (9000m, ~300 hPa) should have higher O3 near 80 ppb
        assert model_o3[-1] > 60, f"High altitude O3 should be >60 ppb, got {model_o3[-1]:.1f}"

        # O3 should generally increase with altitude
        assert model_o3[-1] > model_o3[0], "O3 should increase with altitude"


# =============================================================================
# Tests for ProfileStrategy
# =============================================================================


class TestProfileStrategy:
    """Tests for profile-to-grid pairing."""

    def test_geometry_property(self) -> None:
        """Test geometry property returns PROFILE."""
        strategy = ProfileStrategy()
        assert strategy.geometry == DataGeometry.PROFILE

    def test_pair_basic(self, model_3d: xr.Dataset, profile_obs: xr.Dataset) -> None:
        """Test basic profile pairing."""
        strategy = ProfileStrategy()

        # Need to add a z coordinate that matches model
        model_with_pressure = model_3d.assign_coords(z=model_3d["z"])

        paired = strategy.pair(
            model_with_pressure,
            profile_obs,
            radius_of_influence=200000.0,
        )

        # Should have observation variables
        assert "temperature" in paired.data_vars


# =============================================================================
# Tests for SwathStrategy
# =============================================================================


class TestSwathStrategy:
    """Tests for swath-to-grid pairing."""

    def test_geometry_property(self) -> None:
        """Test geometry property returns SWATH."""
        strategy = SwathStrategy()
        assert strategy.geometry == DataGeometry.SWATH

    def test_pair_basic(self, model_2d: xr.Dataset, swath_obs: xr.Dataset) -> None:
        """Test basic swath pairing."""
        strategy = SwathStrategy()
        paired = strategy.pair(model_2d, swath_obs, radius_of_influence=200000.0)

        # Should have observation variable
        assert "column_ozone" in paired.data_vars

    def test_pair_masks_pixels_outside_radius(self, model_2d: xr.Dataset) -> None:
        """Swath pixels beyond radius_of_influence must be NaN on the obs side too.

        Companion to test_pair_drops_sites_outside_radius (Point) and
        test_pair_drops_track_points_outside_radius (Track). Swath data is
        inherently 2D (scanline x pixel), so we mask rather than drop —
        but the contract is the same: out-of-domain pixels must not contribute
        valid obs values to cross-pixel aggregates.
        """
        # model_2d covers lat 30-50, lon -120 to -80. Build a 4 x 6 swath where
        # the left half is over CONUS and the right half is over Asia.
        n_scanlines, n_pixels = 4, 6
        lats = np.broadcast_to(
            np.linspace(35.0, 45.0, n_scanlines)[:, None], (n_scanlines, n_pixels)
        ).copy()
        # Pixels 0-2 over CONUS, pixels 3-5 over Asia
        lons_row = np.array([-110.0, -100.0, -90.0, 60.0, 80.0, 100.0])
        lons = np.broadcast_to(lons_row[None, :], (n_scanlines, n_pixels)).copy()

        # Sentinel obs values at the out-of-domain pixels. Use "temperature"
        # so it pairs with model_2d's temperature variable.
        column = np.full((n_scanlines, n_pixels), 300.0)
        column[:, 3:] = 9999.0

        obs = xr.Dataset(
            {"temperature": (["scanline", "pixel"], column)},
            coords={
                "scanline": np.arange(n_scanlines),
                "pixel": np.arange(n_pixels),
                "latitude": (["scanline", "pixel"], lats),
                "longitude": (["scanline", "pixel"], lons),
                "time": pd.Timestamp("2024-01-01 13:30"),
            },
        )

        strategy = SwathStrategy()
        paired = strategy.pair(model_2d, obs, radius_of_influence=200000.0)

        # Swath dims preserved
        assert paired.sizes["scanline"] == n_scanlines
        assert paired.sizes["pixel"] == n_pixels

        # Out-of-domain pixels: obs must be NaN, model must be NaN
        out_obs = paired["temperature"].values[:, 3:]
        out_model = paired["model_temperature"].values[:, 3:]
        assert np.isnan(out_obs).all(), (
            "Obs values at out-of-radius swath pixels must be NaN; " f"got {out_obs}"
        )
        assert np.isnan(out_model).all(), "Model values at out-of-radius swath pixels must be NaN."

        # In-domain pixels: obs must retain its 300.0 value, model must be finite
        in_obs = paired["temperature"].values[:, :3]
        in_model = paired["model_temperature"].values[:, :3]
        np.testing.assert_array_equal(in_obs, np.full_like(in_obs, 300.0))
        assert not np.isnan(
            in_model
        ).any(), "Model values must be finite at in-domain swath pixels."


# =============================================================================
# Tests for GridStrategy
# =============================================================================


class TestGridStrategy:
    """Tests for grid-to-grid pairing."""

    def test_geometry_property(self) -> None:
        """Test geometry property returns GRID."""
        strategy = GridStrategy()
        assert strategy.geometry == DataGeometry.GRID

    def test_pair_regrid_to_obs(self, model_2d: xr.Dataset, gridded_obs: xr.Dataset) -> None:
        """Test regridding model to observation grid."""
        strategy = GridStrategy()
        paired = strategy.pair(
            model_2d,
            gridded_obs,
            regrid_to="obs",
        )

        # Should have both variables
        assert "temperature" in paired.data_vars
        assert "model_temperature" in paired.data_vars

        # Should be on observation grid
        assert len(paired["lat"]) == len(gridded_obs["lat"])
        assert len(paired["lon"]) == len(gridded_obs["lon"])

    def test_pair_regrid_to_model(self, model_2d: xr.Dataset, gridded_obs: xr.Dataset) -> None:
        """Test regridding observations to model grid."""
        strategy = GridStrategy()
        paired = strategy.pair(
            model_2d,
            gridded_obs,
            regrid_to="model",
        )

        # Should have both variables
        assert "temperature" in paired.data_vars
        assert "model_temperature" in paired.data_vars

        # Should be on model grid
        assert len(paired["lat"]) == len(model_2d["lat"])
        assert len(paired["lon"]) == len(model_2d["lon"])


# =============================================================================
# Integration tests
# =============================================================================


class TestPairingIntegration:
    """Integration tests for full pairing workflow."""

    def test_engine_pair_point(self, model_2d: xr.Dataset, point_obs: xr.Dataset) -> None:
        """Test full pairing workflow through engine for point data."""
        engine = PairingEngine()
        config = PairingConfig(radius_of_influence=200000.0)

        paired = engine.pair(
            model=model_2d,
            obs=point_obs,
            obs_vars=["temperature"],
            model_vars=["temperature"],
            config=config,
        )

        assert paired is not None
        # Engine prefixes obs vars with "obs_" and model vars with "model_"
        assert "obs_temperature" in paired.data.data_vars
        assert "model_temperature" in paired.data.data_vars

    def test_engine_pair_track(self, model_3d: xr.Dataset, track_obs: xr.Dataset) -> None:
        """Test full pairing workflow through engine for track data."""
        engine = PairingEngine()
        config = PairingConfig(radius_of_influence=200000.0)

        paired = engine.pair(
            model=model_3d,
            obs=track_obs,
            obs_vars=["ozone"],
            model_vars=["ozone"],
            config=config,
        )

        assert paired is not None
        assert "obs_ozone" in paired.data.data_vars
        assert "model_ozone" in paired.data.data_vars

    def test_engine_pair_grid(self, model_2d: xr.Dataset, gridded_obs: xr.Dataset) -> None:
        """Test full pairing workflow through engine for gridded data."""
        engine = PairingEngine()
        config = PairingConfig()

        paired = engine.pair(
            model=model_2d,
            obs=gridded_obs,
            obs_vars=["temperature"],
            model_vars=["temperature"],
            config=config,
        )

        assert paired is not None
        assert "obs_temperature" in paired.data.data_vars
        assert "model_temperature" in paired.data.data_vars


# =============================================================================
# Regression test: grid-to-grid pairing with offset time coordinates
# =============================================================================


def _grid_ds_offset(varname: str, hour: int, minute: int, seed: int) -> xr.Dataset:
    """Synthetic GRID dataset with monthly times at a given hour:minute.

    Mirrors the MERRA2 vs MODIS pattern where MERRA2 monthly AOD is stamped at
    YYYY-MM-01T00:30:00 and MODIS L3 is stamped at YYYY-MM-01T00:00:00.
    """
    rng = np.random.default_rng(seed)
    times = np.array(
        [f"2003-{m:02d}-01T{hour:02d}:{minute:02d}:00" for m in (1, 2, 3)],
        dtype="datetime64[ns]",
    )
    lat = np.linspace(89.5, -89.5, 6)  # descending, MODIS-like
    lon = np.linspace(-179.5, 179.5, 8)
    data = rng.uniform(0.05, 0.8, size=(3, 6, 8))
    ds = xr.Dataset(
        {varname: (("time", "lat", "lon"), data)},
        coords={"time": times, "lat": lat, "lon": lon},
    )
    ds.attrs["geometry"] = DataGeometry.GRID.value
    return ds


def test_grid_pairing_preserves_model_when_times_offset() -> None:
    """Regression: nearest-matched model must NOT become all-NaN.

    Root cause: ``GridStrategy._align_times`` used ``model.sel(time=obs_times,
    method="nearest")`` which keeps the *model's* original time labels.
    ``_create_paired_output`` then reindexes the model onto the obs time
    coordinate, zeroing all model values when model/obs timestamps differ by
    any amount (e.g. MERRA2 monthly 00:30 vs MODIS L3 00:00).

    Fix: relabel the matched model's time coordinate to obs_times immediately
    after the nearest-sel so the two datasets share identical time labels.
    """
    # Model stamped at 00:30, obs at 00:00 (MERRA2 vs MODIS pattern).
    model = _grid_ds_offset("TOTEXTTAU", hour=0, minute=30, seed=1)
    obs = _grid_ds_offset("aod_550nm", hour=0, minute=0, seed=2)

    paired = (
        PairingEngine()
        .pair(
            model,
            obs,
            obs_vars=["aod_550nm"],
            model_vars=["TOTEXTTAU"],
            config=PairingConfig(time_tolerance=timedelta(hours=1), time_method="nearest"),
        )
        .data
    )

    model_var = next(v for v in paired.data_vars if str(v).startswith("model_"))
    obs_var = next(v for v in paired.data_vars if str(v).startswith("obs_"))
    model_finite = int(np.isfinite(paired[model_var]).sum())
    covalid = int((np.isfinite(paired[model_var]) & np.isfinite(paired[obs_var])).sum())
    assert model_finite > 0, (
        f"regridded model is all-NaN (time-label reindex bug); " f"model_finite={model_finite}"
    )
    assert covalid > 0, (
        f"no co-valid model/obs cells -> stats would be all-NaN; " f"covalid={covalid}"
    )
