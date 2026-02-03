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
    alts = np.concatenate([
        np.linspace(0, 5000, 30),
        np.ones(40) * 5000,
        np.linspace(5000, 0, 30),
    ])

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
            "temperature": (["time", "level"], (290 - 0.05 * (1000 - levels) + np.random.randn(8)).reshape(1, -1)),
            "humidity": (["time", "level"], (80 - 0.1 * (1000 - levels) + 5 * np.random.randn(8)).reshape(1, -1)),
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
            "column_ozone": (["scanline", "pixel"], 300 + 50 * np.random.randn(n_scanlines, n_pixels)),
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

        lat_idx, lon_idx = strategy._find_nearest_indices(
            model_lat, model_lon, obs_lat, obs_lon
        )

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
            model_lat, model_lon, obs_lat, obs_lon,
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

        # Should have both obs and model variables (model vars keep original names)
        # Prefixing is done by create_paired_dataset in the engine, not the strategy
        assert "temperature" in paired.data_vars  # Model var (same name as obs in this test)
        assert "humidity" in paired.data_vars

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
            model_with_pressure, profile_obs,
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


# =============================================================================
# Tests for GridStrategy
# =============================================================================


class TestGridStrategy:
    """Tests for grid-to-grid pairing."""

    def test_geometry_property(self) -> None:
        """Test geometry property returns GRID."""
        strategy = GridStrategy()
        assert strategy.geometry == DataGeometry.GRID

    def test_pair_regrid_to_obs(
        self, model_2d: xr.Dataset, gridded_obs: xr.Dataset
    ) -> None:
        """Test regridding model to observation grid."""
        strategy = GridStrategy()
        paired = strategy.pair(
            model_2d, gridded_obs,
            regrid_to="obs",
        )

        # Should have both variables
        assert "temperature" in paired.data_vars
        assert "model_temperature" in paired.data_vars

        # Should be on observation grid
        assert len(paired["lat"]) == len(gridded_obs["lat"])
        assert len(paired["lon"]) == len(gridded_obs["lon"])

    def test_pair_regrid_to_model(
        self, model_2d: xr.Dataset, gridded_obs: xr.Dataset
    ) -> None:
        """Test regridding observations to model grid."""
        strategy = GridStrategy()
        paired = strategy.pair(
            model_2d, gridded_obs,
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

    def test_engine_pair_point(
        self, model_2d: xr.Dataset, point_obs: xr.Dataset
    ) -> None:
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

    def test_engine_pair_track(
        self, model_3d: xr.Dataset, track_obs: xr.Dataset
    ) -> None:
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

    def test_engine_pair_grid(
        self, model_2d: xr.Dataset, gridded_obs: xr.Dataset
    ) -> None:
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
