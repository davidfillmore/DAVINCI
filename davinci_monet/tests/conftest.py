"""Pytest fixtures using synthetic data generators.

This module provides reusable fixtures for testing DAVINCI components.
All fixtures use synthetic data to avoid external dependencies.
"""

from __future__ import annotations

from typing import Any, Generator

import numpy as np
import pytest
import xarray as xr

from davinci_monet.core.protocols import DataGeometry
from davinci_monet.tests.synthetic.generators import Domain, TimeConfig
from davinci_monet.tests.synthetic.models import create_model_dataset
from davinci_monet.tests.synthetic.observations import (
    create_gridded_observations,
    create_point_observations,
    create_profile_observations,
    create_swath_observations,
    create_track_observations,
)
from davinci_monet.tests.synthetic.scenarios import (
    BiasScenario,
    PerfectMatchScenario,
    create_scenario,
)


# =============================================================================
# Domain and Time Fixtures
# =============================================================================


@pytest.fixture
def default_domain() -> Domain:
    """Default CONUS-like domain."""
    return Domain(
        lon_min=-130.0,
        lon_max=-60.0,
        lat_min=20.0,
        lat_max=55.0,
        n_lon=36,
        n_lat=18,
    )


@pytest.fixture
def small_domain() -> Domain:
    """Small domain for fast tests."""
    return Domain(
        lon_min=-100.0,
        lon_max=-90.0,
        lat_min=35.0,
        lat_max=45.0,
        n_lon=12,
        n_lat=12,
    )


@pytest.fixture
def default_time_config() -> TimeConfig:
    """Default 24-hour time configuration."""
    return TimeConfig(
        start="2024-01-15",
        end="2024-01-16",
        freq="1h",
    )


@pytest.fixture
def short_time_config() -> TimeConfig:
    """Short 6-hour time configuration for fast tests."""
    return TimeConfig(
        start="2024-01-15 00:00",
        end="2024-01-15 06:00",
        freq="1h",
    )


# =============================================================================
# Model Fixtures
# =============================================================================


@pytest.fixture
def surface_model(small_domain: Domain, short_time_config: TimeConfig) -> xr.Dataset:
    """Surface-only model dataset."""
    return create_model_dataset(
        variables=["O3", "PM25"],
        domain=small_domain,
        time_config=short_time_config,
        n_levels=0,
        seed=42,
    )


@pytest.fixture
def model_3d(small_domain: Domain, short_time_config: TimeConfig) -> xr.Dataset:
    """3D model dataset with vertical levels."""
    return create_model_dataset(
        variables=["O3", "PM25", "NO2"],
        domain=small_domain,
        time_config=short_time_config,
        n_levels=20,
        seed=42,
    )


@pytest.fixture
def model_single_var(small_domain: Domain, short_time_config: TimeConfig) -> xr.Dataset:
    """Model with single variable (O3)."""
    return create_model_dataset(
        variables=["O3"],
        domain=small_domain,
        time_config=short_time_config,
        n_levels=0,
        seed=42,
    )


# =============================================================================
# Observation Fixtures - Point
# =============================================================================


@pytest.fixture
def point_observations(small_domain: Domain, short_time_config: TimeConfig) -> xr.Dataset:
    """Point observations (surface stations)."""
    return create_point_observations(
        n_sites=10,
        variables=["O3", "PM25"],
        domain=small_domain,
        time_config=short_time_config,
        seed=42,
    )


@pytest.fixture
def point_observations_single_site(
    small_domain: Domain, short_time_config: TimeConfig
) -> xr.Dataset:
    """Single-site point observations."""
    return create_point_observations(
        n_sites=1,
        variables=["O3"],
        domain=small_domain,
        time_config=short_time_config,
        seed=42,
    )


# =============================================================================
# Observation Fixtures - Track
# =============================================================================


@pytest.fixture
def track_observations(small_domain: Domain, short_time_config: TimeConfig) -> xr.Dataset:
    """Track observations (aircraft)."""
    return create_track_observations(
        n_points=100,
        variables=["O3", "CO"],
        domain=small_domain,
        time_config=short_time_config,
        seed=42,
    )


# =============================================================================
# Observation Fixtures - Profile
# =============================================================================


@pytest.fixture
def profile_observations(small_domain: Domain, short_time_config: TimeConfig) -> xr.Dataset:
    """Profile observations (sondes)."""
    return create_profile_observations(
        n_profiles=5,
        n_levels=30,
        variables=["O3", "temperature"],
        domain=small_domain,
        time_config=short_time_config,
        seed=42,
    )


# =============================================================================
# Observation Fixtures - Swath
# =============================================================================


@pytest.fixture
def swath_observations(small_domain: Domain, short_time_config: TimeConfig) -> xr.Dataset:
    """Swath observations (satellite L2)."""
    return create_swath_observations(
        n_scans=20,
        n_pixels=30,
        variables=["NO2"],
        domain=small_domain,
        time_config=short_time_config,
        seed=42,
    )


# =============================================================================
# Observation Fixtures - Grid
# =============================================================================


@pytest.fixture
def gridded_observations(small_domain: Domain, short_time_config: TimeConfig) -> xr.Dataset:
    """Gridded observations (satellite L3)."""
    return create_gridded_observations(
        variables=["NO2"],
        domain=small_domain,
        time_config=short_time_config,
        seed=42,
    )


# =============================================================================
# Scenario Fixtures
# =============================================================================


@pytest.fixture
def perfect_match_scenario() -> PerfectMatchScenario:
    """Perfect match scenario for testing."""
    return PerfectMatchScenario(
        variables=["O3"],
        domain=Domain(n_lon=18, n_lat=9),
        time_config=TimeConfig(end="2024-01-15 06:00"),
        geometry=DataGeometry.POINT,
        n_obs=10,
        seed=42,
    )


@pytest.fixture
def perfect_match_data(
    perfect_match_scenario: PerfectMatchScenario,
) -> tuple[xr.Dataset, xr.Dataset]:
    """Generate perfect match model and observation data."""
    return perfect_match_scenario.generate()


@pytest.fixture
def bias_scenario() -> BiasScenario:
    """Bias scenario for testing."""
    return BiasScenario(
        variables=["O3"],
        domain=Domain(n_lon=18, n_lat=9),
        time_config=TimeConfig(end="2024-01-15 06:00"),
        bias=5.0,
        geometry=DataGeometry.POINT,
        n_obs=10,
        seed=42,
    )


@pytest.fixture
def bias_data(bias_scenario: BiasScenario) -> tuple[xr.Dataset, xr.Dataset]:
    """Generate bias model and observation data."""
    return bias_scenario.generate()


# =============================================================================
# Helper Fixtures
# =============================================================================


@pytest.fixture
def random_seed() -> int:
    """Fixed random seed for reproducibility."""
    return 42


@pytest.fixture
def rng(random_seed: int) -> np.random.Generator:
    """Numpy random generator with fixed seed."""
    return np.random.default_rng(random_seed)


# =============================================================================
# Parametrized Fixtures for Geometry Types
# =============================================================================


@pytest.fixture(
    params=[
        DataGeometry.POINT,
        DataGeometry.TRACK,
        DataGeometry.PROFILE,
        DataGeometry.SWATH,
        DataGeometry.GRID,
    ]
)
def geometry_type(request: pytest.FixtureRequest) -> DataGeometry:
    """Parametrized fixture for all geometry types."""
    return request.param  # type: ignore[no-any-return]


@pytest.fixture
def observations_for_geometry(
    geometry_type: DataGeometry,
    small_domain: Domain,
    short_time_config: TimeConfig,
) -> xr.Dataset:
    """Create observations for the parametrized geometry type."""
    creators = {
        DataGeometry.POINT: lambda: create_point_observations(
            n_sites=5, variables=["O3"], domain=small_domain, time_config=short_time_config
        ),
        DataGeometry.TRACK: lambda: create_track_observations(
            n_points=50, variables=["O3"], domain=small_domain, time_config=short_time_config
        ),
        DataGeometry.PROFILE: lambda: create_profile_observations(
            n_profiles=3, variables=["O3"], domain=small_domain, time_config=short_time_config
        ),
        DataGeometry.SWATH: lambda: create_swath_observations(
            n_scans=10, variables=["O3"], domain=small_domain, time_config=short_time_config
        ),
        DataGeometry.GRID: lambda: create_gridded_observations(
            variables=["O3"], domain=small_domain, time_config=short_time_config
        ),
    }
    return creators[geometry_type]()
