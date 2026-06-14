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
from davinci_monet.tests.synthetic.datasets import (
    create_dataset_dataset,
    create_gridded_geometries,
    create_point_geometries,
    create_profile_geometries,
    create_swath_geometries,
    create_track_geometries,
)
from davinci_monet.tests.synthetic.generators import Domain, TimeConfig
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
# Dataset Fixtures
# =============================================================================


@pytest.fixture
def surface_dataset(small_domain: Domain, short_time_config: TimeConfig) -> xr.Dataset:
    """Surface-only dataset dataset."""
    return create_dataset_dataset(
        variables=["O3", "PM25"],
        domain=small_domain,
        time_config=short_time_config,
        n_levels=0,
        seed=42,
    )


@pytest.fixture
def dataset_3d(small_domain: Domain, short_time_config: TimeConfig) -> xr.Dataset:
    """3D dataset dataset with vertical levels."""
    return create_dataset_dataset(
        variables=["O3", "PM25", "NO2"],
        domain=small_domain,
        time_config=short_time_config,
        n_levels=20,
        seed=42,
    )


@pytest.fixture
def dataset_single_var(small_domain: Domain, short_time_config: TimeConfig) -> xr.Dataset:
    """Dataset with single variable (O3)."""
    return create_dataset_dataset(
        variables=["O3"],
        domain=small_domain,
        time_config=short_time_config,
        n_levels=0,
        seed=42,
    )


# =============================================================================
# Dataset Fixtures - Point
# =============================================================================


@pytest.fixture
def point_geometries(small_domain: Domain, short_time_config: TimeConfig) -> xr.Dataset:
    """Point datasets (surface stations)."""
    return create_point_geometries(
        n_sites=10,
        variables=["O3", "PM25"],
        domain=small_domain,
        time_config=short_time_config,
        seed=42,
    )


@pytest.fixture
def point_geometries_single_site(small_domain: Domain, short_time_config: TimeConfig) -> xr.Dataset:
    """Single-site point datasets."""
    return create_point_geometries(
        n_sites=1,
        variables=["O3"],
        domain=small_domain,
        time_config=short_time_config,
        seed=42,
    )


# =============================================================================
# Dataset Fixtures - Track
# =============================================================================


@pytest.fixture
def track_geometries(small_domain: Domain, short_time_config: TimeConfig) -> xr.Dataset:
    """Track datasets (aircraft)."""
    return create_track_geometries(
        n_points=100,
        variables=["O3", "CO"],
        domain=small_domain,
        time_config=short_time_config,
        seed=42,
    )


# =============================================================================
# Dataset Fixtures - Profile
# =============================================================================


@pytest.fixture
def profile_geometries(small_domain: Domain, short_time_config: TimeConfig) -> xr.Dataset:
    """Profile datasets (sondes)."""
    return create_profile_geometries(
        n_profiles=5,
        n_levels=30,
        variables=["O3", "temperature"],
        domain=small_domain,
        time_config=short_time_config,
        seed=42,
    )


# =============================================================================
# Dataset Fixtures - Swath
# =============================================================================


@pytest.fixture
def swath_geometries(small_domain: Domain, short_time_config: TimeConfig) -> xr.Dataset:
    """Swath datasets (satellite L2)."""
    return create_swath_geometries(
        n_scans=20,
        n_pixels=30,
        variables=["NO2"],
        domain=small_domain,
        time_config=short_time_config,
        seed=42,
    )


# =============================================================================
# Dataset Fixtures - Grid
# =============================================================================


@pytest.fixture
def gridded_geometries(small_domain: Domain, short_time_config: TimeConfig) -> xr.Dataset:
    """Gridded datasets (satellite L3)."""
    return create_gridded_geometries(
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
        n_geometry=10,
        seed=42,
    )


@pytest.fixture
def perfect_match_data(
    perfect_match_scenario: PerfectMatchScenario,
) -> tuple[xr.Dataset, xr.Dataset]:
    """Generate perfect match dataset and dataset data."""
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
        n_geometry=10,
        seed=42,
    )


@pytest.fixture
def bias_data(bias_scenario: BiasScenario) -> tuple[xr.Dataset, xr.Dataset]:
    """Generate bias dataset and dataset data."""
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
    return request.param


@pytest.fixture
def geometries_for_geometry(
    geometry_type: DataGeometry,
    small_domain: Domain,
    short_time_config: TimeConfig,
) -> xr.Dataset:
    """Create datasets for the parametrized geometry type."""
    creators = {
        DataGeometry.POINT: lambda: create_point_geometries(
            n_sites=5, variables=["O3"], domain=small_domain, time_config=short_time_config
        ),
        DataGeometry.TRACK: lambda: create_track_geometries(
            n_points=50, variables=["O3"], domain=small_domain, time_config=short_time_config
        ),
        DataGeometry.PROFILE: lambda: create_profile_geometries(
            n_profiles=3, variables=["O3"], domain=small_domain, time_config=short_time_config
        ),
        DataGeometry.SWATH: lambda: create_swath_geometries(
            n_scans=10, variables=["O3"], domain=small_domain, time_config=short_time_config
        ),
        DataGeometry.GRID: lambda: create_gridded_geometries(
            variables=["O3"], domain=small_domain, time_config=short_time_config
        ),
    }
    return creators[geometry_type]()
