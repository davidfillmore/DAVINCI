"""Base data generators for synthetic test data.

This module provides foundational utilities for generating synthetic
atmospheric data including coordinate grids, time axes, and random
sampling utilities.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import xarray as xr


@dataclass
class Domain:
    """Geographic domain specification.

    Parameters
    ----------
    lon_min
        Western boundary longitude (degrees).
    lon_max
        Eastern boundary longitude (degrees).
    lat_min
        Southern boundary latitude (degrees).
    lat_max
        Northern boundary latitude (degrees).
    n_lon
        Number of longitude grid points.
    n_lat
        Number of latitude grid points.

    Examples
    --------
    >>> domain = Domain(lon_min=-130, lon_max=-60, lat_min=20, lat_max=55)
    >>> domain.lon_range
    70.0
    """

    lon_min: float = -130.0
    lon_max: float = -60.0
    lat_min: float = 20.0
    lat_max: float = 55.0
    n_lon: int = 72
    n_lat: int = 36

    @property
    def lon_range(self) -> float:
        """Longitude range in degrees."""
        return self.lon_max - self.lon_min

    @property
    def lat_range(self) -> float:
        """Latitude range in degrees."""
        return self.lat_max - self.lat_min

    @property
    def lon_resolution(self) -> float:
        """Longitude resolution in degrees."""
        return self.lon_range / (self.n_lon - 1) if self.n_lon > 1 else self.lon_range

    @property
    def lat_resolution(self) -> float:
        """Latitude resolution in degrees."""
        return self.lat_range / (self.n_lat - 1) if self.n_lat > 1 else self.lat_range

    @property
    def center(self) -> tuple[float, float]:
        """Domain center as (lon, lat)."""
        return (
            (self.lon_min + self.lon_max) / 2,
            (self.lat_min + self.lat_max) / 2,
        )


@dataclass
class TimeConfig:
    """Time axis configuration.

    Parameters
    ----------
    start
        Start time as datetime, string, or numpy datetime64.
    end
        End time as datetime, string, or numpy datetime64.
    freq
        Time frequency string (e.g., '1h', '3h', '1D').

    Examples
    --------
    >>> config = TimeConfig(start="2024-01-01", end="2024-01-03", freq="1h")
    >>> len(config.time_range)
    49
    """

    start: str | datetime | np.datetime64 = "2024-01-01"
    end: str | datetime | np.datetime64 = "2024-01-02"
    freq: str = "1h"

    @property
    def time_range(self) -> pd.DatetimeIndex:
        """Generate the time range as DatetimeIndex."""
        return pd.date_range(start=self.start, end=self.end, freq=self.freq)

    @property
    def n_times(self) -> int:
        """Number of time steps."""
        return len(self.time_range)


@dataclass
class VariableSpec:
    """Specification for a synthetic variable.

    Parameters
    ----------
    name
        Variable name.
    units
        Variable units.
    long_name
        Descriptive name.
    mean
        Mean value for random generation.
    std
        Standard deviation for random generation.
    min_val
        Minimum physical value (for clipping).
    max_val
        Maximum physical value (for clipping).
    """

    name: str
    units: str = "units"
    long_name: str = ""
    mean: float = 50.0
    std: float = 10.0
    min_val: float = 0.0
    max_val: float = 1000.0

    def __post_init__(self) -> None:
        if not self.long_name:
            self.long_name = self.name


# Common atmospheric variable specifications
VARIABLE_SPECS: dict[str, VariableSpec] = {
    "O3": VariableSpec(
        name="O3",
        units="ppbv",
        long_name="Ozone",
        mean=40.0,
        std=15.0,
        min_val=0.0,
        max_val=200.0,
    ),
    "PM25": VariableSpec(
        name="PM25",
        units="ug/m3",
        long_name="Fine Particulate Matter",
        mean=12.0,
        std=8.0,
        min_val=0.0,
        max_val=500.0,
    ),
    "NO2": VariableSpec(
        name="NO2",
        units="ppbv",
        long_name="Nitrogen Dioxide",
        mean=15.0,
        std=10.0,
        min_val=0.0,
        max_val=200.0,
    ),
    "CO": VariableSpec(
        name="CO",
        units="ppbv",
        long_name="Carbon Monoxide",
        mean=200.0,
        std=100.0,
        min_val=0.0,
        max_val=5000.0,
    ),
    "SO2": VariableSpec(
        name="SO2",
        units="ppbv",
        long_name="Sulfur Dioxide",
        mean=5.0,
        std=5.0,
        min_val=0.0,
        max_val=100.0,
    ),
    "temperature": VariableSpec(
        name="temperature",
        units="K",
        long_name="Temperature",
        mean=288.0,
        std=15.0,
        min_val=200.0,
        max_val=330.0,
    ),
    "pressure": VariableSpec(
        name="pressure",
        units="hPa",
        long_name="Pressure",
        mean=1013.25,
        std=20.0,
        min_val=100.0,
        max_val=1100.0,
    ),
}


def get_variable_spec(name: str) -> VariableSpec:
    """Get variable specification by name.

    If the variable is not in the predefined specs, returns a default spec.

    Parameters
    ----------
    name
        Variable name.

    Returns
    -------
    VariableSpec
        Variable specification.
    """
    if name in VARIABLE_SPECS:
        return VARIABLE_SPECS[name]
    return VariableSpec(name=name)


def create_coordinate_grid(domain: Domain) -> tuple[xr.DataArray, xr.DataArray]:
    """Create longitude and latitude coordinate arrays.

    Parameters
    ----------
    domain
        Domain specification.

    Returns
    -------
    tuple[xr.DataArray, xr.DataArray]
        Tuple of (longitude, latitude) DataArrays.

    Examples
    --------
    >>> domain = Domain(lon_min=-100, lon_max=-90, lat_min=30, lat_max=40, n_lon=11, n_lat=11)
    >>> lon, lat = create_coordinate_grid(domain)
    >>> lon.values[0], lon.values[-1]
    (-100.0, -90.0)
    """
    lon = xr.DataArray(
        np.linspace(domain.lon_min, domain.lon_max, domain.n_lon),
        dims=["lon"],
        attrs={"units": "degrees_east", "long_name": "Longitude"},
    )
    lat = xr.DataArray(
        np.linspace(domain.lat_min, domain.lat_max, domain.n_lat),
        dims=["lat"],
        attrs={"units": "degrees_north", "long_name": "Latitude"},
    )
    return lon, lat


def create_time_axis(config: TimeConfig) -> xr.DataArray:
    """Create a time coordinate array.

    Parameters
    ----------
    config
        Time configuration.

    Returns
    -------
    xr.DataArray
        Time coordinate array.
    """
    return xr.DataArray(
        config.time_range.values,
        dims=["time"],
        attrs={"long_name": "Time"},
    )


def create_level_axis(
    n_levels: int = 30,
    surface_pressure: float = 1013.25,
    top_pressure: float = 10.0,
    ascending_pressure: bool = False,
) -> xr.DataArray:
    """Create a vertical level coordinate array.

    Uses logarithmic spacing typical of atmospheric datasets.

    Parameters
    ----------
    n_levels
        Number of vertical levels.
    surface_pressure
        Surface pressure in hPa.
    top_pressure
        Top-of-atmosphere pressure in hPa.
    ascending_pressure
        If False (default), pressure *decreases* with index (surface at index 0,
        top of atmosphere last). If True, reproduce the **CESM hybrid
        sigma-pressure convention** where pressure *increases* with index (TOA at
        index 0, surface at the last index); this is the ordering that triggers
        the ``surface_idx = -1`` branch of surface extraction and is otherwise
        not represented in the synthetic data (see the CESM vertical-coordinate
        warning in CLAUDE.md).

    Returns
    -------
    xr.DataArray
        Vertical level array in hPa.
    """
    # Log-spaced levels from surface to top
    levels = np.logspace(
        np.log10(surface_pressure),
        np.log10(top_pressure),
        n_levels,
    )
    if ascending_pressure:
        # Reverse so pressure increases with index (TOA -> surface), matching the
        # CESM ordering where the surface is the last level.
        levels = levels[::-1]
    return xr.DataArray(
        levels,
        dims=["level"],
        attrs={"units": "hPa", "long_name": "Pressure Level", "positive": "down"},
    )


def random_locations_in_domain(
    domain: Domain,
    n_points: int,
    seed: int | None = None,
) -> tuple[npt.NDArray[np.floating[Any]], npt.NDArray[np.floating[Any]]]:
    """Generate random locations within a domain.

    Parameters
    ----------
    domain
        Domain specification.
    n_points
        Number of random points to generate.
    seed
        Random seed for reproducibility.

    Returns
    -------
    tuple[ndarray, ndarray]
        Tuple of (longitudes, latitudes) arrays.

    Examples
    --------
    >>> domain = Domain(lon_min=-100, lon_max=-90, lat_min=30, lat_max=40)
    >>> lons, lats = random_locations_in_domain(domain, 5, seed=42)
    >>> len(lons)
    5
    """
    rng = np.random.default_rng(seed)
    lons = rng.uniform(domain.lon_min, domain.lon_max, n_points)
    lats = rng.uniform(domain.lat_min, domain.lat_max, n_points)
    return lons, lats


def generate_random_field(
    shape: tuple[int, ...],
    spec: VariableSpec,
    seed: int | None = None,
    add_spatial_correlation: bool = True,
) -> npt.NDArray[np.floating[Any]]:
    """Generate a random field with realistic characteristics.

    Parameters
    ----------
    shape
        Shape of the output array.
    spec
        Variable specification with mean, std, and bounds.
    seed
        Random seed for reproducibility.
    add_spatial_correlation
        If True, add spatial smoothing for more realistic fields.

    Returns
    -------
    ndarray
        Random field with values clipped to physical bounds.
    """
    rng = np.random.default_rng(seed)
    data = rng.normal(spec.mean, spec.std, shape)

    # Add spatial correlation via simple smoothing if requested
    if add_spatial_correlation and len(shape) >= 2:
        from scipy.ndimage import gaussian_filter

        # Apply Gaussian smoothing to last two dimensions (assumed lat/lon)
        sigma = 1.5
        for i in range(data.shape[0] if len(shape) > 2 else 1):
            if len(shape) > 2:
                data[i] = gaussian_filter(data[i], sigma=sigma)
            else:
                data = gaussian_filter(data, sigma=sigma)

    # Clip to physical bounds
    data = np.clip(data, spec.min_val, spec.max_val)
    return data


def add_diurnal_cycle(
    data: xr.DataArray,
    amplitude: float = 0.2,
    peak_hour: int = 14,
) -> xr.DataArray:
    """Add a diurnal cycle to time-varying data.

    Parameters
    ----------
    data
        Input DataArray with a 'time' dimension and datetime coordinates.
    amplitude
        Relative amplitude of the diurnal cycle (fraction of mean).
    peak_hour
        Hour of maximum (0-23).

    Returns
    -------
    xr.DataArray
        Data with diurnal cycle added.
    """
    if "time" not in data.dims:
        return data

    # Check if time coordinate exists and has datetime accessor
    if "time" not in data.coords:
        return data

    try:
        hours = data.coords["time"].dt.hour.values
    except AttributeError:
        # time coordinate doesn't have datetime accessor
        return data

    # Sinusoidal diurnal cycle
    cycle = amplitude * np.sin(2 * np.pi * (hours - peak_hour + 6) / 24)

    # Broadcast to data shape
    shape = [1] * len(data.dims)
    time_idx = list(data.dims).index("time")
    shape[time_idx] = len(hours)
    cycle = cycle.reshape(shape)

    mean_val = float(data.mean())
    result: xr.DataArray = data + mean_val * cycle
    # Preserve attributes
    result.attrs = data.attrs
    return result


def add_random_noise(
    data: xr.DataArray,
    noise_fraction: float = 0.1,
    seed: int | None = None,
) -> xr.DataArray:
    """Add random noise to data.

    Parameters
    ----------
    data
        Input DataArray.
    noise_fraction
        Noise standard deviation as fraction of data std (or mean if std is 0).
    seed
        Random seed for reproducibility.

    Returns
    -------
    xr.DataArray
        Data with noise added.
    """
    rng = np.random.default_rng(seed)
    data_std = float(data.std())
    # If data has no variance, use mean as geometry scale
    if data_std == 0:
        data_std = abs(float(data.mean())) if float(data.mean()) != 0 else 1.0
    noise_std = data_std * noise_fraction
    noise = rng.normal(0, noise_std, data.shape)
    return data + noise
