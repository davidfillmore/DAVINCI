"""Synthetic observation generators.

This module generates synthetic observations for different geometries:
- Point: Surface stations, ground sites (time, site)
- Track: Aircraft, mobile platforms (time,) with lat/lon/alt coords
- Profile: Sondes, vertical profiles (time, level)
- Swath: Satellite L2 products (time, scanline, pixel)
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import numpy.typing as npt
import xarray as xr

from davinci_monet.core.protocols import DataGeometry
from davinci_monet.tests.synthetic.generators import (
    Domain,
    TimeConfig,
    VariableSpec,
    add_diurnal_cycle,
    add_random_noise,
    create_level_axis,
    create_time_axis,
    generate_random_field,
    get_variable_spec,
    random_locations_in_domain,
)


def create_point_observations(
    n_sites: int = 20,
    variables: Sequence[str] | None = None,
    domain: Domain | None = None,
    time_config: TimeConfig | None = None,
    seed: int | None = 42,
    site_prefix: str = "SITE",
) -> xr.Dataset:
    """Create synthetic point observations (surface stations).

    Generates data with dimensions (time, site) and lat/lon coordinates
    for each site.

    Parameters
    ----------
    n_sites
        Number of observation sites.
    variables
        List of variable names. Defaults to ["O3", "PM25"].
    domain
        Geographic domain for site locations.
    time_config
        Time axis configuration.
    seed
        Random seed for reproducibility.
    site_prefix
        Prefix for site IDs.

    Returns
    -------
    xr.Dataset
        Point observation dataset with geometry attribute.

    Examples
    --------
    >>> ds = create_point_observations(n_sites=5, variables=["O3"])
    >>> ds.dims
    Frozen({'time': ..., 'site': 5})
    >>> ds.attrs["geometry"]
    'point'
    """
    if variables is None:
        variables = ["O3", "PM25"]
    if domain is None:
        domain = Domain()
    if time_config is None:
        time_config = TimeConfig()

    rng = np.random.default_rng(seed)

    # Create coordinates
    time = create_time_axis(time_config)
    lons, lats = random_locations_in_domain(domain, n_sites, seed=seed)
    site_ids = [f"{site_prefix}{i:03d}" for i in range(n_sites)]

    # Create site dimension
    site = xr.DataArray(range(n_sites), dims=["site"])

    coords: dict[str, Any] = {
        "time": time,
        "site": site,
        "latitude": (["site"], lats, {"units": "degrees_north"}),
        "longitude": (["site"], lons, {"units": "degrees_east"}),
        "site_id": (["site"], site_ids),
    }

    # Generate data variables
    data_vars: dict[str, tuple[list[str], npt.NDArray[Any], dict[str, str]]] = {}
    shape = (len(time), n_sites)

    for var_name in variables:
        spec = get_variable_spec(var_name)
        var_seed = int(rng.integers(0, 2**31)) if seed is not None else None
        data = generate_random_field(shape, spec, seed=var_seed, add_spatial_correlation=False)
        data_vars[var_name] = (
            ["time", "site"],
            data,
            {"units": spec.units, "long_name": spec.long_name},
        )

    ds = xr.Dataset(data_vars, coords=coords)

    # Add diurnal cycle and clip to physical bounds
    for var_name in variables:
        if var_name in ["O3", "NO2", "temperature"]:
            peak = 14 if var_name == "O3" else 8
            ds[var_name] = add_diurnal_cycle(ds[var_name], amplitude=0.15, peak_hour=peak)
            # Clip to physical bounds after diurnal cycle
            spec = get_variable_spec(var_name)
            ds[var_name] = ds[var_name].clip(min=spec.min_val, max=spec.max_val)

    ds.attrs = {
        "title": "Synthetic Point Observations",
        "geometry": "point",
        "source": "davinci_monet.tests.synthetic",
    }

    return ds


def create_track_observations(
    n_points: int = 500,
    variables: Sequence[str] | None = None,
    domain: Domain | None = None,
    time_config: TimeConfig | None = None,
    altitude_range: tuple[float, float] = (500.0, 8000.0),
    seed: int | None = 42,
) -> xr.Dataset:
    """Create synthetic track observations (aircraft).

    Generates data with dimension (time,) and lat/lon/alt as coordinates.
    The track follows a realistic flight pattern.

    Parameters
    ----------
    n_points
        Number of points along track.
    variables
        List of variable names. Defaults to ["O3", "CO"].
    domain
        Geographic domain for track extent.
    time_config
        Time axis configuration.
    altitude_range
        (min, max) altitude in meters.
    seed
        Random seed for reproducibility.

    Returns
    -------
    xr.Dataset
        Track observation dataset with geometry attribute.
    """
    if variables is None:
        variables = ["O3", "CO"]
    if domain is None:
        domain = Domain()
    if time_config is None:
        time_config = TimeConfig()

    rng = np.random.default_rng(seed)

    # Generate track timestamps
    full_time = create_time_axis(time_config)
    # Subsample or interpolate to n_points
    indices = np.linspace(0, len(full_time) - 1, n_points).astype(int)
    time_vals = full_time.values[indices]

    # Generate a smooth track (spiral/survey pattern)
    t = np.linspace(0, 4 * np.pi, n_points)
    center_lon, center_lat = domain.center
    radius_lon = domain.lon_range * 0.3
    radius_lat = domain.lat_range * 0.3

    lons = center_lon + radius_lon * np.sin(t) * np.linspace(0.5, 1.0, n_points)
    lats = center_lat + radius_lat * np.cos(t) * np.linspace(0.5, 1.0, n_points)

    # Add some noise to make it more realistic
    lons += rng.normal(0, 0.1, n_points)
    lats += rng.normal(0, 0.05, n_points)

    # Generate altitude profile (takeoff, cruise, landing pattern)
    alt_min, alt_max = altitude_range
    alt_profile = np.zeros(n_points)
    cruise_start = n_points // 6
    cruise_end = 5 * n_points // 6

    # Takeoff
    alt_profile[:cruise_start] = np.linspace(alt_min, alt_max, cruise_start)
    # Cruise with some variation
    alt_profile[cruise_start:cruise_end] = alt_max + rng.normal(0, 200, cruise_end - cruise_start)
    # Landing
    alt_profile[cruise_end:] = np.linspace(alt_max, alt_min, n_points - cruise_end)

    coords: dict[str, Any] = {
        "time": (["time"], time_vals),
        "latitude": (["time"], lats, {"units": "degrees_north"}),
        "longitude": (["time"], lons, {"units": "degrees_east"}),
        "altitude": (
            ["time"],
            alt_profile,
            {"units": "m", "long_name": "Altitude above sea level"},
        ),
    }

    # Generate data variables
    data_vars: dict[str, tuple[list[str], npt.NDArray[Any], dict[str, str]]] = {}

    for var_name in variables:
        spec = get_variable_spec(var_name)
        var_seed = int(rng.integers(0, 2**31)) if seed is not None else None
        data = generate_random_field(
            (n_points,), spec, seed=var_seed, add_spatial_correlation=False
        )
        data_vars[var_name] = (["time"], data, {"units": spec.units, "long_name": spec.long_name})

    ds = xr.Dataset(data_vars, coords=coords)

    ds.attrs = {
        "title": "Synthetic Track Observations",
        "geometry": "track",
        "source": "davinci_monet.tests.synthetic",
        "platform": "aircraft",
    }

    return ds


def create_profile_observations(
    n_profiles: int = 10,
    n_levels: int = 50,
    variables: Sequence[str] | None = None,
    domain: Domain | None = None,
    time_config: TimeConfig | None = None,
    seed: int | None = 42,
) -> xr.Dataset:
    """Create synthetic profile observations (sondes).

    Generates data with dimensions (time, level) for vertical profiles
    at specific locations.

    Parameters
    ----------
    n_profiles
        Number of profile soundings.
    n_levels
        Number of vertical levels per profile.
    variables
        List of variable names. Defaults to ["O3", "temperature"].
    domain
        Geographic domain for launch sites.
    time_config
        Time configuration (profiles distributed across time range).
    seed
        Random seed for reproducibility.

    Returns
    -------
    xr.Dataset
        Profile observation dataset with geometry attribute.
    """
    if variables is None:
        variables = ["O3", "temperature"]
    if domain is None:
        domain = Domain()
    if time_config is None:
        time_config = TimeConfig()

    rng = np.random.default_rng(seed)

    # Select profile times from the time range
    full_time = time_config.time_range
    profile_indices = np.linspace(0, len(full_time) - 1, n_profiles).astype(int)
    profile_times = full_time[profile_indices]

    # Random launch locations
    lons, lats = random_locations_in_domain(domain, n_profiles, seed=seed)

    # Vertical levels
    level = create_level_axis(n_levels)

    coords: dict[str, Any] = {
        "time": (["time"], profile_times.values),
        "level": level,
        "latitude": (["time"], lats, {"units": "degrees_north"}),
        "longitude": (["time"], lons, {"units": "degrees_east"}),
    }

    # Generate data variables
    data_vars: dict[str, tuple[list[str], npt.NDArray[Any], dict[str, str]]] = {}
    shape = (n_profiles, n_levels)

    for var_name in variables:
        spec = get_variable_spec(var_name)
        var_seed = int(rng.integers(0, 2**31)) if seed is not None else None
        data = generate_random_field(shape, spec, seed=var_seed, add_spatial_correlation=False)

        # Add vertical structure
        if var_name == "O3":
            # Ozone increases with altitude in stratosphere
            vertical_factor = np.linspace(1.0, 3.0, n_levels)
            data = data * vertical_factor
        elif var_name == "temperature":
            # Temperature decreases with altitude (lapse rate)
            vertical_factor = np.linspace(1.0, 0.75, n_levels)
            data = data * vertical_factor

        data_vars[var_name] = (
            ["time", "level"],
            data,
            {"units": spec.units, "long_name": spec.long_name},
        )

    ds = xr.Dataset(data_vars, coords=coords)

    ds.attrs = {
        "title": "Synthetic Profile Observations",
        "geometry": "profile",
        "source": "davinci_monet.tests.synthetic",
        "platform": "ozonesonde",
    }

    return ds


def create_swath_observations(
    n_scans: int = 100,
    n_pixels: int = 60,
    variables: Sequence[str] | None = None,
    domain: Domain | None = None,
    time_config: TimeConfig | None = None,
    seed: int | None = 42,
) -> xr.Dataset:
    """Create synthetic swath observations (satellite L2).

    Generates data with dimensions (time, scanline, pixel) mimicking
    satellite swath data.

    Parameters
    ----------
    n_scans
        Number of scanlines (along-track).
    n_pixels
        Number of pixels per scanline (cross-track).
    variables
        List of variable names. Defaults to ["NO2"].
    domain
        Geographic domain for swath coverage.
    time_config
        Time configuration.
    seed
        Random seed for reproducibility.

    Returns
    -------
    xr.Dataset
        Swath observation dataset with geometry attribute.
    """
    if variables is None:
        variables = ["NO2"]
    if domain is None:
        domain = Domain()
    if time_config is None:
        time_config = TimeConfig()

    rng = np.random.default_rng(seed)

    # Generate scanline times
    full_time = time_config.time_range
    scan_indices = np.linspace(0, len(full_time) - 1, n_scans).astype(int)
    scan_times = full_time[scan_indices]

    # Generate 2D lat/lon arrays for swath
    # Satellite track goes roughly south to north
    track_lats = np.linspace(domain.lat_min, domain.lat_max, n_scans)
    track_lons = np.linspace(
        domain.lon_min + domain.lon_range * 0.3, domain.lon_min + domain.lon_range * 0.7, n_scans
    )

    # Cross-track pixels
    swath_width = 15.0  # degrees
    pixel_offsets = np.linspace(-swath_width / 2, swath_width / 2, n_pixels)

    # Create 2D lat/lon arrays
    lats_2d = np.zeros((n_scans, n_pixels))
    lons_2d = np.zeros((n_scans, n_pixels))

    for i in range(n_scans):
        lats_2d[i, :] = track_lats[i] + rng.normal(0, 0.1, n_pixels)
        lons_2d[i, :] = track_lons[i] + pixel_offsets

    coords: dict[str, Any] = {
        "time": (["scanline"], scan_times.values),
        "scanline": range(n_scans),
        "pixel": range(n_pixels),
        "latitude": (["scanline", "pixel"], lats_2d, {"units": "degrees_north"}),
        "longitude": (["scanline", "pixel"], lons_2d, {"units": "degrees_east"}),
    }

    # Generate data variables
    data_vars: dict[str, tuple[list[str], npt.NDArray[Any], dict[str, str]]] = {}
    shape = (n_scans, n_pixels)

    for var_name in variables:
        spec = get_variable_spec(var_name)
        var_seed = int(rng.integers(0, 2**31)) if seed is not None else None
        data = generate_random_field(shape, spec, seed=var_seed, add_spatial_correlation=True)
        data_vars[var_name] = (
            ["scanline", "pixel"],
            data,
            {"units": spec.units, "long_name": spec.long_name},
        )

    # Add quality flag
    qa_flag = rng.choice([0, 1, 2], size=shape, p=[0.8, 0.15, 0.05])
    data_vars["qa_flag"] = (
        ["scanline", "pixel"],
        qa_flag.astype(np.int8),
        {"long_name": "Quality Flag", "flag_values": "0=good, 1=suspect, 2=bad"},
    )

    ds = xr.Dataset(data_vars, coords=coords)

    ds.attrs = {
        "title": "Synthetic Swath Observations",
        "geometry": "swath",
        "source": "davinci_monet.tests.synthetic",
        "platform": "satellite",
        "sensor": "synthetic",
    }

    return ds


def create_gridded_observations(
    variables: Sequence[str] | None = None,
    domain: Domain | None = None,
    time_config: TimeConfig | None = None,
    seed: int | None = 42,
) -> xr.Dataset:
    """Create synthetic gridded observations (satellite L3 / reanalysis).

    Generates data with dimensions (time, lat, lon) on a regular grid.

    Parameters
    ----------
    variables
        List of variable names. Defaults to ["NO2"].
    domain
        Geographic domain.
    time_config
        Time configuration.
    seed
        Random seed for reproducibility.

    Returns
    -------
    xr.Dataset
        Gridded observation dataset with geometry attribute.
    """
    if variables is None:
        variables = ["NO2"]
    if domain is None:
        domain = Domain(n_lon=36, n_lat=18)  # Coarser grid typical of L3
    if time_config is None:
        time_config = TimeConfig(freq="1D")  # Daily typical for L3

    # Import from models since structure is similar
    from davinci_monet.tests.synthetic.models import create_model_dataset

    ds = create_model_dataset(
        variables=variables,
        domain=domain,
        time_config=time_config,
        n_levels=0,
        seed=seed,
        add_diurnal=False,
    )

    ds.attrs = {
        "title": "Synthetic Gridded Observations",
        "geometry": "grid",
        "source": "davinci_monet.tests.synthetic",
        "platform": "satellite",
        "product_level": "L3",
    }

    return ds
