"""Synthetic model output generators.

This module generates synthetic atmospheric model output (CMAQ-like, WRF-Chem-like)
for testing pairing and analysis components.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import xarray as xr

from davinci_monet.tests.synthetic.generators import (
    Domain,
    TimeConfig,
    VariableSpec,
    add_diurnal_cycle,
    create_coordinate_grid,
    create_level_axis,
    create_time_axis,
    generate_random_field,
    get_variable_spec,
)


@dataclass
class ModelConfig:
    """Configuration for synthetic model output.

    Parameters
    ----------
    domain
        Geographic domain specification.
    time_config
        Time axis configuration.
    variables
        List of variable names to generate.
    n_levels
        Number of vertical levels (0 for surface-only).
    seed
        Random seed for reproducibility.
    add_diurnal
        Whether to add diurnal cycles to variables.
    """

    domain: Domain = field(default_factory=Domain)
    time_config: TimeConfig = field(default_factory=TimeConfig)
    variables: list[str] = field(default_factory=lambda: ["O3", "PM25"])
    n_levels: int = 0
    seed: int | None = 42
    add_diurnal: bool = True


def create_variable_field(
    spec: VariableSpec,
    shape: tuple[int, ...],
    dims: Sequence[str],
    seed: int | None = None,
) -> xr.DataArray:
    """Create a synthetic variable field.

    Parameters
    ----------
    spec
        Variable specification.
    shape
        Shape of the output array.
    dims
        Dimension names corresponding to shape.
    seed
        Random seed for reproducibility.

    Returns
    -------
    xr.DataArray
        Synthetic variable field.
    """
    data = generate_random_field(shape, spec, seed=seed, add_spatial_correlation=True)

    da = xr.DataArray(
        data,
        dims=list(dims),
        attrs={
            "units": spec.units,
            "long_name": spec.long_name,
        },
    )

    return da


def create_model_dataset(
    variables: Sequence[str] | None = None,
    domain: Domain | None = None,
    time_config: TimeConfig | None = None,
    n_levels: int = 0,
    seed: int | None = 42,
    add_diurnal: bool = True,
) -> xr.Dataset:
    """Create a synthetic model output dataset.

    The dataset mimics the structure of atmospheric chemistry model output
    with dimensions (time, level, lat, lon) or (time, lat, lon) for surface.

    Parameters
    ----------
    variables
        List of variable names to generate. Defaults to ["O3", "PM25"].
    domain
        Geographic domain. Defaults to CONUS-like domain.
    time_config
        Time configuration. Defaults to 24 hours.
    n_levels
        Number of vertical levels. 0 for surface-only.
    seed
        Random seed for reproducibility.
    add_diurnal
        Whether to add diurnal cycles.

    Returns
    -------
    xr.Dataset
        Synthetic model dataset.

    Examples
    --------
    >>> ds = create_model_dataset(variables=["O3"], n_levels=0)
    >>> "O3" in ds
    True
    >>> "time" in ds.dims
    True
    """
    if variables is None:
        variables = ["O3", "PM25"]
    if domain is None:
        domain = Domain()
    if time_config is None:
        time_config = TimeConfig()

    # Create coordinates
    lon, lat = create_coordinate_grid(domain)
    time = create_time_axis(time_config)

    coords: dict[str, xr.DataArray] = {
        "lon": lon,
        "lat": lat,
        "time": time,
    }

    # Determine dimensions
    dims: tuple[str, ...]
    shape: tuple[int, ...]
    if n_levels > 0:
        level = create_level_axis(n_levels)
        coords["level"] = level
        dims = ("time", "level", "lat", "lon")
        shape = (len(time), n_levels, len(lat), len(lon))
    else:
        dims = ("time", "lat", "lon")
        shape = (len(time), len(lat), len(lon))

    # Generate data variables
    data_vars: dict[str, xr.DataArray] = {}
    rng = np.random.default_rng(seed)

    for var_name in variables:
        spec = get_variable_spec(var_name)
        var_seed = int(rng.integers(0, 2**31)) if seed is not None else None
        data_vars[var_name] = create_variable_field(spec, shape, dims, seed=var_seed)

    ds = xr.Dataset(data_vars, coords=coords)

    # Add diurnal cycle after coordinates are assigned
    if add_diurnal:
        for var_name in variables:
            if var_name in ["O3", "NO2", "temperature"]:
                peak_hour = 14 if var_name == "O3" else 8 if var_name == "NO2" else 15
                ds[var_name] = add_diurnal_cycle(ds[var_name], amplitude=0.15, peak_hour=peak_hour)

    # Add global attributes
    ds.attrs = {
        "title": "Synthetic Model Output",
        "source": "davinci_monet.tests.synthetic",
        "Conventions": "CF-1.8",
        "history": "Generated for testing",
    }

    return ds


def create_model_dataset_from_config(config: ModelConfig) -> xr.Dataset:
    """Create a synthetic model dataset from a configuration object.

    Parameters
    ----------
    config
        Model configuration.

    Returns
    -------
    xr.Dataset
        Synthetic model dataset.
    """
    return create_model_dataset(
        variables=config.variables,
        domain=config.domain,
        time_config=config.time_config,
        n_levels=config.n_levels,
        seed=config.seed,
        add_diurnal=config.add_diurnal,
    )


def create_surface_model(
    variables: Sequence[str] | None = None,
    domain: Domain | None = None,
    time_config: TimeConfig | None = None,
    seed: int | None = 42,
) -> xr.Dataset:
    """Create surface-only model output (no vertical levels).

    Convenience function for common use case.

    Parameters
    ----------
    variables
        Variables to generate.
    domain
        Geographic domain.
    time_config
        Time configuration.
    seed
        Random seed.

    Returns
    -------
    xr.Dataset
        Surface model dataset with dims (time, lat, lon).
    """
    return create_model_dataset(
        variables=variables,
        domain=domain,
        time_config=time_config,
        n_levels=0,
        seed=seed,
    )


def create_3d_model(
    variables: Sequence[str] | None = None,
    domain: Domain | None = None,
    time_config: TimeConfig | None = None,
    n_levels: int = 30,
    seed: int | None = 42,
) -> xr.Dataset:
    """Create 3D model output with vertical levels.

    Convenience function for 3D model data.

    Parameters
    ----------
    variables
        Variables to generate.
    domain
        Geographic domain.
    time_config
        Time configuration.
    n_levels
        Number of vertical levels.
    seed
        Random seed.

    Returns
    -------
    xr.Dataset
        3D model dataset with dims (time, level, lat, lon).
    """
    return create_model_dataset(
        variables=variables,
        domain=domain,
        time_config=time_config,
        n_levels=n_levels,
        seed=seed,
    )
