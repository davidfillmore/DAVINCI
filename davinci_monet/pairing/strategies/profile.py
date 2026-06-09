"""Profile-to-grid pairing strategy.

This module implements pairing for profile observations (sondes, lidar)
with gridded model output.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import xarray as xr

from davinci_monet.core.exceptions import PairingError
from davinci_monet.core.protocols import DataGeometry
from davinci_monet.core.types import TimeDelta
from davinci_monet.pairing.strategies.base import BasePairingStrategy


class ProfileStrategy(BasePairingStrategy):
    """Pairing strategy for profile observations.

    Matches vertical profile observations (sondes, lidar) to model
    columns at the profile location.

    The strategy:
    1. Finds nearest model grid column for profile location
    2. Interpolates model temporally to profile time
    3. Interpolates both model and obs to common vertical grid
    4. Creates paired dataset with aligned vertical profiles

    Examples
    --------
    >>> strategy = ProfileStrategy()
    >>> paired = strategy.pair(model_data, sonde_data,
    ...                        vertical_method='log')
    """

    @property
    def geometry(self) -> DataGeometry:
        """Return PROFILE geometry."""
        return DataGeometry.PROFILE

    def pair(
        self,
        model: xr.Dataset,
        obs: xr.Dataset,
        radius_of_influence: float | None = None,
        time_tolerance: TimeDelta | None = None,
        vertical_method: str = "linear",
        horizontal_method: str = "nearest",
        **kwargs: Any,
    ) -> xr.Dataset:
        """Pair profile observations with model grid.

        Parameters
        ----------
        model
            Model Dataset with dims (time, z, lat, lon).
        obs
            Observation Dataset with dims (time, level) or (level,).
        radius_of_influence
            Maximum horizontal distance in meters.
        time_tolerance
            Maximum time difference for matching.
        vertical_method
            Vertical interpolation method ('nearest', 'linear', 'log').
        horizontal_method
            Horizontal matching method.
        **kwargs
            Additional options:
            - level_coord: str, name of vertical coordinate in obs
            - interp_to_obs_levels: bool, whether to interp model to obs levels

        Returns
        -------
        xr.Dataset
            Paired dataset with model and obs profiles.
        """
        level_coord = kwargs.get("level_coord", "level")
        interp_to_obs_levels = kwargs.get("interp_to_obs_levels", True)

        # Get coordinates
        model_lat, model_lon = self._get_model_coords(model)
        obs_lat, obs_lon = self._get_obs_coords(obs)

        # Get profile location (may be single point or time-varying)
        if obs_lat.ndim == 0:
            # Single location
            profile_lat = float(obs_lat.values)
            profile_lon = float(obs_lon.values)
        else:
            # Take first value (assume profile at fixed location)
            profile_lat = float(obs_lat.values.flat[0])
            profile_lon = float(obs_lon.values.flat[0])

        # Find nearest model column
        lat_idx, lon_idx = self._find_nearest_indices(
            model_lat,
            model_lon,
            xr.DataArray([profile_lat]),
            xr.DataArray([profile_lon]),
            radius_of_influence=radius_of_influence,
        )

        if lat_idx.values[0] < 0:
            raise PairingError(
                f"Profile location ({profile_lat}, {profile_lon}) is outside "
                f"radius of influence ({radius_of_influence}m) from model grid"
            )

        # Extract model column at profile location
        model_column = self._extract_column(
            model,
            model_lat,
            model_lon,
            int(lat_idx.values[0]),
            int(lon_idx.values[0]),
        )

        # Interpolate model to observation times if needed
        if "time" in model_column.dims and "time" in obs.dims:
            obs_times = obs["time"]
            model_column = self._interpolate_time(
                model_column, obs_times, "nearest", time_tolerance=time_tolerance
            )

        # Handle vertical interpolation
        if interp_to_obs_levels and level_coord in obs.dims:
            obs_levels = obs[level_coord]
            model_column = self._interpolate_vertical(
                model_column, obs_levels, level_coord="z", method=vertical_method
            )

        # Create paired output
        paired = self._create_paired_output(obs, model_column, level_coord)

        return paired

    def _extract_column(
        self,
        model: xr.Dataset,
        model_lat: xr.DataArray,
        model_lon: xr.DataArray,
        lat_idx: int,
        lon_idx: int,
    ) -> xr.Dataset:
        """Extract a single model column.

        Parameters
        ----------
        model
            Model dataset.
        model_lat, model_lon
            Model coordinate arrays.
        lat_idx, lon_idx
            Grid indices for column location.

        Returns
        -------
        xr.Dataset
            Model data at single column with (time, z) dims.
        """
        # Determine dimension names
        if model_lat.ndim == 1:
            lat_dim = model_lat.dims[0]
            lon_dim = model_lon.dims[0]
        else:
            lat_dim = model_lat.dims[0]
            lon_dim = model_lat.dims[1]

        # Extract column
        return model.isel({lat_dim: lat_idx, lon_dim: lon_idx})

    def _create_paired_output(
        self,
        obs: xr.Dataset,
        model_column: xr.Dataset,
        level_coord: str,
    ) -> xr.Dataset:
        """Create the final paired output dataset.

        Parameters
        ----------
        obs
            Observation dataset.
        model_column
            Model column data.
        level_coord
            Name of vertical coordinate.

        Returns
        -------
        xr.Dataset
            Combined dataset.
        """
        # Combine coordinates
        coords = dict(obs.coords)
        for coord in model_column.coords:
            if coord not in coords:
                coords[coord] = model_column.coords[coord]

        # Combine data variables
        data_vars: dict[str, Any] = {}

        # Add observation variables
        for var in obs.data_vars:
            data_vars[str(var)] = obs[var]

        # Add model variables with prefix
        for var in model_column.data_vars:
            model_var_name = f"model_{var}"
            data_vars[model_var_name] = model_column[var]

        return xr.Dataset(data_vars, coords=coords)
