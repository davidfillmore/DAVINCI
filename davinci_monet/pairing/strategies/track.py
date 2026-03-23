"""Track-to-grid pairing strategy.

This module implements pairing for track observations (aircraft, mobile
platforms) with gridded model output, including 3D interpolation.
"""

from __future__ import annotations

import os
from typing import Any, Hashable

import dask
import numpy as np
import xarray as xr

from davinci_monet.core.exceptions import PairingError
from davinci_monet.core.protocols import DataGeometry
from davinci_monet.core.types import TimeDelta
from davinci_monet.pairing.strategies.base import BasePairingStrategy


def altitude_to_pressure(
    altitude_m: np.ndarray[Any, np.dtype[Any]],
) -> np.ndarray[Any, np.dtype[Any]]:
    """Convert altitude (meters) to pressure (hPa) using standard atmosphere.

    Uses the barometric formula for the troposphere (valid up to ~11 km):
    P = P0 * (1 - L*h/T0)^(g*M/(R*L))

    Parameters
    ----------
    altitude_m
        Altitude in meters above sea level.

    Returns
    -------
    np.ndarray
        Pressure in hPa.
    """
    # Standard atmosphere constants
    P0 = 1013.25  # Sea level pressure (hPa)
    L = 0.0065  # Temperature lapse rate (K/m)
    T0 = 288.15  # Sea level temperature (K)
    g = 9.80665  # Gravitational acceleration (m/s²)
    M = 0.0289644  # Molar mass of air (kg/mol)
    R = 8.31447  # Gas constant (J/(mol·K))

    exponent = g * M / (R * L)  # ≈ 5.2559

    # Barometric formula
    pressure = P0 * (1 - L * altitude_m / T0) ** exponent

    return pressure


class TrackStrategy(BasePairingStrategy):
    """Pairing strategy for track observations.

    Matches moving observations (aircraft, mobile platforms) to model
    grid cells along their trajectory, with 3D interpolation.

    The strategy:
    1. For each track point, finds nearest model grid cell
    2. Interpolates model temporally to track time
    3. Interpolates model vertically to track altitude
    4. Creates paired dataset with aligned values

    Examples
    --------
    >>> strategy = TrackStrategy()
    >>> paired = strategy.pair(model_data, aircraft_data,
    ...                        vertical_method='linear')
    """

    @property
    def geometry(self) -> DataGeometry:
        """Return TRACK geometry."""
        return DataGeometry.TRACK

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
        """Pair track observations with model grid.

        Parameters
        ----------
        model
            Model Dataset with dims (time, z, lat, lon).
        obs
            Observation Dataset with dims (time,) and lat/lon/alt coords.
        radius_of_influence
            Maximum horizontal distance in meters.
        time_tolerance
            Maximum time difference for matching.
        vertical_method
            Vertical interpolation method ('nearest', 'linear', 'log').
        horizontal_method
            Horizontal matching method ('nearest' only currently).
        **kwargs
            Additional options:
            - pressure_var: str, name of pressure variable for vertical interp
            - altitude_var: str, name of altitude variable in obs

        Returns
        -------
        xr.Dataset
            Paired dataset with model values along track.
        """
        pressure_var = kwargs.get("pressure_var", "pressure")
        altitude_var = kwargs.get("altitude_var", "altitude")

        # Get coordinates
        model_lat, model_lon = self._get_model_coords(model)
        obs_lat, obs_lon = self._get_obs_coords(obs)

        # Get observation times
        if "time" not in obs.dims:
            raise PairingError("Track observations must have 'time' dimension")
        obs_times = obs["time"]
        n_points = len(obs_times)

        # Find nearest model grid cell for each track point
        lat_idx, lon_idx = self._find_nearest_indices(
            model_lat,
            model_lon,
            obs_lat,
            obs_lon,
            radius_of_influence=radius_of_influence,
        )

        # Determine vertical coordinate
        obs_altitude = self._get_altitude(obs, altitude_var)
        model_has_vertical = any(
            dim in model.dims for dim in ["z", "lev", "level", "altitude", "height"]
        )

        # Extract and interpolate model values along track
        model_along_track = self._extract_along_track(
            model,
            model_lat,
            model_lon,
            lat_idx.values,
            lon_idx.values,
            obs_times,
            obs_altitude,
            vertical_method=vertical_method,
            model_has_vertical=model_has_vertical,
        )

        # Create paired output
        paired = self._create_paired_output(obs, model_along_track)

        return paired

    def _get_altitude(self, obs: xr.Dataset, altitude_var: str) -> xr.DataArray | None:
        """Get altitude/pressure coordinate from observations.

        Parameters
        ----------
        obs
            Observation dataset.
        altitude_var
            Name of altitude/pressure variable.

        Returns
        -------
        xr.DataArray | None
            Altitude values or None if not present.
        """
        # Try various altitude/pressure variable names
        for name in [altitude_var, "altitude", "alt", "z", "pressure", "pressure_obs"]:
            if name in obs.coords:
                return obs.coords[name]
            if name in obs.data_vars:
                return obs[name]
        return None

    def _extract_along_track(
        self,
        model: xr.Dataset,
        model_lat: xr.DataArray,
        model_lon: xr.DataArray,
        lat_idx: np.ndarray[Any, np.dtype[Any]],
        lon_idx: np.ndarray[Any, np.dtype[Any]],
        obs_times: xr.DataArray,
        obs_altitude: xr.DataArray | None,
        vertical_method: str,
        model_has_vertical: bool,
    ) -> xr.Dataset:
        """Extract model values along the track with 3D interpolation.

        Uses vectorized extraction for efficiency. For 3D models, interpolates
        vertically to aircraft altitude at each track point.

        Parameters
        ----------
        model
            Model dataset.
        model_lat, model_lon
            Model coordinate arrays.
        lat_idx, lon_idx
            Nearest grid indices for each track point.
        obs_times
            Track observation times.
        obs_altitude
            Track altitudes in meters (required for 3D interpolation).
        vertical_method
            Vertical interpolation method ('nearest', 'linear').
        model_has_vertical
            Whether model has vertical dimension.

        Returns
        -------
        xr.Dataset
            Model values along track with time dimension.
        """
        n_points = len(obs_times)

        # Determine lat/lon dimension names
        if model_lat.ndim == 1:
            lat_dim = str(model_lat.dims[0])
            lon_dim = str(model_lon.dims[0])
        else:
            lat_dim = str(model_lat.dims[0])
            lon_dim = str(model_lat.dims[1])

        # Handle invalid indices (outside radius of influence)
        valid_mask = (lat_idx >= 0) & (lon_idx >= 0)

        # Create DataArray indexers for vectorized spatial extraction
        lat_indexer = xr.DataArray(np.where(valid_mask, lat_idx, 0), dims=["track_point"])
        lon_indexer = xr.DataArray(np.where(valid_mask, lon_idx, 0), dims=["track_point"])

        # Detect vertical dimension
        level_dim = None
        for dim_name in ["lev", "z", "level", "altitude", "height"]:
            if dim_name in model.dims:
                level_dim = dim_name
                break

        # If model has vertical dimension and we have altitude, do 3D interpolation
        # Otherwise fall back to surface extraction
        if level_dim and model_has_vertical and obs_altitude is not None:
            extracted = self._extract_with_vertical_interp(
                model,
                lat_dim,
                lon_dim,
                level_dim,
                lat_indexer,
                lon_indexer,
                obs_altitude,
                vertical_method,
                valid_mask,
                n_points,
            )
        else:
            # Fall back to surface extraction
            model_2d = self._extract_surface(model)
            extracted = model_2d.isel({lat_dim: lat_indexer, lon_dim: lon_indexer})

        # Load data with optimized parallel scheduler for file I/O
        n_workers = min(32, os.cpu_count() or 4)
        with dask.config.set(scheduler="threads", num_workers=n_workers):
            extracted = extracted.compute()

        # Interpolate in time: for each track point, find nearest model time
        # Use vectorized nearest-neighbor time matching
        model_times = extracted["time"].values.astype("datetime64[ns]").astype(np.int64)
        obs_times_vals = obs_times.values.astype("datetime64[ns]").astype(np.int64)

        # Vectorized nearest time matching using searchsorted
        insert_idx = np.searchsorted(model_times, obs_times_vals)
        # Clamp to valid range
        insert_idx = np.clip(insert_idx, 1, len(model_times) - 1)
        # Check if left or right neighbor is closer
        left_idx = insert_idx - 1
        right_idx = insert_idx
        left_dist = np.abs(obs_times_vals - model_times[left_idx])
        right_dist = np.abs(model_times[right_idx] - obs_times_vals)
        time_idx = np.where(left_dist <= right_dist, left_idx, right_idx)

        # Create time indexer
        time_indexer = xr.DataArray(time_idx, dims=["track_point"])

        # Extract at matched times
        result_vars: dict[str, tuple[tuple[str, ...], np.ndarray[Any, np.dtype[Any]]]] = {}
        for var in extracted.data_vars:
            var_data = extracted[var]
            if "time" in var_data.dims:
                # Select the appropriate time for each track point
                out_data = var_data.isel(time=time_indexer).values
            else:
                out_data = np.full(n_points, var_data.values)

            # Mask invalid points
            out_data = np.where(valid_mask, out_data, np.nan)
            result_vars[str(var)] = (("time",), out_data)

        # Build output dataset
        coords = {"time": obs_times.values}

        return xr.Dataset(result_vars, coords=coords)

    def _extract_with_vertical_interp(
        self,
        model: xr.Dataset,
        lat_dim: str,
        lon_dim: str,
        level_dim: str,
        lat_indexer: xr.DataArray,
        lon_indexer: xr.DataArray,
        obs_altitude: xr.DataArray,
        vertical_method: str,
        valid_mask: np.ndarray[Any, np.dtype[Any]],
        n_points: int,
    ) -> xr.Dataset:
        """Extract model values with vertical interpolation to aircraft altitude.

        Parameters
        ----------
        model
            Model dataset with vertical dimension.
        lat_dim, lon_dim, level_dim
            Dimension names.
        lat_indexer, lon_indexer
            DataArray indexers for horizontal extraction.
        obs_altitude
            Aircraft altitude in meters.
        vertical_method
            Interpolation method ('nearest', 'linear').
        valid_mask
            Boolean mask for valid track points.
        n_points
            Number of track points.

        Returns
        -------
        xr.Dataset
            Model values interpolated to aircraft altitude.
        """
        # Get model pressure levels (hPa)
        model_levels = model.coords[level_dim].values

        # Determine if levels are in pressure (hPa) or something else
        # CESM uses hPa with values like 3.6 to 992.5
        levels_are_pressure = model_levels.max() > 100  # hPa values > 100

        # Convert aircraft altitude (meters) to pressure (hPa)
        altitude_values = obs_altitude.values
        if hasattr(altitude_values, "compute"):
            altitude_values = altitude_values.compute()
        altitude_values = np.asarray(altitude_values, dtype=np.float64)

        # Handle NaN altitudes - use surface pressure as fallback
        altitude_values = np.where(np.isnan(altitude_values), 0.0, altitude_values)
        aircraft_pressure = altitude_to_pressure(altitude_values)

        # Extract 3D columns at each horizontal location
        # This gives us (time, level, track_point) for each variable
        extracted_3d = model.isel({lat_dim: lat_indexer, lon_dim: lon_indexer})

        # Determine surface index (where pressure is highest)
        if model_levels[-1] > model_levels[0]:
            # Pressure increases with index - surface at end (CESM style)
            surface_idx = len(model_levels) - 1
        else:
            # Pressure decreases with index - surface at start
            surface_idx = 0

        # Build result variables
        result_vars = {}
        for var in extracted_3d.data_vars:
            var_data = extracted_3d[var]
            if level_dim not in var_data.dims:
                # Variable has no vertical dimension, keep as-is
                result_vars[var] = var_data
                continue

            # Get dimension order
            dims = list(var_data.dims)
            level_axis = dims.index(level_dim)

            # Interpolate each track point to its aircraft pressure level
            if vertical_method == "nearest":
                # Nearest neighbor interpolation
                out_data = self._interp_nearest_vertical(
                    var_data.values,
                    model_levels,
                    aircraft_pressure,
                    level_axis,
                    n_points,
                    valid_mask,
                    surface_idx,
                )
            else:
                # Linear interpolation (in log-pressure space)
                out_data = self._interp_linear_vertical(
                    var_data.values,
                    model_levels,
                    aircraft_pressure,
                    level_axis,
                    n_points,
                    valid_mask,
                    surface_idx,
                )

            # Remove level dimension from dims
            new_dims = tuple(d for d in dims if d != level_dim)
            result_vars[var] = xr.DataArray(out_data, dims=new_dims)

        # Preserve coordinates except the vertical one
        coords = {k: v for k, v in extracted_3d.coords.items() if level_dim not in v.dims}

        return xr.Dataset(result_vars, coords=coords)

    def _interp_nearest_vertical(
        self,
        data: np.ndarray[Any, np.dtype[Any]],
        model_levels: np.ndarray[Any, np.dtype[Any]],
        aircraft_pressure: np.ndarray[Any, np.dtype[Any]],
        level_axis: int,
        n_points: int,
        valid_mask: np.ndarray[Any, np.dtype[Any]],
        surface_idx: int,
    ) -> np.ndarray[Any, np.dtype[Any]]:
        """Nearest-neighbor vertical interpolation.

        Parameters
        ----------
        data
            Model data array with shape including level dimension.
        model_levels
            Model pressure levels (hPa).
        aircraft_pressure
            Target pressure levels for each track point (hPa).
        level_axis
            Axis index for the level dimension.
        n_points
            Number of track points.
        valid_mask
            Boolean mask for valid points.
        surface_idx
            Index of surface level for out-of-bounds fallback.

        Returns
        -------
        np.ndarray
            Interpolated data with level dimension removed.
        """
        # Find nearest level for each aircraft pressure
        # Model levels might be ordered differently, so handle both cases
        level_indices = np.zeros(n_points, dtype=np.int64)

        for i in range(n_points):
            if not valid_mask[i]:
                level_indices[i] = surface_idx
                continue

            target_p = aircraft_pressure[i]
            # Find nearest level by absolute difference
            diffs = np.abs(model_levels - target_p)
            level_indices[i] = int(np.argmin(diffs))

        # Use advanced indexing to extract values at each level
        # Move level axis to the end for easier indexing
        data_moved = np.moveaxis(data, level_axis, -1)
        original_shape = data_moved.shape[:-1]

        # Flatten all but last axis for indexing, then reshape
        n_other = int(np.prod(original_shape[:-1])) if len(original_shape) > 1 else 1
        n_track = original_shape[-1] if len(original_shape) > 0 else n_points

        # Extract using the level indices for each track point
        # Result shape will be original_shape without the level dimension
        result: np.ndarray[Any, np.dtype[Any]]
        if len(original_shape) == 1:
            # Shape is (track_point, level) -> (track_point,)
            result = data_moved[np.arange(n_track), level_indices[:n_track]]
        elif len(original_shape) == 2:
            # Shape is (time, track_point, level) -> (time, track_point)
            n_time = original_shape[0]
            result = np.zeros((n_time, n_track), dtype=data.dtype)
            for t in range(n_time):
                result[t, :] = data_moved[t, np.arange(n_track), level_indices[:n_track]]
        else:
            # General case - iterate over track points
            result = np.take_along_axis(
                data_moved,
                level_indices.reshape((1,) * (len(original_shape) - 1) + (n_track, 1)),
                axis=-1,
            ).squeeze(axis=-1)

        return result

    def _interp_linear_vertical(
        self,
        data: np.ndarray[Any, np.dtype[Any]],
        model_levels: np.ndarray[Any, np.dtype[Any]],
        aircraft_pressure: np.ndarray[Any, np.dtype[Any]],
        level_axis: int,
        n_points: int,
        valid_mask: np.ndarray[Any, np.dtype[Any]],
        surface_idx: int,
    ) -> np.ndarray[Any, np.dtype[Any]]:
        """Linear vertical interpolation in log-pressure space.

        Parameters
        ----------
        data
            Model data array with shape including level dimension.
        model_levels
            Model pressure levels (hPa).
        aircraft_pressure
            Target pressure levels for each track point (hPa).
        level_axis
            Axis index for the level dimension.
        n_points
            Number of track points.
        valid_mask
            Boolean mask for valid points.
        surface_idx
            Index of surface level for out-of-bounds fallback.

        Returns
        -------
        np.ndarray
            Interpolated data with level dimension removed.
        """
        # Use log-pressure for interpolation (better for atmospheric profiles)
        # Guard against zero/negative pressure values
        safe_model_levels = np.maximum(model_levels, 1e-10)
        log_model_levels = np.log(safe_model_levels)

        # Move level axis to end
        data_moved = np.moveaxis(data, level_axis, -1)
        original_shape = data_moved.shape[:-1]
        n_levels = data_moved.shape[-1]

        # Determine shape of output
        if len(original_shape) == 1:
            n_track = original_shape[0]
            result = np.zeros(n_track, dtype=np.float64)
        elif len(original_shape) == 2:
            n_time, n_track = original_shape
            result = np.zeros((n_time, n_track), dtype=np.float64)
        else:
            result = np.zeros(original_shape, dtype=np.float64)
            n_track = original_shape[-1]

        # Ensure model levels are sorted for searchsorted
        if model_levels[0] > model_levels[-1]:
            # Decreasing pressure - reverse for interpolation
            sorted_indices = np.arange(n_levels - 1, -1, -1)
            log_levels_sorted = log_model_levels[sorted_indices]
        else:
            sorted_indices = np.arange(n_levels)
            log_levels_sorted = log_model_levels

        # Interpolate each track point
        for i in range(min(n_track, n_points)):
            if not valid_mask[i]:
                # Use surface value for invalid points
                if len(original_shape) == 1:
                    result[i] = data_moved[i, surface_idx]
                elif len(original_shape) == 2:
                    result[:, i] = data_moved[:, i, surface_idx]
                continue

            target_log_p = np.log(max(aircraft_pressure[i], 1e-10))

            # Find bracketing levels
            idx = np.searchsorted(log_levels_sorted, target_log_p)

            if idx == 0:
                # Below lowest level - use lowest
                level_idx = sorted_indices[0]
                if len(original_shape) == 1:
                    result[i] = data_moved[i, level_idx]
                elif len(original_shape) == 2:
                    result[:, i] = data_moved[:, i, level_idx]
            elif idx >= n_levels:
                # Above highest level - use highest
                level_idx = sorted_indices[-1]
                if len(original_shape) == 1:
                    result[i] = data_moved[i, level_idx]
                elif len(original_shape) == 2:
                    result[:, i] = data_moved[:, i, level_idx]
            else:
                # Interpolate between bracketing levels
                idx_lo = sorted_indices[idx - 1]
                idx_hi = sorted_indices[idx]
                log_p_lo = log_model_levels[idx_lo]
                log_p_hi = log_model_levels[idx_hi]

                # Linear weight in log-pressure space
                weight = (target_log_p - log_p_lo) / (log_p_hi - log_p_lo)
                weight = np.clip(weight, 0.0, 1.0)

                if len(original_shape) == 1:
                    result[i] = (1 - weight) * data_moved[i, idx_lo] + weight * data_moved[
                        i, idx_hi
                    ]
                elif len(original_shape) == 2:
                    result[:, i] = (1 - weight) * data_moved[:, i, idx_lo] + weight * data_moved[
                        :, i, idx_hi
                    ]

        return result

    def _create_paired_output(
        self,
        obs: xr.Dataset,
        model_along_track: xr.Dataset,
    ) -> xr.Dataset:
        """Create the final paired output dataset.

        Parameters
        ----------
        obs
            Observation dataset.
        model_along_track
            Model values along track.

        Returns
        -------
        xr.Dataset
            Combined dataset with obs and model variables.
        """
        # Combine coordinates
        coords = dict(obs.coords)

        # Get obs variable names to check for collisions
        obs_var_names = set(str(v) for v in obs.data_vars)

        # Combine data variables
        data_vars: dict[str, Any] = {}

        # Add observation variables
        for var in obs.data_vars:
            data_vars[str(var)] = obs[var]

        # Add model variables - add prefix only if name collision
        for var in model_along_track.data_vars:
            var_name = str(var)
            if var_name in obs_var_names:
                # Name collision - add model_ prefix
                data_vars[f"model_{var_name}"] = model_along_track[var]
            else:
                # No collision - keep original name
                data_vars[var_name] = model_along_track[var]

        return xr.Dataset(data_vars, coords=coords)
