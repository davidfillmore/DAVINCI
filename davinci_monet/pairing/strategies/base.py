"""Base class for pairing strategies.

This module provides the abstract base class that all pairing strategies
must inherit from, along with common utility functions.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from davinci_monet.core.exceptions import InterpolationError, PairingError
from davinci_monet.core.protocols import DataGeometry
from davinci_monet.core.types import TimeDelta


class BasePairingStrategy(ABC):
    """Abstract base class for pairing strategies.

    Provides common functionality used by all geometry-specific strategies.
    """

    @property
    @abstractmethod
    def geometry(self) -> DataGeometry:
        """Return the geometry type this strategy handles."""
        ...

    @abstractmethod
    def pair(
        self,
        model: xr.Dataset,
        obs: xr.Dataset,
        radius_of_influence: float | None = None,
        time_tolerance: TimeDelta | None = None,
        vertical_method: str = "nearest",
        horizontal_method: str = "nearest",
        **kwargs: Any,
    ) -> xr.Dataset:
        """Pair model output with observations.

        Parameters
        ----------
        model
            Model Dataset with dims (time, level, lat, lon).
        obs
            Observation Dataset with geometry-specific dimensions.
        radius_of_influence
            Spatial search radius in meters.
        time_tolerance
            Maximum time difference for matching.
        vertical_method
            Vertical interpolation method.
        horizontal_method
            Horizontal interpolation method.
        **kwargs
            Strategy-specific options.

        Returns
        -------
        xr.Dataset
            Dataset with paired model and observation values.
        """
        ...

    def pair_sources(
        self,
        reference: xr.Dataset,
        comparand: xr.Dataset,
        **kwargs: Any,
    ) -> xr.Dataset:
        """Role-neutral pairing entrypoint.

        Resamples ``comparand`` onto ``reference``'s geometry. Strategy
        implementations may still share older helper code internally, but this
        method is the public boundary used by the engine and accepts
        reference/comparand variable names.
        """
        reference_vars = kwargs.pop("reference_vars", None)
        comparand_vars = kwargs.pop("comparand_vars", None)
        reference_var = kwargs.pop("reference_var", None)
        comparand_var = kwargs.pop("comparand_var", None)
        if reference_var is None and reference_vars:
            reference_var = reference_vars[0]
        if comparand_var is None and comparand_vars:
            comparand_var = comparand_vars[0]
        if reference_var is not None:
            kwargs.setdefault("reference_var", reference_var)
            kwargs.setdefault("obs_var", reference_var)
        if comparand_var is not None:
            kwargs.setdefault("comparand_var", comparand_var)
            kwargs.setdefault("model_var", comparand_var)
        return self.pair(model=comparand, obs=reference, **kwargs)

    def _get_model_coords(self, model: xr.Dataset) -> tuple[xr.DataArray, xr.DataArray]:
        """Extract latitude and longitude coordinates from model.

        Parameters
        ----------
        model
            Model dataset.

        Returns
        -------
        tuple[xr.DataArray, xr.DataArray]
            (latitude, longitude) coordinate arrays.

        Raises
        ------
        PairingError
            If coordinates not found.
        """
        lat = None
        lon = None

        for name in ["lat", "latitude", "LAT", "LATITUDE", "XLAT"]:
            if name in model.coords:
                lat = model.coords[name]
                break
            if name in model.data_vars:
                lat = model[name]
                break

        for name in ["lon", "longitude", "LON", "LONGITUDE", "XLONG"]:
            if name in model.coords:
                lon = model.coords[name]
                break
            if name in model.data_vars:
                lon = model[name]
                break

        if lat is None or lon is None:
            raise PairingError(
                f"Cannot find lat/lon coordinates in model. "
                f"Available coords: {list(model.coords)}"
            )

        return lat, lon

    def _get_obs_coords(self, obs: xr.Dataset) -> tuple[xr.DataArray, xr.DataArray]:
        """Extract latitude and longitude from observations.

        Parameters
        ----------
        obs
            Observation dataset.

        Returns
        -------
        tuple[xr.DataArray, xr.DataArray]
            (latitude, longitude) arrays.
        """
        lat = None
        lon = None

        for name in ["lat", "latitude", "LAT", "LATITUDE"]:
            if name in obs.coords:
                lat = obs.coords[name]
                break
            if name in obs.data_vars:
                lat = obs[name]
                break

        for name in ["lon", "longitude", "LON", "LONGITUDE"]:
            if name in obs.coords:
                lon = obs.coords[name]
                break
            if name in obs.data_vars:
                lon = obs[name]
                break

        if lat is None or lon is None:
            raise PairingError(
                f"Cannot find lat/lon coordinates in observations. "
                f"Available coords: {list(obs.coords)}"
            )

        return lat, lon

    def _get_reference_coords(self, reference: xr.Dataset) -> tuple[xr.DataArray, xr.DataArray]:
        """Extract (lat, lon) from the reference source.

        Role-neutral alias of :meth:`_get_obs_coords` (Phase 4).
        """
        return self._get_obs_coords(reference)

    def _get_comparand_coords(self, comparand: xr.Dataset) -> tuple[xr.DataArray, xr.DataArray]:
        """Extract (lat, lon) from the comparand source.

        Role-neutral alias of :meth:`_get_model_coords` (Phase 4).
        """
        return self._get_model_coords(comparand)

    def _find_nearest_indices(
        self,
        model_lat: xr.DataArray,
        model_lon: xr.DataArray,
        obs_lat: xr.DataArray,
        obs_lon: xr.DataArray,
        radius_of_influence: float | None = None,
    ) -> tuple[xr.DataArray, xr.DataArray]:
        """Find nearest model grid indices for observation locations.

        Parameters
        ----------
        model_lat, model_lon
            Model coordinate arrays.
        obs_lat, obs_lon
            Observation coordinate arrays.
        radius_of_influence
            Maximum distance in meters. Points beyond this are masked.

        Returns
        -------
        tuple[xr.DataArray, xr.DataArray]
            (lat_indices, lon_indices) for each observation point.
        """
        # Handle 1D vs 2D model grids
        if model_lat.ndim == 1 and model_lon.ndim == 1:
            # Regular grid - find nearest indices directly
            lat_idx = self._find_nearest_1d(model_lat.values, obs_lat.values)
            lon_idx = self._find_nearest_1d(model_lon.values, obs_lon.values)
        else:
            # Curvilinear grid - need 2D search
            lat_idx, lon_idx = self._find_nearest_2d(
                model_lat.values, model_lon.values, obs_lat.values, obs_lon.values
            )

        # Apply radius of influence filter if specified
        if radius_of_influence is not None:
            distances = self._haversine_distance(
                obs_lat.values,
                obs_lon.values,
                (
                    model_lat.values[lat_idx]
                    if model_lat.ndim == 1
                    else model_lat.values[lat_idx, lon_idx]
                ),
                (
                    model_lon.values[lon_idx]
                    if model_lon.ndim == 1
                    else model_lon.values[lat_idx, lon_idx]
                ),
            )
            mask = distances > radius_of_influence
            lat_idx = np.where(mask, -1, lat_idx)
            lon_idx = np.where(mask, -1, lon_idx)

        return xr.DataArray(lat_idx), xr.DataArray(lon_idx)

    def _find_nearest_1d(
        self, grid: np.ndarray[Any, np.dtype[Any]], points: np.ndarray[Any, np.dtype[Any]]
    ) -> np.ndarray[Any, np.dtype[Any]]:
        """Find nearest indices in a 1D sorted array.

        Parameters
        ----------
        grid
            1D array of grid values.
        points
            Array of points to locate.

        Returns
        -------
        np.ndarray
            Indices of nearest grid points.
        """
        # Use searchsorted for efficiency
        idx = np.searchsorted(grid, points)
        idx = np.clip(idx, 1, len(grid) - 1)

        # Check which neighbor is closer
        left = grid[idx - 1]
        right = grid[idx]
        idx = np.where(np.abs(points - left) < np.abs(points - right), idx - 1, idx)

        return idx

    def _find_nearest_2d(
        self,
        model_lat: np.ndarray[Any, np.dtype[Any]],
        model_lon: np.ndarray[Any, np.dtype[Any]],
        obs_lat: np.ndarray[Any, np.dtype[Any]],
        obs_lon: np.ndarray[Any, np.dtype[Any]],
    ) -> tuple[np.ndarray[Any, np.dtype[Any]], np.ndarray[Any, np.dtype[Any]]]:
        """Find nearest indices in a 2D curvilinear grid.

        Uses brute force distance calculation. For large grids,
        consider using scipy.spatial.cKDTree.

        Parameters
        ----------
        model_lat, model_lon
            2D model coordinate arrays.
        obs_lat, obs_lon
            1D observation coordinate arrays.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            (i_indices, j_indices) in the 2D grid.
        """
        n_obs = len(obs_lat)
        i_idx = np.zeros(n_obs, dtype=int)
        j_idx = np.zeros(n_obs, dtype=int)

        for k in range(n_obs):
            # Calculate distance to all grid points
            dist = self._haversine_distance(obs_lat[k], obs_lon[k], model_lat, model_lon)
            # Find minimum
            flat_idx = np.argmin(dist)
            i_idx[k], j_idx[k] = np.unravel_index(flat_idx, model_lat.shape)

        return i_idx, j_idx

    def _haversine_distance(
        self,
        lat1: np.ndarray[Any, np.dtype[Any]] | float,
        lon1: np.ndarray[Any, np.dtype[Any]] | float,
        lat2: np.ndarray[Any, np.dtype[Any]] | float,
        lon2: np.ndarray[Any, np.dtype[Any]] | float,
    ) -> np.ndarray[Any, np.dtype[Any]]:
        """Calculate haversine distance in meters.

        Parameters
        ----------
        lat1, lon1
            First point(s) in degrees.
        lat2, lon2
            Second point(s) in degrees.

        Returns
        -------
        np.ndarray
            Distance(s) in meters.
        """
        R = 6371000  # Earth radius in meters

        lat1_rad = np.radians(lat1)
        lat2_rad = np.radians(lat2)
        dlat = np.radians(lat2 - lat1)
        dlon = np.radians(lon2 - lon1)

        a = np.sin(dlat / 2) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

        return np.asarray(R * c)

    def _interpolate_time(
        self,
        model: xr.Dataset,
        target_times: xr.DataArray,
        method: str = "nearest",
        time_tolerance: TimeDelta | None = None,
    ) -> xr.Dataset:
        """Interpolate model data to target times.

        Parameters
        ----------
        model
            Model dataset with time dimension.
        target_times
            Target time values.
        method
            Interpolation method ('nearest', 'linear').
        time_tolerance
            Maximum allowed gap between a target (observation) time and the
            matched model time. When set and ``method == "nearest"``, target
            times with no model time within the tolerance are filled with NaN
            instead of being silently snapped to a distant model time. None
            (default) disables the gate and matches the nearest time regardless
            of distance.

        Returns
        -------
        xr.Dataset
            Model data interpolated to target times.
        """
        if "time" not in model.dims:
            return model

        if method == "nearest":
            if time_tolerance is not None:
                # Reindex (not sel) so out-of-tolerance targets become NaN
                # rather than snapping to a far-away model time.
                return model.reindex(
                    time=target_times.values,
                    method="nearest",
                    tolerance=pd.Timedelta(time_tolerance),
                )
            # Select nearest times and assign target times as coordinate.
            # This ensures alignment when combining with observation data.
            result = model.sel(time=target_times, method="nearest")
            return result.assign_coords(time=target_times.values)
        else:
            return model.interp(time=target_times, method=method)  # type: ignore[arg-type]

    def _interpolate_vertical(
        self,
        model: xr.Dataset,
        target_levels: xr.DataArray,
        level_coord: str = "z",
        method: str = "linear",
    ) -> xr.Dataset:
        """Interpolate model data to target vertical levels.

        Parameters
        ----------
        model
            Model dataset with vertical dimension.
        target_levels
            Target level values.
        level_coord
            Name of vertical coordinate.
        method
            Interpolation method ('nearest', 'linear', 'log').

        Returns
        -------
        xr.Dataset
            Model data interpolated to target levels.
        """
        if level_coord not in model.dims:
            return model

        if method == "nearest":
            return model.sel({level_coord: target_levels}, method="nearest")
        elif method == "log":
            # Log-pressure interpolation
            model_log = model.assign_coords({level_coord: np.log(model[level_coord].values)})
            target_log = np.log(target_levels.values)
            result = model_log.interp({level_coord: target_log}, method="linear")
            return result.assign_coords({level_coord: target_levels.values})
        else:
            return model.interp(
                {level_coord: target_levels.values},
                method=method,  # type: ignore[arg-type]
            )

    def _extract_surface(self, model: xr.Dataset, level_dim: str | None = None) -> xr.Dataset:
        """Extract surface level from model data.

        Parameters
        ----------
        model
            Model dataset.
        level_dim
            Name of vertical dimension. If None, auto-detects from common names.

        Returns
        -------
        xr.Dataset
            Model data at surface level only.
        """
        # Auto-detect vertical dimension if not specified
        if level_dim is None:
            for dim_name in ["lev", "z", "level", "altitude", "height"]:
                if dim_name in model.dims:
                    level_dim = dim_name
                    break

        if level_dim is None or level_dim not in model.dims:
            return model

        # Determine correct surface index based on coordinate values
        # For CESM-style hybrid coordinates where pressure increases downward,
        # surface is at the last index (highest pressure), not first (TOA)
        surface_idx = 0  # Default: first level is surface
        if level_dim in model.coords:
            vert_vals = model.coords[level_dim].values
            if len(vert_vals) > 1 and vert_vals[-1] > vert_vals[0]:
                # Values increase (typical hybrid sigma-pressure) -> surface at end
                surface_idx = -1

        return model.isel({level_dim: surface_idx})
