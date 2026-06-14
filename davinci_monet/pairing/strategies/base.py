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
    def pair_sources(
        self,
        x_data: xr.Dataset,
        y_data: xr.Dataset,
        radius_of_influence: float | None = None,
        time_tolerance: TimeDelta | None = None,
        vertical_method: str = "nearest",
        horizontal_method: str = "nearest",
        **kwargs: Any,
    ) -> xr.Dataset:
        """Sample ``y_data`` onto ``x_data``."""
        ...

    def _get_dataset_coords(self, dataset: xr.Dataset) -> tuple[xr.DataArray, xr.DataArray]:
        """Extract latitude and longitude coordinates from dataset.

        Parameters
        ----------
        dataset
            Dataset dataset.

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
            if name in dataset.coords:
                lat = dataset.coords[name]
                break
            if name in dataset.data_vars:
                lat = dataset[name]
                break

        for name in ["lon", "longitude", "LON", "LONGITUDE", "XLONG"]:
            if name in dataset.coords:
                lon = dataset.coords[name]
                break
            if name in dataset.data_vars:
                lon = dataset[name]
                break

        if lat is None or lon is None:
            raise PairingError(
                f"Cannot find lat/lon coordinates in dataset. "
                f"Available coords: {list(dataset.coords)}"
            )

        return lat, lon

    def _get_geometry_coords(self, geometry: xr.Dataset) -> tuple[xr.DataArray, xr.DataArray]:
        """Extract latitude and longitude from datasets.

        Parameters
        ----------
        geometry
            Dataset dataset.

        Returns
        -------
        tuple[xr.DataArray, xr.DataArray]
            (latitude, longitude) arrays.
        """
        lat = None
        lon = None

        for name in ["lat", "latitude", "LAT", "LATITUDE"]:
            if name in geometry.coords:
                lat = geometry.coords[name]
                break
            if name in geometry.data_vars:
                lat = geometry[name]
                break

        for name in ["lon", "longitude", "LON", "LONGITUDE"]:
            if name in geometry.coords:
                lon = geometry.coords[name]
                break
            if name in geometry.data_vars:
                lon = geometry[name]
                break

        if lat is None or lon is None:
            raise PairingError(
                f"Cannot find lat/lon coordinates in datasets. "
                f"Available coords: {list(geometry.coords)}"
            )

        return lat, lon

    def _find_nearest_indices(
        self,
        y_lat: xr.DataArray,
        y_lon: xr.DataArray,
        x_lat: xr.DataArray,
        x_lon: xr.DataArray,
        radius_of_influence: float | None = None,
    ) -> tuple[xr.DataArray, xr.DataArray]:
        """Find nearest dataset grid indices for dataset locations.

        Parameters
        ----------
        dataset_lat, dataset_lon
            Dataset coordinate arrays.
        geometry_lat, geometry_lon
            Dataset coordinate arrays.
        radius_of_influence
            Maximum distance in meters. Points beyond this are masked.

        Returns
        -------
        tuple[xr.DataArray, xr.DataArray]
            (lat_indices, lon_indices) for each dataset point.
        """
        # Handle 1D vs 2D dataset grids
        if y_lat.ndim == 1 and y_lon.ndim == 1:
            # Regular grid - find nearest indices directly
            lat_idx = self._find_nearest_1d(y_lat.values, x_lat.values)
            lon_idx = self._find_nearest_1d(y_lon.values, x_lon.values)
        else:
            # Curvilinear grid - need 2D search
            lat_idx, lon_idx = self._find_nearest_2d(
                y_lat.values, y_lon.values, x_lat.values, x_lon.values
            )

        # Apply radius of influence filter if specified
        if radius_of_influence is not None:
            distances = self._haversine_distance(
                x_lat.values,
                x_lon.values,
                (y_lat.values[lat_idx] if y_lat.ndim == 1 else y_lat.values[lat_idx, lon_idx]),
                (y_lon.values[lon_idx] if y_lon.ndim == 1 else y_lon.values[lat_idx, lon_idx]),
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
        y_lat: np.ndarray[Any, np.dtype[Any]],
        y_lon: np.ndarray[Any, np.dtype[Any]],
        x_lat: np.ndarray[Any, np.dtype[Any]],
        x_lon: np.ndarray[Any, np.dtype[Any]],
    ) -> tuple[np.ndarray[Any, np.dtype[Any]], np.ndarray[Any, np.dtype[Any]]]:
        """Find nearest indices in a 2D curvilinear grid.

        Uses brute force distance calculation. For large grids,
        consider using scipy.spatial.cKDTree.

        Parameters
        ----------
        dataset_lat, dataset_lon
            2D dataset coordinate arrays.
        geometry_lat, geometry_lon
            1D dataset coordinate arrays.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            (i_indices, j_indices) in the 2D grid.
        """
        n_geometry = len(x_lat)
        i_idx = np.zeros(n_geometry, dtype=int)
        j_idx = np.zeros(n_geometry, dtype=int)

        for k in range(n_geometry):
            # Calculate distance to all grid points
            dist = self._haversine_distance(x_lat[k], x_lon[k], y_lat, y_lon)
            # Find minimum
            flat_idx = np.argmin(dist)
            i_idx[k], j_idx[k] = np.unravel_index(flat_idx, y_lat.shape)

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
        dataset: xr.Dataset,
        target_times: xr.DataArray,
        method: str = "nearest",
        time_tolerance: TimeDelta | None = None,
    ) -> xr.Dataset:
        """Interpolate dataset data to target times.

        Parameters
        ----------
        dataset
            Dataset dataset with time dimension.
        target_times
            Target time values.
        method
            Interpolation method ('nearest', 'linear').
        time_tolerance
            Maximum allowed gap between a target (dataset) time and the
            matched dataset time. When set and ``method == "nearest"``, target
            times with no dataset time within the tolerance are filled with NaN
            instead of being silently snapped to a distant dataset time. None
            (default) disables the gate and matches the nearest time regardless
            of distance.

        Returns
        -------
        xr.Dataset
            Dataset data interpolated to target times.
        """
        if "time" not in dataset.dims:
            return dataset

        if method == "nearest":
            if time_tolerance is not None:
                # Reindex (not sel) so out-of-tolerance targets become NaN
                # rather than snapping to a far-away dataset time.
                return dataset.reindex(
                    time=target_times.values,
                    method="nearest",
                    tolerance=pd.Timedelta(time_tolerance),
                )
            # Select nearest times and assign target times as coordinate.
            # This ensures alignment when combining with dataset data.
            result = dataset.sel(time=target_times, method="nearest")
            return result.assign_coords(time=target_times.values)
        else:
            return dataset.interp(time=target_times, method=method)  # type: ignore[arg-type]

    def _interpolate_vertical(
        self,
        dataset: xr.Dataset,
        target_levels: xr.DataArray,
        level_coord: str = "z",
        method: str = "linear",
    ) -> xr.Dataset:
        """Interpolate dataset data to target vertical levels.

        Parameters
        ----------
        dataset
            Dataset dataset with vertical dimension.
        target_levels
            Target level values.
        level_coord
            Name of vertical coordinate.
        method
            Interpolation method ('nearest', 'linear', 'log').

        Returns
        -------
        xr.Dataset
            Dataset data interpolated to target levels.
        """
        if level_coord not in dataset.dims:
            return dataset

        if method == "nearest":
            return dataset.sel({level_coord: target_levels}, method="nearest")
        elif method == "log":
            # Log-pressure interpolation
            y_log = dataset.assign_coords({level_coord: np.log(dataset[level_coord].values)})
            target_log = np.log(target_levels.values)
            result = y_log.interp({level_coord: target_log}, method="linear")
            return result.assign_coords({level_coord: target_levels.values})
        else:
            return dataset.interp(
                {level_coord: target_levels.values},
                method=method,  # type: ignore[arg-type]
            )

    def _extract_surface(self, dataset: xr.Dataset, level_dim: str | None = None) -> xr.Dataset:
        """Extract surface level from dataset data.

        Parameters
        ----------
        dataset
            Dataset dataset.
        level_dim
            Name of vertical dimension. If None, auto-detects from common names.

        Returns
        -------
        xr.Dataset
            Dataset data at surface level only.
        """
        # Auto-detect vertical dimension if not specified
        if level_dim is None:
            for dim_name in ["lev", "z", "level", "altitude", "height"]:
                if dim_name in dataset.dims:
                    level_dim = dim_name
                    break

        if level_dim is None or level_dim not in dataset.dims:
            return dataset

        # Determine correct surface index based on coordinate values
        # For CESM-style hybrid coordinates where pressure increases downward,
        # surface is at the last index (highest pressure), not first (TOA)
        surface_idx = 0  # Default: first level is surface
        if level_dim in dataset.coords:
            vert_vals = dataset.coords[level_dim].values
            if len(vert_vals) > 1 and vert_vals[-1] > vert_vals[0]:
                # Values increase (typical hybrid sigma-pressure) -> surface at end
                surface_idx = -1

        return dataset.isel({level_dim: surface_idx})
