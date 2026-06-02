"""Grid-to-grid pairing strategy.

This module implements pairing for gridded observations (L3 satellite
products, reanalysis) with gridded model output.
"""

from __future__ import annotations

from typing import Any, Hashable

import numpy as np
import xarray as xr

from davinci_monet.core.exceptions import PairingError
from davinci_monet.core.protocols import DataGeometry
from davinci_monet.core.types import TimeDelta
from davinci_monet.pairing.strategies.base import BasePairingStrategy


class GridStrategy(BasePairingStrategy):
    """Pairing strategy for gridded observations.

    Matches gridded observations (L3 satellite products, reanalysis)
    to gridded model output through regridding.

    The strategy:
    1. Determines common grid (obs grid, model grid, or custom)
    2. Regrids model to observation grid (or vice versa)
    3. Aligns temporal dimensions
    4. Creates paired dataset on common grid

    Examples
    --------
    >>> strategy = GridStrategy()
    >>> paired = strategy.pair(model_data, l3_satellite_data,
    ...                        regrid_to='obs')
    """

    @property
    def geometry(self) -> DataGeometry:
        """Return GRID geometry."""
        return DataGeometry.GRID

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
        """Pair gridded observations with model grid.

        Parameters
        ----------
        model
            Model Dataset with dims (time, [z], lat, lon).
        obs
            Observation Dataset with dims (time, lat, lon).
        radius_of_influence
            Not used for grid-to-grid.
        time_tolerance
            Maximum time difference for matching.
        vertical_method
            Vertical interpolation method if needed.
        horizontal_method
            Horizontal interpolation method ('nearest', 'bilinear').
        **kwargs
            Additional options:
            - regrid_to: str, 'obs' or 'model' (default 'obs')
            - extract_surface: bool, whether to extract surface level

        Returns
        -------
        xr.Dataset
            Paired dataset on common grid.
        """
        regrid_to = kwargs.get("regrid_to", "obs")
        extract_surface = kwargs.get("extract_surface", True)

        # Get coordinates
        model_lat, model_lon = self._get_model_coords(model)
        obs_lat, obs_lon = self._get_obs_coords(obs)

        # Extract surface if model is 3D
        if extract_surface and "z" in model.dims:
            model_proc = self._extract_surface(model)
        else:
            model_proc = model

        # Regrid to common grid
        if regrid_to == "obs":
            # Regrid model to observation grid
            model_regridded = self._regrid_to_target(
                model_proc, obs_lat, obs_lon, method=horizontal_method
            )
            obs_aligned = obs
        elif regrid_to == "model":
            # Regrid observations to model grid
            model_regridded = model_proc
            obs_aligned = self._regrid_to_target(
                obs, model_lat, model_lon, method=horizontal_method
            )
        else:
            raise PairingError(f"Invalid regrid_to option: {regrid_to}")

        # Align temporal dimensions
        if "time" in model_regridded.dims and "time" in obs_aligned.dims:
            model_regridded, obs_aligned = self._align_times(
                model_regridded, obs_aligned, time_tolerance
            )

        # Create paired output
        paired = self._create_paired_output(obs_aligned, model_regridded)

        return paired

    def _regrid_to_target(
        self,
        data: xr.Dataset,
        target_lat: xr.DataArray,
        target_lon: xr.DataArray,
        method: str = "nearest",
    ) -> xr.Dataset:
        """Regrid dataset to target lat/lon grid.

        Parameters
        ----------
        data
            Dataset to regrid.
        target_lat, target_lon
            Target coordinate arrays (1D).
        method
            Interpolation method.

        Returns
        -------
        xr.Dataset
            Regridded dataset.
        """
        # Find source lat/lon dimension names
        source_lat, source_lon = self._get_model_coords(data)

        if source_lat.ndim != 1 or source_lon.ndim != 1:
            # Curvilinear grids need more complex regridding
            return self._regrid_curvilinear(data, target_lat, target_lon, method)

        lat_dim = source_lat.dims[0]
        lon_dim = source_lon.dims[0]

        # Use xarray interp for regular grids
        return data.interp(
            {lat_dim: target_lat.values, lon_dim: target_lon.values},
            method=method,  # type: ignore[arg-type]
        )

    def _regrid_curvilinear(
        self,
        data: xr.Dataset,
        target_lat: xr.DataArray,
        target_lon: xr.DataArray,
        method: str,
    ) -> xr.Dataset:
        """Regrid curvilinear grid to regular target grid.

        Parameters
        ----------
        data
            Dataset with curvilinear coordinates.
        target_lat, target_lon
            Target 1D coordinate arrays.
        method
            Interpolation method.

        Returns
        -------
        xr.Dataset
            Regridded dataset on regular grid.
        """
        # This is a simplified implementation using nearest neighbor
        # For production use, consider xesmf or other regridding libraries

        source_lat, source_lon = self._get_model_coords(data)
        source_lat_flat = source_lat.values.flatten()
        source_lon_flat = source_lon.values.flatten()

        # Build output grid
        n_lat = len(target_lat)
        n_lon = len(target_lon)

        data_vars: dict[str, tuple[tuple[str, ...], np.ndarray[Any, np.dtype[Any]]]] = {}

        for var in data.data_vars:
            var_data = data[var]
            source_shape = source_lat.shape

            # Get non-spatial dimensions
            spatial_dims = source_lat.dims
            other_dims: list[str] = [str(d) for d in var_data.dims if d not in spatial_dims]

            # Build output shape
            out_shape: list[int] = [int(var_data.sizes[d]) for d in other_dims]
            out_shape.extend([n_lat, n_lon])

            out_data = np.full(out_shape, np.nan)

            # For each target point, find nearest source point
            for i, lat in enumerate(target_lat.values):
                for j, lon in enumerate(target_lon.values):
                    dist = self._haversine_distance(lat, lon, source_lat_flat, source_lon_flat)
                    nearest_idx = np.argmin(dist)
                    src_i, src_j = np.unravel_index(nearest_idx, source_shape)

                    selection = {spatial_dims[0]: src_i, spatial_dims[1]: src_j}
                    val = var_data.isel(selection).values
                    out_data[..., i, j] = val

            out_dims = tuple(other_dims) + ("lat", "lon")
            data_vars[str(var)] = (out_dims, out_data)

        coords = {
            "lat": target_lat.values,
            "lon": target_lon.values,
        }

        # Add time coordinate if present
        if "time" in data.coords:
            coords["time"] = data.coords["time"].values

        return xr.Dataset(data_vars, coords=coords)

    def _align_times(
        self,
        model: xr.Dataset,
        obs: xr.Dataset,
        time_tolerance: TimeDelta | None,
    ) -> tuple[xr.Dataset, xr.Dataset]:
        """Align model and observation time dimensions.

        Parameters
        ----------
        model
            Model dataset.
        obs
            Observation dataset.
        time_tolerance
            Maximum time difference.

        Returns
        -------
        tuple[xr.Dataset, xr.Dataset]
            Temporally aligned datasets.
        """
        model_times = model["time"].values
        obs_times = obs["time"].values

        # Find common times (within tolerance)
        if time_tolerance is not None:
            # Match each obs time to the nearest model time, then relabel the
            # matched model times to the obs times so the two share identical
            # time labels. Without the relabel, _create_paired_output reindexes
            # the model onto the obs time coordinate and the model becomes all
            # NaN whenever model/obs times differ (e.g. MERRA2 00:30 vs MODIS 00:00).
            model_matched = model.sel(time=obs_times, method="nearest")
            model_matched = model_matched.assign_coords(time=obs_times)
            return model_matched, obs
        else:
            # Interpolate model to obs times
            model_interp = model.interp(time=obs_times)
            return model_interp, obs

    def _create_paired_output(
        self,
        obs: xr.Dataset,
        model: xr.Dataset,
    ) -> xr.Dataset:
        """Create the final paired output dataset.

        Parameters
        ----------
        obs
            Observation dataset (on target grid).
        model
            Model dataset (on target grid).

        Returns
        -------
        xr.Dataset
            Combined dataset.
        """
        # Combine coordinates
        coords = dict(obs.coords)
        for coord in model.coords:
            if coord not in coords:
                coords[coord] = model.coords[coord]

        # Combine data variables
        data_vars: dict[str, Any] = {}

        # Add observation variables
        for var in obs.data_vars:
            data_vars[str(var)] = obs[var]

        # Add model variables with prefix
        for var in model.data_vars:
            model_var_name = f"model_{var}"
            data_vars[model_var_name] = model[var]

        return xr.Dataset(data_vars, coords=coords)
