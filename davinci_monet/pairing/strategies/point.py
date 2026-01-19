"""Point-to-grid pairing strategy.

This module implements pairing for point observations (surface stations,
ground sites) with gridded model output.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Hashable

import dask
import numpy as np
import xarray as xr

from davinci_monet.core.exceptions import PairingError
from davinci_monet.core.protocols import DataGeometry
from davinci_monet.core.types import TimeDelta
from davinci_monet.pairing.strategies.base import BasePairingStrategy


class PointStrategy(BasePairingStrategy):
    """Pairing strategy for point observations.

    Matches fixed-location observations (surface stations, ground sites)
    to the nearest model grid cell within the radius of influence.

    The strategy:
    1. Finds nearest model grid cell for each observation site
    2. Extracts surface level from model (if 3D)
    3. Interpolates model to observation times
    4. Creates paired dataset with aligned values

    Examples
    --------
    >>> strategy = PointStrategy()
    >>> paired = strategy.pair(model_data, obs_data,
    ...                        radius_of_influence=12000)
    """

    @property
    def geometry(self) -> DataGeometry:
        """Return POINT geometry."""
        return DataGeometry.POINT

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
        """Pair point observations with model grid.

        Parameters
        ----------
        model
            Model Dataset with dims (time, [z], lat, lon).
        obs
            Observation Dataset with dims (time, site).
        radius_of_influence
            Maximum distance in meters for matching. Default 12000m.
        time_tolerance
            Maximum time difference for matching.
        vertical_method
            Not used for surface observations.
        horizontal_method
            Horizontal matching method ('nearest' only currently).
        **kwargs
            Additional options:
            - extract_surface: bool, whether to extract surface level (default True)

        Returns
        -------
        xr.Dataset
            Paired dataset with model values at observation locations.
        """
        if radius_of_influence is None:
            radius_of_influence = 12000.0

        extract_surface = kwargs.get("extract_surface", True)

        # Get coordinates
        model_lat, model_lon = self._get_model_coords(model)
        obs_lat, obs_lon = self._get_obs_coords(obs)

        # Extract surface level if model is 3D
        if extract_surface:
            model_surface = self._extract_surface(model)
        else:
            model_surface = model

        # Find site dimension in obs
        site_dim = self._get_site_dim(obs)

        # Get unique site locations
        if obs_lat.dims == (site_dim,):
            site_lats = obs_lat.values
            site_lons = obs_lon.values
        else:
            # Lat/lon may be (time, site) - take first time
            site_lats = obs_lat.isel(time=0).values if "time" in obs_lat.dims else obs_lat.values
            site_lons = obs_lon.isel(time=0).values if "time" in obs_lon.dims else obs_lon.values

        # Find nearest model grid indices for each site
        lat_idx, lon_idx = self._find_nearest_indices(
            model_lat, model_lon,
            xr.DataArray(site_lats), xr.DataArray(site_lons),
            radius_of_influence=radius_of_influence,
        )

        # Extract model values at observation sites
        model_at_sites = self._extract_at_sites(
            model_surface, model_lat, model_lon, lat_idx.values, lon_idx.values, site_dim
        )

        # Interpolate model to observation times
        if "time" in model_at_sites.dims and "time" in obs.dims:
            obs_times = obs["time"]
            model_at_sites = self._interpolate_time(
                model_at_sites, obs_times, method="nearest"
            )

        # Combine into paired dataset
        paired = self._create_paired_output(obs, model_at_sites, site_dim)

        return paired

    def _get_site_dim(self, obs: xr.Dataset) -> str:
        """Find the site dimension name in observations.

        Parameters
        ----------
        obs
            Observation dataset.

        Returns
        -------
        str
            Site dimension name.
        """
        for dim in ["site", "station", "x", "location"]:
            if dim in obs.dims:
                return dim

        raise PairingError(
            f"Cannot find site dimension in observations. "
            f"Available dims: {list(obs.dims)}"
        )

    def _extract_at_sites(
        self,
        model: xr.Dataset,
        model_lat: xr.DataArray,
        model_lon: xr.DataArray,
        lat_idx: np.ndarray[Any, np.dtype[Any]],
        lon_idx: np.ndarray[Any, np.dtype[Any]],
        site_dim: str,
    ) -> xr.Dataset:
        """Extract model values at observation site locations.

        Parameters
        ----------
        model
            Model dataset (surface level).
        model_lat, model_lon
            Model coordinate arrays.
        lat_idx, lon_idx
            Indices of nearest model grid cells.
        site_dim
            Name of site dimension.

        Returns
        -------
        xr.Dataset
            Model values at site locations with site dimension.
        """
        n_sites = len(lat_idx)

        # Determine lat/lon dimension names
        if model_lat.ndim == 1:
            lat_dim = model_lat.dims[0]
            lon_dim = model_lon.dims[0]
        else:
            # Curvilinear grid - assume (y, x) or similar
            lat_dim = model_lat.dims[0]
            lon_dim = model_lat.dims[1]

        # Create site coordinate
        site_coord = np.arange(n_sites)

        # Handle invalid indices (outside radius of influence)
        valid_mask = (lat_idx >= 0) & (lon_idx >= 0)

        # Create DataArray indexers for vectorized extraction
        lat_indexer = xr.DataArray(
            np.where(valid_mask, lat_idx, 0),  # Use 0 for invalid, mask later
            dims=[site_dim]
        )
        lon_indexer = xr.DataArray(
            np.where(valid_mask, lon_idx, 0),
            dims=[site_dim]
        )

        # Extract all sites at once using advanced indexing
        extracted = model.isel({lat_dim: lat_indexer, lon_dim: lon_indexer})

        # Load data to numpy with optimized parallel scheduler
        # Use threaded scheduler with multiple workers for parallel file I/O
        n_workers = min(32, os.cpu_count() or 4)
        with dask.config.set(scheduler='threads', num_workers=n_workers):
            extracted = extracted.compute()

        # Mask invalid sites with NaN
        if not valid_mask.all():
            for var in extracted.data_vars:
                extracted[var] = extracted[var].where(
                    xr.DataArray(valid_mask, dims=[site_dim])
                )

        # Build output dataset with proper coordinates
        coords = {site_dim: site_coord}
        if "time" in model.coords:
            coords["time"] = model.coords["time"].values

        return xr.Dataset(
            {str(var): extracted[var] for var in extracted.data_vars},
            coords=coords
        )

    def _create_paired_output(
        self,
        obs: xr.Dataset,
        model_at_sites: xr.Dataset,
        site_dim: str,
    ) -> xr.Dataset:
        """Create the final paired output dataset.

        Parameters
        ----------
        obs
            Observation dataset.
        model_at_sites
            Model values at site locations.
        site_dim
            Site dimension name.

        Returns
        -------
        xr.Dataset
            Combined dataset with both obs and model values.
        """
        # Combine data variables
        data_vars: dict[str, Any] = {}

        # Add observation variables
        for var in obs.data_vars:
            data_vars[str(var)] = obs[var]

        # Add model variables - reassign to obs coordinates to ensure alignment
        # Model was extracted at same site/time locations, just with integer indices
        for var in model_at_sites.data_vars:
            model_var = model_at_sites[var]
            obs_ref = obs[list(obs.data_vars)[0]]

            # Handle dimension mismatch (e.g., obs has extra y=1 dimension)
            if model_var.ndim != obs_ref.ndim:
                # Find extra dims in obs that aren't in model (usually singleton dims like y=1)
                extra_dims = [d for d in obs_ref.dims if d not in model_var.dims]
                if all(obs_ref.sizes[d] == 1 for d in extra_dims):
                    # Squeeze obs to match model dims, then expand model to match obs
                    target_dims = model_var.dims
                    target_coords = {d: obs.coords[d] for d in target_dims if d in obs.coords}
                    model_da = xr.DataArray(
                        model_var.values,
                        dims=target_dims,
                        coords=target_coords,
                        name=var,
                    )
                    # Expand to include the extra singleton dimensions
                    for ed in extra_dims:
                        model_da = model_da.expand_dims({ed: obs.coords[ed].values})
                    # Reorder to match obs dims
                    model_da = model_da.transpose(*obs_ref.dims)
                else:
                    raise PairingError(
                        f"Dimension mismatch between model ({model_var.dims}) and obs ({obs_ref.dims}). "
                        f"Extra dimensions {extra_dims} are not singletons."
                    )
            else:
                # Dimensions match - use original logic
                model_da = xr.DataArray(
                    model_var.values,
                    dims=obs_ref.dims,
                    coords={d: obs.coords[d] for d in obs_ref.dims if d in obs.coords},
                    name=var,
                )
            data_vars[str(var)] = model_da

        return xr.Dataset(data_vars, coords=dict(obs.coords))
