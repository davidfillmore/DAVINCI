"""Point-to-grid pairing strategy.

This module implements pairing for point datasets (surface stations,
ground sites) with gridded dataset output.
"""

from __future__ import annotations

import logging
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

_logger = logging.getLogger(__name__)


class PointStrategy(BasePairingStrategy):
    """Pairing strategy for point datasets.

    Matches fixed-location datasets (surface stations, ground sites)
    to the nearest dataset grid cell within the radius of influence.

    The strategy:
    1. Finds nearest dataset grid cell for each dataset site
    2. Extracts surface level from dataset (if 3D)
    3. Interpolates dataset to dataset times
    4. Creates paired dataset with aligned values

    Examples
    --------
    >>> strategy = PointStrategy()
    >>> paired = strategy.pair_sources(y_data, x_data,
    ...                        radius_of_influence=12000)
    """

    @property
    def geometry(self) -> DataGeometry:
        """Return POINT geometry."""
        return DataGeometry.POINT

    def pair_sources(
        self,
        x_data: xr.Dataset,
        y_data: xr.Dataset,
        radius_of_influence: float | None = None,
        time_tolerance: TimeDelta | None = None,
        vertical_method: str = "nearest",
        horizontal_method: str = "nearest",
        time_method: str = "nearest",
        **kwargs: Any,
    ) -> xr.Dataset:
        """Pair point datasets with dataset grid.

        Parameters
        ----------
        dataset
            Dataset Dataset with dims (time, [z], lat, lon).
        geometry
            Dataset Dataset with dims (time, site).
        radius_of_influence
            Maximum distance in meters for matching. Default 12000m.
        time_tolerance
            Maximum time difference for matching.
        vertical_method
            Not used for surface datasets.
        horizontal_method
            Horizontal matching method ('nearest' only currently).
        time_method
            Time interpolation method ('nearest' or 'linear'). Use 'linear'
            when the dataset has sparse time output relative to datasets
            (e.g. 6-hourly WRF-Chem snapshots vs hourly AirNow) to avoid the
            step-function artifact produced by 'nearest'. Default 'nearest'
            preserves prior behavior.
        **kwargs
            Additional options:
            - extract_surface: bool, whether to extract surface level (default True)

        Returns
        -------
        xr.Dataset
            Paired dataset with dataset values at dataset locations.
        """
        dataset = y_data
        geometry = x_data

        if radius_of_influence is None:
            radius_of_influence = 12000.0

        extract_surface = kwargs.get("extract_surface", True)
        dask_num_workers = kwargs.get("dask_num_workers")

        # Get coordinates
        dataset_lat, dataset_lon = self._get_dataset_coords(dataset)
        geometry_lat, geometry_lon = self._get_geometry_coords(geometry)

        # Extract surface level if dataset is 3D
        if extract_surface:
            dataset_surface = self._extract_surface(dataset)
        else:
            dataset_surface = dataset

        # Find site dimension in geometry
        site_dim = self._get_site_dim(geometry)

        # Get unique site locations
        if geometry_lat.dims == (site_dim,):
            site_lats = geometry_lat.values
            site_lons = geometry_lon.values
        else:
            # Lat/lon may be (time, site) - take first time
            site_lats = (
                geometry_lat.isel(time=0).values
                if "time" in geometry_lat.dims
                else geometry_lat.values
            )
            site_lons = (
                geometry_lon.isel(time=0).values
                if "time" in geometry_lon.dims
                else geometry_lon.values
            )

        # Find nearest dataset grid indices for each site
        lat_idx, lon_idx = self._find_nearest_indices(
            dataset_lat,
            dataset_lon,
            xr.DataArray(site_lats),
            xr.DataArray(site_lons),
            radius_of_influence=radius_of_influence,
        )

        # Drop geometry sites that fall outside radius_of_influence. Without this,
        # _extract_at_sites masks the dataset to NaN at unpaired sites but leaves
        # the geometry values intact, so cross-site aggregates (timeseries domain-mean)
        # are polluted by sites with no dataset match.
        valid = (lat_idx.values >= 0) & (lon_idx.values >= 0)
        if not valid.all():
            keep = np.where(valid)[0]
            _logger.info(
                "PointStrategy: dropping %d geometry site(s) outside %.0f m "
                "radius of influence (kept %d/%d).",
                int((~valid).sum()),
                radius_of_influence,
                int(valid.sum()),
                len(valid),
            )
            geometry = geometry.isel({site_dim: keep})
            site_lats = site_lats[keep]
            site_lons = site_lons[keep]
            lat_idx = xr.DataArray(lat_idx.values[keep])
            lon_idx = xr.DataArray(lon_idx.values[keep])

        # Extract dataset values at dataset sites
        dataset_at_sites = self._extract_at_sites(
            dataset_surface,
            dataset_lat,
            dataset_lon,
            lat_idx.values,
            lon_idx.values,
            site_dim,
            dask_num_workers=dask_num_workers,
        )

        # Interpolate dataset to dataset times
        if "time" in dataset_at_sites.dims and "time" in geometry.dims:
            geometry_times = geometry["time"]
            dataset_at_sites = self._interpolate_time(
                dataset_at_sites, geometry_times, method=time_method, time_tolerance=time_tolerance
            )

        # Combine into paired dataset
        paired = self._create_paired_output(geometry, dataset_at_sites, site_dim)

        return paired

    def _get_site_dim(self, geometry: xr.Dataset) -> str:
        """Find the site dimension name in datasets.

        Parameters
        ----------
        geometry
            Dataset dataset.

        Returns
        -------
        str
            Site dimension name.
        """
        for dim in ["site", "station", "x", "location"]:
            if dim in geometry.dims:
                return dim

        raise PairingError(
            f"Cannot find site dimension in datasets. " f"Available dims: {list(geometry.dims)}"
        )

    def _extract_at_sites(
        self,
        dataset: xr.Dataset,
        dataset_lat: xr.DataArray,
        dataset_lon: xr.DataArray,
        lat_idx: np.ndarray[Any, np.dtype[Any]],
        lon_idx: np.ndarray[Any, np.dtype[Any]],
        site_dim: str,
        dask_num_workers: int | None = None,
    ) -> xr.Dataset:
        """Extract dataset values at dataset site locations.

        Parameters
        ----------
        dataset
            Dataset dataset (surface level).
        dataset_lat, dataset_lon
            Dataset coordinate arrays.
        lat_idx, lon_idx
            Indices of nearest dataset grid cells.
        site_dim
            Name of site dimension.

        Returns
        -------
        xr.Dataset
            Dataset values at site locations with site dimension.
        """
        n_sites = len(lat_idx)

        # Determine lat/lon dimension names
        if dataset_lat.ndim == 1:
            lat_dim = dataset_lat.dims[0]
            lon_dim = dataset_lon.dims[0]
        else:
            # Curvilinear grid - assume (y, x) or similar
            lat_dim = dataset_lat.dims[0]
            lon_dim = dataset_lat.dims[1]

        # Create site coordinate
        site_coord = np.arange(n_sites)

        # Handle invalid indices (outside radius of influence)
        valid_mask = (lat_idx >= 0) & (lon_idx >= 0)

        # Create DataArray indexers for vectorized extraction
        lat_indexer = xr.DataArray(
            np.where(valid_mask, lat_idx, 0), dims=[site_dim]  # Use 0 for invalid, mask later
        )
        lon_indexer = xr.DataArray(np.where(valid_mask, lon_idx, 0), dims=[site_dim])

        # Extract all sites at once using advanced indexing
        extracted = dataset.isel({lat_dim: lat_indexer, lon_dim: lon_indexer})

        # Load data to numpy with optimized parallel scheduler
        # Use threaded scheduler with multiple workers for parallel file I/O
        if dask_num_workers is not None:
            n_workers = max(1, int(dask_num_workers))
        else:
            n_workers = min(32, os.cpu_count() or 4)
        with dask.config.set(scheduler="threads", num_workers=n_workers):
            extracted = extracted.compute()

        # Mask invalid sites with NaN
        if not valid_mask.all():
            for var in extracted.data_vars:
                extracted[var] = extracted[var].where(xr.DataArray(valid_mask, dims=[site_dim]))

        # Build output dataset with proper coordinates
        coords = {site_dim: site_coord}
        if "time" in dataset.coords:
            coords["time"] = dataset.coords["time"].values

        return xr.Dataset({str(var): extracted[var] for var in extracted.data_vars}, coords=coords)

    def _create_paired_output(
        self,
        geometry: xr.Dataset,
        dataset_at_sites: xr.Dataset,
        site_dim: str,
    ) -> xr.Dataset:
        """Create the final paired output dataset.

        Parameters
        ----------
        geometry
            Dataset dataset.
        dataset_at_sites
            Dataset values at site locations.
        site_dim
            Site dimension name.

        Returns
        -------
        xr.Dataset
            Combined dataset with both geometry and dataset values.
        """
        # Combine data variables
        data_vars: dict[str, Any] = {}

        # Add dataset variables
        geometry_var_names = set()
        for var in geometry.data_vars:
            var_name = str(var)
            data_vars[var_name] = geometry[var]
            geometry_var_names.add(var_name)

        # Add dataset variables - reassign to geometry coordinates to ensure alignment
        # Dataset was extracted at same site/time locations, just with integer indices
        for var in dataset_at_sites.data_vars:
            y_var = dataset_at_sites[var]
            geometry_ref = geometry[list(geometry.data_vars)[0]]

            # Handle dimension mismatch (e.g., geometry has extra y=1 dimension)
            if y_var.ndim != geometry_ref.ndim:
                # Find extra dims in geometry that aren't in dataset (usually singleton dims like y=1)
                extra_dims = [d for d in geometry_ref.dims if d not in y_var.dims]
                if all(geometry_ref.sizes[d] == 1 for d in extra_dims):
                    # Squeeze geometry to match dataset dims, then expand dataset to match geometry
                    target_dims = y_var.dims
                    target_coords = {
                        d: geometry.coords[d] for d in target_dims if d in geometry.coords
                    }
                    dataset_da = xr.DataArray(
                        y_var.values,
                        dims=target_dims,
                        coords=target_coords,
                        name=var,
                    )
                    # Expand to include the extra singleton dimensions
                    for ed in extra_dims:
                        dataset_da = dataset_da.expand_dims({ed: geometry.coords[ed].values})
                    # Reorder to match geometry dims
                    dataset_da = dataset_da.transpose(*geometry_ref.dims)
                else:
                    raise PairingError(
                        f"Dimension mismatch between dataset ({y_var.dims}) and geometry ({geometry_ref.dims}). "
                        f"Extra dimensions {extra_dims} are not singletons."
                    )
            else:
                # Dimensions match - use original logic
                dataset_da = xr.DataArray(
                    y_var.values,
                    dims=geometry_ref.dims,
                    coords={
                        d: geometry.coords[d] for d in geometry_ref.dims if d in geometry.coords
                    },
                    name=var,
                )
            var_name = str(var)
            if var_name in geometry_var_names:
                var_name = f"dataset_{var_name}"
            data_vars[var_name] = dataset_da

        return xr.Dataset(data_vars, coords=dict(geometry.coords))
