"""Swath-to-grid pairing strategy.

This module implements pairing for satellite swath datasets (L2 products)
with gridded dataset output via per-pixel nearest-neighbor matching.

Note
----
For production satellite analyses prefer :class:`SwathGridStrategy` (or the
external `bin_swath_to_grid` helper in ``pairing/grid_binning.py``). Real L2
swaths have 10^5-10^6 pixels and per-pixel nearest-neighbor matching is too
slow; the binning path collapses pixels onto a target grid once and then
pairs grid-to-grid. MODIS L2 geometry (``datasets/satellite/modis_l2.py``)
follow that pattern and emit ``geometry = "GRID"``.

This class is preserved for possible future use cases that genuinely need
direct per-pixel pairing (e.g. small swaths, sparse retrievals, debugging).
It is not on the current production path.
"""

from __future__ import annotations

import logging
from typing import Any, Hashable

import numpy as np
import xarray as xr

from davinci_monet.core.exceptions import PairingError
from davinci_monet.core.protocols import DataGeometry
from davinci_monet.core.types import TimeDelta
from davinci_monet.pairing.strategies.base import BasePairingStrategy

_logger = logging.getLogger(__name__)


class SwathStrategy(BasePairingStrategy):
    """Pairing strategy for satellite swath datasets.

    Matches satellite swath datasets (L2 products) to dataset
    grid, optionally applying averaging kernels.

    The strategy:
    1. For each swath pixel, finds nearest dataset grid cell
    2. Optionally matches dataset to satellite overpass time
    3. Optionally applies averaging kernels to dataset profiles
    4. Creates paired dataset with collocated values

    .. note::
        Production satellite analyses use :class:`SwathGridStrategy` or the
        external ``bin_swath_to_grid`` helper, which collapse pixels onto a
        target grid before pairing. This direct per-pixel class is preserved
        for possible future use and is not on the current production path —
        see the module docstring.

    Examples
    --------
    >>> strategy = SwathStrategy()
    >>> paired = strategy.pair_sources(y_data, satellite_data,
    ...                        apply_averaging_kernel=True)
    """

    @property
    def geometry(self) -> DataGeometry:
        """Return SWATH geometry."""
        return DataGeometry.SWATH

    def pair_sources(
        self,
        x_data: xr.Dataset,
        y_data: xr.Dataset,
        radius_of_influence: float | None = None,
        time_tolerance: TimeDelta | None = None,
        vertical_method: str = "linear",
        horizontal_method: str = "nearest",
        **kwargs: Any,
    ) -> xr.Dataset:
        """Pair swath datasets with dataset grid.

        Parameters
        ----------
        dataset
            Dataset Dataset with dims (time, [z], lat, lon).
        geometry
            Dataset Dataset with dims (time, scanline, pixel) or similar.
        radius_of_influence
            Maximum distance in meters for matching.
        time_tolerance
            Maximum time difference for matching.
        vertical_method
            Vertical interpolation method.
        horizontal_method
            Horizontal matching method.
        **kwargs
            Additional options:
            - apply_averaging_kernel: bool, whether to apply AK
            - ak_var: str, name of averaging kernel variable
            - match_overpass: bool, whether to match dataset to overpass time

        Returns
        -------
        xr.Dataset
            Paired dataset with y values at swath pixels.
        """
        dataset = y_data
        geometry = x_data

        apply_ak = kwargs.get("apply_averaging_kernel", False)
        match_overpass = kwargs.get("match_overpass", False)

        # Get coordinates
        y_lat, y_lon = self._get_dataset_coords(dataset)
        x_lat, x_lon = self._get_geometry_coords(geometry)

        # Flatten swath coordinates for processing
        x_lat_flat = x_lat.values.flatten()
        x_lon_flat = x_lon.values.flatten()
        n_pixels = len(x_lat_flat)

        # Find nearest dataset grid for each pixel
        lat_idx, lon_idx = self._find_nearest_indices(
            y_lat,
            y_lon,
            xr.DataArray(x_lat_flat),
            xr.DataArray(x_lon_flat),
            radius_of_influence=radius_of_influence,
        )

        # Mask geometry values at pixels outside radius_of_influence so the paired
        # output has geometry and dataset NaN at the same locations. Swath data is
        # inherently 2D (scanline x pixel), so we mask rather than drop —
        # preserving the swath geometry for downstream spatial plotting.
        # Without this, _extract_at_pixels NaNs the dataset side but the geometry
        # side retains valid values, polluting cross-pixel aggregates.
        valid_flat = (lat_idx.values >= 0) & (lon_idx.values >= 0)
        if not valid_flat.all():
            spatial_dims = x_lat.dims
            valid_2d = valid_flat.reshape(x_lat.shape)
            valid_da = xr.DataArray(valid_2d, dims=spatial_dims)
            _logger.info(
                "SwathStrategy: masking %d swath pixel(s) outside %.0f m "
                "radius of influence (kept %d/%d).",
                int((~valid_flat).sum()),
                radius_of_influence,
                int(valid_flat.sum()),
                len(valid_flat),
            )
            masked = geometry.copy()
            for var in geometry.data_vars:
                if all(d in geometry[var].dims for d in spatial_dims):
                    masked[var] = geometry[var].where(valid_da)
            geometry = masked

        # Handle time matching
        if match_overpass and "time" in geometry.coords:
            # Get overpass times and match dataset
            y_matched = self._match_to_overpass(dataset, geometry)
        else:
            y_matched = dataset

        # Extract dataset values at pixel locations
        y_at_pixels = self._extract_at_pixels(
            y_matched,
            y_lat,
            y_lon,
            lat_idx.values,
            lon_idx.values,
            geometry.shape if hasattr(geometry, "shape") else x_lat.shape,
        )

        # Apply averaging kernel if requested
        if apply_ak:
            ak_var = kwargs.get("ak_var", "averaging_kernel")
            if ak_var in geometry:
                y_at_pixels = self._apply_averaging_kernel(y_at_pixels, geometry, ak_var)

        # Create paired output
        paired = self._create_paired_output(geometry, y_at_pixels)

        return paired

    def _match_to_overpass(
        self,
        dataset: xr.Dataset,
        geometry: xr.Dataset,
    ) -> xr.Dataset:
        """Match dataset to satellite overpass times.

        Parameters
        ----------
        dataset
            Dataset dataset.
        geometry
            Dataset dataset with time info.

        Returns
        -------
        xr.Dataset
            Dataset data at overpass times.
        """
        if "time" not in dataset.dims:
            return dataset

        # Get dataset times
        if "time" in geometry.coords:
            x_times = geometry["time"]
            if x_times.ndim > 0:
                # Use median time as representative overpass
                x_time = x_times.values.flat[len(x_times.values.flat) // 2]
            else:
                x_time = x_times.values
            return dataset.sel(time=x_time, method="nearest")

        return dataset

    def _extract_at_pixels(
        self,
        dataset: xr.Dataset,
        y_lat: xr.DataArray,
        y_lon: xr.DataArray,
        lat_idx: np.ndarray[Any, np.dtype[Any]],
        lon_idx: np.ndarray[Any, np.dtype[Any]],
        output_shape: tuple[int, ...],
    ) -> xr.Dataset:
        """Extract dataset values at pixel locations.

        Parameters
        ----------
        dataset
            Dataset dataset.
        dataset_lat, dataset_lon
            Dataset coordinate arrays.
        lat_idx, lon_idx
            Flat arrays of nearest grid indices.
        output_shape
            Shape to reshape output to match swath.

        Returns
        -------
        xr.Dataset
            Dataset values at pixel locations.
        """
        n_pixels = len(lat_idx)

        # Determine dimension names
        if y_lat.ndim == 1:
            lat_dim = y_lat.dims[0]
            lon_dim = y_lon.dims[0]
        else:
            lat_dim = y_lat.dims[0]
            lon_dim = y_lat.dims[1]

        # Build output for each variable
        data_vars: dict[str, tuple[tuple[str, ...], np.ndarray[Any, np.dtype[Any]]]] = {}

        for var in dataset.data_vars:
            var_data = dataset[var]

            # Determine which dims to keep (excluding lat/lon)
            keep_dims = [d for d in var_data.dims if d not in (lat_dim, lon_dim)]

            # Build output shape
            out_shape = []
            for d in keep_dims:
                out_shape.append(var_data.sizes[d])
            out_shape.extend(output_shape)

            # Extract values
            out_data = np.full([n_pixels], np.nan)

            for i in range(n_pixels):
                if lat_idx[i] < 0 or lon_idx[i] < 0:
                    continue

                if y_lat.ndim == 1:
                    selection = {lat_dim: lat_idx[i], lon_dim: lon_idx[i]}
                else:
                    selection = {lat_dim: lat_idx[i], lon_dim: lon_idx[i]}

                point_val = var_data.isel(selection)

                # If still has dimensions, take mean (for vertical)
                if point_val.ndim > 0:
                    out_data[i] = float(point_val.mean().values)
                else:
                    out_data[i] = float(point_val.values)

            # Reshape to swath dimensions
            out_data_reshaped = out_data.reshape(output_shape)
            out_dims = ("y", "x") if len(output_shape) == 2 else ("pixel",)

            data_vars[str(var)] = (out_dims, out_data_reshaped)

        return xr.Dataset(data_vars)

    def _apply_averaging_kernel(
        self,
        y_data: xr.Dataset,
        geometry: xr.Dataset,
        ak_var: str,
    ) -> xr.Dataset:
        """Apply satellite averaging kernel to dataset data.

        Parameters
        ----------
        y_data
            Dataset data at pixel locations.
        geometry
            Dataset dataset containing averaging kernel.
        ak_var
            Name of averaging kernel variable.

        Returns
        -------
        xr.Dataset
            Dataset data with averaging kernel applied.
        """
        # This is a placeholder - full AK application requires
        # knowledge of the specific satellite product
        # For now, just return dataset data unchanged
        return y_data

    def _create_paired_output(
        self,
        geometry: xr.Dataset,
        y_at_pixels: xr.Dataset,
    ) -> xr.Dataset:
        """Create the final paired output dataset.

        Parameters
        ----------
        geometry
            Dataset dataset.
        dataset_at_pixels
            Dataset values at pixel locations.

        Returns
        -------
        xr.Dataset
            Combined dataset.
        """
        # Combine coordinates
        coords = dict(geometry.coords)

        # Combine data variables
        data_vars: dict[str, Any] = {}

        # Add dataset variables
        for var in geometry.data_vars:
            data_vars[str(var)] = geometry[var]

        # Add dataset variables with prefix
        for var in y_at_pixels.data_vars:
            y_var_name = f"y_{var}"
            data_vars[y_var_name] = y_at_pixels[var]

        return xr.Dataset(data_vars, coords=coords)
