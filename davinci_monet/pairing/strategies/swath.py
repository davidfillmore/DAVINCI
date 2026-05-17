"""Swath-to-grid pairing strategy.

This module implements pairing for satellite swath observations (L2 products)
with gridded model output.
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
    """Pairing strategy for satellite swath observations.

    Matches satellite swath observations (L2 products) to model
    grid, optionally applying averaging kernels.

    The strategy:
    1. For each swath pixel, finds nearest model grid cell
    2. Optionally matches model to satellite overpass time
    3. Optionally applies averaging kernels to model profiles
    4. Creates paired dataset with collocated values

    Examples
    --------
    >>> strategy = SwathStrategy()
    >>> paired = strategy.pair(model_data, satellite_data,
    ...                        apply_averaging_kernel=True)
    """

    @property
    def geometry(self) -> DataGeometry:
        """Return SWATH geometry."""
        return DataGeometry.SWATH

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
        """Pair swath observations with model grid.

        Parameters
        ----------
        model
            Model Dataset with dims (time, [z], lat, lon).
        obs
            Observation Dataset with dims (time, scanline, pixel) or similar.
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
            - match_overpass: bool, whether to match model to overpass time

        Returns
        -------
        xr.Dataset
            Paired dataset with model values at swath pixels.
        """
        apply_ak = kwargs.get("apply_averaging_kernel", False)
        match_overpass = kwargs.get("match_overpass", False)

        # Get coordinates
        model_lat, model_lon = self._get_model_coords(model)
        obs_lat, obs_lon = self._get_obs_coords(obs)

        # Flatten swath coordinates for processing
        obs_lat_flat = obs_lat.values.flatten()
        obs_lon_flat = obs_lon.values.flatten()
        n_pixels = len(obs_lat_flat)

        # Find nearest model grid for each pixel
        lat_idx, lon_idx = self._find_nearest_indices(
            model_lat,
            model_lon,
            xr.DataArray(obs_lat_flat),
            xr.DataArray(obs_lon_flat),
            radius_of_influence=radius_of_influence,
        )

        # Mask obs values at pixels outside radius_of_influence so the paired
        # output has obs and model NaN at the same locations. Swath data is
        # inherently 2D (scanline x pixel), so we mask rather than drop —
        # preserving the swath geometry for downstream spatial plotting.
        # Without this, _extract_at_pixels NaNs the model side but the obs
        # side retains valid values, polluting cross-pixel aggregates.
        valid_flat = (lat_idx.values >= 0) & (lon_idx.values >= 0)
        if not valid_flat.all():
            spatial_dims = obs_lat.dims
            valid_2d = valid_flat.reshape(obs_lat.shape)
            valid_da = xr.DataArray(valid_2d, dims=spatial_dims)
            _logger.info(
                "SwathStrategy: masking %d swath pixel(s) outside %.0f m "
                "radius of influence (kept %d/%d).",
                int((~valid_flat).sum()),
                radius_of_influence,
                int(valid_flat.sum()),
                len(valid_flat),
            )
            masked = obs.copy()
            for var in obs.data_vars:
                if all(d in obs[var].dims for d in spatial_dims):
                    masked[var] = obs[var].where(valid_da)
            obs = masked

        # Handle time matching
        if match_overpass and "time" in obs.coords:
            # Get overpass times and match model
            model_matched = self._match_to_overpass(model, obs)
        else:
            model_matched = model

        # Extract model values at pixel locations
        model_at_pixels = self._extract_at_pixels(
            model_matched,
            model_lat,
            model_lon,
            lat_idx.values,
            lon_idx.values,
            obs.shape if hasattr(obs, "shape") else obs_lat.shape,
        )

        # Apply averaging kernel if requested
        if apply_ak:
            ak_var = kwargs.get("ak_var", "averaging_kernel")
            if ak_var in obs:
                model_at_pixels = self._apply_averaging_kernel(model_at_pixels, obs, ak_var)

        # Create paired output
        paired = self._create_paired_output(obs, model_at_pixels)

        return paired

    def _match_to_overpass(
        self,
        model: xr.Dataset,
        obs: xr.Dataset,
    ) -> xr.Dataset:
        """Match model to satellite overpass times.

        Parameters
        ----------
        model
            Model dataset.
        obs
            Observation dataset with time info.

        Returns
        -------
        xr.Dataset
            Model data at overpass times.
        """
        if "time" not in model.dims:
            return model

        # Get observation times
        if "time" in obs.coords:
            obs_times = obs["time"]
            if obs_times.ndim > 0:
                # Use median time as representative overpass
                obs_time = obs_times.values.flat[len(obs_times.values.flat) // 2]
            else:
                obs_time = obs_times.values
            return model.sel(time=obs_time, method="nearest")

        return model

    def _extract_at_pixels(
        self,
        model: xr.Dataset,
        model_lat: xr.DataArray,
        model_lon: xr.DataArray,
        lat_idx: np.ndarray[Any, np.dtype[Any]],
        lon_idx: np.ndarray[Any, np.dtype[Any]],
        output_shape: tuple[int, ...],
    ) -> xr.Dataset:
        """Extract model values at pixel locations.

        Parameters
        ----------
        model
            Model dataset.
        model_lat, model_lon
            Model coordinate arrays.
        lat_idx, lon_idx
            Flat arrays of nearest grid indices.
        output_shape
            Shape to reshape output to match swath.

        Returns
        -------
        xr.Dataset
            Model values at pixel locations.
        """
        n_pixels = len(lat_idx)

        # Determine dimension names
        if model_lat.ndim == 1:
            lat_dim = model_lat.dims[0]
            lon_dim = model_lon.dims[0]
        else:
            lat_dim = model_lat.dims[0]
            lon_dim = model_lat.dims[1]

        # Build output for each variable
        data_vars: dict[str, tuple[tuple[str, ...], np.ndarray[Any, np.dtype[Any]]]] = {}

        for var in model.data_vars:
            var_data = model[var]

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

                if model_lat.ndim == 1:
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
        model_data: xr.Dataset,
        obs: xr.Dataset,
        ak_var: str,
    ) -> xr.Dataset:
        """Apply satellite averaging kernel to model data.

        Parameters
        ----------
        model_data
            Model data at pixel locations.
        obs
            Observation dataset containing averaging kernel.
        ak_var
            Name of averaging kernel variable.

        Returns
        -------
        xr.Dataset
            Model data with averaging kernel applied.
        """
        # This is a placeholder - full AK application requires
        # knowledge of the specific satellite product
        # For now, just return model data unchanged
        return model_data

    def _create_paired_output(
        self,
        obs: xr.Dataset,
        model_at_pixels: xr.Dataset,
    ) -> xr.Dataset:
        """Create the final paired output dataset.

        Parameters
        ----------
        obs
            Observation dataset.
        model_at_pixels
            Model values at pixel locations.

        Returns
        -------
        xr.Dataset
            Combined dataset.
        """
        # Combine coordinates
        coords = dict(obs.coords)

        # Combine data variables
        data_vars: dict[str, Any] = {}

        # Add observation variables
        for var in obs.data_vars:
            data_vars[str(var)] = obs[var]

        # Add model variables with prefix
        for var in model_at_pixels.data_vars:
            model_var_name = f"model_{var}"
            data_vars[model_var_name] = model_at_pixels[var]

        return xr.Dataset(data_vars, coords=coords)
