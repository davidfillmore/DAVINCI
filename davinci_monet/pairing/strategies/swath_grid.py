"""Swath-to-grid pairing strategy using numba-accelerated binning.

Bins satellite L2 swath pixels onto a uniform (time, lon, lat) grid,
then pairs with dataset data on the same grid. This is the recommended
strategy for all L2 satellite products at scale.

Ported from the MELODIES-MONET intermediate grid approach
(grid_util.update_data_grid).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from davinci_monet.core.exceptions import PairingError
from davinci_monet.core.protocols import DataGeometry
from davinci_monet.core.types import TimeDelta
from davinci_monet.pairing.grid_binning import bin_swath_to_grid, edges_from_centers, normalize_grid
from davinci_monet.pairing.strategies.base import BasePairingStrategy


class SwathGridStrategy(BasePairingStrategy):
    """Pairing strategy that bins satellite swath data onto a uniform grid.

    The strategy:
    1. Defines a target grid (from dataset, resolution, or explicit dims)
    2. Bins all swath pixels into grid cells using numba-accelerated loop
    3. Normalizes (mean = sum / count)
    4. Aligns dataset data on the same grid
    5. Returns paired dataset with geometry, dataset, and pixel counts

    Examples
    --------
    >>> strategy = SwathGridStrategy()
    >>> paired = strategy.pair_sources(
    ...     y_data, satellite_data,
    ...     grid_mode="match_dataset",
    ...     time_resolution="1D",
    ... )
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
        vertical_method: str = "nearest",
        horizontal_method: str = "nearest",
        **kwargs: Any,
    ) -> xr.Dataset:
        """Pair swath datasets with dataset by binning onto a common grid.

        Parameters
        ----------
        dataset
            Dataset Dataset with dims (time, [z], lat, lon).
        geometry
            Dataset Dataset with swath dimensions and 2D lat/lon.
        radius_of_influence
            Not used (kept for interface compatibility).
        time_tolerance
            Not used (temporal binning is controlled by time_resolution).
        vertical_method
            Not used.
        horizontal_method
            Not used.
        **kwargs
            Strategy-specific options:

            - grid_mode : str
                Grid definition mode: "match_dataset", "resolution", or
                "explicit". Default "match_dataset".
            - time_resolution : str
                Pandas frequency string for temporal binning (e.g. "1D").
                Default "1D".
            - resolution : float
                Grid spacing in degrees (for grid_mode="resolution").
            - ntime, nlat, nlon : int
                Explicit grid dimensions (for grid_mode="explicit").
            - min_geometry_count : int
                Minimum pixel count per cell. Cells below this are
                masked to NaN. Default 1.
            - x_var : str
                Geometry variable name to bin.
            - y_var : str
                Dataset variable name to extract.

        Returns
        -------
        xr.Dataset
            Paired dataset on common grid with geometry, dataset, and count
            variables.
        """
        dataset = y_data
        geometry = x_data

        grid_mode = kwargs.get("grid_mode", "match_dataset")
        time_resolution = kwargs.get("time_resolution", "1D")
        min_geometry_count = kwargs.get("min_geometry_count", 1)
        x_var = kwargs.get("x_var") or kwargs.get("x_var")
        y_var = kwargs.get("y_var") or kwargs.get("y_var")

        # Extract surface if dataset has vertical dimension
        dataset_proc = dataset
        for dim_name in ["lev", "z", "level"]:
            if dim_name in dataset_proc.dims:
                dataset_proc = self._extract_surface(dataset_proc, dim_name)
                break

        # Get dataset coordinates
        dataset_lat, dataset_lon = self._get_dataset_coords(dataset_proc)

        # Determine analysis time range from dataset
        if "time" in dataset_proc.dims:
            dataset_times = pd.DatetimeIndex(dataset_proc["time"].values)
            t_start = dataset_times.min()
            t_end = dataset_times.max()
        else:
            raise PairingError("Dataset dataset must have a time dimension")

        # Build target grid (filter kwargs to avoid duplicating named params)
        consumed = {
            "grid_mode",
            "time_resolution",
            "min_geometry_count",
            "x_var",
            "y_var",
            "x_var",
            "y_var",
        }
        grid_kwargs = {k: v for k, v in kwargs.items() if k not in consumed}
        time_edges, lon_edges, lat_edges, time_centers, lon_centers, lat_centers = self._build_grid(
            grid_mode=grid_mode,
            dataset_lat=dataset_lat,
            dataset_lon=dataset_lon,
            t_start=t_start,
            t_end=t_end,
            time_resolution=time_resolution,
            **grid_kwargs,
        )

        ntime = len(time_centers)
        nlon = len(lon_centers)
        nlat = len(lat_centers)

        # Determine which geometry variable to bin
        if x_var is None:
            data_var_names = list(geometry.data_vars)
            if len(data_var_names) == 0:
                raise PairingError("No data variables in dataset dataset")
            x_var = data_var_names[0]

        # Extract flat arrays from geometry
        geometry_lat, geometry_lon = self._get_geometry_coords(geometry)
        lat_flat = geometry_lat.values.flatten().astype(np.float64)
        lon_flat = geometry_lon.values.flatten().astype(np.float64)
        data_flat = geometry[x_var].values.flatten().astype(np.float64)

        # Handle longitude convention: shift -180..180 to 0..360 if needed
        if lon_edges[0] >= 0 and np.any(lon_flat < 0):
            lon_flat = np.where(lon_flat < 0, lon_flat + 360.0, lon_flat)

        # Get dataset timestamps as epoch seconds, aligned to the same
        # flattening order as ``data_flat`` (i.e. the binned geometry variable).
        time_flat = self._get_geometry_timestamps(geometry, len(data_flat), align_var=x_var)

        # Allocate accumulation arrays
        count_grid = np.zeros((ntime, nlon, nlat), dtype=np.int32)
        data_grid = np.zeros((ntime, nlon, nlat), dtype=np.float64)

        # Bin swath pixels onto grid (numba fast path)
        bin_swath_to_grid(
            time_edges,
            lon_edges,
            lat_edges,
            time_flat,
            lon_flat,
            lat_flat,
            data_flat,
            count_grid,
            data_grid,
        )

        # Normalize: sum → mean
        normalize_grid(count_grid, data_grid)

        # Apply min_geometry_count filter
        if min_geometry_count > 1:
            data_grid[count_grid < min_geometry_count] = np.nan

        # Build datetime coordinates from time centers (epoch seconds)
        time_coords = pd.to_datetime(time_centers, unit="s")

        # Create gridded geometry dataset
        geometry_gridded = xr.Dataset(
            {
                f"geometry_{x_var}": (["time", "lon", "lat"], data_grid.astype(np.float32)),
                "geometry_count": (["time", "lon", "lat"], count_grid),
            },
            coords={
                "time": time_coords,
                "lon": lon_centers,
                "lat": lat_centers,
            },
        )

        # Extract dataset variable on same grid
        if y_var is None:
            dataset_data_vars = list(dataset_proc.data_vars)
            if len(dataset_data_vars) == 0:
                raise PairingError("No data variables in dataset dataset")
            y_var = str(dataset_data_vars[0])

        dataset_on_grid = self._align_dataset_to_grid(
            dataset_proc,
            y_var,
            time_coords,
            lon_centers,
            lat_centers,
            dataset_lat,
            dataset_lon,
            grid_mode,
        )

        # Create paired output
        paired = geometry_gridded.copy()
        paired[f"dataset_{y_var}"] = dataset_on_grid

        return paired

    def _build_grid(
        self,
        grid_mode: str,
        dataset_lat: xr.DataArray,
        dataset_lon: xr.DataArray,
        t_start: pd.Timestamp,
        t_end: pd.Timestamp,
        time_resolution: str,
        **kwargs: Any,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        """Build target grid edges and centers.

        Returns
        -------
        tuple
            (time_edges, lon_edges, lat_edges,
             time_centers, lon_centers, lat_centers)
            All edges/centers as numpy arrays. Time values are in
            epoch seconds (float64).
        """
        # Time grid: from analysis window + resolution
        time_range = pd.date_range(t_start, t_end, freq=time_resolution)
        if len(time_range) < 1:
            time_range = pd.DatetimeIndex([t_start])
        time_centers_epoch = time_range.values.astype("datetime64[s]").astype(np.float64)
        time_edges = edges_from_centers(time_centers_epoch)

        if grid_mode == "match_dataset":
            if dataset_lat.ndim != 1 or dataset_lon.ndim != 1:
                raise PairingError(
                    "match_dataset grid mode requires 1D dataset lat/lon " "(rectilinear grid)"
                )
            lat_centers = dataset_lat.values.astype(np.float64)
            lon_centers = dataset_lon.values.astype(np.float64)
            lat_edges = edges_from_centers(lat_centers)
            lon_edges = edges_from_centers(lon_centers)

        elif grid_mode == "resolution":
            res = kwargs.get("resolution")
            if res is None:
                raise PairingError("resolution must be specified for grid_mode='resolution'")
            res = float(res)
            lat_centers = np.arange(-90 + res / 2, 90, res)
            lon_centers = np.arange(0 + res / 2, 360, res)
            lat_edges = edges_from_centers(lat_centers)
            lon_edges = edges_from_centers(lon_centers)

        elif grid_mode == "explicit":
            nlat = kwargs.get("nlat")
            nlon = kwargs.get("nlon")
            if nlat is None or nlon is None:
                raise PairingError("nlat and nlon must be specified for grid_mode='explicit'")
            lat_centers = np.linspace(-90, 90, int(nlat))
            lon_centers = np.linspace(0, 360, int(nlon), endpoint=False)
            lat_edges = edges_from_centers(lat_centers)
            lon_edges = edges_from_centers(lon_centers)

        else:
            raise PairingError(f"Unknown grid_mode: {grid_mode}")

        return time_edges, lon_edges, lat_edges, time_centers_epoch, lon_centers, lat_centers

    def _get_geometry_timestamps(
        self, geometry: xr.Dataset, n_pixels: int, align_var: str | None = None
    ) -> np.ndarray:
        """Extract dataset timestamps as flat epoch-second array.

        Parameters
        ----------
        geometry
            Dataset dataset.
        n_pixels
            Total number of pixels (for broadcasting).
        align_var
            Name of the data variable whose pixel order ``data_flat`` follows.
            The time coordinate is broadcast against this variable so per-pixel
            timestamps line up with the binned values. Defaults to the first
            data variable.

        Returns
        -------
        np.ndarray
            Flat float64 array of epoch seconds, one per pixel.
        """
        if "time" in geometry.coords:
            time_da = geometry["time"]
            time_vals = time_da.values
            if np.issubdtype(time_vals.dtype, np.datetime64):
                epoch = time_vals.astype("datetime64[s]").astype(np.float64)
                if epoch.ndim == 0:
                    # Scalar time → broadcast to all pixels
                    return np.full(n_pixels, float(epoch), dtype=np.float64)
                # Dim-aware broadcast: a swath ``time`` is commonly per-scanline
                # (dims ``(scanline,)``) while data is per-pixel (dims
                # ``(scanline, pixel)``). broadcast_like expands ``time`` along
                # the data var's dims regardless of axis order, where positional
                # ``np.broadcast_to`` would fail (trailing-axis rule).
                var_name = align_var if align_var in geometry.data_vars else None
                if var_name is None and geometry.data_vars:
                    var_name = str(list(geometry.data_vars)[0])
                if var_name is not None:
                    # ``epoch`` was already converted via .values above (numpy
                    # path), so reuse it directly without recomputing.
                    epoch_da = xr.DataArray(epoch, dims=time_da.dims)
                    broadcast = epoch_da.broadcast_like(geometry[var_name])
                    # Match the C-order .flatten() used for the data values.
                    epoch_flat = broadcast.transpose(*geometry[var_name].dims).values.flatten()
                    if len(epoch_flat) == n_pixels:
                        return epoch_flat.astype(np.float64)
                return np.full(n_pixels, float(epoch.mean()), dtype=np.float64)
            else:
                return np.full(n_pixels, float(time_vals.flat[0]), dtype=np.float64)

        # No time coordinate — use a single default timestamp
        return np.zeros(n_pixels, dtype=np.float64)

    def _align_dataset_to_grid(
        self,
        dataset: xr.Dataset,
        y_var: str,
        time_coords: pd.DatetimeIndex,
        lon_centers: np.ndarray,
        lat_centers: np.ndarray,
        dataset_lat: xr.DataArray,
        dataset_lon: xr.DataArray,
        grid_mode: str,
    ) -> xr.DataArray:
        """Extract dataset variable aligned to the target grid.

        For match_dataset mode, the grid is already the dataset grid so
        we just select the nearest times. For other modes, we interpolate
        spatially.

        Returns
        -------
        xr.DataArray
            Dataset data on (time, lon, lat) grid.
        """
        var_data = dataset[y_var]

        # Select nearest dataset times
        if "time" in var_data.dims:
            var_data = var_data.sel(time=time_coords, method="nearest")
            var_data = var_data.assign_coords(time=time_coords)

        if grid_mode == "match_dataset":
            # Already on dataset grid — just ensure dim order is (time, lon, lat)
            lat_dim = dataset_lat.dims[0]
            lon_dim = dataset_lon.dims[0]
            # Transpose to (time, lon, lat) to match our output convention
            dim_order: list[str] = []
            if "time" in var_data.dims:
                dim_order.append("time")
            dim_order.extend([str(lon_dim), str(lat_dim)])
            var_data = var_data.transpose(*dim_order)
            # Rename dims to match paired output
            rename_map = {}
            if lat_dim != "lat":
                rename_map[lat_dim] = "lat"
            if lon_dim != "lon":
                rename_map[lon_dim] = "lon"
            if rename_map:
                var_data = var_data.rename(rename_map)
            return var_data.astype(np.float32)
        else:
            # Interpolate dataset to target grid
            lat_dim = dataset_lat.dims[0]
            lon_dim = dataset_lon.dims[0]
            var_interp = var_data.interp(
                {lat_dim: lat_centers, lon_dim: lon_centers},
                method="nearest",
            )
            # Rename dims
            rename_map = {}
            if lat_dim != "lat":
                rename_map[lat_dim] = "lat"
            if lon_dim != "lon":
                rename_map[lon_dim] = "lon"
            if rename_map:
                var_interp = var_interp.rename(rename_map)
            # Ensure (time, lon, lat) order
            dim_order = []
            if "time" in var_interp.dims:
                dim_order.append("time")
            dim_order.extend(["lon", "lat"])
            var_interp = var_interp.transpose(*dim_order)
            return var_interp.astype(np.float32)
