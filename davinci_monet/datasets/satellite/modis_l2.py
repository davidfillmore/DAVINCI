"""MODIS L2 dataset reader with swath-to-grid binning.

Reads MODIS L2 HDF4 granules via ``monetio.sat._modis_l2_mm`` and bins
them onto a target grid using the numba-accelerated binning in
``davinci_monet.pairing.grid_binning``.

The primary entry point for the pipeline is :class:`MODISL2Reader`.
"""

from __future__ import annotations

import fnmatch
import logging
from collections import OrderedDict
from glob import glob
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import xarray as xr

from davinci_monet.pairing.grid_binning import bin_swath_to_grid, edges_from_centers, normalize_grid

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# File subsetting
# ---------------------------------------------------------------------------


def subset_modis_l2_files(
    file_pattern: str,
    start_time: str,
    end_time: str,
) -> list[str]:
    """Filter MODIS L2 files by hourly timestamp encoded in filename.

    MODIS L2 filenames follow the convention::

        MOD04_L2.AYYYYDDD.HHMM.CCC.TIMESTAMP.hdf
        MYD04_L2.AYYYYDDD.HHMM.CCC.TIMESTAMP.hdf

    This function expands the glob, builds an hourly interval over the
    analysis window, and keeps only files whose ``YYYYDDD.HH`` matches.

    Mirrors ``melodies_monet.util.time_interval_subset.subset_MODIS_l2``.

    Parameters
    ----------
    file_pattern
        Glob pattern for MODIS HDF4 files.
    start_time, end_time
        ISO-format analysis time bounds.

    Returns
    -------
    list[str]
        Sorted list of matching file paths.
    """
    all_files = sorted(glob(file_pattern))
    if not all_files:
        return []

    subset_interval = pd.date_range(
        start=start_time,
        end=end_time,
        freq="h",
        inclusive="left",
    )

    interval_files: list[str] = []
    for ts in subset_interval:
        pattern = f"*M?D04_L2.A{ts.strftime('%Y%j.%H')}*.hdf"
        matched = fnmatch.filter(all_files, pattern)
        matched.sort()
        interval_files.extend(matched)

    return interval_files


# ---------------------------------------------------------------------------
# Variable dict builder
# ---------------------------------------------------------------------------


def build_modis_variable_dict(
    variables: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Build a monetio-compatible ``variable_dict`` from DAVINCI config.

    The monetio reader expects::

        {"SDS_Name": {"minimum": float, "maximum": float, "scale": float}}

    Parameters
    ----------
    variables
        DAVINCI variable configs keyed by variable name.

    Returns
    -------
    dict
        ``variable_dict`` for ``monetio.sat._modis_l2_mm.read_mfdataset``.
    """
    result: dict[str, dict[str, Any]] = {}
    for var_name, cfg in variables.items():
        source_name = cfg.get("source_name", var_name)
        entry: dict[str, Any] = {}
        scale = cfg.get("unit_scale", 1.0)
        if scale != 1.0:
            entry["scale"] = scale
        valid_min = cfg.get("valid_min")
        if valid_min is not None:
            entry["minimum"] = valid_min
        valid_max = cfg.get("valid_max")
        if valid_max is not None:
            entry["maximum"] = valid_max
        result[source_name] = entry
    return result


# ---------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------


class MODISL2Reader:
    """Read MODIS L2 granules and grid them for pipeline use.

    This reader wraps ``monetio.sat._modis_l2_mm.read_mfdataset``
    (which returns an ``OrderedDict`` of per-granule datasets) and bins
    the swath pixels onto a target grid.

    Examples
    --------
    >>> reader = MODISL2Reader()
    >>> ds = reader.read_and_grid(
    ...     files, variable_dict,
    ...     lat_centers=dataset_lat, lon_centers=dataset_lon,
    ...     start_time="2019-12-21", end_time="2019-12-23",
    ... )
    """

    def read_granules(
        self,
        files: Sequence[str | Path],
        variable_dict: dict[str, dict[str, Any]],
        *,
        debug: bool = False,
    ) -> OrderedDict:
        """Read MODIS L2 granules via monetio.

        Parameters
        ----------
        files
            List of HDF4 file paths (already subset to analysis window).
        variable_dict
            monetio-format variable specification.
        debug
            Enable monetio debug logging.

        Returns
        -------
        OrderedDict[str, xr.Dataset]
            Keyed by ``YYYYjjjHHMM`` datetime strings.
        """
        import monetio.sat._modis_l2_mm as modis_module

        file_list = [str(f) for f in files]
        return modis_module.read_mfdataset(file_list, variable_dict, debug=debug)

    def read_and_grid(
        self,
        files: Sequence[str | Path],
        variable_dict: dict[str, dict[str, Any]],
        lat_centers: np.ndarray,
        lon_centers: np.ndarray,
        start_time: str,
        end_time: str,
        time_resolution: str = "1D",
        min_sample_count: int = 1,
        *,
        debug: bool = False,
        progress_callback: Any | None = None,
    ) -> xr.Dataset:
        """Read granules and bin onto a target grid.

        Parameters
        ----------
        files
            HDF4 file paths (pre-subset).
        variable_dict
            monetio-format variable specification.
        lat_centers, lon_centers
            1-D arrays defining the target grid.
        start_time, end_time
            ISO-format analysis window.
        time_resolution
            Pandas frequency for temporal bins (default ``"1D"``).
        min_sample_count
            Minimum pixel count per cell; cells below are set to NaN.
        debug
            Enable monetio debug logging.
        progress_callback
            Optional callable for progress messages.

        Returns
        -------
        xr.Dataset
            Gridded dataset with dims ``(time, lon, lat)``.
            Contains one gridded variable per entry in *variable_dict*
            plus ``sample_count``.
        """
        granules = self.read_granules(files, variable_dict, debug=debug)
        if progress_callback:
            progress_callback(f"step: Read {len(granules)} MODIS granules")

        return self._grid_granules(
            granules,
            variable_dict=variable_dict,
            lat_centers=lat_centers,
            lon_centers=lon_centers,
            start_time=start_time,
            end_time=end_time,
            time_resolution=time_resolution,
            min_sample_count=min_sample_count,
            progress_callback=progress_callback,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _grid_granules(
        self,
        granules: OrderedDict,
        *,
        variable_dict: dict[str, dict[str, Any]],
        lat_centers: np.ndarray,
        lon_centers: np.ndarray,
        start_time: str,
        end_time: str,
        time_resolution: str,
        min_sample_count: int,
        progress_callback: Any | None = None,
    ) -> xr.Dataset:
        """Bin an OrderedDict of granules onto a (time, lon, lat) grid."""
        nlat = len(lat_centers)
        nlon = len(lon_centers)

        # Build time grid
        time_range = pd.date_range(start_time, end_time, freq=time_resolution)
        if len(time_range) < 1:
            time_range = pd.DatetimeIndex([pd.Timestamp(start_time)])
        ntime = len(time_range)
        time_centers_epoch = time_range.values.astype("datetime64[s]").astype(np.float64)
        time_edges = edges_from_centers(time_centers_epoch)

        lat_edges = edges_from_centers(lat_centers)
        lon_edges = edges_from_centers(lon_centers)

        # One count/data grid per variable
        var_names = list(variable_dict.keys())
        count_grids = {v: np.zeros((ntime, nlon, nlat), dtype=np.int32) for v in var_names}
        data_grids = {v: np.zeros((ntime, nlon, nlat), dtype=np.float64) for v in var_names}

        n_valid_total = 0
        for i, (granule_key, granule) in enumerate(granules.items()):
            geometry_timestamp = pd.to_datetime(granule_key, format="%Y%j%H%M").timestamp()

            lat = granule["lat"].values.flatten().astype(np.float64)
            lon = granule["lon"].values.flatten().astype(np.float64)

            # Mask fill values
            fill_mask = (lat < -900) | (lon < -900)

            # Shift lon from -180..180 to 0..360 if dataset grid is 0..360
            # (first edge may be slightly negative due to wrapping, so check max)
            if lon_edges[-1] > 180:
                lon = np.where(lon < 0, lon + 360.0, lon)

            n_points = len(lat)
            time_flat = np.full(n_points, geometry_timestamp, dtype=np.float64)

            for var_name in var_names:
                if var_name not in granule:
                    continue
                data_flat = granule[var_name].values.flatten().astype(np.float64)
                data_flat[fill_mask] = np.nan
                n_valid_total += int(np.isfinite(data_flat).sum())

                bin_swath_to_grid(
                    time_edges,
                    lon_edges,
                    lat_edges,
                    time_flat,
                    lon,
                    lat,
                    data_flat,
                    count_grids[var_name],
                    data_grids[var_name],
                )

            if progress_callback and (i + 1) % 50 == 0:
                progress_callback(
                    f"step: Binned {i + 1}/{len(granules)} granules, "
                    f"{n_valid_total:,} valid pixels"
                )

        if progress_callback:
            progress_callback(
                f"step: Binning complete: {n_valid_total:,} pixels "
                f"from {len(granules)} granules"
            )

        # Normalize each variable grid
        for var_name in var_names:
            normalize_grid(count_grids[var_name], data_grids[var_name])

        # Apply min_sample_count filter
        if min_sample_count > 1:
            for var_name in var_names:
                data_grids[var_name][count_grids[var_name] < min_sample_count] = np.nan

        # Build xr.Dataset
        time_coords = pd.to_datetime(time_centers_epoch, unit="s")

        data_vars: dict[str, tuple[list[str], np.ndarray]] = {}
        for var_name in var_names:
            data_vars[var_name] = (
                ["time", "lon", "lat"],
                data_grids[var_name].astype(np.float32),
            )

        # Use the maximum count across variables for sample_count
        # (they should be identical when there's one variable)
        max_count = np.zeros((ntime, nlon, nlat), dtype=np.int32)
        for var_name in var_names:
            np.maximum(max_count, count_grids[var_name], out=max_count)
        data_vars["sample_count"] = (["time", "lon", "lat"], max_count)

        ds = xr.Dataset(
            data_vars,
            coords={
                "time": time_coords,
                "lon": lon_centers,
                "lat": lat_centers,
            },
        )
        ds.attrs["geometry"] = "GRID"
        ds.attrs["source_geometry_type"] = "sat_swath_clm"
        ds.attrs["sat_type"] = "modis_l2"

        return ds
