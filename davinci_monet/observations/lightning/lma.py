"""Lightning Mapping Array (LMA) observation reader.

This module provides the LMAReader class for reading Lightning Mapping Array
network data from NCAR EOL archives. LMA networks produce CF-compliant NetCDF
grids of flash and source density on regular lat/lon grids.

Supported networks:
- OKLMA (Oklahoma Lightning Mapping Array)
- COLMA (Colorado Lightning Mapping Array)
- NALMA (North Alabama Lightning Mapping Array)

Data source: NCAR EOL dataset 353.202
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import xarray as xr

from davinci_monet.core.exceptions import DataFormatError, DataNotFoundError
from davinci_monet.core.protocols import DataGeometry
from davinci_monet.core.registry import observation_registry
from davinci_monet.observations.base import ObservationData, create_observation_data

# Standard variable name mappings for LMA data
LMA_VARIABLE_MAPPING: dict[str, str] = {
    "flash_density": "flash_extent_density",
    "flash_extent": "flash_extent_density",
    "source_density": "source_density",
    "flash_init": "flash_init_density",
    "flash_initiation": "flash_init_density",
}

# Known LMA network locations
LMA_NETWORKS: dict[str, dict[str, Any]] = {
    "oklma": {
        "name": "Oklahoma Lightning Mapping Array",
        "center_lat": 35.2,
        "center_lon": -97.4,
    },
    "colma": {
        "name": "Colorado Lightning Mapping Array",
        "center_lat": 40.4,
        "center_lon": -104.6,
    },
    "nalma": {
        "name": "North Alabama Lightning Mapping Array",
        "center_lat": 34.7,
        "center_lon": -86.6,
    },
}


@observation_registry.register("lma")
class LMAReader:
    """Reader for Lightning Mapping Array (LMA) gridded observations.

    Reads LMA flash density data from CF-compliant NetCDF grid files
    produced by LMA network processing pipelines. Data is returned as
    grid geometry with (time, latitude, longitude) dimensions.

    Examples
    --------
    >>> reader = LMAReader()
    >>> ds = reader.open(["oklma_20120530_grid.nc"])
    >>> print(ds.dims)
    Frozen({'time': 24, 'latitude': 100, 'longitude': 100})
    """

    @property
    def name(self) -> str:
        """Return reader name."""
        return "lma"

    def open(
        self,
        file_paths: Sequence[str | Path],
        variables: Sequence[str] | None = None,
        *,
        network: str | None = None,
        **kwargs: Any,
    ) -> xr.Dataset:
        """Open LMA observation files.

        Parameters
        ----------
        file_paths
            Paths to LMA NetCDF grid files.
        variables
            Variables to load. If None, loads all available density fields.
        network
            LMA network identifier ('oklma', 'colma', 'nalma').
            Used to set metadata; auto-detected from filename if None.
        **kwargs
            Additional options passed to xr.open_dataset.

        Returns
        -------
        xr.Dataset
            LMA observations with dimensions (time, latitude, longitude)
            and density variables.
        """
        file_list = [Path(f) for f in file_paths]

        if not file_list:
            raise DataNotFoundError("No LMA files provided")

        missing = [f for f in file_list if not f.exists()]
        if missing:
            raise DataNotFoundError(f"LMA files not found: {missing}")

        ds = self._open_netcdf(file_list, variables, **kwargs)

        if network is None:
            network = self._detect_network(file_list[0])

        ds = self._standardize_dataset(ds, network=network)

        return ds

    def _open_netcdf(
        self,
        file_paths: list[Path],
        variables: Sequence[str] | None,
        **kwargs: Any,
    ) -> xr.Dataset:
        """Open LMA NetCDF files.

        OKLMA grids use non-standard dim names (ntimes, lon, lat) with
        time/longitude/latitude as plain data variables. Each file is
        normalized before concatenation.
        """
        ds_list = []
        for fpath in file_paths:
            try:
                ds = xr.open_dataset(str(fpath), **kwargs)
                ds = self._normalize_dims(ds)
                ds_list.append(ds)
            except Exception as e:
                warnings.warn(f"Failed to open {fpath}: {e}", UserWarning)
                continue

        if not ds_list:
            raise DataNotFoundError("No valid LMA data found")

        if len(ds_list) > 1:
            ds = xr.concat(ds_list, dim="time")
        else:
            ds = ds_list[0]

        if variables is not None:
            available = [v for v in variables if v in ds.data_vars]
            if available:
                # Keep coordinate variables too
                ds = ds[available]

        return ds

    def _normalize_dims(self, ds: xr.Dataset) -> xr.Dataset:
        """Normalize OKLMA dimension names and promote coords.

        Raw OKLMA grids have dims (ntimes, lon, lat) with time, longitude,
        latitude as plain data variables. This method renames dims and
        promotes the 1-D variables to proper coordinates.
        """
        renames: dict[str, str] = {}

        # Rename ntimes → time, lon → longitude, lat → latitude
        if "ntimes" in ds.dims:
            renames["ntimes"] = "time"
        if "lon" in ds.dims and "longitude" not in ds.dims:
            renames["lon"] = "longitude"
        if "lat" in ds.dims and "latitude" not in ds.dims:
            renames["lat"] = "latitude"

        if renames:
            ds = ds.rename(renames)

        # Promote 1-D data variables to coordinates
        for coord_name in ("time", "latitude", "longitude"):
            if coord_name in ds.data_vars and coord_name in ds.dims:
                ds = ds.set_coords(coord_name)

        # Drop CRS variable if present (not needed for analysis)
        if "crs" in ds.data_vars:
            ds = ds.drop_vars("crs")

        return ds

    def _detect_network(self, file_path: Path) -> str | None:
        """Detect LMA network from filename."""
        name_lower = file_path.name.lower()
        for network_id in LMA_NETWORKS:
            if network_id in name_lower:
                return network_id
        return None

    def _standardize_dataset(
        self,
        ds: xr.Dataset,
        network: str | None = None,
    ) -> xr.Dataset:
        """Standardize LMA dataset and add metadata.

        Dimension normalization is handled in _normalize_dims() before
        concatenation. This method handles any remaining cleanup and
        adds network metadata.
        """
        # Ensure time dimension exists (safety net for pre-normalized data)
        if "time" not in ds.dims:
            ds = ds.expand_dims("time")

        # Set geometry attribute
        ds.attrs["geometry"] = DataGeometry.GRID.value

        # Add network metadata
        if network is not None and network in LMA_NETWORKS:
            info = LMA_NETWORKS[network]
            ds.attrs["lma_network"] = info["name"]
            ds.attrs["lma_network_id"] = network

        return ds

    def get_variable_mapping(self) -> Mapping[str, str]:
        """Return LMA variable name mapping."""
        return LMA_VARIABLE_MAPPING


def open_lma(
    files: str | Path | Sequence[str | Path],
    variables: Sequence[str] | None = None,
    label: str = "lma",
    network: str | None = None,
    **kwargs: Any,
) -> ObservationData:
    """Convenience function to open LMA observation data.

    Parameters
    ----------
    files
        File path(s) or glob pattern.
    variables
        Variables to load.
    label
        Observation label.
    network
        LMA network identifier ('oklma', 'colma', 'nalma').
    **kwargs
        Additional reader options.

    Returns
    -------
    ObservationData
        LMA observation data container with GRID geometry.
    """
    from glob import glob

    reader = LMAReader()

    if isinstance(files, (str, Path)):
        file_str = str(files)
        if "*" in file_str or "?" in file_str:
            file_list = sorted(glob(file_str))
            if not file_list:
                raise DataNotFoundError(f"No files match pattern: {files}")
            file_paths: Sequence[str | Path] = file_list
        else:
            file_paths = [files]
    else:
        file_paths = list(files)

    ds = reader.open(file_paths, variables, network=network, **kwargs)

    obs = create_observation_data(
        label=label,
        obs_type="lma",
        data=ds,
        variables=dict.fromkeys(variables) if variables else {},
    )
    obs.geometry = DataGeometry.GRID

    return obs
