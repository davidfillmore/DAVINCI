"""MOPITT L3 CO observation reader.

This module provides the MOPITTL3COReader class for reading MOPITT
(Measurements Of Pollution In The Troposphere) L3 CO gridded products.

MOPITT is on NASA's Terra satellite, providing global CO measurements
since 2000.

Note
----
This reader requires monetio.sat._mopitt_l3_mm for full functionality.
Without monetio, it falls back to basic xarray reading.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Sequence

import xarray as xr

from davinci_monet.core.exceptions import DataNotFoundError
from davinci_monet.core.protocols import DataGeometry
from davinci_monet.core.registry import source_registry
from davinci_monet.io.reader_utils import (
    select_variables,
    set_geometry_attr,
    validate_file_list,
)

# Standard variable name mappings for MOPITT CO
MOPITT_CO_VARIABLE_MAPPING: dict[str, str] = {
    "co": "RetrievedCOTotalColumnDay",
    "co_night": "RetrievedCOTotalColumnNight",
    "co_surface": "RetrievedCOSurfaceMixingRatioDay",
    "averaging_kernel": "TotalColumnAveragingKernelDay",
}


@source_registry.register("mopitt_l3_co")
class MOPITTL3COReader:
    """Reader for MOPITT L3 CO satellite observations.

    Reads MOPITT L3 CO gridded data from HDF/NetCDF files. MOPITT provides
    global CO column and profile measurements from Terra.

    Data is returned as grid geometry with (time, lat, lon) dimensions.

    Note
    ----
    Full functionality requires monetio. Without monetio, the reader falls
    back to basic xarray which may not handle MOPITT-specific features
    correctly.

    Examples
    --------
    >>> reader = MOPITTL3COReader()
    >>> ds = reader.open(["MOP03JM-*.he5"])
    """

    @property
    def name(self) -> str:
        """Return reader name."""
        return "mopitt_l3_co"

    def open(
        self,
        file_paths: Sequence[str | Path],
        variables: Sequence[str] | None = None,
        *,
        day_night: str = "day",
        **kwargs: Any,
    ) -> xr.Dataset:
        """Open MOPITT L3 CO observation files.

        Parameters
        ----------
        file_paths
            Paths to MOPITT L3 files.
        variables
            Variables to load. If None, loads CO total column.
        day_night
            'day', 'night', or 'both' for day/night retrievals.
        **kwargs
            Additional options.

        Returns
        -------
        xr.Dataset
            MOPITT observations with grid dimensions.
        """
        file_list = validate_file_list(file_paths, source_label="MOPITT")

        # Try monetio first
        try:
            ds = self._open_with_monetio(file_list, variables, **kwargs)
        except ImportError:
            warnings.warn(
                "monetio not available, using basic xarray reader. "
                "MOPITT-specific handling may be incomplete.",
                UserWarning,
            )
            ds = self._open_with_xarray(file_list, variables, **kwargs)

        return self._standardize_dataset(ds)

    def _open_with_monetio(
        self,
        file_paths: list[Path],
        variables: Sequence[str] | None,
        **kwargs: Any,
    ) -> xr.Dataset:
        """Open MOPITT files using monetio."""
        import monetio.sat._mopitt_l3_mm as mopitt_mod

        files = [str(f) for f in file_paths]

        if len(files) == 1:
            ds: xr.Dataset = mopitt_mod.open_dataset(files[0], **kwargs)
        else:
            ds_list = []
            for f in files:
                try:
                    ds_i: xr.Dataset = mopitt_mod.open_dataset(f, **kwargs)
                    ds_list.append(ds_i)
                except Exception as e:
                    warnings.warn(f"Failed to open {f}: {e}", UserWarning)
                    continue

            if not ds_list:
                raise DataNotFoundError("No valid MOPITT data found")

            ds = xr.concat(ds_list, dim="time")

        return select_variables(ds, variables)

    def _open_with_xarray(
        self,
        file_paths: list[Path],
        variables: Sequence[str] | None,
        **kwargs: Any,
    ) -> xr.Dataset:
        """Open MOPITT files using xarray."""
        ds_list = []
        for fpath in file_paths:
            try:
                # MOPITT HDF-EOS files have nested groups
                ds = xr.open_dataset(str(fpath), **kwargs)
                ds_list.append(ds)
            except Exception as e:
                warnings.warn(f"Failed to open {fpath}: {e}", UserWarning)
                continue

        if not ds_list:
            raise DataNotFoundError("No valid MOPITT data found")

        if len(ds_list) > 1:
            ds = xr.concat(ds_list, dim="time")
        else:
            ds = ds_list[0]

        return select_variables(ds, variables)

    def _standardize_dataset(self, ds: xr.Dataset) -> xr.Dataset:
        """Standardize MOPITT dataset dimensions and coordinates."""
        coord_renames: dict[str, str] = {}

        if "latitude" in ds.coords and "lat" not in ds.coords:
            coord_renames["latitude"] = "lat"
        if "longitude" in ds.coords and "lon" not in ds.coords:
            coord_renames["longitude"] = "lon"

        if coord_renames:
            ds = ds.rename(coord_renames)

        return set_geometry_attr(ds, DataGeometry.GRID)
