"""OMPS L3 Total Ozone observation reader.

This module provides the OMPSL3O3Reader class for reading OMPS
(Ozone Mapping and Profiler Suite) L3 total ozone gridded products.

OMPS is on the Suomi NPP and NOAA-20 satellites, providing global
total column ozone measurements.

Note
----
This reader requires monetio.sat._omps_l3_mm for full functionality.
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

# Standard variable name mappings for OMPS O3
OMPS_O3_VARIABLE_MAPPING: dict[str, str] = {
    "o3": "ColumnAmountO3",
    "total_ozone": "ColumnAmountO3",
    "reflectivity": "Reflectivity",
    "quality_flag": "QualityFlags",
}


@source_registry.register("omps_l3_o3")
class OMPSL3O3Reader:
    """Reader for OMPS L3 total ozone satellite observations.

    Reads OMPS L3 gridded total ozone data from NetCDF/HDF files.
    OMPS provides daily global total column ozone measurements.

    Data is returned as grid geometry with (time, lat, lon) dimensions.

    Note
    ----
    Full functionality requires monetio. Without monetio, the reader falls
    back to basic xarray which may not handle OMPS-specific features
    correctly.

    Examples
    --------
    >>> reader = OMPSL3O3Reader()
    >>> ds = reader.open(["OMPS-NPP_NMTO3-L3-DAILY_*.nc"])
    """

    @property
    def name(self) -> str:
        """Return reader name."""
        return "omps_l3_o3"

    def open(
        self,
        file_paths: Sequence[str | Path],
        variables: Sequence[str] | None = None,
        **kwargs: Any,
    ) -> xr.Dataset:
        """Open OMPS L3 ozone observation files.

        Parameters
        ----------
        file_paths
            Paths to OMPS L3 files.
        variables
            Variables to load. If None, loads total ozone column.
        **kwargs
            Additional options.

        Returns
        -------
        xr.Dataset
            OMPS observations with grid dimensions.
        """
        file_list = validate_file_list(file_paths, source_label="OMPS")

        # Try monetio first
        try:
            ds = self._open_with_monetio(file_list, variables, **kwargs)
        except ImportError:
            warnings.warn(
                "monetio not available, using basic xarray reader. "
                "OMPS-specific handling may be incomplete.",
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
        """Open OMPS files using monetio."""
        import monetio.sat._omps_l3_mm as omps_mod

        files = [str(f) for f in file_paths]

        if len(files) == 1:
            ds: xr.Dataset = omps_mod.open_dataset(files[0], **kwargs)
        else:
            ds_list = []
            for f in files:
                try:
                    ds_i: xr.Dataset = omps_mod.open_dataset(f, **kwargs)
                    ds_list.append(ds_i)
                except Exception as e:
                    warnings.warn(f"Failed to open {f}: {e}", UserWarning)
                    continue

            if not ds_list:
                raise DataNotFoundError("No valid OMPS data found")

            ds = xr.concat(ds_list, dim="time")

        return select_variables(ds, variables)

    def _open_with_xarray(
        self,
        file_paths: list[Path],
        variables: Sequence[str] | None,
        **kwargs: Any,
    ) -> xr.Dataset:
        """Open OMPS files using xarray."""
        ds_list = []
        for fpath in file_paths:
            try:
                ds = xr.open_dataset(str(fpath), **kwargs)
                ds_list.append(ds)
            except Exception as e:
                warnings.warn(f"Failed to open {fpath}: {e}", UserWarning)
                continue

        if not ds_list:
            raise DataNotFoundError("No valid OMPS data found")

        if len(ds_list) > 1:
            ds = xr.concat(ds_list, dim="time")
        else:
            ds = ds_list[0]

        return select_variables(ds, variables)

    def _standardize_dataset(self, ds: xr.Dataset) -> xr.Dataset:
        """Standardize OMPS dataset dimensions and coordinates."""
        coord_renames: dict[str, str] = {}

        if "latitude" in ds.coords and "lat" not in ds.coords:
            coord_renames["latitude"] = "lat"
        if "longitude" in ds.coords and "lon" not in ds.coords:
            coord_renames["longitude"] = "lon"

        if coord_renames:
            ds = ds.rename(coord_renames)

        return set_geometry_attr(ds, DataGeometry.GRID)
