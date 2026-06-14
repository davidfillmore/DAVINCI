"""Generic L3 satellite gridded dataset reader.

This module provides a generic reader for Level 3 satellite gridded products.
It handles standard NetCDF files with regular lat/lon grids.

Use this reader for any L3 satellite product that doesn't have a
dedicated reader. For satellite-specific features (projections, QA flags,
variable mappings), use the dedicated readers when available.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Mapping, Sequence

import xarray as xr

from davinci_monet.core.exceptions import DataNotFoundError
from davinci_monet.core.protocols import DataGeometry
from davinci_monet.core.registry import source_registry
from davinci_monet.io.reader_utils import (
    select_variables,
    set_geometry_attr,
    validate_file_list,
)


@source_registry.register("satellite_l3")
class GenericL3Reader:
    """Generic reader for Level 3 satellite gridded datasets.

    Reads L3 satellite data from NetCDF files. Data is returned as
    grid geometry with (time, lat, lon) dimensions.

    This reader provides basic functionality for any L3 satellite product.
    For satellite-specific features (QA filtering, variable mappings,
    projection handling), use dedicated readers like GOESL3AODReader.

    Parameters for dimension/coordinate mapping can be provided to adapt
    to different file formats.

    Examples
    --------
    >>> reader = GenericL3Reader()
    >>> ds = reader.open(["satellite_l3_*.nc"])

    >>> # With custom coordinate mapping
    >>> ds = reader.open(
    ...     ["data.nc"],
    ...     coord_mapping={"latitude": "lat", "longitude": "lon"}
    ... )
    """

    @property
    def name(self) -> str:
        """Return reader name."""
        return "satellite_l3"

    @property
    def geometry(self) -> DataGeometry:
        """Return produced geometry."""
        return DataGeometry.GRID

    def open(
        self,
        file_paths: Sequence[str | Path],
        variables: Sequence[str] | None = None,
        *,
        dim_mapping: Mapping[str, str] | None = None,
        coord_mapping: Mapping[str, str] | None = None,
        qa_variable: str | None = None,
        qa_values: Sequence[int] | None = None,
        **kwargs: Any,
    ) -> xr.Dataset:
        """Open generic L3 satellite dataset files.

        Parameters
        ----------
        file_paths
            Paths to L3 satellite files.
        variables
            Variables to load. If None, loads all variables.
        dim_mapping
            Mapping of file dimension names to standard names.
            Example: {"latitude": "lat", "longitude": "lon"}
        coord_mapping
            Mapping of file coordinate names to standard names.
            Example: {"latitude": "lat", "longitude": "lon"}
        qa_variable
            Name of quality/flag variable for filtering.
        qa_values
            Quality flag values to keep. If None, keeps all.
        **kwargs
            Additional xarray.open_dataset options.

        Returns
        -------
        xr.Dataset
            L3 satellite datasets with grid dimensions.
        """
        file_list = validate_file_list(file_paths, dataset_label="Satellite L3")

        ds = self._open_files(file_list, variables, **kwargs)

        # Apply dimension renaming
        if dim_mapping:
            ds = self._apply_dim_mapping(ds, dim_mapping)

        # Apply coordinate renaming
        if coord_mapping:
            ds = self._apply_coord_mapping(ds, coord_mapping)

        # Apply QA filtering
        if qa_variable and qa_values is not None:
            ds = self._apply_qa_filter(ds, qa_variable, qa_values)

        return self._standardize_dataset(ds)

    def _open_files(
        self,
        file_paths: list[Path],
        variables: Sequence[str] | None,
        **kwargs: Any,
    ) -> xr.Dataset:
        """Open satellite files using xarray."""
        ds_list = []
        for fpath in file_paths:
            try:
                ds = xr.open_dataset(str(fpath), **kwargs)
                ds_list.append(ds)
            except Exception as e:
                warnings.warn(f"Failed to open {fpath}: {e}", UserWarning)
                continue

        if not ds_list:
            raise DataNotFoundError("No valid satellite L3 data found")

        if len(ds_list) > 1:
            ds = xr.concat(ds_list, dim="time")
        else:
            ds = ds_list[0]

        return select_variables(ds, variables)

    def _apply_dim_mapping(self, ds: xr.Dataset, dim_mapping: Mapping[str, str]) -> xr.Dataset:
        """Apply dimension name mapping."""
        renames = {}
        for old_name, new_name in dim_mapping.items():
            if old_name in ds.dims and new_name not in ds.dims:
                renames[old_name] = new_name
        if renames:
            ds = ds.rename(renames)
        return ds

    def _apply_coord_mapping(self, ds: xr.Dataset, coord_mapping: Mapping[str, str]) -> xr.Dataset:
        """Apply coordinate name mapping."""
        renames = {}
        for old_name, new_name in coord_mapping.items():
            if old_name in ds.coords and new_name not in ds.coords:
                renames[old_name] = new_name
            elif old_name in ds.data_vars and new_name not in ds.data_vars:
                renames[old_name] = new_name
        if renames:
            ds = ds.rename(renames)
        return ds

    def _apply_qa_filter(
        self, ds: xr.Dataset, qa_variable: str, qa_values: Sequence[int]
    ) -> xr.Dataset:
        """Apply quality flag filtering."""
        if qa_variable not in ds.data_vars and qa_variable not in ds.coords:
            warnings.warn(
                f"QA variable '{qa_variable}' not found in dataset",
                UserWarning,
            )
            return ds

        qa_data = ds[qa_variable]
        mask = qa_data.isin(list(qa_values))

        for var in ds.data_vars:
            if var != qa_variable:
                ds[var] = ds[var].where(mask)

        return ds

    def _standardize_dataset(self, ds: xr.Dataset) -> xr.Dataset:
        """Standardize dataset for grid geometry."""
        coord_renames: dict[str, str] = {}

        # Standardize coordinate names
        if "latitude" in ds.coords and "lat" not in ds.coords:
            coord_renames["latitude"] = "lat"
        if "longitude" in ds.coords and "lon" not in ds.coords:
            coord_renames["longitude"] = "lon"

        if coord_renames:
            ds = ds.rename(coord_renames)

        return set_geometry_attr(ds, DataGeometry.GRID)
