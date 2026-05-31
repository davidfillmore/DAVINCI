"""Generic L2 satellite swath observation reader.

This module provides a generic reader for Level 2 satellite swath products.
It handles standard NetCDF/HDF5 satellite files with swath geometry
(scanline, pixel or similar dimensions).

Use this reader for any L2 satellite product that doesn't have a
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
from davinci_monet.observations.base import ObservationData, create_observation_data


@source_registry.register("satellite_l2")
class GenericL2Reader:
    """Generic reader for Level 2 satellite swath observations.

    Reads L2 satellite data from NetCDF/HDF5 files. Data is returned as
    swath geometry with appropriate dimensions.

    This reader provides basic functionality for any L2 satellite product.
    For satellite-specific features (QA filtering, variable mappings,
    projection handling), use dedicated readers like TROPOMIReader.

    Parameters for dimension/coordinate mapping can be provided to adapt
    to different file formats.

    Examples
    --------
    >>> reader = GenericL2Reader()
    >>> ds = reader.open(["satellite_l2_*.nc"])

    >>> # With custom dimension mapping
    >>> ds = reader.open(
    ...     ["data.nc"],
    ...     dim_mapping={"nscans": "scanline", "npixels": "pixel"}
    ... )
    """

    @property
    def name(self) -> str:
        """Return reader name."""
        return "satellite_l2"

    def open(
        self,
        file_paths: Sequence[str | Path],
        variables: Sequence[str] | None = None,
        *,
        group: str | None = None,
        dim_mapping: Mapping[str, str] | None = None,
        coord_mapping: Mapping[str, str] | None = None,
        qa_variable: str | None = None,
        qa_threshold: float | None = None,
        **kwargs: Any,
    ) -> xr.Dataset:
        """Open generic L2 satellite observation files.

        Parameters
        ----------
        file_paths
            Paths to L2 satellite files.
        variables
            Variables to load. If None, loads all variables.
        group
            HDF5 group to read from (e.g., "PRODUCT" for TROPOMI-like files).
        dim_mapping
            Mapping of file dimension names to standard names.
            Example: {"nscans": "scanline", "npixels": "pixel"}
        coord_mapping
            Mapping of file coordinate names to standard names.
            Example: {"latitude": "lat", "longitude": "lon"}
        qa_variable
            Name of quality assurance variable for filtering.
        qa_threshold
            QA threshold value. Pixels with QA below threshold are masked.
        **kwargs
            Additional xarray.open_dataset options.

        Returns
        -------
        xr.Dataset
            L2 satellite observations with swath dimensions.
        """
        file_list = [Path(f) for f in file_paths]

        if not file_list:
            raise DataNotFoundError("No satellite L2 files provided")

        missing = [f for f in file_list if not f.exists()]
        if missing:
            raise DataNotFoundError(f"Satellite L2 files not found: {missing}")

        ds = self._open_files(file_list, variables, group=group, **kwargs)

        # Apply dimension renaming
        if dim_mapping:
            ds = self._apply_dim_mapping(ds, dim_mapping)

        # Apply coordinate renaming
        if coord_mapping:
            ds = self._apply_coord_mapping(ds, coord_mapping)

        # Apply QA filtering
        if qa_variable and qa_threshold is not None:
            ds = self._apply_qa_filter(ds, qa_variable, qa_threshold)

        return self._standardize_dataset(ds)

    def _open_files(
        self,
        file_paths: list[Path],
        variables: Sequence[str] | None,
        group: str | None = None,
        **kwargs: Any,
    ) -> xr.Dataset:
        """Open satellite files using xarray."""
        ds_list = []
        for fpath in file_paths:
            try:
                open_kwargs = dict(kwargs)
                if group:
                    open_kwargs["group"] = group

                ds = xr.open_dataset(str(fpath), **open_kwargs)
                ds_list.append(ds)
            except Exception as e:
                warnings.warn(f"Failed to open {fpath}: {e}", UserWarning)
                continue

        if not ds_list:
            raise DataNotFoundError("No valid satellite L2 data found")

        if len(ds_list) > 1:
            # Try to concatenate along time or scanline dimension
            concat_dim = self._find_concat_dim(ds_list[0])
            ds = xr.concat(ds_list, dim=concat_dim)
        else:
            ds = ds_list[0]

        if variables is not None:
            available = [v for v in variables if v in ds.data_vars]
            if available:
                ds = ds[available]

        return ds

    def _find_concat_dim(self, ds: xr.Dataset) -> str:
        """Find appropriate dimension for concatenation."""
        for dim in ["time", "scanline", "nscans", "along_track"]:
            if dim in ds.dims:
                return dim
        # Default to first dimension
        return list(ds.dims.keys())[0] if ds.dims else "time"  # type: ignore[return-value]

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

    def _apply_qa_filter(self, ds: xr.Dataset, qa_variable: str, qa_threshold: float) -> xr.Dataset:
        """Apply quality assurance filtering."""
        if qa_variable not in ds.data_vars and qa_variable not in ds.coords:
            warnings.warn(
                f"QA variable '{qa_variable}' not found in dataset",
                UserWarning,
            )
            return ds

        qa_data = ds[qa_variable]
        mask = qa_data >= qa_threshold

        for var in ds.data_vars:
            if var != qa_variable:
                ds[var] = ds[var].where(mask)

        return ds

    def _standardize_dataset(self, ds: xr.Dataset) -> xr.Dataset:
        """Standardize dataset for swath geometry."""
        coord_renames: dict[str, str] = {}

        # Standardize coordinate names
        if "latitude" in ds.coords and "lat" not in ds.coords:
            coord_renames["latitude"] = "lat"
        if "longitude" in ds.coords and "lon" not in ds.coords:
            coord_renames["longitude"] = "lon"

        if coord_renames:
            ds = ds.rename(coord_renames)

        ds.attrs["geometry"] = DataGeometry.SWATH.value

        return ds

    def get_variable_mapping(self) -> Mapping[str, str]:
        """Return empty variable mapping (generic reader)."""
        return {}


def open_satellite_l2(
    files: str | Path | Sequence[str | Path],
    variables: Sequence[str] | None = None,
    label: str = "satellite_l2",
    group: str | None = None,
    dim_mapping: Mapping[str, str] | None = None,
    coord_mapping: Mapping[str, str] | None = None,
    qa_variable: str | None = None,
    qa_threshold: float | None = None,
    **kwargs: Any,
) -> ObservationData:
    """Open generic L2 satellite observation data.

    This is a generic reader for Level 2 satellite swath products.
    Use dedicated readers (e.g., open_tropomi) for satellite-specific
    features when available.

    Parameters
    ----------
    files
        File path(s) or glob pattern.
    variables
        Variables to load.
    label
        Observation label.
    group
        HDF5 group to read from.
    dim_mapping
        Mapping of file dimension names to standard names.
    coord_mapping
        Mapping of file coordinate names to standard names.
    qa_variable
        Name of quality assurance variable for filtering.
    qa_threshold
        QA threshold value.
    **kwargs
        Additional xarray options.

    Returns
    -------
    ObservationData
        Satellite observation data container with SWATH geometry.

    Examples
    --------
    >>> # Basic usage
    >>> obs = open_satellite_l2("satellite_*.nc")

    >>> # With custom mappings
    >>> obs = open_satellite_l2(
    ...     "data.nc",
    ...     dim_mapping={"nscans": "scanline"},
    ...     coord_mapping={"latitude": "lat", "longitude": "lon"},
    ...     qa_variable="quality_flag",
    ...     qa_threshold=0.5,
    ... )
    """
    from glob import glob

    reader = GenericL2Reader()

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

    ds = reader.open(
        file_paths,
        variables,
        group=group,
        dim_mapping=dim_mapping,
        coord_mapping=coord_mapping,
        qa_variable=qa_variable,
        qa_threshold=qa_threshold,
        **kwargs,
    )

    obs = create_observation_data(
        label=label,
        obs_type="satellite",
        data=ds,
        variables=dict.fromkeys(variables) if variables else {},
    )
    obs.geometry = DataGeometry.SWATH

    return obs
