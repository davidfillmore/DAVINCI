"""CMAQ model reader.

This module provides the CMAQReader class for reading Community Multiscale
Air Quality (CMAQ) model output files.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Mapping, Sequence

import xarray as xr

from davinci_monet.core.exceptions import DataFormatError, DataNotFoundError, write_error_log
from davinci_monet.core.registry import model_registry
from davinci_monet.models.base import ModelData, create_model_data


# Standard variable name mappings for CMAQ
CMAQ_VARIABLE_MAPPING: dict[str, str] = {
    "ozone": "O3",
    "pm25": "PM25_TOT",
    "no2": "NO2",
    "no": "NO",
    "co": "CO",
    "so2": "SO2",
    "nox": "NOX",
    "pm10": "PM10",
    "temperature": "TEMP2",
    "temperature_k": "TEMP2",
    "pressure": "PRSFC",
    "pres_pa_mid": "PRES",
    "relative_humidity": "RH",
    "wind_speed": "WSPD10",
    "wind_direction": "WDIR10",
}

# Reverse mapping for lookups
CMAQ_STANDARD_NAMES: dict[str, str] = {v: k for k, v in CMAQ_VARIABLE_MAPPING.items()}


@model_registry.register("cmaq")
class CMAQReader:
    """Reader for CMAQ model output.

    Reads CMAQ CONC (concentration), MET (meteorology), and other output files,
    standardizing them into a consistent xarray Dataset format.

    Examples
    --------
    >>> reader = CMAQReader()
    >>> ds = reader.open(["CCTM_CONC_20240101.nc", "CCTM_CONC_20240102.nc"])
    >>> print(ds.dims)
    Frozen({'time': 48, 'z': 35, 'lat': 100, 'lon': 120})
    """

    @property
    def name(self) -> str:
        """Return reader name."""
        return "cmaq"

    def open(
        self,
        file_paths: Sequence[str | Path],
        variables: Sequence[str] | None = None,
        **kwargs: Any,
    ) -> xr.Dataset:
        """Open CMAQ output files.

        Parameters
        ----------
        file_paths
            Paths to CMAQ output files.
        variables
            Variables to load. If None, loads all variables.
        **kwargs
            Additional options:
            - fname_vert: Path(s) to vertical files (METCRO3D)
            - fname_surf: Path(s) to surface files (METCRO2D)
            - concatenate_forecasts: bool, concatenate forecast cycles
            - surf_only: bool, extract surface level only

        Returns
        -------
        xr.Dataset
            CMAQ data with standardized dimensions (time, z, lat, lon).
        """
        file_list = [Path(f) for f in file_paths]

        if not file_list:
            raise DataNotFoundError("No CMAQ files provided")

        # Check files exist
        missing = [f for f in file_list if not f.exists()]
        if missing:
            raise DataNotFoundError(f"CMAQ files not found: {missing}")

        # Try to use monetio if available
        try:
            ds = self._open_with_monetio(file_list, variables, **kwargs)
        except ImportError:
            warnings.warn(
                "monetio not available, using basic xarray reader. "
                "Some CMAQ-specific features may not work.",
                UserWarning,
            )
            ds = self._open_with_xarray(file_list, variables, **kwargs)

        # Standardize dimensions and coordinates
        ds = self._standardize_dataset(ds)

        return ds

    def _open_with_monetio(
        self,
        file_paths: list[Path],
        variables: Sequence[str] | None,
        **kwargs: Any,
    ) -> xr.Dataset:
        """Open CMAQ files using monetio.

        Parameters
        ----------
        file_paths
            CMAQ file paths.
        variables
            Variables to load.
        **kwargs
            Additional monetio options.

        Returns
        -------
        xr.Dataset
            Raw CMAQ dataset.
        """
        import monetio as mio

        # Prepare kwargs for monetio
        mio_kwargs: dict[str, Any] = {}

        if variables is not None:
            mio_kwargs["var_list"] = list(variables)

        if "fname_vert" in kwargs:
            mio_kwargs["fname_vert"] = kwargs["fname_vert"]

        if "fname_surf" in kwargs:
            mio_kwargs["fname_surf"] = kwargs["fname_surf"]

        if kwargs.get("concatenate_forecasts", False):
            mio_kwargs["concatenate_forecasts"] = True

        # Use monetio CMAQ reader
        files = [str(f) for f in file_paths]
        ds: xr.Dataset = mio.models._cmaq_mm.open_mfdataset(files, **mio_kwargs)

        return ds

    def _open_with_xarray(
        self,
        file_paths: list[Path],
        variables: Sequence[str] | None,
        **kwargs: Any,
    ) -> xr.Dataset:
        """Open CMAQ files using basic xarray.

        Parameters
        ----------
        file_paths
            CMAQ file paths.
        variables
            Variables to load.
        **kwargs
            Additional xarray options.

        Returns
        -------
        xr.Dataset
            Raw CMAQ dataset.
        """
        # Filter out our custom kwargs
        xr_kwargs = {
            k: v for k, v in kwargs.items()
            if k not in ("fname_vert", "fname_surf", "concatenate_forecasts", "surf_only")
        }

        try:
            if len(file_paths) > 1:
                ds = xr.open_mfdataset(
                    [str(f) for f in file_paths],
                    combine="by_coords",
                    parallel=True,
                    **xr_kwargs,
                )
            else:
                ds = xr.open_dataset(str(file_paths[0]), **xr_kwargs)
        except Exception as e:
            error_file = write_error_log(e, "Opening CMAQ files")
            msg = f"Failed to open CMAQ files: {e}"
            if error_file:
                msg += f" (details: {error_file})"
            raise DataFormatError(msg) from e

        # Select variables if specified
        if variables is not None:
            available = [v for v in variables if v in ds.data_vars]
            if available:
                ds = ds[available]

        return ds

    def _standardize_dataset(self, ds: xr.Dataset) -> xr.Dataset:
        """Standardize CMAQ dataset dimensions and coordinates.

        Parameters
        ----------
        ds
            Raw CMAQ dataset.

        Returns
        -------
        xr.Dataset
            Standardized dataset with dims (time, z, lat, lon).
        """
        # Common CMAQ dimension renames
        dim_renames: dict[str, str] = {}

        if "TSTEP" in ds.dims:
            dim_renames["TSTEP"] = "time"
        if "LAY" in ds.dims:
            dim_renames["LAY"] = "z"
        if "ROW" in ds.dims:
            dim_renames["ROW"] = "y"
        if "COL" in ds.dims:
            dim_renames["COL"] = "x"

        if dim_renames:
            ds = ds.rename(dim_renames)

        # Ensure latitude and longitude are proper coordinates
        # CMAQ often has these as data variables
        if "latitude" in ds.data_vars and "latitude" not in ds.coords:
            ds = ds.set_coords("latitude")
        if "longitude" in ds.data_vars and "longitude" not in ds.coords:
            ds = ds.set_coords("longitude")

        # Create lat/lon aliases if needed
        if "latitude" in ds.coords and "lat" not in ds.coords:
            ds = ds.assign_coords(lat=ds["latitude"])
        if "longitude" in ds.coords and "lon" not in ds.coords:
            ds = ds.assign_coords(lon=ds["longitude"])

        return ds

    def get_variable_mapping(self) -> Mapping[str, str]:
        """Return CMAQ variable name mapping.

        Returns
        -------
        Mapping[str, str]
            Standard name to CMAQ name mapping.
        """
        return CMAQ_VARIABLE_MAPPING


def open_cmaq(
    files: str | Path | Sequence[str | Path],
    variables: Sequence[str] | None = None,
    label: str = "cmaq",
    **kwargs: Any,
) -> ModelData:
    """Convenience function to open CMAQ model data.

    Parameters
    ----------
    files
        File path(s) or glob pattern.
    variables
        Variables to load.
    label
        Model label.
    **kwargs
        Additional reader options.

    Returns
    -------
    ModelData
        CMAQ model data container.
    """
    reader = CMAQReader()

    # Handle glob pattern
    if isinstance(files, (str, Path)):
        file_str = str(files)
        if "*" in file_str or "?" in file_str:
            from glob import glob
            file_list = sorted(glob(file_str))
            if not file_list:
                raise DataNotFoundError(f"No files match pattern: {files}")
            file_paths: Sequence[str | Path] = file_list
        else:
            file_paths = [files]
    else:
        file_paths = list(files)

    ds = reader.open(file_paths, variables, **kwargs)

    return create_model_data(
        label=label,
        mod_type="cmaq",
        data=ds,
        files=file_paths,
    )
