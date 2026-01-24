"""WRF-Chem model reader.

This module provides the WRFChemReader class for reading Weather Research and
Forecasting model with Chemistry (WRF-Chem) output files.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Mapping, Sequence

import xarray as xr

from davinci_monet.core.exceptions import (
    DataFormatError,
    DataNotFoundError,
    cleanup_netcdf_state,
    is_transient_error,
    write_error_log,
)
from davinci_monet.core.registry import model_registry
from davinci_monet.models.base import ModelData, create_model_data


# Standard variable name mappings for WRF-Chem
# WRF-Chem variable names vary by chemical mechanism
WRFCHEM_VARIABLE_MAPPING: dict[str, str] = {
    # Common across mechanisms
    "ozone": "o3",
    "pm25": "PM2_5_DRY",
    "pm10": "PM10",
    "no2": "no2",
    "no": "no",
    "co": "co",
    "so2": "so2",
    "nox": "nox",
    # Meteorology
    "temperature": "T2",
    "temperature_2m": "T2",
    "temperature_k": "T2",
    "pressure": "PSFC",
    "pres_pa_mid": "P",
    "relative_humidity": "rh",
    "wind_speed_u": "U10",
    "wind_speed_v": "V10",
    "wind_speed": "WSPD10",
    "wind_direction": "WDIR10",
    # Additional species
    "hcho": "hcho",
    "isop": "isop",
    "nh3": "nh3",
}

# Reverse mapping
WRFCHEM_STANDARD_NAMES: dict[str, str] = {v: k for k, v in WRFCHEM_VARIABLE_MAPPING.items()}


@model_registry.register("wrfchem")
class WRFChemReader:
    """Reader for WRF-Chem model output.

    Reads WRF-Chem output files (wrfout_*), handling various chemical
    mechanisms and output configurations.

    Examples
    --------
    >>> reader = WRFChemReader()
    >>> ds = reader.open(["wrfout_d01_2024-01-01_00:00:00"])
    >>> print(ds.dims)
    Frozen({'time': 24, 'z': 35, 'lat': 100, 'lon': 120})
    """

    @property
    def name(self) -> str:
        """Return reader name."""
        return "wrfchem"

    def open(
        self,
        file_paths: Sequence[str | Path],
        variables: Sequence[str] | None = None,
        **kwargs: Any,
    ) -> xr.Dataset:
        """Open WRF-Chem output files.

        Parameters
        ----------
        file_paths
            Paths to WRF-Chem output files (wrfout_*).
        variables
            Variables to load. If None, loads all variables.
        **kwargs
            Additional options passed to monetio or xarray.

        Returns
        -------
        xr.Dataset
            WRF-Chem data with standardized dimensions (time, z, lat, lon).
        """
        file_list = [Path(f) for f in file_paths]

        if not file_list:
            raise DataNotFoundError("No WRF-Chem files provided")

        # Check files exist
        missing = [f for f in file_list if not f.exists()]
        if missing:
            raise DataNotFoundError(f"WRF-Chem files not found: {missing}")

        # Try monetio first
        try:
            ds = self._open_with_monetio(file_list, variables, **kwargs)
        except ImportError:
            warnings.warn(
                "monetio not available, using basic xarray reader. "
                "Some WRF-Chem specific features may not work.",
                UserWarning,
            )
            ds = self._open_with_xarray(file_list, variables, **kwargs)

        # Standardize dimensions
        ds = self._standardize_dataset(ds)

        return ds

    def _open_with_monetio(
        self,
        file_paths: list[Path],
        variables: Sequence[str] | None,
        **kwargs: Any,
    ) -> xr.Dataset:
        """Open WRF-Chem files using monetio.

        Parameters
        ----------
        file_paths
            WRF-Chem file paths.
        variables
            Variables to load.
        **kwargs
            Additional monetio options.

        Returns
        -------
        xr.Dataset
            Raw WRF-Chem dataset.
        """
        import monetio as mio

        mio_kwargs: dict[str, Any] = {}

        if variables is not None:
            mio_kwargs["var_list"] = list(variables)

        mio_kwargs.update(kwargs)

        files = [str(f) for f in file_paths]
        ds: xr.Dataset = mio.models._wrfchem_mm.open_mfdataset(files, **mio_kwargs)

        return ds

    def _open_with_xarray(
        self,
        file_paths: list[Path],
        variables: Sequence[str] | None,
        **kwargs: Any,
    ) -> xr.Dataset:
        """Open WRF-Chem files using xarray.

        Parameters
        ----------
        file_paths
            WRF-Chem file paths.
        variables
            Variables to load.
        **kwargs
            Additional xarray options.

        Returns
        -------
        xr.Dataset
            Raw WRF-Chem dataset.
        """
        max_retries = 3
        last_error: Exception | None = None

        for attempt in range(max_retries):
            try:
                if len(file_paths) > 1:
                    ds = xr.open_mfdataset(
                        [str(f) for f in file_paths],
                        combine="by_coords",
                        parallel=True,
                        **kwargs,
                    )
                else:
                    ds = xr.open_dataset(str(file_paths[0]), **kwargs)

                if variables is not None:
                    available = [v for v in variables if v in ds.data_vars]
                    if available:
                        ds = ds[available]

                return ds

            except Exception as e:
                last_error = e
                if attempt < max_retries - 1 and is_transient_error(e):
                    warnings.warn(
                        f"Transient NetCDF error (attempt {attempt + 1}/{max_retries}), "
                        f"retrying: {e}",
                        UserWarning,
                    )
                    cleanup_netcdf_state()
                    continue
                error_file = write_error_log(e, "Opening WRF-Chem files")
                msg = f"Failed to open WRF-Chem files: {e}"
                if error_file:
                    msg += f" (details: {error_file})"
                raise DataFormatError(msg) from e

        raise DataFormatError(f"Failed to open WRF-Chem files after {max_retries} attempts") from last_error

    def _standardize_dataset(self, ds: xr.Dataset) -> xr.Dataset:
        """Standardize WRF-Chem dataset dimensions and coordinates.

        Parameters
        ----------
        ds
            Raw WRF-Chem dataset.

        Returns
        -------
        xr.Dataset
            Standardized dataset.
        """
        # WRF dimension renames
        dim_renames: dict[str, str] = {}

        if "Time" in ds.dims:
            dim_renames["Time"] = "time"
        if "bottom_top" in ds.dims:
            dim_renames["bottom_top"] = "z"
        if "south_north" in ds.dims:
            dim_renames["south_north"] = "y"
        if "west_east" in ds.dims:
            dim_renames["west_east"] = "x"
        if "bottom_top_stag" in ds.dims:
            dim_renames["bottom_top_stag"] = "z_stag"
        if "south_north_stag" in ds.dims:
            dim_renames["south_north_stag"] = "y_stag"
        if "west_east_stag" in ds.dims:
            dim_renames["west_east_stag"] = "x_stag"

        if dim_renames:
            ds = ds.rename(dim_renames)

        # Handle WRF lat/lon (XLAT, XLONG)
        if "XLAT" in ds.data_vars or "XLAT" in ds.coords:
            if "XLAT" in ds.data_vars:
                ds = ds.set_coords("XLAT")
            if "lat" not in ds.coords:
                ds = ds.assign_coords(lat=ds["XLAT"])

        if "XLONG" in ds.data_vars or "XLONG" in ds.coords:
            if "XLONG" in ds.data_vars:
                ds = ds.set_coords("XLONG")
            if "lon" not in ds.coords:
                ds = ds.assign_coords(lon=ds["XLONG"])

        # Handle latitude/longitude if present
        if "latitude" in ds.data_vars and "latitude" not in ds.coords:
            ds = ds.set_coords("latitude")
        if "longitude" in ds.data_vars and "longitude" not in ds.coords:
            ds = ds.set_coords("longitude")

        return ds

    def get_variable_mapping(self) -> Mapping[str, str]:
        """Return WRF-Chem variable name mapping.

        Returns
        -------
        Mapping[str, str]
            Standard name to WRF-Chem name mapping.
        """
        return WRFCHEM_VARIABLE_MAPPING


def open_wrfchem(
    files: str | Path | Sequence[str | Path],
    variables: Sequence[str] | None = None,
    label: str = "wrfchem",
    **kwargs: Any,
) -> ModelData:
    """Convenience function to open WRF-Chem model data.

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
        WRF-Chem model data container.
    """
    reader = WRFChemReader()

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
        mod_type="wrfchem",
        data=ds,
        files=file_paths,
    )
