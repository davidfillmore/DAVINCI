"""UFS (Unified Forecast System) model reader.

This module provides the UFSReader class for reading UFS-AQM (Air Quality Model)
output, including RRFS (Rapid Refresh Forecast System) data.
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


# Standard variable name mappings for UFS-AQM
UFS_VARIABLE_MAPPING: dict[str, str] = {
    "ozone": "o3",
    "pm25": "pm25",
    "no2": "no2",
    "no": "no",
    "co": "co",
    "so2": "so2",
    # Meteorology
    "temperature": "tmp2m",
    "temperature_2m": "tmp2m",
    "temperature_k": "tmp2m",
    "pressure": "pressfc",
    "pres_pa_mid": "pres",
    "relative_humidity": "rh2m",
    "specific_humidity": "spfh2m",
    "wind_speed_u": "ugrd10m",
    "wind_speed_v": "vgrd10m",
}

# Reverse mapping
UFS_STANDARD_NAMES: dict[str, str] = {v: k for k, v in UFS_VARIABLE_MAPPING.items()}


@model_registry.register("ufs")
class UFSReader:
    """Reader for UFS-AQM model output.

    Reads UFS Air Quality Model output, including RRFS-CMAQ data.
    Supports both grib2 and netCDF formats.

    Examples
    --------
    >>> reader = UFSReader()
    >>> ds = reader.open(["rrfs.t00z.natlevf024.tm00.grib2"])
    >>> print(ds.dims)
    Frozen({'time': 1, 'z': 65, 'lat': 1059, 'lon': 1799})
    """

    @property
    def name(self) -> str:
        """Return reader name."""
        return "ufs"

    def open(
        self,
        file_paths: Sequence[str | Path],
        variables: Sequence[str] | None = None,
        **kwargs: Any,
    ) -> xr.Dataset:
        """Open UFS-AQM output files.

        Parameters
        ----------
        file_paths
            Paths to UFS output files.
        variables
            Variables to load. If None, loads all variables.
        **kwargs
            Additional options:
            - fname_pm25: Path(s) to PM2.5 specific files
            - surf_only: bool, extract surface level only

        Returns
        -------
        xr.Dataset
            UFS data with standardized dimensions (time, z, lat, lon).
        """
        file_list = [Path(f) for f in file_paths]

        if not file_list:
            raise DataNotFoundError("No UFS files provided")

        # Check files exist
        missing = [f for f in file_list if not f.exists()]
        if missing:
            raise DataNotFoundError(f"UFS files not found: {missing}")

        # Try monetio first
        try:
            ds = self._open_with_monetio(file_list, variables, **kwargs)
        except ImportError:
            warnings.warn(
                "monetio not available, using basic xarray reader. "
                "Some UFS-specific features may not work.",
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
        """Open UFS files using monetio.

        Parameters
        ----------
        file_paths
            UFS file paths.
        variables
            Variables to load.
        **kwargs
            Additional monetio options.

        Returns
        -------
        xr.Dataset
            Raw UFS dataset.
        """
        import monetio as mio

        mio_kwargs: dict[str, Any] = {}

        if variables is not None:
            mio_kwargs["var_list"] = list(variables)

        if "fname_pm25" in kwargs:
            mio_kwargs["fname_pm25"] = kwargs["fname_pm25"]

        # Remove our custom kwargs before passing to monetio
        for key in ("fname_pm25", "surf_only"):
            kwargs.pop(key, None)

        mio_kwargs.update(kwargs)

        files = [str(f) for f in file_paths]

        # Try newer ufs module first, fall back to deprecated _rrfs_cmaq_mm
        ds: xr.Dataset
        if hasattr(mio.models, "ufs"):
            ds = mio.models.ufs.open_mfdataset(files, **mio_kwargs)
        else:
            warnings.warn(
                "Using deprecated _rrfs_cmaq_mm reader. "
                "Update monetio for better UFS support.",
                DeprecationWarning,
            )
            ds = mio.models._rrfs_cmaq_mm.open_mfdataset(files, **mio_kwargs)

        return ds

    def _open_with_xarray(
        self,
        file_paths: list[Path],
        variables: Sequence[str] | None,
        **kwargs: Any,
    ) -> xr.Dataset:
        """Open UFS files using xarray.

        Parameters
        ----------
        file_paths
            UFS file paths.
        variables
            Variables to load.
        **kwargs
            Additional xarray options.

        Returns
        -------
        xr.Dataset
            Raw UFS dataset.
        """
        # Filter out custom kwargs
        xr_kwargs = {
            k: v for k, v in kwargs.items()
            if k not in ("fname_pm25", "surf_only")
        }

        # Check if files are grib2
        is_grib = any(str(f).endswith((".grib2", ".grb2", ".grib")) for f in file_paths)

        if is_grib:
            xr_kwargs.setdefault("engine", "cfgrib")

        max_retries = 3
        last_error: Exception | None = None

        for attempt in range(max_retries):
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
                error_file = write_error_log(e, "Opening UFS files")
                msg = f"Failed to open UFS files: {e}"
                if error_file:
                    msg += f" (details: {error_file})"
                raise DataFormatError(msg) from e

        raise DataFormatError(f"Failed to open UFS files after {max_retries} attempts") from last_error

    def _standardize_dataset(self, ds: xr.Dataset) -> xr.Dataset:
        """Standardize UFS dataset dimensions and coordinates.

        Parameters
        ----------
        ds
            Raw UFS dataset.

        Returns
        -------
        xr.Dataset
            Standardized dataset.
        """
        # UFS dimension renames (depends on file type)
        dim_renames: dict[str, str] = {}

        # Common grib2 dimensions
        if "step" in ds.dims:
            dim_renames["step"] = "time"
        if "valid_time" in ds.dims:
            dim_renames["valid_time"] = "time"
        if "level" in ds.dims:
            dim_renames["level"] = "z"
        if "heightAboveGround" in ds.dims:
            dim_renames["heightAboveGround"] = "z"
        if "isobaricInhPa" in ds.dims:
            dim_renames["isobaricInhPa"] = "z"

        # NetCDF dimensions
        if "pfull" in ds.dims:
            dim_renames["pfull"] = "z"
        if "grid_yt" in ds.dims:
            dim_renames["grid_yt"] = "y"
        if "grid_xt" in ds.dims:
            dim_renames["grid_xt"] = "x"

        if dim_renames:
            ds = ds.rename(dim_renames)

        # Handle lat/lon coordinates
        if "latitude" in ds.coords and "lat" not in ds.coords:
            ds = ds.assign_coords(lat=ds["latitude"])
        if "longitude" in ds.coords and "lon" not in ds.coords:
            ds = ds.assign_coords(lon=ds["longitude"])

        return ds

    def get_variable_mapping(self) -> Mapping[str, str]:
        """Return UFS variable name mapping.

        Returns
        -------
        Mapping[str, str]
            Standard name to UFS name mapping.
        """
        return UFS_VARIABLE_MAPPING


# Alias for backward compatibility
@model_registry.register("rrfs")
class RRFSReader(UFSReader):
    """Alias for UFSReader for backward compatibility with RRFS."""

    @property
    def name(self) -> str:
        """Return reader name."""
        return "rrfs"


def open_ufs(
    files: str | Path | Sequence[str | Path],
    variables: Sequence[str] | None = None,
    label: str = "ufs",
    **kwargs: Any,
) -> ModelData:
    """Convenience function to open UFS model data.

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
        UFS model data container.
    """
    reader = UFSReader()

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
        mod_type="ufs",
        data=ds,
        files=file_paths,
    )
