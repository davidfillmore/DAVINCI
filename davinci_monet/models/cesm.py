"""CESM (Community Earth System Model) readers.

This module provides readers for CESM output, including:
- CESM-FV: Finite volume dynamical core (regular lat-lon grid)
- CESM-SE: Spectral element dynamical core (unstructured grid)
- CAM-chem: Chemistry component
- MUSICA: Multi-Scale Infrastructure for Chemistry and Aerosols
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import xarray as xr

from davinci_monet.core.exceptions import (
    DataFormatError,
    DataNotFoundError,
    cleanup_netcdf_state,
    is_transient_error,
    write_error_log,
)
from davinci_monet.core.protocols import DataGeometry
from davinci_monet.core.registry import source_registry

# Physical constants for column integration
_P0_DEFAULT = 100000.0  # Pa, reference pressure for CESM hybrid coords
_G = 9.80665  # m/s^2, gravitational acceleration
_M_AIR = 0.0289644  # kg/mol, molar mass of dry air


def compute_tropospheric_column(
    ds: xr.Dataset,
    var_name: str,
    ps_name: str = "PS",
    hyai_name: str = "hyai",
    hybi_name: str = "hybi",
    p0: float | None = None,
    z_dim: str = "z",
) -> xr.DataArray:
    """Compute tropospheric column from mixing ratio on hybrid coordinates.

    Integrates a tracer mixing ratio vertically using CESM hybrid sigma-pressure
    coordinates to produce a total column in mol/m².

    Parameters
    ----------
    ds
        Dataset containing the tracer, surface pressure, and hybrid coefficients.
    var_name
        Name of the tracer variable (mixing ratio in mol/mol).
    ps_name
        Name of surface pressure variable (Pa).
    hyai_name
        Name of hybrid A coefficient at interfaces.
    hybi_name
        Name of hybrid B coefficient at interfaces.
    p0
        Reference pressure (Pa). If None, uses 100000 Pa or P0 from dataset.
    z_dim
        Name of the vertical dimension.

    Returns
    -------
    xr.DataArray
        Tropospheric column with dimensions (time, lat, lon) in mol/m².

    Notes
    -----
    The column is computed as:
        Column = Σ (χ * ΔP / g / M_air)
    where χ is mixing ratio (mol/mol), ΔP is layer pressure thickness (Pa),
    g is gravitational acceleration, and M_air is molar mass of air.
    """
    # Get reference pressure
    if p0 is None:
        if "P0" in ds:
            p0 = float(ds["P0"].values)
        else:
            p0 = _P0_DEFAULT

    # Get required variables
    tracer = ds[var_name]
    ps = ds[ps_name]
    hyai = ds[hyai_name]
    hybi = ds[hybi_name]

    # Determine dimension names
    # Handle both 'z' and 'lev' naming
    if z_dim not in tracer.dims:
        if "lev" in tracer.dims:
            z_dim = "lev"
        else:
            raise ValueError(f"Could not find vertical dimension in {tracer.dims}")

    z_int_dim = f"{z_dim}_interface" if f"{z_dim}_interface" in ds.dims else "ilev"

    # Compute pressure at interfaces: P = hyai * P0 + hybi * PS
    # hyai and hybi have shape (n_interfaces,), PS has shape (time, lat, lon)
    # Result should be (n_interfaces, time, lat, lon)
    p_int = hyai * p0 + hybi * ps

    # Compute layer pressure thickness: ΔP = P(k+1) - P(k)
    # diff along the interface dimension gives layer thicknesses
    dp = p_int.diff(dim=z_int_dim if z_int_dim in p_int.dims else hyai.dims[0])

    # Rename the resulting dimension to match tracer's z dimension
    if dp.dims[0] != z_dim:
        dp = dp.rename({dp.dims[0]: z_dim})

    # Compute air column per layer [mol/m²]: dn = ΔP / g / M_air
    dn_air = dp / _G / _M_AIR

    # Compute tracer column [mol/m²]: sum(tracer * dn_air) over vertical
    column = (tracer * dn_air).sum(dim=z_dim)

    # Add attributes
    column.attrs = {
        "long_name": f"{var_name} tropospheric column",
        "units": "mol/m2",
        "derived_from": var_name,
    }

    return column


# Standard variable name mappings for CESM/CAM-chem
CESM_VARIABLE_MAPPING: dict[str, str] = {
    "ozone": "O3",
    "pm25": "PM25",
    "no2": "NO2",
    "no": "NO",
    "co": "CO",
    "so2": "SO2",
    # Aerosols
    "bc": "bc_a4",
    "oc": "pom_a4",
    "so4": "so4_a1",
    "dust": "dst_a1",
    "sea_salt": "ncl_a1",
    # Meteorology
    "temperature": "T",
    "temperature_k": "T",
    "pressure": "PS",
    "pres_pa_mid": "P",
    "relative_humidity": "RELHUM",
    "specific_humidity": "Q",
    "wind_speed_u": "U",
    "wind_speed_v": "V",
    "geopotential_height": "Z3",
}

# Reverse mapping
CESM_STANDARD_NAMES: dict[str, str] = {v: k for k, v in CESM_VARIABLE_MAPPING.items()}


@source_registry.register("cesm_fv")
class CESMFVReader:
    """Reader for CESM Finite Volume (FV) output.

    Reads CESM/CAM-chem output on the regular latitude-longitude grid
    used by the finite volume dynamical core.

    Examples
    --------
    >>> reader = CESMFVReader()
    >>> ds = reader.open(["cam.h0.2024-01.nc"])
    >>> print(ds.dims)
    Frozen({'time': 31, 'lev': 32, 'lat': 192, 'lon': 288})
    """

    @property
    def name(self) -> str:
        """Return reader name."""
        return "cesm_fv"

    @property
    def geometry(self) -> DataGeometry:
        """Model output is gridded."""
        return DataGeometry.GRID

    def open(
        self,
        file_paths: Sequence[str | Path],
        variables: Sequence[str] | None = None,
        **kwargs: Any,
    ) -> xr.Dataset:
        """Open CESM-FV output files.

        Parameters
        ----------
        file_paths
            Paths to CESM output files.
        variables
            Variables to load. If None, loads all variables.
        **kwargs
            Additional options passed to monetio or xarray.

        Returns
        -------
        xr.Dataset
            CESM data with standardized dimensions (time, z, lat, lon).
        """
        file_list = [Path(f) for f in file_paths]

        if not file_list:
            raise DataNotFoundError("No CESM files provided")

        missing = [f for f in file_list if not f.exists()]
        if missing:
            raise DataNotFoundError(f"CESM files not found: {missing}")

        # Try monetio first
        try:
            ds = self._open_with_monetio(file_list, variables, **kwargs)
        except ImportError:
            warnings.warn(
                "monetio not available, using basic xarray reader.",
                UserWarning,
            )
            ds = self._open_with_xarray(file_list, variables, **kwargs)

        ds = self._standardize_dataset(ds)

        return ds

    def _open_with_monetio(
        self,
        file_paths: list[Path],
        variables: Sequence[str] | None,
        **kwargs: Any,
    ) -> xr.Dataset:
        """Open CESM-FV files using monetio."""
        import monetio as mio

        mio_kwargs: dict[str, Any] = {}

        if variables is not None:
            mio_kwargs["var_list"] = list(variables)

        mio_kwargs.update(kwargs)

        files = [str(f) for f in file_paths]
        ds: xr.Dataset = mio.models._cesm_fv_mm.open_mfdataset(files, **mio_kwargs)

        return ds

    def _open_with_xarray(
        self,
        file_paths: list[Path],
        variables: Sequence[str] | None,
        **kwargs: Any,
    ) -> xr.Dataset:
        """Open CESM-FV files using xarray."""
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
                    # Transient error - clean up and retry
                    warnings.warn(
                        f"Transient NetCDF error (attempt {attempt + 1}/{max_retries}), "
                        f"retrying: {e}",
                        UserWarning,
                    )
                    cleanup_netcdf_state()
                    continue
                # Non-transient error or max retries reached
                error_file = write_error_log(e, "Opening CESM-FV files")
                msg = f"Failed to open CESM-FV files: {e}"
                if error_file:
                    msg += f" (details: {error_file})"
                raise DataFormatError(msg) from e

        # Should not reach here, but just in case
        raise DataFormatError(
            f"Failed to open CESM-FV files after {max_retries} attempts"
        ) from last_error

    def _standardize_dataset(self, ds: xr.Dataset) -> xr.Dataset:
        """Standardize CESM-FV dataset dimensions."""
        dim_renames: dict[str, str] = {}

        # CESM standard dimension names
        if "lev" in ds.dims:
            dim_renames["lev"] = "z"
        if "ilev" in ds.dims:
            dim_renames["ilev"] = "z_interface"

        if dim_renames:
            ds = ds.rename(dim_renames)

        return ds

    def get_variable_mapping(self) -> Mapping[str, str]:
        """Return CESM variable name mapping."""
        return CESM_VARIABLE_MAPPING


@source_registry.register("cesm_se")
class CESMSEReader:
    """Reader for CESM Spectral Element (SE) output.

    Reads CESM/CAM-chem output on the unstructured grid used by the
    spectral element dynamical core. Requires a SCRIP file for
    coordinate mapping.

    Examples
    --------
    >>> reader = CESMSEReader()
    >>> ds = reader.open(
    ...     ["cam.h0.2024-01.nc"],
    ...     scrip_file="ne30pg2_scrip.nc"
    ... )
    """

    @property
    def name(self) -> str:
        """Return reader name."""
        return "cesm_se"

    @property
    def geometry(self) -> DataGeometry:
        """Model output is gridded."""
        return DataGeometry.GRID

    def open(
        self,
        file_paths: Sequence[str | Path],
        variables: Sequence[str] | None = None,
        **kwargs: Any,
    ) -> xr.Dataset:
        """Open CESM-SE output files.

        Parameters
        ----------
        file_paths
            Paths to CESM output files.
        variables
            Variables to load. If None, loads all variables.
        **kwargs
            Additional options:
            - scrip_file: Path to SCRIP grid file (required for coordinate mapping)

        Returns
        -------
        xr.Dataset
            CESM data with unstructured grid dimensions.
        """
        file_list = [Path(f) for f in file_paths]

        if not file_list:
            raise DataNotFoundError("No CESM files provided")

        missing = [f for f in file_list if not f.exists()]
        if missing:
            raise DataNotFoundError(f"CESM files not found: {missing}")

        # Try monetio first
        try:
            ds = self._open_with_monetio(file_list, variables, **kwargs)
        except ImportError:
            warnings.warn(
                "monetio not available, using basic xarray reader. "
                "SE grid coordinate mapping may not work correctly.",
                UserWarning,
            )
            ds = self._open_with_xarray(file_list, variables, **kwargs)

        ds = self._standardize_dataset(ds)

        return ds

    def _open_with_monetio(
        self,
        file_paths: list[Path],
        variables: Sequence[str] | None,
        **kwargs: Any,
    ) -> xr.Dataset:
        """Open CESM-SE files using monetio."""
        import monetio as mio

        mio_kwargs: dict[str, Any] = {}

        if variables is not None:
            mio_kwargs["var_list"] = list(variables)

        # SCRIP file is required for SE grid
        if "scrip_file" in kwargs:
            mio_kwargs["scrip_file"] = kwargs.pop("scrip_file")

        mio_kwargs.update(kwargs)

        files = [str(f) for f in file_paths]
        ds: xr.Dataset = mio.models._cesm_se_mm.open_mfdataset(files, **mio_kwargs)

        return ds

    def _open_with_xarray(
        self,
        file_paths: list[Path],
        variables: Sequence[str] | None,
        **kwargs: Any,
    ) -> xr.Dataset:
        """Open CESM-SE files using xarray."""
        # Remove our custom kwargs
        xr_kwargs = {k: v for k, v in kwargs.items() if k != "scrip_file"}

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

                # If SCRIP file provided, add coordinates
                if "scrip_file" in kwargs:
                    scrip_path = kwargs["scrip_file"]
                    if Path(scrip_path).exists():
                        try:
                            scrip = xr.open_dataset(scrip_path)
                        except Exception as e:
                            error_file = write_error_log(e, f"Opening SCRIP file '{scrip_path}'")
                            msg = f"Failed to open SCRIP file '{scrip_path}': {e}"
                            if error_file:
                                msg += f" (details: {error_file})"
                            raise DataFormatError(msg) from e
                        if "grid_center_lat" in scrip:
                            ds = ds.assign_coords(lat=("ncol", scrip["grid_center_lat"].values))
                        if "grid_center_lon" in scrip:
                            ds = ds.assign_coords(lon=("ncol", scrip["grid_center_lon"].values))

                if variables is not None:
                    available = [v for v in variables if v in ds.data_vars]
                    if available:
                        ds = ds[available]

                return ds

            except DataFormatError:
                # Re-raise DataFormatError (from SCRIP file) without retry
                raise
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
                error_file = write_error_log(e, "Opening CESM-SE files")
                msg = f"Failed to open CESM-SE files: {e}"
                if error_file:
                    msg += f" (details: {error_file})"
                raise DataFormatError(msg) from e

        raise DataFormatError(
            f"Failed to open CESM-SE files after {max_retries} attempts"
        ) from last_error

    def _standardize_dataset(self, ds: xr.Dataset) -> xr.Dataset:
        """Standardize CESM-SE dataset dimensions."""
        dim_renames: dict[str, str] = {}

        if "lev" in ds.dims:
            dim_renames["lev"] = "z"
        if "ilev" in ds.dims:
            dim_renames["ilev"] = "z_interface"

        if dim_renames:
            ds = ds.rename(dim_renames)

        return ds

    def get_variable_mapping(self) -> Mapping[str, str]:
        """Return CESM variable name mapping."""
        return CESM_VARIABLE_MAPPING
