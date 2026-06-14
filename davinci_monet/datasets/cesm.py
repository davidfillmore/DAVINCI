"""CESM (Community Earth System Dataset) readers.

This module provides readers for CESM output, including:
- CESM-FV: Finite volume dynamical core (regular lat-lon grid)
- CESM-SE: Spectral element dynamical core (unstructured grid)
- CAM-chem: Chemistry component
- MUSICA: Multi-Scale Infrastructure for Chemistry and Aerosols
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import xarray as xr

from davinci_monet.core.exceptions import DataFormatError, write_error_log
from davinci_monet.core.protocols import DataGeometry
from davinci_monet.core.registry import source_registry
from davinci_monet.io.reader_utils import (
    retry_transient_open,
    select_variables,
    standardize_dims,
    validate_file_list,
)


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
        """Dataset output is gridded."""
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
        file_list = validate_file_list(file_paths, dataset_label="CESM")

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

        def _open() -> xr.Dataset:
            if len(file_paths) > 1:
                ds = xr.open_mfdataset(
                    [str(f) for f in file_paths],
                    combine="by_coords",
                    parallel=True,
                    **kwargs,
                )
            else:
                ds = xr.open_dataset(str(file_paths[0]), **kwargs)

            return select_variables(ds, variables)

        return retry_transient_open(_open, context="Opening CESM-FV files")

    def _standardize_dataset(self, ds: xr.Dataset) -> xr.Dataset:
        """Standardize CESM-FV dataset dimensions."""
        # CESM standard dimension names
        return standardize_dims(ds, {"lev": "z", "ilev": "z_interface"})


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
        """Dataset output is gridded."""
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
        file_list = validate_file_list(file_paths, dataset_label="CESM")

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

        def _open() -> xr.Dataset:
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

            return select_variables(ds, variables)

        # A DataFormatError from a bad SCRIP file must propagate without retry.
        return retry_transient_open(
            _open, context="Opening CESM-SE files", reraise=(DataFormatError,)
        )

    def _standardize_dataset(self, ds: xr.Dataset) -> xr.Dataset:
        """Standardize CESM-SE dataset dimensions."""
        return standardize_dims(ds, {"lev": "z", "ilev": "z_interface"})
