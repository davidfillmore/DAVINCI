"""Generic model reader.

This module provides a fallback reader for model output that doesn't match
any specific model type. It uses xarray's generic NetCDF/grib readers.
"""

from __future__ import annotations

import gc
import os
import sys
import warnings
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import xarray as xr

from davinci_monet.core.exceptions import (
    DataFormatError,
    DataNotFoundError,
    cleanup_netcdf_state,
    is_transient_error,
)
from davinci_monet.core.protocols import DataGeometry
from davinci_monet.core.registry import source_registry
from davinci_monet.models.base import ModelData, create_model_data

# Common coordinate name aliases for standardization
COMMON_COORDINATE_ALIASES: dict[str, list[str]] = {
    "time": ["time", "Time", "TSTEP", "t", "datetime", "date_time"],
    "z": [
        "z",
        "lev",
        "level",
        "levels",
        "z_level",
        "altitude",
        "height",
        "bottom_top",
        "LAY",
        "layer",
        "pfull",
        "phalf",
        "sigma",
    ],
    "lat": [
        "lat",
        "latitude",
        "LAT",
        "LATITUDE",
        "XLAT",
        "south_north",
        "nlat",
        "rlat",
        "grid_yt",
        "y",
    ],
    "lon": [
        "lon",
        "longitude",
        "LON",
        "LONGITUDE",
        "XLONG",
        "west_east",
        "nlon",
        "rlon",
        "grid_xt",
        "x",
    ],
}


def _cleanup_with_suppressed_errors() -> None:
    """Force garbage collection with stderr suppressed to hide cleanup errors.

    When xarray fails to open files, cleanup code in __del__ methods can
    produce ugly tracebacks like "Exception ignored in: CachingFileManager.__del__".
    These can't be caught with try/except, so we suppress stderr during gc.
    """
    old_stderr = sys.stderr
    try:
        sys.stderr = open(os.devnull, "w")
        gc.collect()
    finally:
        sys.stderr.close()
        sys.stderr = old_stderr


@source_registry.register("generic")
class GenericReader:
    """Generic model reader for arbitrary NetCDF/grib files.

    This reader provides a fallback for model types that don't have
    a dedicated reader. It attempts to standardize dimensions and
    coordinates automatically.

    Examples
    --------
    >>> reader = GenericReader()
    >>> ds = reader.open(["model_output.nc"])
    """

    @property
    def name(self) -> str:
        """Return reader name."""
        return "generic"

    @property
    def geometry(self) -> DataGeometry:
        """Model output is gridded."""
        return DataGeometry.GRID

    def open(
        self,
        file_paths: Sequence[str | Path],
        variables: Sequence[str] | None = None,
        progress_callback: Callable[[int, int, str], None] | None = None,
        **kwargs: Any,
    ) -> xr.Dataset:
        """Open model files using xarray.

        Parameters
        ----------
        file_paths
            Paths to model output files.
        variables
            Variables to load. If None, loads all variables.
        progress_callback
            Optional callable ``(i, total, name) -> None`` invoked for each
            file as it is opened.  ``i`` is 1-based, ``total`` is the number
            of files, and ``name`` is the file basename.  When provided with
            multiple files the reader switches from ``parallel=True`` to
            ``parallel=False`` so the preprocess hook runs sequentially and
            the counter is monotonically increasing.  The default path
            (``progress_callback=None``, or a single file) is unchanged.
        **kwargs
            Additional options:
            - engine: xarray engine to use ('netcdf4', 'h5netcdf', 'cfgrib', etc.)
            - standardize: bool, whether to standardize dimension names (default True)
            - concat_dim: Dimension to concatenate along (default 'time')
            - combine: How to combine files ('by_coords', 'nested')

        Returns
        -------
        xr.Dataset
            Model data with optionally standardized dimensions.
        """
        file_list = [Path(f) for f in file_paths]

        if not file_list:
            raise DataNotFoundError("No files provided")

        # Check files exist
        missing = [f for f in file_list if not f.exists()]
        if missing:
            raise DataNotFoundError(f"Files not found: {missing}")

        # Extract our custom kwargs
        standardize = kwargs.pop("standardize", True)
        concat_dim = kwargs.pop("concat_dim", "time")
        combine = kwargs.pop("combine", "by_coords")

        # Detect engine if not specified
        if "engine" not in kwargs:
            engine = self._detect_engine(file_list[0])
            if engine:
                kwargs["engine"] = engine

        # Open files with retry for transient NetCDF errors
        max_retries = 3
        last_error: Exception | None = None

        for attempt in range(max_retries):
            try:
                if len(file_list) > 1:
                    if progress_callback is not None:
                        # Sequential open so the preprocess counter is ordered.
                        total = len(file_list)
                        counter: list[int] = [0]

                        def _progress_preprocess(ds: xr.Dataset) -> xr.Dataset:
                            counter[0] += 1
                            source = ds.encoding.get("source", "")
                            name = Path(source).name if source else ""
                            progress_callback(counter[0], total, name)
                            return ds

                        ds = xr.open_mfdataset(
                            [str(f) for f in file_list],
                            combine=combine,
                            data_vars="all",
                            parallel=False,
                            preprocess=_progress_preprocess,
                            **kwargs,
                        )
                    else:
                        ds = xr.open_mfdataset(
                            [str(f) for f in file_list],
                            combine=combine,
                            data_vars="all",
                            parallel=True,
                            **kwargs,
                        )
                else:
                    ds = xr.open_dataset(str(file_list[0]), **kwargs)
                break  # Success - exit retry loop
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
                _cleanup_with_suppressed_errors()
                raise DataFormatError(f"Failed to open files: {e}") from e
        else:
            # All retries exhausted
            _cleanup_with_suppressed_errors()
            raise DataFormatError(
                f"Failed to open files after {max_retries} attempts: {last_error}"
            ) from last_error

        # Select variables if specified
        if variables is not None:
            available = [v for v in variables if v in ds.data_vars]
            if available:
                ds = ds[available]

        # Standardize dimensions
        if standardize:
            ds = self._standardize_dataset(ds)

        return ds

    def _detect_engine(self, file_path: Path) -> str | None:
        """Detect appropriate xarray engine for file.

        Parameters
        ----------
        file_path
            Path to check.

        Returns
        -------
        str | None
            Engine name or None for default.
        """
        suffix = file_path.suffix.lower()

        if suffix in (".grib", ".grib2", ".grb", ".grb2"):
            return "cfgrib"
        elif suffix in (".zarr",):
            return "zarr"
        elif suffix in (".nc", ".nc4", ".netcdf"):
            return "netcdf4"

        return None

    def _standardize_dataset(self, ds: xr.Dataset) -> xr.Dataset:
        """Standardize dataset dimensions and coordinates.

        Parameters
        ----------
        ds
            Raw dataset.

        Returns
        -------
        xr.Dataset
            Dataset with standardized dimension names.
        """
        dim_renames: dict[str, str] = {}
        coord_renames: dict[str, str] = {}

        # Check each standard name against aliases
        for standard_name, aliases in COMMON_COORDINATE_ALIASES.items():
            for alias in aliases:
                # Check dimensions
                if alias in ds.dims and alias != standard_name:
                    if standard_name not in ds.dims:
                        dim_renames[alias] = standard_name
                    break

                # Check coordinates
                if alias in ds.coords and alias != standard_name:
                    if standard_name not in ds.coords:
                        coord_renames[alias] = standard_name
                    break

        # Apply renames
        if dim_renames:
            ds = ds.rename(dim_renames)

        if coord_renames:
            ds = ds.rename(coord_renames)

        return ds

    def get_variable_mapping(self) -> Mapping[str, str]:
        """Return empty mapping (generic reader has no predefined mapping).

        Returns
        -------
        Mapping[str, str]
            Empty mapping.
        """
        return {}


def open_model(
    files: str | Path | Sequence[str | Path],
    mod_type: str | None = None,
    variables: Sequence[str] | None = None,
    label: str = "model",
    **kwargs: Any,
) -> ModelData:
    """Universal function to open any model type.

    Automatically selects the appropriate reader based on mod_type,
    or uses the generic reader if no specific reader matches.

    Parameters
    ----------
    files
        File path(s) or glob pattern.
    mod_type
        Model type ('cmaq', 'wrfchem', 'ufs', 'cesm_fv', 'cesm_se', or None for generic).
    variables
        Variables to load.
    label
        Model label.
    **kwargs
        Additional reader options.

    Returns
    -------
    ModelData
        Model data container.

    Examples
    --------
    >>> data = open_model("output/*.nc", mod_type="cmaq", label="CMAQ_run1")
    >>> data = open_model("wrfout_d01_*", mod_type="wrfchem")
    >>> data = open_model("generic_output.nc")  # Uses generic reader
    """
    from davinci_monet.core.registry import source_registry

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

    # Get appropriate reader
    if mod_type is not None:
        mod_type_lower = mod_type.lower()
        try:
            reader_cls = source_registry.get(mod_type_lower)
            reader = reader_cls()
        except (KeyError, Exception):
            warnings.warn(
                f"Unknown model type '{mod_type}', using generic reader.",
                UserWarning,
            )
            reader = GenericReader()
            mod_type_lower = "generic"
    else:
        reader = GenericReader()
        mod_type_lower = "generic"

    # Open files
    ds = reader.open(file_paths, variables, **kwargs)

    return create_model_data(
        label=label,
        mod_type=mod_type_lower,
        data=ds,
        files=file_paths,
    )
