"""File writers for various data formats.

This module provides functions for writing data to different file formats
including NetCDF and pickle.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import xarray as xr

from davinci_monet.core.exceptions import DataFormatError


def write_dataset(
    ds: xr.Dataset,
    path: str | Path,
    engine: str | None = None,
    compress: bool = False,
    **kwargs: Any,
) -> None:
    """Write a dataset to file.

    Automatically detects format based on file extension.

    Parameters
    ----------
    ds
        Dataset to write.
    path
        Output file path.
    engine
        NetCDF engine to use.
    compress
        Whether to apply zlib compression to NetCDF variables.
    **kwargs
        Additional arguments passed to xarray.

    Raises
    ------
    DataFormatError
        If write fails.
    """
    path = Path(path)

    # Create parent directory if needed
    path.parent.mkdir(parents=True, exist_ok=True)

    suffix = path.suffix.lower()

    try:
        if suffix in (".pkl", ".pickle"):
            write_pickle(ds, path)
        elif suffix == ".zarr":
            ds.to_zarr(str(path), **kwargs)
        else:
            # Default to NetCDF
            if engine is None:
                engine = "netcdf4"

            # Handle compression
            if compress:
                encoding = kwargs.pop("encoding", {})
                for var in ds.data_vars:
                    if var not in encoding:
                        encoding[var] = {}
                    encoding[var].setdefault("zlib", True)
                    encoding[var].setdefault("complevel", 4)
                kwargs["encoding"] = encoding

            ds.to_netcdf(str(path), engine=engine, **kwargs)  # type: ignore[call-overload]
    except Exception as e:
        raise DataFormatError(f"Failed to write {path}: {e}") from e


def write_pickle(
    data: Any,
    path: str | Path,
    protocol: int = pickle.HIGHEST_PROTOCOL,
) -> None:
    """Write data to pickle file.

    Parameters
    ----------
    data
        Data to pickle.
    path
        Output file path.
    protocol
        Pickle protocol version.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(path, "wb") as f:
            pickle.dump(data, f, protocol=protocol)
    except Exception as e:
        raise DataFormatError(f"Failed to write pickle {path}: {e}") from e
