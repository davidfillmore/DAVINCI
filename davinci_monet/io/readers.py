"""File readers for various data formats.

This module provides functions for reading data from different file formats
including NetCDF, pickle, CSV, and ICARTT.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import xarray as xr

from davinci_monet.core.exceptions import DataFormatError, DataNotFoundError, write_error_log


def read_dataset(
    path: str | Path,
    engine: str | None = None,
    **kwargs: Any,
) -> xr.Dataset:
    """Read a dataset from file.

    Automatically detects format based on file extension.

    Parameters
    ----------
    path
        Path to file.
    engine
        xarray engine to use. If None, auto-detects.
    **kwargs
        Additional arguments passed to xarray.

    Returns
    -------
    xr.Dataset
        Loaded dataset.

    Raises
    ------
    DataNotFoundError
        If file does not exist.
    DataFormatError
        If file format is not supported.
    """
    path = Path(path)

    if not path.exists():
        raise DataNotFoundError(f"File not found: {path}")

    suffix = path.suffix.lower()

    # Auto-detect engine
    if engine is None:
        if suffix in (".nc", ".nc4", ".netcdf"):
            engine = "netcdf4"
        elif suffix in (".grib", ".grib2", ".grb", ".grb2"):
            engine = "cfgrib"
        elif suffix in (".zarr",):
            engine = "zarr"
        elif suffix in (".pkl", ".pickle"):
            return read_pickle(path)
        else:
            engine = "netcdf4"  # Default

    try:
        ds: xr.Dataset = xr.open_dataset(str(path), engine=engine, **kwargs)
        return ds
    except Exception as e:
        raise DataFormatError(f"Failed to read {path}: {e}") from e


def read_mfdataset(
    paths: Sequence[str | Path],
    engine: str | None = None,
    combine: str = "by_coords",
    **kwargs: Any,
) -> xr.Dataset:
    """Read multiple files into a single dataset.

    Parameters
    ----------
    paths
        List of file paths or glob pattern.
    engine
        xarray engine to use.
    combine
        How to combine files ('by_coords', 'nested').
    **kwargs
        Additional arguments passed to xarray.

    Returns
    -------
    xr.Dataset
        Combined dataset.
    """
    from glob import glob

    # Handle glob patterns
    file_list: list[str] = []
    for path in paths:
        path_str = str(path)
        if "*" in path_str or "?" in path_str:
            file_list.extend(sorted(glob(path_str)))
        else:
            file_list.append(path_str)

    if not file_list:
        raise DataNotFoundError(f"No files found matching: {paths}")

    # Check if all files exist
    missing = [f for f in file_list if not Path(f).exists()]
    if missing:
        raise DataNotFoundError(f"Files not found: {missing}")

    # Auto-detect engine from first file
    if engine is None:
        suffix = Path(file_list[0]).suffix.lower()
        if suffix in (".nc", ".nc4", ".netcdf"):
            engine = "netcdf4"
        elif suffix in (".grib", ".grib2", ".grb", ".grb2"):
            engine = "cfgrib"

    try:
        ds: xr.Dataset = xr.open_mfdataset(
            file_list,
            engine=engine,
            combine=combine,
            parallel=True,
            **kwargs,
        )
        return ds
    except Exception as e:
        raise DataFormatError(f"Failed to read files: {e}") from e


def read_pickle(path: str | Path) -> xr.Dataset | pd.DataFrame | Any:
    """Read data from pickle file.

    Parameters
    ----------
    path
        Path to pickle file.

    Returns
    -------
    Any
        Unpickled data.
    """
    path = Path(path)

    if not path.exists():
        raise DataNotFoundError(f"File not found: {path}")

    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
        return data
    except Exception as e:
        raise DataFormatError(f"Failed to read pickle {path}: {e}") from e


def read_csv(
    path: str | Path,
    parse_dates: bool | list[str] = True,
    **kwargs: Any,
) -> pd.DataFrame:
    """Read data from CSV file.

    Parameters
    ----------
    path
        Path to CSV file.
    parse_dates
        Whether to parse date columns.
    **kwargs
        Additional arguments passed to pandas.

    Returns
    -------
    pd.DataFrame
        Loaded data.
    """
    path = Path(path)

    if not path.exists():
        raise DataNotFoundError(f"File not found: {path}")

    try:
        df: pd.DataFrame = pd.read_csv(str(path), parse_dates=parse_dates, **kwargs)
        return df
    except Exception as e:
        raise DataFormatError(f"Failed to read CSV {path}: {e}") from e


def read_csv_to_xarray(
    path: str | Path,
    time_column: str = "time",
    index_columns: list[str] | None = None,
    **kwargs: Any,
) -> xr.Dataset:
    """Read CSV and convert to xarray Dataset.

    Parameters
    ----------
    path
        Path to CSV file.
    time_column
        Name of time column.
    index_columns
        Columns to use as index/dimensions.
    **kwargs
        Additional arguments passed to pandas.

    Returns
    -------
    xr.Dataset
        Converted dataset.
    """
    df = read_csv(path, **kwargs)

    # Set up index
    if index_columns:
        df = df.set_index(index_columns)
    elif time_column in df.columns:
        df = df.set_index(time_column)

    ds: xr.Dataset = df.to_xarray()
    return ds


def read_icartt(path: str | Path, **kwargs: Any) -> pd.DataFrame:
    """Read ICARTT format file.

    Parameters
    ----------
    path
        Path to ICARTT file.
    **kwargs
        Additional options.

    Returns
    -------
    pd.DataFrame
        Loaded data.
    """
    path = Path(path)

    if not path.exists():
        raise DataNotFoundError(f"File not found: {path}")

    try:
        # Try monetio first
        import monetio.profile.icartt as icartt_mod
        df: pd.DataFrame = icartt_mod.add_data(str(path), **kwargs)
        return df
    except ImportError:
        # Fall back to basic parsing
        return _parse_icartt_basic(path)


def _parse_icartt_basic(path: Path) -> pd.DataFrame:
    """Basic ICARTT parser.

    Parameters
    ----------
    path
        Path to ICARTT file.

    Returns
    -------
    pd.DataFrame
        Parsed data.
    """
    try:
        with open(path, "r") as f:
            lines = f.readlines()
    except OSError as e:
        error_file = write_error_log(e, f"Reading ICARTT file '{path}'")
        msg = f"Failed to read ICARTT file '{path}': {e}"
        if error_file:
            msg += f" (details: {error_file})"
        raise DataFormatError(msg) from e

    # Parse header
    first_line = lines[0].strip().split(",")
    n_header_lines = int(first_line[0])

    # Get variable names from last header line
    var_line = lines[n_header_lines - 1].strip()
    var_names = [v.strip() for v in var_line.split(",")]

    # Read data
    data_lines = lines[n_header_lines:]
    data = []
    for line in data_lines:
        if line.strip():
            values = []
            for v in line.strip().split(","):
                try:
                    values.append(float(v.strip()))
                except ValueError:
                    values.append(np.nan)
            data.append(values)

    if not data:
        raise DataFormatError(f"No data found in {path}")

    df = pd.DataFrame(data, columns=var_names[: len(data[0])])
    return df


def read_saved_analysis(
    path: str | Path,
    format: str = "auto",
) -> dict[str, Any]:
    """Read saved analysis results.

    Parameters
    ----------
    path
        Path to saved analysis.
    format
        File format ('pickle', 'netcdf', 'auto').

    Returns
    -------
    dict[str, Any]
        Loaded analysis data.
    """
    path = Path(path)

    if not path.exists():
        raise DataNotFoundError(f"File not found: {path}")

    if format == "auto":
        suffix = path.suffix.lower()
        if suffix in (".pkl", ".pickle"):
            format = "pickle"
        elif suffix in (".nc", ".nc4"):
            format = "netcdf"
        else:
            format = "pickle"

    if format == "pickle":
        return read_pickle(path)
    elif format == "netcdf":
        ds = read_dataset(path)
        return {"data": ds}
    else:
        raise DataFormatError(f"Unsupported format: {format}")
