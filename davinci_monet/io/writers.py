"""File writers for various data formats.

This module provides functions for writing data to different file formats
including NetCDF, pickle, and CSV.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import pandas as pd
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

            ds.to_netcdf(str(path), engine=engine, **kwargs)
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


def write_csv(
    df: pd.DataFrame,
    path: str | Path,
    index: bool = True,
    **kwargs: Any,
) -> None:
    """Write DataFrame to CSV file.

    Parameters
    ----------
    df
        DataFrame to write.
    path
        Output file path.
    index
        Whether to write index column.
    **kwargs
        Additional arguments passed to pandas.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        df.to_csv(str(path), index=index, **kwargs)
    except Exception as e:
        raise DataFormatError(f"Failed to write CSV {path}: {e}") from e


def write_paired_data(
    paired: dict[str, xr.Dataset],
    output_dir: str | Path,
    format: str = "netcdf",
    prefix: str = "",
) -> list[str]:
    """Write paired model-observation data to files.

    Parameters
    ----------
    paired
        Dictionary of paired datasets keyed by pair name.
    output_dir
        Output directory.
    format
        Output format ('netcdf', 'pickle').
    prefix
        Optional filename prefix.

    Returns
    -------
    list[str]
        List of written file paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    written_files = []

    for pair_key, ds in paired.items():
        if ds is None:
            continue

        if prefix:
            filename = f"{prefix}_{pair_key}"
        else:
            filename = pair_key

        if format == "pickle":
            filepath = output_dir / f"{filename}_paired.pkl"
            write_pickle(ds, filepath)
        else:
            filepath = output_dir / f"{filename}_paired.nc"
            write_dataset(ds, filepath)

        written_files.append(str(filepath))

    return written_files


def write_statistics(
    stats: dict[str, Any],
    path: str | Path,
    format: str = "csv",
) -> None:
    """Write statistics results to file.

    Parameters
    ----------
    stats
        Statistics dictionary.
    path
        Output file path.
    format
        Output format ('csv', 'json', 'pickle').
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        if format == "csv":
            # Flatten nested dict to DataFrame
            rows = []
            for pair_key, pair_stats in stats.items():
                for var_name, var_stats in pair_stats.items():
                    row = {"pair": pair_key, "variable": var_name}
                    row.update(var_stats)
                    rows.append(row)
            df = pd.DataFrame(rows)
            write_csv(df, path, index=False)

        elif format == "json":
            import json

            with open(path, "w") as f:
                json.dump(stats, f, indent=2, default=str)

        elif format == "pickle":
            write_pickle(stats, path)

        else:
            raise DataFormatError(f"Unsupported format: {format}")

    except Exception as e:
        raise DataFormatError(f"Failed to write statistics {path}: {e}") from e


def write_analysis_results(
    context: Any,
    output_dir: str | Path,
    save_paired: bool = True,
    save_stats: bool = True,
    format: str = "netcdf",
) -> dict[str, list[str]]:
    """Write complete analysis results from pipeline context.

    Parameters
    ----------
    context
        Pipeline context containing results.
    output_dir
        Output directory.
    save_paired
        Whether to save paired data.
    save_stats
        Whether to save statistics.
    format
        Output format for paired data.

    Returns
    -------
    dict[str, list[str]]
        Dictionary of written file paths by category.
    """
    from davinci_monet.pipeline.stages import PipelineContext

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    written: dict[str, list[str]] = {
        "paired": [],
        "statistics": [],
    }

    if save_paired and hasattr(context, "paired"):
        written["paired"] = write_paired_data(
            context.paired,
            output_dir / "paired",
            format=format,
        )

    if save_stats and hasattr(context, "results"):
        stats_result = context.results.get("statistics")
        if stats_result and stats_result.data:
            stats_path = output_dir / "statistics.csv"
            write_statistics(stats_result.data, stats_path)
            written["statistics"].append(str(stats_path))

    return written
