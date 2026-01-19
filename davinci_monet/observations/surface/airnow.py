"""AirNow real-time air quality observation reader.

This module provides the AirNowReader class for reading AirNow surface
monitoring data.
"""

from __future__ import annotations

import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import xarray as xr

from davinci_monet.core.exceptions import DataFormatError, DataNotFoundError
from davinci_monet.core.protocols import DataGeometry
from davinci_monet.core.registry import observation_registry
from davinci_monet.observations.base import ObservationData, create_observation_data


# Standard variable name mappings for AirNow
AIRNOW_VARIABLE_MAPPING: dict[str, str] = {
    "ozone": "OZONE",
    "o3": "OZONE",
    "pm25": "PM2.5",
    "pm10": "PM10",
    "no2": "NO2",
    "co": "CO",
    "so2": "SO2",
}


@observation_registry.register("airnow")
class AirNowReader:
    """Reader for AirNow real-time air quality data.

    Reads AirNow data from files or via monetio API queries.

    Examples
    --------
    >>> reader = AirNowReader()
    >>> ds = reader.open(dates=["2024-01-01", "2024-01-02"])
    """

    @property
    def name(self) -> str:
        """Return reader name."""
        return "airnow"

    def open(
        self,
        file_paths: Sequence[str | Path] | None = None,
        variables: Sequence[str] | None = None,
        *,
        dates: Sequence[datetime | str] | None = None,
        wide_fmt: bool = True,
        **kwargs: Any,
    ) -> xr.Dataset:
        """Open AirNow observation data.

        Parameters
        ----------
        file_paths
            Paths to pre-downloaded AirNow files.
        variables
            Variables to load.
        dates
            Date range for monetio API query [start, end].
        wide_fmt
            If True, return wide format (one column per variable).
        **kwargs
            Additional options passed to monetio.

        Returns
        -------
        xr.Dataset
            AirNow observations with dimensions (time, x).
        """
        if file_paths is not None:
            return self._open_files(file_paths, variables, **kwargs)
        elif dates is not None:
            return self._open_with_monetio(dates, variables, wide_fmt=wide_fmt, **kwargs)
        else:
            raise DataNotFoundError("Either file_paths or dates must be provided")

    def _open_files(
        self,
        file_paths: Sequence[str | Path],
        variables: Sequence[str] | None,
        **kwargs: Any,
    ) -> xr.Dataset:
        """Open AirNow data from files."""
        file_list = [Path(f) for f in file_paths]

        if not file_list:
            raise DataNotFoundError("No AirNow files provided")

        missing = [f for f in file_list if not f.exists()]
        if missing:
            raise DataNotFoundError(f"AirNow files not found: {missing}")

        try:
            if len(file_list) > 1:
                ds = xr.open_mfdataset(
                    [str(f) for f in file_list],
                    combine="by_coords",
                    parallel=True,
                    **kwargs,
                )
            else:
                ds = xr.open_dataset(str(file_list[0]), **kwargs)
        except Exception as e:
            raise DataFormatError(f"Failed to open AirNow files: {e}") from e

        if variables is not None:
            available = [v for v in variables if v in ds.data_vars]
            if available:
                ds = ds[available]

        return self._standardize_dataset(ds)

    def _open_with_monetio(
        self,
        dates: Sequence[datetime | str],
        variables: Sequence[str] | None,
        wide_fmt: bool = True,
        **kwargs: Any,
    ) -> xr.Dataset:
        """Open AirNow data using monetio API."""
        try:
            import monetio.obs.airnow as airnow_mod
        except ImportError as e:
            raise ImportError(
                "monetio is required for AirNow API queries. "
                "Install with: pip install monetio"
            ) from e

        # Parse dates
        start_date = pd.to_datetime(dates[0])
        end_date = pd.to_datetime(dates[-1])
        date_range = pd.date_range(start_date, end_date, freq="D")

        # Query AirNow data
        ds_list = []
        for date in date_range:
            try:
                df: pd.DataFrame = airnow_mod.add_data(
                    date, wide_fmt=wide_fmt, **kwargs
                )
                if not df.empty:
                    ds_list.append(df)
            except Exception:
                continue

        if not ds_list:
            raise DataNotFoundError(
                f"No AirNow data found for {start_date} to {end_date}"
            )

        # Combine and convert to xarray
        combined_df = pd.concat(ds_list, ignore_index=True)
        ds: xr.Dataset = self._dataframe_to_dataset(combined_df)

        if variables is not None:
            available = [v for v in variables if v in ds.data_vars]
            if available:
                ds = ds[available]

        return self._standardize_dataset(ds)

    def _dataframe_to_dataset(self, df: pd.DataFrame) -> xr.Dataset:
        """Convert AirNow DataFrame to xarray Dataset."""
        if "time" in df.columns and "siteid" in df.columns:
            df = df.set_index(["time", "siteid"])
            ds: xr.Dataset = df.to_xarray()
            if "siteid" in ds.dims:
                ds = ds.rename({"siteid": "x"})
        else:
            ds = df.to_xarray()
        return ds

    def _standardize_dataset(self, ds: xr.Dataset) -> xr.Dataset:
        """Standardize AirNow dataset dimensions and coordinates."""
        coord_renames: dict[str, str] = {}
        if "latitude" in ds.coords and "lat" not in ds.coords:
            coord_renames["latitude"] = "lat"
        if "longitude" in ds.coords and "lon" not in ds.coords:
            coord_renames["longitude"] = "lon"

        if coord_renames:
            ds = ds.rename(coord_renames)

        ds.attrs["geometry"] = DataGeometry.POINT.value

        return ds

    def get_variable_mapping(self) -> Mapping[str, str]:
        """Return AirNow variable name mapping."""
        return AIRNOW_VARIABLE_MAPPING


def open_airnow(
    files: str | Path | Sequence[str | Path] | None = None,
    variables: Sequence[str] | None = None,
    label: str = "airnow",
    dates: Sequence[datetime | str] | None = None,
    **kwargs: Any,
) -> ObservationData:
    """Convenience function to open AirNow observation data.

    Parameters
    ----------
    files
        File path(s) or glob pattern.
    variables
        Variables to load.
    label
        Observation label.
    dates
        Date range for API query [start, end].
    **kwargs
        Additional reader options.

    Returns
    -------
    ObservationData
        AirNow observation data container.
    """
    from glob import glob

    reader = AirNowReader()

    file_paths: Sequence[str | Path] | None = None
    if files is not None:
        if isinstance(files, (str, Path)):
            file_str = str(files)
            if "*" in file_str or "?" in file_str:
                file_list = sorted(glob(file_str))
                if not file_list:
                    raise DataNotFoundError(f"No files match pattern: {files}")
                file_paths = file_list
            else:
                file_paths = [files]
        else:
            file_paths = list(files)

    ds = reader.open(file_paths, variables, dates=dates, **kwargs)

    return create_observation_data(
        label=label,
        obs_type="pt_sfc",
        data=ds,
        variables=dict.fromkeys(variables) if variables else {},
    )
