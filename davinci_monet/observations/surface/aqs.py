"""EPA Air Quality System (AQS) observation reader.

This module provides the AQSReader class for reading EPA AQS surface
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
from davinci_monet.core.registry import source_registry
from davinci_monet.observations.base import ObservationData, create_observation_data

# Standard variable name mappings for AQS
AQS_VARIABLE_MAPPING: dict[str, str] = {
    "ozone": "OZONE",
    "o3": "OZONE",
    "pm25": "PM2.5",
    "pm25_frm": "PM2.5",
    "pm10": "PM10",
    "no2": "NO2",
    "no": "NO",
    "nox": "NOX",
    "co": "CO",
    "so2": "SO2",
    "temperature": "TEMP",
    "wind_speed": "WS",
    "wind_direction": "WD",
    "relative_humidity": "RH_DP",
}


@source_registry.register("aqs")
class AQSReader:
    """Reader for EPA AQS surface observation data.

    Reads EPA Air Quality System data, supporting both pre-generated files
    and monetio API queries.

    Examples
    --------
    >>> reader = AQSReader()
    >>> ds = reader.open(["aqs_data.nc"])
    >>> print(ds.dims)
    Frozen({'time': 744, 'x': 1500})
    """

    @property
    def name(self) -> str:
        """Return reader name."""
        return "aqs"

    def open(
        self,
        file_paths: Sequence[str | Path] | None = None,
        variables: Sequence[str] | None = None,
        *,
        dates: Sequence[datetime | str] | None = None,
        daily: bool = False,
        **kwargs: Any,
    ) -> xr.Dataset:
        """Open AQS observation data.

        Parameters
        ----------
        file_paths
            Paths to pre-downloaded AQS files (NetCDF format).
        variables
            Variables to load. If None, loads all available.
        dates
            Date range for monetio API query [start, end].
        daily
            If True, load daily average data instead of hourly.
        **kwargs
            Additional options passed to monetio.

        Returns
        -------
        xr.Dataset
            AQS observations with dimensions (time, x).
        """
        if file_paths is not None:
            return self._open_files(file_paths, variables, **kwargs)
        elif dates is not None:
            return self._open_with_monetio(dates, variables, daily=daily, **kwargs)
        else:
            raise DataNotFoundError("Either file_paths or dates must be provided")

    def _open_files(
        self,
        file_paths: Sequence[str | Path],
        variables: Sequence[str] | None,
        **kwargs: Any,
    ) -> xr.Dataset:
        """Open AQS data from files."""
        file_list = [Path(f) for f in file_paths]

        if not file_list:
            raise DataNotFoundError("No AQS files provided")

        missing = [f for f in file_list if not f.exists()]
        if missing:
            raise DataNotFoundError(f"AQS files not found: {missing}")

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
            raise DataFormatError(f"Failed to open AQS files: {e}") from e

        if variables is not None:
            available = [v for v in variables if v in ds.data_vars]
            if available:
                ds = ds[available]

        return self._standardize_dataset(ds)

    def _open_with_monetio(
        self,
        dates: Sequence[datetime | str],
        variables: Sequence[str] | None,
        daily: bool = False,
        **kwargs: Any,
    ) -> xr.Dataset:
        """Open AQS data using monetio API."""
        try:
            import monetio.obs.aqs as aqs_mod
        except ImportError as e:
            raise ImportError(
                "monetio is required for AQS API queries. " "Install with: pip install monetio"
            ) from e

        # Parse dates
        start_date = pd.to_datetime(dates[0])
        end_date = pd.to_datetime(dates[-1])
        date_range = pd.date_range(start_date, end_date, freq="D")

        # Query AQS data
        ds_list = []
        for date in date_range:
            try:
                if daily:
                    df: pd.DataFrame = aqs_mod.add_data(date, daily=True, **kwargs)
                else:
                    df = aqs_mod.add_data(date, daily=False, **kwargs)
                if not df.empty:
                    ds_list.append(df)
            except Exception:
                continue

        if not ds_list:
            raise DataNotFoundError(f"No AQS data found for {start_date} to {end_date}")

        # Combine and convert to xarray
        combined_df = pd.concat(ds_list, ignore_index=True)
        ds: xr.Dataset = self._dataframe_to_dataset(combined_df)

        if variables is not None:
            available = [v for v in variables if v in ds.data_vars]
            if available:
                ds = ds[available]

        return self._standardize_dataset(ds)

    def _dataframe_to_dataset(self, df: pd.DataFrame) -> xr.Dataset:
        """Convert AQS DataFrame to xarray Dataset."""
        # AQS data typically has time, site info, and observations
        # Create multi-index on time and site
        if "time" in df.columns and "siteid" in df.columns:
            df = df.set_index(["time", "siteid"])
            ds: xr.Dataset = df.to_xarray()
            # Rename siteid to x for compatibility
            if "siteid" in ds.dims:
                ds = ds.rename({"siteid": "x"})
        else:
            ds = df.to_xarray()
        return ds

    def _standardize_dataset(self, ds: xr.Dataset) -> xr.Dataset:
        """Standardize AQS dataset dimensions and coordinates."""
        # Ensure standard coordinate names
        coord_renames: dict[str, str] = {}
        if "latitude" in ds.coords and "lat" not in ds.coords:
            coord_renames["latitude"] = "lat"
        if "longitude" in ds.coords and "lon" not in ds.coords:
            coord_renames["longitude"] = "lon"

        if coord_renames:
            ds = ds.rename(coord_renames)

        # Set geometry attribute
        ds.attrs["geometry"] = DataGeometry.POINT.value

        return ds

    def get_variable_mapping(self) -> Mapping[str, str]:
        """Return AQS variable name mapping."""
        return AQS_VARIABLE_MAPPING


def open_aqs(
    files: str | Path | Sequence[str | Path] | None = None,
    variables: Sequence[str] | None = None,
    label: str = "aqs",
    dates: Sequence[datetime | str] | None = None,
    daily: bool = False,
    **kwargs: Any,
) -> ObservationData:
    """Convenience function to open AQS observation data.

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
    daily
        If True, load daily averages.
    **kwargs
        Additional reader options.

    Returns
    -------
    ObservationData
        AQS observation data container.
    """
    from glob import glob

    reader = AQSReader()

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

    ds = reader.open(file_paths, variables, dates=dates, daily=daily, **kwargs)

    return create_observation_data(
        label=label,
        obs_type="pt_sfc",
        data=ds,
        variables=dict.fromkeys(variables) if variables else {},
    )
