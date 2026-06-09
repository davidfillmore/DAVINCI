"""OpenAQ global air quality observation reader.

This module provides the OpenAQReader class for reading OpenAQ data
from the global air quality database.
"""

from __future__ import annotations

import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import xarray as xr

from davinci_monet.core.exceptions import DataFormatError, DataNotFoundError
from davinci_monet.core.protocols import DataGeometry
from davinci_monet.core.registry import source_registry
from davinci_monet.io.reader_utils import (
    retry_transient_open,
    select_variables,
    set_geometry_attr,
    validate_file_list,
)


@source_registry.register("openaq")
class OpenAQReader:
    """Reader for OpenAQ global air quality data.

    Reads OpenAQ data from files or via monetio API queries.

    Examples
    --------
    >>> reader = OpenAQReader()
    >>> ds = reader.open(
    ...     dates=["2024-01-01", "2024-01-02"],
    ...     country="US",
    ...     parameter="pm25"
    ... )
    """

    @property
    def name(self) -> str:
        """Return reader name."""
        return "openaq"

    def open(
        self,
        file_paths: Sequence[str | Path] | None = None,
        variables: Sequence[str] | None = None,
        *,
        dates: Sequence[datetime | str] | None = None,
        country: str | None = None,
        city: str | None = None,
        parameter: str | Sequence[str] | None = None,
        api_version: int = 2,
        **kwargs: Any,
    ) -> xr.Dataset:
        """Open OpenAQ observation data.

        Parameters
        ----------
        file_paths
            Paths to pre-downloaded OpenAQ files.
        variables
            Variables to load.
        dates
            Date range for API query [start, end].
        country
            Country code (e.g., 'US', 'GB').
        city
            City name filter.
        parameter
            Parameter(s) to query ('pm25', 'o3', etc.).
        api_version
            OpenAQ API version (2 or 3).
        **kwargs
            Additional options passed to monetio.

        Returns
        -------
        xr.Dataset
            OpenAQ observations with dimensions (time, x).
        """
        if file_paths is not None:
            return self._open_files(file_paths, variables, **kwargs)
        elif dates is not None:
            return self._open_with_monetio(
                dates,
                variables,
                country=country,
                city=city,
                parameter=parameter,
                api_version=api_version,
                **kwargs,
            )
        else:
            raise DataNotFoundError("Either file_paths or dates must be provided")

    def _open_files(
        self,
        file_paths: Sequence[str | Path],
        variables: Sequence[str] | None,
        **kwargs: Any,
    ) -> xr.Dataset:
        """Open OpenAQ data from files."""
        file_list = validate_file_list(file_paths, source_label="OpenAQ")

        def _open() -> xr.Dataset:
            if len(file_list) > 1:
                ds = xr.open_mfdataset(
                    [str(f) for f in file_list],
                    combine="by_coords",
                    parallel=True,
                    **kwargs,
                )
            else:
                ds = xr.open_dataset(str(file_list[0]), **kwargs)
            return select_variables(ds, variables)

        ds = retry_transient_open(_open, context="Opening OpenAQ files")

        return self._standardize_dataset(ds)

    def _open_with_monetio(
        self,
        dates: Sequence[datetime | str],
        variables: Sequence[str] | None,
        country: str | None = None,
        city: str | None = None,
        parameter: str | Sequence[str] | None = None,
        api_version: int = 2,
        **kwargs: Any,
    ) -> xr.Dataset:
        """Open OpenAQ data using monetio API."""
        # Select appropriate monetio module
        if api_version == 3:
            try:
                import monetio.obs.openaq_v3 as openaq_mod
            except ImportError as e:
                raise ImportError(
                    "monetio is required for OpenAQ API queries. "
                    "Install with: pip install monetio"
                ) from e
        else:
            try:
                import monetio.obs.openaq_v2 as openaq_mod
            except ImportError:
                try:
                    import monetio.obs.openaq as openaq_mod
                except ImportError as e:
                    raise ImportError(
                        "monetio is required for OpenAQ API queries. "
                        "Install with: pip install monetio"
                    ) from e

        # Parse dates
        start_date = pd.to_datetime(dates[0])
        end_date = pd.to_datetime(dates[-1])

        # Build kwargs for monetio
        mio_kwargs: dict[str, Any] = {}
        if country is not None:
            mio_kwargs["country"] = country
        if city is not None:
            mio_kwargs["city"] = city
        if parameter is not None:
            if isinstance(parameter, str):
                mio_kwargs["parameter"] = [parameter]
            else:
                mio_kwargs["parameter"] = list(parameter)
        mio_kwargs.update(kwargs)

        try:
            df: pd.DataFrame = openaq_mod.add_data(start_date, end_date, **mio_kwargs)
        except Exception as e:
            raise DataFormatError(f"Failed to query OpenAQ data: {e}") from e

        if df.empty:
            raise DataNotFoundError(f"No OpenAQ data found for {start_date} to {end_date}")

        ds: xr.Dataset = self._dataframe_to_dataset(df)
        ds = select_variables(ds, variables)

        return self._standardize_dataset(ds)

    def _dataframe_to_dataset(self, df: pd.DataFrame) -> xr.Dataset:
        """Convert OpenAQ DataFrame to xarray Dataset."""
        if "time" in df.columns and "location" in df.columns:
            df = df.set_index(["time", "location"])
            ds: xr.Dataset = df.to_xarray()
            if "location" in ds.dims:
                ds = ds.rename({"location": "x"})
        elif "time" in df.columns:
            df = df.set_index("time")
            ds = df.to_xarray()
        else:
            ds = df.to_xarray()
        return ds

    def _standardize_dataset(self, ds: xr.Dataset) -> xr.Dataset:
        """Standardize OpenAQ dataset dimensions and coordinates."""
        coord_renames: dict[str, str] = {}
        if "latitude" in ds.coords and "lat" not in ds.coords:
            coord_renames["latitude"] = "lat"
        if "longitude" in ds.coords and "lon" not in ds.coords:
            coord_renames["longitude"] = "lon"

        if coord_renames:
            ds = ds.rename(coord_renames)

        return set_geometry_attr(ds, DataGeometry.POINT)
