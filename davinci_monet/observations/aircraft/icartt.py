"""ICARTT (International Consortium for Atmospheric Research on Transport
and Transformation) format reader.

This module provides the ICARTTReader class for reading aircraft observations
in ICARTT format, commonly used for field campaign data.
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

# Common variable name mappings for ICARTT aircraft data
ICARTT_VARIABLE_MAPPING: dict[str, str] = {
    "ozone": "O3",
    "o3": "O3",
    "no": "NO",
    "no2": "NO2",
    "nox": "NOx",
    "co": "CO",
    "so2": "SO2",
    "hcho": "HCHO",
    "altitude": "GPS_Altitude",
    "pressure": "Static_Pressure",
    "temperature": "Static_Temperature",
    "latitude": "Latitude",
    "longitude": "Longitude",
}


@observation_registry.register("icartt")
class ICARTTReader:
    """Reader for ICARTT format aircraft observations.

    Reads aircraft data in ICARTT (.ict) format from field campaigns.
    Data is returned as track geometry with (time,) dimension.

    Examples
    --------
    >>> reader = ICARTTReader()
    >>> ds = reader.open(["flight_data.ict"])
    >>> print(ds.dims)
    Frozen({'time': 3600})
    """

    @property
    def name(self) -> str:
        """Return reader name."""
        return "icartt"

    def open(
        self,
        file_paths: Sequence[str | Path],
        variables: Sequence[str] | None = None,
        **kwargs: Any,
    ) -> xr.Dataset:
        """Open ICARTT observation files.

        Parameters
        ----------
        file_paths
            Paths to ICARTT files.
        variables
            Variables to load. If None, loads all available.
        **kwargs
            Additional options passed to monetio or parser.

        Returns
        -------
        xr.Dataset
            Aircraft observations with dimension (time,) and
            lat/lon/alt coordinates.
        """
        file_list = [Path(f) for f in file_paths]

        if not file_list:
            raise DataNotFoundError("No ICARTT files provided")

        missing = [f for f in file_list if not f.exists()]
        if missing:
            raise DataNotFoundError(f"ICARTT files not found: {missing}")

        # Try monetio first
        try:
            ds = self._open_with_monetio(file_list, variables, **kwargs)
        except ImportError:
            warnings.warn(
                "monetio not available, using basic ICARTT parser.",
                UserWarning,
            )
            ds = self._open_with_parser(file_list, variables, **kwargs)

        return self._standardize_dataset(ds)

    def _open_with_monetio(
        self,
        file_paths: list[Path],
        variables: Sequence[str] | None,
        **kwargs: Any,
    ) -> xr.Dataset:
        """Open ICARTT files using monetio."""
        import monetio.profile.icartt as icartt_mod

        ds_list = []
        for fpath in file_paths:
            try:
                result = icartt_mod.add_data(str(fpath), **kwargs)
                # monetio returns xarray Dataset directly
                if isinstance(result, xr.Dataset):
                    if len(result.time) > 0:
                        ds_list.append(result)
                elif isinstance(result, pd.DataFrame):
                    # Fallback for older monetio versions
                    if not result.empty:
                        ds_list.append(self._dataframe_to_dataset(result))
            except Exception as e:
                warnings.warn(f"Failed to read {fpath}: {e}", UserWarning)
                continue

        if not ds_list:
            raise DataNotFoundError("No valid ICARTT data found")

        # Concatenate datasets along time dimension
        if len(ds_list) == 1:
            ds = ds_list[0]
        else:
            ds = xr.concat(ds_list, dim="time")

        if variables is not None:
            available = [v for v in variables if v in ds.data_vars]
            if available:
                ds = ds[available]

        return ds

    def _open_with_parser(
        self,
        file_paths: list[Path],
        variables: Sequence[str] | None,
        **kwargs: Any,
    ) -> xr.Dataset:
        """Open ICARTT files using basic parser."""
        ds_list = []
        for fpath in file_paths:
            try:
                ds = self._parse_icartt_file(fpath)
                ds_list.append(ds)
            except Exception as e:
                warnings.warn(f"Failed to parse {fpath}: {e}", UserWarning)
                continue

        if not ds_list:
            raise DataNotFoundError("No valid ICARTT data found")

        ds = xr.concat(ds_list, dim="time")

        if variables is not None:
            available = [v for v in variables if v in ds.data_vars]
            if available:
                ds = ds[available]

        return ds

    def _parse_icartt_file(self, file_path: Path) -> xr.Dataset:
        """Parse a single ICARTT file.

        This is a simplified parser for basic ICARTT files.
        For full compliance, use monetio.
        """
        with open(file_path, "r") as f:
            lines = f.readlines()

        # Parse header
        first_line = lines[0].strip().split(",")
        n_header_lines = int(first_line[0])

        # Get variable names from header
        var_line_idx = n_header_lines - 1
        if var_line_idx < len(lines):
            var_names = [v.strip() for v in lines[var_line_idx].strip().split(",")]
        else:
            raise DataFormatError(f"Invalid ICARTT header in {file_path}")

        # Read data
        data_lines = lines[n_header_lines:]
        data = []
        for line in data_lines:
            if line.strip():
                values = [
                    float(v) if v.strip() not in ("", "NaN") else np.nan
                    for v in line.strip().split(",")
                ]
                data.append(values)

        if not data:
            raise DataFormatError(f"No data found in {file_path}")

        # Create DataFrame
        df = pd.DataFrame(data, columns=var_names[: len(data[0])])

        # Convert to xarray
        # Look for time variable (commonly first column or named time/Time_UTC)
        time_col = None
        for col in ["time", "Time_UTC", "UTC_Time", "Time"]:
            if col in df.columns:
                time_col = col
                break
        if time_col is None and len(df.columns) > 0:
            time_col = df.columns[0]

        if time_col:
            df = df.set_index(time_col)
            df.index.name = "time"

        ds: xr.Dataset = df.to_xarray()
        return ds

    def _dataframe_to_dataset(self, df: pd.DataFrame) -> xr.Dataset:
        """Convert ICARTT DataFrame to xarray Dataset."""
        # Look for time column
        time_col = None
        for col in ["time", "Time_UTC", "UTC_Time", "Time"]:
            if col in df.columns:
                time_col = col
                break

        if time_col:
            df = df.set_index(time_col)
            df.index.name = "time"

        ds: xr.Dataset = df.to_xarray()
        return ds

    def _standardize_dataset(self, ds: xr.Dataset) -> xr.Dataset:
        """Standardize ICARTT dataset dimensions and coordinates."""
        coord_renames: dict[str, str] = {}

        # Standardize coordinate names (including campaign-specific suffixes)
        lat_aliases = [
            "Latitude",
            "latitude",
            "LAT",
            "lat",
            "Latitude_YANG",
            "Latitude_BENNETT",
            "G_LAT",
            "LATITUDE",
        ]
        for alias in lat_aliases:
            if alias in ds.data_vars and "latitude" not in ds.coords:
                ds = ds.set_coords(alias)
                if alias != "latitude":
                    coord_renames[alias] = "latitude"
                break

        lon_aliases = [
            "Longitude",
            "longitude",
            "LON",
            "lon",
            "Longitude_YANG",
            "Longitude_BENNETT",
            "G_LONG",
            "LONGITUDE",
        ]
        for alias in lon_aliases:
            if alias in ds.data_vars and "longitude" not in ds.coords:
                ds = ds.set_coords(alias)
                if alias != "longitude":
                    coord_renames[alias] = "longitude"
                break

        # For altitude, prefer geometric altitude (meters first, then km, feet)
        # Variables in non-meter units will be converted to meters
        FEET_TO_METERS = 0.3048
        KM_TO_METERS = 1000.0
        feet_vars = {"Pressure_Altitude_BENNETT", "GPS_Altitude_BENNETT"}
        km_vars = {"ALTP", "GPS_ALT", "GPS_ALT_MMS", "RadarAlt"}
        alt_aliases = [
            # Altitude in meters (preferred)
            "GPS_Altitude_m_DIGANGI",
            "Altitude_AGL_m_DIGANGI",
            "MSL_GPS_Altitude_YANG",
            "MSL_GPS_Altitude",
            "GPS_Altitude",
            "Altitude",
            "altitude",
            "ALT",
            "alt",
            # Altitude in km (DC3 merge files, etc.)
            "ALTP",
            "GPS_ALT",
            "GPS_ALT_MMS",
            "RadarAlt",
            # Altitude in feet (will be converted)
            "Pressure_Altitude_BENNETT",
            "GPS_Altitude_BENNETT",
            # Pressure as last resort (not recommended for plotting)
            "Static_Pressure_BENNETT",
            "Static_Pressure",
            "PRESSURE",
        ]
        for alias in alt_aliases:
            if alias in ds.data_vars and "altitude" not in ds.coords:
                # Convert feet to meters if needed
                if alias in feet_vars:
                    ds[alias] = ds[alias] * FEET_TO_METERS
                    ds[alias].attrs["units"] = "m"
                # Convert km to meters if needed
                elif alias in km_vars:
                    ds[alias] = ds[alias] * KM_TO_METERS
                    ds[alias].attrs["units"] = "m"
                ds = ds.set_coords(alias)
                if alias != "altitude":
                    coord_renames[alias] = "altitude"
                break

        if coord_renames:
            ds = ds.rename(coord_renames)

        # Add flight identifier based on date (for per-flight analysis)
        if "time" in ds.dims:
            times = pd.to_datetime(ds["time"].values)
            flight_dates = times.strftime("%Y-%m-%d")
            ds = ds.assign_coords(flight=("time", flight_dates))

        ds.attrs["geometry"] = DataGeometry.TRACK.value

        return ds

    def get_variable_mapping(self) -> Mapping[str, str]:
        """Return ICARTT variable name mapping."""
        return ICARTT_VARIABLE_MAPPING


def open_icartt(
    files: str | Path | Sequence[str | Path],
    variables: Sequence[str] | None = None,
    label: str = "aircraft",
    **kwargs: Any,
) -> ObservationData:
    """Convenience function to open ICARTT observation data.

    Parameters
    ----------
    files
        File path(s) or glob pattern.
    variables
        Variables to load.
    label
        Observation label.
    **kwargs
        Additional reader options.

    Returns
    -------
    ObservationData
        ICARTT observation data container with TRACK geometry.
    """
    from glob import glob

    reader = ICARTTReader()

    if isinstance(files, (str, Path)):
        file_str = str(files)
        if "*" in file_str or "?" in file_str:
            file_list = sorted(glob(file_str))
            if not file_list:
                raise DataNotFoundError(f"No files match pattern: {files}")
            file_paths: Sequence[str | Path] = file_list
        else:
            file_paths = [files]
    else:
        file_paths = list(files)

    ds = reader.open(file_paths, variables, **kwargs)

    obs = create_observation_data(
        label=label,
        obs_type="aircraft",
        data=ds,
        variables=dict.fromkeys(variables) if variables else {},
    )
    obs.geometry = DataGeometry.TRACK

    return obs
