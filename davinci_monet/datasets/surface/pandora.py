"""Pandora spectrometer dataset reader.

This module provides the PandoraReader class for reading Pandora L2
column NO2 and other trace gas measurements from ground-based spectrometers.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import xarray as xr

from davinci_monet.core.exceptions import DataNotFoundError
from davinci_monet.core.protocols import DataGeometry
from davinci_monet.core.registry import source_registry
from davinci_monet.io.reader_utils import (
    select_variables,
    set_geometry_attr,
    validate_file_list,
)

# Column indices in Pandora L2 files (0-indexed)
PANDORA_COLUMNS = {
    "datetime": 0,  # yyyymmddThhmmssZ
    "solar_zenith_angle": 3,  # deg
    "no2_quality_flag": 52,  # 0=high, 1=medium, 2=low quality
    "no2_trop_column": 61,  # mol/m2
    "no2_column_uncertainty": 62,  # mol/m2
}


def _parse_pandora_header(file_path: Path) -> dict[str, Any]:
    """Parse Pandora L2 file header to extract metadata.

    Parameters
    ----------
    file_path
        Path to Pandora L2 file.

    Returns
    -------
    dict
        Metadata including site name, lat, lon, altitude.
    """
    metadata: dict[str, Any] = {}

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if line.startswith("---"):
                break

            if line.startswith("Short location name:"):
                metadata["site_name"] = line.split(":", 1)[1].strip()
            elif line.startswith("Location latitude"):
                metadata["latitude"] = float(line.split(":")[-1].strip())
            elif line.startswith("Location longitude"):
                metadata["longitude"] = float(line.split(":")[-1].strip())
            elif line.startswith("Location altitude"):
                metadata["altitude"] = float(line.split(":")[-1].strip())
            elif line.startswith("Instrument number:"):
                metadata["instrument_number"] = int(line.split(":")[-1].strip())

    return metadata


def _parse_pandora_data(
    file_path: Path,
    quality_flag_max: int = 1,
    solar_zenith_max: float = 80.0,
) -> pd.DataFrame:
    """Parse Pandora L2 file data section.

    Parameters
    ----------
    file_path
        Path to Pandora L2 file.
    quality_flag_max
        Maximum quality flag to include (0=high, 1=medium, 2=low).
        Default 1 includes high and medium quality.
    solar_zenith_max
        Maximum solar zenith angle to include [deg].

    Returns
    -------
    pd.DataFrame
        Parsed data with columns: time, no2_trop_column, no2_column_uncertainty,
        no2_quality_flag, solar_zenith_angle.
    """
    data_lines = []
    separator_count = 0

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if line.startswith("---"):
                separator_count += 1
                continue
            # Data starts after the second separator line
            if separator_count >= 2 and line:
                data_lines.append(line)

    if not data_lines:
        return pd.DataFrame()

    # Parse data lines
    records = []
    for line in data_lines:
        parts = line.split()
        if len(parts) < 63:  # Need at least 63 columns for NO2
            continue

        try:
            # Parse datetime (column 1)
            dt_str = parts[PANDORA_COLUMNS["datetime"]]
            # Format: yyyymmddThhmmss.sZ
            dt = pd.to_datetime(dt_str.rstrip("Z"), format="%Y%m%dT%H%M%S.%f")

            # Parse other columns
            sza = float(parts[PANDORA_COLUMNS["solar_zenith_angle"]])
            qf = int(parts[PANDORA_COLUMNS["no2_quality_flag"]])
            no2_col = float(parts[PANDORA_COLUMNS["no2_trop_column"]])
            no2_unc = float(parts[PANDORA_COLUMNS["no2_column_uncertainty"]])

            # Apply quality filters
            # Quality flags: 0/10=high, 1/11=medium, 2/12=low, 20/21/22=unusable
            # Use last digit for quality level comparison
            qf_level = qf % 10
            if qf >= 20 or qf_level > quality_flag_max:
                continue
            if sza > solar_zenith_max:
                continue
            # Skip invalid retrievals
            if no2_col < -1e90:
                continue

            records.append(
                {
                    "time": dt,
                    "no2_trop_column": no2_col,
                    "no2_column_uncertainty": no2_unc,
                    "no2_quality_flag": qf,
                    "solar_zenith_angle": sza,
                }
            )

        except (ValueError, IndexError):
            # Skip malformed lines
            continue

    return pd.DataFrame(records)


@source_registry.register("pandora")
class PandoraReader:
    """Reader for Pandora spectrometer datasets.

    Reads Pandora L2 data files containing column NO2 measurements.

    Examples
    --------
    >>> reader = PandoraReader()
    >>> ds = reader.open(file_paths=["Pandora190s1_Bangkok_L2_rnvh3p1-8.txt"])
    """

    @property
    def name(self) -> str:
        """Return reader name."""
        return "pandora"

    @property
    def geometry(self) -> DataGeometry:
        """Return produced geometry."""
        return DataGeometry.POINT

    def open(
        self,
        file_paths: Sequence[str | Path],
        variables: Sequence[str] | None = None,
        *,
        quality_flag_max: int = 1,
        solar_zenith_max: float = 80.0,
        start_time: datetime | str | None = None,
        end_time: datetime | str | None = None,
        **kwargs: Any,
    ) -> xr.Dataset:
        """Open Pandora dataset data.

        Parameters
        ----------
        file_paths
            Paths to Pandora L2 files.
        variables
            Variables to load (default: all NO2 variables).
        quality_flag_max
            Maximum quality flag to include (0=high only, 1=high+medium).
        solar_zenith_max
            Maximum solar zenith angle [deg].
        start_time
            Start time for filtering data.
        end_time
            End time for filtering data.
        **kwargs
            Additional options.

        Returns
        -------
        xr.Dataset
            Pandora datasets with dimensions (time, site).
        """
        file_list = validate_file_list(file_paths, source_label="Pandora")

        # Parse time bounds
        if start_time is not None:
            start_time = pd.to_datetime(start_time)
        if end_time is not None:
            end_time = pd.to_datetime(end_time)

        # Read all files
        all_data = []
        site_metadata = {}

        for fpath in file_list:
            # Parse header for site info
            metadata = _parse_pandora_header(fpath)
            site_name = metadata.get("site_name", fpath.stem)

            # Parse data
            df = _parse_pandora_data(
                fpath,
                quality_flag_max=quality_flag_max,
                solar_zenith_max=solar_zenith_max,
            )

            if df.empty:
                continue

            # Filter by time
            if start_time is not None:
                df = df[df["time"] >= start_time]
            if end_time is not None:
                df = df[df["time"] <= end_time]

            if df.empty:
                continue

            # Add site info
            df["site"] = site_name
            df["latitude"] = metadata.get("latitude", np.nan)
            df["longitude"] = metadata.get("longitude", np.nan)
            df["altitude"] = metadata.get("altitude", np.nan)

            all_data.append(df)
            site_metadata[site_name] = metadata

        if not all_data:
            raise DataNotFoundError("No valid Pandora data found in provided files")

        # Combine all data
        combined_df = pd.concat(all_data, ignore_index=True)

        # Convert to xarray Dataset
        ds = self._dataframe_to_dataset(combined_df)

        # Select variables if specified
        return select_variables(ds, variables)

    def _dataframe_to_dataset(self, df: pd.DataFrame) -> xr.Dataset:
        """Convert Pandora DataFrame to xarray Dataset.

        Parameters
        ----------
        df
            DataFrame with columns: time, site, no2_trop_column, etc.

        Returns
        -------
        xr.Dataset
            Dataset with dimensions (time, site).
        """
        # Get unique sites and times
        sites = df["site"].unique().tolist()
        times = pd.to_datetime(df["time"].unique()).sort_values()

        # Get site coordinates
        site_coords = df.groupby("site")[["latitude", "longitude", "altitude"]].first()

        # Pivot data variables
        data_vars = {}
        for col in [
            "no2_trop_column",
            "no2_column_uncertainty",
            "no2_quality_flag",
            "solar_zenith_angle",
        ]:
            if col not in df.columns:
                continue
            pivoted = df.pivot_table(index="time", columns="site", values=col, aggfunc="first")
            pivoted = pivoted.reindex(index=times, columns=sites)
            data_vars[col] = (["time", "site"], pivoted.values)

        # Create coordinates
        coords = {
            "time": times,
            "site": sites,
            "latitude": ("site", [site_coords.loc[s, "latitude"] for s in sites]),
            "longitude": ("site", [site_coords.loc[s, "longitude"] for s in sites]),
            "altitude": ("site", [site_coords.loc[s, "altitude"] for s in sites]),
        }

        # Create dataset
        ds = xr.Dataset(data_vars, coords=coords)

        # Add attributes
        ds.attrs["source"] = "Pandora"
        set_geometry_attr(ds, DataGeometry.POINT)
        ds.attrs["description"] = "Pandora spectrometer column NO2 measurements"

        # Variable attributes
        if "no2_trop_column" in ds.data_vars:
            ds["no2_trop_column"].attrs = {
                "long_name": "NO2 tropospheric vertical column",
                "units": "mol/m2",
            }
        if "no2_column_uncertainty" in ds.data_vars:
            ds["no2_column_uncertainty"].attrs = {
                "long_name": "NO2 column uncertainty",
                "units": "mol/m2",
            }

        return ds
