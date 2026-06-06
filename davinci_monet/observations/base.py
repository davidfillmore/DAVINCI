"""Base observation class for observational data.

This module provides the ObservationData class that wraps observation data
with common operations for loading, processing, and transforming.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from glob import glob
from pathlib import Path
from typing import Any, Mapping, Sequence

import xarray as xr

from davinci_monet.core.base import DataContainer
from davinci_monet.core.exceptions import DataFormatError, DataNotFoundError, DataValidationError
from davinci_monet.core.protocols import DataGeometry
from davinci_monet.core.registry import source_registry
from davinci_monet.core.types import PathLike, TimeRange, VariableMapping


def resample_dataset(
    data: "xr.Dataset",
    freq: str,
    min_count: int | None = None,
    track_count: bool = False,
) -> "xr.Dataset":
    """Resample a dataset along ``time``, masking sparse bins and optionally counting.

    Pure-function form of :meth:`ObservationData.resample_data` so the unified
    source loader can resample bare datasets without an ``ObservationData`` wrapper.
    """
    if "time" not in data.dims:
        return data
    resampler = data.resample(time=freq)
    result = resampler.mean()
    if track_count or min_count is not None:
        data_vars = [v for v in data.data_vars if v not in ("latitude", "longitude", "altitude")]
        if data_vars:
            counts = resampler.count()[data_vars[0]]
            if track_count:
                result["obs_count"] = counts
            if min_count is not None:
                mask = counts >= min_count
                for var in data_vars:
                    if var in result:
                        result[var] = result[var].where(mask)
    return result


@dataclass
class ObservationData(DataContainer):
    """Container for observational data.

    Wraps observation data (surface stations, aircraft, satellite, etc.)
    with metadata and common processing operations.

    Attributes
    ----------
    data : xr.Dataset | None
        The observation dataset.
    label : str
        Observation source identifier.
    obs_type : str
        Observation type (e.g., 'pt_sfc', 'aircraft', 'satellite').
    _geometry : DataGeometry
        Data geometry type.
    files : list[Path]
        List of observation files.
    file_pattern : str | None
        Glob pattern or filename.
    time_var : str | None
        Name of the time variable in the source data.
    data_proc : dict[str, Any]
        Data processing configuration.
    resample : str | None
        Resampling frequency (e.g., 'h', 'D').
    """

    obs_type: str = "pt_sfc"
    _geometry: DataGeometry = DataGeometry.POINT
    files: list[Path] = field(default_factory=list)
    file_pattern: str | None = None
    time_var: str | None = None
    data_proc: dict[str, Any] = field(default_factory=dict)
    resample: str | None = None

    @property
    def geometry(self) -> DataGeometry:
        """Return the data geometry type."""
        return self._geometry

    @geometry.setter
    def geometry(self, value: DataGeometry) -> None:
        """Set the data geometry type."""
        self._geometry = value

    def _copy_with_data(self, data: xr.Dataset) -> ObservationData:
        """Create a copy with new data."""
        return ObservationData(
            data=data,
            label=self.label,
            variables=self.variables.copy(),
            variable_mapping=dict(self.variable_mapping),
            obs_type=self.obs_type,
            _geometry=self._geometry,
            files=self.files.copy(),
            file_pattern=self.file_pattern,
            time_var=self.time_var,
            data_proc=self.data_proc.copy(),
            resample=self.resample,
        )

    def resolve_files(self, pattern: str | None = None) -> list[Path]:
        """Resolve file pattern to list of files.

        Parameters
        ----------
        pattern
            Glob pattern or filename. If None, uses self.file_pattern.

        Returns
        -------
        list[Path]
            Sorted list of matching file paths.

        Raises
        ------
        DataNotFoundError
            If no files match the pattern.
        """
        if pattern is None:
            pattern = self.file_pattern

        if pattern is None:
            return self.files

        # Single file
        if Path(pattern).exists() and Path(pattern).is_file():
            self.files = [Path(pattern)]
            return self.files

        # Glob pattern
        matched = glob(pattern)
        if not matched:
            raise DataNotFoundError(f"No files match pattern: {pattern}")

        sorted_files = sorted(matched)
        self.files = [Path(f) for f in sorted_files]
        return self.files

    @classmethod
    def geometry_from_obs_type(cls, obs_type: str) -> DataGeometry:
        """Determine geometry from observation type string.

        Parameters
        ----------
        obs_type
            Observation type string (e.g., 'pt_sfc', 'aircraft').

        Returns
        -------
        DataGeometry
            The corresponding geometry type.
        """
        obs_type_lower = obs_type.lower()

        # Point surface observations
        if obs_type_lower in ("pt_sfc", "surface", "ground", "airnow", "aeronet"):
            return DataGeometry.POINT

        # Aircraft/track observations
        if obs_type_lower in ("aircraft", "mobile", "ship", "track"):
            return DataGeometry.TRACK

        # Profile observations
        if obs_type_lower in ("sonde", "profile", "lidar", "ozonesonde"):
            return DataGeometry.PROFILE

        # Satellite swath observations
        if obs_type_lower in ("satellite", "swath", "l2", "sat_swath_clm"):
            return DataGeometry.SWATH

        # Lightning mapping array (gridded)
        if obs_type_lower in ("lma",):
            return DataGeometry.GRID

        # Gridded observations (including gridded satellite products)
        if obs_type_lower in ("gridded", "grid", "l3", "reanalysis", "sat_grid_clm"):
            return DataGeometry.GRID

        # Default to point
        return DataGeometry.POINT

    def apply_variable_config(self) -> None:
        """Apply variable configuration (scaling, masking, renaming).

        Processes all variables in self.variables dictionary.
        """
        if self.data is None:
            return

        for var_name, config in list(self.variables.items()):
            # Handle source_name: rename file variable to canonical name first
            source = config.get("source_name")
            if source and source in self.data and var_name not in self.data:
                self.rename_variable(source, var_name)

            if var_name not in self.data:
                continue

            # Apply unit scaling
            if "unit_scale" in config:
                scale = config["unit_scale"]
                method = config.get("unit_scale_method", "*")
                self.apply_unit_scale(var_name, scale, method)

            # Apply masking
            min_val = config.get("obs_min")
            max_val = config.get("obs_max")
            nan_val = config.get("nan_value")
            if any(v is not None for v in [min_val, max_val, nan_val]):
                self.apply_mask(var_name, min_val, max_val, nan_val)

            # Apply renaming (do this last)
            new_name = config.get("rename")
            if new_name:
                self.rename_variable(var_name, new_name)
                # Update the config dict key
                self.variables[new_name] = self.variables.pop(var_name)

    def remove_nans(self, variables: Sequence[str] | None = None) -> None:
        """Remove records with NaN values.

        Parameters
        ----------
        variables
            List of variables to check. If None, checks all data variables.
        """
        if self.data is None:
            return

        if variables is None:
            variables = list(self.data.data_vars)

        # Create mask for valid data
        import numpy as np

        valid_mask = xr.ones_like(self.data[variables[0]], dtype=bool)
        for var in variables:
            if var in self.data:
                valid_mask = valid_mask & ~self.data[var].isnull()

        # Apply mask
        self.data = self.data.where(valid_mask, drop=True)

    def resample_data(
        self,
        freq: str | None = None,
        min_count: int | None = None,
        track_count: bool = False,
    ) -> None:
        """Resample observation data to a different frequency.

        Used to average high-frequency observations (e.g., sub-hourly Pandora)
        to match model output resolution (e.g., hourly). Averaging is applied
        BEFORE pairing for efficiency.

        Parameters
        ----------
        freq
            Pandas frequency string (e.g., 'h', 'D', '30min').
            If None, uses self.resample.
        min_count
            Minimum number of observations required per average.
            Averages with fewer observations are set to NaN.
        track_count
            If True, add 'obs_count' variable tracking number of
            observations in each average.
        """
        if self.data is None:
            return
        if freq is None:
            freq = self.resample
        if freq is None:
            return
        self.data = resample_dataset(self.data, freq, min_count=min_count, track_count=track_count)

    def filter_by_flag(
        self,
        flag_var: str,
        valid_flags: Sequence[int] | None = None,
        invalid_flags: Sequence[int] | None = None,
    ) -> None:
        """Filter observations by quality flag.

        Parameters
        ----------
        flag_var
            Name of the flag variable.
        valid_flags
            List of flag values to keep.
        invalid_flags
            List of flag values to remove.
        """
        if self.data is None or flag_var not in self.data:
            return

        flags = self.data[flag_var]

        if valid_flags is not None:
            # Create mask using xarray isin
            mask = flags.isin(list(valid_flags))
            self.data = self.data.where(mask, drop=True)

        if invalid_flags is not None:
            # Create mask using xarray isin
            mask = ~flags.isin(list(invalid_flags))
            self.data = self.data.where(mask, drop=True)

    def filter_by_time(
        self,
        start: datetime | str | None = None,
        end: datetime | str | None = None,
    ) -> None:
        """Filter observations to a time range in-place.

        Parameters
        ----------
        start
            Start time (inclusive).
        end
            End time (inclusive).
        """
        if self.data is None or "time" not in self.data.dims:
            return

        if start is not None:
            self.data = self.data.sel(time=slice(start, None))
        if end is not None:
            self.data = self.data.sel(time=slice(None, end))

    def filter_by_bbox(
        self,
        lon_min: float,
        lon_max: float,
        lat_min: float,
        lat_max: float,
    ) -> None:
        """Filter observations to a geographic bounding box.

        Parameters
        ----------
        lon_min, lon_max
            Longitude bounds.
        lat_min, lat_max
            Latitude bounds.
        """
        if self.data is None:
            return

        # Find latitude and longitude variables
        lat_var = None
        lon_var = None
        for name in ["lat", "latitude", "LAT", "LATITUDE"]:
            if name in self.data.coords or name in self.data.data_vars:
                lat_var = name
                break
        for name in ["lon", "longitude", "LON", "LONGITUDE"]:
            if name in self.data.coords or name in self.data.data_vars:
                lon_var = name
                break

        if lat_var is None or lon_var is None:
            return

        lat = self.data[lat_var]
        lon = self.data[lon_var]

        mask = (lat >= lat_min) & (lat <= lat_max) & (lon >= lon_min) & (lon <= lon_max)

        self.data = self.data.where(mask, drop=True)

    def add_site_coordinates(
        self,
        lat: float,
        lon: float,
        alt: float | None = None,
    ) -> None:
        """Add site coordinates to a ground observation.

        Parameters
        ----------
        lat
            Latitude in degrees.
        lon
            Longitude in degrees.
        alt
            Altitude in meters (optional).
        """
        if self.data is None:
            return

        self.data = self.data.assign_coords(
            latitude=lat,
            longitude=lon,
        )
        if alt is not None:
            self.data = self.data.assign_coords(altitude=alt)

    def sum_variables(
        self,
        new_var: str,
        source_vars: Sequence[str],
        config: dict[str, Any] | None = None,
    ) -> None:
        """Create a new variable by summing existing variables.

        Parameters
        ----------
        new_var
            Name for the new summed variable.
        source_vars
            List of variable names to sum.
        config
            Optional configuration for the new variable.
        """
        if self.data is None:
            return

        # Check source variables exist
        missing = [v for v in source_vars if v not in self.data]
        if missing:
            raise DataValidationError(f"Cannot sum variables, missing: {missing}")

        if new_var in self.data:
            raise DataValidationError(f"Variable '{new_var}' already exists")

        # Sum the variables
        result = self.data[source_vars[0]].copy()
        for var in source_vars[1:]:
            result = result + self.data[var]

        self.data[new_var] = result

        if config is not None:
            self.variables[new_var] = config

    @property
    def n_sites(self) -> int:
        """Number of observation sites (for point geometry)."""
        if self.data is None:
            return 0
        if self._geometry != DataGeometry.POINT:
            return 0
        for dim in ["site", "x", "station"]:
            if dim in self.data.dims:
                return int(self.data.sizes[dim])
        return 0

    @property
    def n_times(self) -> int:
        """Number of time steps."""
        if self.data is None or "time" not in self.data.dims:
            return 0
        return int(self.data.sizes["time"])

    def to_point_dataframe(self) -> Any:
        """Convert point observations to DataFrame with site info.

        Returns
        -------
        pd.DataFrame
            DataFrame with time, site, lat, lon, and data variables.
        """
        import pandas as pd

        if self.data is None:
            return pd.DataFrame()

        df: pd.DataFrame = self.data.to_dataframe().reset_index()
        return df


class _GenericNetCDFReader:
    """Generic NetCDF source reader keyed only by geometry.

    Base for the generic readers (``pt_sfc``, ``aircraft``, ``ozonesonde``-style
    profiles, gridded) used by unified ``sources:`` configs and by auto-converted
    legacy ``model:``/``obs:`` controls. Opens plain NetCDF with no
    format-specific handling and reports a fixed :class:`DataGeometry`, mirroring
    the role the legacy loader stage filled via
    :meth:`ObservationData.geometry_from_obs_type`.
    """

    _name: str = "generic_obs"
    _geometry: DataGeometry = DataGeometry.POINT

    @property
    def name(self) -> str:
        """Return reader name."""
        return self._name

    @property
    def geometry(self) -> DataGeometry:
        """Geometry of data this reader returns."""
        return self._geometry

    def open(
        self,
        file_paths: Sequence[str | Path],
        variables: Sequence[str] | None = None,
        time_range: tuple[Any, Any] | None = None,
        **kwargs: Any,
    ) -> xr.Dataset:
        """Open one or more generic NetCDF files."""
        files = [str(Path(p).expanduser()) for p in file_paths]
        if not files:
            raise DataNotFoundError(f"No {self._name} observation files provided")
        try:
            if len(files) == 1:
                ds = xr.open_dataset(files[0])
            else:
                ds = xr.open_mfdataset(files, combine="by_coords", parallel=True)
        except OSError as e:
            raise DataFormatError(f"Failed to open {self._name} files: {e}") from e

        if variables:
            keep_vars = [v for v in variables if v in ds.data_vars]
            ds = ds[keep_vars]

        if time_range and "time" in ds:
            start, end = time_range
            ds = ds.sel(time=slice(start, end))

        ds.attrs["geometry"] = self.geometry.name.lower()
        return ds

    def get_variable_mapping(self) -> Mapping[str, str]:
        """Generic files use their native variable names."""
        return {}


@source_registry.register("pt_sfc")
class PointSurfaceReader(_GenericNetCDFReader):
    """Generic point-surface (POINT) reader for unified ``sources:`` configs."""

    _name = "pt_sfc"
    _geometry = DataGeometry.POINT


@source_registry.register("aircraft")
class AircraftReader(_GenericNetCDFReader):
    """Generic aircraft/track (TRACK) NetCDF reader.

    Handles plain-NetCDF track files (the legacy ``obs_type: aircraft`` /
    ``mobile`` / ``ship`` / ``track`` path). For ICARTT ``.ict`` campaign files
    use the dedicated ``icartt`` reader instead.
    """

    _name = "aircraft"
    _geometry = DataGeometry.TRACK


@source_registry.register("profile")
class ProfileReader(_GenericNetCDFReader):
    """Generic vertical-profile (PROFILE) NetCDF reader.

    Handles plain-NetCDF profile files (the legacy ``obs_type: profile`` /
    ``sonde`` path). For ozonesonde campaign files use the dedicated
    ``ozonesonde`` reader instead.
    """

    _name = "profile"
    _geometry = DataGeometry.PROFILE


@source_registry.register("gridded")
class GriddedObsReader(_GenericNetCDFReader):
    """Generic gridded (GRID) NetCDF observation reader.

    Handles plain-NetCDF gridded obs files (the legacy ``obs_type: gridded`` /
    ``grid`` / ``reanalysis`` / ``sat_grid_clm`` non-satellite path). For gridded
    satellite L3 products use the dedicated ``satellite_l3`` reader.
    """

    _name = "gridded"
    _geometry = DataGeometry.GRID


def create_observation_data(
    label: str,
    obs_type: str = "pt_sfc",
    data: xr.Dataset | None = None,
    filename: str | Path | None = None,
    variables: dict[str, Any] | None = None,
    time_var: str | None = None,
    resample: str | None = None,
    data_proc: dict[str, Any] | None = None,
    **kwargs: Any,
) -> ObservationData:
    """Factory function to create ObservationData instance.

    Parameters
    ----------
    label
        Observation source identifier.
    obs_type
        Observation type (e.g., 'pt_sfc', 'aircraft').
    data
        Pre-loaded observation dataset.
    filename
        File path or glob pattern.
    variables
        Variable configuration.
    time_var
        Name of time variable in source data.
    resample
        Resampling frequency.
    data_proc
        Data processing configuration.
    **kwargs
        Additional options.

    Returns
    -------
    ObservationData
        Configured ObservationData instance.
    """
    geometry = ObservationData.geometry_from_obs_type(obs_type)

    obs = ObservationData(
        label=label,
        data=data,
        obs_type=obs_type,
        _geometry=geometry,
        variables=variables or {},
        time_var=time_var,
        resample=resample,
        data_proc=data_proc or {},
    )

    if filename is not None:
        obs.file_pattern = str(filename)
        try:
            obs.resolve_files()
        except DataNotFoundError:
            pass  # Files may be resolved later

    return obs


# Geometry-specific convenience classes


@dataclass
class PointObservation(ObservationData):
    """Observation data with POINT geometry.

    Represents fixed-location observations like surface stations.
    """

    obs_type: str = "pt_sfc"
    _geometry: DataGeometry = DataGeometry.POINT


@dataclass
class TrackObservation(ObservationData):
    """Observation data with TRACK geometry.

    Represents mobile observations like aircraft tracks.
    """

    obs_type: str = "aircraft"
    _geometry: DataGeometry = DataGeometry.TRACK


@dataclass
class ProfileObservation(ObservationData):
    """Observation data with PROFILE geometry.

    Represents vertical profile observations like sondes.
    """

    obs_type: str = "sonde"
    _geometry: DataGeometry = DataGeometry.PROFILE


@dataclass
class SwathObservation(ObservationData):
    """Observation data with SWATH geometry.

    Represents satellite swath observations (L2 products).
    """

    obs_type: str = "satellite"
    _geometry: DataGeometry = DataGeometry.SWATH


@dataclass
class GriddedObservation(ObservationData):
    """Observation data with GRID geometry.

    Represents gridded observations like L3 satellite products.
    """

    obs_type: str = "gridded"
    _geometry: DataGeometry = DataGeometry.GRID
