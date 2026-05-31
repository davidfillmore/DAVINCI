"""Base data classes for DAVINCI.

This module provides the core data container classes that wrap xarray Datasets
with metadata and common operations.

Key Classes:
    - PairedData: Container for paired model-observation data
    - DataContainer: Base class for model and observation containers
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import xarray as xr

from davinci_monet.core.exceptions import DataValidationError, VariableNotFoundError
from davinci_monet.core.protocols import DataGeometry
from davinci_monet.core.types import BoundingBox, PathLike, TimeRange, VariableMapping


@dataclass
class DataContainer(ABC):
    """Abstract base class for data containers.

    Provides common interface for model and observation data containers.
    Wraps xarray Dataset with additional metadata and operations.

    Attributes
    ----------
    data : xr.Dataset | None
        The underlying xarray Dataset.
    label : str
        Identifier for this data source.
    variables : dict[str, Any]
        Variable configuration dictionary.
    variable_mapping : VariableMapping
        Mapping from standard to source-specific variable names.
    """

    data: xr.Dataset | None = None
    label: str = ""
    variables: dict[str, Any] = field(default_factory=dict)
    variable_mapping: VariableMapping = field(default_factory=dict)

    @property
    @abstractmethod
    def geometry(self) -> DataGeometry:
        """Return the data geometry type."""
        ...

    @property
    def is_loaded(self) -> bool:
        """Check if data has been loaded."""
        return self.data is not None

    def get_variable(self, name: str) -> xr.DataArray:
        """Get a variable from the dataset.

        Parameters
        ----------
        name
            Variable name (can be standard or source-specific).

        Returns
        -------
        xr.DataArray
            The requested variable.

        Raises
        ------
        VariableNotFoundError
            If variable not found in dataset.
        """
        if self.data is None:
            raise VariableNotFoundError(f"No data loaded for {self.label}")

        # Try direct name first
        if name in self.data:
            result: xr.DataArray = self.data[name]
            return result

        # Try mapping
        if name in self.variable_mapping:
            mapped_name = self.variable_mapping[name]
            if mapped_name in self.data:
                result = self.data[mapped_name]
                return result

        # Try reverse mapping
        for std_name, source_name in self.variable_mapping.items():
            if source_name == name and std_name in self.data:
                result = self.data[std_name]
                return result

        raise VariableNotFoundError(
            f"Variable '{name}' not found in {self.label}. "
            f"Available: {list(self.data.data_vars)}"
        )

    def has_variable(self, name: str) -> bool:
        """Check if a variable exists in the dataset.

        Parameters
        ----------
        name
            Variable name (can be standard or source-specific).

        Returns
        -------
        bool
            True if variable exists.
        """
        try:
            self.get_variable(name)
            return True
        except VariableNotFoundError:
            return False

    @property
    def available_variables(self) -> list[str]:
        """List of available data variables."""
        if self.data is None:
            return []
        return list(self.data.data_vars)  # type: ignore[arg-type]

    @property
    def time_range(self) -> TimeRange | None:
        """Get the time range of the data."""
        if self.data is None or "time" not in self.data.dims:
            return None
        times = self.data["time"].values
        if len(times) == 0:
            return None
        # Convert to datetime if needed
        start = times[0]
        end = times[-1]
        return (start, end)

    @property
    def bounds(self) -> BoundingBox | None:
        """Get the geographic bounding box."""
        if self.data is None:
            return None

        # Try common coordinate names
        lat_names = ["lat", "latitude", "LAT", "LATITUDE", "y"]
        lon_names = ["lon", "longitude", "LON", "LONGITUDE", "x"]

        lat_coord = None
        lon_coord = None

        for name in lat_names:
            if name in self.data.coords:
                lat_coord = self.data.coords[name]
                break
            if name in self.data.data_vars:
                lat_coord = self.data[name]
                break

        for name in lon_names:
            if name in self.data.coords:
                lon_coord = self.data.coords[name]
                break
            if name in self.data.data_vars:
                lon_coord = self.data[name]
                break

        if lat_coord is None or lon_coord is None:
            return None

        lat_min = float(lat_coord.min())
        lat_max = float(lat_coord.max())
        lon_min = float(lon_coord.min())
        lon_max = float(lon_coord.max())

        return (lon_min, lon_max, lat_min, lat_max)

    def subset_time(
        self,
        start: datetime | str | None = None,
        end: datetime | str | None = None,
    ) -> DataContainer:
        """Subset data to a time range.

        Parameters
        ----------
        start
            Start time (inclusive).
        end
            End time (inclusive).

        Returns
        -------
        DataContainer
            New container with subsetted data.
        """
        if self.data is None or "time" not in self.data.dims:
            return self

        subset = self.data
        if start is not None:
            subset = subset.sel(time=slice(start, None))
        if end is not None:
            subset = subset.sel(time=slice(None, end))

        # Create new instance with subsetted data
        return self._copy_with_data(subset)

    def subset_bbox(self, bbox: BoundingBox) -> DataContainer:
        """Subset data to a geographic bounding box.

        Parameters
        ----------
        bbox
            Bounding box as (lon_min, lon_max, lat_min, lat_max).

        Returns
        -------
        DataContainer
            New container with subsetted data.
        """
        if self.data is None:
            return self

        lon_min, lon_max, lat_min, lat_max = bbox

        # Try common coordinate names
        lat_name = None
        lon_name = None
        for name in ["lat", "latitude"]:
            if name in self.data.dims or name in self.data.coords:
                lat_name = name
                break
        for name in ["lon", "longitude"]:
            if name in self.data.dims or name in self.data.coords:
                lon_name = name
                break

        if lat_name is None or lon_name is None:
            return self

        lat_coord = self.data[lat_name]
        lon_coord = self.data[lon_name]

        # Apply selection
        if (
            lat_name in self.data.dims
            and lon_name in self.data.dims
            and lat_coord.ndim == 1
            and lon_coord.ndim == 1
        ):
            lat_slice = slice(lat_min, lat_max)
            lon_slice = slice(lon_min, lon_max)

            if lat_coord.values[0] > lat_coord.values[-1]:
                lat_slice = slice(lat_max, lat_min)
            if lon_coord.values[0] > lon_coord.values[-1]:
                lon_slice = slice(lon_max, lon_min)

            subset = self.data.sel(
                {
                    lat_name: lat_slice,
                    lon_name: lon_slice,
                }
            )
        else:
            # Curvilinear or non-dimension coords: use mask-based selection
            mask = (
                (lat_coord >= lat_min)
                & (lat_coord <= lat_max)
                & (lon_coord >= lon_min)
                & (lon_coord <= lon_max)
            )
            subset = self.data.where(mask, drop=True)

        return self._copy_with_data(subset)

    @abstractmethod
    def _copy_with_data(self, data: xr.Dataset) -> DataContainer:
        """Create a copy of this container with new data."""
        ...

    def apply_unit_scale(
        self,
        variable: str,
        scale: float,
        method: str = "*",
    ) -> None:
        """Apply unit scaling to a variable in-place.

        Parameters
        ----------
        variable
            Variable name to scale.
        scale
            Scale factor.
        method
            Scaling method: '*', '/', '+', or '-'.
        """
        if self.data is None:
            return

        if variable not in self.data:
            return

        if method == "*":
            self.data[variable] = self.data[variable] * scale
        elif method == "/":
            self.data[variable] = self.data[variable] / scale
        elif method == "+":
            self.data[variable] = self.data[variable] + scale
        elif method == "-":
            self.data[variable] = self.data[variable] - scale

    def rename_variable(self, old_name: str, new_name: str) -> None:
        """Rename a variable in-place.

        Parameters
        ----------
        old_name
            Current variable name.
        new_name
            New variable name.
        """
        if self.data is None or old_name not in self.data:
            return

        self.data = self.data.rename({old_name: new_name})

    def apply_mask(
        self,
        variable: str,
        min_val: float | None = None,
        max_val: float | None = None,
        nan_value: float | None = None,
    ) -> None:
        """Apply masking to a variable in-place.

        Parameters
        ----------
        variable
            Variable name to mask.
        min_val
            Minimum valid value.
        max_val
            Maximum valid value.
        nan_value
            Value to treat as NaN.
        """
        if self.data is None or variable not in self.data:
            return

        import numpy as np

        arr = self.data[variable]

        if nan_value is not None:
            arr = arr.where(arr != nan_value)

        if min_val is not None:
            arr = arr.where(arr >= min_val)

        if max_val is not None:
            arr = arr.where(arr <= max_val)

        self.data[variable] = arr


@dataclass
class PairedData:
    """Container for paired model-observation data.

    Holds the result of pairing model output with observations,
    including metadata about the pairing process.

    Attributes
    ----------
    data : xr.Dataset
        The paired dataset with both model and observation variables.
    model_label : str
        Label of the model source.
    obs_label : str
        Label of the observation source.
    geometry : DataGeometry
        The geometry type of the observation data.
    pairing_info : dict[str, Any]
        Metadata about the pairing process.
    """

    data: xr.Dataset
    model_label: str
    obs_label: str
    geometry: DataGeometry
    pairing_info: dict[str, Any] = field(default_factory=dict)

    @property
    def pair_label(self) -> str:
        """Get combined pair label (obs_model format)."""
        return f"{self.obs_label}_{self.model_label}"

    @property
    def model_variables(self) -> list[str]:
        """List of model variables in the paired data."""
        return [str(v) for v in self.data.data_vars if str(v).startswith("model_")]

    @property
    def obs_variables(self) -> list[str]:
        """List of observation variables in the paired data."""
        return [str(v) for v in self.data.data_vars if str(v).startswith("obs_")]

    @property
    def paired_variable_names(self) -> list[tuple[str, str]]:
        """List of (obs_var, model_var) pairs."""
        pairs: list[tuple[str, str]] = []
        for obs_var in self.obs_variables:
            base_name = obs_var.replace("obs_", "")
            model_var = f"model_{base_name}"
            if model_var in self.model_variables:
                pairs.append((obs_var, model_var))
        return pairs

    def get_obs(self, variable: str) -> xr.DataArray:
        """Get observation variable.

        Parameters
        ----------
        variable
            Variable name (with or without 'obs_' prefix).

        Returns
        -------
        xr.DataArray
            Observation data.
        """
        if not variable.startswith("obs_"):
            variable = f"obs_{variable}"
        if variable not in self.data:
            raise VariableNotFoundError(
                f"Observation variable '{variable}' not found. " f"Available: {self.obs_variables}"
            )
        result: xr.DataArray = self.data[variable]
        return result

    def get_model(self, variable: str) -> xr.DataArray:
        """Get model variable.

        Parameters
        ----------
        variable
            Variable name (with or without 'model_' prefix).

        Returns
        -------
        xr.DataArray
            Model data.
        """
        if not variable.startswith("model_"):
            variable = f"model_{variable}"
        if variable not in self.data:
            raise VariableNotFoundError(
                f"Model variable '{variable}' not found. " f"Available: {self.model_variables}"
            )
        result: xr.DataArray = self.data[variable]
        return result

    def get_pair(self, variable: str) -> tuple[xr.DataArray, xr.DataArray]:
        """Get both observation and model arrays for a variable.

        Parameters
        ----------
        variable
            Base variable name (without prefix).

        Returns
        -------
        tuple[xr.DataArray, xr.DataArray]
            (obs_array, model_array) tuple.
        """
        obs = self.get_obs(variable)
        model = self.get_model(variable)
        return (obs, model)

    @property
    def time_range(self) -> TimeRange | None:
        """Get the time range of paired data."""
        if "time" not in self.data.dims:
            return None
        times = self.data["time"].values
        if len(times) == 0:
            return None
        return (times[0], times[-1])

    @property
    def n_points(self) -> int:
        """Get total number of paired data points."""
        # Count non-NaN pairs for first paired variable
        if not self.paired_variable_names:
            return 0
        obs_var, model_var = self.paired_variable_names[0]
        obs_data = self.data[obs_var]
        model_data = self.data[model_var]
        valid = ~obs_data.isnull() & ~model_data.isnull()
        return int(valid.sum().values)

    def to_dataframe(self) -> Any:
        """Convert paired data to pandas DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame with time/location index and paired variables.
        """
        import pandas as pd

        df: pd.DataFrame = self.data.to_dataframe().reset_index()
        return df

    def subset_time(
        self,
        start: datetime | str | None = None,
        end: datetime | str | None = None,
    ) -> PairedData:
        """Subset paired data to a time range.

        Parameters
        ----------
        start
            Start time (inclusive).
        end
            End time (inclusive).

        Returns
        -------
        PairedData
            New PairedData with subsetted data.
        """
        if "time" not in self.data.dims:
            return self

        subset = self.data
        if start is not None:
            subset = subset.sel(time=slice(start, None))
        if end is not None:
            subset = subset.sel(time=slice(None, end))

        return PairedData(
            data=subset,
            model_label=self.model_label,
            obs_label=self.obs_label,
            geometry=self.geometry,
            pairing_info=self.pairing_info,
        )


def validate_dataset_geometry(
    data: xr.Dataset,
    expected_geometry: DataGeometry,
) -> None:
    """Validate that a dataset has the expected geometry.

    Parameters
    ----------
    data
        Dataset to validate.
    expected_geometry
        Expected geometry type.

    Raises
    ------
    DataValidationError
        If dataset doesn't match expected geometry.
    """
    # Check geometry attribute if present
    if "geometry" in data.attrs:
        actual = data.attrs["geometry"]
        if isinstance(actual, DataGeometry):
            if actual != expected_geometry:
                raise DataValidationError(
                    f"Expected geometry {expected_geometry.name}, got {actual.name}"
                )
            return
        if isinstance(actual, str):
            try:
                actual_geom = DataGeometry[actual.upper()]
                if actual_geom != expected_geometry:
                    raise DataValidationError(
                        f"Expected geometry {expected_geometry.name}, got {actual}"
                    )
                return
            except KeyError:
                pass

    # Validate based on expected dimensions
    dims = set(data.dims)

    if expected_geometry == DataGeometry.POINT:
        # Point: (time, site) or (time, x)
        if not (("time" in dims and ("site" in dims or "x" in dims)) or "time" in dims):
            raise DataValidationError(f"POINT geometry expects dims (time, site), got {dims}")

    elif expected_geometry == DataGeometry.TRACK:
        # Track: (time,) with lat/lon/alt coords
        if "time" not in dims:
            raise DataValidationError(f"TRACK geometry expects 'time' dimension, got {dims}")

    elif expected_geometry == DataGeometry.PROFILE:
        # Profile: (time, level) with lat/lon coords
        if not ("time" in dims and ("level" in dims or "z" in dims)):
            raise DataValidationError(f"PROFILE geometry expects dims (time, level), got {dims}")

    elif expected_geometry == DataGeometry.SWATH:
        # Swath: (time, scanline, pixel) or similar
        if "time" not in dims:
            raise DataValidationError(f"SWATH geometry expects 'time' dimension, got {dims}")

    elif expected_geometry == DataGeometry.GRID:
        # Grid: (time, lat, lon) or (time, y, x)
        has_spatial = ("lat" in dims or "latitude" in dims or "y" in dims) and (
            "lon" in dims or "longitude" in dims or "x" in dims
        )
        if not has_spatial:
            raise DataValidationError(f"GRID geometry expects spatial dims (lat, lon), got {dims}")


def create_paired_dataset(
    obs_data: xr.Dataset,
    model_data: xr.Dataset,
    obs_vars: Sequence[str],
    model_vars: Sequence[str],
    prefix_obs: str = "obs_",
    prefix_model: str = "model_",
) -> xr.Dataset:
    """Create a paired dataset from observation and model data.

    Parameters
    ----------
    obs_data
        Observation dataset.
    model_data
        Model dataset (already interpolated to observation locations).
    obs_vars
        List of observation variable names.
    model_vars
        List of model variable names (same order as obs_vars).
    prefix_obs
        Prefix for observation variables in output.
    prefix_model
        Prefix for model variables in output.

    Returns
    -------
    xr.Dataset
        Combined dataset with prefixed variable names.
    """
    if len(obs_vars) != len(model_vars):
        raise DataValidationError(
            f"obs_vars ({len(obs_vars)}) and model_vars ({len(model_vars)}) "
            "must have the same length"
        )

    # Start with coordinates from obs_data
    coords = dict(obs_data.coords)

    # Create data variables dict
    data_vars: dict[str, xr.DataArray] = {}

    for obs_var, model_var in zip(obs_vars, model_vars):
        if obs_var in obs_data:
            data_vars[f"{prefix_obs}{obs_var}"] = obs_data[obs_var]
        if model_var in model_data:
            # Use obs_var name for consistency
            data_vars[f"{prefix_model}{obs_var}"] = model_data[model_var]

    # Combine into new dataset
    result = xr.Dataset(data_vars, coords=coords)

    # Copy relevant attributes
    result.attrs = {
        "created_by": "davinci_monet",
        "paired": True,
    }

    return result
