"""Type aliases for DAVINCI.

This module provides commonly used type aliases to improve code readability
and maintainability. These aliases are used throughout the codebase for
consistent typing.

Usage:
    from davinci_monet.core.types import PathLike, TimeRange, VariableMapping
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timedelta
from os import PathLike as OSPathLike
from pathlib import Path
from typing import Any, TypeAlias, TypeVar, Union

import numpy as np
import numpy.typing as npt

# =============================================================================
# Path Types
# =============================================================================

PathLike: TypeAlias = Union[str, Path, OSPathLike[str]]
"""Type for file system paths - accepts str, Path, or os.PathLike."""

PathSequence: TypeAlias = Sequence[PathLike]
"""Sequence of file paths."""

# =============================================================================
# Time Types
# =============================================================================

# Note: pandas and numpy datetime types are included for interoperability
Timestamp: TypeAlias = Union[datetime, str, np.datetime64]
"""A timestamp that can be datetime, ISO string, or numpy datetime64."""

TimeDelta: TypeAlias = Union[timedelta, str, np.timedelta64]
"""A time duration that can be timedelta, string (e.g., '1h'), or numpy timedelta64."""

TimeRange: TypeAlias = tuple[Timestamp, Timestamp]
"""A time range as (start, end) tuple."""

# =============================================================================
# Numeric Types
# =============================================================================

Number: TypeAlias = Union[int, float, np.integer[Any], np.floating[Any]]
"""Any numeric type (int, float, or numpy numeric)."""

ArrayLike: TypeAlias = Union[
    Sequence[Number],
    npt.NDArray[np.floating[Any]],
    npt.NDArray[np.integer[Any]],
]
"""Array-like data that can be converted to numpy array."""

FloatArray: TypeAlias = npt.NDArray[np.floating[Any]]
"""Numpy array of floating point values."""

IntArray: TypeAlias = npt.NDArray[np.integer[Any]]
"""Numpy array of integer values."""

BoolArray: TypeAlias = npt.NDArray[np.bool_]
"""Numpy array of boolean values."""

# =============================================================================
# Coordinate Types
# =============================================================================

Longitude: TypeAlias = float
"""Longitude value in degrees (-180 to 180 or 0 to 360)."""

Latitude: TypeAlias = float
"""Latitude value in degrees (-90 to 90)."""

Coordinate: TypeAlias = tuple[Longitude, Latitude]
"""A (longitude, latitude) coordinate pair."""

BoundingBox: TypeAlias = tuple[Longitude, Longitude, Latitude, Latitude]
"""Geographic bounding box as (lon_min, lon_max, lat_min, lat_max)."""

Altitude: TypeAlias = float
"""Altitude/elevation value (typically in meters)."""

Coordinate3D: TypeAlias = tuple[Longitude, Latitude, Altitude]
"""A 3D coordinate as (longitude, latitude, altitude)."""

# =============================================================================
# Variable/Data Types
# =============================================================================

VariableName: TypeAlias = str
"""Name of a data variable."""

VariableMapping: TypeAlias = Mapping[str, str]
"""Mapping from standard variable names to dataset-specific names."""

VariableDict: TypeAlias = dict[str, dict[str, Any]]
"""Variable configuration dictionary with nested properties."""

DimensionName: TypeAlias = str
"""Name of a dataset dimension (e.g., 'time', 'lat', 'lon')."""

Dimensions: TypeAlias = Sequence[DimensionName]
"""Sequence of dimension names."""

AttributeDict: TypeAlias = dict[str, Any]
"""Dictionary of dataset/variable attributes."""

# =============================================================================
# Configuration Types
# =============================================================================

ConfigDict: TypeAlias = dict[str, Any]
"""Generic configuration dictionary (typically from YAML)."""

NestedConfigDict: TypeAlias = dict[str, Union[Any, "NestedConfigDict"]]
"""Nested configuration dictionary."""

# =============================================================================
# Callable Types
# =============================================================================

T = TypeVar("T")
R = TypeVar("R")

Transformer: TypeAlias = Callable[[T], R]
"""A function that transforms input of type T to output of type R."""

Predicate: TypeAlias = Callable[[T], bool]
"""A function that returns True/False for a given input."""

DataTransformer: TypeAlias = Callable[[Any], Any]
"""A function that transforms data (xr.Dataset, xr.DataArray, etc.)."""

# =============================================================================
# Statistics Types
# =============================================================================

MetricName: TypeAlias = str
"""Name of a statistical metric (e.g., 'MB', 'RMSE', 'R2')."""

MetricValue: TypeAlias = float
"""Value of a computed statistic."""

MetricDict: TypeAlias = dict[MetricName, MetricValue]
"""Dictionary of metric names to computed values."""

GroupBy: TypeAlias = Union[str, Sequence[str]]
"""Grouping specification - single dimension or multiple dimensions."""

# =============================================================================
# Plotting Types
# =============================================================================

Color: TypeAlias = Union[str, tuple[float, float, float], tuple[float, float, float, float]]
"""Color specification - name, RGB tuple, or RGBA tuple."""

ColorMap: TypeAlias = str
"""Name of a matplotlib colormap."""

PlotKwargs: TypeAlias = dict[str, Any]
"""Keyword arguments for plotting functions."""

# =============================================================================
# I/O Types
# =============================================================================

FileFormat: TypeAlias = str
"""File format identifier (e.g., 'netcdf', 'csv', 'icartt')."""

Encoding: TypeAlias = dict[str, dict[str, Any]]
"""NetCDF encoding specification for xarray."""

CompressionLevel: TypeAlias = int
"""Compression level (typically 0-9)."""
