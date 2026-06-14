"""Protocol definitions for DAVINCI components.

This module defines the interfaces (Protocols) that all pluggable components
must implement. Using Protocols enables static type checking while maintaining
flexibility for plugin architectures.

Data Geometry Types:
    - point: Fixed locations (surface stations, ground sites)
    - track: 3D trajectories (aircraft, mobile platforms)
    - profile: Vertical profiles (sondes, lidar)
    - swath: 2D satellite footprints (L2 products)
    - grid: Regular gridded data (L3 products, reanalysis)
"""

from __future__ import annotations

from abc import abstractmethod
from enum import Enum, auto
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Protocol, Sequence, runtime_checkable

if TYPE_CHECKING:
    import matplotlib.figure
    import pandas as pd
    import xarray as xr


class DataGeometry(Enum):
    """Enumeration of supported data geometry types.

    The geometry type determines which pairing strategy will be used
    to match datasets with dataset output.
    """

    POINT = auto()
    """Fixed point locations (time, site) - surface stations, ground sites."""

    TRACK = auto()
    """3D trajectory (time,) with lat/lon/alt coords - aircraft, mobile platforms."""

    PROFILE = auto()
    """Vertical profile (time, level) with lat/lon coords - sondes, lidar."""

    SWATH = auto()
    """2D satellite swath (time, scanline, pixel) - L2 products."""

    GRID = auto()
    """Regular grid (time, lat, lon) - L3 products, reanalysis, dataset output."""


# =============================================================================
# Unified Source Protocols
# =============================================================================
#
# A data source is just data of a given geometry (point, track, profile,
# swath, grid). Datasets and datasets are both data sources; the only thing
# that distinguishes them is topology, not origin. A dataset/geometry metadata may
# travel as metadata for labeling and styling, but it never appears in these
# contracts.


@runtime_checkable
class SourceReader(Protocol):
    """Protocol for data source readers (datasets and datasets alike).

    Every source reader declares the geometry it produces and loads files into
    a standardized xarray Dataset whose ``attrs['geometry']`` is set. This is
    the unified reader interface for all source geometries.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this source type (e.g. 'cesm_fv', 'pt_sfc')."""
        ...

    @property
    @abstractmethod
    def geometry(self) -> DataGeometry:
        """The data geometry this reader produces."""
        ...

    @abstractmethod
    def open(
        self,
        file_paths: Sequence[str | Path],
        variables: Sequence[str] | None = None,
        time_range: tuple[Any, Any] | None = None,
        **kwargs: Any,
    ) -> xr.Dataset:
        """Open source files and return a standardized Dataset.

        Parameters
        ----------
        file_paths
            Paths to source files (can include glob patterns).
        variables
            Optional list of variables to load. If None, load all.
        time_range
            Optional (start, end) time range to subset.
        **kwargs
            Additional reader-specific options.

        Returns
        -------
        xr.Dataset
            Source data with geometry-appropriate dimensions and the
            ``geometry`` attribute set.
        """
        ...


@runtime_checkable
class SourceProcessor(Protocol):
    """Protocol for data source post-processing operations.

    Processors handle
    unit conversion, vertical-coordinate handling, resampling, QA/QC,
    subsetting, and aggregation, composed into one chain regardless of origin.
    """

    @abstractmethod
    def process(self, dataset: xr.Dataset, **kwargs: Any) -> xr.Dataset:
        """Apply processing to a source Dataset and return the result."""
        ...


# =============================================================================
# Pairing Protocols
# =============================================================================


@runtime_checkable
class PairingStrategy(Protocol):
    """Protocol for source-pairing strategies.

    ``pair_sources`` is the canonical  API. Concrete strategy
    classes may keep an internal ``pair(dataset, geometry, ...)`` method that
    ``pair_sources`` delegates to, but it is not part of this public contract.
    """

    @property
    @abstractmethod
    def geometry(self) -> DataGeometry:
        """The data geometry this strategy handles."""
        ...

    @abstractmethod
    def pair_sources(
        self,
        x_data: xr.Dataset,
        y_data: xr.Dataset,
        **kwargs: Any,
    ) -> xr.Dataset:
        """Pair two  sources.

        The dataset is sampled onto the geometry geometry.
        """
        ...


@runtime_checkable
class PairingEngine(Protocol):
    """Protocol for the main pairing orchestrator.

    The pairing engine selects the appropriate strategy based on
    geometry geometry and coordinates the pairing process.
    """

    @abstractmethod
    def register_strategy(self, strategy: PairingStrategy) -> None:
        """Register a pairing strategy for a geometry type."""
        ...

    @abstractmethod
    def pair_sources(
        self,
        x_data: xr.Dataset,
        y_data: xr.Dataset,
        **kwargs: Any,
    ) -> xr.Dataset:
        """Pair two  sources using the appropriate strategy.

        The strategy is selected based on the geometry Dataset's geometry.
        """
        ...


# =============================================================================
# Plotting Protocols
# =============================================================================


@runtime_checkable
class Plotter(Protocol):
    """Protocol for plot renderers.

    Each plotter handles a specific plot type (timeseries, scatter,
    spatial, Taylor diagram, etc.).
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this plot type."""
        ...

    @abstractmethod
    def plot(
        self,
        paired_data: xr.Dataset,
        x_var: str,
        y_var: str,
        **kwargs: Any,
    ) -> matplotlib.figure.Figure:
        """Generate a plot from paired data.

        Parameters
        ----------
        paired_data
            Paired Dataset containing x and y variables.
        x_var
            Name of the x variable to plot.
        y_var
            Name of the y variable to plot.
        **kwargs
            Plot-specific options (colors, labels, domains, etc.).

        Returns
        -------
        matplotlib.figure.Figure
            The generated figure.
        """
        ...

    @abstractmethod
    def save(
        self,
        fig: matplotlib.figure.Figure,
        output_path: str | Path,
        **kwargs: Any,
    ) -> Path:
        """Save figure to file.

        Parameters
        ----------
        fig
            Figure to save.
        output_path
            Output file path.
        **kwargs
            Save options (dpi, format, etc.).

        Returns
        -------
        Path
            Path to saved file.
        """
        ...


@runtime_checkable
class SpatialPlotter(Protocol):
    """Protocol for spatial/map-based plotters.

    Extends base plotter with geospatial capabilities.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this plot type."""
        ...

    @abstractmethod
    def plot(
        self,
        paired_data: xr.Dataset,
        x_var: str,
        y_var: str,
        domain: tuple[float, float, float, float] | None = None,
        projection: Any | None = None,
        **kwargs: Any,
    ) -> matplotlib.figure.Figure:
        """Generate a spatial plot.

        Parameters
        ----------
        paired_data
            Paired Dataset.
        x_var
            Compatibility name for the x variable.
        y_var
            Compatibility name for the y variable.
        domain
            Geographic extent (lon_min, lon_max, lat_min, lat_max).
        projection
            Cartopy projection for the map.
        **kwargs
            Additional plot options.

        Returns
        -------
        matplotlib.figure.Figure
            The generated figure.
        """
        ...


# =============================================================================
# Statistics Protocols
# =============================================================================


@runtime_checkable
class StatisticMetric(Protocol):
    """Protocol for individual statistical metrics.

    Each metric computes a single statistic (MB, RMSE, R2, etc.).
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short name/abbreviation (e.g., 'MB', 'RMSE')."""
        ...

    @property
    @abstractmethod
    def long_name(self) -> str:
        """Full descriptive name (e.g., 'Mean Bias')."""
        ...

    @abstractmethod
    def compute(
        self,
        x: xr.DataArray,
        y: xr.DataArray,
        **kwargs: Any,
    ) -> float:
        """Compute the statistic.

        Parameters
        ----------
        x
            Reference values.
        y
            Comparison values (aligned with x).
        **kwargs
            Metric-specific options.

        Returns
        -------
        float
            Computed statistic value.
        """
        ...


@runtime_checkable
class StatisticsCalculator(Protocol):
    """Protocol for computing multiple statistics on paired data."""

    @abstractmethod
    def compute(
        self,
        paired_data: xr.Dataset,
        x_var: str,
        y_var: str,
        metrics: Sequence[str] | None = None,
        groupby: str | Sequence[str] | None = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Compute statistics for paired data.

        Parameters
        ----------
        paired_data
            Paired Dataset.
        x_var
            Name of the x variable.
        y_var
            Name of the y variable.
        metrics
            List of metric names to compute. If None, compute all.
        groupby
            Optional dimension(s) to group by (e.g., 'site', 'time.month').
        **kwargs
            Additional options.

        Returns
        -------
        pd.DataFrame
            Statistics table with metrics as columns.
        """
        ...


# =============================================================================
# Configuration Protocol
# =============================================================================


@runtime_checkable
class Configurable(Protocol):
    """Protocol for components that can be configured from dict/YAML."""

    @classmethod
    @abstractmethod
    def from_config(cls, config: Mapping[str, Any]) -> Configurable:
        """Create instance from configuration dict.

        Parameters
        ----------
        config
            Configuration dictionary (typically from YAML).

        Returns
        -------
        Configurable
            Configured instance.
        """
        ...
