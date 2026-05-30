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
    to match observations with model output.
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
    """Regular grid (time, lat, lon) - L3 products, reanalysis, model output."""


# =============================================================================
# Model Protocols
# =============================================================================


@runtime_checkable
class ModelReader(Protocol):
    """Protocol for model output readers.

    Model readers are responsible for loading atmospheric model output
    (CMAQ, WRF-Chem, UFS, CESM, etc.) into standardized xarray Datasets.

    The output Dataset should have dimensions (time, level, lat, lon) with
    consistent coordinate names and units.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this model type (e.g., 'cmaq', 'wrfchem')."""
        ...

    @abstractmethod
    def open(
        self,
        file_paths: Sequence[str | Path],
        variables: Sequence[str] | None = None,
        **kwargs: Any,
    ) -> xr.Dataset:
        """Open model output files and return a standardized Dataset.

        Parameters
        ----------
        file_paths
            Paths to model output files (can include glob patterns).
        variables
            Optional list of variables to load. If None, load all.
        **kwargs
            Additional reader-specific options.

        Returns
        -------
        xr.Dataset
            Model output with dims (time, level, lat, lon).
        """
        ...

    @abstractmethod
    def get_variable_mapping(self) -> Mapping[str, str]:
        """Return mapping from standard variable names to model-specific names.

        Returns
        -------
        Mapping[str, str]
            Dict mapping standard names (e.g., 'ozone') to model names (e.g., 'O3').
        """
        ...


@runtime_checkable
class ModelProcessor(Protocol):
    """Protocol for model data post-processing operations.

    Processors handle operations like unit conversion, variable derivation,
    spatial subsetting, and temporal aggregation.
    """

    @abstractmethod
    def process(self, dataset: xr.Dataset, **kwargs: Any) -> xr.Dataset:
        """Apply processing to a model Dataset.

        Parameters
        ----------
        dataset
            Input model Dataset.
        **kwargs
            Processing options.

        Returns
        -------
        xr.Dataset
            Processed Dataset.
        """
        ...


# =============================================================================
# Observation Protocols
# =============================================================================


@runtime_checkable
class ObservationReader(Protocol):
    """Protocol for observation data readers.

    Observation readers load observational data from various sources
    (surface networks, aircraft campaigns, satellites) into xarray Datasets
    with geometry metadata.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this observation type."""
        ...

    @property
    @abstractmethod
    def geometry(self) -> DataGeometry:
        """The data geometry type for this observation source."""
        ...

    @abstractmethod
    def open(
        self,
        file_paths: Sequence[str | Path],
        variables: Sequence[str] | None = None,
        time_range: tuple[Any, Any] | None = None,
        **kwargs: Any,
    ) -> xr.Dataset:
        """Open observation files and return a standardized Dataset.

        Parameters
        ----------
        file_paths
            Paths to observation files.
        variables
            Optional list of variables to load.
        time_range
            Optional (start, end) time range to subset.
        **kwargs
            Additional reader-specific options.

        Returns
        -------
        xr.Dataset
            Observation data with geometry-appropriate dimensions and
            'geometry' attribute set.
        """
        ...

    @abstractmethod
    def get_variable_mapping(self) -> Mapping[str, str]:
        """Return mapping from standard variable names to observation-specific names."""
        ...


@runtime_checkable
class ObservationProcessor(Protocol):
    """Protocol for observation data post-processing.

    Handles filtering, QA/QC, unit conversion, resampling, etc.
    """

    @abstractmethod
    def process(self, dataset: xr.Dataset, **kwargs: Any) -> xr.Dataset:
        """Apply processing to an observation Dataset."""
        ...


# =============================================================================
# Unified Source Protocols
# =============================================================================
#
# A data source is just data of a given geometry (point, track, profile,
# swath, grid). Models and observations are both data sources; the only thing
# that distinguishes them is topology, not origin. These protocols unify the
# legacy ModelReader/ModelProcessor and ObservationReader/ObservationProcessor
# pairs. A model/obs "role" may travel as metadata for labeling and styling,
# but it never appears in these contracts.


@runtime_checkable
class SourceReader(Protocol):
    """Protocol for data source readers (models and observations alike).

    Every source reader declares the geometry it produces and loads files into
    a standardized xarray Dataset whose ``attrs['geometry']`` is set. This is
    the unified replacement for ``ModelReader`` and ``ObservationReader``.
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

    @abstractmethod
    def get_variable_mapping(self) -> Mapping[str, str]:
        """Return mapping from standard variable names to source-specific names."""
        ...


@runtime_checkable
class SourceProcessor(Protocol):
    """Protocol for data source post-processing operations.

    Unifies ``ModelProcessor`` and ``ObservationProcessor``. Processors handle
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
    """Protocol for model-observation pairing strategies.

    Each pairing strategy handles a specific data geometry, implementing
    the spatial and temporal matching logic appropriate for that geometry.
    """

    @property
    @abstractmethod
    def geometry(self) -> DataGeometry:
        """The data geometry this strategy handles."""
        ...

    @abstractmethod
    def pair(
        self,
        model: xr.Dataset,
        obs: xr.Dataset,
        radius_of_influence: float | None = None,
        time_tolerance: Any | None = None,
        vertical_method: str = "nearest",
        horizontal_method: str = "nearest",
        **kwargs: Any,
    ) -> xr.Dataset:
        """Pair model output with observations.

        Parameters
        ----------
        model
            Model Dataset with dims (time, level, lat, lon).
        obs
            Observation Dataset with geometry-specific dimensions.
        radius_of_influence
            Spatial search radius in meters.
        time_tolerance
            Maximum time difference for matching (e.g., '1h', timedelta).
        vertical_method
            Vertical interpolation method ('nearest', 'linear', 'log').
        horizontal_method
            Horizontal interpolation method ('nearest', 'bilinear').
        **kwargs
            Strategy-specific options.

        Returns
        -------
        xr.Dataset
            Paired Dataset with aligned model and observation variables.
            Contains both 'model_<var>' and 'obs_<var>' data variables.
        """
        ...


@runtime_checkable
class PairingEngine(Protocol):
    """Protocol for the main pairing orchestrator.

    The pairing engine selects the appropriate strategy based on
    observation geometry and coordinates the pairing process.
    """

    @abstractmethod
    def register_strategy(self, strategy: PairingStrategy) -> None:
        """Register a pairing strategy for a geometry type."""
        ...

    @abstractmethod
    def pair(
        self,
        model: xr.Dataset,
        obs: xr.Dataset,
        **kwargs: Any,
    ) -> xr.Dataset:
        """Pair model and observations using the appropriate strategy.

        The strategy is selected based on the observation Dataset's
        'geometry' attribute.
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
        obs_var: str,
        model_var: str,
        **kwargs: Any,
    ) -> matplotlib.figure.Figure:
        """Generate a plot from paired data.

        Parameters
        ----------
        paired_data
            Paired Dataset containing model and observation variables.
        obs_var
            Name of the observation variable to plot.
        model_var
            Name of the model variable to plot.
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
        obs_var: str,
        model_var: str,
        domain: tuple[float, float, float, float] | None = None,
        projection: Any | None = None,
        **kwargs: Any,
    ) -> matplotlib.figure.Figure:
        """Generate a spatial plot.

        Parameters
        ----------
        paired_data
            Paired Dataset.
        obs_var
            Observation variable name.
        model_var
            Model variable name.
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
        obs: xr.DataArray,
        model: xr.DataArray,
        **kwargs: Any,
    ) -> float:
        """Compute the statistic.

        Parameters
        ----------
        obs
            Observation values.
        model
            Model values (aligned with obs).
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
        obs_var: str,
        model_var: str,
        metrics: Sequence[str] | None = None,
        groupby: str | Sequence[str] | None = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Compute statistics for paired data.

        Parameters
        ----------
        paired_data
            Paired Dataset.
        obs_var
            Observation variable name.
        model_var
            Model variable name.
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
# Pipeline Protocols
# =============================================================================


@runtime_checkable
class PipelineStage(Protocol):
    """Protocol for pipeline execution stages.

    Stages are composable units that perform specific parts of the
    analysis workflow (load data, pair, compute stats, plot).
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Stage identifier."""
        ...

    @abstractmethod
    def execute(self, context: Mapping[str, Any]) -> Mapping[str, Any]:
        """Execute this pipeline stage.

        Parameters
        ----------
        context
            Shared context dict containing data and metadata from
            previous stages.

        Returns
        -------
        Mapping[str, Any]
            Updated context with this stage's outputs.
        """
        ...


@runtime_checkable
class Pipeline(Protocol):
    """Protocol for the analysis pipeline runner."""

    @abstractmethod
    def add_stage(self, stage: PipelineStage) -> None:
        """Add a stage to the pipeline."""
        ...

    @abstractmethod
    def run(self, initial_context: Mapping[str, Any] | None = None) -> Mapping[str, Any]:
        """Execute all stages in sequence.

        Parameters
        ----------
        initial_context
            Optional initial context dict.

        Returns
        -------
        Mapping[str, Any]
            Final context after all stages complete.
        """
        ...


# =============================================================================
# I/O Protocols
# =============================================================================


@runtime_checkable
class DataReader(Protocol):
    """Generic protocol for file readers."""

    @abstractmethod
    def read(self, path: str | Path, **kwargs: Any) -> xr.Dataset:
        """Read data from file(s)."""
        ...

    @abstractmethod
    def supports(self, path: str | Path) -> bool:
        """Check if this reader supports the given file type."""
        ...


@runtime_checkable
class DataWriter(Protocol):
    """Generic protocol for file writers."""

    @abstractmethod
    def write(
        self,
        data: xr.Dataset,
        path: str | Path,
        **kwargs: Any,
    ) -> Path:
        """Write data to file."""
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
