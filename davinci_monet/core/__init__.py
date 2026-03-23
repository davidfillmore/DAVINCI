"""Core module containing protocols, registry, and base classes.

This module provides the foundational components for DAVINCI:
- Protocol definitions for all pluggable components
- Plugin registry system
- Base data container classes
- Custom exceptions
- Type aliases
"""

from davinci_monet.core.base import (
    DataContainer,
    PairedData,
    create_paired_dataset,
    validate_dataset_geometry,
)
from davinci_monet.core.protocols import (
    Configurable,
    DataGeometry,
    DataReader,
    DataWriter,
    ModelProcessor,
    ModelReader,
    ObservationProcessor,
    ObservationReader,
    PairingEngine,
    PairingStrategy,
    Pipeline,
    PipelineStage,
    Plotter,
    SpatialPlotter,
    StatisticMetric,
    StatisticsCalculator,
)
from davinci_monet.core.registry import (
    ComponentAlreadyRegisteredError,
    ComponentNotFoundError,
    Registry,
    RegistryError,
    model_registry,
    observation_registry,
    pairing_registry,
    plotter_registry,
    reader_registry,
    statistic_registry,
    writer_registry,
)
from davinci_monet.core.exceptions import (
    ConfigMigrationError,
    ConfigParseError,
    ConfigurationError,
    ConfigValidationError,
    DataError,
    DataFormatError,
    DataNotFoundError,
    DataValidationError,
    DavinciMonetError,
    GeometryMismatchError,
    InsufficientDataError,
    InterpolationError,
    NoOverlapError,
    PairingError,
    PipelineAbortError,
    PipelineError,
    PlotConfigError,
    PlottingError,
    StageExecutionError,
    StatisticsError,
    VariableNotFoundError,
)
from davinci_monet.core.types import (
    BoundingBox,
    ConfigDict,
    Coordinate,
    PathLike,
    TimeRange,
    Timestamp,
    VariableMapping,
)

__all__ = [
    # Base data classes
    "DataContainer",
    "PairedData",
    "create_paired_dataset",
    "validate_dataset_geometry",
    # Data geometry enum
    "DataGeometry",
    # Model protocols
    "ModelReader",
    "ModelProcessor",
    # Observation protocols
    "ObservationReader",
    "ObservationProcessor",
    # Pairing protocols
    "PairingStrategy",
    "PairingEngine",
    # Plotting protocols
    "Plotter",
    "SpatialPlotter",
    # Statistics protocols
    "StatisticMetric",
    "StatisticsCalculator",
    # Pipeline protocols
    "PipelineStage",
    "Pipeline",
    # I/O protocols
    "DataReader",
    "DataWriter",
    # Configuration protocol
    "Configurable",
    # Registry
    "Registry",
    "RegistryError",
    "ComponentNotFoundError",
    "ComponentAlreadyRegisteredError",
    # Pre-configured registries
    "model_registry",
    "observation_registry",
    "pairing_registry",
    "plotter_registry",
    "statistic_registry",
    "reader_registry",
    "writer_registry",
    # Exceptions
    "DavinciMonetError",
    "ConfigurationError",
    "ConfigValidationError",
    "ConfigParseError",
    "ConfigMigrationError",
    "DataError",
    "DataNotFoundError",
    "DataFormatError",
    "DataValidationError",
    "VariableNotFoundError",
    "PairingError",
    "GeometryMismatchError",
    "NoOverlapError",
    "InterpolationError",
    "PlottingError",
    "PlotConfigError",
    "StatisticsError",
    "InsufficientDataError",
    "PipelineError",
    "StageExecutionError",
    "PipelineAbortError",
    # Type aliases (commonly used)
    "PathLike",
    "Timestamp",
    "TimeRange",
    "Coordinate",
    "BoundingBox",
    "VariableMapping",
    "ConfigDict",
]
