"""Core module containing protocols, registry, and base classes.

This module provides the foundational components for DAVINCI:
- Protocol definitions for all pluggable components
- Plugin registry system
- Base data container classes
- Custom exceptions
- Type aliases
"""

from davinci_monet.core.base import (
    PairedData,
    validate_dataset_geometry,
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
from davinci_monet.core.protocols import (
    DataGeometry,
    PairingStrategy,
    SourceProcessor,
    SourceReader,
)
from davinci_monet.core.registry import (
    ComponentAlreadyRegisteredError,
    ComponentNotFoundError,
    Registry,
    RegistryError,
    plotter_registry,
    source_registry,
    statistic_registry,
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
    "PairedData",
    "validate_dataset_geometry",
    # Data geometry enum
    "DataGeometry",
    # Unified source protocols
    "SourceReader",
    "SourceProcessor",
    # Pairing protocols
    "PairingStrategy",
    # Registry
    "Registry",
    "RegistryError",
    "ComponentNotFoundError",
    "ComponentAlreadyRegisteredError",
    # Pre-configured registries
    "source_registry",
    "plotter_registry",
    "statistic_registry",
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
