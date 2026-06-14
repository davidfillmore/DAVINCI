"""Core stage primitives: status, result, protocol, context, and base class.

This module holds the framework-level building blocks shared by every concrete
stage: :class:`StageStatus`, :class:`StageResult`, the :class:`Stage` protocol,
:class:`PipelineContext`, the :class:`SourceData` / :class:`SourcePairJob`
containers, and the :class:`BaseStage` abstract base.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Protocol, runtime_checkable

import xarray as xr

from davinci_monet.core.protocols import DataGeometry


class StageStatus(Enum):
    """Status of a pipeline stage."""

    PENDING = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()
    SKIPPED = auto()


@dataclass
class StageResult:
    """Result of a pipeline stage execution.

    Attributes
    ----------
    stage_name
        Name of the stage that produced this result.
    status
        Execution status.
    data
        Output data from the stage.
    metadata
        Additional metadata about the execution.
    error
        Error message if the stage failed.
    error_type
        Exception class name if the stage failed (e.g., 'ValueError').
    traceback_str
        Full traceback string if the stage failed with an exception.
    duration_seconds
        Execution time in seconds.
    """

    stage_name: str
    status: StageStatus
    data: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    error_type: str | None = None
    traceback_str: str | None = None
    duration_seconds: float = 0.0


@runtime_checkable
class Stage(Protocol):
    """Protocol for pipeline stages.

    A stage is a single unit of work in the analysis pipeline.
    Stages can be composed and chained together.
    """

    @property
    def name(self) -> str:
        """Stage name."""
        ...

    def execute(self, context: PipelineContext) -> StageResult:
        """Execute the stage.

        Parameters
        ----------
        context
            Pipeline context containing configuration and data.

        Returns
        -------
        StageResult
            Result of stage execution.
        """
        ...

    def validate(self, context: PipelineContext) -> bool:
        """Validate that the stage can run with the given context.

        Parameters
        ----------
        context
            Pipeline context to validate.

        Returns
        -------
        bool
            True if validation passes.
        """
        ...


@dataclass
class PipelineContext:
    """Context passed between pipeline stages.

    Contains configuration, data, and state that flows through the pipeline.

    Attributes
    ----------
    config
        Configuration dictionary from YAML or programmatic setup.
    sources
        Dictionary of loaded datasets keyed by source label.
    paired
        Dictionary of paired source data.
    results
        Results from completed stages.
    metadata
        Pipeline metadata (start time, etc.).
    progress_callback
        Optional callback for reporting progress within stages.
        Called with a message string to display progress updates.
    """

    config: dict[str, Any] = field(default_factory=dict)
    # Unified data-source view keyed by dataset label.
    sources: dict[str, Any] = field(default_factory=dict)
    paired: dict[str, Any] = field(default_factory=dict)
    results: dict[str, StageResult] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    progress_callback: Callable[[str], None] | None = None

    def log_progress(self, message: str) -> None:
        """Log a progress message if callback is set."""
        if self.progress_callback:
            self.progress_callback(message)

    def get_source(self, label: str) -> Any:
        """Get a data source (dataset or dataset) by label.

        Part of the unified data-source abstraction (Phase 3). Sources are
        populated by :class:`LoadSourcesStage`.
        """
        if label not in self.sources:
            raise KeyError(f"Source '{label}' not found in context")
        return self.sources[label]

    def iter_sources(self) -> list[tuple[str, Any]]:
        """Return loaded sources in insertion order."""
        return list(self.sources.items())

    def get_source_dataset(self, label: str) -> xr.Dataset:
        """Return the xarray Dataset for a source label."""
        source = self.get_source(label)
        data = source.data if hasattr(source, "data") else source
        if not isinstance(data, xr.Dataset):
            raise KeyError(f"Source '{label}' does not contain an xarray Dataset")
        return data

    def get_paired(self, key: str) -> Any:
        """Get paired data by key."""
        if key not in self.paired:
            raise KeyError(f"Paired data '{key}' not found in context")
        return self.paired[key]


@dataclass
class SourceData:
    """Container for a unified data source loaded from ``sources:`` config."""

    data: xr.Dataset
    label: str
    source_type: str
    geometry: DataGeometry
    variables: dict[str, Any] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)


@dataclass
class SourcePairJob:
    """Concrete source pair to process."""

    index: int
    pair_key: str
    geometry_label: str
    geometry_obj: Any
    dataset_label: str
    dataset_obj: Any
    geometry_var: str
    dataset_var: str
    radius_of_influence: float
    strategy_options: dict[str, Any] = field(default_factory=dict)


class BaseStage(ABC):
    """Abstract base class for pipeline stages.

    Provides common functionality for stage implementations.
    """

    def __init__(self, name: str | None = None) -> None:
        """Initialize stage.

        Parameters
        ----------
        name
            Optional custom name. If None, uses class name.
        """
        self._name = name or self.__class__.__name__

    @property
    def name(self) -> str:
        """Stage name."""
        return self._name

    def validate(self, context: PipelineContext) -> bool:
        """Default validation - always passes.

        Override in subclasses for specific validation.
        """
        return True

    @abstractmethod
    def execute(self, context: PipelineContext) -> StageResult:
        """Execute the stage."""
        ...

    def _create_result(
        self,
        status: StageStatus,
        data: Any = None,
        error: str | None = None,
        duration: float = 0.0,
        **metadata: Any,
    ) -> StageResult:
        """Create a stage result."""
        return StageResult(
            stage_name=self.name,
            status=status,
            data=data,
            error=error,
            duration_seconds=duration,
            metadata=metadata,
        )
