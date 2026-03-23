"""Custom exception hierarchy for DAVINCI.

This module defines a structured exception hierarchy for consistent error
handling throughout the package. All exceptions inherit from DavinciMonetError.

Exception Hierarchy:
    DavinciMonetError (base)
    ├── ConfigurationError
    │   ├── ConfigValidationError
    │   ├── ConfigParseError
    │   └── ConfigMigrationError
    ├── DataError
    │   ├── DataNotFoundError
    │   ├── DataFormatError
    │   ├── DataValidationError
    │   └── VariableNotFoundError
    ├── PairingError
    │   ├── GeometryMismatchError
    │   ├── NoOverlapError
    │   └── InterpolationError
    ├── PlottingError
    │   └── PlotConfigError
    ├── StatisticsError
    │   └── InsufficientDataError
    └── PipelineError
        ├── StageExecutionError
        └── PipelineAbortError
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


class DavinciMonetError(Exception):
    """Base exception for all DAVINCI errors.

    All custom exceptions in DAVINCI inherit from this class,
    allowing users to catch all package-specific errors with a single
    except clause.

    Parameters
    ----------
    message
        Human-readable error description.
    details
        Optional additional context or data about the error.
    """

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        self.message = message
        self.details = details or {}
        super().__init__(message)

    def __str__(self) -> str:
        if self.details:
            detail_str = ", ".join(f"{k}={v!r}" for k, v in self.details.items())
            return f"{self.message} ({detail_str})"
        return self.message


# =============================================================================
# Configuration Errors
# =============================================================================


class ConfigurationError(DavinciMonetError):
    """Base exception for configuration-related errors."""


class ConfigValidationError(ConfigurationError):
    """Raised when configuration validation fails.

    Parameters
    ----------
    message
        Description of the validation failure.
    field
        The configuration field that failed validation.
    value
        The invalid value.
    expected
        Description of expected value/format.
    """

    def __init__(
        self,
        message: str,
        field: str | None = None,
        value: Any = None,
        expected: str | None = None,
    ) -> None:
        details: dict[str, Any] = {}
        if field is not None:
            details["field"] = field
        if value is not None:
            details["value"] = value
        if expected is not None:
            details["expected"] = expected
        super().__init__(message, details)
        self.field = field
        self.value = value
        self.expected = expected


class ConfigParseError(ConfigurationError):
    """Raised when configuration file parsing fails.

    Parameters
    ----------
    message
        Description of the parse error.
    path
        Path to the configuration file.
    line
        Line number where error occurred (if known).
    """

    def __init__(
        self,
        message: str,
        path: str | Path | None = None,
        line: int | None = None,
    ) -> None:
        details: dict[str, Any] = {}
        if path is not None:
            details["path"] = str(path)
        if line is not None:
            details["line"] = line
        super().__init__(message, details)
        self.path = Path(path) if path else None
        self.line = line


class ConfigMigrationError(ConfigurationError):
    """Raised when configuration migration between versions fails.

    Parameters
    ----------
    message
        Description of the migration failure.
    from_version
        Source configuration version.
    to_version
        Target configuration version.
    """

    def __init__(
        self,
        message: str,
        from_version: str | None = None,
        to_version: str | None = None,
    ) -> None:
        details: dict[str, Any] = {}
        if from_version is not None:
            details["from_version"] = from_version
        if to_version is not None:
            details["to_version"] = to_version
        super().__init__(message, details)
        self.from_version = from_version
        self.to_version = to_version


# =============================================================================
# Data Errors
# =============================================================================


class DataError(DavinciMonetError):
    """Base exception for data-related errors."""


class DataNotFoundError(DataError):
    """Raised when required data cannot be found.

    Parameters
    ----------
    message
        Description of the missing data.
    path
        Path where data was expected.
    pattern
        Glob pattern used to search for data.
    """

    def __init__(
        self,
        message: str,
        path: str | Path | None = None,
        pattern: str | None = None,
    ) -> None:
        details: dict[str, Any] = {}
        if path is not None:
            details["path"] = str(path)
        if pattern is not None:
            details["pattern"] = pattern
        super().__init__(message, details)
        self.path = Path(path) if path else None
        self.pattern = pattern


class DataFormatError(DataError):
    """Raised when data format is invalid or unsupported.

    Parameters
    ----------
    message
        Description of the format issue.
    path
        Path to the problematic file.
    expected_format
        Expected file format.
    actual_format
        Actual file format encountered.
    """

    def __init__(
        self,
        message: str,
        path: str | Path | None = None,
        expected_format: str | None = None,
        actual_format: str | None = None,
    ) -> None:
        details: dict[str, Any] = {}
        if path is not None:
            details["path"] = str(path)
        if expected_format is not None:
            details["expected_format"] = expected_format
        if actual_format is not None:
            details["actual_format"] = actual_format
        super().__init__(message, details)
        self.path = Path(path) if path else None
        self.expected_format = expected_format
        self.actual_format = actual_format


class DataValidationError(DataError):
    """Raised when data fails validation checks.

    Parameters
    ----------
    message
        Description of the validation failure.
    variable
        Variable that failed validation.
    reason
        Specific reason for failure.
    """

    def __init__(
        self,
        message: str,
        variable: str | None = None,
        reason: str | None = None,
    ) -> None:
        details: dict[str, Any] = {}
        if variable is not None:
            details["variable"] = variable
        if reason is not None:
            details["reason"] = reason
        super().__init__(message, details)
        self.variable = variable
        self.reason = reason


class VariableNotFoundError(DataError):
    """Raised when a required variable is not found in a dataset.

    Parameters
    ----------
    message
        Description of the missing variable.
    variable
        Name of the missing variable.
    available
        List of available variables.
    dataset
        Name/description of the dataset searched.
    """

    def __init__(
        self,
        message: str,
        variable: str | None = None,
        available: list[str] | None = None,
        dataset: str | None = None,
    ) -> None:
        details: dict[str, Any] = {}
        if variable is not None:
            details["variable"] = variable
        if available is not None:
            details["available"] = available
        if dataset is not None:
            details["dataset"] = dataset
        super().__init__(message, details)
        self.variable = variable
        self.available = available
        self.dataset = dataset


# =============================================================================
# Pairing Errors
# =============================================================================


class PairingError(DavinciMonetError):
    """Base exception for pairing-related errors."""


class GeometryMismatchError(PairingError):
    """Raised when data geometries are incompatible for pairing.

    Parameters
    ----------
    message
        Description of the geometry mismatch.
    model_geometry
        Geometry type of model data.
    obs_geometry
        Geometry type of observation data.
    """

    def __init__(
        self,
        message: str,
        model_geometry: str | None = None,
        obs_geometry: str | None = None,
    ) -> None:
        details: dict[str, Any] = {}
        if model_geometry is not None:
            details["model_geometry"] = model_geometry
        if obs_geometry is not None:
            details["obs_geometry"] = obs_geometry
        super().__init__(message, details)
        self.model_geometry = model_geometry
        self.obs_geometry = obs_geometry


class NoOverlapError(PairingError):
    """Raised when model and observation data have no spatial/temporal overlap.

    Parameters
    ----------
    message
        Description of the overlap issue.
    dimension
        Dimension with no overlap (e.g., 'time', 'space').
    model_range
        Range of model data.
    obs_range
        Range of observation data.
    """

    def __init__(
        self,
        message: str,
        dimension: str | None = None,
        model_range: tuple[Any, Any] | None = None,
        obs_range: tuple[Any, Any] | None = None,
    ) -> None:
        details: dict[str, Any] = {}
        if dimension is not None:
            details["dimension"] = dimension
        if model_range is not None:
            details["model_range"] = model_range
        if obs_range is not None:
            details["obs_range"] = obs_range
        super().__init__(message, details)
        self.dimension = dimension
        self.model_range = model_range
        self.obs_range = obs_range


class InterpolationError(PairingError):
    """Raised when interpolation fails during pairing.

    Parameters
    ----------
    message
        Description of the interpolation failure.
    method
        Interpolation method that failed.
    dimension
        Dimension being interpolated.
    """

    def __init__(
        self,
        message: str,
        method: str | None = None,
        dimension: str | None = None,
    ) -> None:
        details: dict[str, Any] = {}
        if method is not None:
            details["method"] = method
        if dimension is not None:
            details["dimension"] = dimension
        super().__init__(message, details)
        self.method = method
        self.dimension = dimension


# =============================================================================
# Plotting Errors
# =============================================================================


class PlottingError(DavinciMonetError):
    """Base exception for plotting-related errors."""


class PlotConfigError(PlottingError):
    """Raised when plot configuration is invalid.

    Parameters
    ----------
    message
        Description of the configuration issue.
    plot_type
        Type of plot being configured.
    parameter
        Invalid parameter name.
    """

    def __init__(
        self,
        message: str,
        plot_type: str | None = None,
        parameter: str | None = None,
    ) -> None:
        details: dict[str, Any] = {}
        if plot_type is not None:
            details["plot_type"] = plot_type
        if parameter is not None:
            details["parameter"] = parameter
        super().__init__(message, details)
        self.plot_type = plot_type
        self.parameter = parameter


# =============================================================================
# Statistics Errors
# =============================================================================


class StatisticsError(DavinciMonetError):
    """Base exception for statistics-related errors."""


class InsufficientDataError(StatisticsError):
    """Raised when there is insufficient data to compute statistics.

    Parameters
    ----------
    message
        Description of the data insufficiency.
    metric
        Statistic being computed.
    required
        Minimum number of points required.
    actual
        Actual number of points available.
    """

    def __init__(
        self,
        message: str,
        metric: str | None = None,
        required: int | None = None,
        actual: int | None = None,
    ) -> None:
        details: dict[str, Any] = {}
        if metric is not None:
            details["metric"] = metric
        if required is not None:
            details["required"] = required
        if actual is not None:
            details["actual"] = actual
        super().__init__(message, details)
        self.metric = metric
        self.required = required
        self.actual = actual


# =============================================================================
# Pipeline Errors
# =============================================================================


class PipelineError(DavinciMonetError):
    """Base exception for pipeline-related errors."""


class StageExecutionError(PipelineError):
    """Raised when a pipeline stage fails to execute.

    Parameters
    ----------
    message
        Description of the execution failure.
    stage_name
        Name of the failed stage.
    original_error
        The underlying exception that caused the failure.
    """

    def __init__(
        self,
        message: str,
        stage_name: str | None = None,
        original_error: Exception | None = None,
    ) -> None:
        details: dict[str, Any] = {}
        if stage_name is not None:
            details["stage_name"] = stage_name
        if original_error is not None:
            details["original_error"] = str(original_error)
        super().__init__(message, details)
        self.stage_name = stage_name
        self.original_error = original_error


class PipelineAbortError(PipelineError):
    """Raised when pipeline execution is aborted.

    Parameters
    ----------
    message
        Description of why the pipeline was aborted.
    completed_stages
        List of stages that completed before abort.
    pending_stages
        List of stages that were not executed.
    """

    def __init__(
        self,
        message: str,
        completed_stages: list[str] | None = None,
        pending_stages: list[str] | None = None,
    ) -> None:
        details: dict[str, Any] = {}
        if completed_stages is not None:
            details["completed_stages"] = completed_stages
        if pending_stages is not None:
            details["pending_stages"] = pending_stages
        super().__init__(message, details)
        self.completed_stages = completed_stages
        self.pending_stages = pending_stages


# Known transient NetCDF/HDF5 error patterns that may succeed on retry
TRANSIENT_ERROR_PATTERNS = [
    "Not a valid ID",
    "invalid location identifier",
    "CachingFileManager",
    "h5py",
    "HDF5",
    "NetCDF",
]


def is_transient_error(error: Exception) -> bool:
    """Check if an error appears to be a transient NetCDF/HDF5 error.

    Parameters
    ----------
    error
        The exception to check.

    Returns
    -------
    bool
        True if the error matches known transient patterns.
    """
    error_str = str(error)
    error_type = type(error).__name__
    combined = f"{error_type}: {error_str}"

    return any(pattern in combined for pattern in TRANSIENT_ERROR_PATTERNS)


def cleanup_netcdf_state() -> None:
    """Clean up NetCDF/HDF5 state to help recover from transient errors.

    This function forces garbage collection and clears xarray's file cache,
    which can help recover from stale file handle errors.
    """
    import gc

    # Force garbage collection
    gc.collect()

    # Clear xarray's file manager cache
    try:
        from xarray.backends.file_manager import FILE_CACHE

        FILE_CACHE.clear()
    except (ImportError, AttributeError):
        pass

    # Try to clear any HDF5 state
    try:
        import h5py

        h5py._errors.silence_errors()
    except (ImportError, AttributeError):
        pass


def write_error_log(
    error: Exception,
    context: str,
    log_dir: str | None = None,
) -> str | None:
    """Write error traceback to a log file.

    This utility function saves full exception details to a timestamped log file,
    which is useful for debugging while keeping user-facing error messages clean.

    Parameters
    ----------
    error
        The exception that occurred.
    context
        Description of what operation failed (e.g., "Opening CESM files").
    log_dir
        Directory to write log file. Defaults to "./logs".

    Returns
    -------
    str or None
        Path to the error log file, or None if writing failed.

    Examples
    --------
    >>> try:
    ...     ds = xr.open_dataset("bad_file.nc")
    ... except Exception as e:
    ...     log_path = write_error_log(e, "Opening model data")
    ...     raise DataFormatError(f"Failed to open file. Details: {log_path}") from e
    """
    import traceback
    from datetime import datetime
    from pathlib import Path

    if log_dir is None:
        log_dir = "logs"

    log_path = Path(log_dir)

    try:
        log_path.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        error_file = log_path / f"error_{timestamp}.log"

        with open(error_file, "w") as f:
            f.write(f"Error occurred: {datetime.now().isoformat()}\n")
            f.write(f"Context: {context}\n")
            f.write(f"Error type: {type(error).__name__}\n")
            f.write(f"Error message: {error}\n")
            f.write("\nFull traceback:\n")
            f.write(traceback.format_exc())

        return str(error_file)
    except Exception:
        return None
