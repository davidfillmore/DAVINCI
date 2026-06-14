"""Tests for custom exception hierarchy."""

from __future__ import annotations

from pathlib import Path

import pytest

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


class TestDavinciMonetError:
    """Tests for base exception class."""

    def test_basic_exception(self) -> None:
        """Test creating a basic exception with message only."""
        error = DavinciMonetError("Something went wrong")
        assert str(error) == "Something went wrong"
        assert error.message == "Something went wrong"
        assert error.details == {}

    def test_exception_with_details(self) -> None:
        """Test creating an exception with details."""
        error = DavinciMonetError("Error occurred", {"key": "value", "num": 42})
        assert "key='value'" in str(error)
        assert "num=42" in str(error)
        assert error.details["key"] == "value"
        assert error.details["num"] == 42

    def test_exception_is_catchable(self) -> None:
        """Test that exception can be caught."""
        with pytest.raises(DavinciMonetError):
            raise DavinciMonetError("Test error")

    def test_exception_inherits_from_exception(self) -> None:
        """Test that DavinciMonetError inherits from Exception."""
        assert issubclass(DavinciMonetError, Exception)


class TestConfigurationErrors:
    """Tests for configuration error classes."""

    def test_configuration_error_hierarchy(self) -> None:
        """Test ConfigurationError inherits from DavinciMonetError."""
        assert issubclass(ConfigurationError, DavinciMonetError)

    def test_config_validation_error(self) -> None:
        """Test ConfigValidationError with all parameters."""
        error = ConfigValidationError(
            "Invalid value",
            field="start_time",
            value="not-a-date",
            expected="ISO 8601 datetime string",
        )
        assert error.field == "start_time"
        assert error.value == "not-a-date"
        assert error.expected == "ISO 8601 datetime string"
        assert "start_time" in str(error)

    def test_config_validation_error_minimal(self) -> None:
        """Test ConfigValidationError with minimal parameters."""
        error = ConfigValidationError("Validation failed")
        assert error.field is None
        assert error.value is None
        assert error.expected is None

    def test_config_parse_error(self) -> None:
        """Test ConfigParseError with path and line."""
        error = ConfigParseError("YAML syntax error", path="/path/to/config.yaml", line=42)
        assert error.path == Path("/path/to/config.yaml")
        assert error.line == 42
        assert "/path/to/config.yaml" in str(error)

    def test_config_migration_error(self) -> None:
        """Test ConfigMigrationError with versions."""
        error = ConfigMigrationError(
            "Cannot migrate configuration",
            from_version="1.0",
            to_version="2.0",
        )
        assert error.from_version == "1.0"
        assert error.to_version == "2.0"


class TestDataErrors:
    """Tests for data error classes."""

    def test_data_error_hierarchy(self) -> None:
        """Test DataError inherits from DavinciMonetError."""
        assert issubclass(DataError, DavinciMonetError)

    def test_data_not_found_error(self) -> None:
        """Test DataNotFoundError with path and pattern."""
        error = DataNotFoundError(
            "Dataset files not found",
            path="/data/dataset/",
            pattern="*.nc",
        )
        assert error.path == Path("/data/dataset/")
        assert error.pattern == "*.nc"

    def test_data_format_error(self) -> None:
        """Test DataFormatError with format info."""
        error = DataFormatError(
            "Unsupported format",
            path="/data/file.xyz",
            expected_format="NetCDF",
            actual_format="unknown",
        )
        assert error.path == Path("/data/file.xyz")
        assert error.expected_format == "NetCDF"
        assert error.actual_format == "unknown"

    def test_data_validation_error(self) -> None:
        """Test DataValidationError."""
        error = DataValidationError(
            "Data validation failed",
            variable="temperature",
            reason="Values out of physical range",
        )
        assert error.variable == "temperature"
        assert error.reason == "Values out of physical range"

    def test_variable_not_found_error(self) -> None:
        """Test VariableNotFoundError with available variables."""
        error = VariableNotFoundError(
            "Variable not found",
            variable="O3",
            available=["NO2", "PM25", "CO"],
            dataset="dataset_output",
        )
        assert error.variable == "O3"
        assert error.available == ["NO2", "PM25", "CO"]
        assert error.dataset == "dataset_output"


class TestPairingErrors:
    """Tests for pairing error classes."""

    def test_pairing_error_hierarchy(self) -> None:
        """Test PairingError inherits from DavinciMonetError."""
        assert issubclass(PairingError, DavinciMonetError)

    def test_geometry_mismatch_error(self) -> None:
        """Test GeometryMismatchError."""
        error = GeometryMismatchError(
            "Cannot pair incompatible geometries",
            dataset_geometry="grid",
            geometry_geometry="track",
        )
        assert error.dataset_geometry == "grid"
        assert error.geometry_geometry == "track"

    def test_no_overlap_error(self) -> None:
        """Test NoOverlapError with dimension ranges."""
        error = NoOverlapError(
            "No temporal overlap",
            dimension="time",
            dataset_range=("2024-01-01", "2024-01-31"),
            geometry_range=("2024-02-01", "2024-02-28"),
        )
        assert error.dimension == "time"
        assert error.dataset_range == ("2024-01-01", "2024-01-31")
        assert error.geometry_range == ("2024-02-01", "2024-02-28")

    def test_interpolation_error(self) -> None:
        """Test InterpolationError."""
        error = InterpolationError(
            "Interpolation failed",
            method="bilinear",
            dimension="horizontal",
        )
        assert error.method == "bilinear"
        assert error.dimension == "horizontal"


class TestPlottingErrors:
    """Tests for plotting error classes."""

    def test_plotting_error_hierarchy(self) -> None:
        """Test PlottingError inherits from DavinciMonetError."""
        assert issubclass(PlottingError, DavinciMonetError)

    def test_plot_config_error(self) -> None:
        """Test PlotConfigError."""
        error = PlotConfigError(
            "Invalid plot configuration",
            plot_type="timeseries",
            parameter="color_map",
        )
        assert error.plot_type == "timeseries"
        assert error.parameter == "color_map"


class TestStatisticsErrors:
    """Tests for statistics error classes."""

    def test_statistics_error_hierarchy(self) -> None:
        """Test StatisticsError inherits from DavinciMonetError."""
        assert issubclass(StatisticsError, DavinciMonetError)

    def test_insufficient_data_error(self) -> None:
        """Test InsufficientDataError."""
        error = InsufficientDataError(
            "Not enough data points",
            metric="R2",
            required=30,
            actual=5,
        )
        assert error.metric == "R2"
        assert error.required == 30
        assert error.actual == 5


class TestPipelineErrors:
    """Tests for pipeline error classes."""

    def test_pipeline_error_hierarchy(self) -> None:
        """Test PipelineError inherits from DavinciMonetError."""
        assert issubclass(PipelineError, DavinciMonetError)

    def test_stage_execution_error(self) -> None:
        """Test StageExecutionError with original error."""
        original = ValueError("Something went wrong")
        error = StageExecutionError(
            "Stage failed",
            stage_name="load_dataset",
            original_error=original,
        )
        assert error.stage_name == "load_dataset"
        assert error.original_error is original
        assert "Something went wrong" in str(error)

    def test_pipeline_abort_error(self) -> None:
        """Test PipelineAbortError with stage lists."""
        error = PipelineAbortError(
            "Pipeline aborted due to error",
            completed_stages=["load_dataset", "load_geometry"],
            pending_stages=["pair", "plot", "stats"],
        )
        assert error.completed_stages == ["load_dataset", "load_geometry"]
        assert error.pending_stages == ["pair", "plot", "stats"]


class TestExceptionHierarchy:
    """Tests for overall exception hierarchy structure."""

    def test_all_exceptions_inherit_from_base(self) -> None:
        """Verify all custom exceptions inherit from DavinciMonetError."""
        exceptions = [
            ConfigurationError,
            ConfigValidationError,
            ConfigParseError,
            ConfigMigrationError,
            DataError,
            DataNotFoundError,
            DataFormatError,
            DataValidationError,
            VariableNotFoundError,
            PairingError,
            GeometryMismatchError,
            NoOverlapError,
            InterpolationError,
            PlottingError,
            PlotConfigError,
            StatisticsError,
            InsufficientDataError,
            PipelineError,
            StageExecutionError,
            PipelineAbortError,
        ]
        for exc_class in exceptions:
            assert issubclass(exc_class, DavinciMonetError), f"{exc_class.__name__}"

    def test_catch_all_config_errors(self) -> None:
        """Test catching all config errors with ConfigurationError."""
        config_errors = [
            ConfigValidationError("test"),
            ConfigParseError("test"),
            ConfigMigrationError("test"),
        ]
        for error in config_errors:
            with pytest.raises(ConfigurationError):
                raise error

    def test_catch_all_data_errors(self) -> None:
        """Test catching all data errors with DataError."""
        data_errors = [
            DataNotFoundError("test"),
            DataFormatError("test"),
            DataValidationError("test"),
            VariableNotFoundError("test"),
        ]
        for error in data_errors:
            with pytest.raises(DataError):
                raise error

    def test_catch_all_pairing_errors(self) -> None:
        """Test catching all pairing errors with PairingError."""
        pairing_errors = [
            GeometryMismatchError("test"),
            NoOverlapError("test"),
            InterpolationError("test"),
        ]
        for error in pairing_errors:
            with pytest.raises(PairingError):
                raise error

    def test_catch_all_package_errors(self) -> None:
        """Test catching all errors with DavinciMonetError."""
        all_errors = [
            ConfigValidationError("test"),
            DataNotFoundError("test"),
            GeometryMismatchError("test"),
            PlotConfigError("test"),
            InsufficientDataError("test"),
            StageExecutionError("test"),
        ]
        for error in all_errors:
            with pytest.raises(DavinciMonetError):
                raise error
