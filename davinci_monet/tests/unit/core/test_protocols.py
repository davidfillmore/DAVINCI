"""Tests for core protocol definitions.

These tests verify that:
1. Protocol classes are properly defined as runtime_checkable
2. The DataGeometry enum contains expected values
3. Classes implementing protocols are correctly recognized
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import pytest

from davinci_monet.core.protocols import (
    Configurable,
    DataGeometry,
    PairingEngine,
    PairingStrategy,
    Plotter,
    SpatialPlotter,
    StatisticMetric,
    StatisticsCalculator,
)


class TestDataGeometry:
    """Tests for DataGeometry enum."""

    def test_geometry_values_exist(self) -> None:
        """Verify all expected geometry types are defined."""
        assert hasattr(DataGeometry, "POINT")
        assert hasattr(DataGeometry, "TRACK")
        assert hasattr(DataGeometry, "PROFILE")
        assert hasattr(DataGeometry, "SWATH")
        assert hasattr(DataGeometry, "GRID")

    def test_geometry_count(self) -> None:
        """Verify exactly 5 geometry types."""
        assert len(DataGeometry) == 5

    def test_geometry_iteration(self) -> None:
        """Verify geometry can be iterated."""
        geometries = list(DataGeometry)
        assert len(geometries) == 5
        assert DataGeometry.POINT in geometries
        assert DataGeometry.GRID in geometries


class TestPairingStrategyProtocol:
    """Tests for PairingStrategy protocol."""

    def test_protocol_is_runtime_checkable(self) -> None:
        """Verify PairingStrategy is runtime_checkable."""

        class MockPointStrategy:
            @property
            def geometry(self) -> DataGeometry:
                return DataGeometry.POINT

            def pair_sources(
                self,
                reference: Any,
                comparand: Any,
                **kwargs: Any,
            ) -> Any:
                return None

        strategy = MockPointStrategy()
        assert isinstance(strategy, PairingStrategy)


class TestPlotterProtocol:
    """Tests for Plotter protocol."""

    def test_protocol_is_runtime_checkable(self) -> None:
        """Verify Plotter is runtime_checkable."""

        class MockPlotter:
            @property
            def name(self) -> str:
                return "timeseries"

            def plot(
                self,
                paired_data: Any,
                obs_var: str,
                model_var: str,
                **kwargs: Any,
            ) -> Any:
                return None

            def save(
                self,
                fig: Any,
                output_path: str | Path,
                **kwargs: Any,
            ) -> Path:
                return Path(output_path)

        plotter = MockPlotter()
        assert isinstance(plotter, Plotter)


class TestStatisticMetricProtocol:
    """Tests for StatisticMetric protocol."""

    def test_protocol_is_runtime_checkable(self) -> None:
        """Verify StatisticMetric is runtime_checkable."""

        class MockMetric:
            @property
            def name(self) -> str:
                return "MB"

            @property
            def long_name(self) -> str:
                return "Mean Bias"

            def compute(
                self,
                obs: Any,
                model: Any,
                **kwargs: Any,
            ) -> float:
                return 0.0

        metric = MockMetric()
        assert isinstance(metric, StatisticMetric)


class TestConfigurableProtocol:
    """Tests for Configurable protocol."""

    def test_protocol_is_runtime_checkable(self) -> None:
        """Verify Configurable is runtime_checkable."""

        class MockConfigurable:
            @classmethod
            def from_config(cls, config: Mapping[str, Any]) -> "MockConfigurable":
                return cls()

        obj = MockConfigurable()
        assert isinstance(obj, Configurable)


class TestPairingEngineProtocol:
    """Tests for PairingEngine protocol."""

    def test_protocol_is_runtime_checkable(self) -> None:
        """Verify PairingEngine is runtime_checkable."""

        class MockEngine:
            def register_strategy(self, strategy: PairingStrategy) -> None:
                pass

            def pair_sources(
                self,
                reference: Any,
                comparand: Any,
                **kwargs: Any,
            ) -> Any:
                return None

        engine = MockEngine()
        assert isinstance(engine, PairingEngine)


class TestStatisticsCalculatorProtocol:
    """Tests for StatisticsCalculator protocol."""

    def test_protocol_is_runtime_checkable(self) -> None:
        """Verify StatisticsCalculator is runtime_checkable."""

        class MockCalculator:
            def compute(
                self,
                paired_data: Any,
                obs_var: str,
                model_var: str,
                metrics: Sequence[str] | None = None,
                groupby: str | Sequence[str] | None = None,
                **kwargs: Any,
            ) -> Any:
                return None

        calc = MockCalculator()
        assert isinstance(calc, StatisticsCalculator)


class TestSpatialPlotterProtocol:
    """Tests for SpatialPlotter protocol."""

    def test_protocol_is_runtime_checkable(self) -> None:
        """Verify SpatialPlotter is runtime_checkable."""

        class MockSpatialPlotter:
            @property
            def name(self) -> str:
                return "spatial_bias"

            def plot(
                self,
                paired_data: Any,
                obs_var: str,
                model_var: str,
                domain: tuple[float, float, float, float] | None = None,
                projection: Any | None = None,
                **kwargs: Any,
            ) -> Any:
                return None

        plotter = MockSpatialPlotter()
        assert isinstance(plotter, SpatialPlotter)
