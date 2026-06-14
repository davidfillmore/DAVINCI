"""Stats edge case tests for coverage.

Tests the uncovered branches in metrics.py: empty arrays, single-point,
all-NaN, and all-identical values.
"""

from __future__ import annotations

import numpy as np
import pytest

from davinci_monet.stats.metrics import statistic_registry

# =============================================================================
# Fixtures
# =============================================================================

CORE_METRICS = [
    "N",
    "MB",
    "RMSE",
    "R",
    "NMB",
    "NME",
    "IOA",
    "MG",
    "MD",
    "STDG",
    "STDD",
    "MdnG",
    "MdnD",
]


def _get_metric(name: str):
    """Get an instantiated metric from the registry."""
    metric_cls = statistic_registry.get(name)
    return metric_cls()


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEmptyArrays:
    """Metrics with no valid data after NaN removal."""

    @pytest.mark.parametrize("metric_name", CORE_METRICS)
    def test_all_nan(self, metric_name: str) -> None:
        geometry = np.array([np.nan, np.nan, np.nan])
        dataset = np.array([np.nan, np.nan, np.nan])

        metric = _get_metric(metric_name)
        result = metric.compute(geometry, dataset)

        if metric_name == "N":
            assert result == 0.0
        else:
            assert np.isnan(result), f"{metric_name} should be NaN for all-NaN input, got {result}"

    @pytest.mark.parametrize("metric_name", CORE_METRICS)
    def test_empty_arrays(self, metric_name: str) -> None:
        geometry = np.array([])
        dataset = np.array([])

        metric = _get_metric(metric_name)
        result = metric.compute(geometry, dataset)

        if metric_name == "N":
            assert result == 0.0
        else:
            assert np.isnan(result), f"{metric_name} should be NaN for empty input, got {result}"


class TestSinglePoint:
    """Metrics with exactly one valid data point."""

    @pytest.mark.parametrize("metric_name", CORE_METRICS)
    def test_single_point(self, metric_name: str) -> None:
        geometry = np.array([50.0])
        dataset = np.array([55.0])

        metric = _get_metric(metric_name)
        result = metric.compute(geometry, dataset)

        if metric_name == "N":
            assert result == 1.0
        elif metric_name in ("STDG", "STDD", "R"):
            # Std dev and correlation undefined for single point
            assert np.isnan(result), f"{metric_name} should be NaN for single point"
        elif metric_name == "MB":
            assert result == pytest.approx(5.0)
        elif metric_name in ("MG", "MdnG"):
            assert result == pytest.approx(50.0)
        elif metric_name in ("MD", "MdnD"):
            assert result == pytest.approx(55.0)
        else:
            # RMSE, NMB, NME, IOA — just verify they return a finite number
            assert np.isfinite(result), f"{metric_name} should be finite for single point"


class TestIdenticalValues:
    """Metrics where geometry == dataset exactly (zero bias, zero variance scenarios)."""

    def test_perfect_match(self) -> None:
        geometry = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        dataset = geometry.copy()

        assert _get_metric("MB").compute(geometry, dataset) == pytest.approx(0.0, abs=1e-10)
        assert _get_metric("RMSE").compute(geometry, dataset) == pytest.approx(0.0, abs=1e-10)
        assert _get_metric("NMB").compute(geometry, dataset) == pytest.approx(0.0, abs=1e-10)
        assert _get_metric("NME").compute(geometry, dataset) == pytest.approx(0.0, abs=1e-10)
        assert _get_metric("R").compute(geometry, dataset) == pytest.approx(1.0)
        assert _get_metric("IOA").compute(geometry, dataset) == pytest.approx(1.0)

    def test_constant_geometry_and_dataset(self) -> None:
        """All values identical — std dev is 0, R is undefined."""
        geometry = np.array([42.0, 42.0, 42.0, 42.0])
        dataset = np.array([42.0, 42.0, 42.0, 42.0])

        assert _get_metric("STDG").compute(geometry, dataset) == pytest.approx(0.0, abs=1e-10)
        assert _get_metric("STDD").compute(geometry, dataset) == pytest.approx(0.0, abs=1e-10)
        # R is undefined when variance is 0
        r = _get_metric("R").compute(geometry, dataset)
        assert np.isnan(r) or r == pytest.approx(1.0)


class TestMixedNaN:
    """Arrays with some NaN values — valid pairs should still compute."""

    def test_partial_nan(self) -> None:
        geometry = np.array([10.0, np.nan, 30.0, 40.0, np.nan])
        dataset = np.array([12.0, 20.0, np.nan, 42.0, 50.0])

        # Only indices 0 and 3 have both valid: (10,12) and (40,42)
        assert _get_metric("N").compute(geometry, dataset) == 2.0
        assert _get_metric("MB").compute(geometry, dataset) == pytest.approx(2.0)
        assert _get_metric("MG").compute(geometry, dataset) == pytest.approx(25.0)
        assert _get_metric("MD").compute(geometry, dataset) == pytest.approx(27.0)
