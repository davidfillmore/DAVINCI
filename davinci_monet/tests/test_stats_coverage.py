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
    "MO",
    "MP",
    "STDO",
    "STDP",
    "MdnO",
    "MdnP",
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
        obs = np.array([np.nan, np.nan, np.nan])
        mod = np.array([np.nan, np.nan, np.nan])

        metric = _get_metric(metric_name)
        result = metric.compute(obs, mod)

        if metric_name == "N":
            assert result == 0.0
        else:
            assert np.isnan(result), f"{metric_name} should be NaN for all-NaN input, got {result}"

    @pytest.mark.parametrize("metric_name", CORE_METRICS)
    def test_empty_arrays(self, metric_name: str) -> None:
        obs = np.array([])
        mod = np.array([])

        metric = _get_metric(metric_name)
        result = metric.compute(obs, mod)

        if metric_name == "N":
            assert result == 0.0
        else:
            assert np.isnan(result), f"{metric_name} should be NaN for empty input, got {result}"


class TestSinglePoint:
    """Metrics with exactly one valid data point."""

    @pytest.mark.parametrize("metric_name", CORE_METRICS)
    def test_single_point(self, metric_name: str) -> None:
        obs = np.array([50.0])
        mod = np.array([55.0])

        metric = _get_metric(metric_name)
        result = metric.compute(obs, mod)

        if metric_name == "N":
            assert result == 1.0
        elif metric_name in ("STDO", "STDP", "R"):
            # Std dev and correlation undefined for single point
            assert np.isnan(result), f"{metric_name} should be NaN for single point"
        elif metric_name == "MB":
            assert result == pytest.approx(5.0)
        elif metric_name in ("MO", "MdnO"):
            assert result == pytest.approx(50.0)
        elif metric_name in ("MP", "MdnP"):
            assert result == pytest.approx(55.0)
        else:
            # RMSE, NMB, NME, IOA — just verify they return a finite number
            assert np.isfinite(result), f"{metric_name} should be finite for single point"


class TestIdenticalValues:
    """Metrics where obs == mod exactly (zero bias, zero variance scenarios)."""

    def test_perfect_match(self) -> None:
        obs = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        mod = obs.copy()

        assert _get_metric("MB").compute(obs, mod) == pytest.approx(0.0, abs=1e-10)
        assert _get_metric("RMSE").compute(obs, mod) == pytest.approx(0.0, abs=1e-10)
        assert _get_metric("NMB").compute(obs, mod) == pytest.approx(0.0, abs=1e-10)
        assert _get_metric("NME").compute(obs, mod) == pytest.approx(0.0, abs=1e-10)
        assert _get_metric("R").compute(obs, mod) == pytest.approx(1.0)
        assert _get_metric("IOA").compute(obs, mod) == pytest.approx(1.0)

    def test_constant_obs_and_mod(self) -> None:
        """All values identical — std dev is 0, R is undefined."""
        obs = np.array([42.0, 42.0, 42.0, 42.0])
        mod = np.array([42.0, 42.0, 42.0, 42.0])

        assert _get_metric("STDO").compute(obs, mod) == pytest.approx(0.0, abs=1e-10)
        assert _get_metric("STDP").compute(obs, mod) == pytest.approx(0.0, abs=1e-10)
        # R is undefined when variance is 0
        r = _get_metric("R").compute(obs, mod)
        assert np.isnan(r) or r == pytest.approx(1.0)


class TestMixedNaN:
    """Arrays with some NaN values — valid pairs should still compute."""

    def test_partial_nan(self) -> None:
        obs = np.array([10.0, np.nan, 30.0, 40.0, np.nan])
        mod = np.array([12.0, 20.0, np.nan, 42.0, 50.0])

        # Only indices 0 and 3 have both valid: (10,12) and (40,42)
        assert _get_metric("N").compute(obs, mod) == 2.0
        assert _get_metric("MB").compute(obs, mod) == pytest.approx(2.0)
        assert _get_metric("MO").compute(obs, mod) == pytest.approx(25.0)
        assert _get_metric("MP").compute(obs, mod) == pytest.approx(27.0)
