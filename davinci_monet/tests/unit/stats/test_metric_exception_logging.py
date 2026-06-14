"""Tests for silent-stats-failure fix (WS5-A item 1).

Verifies that a metric that raises an exception is:
1. Logged at WARNING level with the metric name and the exception message.
2. Yields NaN in the result rather than crashing the table.

Note on caplog isolation: some other tests call configure_logging(propagate=False)
which sets davinci_monet logger propagate=False globally for the process.  Our
tests use monkeypatch to restore propagate=True on the davinci_monet logger so
that caplog (attached to the root logger) sees the records regardless of suite
ordering.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

from davinci_monet.stats.calculator import StatisticsCalculator, quick_stats

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_broken_get_metric(exc: Exception):
    """Return a drop-in replacement for get_metric that always raises *exc*."""

    def _broken_get_metric(name: str):  # noqa: ANN202
        raise exc

    return _broken_get_metric


def _restore_propagate(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure davinci_monet logger propagates to root so caplog sees records.

    Some tests in the suite call configure_logging(propagate=False), which
    sets logging.getLogger('davinci_monet').propagate = False globally.  This
    breaks caplog's root-logger handler.  We patch it back for this module.
    """
    dm_logger = logging.getLogger("davinci_monet")
    monkeypatch.setattr(dm_logger, "propagate", True)


# ---------------------------------------------------------------------------
# Tests for StatisticsCalculator._compute_metrics
# ---------------------------------------------------------------------------


class TestComputeMetricsLogsAndNan:
    """Broken metrics are logged and yield NaN without crashing."""

    def _setup(
        self,
        monkeypatch: pytest.MonkeyPatch,
        exc: Exception | None = None,
    ) -> tuple[StatisticsCalculator, np.ndarray, np.ndarray]:
        exc = exc or RuntimeError("simulated metric failure")
        _restore_propagate(monkeypatch)
        monkeypatch.setattr(
            "davinci_monet.stats.calculator.get_metric",
            _make_broken_get_metric(exc),
        )
        calc = StatisticsCalculator()
        geometry = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        dataset = np.array([1.1, 2.1, 3.1, 4.1, 5.1])
        return calc, geometry, dataset

    def test_warning_logged_with_metric_name(
        self,
        caplog: pytest.LogCaptureFixture,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """WARNING record includes the metric name."""
        calc, geometry, dataset = self._setup(monkeypatch)

        with caplog.at_level(logging.WARNING, logger="davinci_monet.stats.calculator"):
            calc._compute_metrics(geometry, dataset, ["FAKE_METRIC"])

        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert warnings, "Expected at least one WARNING log record"
        assert any("FAKE_METRIC" in r.message for r in warnings)

    def test_warning_contains_exception_message(
        self,
        caplog: pytest.LogCaptureFixture,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """WARNING record includes the exception text."""
        calc, geometry, dataset = self._setup(monkeypatch)

        with caplog.at_level(logging.WARNING, logger="davinci_monet.stats.calculator"):
            calc._compute_metrics(geometry, dataset, ["FAKE_METRIC"])

        joined = " ".join(r.message for r in caplog.records)
        assert "simulated metric failure" in joined

    def test_result_is_nan(
        self,
        caplog: pytest.LogCaptureFixture,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Broken metric yields NaN, not a crash."""
        calc, geometry, dataset = self._setup(monkeypatch)

        with caplog.at_level(logging.WARNING, logger="davinci_monet.stats.calculator"):
            results = calc._compute_metrics(geometry, dataset, ["FAKE_METRIC"])

        assert "FAKE_METRIC" in results
        assert np.isnan(results["FAKE_METRIC"])

    def test_no_crash_no_propagation(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Exception in metric does not propagate out of _compute_metrics."""
        calc, geometry, dataset = self._setup(monkeypatch)
        # Should not raise
        result = calc._compute_metrics(geometry, dataset, ["FAKE_METRIC"])
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# Tests for quick_stats
# ---------------------------------------------------------------------------


class TestQuickStatsLogsAndNan:
    """quick_stats broken-metric path is also logged."""

    def test_warning_logged(
        self,
        caplog: pytest.LogCaptureFixture,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _restore_propagate(monkeypatch)
        monkeypatch.setattr(
            "davinci_monet.stats.calculator.get_metric",
            _make_broken_get_metric(ValueError("bad metric value")),
        )
        geometry = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        dataset = np.array([1.1, 2.1, 3.1, 4.1, 5.1])

        with caplog.at_level(logging.WARNING, logger="davinci_monet.stats.calculator"):
            results = quick_stats(geometry, dataset, metrics=["FAKE_Q"])

        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert warnings
        assert any("FAKE_Q" in r.message for r in warnings)

    def test_nan_returned(
        self,
        caplog: pytest.LogCaptureFixture,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _restore_propagate(monkeypatch)
        monkeypatch.setattr(
            "davinci_monet.stats.calculator.get_metric",
            _make_broken_get_metric(ValueError("bad metric value")),
        )
        geometry = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        dataset = np.array([1.1, 2.1, 3.1, 4.1, 5.1])

        with caplog.at_level(logging.WARNING, logger="davinci_monet.stats.calculator"):
            results = quick_stats(geometry, dataset, metrics=["FAKE_Q"])

        assert np.isnan(results["FAKE_Q"])
