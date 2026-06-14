"""Statistical metrics for paired source comparison.

This module provides individual metric implementations for comparing a
``y`` source against an ``x`` source.

All metrics follow the convention:
- x: reference values
- y: comparison values
- Positive bias means y > x
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import xarray as xr

from davinci_monet.core.registry import statistic_registry

# =============================================================================
# Base Classes
# =============================================================================


@dataclass
class MetricResult:
    """Result container for a calculated metric.

    Attributes
    ----------
    name : str
        Metric abbreviation (e.g., 'MB', 'RMSE').
    long_name : str
        Full metric name.
    value : float
        Calculated value.
    units : str | None
        Units of the metric (if applicable).
    """

    name: str
    long_name: str
    value: float
    units: str | None = None


class BaseMetric(ABC):
    """Abstract base class for statistical metrics.

    All metrics must implement the compute method which takes
    x (reference) and y (comparison) arrays and returns a single value.
    """

    # Override in subclasses
    name: str = "base"
    long_name: str = "Base Metric"
    units: str | None = None

    @abstractmethod
    def compute(
        self,
        x: np.ndarray,
        y: np.ndarray,
        **kwargs: Any,
    ) -> float:
        """Compute the metric.

        Parameters
        ----------
        x
            Reference values (1D array).
        y
            Comparison values (1D array, same shape as x).
        **kwargs
            Metric-specific options.

        Returns
        -------
        float
            Computed metric value.
        """
        ...

    def __call__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        **kwargs: Any,
    ) -> MetricResult:
        """Compute metric and return result object.

        Parameters
        ----------
        x
            Reference values.
        y
            Comparison values.
        **kwargs
            Additional options.

        Returns
        -------
        MetricResult
            Result container with value and metadata.
        """
        value = self.compute(x, y, **kwargs)
        return MetricResult(
            name=self.name,
            long_name=self.long_name,
            value=value,
            units=self.units,
        )


def _prepare_arrays(
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Prepare arrays for computation by removing NaN values.

    Parameters
    ----------
    x, y
        Input arrays.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Cleaned arrays with matching valid indices.
    """
    x = np.asarray(x).flatten()
    y = np.asarray(y).flatten()

    # Remove NaN from both arrays at matching indices
    mask = np.isfinite(x) & np.isfinite(y)
    return x[mask], y[mask]


# =============================================================================
# Basic Statistical Measures
# =============================================================================


@statistic_registry.register("N")
class CountMetric(BaseMetric):
    """Number of valid x-y pairs."""

    name = "N"
    long_name = "Sample Size"
    units = None

    def compute(self, x: np.ndarray, y: np.ndarray, **kwargs: Any) -> float:
        x, y = _prepare_arrays(x, y)
        return float(len(x))


@statistic_registry.register("MX")
class MeanXMetric(BaseMetric):
    """Mean of x values."""

    name = "MX"
    long_name = "Mean X"
    units = None  # Inherits from data

    def compute(self, x: np.ndarray, y: np.ndarray, **kwargs: Any) -> float:
        x, _ = _prepare_arrays(x, y)
        if len(x) == 0:
            return np.nan
        return float(np.mean(x))


@statistic_registry.register("MY")
class MeanYMetric(BaseMetric):
    """Mean of y values."""

    name = "MY"
    long_name = "Mean Y"
    units = None

    def compute(self, x: np.ndarray, y: np.ndarray, **kwargs: Any) -> float:
        _, y = _prepare_arrays(x, y)
        if len(y) == 0:
            return np.nan
        return float(np.mean(y))


@statistic_registry.register("STDX")
class StdXMetric(BaseMetric):
    """Standard deviation of x values."""

    name = "STDX"
    long_name = "X Standard Deviation"
    units = None

    def compute(self, x: np.ndarray, y: np.ndarray, **kwargs: Any) -> float:
        x, _ = _prepare_arrays(x, y)
        if len(x) < 2:
            return np.nan
        return float(np.std(x, ddof=1))


@statistic_registry.register("STDY")
class StdYMetric(BaseMetric):
    """Standard deviation of y values."""

    name = "STDY"
    long_name = "Y Standard Deviation"
    units = None

    def compute(self, x: np.ndarray, y: np.ndarray, **kwargs: Any) -> float:
        _, y = _prepare_arrays(x, y)
        if len(y) < 2:
            return np.nan
        return float(np.std(y, ddof=1))


@statistic_registry.register("MdnX")
class MedianXMetric(BaseMetric):
    """Median of x values."""

    name = "MdnX"
    long_name = "Median X"
    units = None

    def compute(self, x: np.ndarray, y: np.ndarray, **kwargs: Any) -> float:
        x, _ = _prepare_arrays(x, y)
        if len(x) == 0:
            return np.nan
        return float(np.median(x))


@statistic_registry.register("MdnY")
class MedianYMetric(BaseMetric):
    """Median of y values."""

    name = "MdnY"
    long_name = "Median Y"
    units = None

    def compute(self, x: np.ndarray, y: np.ndarray, **kwargs: Any) -> float:
        _, y = _prepare_arrays(x, y)
        if len(y) == 0:
            return np.nan
        return float(np.median(y))


# =============================================================================
# Bias Metrics
# =============================================================================


@statistic_registry.register("MB")
class MeanBiasMetric(BaseMetric):
    """Mean Bias: average difference (y - x).

    MB = (1/N) * Σ(y - x)

    Positive values indicate y values are higher than x values.
    """

    name = "MB"
    long_name = "Mean Bias"
    units = None

    def compute(self, x: np.ndarray, y: np.ndarray, **kwargs: Any) -> float:
        x, y = _prepare_arrays(x, y)
        if len(x) == 0:
            return np.nan
        return float(np.mean(y - x))


@statistic_registry.register("MdnB")
class MedianBiasMetric(BaseMetric):
    """Median Bias: median difference (y - x)."""

    name = "MdnB"
    long_name = "Median Bias"
    units = None

    def compute(self, x: np.ndarray, y: np.ndarray, **kwargs: Any) -> float:
        x, y = _prepare_arrays(x, y)
        if len(x) == 0:
            return np.nan
        return float(np.median(y - x))


@statistic_registry.register("NMB")
class NormalizedMeanBiasMetric(BaseMetric):
    """Normalized Mean Bias (%).

    NMB = 100 * Σ(y - x) / Σ(x)
    """

    name = "NMB"
    long_name = "Normalized Mean Bias"
    units = "%"

    def compute(self, x: np.ndarray, y: np.ndarray, **kwargs: Any) -> float:
        x, y = _prepare_arrays(x, y)
        if len(x) == 0:
            return np.nan
        sum_x = np.sum(x)
        if sum_x == 0:
            return np.nan
        return float(100.0 * np.sum(y - x) / sum_x)


@statistic_registry.register("NMdnB")
class NormalizedMedianBiasMetric(BaseMetric):
    """Normalized Median Bias (%)."""

    name = "NMdnB"
    long_name = "Normalized Median Bias"
    units = "%"

    def compute(self, x: np.ndarray, y: np.ndarray, **kwargs: Any) -> float:
        x, y = _prepare_arrays(x, y)
        if len(x) == 0:
            return np.nan
        mdn_x = np.median(x)
        if mdn_x == 0:
            return np.nan
        return float(100.0 * np.median(y - x) / mdn_x)


@statistic_registry.register("FB")
class FractionalBiasMetric(BaseMetric):
    """Fractional Bias (%).

    FB = 200 * (mean(y) - mean(x)) / (mean(y) + mean(x))

    Range: -200% to +200%
    """

    name = "FB"
    long_name = "Fractional Bias"
    units = "%"

    def compute(self, x: np.ndarray, y: np.ndarray, **kwargs: Any) -> float:
        x, y = _prepare_arrays(x, y)
        if len(x) == 0:
            return np.nan
        mean_x = np.mean(x)
        mean_y = np.mean(y)
        denom = mean_x + mean_y
        if denom == 0:
            return np.nan
        return float(200.0 * (mean_y - mean_x) / denom)


@statistic_registry.register("MNB")
class MeanNormalizedBiasMetric(BaseMetric):
    """Mean Normalized Bias (%).

    MNB = 100 * (1/N) * Σ((y - x) / x)
    """

    name = "MNB"
    long_name = "Mean Normalized Bias"
    units = "%"

    def compute(self, x: np.ndarray, y: np.ndarray, **kwargs: Any) -> float:
        x, y = _prepare_arrays(x, y)
        # Exclude zero x values.
        mask = x != 0
        x = x[mask]
        y = y[mask]
        if len(x) == 0:
            return np.nan
        return float(100.0 * np.mean((y - x) / x))


# =============================================================================
# Error Metrics
# =============================================================================


@statistic_registry.register("ME")
class MeanErrorMetric(BaseMetric):
    """Mean (Gross) Error: average absolute difference.

    ME = (1/N) * Σ|y - x|
    """

    name = "ME"
    long_name = "Mean Error"
    units = None

    def compute(self, x: np.ndarray, y: np.ndarray, **kwargs: Any) -> float:
        x, y = _prepare_arrays(x, y)
        if len(x) == 0:
            return np.nan
        return float(np.mean(np.abs(y - x)))


@statistic_registry.register("MdnE")
class MedianErrorMetric(BaseMetric):
    """Median (Gross) Error."""

    name = "MdnE"
    long_name = "Median Error"
    units = None

    def compute(self, x: np.ndarray, y: np.ndarray, **kwargs: Any) -> float:
        x, y = _prepare_arrays(x, y)
        if len(x) == 0:
            return np.nan
        return float(np.median(np.abs(y - x)))


@statistic_registry.register("RMSE")
class RMSEMetric(BaseMetric):
    """Root Mean Square Error.

    RMSE = sqrt((1/N) * Σ(y - x)²)
    """

    name = "RMSE"
    long_name = "Root Mean Square Error"
    units = None

    def compute(self, x: np.ndarray, y: np.ndarray, **kwargs: Any) -> float:
        x, y = _prepare_arrays(x, y)
        if len(x) == 0:
            return np.nan
        return float(np.sqrt(np.mean((y - x) ** 2)))


@statistic_registry.register("NME")
class NormalizedMeanErrorMetric(BaseMetric):
    """Normalized Mean Error (%).

    NME = 100 * Σ|y - x| / Σ(x)
    """

    name = "NME"
    long_name = "Normalized Mean Error"
    units = "%"

    def compute(self, x: np.ndarray, y: np.ndarray, **kwargs: Any) -> float:
        x, y = _prepare_arrays(x, y)
        if len(x) == 0:
            return np.nan
        sum_x = np.sum(x)
        if sum_x == 0:
            return np.nan
        return float(100.0 * np.sum(np.abs(y - x)) / sum_x)


@statistic_registry.register("FE")
class FractionalErrorMetric(BaseMetric):
    """Fractional Error (%).

    FE = 200 * (1/N) * Σ|y - x| / (y + x)

    Range: 0% to 200%
    """

    name = "FE"
    long_name = "Fractional Error"
    units = "%"

    def compute(self, x: np.ndarray, y: np.ndarray, **kwargs: Any) -> float:
        x, y = _prepare_arrays(x, y)
        if len(x) == 0:
            return np.nan
        denom = y + x
        # Exclude where denominator is zero
        mask = denom != 0
        if not np.any(mask):
            return np.nan
        return float(200.0 * np.mean(np.abs(y[mask] - x[mask]) / denom[mask]))


@statistic_registry.register("MNE")
class MeanNormalizedErrorMetric(BaseMetric):
    """Mean Normalized (Gross) Error (%).

    MNE = 100 * (1/N) * Σ(|y - x| / x)
    """

    name = "MNE"
    long_name = "Mean Normalized Error"
    units = "%"

    def compute(self, x: np.ndarray, y: np.ndarray, **kwargs: Any) -> float:
        x, y = _prepare_arrays(x, y)
        # Exclude zero x
        mask = x != 0
        x = x[mask]
        y = y[mask]
        if len(x) == 0:
            return np.nan
        return float(100.0 * np.mean(np.abs(y - x) / x))


# =============================================================================
# Correlation and Agreement Metrics
# =============================================================================


@statistic_registry.register("R")
class CorrelationMetric(BaseMetric):
    """Pearson Correlation Coefficient.

    R = cov(x, y) / (std(x) * std(y))

    Range: -1 to +1
    """

    name = "R"
    long_name = "Correlation Coefficient"
    units = None

    def compute(self, x: np.ndarray, y: np.ndarray, **kwargs: Any) -> float:
        x, y = _prepare_arrays(x, y)
        if len(x) < 2:
            return np.nan
        # Check for zero variance
        if np.std(x) == 0 or np.std(y) == 0:
            return np.nan
        return float(np.corrcoef(x, y)[0, 1])


@statistic_registry.register("R2")
class R2Metric(BaseMetric):
    """Coefficient of Determination (R²).

    R² = R^2 (square of correlation coefficient)

    Range: 0 to 1
    """

    name = "R2"
    long_name = "Coefficient of Determination"
    units = None

    def compute(self, x: np.ndarray, y: np.ndarray, **kwargs: Any) -> float:
        x, y = _prepare_arrays(x, y)
        if len(x) < 2:
            return np.nan
        if np.std(x) == 0 or np.std(y) == 0:
            return np.nan
        r = np.corrcoef(x, y)[0, 1]
        return float(r**2)


@statistic_registry.register("IOA")
class IndexOfAgreementMetric(BaseMetric):
    """Index of Agreement (Willmott, 1981).

    IOA = 1 - Σ(y - x)² / Σ(|y - mean(x)| + |x - mean(x)|)²

    Range: 0 to 1, where 1 indicates perfect agreement.
    """

    name = "IOA"
    long_name = "Index of Agreement"
    units = None

    def compute(self, x: np.ndarray, y: np.ndarray, **kwargs: Any) -> float:
        x, y = _prepare_arrays(x, y)
        if len(x) == 0:
            return np.nan

        mean_x = np.mean(x)
        numerator = np.sum((y - x) ** 2)
        denominator = np.sum((np.abs(y - mean_x) + np.abs(x - mean_x)) ** 2)

        if denominator == 0:
            return np.nan
        return float(1.0 - numerator / denominator)


@statistic_registry.register("d1")
class ModifiedIndexOfAgreementMetric(BaseMetric):
    """Modified Index of Agreement (d1).

    d1 = 1 - Σ|y - x| / Σ(|y - mean(x)| + |x - mean(x)|)

    Range: 0 to 1
    """

    name = "d1"
    long_name = "Modified Index of Agreement"
    units = None

    def compute(self, x: np.ndarray, y: np.ndarray, **kwargs: Any) -> float:
        x, y = _prepare_arrays(x, y)
        if len(x) == 0:
            return np.nan

        mean_x = np.mean(x)
        numerator = np.sum(np.abs(y - x))
        denominator = np.sum(np.abs(y - mean_x) + np.abs(x - mean_x))

        if denominator == 0:
            return np.nan
        return float(1.0 - numerator / denominator)


@statistic_registry.register("E1")
class ModifiedCoefficientOfEfficiencyMetric(BaseMetric):
    """Modified Coefficient of Efficiency (E1).

    E1 = 1 - Σ|y - x| / Σ|x - mean(x)|

    Range: -∞ to 1
    """

    name = "E1"
    long_name = "Modified Coefficient of Efficiency"
    units = None

    def compute(self, x: np.ndarray, y: np.ndarray, **kwargs: Any) -> float:
        x, y = _prepare_arrays(x, y)
        if len(x) == 0:
            return np.nan

        mean_x = np.mean(x)
        numerator = np.sum(np.abs(y - x))
        denominator = np.sum(np.abs(x - mean_x))

        if denominator == 0:
            return np.nan
        return float(1.0 - numerator / denominator)


@statistic_registry.register("AC")
class AnomalyCorrelationMetric(BaseMetric):
    """Anomaly Correlation.

    AC = Σ((x - mean(x)) * (y - mean(y))) /
         sqrt(Σ(x - mean(x))² * Σ(y - mean(y))²)

    Range: -1 to +1
    """

    name = "AC"
    long_name = "Anomaly Correlation"
    units = None

    def compute(self, x: np.ndarray, y: np.ndarray, **kwargs: Any) -> float:
        x, y = _prepare_arrays(x, y)
        if len(x) < 2:
            return np.nan

        x_anom = x - np.mean(x)
        y_anom = y - np.mean(y)

        numerator = np.sum(x_anom * y_anom)
        denominator = np.sqrt(np.sum(x_anom**2) * np.sum(y_anom**2))

        if denominator == 0:
            return np.nan
        return float(numerator / denominator)


# =============================================================================
# Ratio Metrics
# =============================================================================


@statistic_registry.register("RM")
class MeanRatioMetric(BaseMetric):
    """Mean Ratio (x/y)."""

    name = "RM"
    long_name = "Mean Ratio"
    units = None

    def compute(self, x: np.ndarray, y: np.ndarray, **kwargs: Any) -> float:
        x, y = _prepare_arrays(x, y)
        # Exclude zero y values
        mask = y != 0
        x = x[mask]
        y = y[mask]
        if len(x) == 0:
            return np.nan
        return float(np.mean(x / y))


@statistic_registry.register("RMdn")
class MedianRatioMetric(BaseMetric):
    """Median Ratio (x/y)."""

    name = "RMdn"
    long_name = "Median Ratio"
    units = None

    def compute(self, x: np.ndarray, y: np.ndarray, **kwargs: Any) -> float:
        x, y = _prepare_arrays(x, y)
        # Exclude zero y values
        mask = y != 0
        x = x[mask]
        y = y[mask]
        if len(x) == 0:
            return np.nan
        return float(np.median(x / y))


# =============================================================================
# Default Metric Sets
# =============================================================================

#: Standard set of metrics for general evaluation
STANDARD_METRICS = ["N", "MX", "MY", "MB", "RMSE", "R", "R2", "NMB", "NME", "IOA"]

#: Full set of all implemented metrics
ALL_METRICS = [
    # Basic
    "N",
    "MX",
    "MY",
    "STDX",
    "STDY",
    "MdnX",
    "MdnY",
    # Bias
    "MB",
    "MdnB",
    "NMB",
    "NMdnB",
    "FB",
    "MNB",
    # Error
    "ME",
    "MdnE",
    "RMSE",
    "NME",
    "FE",
    "MNE",
    # Correlation/Agreement
    "R",
    "R2",
    "IOA",
    "d1",
    "E1",
    "AC",
    # Ratio
    "RM",
    "RMdn",
]

#: Metrics that are expressed as percentages
PERCENTAGE_METRICS = {"NMB", "NMdnB", "FB", "MNB", "NME", "FE", "MNE"}


def get_metric(name: str) -> BaseMetric:
    """Get a metric instance by name.

    Parameters
    ----------
    name
        Metric name (e.g., 'MB', 'RMSE').

    Returns
    -------
    BaseMetric
        Metric instance.

    Raises
    ------
    ComponentNotFoundError
        If metric is not registered.
    """
    metric_cls = statistic_registry.get(name)
    return metric_cls()


def list_metrics() -> list[str]:
    """List all registered metric names.

    Returns
    -------
    list[str]
        Sorted list of metric names.
    """
    return statistic_registry.list()


def compute_metric(
    name: str,
    x: np.ndarray,
    y: np.ndarray,
    **kwargs: Any,
) -> float:
    """Compute a single metric.

    Parameters
    ----------
    name
        Metric name.
    x
        Reference values.
    y
        Comparison values.
    **kwargs
        Additional options.

    Returns
    -------
    float
        Computed metric value.
    """
    metric = get_metric(name)
    return metric.compute(x, y, **kwargs)
