"""Statistical metrics for paired source comparison.

This module provides individual metric implementations for comparing a
``y`` (dataset/model) source against an ``x`` (geometry/reference) source.

All metrics follow the convention:
- x: reference values (geometry source)
- y: comparison values (dataset source)
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
    """Number of valid geometry-dataset pairs."""

    name = "N"
    long_name = "Sample Size"
    units = None

    def compute(self, x: np.ndarray, y: np.ndarray, **kwargs: Any) -> float:
        x, y = _prepare_arrays(x, y)
        return float(len(x))


@statistic_registry.register("MG")
class MeanGeometryMetric(BaseMetric):
    """Mean of geometry values."""

    name = "MG"
    long_name = "Mean Geometry"
    units = None  # Inherits from data

    def compute(self, x: np.ndarray, y: np.ndarray, **kwargs: Any) -> float:
        x, _ = _prepare_arrays(x, y)
        if len(x) == 0:
            return np.nan
        return float(np.mean(x))


@statistic_registry.register("MD")
class MeanDatasetMetric(BaseMetric):
    """Mean of dataset values."""

    name = "MD"
    long_name = "Mean Dataset"
    units = None

    def compute(self, x: np.ndarray, y: np.ndarray, **kwargs: Any) -> float:
        _, y = _prepare_arrays(x, y)
        if len(y) == 0:
            return np.nan
        return float(np.mean(y))


@statistic_registry.register("STDG")
class StdGeometryMetric(BaseMetric):
    """Standard deviation of geometry values."""

    name = "STDG"
    long_name = "Geometry Standard Deviation"
    units = None

    def compute(self, x: np.ndarray, y: np.ndarray, **kwargs: Any) -> float:
        x, _ = _prepare_arrays(x, y)
        if len(x) < 2:
            return np.nan
        return float(np.std(x, ddof=1))


@statistic_registry.register("STDD")
class StdDatasetMetric(BaseMetric):
    """Standard deviation of dataset values."""

    name = "STDD"
    long_name = "Dataset Standard Deviation"
    units = None

    def compute(self, x: np.ndarray, y: np.ndarray, **kwargs: Any) -> float:
        _, y = _prepare_arrays(x, y)
        if len(y) < 2:
            return np.nan
        return float(np.std(y, ddof=1))


@statistic_registry.register("MdnG")
class MedianGeometryMetric(BaseMetric):
    """Median of geometry values."""

    name = "MdnG"
    long_name = "Median Geometry"
    units = None

    def compute(self, x: np.ndarray, y: np.ndarray, **kwargs: Any) -> float:
        x, _ = _prepare_arrays(x, y)
        if len(x) == 0:
            return np.nan
        return float(np.median(x))


@statistic_registry.register("MdnD")
class MedianDatasetMetric(BaseMetric):
    """Median of dataset values."""

    name = "MdnD"
    long_name = "Median Dataset"
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
    """Mean Bias: average difference (dataset - geometry).

    MB = (1/N) * Σ(dataset - geometry)

    Positive values indicate dataset values are higher than geometry values.
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
    """Median Bias: median difference (dataset - geometry)."""

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

    NMB = 100 * Σ(dataset - geometry) / Σ(geometry)
    """

    name = "NMB"
    long_name = "Normalized Mean Bias"
    units = "%"

    def compute(self, x: np.ndarray, y: np.ndarray, **kwargs: Any) -> float:
        x, y = _prepare_arrays(x, y)
        if len(x) == 0:
            return np.nan
        sum_geometry = np.sum(x)
        if sum_geometry == 0:
            return np.nan
        return float(100.0 * np.sum(y - x) / sum_geometry)


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
        mdn_geometry = np.median(x)
        if mdn_geometry == 0:
            return np.nan
        return float(100.0 * np.median(y - x) / mdn_geometry)


@statistic_registry.register("FB")
class FractionalBiasMetric(BaseMetric):
    """Fractional Bias (%).

    FB = 200 * (mean(dataset) - mean(geometry)) / (mean(dataset) + mean(geometry))

    Range: -200% to +200%
    """

    name = "FB"
    long_name = "Fractional Bias"
    units = "%"

    def compute(self, x: np.ndarray, y: np.ndarray, **kwargs: Any) -> float:
        x, y = _prepare_arrays(x, y)
        if len(x) == 0:
            return np.nan
        mean_geometry = np.mean(x)
        mean_dataset = np.mean(y)
        denom = mean_geometry + mean_dataset
        if denom == 0:
            return np.nan
        return float(200.0 * (mean_dataset - mean_geometry) / denom)


@statistic_registry.register("MNB")
class MeanNormalizedBiasMetric(BaseMetric):
    """Mean Normalized Bias (%).

    MNB = 100 * (1/N) * Σ((dataset - geometry) / geometry)
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

    ME = (1/N) * Σ|dataset - geometry|
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

    RMSE = sqrt((1/N) * Σ(dataset - geometry)²)
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

    NME = 100 * Σ|dataset - geometry| / Σ(geometry)
    """

    name = "NME"
    long_name = "Normalized Mean Error"
    units = "%"

    def compute(self, x: np.ndarray, y: np.ndarray, **kwargs: Any) -> float:
        x, y = _prepare_arrays(x, y)
        if len(x) == 0:
            return np.nan
        sum_geometry = np.sum(x)
        if sum_geometry == 0:
            return np.nan
        return float(100.0 * np.sum(np.abs(y - x)) / sum_geometry)


@statistic_registry.register("FE")
class FractionalErrorMetric(BaseMetric):
    """Fractional Error (%).

    FE = 200 * (1/N) * Σ|dataset - geometry| / (dataset + geometry)

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

    MNE = 100 * (1/N) * Σ(|dataset - geometry| / geometry)
    """

    name = "MNE"
    long_name = "Mean Normalized Error"
    units = "%"

    def compute(self, x: np.ndarray, y: np.ndarray, **kwargs: Any) -> float:
        x, y = _prepare_arrays(x, y)
        # Exclude zero datasets
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

    R = cov(geometry, dataset) / (std(geometry) * std(dataset))

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

    IOA = 1 - Σ(dataset - geometry)² / Σ(|dataset - mean(geometry)| + |geometry - mean(geometry)|)²

    Range: 0 to 1, where 1 indicates perfect agreement.
    """

    name = "IOA"
    long_name = "Index of Agreement"
    units = None

    def compute(self, x: np.ndarray, y: np.ndarray, **kwargs: Any) -> float:
        x, y = _prepare_arrays(x, y)
        if len(x) == 0:
            return np.nan

        mean_geometry = np.mean(x)
        numerator = np.sum((y - x) ** 2)
        denominator = np.sum((np.abs(y - mean_geometry) + np.abs(x - mean_geometry)) ** 2)

        if denominator == 0:
            return np.nan
        return float(1.0 - numerator / denominator)


@statistic_registry.register("d1")
class ModifiedIndexOfAgreementMetric(BaseMetric):
    """Modified Index of Agreement (d1).

    d1 = 1 - Σ|dataset - geometry| / Σ(|dataset - mean(geometry)| + |geometry - mean(geometry)|)

    Range: 0 to 1
    """

    name = "d1"
    long_name = "Modified Index of Agreement"
    units = None

    def compute(self, x: np.ndarray, y: np.ndarray, **kwargs: Any) -> float:
        x, y = _prepare_arrays(x, y)
        if len(x) == 0:
            return np.nan

        mean_geometry = np.mean(x)
        numerator = np.sum(np.abs(y - x))
        denominator = np.sum(np.abs(y - mean_geometry) + np.abs(x - mean_geometry))

        if denominator == 0:
            return np.nan
        return float(1.0 - numerator / denominator)


@statistic_registry.register("E1")
class ModifiedCoefficientOfEfficiencyMetric(BaseMetric):
    """Modified Coefficient of Efficiency (E1).

    E1 = 1 - Σ|dataset - geometry| / Σ|geometry - mean(geometry)|

    Range: -∞ to 1
    """

    name = "E1"
    long_name = "Modified Coefficient of Efficiency"
    units = None

    def compute(self, x: np.ndarray, y: np.ndarray, **kwargs: Any) -> float:
        x, y = _prepare_arrays(x, y)
        if len(x) == 0:
            return np.nan

        mean_geometry = np.mean(x)
        numerator = np.sum(np.abs(y - x))
        denominator = np.sum(np.abs(x - mean_geometry))

        if denominator == 0:
            return np.nan
        return float(1.0 - numerator / denominator)


@statistic_registry.register("AC")
class AnomalyCorrelationMetric(BaseMetric):
    """Anomaly Correlation.

    AC = Σ((geometry - mean(geometry)) * (dataset - mean(dataset))) /
         sqrt(Σ(geometry - mean(geometry))² * Σ(dataset - mean(dataset))²)

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
    """Mean Ratio (geometry/dataset)."""

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
    """Median Ratio (geometry/dataset)."""

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
STANDARD_METRICS = ["N", "MG", "MD", "MB", "RMSE", "R", "R2", "NMB", "NME", "IOA"]

#: Full set of all implemented metrics
ALL_METRICS = [
    # Basic
    "N",
    "MG",
    "MD",
    "STDG",
    "STDD",
    "MdnG",
    "MdnD",
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
