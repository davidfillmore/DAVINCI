"""Statistical metrics for paired source comparison.

This module provides individual metric implementations for comparing a
comparand source against a reference source.

All metrics follow the convention:
- obs: reference values (legacy parameter name)
- mod: comparand values (legacy parameter name)
- Positive bias means comparand > reference
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
    observation and model arrays and returns a single value.
    """

    # Override in subclasses
    name: str = "base"
    long_name: str = "Base Metric"
    units: str | None = None

    @abstractmethod
    def compute(
        self,
        obs: np.ndarray,
        mod: np.ndarray,
        **kwargs: Any,
    ) -> float:
        """Compute the metric.

        Parameters
        ----------
        obs
            Observation values (1D array).
        mod
            Model values (1D array, same shape as obs).
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
        obs: np.ndarray,
        mod: np.ndarray,
        **kwargs: Any,
    ) -> MetricResult:
        """Compute metric and return result object.

        Parameters
        ----------
        obs
            Observation values.
        mod
            Model values.
        **kwargs
            Additional options.

        Returns
        -------
        MetricResult
            Result container with value and metadata.
        """
        value = self.compute(obs, mod, **kwargs)
        return MetricResult(
            name=self.name,
            long_name=self.long_name,
            value=value,
            units=self.units,
        )


def _prepare_arrays(
    obs: np.ndarray,
    mod: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Prepare arrays for computation by removing NaN values.

    Parameters
    ----------
    obs, mod
        Input arrays.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Cleaned arrays with matching valid indices.
    """
    obs = np.asarray(obs).flatten()
    mod = np.asarray(mod).flatten()

    # Remove NaN from both arrays at matching indices
    mask = np.isfinite(obs) & np.isfinite(mod)
    return obs[mask], mod[mask]


# =============================================================================
# Basic Statistical Measures
# =============================================================================


@statistic_registry.register("N")
class CountMetric(BaseMetric):
    """Number of valid observation-model pairs."""

    name = "N"
    long_name = "Sample Size"
    units = None

    def compute(self, obs: np.ndarray, mod: np.ndarray, **kwargs: Any) -> float:
        obs, mod = _prepare_arrays(obs, mod)
        return float(len(obs))


@statistic_registry.register("MO")
class MeanObsMetric(BaseMetric):
    """Mean of reference values."""

    name = "MO"
    long_name = "Mean Reference"
    units = None  # Inherits from data

    def compute(self, obs: np.ndarray, mod: np.ndarray, **kwargs: Any) -> float:
        obs, _ = _prepare_arrays(obs, mod)
        if len(obs) == 0:
            return np.nan
        return float(np.mean(obs))


@statistic_registry.register("MP")
class MeanModMetric(BaseMetric):
    """Mean of comparand values."""

    name = "MP"
    long_name = "Mean Comparand"
    units = None

    def compute(self, obs: np.ndarray, mod: np.ndarray, **kwargs: Any) -> float:
        _, mod = _prepare_arrays(obs, mod)
        if len(mod) == 0:
            return np.nan
        return float(np.mean(mod))


@statistic_registry.register("STDO")
class StdObsMetric(BaseMetric):
    """Standard deviation of reference values."""

    name = "STDO"
    long_name = "Reference Standard Deviation"
    units = None

    def compute(self, obs: np.ndarray, mod: np.ndarray, **kwargs: Any) -> float:
        obs, _ = _prepare_arrays(obs, mod)
        if len(obs) < 2:
            return np.nan
        return float(np.std(obs, ddof=1))


@statistic_registry.register("STDP")
class StdModMetric(BaseMetric):
    """Standard deviation of comparand values."""

    name = "STDP"
    long_name = "Comparand Standard Deviation"
    units = None

    def compute(self, obs: np.ndarray, mod: np.ndarray, **kwargs: Any) -> float:
        _, mod = _prepare_arrays(obs, mod)
        if len(mod) < 2:
            return np.nan
        return float(np.std(mod, ddof=1))


@statistic_registry.register("MdnO")
class MedianObsMetric(BaseMetric):
    """Median of observations."""

    name = "MdnO"
    long_name = "Median Observation"
    units = None

    def compute(self, obs: np.ndarray, mod: np.ndarray, **kwargs: Any) -> float:
        obs, _ = _prepare_arrays(obs, mod)
        if len(obs) == 0:
            return np.nan
        return float(np.median(obs))


@statistic_registry.register("MdnP")
class MedianModMetric(BaseMetric):
    """Median of model predictions."""

    name = "MdnP"
    long_name = "Median Model"
    units = None

    def compute(self, obs: np.ndarray, mod: np.ndarray, **kwargs: Any) -> float:
        _, mod = _prepare_arrays(obs, mod)
        if len(mod) == 0:
            return np.nan
        return float(np.median(mod))


# =============================================================================
# Bias Metrics
# =============================================================================


@statistic_registry.register("MB")
class MeanBiasMetric(BaseMetric):
    """Mean Bias: average difference (model - observation).

    MB = (1/N) * Σ(mod - obs)

    Positive values indicate model overestimation.
    """

    name = "MB"
    long_name = "Mean Bias"
    units = None

    def compute(self, obs: np.ndarray, mod: np.ndarray, **kwargs: Any) -> float:
        obs, mod = _prepare_arrays(obs, mod)
        if len(obs) == 0:
            return np.nan
        return float(np.mean(mod - obs))


@statistic_registry.register("MdnB")
class MedianBiasMetric(BaseMetric):
    """Median Bias: median difference (model - observation)."""

    name = "MdnB"
    long_name = "Median Bias"
    units = None

    def compute(self, obs: np.ndarray, mod: np.ndarray, **kwargs: Any) -> float:
        obs, mod = _prepare_arrays(obs, mod)
        if len(obs) == 0:
            return np.nan
        return float(np.median(mod - obs))


@statistic_registry.register("NMB")
class NormalizedMeanBiasMetric(BaseMetric):
    """Normalized Mean Bias (%).

    NMB = 100 * Σ(mod - obs) / Σ(obs)
    """

    name = "NMB"
    long_name = "Normalized Mean Bias"
    units = "%"

    def compute(self, obs: np.ndarray, mod: np.ndarray, **kwargs: Any) -> float:
        obs, mod = _prepare_arrays(obs, mod)
        if len(obs) == 0:
            return np.nan
        sum_obs = np.sum(obs)
        if sum_obs == 0:
            return np.nan
        return float(100.0 * np.sum(mod - obs) / sum_obs)


@statistic_registry.register("NMdnB")
class NormalizedMedianBiasMetric(BaseMetric):
    """Normalized Median Bias (%)."""

    name = "NMdnB"
    long_name = "Normalized Median Bias"
    units = "%"

    def compute(self, obs: np.ndarray, mod: np.ndarray, **kwargs: Any) -> float:
        obs, mod = _prepare_arrays(obs, mod)
        if len(obs) == 0:
            return np.nan
        mdn_obs = np.median(obs)
        if mdn_obs == 0:
            return np.nan
        return float(100.0 * np.median(mod - obs) / mdn_obs)


@statistic_registry.register("FB")
class FractionalBiasMetric(BaseMetric):
    """Fractional Bias (%).

    FB = 200 * (mean(mod) - mean(obs)) / (mean(mod) + mean(obs))

    Range: -200% to +200%
    """

    name = "FB"
    long_name = "Fractional Bias"
    units = "%"

    def compute(self, obs: np.ndarray, mod: np.ndarray, **kwargs: Any) -> float:
        obs, mod = _prepare_arrays(obs, mod)
        if len(obs) == 0:
            return np.nan
        mean_obs = np.mean(obs)
        mean_mod = np.mean(mod)
        denom = mean_obs + mean_mod
        if denom == 0:
            return np.nan
        return float(200.0 * (mean_mod - mean_obs) / denom)


@statistic_registry.register("MNB")
class MeanNormalizedBiasMetric(BaseMetric):
    """Mean Normalized Bias (%).

    MNB = 100 * (1/N) * Σ((mod - obs) / obs)
    """

    name = "MNB"
    long_name = "Mean Normalized Bias"
    units = "%"

    def compute(self, obs: np.ndarray, mod: np.ndarray, **kwargs: Any) -> float:
        obs, mod = _prepare_arrays(obs, mod)
        # Exclude zero observations
        mask = obs != 0
        obs = obs[mask]
        mod = mod[mask]
        if len(obs) == 0:
            return np.nan
        return float(100.0 * np.mean((mod - obs) / obs))


# =============================================================================
# Error Metrics
# =============================================================================


@statistic_registry.register("ME")
class MeanErrorMetric(BaseMetric):
    """Mean (Gross) Error: average absolute difference.

    ME = (1/N) * Σ|mod - obs|
    """

    name = "ME"
    long_name = "Mean Error"
    units = None

    def compute(self, obs: np.ndarray, mod: np.ndarray, **kwargs: Any) -> float:
        obs, mod = _prepare_arrays(obs, mod)
        if len(obs) == 0:
            return np.nan
        return float(np.mean(np.abs(mod - obs)))


@statistic_registry.register("MdnE")
class MedianErrorMetric(BaseMetric):
    """Median (Gross) Error."""

    name = "MdnE"
    long_name = "Median Error"
    units = None

    def compute(self, obs: np.ndarray, mod: np.ndarray, **kwargs: Any) -> float:
        obs, mod = _prepare_arrays(obs, mod)
        if len(obs) == 0:
            return np.nan
        return float(np.median(np.abs(mod - obs)))


@statistic_registry.register("RMSE")
class RMSEMetric(BaseMetric):
    """Root Mean Square Error.

    RMSE = sqrt((1/N) * Σ(mod - obs)²)
    """

    name = "RMSE"
    long_name = "Root Mean Square Error"
    units = None

    def compute(self, obs: np.ndarray, mod: np.ndarray, **kwargs: Any) -> float:
        obs, mod = _prepare_arrays(obs, mod)
        if len(obs) == 0:
            return np.nan
        return float(np.sqrt(np.mean((mod - obs) ** 2)))


@statistic_registry.register("NME")
class NormalizedMeanErrorMetric(BaseMetric):
    """Normalized Mean Error (%).

    NME = 100 * Σ|mod - obs| / Σ(obs)
    """

    name = "NME"
    long_name = "Normalized Mean Error"
    units = "%"

    def compute(self, obs: np.ndarray, mod: np.ndarray, **kwargs: Any) -> float:
        obs, mod = _prepare_arrays(obs, mod)
        if len(obs) == 0:
            return np.nan
        sum_obs = np.sum(obs)
        if sum_obs == 0:
            return np.nan
        return float(100.0 * np.sum(np.abs(mod - obs)) / sum_obs)


@statistic_registry.register("FE")
class FractionalErrorMetric(BaseMetric):
    """Fractional Error (%).

    FE = 200 * (1/N) * Σ|mod - obs| / (mod + obs)

    Range: 0% to 200%
    """

    name = "FE"
    long_name = "Fractional Error"
    units = "%"

    def compute(self, obs: np.ndarray, mod: np.ndarray, **kwargs: Any) -> float:
        obs, mod = _prepare_arrays(obs, mod)
        if len(obs) == 0:
            return np.nan
        denom = mod + obs
        # Exclude where denominator is zero
        mask = denom != 0
        if not np.any(mask):
            return np.nan
        return float(200.0 * np.mean(np.abs(mod[mask] - obs[mask]) / denom[mask]))


@statistic_registry.register("MNE")
class MeanNormalizedErrorMetric(BaseMetric):
    """Mean Normalized (Gross) Error (%).

    MNE = 100 * (1/N) * Σ(|mod - obs| / obs)
    """

    name = "MNE"
    long_name = "Mean Normalized Error"
    units = "%"

    def compute(self, obs: np.ndarray, mod: np.ndarray, **kwargs: Any) -> float:
        obs, mod = _prepare_arrays(obs, mod)
        # Exclude zero observations
        mask = obs != 0
        obs = obs[mask]
        mod = mod[mask]
        if len(obs) == 0:
            return np.nan
        return float(100.0 * np.mean(np.abs(mod - obs) / obs))


# =============================================================================
# Correlation and Agreement Metrics
# =============================================================================


@statistic_registry.register("R")
class CorrelationMetric(BaseMetric):
    """Pearson Correlation Coefficient.

    R = cov(obs, mod) / (std(obs) * std(mod))

    Range: -1 to +1
    """

    name = "R"
    long_name = "Correlation Coefficient"
    units = None

    def compute(self, obs: np.ndarray, mod: np.ndarray, **kwargs: Any) -> float:
        obs, mod = _prepare_arrays(obs, mod)
        if len(obs) < 2:
            return np.nan
        # Check for zero variance
        if np.std(obs) == 0 or np.std(mod) == 0:
            return np.nan
        return float(np.corrcoef(obs, mod)[0, 1])


@statistic_registry.register("R2")
class R2Metric(BaseMetric):
    """Coefficient of Determination (R²).

    R² = R^2 (square of correlation coefficient)

    Range: 0 to 1
    """

    name = "R2"
    long_name = "Coefficient of Determination"
    units = None

    def compute(self, obs: np.ndarray, mod: np.ndarray, **kwargs: Any) -> float:
        obs, mod = _prepare_arrays(obs, mod)
        if len(obs) < 2:
            return np.nan
        if np.std(obs) == 0 or np.std(mod) == 0:
            return np.nan
        r = np.corrcoef(obs, mod)[0, 1]
        return float(r**2)


@statistic_registry.register("IOA")
class IndexOfAgreementMetric(BaseMetric):
    """Index of Agreement (Willmott, 1981).

    IOA = 1 - Σ(mod - obs)² / Σ(|mod - mean(obs)| + |obs - mean(obs)|)²

    Range: 0 to 1, where 1 indicates perfect agreement.
    """

    name = "IOA"
    long_name = "Index of Agreement"
    units = None

    def compute(self, obs: np.ndarray, mod: np.ndarray, **kwargs: Any) -> float:
        obs, mod = _prepare_arrays(obs, mod)
        if len(obs) == 0:
            return np.nan

        mean_obs = np.mean(obs)
        numerator = np.sum((mod - obs) ** 2)
        denominator = np.sum((np.abs(mod - mean_obs) + np.abs(obs - mean_obs)) ** 2)

        if denominator == 0:
            return np.nan
        return float(1.0 - numerator / denominator)


@statistic_registry.register("d1")
class ModifiedIndexOfAgreementMetric(BaseMetric):
    """Modified Index of Agreement (d1).

    d1 = 1 - Σ|mod - obs| / Σ(|mod - mean(obs)| + |obs - mean(obs)|)

    Range: 0 to 1
    """

    name = "d1"
    long_name = "Modified Index of Agreement"
    units = None

    def compute(self, obs: np.ndarray, mod: np.ndarray, **kwargs: Any) -> float:
        obs, mod = _prepare_arrays(obs, mod)
        if len(obs) == 0:
            return np.nan

        mean_obs = np.mean(obs)
        numerator = np.sum(np.abs(mod - obs))
        denominator = np.sum(np.abs(mod - mean_obs) + np.abs(obs - mean_obs))

        if denominator == 0:
            return np.nan
        return float(1.0 - numerator / denominator)


@statistic_registry.register("E1")
class ModifiedCoefficientOfEfficiencyMetric(BaseMetric):
    """Modified Coefficient of Efficiency (E1).

    E1 = 1 - Σ|mod - obs| / Σ|obs - mean(obs)|

    Range: -∞ to 1
    """

    name = "E1"
    long_name = "Modified Coefficient of Efficiency"
    units = None

    def compute(self, obs: np.ndarray, mod: np.ndarray, **kwargs: Any) -> float:
        obs, mod = _prepare_arrays(obs, mod)
        if len(obs) == 0:
            return np.nan

        mean_obs = np.mean(obs)
        numerator = np.sum(np.abs(mod - obs))
        denominator = np.sum(np.abs(obs - mean_obs))

        if denominator == 0:
            return np.nan
        return float(1.0 - numerator / denominator)


@statistic_registry.register("AC")
class AnomalyCorrelationMetric(BaseMetric):
    """Anomaly Correlation.

    AC = Σ((obs - mean(obs)) * (mod - mean(mod))) /
         sqrt(Σ(obs - mean(obs))² * Σ(mod - mean(mod))²)

    Range: -1 to +1
    """

    name = "AC"
    long_name = "Anomaly Correlation"
    units = None

    def compute(self, obs: np.ndarray, mod: np.ndarray, **kwargs: Any) -> float:
        obs, mod = _prepare_arrays(obs, mod)
        if len(obs) < 2:
            return np.nan

        obs_anom = obs - np.mean(obs)
        mod_anom = mod - np.mean(mod)

        numerator = np.sum(obs_anom * mod_anom)
        denominator = np.sqrt(np.sum(obs_anom**2) * np.sum(mod_anom**2))

        if denominator == 0:
            return np.nan
        return float(numerator / denominator)


# =============================================================================
# Ratio Metrics
# =============================================================================


@statistic_registry.register("RM")
class MeanRatioMetric(BaseMetric):
    """Mean Ratio (obs/mod)."""

    name = "RM"
    long_name = "Mean Ratio"
    units = None

    def compute(self, obs: np.ndarray, mod: np.ndarray, **kwargs: Any) -> float:
        obs, mod = _prepare_arrays(obs, mod)
        # Exclude zero model values
        mask = mod != 0
        obs = obs[mask]
        mod = mod[mask]
        if len(obs) == 0:
            return np.nan
        return float(np.mean(obs / mod))


@statistic_registry.register("RMdn")
class MedianRatioMetric(BaseMetric):
    """Median Ratio (obs/mod)."""

    name = "RMdn"
    long_name = "Median Ratio"
    units = None

    def compute(self, obs: np.ndarray, mod: np.ndarray, **kwargs: Any) -> float:
        obs, mod = _prepare_arrays(obs, mod)
        # Exclude zero model values
        mask = mod != 0
        obs = obs[mask]
        mod = mod[mask]
        if len(obs) == 0:
            return np.nan
        return float(np.median(obs / mod))


# =============================================================================
# Default Metric Sets
# =============================================================================

#: Standard set of metrics for general evaluation
STANDARD_METRICS = ["N", "MO", "MP", "MB", "RMSE", "R", "R2", "NMB", "NME", "IOA"]

#: Full set of all implemented metrics
ALL_METRICS = [
    # Basic
    "N",
    "MO",
    "MP",
    "STDO",
    "STDP",
    "MdnO",
    "MdnP",
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
    obs: np.ndarray,
    mod: np.ndarray,
    **kwargs: Any,
) -> float:
    """Compute a single metric.

    Parameters
    ----------
    name
        Metric name.
    obs
        Observation values.
    mod
        Model values.
    **kwargs
        Additional options.

    Returns
    -------
    float
        Computed metric value.
    """
    metric = get_metric(name)
    return metric.compute(obs, mod, **kwargs)
