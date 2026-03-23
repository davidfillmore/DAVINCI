"""Statistics module for DAVINCI.

This module provides comprehensive statistical analysis capabilities
for evaluating model performance against observations.

Quick Start
-----------
>>> from davinci_monet.stats import calculate_statistics, quick_stats
>>>
>>> # Calculate statistics from paired data
>>> stats = calculate_statistics(paired_data, "obs_o3", "model_o3")
>>>
>>> # Quick stats from arrays
>>> stats = quick_stats(obs_array, model_array)
>>> print(f"RMSE: {stats['RMSE']:.2f}")

Available Metrics
-----------------
Basic:
    N, MO, MP, STDO, STDP, MdnO, MdnP

Bias:
    MB, MdnB, NMB, NMdnB, FB, MNB

Error:
    ME, MdnE, RMSE, NME, FE, MNE

Correlation/Agreement:
    R, R2, IOA, d1, E1, AC

Ratio:
    RM, RMdn
"""

# Metrics
from davinci_monet.stats.metrics import (
    # Base classes
    BaseMetric,
    MetricResult,
    # Metric functions
    get_metric,
    list_metrics,
    compute_metric,
    # Metric sets
    STANDARD_METRICS,
    ALL_METRICS,
    PERCENTAGE_METRICS,
    # Individual metrics (commonly used)
    MeanBiasMetric,
    RMSEMetric,
    CorrelationMetric,
    R2Metric,
    NormalizedMeanBiasMetric,
    NormalizedMeanErrorMetric,
    IndexOfAgreementMetric,
)

# Calculator
from davinci_monet.stats.calculator import (
    StatisticsCalculator,
    StatisticsConfig,
    calculate_statistics,
    quick_stats,
)

# Output
from davinci_monet.stats.output import (
    StatisticsFormatter,
    OutputConfig,
    write_statistics_csv,
    write_statistics_table,
    format_stats_summary,
    create_comparison_table,
    get_metric_fullname,
    METRIC_FULL_NAMES,
)

__all__ = [
    # Base classes
    "BaseMetric",
    "MetricResult",
    # Metric functions
    "get_metric",
    "list_metrics",
    "compute_metric",
    # Metric sets
    "STANDARD_METRICS",
    "ALL_METRICS",
    "PERCENTAGE_METRICS",
    # Individual metrics
    "MeanBiasMetric",
    "RMSEMetric",
    "CorrelationMetric",
    "R2Metric",
    "NormalizedMeanBiasMetric",
    "NormalizedMeanErrorMetric",
    "IndexOfAgreementMetric",
    # Calculator
    "StatisticsCalculator",
    "StatisticsConfig",
    "calculate_statistics",
    "quick_stats",
    # Output
    "StatisticsFormatter",
    "OutputConfig",
    "write_statistics_csv",
    "write_statistics_table",
    "format_stats_summary",
    "create_comparison_table",
    "get_metric_fullname",
    "METRIC_FULL_NAMES",
]
