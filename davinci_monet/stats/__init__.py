"""Statistics module for DAVINCI.

This module provides comprehensive statistical analysis capabilities
for paired x and y values.

Quick Start
-----------
>>> from davinci_monet.stats import calculate_statistics, quick_stats
>>>
>>> # Calculate statistics from paired data
>>> stats = calculate_statistics(paired_data, "x_o3", "y_o3")
>>>
>>> # Quick stats from arrays
>>> stats = quick_stats(x_array, y_array)
>>> print(f"RMSE: {stats['RMSE']:.2f}")

Available Metrics
-----------------
Basic:
    N, MX, MY, STDX, STDY, MdnX, MdnY

Bias:
    MB, MdnB, NMB, NMdnB, FB, MNB

Error:
    ME, MdnE, RMSE, NME, FE, MNE

Correlation/Agreement:
    R, R2, IOA, d1, E1, AC

Ratio:
    RM, RMdn
"""

# Calculator
from davinci_monet.stats.calculator import (
    StatisticsCalculator,
    StatisticsConfig,
    calculate_statistics,
    quick_stats,
)

# Metrics
from davinci_monet.stats.metrics import (  # Base classes; Metric functions; Metric sets; Individual metrics (commonly used)
    ALL_METRICS,
    PERCENTAGE_METRICS,
    STANDARD_METRICS,
    BaseMetric,
    CorrelationMetric,
    IndexOfAgreementMetric,
    MeanBiasMetric,
    MetricResult,
    NormalizedMeanBiasMetric,
    NormalizedMeanErrorMetric,
    R2Metric,
    RMSEMetric,
    compute_metric,
    get_metric,
    list_metrics,
)

# Output
from davinci_monet.stats.output import (
    METRIC_FULL_NAMES,
    OutputConfig,
    StatisticsFormatter,
    create_comparison_table,
    format_stats_summary,
    get_metric_fullname,
    write_statistics_csv,
    write_statistics_table,
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
