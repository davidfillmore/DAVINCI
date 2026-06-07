"""Tests for the statistics module.

This module tests the statistical metrics, calculator, and output
functionality.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def simple_arrays() -> tuple[np.ndarray, np.ndarray]:
    """Create simple observation and model arrays."""
    np.random.seed(42)
    obs = np.array([10, 20, 30, 40, 50], dtype=float)
    mod = np.array([12, 18, 33, 38, 52], dtype=float)
    return obs, mod


@pytest.fixture
def random_arrays() -> tuple[np.ndarray, np.ndarray]:
    """Create random observation and model arrays."""
    np.random.seed(42)
    obs = np.random.normal(50, 10, 1000)
    mod = obs + np.random.normal(2, 5, 1000)  # Slight positive bias
    return obs, mod


@pytest.fixture
def paired_dataset() -> xr.Dataset:
    """Create a paired dataset for testing."""
    np.random.seed(42)
    n_times = 100
    n_sites = 5

    time = pd.date_range("2023-01-01", periods=n_times, freq="h")
    sites = [f"site_{i}" for i in range(n_sites)]

    obs = np.random.normal(50, 10, (n_times, n_sites))
    mod = obs + np.random.normal(2, 5, (n_times, n_sites))

    ds = xr.Dataset(
        {
            "obs_o3": (["time", "site"], obs, {"units": "ppbv"}),
            "model_o3": (["time", "site"], mod, {"units": "ppbv"}),
        },
        coords={
            "time": time,
            "site": sites,
        },
    )
    return ds


# =============================================================================
# Metric Tests
# =============================================================================


class TestBasicMetrics:
    """Tests for basic statistical measures."""

    def test_count(self, simple_arrays):
        """Test sample count."""
        from davinci_monet.stats import get_metric

        obs, mod = simple_arrays
        metric = get_metric("N")
        result = metric.compute(obs, mod)
        assert result == 5

    def test_mean_obs(self, simple_arrays):
        """Test mean observation."""
        from davinci_monet.stats import get_metric

        obs, mod = simple_arrays
        metric = get_metric("MO")
        result = metric.compute(obs, mod)
        assert result == pytest.approx(30.0)

    def test_mean_mod(self, simple_arrays):
        """Test mean model."""
        from davinci_monet.stats import get_metric

        obs, mod = simple_arrays
        metric = get_metric("MP")
        result = metric.compute(obs, mod)
        assert result == pytest.approx(30.6)

    def test_std_obs(self, simple_arrays):
        """Test observation standard deviation."""
        from davinci_monet.stats import get_metric

        obs, mod = simple_arrays
        metric = get_metric("STDO")
        result = metric.compute(obs, mod)
        assert result == pytest.approx(np.std(obs, ddof=1))


class TestBiasMetrics:
    """Tests for bias metrics."""

    def test_mean_bias(self, simple_arrays):
        """Test mean bias."""
        from davinci_monet.stats import get_metric

        obs, mod = simple_arrays
        metric = get_metric("MB")
        result = metric.compute(obs, mod)
        # MB = mean(mod - obs) = (2 + -2 + 3 + -2 + 2) / 5 = 0.6
        assert result == pytest.approx(0.6)

    def test_mean_bias_sign(self):
        """Test that positive bias means model > obs."""
        from davinci_monet.stats import get_metric

        obs = np.array([10, 20, 30])
        mod = np.array([15, 25, 35])  # Model is higher
        metric = get_metric("MB")
        result = metric.compute(obs, mod)
        assert result > 0  # Positive bias

    def test_normalized_mean_bias(self, simple_arrays):
        """Test normalized mean bias."""
        from davinci_monet.stats import get_metric

        obs, mod = simple_arrays
        metric = get_metric("NMB")
        result = metric.compute(obs, mod)
        # NMB = 100 * sum(mod - obs) / sum(obs) = 100 * 3 / 150 = 2%
        assert result == pytest.approx(2.0)

    def test_fractional_bias(self):
        """Test fractional bias."""
        from davinci_monet.stats import get_metric

        obs = np.array([10, 20, 30])
        mod = np.array([10, 20, 30])  # Perfect match
        metric = get_metric("FB")
        result = metric.compute(obs, mod)
        assert result == pytest.approx(0.0)


class TestErrorMetrics:
    """Tests for error metrics."""

    def test_mean_error(self, simple_arrays):
        """Test mean error."""
        from davinci_monet.stats import get_metric

        obs, mod = simple_arrays
        metric = get_metric("ME")
        result = metric.compute(obs, mod)
        # ME = mean(|mod - obs|) = (2 + 2 + 3 + 2 + 2) / 5 = 2.2
        assert result == pytest.approx(2.2)

    def test_rmse(self, simple_arrays):
        """Test RMSE."""
        from davinci_monet.stats import get_metric

        obs, mod = simple_arrays
        metric = get_metric("RMSE")
        result = metric.compute(obs, mod)
        # RMSE = sqrt(mean((mod - obs)^2)) = sqrt((4+4+9+4+4)/5) = sqrt(5) ≈ 2.236
        expected = np.sqrt(np.mean((mod - obs) ** 2))
        assert result == pytest.approx(expected)

    def test_normalized_mean_error(self, simple_arrays):
        """Test normalized mean error."""
        from davinci_monet.stats import get_metric

        obs, mod = simple_arrays
        metric = get_metric("NME")
        result = metric.compute(obs, mod)
        # NME = 100 * sum(|mod - obs|) / sum(obs) = 100 * 11 / 150 ≈ 7.33%
        assert result == pytest.approx(100 * 11 / 150)


class TestCorrelationMetrics:
    """Tests for correlation metrics."""

    def test_correlation_perfect(self):
        """Test perfect correlation."""
        from davinci_monet.stats import get_metric

        obs = np.array([1, 2, 3, 4, 5])
        mod = np.array([2, 4, 6, 8, 10])  # Perfect linear relationship
        metric = get_metric("R")
        result = metric.compute(obs, mod)
        assert result == pytest.approx(1.0)

    def test_correlation_negative(self):
        """Test negative correlation."""
        from davinci_monet.stats import get_metric

        obs = np.array([1, 2, 3, 4, 5])
        mod = np.array([10, 8, 6, 4, 2])  # Perfect negative relationship
        metric = get_metric("R")
        result = metric.compute(obs, mod)
        assert result == pytest.approx(-1.0)

    def test_r2(self, random_arrays):
        """Test R^2."""
        from davinci_monet.stats import get_metric

        obs, mod = random_arrays
        r_metric = get_metric("R")
        r2_metric = get_metric("R2")
        r = r_metric.compute(obs, mod)
        r2 = r2_metric.compute(obs, mod)
        assert r2 == pytest.approx(r**2)

    def test_index_of_agreement(self):
        """Test index of agreement."""
        from davinci_monet.stats import get_metric

        obs = np.array([1, 2, 3, 4, 5])
        mod = np.array([1, 2, 3, 4, 5])  # Perfect match
        metric = get_metric("IOA")
        result = metric.compute(obs, mod)
        assert result == pytest.approx(1.0)


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_arrays(self):
        """Test handling of empty arrays."""
        from davinci_monet.stats import get_metric

        obs = np.array([])
        mod = np.array([])
        metric = get_metric("MB")
        result = metric.compute(obs, mod)
        assert np.isnan(result)

    def test_nan_handling(self):
        """Test NaN handling."""
        from davinci_monet.stats import get_metric

        obs = np.array([1, 2, np.nan, 4, 5])
        mod = np.array([1, 2, 3, np.nan, 5])
        metric = get_metric("N")
        result = metric.compute(obs, mod)
        # Only 3 valid pairs
        assert result == 3

    def test_zero_variance(self):
        """Test handling of zero variance."""
        from davinci_monet.stats import get_metric

        obs = np.array([5, 5, 5, 5, 5])
        mod = np.array([5, 5, 5, 5, 5])
        metric = get_metric("R")
        result = metric.compute(obs, mod)
        assert np.isnan(result)

    def test_single_value(self):
        """Test single value arrays."""
        from davinci_monet.stats import get_metric

        obs = np.array([5])
        mod = np.array([6])
        metric = get_metric("MB")
        result = metric.compute(obs, mod)
        assert result == pytest.approx(1.0)


# =============================================================================
# Calculator Tests
# =============================================================================


class TestStatisticsCalculator:
    """Tests for StatisticsCalculator."""

    def test_basic_calculation(self, paired_dataset):
        """Test basic statistics calculation."""
        from davinci_monet.stats import StatisticsCalculator

        calc = StatisticsCalculator()
        stats = calc.compute(paired_dataset, "obs_o3", "model_o3")

        assert isinstance(stats, pd.DataFrame)
        assert len(stats) == 1
        assert "MB" in stats.columns
        assert "RMSE" in stats.columns
        assert "R" in stats.columns

    def test_custom_metrics(self, paired_dataset):
        """Test custom metric selection."""
        from davinci_monet.stats import StatisticsCalculator

        calc = StatisticsCalculator()
        stats = calc.compute(
            paired_dataset,
            "obs_o3",
            "model_o3",
            metrics=["MB", "RMSE"],
        )

        assert set(stats.columns) == {"MB", "RMSE"}

    def test_groupby_site(self, paired_dataset):
        """Test grouping by site."""
        from davinci_monet.stats import StatisticsCalculator

        calc = StatisticsCalculator()
        stats = calc.compute(
            paired_dataset,
            "obs_o3",
            "model_o3",
            groupby="site",
        )

        assert len(stats) == 5  # 5 sites
        assert "site" in stats.columns

    def test_groupby_time_month(self, paired_dataset):
        """Test grouping by month."""
        from davinci_monet.stats import StatisticsCalculator

        calc = StatisticsCalculator()
        stats = calc.compute(
            paired_dataset,
            "obs_o3",
            "model_o3",
            groupby="time.month",
        )

        assert "time.month" in stats.columns
        # All data is in January
        assert stats["time.month"].iloc[0] == 1

    def test_compute_summary(self, paired_dataset):
        """Test summary dictionary output."""
        from davinci_monet.stats import StatisticsCalculator

        calc = StatisticsCalculator()
        summary = calc.compute_summary(
            paired_dataset,
            "obs_o3",
            "model_o3",
            metrics=["MB", "RMSE", "R"],
        )

        assert isinstance(summary, dict)
        assert "MB" in summary
        assert "RMSE" in summary
        assert "R" in summary

    def test_reference_comparand_keywords(self, paired_dataset):
        """Test neutral reference/comparand variable keywords."""
        from davinci_monet.stats import StatisticsCalculator

        calc = StatisticsCalculator()
        stats = calc.compute(
            paired_dataset,
            reference_var="obs_o3",
            comparand_var="model_o3",
            metrics=["N", "MB"],
        )

        assert int(stats["N"].iloc[0]) == 500
        assert "MB" in stats.columns


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_calculate_statistics(self, paired_dataset):
        """Test calculate_statistics function."""
        from davinci_monet.stats import calculate_statistics

        stats = calculate_statistics(
            paired_dataset,
            "obs_o3",
            "model_o3",
        )

        assert isinstance(stats, pd.DataFrame)
        assert len(stats) == 1

    def test_calculate_statistics_reference_comparand_keywords(self, paired_dataset):
        """Test convenience wrapper accepts neutral variable keywords."""
        from davinci_monet.stats import calculate_statistics

        stats = calculate_statistics(
            paired_dataset,
            reference_var="obs_o3",
            comparand_var="model_o3",
            metrics=["N"],
        )

        assert int(stats["N"].iloc[0]) == 500

    def test_quick_stats(self, random_arrays):
        """Test quick_stats function."""
        from davinci_monet.stats import quick_stats

        obs, mod = random_arrays
        stats = quick_stats(obs, mod)

        assert isinstance(stats, dict)
        assert "MB" in stats
        assert "RMSE" in stats
        assert "R" in stats
        # Check positive bias (we added positive offset)
        assert stats["MB"] > 0


# =============================================================================
# Output Tests
# =============================================================================


class TestStatisticsFormatter:
    """Tests for StatisticsFormatter."""

    def test_format_dataframe(self, paired_dataset):
        """Test DataFrame formatting."""
        from davinci_monet.stats import StatisticsCalculator, StatisticsFormatter

        calc = StatisticsCalculator()
        stats = calc.compute(paired_dataset, "obs_o3", "model_o3")

        formatter = StatisticsFormatter()
        formatted = formatter.format_dataframe(stats, transpose=True)

        assert "Stat_ID" in formatted.columns
        assert "Stat_FullName" in formatted.columns

    def test_to_csv(self, paired_dataset, tmp_path):
        """Test CSV output."""
        from davinci_monet.stats import StatisticsCalculator, StatisticsFormatter

        calc = StatisticsCalculator()
        stats = calc.compute(paired_dataset, "obs_o3", "model_o3")

        formatter = StatisticsFormatter()
        output_path = tmp_path / "stats.csv"
        result_path = formatter.to_csv(stats, output_path)

        assert result_path.exists()

        # Read back and verify
        df = pd.read_csv(result_path)
        assert "Stat_ID" in df.columns

    def test_to_json(self, paired_dataset, tmp_path):
        """Test JSON output."""
        from davinci_monet.stats import StatisticsCalculator, StatisticsFormatter

        calc = StatisticsCalculator()
        stats = calc.compute(paired_dataset, "obs_o3", "model_o3")

        formatter = StatisticsFormatter()
        output_path = tmp_path / "stats.json"
        result_path = formatter.to_json(stats, output_path)

        assert result_path.exists()

    def test_to_table_image(self, paired_dataset, tmp_path):
        """Test table image output."""
        import matplotlib

        matplotlib.use("Agg")

        from davinci_monet.stats import StatisticsCalculator, StatisticsFormatter

        calc = StatisticsCalculator()
        stats = calc.compute(paired_dataset, "obs_o3", "model_o3")

        formatter = StatisticsFormatter()
        output_path = tmp_path / "stats.png"
        result_path = formatter.to_table_image(stats, output_path)

        assert result_path.exists()


class TestOutputFunctions:
    """Tests for output convenience functions."""

    def test_write_statistics_csv(self, paired_dataset, tmp_path):
        """Test write_statistics_csv function."""
        from davinci_monet.stats import calculate_statistics, write_statistics_csv

        stats = calculate_statistics(paired_dataset, "obs_o3", "model_o3")
        output_path = tmp_path / "stats.csv"
        result_path = write_statistics_csv(stats, output_path)

        assert result_path.exists()

    def test_format_stats_summary(self, random_arrays):
        """Test format_stats_summary function."""
        from davinci_monet.stats import format_stats_summary, quick_stats

        obs, mod = random_arrays
        stats = quick_stats(obs, mod, metrics=["MB", "RMSE", "R"])
        summary = format_stats_summary(stats)

        assert "Mean Bias" in summary
        assert "Root Mean Square Error" in summary
        assert "Correlation Coefficient" in summary

    def test_get_metric_fullname(self):
        """Test get_metric_fullname function."""
        from davinci_monet.stats import get_metric_fullname

        assert get_metric_fullname("MB") == "Mean Bias"
        assert get_metric_fullname("RMSE") == "Root Mean Square Error"
        assert get_metric_fullname("MB", use_spaces=False) == "Mean_Bias"

    def test_create_comparison_table(self, paired_dataset):
        """Test create_comparison_table function."""
        from davinci_monet.stats import calculate_statistics, create_comparison_table

        # Create stats for two "models"
        stats1 = calculate_statistics(paired_dataset, "obs_o3", "model_o3")
        stats2 = calculate_statistics(paired_dataset, "obs_o3", "model_o3")

        comparison = create_comparison_table(
            {"Model A": stats1, "Model B": stats2},
            metrics=["MB", "RMSE", "R"],
        )

        assert "Model A" in comparison.columns
        assert "Model B" in comparison.columns
        assert "Stat_ID" in comparison.columns


# =============================================================================
# Registry Tests
# =============================================================================


class TestMetricRegistry:
    """Tests for metric registry."""

    def test_list_metrics(self):
        """Test listing available metrics."""
        from davinci_monet.stats import list_metrics

        metrics = list_metrics()
        assert len(metrics) >= 20
        assert "MB" in metrics
        assert "RMSE" in metrics
        assert "R" in metrics

    def test_standard_metrics(self):
        """Test standard metric set."""
        from davinci_monet.stats import STANDARD_METRICS

        assert "N" in STANDARD_METRICS
        assert "MB" in STANDARD_METRICS
        assert "RMSE" in STANDARD_METRICS
        assert "R" in STANDARD_METRICS

    def test_all_metrics(self):
        """Test all metrics set."""
        from davinci_monet.stats import ALL_METRICS

        assert len(ALL_METRICS) >= 20


# =============================================================================
# Workflow Tests
# =============================================================================


class TestStatsWorkflow:
    """End-to-end workflow tests for the statistics module (calls internal APIs directly)."""

    def test_full_workflow(self, paired_dataset, tmp_path):
        """Test complete statistics workflow."""
        from davinci_monet.stats import (
            calculate_statistics,
            format_stats_summary,
            write_statistics_csv,
        )

        # Calculate statistics
        stats = calculate_statistics(
            paired_dataset,
            "obs_o3",
            "model_o3",
            metrics=["N", "MB", "RMSE", "R", "NMB", "NME"],
        )

        # Verify values
        assert stats["N"].iloc[0] == 500  # 100 times * 5 sites
        assert stats["MB"].iloc[0] != 0  # Should have some bias
        assert 0 < stats["R"].iloc[0] < 1  # Should have positive correlation

        # Write to CSV
        csv_path = write_statistics_csv(stats, tmp_path / "stats.csv")
        assert csv_path.exists()

        # Format summary
        summary = format_stats_summary(stats.iloc[0].to_dict())
        assert "Mean Bias" in summary

    def test_grouped_workflow(self, paired_dataset):
        """Test grouped statistics workflow."""
        from davinci_monet.stats import calculate_statistics

        # Group by site
        site_stats = calculate_statistics(
            paired_dataset,
            "obs_o3",
            "model_o3",
            groupby="site",
            metrics=["N", "MB", "RMSE"],
        )

        assert len(site_stats) == 5
        assert all(site_stats["N"] == 100)  # 100 times per site
