"""Tests for per-site timeseries plot renderer."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr

from davinci_monet.plots.base import PlotConfig
from davinci_monet.plots.renderers.per_site_timeseries import (
    PerSiteTimeSeriesPlotter,
    plot_per_site_timeseries,
    sanitize_site_id,
)


@pytest.fixture
def synthetic_paired_data() -> xr.Dataset:
    """Create synthetic paired data with 3 sites and a time dimension."""
    n_times = 48
    sites = ["Bangkok", "Seoul", "Tokyo"]
    times = np.array([np.datetime64("2024-02-01") + np.timedelta64(i, "h") for i in range(n_times)])

    rng = np.random.default_rng(42)

    obs_data = np.empty((len(sites), n_times))
    mod_data = np.empty((len(sites), n_times))
    lats = [13.75, 37.57, 35.68]
    lons = [100.52, 126.98, 139.69]

    for i in range(len(sites)):
        obs_data[i, :] = 30 + 10 * rng.random(n_times)
        mod_data[i, :] = 28 + 12 * rng.random(n_times)

    ds = xr.Dataset(
        {
            "obs_o3": (["site", "time"], obs_data),
            "model_o3": (["site", "time"], mod_data),
        },
        coords={
            "site": sites,
            "time": times,
            "latitude": ("site", lats),
            "longitude": ("site", lons),
        },
    )
    ds["obs_o3"].attrs["units"] = "ppb"
    ds["model_o3"].attrs["units"] = "ppb"
    return ds


@pytest.fixture
def sparse_paired_data() -> xr.Dataset:
    """Create synthetic data where one site has too few valid points."""
    n_times = 48
    sites = ["Site_A", "Site_B", "Site_C"]
    times = np.array([np.datetime64("2024-02-01") + np.timedelta64(i, "h") for i in range(n_times)])

    rng = np.random.default_rng(99)

    obs_data = np.empty((3, n_times))
    mod_data = np.empty((3, n_times))

    # Site_A: enough data
    obs_data[0, :] = 30 + 10 * rng.random(n_times)
    mod_data[0, :] = 28 + 12 * rng.random(n_times)

    # Site_B: enough data
    obs_data[1, :] = 40 + 5 * rng.random(n_times)
    mod_data[1, :] = 38 + 7 * rng.random(n_times)

    # Site_C: mostly NaN (only 3 valid points)
    obs_data[2, :] = np.nan
    mod_data[2, :] = np.nan
    obs_data[2, :3] = [35, 36, 37]
    mod_data[2, :3] = [33, 34, 35]

    ds = xr.Dataset(
        {
            "obs_pm25": (["site", "time"], obs_data),
            "model_pm25": (["site", "time"], mod_data),
        },
        coords={
            "site": sites,
            "time": times,
        },
    )
    return ds


class TestPerSiteTimeSeriesPlotter:
    """Tests for PerSiteTimeSeriesPlotter."""

    def test_plot_returns_figure(self, synthetic_paired_data: xr.Dataset) -> None:
        plotter = PerSiteTimeSeriesPlotter()
        fig = plotter.plot(synthetic_paired_data, "obs_o3", "model_o3", min_points=5)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_specific_site(self, synthetic_paired_data: xr.Dataset) -> None:
        plotter = PerSiteTimeSeriesPlotter()
        fig = plotter.plot(
            synthetic_paired_data,
            "obs_o3",
            "model_o3",
            site="Seoul",
            min_points=5,
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_per_site_yields_all_sites(self, synthetic_paired_data: xr.Dataset) -> None:
        plotter = PerSiteTimeSeriesPlotter()
        results = list(
            plotter.plot_per_site(
                synthetic_paired_data,
                "obs_o3",
                "model_o3",
                min_points=5,
            )
        )
        assert len(results) == 3
        for site_id, fig in results:
            assert isinstance(site_id, str)
            assert isinstance(fig, plt.Figure)
            plt.close(fig)

    def test_plot_per_site_yields_correct_ids(self, synthetic_paired_data: xr.Dataset) -> None:
        plotter = PerSiteTimeSeriesPlotter()
        results = list(
            plotter.plot_per_site(
                synthetic_paired_data,
                "obs_o3",
                "model_o3",
                min_points=5,
            )
        )
        site_ids = [sid for sid, _ in results]
        assert site_ids == ["Bangkok", "Seoul", "Tokyo"]
        for _, fig in results:
            plt.close(fig)

    def test_min_points_filtering(self, sparse_paired_data: xr.Dataset) -> None:
        plotter = PerSiteTimeSeriesPlotter()
        results = list(
            plotter.plot_per_site(
                sparse_paired_data,
                "obs_pm25",
                "model_pm25",
                min_points=20,
            )
        )
        # Site_C has only 3 valid points, should be skipped
        assert len(results) == 2
        site_ids = [sid for sid, _ in results]
        assert "Site_C" not in site_ids
        for _, fig in results:
            plt.close(fig)

    def test_scale_factor(self, synthetic_paired_data: xr.Dataset) -> None:
        plotter = PerSiteTimeSeriesPlotter()
        # With scale_factor=2, y-values should be doubled
        fig = plotter.plot(
            synthetic_paired_data,
            "obs_o3",
            "model_o3",
            site="Bangkok",
            min_points=5,
            scale_factor=2.0,
        )
        assert isinstance(fig, plt.Figure)
        ax = fig.axes[0]
        # y-axis upper limit should reflect scaled values (roughly 2x the ~40 ppb range)
        assert ax.get_ylim()[1] > 50  # scaled values should push limits higher
        plt.close(fig)

    def test_show_stats_false(self, synthetic_paired_data: xr.Dataset) -> None:
        plotter = PerSiteTimeSeriesPlotter()
        fig = plotter.plot(
            synthetic_paired_data,
            "obs_o3",
            "model_o3",
            site="Bangkok",
            min_points=5,
            show_stats=False,
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_missing_site_dim_raises(self, synthetic_paired_data: xr.Dataset) -> None:
        plotter = PerSiteTimeSeriesPlotter()
        with pytest.raises(ValueError, match="Site dimension"):
            plotter.plot(
                synthetic_paired_data,
                "obs_o3",
                "model_o3",
                site_dim="nonexistent",
            )

    def test_no_valid_sites_raises(self, sparse_paired_data: xr.Dataset) -> None:
        plotter = PerSiteTimeSeriesPlotter()
        with pytest.raises(ValueError, match="No sites"):
            plotter.plot(
                sparse_paired_data,
                "obs_pm25",
                "model_pm25",
                min_points=1000,
            )

    def test_plot_per_site_missing_dim_raises(self, synthetic_paired_data: xr.Dataset) -> None:
        plotter = PerSiteTimeSeriesPlotter()
        with pytest.raises(ValueError, match="Site dimension"):
            list(
                plotter.plot_per_site(
                    synthetic_paired_data,
                    "obs_o3",
                    "model_o3",
                    site_dim="nonexistent",
                )
            )

    def test_with_title(self, synthetic_paired_data: xr.Dataset) -> None:
        config = PlotConfig(title="O3: Model vs AirNow")
        plotter = PerSiteTimeSeriesPlotter(config=config)
        fig = plotter.plot(
            synthetic_paired_data,
            "obs_o3",
            "model_o3",
            site="Bangkok",
            min_points=5,
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_line_obs_style(self, synthetic_paired_data: xr.Dataset) -> None:
        plotter = PerSiteTimeSeriesPlotter()
        fig = plotter.plot(
            synthetic_paired_data,
            "obs_o3",
            "model_o3",
            site="Bangkok",
            min_points=5,
            obs_style="line",
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_scatter_model_style(self, synthetic_paired_data: xr.Dataset) -> None:
        plotter = PerSiteTimeSeriesPlotter()
        fig = plotter.plot(
            synthetic_paired_data,
            "obs_o3",
            "model_o3",
            site="Bangkok",
            min_points=5,
            model_style="scatter",
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestSanitizeSiteId:
    """Tests for the sanitize_site_id function."""

    def test_simple_name(self) -> None:
        assert sanitize_site_id("Bangkok") == "Bangkok"

    def test_spaces_replaced(self) -> None:
        assert sanitize_site_id("New York") == "New_York"

    def test_special_chars_replaced(self) -> None:
        assert sanitize_site_id("Site (A/B)") == "Site_A_B"

    def test_consecutive_underscores_collapsed(self) -> None:
        assert sanitize_site_id("Site   Name") == "Site_Name"

    def test_leading_trailing_stripped(self) -> None:
        assert sanitize_site_id(" Site ") == "Site"

    def test_hyphens_preserved(self) -> None:
        assert sanitize_site_id("Seoul-Gangnam") == "Seoul-Gangnam"


class TestConvenienceFunction:
    """Tests for the plot_per_site_timeseries convenience function."""

    def test_returns_figure(self, synthetic_paired_data: xr.Dataset) -> None:
        fig = plot_per_site_timeseries(
            synthetic_paired_data,
            "obs_o3",
            "model_o3",
            min_points=5,
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_with_title(self, synthetic_paired_data: xr.Dataset) -> None:
        fig = plot_per_site_timeseries(
            synthetic_paired_data,
            "obs_o3",
            "model_o3",
            title="O3 Comparison",
            min_points=5,
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
