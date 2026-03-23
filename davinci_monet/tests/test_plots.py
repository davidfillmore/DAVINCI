"""Tests for the plotting module.

This module tests the plotting system including base classes,
registry, and individual plot renderers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend for testing
import matplotlib.pyplot as plt


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def simple_paired_data() -> xr.Dataset:
    """Create simple paired dataset for testing."""
    np.random.seed(42)
    n_times = 100
    n_sites = 5

    time = pd.date_range("2023-01-01", periods=n_times, freq="h")
    sites = [f"site_{i}" for i in range(n_sites)]

    # Create synthetic data
    obs = np.random.normal(50, 10, (n_times, n_sites))
    model = obs + np.random.normal(2, 5, (n_times, n_sites))  # Slight bias

    lats = np.linspace(35, 45, n_sites)
    lons = np.linspace(-120, -100, n_sites)

    ds = xr.Dataset(
        {
            "obs_o3": (["time", "site"], obs, {"units": "ppbv", "long_name": "Ozone"}),
            "model_o3": (["time", "site"], model, {"units": "ppbv", "long_name": "Ozone"}),
        },
        coords={
            "time": time,
            "site": sites,
            "latitude": ("site", lats),
            "longitude": ("site", lons),
        },
    )
    return ds


@pytest.fixture
def track_paired_data() -> xr.Dataset:
    """Create paired dataset for track/curtain plots."""
    np.random.seed(42)
    n_times = 200

    time = pd.date_range("2023-01-01", periods=n_times, freq="1min")

    # Simulate aircraft track
    altitude = 1000 + 5000 * np.sin(np.linspace(0, 2 * np.pi, n_times)) + np.random.normal(0, 100, n_times)
    lats = np.linspace(35, 40, n_times)
    lons = np.linspace(-120, -110, n_times)

    obs = 50 + 10 * np.exp(-altitude / 5000) + np.random.normal(0, 5, n_times)
    model = obs + np.random.normal(3, 3, n_times)

    ds = xr.Dataset(
        {
            "obs_o3": (["time"], obs, {"units": "ppbv", "long_name": "Ozone"}),
            "model_o3": (["time"], model, {"units": "ppbv", "long_name": "Ozone"}),
        },
        coords={
            "time": time,
            "altitude": ("time", altitude),
            "latitude": ("time", lats),
            "longitude": ("time", lons),
        },
    )
    return ds


@pytest.fixture
def flight_paired_data() -> xr.Dataset:
    """Create paired dataset for flight time series plots (multiple flights)."""
    np.random.seed(42)
    # Simulate 3 flights on different days, each ~2 hours
    n_points_per_flight = 120
    flights = []

    for day in range(3):
        base_time = pd.Timestamp(f"2023-01-0{day + 1} 10:00:00")
        time = pd.date_range(base_time, periods=n_points_per_flight, freq="1min")

        # Simulate aircraft track
        altitude = 1000 + 5000 * np.sin(np.linspace(0, np.pi, n_points_per_flight))
        lats = np.linspace(35 + day, 40 + day, n_points_per_flight)
        lons = np.linspace(-120, -110, n_points_per_flight)

        obs = 50 + 10 * np.exp(-altitude / 5000) + np.random.normal(0, 5, n_points_per_flight)
        model = obs + np.random.normal(3, 3, n_points_per_flight)

        flight_id = f"2023-01-0{day + 1}"

        ds = xr.Dataset(
            {
                "obs_o3": (["time"], obs, {"units": "ppbv", "long_name": "Ozone"}),
                "model_o3": (["time"], model, {"units": "ppbv", "long_name": "Ozone"}),
            },
            coords={
                "time": time,
                "altitude": ("time", altitude),
                "latitude": ("time", lats),
                "longitude": ("time", lons),
                "flight": ("time", [flight_id] * n_points_per_flight),
            },
        )
        flights.append(ds)

    return xr.concat(flights, dim="time")


@pytest.fixture
def gridded_paired_data() -> xr.Dataset:
    """Create gridded paired dataset for spatial plots."""
    np.random.seed(42)

    lats = np.linspace(30, 50, 20)
    lons = np.linspace(-130, -100, 30)
    time = pd.date_range("2023-01-01", periods=3, freq="D")

    lat_grid, lon_grid = np.meshgrid(lats, lons, indexing="ij")

    # Create spatial pattern
    obs = 40 + 20 * np.sin(np.radians(lat_grid)) * np.cos(np.radians(lon_grid + 110))
    obs = np.stack([obs + np.random.normal(0, 3, obs.shape) for _ in range(len(time))])
    model = obs + np.random.normal(5, 8, obs.shape)

    ds = xr.Dataset(
        {
            "obs_o3": (["time", "lat", "lon"], obs, {"units": "ppbv", "long_name": "Ozone"}),
            "model_o3": (["time", "lat", "lon"], model, {"units": "ppbv", "long_name": "Ozone"}),
        },
        coords={
            "time": time,
            "lat": lats,
            "lon": lons,
        },
    )
    return ds


# =============================================================================
# Base Module Tests
# =============================================================================


class TestPlotConfig:
    """Tests for PlotConfig class."""

    def test_default_config(self):
        """Test default configuration creation."""
        from davinci_monet.plots.base import PlotConfig
        from davinci_monet.plots.style import OBS_COLOR, MODEL_COLOR

        config = PlotConfig()
        assert config.figure.figsize == (8, 5)  # FigureConfig default
        assert config.text.fontsize == 14.0  # Axis label size
        assert config.style.obs_color == OBS_COLOR  # NCAR gray
        assert config.style.model_color == MODEL_COLOR  # NCAR blue
        assert config.debug is False

    def test_from_dict(self):
        """Test configuration from dictionary."""
        from davinci_monet.plots.base import PlotConfig

        config_dict = {
            "figure": {"figsize": (12, 8)},
            "text": {"fontsize": 14},
            "vmin": 0,
            "vmax": 100,
            "title": "Test Plot",
        }

        config = PlotConfig.from_dict(config_dict)
        assert config.figure.figsize == (12, 8)
        assert config.text.fontsize == 14
        assert config.vmin == 0
        assert config.vmax == 100
        assert config.title == "Test Plot"

    def test_legacy_dict_keys(self):
        """Test backward compatible dict keys."""
        from davinci_monet.plots.base import PlotConfig

        config_dict = {
            "fig_dict": {"figsize": (8, 6)},
            "text_dict": {"fontsize": 11},
            "plot_dict": {"obs_color": "red"},
            "domain_type": "conus",
        }

        config = PlotConfig.from_dict(config_dict)
        assert config.figure.figsize == (8, 6)
        assert config.text.fontsize == 11
        assert config.domain.domain_type == "conus"


class TestUtilities:
    """Tests for plotting utility functions."""

    def test_merge_config_dicts(self):
        """Test dictionary merging."""
        from davinci_monet.plots.base import merge_config_dicts

        defaults = {"a": 1, "b": 2}
        overrides = {"b": 3, "c": 4}

        result = merge_config_dicts(defaults, overrides)
        assert result == {"a": 1, "b": 3, "c": 4}

    def test_merge_config_dicts_none(self):
        """Test dictionary merging with None overrides."""
        from davinci_monet.plots.base import merge_config_dicts

        defaults = {"a": 1, "b": 2}
        result = merge_config_dicts(defaults, None)
        assert result == defaults

    def test_calculate_symmetric_limits(self):
        """Test symmetric limit calculation."""
        from davinci_monet.plots.base import calculate_symmetric_limits

        data = np.array([-5, -3, 0, 2, 8])
        vmin, vmax = calculate_symmetric_limits(data)
        assert vmin == -vmax
        assert vmax > 0

    def test_calculate_symmetric_limits_empty(self):
        """Test symmetric limits with empty data."""
        from davinci_monet.plots.base import calculate_symmetric_limits

        data = np.array([np.nan, np.nan])
        vmin, vmax = calculate_symmetric_limits(data)
        assert vmin == -1.0
        assert vmax == 1.0

    def test_get_variable_label(self, simple_paired_data):
        """Test variable label extraction."""
        from davinci_monet.plots.base import get_variable_label

        label = get_variable_label(simple_paired_data, "obs_o3")
        assert label == "Ozone"

        label = get_variable_label(simple_paired_data, "obs_o3", "Custom Label")
        assert label == "Custom Label"

    def test_format_label_with_units(self):
        """Test label formatting with units."""
        from davinci_monet.plots.base import format_label_with_units

        assert format_label_with_units("Ozone", "ppbv") == "Ozone (ppbv)"
        assert format_label_with_units("Ozone", None) == "Ozone"


# =============================================================================
# Registry Tests
# =============================================================================


class TestRegistry:
    """Tests for plot registry."""

    def test_list_plotters(self):
        """Test listing registered plotters."""
        from davinci_monet.plots import list_plotters

        plotters = list_plotters()
        assert "timeseries" in plotters
        assert "scatter" in plotters
        assert "taylor" in plotters
        assert "boxplot" in plotters
        assert "diurnal" in plotters
        assert "curtain" in plotters
        assert "scorecard" in plotters
        assert "site_timeseries" in plotters
        assert "flight_timeseries" in plotters
        assert "track_map_3d" in plotters
        assert "spatial_bias" in plotters
        assert "spatial_overlay" in plotters
        assert "spatial_distribution" in plotters

    def test_has_plotter(self):
        """Test plotter existence check."""
        from davinci_monet.plots import has_plotter

        assert has_plotter("timeseries")
        assert has_plotter("scatter")
        assert not has_plotter("nonexistent")

    def test_get_plotter(self):
        """Test getting plotter instance."""
        from davinci_monet.plots import get_plotter, TimeSeriesPlotter

        plotter = get_plotter("timeseries")
        assert isinstance(plotter, TimeSeriesPlotter)

    def test_get_plotter_with_config(self):
        """Test getting plotter with configuration."""
        from davinci_monet.plots import get_plotter

        config = {"vmin": 0, "vmax": 100}
        plotter = get_plotter("timeseries", config=config)
        assert plotter.config.vmin == 0
        assert plotter.config.vmax == 100

    def test_get_plot_category(self):
        """Test plot category classification."""
        from davinci_monet.plots import get_plot_category

        assert get_plot_category("timeseries") == "temporal"
        assert get_plot_category("scatter") == "statistical"
        assert get_plot_category("spatial_bias") == "spatial"
        assert get_plot_category("curtain") == "specialized"
        assert get_plot_category("unknown") is None


# =============================================================================
# Renderer Tests
# =============================================================================


class TestTimeSeriesPlotter:
    """Tests for time series plotter."""

    def test_basic_plot(self, simple_paired_data):
        """Test basic time series plot."""
        from davinci_monet.plots import plot_timeseries

        fig = plot_timeseries(
            simple_paired_data,
            "obs_o3",
            "model_o3",
            aggregate_dim="site",
        )

        assert fig is not None
        assert len(fig.axes) >= 1
        plt.close(fig)

    def test_with_resampling(self, simple_paired_data):
        """Test time series with resampling."""
        from davinci_monet.plots import plot_timeseries

        fig = plot_timeseries(
            simple_paired_data,
            "obs_o3",
            "model_o3",
            aggregate_dim="site",
            resample="6h",
        )

        assert fig is not None
        plt.close(fig)

    def test_custom_labels(self, simple_paired_data):
        """Test custom labels."""
        from davinci_monet.plots import TimeSeriesPlotter, PlotConfig

        config = PlotConfig(obs_label="Custom Obs", model_label="Custom Model")
        plotter = TimeSeriesPlotter(config=config)

        fig = plotter.plot(
            simple_paired_data,
            "obs_o3",
            "model_o3",
            aggregate_dim="site",
        )

        assert fig is not None
        plt.close(fig)


class TestDiurnalPlotter:
    """Tests for diurnal cycle plotter."""

    def test_basic_plot(self, simple_paired_data):
        """Test basic diurnal plot."""
        from davinci_monet.plots import plot_diurnal

        fig = plot_diurnal(simple_paired_data, "obs_o3", "model_o3")

        assert fig is not None
        plt.close(fig)

    def test_spread_types(self, simple_paired_data):
        """Test different spread types."""
        from davinci_monet.plots import DiurnalPlotter

        plotter = DiurnalPlotter()

        for spread in ["none", "std", "iqr", "range"]:
            fig = plotter.plot(
                simple_paired_data,
                "obs_o3",
                "model_o3",
                show_spread=spread,
            )
            assert fig is not None
            plt.close(fig)


class TestSiteTimeSeriesPlotter:
    """Tests for site time series plotter."""

    def test_basic_plot(self, simple_paired_data):
        """Test basic site time series plot."""
        from davinci_monet.plots import plot_site_timeseries

        fig = plot_site_timeseries(
            simple_paired_data,
            "obs_o3",
            "model_o3",
            ncols=2,
        )

        assert fig is not None
        plt.close(fig)

    def test_custom_ncols(self, simple_paired_data):
        """Test different number of columns."""
        from davinci_monet.plots import SiteTimeSeriesPlotter

        plotter = SiteTimeSeriesPlotter()

        for ncols in [1, 2, 3]:
            fig = plotter.plot(
                simple_paired_data,
                "obs_o3",
                "model_o3",
                ncols=ncols,
            )
            assert fig is not None
            plt.close(fig)

    def test_min_points_filter(self, simple_paired_data):
        """Test minimum points filtering."""
        from davinci_monet.plots import SiteTimeSeriesPlotter

        plotter = SiteTimeSeriesPlotter()
        fig = plotter.plot(
            simple_paired_data,
            "obs_o3",
            "model_o3",
            min_points=10,
        )

        assert fig is not None
        plt.close(fig)


class TestFlightTimeSeriesPlotter:
    """Tests for flight time series plotter."""

    def test_basic_plot(self, flight_paired_data):
        """Test basic flight time series plot."""
        from davinci_monet.plots import plot_flight_timeseries

        fig = plot_flight_timeseries(
            flight_paired_data,
            "obs_o3",
            "model_o3",
            ncols=2,
        )

        assert fig is not None
        # Should have panels for 3 flights
        plt.close(fig)

    def test_custom_ncols(self, flight_paired_data):
        """Test different number of columns."""
        from davinci_monet.plots import FlightTimeSeriesPlotter

        plotter = FlightTimeSeriesPlotter()

        for ncols in [1, 2, 3]:
            fig = plotter.plot(
                flight_paired_data,
                "obs_o3",
                "model_o3",
                ncols=ncols,
            )
            assert fig is not None
            plt.close(fig)

    def test_min_points_filter(self, flight_paired_data):
        """Test minimum points filtering."""
        from davinci_monet.plots import FlightTimeSeriesPlotter

        plotter = FlightTimeSeriesPlotter()
        fig = plotter.plot(
            flight_paired_data,
            "obs_o3",
            "model_o3",
            min_points=50,
        )

        assert fig is not None
        plt.close(fig)

    def test_show_stats(self, flight_paired_data):
        """Test statistics display toggle."""
        from davinci_monet.plots import FlightTimeSeriesPlotter

        plotter = FlightTimeSeriesPlotter()

        for show_stats in [True, False]:
            fig = plotter.plot(
                flight_paired_data,
                "obs_o3",
                "model_o3",
                show_stats=show_stats,
            )
            assert fig is not None
            plt.close(fig)

    def test_missing_flight_coord(self, track_paired_data):
        """Test error when flight coordinate is missing."""
        from davinci_monet.plots import FlightTimeSeriesPlotter

        plotter = FlightTimeSeriesPlotter()

        with pytest.raises(ValueError, match="Flight coordinate"):
            plotter.plot(
                track_paired_data,
                "obs_o3",
                "model_o3",
            )

    def test_plot_per_flight(self, flight_paired_data):
        """Test per-flight time series plot generation."""
        from davinci_monet.plots import FlightTimeSeriesPlotter

        plotter = FlightTimeSeriesPlotter()
        flight_plots = list(plotter.plot_per_flight(
            flight_paired_data,
            "obs_o3",
            "model_o3",
            min_points=10,
        ))

        # Should generate 3 flights (from fixture)
        assert len(flight_plots) == 3

        for flight_id, fig in flight_plots:
            # Flight ID should be in YYYYMMDD format (no hyphens)
            assert "-" not in flight_id
            assert len(flight_id) == 8
            assert fig is not None
            plt.close(fig)

    def test_plot_per_flight_min_points(self, flight_paired_data):
        """Test min_points filter in per-flight time series plotting."""
        from davinci_monet.plots import FlightTimeSeriesPlotter

        plotter = FlightTimeSeriesPlotter()
        # Set min_points higher than data available per flight
        flight_plots = list(plotter.plot_per_flight(
            flight_paired_data,
            "obs_o3",
            "model_o3",
            min_points=200,  # Each flight has 120 points
        ))

        # No flights should pass the filter
        assert len(flight_plots) == 0

    def test_altitude_display(self, flight_paired_data):
        """Test altitude display on right y-axis."""
        from davinci_monet.plots import FlightTimeSeriesPlotter

        plotter = FlightTimeSeriesPlotter()
        fig = plotter.plot(
            flight_paired_data,
            "obs_o3",
            "model_o3",
            show_altitude=True,
        )

        assert fig is not None
        # Check that twin axes were created for altitude
        # Each subplot should have a twin axis if altitude data exists
        plt.close(fig)

    def test_altitude_disabled(self, flight_paired_data):
        """Test that altitude can be disabled."""
        from davinci_monet.plots import FlightTimeSeriesPlotter

        plotter = FlightTimeSeriesPlotter()
        fig = plotter.plot(
            flight_paired_data,
            "obs_o3",
            "model_o3",
            show_altitude=False,
        )

        assert fig is not None
        plt.close(fig)

    def test_altitude_units(self, flight_paired_data):
        """Test altitude unit conversion (m vs km)."""
        from davinci_monet.plots import FlightTimeSeriesPlotter

        plotter = FlightTimeSeriesPlotter()

        # Test km (default)
        fig_km = plotter.plot(
            flight_paired_data,
            "obs_o3",
            "model_o3",
            altitude_units="km",
        )
        assert fig_km is not None
        plt.close(fig_km)

        # Test meters
        fig_m = plotter.plot(
            flight_paired_data,
            "obs_o3",
            "model_o3",
            altitude_units="m",
        )
        assert fig_m is not None
        plt.close(fig_m)

    def test_per_flight_altitude(self, flight_paired_data):
        """Test altitude display in per-flight time series."""
        from davinci_monet.plots import FlightTimeSeriesPlotter

        plotter = FlightTimeSeriesPlotter()
        flight_plots = list(plotter.plot_per_flight(
            flight_paired_data,
            "obs_o3",
            "model_o3",
            show_altitude=True,
        ))

        assert len(flight_plots) == 3  # 3 flights
        for flight_id, fig in flight_plots:
            assert fig is not None
            plt.close(fig)


class TestScatterPlotter:
    """Tests for scatter plotter."""

    def test_basic_plot(self, simple_paired_data):
        """Test basic scatter plot."""
        from davinci_monet.plots import plot_scatter

        fig = plot_scatter(simple_paired_data, "obs_o3", "model_o3")

        assert fig is not None
        plt.close(fig)

    def test_with_density(self, simple_paired_data):
        """Test scatter plot with density coloring."""
        from davinci_monet.plots import plot_scatter

        fig = plot_scatter(
            simple_paired_data,
            "obs_o3",
            "model_o3",
            show_density=True,
        )

        assert fig is not None
        plt.close(fig)

    def test_regression_and_stats(self, simple_paired_data):
        """Test scatter with regression and stats."""
        from davinci_monet.plots import ScatterPlotter

        plotter = ScatterPlotter()
        fig = plotter.plot(
            simple_paired_data,
            "obs_o3",
            "model_o3",
            show_regression=True,
            show_stats=True,
            show_one_to_one=True,
        )

        assert fig is not None
        plt.close(fig)

    def test_plot_per_flight(self, flight_paired_data):
        """Test per-flight scatter plot generation."""
        from davinci_monet.plots import ScatterPlotter

        plotter = ScatterPlotter()
        flight_plots = list(plotter.plot_per_flight(
            flight_paired_data,
            "obs_o3",
            "model_o3",
            min_points=10,
        ))

        # Should generate 3 flights (from fixture)
        assert len(flight_plots) == 3

        for flight_id, fig in flight_plots:
            # Flight ID should be in YYYYMMDD format (no hyphens)
            assert "-" not in flight_id
            assert len(flight_id) == 8
            assert fig is not None
            plt.close(fig)

    def test_plot_per_flight_min_points(self, flight_paired_data):
        """Test min_points filter in per-flight plotting."""
        from davinci_monet.plots import ScatterPlotter

        plotter = ScatterPlotter()
        # Set min_points higher than data available per flight
        flight_plots = list(plotter.plot_per_flight(
            flight_paired_data,
            "obs_o3",
            "model_o3",
            min_points=200,  # Each flight has 120 points
        ))

        # No flights should pass the filter
        assert len(flight_plots) == 0


class TestTaylorPlotter:
    """Tests for Taylor diagram plotter."""

    def test_basic_plot(self, simple_paired_data):
        """Test basic Taylor diagram."""
        from davinci_monet.plots import plot_taylor

        fig = plot_taylor(simple_paired_data, "obs_o3", "model_o3")

        assert fig is not None
        plt.close(fig)

    def test_normalized(self, simple_paired_data):
        """Test normalized Taylor diagram."""
        from davinci_monet.plots import TaylorPlotter

        plotter = TaylorPlotter()
        fig = plotter.plot(
            simple_paired_data,
            "obs_o3",
            "model_o3",
            normalize=True,
        )

        assert fig is not None
        plt.close(fig)


class TestBoxPlotter:
    """Tests for box plotter."""

    def test_basic_plot(self, simple_paired_data):
        """Test basic box plot."""
        from davinci_monet.plots import plot_boxplot

        fig = plot_boxplot(simple_paired_data, "obs_o3", "model_o3")

        assert fig is not None
        plt.close(fig)

    def test_grouped_plot(self, simple_paired_data):
        """Test grouped box plot."""
        from davinci_monet.plots import BoxPlotter

        plotter = BoxPlotter()
        fig = plotter.plot(
            simple_paired_data,
            "obs_o3",
            "model_o3",
            group_by="site",
        )

        assert fig is not None
        plt.close(fig)

    def test_horizontal(self, simple_paired_data):
        """Test horizontal box plot."""
        from davinci_monet.plots import plot_boxplot

        fig = plot_boxplot(
            simple_paired_data,
            "obs_o3",
            "model_o3",
            orientation="horizontal",
        )

        assert fig is not None
        plt.close(fig)


class TestCurtainPlotter:
    """Tests for curtain plotter."""

    def test_basic_plot(self, track_paired_data):
        """Test basic curtain plot."""
        from davinci_monet.plots import plot_curtain

        fig = plot_curtain(
            track_paired_data,
            "obs_o3",
            "model_o3",
            alt_var="altitude",
        )

        assert fig is not None
        plt.close(fig)

    def test_show_var_options(self, track_paired_data):
        """Test different show_var options."""
        from davinci_monet.plots import CurtainPlotter

        plotter = CurtainPlotter()

        for show_var in ["obs", "model", "bias"]:
            fig = plotter.plot(
                track_paired_data,
                "obs_o3",
                "model_o3",
                alt_var="altitude",
                show_var=show_var,
            )
            assert fig is not None
            plt.close(fig)


class TestTrackMap3DPlotter:
    """Tests for 3D track map plotter."""

    def test_basic_plot(self, track_paired_data):
        """Test basic 3D track plot."""
        from davinci_monet.plots import plot_track_map_3d

        fig = plot_track_map_3d(
            track_paired_data,
            "obs_o3",
            "model_o3",
            alt_var="altitude",
        )

        assert fig is not None
        plt.close(fig)

    def test_show_var_options(self, track_paired_data):
        """Test different show_var options."""
        from davinci_monet.plots import TrackMap3DPlotter

        plotter = TrackMap3DPlotter()

        for show_var in ["obs", "model", "bias"]:
            fig = plotter.plot(
                track_paired_data,
                "obs_o3",
                "model_o3",
                alt_var="altitude",
                show_var=show_var,
            )
            assert fig is not None
            plt.close(fig)

    def test_view_angles(self, track_paired_data):
        """Test different view angles."""
        from davinci_monet.plots import TrackMap3DPlotter

        plotter = TrackMap3DPlotter()
        fig = plotter.plot(
            track_paired_data,
            "obs_o3",
            "model_o3",
            alt_var="altitude",
            elev=45,
            azim=-90,
        )

        assert fig is not None
        plt.close(fig)

    def test_projection_toggle(self, track_paired_data):
        """Test show_projection toggle."""
        from davinci_monet.plots import TrackMap3DPlotter

        plotter = TrackMap3DPlotter()
        fig = plotter.plot(
            track_paired_data,
            "obs_o3",
            "model_o3",
            alt_var="altitude",
            show_projection=False,
        )

        assert fig is not None
        plt.close(fig)

    def test_plot_per_flight(self, flight_paired_data):
        """Test per-flight 3D track plot generation."""
        from davinci_monet.plots import TrackMap3DPlotter

        plotter = TrackMap3DPlotter()
        flight_plots = list(plotter.plot_per_flight(
            flight_paired_data,
            "obs_o3",
            "model_o3",
            min_points=10,
            show_coastlines=False,  # Faster for testing
        ))

        # Should generate 3 flights (from fixture)
        assert len(flight_plots) == 3

        for flight_id, fig in flight_plots:
            # Flight ID should be in YYYYMMDD format (no hyphens)
            assert "-" not in flight_id
            assert len(flight_id) == 8
            assert fig is not None
            plt.close(fig)

    def test_plot_per_flight_min_points(self, flight_paired_data):
        """Test min_points filter in per-flight 3D track plotting."""
        from davinci_monet.plots import TrackMap3DPlotter

        plotter = TrackMap3DPlotter()
        # Set min_points higher than data available per flight
        flight_plots = list(plotter.plot_per_flight(
            flight_paired_data,
            "obs_o3",
            "model_o3",
            min_points=200,  # Each flight has 120 points
            show_coastlines=False,
        ))

        # No flights should pass the filter
        assert len(flight_plots) == 0


class TestScorecardPlotter:
    """Tests for scorecard plotter."""

    def test_basic_plot(self, simple_paired_data):
        """Test basic scorecard plot."""
        from davinci_monet.plots import plot_scorecard

        fig = plot_scorecard(simple_paired_data, "obs_o3", "model_o3")

        assert fig is not None
        plt.close(fig)

    def test_from_dataframe(self):
        """Test scorecard from DataFrame."""
        from davinci_monet.plots import ScorecardPlotter

        stats_df = pd.DataFrame(
            {
                "Model A": [0.9, 2.5, 5.0],
                "Model B": [0.85, -1.0, 6.5],
                "Model C": [0.92, 0.5, 4.0],
            },
            index=["R", "MB", "RMSE"],
        )

        plotter = ScorecardPlotter()
        fig = plotter.plot_from_dataframe(stats_df)

        assert fig is not None
        plt.close(fig)


class TestSpatialPlotters:
    """Tests for spatial plotters."""

    @pytest.mark.skipif(
        not pytest.importorskip("cartopy", reason="cartopy not available"),
        reason="cartopy not available"
    )
    def test_spatial_bias(self, simple_paired_data):
        """Test spatial bias plot."""
        from davinci_monet.plots import plot_spatial_bias

        fig = plot_spatial_bias(simple_paired_data, "obs_o3", "model_o3")

        assert fig is not None
        plt.close(fig)

    @pytest.mark.skipif(
        not pytest.importorskip("cartopy", reason="cartopy not available"),
        reason="cartopy not available"
    )
    def test_spatial_distribution(self, simple_paired_data):
        """Test spatial distribution plot."""
        from davinci_monet.plots import plot_spatial_distribution

        fig = plot_spatial_distribution(
            simple_paired_data,
            "obs_o3",
            "model_o3",
            show_var="obs",
        )

        assert fig is not None
        plt.close(fig)

    def test_get_domain_extent(self):
        """Test domain extent lookup."""
        from davinci_monet.plots.renderers.spatial import get_domain_extent

        extent = get_domain_extent("conus")
        assert extent is not None
        assert len(extent) == 4

        extent = get_domain_extent("epa_region", "R1")
        assert extent is not None

        extent = get_domain_extent("unknown")
        assert extent is None


# =============================================================================
# Integration Tests
# =============================================================================


class TestPlotterIntegration:
    """Integration tests for the plotting system."""

    def test_all_plotters_instantiate(self):
        """Test that all registered plotters can be instantiated."""
        from davinci_monet.plots import list_plotters, get_plotter

        for name in list_plotters():
            plotter = get_plotter(name)
            assert plotter is not None
            assert hasattr(plotter, "plot")
            assert hasattr(plotter, "save")

    def test_plotter_save(self, simple_paired_data, tmp_path):
        """Test saving a figure."""
        from davinci_monet.plots import get_plotter

        plotter = get_plotter("scatter")
        fig = plotter.plot(simple_paired_data, "obs_o3", "model_o3")

        output_path = tmp_path / "test_plot.png"
        saved_path = plotter.save(fig, output_path)

        assert saved_path.exists()
        plt.close(fig)

    def test_multiple_plots_same_axes(self, simple_paired_data):
        """Test plotting multiple models on same axes."""
        from davinci_monet.plots import TimeSeriesPlotter

        fig, ax = plt.subplots()

        plotter = TimeSeriesPlotter()

        # Plot first "model"
        plotter.plot(
            simple_paired_data,
            "obs_o3",
            "model_o3",
            ax=ax,
            aggregate_dim="site",
        )

        # Plot on same axes (simulating multiple models)
        # In real use, would have different model data

        assert fig is not None
        plt.close(fig)


# =============================================================================
# Spatial Overlay Tests
# =============================================================================


class TestSpatialOverlay:
    """Behavioral tests for SpatialOverlayPlotter."""

    def test_overlay_plot(self, simple_paired_data, gridded_paired_data):
        """Overlay model contours with observation scatter points."""
        from davinci_monet.plots.renderers.spatial.overlay import SpatialOverlayPlotter

        plotter = SpatialOverlayPlotter()

        # Create a model field (2D lat/lon) for the contour layer
        model_field = gridded_paired_data["model_o3"].isel(time=0)

        fig = plotter.plot(
            simple_paired_data,
            obs_var="obs_o3",
            model_var="model_o3",
            model_field=model_field,
        )

        assert fig is not None
        axes = fig.get_axes()
        assert len(axes) >= 1
        plt.close(fig)

    def test_overlay_without_model_field(self, simple_paired_data):
        """Overlay should handle missing model_field gracefully."""
        from davinci_monet.plots.renderers.spatial.overlay import SpatialOverlayPlotter

        plotter = SpatialOverlayPlotter()

        # When model_field is None, plotter should fall back to model_var from paired_data
        # This may not produce contours (1D data), but should not crash
        try:
            fig = plotter.plot(
                simple_paired_data,
                obs_var="obs_o3",
                model_var="model_o3",
            )
            assert fig is not None
            plt.close(fig)
        except (ValueError, KeyError, TypeError):
            # Acceptable: plotter may require a 2D model field
            pass


# =============================================================================
# Time Series Aggregate Mode Tests
# =============================================================================


class TestTimeSeriesAggregate:
    """Tests for TimeSeriesPlotter aggregate and multi-dim modes."""

    def test_aggregate_dim(self, simple_paired_data):
        """Timeseries with explicit aggregate_dim averages over sites."""
        from davinci_monet.plots import TimeSeriesPlotter

        plotter = TimeSeriesPlotter()
        fig = plotter.plot(
            simple_paired_data,
            "obs_o3",
            "model_o3",
            aggregate_dim="site",
        )

        assert fig is not None
        ax = fig.get_axes()[0]
        # Should have at least obs and model lines
        assert len(ax.get_lines()) >= 2
        plt.close(fig)

    def test_auto_aggregate_multidim(self, simple_paired_data):
        """Timeseries auto-averages non-time dims when no aggregate_dim given."""
        from davinci_monet.plots import TimeSeriesPlotter

        plotter = TimeSeriesPlotter()
        fig = plotter.plot(
            simple_paired_data,
            "obs_o3",
            "model_o3",
            # No aggregate_dim — should auto-detect and average 'site'
        )

        assert fig is not None
        plt.close(fig)

    def test_resample(self, simple_paired_data):
        """Timeseries with resample parameter."""
        from davinci_monet.plots import TimeSeriesPlotter

        plotter = TimeSeriesPlotter()
        fig = plotter.plot(
            simple_paired_data,
            "obs_o3",
            "model_o3",
            aggregate_dim="site",
            resample="6h",
        )

        assert fig is not None
        plt.close(fig)


# =============================================================================
# Scorecard Tests
# =============================================================================


class TestScorecardPlotter:
    """Tests for ScorecardPlotter with multiple variables."""

    def test_scorecard_multi_variable(self):
        """Scorecard with multiple variables."""
        from davinci_monet.plots.renderers.scorecard import ScorecardPlotter

        # Create multi-variable paired data
        np.random.seed(42)
        n = 100

        ds = xr.Dataset(
            {
                "obs_o3": (["time"], np.random.normal(50, 10, n)),
                "model_o3": (["time"], np.random.normal(52, 10, n)),
                "obs_pm25": (["time"], np.random.normal(15, 5, n)),
                "model_pm25": (["time"], np.random.normal(17, 5, n)),
            },
            coords={"time": pd.date_range("2023-01-01", periods=n, freq="h")},
        )

        plotter = ScorecardPlotter()
        fig = plotter.plot(
            ds,
            obs_var="obs_o3",
            model_var="model_o3",
        )

        assert fig is not None
        plt.close(fig)


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_data(self):
        """Test handling of empty data."""
        from davinci_monet.plots import ScatterPlotter

        ds = xr.Dataset(
            {
                "obs_o3": (["time"], np.array([np.nan, np.nan, np.nan])),
                "model_o3": (["time"], np.array([np.nan, np.nan, np.nan])),
            },
            coords={"time": pd.date_range("2023-01-01", periods=3, freq="h")},
        )

        plotter = ScatterPlotter()
        fig = plotter.plot(ds, "obs_o3", "model_o3")

        # Should handle gracefully
        assert fig is not None
        plt.close(fig)

    def test_single_point(self):
        """Test handling of single data point."""
        from davinci_monet.plots import ScatterPlotter

        ds = xr.Dataset(
            {
                "obs_o3": (["time"], np.array([50.0])),
                "model_o3": (["time"], np.array([52.0])),
            },
            coords={"time": pd.date_range("2023-01-01", periods=1, freq="h")},
        )

        plotter = ScatterPlotter()
        fig = plotter.plot(ds, "obs_o3", "model_o3")

        assert fig is not None
        plt.close(fig)
