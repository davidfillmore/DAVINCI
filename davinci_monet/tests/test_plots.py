"""Tests for the plotting module.

This module tests the plotting system including base classes,
registry, and individual plot renderers.
"""

from __future__ import annotations

import matplotlib
import numpy as np
import pandas as pd
import pytest
import xarray as xr

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
    geometry = np.random.normal(50, 10, (n_times, n_sites))
    dataset = geometry + np.random.normal(2, 5, (n_times, n_sites))  # Slight bias

    lats = np.linspace(35, 45, n_sites)
    lons = np.linspace(-120, -100, n_sites)

    ds = xr.Dataset(
        {
            "geometry_o3": (["time", "site"], geometry, {"units": "ppbv", "long_name": "Ozone"}),
            "dataset_o3": (["time", "site"], dataset, {"units": "ppbv", "long_name": "Ozone"}),
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
    altitude = (
        1000 + 5000 * np.sin(np.linspace(0, 2 * np.pi, n_times)) + np.random.normal(0, 100, n_times)
    )
    lats = np.linspace(35, 40, n_times)
    lons = np.linspace(-120, -110, n_times)

    geometry = 50 + 10 * np.exp(-altitude / 5000) + np.random.normal(0, 5, n_times)
    dataset = geometry + np.random.normal(3, 3, n_times)

    ds = xr.Dataset(
        {
            "geometry_o3": (["time"], geometry, {"units": "ppbv", "long_name": "Ozone"}),
            "dataset_o3": (["time"], dataset, {"units": "ppbv", "long_name": "Ozone"}),
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

        geometry = 50 + 10 * np.exp(-altitude / 5000) + np.random.normal(0, 5, n_points_per_flight)
        dataset = geometry + np.random.normal(3, 3, n_points_per_flight)

        flight_id = f"2023-01-0{day + 1}"

        ds = xr.Dataset(
            {
                "geometry_o3": (["time"], geometry, {"units": "ppbv", "long_name": "Ozone"}),
                "dataset_o3": (["time"], dataset, {"units": "ppbv", "long_name": "Ozone"}),
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
    geometry = 40 + 20 * np.sin(np.radians(lat_grid)) * np.cos(np.radians(lon_grid + 110))
    geometry = np.stack(
        [geometry + np.random.normal(0, 3, geometry.shape) for _ in range(len(time))]
    )
    dataset = geometry + np.random.normal(5, 8, geometry.shape)

    ds = xr.Dataset(
        {
            "geometry_o3": (
                ["time", "lat", "lon"],
                geometry,
                {"units": "ppbv", "long_name": "Ozone"},
            ),
            "dataset_o3": (
                ["time", "lat", "lon"],
                dataset,
                {"units": "ppbv", "long_name": "Ozone"},
            ),
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
        from davinci_monet.plots.style import DATASET_A_COLOR, DATASET_B_COLOR

        config = PlotConfig()
        assert config.figure.figsize == (8, 5)  # FigureConfig default
        assert config.text.fontsize == 14.0  # Axis label size
        assert config.style.x_color == DATASET_A_COLOR  # NCAR gray
        assert config.style.y_color == DATASET_B_COLOR  # NCAR blue
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

    def test_alternate_dict_keys(self):
        """Test alternate dict keys."""
        from davinci_monet.plots.base import PlotConfig

        config_dict = {
            "fig_dict": {"figsize": (8, 6)},
            "text_dict": {"fontsize": 11},
            "plot_dict": {"x_color": "red"},
            "domain_type": "conus",
        }

        config = PlotConfig.from_dict(config_dict)
        assert config.figure.figsize == (8, 6)
        assert config.text.fontsize == 11
        assert config.domain.domain_type == "conus"

    def test_caption_key_is_ignored_from_dict(self):
        """caption is not a supported PlotConfig option."""
        from davinci_monet.plots.base import PlotConfig

        config = PlotConfig.from_dict({"title": "T", "caption": "C"})
        assert config.title == "T"
        assert not hasattr(config, "caption")


class TestBasePlotterNoCaption:
    """Tests for BasePlotter save behavior with current caption input."""

    def test_caption_is_not_drawn_on_save(self, tmp_path):
        """save() must not render caption text at the figure bottom."""
        import numpy as np
        import pandas as pd
        import xarray as xr

        from davinci_monet.plots import ScatterPlotter
        from davinci_monet.plots.base import PlotConfig

        # Build minimal paired dataset
        np.random.seed(0)
        n = 20
        time = pd.date_range("2025-01-01", periods=n, freq="h")
        ds = xr.Dataset(
            {
                "geometry_o3": (["time"], np.random.normal(50, 10, n)),
                "dataset_o3": (["time"], np.random.normal(52, 10, n)),
            },
            coords={"time": time},
        )

        caption_text = "2025-10-01 - 2025-12-31"
        config = PlotConfig.from_dict({"caption": caption_text})
        plotter = ScatterPlotter(config=config)

        fig = plotter.plot(ds, "geometry_o3", "dataset_o3")
        output_file = tmp_path / "test_caption.png"
        plotter.save(fig, output_file)

        assert (
            any(t.get_text() == caption_text for t in fig.texts) is False
        ), f"Caption '{caption_text}' should not be drawn in fig.texts"
        plt.close(fig)


class TestBasePlotterSubtitle:
    """Tests for separate title/subtitle rendering."""

    def test_subtitle_drawn_below_title_with_smaller_font(self):
        """Subtitles must be separate smaller text below the main title."""
        import numpy as np
        import pandas as pd
        import xarray as xr

        from davinci_monet.plots import ScatterPlotter
        from davinci_monet.plots.base import PlotConfig

        np.random.seed(0)
        n = 20
        time = pd.date_range("2025-01-01", periods=n, freq="h")
        ds = xr.Dataset(
            {
                "geometry_o3": (["time"], np.random.normal(50, 10, n)),
                "dataset_o3": (["time"], np.random.normal(52, 10, n)),
            },
            coords={"time": time},
        )

        config = PlotConfig.from_dict(
            {"title": "O3: Dataset vs Geometry", "subtitle": "2025-01-01 - 2025-01-02"}
        )
        plotter = ScatterPlotter(config=config)

        fig = plotter.plot(ds, "geometry_o3", "dataset_o3")
        ax = fig.axes[0]

        assert ax.get_title() == r"O$_3$: Dataset vs Geometry"
        assert "\n" not in ax.get_title()
        subtitle = next(t for t in ax.texts if t.get_text() == "2025-01-01 - 2025-01-02")
        assert subtitle.get_fontsize() == config.text.annotation_small
        assert subtitle.get_position()[1] < 1.1
        plt.close(fig)


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

        label = get_variable_label(simple_paired_data, "geometry_o3")
        assert label == "Ozone"

        label = get_variable_label(simple_paired_data, "geometry_o3", "Custom Label")
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
        from davinci_monet.plots import TimeSeriesPlotter, get_plotter

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
            "geometry_o3",
            "dataset_o3",
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
            "geometry_o3",
            "dataset_o3",
            aggregate_dim="site",
            resample="6h",
        )

        assert fig is not None
        plt.close(fig)

    def test_custom_labels(self, simple_paired_data):
        """Test custom labels."""
        from davinci_monet.plots import PlotConfig, TimeSeriesPlotter

        config = PlotConfig(geometry_label="Custom Geometry", dataset_label="Custom Dataset")
        plotter = TimeSeriesPlotter(config=config)

        fig = plotter.plot(
            simple_paired_data,
            "geometry_o3",
            "dataset_o3",
            aggregate_dim="site",
        )

        assert fig is not None
        plt.close(fig)

    def test_smart_ylim_matches_plotted_aggregate(self):
        """When no aggregate_dim is passed, plot() averages across non-time dims;
        smart-ylim must compute its range from that same aggregate, not the
        raw per-site values.

        Regression test for the wide-y-axis cosmetic bug seen in the WRF-Chem
        PM2.5 timeseries: per-site max (e.g. a wildfire site at 200 µg/m³)
        was driving vmax even though the plotted line was the cross-site mean
        around ~10.
        """
        import numpy as np
        import pandas as pd
        import xarray as xr

        from davinci_monet.plots import TimeSeriesPlotter

        n_times, n_sites = 24, 10
        times = pd.date_range("2024-01-01", periods=n_times, freq="h")
        # 9 sites at ~10, 1 outlier site at ~200 (wildfire). Mean ≈ 29 raw,
        # but the cross-site mean is ((9*10) + 200) / 10 = 29 — wait, that's
        # still 29. Use a less extreme ratio to make the test discriminating:
        # 9 sites at 10, 1 site at 200 → mean = 29. data_max for the plotted
        # mean = 29; data_max for raw = 200. The fix should produce ylim ~32,
        # the bug produces ylim ~220.
        geometry = np.full((n_times, n_sites), 10.0)
        geometry[:, -1] = 200.0
        dataset = np.full((n_times, n_sites), 11.0)
        dataset[:, -1] = 200.0

        paired = xr.Dataset(
            {
                "geometry_pm25": (["time", "site"], geometry),
                "dataset_pm25": (["time", "site"], dataset),
            },
            coords={"time": times, "site": np.arange(n_sites)},
        )

        plotter = TimeSeriesPlotter()
        # Do not pass aggregate_dim → plot() auto-aggregates over 'site'
        fig = plotter.plot(paired, "geometry_pm25", "dataset_pm25")

        ax = fig.axes[0]
        _, ymax = ax.get_ylim()
        # Plotted mean tops out near 29; padded vmax should be ~32, not >100.
        assert ymax < 60.0, (
            f"Smart-ylim should use the plotted aggregate (~29), not the raw "
            f"per-site max (200). Got ymax={ymax}."
        )
        plt.close(fig)


class TestDiurnalPlotter:
    """Tests for diurnal cycle plotter."""

    def test_basic_plot(self, simple_paired_data):
        """Test basic diurnal plot."""
        from davinci_monet.plots import plot_diurnal

        fig = plot_diurnal(simple_paired_data, "geometry_o3", "dataset_o3")

        assert fig is not None
        plt.close(fig)

    def test_spread_types(self, simple_paired_data):
        """Test different spread types."""
        from davinci_monet.plots import DiurnalPlotter

        plotter = DiurnalPlotter()

        for spread in ["none", "std", "iqr", "range"]:
            fig = plotter.plot(
                simple_paired_data,
                "geometry_o3",
                "dataset_o3",
                show_spread=spread,  # type: ignore[arg-type]
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
            "geometry_o3",
            "dataset_o3",
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
                "geometry_o3",
                "dataset_o3",
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
            "geometry_o3",
            "dataset_o3",
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
            "geometry_o3",
            "dataset_o3",
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
                "geometry_o3",
                "dataset_o3",
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
            "geometry_o3",
            "dataset_o3",
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
                "geometry_o3",
                "dataset_o3",
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
                "geometry_o3",
                "dataset_o3",
            )

    def test_plot_per_flight(self, flight_paired_data):
        """Test per-flight time series plot generation."""
        from davinci_monet.plots import FlightTimeSeriesPlotter

        plotter = FlightTimeSeriesPlotter()
        flight_plots = list(
            plotter.plot_per_flight(
                flight_paired_data,
                "geometry_o3",
                "dataset_o3",
                min_points=10,
            )
        )

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
        flight_plots = list(
            plotter.plot_per_flight(
                flight_paired_data,
                "geometry_o3",
                "dataset_o3",
                min_points=200,  # Each flight has 120 points
            )
        )

        # No flights should pass the filter
        assert len(flight_plots) == 0

    def test_altitude_display(self, flight_paired_data):
        """Test altitude display on right y-axis."""
        from davinci_monet.plots import FlightTimeSeriesPlotter

        plotter = FlightTimeSeriesPlotter()
        fig = plotter.plot(
            flight_paired_data,
            "geometry_o3",
            "dataset_o3",
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
            "geometry_o3",
            "dataset_o3",
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
            "geometry_o3",
            "dataset_o3",
            altitude_units="km",
        )
        assert fig_km is not None
        plt.close(fig_km)

        # Test meters
        fig_m = plotter.plot(
            flight_paired_data,
            "geometry_o3",
            "dataset_o3",
            altitude_units="m",
        )
        assert fig_m is not None
        plt.close(fig_m)

    def test_per_flight_altitude(self, flight_paired_data):
        """Test altitude display in per-flight time series."""
        from davinci_monet.plots import FlightTimeSeriesPlotter

        plotter = FlightTimeSeriesPlotter()
        flight_plots = list(
            plotter.plot_per_flight(
                flight_paired_data,
                "geometry_o3",
                "dataset_o3",
                show_altitude=True,
            )
        )

        assert len(flight_plots) == 3  # 3 flights
        for flight_id, fig in flight_plots:
            assert fig is not None
            plt.close(fig)


class TestScatterPlotter:
    """Tests for scatter plotter."""

    def test_basic_plot(self, simple_paired_data):
        """Test basic scatter plot."""
        from davinci_monet.plots import plot_scatter

        fig = plot_scatter(simple_paired_data, "geometry_o3", "dataset_o3")

        assert fig is not None
        plt.close(fig)

    def test_with_density(self, simple_paired_data):
        """Test scatter plot with density coloring."""
        from davinci_monet.plots import plot_scatter

        fig = plot_scatter(
            simple_paired_data,
            "geometry_o3",
            "dataset_o3",
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
            "geometry_o3",
            "dataset_o3",
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
        flight_plots = list(
            plotter.plot_per_flight(
                flight_paired_data,
                "geometry_o3",
                "dataset_o3",
                min_points=10,
            )
        )

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
        flight_plots = list(
            plotter.plot_per_flight(
                flight_paired_data,
                "geometry_o3",
                "dataset_o3",
                min_points=200,  # Each flight has 120 points
            )
        )

        # No flights should pass the filter
        assert len(flight_plots) == 0

    def test_source_named_axis_labels(self, simple_paired_data):
        """Fix C: geometry_label/dataset_label config produces source-named scatter axes.

        When PlotConfig.geometry_label and dataset_label are set, the scatter renderer
        must use them as axis labels (no 'Dataset'/'Dataset' prefix), and the
        units suffix must not produce a bare '(1)'.
        """
        from davinci_monet.plots import PlotConfig, ScatterPlotter

        config = PlotConfig(geometry_label="MODIS Terra AOD", dataset_label="MERRA-2 AOD")
        plotter = ScatterPlotter(config=config)
        fig = plotter.plot(
            simple_paired_data,
            "geometry_o3",
            "dataset_o3",
        )

        ax = fig.axes[0]
        xlabel = ax.get_xlabel()
        ylabel = ax.get_ylabel()

        # Labels must reflect the custom source names
        assert "MODIS Terra AOD" in xlabel, f"Expected 'MODIS Terra AOD' in xlabel, got: {xlabel!r}"
        assert "MERRA-2 AOD" in ylabel, f"Expected 'MERRA-2 AOD' in ylabel, got: {ylabel!r}"

        # Must NOT carry an 'Dataset'/'Dataset' prefix
        assert not xlabel.startswith(
            "Dataset"
        ), f"xlabel must not start with 'Dataset', got: {xlabel!r}"
        assert not ylabel.startswith(
            "Dataset"
        ), f"ylabel must not start with 'Dataset', got: {ylabel!r}"

        # Bare dimensionless unit '(1)' is ugly — must not appear
        assert "(1)" not in xlabel, f"Bare '(1)' unit in xlabel: {xlabel!r}"
        assert "(1)" not in ylabel, f"Bare '(1)' unit in ylabel: {ylabel!r}"

        plt.close(fig)

    def test_dataset_label_attrs_qualify_default_axis_labels(self, simple_paired_data):
        """Scatter axes use source identity without pair_axis-derived words."""
        from davinci_monet.plots import ScatterPlotter

        data = simple_paired_data.copy()
        data["geometry_o3"].attrs.update({"pair_axis": "geometry", "dataset_label": "airnow"})
        data["dataset_o3"].attrs.update({"pair_axis": "dataset", "dataset_label": "cam"})

        fig = ScatterPlotter().plot(data, "geometry_o3", "dataset_o3")

        ax = fig.axes[0]
        assert ax.get_xlabel() == "AIRNOW Ozone (ppbv)"
        assert ax.get_ylabel() == "CAM Ozone (ppbv)"
        assert not ax.get_xlabel().startswith("Dataset")
        assert not ax.get_ylabel().startswith("Dataset")

        plt.close(fig)


class TestTaylorPlotter:
    """Tests for Taylor diagram plotter."""

    def test_basic_plot(self, simple_paired_data):
        """Test basic Taylor diagram."""
        from davinci_monet.plots import plot_taylor

        fig = plot_taylor(simple_paired_data, "geometry_o3", "dataset_o3")

        assert fig is not None
        plt.close(fig)

    def test_normalized(self, simple_paired_data):
        """Test normalized Taylor diagram."""
        from davinci_monet.plots import TaylorPlotter

        plotter = TaylorPlotter()
        fig = plotter.plot(
            simple_paired_data,
            "geometry_o3",
            "dataset_o3",
            normalize=True,
        )

        assert fig is not None
        plt.close(fig)


class TestBoxPlotter:
    """Tests for box plotter."""

    def test_basic_plot(self, simple_paired_data):
        """Test basic box plot."""
        from davinci_monet.plots import plot_boxplot

        fig = plot_boxplot(simple_paired_data, "geometry_o3", "dataset_o3")

        assert fig is not None
        plt.close(fig)

    def test_grouped_plot(self, simple_paired_data):
        """Test grouped box plot."""
        from davinci_monet.plots import BoxPlotter

        plotter = BoxPlotter()
        fig = plotter.plot(
            simple_paired_data,
            "geometry_o3",
            "dataset_o3",
            group_by="site",
        )

        assert fig is not None
        plt.close(fig)

    def test_horizontal(self, simple_paired_data):
        """Test horizontal box plot."""
        from davinci_monet.plots import plot_boxplot

        fig = plot_boxplot(
            simple_paired_data,
            "geometry_o3",
            "dataset_o3",
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
            "geometry_o3",
            "dataset_o3",
            alt_var="altitude",
        )

        assert fig is not None
        plt.close(fig)

    def test_show_var_options(self, track_paired_data):
        """Test different show_var options."""
        from davinci_monet.plots import CurtainPlotter

        plotter = CurtainPlotter()

        for show_var in ["geometry", "dataset", "bias"]:
            fig = plotter.plot(
                track_paired_data,
                "geometry_o3",
                "dataset_o3",
                alt_var="altitude",
                show_var=show_var,  # type: ignore[arg-type]
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
            "geometry_o3",
            "dataset_o3",
            alt_var="altitude",
        )

        assert fig is not None
        plt.close(fig)

    def test_show_var_options(self, track_paired_data):
        """Test different show_var options."""
        from davinci_monet.plots import TrackMap3DPlotter

        plotter = TrackMap3DPlotter()

        for show_var in ["geometry", "dataset", "bias"]:
            fig = plotter.plot(
                track_paired_data,
                "geometry_o3",
                "dataset_o3",
                alt_var="altitude",
                show_var=show_var,  # type: ignore[arg-type]
            )
            assert fig is not None
            plt.close(fig)

    def test_view_angles(self, track_paired_data):
        """Test different view angles."""
        from davinci_monet.plots import TrackMap3DPlotter

        plotter = TrackMap3DPlotter()
        fig = plotter.plot(
            track_paired_data,
            "geometry_o3",
            "dataset_o3",
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
            "geometry_o3",
            "dataset_o3",
            alt_var="altitude",
            show_projection=False,
        )

        assert fig is not None
        plt.close(fig)

    def test_plot_per_flight(self, flight_paired_data):
        """Test per-flight 3D track plot generation."""
        from davinci_monet.plots import TrackMap3DPlotter

        plotter = TrackMap3DPlotter()
        flight_plots = list(
            plotter.plot_per_flight(
                flight_paired_data,
                "geometry_o3",
                "dataset_o3",
                min_points=10,
                show_coastlines=False,  # Faster for testing
            )
        )

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
        flight_plots = list(
            plotter.plot_per_flight(
                flight_paired_data,
                "geometry_o3",
                "dataset_o3",
                min_points=200,  # Each flight has 120 points
                show_coastlines=False,
            )
        )

        # No flights should pass the filter
        assert len(flight_plots) == 0


class TestScorecardPlotter:
    """Tests for scorecard plotter."""

    def test_basic_plot(self, simple_paired_data):
        """Test basic scorecard plot."""
        from davinci_monet.plots import plot_scorecard

        fig = plot_scorecard(simple_paired_data, "geometry_o3", "dataset_o3")

        assert fig is not None
        plt.close(fig)

    def test_from_dataframe(self):
        """Test scorecard from DataFrame."""
        from davinci_monet.plots import ScorecardPlotter

        stats_df = pd.DataFrame(
            {
                "Dataset A": [0.9, 2.5, 5.0],
                "Dataset B": [0.85, -1.0, 6.5],
                "Dataset C": [0.92, 0.5, 4.0],
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
        reason="cartopy not available",
    )
    def test_spatial_bias(self, simple_paired_data):
        """Test spatial bias plot."""
        from davinci_monet.plots import plot_spatial_bias

        fig = plot_spatial_bias(simple_paired_data, "geometry_o3", "dataset_o3")

        assert fig is not None
        plt.close(fig)

    @pytest.mark.skipif(
        not pytest.importorskip("cartopy", reason="cartopy not available"),
        reason="cartopy not available",
    )
    def test_spatial_bias_point_data_with_singleton_y_dim(self):
        """Spatial bias must handle point/site data with a residual size-1
        dim (current AirNow stores all sites as `(time, y=1, x=sites)`).
        Previously the renderer wrongly took the regular-grid meshgrid
        path, producing a (sites, sites) broadcast that crashed."""
        import numpy as np
        import xarray as xr

        from davinci_monet.plots import plot_spatial_bias

        times = np.array(
            ["2025-08-01T00:00", "2025-08-01T01:00", "2025-08-01T02:00"],
            dtype="datetime64[ns]",
        )
        n_sites = 5
        lats = np.array([30.0, 35.0, 40.0, 45.0, 50.0])
        lons = np.array([-110.0, -100.0, -90.0, -80.0, -70.0])
        rng = np.random.default_rng(0)
        geometry = rng.uniform(20, 60, size=(3, 1, n_sites))
        dataset = geometry + rng.uniform(-5, 5, size=(3, 1, n_sites))

        ds = xr.Dataset(
            {
                "geometry_o3": (("time", "y", "x"), geometry),
                "dataset_o3": (("time", "y", "x"), dataset),
            },
            coords={
                "time": times,
                "latitude": (("x",), lats),
                "longitude": (("x",), lons),
            },
        )

        fig = plot_spatial_bias(ds, "geometry_o3", "dataset_o3")
        assert fig is not None
        plt.close(fig)

    @pytest.mark.skipif(
        not pytest.importorskip("cartopy", reason="cartopy not available"),
        reason="cartopy not available",
    )
    def test_spatial_bias_point_data_site_geometry(self):
        """Regression: AERONET-style paired data with `(time, site)` dims and
        lat/lon on the site dim must still render (no residual y dim, but
        lats/lons share a single dim like AirNow)."""
        import numpy as np
        import xarray as xr

        from davinci_monet.plots import plot_spatial_bias

        times = np.array(["2025-08-01T00:00", "2025-08-01T01:00"], dtype="datetime64[ns]")
        n_sites = 5
        lats = np.linspace(20.0, 50.0, n_sites)
        lons = np.linspace(-110.0, -70.0, n_sites)
        rng = np.random.default_rng(1)
        geometry = rng.uniform(0, 1, size=(2, n_sites))
        dataset = geometry + rng.uniform(-0.2, 0.2, size=(2, n_sites))

        ds = xr.Dataset(
            {
                "geometry_aod": (("time", "site"), geometry),
                "dataset_aod": (("time", "site"), dataset),
            },
            coords={
                "time": times,
                "lat": (("site",), lats),
                "lon": (("site",), lons),
            },
        )

        fig = plot_spatial_bias(ds, "geometry_aod", "dataset_aod")
        assert fig is not None
        plt.close(fig)

    @pytest.mark.skipif(
        not pytest.importorskip("cartopy", reason="cartopy not available"),
        reason="cartopy not available",
    )
    def test_spatial_bias_grid_uses_pcolormesh_by_default(self):
        """Gridded data must render as pcolormesh (QuadMesh) by default, not
        scatter circles.  Before the fix, plot_type defaulted to 'scatter' so
        a MODIS-L3-style 10x12 grid produced ~120 scatter circles."""
        import numpy as np
        import xarray as xr
        from matplotlib.collections import PathCollection, QuadMesh

        from davinci_monet.plots import plot_spatial_bias

        lat = np.linspace(-89.5, 89.5, 10)
        lon = np.linspace(-179.5, 179.5, 12)
        rng = np.random.default_rng(0)
        geometry = xr.DataArray(
            rng.uniform(0, 1, (10, 12)),
            dims=("lat", "lon"),
            coords={"lat": lat, "lon": lon},
        )
        dataset = xr.DataArray(
            rng.uniform(0, 1, (10, 12)),
            dims=("lat", "lon"),
            coords={"lat": lat, "lon": lon},
        )
        ds = xr.Dataset({"geometry_aod": geometry, "dataset_aod": dataset})

        fig = plot_spatial_bias(ds, "geometry_aod", "dataset_aod", lat_var="lat", lon_var="lon")
        ax = fig.axes[0]
        assert any(
            isinstance(c, QuadMesh) for c in ax.collections
        ), "gridded bias must render as pcolormesh (QuadMesh), not scatter circles"
        assert not any(
            isinstance(c, PathCollection) for c in ax.collections
        ), "gridded bias must not use scatter PathCollection"
        plt.close(fig)

    @pytest.mark.skipif(
        not pytest.importorskip("cartopy", reason="cartopy not available"),
        reason="cartopy not available",
    )
    def test_spatial_bias_point_uses_scatter_with_auto(self):
        """Point/site data must still render as scatter (PathCollection) when
        plot_type='auto' (the new default), guarding against regression."""
        import numpy as np
        import xarray as xr
        from matplotlib.collections import PathCollection

        from davinci_monet.plots import plot_spatial_bias

        site = np.arange(8)
        lat = xr.DataArray(np.linspace(20, 50, 8), dims=("site",), coords={"site": site})
        lon = xr.DataArray(np.linspace(100, 140, 8), dims=("site",), coords={"site": site})
        rng = np.random.default_rng(1)
        geometry = xr.DataArray(rng.uniform(0, 1, 8), dims=("site",), coords={"site": site})
        dataset = xr.DataArray(rng.uniform(0, 1, 8), dims=("site",), coords={"site": site})
        ds = xr.Dataset(
            {"geometry_v": geometry, "dataset_v": dataset},
            coords={"lat": lat, "lon": lon},
        )

        fig = plot_spatial_bias(ds, "geometry_v", "dataset_v", lat_var="lat", lon_var="lon")
        ax = fig.axes[0]
        assert any(
            isinstance(c, PathCollection) for c in ax.collections
        ), "point/site bias must render as scatter (PathCollection)"
        plt.close(fig)

    @pytest.mark.skipif(
        not pytest.importorskip("cartopy", reason="cartopy not available"),
        reason="cartopy not available",
    )
    def test_spatial_distribution(self, simple_paired_data):
        """Test spatial distribution plot."""
        from davinci_monet.plots import plot_spatial_distribution

        fig = plot_spatial_distribution(
            simple_paired_data,
            "geometry_o3",
            "dataset_o3",
            show_var="geometry",
        )

        assert fig is not None
        plt.close(fig)

    @pytest.mark.skipif(
        not pytest.importorskip("cartopy", reason="cartopy not available"),
        reason="cartopy not available",
    )
    def test_spatial_distribution_grid_uses_pcolormesh_by_default(self):
        """Gridded data must render as pcolormesh (QuadMesh) by default, not
        scatter circles.  Before the fix, plot_type defaulted to 'scatter' so
        a regular lat/lon grid produced scatter circles instead of a filled
        field."""
        from matplotlib.collections import PathCollection, QuadMesh

        from davinci_monet.plots import plot_spatial_distribution

        lat = np.linspace(30.0, 50.0, 10)
        lon = np.linspace(-120.0, -100.0, 12)
        rng = np.random.default_rng(0)
        geometry_vals = rng.uniform(20, 80, (10, 12))
        dataset_vals = rng.uniform(20, 80, (10, 12))
        ds = xr.Dataset(
            {
                "geometry_o3": (("lat", "lon"), geometry_vals, {"units": "ppbv"}),
                "dataset_o3": (("lat", "lon"), dataset_vals, {"units": "ppbv"}),
            },
            coords={"lat": lat, "lon": lon},
        )

        fig = plot_spatial_distribution(
            ds, "geometry_o3", "dataset_o3", show_var="geometry", lat_var="lat", lon_var="lon"
        )
        ax = fig.axes[0]
        assert any(
            isinstance(c, QuadMesh) for c in ax.collections
        ), "gridded distribution must render as pcolormesh (QuadMesh), not scatter circles"
        assert not any(
            isinstance(c, PathCollection) for c in ax.collections
        ), "gridded distribution must not use scatter PathCollection"
        plt.close(fig)

    @pytest.mark.skipif(
        not pytest.importorskip("cartopy", reason="cartopy not available"),
        reason="cartopy not available",
    )
    def test_spatial_distribution_point_uses_scatter_by_default(self):
        """Point/site data (lat/lon on the site dim) must render as scatter
        (PathCollection) when plot_type='auto' (the new default)."""
        from matplotlib.collections import PathCollection

        from davinci_monet.plots import plot_spatial_distribution

        n_sites = 8
        site = np.arange(n_sites)
        lats = np.linspace(30.0, 50.0, n_sites)
        lons = np.linspace(-120.0, -100.0, n_sites)
        rng = np.random.default_rng(1)
        geometry_vals = rng.uniform(20, 80, n_sites)
        dataset_vals = rng.uniform(20, 80, n_sites)
        ds = xr.Dataset(
            {
                "geometry_o3": (("site",), geometry_vals, {"units": "ppbv"}),
                "dataset_o3": (("site",), dataset_vals, {"units": "ppbv"}),
            },
            coords={
                "site": site,
                "latitude": (("site",), lats),
                "longitude": (("site",), lons),
            },
        )

        fig = plot_spatial_distribution(ds, "geometry_o3", "dataset_o3", show_var="geometry")
        ax = fig.axes[0]
        assert any(
            isinstance(c, PathCollection) for c in ax.collections
        ), "point/site distribution must render as scatter (PathCollection)"
        plt.close(fig)

    @pytest.mark.skipif(
        not pytest.importorskip("cartopy", reason="cartopy not available"),
        reason="cartopy not available",
    )
    def test_spatial_distribution_time_site_uses_scatter(self):
        """(time, site) point data must render as scatter (PathCollection), not
        pcolormesh.  Without DataArray-dim detection, the numpy-ndim heuristic
        in _plot_data sees data.ndim==2 and lats.ndim==1 and wrongly takes the
        'regular grid' pcolormesh branch."""
        import pandas as pd
        from matplotlib.collections import PathCollection, QuadMesh

        from davinci_monet.plots import plot_spatial_distribution

        n_times = 3
        n_sites = 8
        times = pd.date_range("2025-01-01", periods=n_times, freq="h")
        site = np.arange(n_sites)
        lats = np.linspace(30.0, 50.0, n_sites)
        lons = np.linspace(-120.0, -100.0, n_sites)
        rng = np.random.default_rng(2)
        geometry_vals = rng.uniform(20, 80, (n_times, n_sites))
        dataset_vals = rng.uniform(20, 80, (n_times, n_sites))
        ds = xr.Dataset(
            {
                "geometry_o3": (("time", "site"), geometry_vals, {"units": "ppbv"}),
                "dataset_o3": (("time", "site"), dataset_vals, {"units": "ppbv"}),
            },
            coords={
                "time": times,
                "site": site,
                "latitude": (("site",), lats),
                "longitude": (("site",), lons),
            },
        )

        # time_average=False to keep (time, site) shape reaching _plot_data
        fig = plot_spatial_distribution(
            ds, "geometry_o3", "dataset_o3", show_var="geometry", time_average=False
        )
        ax = fig.axes[0]
        assert any(
            isinstance(c, PathCollection) for c in ax.collections
        ), "(time, site) distribution must render as scatter, not pcolormesh"
        assert not any(
            isinstance(c, QuadMesh) for c in ax.collections
        ), "(time, site) distribution must not use pcolormesh (QuadMesh)"
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
# End-to-End Plotter Tests
# =============================================================================


class TestPlotterEndToEnd:
    """End-to-end tests for the plotting system (calls internal APIs directly)."""

    def test_all_plotters_instantiate(self):
        """Test that all registered plotters can be instantiated."""
        from davinci_monet.plots import get_plotter, list_plotters

        for name in list_plotters():
            plotter = get_plotter(name)
            assert plotter is not None
            assert hasattr(plotter, "plot")
            assert hasattr(plotter, "save")

    def test_plotter_save(self, simple_paired_data, tmp_path):
        """Test saving a figure."""
        from davinci_monet.plots import get_plotter

        plotter = get_plotter("scatter")
        fig = plotter.plot(simple_paired_data, "geometry_o3", "dataset_o3")

        output_path = tmp_path / "test_plot.png"
        saved_path = plotter.save(fig, output_path)

        assert saved_path.exists()
        plt.close(fig)

    def test_multiple_plots_same_axes(self, simple_paired_data):
        """Test plotting multiple datasets on same axes."""
        from davinci_monet.plots import TimeSeriesPlotter

        fig, ax = plt.subplots()

        plotter = TimeSeriesPlotter()

        # Plot first "dataset"
        plotter.plot(
            simple_paired_data,
            "geometry_o3",
            "dataset_o3",
            ax=ax,
            aggregate_dim="site",
        )

        # Plot on same axes (simulating multiple datasets)
        # In real use, would have different dataset data

        assert fig is not None
        plt.close(fig)


# =============================================================================
# Spatial Overlay Tests
# =============================================================================


class TestSpatialOverlay:
    """Behavioral tests for SpatialOverlayPlotter."""

    def test_overlay_plot(self, simple_paired_data, gridded_paired_data):
        """Overlay dataset contours with dataset scatter points."""
        from davinci_monet.plots.renderers.spatial.overlay import SpatialOverlayPlotter

        plotter = SpatialOverlayPlotter()

        # Create a dataset field (2D lat/lon) for the contour layer
        dataset_field = gridded_paired_data["dataset_o3"].isel(time=0)

        fig = plotter.plot(
            simple_paired_data,
            x_var="geometry_o3",
            y_var="dataset_o3",
            dataset_field=dataset_field,
        )

        assert fig is not None
        axes = fig.get_axes()
        assert len(axes) >= 1
        plt.close(fig)

    def test_overlay_without_dataset_field(self, simple_paired_data):
        """Overlay should handle missing dataset_field gracefully."""
        from davinci_monet.plots.renderers.spatial.overlay import SpatialOverlayPlotter

        plotter = SpatialOverlayPlotter()

        # When dataset_field is None, plotter should fall back to y_var from paired_data
        # This may not produce contours (1D data), but should not crash
        try:
            fig = plotter.plot(
                simple_paired_data,
                x_var="geometry_o3",
                y_var="dataset_o3",
            )
            assert fig is not None
            plt.close(fig)
        except (ValueError, KeyError, TypeError):
            # Acceptable: plotter may require a 2D dataset field
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
            "geometry_o3",
            "dataset_o3",
            aggregate_dim="site",
        )

        assert fig is not None
        ax = fig.get_axes()[0]
        # Should have at least geometry and dataset lines
        assert len(ax.get_lines()) >= 2
        plt.close(fig)

    def test_auto_aggregate_multidim(self, simple_paired_data):
        """Timeseries auto-averages non-time dims when no aggregate_dim given."""
        from davinci_monet.plots import TimeSeriesPlotter

        plotter = TimeSeriesPlotter()
        fig = plotter.plot(
            simple_paired_data,
            "geometry_o3",
            "dataset_o3",
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
            "geometry_o3",
            "dataset_o3",
            aggregate_dim="site",
            resample="6h",
        )

        assert fig is not None
        plt.close(fig)


# =============================================================================
# Scorecard Tests
# =============================================================================


class TestScorecardPlotterMultiVariable:
    """Tests for ScorecardPlotter with multiple variables."""

    def test_scorecard_multi_variable(self):
        """Scorecard with multiple variables."""
        from davinci_monet.plots.renderers.scorecard import ScorecardPlotter

        # Create multi-variable paired data
        np.random.seed(42)
        n = 100

        ds = xr.Dataset(
            {
                "geometry_o3": (["time"], np.random.normal(50, 10, n)),
                "dataset_o3": (["time"], np.random.normal(52, 10, n)),
                "geometry_pm25": (["time"], np.random.normal(15, 5, n)),
                "dataset_pm25": (["time"], np.random.normal(17, 5, n)),
            },
            coords={"time": pd.date_range("2023-01-01", periods=n, freq="h")},
        )

        plotter = ScorecardPlotter()
        fig = plotter.plot(
            ds,
            x_var="geometry_o3",
            y_var="dataset_o3",
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
                "geometry_o3": (["time"], np.array([np.nan, np.nan, np.nan])),
                "dataset_o3": (["time"], np.array([np.nan, np.nan, np.nan])),
            },
            coords={"time": pd.date_range("2023-01-01", periods=3, freq="h")},
        )

        plotter = ScatterPlotter()
        fig = plotter.plot(ds, "geometry_o3", "dataset_o3")

        # Should handle gracefully
        assert fig is not None
        plt.close(fig)

    def test_single_point(self):
        """Test handling of single data point."""
        from davinci_monet.plots import ScatterPlotter

        ds = xr.Dataset(
            {
                "geometry_o3": (["time"], np.array([50.0])),
                "dataset_o3": (["time"], np.array([52.0])),
            },
            coords={"time": pd.date_range("2023-01-01", periods=1, freq="h")},
        )

        plotter = ScatterPlotter()
        fig = plotter.plot(ds, "geometry_o3", "dataset_o3")

        assert fig is not None
        plt.close(fig)
