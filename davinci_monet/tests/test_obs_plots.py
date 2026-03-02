"""Tests for observation-only plotting module.

This module tests the ObsPlotter base class and observation-only
plot renderers (flight track map, vertical profile, time series, histogram).
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
def track_obs_data() -> xr.Dataset:
    """Create observation-only aircraft track dataset.

    200 time points with O3 and NO variables, altitude/lat/lon coords.
    Simulates a single flight track with realistic vertical structure.
    """
    np.random.seed(42)
    n_times = 200

    time = pd.date_range("2012-05-18 14:00", periods=n_times, freq="1min")

    # Simulate aircraft track with ascent/descent profile
    altitude = 1000 + 5000 * np.sin(np.linspace(0, 2 * np.pi, n_times)) + np.random.normal(0, 100, n_times)
    lats = np.linspace(35, 40, n_times)
    lons = np.linspace(-102, -96, n_times)

    # O3 inversely correlated with altitude (higher near surface)
    o3 = 50 + 10 * np.exp(-altitude / 5000) + np.random.normal(0, 5, n_times)
    # NO with altitude dependence and noise
    no = 0.5 + 2.0 * np.exp(-altitude / 3000) + np.random.normal(0, 0.3, n_times)

    ds = xr.Dataset(
        {
            "O3": (["time"], o3, {"units": "ppbv", "long_name": "Ozone"}),
            "NO": (["time"], no, {"units": "ppbv", "long_name": "Nitric Oxide"}),
        },
        coords={
            "time": time,
            "altitude": ("time", altitude, {"units": "m", "long_name": "GPS Altitude"}),
            "latitude": ("time", lats),
            "longitude": ("time", lons),
        },
        attrs={"geometry": "track"},
    )
    return ds


@pytest.fixture
def multi_flight_obs_data() -> xr.Dataset:
    """Create observation-only dataset with multiple flights.

    3 flights with 120 points each, O3 variable, flight coord for
    per-flight analysis. Simulates DC3 campaign structure.
    """
    np.random.seed(42)
    n_points_per_flight = 120
    flights = []

    for day in range(3):
        base_time = pd.Timestamp(f"2012-05-{18 + day} 14:00:00")
        time = pd.date_range(base_time, periods=n_points_per_flight, freq="1min")

        # Simulate aircraft track
        altitude = 1000 + 5000 * np.sin(np.linspace(0, np.pi, n_points_per_flight))
        lats = np.linspace(35 + day, 40 + day, n_points_per_flight)
        lons = np.linspace(-102, -96, n_points_per_flight)

        o3 = 50 + 10 * np.exp(-altitude / 5000) + np.random.normal(0, 5, n_points_per_flight)

        flight_id = f"2012-05-{18 + day:02d}"

        ds = xr.Dataset(
            {
                "O3": (["time"], o3, {"units": "ppbv", "long_name": "Ozone"}),
            },
            coords={
                "time": time,
                "altitude": ("time", altitude, {"units": "m", "long_name": "GPS Altitude"}),
                "latitude": ("time", lats),
                "longitude": ("time", lons),
                "flight": ("time", [flight_id] * n_points_per_flight),
            },
            attrs={"geometry": "track"},
        )
        flights.append(ds)

    return xr.concat(flights, dim="time")


@pytest.fixture
def grid_lma_data() -> xr.Dataset:
    """Create synthetic LMA gridded flash density dataset (one hour)."""
    np.random.seed(42)
    n_times = 12  # 5-min steps in one hour
    n_lat = 20
    n_lon = 25

    time = pd.date_range("2012-05-29 22:00", periods=n_times, freq="5min")
    lats = np.linspace(33.5, 37.0, n_lat)
    lons = np.linspace(-101.0, -96.0, n_lon)

    # Create a hotspot in the center
    lat_grid, lon_grid = np.meshgrid(lats, lons, indexing="ij")
    hotspot = np.exp(
        -((lat_grid - 35.2) ** 2 + (lon_grid - 98.5) ** 2) / 0.5
    )
    # Vary intensity over time (ramp up then down)
    time_profile = np.sin(np.linspace(0, np.pi, n_times))
    flash_extent = np.zeros((n_times, n_lat, n_lon))
    for t in range(n_times):
        flash_extent[t] = hotspot * time_profile[t] * 10 + np.random.poisson(
            0.5, (n_lat, n_lon)
        )

    ds = xr.Dataset(
        {
            "flash_extent": (
                ["time", "latitude", "longitude"],
                flash_extent,
                {"units": "flashes/grid cell", "long_name": "Flash Extent Density"},
            ),
        },
        coords={
            "time": time,
            "latitude": lats,
            "longitude": lons,
        },
        attrs={"geometry": "grid", "lma_network_id": "oklma"},
    )
    return ds


@pytest.fixture
def grid_lma_data_multihour() -> xr.Dataset:
    """Create synthetic LMA data spanning 3 hours with varying activity."""
    np.random.seed(42)
    n_lat = 20
    n_lon = 25

    lats = np.linspace(33.5, 37.0, n_lat)
    lons = np.linspace(-101.0, -96.0, n_lon)
    lat_grid, lon_grid = np.meshgrid(lats, lons, indexing="ij")
    hotspot = np.exp(
        -((lat_grid - 35.2) ** 2 + (lon_grid - 98.5) ** 2) / 0.5
    )

    # 3 hours x 12 steps = 36 time steps
    time = pd.date_range("2012-05-29 22:00", periods=36, freq="5min")
    hourly_scale = [5.0, 10.0, 3.0]  # hour 2 is peak

    flash_extent = np.zeros((36, n_lat, n_lon))
    for t in range(36):
        hour_idx = t // 12
        flash_extent[t] = hotspot * hourly_scale[hour_idx] + np.random.poisson(
            0.5, (n_lat, n_lon)
        )

    ds = xr.Dataset(
        {
            "flash_extent": (
                ["time", "latitude", "longitude"],
                flash_extent,
                {"units": "flashes/grid cell", "long_name": "Flash Extent Density"},
            ),
        },
        coords={
            "time": time,
            "latitude": lats,
            "longitude": lons,
        },
        attrs={"geometry": "grid", "lma_network_id": "oklma"},
    )
    return ds


# =============================================================================
# ObsPlotter Base Class Tests
# =============================================================================


class TestObsPlotterBase:
    """Tests for the ObsPlotter abstract base class."""

    def test_cannot_instantiate_abstract(self):
        """ObsPlotter is abstract and cannot be instantiated directly."""
        from davinci_monet.plots.obs_base import ObsPlotter

        with pytest.raises(TypeError):
            ObsPlotter()

    def test_subclass_must_implement_plot(self):
        """Subclass without plot() raises TypeError on instantiation."""
        from davinci_monet.plots.obs_base import ObsPlotter

        class IncompleteObsPlotter(ObsPlotter):
            name = "incomplete"

        with pytest.raises(TypeError):
            IncompleteObsPlotter()

    def test_subclass_with_plot_works(self, track_obs_data):
        """Concrete subclass with plot() can be instantiated and used."""
        from davinci_monet.plots.obs_base import ObsPlotter

        class ConcreteObsPlotter(ObsPlotter):
            name = "concrete"

            def plot(self, obs_data, variable, ax=None, **kwargs):
                fig, ax = self.create_figure()
                data = obs_data[variable].values
                ax.plot(data)
                ax.set_title(f"{variable} obs-only plot")
                return fig

        plotter = ConcreteObsPlotter()
        fig = plotter.plot(track_obs_data, "O3")

        assert fig is not None
        assert len(fig.axes) >= 1
        plt.close(fig)

    def test_create_figure(self):
        """create_figure() returns a (fig, ax) tuple with correct defaults."""
        from davinci_monet.plots.obs_base import ObsPlotter

        class MinimalPlotter(ObsPlotter):
            name = "minimal"

            def plot(self, obs_data, variable, ax=None, **kwargs):
                fig, ax = self.create_figure()
                return fig

        plotter = MinimalPlotter()
        fig, ax = plotter.create_figure()

        assert isinstance(fig, matplotlib.figure.Figure)
        assert isinstance(ax, matplotlib.axes.Axes)
        plt.close(fig)

    def test_save(self, track_obs_data, tmp_path):
        """save() writes a file to disk and returns the path."""
        from davinci_monet.plots.obs_base import ObsPlotter

        class SimplePlotter(ObsPlotter):
            name = "simple"

            def plot(self, obs_data, variable, ax=None, **kwargs):
                fig, ax = self.create_figure()
                ax.plot(obs_data[variable].values)
                return fig

        plotter = SimplePlotter()
        fig = plotter.plot(track_obs_data, "O3")

        output_path = tmp_path / "test_obs_plot.png"
        saved_path = plotter.save(fig, output_path)

        assert saved_path.exists()
        assert saved_path == output_path
        plt.close(fig)

    def test_close(self, track_obs_data):
        """close() closes the figure without error."""
        from davinci_monet.plots.obs_base import ObsPlotter

        class SimplePlotter(ObsPlotter):
            name = "simple"

            def plot(self, obs_data, variable, ax=None, **kwargs):
                fig, ax = self.create_figure()
                ax.plot(obs_data[variable].values)
                return fig

        plotter = SimplePlotter()
        fig = plotter.plot(track_obs_data, "O3")

        # Should not raise
        plotter.close(fig)

    def test_default_config(self):
        """Default config uses PlotConfig defaults."""
        from davinci_monet.plots.obs_base import ObsPlotter
        from davinci_monet.plots.base import PlotConfig

        class MinimalPlotter(ObsPlotter):
            name = "minimal"

            def plot(self, obs_data, variable, ax=None, **kwargs):
                fig, ax = self.create_figure()
                return fig

        plotter = MinimalPlotter()
        assert plotter.config is not None
        assert plotter.config.figure.dpi == 300
        assert plotter.config.figure.facecolor == "white"

    def test_custom_config(self):
        """Custom PlotConfig is used when provided."""
        from davinci_monet.plots.obs_base import ObsPlotter
        from davinci_monet.plots.base import PlotConfig, FigureConfig

        class MinimalPlotter(ObsPlotter):
            name = "minimal"

            def plot(self, obs_data, variable, ax=None, **kwargs):
                fig, ax = self.create_figure()
                return fig

        config = PlotConfig(figure=FigureConfig(figsize=(12, 8), dpi=150))
        plotter = MinimalPlotter(config=config)

        assert plotter.config.figure.figsize == (12, 8)
        assert plotter.config.figure.dpi == 150

    def test_default_figsize_override(self):
        """Subclass default_figsize overrides PlotConfig default when not explicitly set."""
        from davinci_monet.plots.obs_base import ObsPlotter

        class WideObsPlotter(ObsPlotter):
            name = "wide"
            default_figsize = (12, 4)

            def plot(self, obs_data, variable, ax=None, **kwargs):
                fig, ax = self.create_figure()
                return fig

        plotter = WideObsPlotter()
        # When no explicit figsize is set, default_figsize should be used
        assert plotter.config.figure.figsize == (12, 4)

    def test_save_creates_parent_dirs(self, track_obs_data, tmp_path):
        """save() creates parent directories if they don't exist."""
        from davinci_monet.plots.obs_base import ObsPlotter

        class SimplePlotter(ObsPlotter):
            name = "simple"

            def plot(self, obs_data, variable, ax=None, **kwargs):
                fig, ax = self.create_figure()
                ax.plot(obs_data[variable].values)
                return fig

        plotter = SimplePlotter()
        fig = plotter.plot(track_obs_data, "O3")

        output_path = tmp_path / "nested" / "dirs" / "test_plot.png"
        saved_path = plotter.save(fig, output_path)

        assert saved_path.exists()
        plt.close(fig)


# =============================================================================
# Flight Track Map Plotter Tests
# =============================================================================


class TestFlightTrackMapPlotter:
    """Tests for the FlightTrackMapPlotter (obs_flight_track)."""

    def test_basic_plot(self, track_obs_data):
        """Basic flight track map renders without error."""
        from davinci_monet.plots.renderers.obs.flight_track_map import FlightTrackMapPlotter

        plotter = FlightTrackMapPlotter()
        fig = plotter.plot(track_obs_data, "O3")

        assert fig is not None
        assert len(fig.axes) >= 1
        plt.close(fig)

    def test_plot_has_colorbar(self, track_obs_data):
        """Flight track map has a colorbar (figure should have >= 2 axes)."""
        from davinci_monet.plots.renderers.obs.flight_track_map import FlightTrackMapPlotter

        plotter = FlightTrackMapPlotter()
        fig = plotter.plot(track_obs_data, "O3")

        # Main axes + colorbar axes
        assert len(fig.axes) >= 2
        plt.close(fig)

    def test_multi_flight(self, multi_flight_obs_data):
        """Flight track map works with multi-flight data."""
        from davinci_monet.plots.renderers.obs.flight_track_map import FlightTrackMapPlotter

        plotter = FlightTrackMapPlotter()
        fig = plotter.plot(multi_flight_obs_data, "O3")

        assert fig is not None
        plt.close(fig)

    def test_save_output(self, track_obs_data, tmp_path):
        """Flight track map can be saved to disk."""
        from davinci_monet.plots.renderers.obs.flight_track_map import FlightTrackMapPlotter

        plotter = FlightTrackMapPlotter()
        fig = plotter.plot(track_obs_data, "O3")

        output_path = tmp_path / "flight_track.png"
        saved_path = plotter.save(fig, output_path)

        assert saved_path.exists()
        assert saved_path.stat().st_size > 0
        plt.close(fig)

    def test_registered_in_registry(self):
        """FlightTrackMapPlotter is registered as 'obs_flight_track'."""
        from davinci_monet.plots.renderers.obs.flight_track_map import FlightTrackMapPlotter  # noqa: F401
        from davinci_monet.plots.registry import has_plotter

        assert has_plotter("obs_flight_track")


# =============================================================================
# Vertical Profile Plotter Tests
# =============================================================================


class TestVerticalProfilePlotter:
    """Tests for the VerticalProfilePlotter (obs_vertical_profile)."""

    def test_basic_scatter(self, track_obs_data):
        """Scatter mode renders without error."""
        from davinci_monet.plots.renderers.obs.vertical_profile import VerticalProfilePlotter

        plotter = VerticalProfilePlotter()
        fig = plotter.plot(track_obs_data, "O3", mode="scatter")

        assert fig is not None
        assert len(fig.axes) >= 1
        plt.close(fig)

    def test_binned_mode(self, track_obs_data):
        """Binned mode renders means with std envelope."""
        from davinci_monet.plots.renderers.obs.vertical_profile import VerticalProfilePlotter

        plotter = VerticalProfilePlotter()
        fig = plotter.plot(track_obs_data, "O3", mode="binned", n_bins=10)

        assert fig is not None
        assert len(fig.axes) >= 1
        plt.close(fig)

    def test_multi_flight(self, multi_flight_obs_data):
        """Vertical profile works with multi-flight data."""
        from davinci_monet.plots.renderers.obs.vertical_profile import VerticalProfilePlotter

        plotter = VerticalProfilePlotter()
        fig = plotter.plot(multi_flight_obs_data, "O3")

        assert fig is not None
        plt.close(fig)

    def test_registered(self):
        """VerticalProfilePlotter is registered as 'obs_vertical_profile'."""
        from davinci_monet.plots.renderers.obs.vertical_profile import VerticalProfilePlotter  # noqa: F401
        from davinci_monet.plots.registry import has_plotter

        assert has_plotter("obs_vertical_profile")


# =============================================================================
# Obs Time Series Plotter Tests
# =============================================================================


class TestObsTimeSeriesPlotter:
    """Tests for the ObsTimeSeriesPlotter (obs_timeseries)."""

    def test_basic_plot(self, track_obs_data):
        """Basic obs time series renders without error."""
        from davinci_monet.plots.renderers.obs.obs_timeseries import ObsTimeSeriesPlotter

        plotter = ObsTimeSeriesPlotter()
        fig = plotter.plot(track_obs_data, "O3")

        assert fig is not None
        assert len(fig.axes) >= 1
        plt.close(fig)

    def test_with_altitude_axis(self, track_obs_data):
        """Altitude overlay adds a secondary y-axis (>= 2 axes)."""
        from davinci_monet.plots.renderers.obs.obs_timeseries import ObsTimeSeriesPlotter

        plotter = ObsTimeSeriesPlotter()
        fig = plotter.plot(track_obs_data, "O3", show_altitude=True)

        # Main axes + secondary y-axis
        assert len(fig.axes) >= 2
        plt.close(fig)

    def test_multi_flight(self, multi_flight_obs_data):
        """Multi-flight data plots each flight in different color."""
        from davinci_monet.plots.renderers.obs.obs_timeseries import ObsTimeSeriesPlotter

        plotter = ObsTimeSeriesPlotter()
        fig = plotter.plot(multi_flight_obs_data, "O3")

        assert fig is not None
        plt.close(fig)

    def test_registered(self):
        """ObsTimeSeriesPlotter is registered as 'obs_timeseries'."""
        from davinci_monet.plots.renderers.obs.obs_timeseries import ObsTimeSeriesPlotter  # noqa: F401
        from davinci_monet.plots.registry import has_plotter

        assert has_plotter("obs_timeseries")


# =============================================================================
# Obs Histogram Plotter Tests
# =============================================================================


class TestObsHistogramPlotter:
    """Tests for the ObsHistogramPlotter (obs_histogram)."""

    def test_basic_plot(self, track_obs_data):
        """Basic histogram renders without error."""
        from davinci_monet.plots.renderers.obs.obs_histogram import ObsHistogramPlotter

        plotter = ObsHistogramPlotter()
        fig = plotter.plot(track_obs_data, "O3")

        assert fig is not None
        assert len(fig.axes) >= 1
        plt.close(fig)

    def test_shows_stats(self, track_obs_data):
        """Stats annotation box contains N= or Mean text."""
        from davinci_monet.plots.renderers.obs.obs_histogram import ObsHistogramPlotter

        plotter = ObsHistogramPlotter()
        fig = plotter.plot(track_obs_data, "O3", show_stats=True)

        ax = fig.axes[0]
        text_contents = [t.get_text() for t in ax.texts]
        stats_found = any("N=" in t or "Mean" in t for t in text_contents)
        assert stats_found, f"Expected stats text, got: {text_contents}"
        plt.close(fig)

    def test_custom_bins(self, track_obs_data):
        """Custom n_bins parameter is accepted."""
        from davinci_monet.plots.renderers.obs.obs_histogram import ObsHistogramPlotter

        plotter = ObsHistogramPlotter()
        fig = plotter.plot(track_obs_data, "O3", n_bins=15)

        assert fig is not None
        plt.close(fig)

    def test_registered(self):
        """ObsHistogramPlotter is registered as 'obs_histogram'."""
        from davinci_monet.plots.renderers.obs.obs_histogram import ObsHistogramPlotter  # noqa: F401
        from davinci_monet.plots.registry import has_plotter

        assert has_plotter("obs_histogram")


# =============================================================================
# Obs LMA Density Plotter Tests
# =============================================================================


class TestObsLMADensityPlotter:
    """Tests for the ObsLMADensityPlotter (obs_lma_density)."""

    def test_basic_plot(self, grid_lma_data):
        """Basic LMA density map renders without error."""
        from davinci_monet.plots.renderers.obs.obs_lma_density import ObsLMADensityPlotter

        plotter = ObsLMADensityPlotter()
        fig = plotter.plot(grid_lma_data, "flash_extent")

        assert fig is not None
        assert len(fig.axes) >= 1
        plt.close(fig)

    def test_registered_in_registry(self):
        """ObsLMADensityPlotter is registered as 'obs_lma_density'."""
        from davinci_monet.plots.renderers.obs.obs_lma_density import ObsLMADensityPlotter  # noqa: F401
        from davinci_monet.plots.registry import has_plotter

        assert has_plotter("obs_lma_density")

    def test_save_output(self, grid_lma_data, tmp_path):
        """LMA density map can be saved to disk."""
        from davinci_monet.plots.renderers.obs.obs_lma_density import ObsLMADensityPlotter

        plotter = ObsLMADensityPlotter()
        fig = plotter.plot(grid_lma_data, "flash_extent")

        output_path = tmp_path / "lma_density.png"
        saved_path = plotter.save(fig, output_path)

        assert saved_path.exists()
        assert saved_path.stat().st_size > 0
        plt.close(fig)
