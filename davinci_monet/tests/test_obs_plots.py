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
