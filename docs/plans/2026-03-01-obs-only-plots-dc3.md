# Obs-Only Plot Types + DC3 Data Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add observation-only plotting infrastructure (4 plot types) and integrate DC3 field campaign aircraft data for obs-first analysis workflows.

**Architecture:** Parallel `ObsPlotter` base class alongside existing `BasePlotter`, with 4 new renderers in `plots/renderers/obs/`. New `ObsPlottingStage` and `ObsStatisticsStage` in the pipeline. Auto-detection of obs-only mode when config has no `model` section. DC3 aircraft merge files downloaded via `earthaccess` and read by the existing `ICARTTReader`.

**Tech Stack:** matplotlib, cartopy, xarray, numpy, earthaccess (NASA data download), existing DAVINCI-MONET infrastructure

---

## Task 1: ObsPlotter Base Class

**Files:**
- Create: `davinci_monet/plots/obs_base.py`
- Create: `davinci_monet/tests/test_obs_plots.py`

**Step 1: Write the test file with fixtures and base class tests**

```python
"""Tests for observation-only plotting module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def track_obs_data() -> xr.Dataset:
    """Create aircraft track observation dataset (no model data)."""
    np.random.seed(42)
    n_times = 200

    time = pd.date_range("2012-05-29 14:00", periods=n_times, freq="10s")

    # Simulate aircraft track
    altitude = 1000 + 5000 * np.sin(np.linspace(0, 2 * np.pi, n_times))
    lats = np.linspace(35, 40, n_times)
    lons = np.linspace(-100, -95, n_times)

    o3 = 50 + 10 * np.exp(-altitude / 5000) + np.random.normal(0, 5, n_times)
    no = 0.5 + 2 * np.exp(-altitude / 3000) + np.random.normal(0, 0.2, n_times)
    no = np.clip(no, 0, None)

    ds = xr.Dataset(
        {
            "O3": (["time"], o3, {"units": "ppbv", "long_name": "Ozone"}),
            "NO": (["time"], no, {"units": "ppbv", "long_name": "Nitric Oxide"}),
        },
        coords={
            "time": time,
            "altitude": ("time", altitude, {"units": "m"}),
            "latitude": ("time", lats),
            "longitude": ("time", lons),
        },
        attrs={"geometry": "track"},
    )
    return ds


@pytest.fixture
def multi_flight_obs_data() -> xr.Dataset:
    """Create multi-flight observation dataset."""
    np.random.seed(42)
    n_per_flight = 120
    flights = []

    for day in range(3):
        base_time = pd.Timestamp(f"2012-05-{29 + day} 14:00:00")
        time = pd.date_range(base_time, periods=n_per_flight, freq="10s")

        altitude = 1000 + 5000 * np.sin(np.linspace(0, np.pi, n_per_flight))
        lats = np.linspace(35 + day, 40 + day, n_per_flight)
        lons = np.linspace(-100, -95, n_per_flight)

        o3 = 50 + 10 * np.exp(-altitude / 5000) + np.random.normal(0, 5, n_per_flight)

        flight_id = f"2012-05-{29 + day:02d}"

        ds = xr.Dataset(
            {
                "O3": (["time"], o3, {"units": "ppbv", "long_name": "Ozone"}),
            },
            coords={
                "time": time,
                "altitude": ("time", altitude, {"units": "m"}),
                "latitude": ("time", lats),
                "longitude": ("time", lons),
                "flight": ("time", [flight_id] * n_per_flight),
            },
        )
        flights.append(ds)

    return xr.concat(flights, dim="time")


class TestObsPlotterBase:
    """Test that ObsPlotter base class enforces the interface."""

    def test_cannot_instantiate_abstract(self):
        from davinci_monet.plots.obs_base import ObsPlotter
        with pytest.raises(TypeError):
            ObsPlotter()

    def test_subclass_must_implement_plot(self):
        from davinci_monet.plots.obs_base import ObsPlotter

        class IncompleteObs(ObsPlotter):
            name = "incomplete"

        with pytest.raises(TypeError):
            IncompleteObs()

    def test_subclass_with_plot_works(self, track_obs_data):
        from davinci_monet.plots.obs_base import ObsPlotter

        class DummyObsPlotter(ObsPlotter):
            name = "dummy"

            def plot(self, obs_data, variable, ax=None, **kwargs):
                fig, ax = plt.subplots()
                ax.plot(obs_data[variable].values)
                return fig

        plotter = DummyObsPlotter()
        fig = plotter.plot(track_obs_data, "O3")
        assert fig is not None
        plt.close(fig)

    def test_create_figure(self):
        from davinci_monet.plots.obs_base import ObsPlotter

        class DummyObsPlotter(ObsPlotter):
            name = "dummy"

            def plot(self, obs_data, variable, ax=None, **kwargs):
                fig, ax = self.create_figure()
                return fig

        plotter = DummyObsPlotter()
        fig, ax = plotter.create_figure()
        assert fig is not None
        plt.close(fig)

    def test_save(self, track_obs_data, tmp_path):
        from davinci_monet.plots.obs_base import ObsPlotter

        class DummyObsPlotter(ObsPlotter):
            name = "dummy"

            def plot(self, obs_data, variable, ax=None, **kwargs):
                fig, ax = self.create_figure()
                ax.plot(obs_data[variable].values)
                return fig

        plotter = DummyObsPlotter()
        fig = plotter.plot(track_obs_data, "O3")
        out = tmp_path / "test.png"
        plotter.save(fig, out)
        assert out.exists()
        plt.close(fig)
```

**Step 2: Run test to verify it fails**

Run: `pytest davinci_monet/tests/test_obs_plots.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'davinci_monet.plots.obs_base'`

**Step 3: Write ObsPlotter base class**

```python
"""Base class for observation-only plotters.

This module provides ObsPlotter, a parallel base class to BasePlotter
for plots that visualize observation data without model comparison.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt

from davinci_monet.plots.base import (
    FigureConfig,
    PlotConfig,
    TextConfig,
)
from davinci_monet.plots.style import OBS_COLOR

if TYPE_CHECKING:
    import matplotlib.axes
    import matplotlib.figure
    import xarray as xr


class ObsPlotter(ABC):
    """Abstract base class for observation-only plotters.

    Provides common functionality for figure management, styling,
    and saving. Subclasses implement specific obs-only plot types.

    Unlike BasePlotter which requires paired (obs + model) data,
    ObsPlotter works with raw observation datasets and a single variable.
    """

    name: str = "obs_base"
    default_figsize: tuple[float, float] = (8, 5)

    def __init__(self, config: PlotConfig | None = None) -> None:
        self.config = config or PlotConfig()
        if self.config.figure.figsize == (8, 5):
            self.config.figure.figsize = self.default_figsize

    @abstractmethod
    def plot(
        self,
        obs_data: xr.Dataset,
        variable: str,
        ax: matplotlib.axes.Axes | None = None,
        **kwargs: Any,
    ) -> matplotlib.figure.Figure:
        """Generate the plot.

        Parameters
        ----------
        obs_data
            Observation dataset with the variable to plot.
        variable
            Name of the variable to plot.
        ax
            Optional axes to plot on. If None, creates new figure.
        **kwargs
            Additional plot-specific options.

        Returns
        -------
        matplotlib.figure.Figure
            The generated figure.
        """
        ...

    def create_figure(
        self,
        config: FigureConfig | None = None,
        **kwargs: Any,
    ) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
        """Create a new figure and axes.

        Parameters
        ----------
        config
            Figure configuration. Uses self.config.figure if None.
        **kwargs
            Additional kwargs passed to plt.subplots().

        Returns
        -------
        tuple[Figure, Axes]
            The created figure and axes.
        """
        cfg = config or self.config.figure
        fig, ax = plt.subplots(
            figsize=cfg.figsize,
            dpi=cfg.dpi,
            facecolor=cfg.facecolor,
            constrained_layout=cfg.constrained_layout,
            **kwargs,
        )
        return fig, ax

    def save(
        self,
        fig: matplotlib.figure.Figure,
        filepath: str | Path,
        dpi: int | None = None,
        **kwargs: Any,
    ) -> Path:
        """Save figure to file.

        Parameters
        ----------
        fig
            Figure to save.
        filepath
            Output file path.
        dpi
            DPI override.
        **kwargs
            Additional kwargs passed to fig.savefig().

        Returns
        -------
        Path
            Path to saved file.
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            filepath,
            dpi=dpi or self.config.figure.dpi,
            bbox_inches="tight",
            facecolor=fig.get_facecolor(),
            **kwargs,
        )
        return filepath

    def close(self, fig: matplotlib.figure.Figure) -> None:
        """Close a figure to free memory."""
        plt.close(fig)
```

**Step 4: Run test to verify it passes**

Run: `pytest davinci_monet/tests/test_obs_plots.py -v`
Expected: All 5 tests PASS

**Step 5: Commit**

```bash
git add davinci_monet/plots/obs_base.py davinci_monet/tests/test_obs_plots.py
git commit -m "feat: add ObsPlotter base class for observation-only plots"
```

---

## Task 2: Flight Track Map Renderer

**Files:**
- Create: `davinci_monet/plots/renderers/obs/__init__.py`
- Create: `davinci_monet/plots/renderers/obs/flight_track_map.py`
- Modify: `davinci_monet/tests/test_obs_plots.py`

**Step 1: Write tests**

Add to `test_obs_plots.py`:

```python
class TestFlightTrackMapPlotter:
    """Test flight track map renderer."""

    def test_basic_plot(self, track_obs_data):
        from davinci_monet.plots.renderers.obs.flight_track_map import FlightTrackMapPlotter
        plotter = FlightTrackMapPlotter()
        fig = plotter.plot(track_obs_data, "O3")
        assert fig is not None
        plt.close(fig)

    def test_plot_has_colorbar(self, track_obs_data):
        from davinci_monet.plots.renderers.obs.flight_track_map import FlightTrackMapPlotter
        plotter = FlightTrackMapPlotter()
        fig = plotter.plot(track_obs_data, "O3")
        # Figure should have more than one axes (map + colorbar)
        assert len(fig.axes) >= 2
        plt.close(fig)

    def test_multi_flight(self, multi_flight_obs_data):
        from davinci_monet.plots.renderers.obs.flight_track_map import FlightTrackMapPlotter
        plotter = FlightTrackMapPlotter()
        fig = plotter.plot(multi_flight_obs_data, "O3")
        assert fig is not None
        plt.close(fig)

    def test_save_output(self, track_obs_data, tmp_path):
        from davinci_monet.plots.renderers.obs.flight_track_map import FlightTrackMapPlotter
        plotter = FlightTrackMapPlotter()
        fig = plotter.plot(track_obs_data, "O3")
        out = tmp_path / "track.png"
        plotter.save(fig, out)
        assert out.exists()
        plt.close(fig)

    def test_registered_in_registry(self):
        from davinci_monet.plots.registry import has_plotter
        # Import to trigger registration
        import davinci_monet.plots.renderers.obs.flight_track_map  # noqa: F401
        assert has_plotter("obs_flight_track")
```

**Step 2: Run test to verify it fails**

Run: `pytest davinci_monet/tests/test_obs_plots.py::TestFlightTrackMapPlotter -v`
Expected: FAIL

**Step 3: Create renderers/obs/ package and flight track map renderer**

`davinci_monet/plots/renderers/obs/__init__.py`:
```python
"""Observation-only plot renderers."""

from davinci_monet.plots.renderers.obs.flight_track_map import FlightTrackMapPlotter
from davinci_monet.plots.renderers.obs.vertical_profile import VerticalProfilePlotter
from davinci_monet.plots.renderers.obs.obs_timeseries import ObsTimeSeriesPlotter
from davinci_monet.plots.renderers.obs.obs_histogram import ObsHistogramPlotter

__all__ = [
    "FlightTrackMapPlotter",
    "VerticalProfilePlotter",
    "ObsTimeSeriesPlotter",
    "ObsHistogramPlotter",
]
```

Note: The `__init__.py` imports all 4 renderers for registration. Create it now but expect import errors until Tasks 3-4 are done. For now, only import FlightTrackMapPlotter and add the others as they're implemented.

`davinci_monet/plots/renderers/obs/flight_track_map.py`:
```python
"""Flight track map renderer for observation-only data.

Renders 2D Cartopy map with flight path colored by a variable value.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np

from davinci_monet.plots.obs_base import ObsPlotter
from davinci_monet.plots.registry import register_plotter
from davinci_monet.plots.style import OBS_COLOR, get_sequential_cmap

if TYPE_CHECKING:
    import matplotlib.axes
    import matplotlib.figure
    import xarray as xr


@register_plotter("obs_flight_track")
class FlightTrackMapPlotter(ObsPlotter):
    """Plot flight tracks on a map, colored by variable value.

    Creates a Cartopy map showing the aircraft flight path with
    points colored by the value of the selected variable.
    """

    name: str = "obs_flight_track"
    default_figsize: tuple[float, float] = (10, 8)

    def plot(
        self,
        obs_data: xr.Dataset,
        variable: str,
        ax: matplotlib.axes.Axes | None = None,
        title: str | None = None,
        cmap: str | None = None,
        vmin: float | None = None,
        vmax: float | None = None,
        marker_size: float = 10,
        **kwargs: Any,
    ) -> matplotlib.figure.Figure:
        """Generate flight track map.

        Parameters
        ----------
        obs_data
            Observation dataset with lat/lon coordinates.
        variable
            Variable to color the track by.
        ax
            Optional GeoAxes. If None, creates new figure with PlateCarree.
        title
            Plot title. Defaults to variable name.
        cmap
            Colormap name. Defaults to sequential NCAR cmap.
        vmin, vmax
            Color scale limits. Auto-computed if None.
        marker_size
            Size of scatter points.
        """
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature

        lats = obs_data["latitude"].values
        lons = obs_data["longitude"].values
        values = obs_data[variable].values

        # Filter NaN
        mask = np.isfinite(values) & np.isfinite(lats) & np.isfinite(lons)
        lats, lons, values = lats[mask], lons[mask], values[mask]

        if ax is None:
            fig = plt.figure(
                figsize=self.default_figsize,
                dpi=self.config.figure.dpi,
                constrained_layout=True,
            )
            ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
        else:
            fig = ax.get_figure()

        # Map features
        ax.add_feature(cfeature.LAND, facecolor="#f0f0f0", edgecolor="gray", linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linewidth=0.5)
        ax.add_feature(cfeature.STATES, linewidth=0.3, edgecolor="gray")
        ax.coastlines(linewidth=0.5)

        # Auto-zoom to data extent with padding
        lon_pad = max(1.0, (lons.max() - lons.min()) * 0.1)
        lat_pad = max(1.0, (lats.max() - lats.min()) * 0.1)
        ax.set_extent([
            lons.min() - lon_pad, lons.max() + lon_pad,
            lats.min() - lat_pad, lats.max() + lat_pad,
        ])

        # Scatter plot colored by variable
        use_cmap = cmap or "viridis"
        sc = ax.scatter(
            lons, lats, c=values,
            cmap=use_cmap, vmin=vmin, vmax=vmax,
            s=marker_size, transform=ccrs.PlateCarree(),
            zorder=5,
        )

        # Colorbar
        units = obs_data[variable].attrs.get("units", "")
        label = f"{variable} ({units})" if units else variable
        plt.colorbar(sc, ax=ax, label=label, shrink=0.8)

        ax.set_title(title or f"Flight Track: {variable}")
        ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5)

        return fig
```

**Step 4: Run test to verify it passes**

Run: `pytest davinci_monet/tests/test_obs_plots.py::TestFlightTrackMapPlotter -v`
Expected: All 5 tests PASS

**Step 5: Commit**

```bash
git add davinci_monet/plots/renderers/obs/ davinci_monet/tests/test_obs_plots.py
git commit -m "feat: add flight track map obs-only renderer"
```

---

## Task 3: Vertical Profile Renderer

**Files:**
- Create: `davinci_monet/plots/renderers/obs/vertical_profile.py`
- Modify: `davinci_monet/tests/test_obs_plots.py`
- Modify: `davinci_monet/plots/renderers/obs/__init__.py` (add import)

**Step 1: Write tests**

Add to `test_obs_plots.py`:

```python
class TestVerticalProfilePlotter:
    """Test vertical profile renderer."""

    def test_basic_scatter(self, track_obs_data):
        from davinci_monet.plots.renderers.obs.vertical_profile import VerticalProfilePlotter
        plotter = VerticalProfilePlotter()
        fig = plotter.plot(track_obs_data, "O3")
        assert fig is not None
        plt.close(fig)

    def test_binned_mode(self, track_obs_data):
        from davinci_monet.plots.renderers.obs.vertical_profile import VerticalProfilePlotter
        plotter = VerticalProfilePlotter()
        fig = plotter.plot(track_obs_data, "O3", mode="binned")
        assert fig is not None
        plt.close(fig)

    def test_multi_flight(self, multi_flight_obs_data):
        from davinci_monet.plots.renderers.obs.vertical_profile import VerticalProfilePlotter
        plotter = VerticalProfilePlotter()
        fig = plotter.plot(multi_flight_obs_data, "O3")
        assert fig is not None
        plt.close(fig)

    def test_registered(self):
        import davinci_monet.plots.renderers.obs.vertical_profile  # noqa: F401
        from davinci_monet.plots.registry import has_plotter
        assert has_plotter("obs_vertical_profile")
```

**Step 2: Run test to verify it fails**

Run: `pytest davinci_monet/tests/test_obs_plots.py::TestVerticalProfilePlotter -v`
Expected: FAIL

**Step 3: Implement vertical profile renderer**

`davinci_monet/plots/renderers/obs/vertical_profile.py`:
```python
"""Vertical profile renderer for observation-only data.

Renders altitude vs. concentration plots as scatter or binned-mean
with standard deviation envelope.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import matplotlib.pyplot as plt
import numpy as np

from davinci_monet.plots.obs_base import ObsPlotter
from davinci_monet.plots.registry import register_plotter
from davinci_monet.plots.style import OBS_COLOR

if TYPE_CHECKING:
    import matplotlib.axes
    import matplotlib.figure
    import xarray as xr


@register_plotter("obs_vertical_profile")
class VerticalProfilePlotter(ObsPlotter):
    """Plot vertical profiles of observation data.

    Shows altitude vs. concentration as scatter points or
    altitude-binned means with standard deviation shading.
    """

    name: str = "obs_vertical_profile"
    default_figsize: tuple[float, float] = (6, 8)

    def plot(
        self,
        obs_data: xr.Dataset,
        variable: str,
        ax: matplotlib.axes.Axes | None = None,
        mode: Literal["scatter", "binned"] = "scatter",
        n_bins: int = 20,
        alt_coord: str = "altitude",
        title: str | None = None,
        color: str | None = None,
        **kwargs: Any,
    ) -> matplotlib.figure.Figure:
        """Generate vertical profile plot.

        Parameters
        ----------
        obs_data
            Observation dataset with altitude coordinate.
        variable
            Variable to plot on x-axis.
        ax
            Optional axes. If None, creates new figure.
        mode
            "scatter" for raw points, "binned" for altitude-binned means.
        n_bins
            Number of altitude bins (for binned mode).
        alt_coord
            Name of altitude coordinate.
        title
            Plot title.
        color
            Point/line color. Defaults to OBS_COLOR.
        """
        values = obs_data[variable].values
        altitudes = obs_data[alt_coord].values

        # Filter NaN
        mask = np.isfinite(values) & np.isfinite(altitudes)
        values, altitudes = values[mask], altitudes[mask]

        if ax is None:
            fig, ax = self.create_figure()
        else:
            fig = ax.get_figure()

        use_color = color or OBS_COLOR

        if mode == "scatter":
            ax.scatter(values, altitudes, s=3, alpha=0.5, color=use_color, zorder=3)
        else:
            # Binned mean with std envelope
            bin_edges = np.linspace(altitudes.min(), altitudes.max(), n_bins + 1)
            bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
            bin_indices = np.digitize(altitudes, bin_edges) - 1
            bin_indices = np.clip(bin_indices, 0, n_bins - 1)

            bin_means = np.full(n_bins, np.nan)
            bin_stds = np.full(n_bins, np.nan)
            for i in range(n_bins):
                in_bin = values[bin_indices == i]
                if len(in_bin) >= 3:
                    bin_means[i] = np.nanmean(in_bin)
                    bin_stds[i] = np.nanstd(in_bin)

            valid = np.isfinite(bin_means)
            ax.plot(bin_means[valid], bin_centers[valid], color=use_color, linewidth=2, zorder=4)
            ax.fill_betweenx(
                bin_centers[valid],
                bin_means[valid] - bin_stds[valid],
                bin_means[valid] + bin_stds[valid],
                alpha=0.2, color=use_color, zorder=2,
            )

        units = obs_data[variable].attrs.get("units", "")
        xlabel = f"{variable} ({units})" if units else variable
        alt_units = obs_data[alt_coord].attrs.get("units", "m")
        ylabel = f"Altitude ({alt_units})"

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title or f"Vertical Profile: {variable}")
        ax.grid(True, alpha=0.3)

        return fig
```

**Step 4: Run test to verify it passes**

Run: `pytest davinci_monet/tests/test_obs_plots.py::TestVerticalProfilePlotter -v`
Expected: All 4 tests PASS

**Step 5: Commit**

```bash
git add davinci_monet/plots/renderers/obs/vertical_profile.py davinci_monet/tests/test_obs_plots.py davinci_monet/plots/renderers/obs/__init__.py
git commit -m "feat: add vertical profile obs-only renderer"
```

---

## Task 4: Obs Time Series Renderer

**Files:**
- Create: `davinci_monet/plots/renderers/obs/obs_timeseries.py`
- Modify: `davinci_monet/tests/test_obs_plots.py`
- Modify: `davinci_monet/plots/renderers/obs/__init__.py` (add import)

**Step 1: Write tests**

Add to `test_obs_plots.py`:

```python
class TestObsTimeSeriesPlotter:
    """Test obs time series renderer."""

    def test_basic_plot(self, track_obs_data):
        from davinci_monet.plots.renderers.obs.obs_timeseries import ObsTimeSeriesPlotter
        plotter = ObsTimeSeriesPlotter()
        fig = plotter.plot(track_obs_data, "O3")
        assert fig is not None
        plt.close(fig)

    def test_with_altitude_axis(self, track_obs_data):
        from davinci_monet.plots.renderers.obs.obs_timeseries import ObsTimeSeriesPlotter
        plotter = ObsTimeSeriesPlotter()
        fig = plotter.plot(track_obs_data, "O3", show_altitude=True)
        # Should have two y-axes (variable + altitude)
        assert len(fig.axes) >= 2
        plt.close(fig)

    def test_multi_flight(self, multi_flight_obs_data):
        from davinci_monet.plots.renderers.obs.obs_timeseries import ObsTimeSeriesPlotter
        plotter = ObsTimeSeriesPlotter()
        fig = plotter.plot(multi_flight_obs_data, "O3")
        assert fig is not None
        plt.close(fig)

    def test_registered(self):
        import davinci_monet.plots.renderers.obs.obs_timeseries  # noqa: F401
        from davinci_monet.plots.registry import has_plotter
        assert has_plotter("obs_timeseries")
```

**Step 2: Run test to verify it fails**

Run: `pytest davinci_monet/tests/test_obs_plots.py::TestObsTimeSeriesPlotter -v`
Expected: FAIL

**Step 3: Implement obs time series renderer**

`davinci_monet/plots/renderers/obs/obs_timeseries.py`:
```python
"""Time series renderer for observation-only data.

Renders variable value vs. time along flights, with optional
altitude overlay on secondary y-axis.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np

from davinci_monet.plots.obs_base import ObsPlotter
from davinci_monet.plots.registry import register_plotter
from davinci_monet.plots.style import OBS_COLOR, NCAR_PALETTE

if TYPE_CHECKING:
    import matplotlib.axes
    import matplotlib.figure
    import xarray as xr


@register_plotter("obs_timeseries")
class ObsTimeSeriesPlotter(ObsPlotter):
    """Plot observation variable as time series.

    Shows variable value vs. time. For multi-flight datasets,
    plots each flight in a different color. Optionally shows
    altitude on a secondary y-axis.
    """

    name: str = "obs_timeseries"
    default_figsize: tuple[float, float] = (10, 4)

    def plot(
        self,
        obs_data: xr.Dataset,
        variable: str,
        ax: matplotlib.axes.Axes | None = None,
        show_altitude: bool = False,
        alt_coord: str = "altitude",
        title: str | None = None,
        color: str | None = None,
        **kwargs: Any,
    ) -> matplotlib.figure.Figure:
        """Generate time series plot.

        Parameters
        ----------
        obs_data
            Observation dataset with time dimension.
        variable
            Variable to plot.
        ax
            Optional axes. If None, creates new figure.
        show_altitude
            If True, show altitude on secondary y-axis.
        alt_coord
            Name of altitude coordinate.
        title
            Plot title.
        color
            Line color (ignored for multi-flight, uses palette).
        """
        if ax is None:
            fig, ax = self.create_figure()
        else:
            fig = ax.get_figure()

        has_flights = "flight" in obs_data.coords

        if has_flights:
            flight_ids = np.unique(obs_data["flight"].values)
            colors = NCAR_PALETTE[:len(flight_ids)]
            for i, fid in enumerate(flight_ids):
                flight_mask = obs_data["flight"].values == fid
                flight_data = obs_data.isel(time=flight_mask)
                ax.plot(
                    flight_data["time"].values,
                    flight_data[variable].values,
                    color=colors[i % len(colors)],
                    linewidth=1,
                    label=fid,
                    alpha=0.8,
                )
            ax.legend(fontsize=8)
        else:
            ax.plot(
                obs_data["time"].values,
                obs_data[variable].values,
                color=color or OBS_COLOR,
                linewidth=1,
            )

        units = obs_data[variable].attrs.get("units", "")
        ylabel = f"{variable} ({units})" if units else variable
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Time")
        ax.set_title(title or f"Time Series: {variable}")
        ax.grid(True, alpha=0.3)

        # Optional altitude overlay
        if show_altitude and alt_coord in obs_data.coords:
            ax2 = ax.twinx()
            ax2.plot(
                obs_data["time"].values,
                obs_data[alt_coord].values,
                color="gray", alpha=0.3, linewidth=0.5,
            )
            alt_units = obs_data[alt_coord].attrs.get("units", "m")
            ax2.set_ylabel(f"Altitude ({alt_units})", color="gray")
            ax2.tick_params(axis="y", labelcolor="gray")

        fig.autofmt_xdate()

        return fig
```

**Step 4: Run test to verify it passes**

Run: `pytest davinci_monet/tests/test_obs_plots.py::TestObsTimeSeriesPlotter -v`
Expected: All 4 tests PASS

**Step 5: Commit**

```bash
git add davinci_monet/plots/renderers/obs/obs_timeseries.py davinci_monet/tests/test_obs_plots.py davinci_monet/plots/renderers/obs/__init__.py
git commit -m "feat: add obs time series renderer"
```

---

## Task 5: Obs Histogram Renderer

**Files:**
- Create: `davinci_monet/plots/renderers/obs/obs_histogram.py`
- Modify: `davinci_monet/tests/test_obs_plots.py`
- Modify: `davinci_monet/plots/renderers/obs/__init__.py` (add import)

**Step 1: Write tests**

Add to `test_obs_plots.py`:

```python
class TestObsHistogramPlotter:
    """Test obs histogram renderer."""

    def test_basic_plot(self, track_obs_data):
        from davinci_monet.plots.renderers.obs.obs_histogram import ObsHistogramPlotter
        plotter = ObsHistogramPlotter()
        fig = plotter.plot(track_obs_data, "O3")
        assert fig is not None
        plt.close(fig)

    def test_shows_stats(self, track_obs_data):
        from davinci_monet.plots.renderers.obs.obs_histogram import ObsHistogramPlotter
        plotter = ObsHistogramPlotter()
        fig = plotter.plot(track_obs_data, "O3", show_stats=True)
        # Check that text annotation exists on the axes
        ax = fig.axes[0]
        texts = [t.get_text() for t in ax.texts]
        stats_text = " ".join(texts)
        assert "N=" in stats_text or "Mean" in stats_text
        plt.close(fig)

    def test_custom_bins(self, track_obs_data):
        from davinci_monet.plots.renderers.obs.obs_histogram import ObsHistogramPlotter
        plotter = ObsHistogramPlotter()
        fig = plotter.plot(track_obs_data, "O3", n_bins=50)
        assert fig is not None
        plt.close(fig)

    def test_registered(self):
        import davinci_monet.plots.renderers.obs.obs_histogram  # noqa: F401
        from davinci_monet.plots.registry import has_plotter
        assert has_plotter("obs_histogram")
```

**Step 2: Run test to verify it fails**

Run: `pytest davinci_monet/tests/test_obs_plots.py::TestObsHistogramPlotter -v`
Expected: FAIL

**Step 3: Implement obs histogram renderer**

`davinci_monet/plots/renderers/obs/obs_histogram.py`:
```python
"""Histogram renderer for observation-only data.

Renders distribution histogram with summary statistics annotation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np

from davinci_monet.plots.obs_base import ObsPlotter
from davinci_monet.plots.registry import register_plotter
from davinci_monet.plots.style import OBS_COLOR

if TYPE_CHECKING:
    import matplotlib.axes
    import matplotlib.figure
    import xarray as xr


@register_plotter("obs_histogram")
class ObsHistogramPlotter(ObsPlotter):
    """Plot histogram of observation variable distribution.

    Shows distribution with optional summary statistics annotation
    (N, mean, median, std, percentiles).
    """

    name: str = "obs_histogram"
    default_figsize: tuple[float, float] = (7, 5)

    def plot(
        self,
        obs_data: xr.Dataset,
        variable: str,
        ax: matplotlib.axes.Axes | None = None,
        n_bins: int = 30,
        show_stats: bool = True,
        title: str | None = None,
        color: str | None = None,
        **kwargs: Any,
    ) -> matplotlib.figure.Figure:
        """Generate histogram plot.

        Parameters
        ----------
        obs_data
            Observation dataset.
        variable
            Variable to plot distribution of.
        ax
            Optional axes. If None, creates new figure.
        n_bins
            Number of histogram bins.
        show_stats
            If True, annotate with summary statistics.
        title
            Plot title.
        color
            Bar color. Defaults to OBS_COLOR.
        """
        values = obs_data[variable].values.flatten()
        values = values[np.isfinite(values)]

        if ax is None:
            fig, ax = self.create_figure()
        else:
            fig = ax.get_figure()

        use_color = color or OBS_COLOR

        ax.hist(values, bins=n_bins, color=use_color, alpha=0.7, edgecolor="white", linewidth=0.5)

        # Median line
        median = np.median(values)
        ax.axvline(median, color="red", linestyle="--", linewidth=1.5, label=f"Median: {median:.1f}")
        ax.legend(fontsize=8)

        units = obs_data[variable].attrs.get("units", "")
        xlabel = f"{variable} ({units})" if units else variable
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Count")
        ax.set_title(title or f"Distribution: {variable}")
        ax.grid(True, alpha=0.3, axis="y")

        if show_stats:
            stats_text = (
                f"N={len(values)}\n"
                f"Mean={np.mean(values):.1f}\n"
                f"Median={median:.1f}\n"
                f"Std={np.std(values):.1f}\n"
                f"P10={np.percentile(values, 10):.1f}\n"
                f"P90={np.percentile(values, 90):.1f}"
            )
            ax.text(
                0.97, 0.95, stats_text,
                transform=ax.transAxes,
                verticalalignment="top",
                horizontalalignment="right",
                fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
            )

        return fig
```

**Step 4: Run test to verify it passes**

Run: `pytest davinci_monet/tests/test_obs_plots.py::TestObsHistogramPlotter -v`
Expected: All 4 tests PASS

**Step 5: Commit**

```bash
git add davinci_monet/plots/renderers/obs/obs_histogram.py davinci_monet/tests/test_obs_plots.py davinci_monet/plots/renderers/obs/__init__.py
git commit -m "feat: add obs histogram renderer"
```

---

## Task 6: Wire Obs Renderers into Plots Module

**Files:**
- Modify: `davinci_monet/plots/__init__.py` (lines 92-130, 132-215)
- Modify: `davinci_monet/plots/renderers/obs/__init__.py`

**Step 1: Write test**

Add to `test_obs_plots.py`:

```python
class TestObsPlotRegistration:
    """Test that all obs plotters are importable and registered."""

    def test_all_obs_plotters_registered(self):
        from davinci_monet.plots import list_plotters
        # Trigger registration
        import davinci_monet.plots.renderers.obs  # noqa: F401
        available = list_plotters()
        for name in ["obs_flight_track", "obs_vertical_profile", "obs_timeseries", "obs_histogram"]:
            assert name in available, f"{name} not registered"

    def test_get_obs_plotter(self):
        from davinci_monet.plots import get_plotter
        import davinci_monet.plots.renderers.obs  # noqa: F401
        plotter = get_plotter("obs_flight_track")
        assert plotter is not None

    def test_obs_base_importable(self):
        from davinci_monet.plots import ObsPlotter
        assert ObsPlotter is not None
```

**Step 2: Run test to verify it fails**

Run: `pytest davinci_monet/tests/test_obs_plots.py::TestObsPlotRegistration -v`
Expected: FAIL (ObsPlotter not exported from plots module)

**Step 3: Update plots __init__.py**

Add after line 52 in `davinci_monet/plots/__init__.py`:

```python
# Observation-only base class
from davinci_monet.plots.obs_base import ObsPlotter
```

Add after line 130 (the spatial imports):

```python
# Observation-only renderers
from davinci_monet.plots.renderers.obs import (
    FlightTrackMapPlotter,
    VerticalProfilePlotter,
    ObsTimeSeriesPlotter,
    ObsHistogramPlotter,
)
```

Add to `__all__` list:

```python
    # Obs-only base
    "ObsPlotter",
    # Obs-only renderers
    "FlightTrackMapPlotter",
    "VerticalProfilePlotter",
    "ObsTimeSeriesPlotter",
    "ObsHistogramPlotter",
```

**Step 4: Run test to verify it passes**

Run: `pytest davinci_monet/tests/test_obs_plots.py::TestObsPlotRegistration -v`
Expected: All 3 tests PASS

**Step 5: Commit**

```bash
git add davinci_monet/plots/__init__.py davinci_monet/plots/renderers/obs/__init__.py
git commit -m "feat: wire obs-only renderers into plots module"
```

---

## Task 7: ObsPlottingStage + ObsStatisticsStage

**Files:**
- Modify: `davinci_monet/pipeline/stages.py` (add new stages before `create_standard_pipeline` at line 1545)
- Create: `davinci_monet/tests/test_obs_pipeline.py`

**Step 1: Write tests**

`davinci_monet/tests/test_obs_pipeline.py`:

```python
"""Tests for observation-only pipeline stages."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import matplotlib
matplotlib.use("Agg")

from davinci_monet.pipeline.stages import (
    PipelineContext,
    StageStatus,
)


@pytest.fixture
def obs_context() -> PipelineContext:
    """Create pipeline context with observation data only (no models)."""
    np.random.seed(42)
    n_times = 200

    time = pd.date_range("2012-05-29 14:00", periods=n_times, freq="10s")
    altitude = 1000 + 5000 * np.sin(np.linspace(0, 2 * np.pi, n_times))
    lats = np.linspace(35, 40, n_times)
    lons = np.linspace(-100, -95, n_times)
    o3 = 50 + 10 * np.exp(-altitude / 5000) + np.random.normal(0, 5, n_times)
    flight_ids = ["2012-05-29"] * n_times

    ds = xr.Dataset(
        {
            "O3": (["time"], o3, {"units": "ppbv", "long_name": "Ozone"}),
        },
        coords={
            "time": time,
            "altitude": ("time", altitude, {"units": "m"}),
            "latitude": ("time", lats),
            "longitude": ("time", lons),
            "flight": ("time", flight_ids),
        },
        attrs={"geometry": "track"},
    )

    config = {
        "obs": {
            "dc8": {
                "obs_type": "icartt",
                "variables": {"O3": {}},
            },
        },
        "plots": {
            "track": {
                "type": "obs_flight_track",
                "obs": "dc8",
                "variable": "O3",
                "title": "Test Track",
            },
            "profile": {
                "type": "obs_vertical_profile",
                "obs": "dc8",
                "variable": "O3",
            },
        },
        "analysis": {
            "output_dir": "/tmp/test_obs_pipeline",
        },
    }

    context = PipelineContext(config=config)

    # Simulate loaded observation (normally done by LoadObservationsStage)
    from davinci_monet.observations.base import create_observation_data
    from davinci_monet.core.protocols import DataGeometry
    obs_data = create_observation_data(
        label="dc8",
        obs_type="icartt",
        data=ds,
        variables={"O3": {}},
    )
    obs_data.geometry = DataGeometry.TRACK
    context.observations["dc8"] = obs_data

    return context


class TestObsPlottingStage:
    """Test observation-only plotting stage."""

    def test_validate_with_obs(self, obs_context):
        from davinci_monet.pipeline.stages import ObsPlottingStage
        stage = ObsPlottingStage()
        assert stage.validate(obs_context) is True

    def test_validate_without_obs(self):
        from davinci_monet.pipeline.stages import ObsPlottingStage
        stage = ObsPlottingStage()
        context = PipelineContext(config={})
        assert stage.validate(context) is False

    def test_execute_creates_plots(self, obs_context, tmp_path):
        from davinci_monet.pipeline.stages import ObsPlottingStage
        obs_context.config["analysis"]["output_dir"] = str(tmp_path)
        stage = ObsPlottingStage()
        result = stage.execute(obs_context)
        assert result.status == StageStatus.COMPLETED

    def test_stage_name(self):
        from davinci_monet.pipeline.stages import ObsPlottingStage
        stage = ObsPlottingStage()
        assert stage.name == "obs_plotting"


class TestObsStatisticsStage:
    """Test observation-only statistics stage."""

    def test_validate_with_obs(self, obs_context):
        from davinci_monet.pipeline.stages import ObsStatisticsStage
        stage = ObsStatisticsStage()
        assert stage.validate(obs_context) is True

    def test_execute_computes_stats(self, obs_context):
        from davinci_monet.pipeline.stages import ObsStatisticsStage
        stage = ObsStatisticsStage()
        result = stage.execute(obs_context)
        assert result.status == StageStatus.COMPLETED
        # Should have stats in result data
        assert result.data is not None
        assert "dc8" in result.data
        dc8_stats = result.data["dc8"]
        assert "O3" in dc8_stats
        o3_stats = dc8_stats["O3"]
        assert "N" in o3_stats
        assert "mean" in o3_stats
        assert "median" in o3_stats

    def test_stage_name(self):
        from davinci_monet.pipeline.stages import ObsStatisticsStage
        stage = ObsStatisticsStage()
        assert stage.name == "obs_statistics"
```

**Step 2: Run test to verify it fails**

Run: `pytest davinci_monet/tests/test_obs_pipeline.py -v`
Expected: FAIL — `ImportError: cannot import name 'ObsPlottingStage'`

**Step 3: Implement ObsPlottingStage and ObsStatisticsStage**

Add before `create_standard_pipeline()` (line 1545) in `davinci_monet/pipeline/stages.py`:

```python
class ObsPlottingStage(BaseStage):
    """Pipeline stage for observation-only plots.

    Generates plots from raw observation data without requiring
    paired model-observation data.
    """

    name: str = "obs_plotting"

    def validate(self, context: PipelineContext) -> bool:
        """Validate that observation data exists."""
        return bool(context.observations)

    def execute(self, context: PipelineContext) -> StageResult:
        """Execute observation-only plotting."""
        import matplotlib.pyplot as plt
        from davinci_monet.plots.registry import get_plotter

        start = time.time()
        plots_config = context.config.get("plots", {})
        output_dir = Path(context.config.get("analysis", {}).get("output_dir", "."))
        output_dir.mkdir(parents=True, exist_ok=True)

        plot_count = 0
        errors: list[str] = []

        for plot_name, plot_spec in plots_config.items():
            plot_type = plot_spec.get("type", "")
            if not plot_type.startswith("obs_"):
                continue

            obs_label = plot_spec.get("obs", "")
            variable = plot_spec.get("variable", "")

            if obs_label not in context.observations:
                errors.append(f"Observation '{obs_label}' not found for plot '{plot_name}'")
                continue

            obs_data = context.observations[obs_label]
            ds = obs_data.data if hasattr(obs_data, "data") else obs_data

            if variable not in ds.data_vars:
                errors.append(f"Variable '{variable}' not found in '{obs_label}' for plot '{plot_name}'")
                continue

            try:
                plotter = get_plotter(plot_type)
                plot_kwargs = {
                    k: v for k, v in plot_spec.items()
                    if k not in ("type", "obs", "variable")
                }
                fig = plotter.plot(ds, variable, **plot_kwargs)

                # Save
                out_path = output_dir / f"{plot_name}.png"
                plotter.save(fig, out_path)
                plt.close(fig)
                plot_count += 1
                logger.info(f"Saved obs plot: {out_path}")
            except Exception as e:
                errors.append(f"Plot '{plot_name}' failed: {e}")
                logger.warning(f"Obs plot '{plot_name}' failed: {e}")

        message = f"Generated {plot_count} obs-only plots"
        if errors:
            message += f" ({len(errors)} errors)"

        return StageResult(
            stage_name=self.name,
            status=StageStatus.COMPLETED if plot_count > 0 or not plots_config else StageStatus.SKIPPED,
            data={"plot_count": plot_count, "errors": errors},
            duration=time.time() - start,
        )


class ObsStatisticsStage(BaseStage):
    """Pipeline stage for observation-only descriptive statistics.

    Computes summary statistics (N, mean, median, std, percentiles)
    for each observation variable without requiring model data.
    """

    name: str = "obs_statistics"

    def validate(self, context: PipelineContext) -> bool:
        """Validate that observation data exists."""
        return bool(context.observations)

    def execute(self, context: PipelineContext) -> StageResult:
        """Compute descriptive statistics for observation variables."""
        import numpy as np

        start = time.time()
        all_stats: dict[str, dict[str, dict[str, float]]] = {}

        for obs_label, obs_data in context.observations.items():
            ds = obs_data.data if hasattr(obs_data, "data") else obs_data
            obs_stats: dict[str, dict[str, float]] = {}

            for var_name in ds.data_vars:
                values = ds[var_name].values.flatten()
                values = values[np.isfinite(values)]

                if len(values) < 1:
                    continue

                obs_stats[var_name] = {
                    "N": len(values),
                    "mean": float(np.mean(values)),
                    "median": float(np.median(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "p10": float(np.percentile(values, 10)),
                    "p25": float(np.percentile(values, 25)),
                    "p75": float(np.percentile(values, 75)),
                    "p90": float(np.percentile(values, 90)),
                }

            all_stats[obs_label] = obs_stats

        return StageResult(
            stage_name=self.name,
            status=StageStatus.COMPLETED,
            data=all_stats,
            duration=time.time() - start,
        )
```

Also add the necessary import at the top of `stages.py` (near line 25):
```python
from pathlib import Path
```
(Check if `Path` is already imported — if so, skip.)

**Step 4: Run test to verify it passes**

Run: `pytest davinci_monet/tests/test_obs_pipeline.py -v`
Expected: All 7 tests PASS

**Step 5: Commit**

```bash
git add davinci_monet/pipeline/stages.py davinci_monet/tests/test_obs_pipeline.py
git commit -m "feat: add ObsPlottingStage and ObsStatisticsStage"
```

---

## Task 8: Obs-Only Pipeline Auto-Detection

**Files:**
- Modify: `davinci_monet/pipeline/stages.py:1545-1560` (`create_standard_pipeline`)
- Modify: `davinci_monet/pipeline/runner.py:1700-1710` (`run_from_config`)
- Modify: `davinci_monet/tests/test_obs_pipeline.py`

**Step 1: Write tests**

Add to `test_obs_pipeline.py`:

```python
class TestObsOnlyPipelineDetection:
    """Test auto-detection of obs-only pipeline mode."""

    def test_create_obs_pipeline(self):
        from davinci_monet.pipeline.stages import create_obs_pipeline
        stages = create_obs_pipeline()
        stage_names = [s.name for s in stages]
        assert "load_observations" in stage_names
        assert "obs_plotting" in stage_names
        assert "obs_statistics" in stage_names
        # Should NOT have model/pairing stages
        assert "load_models" not in stage_names
        assert "pairing" not in stage_names

    def test_run_from_config_detects_obs_only(self, obs_context, tmp_path):
        """Test that PipelineRunner auto-selects obs pipeline when no model section."""
        from davinci_monet.pipeline.runner import PipelineRunner
        config = obs_context.config.copy()
        config["analysis"]["output_dir"] = str(tmp_path)
        runner = PipelineRunner()
        result = runner.run_from_config(config)
        assert result.success
```

**Step 2: Run test to verify it fails**

Run: `pytest davinci_monet/tests/test_obs_pipeline.py::TestObsOnlyPipelineDetection -v`
Expected: FAIL — `ImportError: cannot import name 'create_obs_pipeline'`

**Step 3: Add create_obs_pipeline and update runner**

Add to `davinci_monet/pipeline/stages.py` after `create_standard_pipeline()`:

```python
def create_obs_pipeline() -> list[BaseStage]:
    """Create an observation-only pipeline (no model/pairing stages).

    Used when configuration has no model section.

    Returns
    -------
    list[BaseStage]
        List of stages for obs-only analysis.
    """
    return [
        LoadObservationsStage(),
        ObsStatisticsStage(),
        ObsPlottingStage(),
        SaveResultsStage(),
    ]
```

In `davinci_monet/pipeline/runner.py`, update `run_from_config` (around line 1700-1713):

Replace the section:
```python
        model_config = config.get("model") or {}
        obs_config = config.get("obs") or {}

        if not model_config and not obs_config:
            raise ConfigurationError(
                "Configuration is empty or incomplete. "
                "At least one model or observation must be defined."
            )

        context = PipelineContext(config=config)
```

With:
```python
        model_config = config.get("model") or {}
        obs_config = config.get("obs") or {}

        if not model_config and not obs_config:
            raise ConfigurationError(
                "Configuration is empty or incomplete. "
                "At least one model or observation must be defined."
            )

        # Auto-detect obs-only mode
        if not model_config and obs_config:
            from davinci_monet.pipeline.stages import create_obs_pipeline
            self._stages = create_obs_pipeline()

        context = PipelineContext(config=config)
```

Also update the import at top of `runner.py` to include `create_obs_pipeline`:

```python
from davinci_monet.pipeline.stages import (
    BaseStage,
    PipelineContext,
    Stage,
    StageResult,
    StageStatus,
    create_standard_pipeline,
)
```

Note: We do NOT add `create_obs_pipeline` to the top-level import — it's imported lazily inside `run_from_config` to avoid circular imports if any arise.

**Step 4: Run test to verify it passes**

Run: `pytest davinci_monet/tests/test_obs_pipeline.py -v`
Expected: All tests PASS

**Step 5: Also run existing tests to ensure no regressions**

Run: `pytest davinci_monet/tests/test_plots.py davinci_monet/tests/test_pipeline.py -v`
Expected: All existing tests still PASS

**Step 6: Commit**

```bash
git add davinci_monet/pipeline/stages.py davinci_monet/pipeline/runner.py davinci_monet/tests/test_obs_pipeline.py
git commit -m "feat: add obs-only pipeline auto-detection"
```

---

## Task 9: DC3 Download Script

**Files:**
- Create: `analyses/dc3/scripts/download_dc3_aircraft.py`
- Create: `analyses/dc3/configs/` (empty directory for now)
- Create: `analyses/dc3/README.md`

**Step 1: Create directory structure**

```bash
mkdir -p analyses/dc3/{scripts,configs,data,output,logs}
```

**Step 2: Write download script**

`analyses/dc3/scripts/download_dc3_aircraft.py`:

```python
#!/usr/bin/env python
"""Download DC3 aircraft merge data from NASA ASDC.

Uses earthaccess library for Earthdata Login authentication.
Downloads ICARTT merge files to ~/Data/DC3/aircraft/merge/.

Usage:
    python download_dc3_aircraft.py [--data-dir ~/Data/DC3]

Requirements:
    pip install earthaccess

Authentication:
    Either set up ~/.netrc with Earthdata credentials, or
    earthaccess will prompt interactively on first use.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def download_dc3_merge(data_dir: Path) -> None:
    """Download DC3 merge data files."""
    try:
        import earthaccess
    except ImportError:
        print("ERROR: earthaccess not installed. Run: pip install earthaccess")
        sys.exit(1)

    # Authenticate
    print("Authenticating with Earthdata Login...")
    earthaccess.login()

    # Create output directory
    merge_dir = data_dir / "aircraft" / "merge"
    merge_dir.mkdir(parents=True, exist_ok=True)

    # Search for DC3 merge data
    print("Searching for DC3_Merge_Data_1...")
    results = earthaccess.search_data(
        short_name="DC3_Merge_Data_1",
        temporal=("2012-05-01", "2012-07-01"),
    )

    if not results:
        print("No results found. Trying alternative search...")
        results = earthaccess.search_data(
            keyword="DC3 merge",
            temporal=("2012-05-01", "2012-07-01"),
        )

    if not results:
        print("No DC3 merge data found via earthaccess.")
        print("\nManual download alternative:")
        print("  1. Go to: https://asdc.larc.nasa.gov/project/DC3/DC3_Merge_Data_1")
        print("  2. Click 'Get Dataset'")
        print(f"  3. Download ICARTT files to: {merge_dir}")
        sys.exit(1)

    print(f"Found {len(results)} granules. Downloading to {merge_dir}...")
    downloaded = earthaccess.download(results, str(merge_dir))
    print(f"Downloaded {len(downloaded)} files to {merge_dir}")

    # List downloaded files
    ict_files = sorted(merge_dir.glob("*.ict"))
    print(f"\nICARTT files available: {len(ict_files)}")
    for f in ict_files[:10]:
        print(f"  {f.name}")
    if len(ict_files) > 10:
        print(f"  ... and {len(ict_files) - 10} more")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download DC3 aircraft merge data")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path.home() / "Data" / "DC3",
        help="Root data directory (default: ~/Data/DC3)",
    )
    args = parser.parse_args()

    download_dc3_merge(args.data_dir)


if __name__ == "__main__":
    main()
```

**Step 3: Write README**

`analyses/dc3/README.md`:

```markdown
# DC3 Analysis

Deep Convective Clouds and Chemistry (DC3) field campaign analysis.

## Quick Start

### 1. Download data

```bash
# Requires earthaccess: pip install earthaccess
# Requires Earthdata Login account: https://urs.earthdata.nasa.gov/
python scripts/download_dc3_aircraft.py
```

### 2. Run obs-only analysis

```bash
export DC3_DATA=~/Data/DC3
export DC3_ANALYSIS=$(pwd)
davinci-monet run configs/dc3-obs-dc8.yaml
```

## Data

Aircraft merge files (ICARTT format) from NASA ASDC:
- DC-8: `dc3-mrg10-dc8_merge_*.ict`
- GV: `dc3-mrg10-gv_merge_*.ict`
- Falcon: `dc3-mrg10-falcon_merge_*.ict`

## Campaign Period

- Field phase: 15 May - 30 June 2012
- Intensive operations: 18 May - 22 June 2012
- Benchmark case: 29-30 May 2012 Oklahoma Supercell
```

**Step 4: Commit**

```bash
git add analyses/dc3/
git commit -m "feat: add DC3 analysis directory and download script"
```

---

## Task 10: DC3 YAML Configs

**Files:**
- Create: `analyses/dc3/configs/dc3-obs-dc8.yaml`
- Create: `analyses/dc3/configs/dc3-obs-gv.yaml`
- Create: `analyses/dc3/configs/dc3-obs-all-aircraft.yaml`
- Create: `analyses/dc3/scripts/run_obs_analysis.py`

**Step 1: Create DC-8 obs-only config**

`analyses/dc3/configs/dc3-obs-dc8.yaml`:

```yaml
analysis:
  start_time: "2012-05-18"
  end_time: "2012-06-22"
  output_dir: ${DC3_ANALYSIS}/output/dc8
  log_dir: ${DC3_ANALYSIS}/logs
  style:
    theme: ncar

obs:
  dc8:
    obs_type: icartt
    filename: ${DC3_DATA}/aircraft/merge/dc3-mrg10-dc8_merge_*.ict
    variables:
      NO:  { source_name: NO_ESRL }
      NO2: { source_name: NO2_TDLIF }
      O3:  { source_name: O3_ESRL }
      CO:  { source_name: CO_DACOM }
      NOy: { source_name: NOy_ESRL }

plots:
  dc8_track_o3:
    type: obs_flight_track
    obs: dc8
    variable: O3
    title: "DC-8 Flight Tracks: O3 (ppbv)"

  dc8_track_no:
    type: obs_flight_track
    obs: dc8
    variable: NO
    title: "DC-8 Flight Tracks: NO (ppbv)"

  dc8_profile_o3:
    type: obs_vertical_profile
    obs: dc8
    variable: O3
    title: "DC-8 O3 Vertical Profile"

  dc8_profile_no:
    type: obs_vertical_profile
    obs: dc8
    variable: NO
    mode: binned
    title: "DC-8 NO Vertical Profile (binned)"

  dc8_ts_o3:
    type: obs_timeseries
    obs: dc8
    variable: O3
    show_altitude: true
    title: "DC-8 O3 Time Series"

  dc8_hist_o3:
    type: obs_histogram
    obs: dc8
    variable: O3
    title: "DC-8 O3 Distribution"

  dc8_hist_co:
    type: obs_histogram
    obs: dc8
    variable: CO
    title: "DC-8 CO Distribution"

stats:
  metrics: [N, mean, median, std, min, max, p10, p25, p75, p90]
```

**Step 2: Create GV config**

`analyses/dc3/configs/dc3-obs-gv.yaml`:

```yaml
analysis:
  start_time: "2012-05-18"
  end_time: "2012-06-30"
  output_dir: ${DC3_ANALYSIS}/output/gv
  log_dir: ${DC3_ANALYSIS}/logs
  style:
    theme: ncar

obs:
  gv:
    obs_type: icartt
    filename: ${DC3_DATA}/aircraft/merge/dc3-mrg10-gv_merge_*.ict
    variables:
      NO:  { source_name: NO_NOxyO3 }
      NO2: { source_name: NO2_NOxyO3 }
      O3:  { source_name: O3_NOxyO3 }
      CO:  { source_name: CO_ACOMCO }
      NOy: { source_name: NOy_NOxyO3 }

plots:
  gv_track_o3:
    type: obs_flight_track
    obs: gv
    variable: O3
    title: "GV Flight Tracks: O3 (ppbv)"

  gv_profile_o3:
    type: obs_vertical_profile
    obs: gv
    variable: O3
    title: "GV O3 Vertical Profile"

  gv_profile_no:
    type: obs_vertical_profile
    obs: gv
    variable: NO
    mode: binned
    title: "GV NO Vertical Profile (binned)"

  gv_ts_o3:
    type: obs_timeseries
    obs: gv
    variable: O3
    show_altitude: true
    title: "GV O3 Time Series"

  gv_hist_o3:
    type: obs_histogram
    obs: gv
    variable: O3
    title: "GV O3 Distribution"

stats:
  metrics: [N, mean, median, std, min, max, p10, p25, p75, p90]
```

**Step 3: Create combined config**

`analyses/dc3/configs/dc3-obs-all-aircraft.yaml`:

```yaml
analysis:
  start_time: "2012-05-18"
  end_time: "2012-06-30"
  output_dir: ${DC3_ANALYSIS}/output/all_aircraft
  log_dir: ${DC3_ANALYSIS}/logs
  style:
    theme: ncar

obs:
  dc8:
    obs_type: icartt
    filename: ${DC3_DATA}/aircraft/merge/dc3-mrg10-dc8_merge_*.ict
    variables:
      NO:  { source_name: NO_ESRL }
      O3:  { source_name: O3_ESRL }
      CO:  { source_name: CO_DACOM }

  gv:
    obs_type: icartt
    filename: ${DC3_DATA}/aircraft/merge/dc3-mrg10-gv_merge_*.ict
    variables:
      NO:  { source_name: NO_NOxyO3 }
      O3:  { source_name: O3_NOxyO3 }
      CO:  { source_name: CO_ACOMCO }

plots:
  dc8_track_o3:
    type: obs_flight_track
    obs: dc8
    variable: O3
    title: "DC-8 O3 Flight Tracks"

  gv_track_o3:
    type: obs_flight_track
    obs: gv
    variable: O3
    title: "GV O3 Flight Tracks"

  dc8_profile_o3:
    type: obs_vertical_profile
    obs: dc8
    variable: O3
    mode: binned
    title: "DC-8 O3 Vertical Profile"

  gv_profile_o3:
    type: obs_vertical_profile
    obs: gv
    variable: O3
    mode: binned
    title: "GV O3 Vertical Profile"

stats:
  metrics: [N, mean, median, std, min, max, p10, p25, p75, p90]
```

**Step 4: Create run script**

`analyses/dc3/scripts/run_obs_analysis.py`:

```python
#!/usr/bin/env python
"""Run DC3 observation-only analysis pipeline.

Usage:
    python run_obs_analysis.py [config_name]

Examples:
    python run_obs_analysis.py dc3-obs-dc8
    python run_obs_analysis.py dc3-obs-gv
    python run_obs_analysis.py dc3-obs-all-aircraft
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Set analysis directory
ANALYSIS_DIR = Path(__file__).resolve().parent.parent
os.environ.setdefault("DC3_ANALYSIS", str(ANALYSIS_DIR))
os.environ.setdefault("DC3_DATA", str(Path.home() / "Data" / "DC3"))


def main() -> None:
    config_name = sys.argv[1] if len(sys.argv) > 1 else "dc3-obs-dc8"
    config_path = ANALYSIS_DIR / "configs" / f"{config_name}.yaml"

    if not config_path.exists():
        print(f"Config not found: {config_path}")
        available = sorted(ANALYSIS_DIR.glob("configs/*.yaml"))
        print("Available configs:")
        for c in available:
            print(f"  {c.stem}")
        sys.exit(1)

    print(f"Running DC3 analysis: {config_path.name}")
    print(f"  DC3_DATA={os.environ['DC3_DATA']}")
    print(f"  DC3_ANALYSIS={os.environ['DC3_ANALYSIS']}")

    from davinci_monet.pipeline.runner import run_analysis
    result = run_analysis(str(config_path))

    if result.success:
        print(f"\nAnalysis complete in {result.total_duration_seconds:.1f}s")
    else:
        print(f"\nAnalysis failed: {result.error}")
        sys.exit(1)


if __name__ == "__main__":
    main()
```

**Step 5: Commit**

```bash
git add analyses/dc3/configs/ analyses/dc3/scripts/run_obs_analysis.py
git commit -m "feat: add DC3 obs-only YAML configs and run script"
```

---

## Task 11: Run Full Test Suite + Integration Smoke Test

**Step 1: Run complete test suite**

Run: `pytest davinci_monet/tests/ -v --tb=short`
Expected: All tests PASS (existing + new)

**Step 2: Verify obs pipeline with synthetic data**

Run a quick Python smoke test:

```python
# Quick integration test (run in Python REPL or as a script)
import numpy as np
import pandas as pd
import xarray as xr
from davinci_monet.pipeline.runner import PipelineRunner
from davinci_monet.pipeline.stages import PipelineContext, create_obs_pipeline
from davinci_monet.observations.base import create_observation_data
from davinci_monet.core.protocols import DataGeometry
import tempfile, os

# Create synthetic aircraft data
np.random.seed(42)
n = 300
ds = xr.Dataset(
    {"O3": (["time"], 50 + np.random.normal(0, 10, n), {"units": "ppbv"})},
    coords={
        "time": pd.date_range("2012-05-29", periods=n, freq="10s"),
        "altitude": ("time", 1000 + 5000 * np.sin(np.linspace(0, 2*np.pi, n))),
        "latitude": ("time", np.linspace(35, 40, n)),
        "longitude": ("time", np.linspace(-100, -95, n)),
        "flight": ("time", ["2012-05-29"] * n),
    },
)

with tempfile.TemporaryDirectory() as tmpdir:
    config = {
        "obs": {"test": {"obs_type": "icartt", "variables": {"O3": {}}}},
        "plots": {
            "track": {"type": "obs_flight_track", "obs": "test", "variable": "O3"},
            "profile": {"type": "obs_vertical_profile", "obs": "test", "variable": "O3"},
            "ts": {"type": "obs_timeseries", "obs": "test", "variable": "O3"},
            "hist": {"type": "obs_histogram", "obs": "test", "variable": "O3"},
        },
        "analysis": {"output_dir": tmpdir},
    }
    context = PipelineContext(config=config)
    obs = create_observation_data(label="test", obs_type="icartt", data=ds, variables={"O3": {}})
    obs.geometry = DataGeometry.TRACK
    context.observations["test"] = obs

    runner = PipelineRunner(stages=create_obs_pipeline())
    result = runner.run(context)
    print(f"Success: {result.success}")
    print(f"Plots created: {len(list(Path(tmpdir).glob('*.png')))}")
    for f in sorted(Path(tmpdir).glob("*.png")):
        print(f"  {f.name}")
```

Expected: 4 PNG files created (track, profile, ts, hist)

**Step 3: If all passes, commit final state**

```bash
git add -A
git commit -m "feat: complete obs-only plots + DC3 data integration"
```

---

## Task 12: Download DC3 Data

**After all code is working**, download the actual DC3 data:

**Step 1: Install earthaccess if needed**

Run: `pip install earthaccess`

**Step 2: Run download script**

Run: `cd analyses/dc3 && python scripts/download_dc3_aircraft.py`

If earthaccess cannot find the data automatically, fall back to manual download:
1. Go to https://asdc.larc.nasa.gov/project/DC3/DC3_Merge_Data_1
2. Click "Get Dataset" → Earthdata Search
3. Download all ICARTT (`.ict`) files
4. Place in `~/Data/DC3/aircraft/merge/`

**Step 3: Verify files exist**

Run: `ls ~/Data/DC3/aircraft/merge/*.ict | head -20`

**Step 4: Run actual DC3 analysis**

Run:
```bash
cd analyses/dc3
export DC3_DATA=~/Data/DC3
export DC3_ANALYSIS=$(pwd)
python scripts/run_obs_analysis.py dc3-obs-dc8
```

Check output in `analyses/dc3/output/dc8/` for generated plots.
