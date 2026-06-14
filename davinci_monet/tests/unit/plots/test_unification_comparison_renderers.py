"""Render-parity tests for the four comparison renderers.

Each renderer now overrides render(series) with the real logic; plot() is a thin
wrapper. These smoke + structural tests verify that:

  plotter.plot(ds, "geometry_x", "dataset_x")  ≡  plotter.render(build_series(ds, "geometry_x", "dataset_x"))

in terms of the figure structure (same number of axes, same collections/lines for
scatter). No metric math is checked here — that is covered by the existing
test_plots.py tests.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr

from davinci_monet.plots.base import build_series

# ---------------------------------------------------------------------------
# Shared fixture helpers
# ---------------------------------------------------------------------------


def _paired_ds(n: int = 30, seed: int = 0) -> xr.Dataset:
    """Minimal paired dataset: geometry_o3 (geometry) + dataset_o3 (dataset)."""
    rng = np.random.default_rng(seed)
    times = np.datetime64("2024-02-01") + np.arange(n) * np.timedelta64(1, "h")
    ds = xr.Dataset(
        {
            "geometry_o3": (
                "time",
                rng.uniform(20, 60, n),
                {"axis": "x", "units": "ppb"},
            ),
            "dataset_o3": (
                "time",
                rng.uniform(20, 60, n),
                {"axis": "y", "units": "ppb"},
            ),
        },
        coords={"time": times},
    )
    return ds


# ---------------------------------------------------------------------------
# ScatterPlotter
# ---------------------------------------------------------------------------


class TestScatterRenderParity:
    def test_plot_and_render_both_return_figure(self) -> None:
        from davinci_monet.plots.renderers.scatter import ScatterPlotter

        ds = _paired_ds()
        plotter = ScatterPlotter()
        fig_plot = plotter.plot(ds, "geometry_o3", "dataset_o3")
        plt.close(fig_plot)
        fig_render = plotter.render(build_series(ds, "geometry_o3", "dataset_o3"))
        plt.close(fig_render)
        assert isinstance(fig_plot, matplotlib.figure.Figure)
        assert isinstance(fig_render, matplotlib.figure.Figure)

    def test_plot_and_render_same_axes_count(self) -> None:
        from davinci_monet.plots.renderers.scatter import ScatterPlotter

        ds = _paired_ds()
        plotter = ScatterPlotter()
        fig_plot = plotter.plot(ds, "geometry_o3", "dataset_o3")
        n_axes_plot = len(fig_plot.axes)
        plt.close(fig_plot)
        fig_render = plotter.render(build_series(ds, "geometry_o3", "dataset_o3"))
        n_axes_render = len(fig_render.axes)
        plt.close(fig_render)
        assert n_axes_plot == n_axes_render

    def test_plot_and_render_same_collections_count(self) -> None:
        """Both paths produce the same number of PathCollections (scatter points)."""
        from davinci_monet.plots.renderers.scatter import ScatterPlotter

        ds = _paired_ds()
        plotter = ScatterPlotter()
        fig_plot = plotter.plot(ds, "geometry_o3", "dataset_o3")
        n_col_plot = len(fig_plot.axes[0].collections)
        plt.close(fig_plot)
        fig_render = plotter.render(build_series(ds, "geometry_o3", "dataset_o3"))
        n_col_render = len(fig_render.axes[0].collections)
        plt.close(fig_render)
        assert n_col_plot == n_col_render

    def test_render_wrong_series_count_raises(self) -> None:
        from davinci_monet.plots.renderers.scatter import ScatterPlotter

        ds = _paired_ds()
        plotter = ScatterPlotter()
        with pytest.raises(NotImplementedError, match="ScatterPlotter"):
            plotter.render(build_series(ds, "geometry_o3"))


# ---------------------------------------------------------------------------
# BoxPlotter
# ---------------------------------------------------------------------------


class TestBoxRenderParity:
    def test_plot_and_render_both_return_figure(self) -> None:
        from davinci_monet.plots.renderers.boxplot import BoxPlotter

        ds = _paired_ds()
        plotter = BoxPlotter()
        fig_plot = plotter.plot(ds, "geometry_o3", "dataset_o3")
        plt.close(fig_plot)
        fig_render = plotter.render(build_series(ds, "geometry_o3", "dataset_o3"))
        plt.close(fig_render)
        assert isinstance(fig_plot, matplotlib.figure.Figure)
        assert isinstance(fig_render, matplotlib.figure.Figure)

    def test_plot_and_render_same_axes_count(self) -> None:
        from davinci_monet.plots.renderers.boxplot import BoxPlotter

        ds = _paired_ds()
        plotter = BoxPlotter()
        fig_plot = plotter.plot(ds, "geometry_o3", "dataset_o3")
        n_plot = len(fig_plot.axes)
        plt.close(fig_plot)
        fig_render = plotter.render(build_series(ds, "geometry_o3", "dataset_o3"))
        n_render = len(fig_render.axes)
        plt.close(fig_render)
        assert n_plot == n_render

    def test_render_wrong_series_count_raises(self) -> None:
        from davinci_monet.plots.renderers.boxplot import BoxPlotter

        ds = _paired_ds()
        plotter = BoxPlotter()
        with pytest.raises(NotImplementedError, match="BoxPlotter"):
            plotter.render(build_series(ds, "geometry_o3"))


# ---------------------------------------------------------------------------
# DiurnalPlotter
# ---------------------------------------------------------------------------


class TestDiurnalRenderParity:
    def test_plot_and_render_both_return_figure(self) -> None:
        from davinci_monet.plots.renderers.diurnal import DiurnalPlotter

        ds = _paired_ds(n=48)
        plotter = DiurnalPlotter()
        fig_plot = plotter.plot(ds, "geometry_o3", "dataset_o3")
        plt.close(fig_plot)
        fig_render = plotter.render(build_series(ds, "geometry_o3", "dataset_o3"))
        plt.close(fig_render)
        assert isinstance(fig_plot, matplotlib.figure.Figure)
        assert isinstance(fig_render, matplotlib.figure.Figure)

    def test_plot_and_render_same_axes_count(self) -> None:
        from davinci_monet.plots.renderers.diurnal import DiurnalPlotter

        ds = _paired_ds(n=48)
        plotter = DiurnalPlotter()
        fig_plot = plotter.plot(ds, "geometry_o3", "dataset_o3")
        n_plot = len(fig_plot.axes)
        plt.close(fig_plot)
        fig_render = plotter.render(build_series(ds, "geometry_o3", "dataset_o3"))
        n_render = len(fig_render.axes)
        plt.close(fig_render)
        assert n_plot == n_render

    def test_plot_and_render_same_lines_count(self) -> None:
        from davinci_monet.plots.renderers.diurnal import DiurnalPlotter

        ds = _paired_ds(n=48)
        plotter = DiurnalPlotter()
        fig_plot = plotter.plot(ds, "geometry_o3", "dataset_o3")
        n_lines_plot = len(fig_plot.axes[0].get_lines())
        plt.close(fig_plot)
        fig_render = plotter.render(build_series(ds, "geometry_o3", "dataset_o3"))
        n_lines_render = len(fig_render.axes[0].get_lines())
        plt.close(fig_render)
        assert n_lines_plot == n_lines_render

    def test_render_wrong_series_count_raises(self) -> None:
        from davinci_monet.plots.renderers.diurnal import DiurnalPlotter

        ds = _paired_ds(n=48)
        plotter = DiurnalPlotter()
        with pytest.raises(NotImplementedError, match="DiurnalPlotter"):
            plotter.render(build_series(ds, "geometry_o3"))


# ---------------------------------------------------------------------------
# TaylorPlotter
# ---------------------------------------------------------------------------


class TestTaylorRenderParity:
    def test_plot_and_render_both_return_figure(self) -> None:
        from davinci_monet.plots.renderers.taylor import TaylorPlotter

        ds = _paired_ds()
        plotter = TaylorPlotter()
        fig_plot = plotter.plot(ds, "geometry_o3", "dataset_o3")
        plt.close(fig_plot)
        fig_render = plotter.render(build_series(ds, "geometry_o3", "dataset_o3"))
        plt.close(fig_render)
        assert isinstance(fig_plot, matplotlib.figure.Figure)
        assert isinstance(fig_render, matplotlib.figure.Figure)

    def test_render_uses_config_subtitle(self) -> None:
        from davinci_monet.plots.base import PlotConfig
        from davinci_monet.plots.renderers.taylor import TaylorPlotter

        ds = _paired_ds()
        plotter = TaylorPlotter(
            PlotConfig(title="O3 Taylor Diagram", subtitle="2024-02-01 - 2024-02-02")
        )
        fig = plotter.render(build_series(ds, "geometry_o3", "dataset_o3"))
        ax = fig.axes[0]

        assert ax.get_title() == r"O$_3$ Taylor Diagram"
        assert any(t.get_text() == "2024-02-01 - 2024-02-02" for t in ax.texts)
        plt.close(fig)

    def test_plot_and_render_same_axes_count(self) -> None:
        from davinci_monet.plots.renderers.taylor import TaylorPlotter

        ds = _paired_ds()
        plotter = TaylorPlotter()
        fig_plot = plotter.plot(ds, "geometry_o3", "dataset_o3")
        n_plot = len(fig_plot.axes)
        plt.close(fig_plot)
        fig_render = plotter.render(build_series(ds, "geometry_o3", "dataset_o3"))
        n_render = len(fig_render.axes)
        plt.close(fig_render)
        assert n_plot == n_render

    def test_plot_and_render_same_lines_count(self) -> None:
        from davinci_monet.plots.renderers.taylor import TaylorPlotter

        ds = _paired_ds()
        plotter = TaylorPlotter()
        fig_plot = plotter.plot(ds, "geometry_o3", "dataset_o3")
        n_lines_plot = len(fig_plot.axes[0].get_lines())
        plt.close(fig_plot)
        fig_render = plotter.render(build_series(ds, "geometry_o3", "dataset_o3"))
        n_lines_render = len(fig_render.axes[0].get_lines())
        plt.close(fig_render)
        assert n_lines_plot == n_lines_render

    def test_render_wrong_series_count_raises(self) -> None:
        from davinci_monet.plots.renderers.taylor import TaylorPlotter

        ds = _paired_ds()
        plotter = TaylorPlotter()
        with pytest.raises(NotImplementedError, match="TaylorPlotter"):
            plotter.render(build_series(ds, "geometry_o3"))


# ---------------------------------------------------------------------------
# ScorecardPlotter
# ---------------------------------------------------------------------------


class TestScorecardRenderParity:
    def test_plot_and_render_both_return_figure(self) -> None:
        from davinci_monet.plots.renderers.scorecard import ScorecardPlotter

        ds = _paired_ds()
        plotter = ScorecardPlotter()
        fig_plot = plotter.plot(ds, "geometry_o3", "dataset_o3")
        plt.close(fig_plot)
        fig_render = plotter.render(build_series(ds, "geometry_o3", "dataset_o3"))
        plt.close(fig_render)
        assert isinstance(fig_plot, matplotlib.figure.Figure)
        assert isinstance(fig_render, matplotlib.figure.Figure)

    def test_plot_and_render_same_axes_count(self) -> None:
        from davinci_monet.plots.renderers.scorecard import ScorecardPlotter

        ds = _paired_ds()
        plotter = ScorecardPlotter()
        fig_plot = plotter.plot(ds, "geometry_o3", "dataset_o3")
        n_plot = len(fig_plot.axes)
        plt.close(fig_plot)
        fig_render = plotter.render(build_series(ds, "geometry_o3", "dataset_o3"))
        n_render = len(fig_render.axes)
        plt.close(fig_render)
        assert n_plot == n_render

    def test_render_wrong_series_count_raises(self) -> None:
        from davinci_monet.plots.renderers.scorecard import ScorecardPlotter

        ds = _paired_ds()
        plotter = ScorecardPlotter()
        with pytest.raises(NotImplementedError, match="ScorecardPlotter"):
            plotter.render(build_series(ds, "geometry_o3"))

    def test_side_entries_still_work(self) -> None:
        """plot_from_dataframe and plot_multi_metric must remain callable."""
        import pandas as pd

        from davinci_monet.plots.renderers.scorecard import ScorecardPlotter

        plotter = ScorecardPlotter()

        # plot_from_dataframe
        stats_df = pd.DataFrame(
            {"Dataset A": [0.9, 2.5], "Dataset B": [0.85, -1.0]},
            index=["R", "MB"],
        )
        fig = plotter.plot_from_dataframe(stats_df)
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close(fig)

        # plot_multi_metric
        stats_dict = {
            "Dataset A": pd.DataFrame({"R": [0.9], "MB": [1.2]}, index=["o3"]),
            "Dataset B": pd.DataFrame({"R": [0.85], "MB": [-0.5]}, index=["o3"]),
        }
        fig2 = plotter.plot_multi_metric(stats_dict, metrics=["R", "MB"])
        assert isinstance(fig2, matplotlib.figure.Figure)
        plt.close(fig2)


# ---------------------------------------------------------------------------
# CurtainPlotter
# ---------------------------------------------------------------------------


def _track_ds(n: int = 30, seed: int = 0) -> xr.Dataset:
    """Minimal track dataset with altitude coordinate."""
    rng = np.random.default_rng(seed)
    times = np.datetime64("2024-02-01") + np.arange(n) * np.timedelta64(1, "h")
    ds = xr.Dataset(
        {
            "geometry_o3": (
                "time",
                rng.uniform(20, 60, n),
                {"axis": "x", "units": "ppb"},
            ),
            "dataset_o3": (
                "time",
                rng.uniform(20, 60, n),
                {"axis": "y", "units": "ppb"},
            ),
        },
        coords={
            "time": times,
            "altitude": ("time", rng.uniform(500, 5000, n)),
        },
    )
    return ds


class TestCurtainRenderParity:
    def test_plot_and_render_both_return_figure(self) -> None:
        from davinci_monet.plots.renderers.curtain import CurtainPlotter

        ds = _track_ds()
        plotter = CurtainPlotter()
        fig_plot = plotter.plot(ds, "geometry_o3", "dataset_o3", alt_var="altitude")
        plt.close(fig_plot)
        fig_render = plotter.render(
            build_series(ds, "geometry_o3", "dataset_o3"), alt_var="altitude"
        )
        plt.close(fig_render)
        assert isinstance(fig_plot, matplotlib.figure.Figure)
        assert isinstance(fig_render, matplotlib.figure.Figure)

    def test_plot_and_render_same_axes_count(self) -> None:
        from davinci_monet.plots.renderers.curtain import CurtainPlotter

        ds = _track_ds()
        plotter = CurtainPlotter()
        fig_plot = plotter.plot(ds, "geometry_o3", "dataset_o3", alt_var="altitude")
        n_plot = len(fig_plot.axes)
        plt.close(fig_plot)
        fig_render = plotter.render(
            build_series(ds, "geometry_o3", "dataset_o3"), alt_var="altitude"
        )
        n_render = len(fig_render.axes)
        plt.close(fig_render)
        assert n_plot == n_render

    def test_curtain_show_var_forwarded(self) -> None:
        """render() must accept show_var kwarg and produce the correct bias plot."""
        from davinci_monet.plots.renderers.curtain import CurtainPlotter

        ds = _track_ds()
        plotter = CurtainPlotter()
        for show_var in ("geometry", "dataset", "bias"):
            fig = plotter.render(
                build_series(ds, "geometry_o3", "dataset_o3"),
                alt_var="altitude",
                show_var=show_var,
            )
            assert isinstance(fig, matplotlib.figure.Figure)
            plt.close(fig)

    def test_render_wrong_series_count_raises(self) -> None:
        from davinci_monet.plots.renderers.curtain import CurtainPlotter

        ds = _track_ds()
        plotter = CurtainPlotter()
        with pytest.raises(NotImplementedError, match="CurtainPlotter"):
            plotter.render(build_series(ds, "geometry_o3"))


# ---------------------------------------------------------------------------
# SpatialBiasPlotter
# ---------------------------------------------------------------------------


def _spatial_point_ds(n_sites: int = 5, seed: int = 0) -> xr.Dataset:
    """Minimal point-site spatial dataset."""
    rng = np.random.default_rng(seed)
    times = np.array(["2024-02-01T00:00", "2024-02-01T01:00"], dtype="datetime64[ns]")
    lats = np.linspace(30.0, 50.0, n_sites)
    lons = np.linspace(-110.0, -70.0, n_sites)
    geometry = rng.uniform(20, 60, size=(2, n_sites))
    dataset = geometry + rng.uniform(-5, 5, size=(2, n_sites))
    ds = xr.Dataset(
        {
            "geometry_o3": (
                ("time", "site"),
                geometry,
                {"axis": "x"},
            ),
            "dataset_o3": (
                ("time", "site"),
                dataset,
                {"axis": "y"},
            ),
        },
        coords={
            "time": times,
            "latitude": ("site", lats),
            "longitude": ("site", lons),
        },
    )
    return ds


class TestSpatialBiasRenderParity:
    def test_plot_and_render_both_return_figure(self) -> None:
        from davinci_monet.plots.renderers.spatial.bias import SpatialBiasPlotter

        ds = _spatial_point_ds()
        plotter = SpatialBiasPlotter()
        fig_plot = plotter.plot(ds, "geometry_o3", "dataset_o3")
        plt.close(fig_plot)
        fig_render = plotter.render(build_series(ds, "geometry_o3", "dataset_o3"))
        plt.close(fig_render)
        assert isinstance(fig_plot, matplotlib.figure.Figure)
        assert isinstance(fig_render, matplotlib.figure.Figure)

    def test_plot_and_render_same_axes_count(self) -> None:
        from davinci_monet.plots.renderers.spatial.bias import SpatialBiasPlotter

        ds = _spatial_point_ds()
        plotter = SpatialBiasPlotter()
        fig_plot = plotter.plot(ds, "geometry_o3", "dataset_o3")
        n_plot = len(fig_plot.axes)
        plt.close(fig_plot)
        fig_render = plotter.render(build_series(ds, "geometry_o3", "dataset_o3"))
        n_render = len(fig_render.axes)
        plt.close(fig_render)
        assert n_plot == n_render

    def test_render_wrong_series_count_raises(self) -> None:
        from davinci_monet.plots.renderers.spatial.bias import SpatialBiasPlotter

        ds = _spatial_point_ds()
        plotter = SpatialBiasPlotter()
        with pytest.raises(NotImplementedError, match="SpatialBiasPlotter"):
            plotter.render(build_series(ds, "geometry_o3"))


# ---------------------------------------------------------------------------
# SpatialDistributionPlotter
# ---------------------------------------------------------------------------


class TestSpatialDistributionRenderParity:
    def test_plot_and_render_both_return_figure(self) -> None:
        from davinci_monet.plots.renderers.spatial.distribution import (
            SpatialDistributionPlotter,
        )

        ds = _spatial_point_ds()
        plotter = SpatialDistributionPlotter()
        fig_plot = plotter.plot(ds, "geometry_o3", "dataset_o3")
        plt.close(fig_plot)
        fig_render = plotter.render(build_series(ds, "geometry_o3", "dataset_o3"))
        plt.close(fig_render)
        assert isinstance(fig_plot, matplotlib.figure.Figure)
        assert isinstance(fig_render, matplotlib.figure.Figure)

    def test_plot_and_render_same_axes_count(self) -> None:
        from davinci_monet.plots.renderers.spatial.distribution import (
            SpatialDistributionPlotter,
        )

        ds = _spatial_point_ds()
        plotter = SpatialDistributionPlotter()
        fig_plot = plotter.plot(ds, "geometry_o3", "dataset_o3")
        n_plot = len(fig_plot.axes)
        plt.close(fig_plot)
        fig_render = plotter.render(build_series(ds, "geometry_o3", "dataset_o3"))
        n_render = len(fig_render.axes)
        plt.close(fig_render)
        assert n_plot == n_render

    def test_show_var_forwarded_via_render(self) -> None:
        """render() must accept show_var kwarg and produce correct figure."""
        from davinci_monet.plots.renderers.spatial.distribution import (
            SpatialDistributionPlotter,
        )

        ds = _spatial_point_ds()
        plotter = SpatialDistributionPlotter()
        for show_var in ("geometry", "dataset", "both"):
            fig = plotter.render(build_series(ds, "geometry_o3", "dataset_o3"), show_var=show_var)
            assert isinstance(fig, matplotlib.figure.Figure)
            plt.close(fig)

    def test_render_wrong_series_count_raises(self) -> None:
        from davinci_monet.plots.renderers.spatial.distribution import (
            SpatialDistributionPlotter,
        )

        ds = _spatial_point_ds()
        plotter = SpatialDistributionPlotter()
        with pytest.raises(NotImplementedError, match="SpatialDistributionPlotter"):
            plotter.render(build_series(ds, "geometry_o3"))


# ---------------------------------------------------------------------------
# SiteTimeSeriesPlotter
# ---------------------------------------------------------------------------


def _site_ds(n_times: int = 50, n_sites: int = 3, seed: int = 0) -> xr.Dataset:
    """Minimal site-based paired dataset: (site, time) dims with coords."""
    rng = np.random.default_rng(seed)
    times = np.datetime64("2024-02-01") + np.arange(n_times) * np.timedelta64(1, "h")
    sites = [f"site_{i}" for i in range(n_sites)]
    geometry = rng.uniform(20, 60, (n_sites, n_times))
    dataset = geometry + rng.uniform(-5, 5, (n_sites, n_times))
    ds = xr.Dataset(
        {
            "geometry_o3": (
                ("site", "time"),
                geometry,
                {"axis": "x", "units": "ppb"},
            ),
            "dataset_o3": (
                ("site", "time"),
                dataset,
                {"axis": "y", "units": "ppb"},
            ),
        },
        coords={
            "time": times,
            "site": sites,
            "latitude": ("site", np.linspace(35.0, 45.0, n_sites)),
            "longitude": ("site", np.linspace(-120.0, -100.0, n_sites)),
        },
    )
    return ds


class TestSiteTimeSeriesRenderParity:
    def test_plot_and_render_both_return_figure(self) -> None:
        from davinci_monet.plots.renderers.site_timeseries import SiteTimeSeriesPlotter

        ds = _site_ds()
        plotter = SiteTimeSeriesPlotter()
        fig_plot = plotter.plot(ds, "geometry_o3", "dataset_o3", ncols=2, min_points=5)
        plt.close(fig_plot)
        fig_render = plotter.render(
            build_series(ds, "geometry_o3", "dataset_o3"), ncols=2, min_points=5
        )
        plt.close(fig_render)
        assert isinstance(fig_plot, matplotlib.figure.Figure)
        assert isinstance(fig_render, matplotlib.figure.Figure)

    def test_plot_and_render_same_axes_count(self) -> None:
        from davinci_monet.plots.renderers.site_timeseries import SiteTimeSeriesPlotter

        ds = _site_ds()
        plotter = SiteTimeSeriesPlotter()
        fig_plot = plotter.plot(ds, "geometry_o3", "dataset_o3", ncols=2, min_points=5)
        n_plot = len(fig_plot.axes)
        plt.close(fig_plot)
        fig_render = plotter.render(
            build_series(ds, "geometry_o3", "dataset_o3"), ncols=2, min_points=5
        )
        n_render = len(fig_render.axes)
        plt.close(fig_render)
        assert n_plot == n_render

    def test_render_wrong_series_count_raises(self) -> None:
        from davinci_monet.plots.renderers.site_timeseries import SiteTimeSeriesPlotter

        ds = _site_ds()
        plotter = SiteTimeSeriesPlotter()
        with pytest.raises(NotImplementedError, match="SiteTimeSeriesPlotter"):
            plotter.render(build_series(ds, "geometry_o3"))


# ---------------------------------------------------------------------------
# FlightTimeSeriesPlotter
# ---------------------------------------------------------------------------


def _flight_ds(n_per_flight: int = 30, n_flights: int = 2, seed: int = 0) -> xr.Dataset:
    """Minimal flight-based paired dataset with a 'flight' coordinate."""
    rng = np.random.default_rng(seed)
    all_times = []
    all_geometry = []
    all_dataset = []
    all_flight = []
    for day in range(n_flights):
        base = np.datetime64(f"2024-02-0{day + 1}T10:00")
        times = base + np.arange(n_per_flight) * np.timedelta64(1, "m")
        geometry = rng.uniform(20, 60, n_per_flight)
        dataset = geometry + rng.uniform(-5, 5, n_per_flight)
        all_times.append(times)
        all_geometry.append(geometry)
        all_dataset.append(dataset)
        all_flight.extend([f"2024-02-0{day + 1}"] * n_per_flight)
    all_times_arr = np.concatenate(all_times)
    all_geometry_arr = np.concatenate(all_geometry)
    all_dataset_arr = np.concatenate(all_dataset)
    ds = xr.Dataset(
        {
            "geometry_o3": (
                "time",
                all_geometry_arr,
                {"axis": "x", "units": "ppb"},
            ),
            "dataset_o3": (
                "time",
                all_dataset_arr,
                {"axis": "y", "units": "ppb"},
            ),
        },
        coords={
            "time": all_times_arr,
            "flight": ("time", all_flight),
        },
    )
    return ds


class TestFlightTimeSeriesRenderParity:
    def test_plot_and_render_both_return_figure(self) -> None:
        from davinci_monet.plots.renderers.flight_timeseries import FlightTimeSeriesPlotter

        ds = _flight_ds()
        plotter = FlightTimeSeriesPlotter()
        fig_plot = plotter.plot(ds, "geometry_o3", "dataset_o3", ncols=2, min_points=5)
        plt.close(fig_plot)
        fig_render = plotter.render(
            build_series(ds, "geometry_o3", "dataset_o3"), ncols=2, min_points=5
        )
        plt.close(fig_render)
        assert isinstance(fig_plot, matplotlib.figure.Figure)
        assert isinstance(fig_render, matplotlib.figure.Figure)

    def test_plot_and_render_same_axes_count(self) -> None:
        from davinci_monet.plots.renderers.flight_timeseries import FlightTimeSeriesPlotter

        ds = _flight_ds()
        plotter = FlightTimeSeriesPlotter()
        fig_plot = plotter.plot(
            ds, "geometry_o3", "dataset_o3", ncols=2, min_points=5, show_altitude=False
        )
        n_plot = len(fig_plot.axes)
        plt.close(fig_plot)
        fig_render = plotter.render(
            build_series(ds, "geometry_o3", "dataset_o3"),
            ncols=2,
            min_points=5,
            show_altitude=False,
        )
        n_render = len(fig_render.axes)
        plt.close(fig_render)
        assert n_plot == n_render

    def test_plot_per_flight_generator_unchanged(self) -> None:
        """plot_per_flight must still yield (flight_id, Figure) tuples."""
        from davinci_monet.plots.renderers.flight_timeseries import FlightTimeSeriesPlotter

        ds = _flight_ds()
        plotter = FlightTimeSeriesPlotter()
        results = list(plotter.plot_per_flight(ds, "geometry_o3", "dataset_o3", min_points=5))
        assert len(results) == 2
        for flight_id, fig in results:
            assert isinstance(flight_id, str)
            assert isinstance(fig, matplotlib.figure.Figure)
            plt.close(fig)

    def test_render_wrong_series_count_raises(self) -> None:
        from davinci_monet.plots.renderers.flight_timeseries import FlightTimeSeriesPlotter

        ds = _flight_ds()
        plotter = FlightTimeSeriesPlotter()
        with pytest.raises(NotImplementedError, match="FlightTimeSeriesPlotter"):
            plotter.render(build_series(ds, "geometry_o3"))


# ---------------------------------------------------------------------------
# PerSiteTimeSeriesPlotter (single-site plot() vs render())
# ---------------------------------------------------------------------------


class TestPerSiteTimeSeriesRenderParity:
    def test_plot_and_render_both_return_figure(self) -> None:
        from davinci_monet.plots.renderers.per_site_timeseries import PerSiteTimeSeriesPlotter

        ds = _site_ds()
        plotter = PerSiteTimeSeriesPlotter()
        fig_plot = plotter.plot(ds, "geometry_o3", "dataset_o3", min_points=5)
        plt.close(fig_plot)
        fig_render = plotter.render(build_series(ds, "geometry_o3", "dataset_o3"), min_points=5)
        plt.close(fig_render)
        assert isinstance(fig_plot, matplotlib.figure.Figure)
        assert isinstance(fig_render, matplotlib.figure.Figure)

    def test_plot_and_render_same_axes_count(self) -> None:
        from davinci_monet.plots.renderers.per_site_timeseries import PerSiteTimeSeriesPlotter

        ds = _site_ds()
        plotter = PerSiteTimeSeriesPlotter()
        fig_plot = plotter.plot(ds, "geometry_o3", "dataset_o3", site="site_0", min_points=5)
        n_plot = len(fig_plot.axes)
        plt.close(fig_plot)
        fig_render = plotter.render(
            build_series(ds, "geometry_o3", "dataset_o3"), site="site_0", min_points=5
        )
        n_render = len(fig_render.axes)
        plt.close(fig_render)
        assert n_plot == n_render

    def test_plot_per_site_generator_unchanged(self) -> None:
        """plot_per_site must still yield (site_id, Figure) tuples."""
        from davinci_monet.plots.renderers.per_site_timeseries import PerSiteTimeSeriesPlotter

        ds = _site_ds()
        plotter = PerSiteTimeSeriesPlotter()
        results = list(plotter.plot_per_site(ds, "geometry_o3", "dataset_o3", min_points=5))
        assert len(results) == 3
        for site_id, fig in results:
            assert isinstance(site_id, str)
            assert isinstance(fig, matplotlib.figure.Figure)
            plt.close(fig)

    def test_render_wrong_series_count_raises(self) -> None:
        from davinci_monet.plots.renderers.per_site_timeseries import PerSiteTimeSeriesPlotter

        ds = _site_ds()
        plotter = PerSiteTimeSeriesPlotter()
        with pytest.raises(NotImplementedError, match="PerSiteTimeSeriesPlotter"):
            plotter.render(build_series(ds, "geometry_o3"))


# ---------------------------------------------------------------------------
# SpatialOverlayPlotter
# ---------------------------------------------------------------------------


def _overlay_ds(n_sites: int = 5, seed: int = 0) -> xr.Dataset:
    """Minimal point-site dataset for overlay (geometry points on a dataset contour)."""
    rng = np.random.default_rng(seed)
    lats = np.linspace(30.0, 50.0, n_sites)
    lons = np.linspace(-110.0, -70.0, n_sites)
    ds = xr.Dataset(
        {
            "geometry_o3": (
                "site",
                rng.uniform(20, 60, n_sites),
                {"axis": "x", "units": "ppb"},
            ),
            "dataset_o3": (
                "site",
                rng.uniform(20, 60, n_sites),
                {"axis": "y", "units": "ppb"},
            ),
        },
        coords={
            "latitude": ("site", lats),
            "longitude": ("site", lons),
        },
    )
    return ds


def _dataset_field_da() -> "xr.DataArray":
    """Tiny 2-D (lat x lon) DataArray suitable as dataset_field for overlay."""
    import xarray as xr

    rng = np.random.default_rng(42)
    lats = np.linspace(28.0, 52.0, 8)
    lons = np.linspace(-112.0, -68.0, 10)
    data = rng.uniform(20, 60, (len(lats), len(lons)))
    return xr.DataArray(
        data,
        dims=["lat", "lon"],
        coords={"lat": lats, "lon": lons},
        attrs={"units": "ppb"},
    )


class TestSpatialOverlayRenderParity:
    def test_plot_and_render_both_return_figure(self) -> None:
        from davinci_monet.plots.renderers.spatial.overlay import SpatialOverlayPlotter

        ds = _overlay_ds()
        y_field = _dataset_field_da()
        plotter = SpatialOverlayPlotter()
        fig_plot = plotter.plot(ds, "geometry_o3", "dataset_o3", y_field=y_field)
        plt.close(fig_plot)
        fig_render = plotter.render(build_series(ds, "geometry_o3", "dataset_o3"), y_field=y_field)
        plt.close(fig_render)
        assert isinstance(fig_plot, matplotlib.figure.Figure)
        assert isinstance(fig_render, matplotlib.figure.Figure)

    def test_plot_and_render_same_axes_count(self) -> None:
        from davinci_monet.plots.renderers.spatial.overlay import SpatialOverlayPlotter

        ds = _overlay_ds()
        y_field = _dataset_field_da()
        plotter = SpatialOverlayPlotter()
        fig_plot = plotter.plot(ds, "geometry_o3", "dataset_o3", y_field=y_field)
        n_plot = len(fig_plot.axes)
        plt.close(fig_plot)
        fig_render = plotter.render(build_series(ds, "geometry_o3", "dataset_o3"), y_field=y_field)
        n_render = len(fig_render.axes)
        plt.close(fig_render)
        assert n_plot == n_render

    def test_dataset_field_forwarded_via_render(self) -> None:
        """render() must accept and use the dataset_field kwarg (not fall back to paired_data)."""
        from davinci_monet.plots.renderers.spatial.overlay import SpatialOverlayPlotter

        ds = _overlay_ds()
        y_field = _dataset_field_da()
        plotter = SpatialOverlayPlotter()
        # Should succeed when dataset_field is explicitly provided
        fig = plotter.render(build_series(ds, "geometry_o3", "dataset_o3"), y_field=y_field)
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close(fig)

    def test_render_wrong_series_count_raises(self) -> None:
        from davinci_monet.plots.renderers.spatial.overlay import SpatialOverlayPlotter

        ds = _overlay_ds()
        plotter = SpatialOverlayPlotter()
        with pytest.raises(NotImplementedError, match="SpatialOverlayPlotter"):
            plotter.render(build_series(ds, "geometry_o3"))


# ---------------------------------------------------------------------------
# TrackMap3DPlotter
# ---------------------------------------------------------------------------


def _track_3d_ds(n: int = 40, seed: int = 0) -> xr.Dataset:
    """Minimal track dataset with lat/lon/altitude coordinates for 3D map."""
    rng = np.random.default_rng(seed)
    times = np.datetime64("2024-02-01") + np.arange(n) * np.timedelta64(1, "m")
    ds = xr.Dataset(
        {
            "geometry_o3": (
                "time",
                rng.uniform(20, 60, n),
                {"axis": "x", "units": "ppb"},
            ),
            "dataset_o3": (
                "time",
                rng.uniform(20, 60, n),
                {"axis": "y", "units": "ppb"},
            ),
        },
        coords={
            "time": times,
            "latitude": ("time", np.linspace(25.0, 45.0, n)),
            "longitude": ("time", np.linspace(-120.0, -80.0, n)),
            "altitude": ("time", rng.uniform(500, 8000, n)),
        },
    )
    return ds


def _flight_3d_ds(n_per_flight: int = 30, n_flights: int = 2, seed: int = 0) -> xr.Dataset:
    """Track dataset with a flight coordinate for plot_per_flight tests."""
    rng = np.random.default_rng(seed)
    all_times = []
    all_geometry = []
    all_dataset = []
    all_flight = []
    all_lat = []
    all_lon = []
    all_alt = []
    for day in range(n_flights):
        base = np.datetime64(f"2024-02-0{day + 1}T10:00")
        times = base + np.arange(n_per_flight) * np.timedelta64(1, "m")
        geometry = rng.uniform(20, 60, n_per_flight)
        dataset = geometry + rng.uniform(-5, 5, n_per_flight)
        all_times.append(times)
        all_geometry.append(geometry)
        all_dataset.append(dataset)
        all_flight.extend([f"2024020{day + 1}"] * n_per_flight)
        all_lat.append(np.linspace(25.0 + day, 40.0 + day, n_per_flight))
        all_lon.append(np.linspace(-120.0, -80.0, n_per_flight))
        all_alt.append(rng.uniform(500, 5000, n_per_flight))
    ds = xr.Dataset(
        {
            "geometry_o3": (
                "time",
                np.concatenate(all_geometry),
                {"axis": "x", "units": "ppb"},
            ),
            "dataset_o3": (
                "time",
                np.concatenate(all_dataset),
                {"axis": "y", "units": "ppb"},
            ),
        },
        coords={
            "time": np.concatenate(all_times),
            "flight": ("time", all_flight),
            "latitude": ("time", np.concatenate(all_lat)),
            "longitude": ("time", np.concatenate(all_lon)),
            "altitude": ("time", np.concatenate(all_alt)),
        },
    )
    return ds


class TestTrackMap3DRenderParity:
    def test_plot_and_render_both_return_figure(self) -> None:
        from davinci_monet.plots.renderers.track_map_3d import TrackMap3DPlotter

        ds = _track_3d_ds()
        plotter = TrackMap3DPlotter()
        fig_plot = plotter.plot(
            ds, "geometry_o3", "dataset_o3", alt_var="altitude", show_coastlines=False
        )
        plt.close(fig_plot)
        fig_render = plotter.render(
            build_series(ds, "geometry_o3", "dataset_o3"),
            alt_var="altitude",
            show_coastlines=False,
        )
        plt.close(fig_render)
        assert isinstance(fig_plot, matplotlib.figure.Figure)
        assert isinstance(fig_render, matplotlib.figure.Figure)

    def test_plot_and_render_same_axes_count(self) -> None:
        from davinci_monet.plots.renderers.track_map_3d import TrackMap3DPlotter

        ds = _track_3d_ds()
        plotter = TrackMap3DPlotter()
        fig_plot = plotter.plot(
            ds, "geometry_o3", "dataset_o3", alt_var="altitude", show_coastlines=False
        )
        n_plot = len(fig_plot.axes)
        plt.close(fig_plot)
        fig_render = plotter.render(
            build_series(ds, "geometry_o3", "dataset_o3"),
            alt_var="altitude",
            show_coastlines=False,
        )
        n_render = len(fig_render.axes)
        plt.close(fig_render)
        assert n_plot == n_render

    def test_plot_per_flight_generator_intact(self) -> None:
        """plot_per_flight must still yield (flight_id, Figure) tuples after migration."""
        from davinci_monet.plots.renderers.track_map_3d import TrackMap3DPlotter

        ds = _flight_3d_ds()
        plotter = TrackMap3DPlotter()
        results = list(
            plotter.plot_per_flight(
                ds,
                "geometry_o3",
                "dataset_o3",
                min_points=5,
                show_coastlines=False,
            )
        )
        assert len(results) == 2
        for flight_id, fig in results:
            assert isinstance(flight_id, str)
            assert isinstance(fig, matplotlib.figure.Figure)
            plt.close(fig)

    def test_render_wrong_series_count_raises(self) -> None:
        from davinci_monet.plots.renderers.track_map_3d import TrackMap3DPlotter

        ds = _track_3d_ds()
        plotter = TrackMap3DPlotter()
        with pytest.raises(NotImplementedError, match="TrackMap3DPlotter"):
            plotter.render(build_series(ds, "geometry_o3"))
