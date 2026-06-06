"""Render-parity tests for the four comparison renderers.

Each renderer now overrides render(series) with the real logic; plot() is a thin
wrapper. These smoke + structural tests verify that:

  plotter.plot(ds, "obs_x", "model_x")  ≡  plotter.render(build_series(ds, "obs_x", "model_x"))

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
    """Minimal paired dataset: obs_o3 (reference) + model_o3 (comparand)."""
    rng = np.random.default_rng(seed)
    times = np.datetime64("2024-02-01") + np.arange(n) * np.timedelta64(1, "h")
    ds = xr.Dataset(
        {
            "obs_o3": (
                "time",
                rng.uniform(20, 60, n),
                {"role": "obs", "pair_role": "reference", "units": "ppb"},
            ),
            "model_o3": (
                "time",
                rng.uniform(20, 60, n),
                {"role": "model", "pair_role": "comparand", "units": "ppb"},
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
        fig_plot = plotter.plot(ds, "obs_o3", "model_o3")
        plt.close(fig_plot)
        fig_render = plotter.render(build_series(ds, "obs_o3", "model_o3"))
        plt.close(fig_render)
        assert isinstance(fig_plot, matplotlib.figure.Figure)
        assert isinstance(fig_render, matplotlib.figure.Figure)

    def test_plot_and_render_same_axes_count(self) -> None:
        from davinci_monet.plots.renderers.scatter import ScatterPlotter

        ds = _paired_ds()
        plotter = ScatterPlotter()
        fig_plot = plotter.plot(ds, "obs_o3", "model_o3")
        n_axes_plot = len(fig_plot.axes)
        plt.close(fig_plot)
        fig_render = plotter.render(build_series(ds, "obs_o3", "model_o3"))
        n_axes_render = len(fig_render.axes)
        plt.close(fig_render)
        assert n_axes_plot == n_axes_render

    def test_plot_and_render_same_collections_count(self) -> None:
        """Both paths produce the same number of PathCollections (scatter points)."""
        from davinci_monet.plots.renderers.scatter import ScatterPlotter

        ds = _paired_ds()
        plotter = ScatterPlotter()
        fig_plot = plotter.plot(ds, "obs_o3", "model_o3")
        n_col_plot = len(fig_plot.axes[0].collections)
        plt.close(fig_plot)
        fig_render = plotter.render(build_series(ds, "obs_o3", "model_o3"))
        n_col_render = len(fig_render.axes[0].collections)
        plt.close(fig_render)
        assert n_col_plot == n_col_render

    def test_render_wrong_series_count_raises(self) -> None:
        from davinci_monet.plots.renderers.scatter import ScatterPlotter

        ds = _paired_ds()
        plotter = ScatterPlotter()
        with pytest.raises(NotImplementedError, match="ScatterPlotter"):
            plotter.render(build_series(ds, "obs_o3"))


# ---------------------------------------------------------------------------
# BoxPlotter
# ---------------------------------------------------------------------------


class TestBoxRenderParity:
    def test_plot_and_render_both_return_figure(self) -> None:
        from davinci_monet.plots.renderers.boxplot import BoxPlotter

        ds = _paired_ds()
        plotter = BoxPlotter()
        fig_plot = plotter.plot(ds, "obs_o3", "model_o3")
        plt.close(fig_plot)
        fig_render = plotter.render(build_series(ds, "obs_o3", "model_o3"))
        plt.close(fig_render)
        assert isinstance(fig_plot, matplotlib.figure.Figure)
        assert isinstance(fig_render, matplotlib.figure.Figure)

    def test_plot_and_render_same_axes_count(self) -> None:
        from davinci_monet.plots.renderers.boxplot import BoxPlotter

        ds = _paired_ds()
        plotter = BoxPlotter()
        fig_plot = plotter.plot(ds, "obs_o3", "model_o3")
        n_plot = len(fig_plot.axes)
        plt.close(fig_plot)
        fig_render = plotter.render(build_series(ds, "obs_o3", "model_o3"))
        n_render = len(fig_render.axes)
        plt.close(fig_render)
        assert n_plot == n_render

    def test_render_wrong_series_count_raises(self) -> None:
        from davinci_monet.plots.renderers.boxplot import BoxPlotter

        ds = _paired_ds()
        plotter = BoxPlotter()
        with pytest.raises(NotImplementedError, match="BoxPlotter"):
            plotter.render(build_series(ds, "obs_o3"))


# ---------------------------------------------------------------------------
# DiurnalPlotter
# ---------------------------------------------------------------------------


class TestDiurnalRenderParity:
    def test_plot_and_render_both_return_figure(self) -> None:
        from davinci_monet.plots.renderers.diurnal import DiurnalPlotter

        ds = _paired_ds(n=48)
        plotter = DiurnalPlotter()
        fig_plot = plotter.plot(ds, "obs_o3", "model_o3")
        plt.close(fig_plot)
        fig_render = plotter.render(build_series(ds, "obs_o3", "model_o3"))
        plt.close(fig_render)
        assert isinstance(fig_plot, matplotlib.figure.Figure)
        assert isinstance(fig_render, matplotlib.figure.Figure)

    def test_plot_and_render_same_axes_count(self) -> None:
        from davinci_monet.plots.renderers.diurnal import DiurnalPlotter

        ds = _paired_ds(n=48)
        plotter = DiurnalPlotter()
        fig_plot = plotter.plot(ds, "obs_o3", "model_o3")
        n_plot = len(fig_plot.axes)
        plt.close(fig_plot)
        fig_render = plotter.render(build_series(ds, "obs_o3", "model_o3"))
        n_render = len(fig_render.axes)
        plt.close(fig_render)
        assert n_plot == n_render

    def test_plot_and_render_same_lines_count(self) -> None:
        from davinci_monet.plots.renderers.diurnal import DiurnalPlotter

        ds = _paired_ds(n=48)
        plotter = DiurnalPlotter()
        fig_plot = plotter.plot(ds, "obs_o3", "model_o3")
        n_lines_plot = len(fig_plot.axes[0].get_lines())
        plt.close(fig_plot)
        fig_render = plotter.render(build_series(ds, "obs_o3", "model_o3"))
        n_lines_render = len(fig_render.axes[0].get_lines())
        plt.close(fig_render)
        assert n_lines_plot == n_lines_render

    def test_render_wrong_series_count_raises(self) -> None:
        from davinci_monet.plots.renderers.diurnal import DiurnalPlotter

        ds = _paired_ds(n=48)
        plotter = DiurnalPlotter()
        with pytest.raises(NotImplementedError, match="DiurnalPlotter"):
            plotter.render(build_series(ds, "obs_o3"))


# ---------------------------------------------------------------------------
# TaylorPlotter
# ---------------------------------------------------------------------------


class TestTaylorRenderParity:
    def test_plot_and_render_both_return_figure(self) -> None:
        from davinci_monet.plots.renderers.taylor import TaylorPlotter

        ds = _paired_ds()
        plotter = TaylorPlotter()
        fig_plot = plotter.plot(ds, "obs_o3", "model_o3")
        plt.close(fig_plot)
        fig_render = plotter.render(build_series(ds, "obs_o3", "model_o3"))
        plt.close(fig_render)
        assert isinstance(fig_plot, matplotlib.figure.Figure)
        assert isinstance(fig_render, matplotlib.figure.Figure)

    def test_plot_and_render_same_axes_count(self) -> None:
        from davinci_monet.plots.renderers.taylor import TaylorPlotter

        ds = _paired_ds()
        plotter = TaylorPlotter()
        fig_plot = plotter.plot(ds, "obs_o3", "model_o3")
        n_plot = len(fig_plot.axes)
        plt.close(fig_plot)
        fig_render = plotter.render(build_series(ds, "obs_o3", "model_o3"))
        n_render = len(fig_render.axes)
        plt.close(fig_render)
        assert n_plot == n_render

    def test_plot_and_render_same_lines_count(self) -> None:
        from davinci_monet.plots.renderers.taylor import TaylorPlotter

        ds = _paired_ds()
        plotter = TaylorPlotter()
        fig_plot = plotter.plot(ds, "obs_o3", "model_o3")
        n_lines_plot = len(fig_plot.axes[0].get_lines())
        plt.close(fig_plot)
        fig_render = plotter.render(build_series(ds, "obs_o3", "model_o3"))
        n_lines_render = len(fig_render.axes[0].get_lines())
        plt.close(fig_render)
        assert n_lines_plot == n_lines_render

    def test_render_wrong_series_count_raises(self) -> None:
        from davinci_monet.plots.renderers.taylor import TaylorPlotter

        ds = _paired_ds()
        plotter = TaylorPlotter()
        with pytest.raises(NotImplementedError, match="TaylorPlotter"):
            plotter.render(build_series(ds, "obs_o3"))


# ---------------------------------------------------------------------------
# ScorecardPlotter
# ---------------------------------------------------------------------------


class TestScorecardRenderParity:
    def test_plot_and_render_both_return_figure(self) -> None:
        from davinci_monet.plots.renderers.scorecard import ScorecardPlotter

        ds = _paired_ds()
        plotter = ScorecardPlotter()
        fig_plot = plotter.plot(ds, "obs_o3", "model_o3")
        plt.close(fig_plot)
        fig_render = plotter.render(build_series(ds, "obs_o3", "model_o3"))
        plt.close(fig_render)
        assert isinstance(fig_plot, matplotlib.figure.Figure)
        assert isinstance(fig_render, matplotlib.figure.Figure)

    def test_plot_and_render_same_axes_count(self) -> None:
        from davinci_monet.plots.renderers.scorecard import ScorecardPlotter

        ds = _paired_ds()
        plotter = ScorecardPlotter()
        fig_plot = plotter.plot(ds, "obs_o3", "model_o3")
        n_plot = len(fig_plot.axes)
        plt.close(fig_plot)
        fig_render = plotter.render(build_series(ds, "obs_o3", "model_o3"))
        n_render = len(fig_render.axes)
        plt.close(fig_render)
        assert n_plot == n_render

    def test_render_wrong_series_count_raises(self) -> None:
        from davinci_monet.plots.renderers.scorecard import ScorecardPlotter

        ds = _paired_ds()
        plotter = ScorecardPlotter()
        with pytest.raises(NotImplementedError, match="ScorecardPlotter"):
            plotter.render(build_series(ds, "obs_o3"))

    def test_side_entries_still_work(self) -> None:
        """plot_from_dataframe and plot_multi_metric must remain callable."""
        import pandas as pd

        from davinci_monet.plots.renderers.scorecard import ScorecardPlotter

        plotter = ScorecardPlotter()

        # plot_from_dataframe
        stats_df = pd.DataFrame(
            {"Model A": [0.9, 2.5], "Model B": [0.85, -1.0]},
            index=["R", "MB"],
        )
        fig = plotter.plot_from_dataframe(stats_df)
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close(fig)

        # plot_multi_metric
        stats_dict = {
            "Model A": pd.DataFrame({"R": [0.9], "MB": [1.2]}, index=["o3"]),
            "Model B": pd.DataFrame({"R": [0.85], "MB": [-0.5]}, index=["o3"]),
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
            "obs_o3": (
                "time",
                rng.uniform(20, 60, n),
                {"role": "obs", "pair_role": "reference", "units": "ppb"},
            ),
            "model_o3": (
                "time",
                rng.uniform(20, 60, n),
                {"role": "model", "pair_role": "comparand", "units": "ppb"},
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
        fig_plot = plotter.plot(ds, "obs_o3", "model_o3", alt_var="altitude")
        plt.close(fig_plot)
        fig_render = plotter.render(build_series(ds, "obs_o3", "model_o3"), alt_var="altitude")
        plt.close(fig_render)
        assert isinstance(fig_plot, matplotlib.figure.Figure)
        assert isinstance(fig_render, matplotlib.figure.Figure)

    def test_plot_and_render_same_axes_count(self) -> None:
        from davinci_monet.plots.renderers.curtain import CurtainPlotter

        ds = _track_ds()
        plotter = CurtainPlotter()
        fig_plot = plotter.plot(ds, "obs_o3", "model_o3", alt_var="altitude")
        n_plot = len(fig_plot.axes)
        plt.close(fig_plot)
        fig_render = plotter.render(build_series(ds, "obs_o3", "model_o3"), alt_var="altitude")
        n_render = len(fig_render.axes)
        plt.close(fig_render)
        assert n_plot == n_render

    def test_curtain_show_var_forwarded(self) -> None:
        """render() must accept show_var kwarg and produce the correct bias plot."""
        from davinci_monet.plots.renderers.curtain import CurtainPlotter

        ds = _track_ds()
        plotter = CurtainPlotter()
        for show_var in ("obs", "model", "bias"):
            fig = plotter.render(
                build_series(ds, "obs_o3", "model_o3"),
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
            plotter.render(build_series(ds, "obs_o3"))


# ---------------------------------------------------------------------------
# SpatialBiasPlotter
# ---------------------------------------------------------------------------


def _spatial_point_ds(n_sites: int = 5, seed: int = 0) -> xr.Dataset:
    """Minimal point-site spatial dataset."""
    rng = np.random.default_rng(seed)
    times = np.array(["2024-02-01T00:00", "2024-02-01T01:00"], dtype="datetime64[ns]")
    lats = np.linspace(30.0, 50.0, n_sites)
    lons = np.linspace(-110.0, -70.0, n_sites)
    obs = rng.uniform(20, 60, size=(2, n_sites))
    mod = obs + rng.uniform(-5, 5, size=(2, n_sites))
    ds = xr.Dataset(
        {
            "obs_o3": (
                ("time", "site"),
                obs,
                {"role": "obs", "pair_role": "reference"},
            ),
            "model_o3": (
                ("time", "site"),
                mod,
                {"role": "model", "pair_role": "comparand"},
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
        fig_plot = plotter.plot(ds, "obs_o3", "model_o3")
        plt.close(fig_plot)
        fig_render = plotter.render(build_series(ds, "obs_o3", "model_o3"))
        plt.close(fig_render)
        assert isinstance(fig_plot, matplotlib.figure.Figure)
        assert isinstance(fig_render, matplotlib.figure.Figure)

    def test_plot_and_render_same_axes_count(self) -> None:
        from davinci_monet.plots.renderers.spatial.bias import SpatialBiasPlotter

        ds = _spatial_point_ds()
        plotter = SpatialBiasPlotter()
        fig_plot = plotter.plot(ds, "obs_o3", "model_o3")
        n_plot = len(fig_plot.axes)
        plt.close(fig_plot)
        fig_render = plotter.render(build_series(ds, "obs_o3", "model_o3"))
        n_render = len(fig_render.axes)
        plt.close(fig_render)
        assert n_plot == n_render

    def test_render_wrong_series_count_raises(self) -> None:
        from davinci_monet.plots.renderers.spatial.bias import SpatialBiasPlotter

        ds = _spatial_point_ds()
        plotter = SpatialBiasPlotter()
        with pytest.raises(NotImplementedError, match="SpatialBiasPlotter"):
            plotter.render(build_series(ds, "obs_o3"))


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
        fig_plot = plotter.plot(ds, "obs_o3", "model_o3")
        plt.close(fig_plot)
        fig_render = plotter.render(build_series(ds, "obs_o3", "model_o3"))
        plt.close(fig_render)
        assert isinstance(fig_plot, matplotlib.figure.Figure)
        assert isinstance(fig_render, matplotlib.figure.Figure)

    def test_plot_and_render_same_axes_count(self) -> None:
        from davinci_monet.plots.renderers.spatial.distribution import (
            SpatialDistributionPlotter,
        )

        ds = _spatial_point_ds()
        plotter = SpatialDistributionPlotter()
        fig_plot = plotter.plot(ds, "obs_o3", "model_o3")
        n_plot = len(fig_plot.axes)
        plt.close(fig_plot)
        fig_render = plotter.render(build_series(ds, "obs_o3", "model_o3"))
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
        for show_var in ("obs", "model", "both"):
            fig = plotter.render(build_series(ds, "obs_o3", "model_o3"), show_var=show_var)
            assert isinstance(fig, matplotlib.figure.Figure)
            plt.close(fig)

    def test_render_wrong_series_count_raises(self) -> None:
        from davinci_monet.plots.renderers.spatial.distribution import (
            SpatialDistributionPlotter,
        )

        ds = _spatial_point_ds()
        plotter = SpatialDistributionPlotter()
        with pytest.raises(NotImplementedError, match="SpatialDistributionPlotter"):
            plotter.render(build_series(ds, "obs_o3"))


# ---------------------------------------------------------------------------
# SiteTimeSeriesPlotter
# ---------------------------------------------------------------------------


def _site_ds(n_times: int = 50, n_sites: int = 3, seed: int = 0) -> xr.Dataset:
    """Minimal site-based paired dataset: (site, time) dims with coords."""
    rng = np.random.default_rng(seed)
    times = np.datetime64("2024-02-01") + np.arange(n_times) * np.timedelta64(1, "h")
    sites = [f"site_{i}" for i in range(n_sites)]
    obs = rng.uniform(20, 60, (n_sites, n_times))
    mod = obs + rng.uniform(-5, 5, (n_sites, n_times))
    ds = xr.Dataset(
        {
            "obs_o3": (
                ("site", "time"),
                obs,
                {"role": "obs", "pair_role": "reference", "units": "ppb"},
            ),
            "model_o3": (
                ("site", "time"),
                mod,
                {"role": "model", "pair_role": "comparand", "units": "ppb"},
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
        fig_plot = plotter.plot(ds, "obs_o3", "model_o3", ncols=2, min_points=5)
        plt.close(fig_plot)
        fig_render = plotter.render(build_series(ds, "obs_o3", "model_o3"), ncols=2, min_points=5)
        plt.close(fig_render)
        assert isinstance(fig_plot, matplotlib.figure.Figure)
        assert isinstance(fig_render, matplotlib.figure.Figure)

    def test_plot_and_render_same_axes_count(self) -> None:
        from davinci_monet.plots.renderers.site_timeseries import SiteTimeSeriesPlotter

        ds = _site_ds()
        plotter = SiteTimeSeriesPlotter()
        fig_plot = plotter.plot(ds, "obs_o3", "model_o3", ncols=2, min_points=5)
        n_plot = len(fig_plot.axes)
        plt.close(fig_plot)
        fig_render = plotter.render(build_series(ds, "obs_o3", "model_o3"), ncols=2, min_points=5)
        n_render = len(fig_render.axes)
        plt.close(fig_render)
        assert n_plot == n_render

    def test_render_wrong_series_count_raises(self) -> None:
        from davinci_monet.plots.renderers.site_timeseries import SiteTimeSeriesPlotter

        ds = _site_ds()
        plotter = SiteTimeSeriesPlotter()
        with pytest.raises(NotImplementedError, match="SiteTimeSeriesPlotter"):
            plotter.render(build_series(ds, "obs_o3"))


# ---------------------------------------------------------------------------
# FlightTimeSeriesPlotter
# ---------------------------------------------------------------------------


def _flight_ds(n_per_flight: int = 30, n_flights: int = 2, seed: int = 0) -> xr.Dataset:
    """Minimal flight-based paired dataset with a 'flight' coordinate."""
    rng = np.random.default_rng(seed)
    all_times = []
    all_obs = []
    all_mod = []
    all_flight = []
    for day in range(n_flights):
        base = np.datetime64(f"2024-02-0{day + 1}T10:00")
        times = base + np.arange(n_per_flight) * np.timedelta64(1, "m")
        obs = rng.uniform(20, 60, n_per_flight)
        mod = obs + rng.uniform(-5, 5, n_per_flight)
        all_times.append(times)
        all_obs.append(obs)
        all_mod.append(mod)
        all_flight.extend([f"2024-02-0{day + 1}"] * n_per_flight)
    all_times_arr = np.concatenate(all_times)
    all_obs_arr = np.concatenate(all_obs)
    all_mod_arr = np.concatenate(all_mod)
    ds = xr.Dataset(
        {
            "obs_o3": (
                "time",
                all_obs_arr,
                {"role": "obs", "pair_role": "reference", "units": "ppb"},
            ),
            "model_o3": (
                "time",
                all_mod_arr,
                {"role": "model", "pair_role": "comparand", "units": "ppb"},
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
        fig_plot = plotter.plot(ds, "obs_o3", "model_o3", ncols=2, min_points=5)
        plt.close(fig_plot)
        fig_render = plotter.render(build_series(ds, "obs_o3", "model_o3"), ncols=2, min_points=5)
        plt.close(fig_render)
        assert isinstance(fig_plot, matplotlib.figure.Figure)
        assert isinstance(fig_render, matplotlib.figure.Figure)

    def test_plot_and_render_same_axes_count(self) -> None:
        from davinci_monet.plots.renderers.flight_timeseries import FlightTimeSeriesPlotter

        ds = _flight_ds()
        plotter = FlightTimeSeriesPlotter()
        fig_plot = plotter.plot(
            ds, "obs_o3", "model_o3", ncols=2, min_points=5, show_altitude=False
        )
        n_plot = len(fig_plot.axes)
        plt.close(fig_plot)
        fig_render = plotter.render(
            build_series(ds, "obs_o3", "model_o3"), ncols=2, min_points=5, show_altitude=False
        )
        n_render = len(fig_render.axes)
        plt.close(fig_render)
        assert n_plot == n_render

    def test_plot_per_flight_generator_unchanged(self) -> None:
        """plot_per_flight must still yield (flight_id, Figure) tuples."""
        from davinci_monet.plots.renderers.flight_timeseries import FlightTimeSeriesPlotter

        ds = _flight_ds()
        plotter = FlightTimeSeriesPlotter()
        results = list(plotter.plot_per_flight(ds, "obs_o3", "model_o3", min_points=5))
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
            plotter.render(build_series(ds, "obs_o3"))


# ---------------------------------------------------------------------------
# PerSiteTimeSeriesPlotter (single-site plot() vs render())
# ---------------------------------------------------------------------------


class TestPerSiteTimeSeriesRenderParity:
    def test_plot_and_render_both_return_figure(self) -> None:
        from davinci_monet.plots.renderers.per_site_timeseries import PerSiteTimeSeriesPlotter

        ds = _site_ds()
        plotter = PerSiteTimeSeriesPlotter()
        fig_plot = plotter.plot(ds, "obs_o3", "model_o3", min_points=5)
        plt.close(fig_plot)
        fig_render = plotter.render(build_series(ds, "obs_o3", "model_o3"), min_points=5)
        plt.close(fig_render)
        assert isinstance(fig_plot, matplotlib.figure.Figure)
        assert isinstance(fig_render, matplotlib.figure.Figure)

    def test_plot_and_render_same_axes_count(self) -> None:
        from davinci_monet.plots.renderers.per_site_timeseries import PerSiteTimeSeriesPlotter

        ds = _site_ds()
        plotter = PerSiteTimeSeriesPlotter()
        fig_plot = plotter.plot(ds, "obs_o3", "model_o3", site="site_0", min_points=5)
        n_plot = len(fig_plot.axes)
        plt.close(fig_plot)
        fig_render = plotter.render(
            build_series(ds, "obs_o3", "model_o3"), site="site_0", min_points=5
        )
        n_render = len(fig_render.axes)
        plt.close(fig_render)
        assert n_plot == n_render

    def test_plot_per_site_generator_unchanged(self) -> None:
        """plot_per_site must still yield (site_id, Figure) tuples."""
        from davinci_monet.plots.renderers.per_site_timeseries import PerSiteTimeSeriesPlotter

        ds = _site_ds()
        plotter = PerSiteTimeSeriesPlotter()
        results = list(plotter.plot_per_site(ds, "obs_o3", "model_o3", min_points=5))
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
            plotter.render(build_series(ds, "obs_o3"))
