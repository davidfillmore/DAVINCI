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
