"""P3 — obs_histogram promoted to the canonical ``histogram`` renderer."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from davinci_monet.core.base import PlotSeries
from davinci_monet.plots.renderers.histogram import HistogramPlotter


def _series(label: str, index: int = 0, role: str = "obs") -> PlotSeries:
    rng = np.random.default_rng(index)
    ds = xr.Dataset(
        {"o3": (("time", "site"), rng.uniform(10, 60, (20, 4)), {"units": "ppb"})},
        coords={"time": np.arange(20), "site": np.arange(4)},
    )
    ds["o3"].attrs["role"] = role
    ds["o3"].attrs["source_label"] = label
    return PlotSeries(ds, "o3", "o3", role, "reference", label, index)


class TestHistogramRender:
    def test_single_source_draws_histogram_with_median(self) -> None:
        fig = HistogramPlotter().render([_series("airnow")])
        ax = fig.axes[0]
        assert len(ax.patches) > 0  # histogram bars
        assert len(ax.get_lines()) >= 1  # median axvline
        plt.close(fig)

    def test_multi_source_overlay_has_legend(self) -> None:
        fig = HistogramPlotter().render([_series("airnow", 0), _series("pandora", 1)])
        ax = fig.axes[0]
        _, labels = ax.get_legend_handles_labels()
        assert {"airnow", "pandora"}.issubset(set(labels))
        plt.close(fig)

    def test_obs_histogram_alias(self) -> None:
        import warnings

        from davinci_monet.plots.registry import get_plotter_class

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            assert get_plotter_class("obs_histogram") is HistogramPlotter
