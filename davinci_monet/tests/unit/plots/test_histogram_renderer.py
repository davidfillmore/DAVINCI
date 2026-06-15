"""Histogram renderer tests."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr

from davinci_monet.core.base import PlotSeries
from davinci_monet.plots.renderers.histogram import HistogramPlotter


def _series(label: str, index: int = 0, axis: str = "x") -> PlotSeries:
    rng = np.random.default_rng(index)
    ds = xr.Dataset(
        {"o3": (("time", "site"), rng.uniform(10, 60, (20, 4)), {"units": "ppb"})},
        coords={"time": np.arange(20), "site": np.arange(4)},
    )
    ds["o3"].attrs["axis"] = axis
    ds["o3"].attrs["source_label"] = label
    return PlotSeries(ds, "o3", "o3", axis, label, index)


class TestHistogramRender:
    def test_single_source_draws_histogram_with_median(self) -> None:
        fig = HistogramPlotter().render([_series("airnow")])
        ax = fig.axes[0]
        assert len(ax.patches) > 0  # histogram bars
        assert len(ax.get_lines()) >= 1  # median axvline
        plt.close(fig)

    def test_rejects_multiple_series(self) -> None:
        with pytest.raises(NotImplementedError, match="requires exactly 1 series"):
            HistogramPlotter().render([_series("airnow", 0), _series("pandora", 1)])

    def test_geometry_named_alias_is_rejected(self) -> None:
        from davinci_monet.plots.registry import get_plotter_class

        with pytest.raises(Exception):
            get_plotter_class("geometry_histogram")
