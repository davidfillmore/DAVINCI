"""Vertical profile renderer tests."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from davinci_monet.core.base import PlotSeries
from davinci_monet.plots.renderers.vertical_profile import VerticalProfilePlotter


def _profile_series(label: str = "dc8", index: int = 0) -> PlotSeries:
    rng = np.random.default_rng(index)
    n = 200
    ds = xr.Dataset(
        {"O3": ("time", rng.uniform(20, 120, n), {"units": "ppbv"})},
        coords={
            "time": np.arange(n),
            "altitude": ("time", rng.uniform(0, 12000, n), {"units": "m"}),
        },
    )
    ds["O3"].attrs["pair_axis"] = "geometry"
    ds["O3"].attrs["dataset_label"] = label
    return PlotSeries(ds, "O3", "O3", "geometry", label, index)


class TestVerticalProfileRender:
    def test_scatter_mode_draws_points(self) -> None:
        fig = VerticalProfilePlotter().render([_profile_series()], mode="scatter")
        assert len(fig.axes[0].collections) >= 1  # scatter PathCollection
        plt.close(fig)

    def test_binned_mode_draws_line(self) -> None:
        fig = VerticalProfilePlotter().render([_profile_series()], mode="binned")
        assert len(fig.axes[0].get_lines()) >= 1
        plt.close(fig)

    def test_multi_source_overlay_legend(self) -> None:
        fig = VerticalProfilePlotter().render(
            [_profile_series("dc8", 0), _profile_series("gv", 1)], mode="binned"
        )
        _, labels = fig.axes[0].get_legend_handles_labels()
        assert {"dc8", "gv"}.issubset(set(labels))
        plt.close(fig)

    def test_geometry_named_alias_is_rejected(self) -> None:
        import pytest

        from davinci_monet.plots.registry import get_plotter_class

        with pytest.raises(Exception):
            get_plotter_class("geometry_vertical_profile")
