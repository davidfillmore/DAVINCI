"""The public plotting package exposes canonical plotters, not legacy wrappers."""

from __future__ import annotations

import numpy as np
import xarray as xr

import davinci_monet.plots as plots
import davinci_monet.plots.renderers as renderers
import davinci_monet.plots.renderers.spatial as spatial_renderers
from davinci_monet.plots.registry import get_plotter_class, list_plotters


def test_every_registered_plotter_class_is_exported_from_plots() -> None:
    exported = set(getattr(plots, "__all__", []))
    for plot_type in list_plotters():
        cls = get_plotter_class(plot_type)
        assert getattr(plots, cls.__name__) is cls
        assert cls.__name__ in exported


def test_build_series_is_exported_from_plots() -> None:
    assert hasattr(plots, "build_series")
    assert "build_series" in getattr(plots, "__all__", [])


def test_legacy_plot_wrappers_are_not_exported_from_plots() -> None:
    exported = set(getattr(plots, "__all__", []))
    assert not {name for name in exported if name.startswith("plot_")}
    assert not {name for name in dir(plots) if name.startswith("plot_")}


def test_legacy_plot_wrappers_are_not_exported_from_renderer_packages() -> None:
    for module in (renderers, spatial_renderers):
        exported = set(getattr(module, "__all__", []))
        assert not {name for name in exported if name.startswith("plot_")}
        assert not {name for name in dir(module) if name.startswith("plot_")}


def test_registered_plotters_do_not_expose_public_plot_method() -> None:
    for plot_type in list_plotters():
        cls = get_plotter_class(plot_type)
        assert "plot" not in cls.__dict__, f"{cls.__name__} exposes legacy .plot()"


def test_taylor_multi_series_uses_canonical_render_contract() -> None:
    ds = xr.Dataset(
        {
            "obs_o3": ("time", np.array([1.0, 2.0, 3.0, 4.0])),
            "model_a_o3": ("time", np.array([1.1, 2.1, 2.9, 4.2])),
            "model_b_o3": ("time", np.array([0.8, 1.9, 3.2, 3.8])),
        },
        coords={"time": np.arange(4)},
    )
    ds["obs_o3"].attrs.update(axis="x", source_label="obs", canonical_name="o3")
    ds["model_a_o3"].attrs.update(axis="y", source_label="model_a", canonical_name="o3")
    ds["model_b_o3"].attrs.update(axis="y", source_label="model_b", canonical_name="o3")

    plotter = plots.TaylorPlotter()
    fig = plotter.render(plots.build_series(ds, ["obs_o3", "model_a_o3", "model_b_o3"]))

    labels = {line.get_label() for line in fig.axes[0].lines}
    # Raw source keys must be routed through labeling.source_display_name:
    # "obs" -> "Obs", "model_a" -> "Model A", "model_b" -> "Model B".
    assert {"Obs", "Model A", "Model B"}.issubset(labels)
    # Raw config keys must never appear in the legend.
    assert not {"obs", "model_a", "model_b"} & labels
