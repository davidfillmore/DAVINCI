"""P0 of the renderer unification (docs/superpowers/specs/2026-06-06-renderer-unification-design.md).

Pure-addition foundation — no behavior change to existing renderers:
- PlotSeries value object + iter_canonical_variable_series (N-capable sibling of
  iter_paired_variable_xy), with a guard that the binary read path stays stable.
- build_series: resolve facade var-args (paired / single / N-list / trailing-Axes)
  into a PlotSeries list.
- BasePlotter.render() default delegating to plot() for the 2-series case.
- Registry alternate-name support (register_alias) resolved by get/has.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from davinci_monet.core.base import (
    PlotSeries,
    iter_canonical_variable_series,
    iter_paired_variable_xy,
)


def _paired(*specs: tuple[str, str, str]) -> xr.Dataset:
    """Build a paired-style dataset.

    Each spec is (var_name, axis, dataset_label); values are a 3-long
    time series so the dataset is realistic.
    """
    data = {}
    for name, axis, label in specs:
        canonical_name = name.split("_", 1)[1] if "_" in name else name
        data[name] = xr.DataArray(
            np.array([1.0, 2.0, 3.0]),
            dims=("time",),
            attrs={
                "axis": axis,
                "canonical_name": canonical_name,
                "source_label": label,
            },
        )
    return xr.Dataset(data, coords={"time": np.arange(3)})


# ---------------------------------------------------------------------------
# PlotSeries + iter_canonical_variable_series
# ---------------------------------------------------------------------------


class TestIterCanonicalVariableSeries:
    def test_single_source_one_series(self) -> None:
        ds = _paired(("airnow_o3", "x", "airnow"))
        groups = iter_canonical_variable_series(ds)
        assert list(groups) == ["o3"]
        assert len(groups["o3"]) == 1
        s = groups["o3"][0]
        assert isinstance(s, PlotSeries)
        assert (s.var_name, s.canonical, s.axis, s.source_label, s.index) == (
            "airnow_o3",
            "o3",
            "x",
            "airnow",
            0,
        )

    def test_paired_two_series_grouped_by_canonical(self) -> None:
        ds = _paired(
            ("airnow_o3", "x", "airnow"),
            ("cam_o3", "y", "cam"),
        )
        groups = iter_canonical_variable_series(ds)
        assert list(groups) == ["o3"]
        series = groups["o3"]
        assert [s.axis for s in series] == ["x", "y"]
        assert [s.index for s in series] == [0, 1]

    def test_n_source_same_canonical_grouped(self) -> None:
        ds = _paired(
            ("airnow_o3", "x", "airnow"),
            ("cam_o3", "y", "cam"),
            ("cam2_o3", "y", "cam2"),
        )
        groups = iter_canonical_variable_series(ds)
        assert list(groups) == ["o3"]
        assert [s.source_label for s in groups["o3"]] == ["airnow", "cam", "cam2"]
        assert [s.index for s in groups["o3"]] == [0, 1, 2]

    def test_multiple_canonicals_kept_separate(self) -> None:
        ds = _paired(
            ("airnow_o3", "x", "airnow"),
            ("cam_o3", "y", "cam"),
            ("airnow_pm25", "x", "airnow"),
            ("cam_pm25", "y", "cam"),
        )
        groups = iter_canonical_variable_series(ds)
        assert set(groups) == {"o3", "pm25"}
        assert all(len(v) == 2 for v in groups.values())


class TestIterPairedVariablePairsUnchanged:
    """Guard: the binary read path keeps first-geometry + first-dataset
    selection and dataset-appearance order, even when a third same-canonical
    var would change [0] selection under reordering."""

    def test_first_dataset_wins_with_extra_source(self) -> None:
        ds = _paired(
            ("airnow_o3", "x", "airnow"),
            ("cam_o3", "y", "cam"),
            ("cam2_o3", "y", "cam2"),
        )
        # Only the FIRST dataset (cam_o3) pairs with the geometry.
        assert iter_paired_variable_xy(ds) == [("airnow_o3", "cam_o3", "o3")]

    def test_prefix_order_preserved(self) -> None:
        # geometry_/dataset_ names, dataset variables appearing in b-then-a order.
        ds = xr.Dataset(
            {
                "geometry_a": ("t", [1.0]),
                "geometry_b": ("t", [1.0]),
                "dataset_b": ("t", [1.0]),
                "dataset_a": ("t", [1.0]),
            },
            coords={"t": [0]},
        )
        # Dataset-appearance order is b, then a.
        assert iter_paired_variable_xy(ds) == [
            ("geometry_b", "dataset_b", "b"),
            ("geometry_a", "dataset_a", "a"),
        ]


# ---------------------------------------------------------------------------
# build_series facade resolution
# ---------------------------------------------------------------------------


class TestBuildSeries:
    def _ds(self) -> xr.Dataset:
        return _paired(
            ("airnow_o3", "x", "airnow"),
            ("cam_o3", "y", "cam"),
            ("cam2_o3", "y", "cam2"),
        )

    def test_paired_two_args(self) -> None:
        from davinci_monet.plots.base import build_series

        series = build_series(self._ds(), "airnow_o3", "cam_o3")
        assert [s.var_name for s in series] == ["airnow_o3", "cam_o3"]
        assert [s.index for s in series] == [0, 1]

    def test_single_arg(self) -> None:
        from davinci_monet.plots.base import build_series

        series = build_series(self._ds(), "airnow_o3")
        assert [s.var_name for s in series] == ["airnow_o3"]

    def test_list_of_n(self) -> None:
        from davinci_monet.plots.base import build_series

        series = build_series(self._ds(), ["airnow_o3", "cam_o3", "cam2_o3"])
        assert [s.var_name for s in series] == ["airnow_o3", "cam_o3", "cam2_o3"]
        assert [s.index for s in series] == [0, 1, 2]

    def test_trailing_axes_not_treated_as_variable(self) -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from davinci_monet.plots.base import build_series

        _, ax = plt.subplots()
        series = build_series(self._ds(), "airnow_o3", ax)
        assert [s.var_name for s in series] == ["airnow_o3"]
        plt.close("all")


# ---------------------------------------------------------------------------
# BasePlotter.render() default
# ---------------------------------------------------------------------------


class TestRenderDefault:
    def test_two_series_default_delegates_to_plot(self) -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from davinci_monet.plots.base import BasePlotter, build_series

        calls: list[tuple] = []

        class StubPlotter(BasePlotter):
            name = "stub"

            def plot(self, paired_data, x_var, y_var, ax=None, **kwargs):
                calls.append((x_var, y_var))
                fig, _ax = self.create_figure()
                return fig

        ds = _paired(
            ("airnow_o3", "x", "airnow"),
            ("cam_o3", "y", "cam"),
        )
        series = build_series(ds, "airnow_o3", "cam_o3")
        fig = StubPlotter().render(series)
        assert calls == [("airnow_o3", "cam_o3")]
        assert fig is not None
        plt.close("all")

    def test_render_uses_geometry_series_dataset(self) -> None:
        """render() must pass the geometry dataset (not series[0].dataset) so that
        when axis ordering flips the positional assumption, the correct
        paired dataset is still forwarded to plot()."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from davinci_monet.plots.base import BasePlotter, build_series

        received_datasets: list = []

        class StubPlotter(BasePlotter):
            name = "stub_geometry_ds"

            def plot(self, paired_data, x_var, y_var, ax=None, **kwargs):
                received_datasets.append(id(paired_data))
                fig, _ax = self.create_figure()
                return fig

        ds = _paired(
            ("airnow_o3", "x", "airnow"),
            ("cam_o3", "y", "cam"),
        )
        # Build series dataset-first so series[0] is NOT the geometry.
        series = build_series(ds, "cam_o3", "airnow_o3")
        x_series = next(s for s in series if s.axis == "x")
        StubPlotter().render(series)
        # plot() must receive the geometry dataset, not series[0].dataset.
        assert received_datasets == [id(x_series.dataset)]
        plt.close("all")


# ---------------------------------------------------------------------------
# Registry alias support
# ---------------------------------------------------------------------------


class TestRegistryAliases:
    def test_alias_resolves_to_target(self) -> None:
        from davinci_monet.core.registry import Registry

        reg: Registry[type] = Registry("test")

        class Real:
            pass

        reg.register("real", Real)
        reg.register_alias("shortcut", "real")
        assert reg.get("shortcut") is Real
        assert "shortcut" in reg
        assert reg.is_alias("shortcut")
        assert not reg.is_alias("real")
        # Aliases do not pollute the canonical listing.
        assert reg.list() == ["real"]

    def test_plotter_alias_resolves_to_target(self) -> None:
        from davinci_monet.core.registry import plotter_registry
        from davinci_monet.plots.base import BasePlotter
        from davinci_monet.plots.registry import (
            get_plotter_class,
            has_plotter,
            register_plotter,
        )

        @register_plotter("p0_real_plot")
        class _RealPlot(BasePlotter):
            name = "p0_real_plot"

            def plot(self, paired_data, x_var, y_var, ax=None, **kwargs):
                fig, _ = self.create_figure()
                return fig

        plotter_registry.register_alias("p0_shortcut_plot", "p0_real_plot")
        assert has_plotter("p0_shortcut_plot")
        assert get_plotter_class("p0_shortcut_plot") is _RealPlot
