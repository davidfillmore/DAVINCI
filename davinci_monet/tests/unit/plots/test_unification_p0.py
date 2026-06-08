"""P0 of the renderer unification (docs/superpowers/specs/2026-06-06-renderer-unification-design.md).

Pure-addition foundation — no behavior change to existing renderers:
- PlotSeries value object + iter_canonical_variable_series (N-capable sibling of
  iter_paired_variable_pairs), with a guard that the binary read path is unchanged.
- build_series: resolve facade var-args (paired / single / N-list / trailing-Axes)
  into a PlotSeries list.
- BasePlotter.render() default delegating to legacy plot() for the 2-series case.
- Registry alias support (register_alias) resolved by get/has with a one-time
  LegacyConfigWarning.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import xarray as xr

from davinci_monet.core.base import (
    PlotSeries,
    iter_canonical_variable_series,
    iter_paired_variable_pairs,
)


def _paired(*specs: tuple[str, str, str, str]) -> xr.Dataset:
    """Build a paired-style dataset.

    Each spec is (var_name, role, pair_role, source_label); values are a 3-long
    time series so the dataset is realistic.
    """
    data = {}
    for name, role, pair_role, label in specs:
        data[name] = xr.DataArray(
            np.array([1.0, 2.0, 3.0]),
            dims=("time",),
            attrs={"role": role, "pair_role": pair_role, "source_label": label},
        )
    return xr.Dataset(data, coords={"time": np.arange(3)})


# ---------------------------------------------------------------------------
# PlotSeries + iter_canonical_variable_series
# ---------------------------------------------------------------------------


class TestIterCanonicalVariableSeries:
    def test_single_source_one_series(self) -> None:
        ds = _paired(("airnow_o3", "obs", "reference", "airnow"))
        groups = iter_canonical_variable_series(ds)
        assert list(groups) == ["o3"]
        assert len(groups["o3"]) == 1
        s = groups["o3"][0]
        assert isinstance(s, PlotSeries)
        assert (s.var_name, s.canonical, s.role, s.pair_role, s.source_label, s.index) == (
            "airnow_o3",
            "o3",
            "obs",
            "reference",
            "airnow",
            0,
        )

    def test_paired_two_series_grouped_by_canonical(self) -> None:
        ds = _paired(
            ("airnow_o3", "obs", "reference", "airnow"),
            ("cam_o3", "model", "comparand", "cam"),
        )
        groups = iter_canonical_variable_series(ds)
        assert list(groups) == ["o3"]
        series = groups["o3"]
        assert [s.role for s in series] == ["obs", "model"]
        assert [s.pair_role for s in series] == ["reference", "comparand"]
        assert [s.index for s in series] == [0, 1]

    def test_n_source_same_canonical_grouped(self) -> None:
        ds = _paired(
            ("airnow_o3", "obs", "reference", "airnow"),
            ("cam_o3", "model", "comparand", "cam"),
            ("cam2_o3", "model", "comparand", "cam2"),
        )
        groups = iter_canonical_variable_series(ds)
        assert list(groups) == ["o3"]
        assert [s.source_label for s in groups["o3"]] == ["airnow", "cam", "cam2"]
        assert [s.index for s in groups["o3"]] == [0, 1, 2]

    def test_multiple_canonicals_kept_separate(self) -> None:
        ds = _paired(
            ("airnow_o3", "obs", "reference", "airnow"),
            ("cam_o3", "model", "comparand", "cam"),
            ("airnow_pm25", "obs", "reference", "airnow"),
            ("cam_pm25", "model", "comparand", "cam"),
        )
        groups = iter_canonical_variable_series(ds)
        assert set(groups) == {"o3", "pm25"}
        assert all(len(v) == 2 for v in groups.values())


class TestIterPairedVariablePairsUnchanged:
    """Guard: the binary read path keeps first-reference + first-comparand
    selection and comparand-appearance order, even when a third same-canonical
    var would change [0] selection under reordering."""

    def test_first_comparand_wins_with_extra_source(self) -> None:
        ds = _paired(
            ("airnow_o3", "obs", "reference", "airnow"),
            ("cam_o3", "model", "comparand", "cam"),
            ("cam2_o3", "model", "comparand", "cam2"),
        )
        # Only the FIRST comparand (cam_o3) pairs with the reference.
        assert iter_paired_variable_pairs(ds) == [("airnow_o3", "cam_o3", "o3")]

    def test_legacy_prefix_order_preserved(self) -> None:
        # Legacy obs_/model_ names, comparands appearing in b-then-a order.
        ds = xr.Dataset(
            {
                "obs_a": ("t", [1.0]),
                "obs_b": ("t", [1.0]),
                "model_b": ("t", [1.0]),
                "model_a": ("t", [1.0]),
            },
            coords={"t": [0]},
        )
        # Comparand-appearance order is b, then a.
        assert iter_paired_variable_pairs(ds) == [
            ("obs_b", "model_b", "b"),
            ("obs_a", "model_a", "a"),
        ]


# ---------------------------------------------------------------------------
# build_series facade resolution
# ---------------------------------------------------------------------------


class TestBuildSeries:
    def _ds(self) -> xr.Dataset:
        return _paired(
            ("airnow_o3", "obs", "reference", "airnow"),
            ("cam_o3", "model", "comparand", "cam"),
            ("cam2_o3", "model", "comparand", "cam2"),
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

            def plot(self, paired_data, obs_var, model_var, ax=None, **kwargs):
                calls.append((obs_var, model_var))
                fig, _ax = self.create_figure()
                return fig

        ds = _paired(
            ("airnow_o3", "obs", "reference", "airnow"),
            ("cam_o3", "model", "comparand", "cam"),
        )
        series = build_series(ds, "airnow_o3", "cam_o3")
        fig = StubPlotter().render(series)
        assert calls == [("airnow_o3", "cam_o3")]
        assert fig is not None
        plt.close("all")

    def test_render_uses_reference_series_dataset(self) -> None:
        """render() must pass ref.dataset (not series[0].dataset) so that
        when pair_role ordering flips the positional assumption, the correct
        paired dataset is still forwarded to plot()."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from davinci_monet.plots.base import BasePlotter, build_series

        received_datasets: list = []

        class StubPlotter(BasePlotter):
            name = "stub_ref_ds"

            def plot(self, paired_data, obs_var, model_var, ax=None, **kwargs):
                received_datasets.append(id(paired_data))
                fig, _ax = self.create_figure()
                return fig

        ds = _paired(
            ("airnow_o3", "obs", "reference", "airnow"),
            ("cam_o3", "model", "comparand", "cam"),
        )
        # Build series comparand-first so series[0] is NOT the reference.
        series = build_series(ds, "cam_o3", "airnow_o3")
        ref = next(s for s in series if s.pair_role == "reference")
        StubPlotter().render(series)
        # plot() must receive ref.dataset, not series[0].dataset
        assert received_datasets == [id(ref.dataset)]
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
        reg.register_alias("old", "real")
        assert reg.get("old") is Real
        assert "old" in reg
        assert reg.is_alias("old")
        assert not reg.is_alias("real")
        # Aliases do not pollute the canonical listing.
        assert reg.list() == ["real"]

    def test_plotter_alias_warns_once(self) -> None:
        from davinci_monet.config.migration import LegacyConfigWarning
        from davinci_monet.plots.base import BasePlotter
        from davinci_monet.plots.registry import (
            get_plotter_class,
            has_plotter,
            register_alias,
            register_plotter,
        )

        @register_plotter("p0_real_plot")
        class _RealPlot(BasePlotter):
            name = "p0_real_plot"

            def plot(self, paired_data, obs_var, model_var, ax=None, **kwargs):
                fig, _ = self.create_figure()
                return fig

        register_alias("p0_old_plot", "p0_real_plot")
        assert has_plotter("p0_old_plot")
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            assert get_plotter_class("p0_old_plot") is _RealPlot
        assert any(issubclass(w.category, LegacyConfigWarning) for w in rec)
