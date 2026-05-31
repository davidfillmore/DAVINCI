"""Phase 5 tests: role-based color resolver, source-label variable resolver,
and obs_timeseries site-aggregation.

Phase 5 is additive: new plotting helpers and an opt-in aggregation mode are
added alongside the existing prefix-based color/variable logic and the current
obs_timeseries behavior, which are left unchanged so the suite stays green.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np
import xarray as xr

from davinci_monet.plots.base import resolve_source_variable
from davinci_monet.plots.renderers.obs.obs_timeseries import ObsTimeSeriesPlotter
from davinci_monet.plots.style import (
    MODEL_COLOR,
    NCAR_PALETTE,
    OBS_COLOR,
    get_color_for_role,
)


class TestGetColorForRole:
    def test_obs_role_is_obs_color(self) -> None:
        assert get_color_for_role("obs") == OBS_COLOR

    def test_model_role_is_model_color(self) -> None:
        assert get_color_for_role("model") == MODEL_COLOR

    def test_roleless_cycles_palette_by_index(self) -> None:
        assert get_color_for_role(None, index=0) == NCAR_PALETTE[0]
        assert get_color_for_role(None, index=1) == NCAR_PALETTE[1]

    def test_index_wraps_around_palette(self) -> None:
        assert get_color_for_role("", index=len(NCAR_PALETTE)) == NCAR_PALETTE[0]


class TestResolveSourceVariable:
    def test_prefers_source_label_prefixed_name(self) -> None:
        ds = xr.Dataset({"cam_o3": ("x", [1.0]), "o3": ("x", [2.0])})
        assert resolve_source_variable(ds, "o3", "cam") == "cam_o3"

    def test_falls_back_to_canonical_name(self) -> None:
        ds = xr.Dataset({"o3": ("x", [2.0])})
        assert resolve_source_variable(ds, "o3", "airnow") == "o3"

    def test_returns_none_when_absent(self) -> None:
        ds = xr.Dataset({"pm25": ("x", [2.0])})
        assert resolve_source_variable(ds, "o3", "airnow") is None


def _point_obs(n_t: int = 12, n_s: int = 5) -> xr.Dataset:
    rng = np.random.default_rng(0)
    times = np.datetime64("2024-02-01") + np.arange(n_t) * np.timedelta64(1, "h")
    return xr.Dataset(
        {"o3": (("time", "site"), rng.uniform(10, 60, (n_t, n_s)), {"units": "ppb"})},
        coords={"time": times, "site": np.arange(n_s)},
    )


class TestObsTimeseriesAggregation:
    def test_default_plots_one_line_per_site(self) -> None:
        ds = _point_obs(n_s=5)
        fig = ObsTimeSeriesPlotter().plot(ds, "o3")
        ax = fig.axes[0]
        # Current behavior preserved: one line per site.
        assert len(ax.get_lines()) == 5

    def test_aggregate_plots_single_mean_line(self) -> None:
        ds = _point_obs(n_s=5)
        fig = ObsTimeSeriesPlotter().plot(ds, "o3", aggregate=True)
        ax = fig.axes[0]
        assert len(ax.get_lines()) == 1

    def test_aggregate_with_uncertainty_adds_band(self) -> None:
        ds = _point_obs(n_s=5)
        fig = ObsTimeSeriesPlotter().plot(ds, "o3", aggregate=True, show_uncertainty=True)
        ax = fig.axes[0]
        assert len(ax.get_lines()) == 1
        # The +/- sigma band is a filled PolyCollection.
        assert len(ax.collections) >= 1
