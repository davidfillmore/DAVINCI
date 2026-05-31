"""Tests for role-based series color + source-label legends (renderer rewire R-3).

R-3: separate-series renderers (timeseries, diurnal, boxplot, taylor, and the
per-site/site/flight timeseries) color each series by its source ``role`` and
label it by ``source_label`` rather than the hard-coded obs/model color and the
"Observed"/"Modeled" text. The standard obs-vs-model case stays visually the
same (obs gray, model blue); same-role / role-less series cycle the palette.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from davinci_monet.pipeline.stages import tag_paired_roles
from davinci_monet.plots.base import get_role_color, get_series_label
from davinci_monet.plots.style import MODEL_COLOR, NCAR_PALETTE, OBS_COLOR


def _paired_with_aliases() -> xr.Dataset:
    rng = np.random.default_rng(0)
    n = 8
    ds = xr.Dataset(
        {
            "model_o3": ("time", rng.uniform(10, 60, n)),
            "obs_o3": ("time", rng.uniform(10, 60, n)),
        },
        coords={"time": np.arange(n)},
    )
    tag_paired_roles(ds, reference_label="airnow", comparand_label="cam")
    return ds


class TestGetRoleColor:
    def test_obs_role_is_obs_color(self) -> None:
        ds = _paired_with_aliases()
        assert get_role_color(ds, "obs_o3", index=0) == OBS_COLOR
        assert get_role_color(ds, "airnow_o3", index=0) == OBS_COLOR

    def test_model_role_is_model_color(self) -> None:
        ds = _paired_with_aliases()
        assert get_role_color(ds, "model_o3", index=1) == MODEL_COLOR
        assert get_role_color(ds, "cam_o3", index=1) == MODEL_COLOR

    def test_roleless_series_cycle_palette_by_index(self) -> None:
        ds = xr.Dataset(
            {"a_o3": ("time", np.zeros(3)), "b_o3": ("time", np.zeros(3))},
            coords={"time": np.arange(3)},
        )
        c0 = get_role_color(ds, "a_o3", index=0)
        c1 = get_role_color(ds, "b_o3", index=1)
        assert c0 == NCAR_PALETTE[0]
        assert c1 == NCAR_PALETTE[1]
        assert c0 != c1  # two same-role/role-less series get distinct colors

    def test_missing_var_is_roleless(self) -> None:
        ds = _paired_with_aliases()
        assert get_role_color(ds, "not_present", index=2) == NCAR_PALETTE[2 % len(NCAR_PALETTE)]

    def test_infers_role_from_legacy_prefix_without_attrs(self) -> None:
        # Direct callers (tests, examples, user scripts) pass obs_/model_ vars
        # that carry no role attr; they must still map to obs gray / model blue,
        # not the palette.
        ds = xr.Dataset(
            {"obs_o3": ("time", np.zeros(3)), "model_o3": ("time", np.zeros(3))},
            coords={"time": np.arange(3)},
        )
        assert get_role_color(ds, "obs_o3", index=0) == OBS_COLOR
        assert get_role_color(ds, "model_o3", index=1) == MODEL_COLOR

    def test_style_overrides_honoured_for_obs_model_roles(self) -> None:
        # A customised StyleConfig color is used for obs/model roles.
        ds = _paired_with_aliases()
        assert get_role_color(ds, "obs_o3", obs_color="#111111") == "#111111"
        assert get_role_color(ds, "model_o3", index=1, model_color="#222222") == "#222222"
        # Role-less series ignore the obs/model overrides and cycle the palette.
        roleless = xr.Dataset({"x_o3": ("time", np.zeros(2))}, coords={"time": np.arange(2)})
        assert get_role_color(roleless, "x_o3", index=1, obs_color="#111111") == NCAR_PALETTE[1]


class TestGetSeriesLabel:
    def test_custom_label_wins(self) -> None:
        ds = _paired_with_aliases()
        assert get_series_label(ds, "obs_o3", custom_label="My Obs") == "My Obs"

    def test_source_label_used_when_no_custom(self) -> None:
        ds = _paired_with_aliases()
        assert get_series_label(ds, "obs_o3") == "airnow"
        assert get_series_label(ds, "model_o3") == "cam"
        assert get_series_label(ds, "airnow_o3") == "airnow"
        assert get_series_label(ds, "cam_o3") == "cam"

    def test_falls_back_to_variable_label_without_source_label(self) -> None:
        # A var without role/source_label attrs falls back to the standard label.
        ds = xr.Dataset({"obs_o3": ("time", np.zeros(3))}, coords={"time": np.arange(3)})
        from davinci_monet.plots.base import get_variable_label

        assert get_series_label(ds, "obs_o3") == get_variable_label(ds, "obs_o3")


def _ts_paired() -> xr.Dataset:
    rng = np.random.default_rng(1)
    n = 24
    ds = xr.Dataset(
        {
            "model_o3": ("time", rng.uniform(20, 60, n)),
            "obs_o3": ("time", rng.uniform(20, 60, n)),
        },
        coords={"time": np.arange(n)},
    )
    tag_paired_roles(ds, reference_label="airnow", comparand_label="cam")
    return ds


class TestTimeseriesRoleStyling:
    """Renderer-level: the standard obs-vs-model timeseries keeps gray/blue and
    labels each series by its source (R-3)."""

    def _line_colors_by_label(self, fig: object) -> dict:
        ax = fig.axes[0]  # type: ignore[attr-defined]
        return {ln.get_label(): ln.get_color() for ln in ax.get_lines()}

    def test_obs_gray_model_blue_unchanged(self) -> None:
        from davinci_monet.plots.renderers.timeseries import TimeSeriesPlotter

        fig = TimeSeriesPlotter().plot(_ts_paired(), "obs_o3", "model_o3")
        colors = self._line_colors_by_label(fig)
        assert colors["airnow"] == OBS_COLOR
        assert colors["cam"] == MODEL_COLOR

    def test_legend_uses_source_labels(self) -> None:
        from davinci_monet.plots.renderers.timeseries import TimeSeriesPlotter

        fig = TimeSeriesPlotter().plot(_ts_paired(), "obs_o3", "model_o3")
        labels = {ln.get_label() for ln in fig.axes[0].get_lines()}
        assert "airnow" in labels
        assert "cam" in labels
        assert "Observations" not in labels
        assert "Model" not in labels

    def test_two_roleless_sources_get_distinct_palette_colors(self) -> None:
        # A model-vs-model style pair (both role-less here) yields two distinct
        # palette colors rather than a single shared color.
        from davinci_monet.plots.renderers.timeseries import TimeSeriesPlotter

        rng = np.random.default_rng(2)
        ds = xr.Dataset(
            {
                "wrf_o3": ("time", rng.uniform(20, 60, 12)),
                "cam_o3": ("time", rng.uniform(20, 60, 12)),
            },
            coords={"time": np.arange(12)},
        )
        for name, label in (("wrf_o3", "wrf"), ("cam_o3", "cam")):
            ds[name].attrs["source_label"] = label  # role-less (no obs/model)
        fig = TimeSeriesPlotter().plot(ds, "wrf_o3", "cam_o3")
        colors = self._line_colors_by_label(fig)
        assert colors["wrf"] == NCAR_PALETTE[0]
        assert colors["cam"] == NCAR_PALETTE[1]
        assert colors["wrf"] != colors["cam"]


class TestTaylorRoleStyling:
    def test_reference_label_respects_config_obs_label(self) -> None:
        # A PlotConfig.obs_label override must win over the source_label for the
        # Taylor reference point (parity with the model label chain).
        from davinci_monet.plots.base import PlotConfig
        from davinci_monet.plots.renderers.taylor import TaylorPlotter

        ds = _paired_with_aliases()  # obs carries source_label "airnow"
        fig = TaylorPlotter(config=PlotConfig(obs_label="CustomRef")).plot(ds, "obs_o3", "model_o3")
        labels = {ln.get_label() for ln in fig.axes[0].get_lines()}
        assert "CustomRef" in labels
        assert "airnow" not in labels
