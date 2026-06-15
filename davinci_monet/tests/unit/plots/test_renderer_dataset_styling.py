"""Tests for pair-axis series color and source-label legends."""

from __future__ import annotations

import numpy as np
import xarray as xr

from davinci_monet.plots.base import build_series, get_axis_color, get_series_label
from davinci_monet.plots.style import NCAR_PALETTE, X_COLOR, Y_COLOR


def _paired_with_aliases() -> xr.Dataset:
    rng = np.random.default_rng(0)
    n = 8
    ds = xr.Dataset(
        {
            "cam_o3": ("time", rng.uniform(10, 60, n)),
            "airnow_o3": ("time", rng.uniform(10, 60, n)),
        },
        coords={"time": np.arange(n)},
    )
    ds["cam_o3"].attrs.update({"axis": "y", "source_label": "cam"})
    ds["airnow_o3"].attrs.update({"axis": "x", "source_label": "airnow"})
    return ds


class TestGetDatasetColor:
    def test_geometry_axis_is_geometry_color(self) -> None:
        ds = _paired_with_aliases()
        assert get_axis_color(ds, "airnow_o3", index=0) == X_COLOR

    def test_dataset_axis_is_dataset_color(self) -> None:
        ds = _paired_with_aliases()
        assert get_axis_color(ds, "cam_o3", index=1) == Y_COLOR

    def test_unpaired_series_cycle_palette_by_index(self) -> None:
        ds = xr.Dataset(
            {"a_o3": ("time", np.zeros(3)), "b_o3": ("time", np.zeros(3))},
            coords={"time": np.arange(3)},
        )
        c0 = get_axis_color(ds, "a_o3", index=0)
        c1 = get_axis_color(ds, "b_o3", index=1)
        assert c0 == NCAR_PALETTE[0]
        assert c1 == NCAR_PALETTE[1]
        assert c0 != c1

    def test_missing_var_uses_palette(self) -> None:
        ds = _paired_with_aliases()
        assert get_axis_color(ds, "not_present", index=2) == NCAR_PALETTE[2 % len(NCAR_PALETTE)]

    def test_infers_axis_from_prefix_without_attrs(self) -> None:
        # Direct callers (tests, examples, user scripts) pass x_/y_ vars
        # that carry no axis attr; they must still map to x gray / y blue,
        # not the palette.
        ds = xr.Dataset(
            {"x_o3": ("time", np.zeros(3)), "y_o3": ("time", np.zeros(3))},
            coords={"time": np.arange(3)},
        )
        assert get_axis_color(ds, "x_o3", index=0) == X_COLOR
        assert get_axis_color(ds, "y_o3", index=1) == Y_COLOR

    def test_style_overrides_honoured_for_xy_axes(self) -> None:
        # A customised StyleConfig color is used for x/y axes.
        ds = _paired_with_aliases()
        assert get_axis_color(ds, "airnow_o3", x_color="#111111") == "#111111"
        assert get_axis_color(ds, "cam_o3", index=1, y_color="#222222") == "#222222"
        # A lone series with no axis prefix/attr ignores the override.
        unpaired = xr.Dataset({"o3": ("time", np.zeros(2))}, coords={"time": np.arange(2)})
        assert get_axis_color(unpaired, "o3", index=1, x_color="#111111") == NCAR_PALETTE[1]


class TestGetSeriesLabel:
    def test_custom_label_wins(self) -> None:
        ds = _paired_with_aliases()
        assert get_series_label(ds, "x_o3", custom_label="My Geometry") == "My Geometry"

    def test_dataset_label_used_when_no_custom(self) -> None:
        # After R-5 the paired vars are renamed to their source labels.
        ds = _paired_with_aliases()
        assert get_series_label(ds, "airnow_o3") == "airnow"
        assert get_series_label(ds, "cam_o3") == "cam"

    def test_falls_back_to_variable_label_without_source_label(self) -> None:
        # A var without axis/source_label attrs falls back to the standard label.
        ds = xr.Dataset({"x_o3": ("time", np.zeros(3))}, coords={"time": np.arange(3)})
        from davinci_monet.plots.base import get_variable_label

        assert get_series_label(ds, "x_o3") == get_variable_label(ds, "x_o3")


def _ts_paired() -> xr.Dataset:
    rng = np.random.default_rng(1)
    n = 24
    ds = xr.Dataset(
        {
            "cam_o3": ("time", rng.uniform(20, 60, n)),
            "airnow_o3": ("time", rng.uniform(20, 60, n)),
        },
        coords={"time": np.arange(n)},
    )
    ds["cam_o3"].attrs.update({"axis": "y", "source_label": "cam"})
    ds["airnow_o3"].attrs.update({"axis": "x", "source_label": "airnow"})
    return ds


class TestTimeseriesAxisStyling:
    """Renderer-level: the standard x-vs-y timeseries keeps gray/blue and
    labels each series by its source (R-3)."""

    def _line_colors_by_label(self, fig: object) -> dict:
        ax = fig.axes[0]  # type: ignore[attr-defined]
        return {ln.get_label(): ln.get_color() for ln in ax.get_lines()}

    def test_geometry_gray_dataset_blue_unchanged(self) -> None:
        from davinci_monet.plots.renderers.timeseries import TimeSeriesPlotter

        ds = _ts_paired()
        fig = TimeSeriesPlotter().render(build_series(ds, "airnow_o3", "cam_o3"))
        colors = self._line_colors_by_label(fig)
        # labeling.legend_label returns friendly display names ("AirNow", "CAM")
        assert colors["AirNow"] == X_COLOR
        assert colors["CAM"] == Y_COLOR

    def test_legend_uses_dataset_labels(self) -> None:
        from davinci_monet.plots.renderers.timeseries import TimeSeriesPlotter

        ds = _ts_paired()
        fig = TimeSeriesPlotter().render(build_series(ds, "airnow_o3", "cam_o3"))
        labels = {ln.get_label() for ln in fig.axes[0].get_lines()}
        # labeling.legend_label now returns friendly display names
        assert "AirNow" in labels
        assert "CAM" in labels
        assert "Datasets" not in labels
        assert "Dataset" not in labels

    def test_two_unpaired_sources_get_distinct_palette_colors(self) -> None:
        # A dataset-vs-dataset style pair (both axis-less here) yields two distinct
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
            ds[name].attrs["source_label"] = label  # axis-less (no geometry/dataset)
        fig = TimeSeriesPlotter().render(build_series(ds, "wrf_o3", "cam_o3"))
        colors = self._line_colors_by_label(fig)
        # labeling.legend_label returns friendly display names ("WRF", "CAM")
        assert colors["WRF"] == NCAR_PALETTE[0]
        assert colors["CAM"] == NCAR_PALETTE[1]
        assert colors["WRF"] != colors["CAM"]


class TestTaylorRoleStyling:
    def test_geometry_label_respects_config_x_label(self) -> None:
        # A PlotConfig.x_label override must win over the source label for the
        # Taylor x-axis point follows the same source-label chain.
        from davinci_monet.plots.base import PlotConfig
        from davinci_monet.plots.renderers.taylor import TaylorPlotter

        ds = _paired_with_aliases()  # geometry renamed to airnow_o3 (source_label "airnow")
        fig = TaylorPlotter(config=PlotConfig(x_label="CustomRef")).render(
            build_series(ds, "airnow_o3", "cam_o3")
        )
        labels = {ln.get_label() for ln in fig.axes[0].get_lines()}
        assert "CustomRef" in labels
        # Raw config keys must never appear — friendly names only.
        assert "airnow" not in labels
        assert "cam" not in labels
        assert "CAM" in labels
