"""P3 of the renderer unification — timeseries merged onto render(series).

The canonical TimeSeriesPlotter gains a render() that handles 1/2/N source
series: 1 → single aggregated line (the spaghetti fix), 2 → obs-vs-model
(delegates to the legacy paired plot), N → overlay. ``obs_timeseries`` becomes a
deprecated alias of ``timeseries``. The unified PlottingStage routes obs-only
specs through render() for migrated renderers.
"""

from __future__ import annotations

from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from davinci_monet.core.base import PlotSeries
from davinci_monet.plots.renderers.timeseries import TimeSeriesPlotter
from davinci_monet.plots.style import NCAR_PRIMARY


def _multisite_series(n_t: int = 12, n_s: int = 6, source_label: str = "airnow") -> PlotSeries:
    rng = np.random.default_rng(0)
    times = np.datetime64("2024-02-01") + np.arange(n_t) * np.timedelta64(1, "h")
    ds = xr.Dataset(
        {"o3": (("time", "site"), rng.uniform(10, 60, (n_t, n_s)), {"units": "ppb"})},
        coords={"time": times, "site": np.arange(n_s)},
    )
    ds["o3"].attrs["role"] = "obs"
    ds["o3"].attrs["source_label"] = source_label
    return PlotSeries(ds, "o3", "o3", "obs", "reference", source_label, 0)


class TestTimeseriesRenderSingleSource:
    def test_single_multisite_aggregates_to_one_line(self) -> None:
        fig = TimeSeriesPlotter().render([_multisite_series(n_s=6)])
        ax = fig.axes[0]
        assert len(ax.get_lines()) == 1  # mean over sites, not 6 spaghetti lines
        plt.close(fig)

    def test_single_source_uses_brand_blue(self) -> None:
        fig = TimeSeriesPlotter().render([_multisite_series()])
        assert fig.axes[0].get_lines()[0].get_color() == NCAR_PRIMARY
        plt.close(fig)

    def test_single_source_labelled_by_source(self) -> None:
        fig = TimeSeriesPlotter().render([_multisite_series(source_label="pandora")])
        assert fig.axes[0].get_lines()[0].get_label() == "pandora"
        plt.close(fig)

    def test_show_individual_sites_opt_in(self) -> None:
        fig = TimeSeriesPlotter().render([_multisite_series(n_s=6)], show_individual_sites=True)
        # One line per site when explicitly requested.
        assert len(fig.axes[0].get_lines()) == 6
        plt.close(fig)

    def test_show_uncertainty_adds_band(self) -> None:
        fig = TimeSeriesPlotter().render([_multisite_series(n_s=6)], show_uncertainty=True)
        ax = fig.axes[0]
        assert len(ax.get_lines()) == 1
        assert len(ax.collections) >= 1  # +/-1 sigma PolyCollection
        plt.close(fig)


class TestTimeseriesRenderPaired:
    def test_two_series_delegates_to_paired_plot(self) -> None:
        rng = np.random.default_rng(1)
        t = np.datetime64("2024-02-01") + np.arange(10) * np.timedelta64(1, "h")
        ds = xr.Dataset(
            {
                "airnow_o3": (
                    "time",
                    rng.uniform(10, 60, 10),
                    {"role": "obs", "pair_role": "reference", "source_label": "airnow"},
                ),
                "cam_o3": (
                    "time",
                    rng.uniform(10, 60, 10),
                    {"role": "model", "pair_role": "comparand", "source_label": "cam"},
                ),
            },
            coords={"time": t},
        )
        ref = PlotSeries(ds, "airnow_o3", "o3", "obs", "reference", "airnow", 0)
        comp = PlotSeries(ds, "cam_o3", "o3", "model", "comparand", "cam", 1)
        fig = TimeSeriesPlotter().render([ref, comp])
        # Two series (obs + model) on the axes.
        assert len(fig.axes[0].get_lines()) == 2
        plt.close(fig)


class TestObsTimeseriesAlias:
    def test_alias_resolves_to_canonical(self) -> None:
        from davinci_monet.plots.registry import get_plotter_class

        assert get_plotter_class("obs_timeseries") is TimeSeriesPlotter


class TestUnifiedStageRoutesTimeseriesThroughRender:
    def test_obs_timeseries_spec_renders_single_line(self, tmp_path: Any) -> None:
        from davinci_monet.core.protocols import DataGeometry
        from davinci_monet.pipeline.stages import (
            PipelineContext,
            PlottingStage,
            SourceData,
            StageStatus,
        )

        rng = np.random.default_rng(0)
        times = np.datetime64("2024-02-01") + np.arange(12) * np.timedelta64(1, "h")
        ds = xr.Dataset(
            {"o3": (("time", "site"), rng.uniform(10, 60, (12, 6)), {"units": "ppb"})},
            coords={"time": times, "site": np.arange(6)},
        )
        obs = SourceData(
            data=ds,
            label="airnow",
            source_type="pt_sfc",
            geometry=DataGeometry.POINT,
            role="obs",
        )
        ctx = PipelineContext(
            config={
                "analysis": {"output_dir": str(tmp_path / "out")},
                "plots": {
                    "o3_ts": {
                        "type": "obs_timeseries",
                        "obs": "airnow",
                        "variable": "o3",
                        "title": "O3",
                    }
                },
            },
            observations={"airnow": obs},
        )
        res = PlottingStage().execute(ctx)
        assert res.status == StageStatus.COMPLETED
        assert any("o3_ts" in p.name for p in (tmp_path / "out").glob("*.png"))
