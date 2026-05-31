"""Tests for source-label naming on obs-only/single-source renderers (R-4).

R-4: obs-only plots draw a single source. They keep the obs-only brand color
(NCAR blue, *not* the paired-obs gray) but now label their primary series by the
dataset's ``source_label`` so the plot self-identifies its source, consistent
with the source-label model used by the paired renderers (R-3).
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from davinci_monet.plots.base import dataset_source_label
from davinci_monet.plots.style import NCAR_PRIMARY


def _obs_timeseries_ds(source_label: str | None = None) -> xr.Dataset:
    rng = np.random.default_rng(0)
    n = 24
    ds = xr.Dataset(
        {"O3": ("time", rng.uniform(20, 60, n))},
        coords={"time": np.arange(n)},
    )
    if source_label is not None:
        ds.attrs["source_label"] = source_label
    return ds


class TestDatasetSourceLabel:
    def test_returns_source_label_attr(self) -> None:
        ds = _obs_timeseries_ds(source_label="pandora")
        assert dataset_source_label(ds) == "pandora"

    def test_returns_none_when_absent(self) -> None:
        ds = _obs_timeseries_ds()
        assert dataset_source_label(ds) is None


class TestObsTimeseriesSourceLabel:
    def test_single_series_labelled_by_source(self) -> None:
        from davinci_monet.plots.renderers.obs.obs_timeseries import ObsTimeSeriesPlotter

        ds = _obs_timeseries_ds(source_label="pandora")
        fig = ObsTimeSeriesPlotter().plot(ds, "O3")
        line = fig.axes[0].get_lines()[0]
        assert line.get_label() == "pandora"

    def test_single_series_keeps_obs_only_blue(self) -> None:
        # Obs-only convention: NCAR blue, NOT the paired-obs gray.
        from davinci_monet.plots.renderers.obs.obs_timeseries import ObsTimeSeriesPlotter

        ds = _obs_timeseries_ds(source_label="pandora")
        fig = ObsTimeSeriesPlotter().plot(ds, "O3")
        assert fig.axes[0].get_lines()[0].get_color() == NCAR_PRIMARY


class TestObsHistogramSourceLabel:
    def test_bars_labelled_by_source(self) -> None:
        from davinci_monet.plots.renderers.obs.obs_histogram import ObsHistogramPlotter

        ds = _obs_timeseries_ds(source_label="pandora")
        fig = ObsHistogramPlotter().plot(ds, "O3")
        _, labels = fig.axes[0].get_legend_handles_labels()
        assert "pandora" in labels


class TestVerticalProfileSourceLabel:
    def test_single_series_labelled_by_source(self) -> None:
        from davinci_monet.plots.renderers.obs.vertical_profile import VerticalProfilePlotter

        rng = np.random.default_rng(1)
        n = 50
        ds = xr.Dataset(
            {"O3": ("time", rng.uniform(20, 60, n))},
            coords={
                "time": np.arange(n),
                "altitude": ("time", rng.uniform(0, 10000, n)),
            },
        )
        ds.attrs["source_label"] = "dc8"
        fig = VerticalProfilePlotter().plot(ds, "O3")  # default scatter mode
        _, labels = fig.axes[0].get_legend_handles_labels()
        assert "dc8" in labels
