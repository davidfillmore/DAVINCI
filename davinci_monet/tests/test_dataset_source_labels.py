"""Tests for source-label naming on geometry-only/single-source renderers (R-4).

R-4: geometry-only plots draw a single source. They keep the geometry-only brand color
(NCAR blue, *not* the paired-geometry gray) but now label their primary series by the
dataset's ``dataset_label`` so the plot self-identifies its source, consistent
with the source-label dataset used by the paired renderers (R-3).
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from davinci_monet.plots.base import source_label
from davinci_monet.plots.style import NCAR_PRIMARY


def _geometry_timeseries_ds(dataset_label: str | None = None) -> xr.Dataset:
    rng = np.random.default_rng(0)
    n = 24
    ds = xr.Dataset(
        {"O3": ("time", rng.uniform(20, 60, n))},
        coords={"time": np.arange(n)},
    )
    if dataset_label is not None:
        ds.attrs["dataset_label"] = dataset_label
    return ds


class TestDatasetSourceLabel:
    def test_returns_dataset_label_attr(self) -> None:
        ds = _geometry_timeseries_ds(dataset_label="pandora")
        assert source_label(ds) == "pandora"

    def test_returns_none_when_absent(self) -> None:
        ds = _geometry_timeseries_ds()
        assert source_label(ds) is None


class TestGeometryTimeseriesSourceLabel:
    """R-4 now flows through the unified TimeSeriesPlotter.render (single source)."""

    def test_single_series_labelled_by_source(self) -> None:
        from davinci_monet.plots.base import build_series
        from davinci_monet.plots.renderers.timeseries import TimeSeriesPlotter

        ds = _geometry_timeseries_ds(dataset_label="pandora")
        fig = TimeSeriesPlotter().render(build_series(ds, "O3"))
        assert fig.axes[0].get_lines()[0].get_label() == "pandora"

    def test_single_series_keeps_geometry_only_blue(self) -> None:
        # Geometry-only convention: NCAR blue, NOT the paired-geometry gray.
        from davinci_monet.plots.base import build_series
        from davinci_monet.plots.renderers.timeseries import TimeSeriesPlotter

        ds = _geometry_timeseries_ds(dataset_label="pandora")
        fig = TimeSeriesPlotter().render(build_series(ds, "O3"))
        assert fig.axes[0].get_lines()[0].get_color() == NCAR_PRIMARY


class TestGeometryHistogramSourceLabel:
    def test_bars_labelled_by_source(self) -> None:
        from davinci_monet.plots.renderers.histogram import HistogramPlotter

        ds = _geometry_timeseries_ds(dataset_label="pandora")
        fig = HistogramPlotter().plot(ds, "O3")
        _, labels = fig.axes[0].get_legend_handles_labels()
        assert "pandora" in labels


class TestVerticalProfileSourceLabel:
    def test_single_series_labelled_by_source(self) -> None:
        from davinci_monet.plots.renderers.vertical_profile import VerticalProfilePlotter

        rng = np.random.default_rng(1)
        n = 50
        ds = xr.Dataset(
            {"O3": ("time", rng.uniform(20, 60, n))},
            coords={
                "time": np.arange(n),
                "altitude": ("time", rng.uniform(0, 10000, n)),
            },
        )
        ds.attrs["dataset_label"] = "dc8"
        fig = VerticalProfilePlotter().plot(ds, "O3")  # default scatter mode
        _, labels = fig.axes[0].get_legend_handles_labels()
        assert "dc8" in labels
