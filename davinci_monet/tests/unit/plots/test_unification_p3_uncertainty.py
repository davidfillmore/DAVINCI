"""Single-source timeseries honors uncertainty_type (std/iqr/range).

The paired TimeSeriesPlotter.plot supported std/iqr/range bands; this extends the
single-source render path (_render_single) to honor the same option instead of
always using std.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from davinci_monet.core.base import PlotSeries
from davinci_monet.plots.renderers.timeseries import TimeSeriesPlotter


def _series(n_t: int = 10, n_s: int = 8) -> PlotSeries:
    rng = np.random.default_rng(3)
    times = np.datetime64("2024-02-01") + np.arange(n_t) * np.timedelta64(1, "h")
    ds = xr.Dataset(
        {"o3": (("time", "site"), rng.uniform(10, 60, (n_t, n_s)), {"units": "ppb"})},
        coords={"time": times, "site": np.arange(n_s)},
    )
    ds["o3"].attrs["axis"] = "x"
    ds["o3"].attrs["source_label"] = "airnow"
    return PlotSeries(ds, "o3", "o3", "x", "airnow", 0)


def _band_extents(uncertainty_type: str) -> tuple[float, float]:
    s = _series()
    fig = TimeSeriesPlotter().render([s], show_uncertainty=True, uncertainty_type=uncertainty_type)
    verts = np.asarray(fig.axes[0].collections[0].get_paths()[0].vertices)
    top, bot = float(verts[:, 1].max()), float(verts[:, 1].min())
    plt.close(fig)
    return top, bot


class TestUncertaintyType:
    def test_range_band_reaches_data_extremes(self) -> None:
        s = _series()
        da = s.dataset[s.var_name]
        top, bot = _band_extents("range")
        # The full-range band spans the global data min/max.
        assert np.isclose(top, float(da.max()))
        assert np.isclose(bot, float(da.min()))

    def test_std_band_strictly_narrower_than_range(self) -> None:
        std_top, std_bot = _band_extents("std")
        rng_top, rng_bot = _band_extents("range")
        assert std_top < rng_top
        assert std_bot > rng_bot

    def test_iqr_band_within_range(self) -> None:
        iqr_top, iqr_bot = _band_extents("iqr")
        rng_top, rng_bot = _band_extents("range")
        assert iqr_top <= rng_top
        assert iqr_bot >= rng_bot
        # IQR is a real band (q75 > q25 somewhere), so top strictly above bottom.
        assert iqr_top > iqr_bot
