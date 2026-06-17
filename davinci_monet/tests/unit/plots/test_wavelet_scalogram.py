"""wavelet_scalogram draws a QuadMesh + a global-spectrum side panel."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xarray as xr  # noqa: E402
from matplotlib.collections import QuadMesh  # noqa: E402

from davinci_monet.plots.base import build_series  # noqa: E402
from davinci_monet.plots.renderers.wavelet_scalogram import WaveletScalogramPlotter  # noqa: E402


def _spectrum() -> xr.Dataset:
    nt, npd = 60, 6
    time = pd.date_range("2024-01-01", periods=nt, freq="D")
    period = np.array([2.0, 4.0, 8.0, 16.0, 32.0, 64.0])
    rng = np.random.default_rng(0)
    power = rng.random((nt, npd))
    ds = xr.Dataset(
        {
            "power": (("time", "period"), power, {"kind": "power"}),
            "power_significance": (("time", "period"), power / power.mean(0), {"kind": "power"}),
            "coi": (("time",), np.full(nt, 16.0), {"kind": "coi", "units": "days"}),
            "global_power": (("period",), power.mean(0), {"kind": "global"}),
            "global_significance": (("period",), np.ones(npd), {"kind": "global"}),
        },
        coords={"time": time, "period": ("period", period, {"units": "days"})},
    )
    ds.attrs["wavelet_quantity"] = "O3"
    return ds


def test_scalogram_quadmesh_and_global_panel() -> None:
    fig = WaveletScalogramPlotter().render(build_series(_spectrum(), "power"))
    meshes = [c for ax in fig.axes for c in ax.collections if isinstance(c, QuadMesh)]
    assert meshes, "expected a QuadMesh power layer"
    # Dense data layer is rasterized so the vector PDF stays small.
    assert meshes[0].get_rasterized() is True
    assert len(fig.axes) >= 2
    plt.close(fig)
