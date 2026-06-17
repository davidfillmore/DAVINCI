"""eof_scree renders explained-variance bars with North error bars."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
import xarray as xr  # noqa: E402
from matplotlib.container import BarContainer  # noqa: E402

from davinci_monet.plots.base import build_series  # noqa: E402
from davinci_monet.plots.renderers.eof_scree import EOFScreePlotter  # noqa: E402


def _ds() -> xr.Dataset:
    ds = xr.Dataset(
        {
            "explained_variance": ("mode", np.array([0.6, 0.25, 0.1])),
            "explained_variance_error": ("mode", np.array([0.05, 0.03, 0.02])),
        },
        coords={"mode": [1, 2, 3]},
    )
    ds.attrs["eof_quantity"] = "O3"
    return ds


def test_eof_scree_bars_and_errorbars() -> None:
    fig = EOFScreePlotter().render(build_series(_ds(), "explained_variance"))
    ax = fig.axes[0]
    assert any(isinstance(c, BarContainer) for c in ax.containers)
    assert ax.get_ylabel().startswith("Explained variance")
    heights = sorted(rect.get_height() for rect in ax.patches)  # type: ignore[attr-defined]
    assert heights[-1] == pytest.approx(60.0, abs=0.1)
    plt.close(fig)
