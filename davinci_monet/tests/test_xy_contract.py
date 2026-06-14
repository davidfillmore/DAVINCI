"""Semantic contract: x is the horizontal/reference axis, y is vertical; diffs are y - x.

This test is the behavior-preservation anchor for the x/y rename. It is written
against the current geometry/dataset API and will be migrated to x/y in lockstep
with the rename tasks; its ASSERTIONS (axis assignment + diff sign) must never change.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from davinci_monet.plots.base import build_series
from davinci_monet.plots.renderers.scatter import ScatterPlotter


def _paired() -> xr.Dataset:
    time = np.arange(5)
    obs = xr.DataArray(np.arange(5.0), dims="time", name="obs_o3")
    obs.attrs.update({"axis": "x", "dataset_label": "obs", "units": "ppb"})
    mod = xr.DataArray(np.arange(5.0) + 2.0, dims="time", name="mod_o3")
    mod.attrs.update({"axis": "y", "dataset_label": "mod", "units": "ppb"})
    ds = xr.Dataset({"obs_o3": obs, "mod_o3": mod}, coords={"time": time})
    return ds


def test_scatter_x_is_geometry_y_is_dataset() -> None:
    ds = _paired()
    fig = ScatterPlotter().render(build_series(ds, "obs_o3", "mod_o3"))
    ax = fig.axes[0]
    # x axis names the geometry/x source; y axis names the dataset/y source.
    assert "OBS" in ax.get_xlabel().upper() or "obs" in ax.get_xlabel().lower()
    assert "MOD" in ax.get_ylabel().upper() or "mod" in ax.get_ylabel().lower()


def test_diff_sign_is_y_minus_x() -> None:
    ds = _paired()
    # mod - obs == +2 everywhere (the "y - x" convention).
    diff = ds["mod_o3"] - ds["obs_o3"]
    assert float(diff.mean()) == 2.0
