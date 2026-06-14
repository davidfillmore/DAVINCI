"""IntermediateGridStrategy symmetric binning (method: grid, 2-D)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from davinci_monet.pairing.strategies.intermediate_grid import IntermediateGridStrategy


def _point_ds(lats, lons, vals, var, t="2024-02-01"):
    n = len(lats)
    time = pd.to_datetime([t] * n)
    return xr.Dataset(
        {var: (["site"], np.asarray(vals, float), {"units": "1"})},
        coords={
            "site": np.arange(n),
            "time": ("site", time),
            "latitude": ("site", np.asarray(lats, float)),
            "longitude": ("site", np.asarray(lons, float)),
        },
    )


def test_symmetric_bins_both_point_sources_cell_means():
    # Two points fall in the SAME 1-degree cell -> their mean; one alone elsewhere.
    x = _point_ds([10.2, 10.7, 40.5], [20.2, 20.6, 50.5], [1.0, 3.0, 9.0], "aod")
    y = _point_ds([10.4, 40.4], [20.4, 50.4], [2.0, 8.0], "AOD")
    paired = IntermediateGridStrategy().pair_sources(
        x_data=x,
        y_data=y,
        x_var="aod",
        y_var="AOD",
        x_source="obs",
        y_source="mod",
        horizontal_res=1.0,
        time_resolution="1D",
        min_sample_count=1,
    )
    assert list(paired["x_aod"].dims) == ["time", "lon", "lat"]
    assert "y_AOD" in paired and "x_sample_count" in paired and "y_sample_count" in paired
    xa = paired["x_aod"].squeeze("time", drop=True)
    ya = paired["y_AOD"].squeeze("time", drop=True)
    # the cell containing (~10.x, ~20.x): x mean = (1+3)/2 = 2.0, y = 2.0, counts 2 and 1
    cell_x = xa.sel(lat=10.5, lon=20.5, method="nearest").item()
    cell_y = ya.sel(lat=10.5, lon=20.5, method="nearest").item()
    assert cell_x == pytest.approx(2.0)
    assert cell_y == pytest.approx(2.0)
    assert paired["x_sample_count"].sel(lat=10.5, lon=20.5, method="nearest").max().item() == 2
    # tagged for downstream stats/plots
    assert paired["x_aod"].attrs["axis"] == "x" and paired["x_aod"].attrs["source_label"] == "obs"
    assert paired["y_AOD"].attrs["axis"] == "y" and paired["y_AOD"].attrs["source_label"] == "mod"


def test_min_sample_count_masks_sparse_cells():
    x = _point_ds([10.2, 10.7], [20.2, 20.6], [1.0, 3.0], "aod")  # 2 in one cell
    y = _point_ds([10.4], [20.4], [2.0], "AOD")  # 1 in that cell
    paired = IntermediateGridStrategy().pair_sources(
        x_data=x,
        y_data=y,
        x_var="aod",
        y_var="AOD",
        x_source="obs",
        y_source="mod",
        horizontal_res=1.0,
        time_resolution="1D",
        min_sample_count=2,
    )
    # y has only 1 sample in the cell -> masked to NaN under min_sample_count=2
    ya = paired["y_AOD"].squeeze("time", drop=True)
    assert np.isnan(ya.sel(lat=10.5, lon=20.5, method="nearest").item())
