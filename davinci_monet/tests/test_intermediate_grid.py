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


def test_small_span_auto_extent_keeps_all_points():
    # Regression: data span (~0.9 deg) smaller than horizontal_res (2.0) with the
    # default (auto) extent must NOT silently drop edge points — the grid must
    # still span the full data extent.
    x = _point_ds(
        [30.0, 30.3, 30.6, 30.9], [-110.0, -110.3, -110.6, -110.9], [1.0, 1.0, 1.0, 1.0], "aod"
    )
    y = _point_ds([30.4, 30.5], [-110.4, -110.5], [2.0, 2.0], "AOD")
    paired = IntermediateGridStrategy().pair_sources(
        x_data=x,
        y_data=y,
        x_var="aod",
        y_var="AOD",
        x_source="obs",
        y_source="mod",
        horizontal_res=2.0,
        time_resolution="1D",
        min_sample_count=1,
    )
    assert int(paired["x_sample_count"].sum().item()) == 4  # all 4 x points retained
    assert int(paired["y_sample_count"].sum().item()) == 2  # all 2 y points retained


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


def test_engine_routes_method_grid_to_symmetric():
    from davinci_monet.pairing.engine import PairingEngine

    x = _point_ds([10.2, 10.7], [20.2, 20.6], [1.0, 3.0], "aod")
    y = _point_ds([10.4], [20.4], [2.0], "AOD")
    paired = PairingEngine().pair_sources(
        x_data=x,
        y_data=y,
        x_vars=["aod"],
        y_vars=["AOD"],
        x_source="obs",
        y_source="mod",
        method="grid",
        horizontal_res=1.0,
        time_resolution="1D",
        min_sample_count=1,
    )
    data = getattr(paired, "data", paired)
    assert isinstance(data, xr.Dataset)
    assert "x_aod" in data and "y_AOD" in data
    assert list(data["x_aod"].dims) == ["time", "lon", "lat"]


@pytest.mark.integration
def test_method_grid_runs_through_pipeline(tmp_path):
    from davinci_monet.pipeline.runner import PipelineRunner

    x = _point_ds([10.2, 10.7, 40.5], [20.2, 20.6, 50.5], [1.0, 3.0, 9.0], "aod")
    y = _point_ds([10.4, 40.4], [20.4, 50.4], [2.0, 8.0], "AOD")
    xp, yp = tmp_path / "x.nc", tmp_path / "y.nc"
    x.to_netcdf(xp)
    y.to_netcdf(yp)

    config = {
        "analysis": {"output_dir": str(tmp_path / "out")},
        "sources": {
            "obs": {"type": "generic", "files": str(xp), "variables": {"aod": {"units": "1"}}},
            "mod": {"type": "generic", "files": str(yp), "variables": {"AOD": {"units": "1"}}},
        },
        "pairs": {
            "obs_vs_mod": {
                "x": {"source": "obs", "variable": "aod"},
                "y": {"source": "mod", "variable": "AOD"},
                "method": "grid",
                "grid": {"horizontal_res": 1.0, "time_resolution": "1D", "min_sample_count": 1},
            }
        },
        "plots": {"sc": {"type": "scatter", "data": ["obs_vs_mod"]}},
    }
    result = PipelineRunner(show_progress=False).run_from_config(config)
    assert result.success, getattr(result, "error", None)
    ctx = result.context
    assert ctx is not None
    assert "obs_vs_mod" in ctx.paired


def test_pressure_to_altitude_standard_atmosphere():
    import numpy as np

    from davinci_monet.pairing.strategies.track import pressure_to_altitude

    p = np.array([1013.25, 500.0, 700.0])
    z = pressure_to_altitude(p)
    assert z[0] == pytest.approx(0.0, abs=1.0)  # sea level
    assert z[1] == pytest.approx(5572.0, abs=50.0)  # ~500 hPa
    assert z[2] == pytest.approx(3012.0, abs=50.0)  # ~700 hPa
    # round-trips with the existing forward conversion
    from davinci_monet.pairing.strategies.track import altitude_to_pressure

    assert altitude_to_pressure(z)[1] == pytest.approx(500.0, rel=1e-3)
