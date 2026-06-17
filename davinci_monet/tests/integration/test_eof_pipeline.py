"""Integration: an eof analysis runs through the pipeline and produces its plots."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from davinci_monet.pipeline.runner import PipelineRunner


def _grid_nc(path: Path) -> None:
    times = pd.date_range("2024-01-01", periods=120, freq="D")
    lat = np.linspace(-5, 5, 6)
    lon = np.linspace(0, 30, 8)
    x = np.linspace(0, np.pi, len(lon))
    rng = np.random.default_rng(0)
    p1 = np.cos(x)[None, :] * np.ones((len(lat), 1))
    pc1 = rng.normal(size=len(times))
    field = 3.0 * pc1[:, None, None] * p1[None] + 0.1 * rng.normal(
        size=(len(times), len(lat), len(lon))
    )
    xr.Dataset(
        {"O3": (("time", "lat", "lon"), field, {"units": "ppb"})},
        coords={
            "time": times,
            "lat": ("lat", lat),
            "lon": ("lon", lon),
            "latitude": ("lat", lat),
            "longitude": ("lon", lon),
        },
    ).to_netcdf(path)


@pytest.mark.integration
def test_eof_plots_through_pipeline(tmp_path: Path) -> None:
    src = tmp_path / "grid.nc"
    _grid_nc(src)
    config = {
        "analysis": {"output_dir": str(tmp_path / "out")},
        "sources": {
            "cam": {"type": "generic", "files": str(src), "variables": {"O3": {"units": "ppb"}}}
        },
        "analyses": {
            "cam_O3_eof": {"type": "eof", "source": "cam", "variable": "O3", "n_modes": 3}
        },
        "plots": {
            "eof_maps": {"type": "eof_pattern", "source": "cam_O3_eof", "variable": "eofs"},
            "eof_var": {
                "type": "eof_scree",
                "source": "cam_O3_eof",
                "variable": "explained_variance",
            },
            "pc1": {"type": "timeseries", "source": "cam_O3_eof", "variable": "pc", "mode": 1},
        },
    }

    result = PipelineRunner(show_progress=False).run_from_config(config)

    assert result.success, getattr(result, "error", None)
    ctx = result.context
    assert ctx is not None
    plots = ctx.results["plotting"].data["plots_generated"]
    pngs = [p for p in plots if p.endswith(".png")]
    assert sum("eof_maps" in p for p in pngs) == 3  # one map per mode
    assert any("eof_var" in p for p in pngs)
    assert any("pc1" in p for p in pngs)
