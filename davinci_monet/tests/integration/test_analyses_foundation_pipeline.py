"""Integration: a derived analysis runs through the full pipeline and its output
is registered as a pseudo-source (proves the foundation end-to-end)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from davinci_monet.analysis import DerivedAnalysis
from davinci_monet.core.protocols import DataGeometry
from davinci_monet.core.registry import analysis_registry
from davinci_monet.pipeline.runner import PipelineRunner


@pytest.fixture
def _passthrough_eof():
    """Register a trivial 'eof' that emits a (time, mode) pc + (mode) variance."""

    @analysis_registry.register("eof", replace=True)
    class _PassEOF(DerivedAnalysis):
        name = "eof"
        output_geometry = DataGeometry.GRID

        def analyze(self, data, spec):  # noqa: ANN001
            nt = data.sizes["time"]
            return xr.Dataset(
                {
                    "pc": (("time", "mode"), np.zeros((nt, 2)), {"kind": "pc", "units": "1"}),
                    "explained_variance": ("mode", np.array([0.7, 0.3]), {"kind": "scalar"}),
                },
                coords={"time": data["time"].values, "mode": [1, 2]},
            )

    yield
    analysis_registry.unregister("eof")


def _grid_nc(path: Path) -> None:
    times = pd.date_range("2024-01-01", periods=6, freq="D")
    lat = np.linspace(20, 50, 4)
    lon = np.linspace(-120, -90, 5)
    rng = np.random.default_rng(0)
    data = rng.normal(size=(len(times), len(lat), len(lon)))
    xr.Dataset(
        {"O3": (("time", "lat", "lon"), data, {"units": "ppb"})},
        coords={
            "time": times,
            "lat": ("lat", lat),
            "lon": ("lon", lon),
            "latitude": ("lat", lat),
            "longitude": ("lon", lon),
        },
    ).to_netcdf(path)


@pytest.mark.integration
def test_analysis_runs_through_pipeline(tmp_path: Path, _passthrough_eof) -> None:
    src = tmp_path / "grid.nc"
    _grid_nc(src)
    config = {
        "analysis": {"output_dir": str(tmp_path / "out")},
        "sources": {
            "cam": {"type": "generic", "files": str(src), "variables": {"O3": {"units": "ppb"}}}
        },
        "analyses": {
            "cam_O3_eof": {"type": "eof", "source": "cam", "variable": "O3", "n_modes": 2}
        },
    }

    result = PipelineRunner(show_progress=False).run_from_config(config)

    assert result.success, getattr(result, "error", None)
    ctx = result.context
    assert ctx is not None
    assert "cam_O3_eof" in ctx.sources
    derived = ctx.sources["cam_O3_eof"]
    assert derived.source_type == "eof"
    assert derived.geometry is DataGeometry.GRID
    assert derived.data.attrs["derived"] is True
    assert set(derived.data.data_vars) == {"pc", "explained_variance"}
    assert "cam_O3_eof" in ctx.results["analyses"].data
