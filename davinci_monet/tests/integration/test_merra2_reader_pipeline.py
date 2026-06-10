"""Integration: MERRA-2 reader through the full pipeline.

Exercises PipelineRunner.run_from_config() with a ``type: merra2`` GRID source
paired against a synthetic GRID obs source, mirroring MERRA-2 AOD vs a gridded
AOD reference. This is the pipeline path a user takes with ``davinci-monet run``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import xarray as xr
import yaml

from davinci_monet.pipeline.runner import PipelineRunner

pytestmark = pytest.mark.integration


def _grid(varname: str, seed: int) -> xr.Dataset:
    times = np.array(
        ["2026-04-01", "2026-04-02", "2026-04-03"], dtype="datetime64[ns]"
    )
    lat = np.linspace(-87.5, 87.5, 6)
    lon = np.linspace(-175.0, 175.0, 8)
    rng = np.random.default_rng(seed)
    data = rng.uniform(0.05, 0.8, size=(3, 6, 8)).astype(np.float32)
    return xr.Dataset(
        {varname: (("time", "lat", "lon"), data)},
        coords={"time": times, "lat": lat, "lon": lon},
    )


def test_merra2_reader_pipeline(tmp_path: Path) -> None:
    m_dir = tmp_path / "merra2"
    o_dir = tmp_path / "obs"
    m_dir.mkdir()
    o_dir.mkdir()
    _grid("TOTEXTTAU", seed=1).to_netcdf(
        m_dir / "MERRA2_400.tavgM_2d_aer_Nx.202604.nc4"
    )
    _grid("aod_550nm", seed=2).to_netcdf(o_dir / "obs.nc")

    out_dir = tmp_path / "output"
    config = {
        "analysis": {
            "start_time": "2026-04-01",
            "end_time": "2026-04-03",
            "output_dir": str(out_dir),
            "log_dir": str(tmp_path / "logs"),
        },
        "sources": {
            "merra2": {
                "type": "merra2",
                "role": "model",
                "files": str(m_dir / "*.nc4"),
                "variables": {"TOTEXTTAU": {"units": "1"}},
            },
            "ref": {
                "type": "generic",
                "role": "obs",
                "files": str(o_dir / "*.nc"),
                "variables": {"aod_550nm": {"units": "1"}},
            },
        },
        "pairs": {
            "merra2_vs_ref": {
                "sources": ["merra2", "ref"],
                "reference": "ref",
                "variables": {"merra2": "TOTEXTTAU", "ref": "aod_550nm"},
            }
        },
        "plots": {
            "bias": {
                "type": "spatial_bias",
                "pairs": ["merra2_vs_ref"],
                "title": "AOD Bias",
            },
            "sc": {
                "type": "scatter",
                "pairs": ["merra2_vs_ref"],
                "title": "AOD Scatter",
            },
        },
        "stats": {"output_table": True, "metrics": ["N", "MB", "RMSE", "R"]},
    }
    cfg = tmp_path / "config.yaml"
    cfg.write_text(yaml.safe_dump(config))

    result = PipelineRunner(show_progress=False).run_from_config(str(cfg))

    failed = [
        f"{s.stage_name}: {s.error}"
        for s in result.stage_results
        if s.status.name == "FAILED"
    ]
    assert result.success, f"Pipeline failed: {failed}"
    assert sorted(out_dir.rglob("*.png")), "expected plots"
    assert list(out_dir.rglob("*.csv")), "expected a stats CSV"
