"""Integration: CERES readers through the full pipeline.

Exercises PipelineRunner.run_from_config() with a ``type: ceres_ebaf`` GRID
source as the pairing reference against a synthetic gridded model — the same
path a user takes with ``davinci-monet run``. SSF/SYN1deg pipeline tests
arrive with their phases.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import xarray as xr
import yaml

from davinci_monet.pipeline.runner import PipelineRunner

pytestmark = pytest.mark.integration


def _monthly_grid(varname: str, seed: int, lon0360: bool = False) -> xr.Dataset:
    times = np.array(["2025-10-01", "2025-11-01", "2025-12-01"], dtype="datetime64[ns]")
    lat = np.linspace(-87.5, 87.5, 6)
    lon = np.linspace(2.5, 357.5, 8) if lon0360 else np.linspace(-175.0, 175.0, 8)
    rng = np.random.default_rng(seed)
    data = rng.uniform(150.0, 300.0, size=(3, 6, 8)).astype(np.float32)
    return xr.Dataset(
        {varname: (("time", "lat", "lon"), data)},
        coords={"time": times, "lat": lat, "lon": lon},
    )


def test_ceres_ebaf_pipeline(tmp_path: Path) -> None:
    e_dir = tmp_path / "ebaf"
    m_dir = tmp_path / "model"
    e_dir.mkdir()
    m_dir.mkdir()
    # EBAF side uses 0-360 longitudes — the reader must normalize them so
    # GRID-GRID pairing aligns with the model's -180..180 grid.
    _monthly_grid("toa_lw_all_mon", seed=1, lon0360=True).to_netcdf(
        e_dir / "CERES_EBAF_Edition4.2.1_202510-202512.nc"
    )
    _monthly_grid("OLR", seed=2).to_netcdf(m_dir / "model.nc")

    out_dir = tmp_path / "output"
    config = {
        "analysis": {
            "start_time": "2025-10-01",
            "end_time": "2025-12-31",
            "output_dir": str(out_dir),
            "log_dir": str(tmp_path / "logs"),
        },
        "sources": {
            "ceres": {
                "type": "ceres_ebaf",
                "role": "obs",
                "files": str(e_dir / "*.nc"),
                "variables": {"toa_lw_all_mon": {"units": "W m-2"}},
            },
            "model": {
                "type": "generic",
                "role": "model",
                "files": str(m_dir / "*.nc"),
                "variables": {"OLR": {"units": "W m-2"}},
            },
        },
        "pairs": {
            "model_vs_ceres_olr": {
                "sources": ["model", "ceres"],
                "reference": "ceres",
                "variables": {"model": "OLR", "ceres": "toa_lw_all_mon"},
            }
        },
        "plots": {
            "bias": {
                "type": "spatial_bias",
                "pairs": ["model_vs_ceres_olr"],
                "title": "OLR Bias",
            },
            "sc": {
                "type": "scatter",
                "pairs": ["model_vs_ceres_olr"],
                "title": "OLR Scatter",
            },
        },
        "stats": {"output_table": True, "metrics": ["N", "MB", "RMSE", "R"]},
    }
    cfg = tmp_path / "config.yaml"
    cfg.write_text(yaml.safe_dump(config))

    result = PipelineRunner(show_progress=False).run_from_config(str(cfg))

    failed = [
        f"{s.stage_name}: {s.error}" for s in result.stage_results if s.status.name == "FAILED"
    ]
    assert result.success, f"Pipeline failed: {failed}"
    assert sorted(out_dir.rglob("*.png")), "expected plots"
    assert list(out_dir.rglob("*.csv")), "expected a stats CSV"
