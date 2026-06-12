"""Integration: CERES readers through the full pipeline.

Exercises PipelineRunner.run_from_config() with a ``type: ceres_ebaf`` GRID
source as the pairing reference against a synthetic gridded model — the same
path a user takes with ``davinci-monet run``. SSF/SYN1deg pipeline tests
arrive with their phases.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import yaml

from davinci_monet.pipeline.runner import PipelineRunner

pytestmark = pytest.mark.integration


def _monthly_grid(
    varname: str,
    seed: int,
    lon0360: bool = False,
    lat_span: float | None = None,
    lon_span: float | None = None,
) -> xr.Dataset:
    """Return a synthetic 3-month gridded dataset (6 lat × 8 lon by default).

    Parameters
    ----------
    varname:
        Name of the single data variable.
    seed:
        RNG seed for reproducibility.
    lon0360:
        If True, longitudes are in 0-360 range (EBAF convention).
    lat_span:
        Half-span of the latitude axis in degrees (default 87.5, giving ±87.5).
        Pass 89.5 to cover the full SYN1deg domain and avoid edge NaNs.
    lon_span:
        Half-span of the longitude axis in degrees (default 175.0, giving ±175).
        Pass 179.5 to cover the full SYN1deg domain and avoid edge NaNs.
    """
    times = np.array(["2025-10-01", "2025-11-01", "2025-12-01"], dtype="datetime64[ns]")
    _lat_span = lat_span if lat_span is not None else 87.5
    _lon_span = lon_span if lon_span is not None else 175.0
    lat = np.linspace(-_lat_span, _lat_span, 6)
    if lon0360:
        lon = np.linspace(2.5, 357.5, 8)
    else:
        lon = np.linspace(-_lon_span, _lon_span, 8)
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
    csvs = list(out_dir.rglob("statistics_summary.csv"))
    assert csvs, "expected a stats CSV"
    # All 3 times x 6 lats x 8 lons must pair; a broken lon normalization
    # NaNs the 0-360 half of the grid and silently halves N (review finding).
    stats = pd.read_csv(csvs[0])
    n_col = next(c for c in stats.columns if c.strip().upper() == "N")
    assert int(stats[n_col].iloc[0]) == 144, f"expected N=144, got\n{stats}"


_IO_EBAF = Path("/Volumes/Io/CERES/EBAF")
_RUN_REAL = bool(os.environ.get("CERES_DATA"))


@pytest.mark.skipif(
    not (_RUN_REAL and _IO_EBAF.is_dir()),
    reason="real-data smoke is opt-in (set CERES_DATA) and needs /Volumes/Io",
)
def test_real_ebaf_file_opens() -> None:
    """Smoke: open the staged EBAF record via the reader, check physics.

    Opt-in only — set ``CERES_DATA`` to activate::

        export CERES_DATA=/Volumes/Io/CERES

    Not auto-run on mount: opening the ~2 GB netCDF over the SMB volume
    contaminates global netCDF4/HDF5 state, and unrelated dask-parallel
    tests then fail transiently when this runs inside the full suite.
    """
    from davinci_monet.observations.satellite.ceres_l3 import CERESEBAFReader

    files = sorted(f for f in _IO_EBAF.glob("CERES_EBAF_*.nc") if not f.name.startswith("._"))
    if not files:
        pytest.skip("no EBAF .nc files present")

    ds = CERESEBAFReader().open([files[0]], variables=["toa_lw_all_mon"])

    assert set(ds.data_vars) == {"toa_lw_all_mon"}
    assert ds.attrs["geometry"] == "grid"
    assert "ctime" not in ds.dims
    lon = ds["lon"].values
    assert lon.min() >= -180.0 and lon.max() < 180.0
    # Area-weighted global-mean OLR for one month must be physical.
    da = ds["toa_lw_all_mon"].isel(time=-1)
    weights = np.cos(np.deg2rad(ds["lat"]))
    gmean = float(da.weighted(weights).mean())
    assert 220.0 <= gmean <= 260.0, f"global-mean OLR {gmean:.1f} W m-2 unphysical"


# ---------------------------------------------------------------------------
# SYN1deg pipeline (Phase 3)
# ---------------------------------------------------------------------------


def test_ceres_syn1deg_pipeline(tmp_path: Path) -> None:
    from davinci_monet.tests.test_ceres_l3_readers import _write_syn_hdf4

    s_dir = tmp_path / "syn"
    m_dir = tmp_path / "model"
    s_dir.mkdir()
    m_dir.mkdir()
    for stamp in ("202510", "202511", "202512"):
        _write_syn_hdf4(
            s_dir / f"CER_SYN1deg-Month_Terra-Aqua-NOAA20_Edition4B_415412.{stamp}",
            nlat=6,
            nlon=8,
        )
    # Extend lat/lon span to ±89.5/±179.5 so the model fully covers the SYN
    # grid edges.  Without this, xarray.interp NaNs the edge rows/cols of the
    # model-regridded field and N falls below 3×6×8=144.
    _monthly_grid("OLR", seed=2, lat_span=89.5, lon_span=179.5).to_netcdf(m_dir / "model.nc")

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
                "type": "ceres_syn1deg",
                "role": "obs",
                "files": str(s_dir / "*.2025*"),
                "variables": {"obs_all_toa_lw_reg": {"units": "W m-2"}},
            },
            "model": {
                "type": "generic",
                "role": "model",
                "files": str(m_dir / "*.nc"),
                "variables": {"OLR": {"units": "W m-2"}},
            },
        },
        "pairs": {
            "model_vs_syn_olr": {
                "sources": ["model", "ceres"],
                "reference": "ceres",
                "variables": {"model": "OLR", "ceres": "obs_all_toa_lw_reg"},
            }
        },
        "plots": {
            "sc": {"type": "scatter", "pairs": ["model_vs_syn_olr"], "title": "OLR"},
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
    csvs = list(out_dir.rglob("statistics_summary.csv"))
    assert csvs, "expected a stats CSV"
    stats = pd.read_csv(csvs[0])
    n_col = next(c for c in stats.columns if c.strip().upper() == "N")
    # 3 months x 6 lats x 8 lons must pair; shrinkage means lost coverage.
    assert int(stats[n_col].iloc[0]) == 144, f"expected N=144, got\n{stats}"


# ---------------------------------------------------------------------------
# SYN1deg real-data smoke (Phase 3)
# ---------------------------------------------------------------------------

_IO_SYN_MONTH = Path("/Volumes/Io/CERES/SYN1deg/month")


@pytest.mark.skipif(
    not (_RUN_REAL and _IO_SYN_MONTH.is_dir() and _IO_EBAF.is_dir()),
    reason="real-data smoke is opt-in (set CERES_DATA) and needs /Volumes/Io",
)
def test_real_syn1deg_zonal_means_correlate_with_ebaf() -> None:
    """Smoke: SYN1deg 2025-12 zonal-mean OLR must track EBAF's (lat axis check)."""
    from davinci_monet.observations.satellite.ceres_l3 import (
        CERESEBAFReader,
        CERESSYN1degReader,
    )

    syn_files = sorted(_IO_SYN_MONTH.glob("CER_SYN1deg-Month_*.202512"))
    ebaf_files = sorted(f for f in _IO_EBAF.glob("CERES_EBAF_*.nc") if not f.name.startswith("._"))
    if not syn_files or not ebaf_files:
        pytest.skip("staged SYN1deg/EBAF samples not found")

    syn = CERESSYN1degReader().open([syn_files[0]], variables=["obs_all_toa_lw_reg"])
    ebaf = CERESEBAFReader().open([ebaf_files[0]], variables=["toa_lw_all_mon"])

    syn_zonal = syn["obs_all_toa_lw_reg"].isel(time=0).mean("lon")
    ebaf_zonal = ebaf["toa_lw_all_mon"].sel(time="2025-12").squeeze().mean("lon")

    r = float(np.corrcoef(syn_zonal.values, ebaf_zonal.values)[0, 1])
    assert r > 0.9, f"SYN vs EBAF zonal correlation {r:.3f} — latitude axis suspect"
    assert syn.sizes["time"] == 1 and set(syn["obs_all_toa_lw_reg"].dims) == {
        "time",
        "lat",
        "lon",
    }
