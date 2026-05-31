"""Runtime regressions for the unified data-source refactor."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr


def _write_grid_source(path: Path, *, offset: float = 0.0) -> None:
    times = np.array(["2024-01-01T00:00", "2024-01-01T01:00"], dtype="datetime64[m]")
    lat = np.array([40.0, 41.0])
    lon = np.array([-105.0, -104.0])
    values = np.arange(8, dtype=float).reshape(2, 2, 2) + offset
    ds = xr.Dataset(
        {"O3": (("time", "lat", "lon"), values)},
        coords={"time": times, "lat": lat, "lon": lon},
        attrs={"geometry": "grid"},
    )
    ds.to_netcdf(path)


def _write_point_source(path: Path) -> None:
    times = np.array(["2024-01-01T00:00", "2024-01-01T01:00"], dtype="datetime64[m]")
    ds = xr.Dataset(
        {"o3": (("time", "site"), np.array([[1.0, 2.0], [3.0, 4.0]]))},
        coords={
            "time": times,
            "site": np.array([0, 1]),
            "latitude": ("site", np.array([40.0, 41.0])),
            "longitude": ("site", np.array([-105.0, -104.0])),
        },
        attrs={"geometry": "point"},
    )
    ds.to_netcdf(path)


def test_sources_config_pairs_from_pair_variables(tmp_path: Path) -> None:
    from davinci_monet.pipeline.runner import PipelineRunner

    model_path = tmp_path / "model.nc"
    obs_path = tmp_path / "obs.nc"
    _write_grid_source(model_path)
    _write_point_source(obs_path)

    config = {
        "analysis": {"output_dir": str(tmp_path / "out")},
        "sources": {
            "cam": {
                "type": "generic",
                "role": "model",
                "files": str(model_path),
                "radius_of_influence": 200000,
                "variables": {"O3": {"units": "ppb"}},
            },
            "airnow": {
                "type": "pt_sfc",
                "role": "obs",
                "filename": str(obs_path),
                "variables": {"o3": {"units": "ppb"}},
            },
        },
        "pairs": {
            "cam_airnow_o3": {
                "sources": ["cam", "airnow"],
                "reference": "airnow",
                "variables": {"cam": "O3", "airnow": "o3"},
            }
        },
        "stats": {"metrics": ["N", "MB"]},
    }

    result = PipelineRunner(show_progress=False).run_from_config(config)

    assert result.success
    assert result.context is not None
    assert set(result.context.sources) == {"cam", "airnow"}
    assert set(result.context.paired) == {"cam_airnow_o3"}
    paired = result.context.paired["cam_airnow_o3"].data
    assert set(paired.data_vars) == {"cam_o3", "airnow_o3"}
    assert paired["cam_o3"].attrs["source_label"] == "cam"
    assert paired["airnow_o3"].attrs["source_label"] == "airnow"


def test_sources_config_supports_model_model_pair(tmp_path: Path) -> None:
    from davinci_monet.pipeline.runner import PipelineRunner

    ref_path = tmp_path / "ref.nc"
    comp_path = tmp_path / "comp.nc"
    _write_grid_source(ref_path, offset=0.0)
    _write_grid_source(comp_path, offset=1.0)

    config = {
        "analysis": {"output_dir": str(tmp_path / "out")},
        "sources": {
            "cam_ref": {"type": "generic", "role": "model", "files": str(ref_path)},
            "cam_cmp": {"type": "generic", "role": "model", "files": str(comp_path)},
        },
        "pairs": {
            "cam_ref_cam_cmp_o3": {
                "sources": ["cam_ref", "cam_cmp"],
                "reference": "cam_ref",
                "variables": {"cam_ref": "O3", "cam_cmp": "O3"},
            }
        },
        "stats": {"metrics": ["N", "MB"]},
    }

    result = PipelineRunner(show_progress=False).run_from_config(config)

    assert result.success
    assert result.context is not None
    paired = result.context.paired["cam_ref_cam_cmp_o3"].data
    assert set(paired.data_vars) == {"cam_ref_O3", "cam_cmp_O3"}
    assert paired["cam_ref_O3"].attrs["pair_role"] == "reference"
    assert paired["cam_cmp_O3"].attrs["pair_role"] == "comparand"
    assert paired["cam_ref_O3"].attrs["role"] == "model"
    assert paired["cam_cmp_O3"].attrs["role"] == "model"


def test_plot_data_reference_without_pair_spec_uses_paired_dataset(
    tmp_path: Path,
) -> None:
    from davinci_monet.pipeline.stages import PipelineContext, PlottingStage, StageStatus

    times = np.array(["2024-01-01T00:00", "2024-01-01T01:00"], dtype="datetime64[m]")
    paired = xr.Dataset(
        {
            "airnow_o3": ("time", np.array([1.0, 2.0])),
            "cam_o3": ("time", np.array([1.1, 2.2])),
        },
        coords={"time": times},
    )
    paired["airnow_o3"].attrs.update(
        {"role": "obs", "pair_role": "reference", "source_label": "airnow"}
    )
    paired["cam_o3"].attrs.update(
        {"role": "model", "pair_role": "comparand", "source_label": "cam"}
    )
    ctx = PipelineContext(
        config={
            "analysis": {"output_dir": str(tmp_path)},
            "plots": {"scatter_o3": {"type": "scatter", "data": ["cam_airnow"]}},
        }
    )
    ctx.paired["cam_airnow"] = paired

    result = PlottingStage().execute(ctx)

    assert result.status is StageStatus.COMPLETED
    assert len(result.data["plots_generated"]) == 2
