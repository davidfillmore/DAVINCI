"""Unit tests for demo stages."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from davinci_monet.addons.plume_sentinel.demo_stages import (
    PlumeSentinelDemoLoadStage,
    PlumeSentinelDemoPlotStage,
    PlumeSentinelDemoPrepareStage,
)
from davinci_monet.addons.plume_sentinel.schema import PlumeSentinelConfig
from davinci_monet.pipeline.stages import PipelineContext, StageStatus


def _make_context(tmp_path: Path) -> PipelineContext:
    config = {
        "analysis": {"output_dir": str(tmp_path), "start_time": "2020-09-09"},
        "plume_sentinel": {"inputs": {}, "plots": {}},
    }
    ctx = PipelineContext(config=config)
    ctx.metadata["plume_sentinel_config"] = PlumeSentinelConfig(**config["plume_sentinel"])
    return ctx


def test_demo_load_stage_populates_loaded_metadata(tmp_path):
    stage = PlumeSentinelDemoLoadStage()
    ctx = _make_context(tmp_path)
    with patch("davinci_monet.addons.plume_sentinel.demo_stages.time.sleep") as sleep:
        result = stage.execute(ctx)
    assert result.status == StageStatus.COMPLETED
    assert "plume_sentinel_loaded" in ctx.metadata
    assert isinstance(ctx.metadata["plume_sentinel_loaded"], dict)
    sleep.assert_called()  # at least one sleep call
    assert sum(c.args[0] for c in sleep.call_args_list) == pytest.approx(3.0, abs=0.5)


def test_demo_prepare_stage_populates_prepared_and_input_datasets(tmp_path):
    stage = PlumeSentinelDemoPrepareStage()
    ctx = _make_context(tmp_path)
    ctx.metadata["plume_sentinel_loaded"] = {}  # set by demo load
    with patch("davinci_monet.addons.plume_sentinel.demo_stages.time.sleep") as sleep:
        result = stage.execute(ctx)
    assert result.status == StageStatus.COMPLETED
    assert "plume_sentinel_prepared" in ctx.metadata
    datasets = ctx.metadata.get("plume_sentinel_input_datasets", [])
    names = [d["name"] for d in datasets]
    assert any("MODIS" in n for n in names)
    assert any("GOES" in n for n in names)
    assert any("HMS" in n for n in names)
    assert sum(c.args[0] for c in sleep.call_args_list) == pytest.approx(4.0, abs=0.5)


def test_demo_plot_stage_scans_existing_pngs(tmp_path):
    # Pre-populate output_dir with two PNGs as a previous run would.
    (tmp_path / "modis_aod_truecolor.png").write_bytes(b"\x89PNG")
    (tmp_path / "goes_hms_smoke.png").write_bytes(b"\x89PNG")
    (tmp_path / "ignore.txt").write_text("not a plot")

    stage = PlumeSentinelDemoPlotStage()
    ctx = _make_context(tmp_path)
    with patch("davinci_monet.addons.plume_sentinel.demo_stages.time.sleep") as sleep:
        result = stage.execute(ctx)
    assert result.status == StageStatus.COMPLETED
    plots = ctx.metadata.get("plume_sentinel_plots_generated", [])
    assert len(plots) == 2
    assert all(p.endswith(".png") for p in plots)
    assert all("ignore" not in p for p in plots)
    assert sum(c.args[0] for c in sleep.call_args_list) == pytest.approx(3.0, abs=0.5)


def test_demo_plot_stage_handles_empty_output_dir(tmp_path):
    stage = PlumeSentinelDemoPlotStage()
    ctx = _make_context(tmp_path)
    with patch("davinci_monet.addons.plume_sentinel.demo_stages.time.sleep"):
        result = stage.execute(ctx)
    assert result.status == StageStatus.COMPLETED
    assert ctx.metadata.get("plume_sentinel_plots_generated", []) == []
