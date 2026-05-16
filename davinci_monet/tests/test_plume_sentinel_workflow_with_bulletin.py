"""End-to-end test: plume_sentinel workflow with the bulletin stage."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from davinci_monet.addons.plume_sentinel.bulletin import BulletinResponse
from davinci_monet.addons.plume_sentinel.workflow import (
    create_plume_sentinel_pipeline,
)


@pytest.fixture
def minimal_config(tmp_path):
    """A minimal config that exercises the full plume_sentinel pipeline."""
    return {
        "analysis": {
            "output_dir": str(tmp_path),
            "start_time": "2020-09-09",
            "end_time": "2020-09-09",
            "workflow": "plume_sentinel",
        },
        "plume_sentinel": {
            "inputs": {},
            "plots": {},
            "bulletin": {
                "output_filename": "bulletin.txt",
                "model": "claude-sonnet-4-6",
                "include_images": False,
            },
        },
    }


def test_create_plume_sentinel_pipeline_includes_bulletin_stage():
    pipeline = create_plume_sentinel_pipeline()
    names = [s.name for s in pipeline]
    assert names == ["load_inputs", "prepare_geospatial", "plotting", "bulletin"]


def test_workflow_writes_bulletin_when_block_present(minimal_config, tmp_path, monkeypatch):
    from davinci_monet.pipeline.runner import PipelineRunner

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-fake")
    fake_resp = BulletinResponse(
        text="==BULLETIN==",
        model="claude-sonnet-4-6",
        input_tokens=10,
        cache_read_tokens=0,
        output_tokens=5,
    )
    with patch(
        "davinci_monet.addons.plume_sentinel.stages.generate_bulletin",
        return_value=fake_resp,
    ):
        runner = PipelineRunner(show_progress=False)
        result = runner.run_from_config(minimal_config)
    assert result.success
    assert (tmp_path / "bulletin.txt").read_text() == "==BULLETIN=="


def test_workflow_omits_bulletin_when_block_absent(minimal_config, tmp_path):
    from davinci_monet.pipeline.runner import PipelineRunner

    cfg = json.loads(json.dumps(minimal_config))
    cfg["plume_sentinel"].pop("bulletin")
    runner = PipelineRunner(show_progress=False)
    result = runner.run_from_config(cfg)
    assert result.success
    assert not (tmp_path / "bulletin.txt").exists()


def test_metrics_payload_includes_bulletin_field_when_present(
    minimal_config, tmp_path, monkeypatch
):
    from davinci_monet.addons.plume_sentinel.workflow import run as run_workflow

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-fake")
    metrics_path = tmp_path / "metrics.json"
    fake_resp = BulletinResponse(
        text="==BULLETIN==",
        model="claude-sonnet-4-6",
        input_tokens=10,
        cache_read_tokens=8,
        output_tokens=5,
    )
    with patch(
        "davinci_monet.addons.plume_sentinel.stages.generate_bulletin",
        return_value=fake_resp,
    ):
        run_workflow(
            minimal_config,
            emit_metrics_json=metrics_path,
            region="westcoast",
            config_slug="modis-aod-truecolor",
        )
    payload = json.loads(metrics_path.read_text())
    assert "bulletin" in payload
    assert payload["bulletin"]["model"] == "claude-sonnet-4-6"
    assert payload["bulletin"]["input_tokens"] == 10
    assert payload["bulletin"]["cache_read_tokens"] == 8
    assert payload["bulletin"]["output_tokens"] == 5
    assert payload["bulletin"]["path"].endswith("bulletin.txt")


def test_create_plume_sentinel_pipeline_demo_mode_uses_demo_stages():
    from davinci_monet.addons.plume_sentinel.demo_stages import (
        PlumeSentinelDemoLoadStage,
        PlumeSentinelDemoPlotStage,
        PlumeSentinelDemoPrepareStage,
    )

    pipeline = create_plume_sentinel_pipeline(demo_mode=True)
    types = [type(s) for s in pipeline]
    assert types[0] is PlumeSentinelDemoLoadStage
    assert types[1] is PlumeSentinelDemoPrepareStage
    assert types[2] is PlumeSentinelDemoPlotStage
    # Bulletin stage is the same in both modes
    assert pipeline[3].name == "bulletin"


def test_runner_dispatches_to_demo_pipeline_when_flag_set(tmp_path, monkeypatch):
    """When config.analysis._demo.enabled is True, runner uses demo stages."""
    from davinci_monet.pipeline.runner import PipelineRunner

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-fake")
    (tmp_path / "modis_aod_truecolor.png").write_bytes(b"\x89PNG")
    config = {
        "analysis": {
            "output_dir": str(tmp_path),
            "start_time": "2020-09-09",
            "end_time": "2020-09-09",
            "workflow": "plume_sentinel",
            "_demo": {"enabled": True, "canned_bulletin": None},
        },
        "plume_sentinel": {"inputs": {}, "plots": {}, "bulletin": {"include_images": False}},
    }
    fake_resp = BulletinResponse(
        text="==DEMO BULLETIN==",
        model="claude-sonnet-4-6",
        input_tokens=1,
        cache_read_tokens=0,
        output_tokens=1,
    )
    with (
        patch(
            "davinci_monet.addons.plume_sentinel.stages.generate_bulletin",
            return_value=fake_resp,
        ),
        patch("davinci_monet.addons.plume_sentinel.demo_stages.time.sleep"),
    ):
        runner = PipelineRunner(show_progress=False)
        result = runner.run_from_config(config)
    assert result.success
    assert (tmp_path / "bulletin.txt").read_text() == "==DEMO BULLETIN=="
