"""Unit tests for PlumeSentinelBulletinStage."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import anthropic as _anthropic_module

from davinci_monet.addons.plume_sentinel.bulletin import BulletinResponse
from davinci_monet.addons.plume_sentinel.schema import PlumeSentinelConfig
from davinci_monet.addons.plume_sentinel.stages import PlumeSentinelBulletinStage
from davinci_monet.pipeline.stages import PipelineContext, StageStatus


def _make_context(tmp_path: Path, *, bulletin: dict | None = None) -> PipelineContext:
    """Build a minimal pipeline context that mimics post-plot state."""
    config: dict = {
        "analysis": {"output_dir": str(tmp_path), "start_time": "2020-09-09"},
        "plume_sentinel": {
            "inputs": {},
            "plots": {},
            **({"bulletin": bulletin} if bulletin is not None else {}),
        },
    }
    ctx = PipelineContext(config=config)
    ctx.metadata["plume_sentinel_config"] = PlumeSentinelConfig(**config["plume_sentinel"])
    ctx.metadata["plume_sentinel_prepared"] = {}
    ctx.metadata["plume_sentinel_plots_generated"] = []
    return ctx


def test_stage_no_op_when_no_bulletin_config(tmp_path):
    stage = PlumeSentinelBulletinStage()
    ctx = _make_context(tmp_path, bulletin=None)
    result = stage.execute(ctx)
    assert result.status == StageStatus.COMPLETED
    assert result.data.get("bulletin") == "skipped (no config)"
    assert not (tmp_path / "bulletin.txt").exists()


def test_stage_skips_when_api_key_env_missing(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    stage = PlumeSentinelBulletinStage()
    ctx = _make_context(tmp_path, bulletin={})
    result = stage.execute(ctx)
    assert result.status == StageStatus.COMPLETED
    assert result.data.get("bulletin") == "skipped (no api key)"
    flags = ctx.metadata.get("plume_sentinel_quality_flags", [])
    assert any(f.get("category") == "bulletin" and "not set" in f.get("message", "") for f in flags)
    assert not (tmp_path / "bulletin.txt").exists()


def test_stage_writes_file_on_success(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-fake")
    stage = PlumeSentinelBulletinStage()
    ctx = _make_context(tmp_path, bulletin={})
    fake_resp = BulletinResponse(
        text="RENDERED BULLETIN BODY",
        model="claude-sonnet-4-6",
        input_tokens=120,
        cache_read_tokens=100,
        output_tokens=60,
    )
    with (
        patch(
            "davinci_monet.addons.plume_sentinel.stages.generate_bulletin",
            return_value=fake_resp,
        ) as gen,
        patch(
            "davinci_monet.addons.plume_sentinel.stages.build_metrics_payload",
            return_value={"event_date": "2020-09-09", "region": "westcoast", "input_datasets": []},
        ),
    ):
        result = stage.execute(ctx)
    assert result.status == StageStatus.COMPLETED
    bulletin_path = tmp_path / "bulletin.txt"
    assert bulletin_path.is_file()
    assert bulletin_path.read_text() == "RENDERED BULLETIN BODY"
    assert result.data["bulletin_path"] == str(bulletin_path)
    assert result.data["input_tokens"] == 120
    assert result.data["cache_read_tokens"] == 100
    assert result.data["output_tokens"] == 60
    gen.assert_called_once()


def test_stage_passes_images_when_include_images_true(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-fake")
    stage = PlumeSentinelBulletinStage()
    ctx = _make_context(tmp_path, bulletin={"include_images": True})
    plot1 = tmp_path / "modis_aod_truecolor.png"
    plot1.write_bytes(b"\x89PNG")
    ctx.metadata["plume_sentinel_plots_generated"] = [str(plot1)]
    fake_resp = BulletinResponse(
        text="BODY",
        model="claude-sonnet-4-6",
        input_tokens=1,
        cache_read_tokens=0,
        output_tokens=1,
    )
    with (
        patch(
            "davinci_monet.addons.plume_sentinel.stages.generate_bulletin",
            return_value=fake_resp,
        ) as gen,
        patch(
            "davinci_monet.addons.plume_sentinel.stages.build_metrics_payload",
            return_value={"event_date": "2020-09-09", "region": "westcoast", "input_datasets": []},
        ),
    ):
        stage.execute(ctx)
    _args, kwargs = gen.call_args
    assert kwargs["image_paths"] == [plot1]


def test_stage_omits_images_when_include_images_false(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-fake")
    stage = PlumeSentinelBulletinStage()
    ctx = _make_context(tmp_path, bulletin={"include_images": False})
    plot1 = tmp_path / "p.png"
    plot1.write_bytes(b"\x89PNG")
    ctx.metadata["plume_sentinel_plots_generated"] = [str(plot1)]
    fake_resp = BulletinResponse(
        text="BODY",
        model="claude-sonnet-4-6",
        input_tokens=1,
        cache_read_tokens=0,
        output_tokens=1,
    )
    with (
        patch(
            "davinci_monet.addons.plume_sentinel.stages.generate_bulletin",
            return_value=fake_resp,
        ) as gen,
        patch(
            "davinci_monet.addons.plume_sentinel.stages.build_metrics_payload",
            return_value={"event_date": "2020-09-09", "region": "westcoast", "input_datasets": []},
        ),
    ):
        stage.execute(ctx)
    _args, kwargs = gen.call_args
    assert kwargs["image_paths"] == []


def test_stage_records_quality_flag_on_api_error(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-fake")
    stage = PlumeSentinelBulletinStage()
    ctx = _make_context(tmp_path, bulletin={})
    with (
        patch(
            "davinci_monet.addons.plume_sentinel.stages.build_metrics_payload",
            return_value={"event_date": "2020-09-09", "region": "westcoast", "input_datasets": []},
        ),
        patch(
            "davinci_monet.addons.plume_sentinel.stages.generate_bulletin",
            side_effect=_anthropic_module.APIError("boom", request=None, body=None),
        ),
    ):
        result = stage.execute(ctx)
    assert result.status == StageStatus.COMPLETED
    assert result.data.get("bulletin") == "skipped (api error)"
    flags = ctx.metadata.get("plume_sentinel_quality_flags", [])
    assert any(f["category"] == "bulletin" and "API call failed" in f["message"] for f in flags)
    assert not (tmp_path / "bulletin.txt").exists()


def test_stage_records_quality_flag_on_mqtt_error(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-fake")
    stage = PlumeSentinelBulletinStage()
    ctx = _make_context(
        tmp_path,
        bulletin={"mqtt": {"topic": "t/test", "broker": "broker.example.com"}},
    )
    fake_resp = BulletinResponse(
        text="HELLO",
        model="claude-sonnet-4-6",
        input_tokens=1,
        cache_read_tokens=0,
        output_tokens=1,
    )
    with (
        patch(
            "davinci_monet.addons.plume_sentinel.stages.build_metrics_payload",
            return_value={"event_date": "2020-09-09", "region": "westcoast", "input_datasets": []},
        ),
        patch(
            "davinci_monet.addons.plume_sentinel.stages.generate_bulletin",
            return_value=fake_resp,
        ),
        patch(
            "davinci_monet.addons.plume_sentinel.stages.publish_mqtt",
            side_effect=OSError("broker unreachable"),
        ),
    ):
        result = stage.execute(ctx)
    assert result.status == StageStatus.COMPLETED
    # File still written
    assert (tmp_path / "bulletin.txt").read_text() == "HELLO"
    assert result.data["mqtt_published"] is False
    flags = ctx.metadata.get("plume_sentinel_quality_flags", [])
    assert any(f["category"] == "bulletin" and "MQTT publish" in f["message"] for f in flags)


def test_stage_records_quality_flag_on_missing_template(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-fake")
    stage = PlumeSentinelBulletinStage()
    ctx = _make_context(
        tmp_path,
        bulletin={"template": str(tmp_path / "nope.template")},
    )
    with patch(
        "davinci_monet.addons.plume_sentinel.stages.build_metrics_payload",
        return_value={"event_date": "2020-09-09", "region": "westcoast", "input_datasets": []},
    ):
        result = stage.execute(ctx)
    assert result.status == StageStatus.COMPLETED
    assert result.data.get("bulletin") == "skipped (template missing)"
    flags = ctx.metadata.get("plume_sentinel_quality_flags", [])
    assert any(f["category"] == "bulletin" and "template not found" in f["message"] for f in flags)
