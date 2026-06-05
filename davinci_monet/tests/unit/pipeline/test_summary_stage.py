"""Unit tests for SummaryStage (engine mocked via monkeypatch)."""

from __future__ import annotations

from pathlib import Path

import davinci_monet.ai.summarizer as summarizer_mod
from davinci_monet.ai.summarizer import SummaryError, SummaryResult
from davinci_monet.pipeline.stages import (
    PipelineContext,
    StageStatus,
    SummaryStage,
)


def _ctx(tmp_path: Path, enabled: bool = True) -> PipelineContext:
    return PipelineContext(
        config={
            "analysis": {
                "start_time": "2024-02-01",
                "end_time": "2024-02-03",
                "output_dir": str(tmp_path / "output"),
            },
            "summary": {"enabled": enabled},
        }
    )


def test_summary_stage_disabled_is_skipped(tmp_path: Path) -> None:
    result = SummaryStage().execute(_ctx(tmp_path, enabled=False))
    assert result.status == StageStatus.SKIPPED


def test_summary_stage_writes_file(monkeypatch, tmp_path: Path) -> None:
    def _fake_client(cfg):
        class _Msgs:
            def create(self, **kwargs):
                class _Block:
                    text = "## What this run is\nok\n## Caveats\n"

                class _Usage:
                    input_tokens = 5
                    output_tokens = 6

                class _Resp:
                    content = [_Block()]
                    usage = _Usage()
                    model = cfg.model

                return _Resp()

        class _Client:
            messages = _Msgs()

        return _Client()

    monkeypatch.setattr(summarizer_mod, "_build_client", _fake_client)

    result = SummaryStage().execute(_ctx(tmp_path))
    assert result.status == StageStatus.COMPLETED
    out = Path(result.data["summary_file"])
    assert out.exists()
    assert out.name == "AI_summary.md"
    assert "## Caveats" in out.read_text()
    assert result.data["usage"] == {"input_tokens": 5, "output_tokens": 6}


def test_summary_stage_error_is_nonfatal(monkeypatch, tmp_path: Path) -> None:
    def _boom(cfg):
        raise SummaryError("no key")

    monkeypatch.setattr(summarizer_mod, "_build_client", _boom)

    result = SummaryStage().execute(_ctx(tmp_path))
    assert result.status == StageStatus.SKIPPED
    assert "no key" in str(result.data)


def test_summary_stage_unexpected_error_is_nonfatal(monkeypatch, tmp_path: Path) -> None:
    """A non-SummaryError failure must still degrade to SKIPPED, never FAILED."""

    def _boom(cfg):
        raise RuntimeError("kaboom")

    monkeypatch.setattr(summarizer_mod, "_build_client", _boom)

    result = SummaryStage().execute(_ctx(tmp_path))
    assert result.status == StageStatus.SKIPPED
    assert "kaboom" in str(result.data)


def test_summary_stage_invalid_config_is_nonfatal(tmp_path: Path) -> None:
    """A malformed summary config (validation error) must skip, not fail the run."""
    ctx = PipelineContext(
        config={
            "analysis": {
                "start_time": "2024-02-01",
                "end_time": "2024-02-03",
                "output_dir": str(tmp_path / "output"),
            },
            # max_images must be an int; this fails SummaryConfig validation
            "summary": {"enabled": True, "max_images": "not-an-int"},
        }
    )
    result = SummaryStage().execute(ctx)
    assert result.status == StageStatus.SKIPPED
