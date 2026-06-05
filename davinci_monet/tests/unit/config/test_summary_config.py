"""Unit tests for SummaryConfig and the MonetConfig.summary field."""

from __future__ import annotations

from davinci_monet.config.schema import MonetConfig, SummaryConfig


def test_summary_config_defaults() -> None:
    cfg = SummaryConfig()
    assert cfg.enabled is False
    assert cfg.model == "claude-haiku-4-5"
    assert cfg.max_tokens == 2000
    assert cfg.api_key_env == "ANTHROPIC_API_KEY"
    assert cfg.plots is None
    assert cfg.max_images == 8
    assert cfg.output_filename == "AI_summary.md"
    assert cfg.instructions is None


def test_summary_config_overrides() -> None:
    cfg = SummaryConfig.model_validate(
        {
            "enabled": True,
            "model": "claude-sonnet-4-6",
            "plots": ["pm25_spatial_bias", "o3_scatter"],
            "max_images": 3,
            "instructions": "Focus on coastal sites.",
        }
    )
    assert cfg.enabled is True
    assert cfg.model == "claude-sonnet-4-6"
    assert cfg.plots == ["pm25_spatial_bias", "o3_scatter"]
    assert cfg.max_images == 3
    assert cfg.instructions == "Focus on coastal sites."


def test_monetconfig_summary_field_defaults_none() -> None:
    cfg = MonetConfig.model_validate(
        {"analysis": {"start_time": "2024-01-01", "end_time": "2024-01-02"}}
    )
    assert cfg.summary is None


def test_monetconfig_parses_summary_block() -> None:
    cfg = MonetConfig.model_validate(
        {
            "analysis": {"start_time": "2024-01-01", "end_time": "2024-01-02"},
            "summary": {"enabled": True, "model": "claude-haiku-4-5"},
        }
    )
    assert cfg.summary is not None
    assert cfg.summary.enabled is True
    assert cfg.summary.model == "claude-haiku-4-5"
    # model_dump round-trips to a plain dict for the pipeline
    dumped = cfg.model_dump()
    assert dumped["summary"]["enabled"] is True
