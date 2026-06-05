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


def test_summary_config_provider_defaults_to_anthropic() -> None:
    cfg = SummaryConfig()
    assert cfg.provider == "anthropic"
    assert cfg.api_key_file is None
    # anthropic defaults are untouched
    assert cfg.model == "claude-haiku-4-5"
    assert cfg.api_key_env == "ANTHROPIC_API_KEY"


def test_summary_config_openrouter_flips_defaults() -> None:
    cfg = SummaryConfig.model_validate({"provider": "openrouter"})
    assert cfg.provider == "openrouter"
    # sentinels flip to OpenRouter-appropriate defaults
    assert cfg.model == "anthropic/claude-3.5-haiku"
    assert cfg.api_key_env == "OPENROUTER_API_KEY"


def test_summary_config_openrouter_preserves_explicit_values() -> None:
    cfg = SummaryConfig.model_validate(
        {
            "provider": "openrouter",
            "model": "anthropic/claude-sonnet-4",
            "api_key_env": "MY_KEY",
            "api_key_file": "OpenRouter.api",
        }
    )
    assert cfg.model == "anthropic/claude-sonnet-4"
    assert cfg.api_key_env == "MY_KEY"
    assert cfg.api_key_file == "OpenRouter.api"
