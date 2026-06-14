"""Unit tests for ai.summarizer.resolve_api_key."""

from __future__ import annotations

from pathlib import Path

import pytest

from davinci_monet.ai.summarizer import SummaryError, resolve_api_key
from davinci_monet.config.schema import SummaryConfig
from davinci_monet.core.schema_utils import validate_schema


def test_resolve_from_file_stripped(tmp_path: Path) -> None:
    p = tmp_path / "key.api"
    p.write_text("sk-or-secret\n")
    cfg = validate_schema(SummaryConfig, {"provider": "openrouter", "api_key_file": str(p)})
    assert resolve_api_key(cfg) == "sk-or-secret"


def test_resolve_missing_file_raises(tmp_path: Path) -> None:
    cfg = validate_schema(
        SummaryConfig, {"provider": "openrouter", "api_key_file": str(tmp_path / "nope.api")}
    )
    with pytest.raises(SummaryError, match="api_key_file"):
        resolve_api_key(cfg)


def test_resolve_empty_file_raises(tmp_path: Path) -> None:
    p = tmp_path / "empty.api"
    p.write_text("   \n")
    cfg = validate_schema(SummaryConfig, {"provider": "openrouter", "api_key_file": str(p)})
    with pytest.raises(SummaryError, match="empty"):
        resolve_api_key(cfg)


def test_resolve_from_env(monkeypatch) -> None:
    monkeypatch.setenv("MY_TEST_KEY", "env-secret")
    cfg = validate_schema(SummaryConfig, {"api_key_env": "MY_TEST_KEY"})
    assert resolve_api_key(cfg) == "env-secret"


def test_resolve_none_raises(monkeypatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    cfg = SummaryConfig()  # no file, default env unset
    with pytest.raises(SummaryError, match="API key not found"):
        resolve_api_key(cfg)
