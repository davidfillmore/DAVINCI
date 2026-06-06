"""Unit tests for the OpenRouter provider (no network)."""

from __future__ import annotations

from pathlib import Path

import pytest

import davinci_monet.ai.openrouter as orouter
from davinci_monet.ai.images import EncodedImage
from davinci_monet.ai.openrouter import build_openrouter_messages, call_openrouter
from davinci_monet.ai.summarizer import SummaryError, SummaryResult
from davinci_monet.config.schema import SummaryConfig


def test_build_openrouter_messages_shape() -> None:
    encoded = [("01_o3_scatter", EncodedImage(media_type="image/png", data="QUJD"))]
    messages = build_openrouter_messages("SYS", "USER TEXT", encoded)
    assert messages[0] == {"role": "system", "content": "SYS"}
    user = messages[1]
    assert user["role"] == "user"
    assert user["content"][0] == {"type": "text", "text": "USER TEXT"}
    assert user["content"][1] == {"type": "text", "text": "Figure: 01_o3_scatter"}
    assert user["content"][2] == {
        "type": "image_url",
        "image_url": {"url": "data:image/png;base64,QUJD"},
    }


def _canned() -> dict:
    return {
        "model": "anthropic/claude-haiku-4.5",
        "choices": [
            {"message": {"role": "assistant", "content": "## What this run is\nx\n## Caveats\n"}}
        ],
        "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
    }


def test_call_openrouter_maps_response(monkeypatch, tmp_path: Path) -> None:
    keyfile = tmp_path / "k.api"
    keyfile.write_text("sk-or-test")
    cfg = SummaryConfig.model_validate({"provider": "openrouter", "api_key_file": str(keyfile)})

    captured = {}

    def _fake_send(cfg_arg, key, body):
        captured["key"] = key
        captured["body"] = body
        return _canned()

    monkeypatch.setattr(orouter, "_send_openrouter_request", _fake_send)
    monkeypatch.setattr(orouter, "_fetch_credits_remaining", lambda c, k: None)

    encoded = [("fig", EncodedImage(media_type="image/png", data="QUJD"))]
    result = call_openrouter("SYS", "USER", encoded, cfg)

    assert isinstance(result, SummaryResult)
    assert "## Caveats" in result.markdown
    assert result.model == "anthropic/claude-haiku-4.5"
    assert result.usage == {"input_tokens": 100, "output_tokens": 50}
    assert result.images_sent == 1
    assert captured["key"] == "sk-or-test"
    assert captured["body"]["model"] == "anthropic/claude-haiku-4.5"
    assert captured["body"]["max_tokens"] == 2000


def test_call_openrouter_malformed_response_raises(monkeypatch, tmp_path: Path) -> None:
    keyfile = tmp_path / "k.api"
    keyfile.write_text("sk-or-test")
    cfg = SummaryConfig.model_validate({"provider": "openrouter", "api_key_file": str(keyfile)})
    monkeypatch.setattr(orouter, "_send_openrouter_request", lambda c, k, b: {"oops": 1})

    with pytest.raises(SummaryError, match="Unexpected OpenRouter response shape"):
        call_openrouter("SYS", "USER", [], cfg)


def test_fetch_credits_remaining_parses(monkeypatch, tmp_path: Path) -> None:
    class _Resp:
        status_code = 200

        def json(self):
            return {"data": {"limit_remaining": 99.97}}

    monkeypatch.setattr("httpx.get", lambda *a, **k: _Resp())
    cfg = SummaryConfig.model_validate({"provider": "openrouter"})
    assert orouter._fetch_credits_remaining(cfg, "sk-or-test") == 99.97


def test_fetch_credits_remaining_none_on_non_200(monkeypatch) -> None:
    class _Resp:
        status_code = 402

        def json(self):
            return {}

    monkeypatch.setattr("httpx.get", lambda *a, **k: _Resp())
    cfg = SummaryConfig.model_validate({"provider": "openrouter"})
    assert orouter._fetch_credits_remaining(cfg, "k") is None


def test_fetch_credits_remaining_none_on_missing_field(monkeypatch) -> None:
    class _Resp:
        status_code = 200

        def json(self):
            return {"data": {}}

    monkeypatch.setattr("httpx.get", lambda *a, **k: _Resp())
    cfg = SummaryConfig.model_validate({"provider": "openrouter"})
    assert orouter._fetch_credits_remaining(cfg, "k") is None


def test_fetch_credits_remaining_none_on_error(monkeypatch) -> None:
    def _boom(*a, **k):
        raise RuntimeError("network down")

    monkeypatch.setattr("httpx.get", _boom)
    cfg = SummaryConfig.model_validate({"provider": "openrouter"})
    assert orouter._fetch_credits_remaining(cfg, "k") is None


def test_call_openrouter_sets_credits(monkeypatch, tmp_path: Path) -> None:
    keyfile = tmp_path / "k.api"
    keyfile.write_text("sk-or-test")
    cfg = SummaryConfig.model_validate({"provider": "openrouter", "api_key_file": str(keyfile)})
    monkeypatch.setattr(orouter, "_send_openrouter_request", lambda c, k, b: _canned())
    monkeypatch.setattr(orouter, "_fetch_credits_remaining", lambda c, k: 42.0)

    result = call_openrouter("SYS", "USER", [], cfg)
    assert result.credits_remaining == 42.0
