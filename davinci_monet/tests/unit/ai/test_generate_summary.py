"""Unit tests for ai.summarizer.generate_summary with a stub client (no network)."""

from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

from davinci_monet.ai.payload import ImageRef, SummaryPayload
from davinci_monet.ai.summarizer import (
    SummaryError,
    SummaryResult,
    generate_summary,
)
from davinci_monet.config.schema import SummaryConfig


class _StubUsage:
    input_tokens = 1234
    output_tokens = 567


class _StubBlock:
    text = "## What this run is\nstub\n## Headline metrics\n## Interpretation\n## Caveats\n"


class _StubResponse:
    content = [_StubBlock()]
    usage = _StubUsage()
    model = "claude-haiku-4-5"


class _StubMessages:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return _StubResponse()


class _StubClient:
    def __init__(self) -> None:
        self.messages = _StubMessages()


def _png_path(tmp_path) -> str:
    p = tmp_path / "00_o3_scatter.png"
    arr = (np.random.default_rng(0).random((40, 50, 3)) * 255).astype("uint8")
    Image.fromarray(arr).save(p)
    return str(p)


def _payload(path: str) -> SummaryPayload:
    return SummaryPayload(
        period={"start": "2024-02-01", "end": "2024-02-03"},
        sources_summary=["cam (cesm_fv)"],
        pairs_summary=["cam_vs_airnow_o3"],
        stats_rows=[{"pair": "p", "variable": "O3", "metrics": {"N": 10}}],
        images=[ImageRef(caption="00_o3_scatter", path=path)],
        instructions=None,
    )


def test_generate_summary_with_injected_client(tmp_path) -> None:
    client = _StubClient()
    result = generate_summary(_payload(_png_path(tmp_path)), cfg=SummaryConfig(), client=client)
    assert isinstance(result, SummaryResult)
    assert "## Caveats" in result.markdown
    assert result.model == "claude-haiku-4-5"
    assert result.usage == {"input_tokens": 1234, "output_tokens": 567}
    assert result.images_sent == 1
    # the client was called with model + system + messages
    call = client.messages.calls[0]
    assert call["model"] == "claude-haiku-4-5"
    assert call["max_tokens"] == 2000
    assert isinstance(call["system"], list)
    assert call["messages"][0]["role"] == "user"


def test_generate_summary_missing_key_raises(monkeypatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(SummaryError):
        generate_summary(
            _payload("/nonexistent.png"), cfg=SummaryConfig(api_key_env="ANTHROPIC_API_KEY")
        )


def test_generate_summary_api_error_wrapped(tmp_path) -> None:
    class _BoomMessages:
        def create(self, **kwargs):
            raise RuntimeError("boom")

    class _BoomClient:
        messages = _BoomMessages()

    with pytest.raises(SummaryError, match="Claude API request failed"):
        generate_summary(_payload(_png_path(tmp_path)), cfg=SummaryConfig(), client=_BoomClient())


def test_generate_summary_routes_to_openrouter(monkeypatch, tmp_path) -> None:
    import davinci_monet.ai.openrouter as orouter
    from davinci_monet.config.schema import SummaryConfig

    keyfile = tmp_path / "k.api"
    keyfile.write_text("sk-or-test")
    cfg = SummaryConfig.model_validate({"provider": "openrouter", "api_key_file": str(keyfile)})

    def _fake_send(cfg_arg, key, body):
        return {
            "model": body["model"],
            "choices": [{"message": {"content": "## What this run is\nok\n## Caveats\n"}}],
            "usage": {"prompt_tokens": 7, "completion_tokens": 8},
        }

    monkeypatch.setattr(orouter, "_send_openrouter_request", _fake_send)

    result = generate_summary(_payload(_png_path(tmp_path)), cfg=cfg)
    assert "## Caveats" in result.markdown
    assert result.usage == {"input_tokens": 7, "output_tokens": 8}
    assert result.model == "anthropic/claude-haiku-4.5"
