"""Unit tests for the plume_sentinel bulletin helpers."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from davinci_monet.addons.plume_sentinel.bulletin import build_prompt, publish_mqtt

TEMPLATE = """BULLETIN ID: {{BULLETIN_ID}}
ISSUED: {{ISSUED_DATE}}
EVENT DATE: {{EVENT_DATE}}
OBSERVATION TIME: {{OBSERVATION_TIME}}
SENSORS: {{SENSOR_SOURCES}}

REGION: {{REGION}}
SEVERITY: {{SEVERITY}}

{{SYNOPTIC_OVERVIEW}}
"""


@pytest.fixture
def metrics_payload() -> dict:
    return {
        "schema": "plumesentinel.metrics.v1",
        "run_id": "2020-09-09-westcoast-modis",
        "region": "westcoast",
        "config_slug": "modis-aod-truecolor",
        "event_date": "2020-09-09",
        "valid_time": "2020-09-09T19:05:00+00:00",
        "produced_at": "2026-05-14T00:00:00+00:00",
        "metrics": {"peak_aod": 4.2},
        "plot_urls": {"modis_aod_truecolor": "http://localhost:8080/x.png"},
        "input_datasets": [
            {"name": "MODIS L2 AOD (MOD04)", "valid_time": "2020-09-09T19:05:00+00:00"},
            {"name": "GOES-16 ABI L2 MCMIP", "valid_time": "2020-09-09T20:01:00+00:00"},
        ],
    }


def test_build_prompt_fills_deterministic_placeholders(metrics_payload):
    out = build_prompt(TEMPLATE, metrics_payload, issued_date="May 14, 2026")
    assert "{{BULLETIN_ID}}" not in out
    assert "PS-2020253-WESTCOAST-001" in out
    assert "May 14, 2026" in out
    assert "September 9, 2020" in out  # human-formatted EVENT_DATE
    assert "MODIS L2 AOD" in out
    assert "GOES-16 ABI" in out


def test_build_prompt_leaves_ai_placeholders(metrics_payload):
    out = build_prompt(TEMPLATE, metrics_payload, issued_date="May 14, 2026")
    assert "{{REGION}}" in out
    assert "{{SEVERITY}}" in out
    assert "{{SYNOPTIC_OVERVIEW}}" in out


def test_build_prompt_handles_missing_input_datasets():
    payload = {
        "event_date": "2020-09-09",
        "region": "westcoast",
        "valid_time": "2020-09-09T19:05:00+00:00",
        "input_datasets": [],
    }
    out = build_prompt(TEMPLATE, payload, issued_date="May 14, 2026")
    # Falls back to "unknown" or empty rather than crashing
    assert "{{SENSOR_SOURCES}}" not in out


def test_publish_mqtt_uses_config():
    fake_client = MagicMock()
    with patch(
        "davinci_monet.addons.plume_sentinel.bulletin.paho_mqtt.Client",
        return_value=fake_client,
    ):
        publish_mqtt(
            text="HELLO",
            broker="broker.example.com",
            topic="t/test",
            port=1884,
            qos=1,
        )
    fake_client.connect.assert_called_once_with("broker.example.com", 1884, 60)
    fake_client.publish.assert_called_once_with("t/test", payload="HELLO", qos=1)
    fake_client.disconnect.assert_called_once()


def test_publish_mqtt_raises_on_connect_failure():
    fake_client = MagicMock()
    fake_client.connect.side_effect = OSError("broker unreachable")
    with patch(
        "davinci_monet.addons.plume_sentinel.bulletin.paho_mqtt.Client",
        return_value=fake_client,
    ):
        with pytest.raises(OSError, match="broker unreachable"):
            publish_mqtt(
                text="HELLO",
                broker="broker.example.com",
                topic="t/test",
            )
    fake_client.publish.assert_not_called()


import base64
import json as _json
from pathlib import Path
from types import SimpleNamespace

from davinci_monet.addons.plume_sentinel.bulletin import (
    BulletinResponse,
    generate_bulletin,
)


def _fake_anthropic_response(text="RENDERED", input_tok=100, cache_read=80, out_tok=50):
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text=text)],
        model="claude-sonnet-4-6",
        usage=SimpleNamespace(
            input_tokens=input_tok,
            cache_read_input_tokens=cache_read,
            output_tokens=out_tok,
        ),
    )


def test_generate_bulletin_returns_response_and_token_counts():
    fake_messages = MagicMock()
    fake_messages.create.return_value = _fake_anthropic_response()
    fake_client = SimpleNamespace(messages=fake_messages)
    with patch(
        "davinci_monet.addons.plume_sentinel.bulletin.anthropic.Anthropic",
        return_value=fake_client,
    ):
        resp = generate_bulletin(
            prompt="prepared partial bulletin",
            metrics_json={"event_date": "2020-09-09"},
            image_paths=[],
            model="claude-sonnet-4-6",
            api_key="sk-fake",
        )
    assert isinstance(resp, BulletinResponse)
    assert resp.text == "RENDERED"
    assert resp.model == "claude-sonnet-4-6"
    assert resp.input_tokens == 100
    assert resp.cache_read_tokens == 80
    assert resp.output_tokens == 50

    args, kwargs = fake_messages.create.call_args
    # System content is a list with cache_control on the last block
    assert isinstance(kwargs["system"], list)
    assert kwargs["system"][-1]["cache_control"] == {"type": "ephemeral"}
    # Three system blocks: persona, partial bulletin, metrics JSON
    assert len(kwargs["system"]) == 3


def test_generate_bulletin_attaches_image_blocks_when_present(tmp_path: Path):
    png_path = tmp_path / "plot.png"
    png_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16
    png_path.write_bytes(png_bytes)

    fake_messages = MagicMock()
    fake_messages.create.return_value = _fake_anthropic_response()
    fake_client = SimpleNamespace(messages=fake_messages)
    with patch(
        "davinci_monet.addons.plume_sentinel.bulletin.anthropic.Anthropic",
        return_value=fake_client,
    ):
        generate_bulletin(
            prompt="x",
            metrics_json={},
            image_paths=[png_path],
            model="claude-sonnet-4-6",
            api_key="sk-fake",
        )
    _args, kwargs = fake_messages.create.call_args
    user_content = kwargs["messages"][0]["content"]
    image_blocks = [b for b in user_content if isinstance(b, dict) and b.get("type") == "image"]
    assert len(image_blocks) == 1
    assert image_blocks[0]["source"]["media_type"] == "image/png"
    assert image_blocks[0]["source"]["data"] == base64.b64encode(png_bytes).decode("ascii")


def test_generate_bulletin_skips_missing_images(tmp_path: Path):
    missing = tmp_path / "nope.png"  # never written
    fake_messages = MagicMock()
    fake_messages.create.return_value = _fake_anthropic_response()
    fake_client = SimpleNamespace(messages=fake_messages)
    with patch(
        "davinci_monet.addons.plume_sentinel.bulletin.anthropic.Anthropic",
        return_value=fake_client,
    ):
        resp = generate_bulletin(
            prompt="x",
            metrics_json={},
            image_paths=[missing],
            model="claude-sonnet-4-6",
            api_key="sk-fake",
        )
    assert resp.skipped_images == [str(missing)]
    _args, kwargs = fake_messages.create.call_args
    user_content = kwargs["messages"][0]["content"]
    image_blocks = [b for b in user_content if isinstance(b, dict) and b.get("type") == "image"]
    assert image_blocks == []
