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
