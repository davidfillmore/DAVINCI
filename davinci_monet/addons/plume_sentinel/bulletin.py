"""Bulletin generation helpers for the PlumeSentinel add-on.

Three responsibilities:
  * ``build_prompt`` — fill the deterministic placeholders in the bulletin
    template from a metrics payload, leaving the AI-analyzed placeholders
    in place.
  * ``generate_bulletin`` — call the Anthropic API with the prepared prompt
    and (optionally) image content blocks, returning the rendered bulletin.
  * ``publish_mqtt`` — publish the bulletin text to an MQTT broker.
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import anthropic
import paho.mqtt.client as paho_mqtt


def _julian_day(date_str: str) -> str:
    """Convert YYYY-MM-DD to a 7-digit YYYYDDD julian day string."""
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    return f"{dt.year}{dt.timetuple().tm_yday:03d}"


def _format_event_date_long(date_str: str) -> str:
    """Convert YYYY-MM-DD to e.g. 'September 9, 2020'."""
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    return dt.strftime("%B %-d, %Y")


def _sensor_sources(input_datasets: list[dict[str, Any]]) -> str:
    """Join input dataset 'name' values into a human-readable sensor list."""
    names = [d.get("name", "unknown") for d in input_datasets if isinstance(d, dict)]
    return ", ".join(names)


def _observation_time(input_datasets: list[dict[str, Any]]) -> str:
    """Pick the earliest valid_time from input_datasets, formatted as HH:MM UTC.

    Falls back to "unknown" if no parseable timestamps are present.
    """
    times: list[datetime] = []
    for d in input_datasets:
        if not isinstance(d, dict):
            continue
        vt = d.get("valid_time")
        if isinstance(vt, str):
            try:
                times.append(datetime.fromisoformat(vt.replace("Z", "+00:00")))
            except ValueError:
                continue
    if not times:
        return "unknown"
    times.sort()
    return f"~{times[0].strftime('%H:%M')} UTC"


def build_prompt(
    template_text: str,
    metrics_payload: dict[str, Any],
    *,
    issued_date: str,
) -> str:
    """Fill deterministic placeholders in the bulletin template.

    Parameters
    ----------
    template_text:
        Raw template content with ``{{PLACEHOLDER}}`` tokens.
    metrics_payload:
        A ``plumesentinel.metrics.v1`` payload (or partial). Reads
        ``event_date``, ``region``, and ``input_datasets``.
    issued_date:
        Human-formatted issue date (e.g. ``"May 14, 2026"``).

    Returns
    -------
    str
        Template with ``{{BULLETIN_ID}}``, ``{{ISSUED_DATE}}``,
        ``{{EVENT_DATE}}``, ``{{OBSERVATION_TIME}}``, ``{{SENSOR_SOURCES}}``
        replaced. AI-analyzed placeholders are left in place.
    """
    event_date = str(metrics_payload.get("event_date", "1970-01-01"))
    region_slug = str(metrics_payload.get("region", "unknown")).upper().replace("-", "")
    input_datasets = metrics_payload.get("input_datasets", []) or []

    bulletin_id = f"PS-{_julian_day(event_date)}-{region_slug}-001"
    long_date = _format_event_date_long(event_date)
    sensors = _sensor_sources(input_datasets)
    obs_time = _observation_time(input_datasets)

    out = template_text
    out = out.replace("{{BULLETIN_ID}}", bulletin_id)
    out = out.replace("{{ISSUED_DATE}}", issued_date)
    out = out.replace("{{EVENT_DATE}}", long_date)
    out = out.replace("{{OBSERVATION_TIME}}", obs_time)
    out = out.replace("{{SENSOR_SOURCES}}", sensors)
    return out


def publish_mqtt(
    *,
    text: str,
    broker: str,
    topic: str,
    port: int = 1883,
    qos: int = 0,
) -> None:
    """Publish ``text`` to ``topic`` on the given MQTT broker.

    Synchronous: connect, publish, disconnect. Raises on connect or publish
    failure; the caller is responsible for converting that into a
    ``quality_flag``.
    """
    client = paho_mqtt.Client()
    client.connect(broker, port, 60)
    try:
        client.publish(topic, payload=text, qos=qos)
    finally:
        client.disconnect()


PERSONA_SYSTEM_BLOCK = (
    "You are PlumeSentinel AI, an automated meteorological analysis system. "
    "You will be given a partial bulletin with {{PLACEHOLDER}} tokens already "
    "filled for the deterministic fields, plus a <metrics> block containing "
    "the quantitative event data. Replace every remaining {{PLACEHOLDER}} "
    "token with your authoritative analysis written in formal meteorological "
    "tone. Return ONLY the rendered bulletin text, with no preamble, no "
    "code fences, and no commentary."
)


@dataclass
class BulletinResponse:
    """Result of a Claude API bulletin call."""

    text: str
    model: str
    input_tokens: int
    cache_read_tokens: int
    output_tokens: int
    skipped_images: list[str] = field(default_factory=list)


def _encode_image_block(path: Path) -> dict[str, Any]:
    """Encode a PNG on disk as an Anthropic image content block."""
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": data},
    }


def generate_bulletin(
    *,
    prompt: str,
    metrics_json: dict[str, Any],
    image_paths: list[Path],
    model: str,
    api_key: str,
    max_tokens: int = 4096,
    temperature: float = 0.2,
) -> BulletinResponse:
    """Call the Anthropic API to render the bulletin from ``prompt`` + metrics.

    Builds a three-block cached system prompt (persona, partial bulletin,
    metrics JSON) plus a user message containing the directive and optional
    vision blocks for each image path on disk.
    """
    metrics_text = json.dumps(
        metrics_json, indent=2, sort_keys=True, default=str, ensure_ascii=False
    )

    system_blocks = [
        {"type": "text", "text": PERSONA_SYSTEM_BLOCK},
        {"type": "text", "text": prompt},
        {
            "type": "text",
            "text": f"<metrics>\n{metrics_text}\n</metrics>",
            "cache_control": {"type": "ephemeral"},
        },
    ]

    user_content: list[dict[str, Any]] = []
    skipped: list[str] = []
    for p in image_paths:
        if not p.is_file():
            skipped.append(str(p))
            continue
        user_content.append(_encode_image_block(p))
    user_content.append({"type": "text", "text": "Render the bulletin for this event."})

    client = anthropic.Anthropic(api_key=api_key)
    resp = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        system=system_blocks,
        messages=[{"role": "user", "content": user_content}],
    )

    text_parts = [b.text for b in resp.content if getattr(b, "type", None) == "text"]
    text = "".join(text_parts)

    cache_read = getattr(resp.usage, "cache_read_input_tokens", 0) or 0
    return BulletinResponse(
        text=text,
        model=resp.model,
        input_tokens=resp.usage.input_tokens,
        cache_read_tokens=cache_read,
        output_tokens=resp.usage.output_tokens,
        skipped_images=skipped,
    )
