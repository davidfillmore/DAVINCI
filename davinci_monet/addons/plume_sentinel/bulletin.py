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

from datetime import datetime
from typing import Any


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
