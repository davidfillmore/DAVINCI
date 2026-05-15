# Plume Sentinel Bulletin Stage Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `PlumeSentinelBulletinStage` to the plume_sentinel addon that generates a meteorological bulletin via the Anthropic Claude API and optionally publishes it to MQTT, all from inside DAVINCI with no interactive Claude Code session.

**Architecture:** New stage appended after `PlumeSentinelPlotStage` in `create_plume_sentinel_pipeline()`. A new `bulletin.py` module hosts pure helpers (`build_prompt`, `generate_bulletin`, `publish_mqtt`). Schema gains an optional `bulletin:` block; absent block → stage no-ops. All failures warn-and-continue via `quality_flags`. Anthropic system prompt uses prompt caching across persona + partial bulletin + metrics JSON; user message holds the directive and optional vision blocks.

**Tech Stack:** Python 3.11, Pydantic v2, `anthropic` SDK, `paho-mqtt`, `importlib.resources`, pytest with `unittest.mock`.

**Spec:** `docs/superpowers/specs/2026-05-14-plume-sentinel-bulletin-stage-design.md`

---

## File Structure

**Create:**
- `davinci_monet/addons/plume_sentinel/metrics_payload.py` — extracted from `workflow.py`
- `davinci_monet/addons/plume_sentinel/bulletin.py` — `BulletinResponse`, `build_prompt`, `generate_bulletin`, `publish_mqtt`
- `davinci_monet/addons/plume_sentinel/demo_stages.py` — `PlumeSentinelDemoLoadStage`, `PlumeSentinelDemoPrepareStage`, `PlumeSentinelDemoPlotStage`
- `davinci_monet/addons/plume_sentinel/templates/__init__.py` — empty package marker
- `davinci_monet/addons/plume_sentinel/templates/bulletin.template` — packaged copy of the existing template
- `davinci_monet/tests/unit/addons/plume_sentinel/test_bulletin.py` — unit tests for `bulletin.py`
- `davinci_monet/tests/unit/addons/plume_sentinel/test_bulletin_stage.py` — unit tests for the stage
- `davinci_monet/tests/unit/addons/plume_sentinel/test_demo_stages.py` — unit tests for demo stages
- `davinci_monet/tests/unit/cli/test_demo_flags.py` — unit tests for the CLI flag translation helper
- `davinci_monet/tests/test_plume_sentinel_workflow_with_bulletin.py` — integration through `PipelineRunner`

**Modify:**
- `davinci_monet/addons/plume_sentinel/schema.py` — add `MqttConfig`, `BulletinConfig`, `bulletin` field
- `davinci_monet/addons/plume_sentinel/stages.py` — add `PlumeSentinelBulletinStage` with live + canned branches
- `davinci_monet/addons/plume_sentinel/workflow.py` — import refactored payload builder; add bulletin stage; add `demo_mode` parameter to factory
- `davinci_monet/pipeline/runner.py` — read `analysis._demo.enabled` and pass `demo_mode` to the plume_sentinel factory
- `davinci_monet/cli/commands/run.py` — add `--demo-mode` and `--demo-bulletin` flags + `_apply_demo_flags` helper
- `pyproject.toml` — add `anthropic`, `paho-mqtt` to `dependencies`; add template glob to `package-data`
- `environment.yml` — add `anthropic`, `paho-mqtt` under `pip:` block (defensive — also installable via pip extras)

---

## Task 1: Refactor — move `_build_metrics_payload` out of `workflow.py`

**Files:**
- Create: `davinci_monet/addons/plume_sentinel/metrics_payload.py`
- Modify: `davinci_monet/addons/plume_sentinel/workflow.py`
- Test: `davinci_monet/tests/test_metrics_extraction.py` (existing — no edits, must keep passing)

This is a pure code move. Behavior is unchanged. The existing test suite is the safety net.

- [ ] **Step 1: Run existing metrics-extraction tests to confirm baseline**

```bash
pytest davinci_monet/tests/test_metrics_extraction.py -v
```

Expected: all tests pass.

- [ ] **Step 2: Create the new module by moving the relevant helpers**

Move these from `workflow.py` to a new file `davinci_monet/addons/plume_sentinel/metrics_payload.py`:
- `_gridded_aod_to_dataarray`
- `_to_iso`
- `_event_date_from_config`
- `_derive_input_datasets`
- `_provenance_quality_flag`
- `_build_metrics_payload` (rename to public `build_metrics_payload` — drop the leading underscore since it now has external callers)

Top of the new file:

```python
"""Assemble plumesentinel.metrics.v1 payloads from pipeline outputs.

Extracted from workflow.py so both ``run(..., emit_metrics_json=...)`` and
``PlumeSentinelBulletinStage`` can call the same payload builder.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import xarray as xr

from davinci_monet.addons.plume_sentinel import metrics as _metrics
from davinci_monet.addons.plume_sentinel.processing import GriddedAodResult
```

- [ ] **Step 3: Update `workflow.py` to import from the new module**

In `workflow.py`, delete the moved functions and replace with:

```python
from davinci_monet.addons.plume_sentinel.metrics_payload import (
    build_metrics_payload,
)
```

Update the one call site in `run(...)`:

```python
payload = build_metrics_payload(
    context_metadata=result.context.metadata,
    config=config,
    config_path=config_path,
    run_id=run_id,
    region=region,
    config_slug=config_slug,
    wallclock_s=wallclock_s,
    stage_results=result.stage_results,
)
```

Keep `_apply_event_date_override` in `workflow.py` — it's not used by the bulletin stage.

- [ ] **Step 4: Run all plume_sentinel tests to verify the refactor is behavior-preserving**

```bash
pytest davinci_monet/tests/test_metrics_extraction.py davinci_monet/tests/test_plume_sentinel_workflow.py davinci_monet/tests/unit/addons/plume_sentinel/ -v
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add davinci_monet/addons/plume_sentinel/metrics_payload.py davinci_monet/addons/plume_sentinel/workflow.py
git commit -m "refactor(plume-sentinel): extract metrics_payload module"
```

---

## Task 2: Add dependencies to pyproject.toml and environment.yml

**Files:**
- Modify: `pyproject.toml`
- Modify: `environment.yml`

- [ ] **Step 1: Add deps to `pyproject.toml`**

In the `[project]` `dependencies = [...]` list, add:

```toml
    "anthropic>=0.40",
    "paho-mqtt>=2.0",
```

- [ ] **Step 2: Add deps to `environment.yml`**

Update the `pip:` block at the bottom of `environment.yml`:

```yaml
  - pip
  - pip:
    - -e ".[dev]"
    - anthropic>=0.40
    - paho-mqtt>=2.0
```

(Listing them explicitly under `pip:` is belt-and-suspenders; they will also be pulled in by the editable install from pyproject.toml.)

- [ ] **Step 3: Install the new packages into the active env**

```bash
pip install "anthropic>=0.40" "paho-mqtt>=2.0"
```

- [ ] **Step 4: Smoke-import the new packages**

```bash
python -c "import anthropic, paho.mqtt.client; print(anthropic.__version__, paho.mqtt.client.Client)"
```

Expected: prints a version string and a class repr; no ImportError.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml environment.yml
git commit -m "feat(plume-sentinel): add anthropic and paho-mqtt deps for bulletin stage"
```

---

## Task 3: Package the bulletin template

**Files:**
- Create: `davinci_monet/addons/plume_sentinel/templates/__init__.py`
- Create: `davinci_monet/addons/plume_sentinel/templates/bulletin.template`
- Modify: `pyproject.toml` (extend `package-data`)
- Test: smoke test via `importlib.resources`

- [ ] **Step 1: Create the package marker and copy the template**

Create empty file `davinci_monet/addons/plume_sentinel/templates/__init__.py`.

Copy the existing template:

```bash
cp analyses/plume-sentinel/templates/bulletin.template davinci_monet/addons/plume_sentinel/templates/bulletin.template
```

- [ ] **Step 2: Extend `pyproject.toml` package-data**

Update the existing `[tool.setuptools.package-data]` section to include the template glob:

```toml
[tool.setuptools.package-data]
davinci_monet = ["py.typed", "addons/plume_sentinel/templates/*.template"]
```

- [ ] **Step 3: Reinstall in editable mode so package-data is picked up**

```bash
pip install -e .
```

- [ ] **Step 4: Smoke-test resource loading**

```bash
python -c "from importlib.resources import files; t = files('davinci_monet.addons.plume_sentinel.templates').joinpath('bulletin.template').read_text(); print(len(t), 'chars'); assert '{{BULLETIN_ID}}' in t"
```

Expected: prints the character count and exits cleanly (no AssertionError).

- [ ] **Step 5: Commit**

```bash
git add davinci_monet/addons/plume_sentinel/templates/ pyproject.toml
git commit -m "feat(plume-sentinel): package bulletin.template inside the addon"
```

---

## Task 4: Schema — add `MqttConfig`, `BulletinConfig`, optional `bulletin` field

**Files:**
- Modify: `davinci_monet/addons/plume_sentinel/schema.py`
- Test: `davinci_monet/tests/unit/addons/plume_sentinel/test_schema.py`

- [ ] **Step 1: Write failing tests for the new schema**

Append to `davinci_monet/tests/unit/addons/plume_sentinel/test_schema.py`:

```python
import pytest
from pydantic import ValidationError

from davinci_monet.addons.plume_sentinel.schema import (
    BulletinConfig,
    MqttConfig,
    PlumeSentinelConfig,
)


def test_mqtt_config_defaults():
    cfg = MqttConfig(topic="plume-sentinel-ai/reports/test")
    assert cfg.broker == "broker.hivemq.com"
    assert cfg.port == 1883
    assert cfg.qos == 0
    assert cfg.topic == "plume-sentinel-ai/reports/test"


def test_bulletin_config_defaults():
    cfg = BulletinConfig()
    assert cfg.template is None
    assert cfg.output_filename == "bulletin.txt"
    assert cfg.model == "claude-sonnet-4-6"
    assert cfg.include_images is False
    assert cfg.api_key_env == "ANTHROPIC_API_KEY"
    assert cfg.mqtt is None
    assert cfg.on_error == "warn"


def test_bulletin_config_on_error_only_accepts_warn():
    with pytest.raises(ValidationError):
        BulletinConfig(on_error="fail")


def test_bulletin_config_parses_mqtt_subblock():
    cfg = BulletinConfig(
        mqtt={"topic": "plume-sentinel-ai/reports/west-coast", "qos": 1}
    )
    assert isinstance(cfg.mqtt, MqttConfig)
    assert cfg.mqtt.topic == "plume-sentinel-ai/reports/west-coast"
    assert cfg.mqtt.qos == 1


def test_plume_sentinel_config_bulletin_optional():
    cfg = PlumeSentinelConfig(inputs={}, plots={})
    assert cfg.bulletin is None


def test_plume_sentinel_config_bulletin_present():
    cfg = PlumeSentinelConfig(
        inputs={},
        plots={},
        bulletin={"model": "claude-opus-4-7", "include_images": True},
    )
    assert cfg.bulletin is not None
    assert cfg.bulletin.model == "claude-opus-4-7"
    assert cfg.bulletin.include_images is True
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest davinci_monet/tests/unit/addons/plume_sentinel/test_schema.py -v -k "mqtt or bulletin"
```

Expected: FAIL with `ImportError: cannot import name 'BulletinConfig'`.

- [ ] **Step 3: Implement the new schema classes**

Add to `davinci_monet/addons/plume_sentinel/schema.py`:

```python
from pydantic import BaseModel, field_validator


class MqttConfig(BaseModel):
    """MQTT broker connection and publish parameters."""

    broker: str = "broker.hivemq.com"
    topic: str
    port: int = 1883
    qos: int = 0


class BulletinConfig(BaseModel):
    """Configuration for the bulletin generation stage."""

    template: str | None = None
    output_filename: str = "bulletin.txt"
    model: str = "claude-sonnet-4-6"
    include_images: bool = False
    api_key_env: str = "ANTHROPIC_API_KEY"
    mqtt: MqttConfig | None = None
    on_error: str = "warn"

    @field_validator("on_error")
    @classmethod
    def _on_error_must_be_warn(cls, v: str) -> str:
        if v != "warn":
            raise ValueError(
                'on_error currently only supports "warn"; "fail" is reserved'
            )
        return v

    @field_validator("mqtt", mode="before")
    @classmethod
    def _parse_mqtt(cls, v):  # noqa: ANN001, ANN201
        if isinstance(v, dict):
            return MqttConfig(**v)
        return v
```

Modify the existing `PlumeSentinelConfig` to add the optional field:

```python
class PlumeSentinelConfig(BaseModel):
    """Top-level configuration for the PlumeSentinel add-on workflow."""

    inputs: dict[str, InputSpec]
    plots: dict[str, PlotSpec]
    bulletin: BulletinConfig | None = None
```

- [ ] **Step 4: Run tests to verify pass**

```bash
pytest davinci_monet/tests/unit/addons/plume_sentinel/test_schema.py -v
```

Expected: all schema tests pass (new ones plus all pre-existing ones).

- [ ] **Step 5: Commit**

```bash
git add davinci_monet/addons/plume_sentinel/schema.py davinci_monet/tests/unit/addons/plume_sentinel/test_schema.py
git commit -m "feat(plume-sentinel): add BulletinConfig and MqttConfig schema"
```

---

## Task 5: Implement `build_prompt()` in `bulletin.py`

**Files:**
- Create: `davinci_monet/addons/plume_sentinel/bulletin.py`
- Test: `davinci_monet/tests/unit/addons/plume_sentinel/test_bulletin.py`

- [ ] **Step 1: Write failing tests for `build_prompt`**

Create `davinci_monet/tests/unit/addons/plume_sentinel/test_bulletin.py`:

```python
"""Unit tests for the plume_sentinel bulletin helpers."""

from __future__ import annotations

import pytest

from davinci_monet.addons.plume_sentinel.bulletin import build_prompt


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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest davinci_monet/tests/unit/addons/plume_sentinel/test_bulletin.py -v
```

Expected: FAIL with `ImportError: cannot import name 'build_prompt'`.

- [ ] **Step 3: Implement `build_prompt`**

Create `davinci_monet/addons/plume_sentinel/bulletin.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify pass**

```bash
pytest davinci_monet/tests/unit/addons/plume_sentinel/test_bulletin.py -v
```

Expected: 3 tests pass.

- [ ] **Step 5: Commit**

```bash
git add davinci_monet/addons/plume_sentinel/bulletin.py davinci_monet/tests/unit/addons/plume_sentinel/test_bulletin.py
git commit -m "feat(plume-sentinel): bulletin.build_prompt fills deterministic placeholders"
```

---

## Task 6: Implement `publish_mqtt()` in `bulletin.py`

**Files:**
- Modify: `davinci_monet/addons/plume_sentinel/bulletin.py`
- Modify: `davinci_monet/tests/unit/addons/plume_sentinel/test_bulletin.py`

- [ ] **Step 1: Write failing tests for `publish_mqtt`**

Append to `test_bulletin.py`:

```python
from unittest.mock import MagicMock, patch

from davinci_monet.addons.plume_sentinel.bulletin import publish_mqtt


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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest davinci_monet/tests/unit/addons/plume_sentinel/test_bulletin.py::test_publish_mqtt_uses_config -v
```

Expected: FAIL with `ImportError: cannot import name 'publish_mqtt'`.

- [ ] **Step 3: Implement `publish_mqtt`**

Add to top of `bulletin.py`:

```python
import paho.mqtt.client as paho_mqtt
```

Add at the end of `bulletin.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify pass**

```bash
pytest davinci_monet/tests/unit/addons/plume_sentinel/test_bulletin.py -v
```

Expected: all 5 tests pass.

- [ ] **Step 5: Commit**

```bash
git add davinci_monet/addons/plume_sentinel/bulletin.py davinci_monet/tests/unit/addons/plume_sentinel/test_bulletin.py
git commit -m "feat(plume-sentinel): bulletin.publish_mqtt publishes text via paho-mqtt"
```

---

## Task 7: Implement `BulletinResponse` and `generate_bulletin()` in `bulletin.py`

**Files:**
- Modify: `davinci_monet/addons/plume_sentinel/bulletin.py`
- Modify: `davinci_monet/tests/unit/addons/plume_sentinel/test_bulletin.py`

- [ ] **Step 1: Write failing tests for `generate_bulletin`**

Append to `test_bulletin.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest davinci_monet/tests/unit/addons/plume_sentinel/test_bulletin.py -v -k generate
```

Expected: FAIL with `ImportError: cannot import name 'BulletinResponse'`.

- [ ] **Step 3: Implement `BulletinResponse` and `generate_bulletin`**

Add to top of `bulletin.py`:

```python
import base64
import json
from dataclasses import dataclass, field
from pathlib import Path

import anthropic
```

Add at the end:

```python
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
    user_content.append(
        {"type": "text", "text": "Render the bulletin for this event."}
    )

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
```

- [ ] **Step 4: Run tests to verify pass**

```bash
pytest davinci_monet/tests/unit/addons/plume_sentinel/test_bulletin.py -v
```

Expected: 8 tests pass (3 build_prompt + 2 publish_mqtt + 3 generate_bulletin).

- [ ] **Step 5: Commit**

```bash
git add davinci_monet/addons/plume_sentinel/bulletin.py davinci_monet/tests/unit/addons/plume_sentinel/test_bulletin.py
git commit -m "feat(plume-sentinel): bulletin.generate_bulletin calls Claude with cached system prompt"
```

---

## Task 8: Implement `PlumeSentinelBulletinStage` — happy path + skip cases

**Files:**
- Modify: `davinci_monet/addons/plume_sentinel/stages.py`
- Create: `davinci_monet/tests/unit/addons/plume_sentinel/test_bulletin_stage.py`

- [ ] **Step 1: Write failing tests for the stage's basic shapes**

Create `davinci_monet/tests/unit/addons/plume_sentinel/test_bulletin_stage.py`:

```python
"""Unit tests for PlumeSentinelBulletinStage."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from davinci_monet.addons.plume_sentinel.bulletin import BulletinResponse
from davinci_monet.addons.plume_sentinel.schema import PlumeSentinelConfig
from davinci_monet.addons.plume_sentinel.stages import PlumeSentinelBulletinStage
from davinci_monet.pipeline.stages import PipelineContext, StageStatus


def _make_context(tmp_path: Path, *, bulletin: dict | None = None) -> PipelineContext:
    """Build a minimal pipeline context that mimics post-plot state."""
    config: dict = {
        "analysis": {"output_dir": str(tmp_path), "start_time": "2020-09-09"},
        "plume_sentinel": {
            "inputs": {},
            "plots": {},
            **({"bulletin": bulletin} if bulletin is not None else {}),
        },
    }
    ctx = PipelineContext(config=config)
    ctx.metadata["plume_sentinel_config"] = PlumeSentinelConfig(
        **config["plume_sentinel"]
    )
    ctx.metadata["plume_sentinel_prepared"] = {}
    ctx.metadata["plume_sentinel_plots_generated"] = []
    return ctx


def test_stage_no_op_when_no_bulletin_config(tmp_path):
    stage = PlumeSentinelBulletinStage()
    ctx = _make_context(tmp_path, bulletin=None)
    result = stage.execute(ctx)
    assert result.status == StageStatus.COMPLETED
    assert result.data.get("bulletin") == "skipped (no config)"
    assert not (tmp_path / "bulletin.txt").exists()


def test_stage_skips_when_api_key_env_missing(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    stage = PlumeSentinelBulletinStage()
    ctx = _make_context(tmp_path, bulletin={})
    result = stage.execute(ctx)
    assert result.status == StageStatus.COMPLETED
    assert result.data.get("bulletin") == "skipped (no api key)"
    flags = ctx.metadata.get("plume_sentinel_quality_flags", [])
    assert any(
        f.get("category") == "bulletin" and "not set" in f.get("message", "")
        for f in flags
    )
    assert not (tmp_path / "bulletin.txt").exists()


def test_stage_writes_file_on_success(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-fake")
    stage = PlumeSentinelBulletinStage()
    ctx = _make_context(tmp_path, bulletin={})
    fake_resp = BulletinResponse(
        text="RENDERED BULLETIN BODY",
        model="claude-sonnet-4-6",
        input_tokens=120,
        cache_read_tokens=100,
        output_tokens=60,
    )
    with patch(
        "davinci_monet.addons.plume_sentinel.stages.generate_bulletin",
        return_value=fake_resp,
    ) as gen, patch(
        "davinci_monet.addons.plume_sentinel.stages.build_metrics_payload",
        return_value={"event_date": "2020-09-09", "region": "westcoast", "input_datasets": []},
    ):
        result = stage.execute(ctx)
    assert result.status == StageStatus.COMPLETED
    bulletin_path = tmp_path / "bulletin.txt"
    assert bulletin_path.is_file()
    assert bulletin_path.read_text() == "RENDERED BULLETIN BODY"
    assert result.data["bulletin_path"] == str(bulletin_path)
    assert result.data["input_tokens"] == 120
    assert result.data["cache_read_tokens"] == 100
    assert result.data["output_tokens"] == 60
    gen.assert_called_once()


def test_stage_passes_images_when_include_images_true(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-fake")
    stage = PlumeSentinelBulletinStage()
    ctx = _make_context(tmp_path, bulletin={"include_images": True})
    plot1 = tmp_path / "modis_aod_truecolor.png"
    plot1.write_bytes(b"\x89PNG")
    ctx.metadata["plume_sentinel_plots_generated"] = [str(plot1)]
    fake_resp = BulletinResponse(
        text="BODY", model="claude-sonnet-4-6",
        input_tokens=1, cache_read_tokens=0, output_tokens=1,
    )
    with patch(
        "davinci_monet.addons.plume_sentinel.stages.generate_bulletin",
        return_value=fake_resp,
    ) as gen, patch(
        "davinci_monet.addons.plume_sentinel.stages.build_metrics_payload",
        return_value={"event_date": "2020-09-09", "region": "westcoast", "input_datasets": []},
    ):
        stage.execute(ctx)
    _args, kwargs = gen.call_args
    assert kwargs["image_paths"] == [plot1]


def test_stage_omits_images_when_include_images_false(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-fake")
    stage = PlumeSentinelBulletinStage()
    ctx = _make_context(tmp_path, bulletin={"include_images": False})
    plot1 = tmp_path / "p.png"
    plot1.write_bytes(b"\x89PNG")
    ctx.metadata["plume_sentinel_plots_generated"] = [str(plot1)]
    fake_resp = BulletinResponse(
        text="BODY", model="claude-sonnet-4-6",
        input_tokens=1, cache_read_tokens=0, output_tokens=1,
    )
    with patch(
        "davinci_monet.addons.plume_sentinel.stages.generate_bulletin",
        return_value=fake_resp,
    ) as gen, patch(
        "davinci_monet.addons.plume_sentinel.stages.build_metrics_payload",
        return_value={"event_date": "2020-09-09", "region": "westcoast", "input_datasets": []},
    ):
        stage.execute(ctx)
    _args, kwargs = gen.call_args
    assert kwargs["image_paths"] == []
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest davinci_monet/tests/unit/addons/plume_sentinel/test_bulletin_stage.py -v
```

Expected: FAIL with `ImportError: cannot import name 'PlumeSentinelBulletinStage'`.

- [ ] **Step 3: Implement the stage (and surface `plots_generated` in context metadata)**

First, modify `PlumeSentinelPlotStage.execute` in `stages.py` so the bulletin stage can read the plot list from `context.metadata`. Append before the `return self._create_result(...)` line:

```python
        context.metadata["plume_sentinel_plots_generated"] = all_paths
```

Then add new imports at the top of `stages.py`:

```python
import os
from datetime import datetime, timezone

from davinci_monet.addons.plume_sentinel.bulletin import (
    BulletinResponse,
    build_prompt,
    generate_bulletin,
    publish_mqtt,
)
from davinci_monet.addons.plume_sentinel.metrics_payload import build_metrics_payload
```

Then add the new stage class at the end of `stages.py`:

```python
class PlumeSentinelBulletinStage(BaseStage):
    """Generate a meteorological bulletin via the Claude API; optionally publish to MQTT."""

    DEFAULT_TEMPLATE_PACKAGE = "davinci_monet.addons.plume_sentinel.templates"
    DEFAULT_TEMPLATE_NAME = "bulletin.template"

    def __init__(self) -> None:
        super().__init__(name="bulletin")

    def execute(self, context: PipelineContext) -> StageResult:
        start = time.time()
        cfg: PlumeSentinelConfig = context.metadata["plume_sentinel_config"]

        if cfg.bulletin is None:
            return self._create_result(
                StageStatus.COMPLETED,
                data={"bulletin": "skipped (no config)"},
                duration=time.time() - start,
            )

        bcfg = cfg.bulletin
        api_key = os.environ.get(bcfg.api_key_env)
        if not api_key:
            _append_quality_flag(
                context,
                "warning",
                f"API key env var {bcfg.api_key_env} not set; bulletin skipped",
            )
            return self._create_result(
                StageStatus.COMPLETED,
                data={"bulletin": "skipped (no api key)"},
                duration=time.time() - start,
            )

        # Build metrics payload from the same helper the CLI uses.
        try:
            payload = build_metrics_payload(
                context_metadata=context.metadata,
                config=context.config,
                config_path=None,
                run_id=None,
                region=None,
                config_slug=None,
                wallclock_s=0.0,
                stage_results=[],
            )
        except Exception as exc:  # noqa: BLE001
            _append_quality_flag(
                context, "warning",
                f"Metrics payload build failed: {exc}; bulletin skipped",
            )
            return self._create_result(
                StageStatus.COMPLETED,
                data={"bulletin": "skipped (metrics payload failed)"},
                duration=time.time() - start,
            )

        # Load template (packaged default unless overridden).
        try:
            template_text = _load_template(bcfg.template)
        except FileNotFoundError as exc:
            _append_quality_flag(
                context, "warning",
                f"Bulletin template not found at {exc}; bulletin skipped",
            )
            return self._create_result(
                StageStatus.COMPLETED,
                data={"bulletin": "skipped (template missing)"},
                duration=time.time() - start,
            )

        issued = datetime.now(timezone.utc).strftime("%B %-d, %Y")
        prompt = build_prompt(template_text, payload, issued_date=issued)

        plots_generated = context.metadata.get(
            "plume_sentinel_plots_generated", []
        ) or []
        image_paths = (
            [Path(p) for p in plots_generated] if bcfg.include_images else []
        )

        # Call Claude.
        try:
            resp: BulletinResponse = generate_bulletin(
                prompt=prompt,
                metrics_json=payload,
                image_paths=image_paths,
                model=bcfg.model,
                api_key=api_key,
            )
        except Exception as exc:  # noqa: BLE001 - record + continue
            _append_quality_flag(
                context, "warning",
                f"Claude API call failed: {exc}; bulletin skipped",
            )
            return self._create_result(
                StageStatus.COMPLETED,
                data={"bulletin": "skipped (api error)"},
                duration=time.time() - start,
            )

        for missing in resp.skipped_images:
            _append_quality_flag(
                context, "info", f"Bulletin image not found: {missing}",
            )

        # Write file.
        output_dir = _get_output_dir(context)
        output_dir.mkdir(parents=True, exist_ok=True)
        bulletin_path = output_dir / bcfg.output_filename
        try:
            bulletin_path.write_text(resp.text)
        except OSError as exc:
            _append_quality_flag(
                context, "warning",
                f"Bulletin file write failed: {exc}",
            )
            return self._create_result(
                StageStatus.COMPLETED,
                data={"bulletin": "skipped (file write failed)"},
                duration=time.time() - start,
            )

        # Publish to MQTT if configured.
        mqtt_published = False
        if bcfg.mqtt is not None:
            try:
                publish_mqtt(
                    text=resp.text,
                    broker=bcfg.mqtt.broker,
                    topic=bcfg.mqtt.topic,
                    port=bcfg.mqtt.port,
                    qos=bcfg.mqtt.qos,
                )
                mqtt_published = True
            except Exception as exc:  # noqa: BLE001
                _append_quality_flag(
                    context, "warning",
                    f"MQTT publish to {bcfg.mqtt.broker}:{bcfg.mqtt.port} failed: {exc}",
                )

        # Surface a structured summary into context.metadata so the metrics
        # payload extension (Task 10) can include it.
        context.metadata["plume_sentinel_bulletin"] = {
            "path": str(bulletin_path),
            "model": resp.model,
            "input_tokens": resp.input_tokens,
            "cache_read_tokens": resp.cache_read_tokens,
            "output_tokens": resp.output_tokens,
        }

        return self._create_result(
            StageStatus.COMPLETED,
            data={
                "bulletin_path": str(bulletin_path),
                "mqtt_published": mqtt_published,
                "model": resp.model,
                "input_tokens": resp.input_tokens,
                "cache_read_tokens": resp.cache_read_tokens,
                "output_tokens": resp.output_tokens,
            },
            duration=time.time() - start,
        )


def _load_template(template_path: str | None) -> str:
    """Load the bulletin template; default to the packaged copy."""
    if template_path:
        p = Path(template_path)
        if not p.is_file():
            raise FileNotFoundError(str(p))
        return p.read_text()
    from importlib.resources import files
    return (
        files(PlumeSentinelBulletinStage.DEFAULT_TEMPLATE_PACKAGE)
        .joinpath(PlumeSentinelBulletinStage.DEFAULT_TEMPLATE_NAME)
        .read_text()
    )


def _append_quality_flag(
    context: PipelineContext, severity: str, message: str
) -> None:
    """Append a bulletin-category quality_flag onto context.metadata."""
    flags = context.metadata.setdefault("plume_sentinel_quality_flags", [])
    flags.append(
        {"category": "bulletin", "severity": severity, "message": message}
    )
```

- [ ] **Step 4: Run tests to verify pass**

```bash
pytest davinci_monet/tests/unit/addons/plume_sentinel/test_bulletin_stage.py -v
```

Expected: 5 tests pass.

- [ ] **Step 5: Commit**

```bash
git add davinci_monet/addons/plume_sentinel/stages.py davinci_monet/tests/unit/addons/plume_sentinel/test_bulletin_stage.py
git commit -m "feat(plume-sentinel): PlumeSentinelBulletinStage with skip-and-warn semantics"
```

---

## Task 9: Bulletin stage error paths — API error, MQTT error, template missing

**Files:**
- Modify: `davinci_monet/tests/unit/addons/plume_sentinel/test_bulletin_stage.py`

The error paths are already implemented in Task 8. This task adds explicit regression tests for them.

- [ ] **Step 1: Add failure-path tests**

Append to `test_bulletin_stage.py`:

```python
import anthropic as _anthropic_module


def test_stage_records_quality_flag_on_api_error(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-fake")
    stage = PlumeSentinelBulletinStage()
    ctx = _make_context(tmp_path, bulletin={})
    with patch(
        "davinci_monet.addons.plume_sentinel.stages.build_metrics_payload",
        return_value={"event_date": "2020-09-09", "region": "westcoast", "input_datasets": []},
    ), patch(
        "davinci_monet.addons.plume_sentinel.stages.generate_bulletin",
        side_effect=_anthropic_module.APIError("boom", request=None, body=None),
    ):
        result = stage.execute(ctx)
    assert result.status == StageStatus.COMPLETED
    assert result.data.get("bulletin") == "skipped (api error)"
    flags = ctx.metadata.get("plume_sentinel_quality_flags", [])
    assert any(
        f["category"] == "bulletin" and "API call failed" in f["message"]
        for f in flags
    )
    assert not (tmp_path / "bulletin.txt").exists()


def test_stage_records_quality_flag_on_mqtt_error(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-fake")
    stage = PlumeSentinelBulletinStage()
    ctx = _make_context(
        tmp_path,
        bulletin={"mqtt": {"topic": "t/test", "broker": "broker.example.com"}},
    )
    fake_resp = BulletinResponse(
        text="HELLO", model="claude-sonnet-4-6",
        input_tokens=1, cache_read_tokens=0, output_tokens=1,
    )
    with patch(
        "davinci_monet.addons.plume_sentinel.stages.build_metrics_payload",
        return_value={"event_date": "2020-09-09", "region": "westcoast", "input_datasets": []},
    ), patch(
        "davinci_monet.addons.plume_sentinel.stages.generate_bulletin",
        return_value=fake_resp,
    ), patch(
        "davinci_monet.addons.plume_sentinel.stages.publish_mqtt",
        side_effect=OSError("broker unreachable"),
    ):
        result = stage.execute(ctx)
    assert result.status == StageStatus.COMPLETED
    # File still written
    assert (tmp_path / "bulletin.txt").read_text() == "HELLO"
    assert result.data["mqtt_published"] is False
    flags = ctx.metadata.get("plume_sentinel_quality_flags", [])
    assert any(
        f["category"] == "bulletin" and "MQTT publish" in f["message"]
        for f in flags
    )


def test_stage_records_quality_flag_on_missing_template(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-fake")
    stage = PlumeSentinelBulletinStage()
    ctx = _make_context(
        tmp_path, bulletin={"template": str(tmp_path / "nope.template")},
    )
    with patch(
        "davinci_monet.addons.plume_sentinel.stages.build_metrics_payload",
        return_value={"event_date": "2020-09-09", "region": "westcoast", "input_datasets": []},
    ):
        result = stage.execute(ctx)
    assert result.status == StageStatus.COMPLETED
    assert result.data.get("bulletin") == "skipped (template missing)"
    flags = ctx.metadata.get("plume_sentinel_quality_flags", [])
    assert any(
        f["category"] == "bulletin" and "template not found" in f["message"]
        for f in flags
    )
```

- [ ] **Step 2: Run tests to verify pass (already implemented in Task 8)**

```bash
pytest davinci_monet/tests/unit/addons/plume_sentinel/test_bulletin_stage.py -v
```

Expected: 8 tests pass (5 from Task 8 + 3 new).

- [ ] **Step 3: Commit**

```bash
git add davinci_monet/tests/unit/addons/plume_sentinel/test_bulletin_stage.py
git commit -m "test(plume-sentinel): regression tests for bulletin stage error paths"
```

---

## Task 10: Wire stage into pipeline factory + extend metrics payload with `bulletin` field

**Files:**
- Modify: `davinci_monet/addons/plume_sentinel/workflow.py`
- Modify: `davinci_monet/addons/plume_sentinel/metrics_payload.py`
- Create: `davinci_monet/tests/test_plume_sentinel_workflow_with_bulletin.py`

- [ ] **Step 1: Write failing integration test**

Create `davinci_monet/tests/test_plume_sentinel_workflow_with_bulletin.py`. Pattern this after the existing `test_plume_sentinel_workflow.py`:

```python
"""End-to-end test: plume_sentinel workflow with the bulletin stage."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from davinci_monet.addons.plume_sentinel.bulletin import BulletinResponse
from davinci_monet.addons.plume_sentinel.workflow import (
    create_plume_sentinel_pipeline,
)


@pytest.fixture
def minimal_config(tmp_path):
    """A minimal config that exercises the full plume_sentinel pipeline."""
    return {
        "analysis": {
            "output_dir": str(tmp_path),
            "start_time": "2020-09-09",
            "end_time": "2020-09-09",
            "workflow": "plume_sentinel",
        },
        "plume_sentinel": {
            "inputs": {},
            "plots": {},
            "bulletin": {
                "output_filename": "bulletin.txt",
                "model": "claude-sonnet-4-6",
                "include_images": False,
            },
        },
    }


def test_create_plume_sentinel_pipeline_includes_bulletin_stage():
    pipeline = create_plume_sentinel_pipeline()
    names = [s.name for s in pipeline]
    assert names == ["load_inputs", "prepare_geospatial", "plotting", "bulletin"]


def test_workflow_writes_bulletin_when_block_present(minimal_config, tmp_path, monkeypatch):
    from davinci_monet.pipeline.runner import PipelineRunner

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-fake")
    fake_resp = BulletinResponse(
        text="==BULLETIN==", model="claude-sonnet-4-6",
        input_tokens=10, cache_read_tokens=0, output_tokens=5,
    )
    with patch(
        "davinci_monet.addons.plume_sentinel.stages.generate_bulletin",
        return_value=fake_resp,
    ):
        runner = PipelineRunner(show_progress=False)
        result = runner.run_from_config(minimal_config)
    assert result.success
    assert (tmp_path / "bulletin.txt").read_text() == "==BULLETIN=="


def test_workflow_omits_bulletin_when_block_absent(minimal_config, tmp_path):
    from davinci_monet.pipeline.runner import PipelineRunner

    cfg = json.loads(json.dumps(minimal_config))
    cfg["plume_sentinel"].pop("bulletin")
    runner = PipelineRunner(show_progress=False)
    result = runner.run_from_config(cfg)
    assert result.success
    assert not (tmp_path / "bulletin.txt").exists()


def test_metrics_payload_includes_bulletin_field_when_present(minimal_config, tmp_path, monkeypatch):
    from davinci_monet.addons.plume_sentinel.workflow import run as run_workflow

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-fake")
    metrics_path = tmp_path / "metrics.json"
    fake_resp = BulletinResponse(
        text="==BULLETIN==", model="claude-sonnet-4-6",
        input_tokens=10, cache_read_tokens=8, output_tokens=5,
    )
    with patch(
        "davinci_monet.addons.plume_sentinel.stages.generate_bulletin",
        return_value=fake_resp,
    ):
        run_workflow(
            minimal_config,
            emit_metrics_json=metrics_path,
            region="westcoast",
            config_slug="modis-aod-truecolor",
        )
    payload = json.loads(metrics_path.read_text())
    assert "bulletin" in payload
    assert payload["bulletin"]["model"] == "claude-sonnet-4-6"
    assert payload["bulletin"]["input_tokens"] == 10
    assert payload["bulletin"]["cache_read_tokens"] == 8
    assert payload["bulletin"]["output_tokens"] == 5
    assert payload["bulletin"]["path"].endswith("bulletin.txt")
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest davinci_monet/tests/test_plume_sentinel_workflow_with_bulletin.py -v
```

Expected: at least `test_create_plume_sentinel_pipeline_includes_bulletin_stage` and `test_metrics_payload_includes_bulletin_field_when_present` FAIL.

- [ ] **Step 3: Add the bulletin stage to the pipeline factory**

In `davinci_monet/addons/plume_sentinel/workflow.py`, update `create_plume_sentinel_pipeline`:

```python
def create_plume_sentinel_pipeline() -> list[BaseStage]:
    """Create the four-stage Plume Sentinel pipeline.

    Returns
    -------
    list[BaseStage]
        Ordered list: load_inputs -> prepare_geospatial -> plotting -> bulletin.
    """
    return [
        PlumeSentinelLoadStage(),
        PlumeSentinelPrepareStage(),
        PlumeSentinelPlotStage(),
        PlumeSentinelBulletinStage(),
    ]
```

Update the import at the top of `workflow.py` to add `PlumeSentinelBulletinStage`:

```python
from davinci_monet.addons.plume_sentinel.stages import (
    PlumeSentinelBulletinStage,
    PlumeSentinelLoadStage,
    PlumeSentinelPlotStage,
    PlumeSentinelPrepareStage,
)
```

- [ ] **Step 4: Extend `build_metrics_payload` to include the bulletin field**

In `davinci_monet/addons/plume_sentinel/metrics_payload.py`, inside the `build_metrics_payload(...)` function, after the existing payload dict is built and before `return payload`, add:

```python
    bulletin_info = context_metadata.get("plume_sentinel_bulletin")
    if bulletin_info:
        payload["bulletin"] = bulletin_info
```

- [ ] **Step 5: Run all plume_sentinel tests**

```bash
pytest davinci_monet/tests/test_plume_sentinel_workflow.py davinci_monet/tests/test_plume_sentinel_workflow_with_bulletin.py davinci_monet/tests/test_metrics_extraction.py davinci_monet/tests/unit/addons/plume_sentinel/ -v
```

Expected: all pass. Counts:
- Existing workflow integration test: still green
- New `test_plume_sentinel_workflow_with_bulletin.py`: 4 pass
- `test_bulletin.py`: 8 pass
- `test_bulletin_stage.py`: 8 pass
- `test_schema.py`: existing + 6 new = all pass
- `test_metrics_extraction.py`: still green

- [ ] **Step 6: Commit**

```bash
git add davinci_monet/addons/plume_sentinel/workflow.py davinci_monet/addons/plume_sentinel/metrics_payload.py davinci_monet/tests/test_plume_sentinel_workflow_with_bulletin.py
git commit -m "feat(plume-sentinel): wire bulletin stage into pipeline + add bulletin field to metrics payload"
```

---

## Task 11: Implement demo stages module

**Files:**
- Create: `davinci_monet/addons/plume_sentinel/demo_stages.py`
- Create: `davinci_monet/tests/unit/addons/plume_sentinel/test_demo_stages.py`

Three demo stages mirror the real load/prepare/plot stages but skip data work:
they sleep, emit progress, and populate stub metadata so downstream stages
(notably the bulletin) see realistic-looking state.

- [ ] **Step 1: Write failing tests for the three demo stages**

Create `davinci_monet/tests/unit/addons/plume_sentinel/test_demo_stages.py`:

```python
"""Unit tests for demo stages."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from davinci_monet.addons.plume_sentinel.demo_stages import (
    PlumeSentinelDemoLoadStage,
    PlumeSentinelDemoPlotStage,
    PlumeSentinelDemoPrepareStage,
)
from davinci_monet.addons.plume_sentinel.schema import PlumeSentinelConfig
from davinci_monet.pipeline.stages import PipelineContext, StageStatus


def _make_context(tmp_path: Path) -> PipelineContext:
    config = {
        "analysis": {"output_dir": str(tmp_path), "start_time": "2020-09-09"},
        "plume_sentinel": {"inputs": {}, "plots": {}},
    }
    ctx = PipelineContext(config=config)
    ctx.metadata["plume_sentinel_config"] = PlumeSentinelConfig(**config["plume_sentinel"])
    return ctx


def test_demo_load_stage_populates_loaded_metadata(tmp_path):
    stage = PlumeSentinelDemoLoadStage()
    ctx = _make_context(tmp_path)
    with patch("davinci_monet.addons.plume_sentinel.demo_stages.time.sleep") as sleep:
        result = stage.execute(ctx)
    assert result.status == StageStatus.COMPLETED
    assert "plume_sentinel_loaded" in ctx.metadata
    assert isinstance(ctx.metadata["plume_sentinel_loaded"], dict)
    sleep.assert_called()  # at least one sleep call
    assert sum(c.args[0] for c in sleep.call_args_list) == pytest.approx(3.0, abs=0.5)


def test_demo_prepare_stage_populates_prepared_and_input_datasets(tmp_path):
    stage = PlumeSentinelDemoPrepareStage()
    ctx = _make_context(tmp_path)
    ctx.metadata["plume_sentinel_loaded"] = {}  # set by demo load
    with patch("davinci_monet.addons.plume_sentinel.demo_stages.time.sleep") as sleep:
        result = stage.execute(ctx)
    assert result.status == StageStatus.COMPLETED
    assert "plume_sentinel_prepared" in ctx.metadata
    datasets = ctx.metadata.get("plume_sentinel_input_datasets", [])
    names = [d["name"] for d in datasets]
    assert any("MODIS" in n for n in names)
    assert any("GOES" in n for n in names)
    assert any("HMS" in n for n in names)
    assert sum(c.args[0] for c in sleep.call_args_list) == pytest.approx(4.0, abs=0.5)


def test_demo_plot_stage_scans_existing_pngs(tmp_path):
    # Pre-populate output_dir with two PNGs as a previous run would.
    (tmp_path / "modis_aod_truecolor.png").write_bytes(b"\x89PNG")
    (tmp_path / "goes_hms_smoke.png").write_bytes(b"\x89PNG")
    (tmp_path / "ignore.txt").write_text("not a plot")

    stage = PlumeSentinelDemoPlotStage()
    ctx = _make_context(tmp_path)
    with patch("davinci_monet.addons.plume_sentinel.demo_stages.time.sleep") as sleep:
        result = stage.execute(ctx)
    assert result.status == StageStatus.COMPLETED
    plots = ctx.metadata.get("plume_sentinel_plots_generated", [])
    assert len(plots) == 2
    assert all(p.endswith(".png") for p in plots)
    assert all("ignore" not in p for p in plots)
    assert sum(c.args[0] for c in sleep.call_args_list) == pytest.approx(3.0, abs=0.5)


def test_demo_plot_stage_handles_empty_output_dir(tmp_path):
    stage = PlumeSentinelDemoPlotStage()
    ctx = _make_context(tmp_path)
    with patch("davinci_monet.addons.plume_sentinel.demo_stages.time.sleep"):
        result = stage.execute(ctx)
    assert result.status == StageStatus.COMPLETED
    assert ctx.metadata.get("plume_sentinel_plots_generated", []) == []
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest davinci_monet/tests/unit/addons/plume_sentinel/test_demo_stages.py -v
```

Expected: FAIL with `ImportError: cannot import name 'PlumeSentinelDemoLoadStage'`.

- [ ] **Step 3: Implement the demo stages**

Create `davinci_monet/addons/plume_sentinel/demo_stages.py`:

```python
"""Demo stages for the PlumeSentinel add-on.

These stages replace the real load/prepare/plot stages when ``--demo-mode``
is set. They sleep, emit realistic progress messages, and populate stub
metadata so the bulletin stage downstream sees a plausible pipeline state.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from davinci_monet.pipeline.stages import BaseStage, PipelineContext, StageResult, StageStatus

LOAD_SLEEP_SECONDS = 3.0
PREPARE_SLEEP_SECONDS = 4.0
PLOT_SLEEP_SECONDS = 3.0


def _get_output_dir(context: PipelineContext) -> Path:
    analysis = context.config.get("analysis", {})
    return Path(analysis.get("output_dir", "output"))


class PlumeSentinelDemoLoadStage(BaseStage):
    """Simulate loading inputs (~3 s) without touching disk or network."""

    def __init__(self) -> None:
        super().__init__(name="load_inputs")

    def execute(self, context: PipelineContext) -> StageResult:
        start = time.time()
        stub_inputs = {
            "modis_terra_aod_event": "<demo-stub>",
            "goes_event_image": "<demo-stub>",
            "hms_smoke_event": "<demo-stub>",
        }
        steps = list(stub_inputs.keys())
        per_step = LOAD_SLEEP_SECONDS / max(len(steps), 1)
        for i, name in enumerate(steps, 1):
            context.log_progress(f"Loading input: {name} ({i}/{len(steps)})")
            time.sleep(per_step)

        context.metadata["plume_sentinel_loaded"] = stub_inputs
        return self._create_result(
            StageStatus.COMPLETED,
            data={"inputs_loaded": list(stub_inputs.keys()), "demo": True},
            duration=time.time() - start,
        )


class PlumeSentinelDemoPrepareStage(BaseStage):
    """Simulate geospatial preparation (~4 s) and populate stub provenance."""

    def __init__(self) -> None:
        super().__init__(name="prepare_geospatial")

    def execute(self, context: PipelineContext) -> StageResult:
        start = time.time()
        analysis = context.config.get("analysis", {})
        valid_time = str(analysis.get("start_time", "1970-01-01"))
        if "T" not in valid_time:
            valid_time = f"{valid_time}T00:00:00+00:00"

        input_datasets: list[dict[str, Any]] = [
            {
                "name": "MODIS L2 AOD (MOD04)",
                "version": "Collection 6.1",
                "agency": "NASA LAADS",
                "valid_time": valid_time,
                "granules": [],
            },
            {
                "name": "GOES-16 ABI L2 MCMIP",
                "agency": "NOAA NESDIS",
                "valid_time": valid_time,
                "granules": [],
            },
            {
                "name": "NOAA NESDIS HMS Smoke",
                "agency": "NOAA NESDIS",
                "valid_time": valid_time,
                "granules": [],
            },
        ]

        steps = [
            "assembling GOES RGB for goes_event_image",
            "cleaning HMS polygons for hms_smoke_event",
            "binning MODIS AOD to grid for modis_terra_aod_event",
        ]
        per_step = PREPARE_SLEEP_SECONDS / len(steps)
        for s in steps:
            context.log_progress(f"step: {s}")
            time.sleep(per_step)

        context.metadata["plume_sentinel_prepared"] = {
            name: "<demo-stub>" for name in ("modis_terra_aod_event", "goes_event_image", "hms_smoke_event")
        }
        context.metadata["plume_sentinel_input_datasets"] = input_datasets

        return self._create_result(
            StageStatus.COMPLETED,
            data={"inputs_prepared": list(context.metadata["plume_sentinel_prepared"].keys()), "demo": True},
            duration=time.time() - start,
        )


class PlumeSentinelDemoPlotStage(BaseStage):
    """Simulate plotting (~3 s) by scanning ``output_dir`` for existing PNGs."""

    def __init__(self) -> None:
        super().__init__(name="plotting")

    def execute(self, context: PipelineContext) -> StageResult:
        start = time.time()
        output_dir = _get_output_dir(context)
        pngs = sorted(p for p in output_dir.glob("*.png")) if output_dir.is_dir() else []

        if pngs:
            per_step = PLOT_SLEEP_SECONDS / len(pngs)
            for i, p in enumerate(pngs, 1):
                context.log_progress(f"Plot: {p.stem} ({i}/{len(pngs)})")
                time.sleep(per_step)
                context.log_progress(f"done: saved to {p}")
        else:
            time.sleep(PLOT_SLEEP_SECONDS)

        paths = [str(p) for p in pngs]
        context.metadata["plume_sentinel_plots_generated"] = paths
        return self._create_result(
            StageStatus.COMPLETED,
            data={"plots_generated": paths, "demo": True},
            duration=time.time() - start,
        )
```

- [ ] **Step 4: Run tests to verify pass**

```bash
pytest davinci_monet/tests/unit/addons/plume_sentinel/test_demo_stages.py -v
```

Expected: 4 tests pass.

- [ ] **Step 5: Commit**

```bash
git add davinci_monet/addons/plume_sentinel/demo_stages.py davinci_monet/tests/unit/addons/plume_sentinel/test_demo_stages.py
git commit -m "feat(plume-sentinel): demo stages with simulated processing delay"
```

---

## Task 12: Pipeline factory + runner dispatch for `--demo-mode`

**Files:**
- Modify: `davinci_monet/addons/plume_sentinel/workflow.py`
- Modify: `davinci_monet/pipeline/runner.py`
- Modify: `davinci_monet/tests/test_plume_sentinel_workflow_with_bulletin.py`

- [ ] **Step 1: Write failing tests for demo-mode dispatch**

Append to `davinci_monet/tests/test_plume_sentinel_workflow_with_bulletin.py`:

```python
def test_create_plume_sentinel_pipeline_demo_mode_uses_demo_stages():
    from davinci_monet.addons.plume_sentinel.demo_stages import (
        PlumeSentinelDemoLoadStage,
        PlumeSentinelDemoPlotStage,
        PlumeSentinelDemoPrepareStage,
    )

    pipeline = create_plume_sentinel_pipeline(demo_mode=True)
    types = [type(s) for s in pipeline]
    assert types[0] is PlumeSentinelDemoLoadStage
    assert types[1] is PlumeSentinelDemoPrepareStage
    assert types[2] is PlumeSentinelDemoPlotStage
    # Bulletin stage is the same in both modes
    assert pipeline[3].name == "bulletin"


def test_runner_dispatches_to_demo_pipeline_when_flag_set(tmp_path, monkeypatch):
    """When config.analysis._demo.enabled is True, runner uses demo stages."""
    from davinci_monet.pipeline.runner import PipelineRunner

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-fake")
    (tmp_path / "modis_aod_truecolor.png").write_bytes(b"\x89PNG")
    config = {
        "analysis": {
            "output_dir": str(tmp_path),
            "start_time": "2020-09-09",
            "end_time": "2020-09-09",
            "workflow": "plume_sentinel",
            "_demo": {"enabled": True, "canned_bulletin": None},
        },
        "plume_sentinel": {"inputs": {}, "plots": {}, "bulletin": {"include_images": False}},
    }
    fake_resp = BulletinResponse(
        text="==DEMO BULLETIN==", model="claude-sonnet-4-6",
        input_tokens=1, cache_read_tokens=0, output_tokens=1,
    )
    with patch(
        "davinci_monet.addons.plume_sentinel.stages.generate_bulletin",
        return_value=fake_resp,
    ), patch(
        "davinci_monet.addons.plume_sentinel.demo_stages.time.sleep"
    ):
        runner = PipelineRunner(show_progress=False)
        result = runner.run_from_config(config)
    assert result.success
    assert (tmp_path / "bulletin.txt").read_text() == "==DEMO BULLETIN=="
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest davinci_monet/tests/test_plume_sentinel_workflow_with_bulletin.py -v -k demo
```

Expected: FAIL — `create_plume_sentinel_pipeline()` does not accept `demo_mode`.

- [ ] **Step 3: Update the factory signature**

In `davinci_monet/addons/plume_sentinel/workflow.py`, update imports and factory:

```python
from davinci_monet.addons.plume_sentinel.demo_stages import (
    PlumeSentinelDemoLoadStage,
    PlumeSentinelDemoPlotStage,
    PlumeSentinelDemoPrepareStage,
)


def create_plume_sentinel_pipeline(*, demo_mode: bool = False) -> list[BaseStage]:
    """Create the four-stage Plume Sentinel pipeline.

    Parameters
    ----------
    demo_mode:
        When True, replace the load/prepare/plot stages with their demo
        counterparts (simulated delays, no data work). The bulletin stage
        is unchanged. See ``demo_stages`` and the spec at
        ``docs/superpowers/specs/2026-05-14-plume-sentinel-bulletin-stage-design.md``.
    """
    if demo_mode:
        return [
            PlumeSentinelDemoLoadStage(),
            PlumeSentinelDemoPrepareStage(),
            PlumeSentinelDemoPlotStage(),
            PlumeSentinelBulletinStage(),
        ]
    return [
        PlumeSentinelLoadStage(),
        PlumeSentinelPrepareStage(),
        PlumeSentinelPlotStage(),
        PlumeSentinelBulletinStage(),
    ]
```

- [ ] **Step 4: Update the runner dispatch**

In `davinci_monet/pipeline/runner.py`, locate the workflow-dispatch branch that
selects `create_plume_sentinel_pipeline()`. Update it to read the demo flag:

```python
# In whatever method or function dispatches workflows (typically
# PipelineRunner.run_from_config or a helper called by it):
analysis = config.get("analysis", {}) or {}
workflow_name = analysis.get("workflow")
if workflow_name == "plume_sentinel":
    demo_enabled = bool(
        analysis.get("_demo", {}).get("enabled", False)
    )
    stages = create_plume_sentinel_pipeline(demo_mode=demo_enabled)
```

(If the existing dispatch is structured differently — e.g., a registry —
adapt the read of `_demo.enabled` to wherever the factory call happens.)

- [ ] **Step 5: Run tests to verify pass**

```bash
pytest davinci_monet/tests/test_plume_sentinel_workflow_with_bulletin.py -v
```

Expected: all tests pass (4 original + 2 new demo tests = 6).

- [ ] **Step 6: Commit**

```bash
git add davinci_monet/addons/plume_sentinel/workflow.py davinci_monet/pipeline/runner.py davinci_monet/tests/test_plume_sentinel_workflow_with_bulletin.py
git commit -m "feat(plume-sentinel): wire demo_mode through factory + runner dispatch"
```

---

## Task 13: Canned-bulletin handling in `PlumeSentinelBulletinStage`

**Files:**
- Modify: `davinci_monet/addons/plume_sentinel/stages.py`
- Modify: `davinci_monet/tests/unit/addons/plume_sentinel/test_bulletin_stage.py`

- [ ] **Step 1: Write failing tests for canned-bulletin behavior**

Append to `davinci_monet/tests/unit/addons/plume_sentinel/test_bulletin_stage.py`:

```python
def test_stage_existing_live_path_reports_mode_live(tmp_path, monkeypatch):
    """The existing live path must include `mode: live` in stage data."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-fake")
    stage = PlumeSentinelBulletinStage()
    ctx = _make_context(tmp_path, bulletin={})
    fake_resp = BulletinResponse(
        text="BODY", model="claude-sonnet-4-6",
        input_tokens=1, cache_read_tokens=0, output_tokens=1,
    )
    with patch(
        "davinci_monet.addons.plume_sentinel.stages.generate_bulletin",
        return_value=fake_resp,
    ), patch(
        "davinci_monet.addons.plume_sentinel.stages.build_metrics_payload",
        return_value={"event_date": "2020-09-09", "region": "westcoast", "input_datasets": []},
    ):
        result = stage.execute(ctx)
    assert result.data.get("mode") == "live"


def _make_demo_context(tmp_path: Path, canned_bulletin: str | None) -> PipelineContext:
    config = {
        "analysis": {
            "output_dir": str(tmp_path),
            "start_time": "2020-09-09",
            "_demo": {"enabled": True, "canned_bulletin": canned_bulletin},
        },
        "plume_sentinel": {"inputs": {}, "plots": {}, "bulletin": {}},
    }
    ctx = PipelineContext(config=config)
    ctx.metadata["plume_sentinel_config"] = PlumeSentinelConfig(**config["plume_sentinel"])
    ctx.metadata["plume_sentinel_prepared"] = {}
    ctx.metadata["plume_sentinel_plots_generated"] = []
    return ctx


def test_stage_canned_mode_copies_file_and_skips_api(tmp_path):
    canned = tmp_path / "saved.txt"
    canned.write_text("==PRE-SAVED BULLETIN==")
    stage = PlumeSentinelBulletinStage()
    ctx = _make_demo_context(tmp_path, canned_bulletin=str(canned))
    with patch(
        "davinci_monet.addons.plume_sentinel.stages.generate_bulletin"
    ) as gen, patch(
        "davinci_monet.addons.plume_sentinel.stages.build_metrics_payload",
        return_value={"event_date": "2020-09-09", "region": "westcoast", "input_datasets": []},
    ):
        result = stage.execute(ctx)
    assert result.status == StageStatus.COMPLETED
    bulletin_path = tmp_path / "bulletin.txt"
    assert bulletin_path.read_text() == "==PRE-SAVED BULLETIN=="
    assert result.data.get("mode") == "canned"
    assert result.data.get("source") == str(canned)
    gen.assert_not_called()  # API skipped


def test_stage_canned_mode_skips_when_file_missing(tmp_path):
    missing = tmp_path / "nope.txt"
    stage = PlumeSentinelBulletinStage()
    ctx = _make_demo_context(tmp_path, canned_bulletin=str(missing))
    with patch(
        "davinci_monet.addons.plume_sentinel.stages.build_metrics_payload",
        return_value={"event_date": "2020-09-09", "region": "westcoast", "input_datasets": []},
    ):
        result = stage.execute(ctx)
    assert result.status == StageStatus.COMPLETED
    assert result.data.get("bulletin") == "skipped (canned bulletin missing)"
    flags = ctx.metadata.get("plume_sentinel_quality_flags", [])
    assert any(
        f["category"] == "bulletin" and "Canned bulletin not found" in f["message"]
        for f in flags
    )


def test_stage_canned_mode_works_without_api_key(tmp_path, monkeypatch):
    """Canned mode must short-circuit before the API key check."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    canned = tmp_path / "saved.txt"
    canned.write_text("==NO KEY NEEDED==")
    stage = PlumeSentinelBulletinStage()
    ctx = _make_demo_context(tmp_path, canned_bulletin=str(canned))
    with patch(
        "davinci_monet.addons.plume_sentinel.stages.generate_bulletin"
    ) as gen:
        result = stage.execute(ctx)
    assert result.status == StageStatus.COMPLETED
    assert result.data.get("mode") == "canned"
    assert (tmp_path / "bulletin.txt").read_text() == "==NO KEY NEEDED=="
    gen.assert_not_called()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest davinci_monet/tests/unit/addons/plume_sentinel/test_bulletin_stage.py -v -k "canned or mode_live"
```

Expected: FAIL — `mode` key is not in stage data; canned branch not implemented.

- [ ] **Step 3: Update `PlumeSentinelBulletinStage` for canned mode + add `mode` to live result**

In `davinci_monet/addons/plume_sentinel/stages.py`, modify
`PlumeSentinelBulletinStage.execute`. The canned branch must run **before** the
API-key check — canned demo mode does not require an API key. Place it
immediately after the "no bulletin config" early-return:

```python
        # Canned-bulletin mode (demo): read pre-saved bulletin and skip API.
        demo_block = context.config.get("analysis", {}).get("_demo", {}) or {}
        canned_path = demo_block.get("canned_bulletin")
        if canned_path:
            canned = Path(canned_path)
            if not canned.is_file():
                _append_quality_flag(
                    context, "warning",
                    f"Canned bulletin not found at {canned}; bulletin skipped",
                )
                return self._create_result(
                    StageStatus.COMPLETED,
                    data={"bulletin": "skipped (canned bulletin missing)"},
                    duration=time.time() - start,
                )
            output_dir = _get_output_dir(context)
            output_dir.mkdir(parents=True, exist_ok=True)
            bulletin_path = output_dir / bcfg.output_filename
            text = canned.read_text()
            try:
                bulletin_path.write_text(text)
            except OSError as exc:
                _append_quality_flag(
                    context, "warning",
                    f"Bulletin file write failed: {exc}",
                )
                return self._create_result(
                    StageStatus.COMPLETED,
                    data={"bulletin": "skipped (file write failed)"},
                    duration=time.time() - start,
                )

            mqtt_published = False
            if bcfg.mqtt is not None:
                try:
                    publish_mqtt(
                        text=text,
                        broker=bcfg.mqtt.broker,
                        topic=bcfg.mqtt.topic,
                        port=bcfg.mqtt.port,
                        qos=bcfg.mqtt.qos,
                    )
                    mqtt_published = True
                except Exception as exc:  # noqa: BLE001
                    _append_quality_flag(
                        context, "warning",
                        f"MQTT publish to {bcfg.mqtt.broker}:{bcfg.mqtt.port} failed: {exc}",
                    )

            context.metadata["plume_sentinel_bulletin"] = {
                "path": str(bulletin_path),
                "mode": "canned",
                "source": str(canned),
            }
            return self._create_result(
                StageStatus.COMPLETED,
                data={
                    "bulletin_path": str(bulletin_path),
                    "mqtt_published": mqtt_published,
                    "mode": "canned",
                    "source": str(canned),
                },
                duration=time.time() - start,
            )
```

In the existing live-mode `return self._create_result(...)` at the end of
`execute`, add `"mode": "live"` to the data dict:

```python
        return self._create_result(
            StageStatus.COMPLETED,
            data={
                "bulletin_path": str(bulletin_path),
                "mqtt_published": mqtt_published,
                "model": resp.model,
                "input_tokens": resp.input_tokens,
                "cache_read_tokens": resp.cache_read_tokens,
                "output_tokens": resp.output_tokens,
                "mode": "live",
            },
            duration=time.time() - start,
        )
```

Also update the `context.metadata["plume_sentinel_bulletin"]` write in the live
branch to include `"mode": "live"`:

```python
        context.metadata["plume_sentinel_bulletin"] = {
            "path": str(bulletin_path),
            "mode": "live",
            "model": resp.model,
            "input_tokens": resp.input_tokens,
            "cache_read_tokens": resp.cache_read_tokens,
            "output_tokens": resp.output_tokens,
        }
```

- [ ] **Step 4: Run all bulletin-stage tests**

```bash
pytest davinci_monet/tests/unit/addons/plume_sentinel/test_bulletin_stage.py -v
```

Expected: 12 tests pass (5 from Task 8 + 3 from Task 9 + 4 new from Task 13).

- [ ] **Step 5: Commit**

```bash
git add davinci_monet/addons/plume_sentinel/stages.py davinci_monet/tests/unit/addons/plume_sentinel/test_bulletin_stage.py
git commit -m "feat(plume-sentinel): canned-bulletin mode in BulletinStage; emit mode field"
```

---

## Task 14: CLI flags `--demo-mode` and `--demo-bulletin`

**Files:**
- Modify: `davinci_monet/cli/commands/run.py`
- Create or modify: `davinci_monet/tests/unit/cli/test_demo_flags.py`

The CLI flags translate to a transient `config["analysis"]["_demo"]` block
that the runner dispatch reads.

- [ ] **Step 1: Write failing tests for the CLI translation**

Create `davinci_monet/tests/unit/cli/test_demo_flags.py` (or extend an existing
CLI test file if one exists for `run`):

```python
"""Tests that the --demo-mode and --demo-bulletin flags populate config correctly."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from davinci_monet.cli.commands.run import _apply_demo_flags


def test_apply_demo_flags_default_off():
    cfg = {"analysis": {"output_dir": "out"}}
    _apply_demo_flags(cfg, demo_mode=False, demo_bulletin=None)
    assert "_demo" not in cfg["analysis"]


def test_apply_demo_flags_enabled_no_canned():
    cfg = {"analysis": {"output_dir": "out"}}
    _apply_demo_flags(cfg, demo_mode=True, demo_bulletin=None)
    assert cfg["analysis"]["_demo"] == {"enabled": True, "canned_bulletin": None}


def test_apply_demo_flags_enabled_with_canned():
    cfg = {"analysis": {"output_dir": "out"}}
    _apply_demo_flags(cfg, demo_mode=True, demo_bulletin="/tmp/saved.txt")
    assert cfg["analysis"]["_demo"] == {
        "enabled": True, "canned_bulletin": "/tmp/saved.txt",
    }


def test_apply_demo_flags_bulletin_without_demo_mode_raises():
    cfg = {"analysis": {"output_dir": "out"}}
    with pytest.raises(ValueError, match="--demo-bulletin requires --demo-mode"):
        _apply_demo_flags(cfg, demo_mode=False, demo_bulletin="/tmp/x.txt")


def test_apply_demo_flags_initialises_analysis_block():
    cfg: dict = {}
    _apply_demo_flags(cfg, demo_mode=True, demo_bulletin=None)
    assert "analysis" in cfg
    assert cfg["analysis"]["_demo"] == {"enabled": True, "canned_bulletin": None}
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest davinci_monet/tests/unit/cli/test_demo_flags.py -v
```

Expected: FAIL with `ImportError: cannot import name '_apply_demo_flags'`.

- [ ] **Step 3: Add `_apply_demo_flags` helper and `--demo-*` CLI flags**

In `davinci_monet/cli/commands/run.py`, add the helper near the top of the file:

```python
def _apply_demo_flags(
    config: dict,
    *,
    demo_mode: bool,
    demo_bulletin: str | None,
) -> None:
    """Translate ``--demo-mode`` / ``--demo-bulletin`` into config state.

    Mutates ``config['analysis']['_demo']`` when ``demo_mode`` is True. The
    leading underscore signals "not part of the YAML schema" — these values
    come from CLI flags only.
    """
    if demo_bulletin is not None and not demo_mode:
        raise ValueError("--demo-bulletin requires --demo-mode")
    if not demo_mode:
        return
    analysis = config.setdefault("analysis", {})
    analysis["_demo"] = {"enabled": True, "canned_bulletin": demo_bulletin}
```

Then, in the existing CLI command function (typically decorated with `@click.command()`
or `@typer.Typer().command()` — match the existing convention), add the two
flags. The existing flag pattern in this file (e.g., the existing
`--emit-metrics-json` flag) is the reference; this snippet shows the Typer style
but adapt to whatever the file uses:

```python
# Inside the run command's parameter list, alongside --emit-metrics-json:
demo_mode: bool = typer.Option(
    False, "--demo-mode",
    help="Skip data load/prepare/plot. Reuse PNGs in output_dir; simulate processing.",
),
demo_bulletin: str | None = typer.Option(
    None, "--demo-bulletin",
    help="Path to a pre-saved bulletin; skips the Claude API call (requires --demo-mode).",
),
```

And call the helper before the existing pipeline launch:

```python
_apply_demo_flags(config, demo_mode=demo_mode, demo_bulletin=demo_bulletin)
```

- [ ] **Step 4: Run tests to verify pass**

```bash
pytest davinci_monet/tests/unit/cli/test_demo_flags.py -v
```

Expected: 5 tests pass.

- [ ] **Step 5: Manual smoke test of the CLI surface**

```bash
davinci-monet run --help | grep -E "demo-mode|demo-bulletin"
```

Expected: both flags appear in the help text.

- [ ] **Step 6: Commit**

```bash
git add davinci_monet/cli/commands/run.py davinci_monet/tests/unit/cli/test_demo_flags.py
git commit -m "feat(cli): add --demo-mode and --demo-bulletin flags for plume-sentinel"
```

---

## Task 15: Final verification — full suite + smoke runs

**Files:** none modified.

- [ ] **Step 1: Run the full pytest suite**

```bash
pytest davinci_monet/tests/ -v
```

Expected: all tests pass (961 prior + ~40 new ≈ 1001 tests).

- [ ] **Step 2: Type-check the addon and CLI**

```bash
mypy davinci_monet/addons/plume_sentinel/ davinci_monet/cli/commands/run.py
```

Expected: no new errors.

- [ ] **Step 3: Format and lint**

```bash
black davinci_monet/addons/plume_sentinel/ davinci_monet/cli/commands/run.py davinci_monet/tests/unit/addons/plume_sentinel/ davinci_monet/tests/unit/cli/ davinci_monet/tests/test_plume_sentinel_workflow_with_bulletin.py
isort davinci_monet/addons/plume_sentinel/ davinci_monet/cli/commands/run.py davinci_monet/tests/unit/addons/plume_sentinel/ davinci_monet/tests/unit/cli/ davinci_monet/tests/test_plume_sentinel_workflow_with_bulletin.py
```

- [ ] **Step 4: Demo-mode smoke test (canned bulletin, no API key needed)**

```bash
# Uses existing PNGs in west-coast-smoke/output and the saved report.txt
davinci-monet run analyses/plume-sentinel/configs/modis-aod-truecolor-gemini.yaml \
    --demo-mode \
    --demo-bulletin report.txt
```

Watch the terminal — you should see ~10 s of simulated load/prepare/plot progress
followed by the bulletin appearing at `analyses/west-coast-smoke/output/bulletin.txt`,
identical to `report.txt`.

- [ ] **Step 5: Live demo-mode smoke test (requires `ANTHROPIC_API_KEY`)**

```bash
ANTHROPIC_API_KEY=sk-... davinci-monet run \
    analyses/plume-sentinel/configs/modis-aod-truecolor-gemini.yaml \
    --demo-mode
```

Inspect the freshly generated `bulletin.txt` and diff it against the canned
`report.txt` to sanity-check the live path. Do not commit either artifact.

- [ ] **Step 6: Commit any formatting changes**

```bash
git add -A
git status   # verify only formatting changes
git commit -m "style: format plume-sentinel bulletin stage + demo mode"
```

---

## Self-Review Notes

**Spec coverage check:**
- New stage `PlumeSentinelBulletinStage` — Task 8.
- `bulletin.py` with `build_prompt`, `generate_bulletin`, `publish_mqtt`, `BulletinResponse` — Tasks 5, 6, 7.
- Schema (`MqttConfig`, `BulletinConfig`, optional `bulletin` field) — Task 4.
- Packaged template — Task 3.
- Refactor `_build_metrics_payload` → `build_metrics_payload` — Task 1.
- Dependencies (`anthropic`, `paho-mqtt`) — Task 2.
- Five-step execute flow (read config, assemble metrics, build prompt, call API, write + publish) — Task 8.
- Stage result data + bulletin field in metrics payload — Tasks 8 and 10.
- Three-block cached system prompt (persona + partial bulletin + `<metrics>`) with single `cache_control` marker on last block — Task 7.
- Vision opt-in via `include_images` — Tasks 7 and 8 (includes-images / omits-images tests).
- Error handling (warn-and-continue, all six conditions from the spec) — Tasks 8 and 9.
- Demo stages with simulated processing delay — Task 11.
- Pipeline factory + runner dispatch threading `demo_mode` — Task 12.
- Canned-bulletin mode in `PlumeSentinelBulletinStage` (skip API, copy file, `mode: canned`) — Task 13.
- CLI flags `--demo-mode` and `--demo-bulletin` — Task 14.
- Stage result `mode` field for live vs canned — Tasks 8/10/13.

**Internal consistency:**
- `build_metrics_payload(...)` keyword signature stays consistent across workflow.py (Task 1), stages.py (Task 8), and tests (Tasks 8, 9, 10).
- `BulletinResponse` dataclass fields (`text`, `model`, `input_tokens`, `cache_read_tokens`, `output_tokens`, `skipped_images`) used uniformly in Tasks 7, 8, 9, 10.
- Stage data keys (`bulletin_path`, `mqtt_published`, `model`, `input_tokens`, `cache_read_tokens`, `output_tokens`, `mode`) consistent between Tasks 8, 10, and 13 (Task 13 adds the `mode` field to both branches).
- `plume_sentinel_bulletin` metadata key shape: live mode (`path`, `mode`, `model`, token counts) vs canned mode (`path`, `mode`, `source`) — both branches set it; the metrics payload extension in Task 10 just copies whichever variant is present.
- `config["analysis"]["_demo"]` shape (`{"enabled": bool, "canned_bulletin": str | None}`) consistent between CLI translation (Task 14), runner dispatch (Task 12), and bulletin-stage canned branch (Task 13).
- Demo-stage metadata keys (`plume_sentinel_loaded`, `plume_sentinel_prepared`, `plume_sentinel_input_datasets`, `plume_sentinel_plots_generated`) match the keys read by `build_metrics_payload` and the bulletin stage.

**Placeholder scan:** no TBDs, TODOs, or "implement later" markers.
