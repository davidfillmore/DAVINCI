# Plume Sentinel Bulletin Stage — Design

**Date:** 2026-05-14
**Branch:** `feature/plume-sentinel-addon` (extension)
**Status:** Approved for implementation planning

## Context

The PlumeSentinel add-on currently produces two artifacts per run: rendered PNG plots
(true-color + AOD, true-color + HMS) and a `plumesentinel.metrics.v1` JSON sidecar.
The accompanying meteorological bulletin (`report.txt`) is generated outside the pipeline
by an interactive Claude Code session driven by the `plume-sentinel` skill: the model
inspects the plots, fills a template, and publishes the result via MQTT.

This design moves bulletin generation and MQTT publishing inside the DAVINCI pipeline
itself, so that `davinci-monet run config.yaml` produces a complete, ready-to-publish
event bulletin with no interactive Claude Code step. The bulletin becomes a first-class
build artifact of the pipeline, reproducible from the config alone.

## Goals

- One command (`davinci-monet run config.yaml`) produces plots, metrics JSON, the
  rendered bulletin, and optionally publishes to an MQTT broker.
- The bulletin is opt-in via a `bulletin:` block in the `plume_sentinel` config — runs
  without that block are unaffected.
- The pipeline is robust to bulletin-stage failures: missing API key, transient
  Anthropic outages, MQTT broker failures, and template errors all degrade to
  warnings recorded as `quality_flags`, never failing an otherwise-successful run.
- Bulletins remain narrative-quality. The default mode uses the structured metrics
  payload as input; an opt-in vision mode additionally attaches the rendered PNGs so
  the model can reason about plume shape, retrieval gaps, and visual context.

## Non-Goals

- Generating bulletins for non-PlumeSentinel workflows.
- Supporting LLM providers other than Anthropic.
- Live-API integration tests in CI (the Anthropic client is mocked at the SDK
  boundary in all automated tests).
- A general-purpose publish framework. MQTT is the only sink in scope; webhook/S3/email
  sinks are deliberately deferred.
- Retrying failed API or MQTT calls. The stage fails fast and records the error.

## Architecture

### New stage

`PlumeSentinelBulletinStage`, added to
`davinci_monet/addons/plume_sentinel/stages.py` and appended after
`PlumeSentinelPlotStage` in `create_plume_sentinel_pipeline()`:

```
[Load] → [Prepare] → [Plot] → [Bulletin (new)]
```

### New module

`davinci_monet/addons/plume_sentinel/bulletin.py` — pure helpers:

- `build_prompt(template_text: str, metrics_payload: dict, analysis_cfg: dict) -> str`
  Fills the deterministic placeholders (`{{BULLETIN_ID}}`, `{{ISSUED_DATE}}`,
  `{{EVENT_DATE}}`, `{{OBSERVATION_TIME}}`, `{{SENSOR_SOURCES}}`) from the metrics
  payload and analysis config. Leaves the AI-analyzed placeholders (`{{REGION}}`,
  `{{SEVERITY}}`, `{{SYNOPTIC_OVERVIEW}}`, `{{AOD_ANALYSIS}}`, `{{HMS_ANALYSIS}}`,
  `{{HEALTH_IMPACTS}}`, `{{ASSESSMENT}}`) in place for the model to fill. Returns
  the fully-prepared partial bulletin to embed in the API system prompt.

- `generate_bulletin(*, prompt, image_paths, model, api_key, max_tokens=4096,
  temperature=0.2) -> BulletinResponse`
  Calls `anthropic.Anthropic(api_key=...).messages.create(...)`. Uses prompt caching
  (`cache_control: {type: "ephemeral"}`) on the persona-and-template system content
  so repeated runs amortize the static prefix. When `image_paths` is non-empty,
  attaches each PNG as a `{type: image, source: {type: base64, media_type: image/png,
  data: ...}}` content block in the user message. Returns a small dataclass
  `BulletinResponse(text, model, input_tokens, cache_read_tokens, output_tokens)`.

- `publish_mqtt(*, text, broker, topic, port=1883, qos=0) -> None`
  Uses `paho-mqtt` to connect, publish the bulletin text, and disconnect.
  Synchronous; raises on connect/publish failure (caller catches and converts to a
  `quality_flag`).

### Schema additions

In `davinci_monet/addons/plume_sentinel/schema.py`:

```python
class MqttConfig(BaseModel):
    broker: str = "broker.hivemq.com"
    topic: str
    port: int = 1883
    qos: int = 0

class BulletinConfig(BaseModel):
    template: str | None = None
    output_filename: str = "bulletin.txt"
    model: str = "claude-sonnet-4-6"
    include_images: bool = False
    api_key_env: str = "ANTHROPIC_API_KEY"
    mqtt: MqttConfig | None = None
    on_error: str = "warn"

class PlumeSentinelConfig(BaseModel):
    inputs: dict[str, InputSpec]
    plots: dict[str, PlotSpec]
    bulletin: BulletinConfig | None = None
```

`on_error` is fixed at `"warn"` for the initial implementation. The field is reserved
for a future `"fail"` mode but the schema only accepts `"warn"` for now (validated by
a `field_validator`). This keeps the surface stable without committing to fail-mode
semantics today.

### Packaged template

The current template at
`analyses/plume-sentinel/templates/bulletin.template` is copied to
`davinci_monet/addons/plume_sentinel/templates/bulletin.template` and bundled with the
package (`importlib.resources`). The packaged template is the default when
`BulletinConfig.template` is `None`. The analyses copy is kept so the existing skill
flow continues to work; the packaged copy is the authoritative one for in-pipeline use.

### Refactor

`_build_metrics_payload(...)` in `workflow.py` is moved to a new
`davinci_monet/addons/plume_sentinel/metrics_payload.py` module so it can be called by
both the `--emit-metrics-json` path in `run()` and the new bulletin stage. The behavior
is unchanged — this is a pure code move, covered by the existing
`test_metrics_extraction.py` tests. A new field `bulletin: {path, model, tokens}` is
appended to the payload schema when the bulletin stage runs.

### Dependencies

Added to `environment.yml` under the `pip:` block:

- `anthropic` (Claude API SDK)
- `paho-mqtt` (MQTT client)

## Data Flow

`PlumeSentinelBulletinStage.execute(context)` proceeds in five steps:

1. **Read config.** Pull `cfg.bulletin` from `context.metadata["plume_sentinel_config"]`.
   If `None`, return `StageStatus.COMPLETED` with `data={"bulletin": "skipped (no
   config)"}` — no-op. If the env var named by `cfg.bulletin.api_key_env` is unset,
   append a `quality_flag` and return `COMPLETED` with `data={"bulletin": "skipped
   (no api key)"}`.

2. **Assemble metrics payload.** Call the refactored
   `metrics_payload.build_metrics_payload(...)` with the run's
   `context.metadata`, config, and stage results. This is the same payload that
   `--emit-metrics-json` writes.

3. **Build prompt.** Load template text from `cfg.bulletin.template` (or
   `importlib.resources.read_text("davinci_monet.addons.plume_sentinel.templates",
   "bulletin.template")`). Call `build_prompt()` to fill deterministic placeholders.
   The unfilled AI placeholders remain in the text.

4. **Call Anthropic API.** Build the system content as a list of three text blocks:
   (a) a persona block ("You are PlumeSentinel AI, an automated meteorological
   analysis system. Replace each remaining `{{PLACEHOLDER}}` token with your
   authoritative analysis. Return only the rendered bulletin text, no commentary.");
   (b) the partial-bulletin block (template with deterministic placeholders already
   filled); (c) a `<metrics>` block containing the full metrics payload as JSON, so
   the model has the quantitative facts (peak AOD, HMS areas, retrieval gaps, plot
   URLs) to write the analysis sections. The last block carries
   `cache_control={"type": "ephemeral"}`, which caches the entire preceding system
   prefix as a single segment. The user message contains a short directive ("Render
   the bulletin for this event."), preceded by image content blocks when
   `include_images=True` and the plotting stage produced files. Temperature 0.2,
   `max_tokens=4096`.

5. **Write output + publish.**
   Write the response text to `<output_dir>/<cfg.bulletin.output_filename>`.
   If `cfg.bulletin.mqtt` is set, call `publish_mqtt(...)`. Both operations are
   independently wrapped in try/except; failures become `quality_flags` and do not
   abort the stage.

### Stage result

```python
StageResult.data = {
    "bulletin_path": "/.../output/bulletin.txt",
    "mqtt_published": True | False,
    "model": "claude-sonnet-4-6",
    "input_tokens": int,
    "cache_read_tokens": int,
    "output_tokens": int,
}
```

The metrics payload gains an optional top-level field:

```json
"bulletin": {
  "path": "bulletin.txt",
  "model": "claude-sonnet-4-6",
  "input_tokens": 4231,
  "cache_read_tokens": 3890,
  "output_tokens": 1402
}
```

## Error Handling

All bulletin-stage failures are non-fatal. Each is recorded as a `quality_flag` with
`category: "bulletin"` and an appropriate `severity` (`warning` or `info`):

| Failure condition | Severity | Message |
|---|---|---|
| `bulletin:` block absent | — | no flag, stage is a no-op |
| `ANTHROPIC_API_KEY` env var missing | warning | `"API key env var <name> not set; bulletin skipped"` |
| `anthropic.APIError` / network failure | warning | `"Claude API call failed: <error>; bulletin skipped"` |
| Template file missing or unreadable | warning | `"Bulletin template not found at <path>; bulletin skipped"` |
| Response missing expected placeholders | info | `"Bulletin response missing placeholders: [<list>]"` (raw text still written) |
| MQTT broker connection or publish failure | warning | `"MQTT publish to <broker>:<port> failed: <error>"` (bulletin.txt still written) |
| Image file referenced but missing on disk | info | `"Bulletin image not found: <path>"` (skipped, remaining images still attached) |

The bulletin stage never returns `StageStatus.FAILED`. The pipeline's overall
success/failure status reflects the upstream load/prepare/plot stages only.

## Prompt Caching

The system content (persona block + partial-bulletin block + metrics JSON block) is
the largest static portion of each request. The last block carries
`cache_control: {type: "ephemeral"}`, which caches the entire preceding system
prefix as one segment (Anthropic caches up to the marker). To preserve cache hits
across runs:

- The packaged template is loaded as text and not reformatted.
- The metrics-payload JSON in block (c) is serialized with
  `json.dumps(payload, indent=2, sort_keys=True, default=str, ensure_ascii=False)`
  so byte-identical inputs produce byte-identical prompts.
- Image content blocks live in the user message (outside the cached region), so
  toggling `include_images` does not invalidate the cached system prefix.

Cache-read token counts are surfaced in `StageResult.data` and the metrics payload to
make cache hit rate observable.

## Testing

All tests live in `davinci_monet/tests/unit/addons/plume_sentinel/`. The
`anthropic.Anthropic` class is mocked at the SDK boundary; no live-API calls in CI.

### Unit tests (`test_bulletin.py`)

1. `test_build_prompt_fills_deterministic_placeholders` — fixture metrics payload →
   `{{BULLETIN_ID}}`, `{{EVENT_DATE}}`, `{{OBSERVATION_TIME}}`, `{{SENSOR_SOURCES}}`,
   `{{ISSUED_DATE}}` are populated; AI placeholders remain literal.
2. `test_build_prompt_omits_when_metrics_field_missing` — partial metrics payload
   (e.g. no HMS data) still renders without crashing.
3. `test_publish_mqtt_uses_config` — mock `paho.mqtt.client.Client`, verify
   `connect(broker, port)` and `publish(topic, payload, qos)` called with config
   values.
4. `test_bulletin_stage_skips_when_no_config` — `cfg.bulletin=None` → stage returns
   `COMPLETED`, no API calls, no file written, no quality_flag.
5. `test_bulletin_stage_skips_when_no_api_key` — env var unset → `COMPLETED` with
   one `quality_flag`, no API call, no file written.
6. `test_bulletin_stage_handles_api_error` — mocked `Anthropic.messages.create`
   raises `anthropic.APIError` → `COMPLETED` with one `quality_flag`, no file written.
7. `test_bulletin_stage_writes_file_on_success` — mocked SDK returns canned response
   → `bulletin.txt` contains the canned text; stage `data` contains
   `bulletin_path` and token counts.
8. `test_bulletin_stage_includes_images_when_configured` — `include_images=True` →
   the API call's `messages[0].content` includes one `image` block per file in
   `plots_generated`, base64-encoded; absent files are skipped with an `info` flag.
9. `test_bulletin_stage_mqtt_failure_does_not_break_stage` — bulletin.txt written
   successfully but `publish_mqtt` raises → `COMPLETED`, `quality_flag` recorded,
   `mqtt_published=False` in stage data.

### Integration test

`davinci_monet/tests/test_plume_sentinel_workflow_with_bulletin.py` extends the
existing `test_plume_sentinel_workflow.py` with synthetic inputs + a mocked
`anthropic.Anthropic`. Asserts:

- Pipeline `result.success == True`
- `bulletin.txt` exists in the run output dir with the mocked response text
- The emitted metrics JSON includes the `bulletin` field with `model`, `path`, and
  token counts
- `quality_flags` is empty for the happy path

A regression test confirms that omitting the `bulletin:` block leaves the existing
test_plume_sentinel_workflow.py behavior unchanged (no new outputs, no new flags).

## Open Questions

None. All design decisions agreed during brainstorming on 2026-05-14:

- Vision is opt-in via `include_images: true`
- Bulletin is triggered by presence of `bulletin:` config block (no CLI flag)
- MQTT publishing lives inside the bulletin stage
- All failures warn-and-continue, recorded as `quality_flags`
- Default model is `claude-sonnet-4-6` (vision-capable, balanced)
- Template defaults to the packaged copy; user configs may override

## Demo Mode

A demo mode lets users showcase the PlumeSentinel pipeline without re-running
data ingest, regridding, or plotting. Pre-existing PNGs in the configured
`output_dir` stand in for freshly generated plots, and the run includes a
simulated processing delay so the terminal output is indistinguishable from a
real run.

### CLI

Two flags added to `davinci-monet run`:

```
--demo-mode                  Skip load/prepare/plot stages. Reuse existing PNGs in output_dir.
--demo-bulletin PATH         (Requires --demo-mode.) Use the saved bulletin at PATH instead
                             of calling the Claude API. Implies "canned" bulletin mode.
```

Without `--demo-bulletin`, demo mode runs *live* — it still calls the Claude API
(typically with `include_images: true` so the model sees the pre-existing plots).

### Architecture

Three demo stages mirror the real ones, in a new module
`davinci_monet/addons/plume_sentinel/demo_stages.py`:

- `PlumeSentinelDemoLoadStage` — sleeps ~3 s, emits realistic "Loading input"
  progress messages, populates `context.metadata["plume_sentinel_loaded"]`
  with stub markers.
- `PlumeSentinelDemoPrepareStage` — sleeps ~4 s, emits "GeoOp" progress,
  populates `plume_sentinel_prepared` stubs and a canonical
  `plume_sentinel_input_datasets` list (MODIS + GOES + HMS) so the metrics
  payload provenance is plausible.
- `PlumeSentinelDemoPlotStage` — sleeps ~3 s, scans `<output_dir>/*.png` for
  pre-existing files and populates `context.metadata["plume_sentinel_plots_generated"]`
  with the matching paths. Emits a `"done: saved to <path>"` line per file
  so the terminal output looks identical to a real plot stage.

Total simulated wallclock is ~10 s, split as `load=3s, prepare=4s, plot=3s`.
All sleeps are `time.sleep()` so tests can patch them to zero.

### Pipeline factory

`create_plume_sentinel_pipeline()` gains a keyword-only `demo_mode: bool = False`
parameter. When `True`, returns `[DemoLoad, DemoPrepare, DemoPlot, Bulletin]`;
otherwise returns the existing four-stage list. The bulletin stage is the same
class in both modes.

### Configuration plumbing

CLI flags are stashed onto a transient config field at runtime — they do *not*
live in user YAML. The CLI writes:

```python
config["analysis"]["_demo"] = {
    "enabled": True,
    "canned_bulletin": "/path/to/saved.txt" | None,
}
```

The leading underscore signals "not part of the YAML schema". The runner's
workflow-dispatch logic reads `_demo.enabled` and passes `demo_mode=True` to
the factory. The bulletin stage reads `_demo.canned_bulletin` from
`context.config["analysis"]` to decide between live and canned modes.

### Canned bulletin handling

When `_demo.canned_bulletin` is set, `PlumeSentinelBulletinStage`:

1. Reads the file at that path
2. Writes it to `<output_dir>/<bulletin.output_filename>` unchanged
3. Skips the Anthropic API call entirely
4. Still publishes to MQTT if `mqtt:` is configured
5. Reports `mode: "canned"` in `StageResult.data`; no token counts

Missing-file or unreadable canned bulletin warns and skips, same as the other
bulletin stage failures.

### Stage result contract

```python
# Live mode (real or demo)
data = {
    "bulletin_path": str,
    "mqtt_published": bool,
    "model": str,
    "input_tokens": int,
    "cache_read_tokens": int,
    "output_tokens": int,
    "mode": "live",
}

# Canned mode
data = {
    "bulletin_path": str,
    "mqtt_published": bool,
    "mode": "canned",
    "source": str,  # path of the canned file
}
```

## Out of Scope (deferred)

- Configurable retry behavior on API/MQTT failures.
- Live-API smoke tests in CI.
- Additional output sinks (S3, webhook, email).
- `on_error: "fail"` mode (schema field is reserved but only `"warn"` is accepted).
- Streaming response handling — full responses only.
- Migrating other workflows to use the bulletin stage.
- Demo mode for non-PlumeSentinel workflows.
- Configurable demo timing (sleep durations are constants in this iteration).
