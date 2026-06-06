# AI Analysis Summary — Design Spec

**Date:** 2026-06-05
**Status:** Approved design, ready for implementation plan
**Branch:** develop

## Context

DAVINCI was rebranded to **Data Analysis and Visual Intelligence for Climate**.
This feature delivers the "Visual Intelligence" payoff: a single-prompt AI
summary generated for each analysis run, using the Claude API (Anthropic SDK)
with **vision** so the model reads the generated plots, not just the numbers.

The summary is an **opt-in final pipeline stage**. When enabled it produces a
structured markdown brief from the run's statistics, config metadata, and a
selected subset of the generated plot images. It is always **non-fatal**: the
analysis (stats, plots, CSVs) is already complete before this stage runs, so any
failure (missing key, missing dependency, network/API error) is logged as a
warning and the stage is skipped — the run is still reported successful.

### Decisions locked during brainstorming

| Decision | Choice |
|----------|--------|
| Summary purpose | **Both, structured**: "what this run is" + headline metrics + interpretation + caveats |
| Trigger | **Opt-in pipeline stage** (runs after `save_results` when `summary.enabled: true`) |
| Prompt input | **Stats + plot images (vision)** |
| Plot selection | **Config-selected subset**, defaulting to all generated plots capped at `max_images` |
| Output | **Markdown file + terminal** |
| Failure mode | **Non-fatal, warn** (always; no `required` flag) |
| Code structure | **Approach A**: pure engine module (`davinci_monet/ai/`) + thin `SummaryStage` |
| Default model | **`claude-haiku-4-5`** (cheapest, fastest, vision-capable; configurable) |
| Provider | **Anthropic Claude API directly** (not the `OpenRouter.api` key present in repo) |

## Goals

- Generate a readable, structured markdown brief per analysis run.
- Let Claude visually interpret the run's figures (spatial bias maps, scatter,
  time series) alongside the statistics.
- Keep the summary entirely optional and never able to fail an otherwise-complete
  run.
- Keep API/prompt logic out of the pipeline plumbing and fully unit-testable
  without a network call.

## Non-Goals

- No standalone `davinci-monet summarize` CLI command (engine leaves the door
  open to add one later).
- No use of the OpenRouter key in the repo.
- No multi-turn / agentic interaction — strictly a single prompt, single
  response.
- No streaming UI.

## Architecture

### New subpackage: `davinci_monet/ai/`

Small, focused modules following the repo's "<500 lines each" principle.

- **`ai/__init__.py`** — public exports: `generate_summary`, `SummaryResult`,
  `SummaryPayload`, `collect_payload`, `SummaryConfig`.

- **`ai/config.py`** — `SummaryConfig` Pydantic model (mirrors how `stats`/`plots`
  blocks are parsed elsewhere in `davinci_monet/config/`):

  | Field | Type | Default | Meaning |
  |-------|------|---------|---------|
  | `enabled` | `bool` | `False` | Stage is a no-op unless true |
  | `model` | `str` | `"claude-haiku-4-5"` | Anthropic model id |
  | `max_tokens` | `int` | `2000` | Response token cap |
  | `api_key_env` | `str` | `"ANTHROPIC_API_KEY"` | Env var holding the key |
  | `plots` | `list[str] \| None` | `None` | Plot keys to attach; `None` → all generated, capped |
  | `max_images` | `int` | `8` | Cap when `plots` is not given |
  | `output_filename` | `str` | `"AI_summary.md"` | Written into `analysis.output_dir` |
  | `instructions` | `str \| None` | `None` | Optional extra steering appended to the prompt |

- **`ai/payload.py`** — `SummaryPayload` dataclass and
  `collect_payload(context, cfg) -> SummaryPayload`. Pulls from `PipelineContext`:
  - statistics from `context.results["statistics"].data` (and
    `context.results["obs_statistics"].data` when present) — richer than the CSV;
  - config metadata from `context.config`: analysis period (`start_time`/
    `end_time`), `output_dir`, sources, variables, and pair definitions;
  - plot image paths from the plotting stages' `StageResult.data["plots_generated"]`,
    selecting `cfg.plots` (by plot key) or the first `cfg.max_images` PNGs.
  - `SummaryPayload` fields: `period`, `sources_summary`, `pairs_summary`,
    `stats_rows` (flattened metrics, reusing the same metric-key resolution as
    `SaveResultsStage`), `images` (list of `(plot_key, title, path)`),
    `instructions`.

- **`ai/images.py`** — `encode_image(path, max_edge=1568) -> EncodedImage`:
  load PNG via Pillow, downscale the long edge to ≤1568px (keeps vision token
  cost predictable on 300-DPI plots), re-encode PNG, base64. Returns
  `{media_type: "image/png", data: <base64>}`. Non-PNG paths (e.g. PDF) are
  skipped with a debug log.

- **`ai/summarizer.py`** — the engine:
  - `build_prompt(payload: SummaryPayload) -> (system: list[block], user_content: list[block])`
    — pure, deterministic, no network. Builds the system prompt (static, marked
    with `cache_control` for prompt caching) and the user content (one text block
    + N image blocks, each preceded by a caption).
  - `generate_summary(payload, *, client=None, cfg) -> SummaryResult` — lazy-imports
    `anthropic`; uses an injected `client` or constructs
    `anthropic.Anthropic(api_key=os.environ[cfg.api_key_env])`; calls
    `client.messages.create(...)`; returns
    `SummaryResult(markdown, model, usage, plots_used, images_sent)`.
  - Raises a typed `SummaryError` for: missing `anthropic` package, missing/empty
    API key, or API/network failure. The stage catches these and degrades
    non-fatally.

### New stage: `SummaryStage` (in `davinci_monet/pipeline/stages.py`)

Thin wiring layer. Appended after `SaveResultsStage` in **both**
`create_standard_pipeline()` and `create_obs_pipeline()`.

```
class SummaryStage(BaseStage):
    name = "summary"

    def execute(self, context):
        cfg = SummaryConfig.parse(context.config.get("summary"))
        if not cfg.enabled:
            return SKIPPED("summary disabled")
        payload = collect_payload(context, cfg)
        try:
            result = generate_summary(payload, cfg=cfg)
        except SummaryError as e:
            context.log.warning(f"AI summary skipped: {e}")
            return SKIPPED(reason=str(e))            # NON-FATAL
        out = Path(context.config["analysis"]["output_dir"]) / cfg.output_filename
        out.write_text(result.markdown)
        context.log_progress(<print rendered markdown to terminal>)
        return COMPLETED(data={
            "summary_file": str(out),
            "model": result.model,
            "usage": result.usage,
        })
```

If `StageStatus` has no `SKIPPED` member, the stage returns `COMPLETED` with
`data={"skipped": reason}` instead — to be confirmed against the enum at
`stages.py:22` during planning. Either way the run is **not** failed.

## Data Flow

```
StatisticsStage ─┐
PlottingStage   ─┤ results["statistics"].data, plots_generated, config
ObsStatistics   ─┤
ObsPlotting     ─┤
SaveResultsStage─┘
        │
        ▼
   SummaryStage
        │ collect_payload() → stats rows + config recap + selected PNG paths
        ▼
   ai.generate_summary()
        │ build_prompt() → system + (text + image blocks)
        │ anthropic.messages.create(model, max_tokens, system, messages)
        ▼
   SummaryResult(markdown, model, usage)
        │
        ├── write output_dir/AI_summary.md
        └── print markdown to terminal
```

## Prompt Design (single prompt, vision)

**System prompt** (static; `cache_control: ephemeral`): establishes the role — a
climate / atmospheric-composition model-evaluation analyst — and requires a
structured markdown brief with these fixed sections:

- `## What this run is` — sources, period, variables, pairing.
- `## Headline metrics` — the key statistics, called out per variable / pair.
- `## Interpretation` — where the model agrees/disagrees with observations, and
  patterns visible in the attached figures.
- `## Caveats` — sample size, what the metrics do not capture.

**User message** — a single content array:

1. One text block: config recap (period, sources, pairs) + a compact stats table
   rendered from `payload.stats_rows` + optional `payload.instructions`.
2. N image blocks, each preceded by a one-line caption text block:
   `Figure: <plot_key> — <title>`.

## Error Handling — always non-fatal

| Condition | Behavior |
|-----------|----------|
| `summary.enabled` false / absent | Stage SKIPPED, no API touch |
| `anthropic` not installed | warn "install davinci-monet[ai]", SKIPPED |
| API key env var missing/empty | warn, SKIPPED |
| Network / API error | warn with error, SKIPPED |
| No plots / no stats available | still summarize from whatever exists; if truly empty, SKIPPED with note |

The analysis run's success status is **never** affected by this stage.

## Dependencies

- Add `anthropic` to a new `[ai]` optional-extra in `pyproject.toml`
  (`optional-dependencies.ai = ["anthropic>=0.40"]`) and to the `pip:` section of
  `environment.yml` so the `davinci` conda env has it.
- `Pillow` is already present transitively via matplotlib; confirm during
  planning and add explicitly to `[ai]` if not guaranteed.
- All AI imports are **lazy** (inside the engine), so a base install without
  `[ai]` runs the pipeline fine and the stage skips with a clear install hint.

## Testing

Per the repo testing rules (integration tests must run through
`PipelineRunner.run_from_config()`; no shortcuts to green).

**Unit tests (pure, no network):**
- `build_prompt` with a synthetic `SummaryPayload` → assert all four section
  headings are requested, image-block count equals selected plots, captions
  present, instructions appended when set.
- `collect_payload` with a synthetic `PipelineContext` (stats dict + config +
  fake plot paths) → assert stats rows flattened correctly, plot selection honors
  `cfg.plots` and the `max_images` cap.
- `encode_image` on a tiny generated PNG → assert downscale to ≤1568px and valid
  base64 / media type.
- `SummaryConfig` parsing → defaults applied, types validated.

**Integration tests (through `PipelineRunner.run_from_config`):**
- Synthetic-data config with `summary.enabled: true` and an **injected stub
  Anthropic client** returning canned markdown → assert `AI_summary.md` is
  written into `output_dir`, terminal output present, `StageResult.data`
  populated with `summary_file`/`model`/`usage`, and overall run success.
- Same config with **no API key set** → assert the run still succeeds and the
  summary stage is SKIPPED (non-fatal), no file written.

The injection seam is `generate_summary(..., client=<stub>)`; the integration
test patches the stage's client construction (e.g. monkeypatch a factory in
`ai.summarizer`) so the real API is never called.

## Files Touched (anticipated)

- **New:** `davinci_monet/ai/__init__.py`, `ai/config.py`, `ai/payload.py`,
  `ai/images.py`, `ai/summarizer.py`.
- **New tests:** `tests/.../test_ai_summary_unit.py`,
  `tests/.../test_ai_summary_integration.py` (locations to match existing test
  layout).
- **Modified:** `davinci_monet/pipeline/stages.py` (add `SummaryStage`, append to
  both pipeline factories), `davinci_monet/config/` schema (register the
  `summary` block), `pyproject.toml` (`[ai]` extra), `environment.yml` (pip:
  `anthropic`).
- **Docs:** README + CLAUDE.md config examples gain a `summary:` block (during
  implementation, not a blocker).

## Open Items for the Plan

- Confirm `StageStatus` has a `SKIPPED` member (`stages.py:22`); pick the
  non-fatal return shape accordingly.
- Confirm the exact config-parsing entry point in `davinci_monet/config/` and how
  unknown top-level blocks are currently handled (so `summary:` validates rather
  than warns).
- Confirm how `plots_generated` is keyed (plot key vs bare path) so
  `cfg.plots` selection and figure captions can map key → title.
- Confirm `Pillow` availability in the `davinci` env.
