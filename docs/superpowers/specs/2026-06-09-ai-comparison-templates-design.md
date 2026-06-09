# Templated AI comparison summaries — Design

**Date:** 2026-06-09
**Status:** Design approved; pending spec review → implementation plan
**Scope:** Extends the optional AI summary stage (`davinci_monet/ai/`, `summary:` config). No changes to pairing, statistics, or plotting.

## Problem

The AI summary stage hardcodes one prompt (`SYSTEM_PROMPT` in `ai/summarizer.py`) with a
fixed four-section brief ("What this run is / Headline metrics / Interpretation / Caveats").
There is no way to vary the brief's structure by the *kind* of comparison being made, and no
mechanism to enforce brevity per section. We want the model to craft its reply from a
**template**: an ordered set of sections, each with a **format** and a **word budget**, chosen
to fit the comparison.

## Decisions (from brainstorming)

| # | Decision | Choice |
|---|----------|--------|
| 1 | Where templates live | Built-in YAML **library + inline override** in config |
| 2 | What one template governs | **Per pair/comparison** — the brief is a concatenation of per-pair templated section blocks |
| 3 | Template selection | **By variable / science scenario** — a variable→template mapping with a generic fallback |
| 4 | Word-limit enforcement | **Soft (prompt budget only)** — the template states each section's budget; no post-generation truncation or validation pass |
| 5 | Section "format" | **Small fixed vocabulary**: `prose`, `bullets`, `headline`, `table`, `metric_callout` |
| 6 | Initial built-in set | **Focused starter set**: `generic_eval` (fallback) + `ozone_eval` + `aerosol_aod_eval` + `pm_eval` + `trace_gas_eval` |
| 7 | Architecture | **Approach A** — declarative YAML library + Pydantic schema + registry (mirrors the satellite catalog), single model call assembling per-pair sections |

## Architecture

Mirror the established `observations/satellite/catalog/` pattern (YAML data + Pydantic
`extra="forbid"` schema + `@lru_cache` registry with `difflib` close-match hints).

```
davinci_monet/ai/templates/
├── __init__.py        # public API: get_template_registry(), resolve_template_for(), schema re-exports
├── schema.py          # Pydantic: SummaryTemplate, TemplateSection, SectionFormat
├── registry.py        # @lru_cache loader of data/*.yaml + inline merge + resolution; UnknownTemplateError
└── data/
    ├── generic_eval.yaml        # fallback; reproduces today's four sections (per pair)
    ├── ozone_eval.yaml
    ├── aerosol_aod_eval.yaml
    ├── pm_eval.yaml
    └── trace_gas_eval.yaml
```

`pyproject.toml` `[tool.setuptools.package-data]` gains `ai/templates/data/*.yaml`
(same form as the existing `observations/satellite/catalog/data/*.yaml` entry).

## Template schema

YAML, one template per file, validated by Pydantic with `model_config = ConfigDict(extra="forbid")`.

```yaml
name: ozone_eval
title: "Surface Ozone Evaluation"     # optional, human-facing
description: "..."                      # optional
matches: ["o3", "ozone"]               # case-insensitive fnmatch patterns on the comparand variable
sections:
  - heading: "Bottom line"
    format: headline          # prose | bullets | headline | table | metric_callout
    words: 20                 # soft per-section word budget (> 0)
  - heading: "Key metrics"
    format: metric_callout
    words: 40
  - heading: "Agreement & patterns"
    format: prose
    words: 80
  - heading: "Caveats"
    format: bullets
    words: 50
```

Models:

- `SectionFormat = Literal["prose", "bullets", "headline", "table", "metric_callout"]`.
- `TemplateSection`: `heading: str`, `format: SectionFormat`, `words: int` (validated `> 0`),
  `instruction: str | None = None` (optional extra nudge appended to the format instruction).
- `SummaryTemplate`: `name: str`, `title: str | None = None`, `description: str | None = None`,
  `matches: list[str] = []`, `sections: list[TemplateSection]` (validated non-empty).

The variable→template mapping is **co-located inside each template** via `matches`; the registry
builds the lookup index from all templates' `matches`. `generic_eval` declares no `matches` and is
the hard-coded fallback (never matched by pattern, always available by name).

### Format → instruction phrases (applied at render time)

| `format` | Instruction emitted to the model |
|----------|----------------------------------|
| `prose` | "≤{words} words of prose." |
| `bullets` | "A short bullet list, ≤{words} words total." |
| `headline` | "One line, ≤{words} words." |
| `table` | "A compact markdown table, ≤{words} words total." |
| `metric_callout` | "A few `key: value` metric lines, ≤{words} words total." |

A section's optional `instruction` is appended after the format phrase.

## Resolution

`resolve_template_for(variable: str, *, override: str | None = None) -> SummaryTemplate`,
precedence:

1. **Explicit override** — if `override` is a template name, return `get_template(override)`;
   raise `UnknownTemplateError` (with a `difflib.get_close_matches` hint, like the catalog) if unknown.
2. **Variable match** — lowercase the comparand variable name and test it against every template's
   `matches` patterns with `fnmatch`; the most specific match wins (longest non-wildcard pattern;
   ties resolved by template name for determinism).
3. **Fallback** — `generic_eval`.

Built-in templates load once via `@lru_cache`. User-supplied inline templates (see config) are
merged over the built-ins by `name` and their `matches` extend the index.

## Config surface (`SummaryConfig`)

All additions are optional and backward-compatible:

- `templates: dict[str, dict] | None = None` — inline template definitions/overrides. Each value is
  validated by `SummaryTemplate` and merged over the built-in library by `name`.
- `template_overrides: dict[str, str] | None = None` — explicit `{pair_name: template_name}` map that
  forces a pair's template, bypassing variable matching (precedence step 1).

Existing fields are unchanged. `instructions` continues to be appended as global guidance to the
user message.

## Prompt assembly (`ai/summarizer.py`)

The hardcoded four-section `SYSTEM_PROMPT` is replaced by a generic instruction:

> "You are a climate and atmospheric-composition data-comparison analyst. For EACH comparison
> below, write a markdown block headed by the comparison name containing EXACTLY the sections listed
> for it, in order. Obey each section's stated format and word budget. Be specific and quantitative;
> never invent numbers not present in the statistics or visible in the figures."

`render_text(payload)` stays pure (no IO) and is extended to:

1. Emit the run header it already produces (period, sources, pairs).
2. Treat each `payload.stats_rows` entry — a `(pair, variable)` comparison — as the unit. A pair with
   multiple variables yields multiple blocks (each resolved by its own variable). For each
   `(pair, variable)`:
   - resolve the template (`template_overrides[pair]` → variable match → `generic_eval`),
   - emit `## {pair} — {variable}`,
   - emit each section as an instruction line: `### {heading} — {format instruction}`,
   - emit that comparison's statistics beneath.
3. Append `instructions` (global) as today.

`build_prompt` uses the new system prompt plus this `render_text`; figure attachment is unchanged.
A single model call still produces the whole brief.

### Backward compatibility

`generic_eval` is authored to reproduce today's four sections, so a run with no configured templates
and no mapped variables reads almost identically — now emitted per `(pair, variable)` comparison
rather than once for the run.
The existing `summary` unit/integration tests are updated to the per-pair structure using the existing
injectable fake-client seam (`generate_summary(..., client=...)`).

## Testing (through the pipeline, per project rules)

**Unit:**
- Schema: valid template round-trips; `extra="forbid"` rejects unknown keys; `words <= 0` rejected;
  empty `sections` rejected.
- Registry: variable match (e.g. `o3` → `ozone_eval`, `aod_550nm` → `aerosol_aod_eval`); explicit
  override; fallback to `generic_eval`; `UnknownTemplateError` carries a close-match hint; inline
  templates merge over and extend built-ins.
- Render: each `SectionFormat` produces its instruction phrase with the budget; `render_text`
  groups by pair and carries each pair's headings + budgets; pure (no IO).

**Integration** (`PipelineRunner.run_from_config` with an injected fake client, matching
`tests/integration/test_ai_summary_pipeline.py`):
- A run with two pairs of different species (e.g. O3 and AOD) completes, writes `AI_summary.md`, and
  the captured prompt contains both distinct templates' sections.

## Out of scope (YAGNI)

- Hard word-limit enforcement / truncation / overflow validation (soft budgets only).
- Per-pair separate API calls (single call assembles all sections).
- Run-level wrapper template (the run header is a fixed preamble, not itself a template).
- Geometry- or scenario-tag-based selection beyond the variable→template `matches` index.
- A meteorology template and per-species templates beyond the starter set (add later by dropping in
  a YAML file).

## Build sequence (for the implementation plan)

1. `schema.py` (+ unit tests).
2. `registry.py` + `data/*.yaml` starter templates (+ resolution unit tests).
3. `SummaryConfig` fields + validation (+ config unit tests).
4. Rewire `summarizer.py` prompt assembly + `render_text` (+ render unit tests).
5. Update existing AI-summary unit/integration tests to the per-pair structure; add the
   two-species integration test.
6. `pyproject.toml` package-data; verify all gates (pytest/mypy/black/isort).
