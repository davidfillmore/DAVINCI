# Templated AI comparison summaries — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the AI summary stage build its prompt from per-comparison templates — each section declares a format and a soft word budget — selected by the comparand variable, with a built-in YAML library plus inline config override.

**Architecture:** New `davinci_monet/ai/templates/` subpackage mirrors the satellite catalog (YAML `data/*.yaml` + Pydantic `extra="forbid"` schema + `@lru_cache` registry with `difflib` hints). `collect_payload` resolves a template per `(pair, variable)` stats row (override → variable `matches` → `generic_eval` fallback) and attaches it; `render_text` renders the attached template's sections as instruction lines; the hardcoded four-section `SYSTEM_PROMPT` becomes a generic "obey each section's format and budget" instruction. A single model call still assembles the whole brief; soft budgets are prompt-only (no truncation).

**Tech Stack:** Python 3.11, Pydantic v2, PyYAML, pytest. Env: `conda activate davinci`; run tests with `HDF5_USE_FILE_LOCKING=FALSE python -m pytest`.

**Spec:** `docs/superpowers/specs/2026-06-09-ai-comparison-templates-design.md`

---

## File Structure

**Create:**
- `davinci_monet/ai/templates/__init__.py` — public API re-exports.
- `davinci_monet/ai/templates/schema.py` — `SectionFormat`, `TemplateSection`, `SummaryTemplate`.
- `davinci_monet/ai/templates/registry.py` — `TemplateRegistry`, `UnknownTemplateError`, `get_template_registry()`, `resolve_template_for()`.
- `davinci_monet/ai/templates/data/{generic_eval,ozone_eval,aerosol_aod_eval,pm_eval,trace_gas_eval}.yaml` — built-in templates.
- `davinci_monet/tests/unit/ai/templates/__init__.py`
- `davinci_monet/tests/unit/ai/templates/test_schema.py`
- `davinci_monet/tests/unit/ai/templates/test_registry.py`

**Modify:**
- `davinci_monet/config/schema.py` — add `templates` + `template_overrides` to `SummaryConfig` (after line 540, `instructions`).
- `davinci_monet/ai/payload.py` — resolve + attach a template per stats row in `collect_payload`.
- `davinci_monet/ai/summarizer.py` — new `SYSTEM_PROMPT`; per-comparison `render_text`.
- `davinci_monet/ai/__init__.py` — export the template API.
- `pyproject.toml` — add `ai/templates/data/*.yaml` to `package-data`.
- `davinci_monet/tests/unit/ai/test_build_prompt.py` — replace the four-section `SYSTEM_PROMPT` assertion.
- `davinci_monet/tests/integration/test_ai_summary_pipeline.py` — add a two-species end-to-end prompt test.

---

## Task 1: Template schema

**Files:**
- Create: `davinci_monet/ai/templates/schema.py`
- Create: `davinci_monet/tests/unit/ai/templates/__init__.py` (empty)
- Test: `davinci_monet/tests/unit/ai/templates/test_schema.py`

- [ ] **Step 1: Create the empty test package init**

```bash
mkdir -p davinci_monet/ai/templates/data davinci_monet/tests/unit/ai/templates
: > davinci_monet/tests/unit/ai/templates/__init__.py
```

- [ ] **Step 2: Write the failing test**

Create `davinci_monet/tests/unit/ai/templates/test_schema.py`:

```python
"""Unit tests for the AI summary template schema."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from davinci_monet.ai.templates.schema import SummaryTemplate, TemplateSection


def _valid_template() -> dict:
    return {
        "name": "demo",
        "title": "Demo",
        "matches": ["o3"],
        "sections": [
            {"heading": "Bottom line", "format": "headline", "words": 20},
            {"heading": "Detail", "format": "prose", "words": 80, "instruction": "Be quantitative."},
        ],
    }


def test_valid_template_round_trips() -> None:
    tmpl = SummaryTemplate(**_valid_template())
    assert tmpl.name == "demo"
    assert tmpl.matches == ["o3"]
    assert len(tmpl.sections) == 2
    assert tmpl.sections[0].format == "headline"


def test_extra_key_is_rejected() -> None:
    bad = _valid_template()
    bad["unexpected"] = "x"
    with pytest.raises(ValidationError):
        SummaryTemplate(**bad)


def test_zero_or_negative_words_rejected() -> None:
    with pytest.raises(ValidationError):
        TemplateSection(heading="h", format="prose", words=0)
    with pytest.raises(ValidationError):
        TemplateSection(heading="h", format="prose", words=-5)


def test_empty_sections_rejected() -> None:
    bad = _valid_template()
    bad["sections"] = []
    with pytest.raises(ValidationError):
        SummaryTemplate(**bad)


def test_unknown_format_rejected() -> None:
    with pytest.raises(ValidationError):
        TemplateSection(heading="h", format="paragraph", words=10)


def test_format_instruction_includes_budget_and_extra() -> None:
    plain = TemplateSection(heading="h", format="bullets", words=40)
    assert "40 words or fewer" in plain.format_instruction()
    assert "bullet" in plain.format_instruction().lower()

    extra = TemplateSection(heading="h", format="prose", words=30, instruction="Lead with bias.")
    assert "30 words or fewer" in extra.format_instruction()
    assert "Lead with bias." in extra.format_instruction()
```

- [ ] **Step 3: Run test to verify it fails**

Run: `conda activate davinci && python -m pytest davinci_monet/tests/unit/ai/templates/test_schema.py -q`
Expected: FAIL — `ModuleNotFoundError: davinci_monet.ai.templates.schema`.

- [ ] **Step 4: Write the schema**

Create `davinci_monet/ai/templates/schema.py`:

```python
"""Pydantic schema for AI summary comparison templates.

A template is an ordered list of sections; each section declares a fixed
``format`` and a soft word ``budget`` that become an instruction to the model.
Mirrors the satellite-catalog schema style (``extra='forbid'``).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

SectionFormat = Literal["prose", "bullets", "headline", "table", "metric_callout"]

# Each format maps to a fixed instruction phrase; ``{words}`` is the soft budget.
_FORMAT_PHRASES: dict[str, str] = {
    "prose": "{words} words or fewer of prose",
    "bullets": "a short bullet list, {words} words or fewer total",
    "headline": "one line, {words} words or fewer",
    "table": "a compact markdown table, {words} words or fewer total",
    "metric_callout": "a few `key: value` metric lines, {words} words or fewer total",
}


class TemplateSection(BaseModel):
    """One section of a comparison template."""

    model_config = ConfigDict(extra="forbid")

    heading: str
    format: SectionFormat
    words: int = Field(gt=0, description="Soft per-section word budget (prompt-only).")
    instruction: str | None = None

    def format_instruction(self) -> str:
        """Render the model-facing instruction for this section."""
        phrase = _FORMAT_PHRASES[self.format].format(words=self.words)
        if self.instruction:
            return f"{phrase}. {self.instruction}"
        return phrase


class SummaryTemplate(BaseModel):
    """A named comparison template selected by the comparand variable."""

    model_config = ConfigDict(extra="forbid")

    name: str
    title: str | None = None
    description: str | None = None
    matches: list[str] = Field(default_factory=list)
    sections: list[TemplateSection] = Field(min_length=1)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest davinci_monet/tests/unit/ai/templates/test_schema.py -q`
Expected: PASS (6 tests).

- [ ] **Step 6: Format + commit**

```bash
black davinci_monet/ai/templates/schema.py davinci_monet/tests/unit/ai/templates/
isort davinci_monet/ai/templates/schema.py davinci_monet/tests/unit/ai/templates/
git add davinci_monet/ai/templates/schema.py davinci_monet/tests/unit/ai/templates/
git commit -m "feat(ai): add summary template schema (section format + word budget)"
```

---

## Task 2: Built-in template library + registry

**Files:**
- Create: `davinci_monet/ai/templates/data/generic_eval.yaml`, `ozone_eval.yaml`, `aerosol_aod_eval.yaml`, `pm_eval.yaml`, `trace_gas_eval.yaml`
- Create: `davinci_monet/ai/templates/registry.py`
- Create: `davinci_monet/ai/templates/__init__.py`
- Test: `davinci_monet/tests/unit/ai/templates/test_registry.py`

- [ ] **Step 1: Write the built-in template YAML files**

Create `davinci_monet/ai/templates/data/generic_eval.yaml`:

```yaml
name: generic_eval
title: Generic comparison
description: Default brief used when no scenario template matches the variable.
matches: []
sections:
  - heading: What this run is
    format: prose
    words: 60
  - heading: Headline metrics
    format: metric_callout
    words: 50
  - heading: Interpretation
    format: prose
    words: 90
  - heading: Caveats
    format: bullets
    words: 50
```

Create `davinci_monet/ai/templates/data/ozone_eval.yaml`:

```yaml
name: ozone_eval
title: Surface ozone evaluation
description: Model-vs-observation evaluation of ozone (O3).
matches: ["o3", "ozone"]
sections:
  - heading: Bottom line
    format: headline
    words: 20
  - heading: Key metrics
    format: metric_callout
    words: 45
    instruction: Lead with mean bias and normalized mean bias.
  - heading: Bias and timing
    format: prose
    words: 90
    instruction: Note daytime versus nighttime or diurnal-peak behavior if visible.
  - heading: Caveats
    format: bullets
    words: 50
```

Create `davinci_monet/ai/templates/data/aerosol_aod_eval.yaml`:

```yaml
name: aerosol_aod_eval
title: Aerosol optical depth evaluation
description: Evaluation of aerosol optical depth against AERONET or satellite retrievals.
matches: ["aod", "aod_*", "aot", "aot_*", "*aod*"]
sections:
  - heading: Bottom line
    format: headline
    words: 20
  - heading: Key metrics
    format: metric_callout
    words: 45
    instruction: Report correlation and normalized mean bias; AOD is unitless.
  - heading: Spatial and loading patterns
    format: prose
    words: 90
  - heading: Caveats
    format: bullets
    words: 50
    instruction: Note wavelength, retrieval sampling, and cloud screening limits.
```

Create `davinci_monet/ai/templates/data/pm_eval.yaml`:

```yaml
name: pm_eval
title: Particulate matter evaluation
description: Evaluation of particulate matter (PM2.5 / PM10) concentrations.
matches: ["pm25", "pm2.5", "pm10", "pm1"]
sections:
  - heading: Bottom line
    format: headline
    words: 20
  - heading: Key metrics
    format: metric_callout
    words: 45
  - heading: Bias and episodes
    format: prose
    words: 90
    instruction: Call out high-concentration episodes and whether the model captures peaks.
  - heading: Caveats
    format: bullets
    words: 50
```

Create `davinci_monet/ai/templates/data/trace_gas_eval.yaml`:

```yaml
name: trace_gas_eval
title: Trace gas evaluation
description: Evaluation of trace gases (NO2, CO, SO2, HCHO, NH3).
matches: ["no2", "co", "so2", "hcho", "nh3"]
sections:
  - heading: Bottom line
    format: headline
    words: 20
  - heading: Key metrics
    format: metric_callout
    words: 45
  - heading: Sources and patterns
    format: prose
    words: 90
    instruction: Relate disagreements to emission sources or chemistry if visible.
  - heading: Caveats
    format: bullets
    words: 50
```

- [ ] **Step 2: Write the failing registry test**

Create `davinci_monet/tests/unit/ai/templates/test_registry.py`:

```python
"""Unit tests for template loading and resolution."""

from __future__ import annotations

import pytest

from davinci_monet.ai.templates.registry import (
    UnknownTemplateError,
    get_template_registry,
    resolve_template_for,
)


def test_builtin_library_loads() -> None:
    reg = get_template_registry()
    assert {"generic_eval", "ozone_eval", "aerosol_aod_eval", "pm_eval", "trace_gas_eval"} <= set(
        reg.names()
    )


def test_resolve_by_variable_match() -> None:
    assert resolve_template_for("O3").name == "ozone_eval"
    assert resolve_template_for("aod_550nm").name == "aerosol_aod_eval"
    assert resolve_template_for("PM25").name == "pm_eval"
    assert resolve_template_for("NO2").name == "trace_gas_eval"


def test_unmatched_variable_falls_back_to_generic() -> None:
    assert resolve_template_for("relative_humidity").name == "generic_eval"


def test_explicit_override_wins() -> None:
    assert resolve_template_for("O3", override="pm_eval").name == "pm_eval"


def test_unknown_override_raises_with_hint() -> None:
    with pytest.raises(UnknownTemplateError) as exc:
        resolve_template_for("O3", override="ozon_eval")
    assert "ozone_eval" in str(exc.value)  # difflib close-match hint


def test_inline_template_merges_and_extends_index() -> None:
    inline = {
        "ozone_eval": {  # override the built-in by name
            "title": "Custom O3",
            "matches": ["o3"],
            "sections": [{"heading": "Just one", "format": "headline", "words": 10}],
        },
        "custom_scn": {  # brand-new template with its own match
            "matches": ["mytracer"],
            "sections": [{"heading": "X", "format": "prose", "words": 10}],
        },
    }
    assert resolve_template_for("O3", inline=inline).title == "Custom O3"
    assert resolve_template_for("mytracer", inline=inline).name == "custom_scn"
```

- [ ] **Step 3: Run test to verify it fails**

Run: `python -m pytest davinci_monet/tests/unit/ai/templates/test_registry.py -q`
Expected: FAIL — `ModuleNotFoundError: davinci_monet.ai.templates.registry`.

- [ ] **Step 4: Write the registry**

Create `davinci_monet/ai/templates/registry.py`:

```python
"""Loading and resolution of AI summary comparison templates.

Built-in templates are YAML files under ``data/``; resolution picks a template
for a comparand variable by matching the templates' ``matches`` patterns, with
``generic_eval`` as the fallback. Mirrors the satellite-catalog registry.
"""

from __future__ import annotations

import difflib
from fnmatch import fnmatchcase
from functools import lru_cache
from pathlib import Path

import yaml

from davinci_monet.ai.templates.schema import SummaryTemplate

_DATA_DIR = Path(__file__).parent / "data"
FALLBACK_TEMPLATE = "generic_eval"


class UnknownTemplateError(LookupError):
    """Raised when a template name is not in the registry."""


class TemplateRegistry:
    """An immutable set of templates indexed by name, with variable resolution."""

    def __init__(self, templates: list[SummaryTemplate]) -> None:
        self._by_name: dict[str, SummaryTemplate] = {t.name: t for t in templates}

    def names(self) -> list[str]:
        return sorted(self._by_name)

    def get(self, name: str) -> SummaryTemplate:
        if name in self._by_name:
            return self._by_name[name]
        close = difflib.get_close_matches(name, list(self._by_name), n=3)
        hint = f" Did you mean: {', '.join(close)}?" if close else ""
        raise UnknownTemplateError(f"Unknown summary template '{name}'.{hint}")

    def merged_with(self, inline: dict[str, dict] | None) -> "TemplateRegistry":
        """Return a new registry with inline templates merged over the built-ins.

        Inline entries are keyed by name; the key is injected as ``name`` when
        the body omits it. An inline name equal to a built-in replaces it.
        """
        templates = dict(self._by_name)
        for name, spec in (inline or {}).items():
            body = dict(spec)
            body.setdefault("name", name)
            templates[name] = SummaryTemplate(**body)
        return TemplateRegistry(list(templates.values()))

    def resolve_for(self, variable: str, *, override: str | None = None) -> SummaryTemplate:
        """Resolve a template: explicit override, else variable match, else fallback."""
        if override:
            return self.get(override)
        var = (variable or "").lower()
        best: tuple[tuple[int, str], SummaryTemplate] | None = None
        for template in self._by_name.values():
            for pattern in template.matches:
                if fnmatchcase(var, pattern.lower()):
                    score = len(pattern.replace("*", "").replace("?", ""))
                    key = (score, template.name)
                    if best is None or key > best[0]:
                        best = (key, template)
        if best is not None:
            return best[1]
        return self.get(FALLBACK_TEMPLATE)


@lru_cache(maxsize=1)
def get_template_registry() -> TemplateRegistry:
    """Load and cache the built-in template library from ``data/*.yaml``."""
    templates: list[SummaryTemplate] = []
    for path in sorted(_DATA_DIR.glob("*.yaml")):
        raw = yaml.safe_load(path.read_text()) or {}
        templates.append(SummaryTemplate(**raw))
    return TemplateRegistry(templates)


def resolve_template_for(
    variable: str,
    *,
    override: str | None = None,
    inline: dict[str, dict] | None = None,
) -> SummaryTemplate:
    """Resolve a template for ``variable`` against the built-ins plus ``inline``."""
    registry = get_template_registry()
    if inline:
        registry = registry.merged_with(inline)
    return registry.resolve_for(variable, override=override)
```

- [ ] **Step 5: Write the subpackage `__init__`**

Create `davinci_monet/ai/templates/__init__.py`:

```python
"""AI summary comparison templates (built-in YAML library + resolution)."""

from __future__ import annotations

from davinci_monet.ai.templates.registry import (
    TemplateRegistry,
    UnknownTemplateError,
    get_template_registry,
    resolve_template_for,
)
from davinci_monet.ai.templates.schema import SummaryTemplate, TemplateSection

__all__ = [
    "SummaryTemplate",
    "TemplateSection",
    "TemplateRegistry",
    "UnknownTemplateError",
    "get_template_registry",
    "resolve_template_for",
]
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `python -m pytest davinci_monet/tests/unit/ai/templates/ -q`
Expected: PASS (schema + registry tests).

- [ ] **Step 7: Format + commit**

```bash
black davinci_monet/ai/templates/ davinci_monet/tests/unit/ai/templates/
isort davinci_monet/ai/templates/ davinci_monet/tests/unit/ai/templates/
git add davinci_monet/ai/templates/
git commit -m "feat(ai): built-in summary template library + registry resolution"
```

---

## Task 3: `SummaryConfig` template fields

**Files:**
- Modify: `davinci_monet/config/schema.py` (`SummaryConfig`, after `instructions` at line ~540)
- Test: `davinci_monet/tests/unit/config/test_summary_template_config.py` (create)

- [ ] **Step 1: Write the failing test**

Create `davinci_monet/tests/unit/config/test_summary_template_config.py`:

```python
"""SummaryConfig gains optional template fields."""

from __future__ import annotations

from davinci_monet.config.schema import SummaryConfig


def test_template_fields_default_none() -> None:
    cfg = SummaryConfig()
    assert cfg.templates is None
    assert cfg.template_overrides is None


def test_inline_templates_and_overrides_parse() -> None:
    cfg = SummaryConfig.model_validate(
        {
            "enabled": True,
            "templates": {
                "my_o3": {
                    "matches": ["o3"],
                    "sections": [{"heading": "h", "format": "headline", "words": 12}],
                }
            },
            "template_overrides": {"pair_a": "my_o3"},
        }
    )
    assert cfg.templates is not None and "my_o3" in cfg.templates
    assert cfg.template_overrides == {"pair_a": "my_o3"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest davinci_monet/tests/unit/config/test_summary_template_config.py -q`
Expected: FAIL — `AttributeError: 'SummaryConfig' object has no attribute 'templates'`.

- [ ] **Step 3: Add the fields**

In `davinci_monet/config/schema.py`, in `class SummaryConfig`, immediately after the line `instructions: str | None = None` (line ~540), add:

```python
    templates: dict[str, dict] | None = None
    template_overrides: dict[str, str] | None = None
```

(They are left as raw dicts here — validated into `SummaryTemplate` by the registry at resolution time — to keep the config schema free of an `ai.templates` import.)

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest davinci_monet/tests/unit/config/test_summary_template_config.py -q`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
black davinci_monet/config/schema.py davinci_monet/tests/unit/config/test_summary_template_config.py
git add davinci_monet/config/schema.py davinci_monet/tests/unit/config/test_summary_template_config.py
git commit -m "feat(config): SummaryConfig templates + template_overrides fields"
```

---

## Task 4: Resolve + attach a template per stats row in `collect_payload`

**Files:**
- Modify: `davinci_monet/ai/payload.py` (`collect_payload`, after the stats-rows loop ~line 75, before `all_plots`)
- Test: `davinci_monet/tests/unit/ai/test_collect_payload_templates.py` (create)

- [ ] **Step 1: Write the failing test**

Create `davinci_monet/tests/unit/ai/test_collect_payload_templates.py`:

```python
"""collect_payload attaches a resolved template to each stats row."""

from __future__ import annotations

from types import SimpleNamespace

from davinci_monet.ai.payload import collect_payload
from davinci_monet.config.schema import SummaryConfig


def _context(stats: dict) -> SimpleNamespace:
    return SimpleNamespace(
        config={
            "analysis": {"start_time": "2024-02-01", "end_time": "2024-02-03"},
            "sources": {"cam": {"type": "cesm_fv", "role": "model"}},
            "pairs": {"p_o3": {}, "p_pm": {}},
        },
        results={"statistics": SimpleNamespace(data=stats)},
    )


def test_each_row_gets_template_by_variable() -> None:
    ctx = _context(
        {
            "p_o3": {"O3": {"N": 10, "MB": 1.0}},
            "p_pm": {"PM25": {"N": 12, "MB": -2.0}},
        }
    )
    payload = collect_payload(ctx, SummaryConfig())
    by_var = {row["variable"]: row["template"].name for row in payload.stats_rows}
    assert by_var["O3"] == "ozone_eval"
    assert by_var["PM25"] == "pm_eval"


def test_override_forces_template_for_pair() -> None:
    ctx = _context({"p_o3": {"O3": {"N": 10}}})
    cfg = SummaryConfig.model_validate({"template_overrides": {"p_o3": "trace_gas_eval"}})
    payload = collect_payload(ctx, cfg)
    assert payload.stats_rows[0]["template"].name == "trace_gas_eval"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest davinci_monet/tests/unit/ai/test_collect_payload_templates.py -q`
Expected: FAIL — `KeyError: 'template'`.

- [ ] **Step 3: Attach the template in `collect_payload`**

In `davinci_monet/ai/payload.py`, after the `stats_rows` loop completes (after line 75, before the `all_plots` block), add:

```python
    from davinci_monet.ai.templates import resolve_template_for

    overrides = cfg.template_overrides or {}
    for row in stats_rows:
        row["template"] = resolve_template_for(
            row["variable"],
            override=overrides.get(row["pair"]),
            inline=cfg.templates,
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest davinci_monet/tests/unit/ai/test_collect_payload_templates.py -q`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
black davinci_monet/ai/payload.py davinci_monet/tests/unit/ai/test_collect_payload_templates.py
isort davinci_monet/ai/payload.py
git add davinci_monet/ai/payload.py davinci_monet/tests/unit/ai/test_collect_payload_templates.py
git commit -m "feat(ai): resolve and attach a comparison template per stats row"
```

---

## Task 5: Template-driven `SYSTEM_PROMPT` and `render_text`

**Files:**
- Modify: `davinci_monet/ai/summarizer.py` (`SYSTEM_PROMPT` lines 18-34; `render_text` lines 96-120)
- Modify: `davinci_monet/tests/unit/ai/test_build_prompt.py` (replace `test_system_prompt_requests_four_sections`)
- Test: `davinci_monet/tests/unit/ai/test_render_text_templates.py` (create)

- [ ] **Step 1: Write the failing test**

Create `davinci_monet/tests/unit/ai/test_render_text_templates.py`:

```python
"""render_text emits per-comparison template section instructions."""

from __future__ import annotations

from davinci_monet.ai.payload import ImageRef, SummaryPayload
from davinci_monet.ai.summarizer import render_text
from davinci_monet.ai.templates import get_template_registry


def _payload_with_template() -> SummaryPayload:
    ozone = get_template_registry().get("ozone_eval")
    return SummaryPayload(
        period={"start": "2024-02-01", "end": "2024-02-03"},
        sources_summary=["cam (cesm_fv)"],
        pairs_summary=["cam_vs_airnow_o3"],
        stats_rows=[
            {
                "pair": "cam_vs_airnow_o3",
                "variable": "O3",
                "metrics": {"N": 120, "MB": -2.5},
                "template": ozone,
            }
        ],
        images=[ImageRef(caption="01_o3", path="/x/01_o3.png")],
        instructions=None,
    )


def test_render_emits_template_headings_and_budget() -> None:
    text = render_text(_payload_with_template())
    assert "## cam_vs_airnow_o3 — O3" in text
    assert "### Bottom line" in text  # an ozone_eval section heading
    assert "20 words or fewer" in text  # its budget phrase
    assert "N=120" in text  # the comparison's stats still present


def test_render_falls_back_to_generic_without_attached_template() -> None:
    payload = _payload_with_template()
    del payload.stats_rows[0]["template"]  # standalone use (no resolution)
    text = render_text(payload)
    assert "### What this run is" in text  # generic_eval heading
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest davinci_monet/tests/unit/ai/test_render_text_templates.py -q`
Expected: FAIL — current `render_text` emits a flat `## Statistics` block, not per-comparison headings.

- [ ] **Step 3: Rewrite `SYSTEM_PROMPT`**

In `davinci_monet/ai/summarizer.py`, replace the `SYSTEM_PROMPT` assignment (lines 18-34) with:

```python
SYSTEM_PROMPT = """You are a climate and atmospheric-composition data-comparison analyst.
You are given the configuration, summary statistics, and figures from a single DAVINCI
analysis run. The run contains one or more comparisons; each lists the exact sections to
write, with a required format and a word budget.

For EACH comparison below, output a markdown block headed by the comparison name that
contains EXACTLY its listed sections, in order. Obey each section's stated format and stay
within its word budget. Be specific and quantitative. Do not invent numbers that are not
present in the provided statistics or visible in the figures."""
```

- [ ] **Step 4: Rewrite `render_text`**

In `davinci_monet/ai/summarizer.py`, replace the body of `render_text` (lines 96-120) with:

```python
def render_text(payload: SummaryPayload) -> str:
    """Render the textual portion of the user message from the payload.

    Emits the run header, then one templated block per ``(pair, variable)``
    comparison: the comparison heading, its template's section instructions
    (heading + format + word budget), and that comparison's statistics. Each row
    carries a resolved ``template``; rows without one fall back to ``generic_eval``.
    """
    from davinci_monet.ai.templates import get_template_registry

    lines: list[str] = ["# Analysis run"]
    period = payload.period
    lines.append(f"Period: {period.get('start')} to {period.get('end')}")
    if payload.sources_summary:
        lines.append("Sources: " + ", ".join(payload.sources_summary))
    if payload.pairs_summary:
        lines.append("Pairs: " + ", ".join(payload.pairs_summary))

    if payload.stats_rows:
        for row in payload.stats_rows:
            template = row.get("template") or get_template_registry().get("generic_eval")
            lines.append("")
            lines.append(f"## {row['pair']} — {row['variable']}")
            lines.append("Write these sections, obeying each format and word budget:")
            for section in template.sections:
                lines.append(f"### {section.heading} — {section.format_instruction()}")
            metric_str = ", ".join(f"{k}={_fmt(v)}" for k, v in row["metrics"].items())
            lines.append(f"Statistics: {metric_str}")
    else:
        lines.append("")
        lines.append("## Statistics")
        lines.append("(no statistics available)")

    if payload.instructions:
        lines.append("")
        lines.append("## Additional instructions")
        lines.append(payload.instructions)

    return "\n".join(lines)
```

- [ ] **Step 5: Update the obsolete `SYSTEM_PROMPT` assertion**

In `davinci_monet/tests/unit/ai/test_build_prompt.py`, replace `test_system_prompt_requests_four_sections` (lines 40-47) with:

```python
def test_system_prompt_is_template_driven() -> None:
    lowered = SYSTEM_PROMPT.lower()
    assert "section" in lowered
    assert "format" in lowered
    assert "word budget" in lowered
    assert "comparison" in lowered
```

- [ ] **Step 6: Run the affected tests**

Run: `python -m pytest davinci_monet/tests/unit/ai/test_render_text_templates.py davinci_monet/tests/unit/ai/test_build_prompt.py -q`
Expected: PASS (render-template tests + all of test_build_prompt, including the unchanged period/sources/stats and build_prompt-structure tests).

- [ ] **Step 7: Commit**

```bash
black davinci_monet/ai/summarizer.py davinci_monet/tests/unit/ai/test_render_text_templates.py davinci_monet/tests/unit/ai/test_build_prompt.py
isort davinci_monet/ai/summarizer.py davinci_monet/tests/unit/ai/test_render_text_templates.py
git add davinci_monet/ai/summarizer.py davinci_monet/tests/unit/ai/test_render_text_templates.py davinci_monet/tests/unit/ai/test_build_prompt.py
git commit -m "feat(ai): template-driven summary prompt (per-comparison sections)"
```

---

## Task 6: Public API exports + package data

**Files:**
- Modify: `davinci_monet/ai/__init__.py`
- Modify: `pyproject.toml` (line ~71, `[tool.setuptools.package-data]`)
- Test: `davinci_monet/tests/unit/ai/templates/test_packaging.py` (create)

- [ ] **Step 1: Write the failing test**

Create `davinci_monet/tests/unit/ai/templates/test_packaging.py`:

```python
"""Template API is exported and the YAML data is declared as package data."""

from __future__ import annotations

import tomllib
from pathlib import Path


def test_template_api_exported_from_ai() -> None:
    import davinci_monet.ai as ai

    assert hasattr(ai, "get_template_registry")
    assert hasattr(ai, "resolve_template_for")
    assert hasattr(ai, "SummaryTemplate")


def test_template_yaml_declared_as_package_data() -> None:
    repo_root = Path(__file__).resolve().parents[4]
    data = tomllib.loads((repo_root / "pyproject.toml").read_text())
    pkg_data = data["tool"]["setuptools"]["package-data"]["davinci_monet"]
    assert any("ai/templates/data/*.yaml" in entry for entry in pkg_data)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest davinci_monet/tests/unit/ai/templates/test_packaging.py -q`
Expected: FAIL — `get_template_registry` not on `davinci_monet.ai`; package-data assertion fails.

- [ ] **Step 3: Export from `ai/__init__.py`**

In `davinci_monet/ai/__init__.py`, add after the `from davinci_monet.ai.summarizer import (...)` block:

```python
from davinci_monet.ai.templates import (
    SummaryTemplate,
    TemplateSection,
    get_template_registry,
    resolve_template_for,
)
```

and add these names to `__all__`:

```python
    "SummaryTemplate",
    "TemplateSection",
    "get_template_registry",
    "resolve_template_for",
```

- [ ] **Step 4: Declare the package data**

In `pyproject.toml`, change the `package-data` line:

```toml
davinci_monet = ["py.typed", "observations/satellite/catalog/data/*.yaml"]
```

to:

```toml
davinci_monet = ["py.typed", "observations/satellite/catalog/data/*.yaml", "ai/templates/data/*.yaml"]
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest davinci_monet/tests/unit/ai/templates/test_packaging.py -q`
Expected: PASS (2 tests).

- [ ] **Step 6: Commit**

```bash
black davinci_monet/ai/__init__.py davinci_monet/tests/unit/ai/templates/test_packaging.py
isort davinci_monet/ai/__init__.py
git add davinci_monet/ai/__init__.py pyproject.toml davinci_monet/tests/unit/ai/templates/test_packaging.py
git commit -m "feat(ai): export template API; ship template YAML as package data"
```

---

## Task 7: End-to-end integration test (two species → two templates in the prompt)

**Files:**
- Modify: `davinci_monet/tests/integration/test_ai_summary_pipeline.py` (add a config builder + test; reuse `_StubClient`)

- [ ] **Step 1: Add a two-variable config builder and test**

Append to `davinci_monet/tests/integration/test_ai_summary_pipeline.py`:

```python
def _build_two_species_config(tmp_path: Path) -> dict:
    domain = Domain(lon_min=-105.0, lon_max=-95.0, lat_min=35.0, lat_max=45.0, n_lon=12, n_lat=12)
    time_cfg = TimeConfig(start="2024-01-15 00:00", end="2024-01-17 00:00", freq="1h")

    model_ds = create_model_dataset(
        variables=["O3", "PM25"], domain=domain, time_config=time_cfg, seed=7
    )
    scenario = PerfectMatchScenario(
        variables=["O3", "PM25"],
        domain=domain,
        time_config=time_cfg,
        geometry=DataGeometry.POINT,
        n_obs=10,
        noise_level=0.0,
        seed=7,
    )
    obs_ds = sample_obs_from(model_ds, "point", scenario=scenario)

    model_path = tmp_path / "model2.nc"
    obs_path = tmp_path / "obs2.nc"
    model_ds.to_netcdf(model_path)
    obs_ds.to_netcdf(obs_path)

    return {
        "analysis": {
            "start_time": "2024-01-15 00:00",
            "end_time": "2024-01-17 00:00",
            "output_dir": str(tmp_path / "output"),
            "log_dir": str(tmp_path / "logs"),
        },
        "sources": {
            "synthetic": {
                "type": "generic",
                "role": "model",
                "files": str(model_path),
                "radius_of_influence": 50000,
                "mapping": {"surface": {"O3": "O3", "PM25": "PM25"}},
                "variables": {"O3": {"units": "ppb"}, "PM25": {"units": "ug/m3"}},
            },
            "surface": {
                "type": "pt_sfc",
                "role": "obs",
                "filename": str(obs_path),
                "variables": {"O3": {"units": "ppb"}, "PM25": {"units": "ug/m3"}},
            },
        },
        "pairs": {
            "o3_pair": {
                "sources": ["synthetic", "surface"],
                "reference": "surface",
                "variables": {"synthetic": "O3", "surface": "O3"},
            },
            "pm_pair": {
                "sources": ["synthetic", "surface"],
                "reference": "surface",
                "variables": {"synthetic": "PM25", "surface": "PM25"},
            },
        },
        "stats": {"metrics": ["N", "MB", "RMSE", "R"]},
    }


def test_two_species_prompt_carries_distinct_templates(monkeypatch, tmp_path: Path) -> None:
    from davinci_monet.pipeline.runner import PipelineRunner

    stub = _StubClient()
    monkeypatch.setattr(summarizer_mod, "_build_client", lambda cfg: stub)

    config = _build_two_species_config(tmp_path)
    config["summary"] = {"enabled": True, "model": "claude-haiku-4-5"}

    runner = PipelineRunner(show_progress=False)
    result = runner.run_from_config(config)

    assert result.success, "pipeline run failed"
    assert stub.calls, "client was not called"
    user_text = stub.calls[0]["messages"][0]["content"][0]["text"]
    # O3 resolved ozone_eval, PM25 resolved pm_eval — both per-comparison blocks present.
    assert "## o3_pair — O3" in user_text
    assert "## pm_pair — PM25" in user_text
    assert "Bias and timing" in user_text  # an ozone_eval-only section heading
    assert "Bias and episodes" in user_text  # a pm_eval-only section heading
```

- [ ] **Step 2: Run the integration test**

Run: `HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/integration/test_ai_summary_pipeline.py -q`
Expected: PASS (existing 3 tests + the new one). The pre-existing tests assert on the stub's fixed output and image presence and are unaffected by the prompt change.

- [ ] **Step 3: Commit**

```bash
black davinci_monet/tests/integration/test_ai_summary_pipeline.py
git add davinci_monet/tests/integration/test_ai_summary_pipeline.py
git commit -m "test(ai): integration — per-species templates reach the prompt end-to-end"
```

---

## Task 8: Full gate sweep

**Files:** none (verification only).

- [ ] **Step 1: Run the full suite + static gates**

```bash
conda activate davinci
HDF5_USE_FILE_LOCKING=FALSE python -m pytest -q
mypy davinci_monet
black --check davinci_monet
isort --check-only davinci_monet
```
Expected: all green; the suite count increases by the new tests; **zero warnings**.

- [ ] **Step 2: Fix any gate failures, then final commit (if any fixes were needed)**

```bash
git add -A && git commit -m "chore(ai): satisfy gates for templated summaries"
```

---

## Self-Review

**Spec coverage:**
- Built-in YAML library + Pydantic schema + registry (Approach A) → Tasks 1–2. ✓
- Inline override + per-pair override config → Task 3 (`templates`, `template_overrides`). ✓
- Per-`(pair, variable)` scope; selection by variable with generic fallback → Task 4 (`collect_payload`) + Task 2 (`resolve_for`). ✓
- Section format vocabulary + soft word budget, no enforcement pass → Task 1 (`format_instruction`); no truncation anywhere. ✓
- Single-call template-driven prompt replacing the four-section `SYSTEM_PROMPT`; backward-compatible `generic_eval` → Task 5. ✓
- Focused starter set (generic + ozone + aerosol_aod + pm + trace_gas) → Task 2 data files. ✓
- Package data + exports → Task 6. Tests through the pipeline → Task 7. ✓

**Placeholder scan:** every code/test/command step contains literal content; no TBD/TODO/"similar to". ✓

**Type/name consistency:** `SummaryTemplate`/`TemplateSection`/`SectionFormat` (Task 1) used identically in Tasks 2/4/5/6; `resolve_template_for(variable, *, override, inline)` signature (Task 2) matches its call in Task 4; `format_instruction()` defined in Task 1 and asserted in Tasks 1/5; `get_template_registry().get("generic_eval")` fallback used in Tasks 4-test and 5. ✓
