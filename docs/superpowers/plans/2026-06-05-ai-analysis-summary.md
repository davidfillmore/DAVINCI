# AI Analysis Summary Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in final pipeline stage that generates a single-prompt, vision-enabled Claude summary (`AI_summary.md`) for each analysis run.

**Architecture:** A pure engine subpackage `davinci_monet/ai/` (config-driven prompt building, image encoding, Anthropic call with an injectable client) plus a thin `SummaryStage` in `pipeline/stages.py` that wires `PipelineContext` → engine → file/terminal output. Always non-fatal: any failure logs a warning and the stage is `SKIPPED`, never `FAILED`.

**Tech Stack:** Python 3.11+, Pydantic v2 (config), `anthropic` SDK (lazy-imported, optional `[ai]` extra), Pillow (image downscaling), pytest.

**Spec:** `docs/superpowers/specs/2026-06-05-ai-analysis-summary-design.md`

**Environment:** Run all tests in the `davinci` conda env:
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate davinci
HDF5_USE_FILE_LOCKING=FALSE python -m pytest <path> -v
```

**Layering note (deviation from spec):** `SummaryConfig` lives in `davinci_monet/config/schema.py` (next to the other config models, so `MonetConfig` can reference it without a circular import) and is re-exported from `davinci_monet.ai`. The `ai/` package imports `SummaryConfig` from `config.schema` (one-way `ai → config` dependency).

---

## Task 1: Add the `[ai]` optional dependency

**Files:**
- Modify: `pyproject.toml`
- Modify: `environment.yml`

- [ ] **Step 1: Add the `[ai]` extra to pyproject.toml**

Find the `[project.optional-dependencies]` table in `pyproject.toml` (it already has a `dev` extra). Add an `ai` entry:

```toml
ai = [
    "anthropic>=0.40",
    "pillow>=10.0",
]
```

If no `[project.optional-dependencies]` table exists, add one above `[project.urls]`:

```toml
[project.optional-dependencies]
ai = [
    "anthropic>=0.40",
    "pillow>=10.0",
]
```

- [ ] **Step 2: Add anthropic to environment.yml pip section**

In `environment.yml`, find the `pip:` list under dependencies and add:

```yaml
    - anthropic>=0.40
```

(Pillow is already pulled in by matplotlib in this env; anthropic is the new one.)

- [ ] **Step 3: Install the extra into the davinci env**

Run:
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate davinci
pip install -e ".[ai]"
```
Expected: anthropic installs successfully.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml environment.yml
git commit -m "build: add optional [ai] extra (anthropic, pillow)"
```

---

## Task 2: `SummaryConfig` schema + wire into `MonetConfig`

**Files:**
- Modify: `davinci_monet/config/schema.py` (add `SummaryConfig` class; add `summary` field to `MonetConfig` at line ~713)
- Test: `davinci_monet/tests/unit/config/test_summary_config.py`

- [ ] **Step 1: Write the failing test**

Create `davinci_monet/tests/unit/config/test_summary_config.py`:

```python
"""Unit tests for SummaryConfig and the MonetConfig.summary field."""

from __future__ import annotations

from davinci_monet.config.schema import MonetConfig, SummaryConfig


def test_summary_config_defaults() -> None:
    cfg = SummaryConfig()
    assert cfg.enabled is False
    assert cfg.model == "claude-haiku-4-5"
    assert cfg.max_tokens == 2000
    assert cfg.api_key_env == "ANTHROPIC_API_KEY"
    assert cfg.plots is None
    assert cfg.max_images == 8
    assert cfg.output_filename == "AI_summary.md"
    assert cfg.instructions is None


def test_summary_config_overrides() -> None:
    cfg = SummaryConfig.model_validate(
        {
            "enabled": True,
            "model": "claude-sonnet-4-6",
            "plots": ["pm25_spatial_bias", "o3_scatter"],
            "max_images": 3,
            "instructions": "Focus on coastal sites.",
        }
    )
    assert cfg.enabled is True
    assert cfg.model == "claude-sonnet-4-6"
    assert cfg.plots == ["pm25_spatial_bias", "o3_scatter"]
    assert cfg.max_images == 3
    assert cfg.instructions == "Focus on coastal sites."


def test_monetconfig_summary_field_defaults_none() -> None:
    cfg = MonetConfig.model_validate(
        {"analysis": {"start_time": "2024-01-01", "end_time": "2024-01-02"}}
    )
    assert cfg.summary is None


def test_monetconfig_parses_summary_block() -> None:
    cfg = MonetConfig.model_validate(
        {
            "analysis": {"start_time": "2024-01-01", "end_time": "2024-01-02"},
            "summary": {"enabled": True, "model": "claude-haiku-4-5"},
        }
    )
    assert cfg.summary is not None
    assert cfg.summary.enabled is True
    assert cfg.summary.model == "claude-haiku-4-5"
    # model_dump round-trips to a plain dict for the pipeline
    dumped = cfg.model_dump()
    assert dumped["summary"]["enabled"] is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/unit/config/test_summary_config.py -v`
Expected: FAIL with `ImportError: cannot import name 'SummaryConfig'`.

- [ ] **Step 3: Add `SummaryConfig` and the `summary` field**

In `davinci_monet/config/schema.py`, add this class just above the `MonetConfig` class definition (around line 677):

```python
class SummaryConfig(FlexibleModel):
    """Configuration for the optional AI analysis summary stage.

    When ``enabled`` is true, a final pipeline stage sends the run's
    statistics, config metadata, and selected plot images to the Claude API
    and writes a markdown brief into the analysis output directory.
    """

    enabled: bool = False
    model: str = "claude-haiku-4-5"
    max_tokens: int = 2000
    api_key_env: str = "ANTHROPIC_API_KEY"
    plots: list[str] | None = None
    max_images: int = 8
    output_filename: str = "AI_summary.md"
    instructions: str | None = None
```

Then add the field to `MonetConfig` (immediately after the `stats` field, ~line 713):

```python
    summary: SummaryConfig | None = None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/unit/config/test_summary_config.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add davinci_monet/config/schema.py davinci_monet/tests/unit/config/test_summary_config.py
git commit -m "feat(config): add SummaryConfig and MonetConfig.summary"
```

---

## Task 3: `ai/images.py` — image encoding with downscale

**Files:**
- Create: `davinci_monet/ai/__init__.py` (empty for now)
- Create: `davinci_monet/ai/images.py`
- Test: `davinci_monet/tests/unit/ai/test_images.py`

- [ ] **Step 1: Write the failing test**

Create `davinci_monet/tests/unit/ai/test_images.py`:

```python
"""Unit tests for ai.images.encode_image."""

from __future__ import annotations

import base64

from pathlib import Path

import numpy as np
from PIL import Image

from davinci_monet.ai.images import EncodedImage, encode_image


def _write_png(path: Path, width: int, height: int) -> None:
    arr = (np.random.default_rng(0).random((height, width, 3)) * 255).astype("uint8")
    Image.fromarray(arr).save(path)


def test_encode_image_returns_base64_png(tmp_path: Path) -> None:
    p = tmp_path / "small.png"
    _write_png(p, 100, 80)
    enc = encode_image(p)
    assert isinstance(enc, EncodedImage)
    assert enc.media_type == "image/png"
    # data must be valid base64
    decoded = base64.b64decode(enc.data)
    assert decoded[:8] == b"\x89PNG\r\n\x1a\n"


def test_encode_image_downscales_large(tmp_path: Path) -> None:
    p = tmp_path / "big.png"
    _write_png(p, 4000, 2000)
    enc = encode_image(p, max_edge=1568)
    img = Image.open(__import__("io").BytesIO(base64.b64decode(enc.data)))
    assert max(img.size) <= 1568


def test_encode_image_keeps_small_unscaled(tmp_path: Path) -> None:
    p = tmp_path / "ok.png"
    _write_png(p, 800, 600)
    enc = encode_image(p, max_edge=1568)
    img = Image.open(__import__("io").BytesIO(base64.b64decode(enc.data)))
    assert img.size == (800, 600)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/unit/ai/test_images.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'davinci_monet.ai'`.

- [ ] **Step 3: Create the package and implementation**

Create `davinci_monet/ai/__init__.py`:

```python
"""DAVINCI AI subpackage: single-prompt analysis summaries via the Claude API."""
```

Create `davinci_monet/ai/images.py`:

```python
"""Encode plot images for the Claude vision prompt.

Loads a PNG, downscales its long edge to keep vision token cost predictable,
and returns base64-encoded PNG bytes ready for an Anthropic image content block.
"""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from pathlib import Path


@dataclass
class EncodedImage:
    """A base64-encoded image ready for an Anthropic image block."""

    media_type: str
    data: str


def encode_image(path: str | Path, max_edge: int = 1568) -> EncodedImage:
    """Load a PNG, downscale to ``max_edge`` on its long side, return base64.

    Parameters
    ----------
    path
        Path to a PNG file.
    max_edge
        Maximum length (px) of the longer image edge. Larger images are
        downscaled preserving aspect ratio. Anthropic recommends <=1568px.

    Returns
    -------
    EncodedImage
        media_type ``"image/png"`` and base64-encoded PNG data.
    """
    from PIL import Image

    with Image.open(path) as img:
        img = img.convert("RGB")
        longest = max(img.size)
        if longest > max_edge:
            scale = max_edge / longest
            new_size = (round(img.size[0] * scale), round(img.size[1] * scale))
            img = img.resize(new_size, Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="PNG")

    data = base64.b64encode(buf.getvalue()).decode("ascii")
    return EncodedImage(media_type="image/png", data=data)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/unit/ai/test_images.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add davinci_monet/ai/__init__.py davinci_monet/ai/images.py davinci_monet/tests/unit/ai/test_images.py
git commit -m "feat(ai): add image encoding with downscale for vision prompt"
```

---

## Task 4: `ai/payload.py` — collect summary payload from context

**Files:**
- Create: `davinci_monet/ai/payload.py`
- Test: `davinci_monet/tests/unit/ai/test_payload.py`

- [ ] **Step 1: Write the failing test**

Create `davinci_monet/tests/unit/ai/test_payload.py`:

```python
"""Unit tests for ai.payload.collect_payload."""

from __future__ import annotations

from pathlib import Path

from davinci_monet.ai.payload import ImageRef, SummaryPayload, collect_payload
from davinci_monet.config.schema import SummaryConfig
from davinci_monet.pipeline.stages import (
    PipelineContext,
    StageResult,
    StageStatus,
)


def _context_with_results(plot_paths: list[str]) -> PipelineContext:
    ctx = PipelineContext(
        config={
            "analysis": {"start_time": "2024-02-01", "end_time": "2024-02-03"},
            "sources": {"cam": {"type": "cesm_fv"}, "airnow": {"type": "pt_sfc"}},
            "pairs": {"cam_vs_airnow_o3": {"sources": ["cam", "airnow"]}},
        }
    )
    ctx.results["statistics"] = StageResult(
        stage_name="statistics",
        status=StageStatus.COMPLETED,
        data={
            "cam_vs_airnow_o3": {
                "O3": {"N": 120, "MB": -2.5, "RMSE": 6.1, "R": 0.82, "_internal": 1},
                "_per_flight": [{"flight": "x"}],
            }
        },
    )
    ctx.results["plotting"] = StageResult(
        stage_name="plotting",
        status=StageStatus.COMPLETED,
        data={"plots_generated": plot_paths},
    )
    return ctx


def test_collect_payload_flattens_stats() -> None:
    ctx = _context_with_results(["00_o3_scatter.png"])
    payload = collect_payload(ctx, SummaryConfig(enabled=True))
    assert isinstance(payload, SummaryPayload)
    assert payload.period == {"start": "2024-02-01", "end": "2024-02-03"}
    assert any("cam" in s for s in payload.sources_summary)
    assert payload.pairs_summary == ["cam_vs_airnow_o3"]
    assert len(payload.stats_rows) == 1
    row = payload.stats_rows[0]
    assert row["pair"] == "cam_vs_airnow_o3"
    assert row["variable"] == "O3"
    assert row["metrics"]["N"] == 120
    # internal keys are dropped
    assert "_internal" not in row["metrics"]


def test_collect_payload_caps_images_when_no_plots_list() -> None:
    paths = [f"{i:02d}_plot.png" for i in range(12)]
    ctx = _context_with_results(paths)
    payload = collect_payload(ctx, SummaryConfig(enabled=True, max_images=3))
    assert len(payload.images) == 3
    assert all(isinstance(i, ImageRef) for i in payload.images)


def test_collect_payload_selects_named_plots() -> None:
    paths = ["00_o3_scatter.png", "01_pm25_spatial_bias.png", "02_o3_timeseries.png"]
    ctx = _context_with_results(paths)
    payload = collect_payload(
        ctx, SummaryConfig(enabled=True, plots=["pm25_spatial_bias"])
    )
    assert len(payload.images) == 1
    assert "pm25_spatial_bias" in payload.images[0].path
    assert payload.images[0].caption == Path(payload.images[0].path).stem


def test_collect_payload_ignores_non_png() -> None:
    ctx = _context_with_results(["00_o3_scatter.png", "00_o3_scatter.pdf"])
    payload = collect_payload(ctx, SummaryConfig(enabled=True))
    assert len(payload.images) == 1
    assert payload.images[0].path.endswith(".png")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/unit/ai/test_payload.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'davinci_monet.ai.payload'`.

- [ ] **Step 3: Implement payload collection**

Create `davinci_monet/ai/payload.py`:

```python
"""Collect the data the summary prompt needs out of a PipelineContext."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from davinci_monet.config.schema import SummaryConfig
    from davinci_monet.pipeline.stages import PipelineContext


@dataclass
class ImageRef:
    """A plot image selected for the prompt."""

    caption: str
    path: str


@dataclass
class SummaryPayload:
    """Everything the summary prompt is built from."""

    period: dict[str, Any]
    sources_summary: list[str]
    pairs_summary: list[str]
    stats_rows: list[dict[str, Any]]
    images: list[ImageRef]
    instructions: str | None


_STATS_STAGES = ("statistics", "obs_statistics")
_PLOT_STAGES = ("plotting", "obs_plotting")


def collect_payload(context: "PipelineContext", cfg: "SummaryConfig") -> SummaryPayload:
    """Build a :class:`SummaryPayload` from the run's config and stage results."""
    config = context.config
    analysis = config.get("analysis", {}) or {}
    period = {"start": analysis.get("start_time"), "end": analysis.get("end_time")}

    sources_summary: list[str] = []
    for block in ("sources", "model", "obs"):
        for label, spec in (config.get(block) or {}).items():
            if isinstance(spec, dict):
                stype = (
                    spec.get("type")
                    or spec.get("mod_type")
                    or spec.get("obs_type")
                    or "?"
                )
                sources_summary.append(f"{label} ({stype})")

    pairs_summary = list((config.get("pairs") or {}).keys())

    stats_rows: list[dict[str, Any]] = []
    for stage_key in _STATS_STAGES:
        result = context.results.get(stage_key)
        data = getattr(result, "data", None)
        if not isinstance(data, dict):
            continue
        for pair_key, pair_stats in data.items():
            if not isinstance(pair_stats, dict):
                continue
            for var_name, var_stats in pair_stats.items():
                if var_name.startswith("_") or not isinstance(var_stats, dict):
                    continue
                metrics = {
                    k: v for k, v in var_stats.items() if not k.startswith("_")
                }
                stats_rows.append(
                    {"pair": pair_key, "variable": var_name, "metrics": metrics}
                )

    all_plots: list[str] = []
    for stage_key in _PLOT_STAGES:
        result = context.results.get(stage_key)
        data = getattr(result, "data", None)
        if not isinstance(data, dict):
            continue
        for path in data.get("plots_generated", []) or []:
            if str(path).lower().endswith(".png"):
                all_plots.append(str(path))

    if cfg.plots:
        selected = [p for p in all_plots if any(k in Path(p).stem for k in cfg.plots)]
    else:
        selected = all_plots[: cfg.max_images]

    images = [ImageRef(caption=Path(p).stem, path=p) for p in selected]

    return SummaryPayload(
        period=period,
        sources_summary=sources_summary,
        pairs_summary=pairs_summary,
        stats_rows=stats_rows,
        images=images,
        instructions=cfg.instructions,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/unit/ai/test_payload.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add davinci_monet/ai/payload.py davinci_monet/tests/unit/ai/test_payload.py
git commit -m "feat(ai): collect summary payload (stats + config + plots) from context"
```

---

## Task 5: `ai/summarizer.py` — prompt building (pure)

**Files:**
- Create: `davinci_monet/ai/summarizer.py`
- Test: `davinci_monet/tests/unit/ai/test_build_prompt.py`

- [ ] **Step 1: Write the failing test**

Create `davinci_monet/tests/unit/ai/test_build_prompt.py`:

```python
"""Unit tests for ai.summarizer.build_prompt and render_text (pure, no network)."""

from __future__ import annotations

from davinci_monet.ai.images import EncodedImage
from davinci_monet.ai.payload import ImageRef, SummaryPayload
from davinci_monet.ai.summarizer import SYSTEM_PROMPT, build_prompt, render_text


def _payload(instructions: str | None = None) -> SummaryPayload:
    return SummaryPayload(
        period={"start": "2024-02-01", "end": "2024-02-03"},
        sources_summary=["cam (cesm_fv)", "airnow (pt_sfc)"],
        pairs_summary=["cam_vs_airnow_o3"],
        stats_rows=[
            {
                "pair": "cam_vs_airnow_o3",
                "variable": "O3",
                "metrics": {"N": 120, "MB": -2.5, "R": 0.82},
            }
        ],
        images=[ImageRef(caption="01_o3_scatter", path="/x/01_o3_scatter.png")],
        instructions=instructions,
    )


def test_render_text_includes_period_sources_and_stats() -> None:
    text = render_text(_payload())
    assert "2024-02-01" in text and "2024-02-03" in text
    assert "cam (cesm_fv)" in text
    assert "cam_vs_airnow_o3" in text
    assert "O3" in text and "N=120" in text


def test_render_text_appends_instructions() -> None:
    text = render_text(_payload(instructions="Focus on coastal sites."))
    assert "Focus on coastal sites." in text


def test_system_prompt_requests_four_sections() -> None:
    for heading in (
        "## What this run is",
        "## Headline metrics",
        "## Interpretation",
        "## Caveats",
    ):
        assert heading in SYSTEM_PROMPT


def test_build_prompt_structure() -> None:
    encoded = [("01_o3_scatter", EncodedImage(media_type="image/png", data="QUJD"))]
    system, content = build_prompt(_payload(), encoded)

    # system is a cache-controlled text block
    assert system[0]["type"] == "text"
    assert system[0]["cache_control"] == {"type": "ephemeral"}

    # content: one text block, then caption + image per figure
    assert content[0]["type"] == "text"
    assert content[1]["type"] == "text" and "Figure: 01_o3_scatter" in content[1]["text"]
    assert content[2]["type"] == "image"
    assert content[2]["source"]["media_type"] == "image/png"
    assert content[2]["source"]["data"] == "QUJD"


def test_build_prompt_no_images() -> None:
    payload = _payload()
    payload.images = []
    system, content = build_prompt(payload, [])
    assert len(content) == 1
    assert content[0]["type"] == "text"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/unit/ai/test_build_prompt.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'davinci_monet.ai.summarizer'`.

- [ ] **Step 3: Implement prompt building**

Create `davinci_monet/ai/summarizer.py` with the prompt-building portion (the API call is added in Task 6):

```python
"""Build and run the single-prompt Claude summary for an analysis run."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from davinci_monet.ai.images import EncodedImage
from davinci_monet.ai.payload import SummaryPayload

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are a climate and atmospheric-composition model-evaluation analyst.
You are given the configuration, summary statistics, and figures from a single DAVINCI
model-evaluation run. Write a concise markdown brief with EXACTLY these four sections:

## What this run is
## Headline metrics
## Interpretation
## Caveats

In "What this run is", describe the data sources, time period, variables, and pairing.
In "Headline metrics", call out the most important statistics per variable and pair.
In "Interpretation", describe where the model agrees or disagrees with observations and
any spatial or temporal patterns visible in the attached figures.
In "Caveats", note the sample size and what the metrics do not capture.

Be specific and quantitative. Do not invent numbers that are not present in the provided
statistics or visible in the figures."""


class SummaryError(Exception):
    """Raised when the summary cannot be produced (degraded non-fatally)."""


@dataclass
class SummaryResult:
    """Result of a successful summary generation."""

    markdown: str
    model: str
    usage: dict[str, Any]
    plots_used: list[str]
    images_sent: int


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.3g}"
    return str(value)


def render_text(payload: SummaryPayload) -> str:
    """Render the textual portion of the user message from the payload."""
    lines: list[str] = ["# Analysis run"]
    period = payload.period
    lines.append(f"Period: {period.get('start')} to {period.get('end')}")
    if payload.sources_summary:
        lines.append("Sources: " + ", ".join(payload.sources_summary))
    if payload.pairs_summary:
        lines.append("Pairs: " + ", ".join(payload.pairs_summary))

    lines.append("")
    lines.append("## Statistics")
    if payload.stats_rows:
        for row in payload.stats_rows:
            metric_str = ", ".join(
                f"{k}={_fmt(v)}" for k, v in row["metrics"].items()
            )
            lines.append(f"- {row['pair']} / {row['variable']}: {metric_str}")
    else:
        lines.append("(no statistics available)")

    if payload.instructions:
        lines.append("")
        lines.append("## Additional instructions")
        lines.append(payload.instructions)

    return "\n".join(lines)


def build_prompt(
    payload: SummaryPayload,
    encoded_images: list[tuple[str, EncodedImage]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build the (system, user_content) blocks for messages.create. Pure, no IO."""
    system: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": SYSTEM_PROMPT,
            "cache_control": {"type": "ephemeral"},
        }
    ]

    content: list[dict[str, Any]] = [{"type": "text", "text": render_text(payload)}]
    for caption, enc in encoded_images:
        content.append({"type": "text", "text": f"Figure: {caption}"})
        content.append(
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": enc.media_type,
                    "data": enc.data,
                },
            }
        )
    return system, content
```

- [ ] **Step 4: Run test to verify it passes**

Run: `HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/unit/ai/test_build_prompt.py -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add davinci_monet/ai/summarizer.py davinci_monet/tests/unit/ai/test_build_prompt.py
git commit -m "feat(ai): build system+user prompt blocks for the summary"
```

---

## Task 6: `ai/summarizer.py` — `generate_summary` with injectable client

**Files:**
- Modify: `davinci_monet/ai/summarizer.py` (append `_build_client` and `generate_summary`)
- Modify: `davinci_monet/ai/__init__.py` (export public API)
- Test: `davinci_monet/tests/unit/ai/test_generate_summary.py`

- [ ] **Step 1: Write the failing test**

Create `davinci_monet/tests/unit/ai/test_generate_summary.py`:

```python
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
    result = generate_summary(
        _payload(_png_path(tmp_path)), cfg=SummaryConfig(), client=client
    )
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
        generate_summary(
            _payload(_png_path(tmp_path)), cfg=SummaryConfig(), client=_BoomClient()
        )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/unit/ai/test_generate_summary.py -v`
Expected: FAIL with `ImportError: cannot import name 'generate_summary'`.

- [ ] **Step 3: Append client + generation logic to `summarizer.py`**

Add to the top imports of `davinci_monet/ai/summarizer.py`:

```python
import os
```

Append these functions to the end of `davinci_monet/ai/summarizer.py`:

```python
def _build_client(cfg: Any) -> Any:
    """Construct a real Anthropic client (lazy import). Raises SummaryError."""
    try:
        import anthropic
    except ImportError as exc:  # pragma: no cover - exercised via stage tests
        raise SummaryError(
            "anthropic package not installed; install with: pip install 'davinci-monet[ai]'"
        ) from exc

    key = os.environ.get(cfg.api_key_env, "")
    if not key:
        raise SummaryError(
            f"API key environment variable '{cfg.api_key_env}' is not set"
        )
    return anthropic.Anthropic(api_key=key)


def generate_summary(
    payload: SummaryPayload,
    *,
    cfg: Any,
    client: Any | None = None,
) -> SummaryResult:
    """Encode images, build the prompt, call Claude, and return the markdown.

    ``client`` is injectable for testing; when ``None`` a real Anthropic client
    is constructed from ``cfg.api_key_env``.
    """
    from davinci_monet.ai.images import encode_image

    if client is None:
        client = _build_client(cfg)

    encoded: list[tuple[str, EncodedImage]] = []
    for img in payload.images:
        try:
            encoded.append((img.caption, encode_image(img.path)))
        except Exception as exc:  # noqa: BLE001 - bad figure must not abort summary
            logger.warning("Skipping figure %s: %s", img.path, exc)

    system, content = build_prompt(payload, encoded)

    try:
        response = client.messages.create(
            model=cfg.model,
            max_tokens=cfg.max_tokens,
            system=system,
            messages=[{"role": "user", "content": content}],
        )
    except Exception as exc:  # noqa: BLE001 - any API/network failure degrades
        raise SummaryError(f"Claude API request failed: {exc}") from exc

    try:
        markdown = response.content[0].text
        usage = {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        }
    except (AttributeError, IndexError) as exc:
        raise SummaryError(f"Unexpected API response shape: {exc}") from exc

    return SummaryResult(
        markdown=markdown,
        model=getattr(response, "model", cfg.model),
        usage=usage,
        plots_used=[caption for caption, _ in encoded],
        images_sent=len(encoded),
    )
```

- [ ] **Step 4: Export the public API from `ai/__init__.py`**

Replace the contents of `davinci_monet/ai/__init__.py` with:

```python
"""DAVINCI AI subpackage: single-prompt analysis summaries via the Claude API."""

from __future__ import annotations

from davinci_monet.ai.payload import ImageRef, SummaryPayload, collect_payload
from davinci_monet.ai.summarizer import (
    SummaryError,
    SummaryResult,
    build_prompt,
    generate_summary,
)
from davinci_monet.config.schema import SummaryConfig

__all__ = [
    "ImageRef",
    "SummaryPayload",
    "collect_payload",
    "SummaryConfig",
    "SummaryError",
    "SummaryResult",
    "build_prompt",
    "generate_summary",
]
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/unit/ai/ -v`
Expected: PASS (all ai unit tests).

- [ ] **Step 6: Commit**

```bash
git add davinci_monet/ai/summarizer.py davinci_monet/ai/__init__.py davinci_monet/tests/unit/ai/test_generate_summary.py
git commit -m "feat(ai): generate_summary with injectable Anthropic client"
```

---

## Task 7: `SummaryStage` + wire into pipeline factories

**Files:**
- Modify: `davinci_monet/pipeline/stages.py` (add `SummaryStage` after `ObsStatisticsStage` ~line 2945; append to `create_standard_pipeline` and `create_obs_pipeline`)
- Test: `davinci_monet/tests/unit/pipeline/test_summary_stage.py`

- [ ] **Step 1: Write the failing test**

Create `davinci_monet/tests/unit/pipeline/test_summary_stage.py`:

```python
"""Unit tests for SummaryStage (engine mocked via monkeypatch)."""

from __future__ import annotations

from pathlib import Path

import davinci_monet.ai.summarizer as summarizer_mod
from davinci_monet.ai.summarizer import SummaryError, SummaryResult
from davinci_monet.pipeline.stages import (
    PipelineContext,
    StageStatus,
    SummaryStage,
)


def _ctx(tmp_path: Path, enabled: bool = True) -> PipelineContext:
    return PipelineContext(
        config={
            "analysis": {
                "start_time": "2024-02-01",
                "end_time": "2024-02-03",
                "output_dir": str(tmp_path / "output"),
            },
            "summary": {"enabled": enabled},
        }
    )


def test_summary_stage_disabled_is_skipped(tmp_path: Path) -> None:
    result = SummaryStage().execute(_ctx(tmp_path, enabled=False))
    assert result.status == StageStatus.SKIPPED


def test_summary_stage_writes_file(monkeypatch, tmp_path: Path) -> None:
    def _fake_client(cfg):
        class _Msgs:
            def create(self, **kwargs):
                class _Block:
                    text = "## What this run is\nok\n## Caveats\n"

                class _Usage:
                    input_tokens = 5
                    output_tokens = 6

                class _Resp:
                    content = [_Block()]
                    usage = _Usage()
                    model = cfg.model

                return _Resp()

        class _Client:
            messages = _Msgs()

        return _Client()

    monkeypatch.setattr(summarizer_mod, "_build_client", _fake_client)

    result = SummaryStage().execute(_ctx(tmp_path))
    assert result.status == StageStatus.COMPLETED
    out = Path(result.data["summary_file"])
    assert out.exists()
    assert out.name == "AI_summary.md"
    assert "## Caveats" in out.read_text()
    assert result.data["usage"] == {"input_tokens": 5, "output_tokens": 6}


def test_summary_stage_error_is_nonfatal(monkeypatch, tmp_path: Path) -> None:
    def _boom(cfg):
        raise SummaryError("no key")

    monkeypatch.setattr(summarizer_mod, "_build_client", _boom)

    result = SummaryStage().execute(_ctx(tmp_path))
    assert result.status == StageStatus.SKIPPED
    assert "no key" in str(result.data)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/unit/pipeline/test_summary_stage.py -v`
Expected: FAIL with `ImportError: cannot import name 'SummaryStage'`.

- [ ] **Step 3: Implement `SummaryStage`**

In `davinci_monet/pipeline/stages.py`, add this class immediately after `ObsStatisticsStage` and before `def create_standard_pipeline()` (around line 2945):

```python
class SummaryStage(BaseStage):
    """Optional final stage: AI summary of the analysis run via the Claude API.

    Always non-fatal. When ``summary.enabled`` is false the stage is skipped.
    Any failure (missing dependency/key, network/API error) logs a warning and
    returns SKIPPED so an otherwise-complete run is still reported successful.
    """

    def __init__(self) -> None:
        super().__init__("summary")

    def execute(self, context: PipelineContext) -> StageResult:
        import logging
        import time
        from pathlib import Path

        from davinci_monet.ai import collect_payload, generate_summary
        from davinci_monet.ai.summarizer import SummaryError
        from davinci_monet.config.schema import SummaryConfig

        start = time.time()
        logger = logging.getLogger(__name__)

        cfg = SummaryConfig.model_validate(context.config.get("summary") or {})
        if not cfg.enabled:
            return self._create_result(
                StageStatus.SKIPPED,
                data={"skipped": "summary disabled"},
                duration=time.time() - start,
            )

        payload = collect_payload(context, cfg)
        try:
            result = generate_summary(payload, cfg=cfg)
        except SummaryError as exc:
            logger.warning("AI summary skipped: %s", exc)
            return self._create_result(
                StageStatus.SKIPPED,
                data={"skipped": str(exc)},
                duration=time.time() - start,
            )

        output_dir = Path(context.config.get("analysis", {}).get("output_dir") or ".")
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / cfg.output_filename
        out_path.write_text(result.markdown)

        context.log_progress(f"AI summary written: {out_path}")
        context.log_progress(result.markdown)

        return self._create_result(
            StageStatus.COMPLETED,
            data={
                "summary_file": str(out_path),
                "model": result.model,
                "usage": result.usage,
                "images_sent": result.images_sent,
            },
            duration=time.time() - start,
        )
```

- [ ] **Step 4: Append `SummaryStage` to both pipeline factories**

In `create_standard_pipeline()`, add `SummaryStage()` as the last entry:

```python
    return [
        LoadSourcesStage(),
        PairingStage(),
        StatisticsStage(),
        PlottingStage(),
        ObsStatisticsStage(),
        ObsPlottingStage(),
        SaveResultsStage(),
        SummaryStage(),
    ]
```

In `create_obs_pipeline()`, add `SummaryStage()` as the last entry:

```python
    return [
        LoadSourcesStage(),
        ObsStatisticsStage(),
        ObsPlottingStage(),
        SaveResultsStage(),
        SummaryStage(),
    ]
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/unit/pipeline/test_summary_stage.py -v`
Expected: PASS (3 tests).

- [ ] **Step 6: Commit**

```bash
git add davinci_monet/pipeline/stages.py davinci_monet/tests/unit/pipeline/test_summary_stage.py
git commit -m "feat(pipeline): add non-fatal SummaryStage to both pipelines"
```

---

## Task 8: Integration test through `run_from_config`

**Files:**
- Test: `davinci_monet/tests/integration/test_ai_summary_pipeline.py`

- [ ] **Step 1: Write the integration test**

Create `davinci_monet/tests/integration/test_ai_summary_pipeline.py`. This drives the real pipeline via `PipelineRunner.run_from_config` with synthetic data, mocking only the Anthropic client construction:

```python
"""Integration: AI summary stage runs through PipelineRunner.run_from_config.

The pipeline runs for real on synthetic data; only the Anthropic client is
stubbed so no network call is made.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np

import davinci_monet.ai.summarizer as summarizer_mod
from davinci_monet.config.parser import LegacyConfigWarning
from davinci_monet.core.protocols import DataGeometry
from davinci_monet.tests.synthetic.generators import Domain, TimeConfig
from davinci_monet.tests.synthetic.models import create_model_dataset
from davinci_monet.tests.synthetic.scenarios import PerfectMatchScenario


class _StubClient:
    """Returns a fixed markdown brief; records calls."""

    def __init__(self) -> None:
        self.calls: list[dict] = []
        outer = self

        class _Msgs:
            def create(self, **kwargs):
                outer.calls.append(kwargs)

                class _Block:
                    text = (
                        "## What this run is\nSynthetic O3 run.\n"
                        "## Headline metrics\nN counted.\n"
                        "## Interpretation\nClose match.\n"
                        "## Caveats\nSynthetic data.\n"
                    )

                class _Usage:
                    input_tokens = 100
                    output_tokens = 50

                class _Resp:
                    content = [_Block()]
                    usage = _Usage()
                    model = kwargs["model"]

                return _Resp()

        self.messages = _Msgs()


def _build_config(tmp_path: Path) -> dict:
    domain = Domain(
        lon_min=-105.0, lon_max=-95.0, lat_min=35.0, lat_max=45.0, n_lon=12, n_lat=12
    )
    time_cfg = TimeConfig(start="2024-01-15 00:00", end="2024-01-17 00:00", freq="1h")

    model_ds = create_model_dataset(
        variables=["O3"], domain=domain, time_config=time_cfg, seed=42
    )
    scenario = PerfectMatchScenario(
        variables=["O3"],
        domain=domain,
        time_config=time_cfg,
        geometry=DataGeometry.POINT,
        n_obs=10,
        noise_level=0.0,
        seed=42,
    )
    obs_ds = scenario._generate_point_obs(model_ds)

    rng = np.random.default_rng(42)
    model_ds["O3"] = model_ds["O3"] + 5.0 + rng.normal(0, 3.0, size=model_ds["O3"].shape)

    model_path = tmp_path / "model.nc"
    obs_path = tmp_path / "obs.nc"
    model_ds.to_netcdf(model_path)
    obs_ds.to_netcdf(obs_path)

    return {
        "analysis": {
            "start_time": "2024-01-15 00:00",
            "end_time": "2024-01-17 00:00",
            "output_dir": str(tmp_path / "output"),
            "log_dir": str(tmp_path / "logs"),
        },
        "model": {
            "synthetic": {
                "mod_type": "generic",
                "files": str(model_path),
                "radius_of_influence": 50000,
                "mapping": {"surface": {"O3": "O3"}},
                "variables": {"O3": {"units": "ppb"}},
            },
        },
        "obs": {
            "surface": {
                "obs_type": "pt_sfc",
                "filename": str(obs_path),
                "variables": {"O3": {"obs_min": 0, "obs_max": 200, "units": "ppb"}},
            },
        },
        "pairs": {
            "synthetic_surface": {
                "model": "synthetic",
                "obs": "surface",
                "variable": {"model_var": "O3", "obs_var": "O3"},
            },
        },
        "plots": {
            "scatter_o3": {
                "type": "scatter",
                "pairs": ["synthetic_surface"],
                "title": "O3: Model vs Observations",
            },
        },
        "stats": {"metrics": ["N", "MB", "RMSE", "R", "NMB", "NME", "IOA"]},
    }


def test_summary_stage_writes_file_through_pipeline(monkeypatch, tmp_path: Path) -> None:
    from davinci_monet.pipeline.runner import PipelineRunner

    stub = _StubClient()
    monkeypatch.setattr(summarizer_mod, "_build_client", lambda cfg: stub)

    config = _build_config(tmp_path)
    config["summary"] = {"enabled": True, "model": "claude-haiku-4-5"}

    runner = PipelineRunner(show_progress=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", LegacyConfigWarning)
        result = runner.run_from_config(config)

    assert result.success, "pipeline run failed"
    summary_file = tmp_path / "output" / "AI_summary.md"
    assert summary_file.exists(), "AI_summary.md was not written"
    assert "## Caveats" in summary_file.read_text()
    # the stubbed client actually received the scatter plot image
    assert stub.calls, "Anthropic client was not called"
    user_content = stub.calls[0]["messages"][0]["content"]
    assert any(block["type"] == "image" for block in user_content)


def test_summary_stage_skips_without_api_key(monkeypatch, tmp_path: Path) -> None:
    from davinci_monet.pipeline.runner import PipelineRunner

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    config = _build_config(tmp_path)
    config["summary"] = {"enabled": True}

    runner = PipelineRunner(show_progress=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", LegacyConfigWarning)
        result = runner.run_from_config(config)

    # run still succeeds; summary file is absent (stage skipped non-fatally)
    assert result.success, "pipeline run should still succeed without a key"
    assert not (tmp_path / "output" / "AI_summary.md").exists()
```

- [ ] **Step 2: Verify the synthetic import paths**

Before running, confirm the synthetic helper import paths used above match the repo (they are taken from `davinci_monet/tests/test_integration.py`). Run:
```bash
HDF5_USE_FILE_LOCKING=FALSE python -c "from davinci_monet.tests.synthetic.generators import Domain, TimeConfig; from davinci_monet.tests.synthetic.models import create_model_dataset; from davinci_monet.tests.synthetic.scenarios import PerfectMatchScenario; from davinci_monet.config.parser import LegacyConfigWarning; print('ok')"
```
Expected: prints `ok`. If any import fails, open `davinci_monet/tests/test_integration.py` (top imports) and adjust the paths to match — do not change the test logic.

- [ ] **Step 3: Run the integration test**

Run: `HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/integration/test_ai_summary_pipeline.py -v`
Expected: PASS (2 tests). If the run reports failure, inspect `result` and the pipeline log under `tmp_path/logs`; do not weaken the assertions to pass.

- [ ] **Step 4: Commit**

```bash
git add davinci_monet/tests/integration/test_ai_summary_pipeline.py
git commit -m "test(ai): integration test for SummaryStage through run_from_config"
```

---

## Task 9: Documentation — `summary:` config block

**Files:**
- Modify: `README.md` (add a short "AI Summary" subsection)
- Modify: `CLAUDE.md` (add `summary:` to the YAML config pattern + a Common Gotchas entry)

- [ ] **Step 1: Add an AI Summary section to README.md**

In `README.md`, after the "Plot Styling" / before "Common Gotchas" area (pick a sensible spot near configuration docs), add:

````markdown
## AI Summary (Visual Intelligence)

Enable an optional final stage that asks Claude to read the run's statistics and
plots and write a structured markdown brief (`AI_summary.md`) into the output
directory. Requires the `[ai]` extra and an Anthropic API key.

```bash
pip install -e ".[ai]"
export ANTHROPIC_API_KEY=sk-ant-...
```

```yaml
summary:
  enabled: true
  model: claude-haiku-4-5          # cheapest vision model; bump to claude-sonnet-4-6
  plots: [scatter_o3, spatial_bias_o3]   # optional; omit to send up to max_images
  max_images: 8
  instructions: "Focus on coastal sites."   # optional steering
```

The stage is always non-fatal: with no key or no network it logs a warning and
is skipped, and the analysis run still succeeds.
````

- [ ] **Step 2: Add `summary:` to the CLAUDE.md YAML config pattern**

In `CLAUDE.md`, in the "YAML Configuration Pattern" block, add a `summary:` section after the `stats:` block:

```yaml
summary:
  enabled: true
  model: claude-haiku-4-5  # cheapest vision model
  max_images: 8
```

- [ ] **Step 3: Add a Common Gotchas entry**

In `CLAUDE.md` "Common Gotchas" list, add:

```markdown
9. **AI summary stage**: The `summary:` block enables an opt-in final stage that
   sends stats + plot images to the Claude API (`pip install -e ".[ai]"`,
   `ANTHROPIC_API_KEY`). It is always non-fatal — missing key/network just skips
   it. Default model `claude-haiku-4-5`. Vision images are downscaled to ≤1568px.
```

- [ ] **Step 4: Commit**

```bash
git add README.md CLAUDE.md
git commit -m "docs: document the summary: AI analysis stage"
```

---

## Task 10: Full suite + formatting/type gates

**Files:** none (verification only)

- [ ] **Step 1: Run the full test suite**

Run:
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate davinci
HDF5_USE_FILE_LOCKING=FALSE python -m pytest -q
```
Expected: all tests pass (previous count + the new ai/summary tests), 0 unexpected warnings.

- [ ] **Step 2: Format and import-sort the new code**

Run:
```bash
black davinci_monet && isort davinci_monet
```
Expected: reformats only if needed; no errors.

- [ ] **Step 3: Type-check**

Run:
```bash
mypy davinci_monet
```
Expected: no new errors in `davinci_monet/ai/` or `pipeline/stages.py`. If `anthropic` is untyped, the lazy import inside functions keeps it out of module-level type analysis; add `# type: ignore[import-untyped]` on the `import anthropic` line only if mypy complains.

- [ ] **Step 4: Commit any formatting fixes**

```bash
git add -A
git commit -m "style: black/isort/mypy fixes for ai summary"
```

(Skip if nothing changed.)

---

## Self-Review Notes (for the implementer)

- **Spec coverage:** config block (T2), engine modules `images`/`payload`/`summarizer` (T3–T6), vision image attachment (T3, T5), `SummaryStage` non-fatal wiring into both pipelines (T7), unit + pipeline-integration tests with mocked client (T3–T8), dependencies (T1), docs (T9).
- **Deviation from spec:** `SummaryConfig` lives in `config/schema.py` (not `ai/config.py`) to avoid a circular import; re-exported from `davinci_monet.ai`. Captions use the plot filename stem (the config plot key is embedded there) because `plots_generated` carries paths only, not a key→title map.
- **Injection seam:** tests monkeypatch `davinci_monet.ai.summarizer._build_client`; the real network client is never constructed in tests.
- **Non-fatal guarantee:** every failure path returns `StageStatus.SKIPPED`; the run's `result.success` is unaffected (verified in T8).
