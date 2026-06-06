# Itemized AI Summary Display Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the terminal show a condensed itemized bullet list derived from the AI summary brief, while `AI_summary.md` keeps the full brief.

**Architecture:** A pure `extract_bullets(markdown)` in `ai/summarizer.py` condenses the brief to a list; `SummaryStage` stores it in `StageResult.data["bullets"]`; `ProgressFormatter.print_summary` renders the list + a file pointer instead of the full markdown. No prompt change, no second model call.

**Tech Stack:** Python 3.11+, Rich (terminal panels), pytest.

**Spec:** `docs/superpowers/specs/2026-06-06-itemized-summary-display-design.md`

**Environment:** Run tests in the `davinci` conda env:
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate davinci
HDF5_USE_FILE_LOCKING=FALSE python -m pytest <path> -v
```
The **full** suite needs `DASK_NUM_WORKERS=1 HDF5_USE_FILE_LOCKING=FALSE` (HDF5 segfault, CLAUDE.md gotcha #8).

---

## Task 1: `extract_bullets` pure function

**Files:**
- Modify: `davinci_monet/ai/summarizer.py` (add `import re` + `extract_bullets`)
- Modify: `davinci_monet/ai/__init__.py` (export `extract_bullets`)
- Test: `davinci_monet/tests/unit/ai/test_extract_bullets.py`

- [ ] **Step 1: Write the failing test**

Create `davinci_monet/tests/unit/ai/test_extract_bullets.py`:

```python
"""Unit tests for ai.summarizer.extract_bullets."""

from __future__ import annotations

from davinci_monet.ai.summarizer import extract_bullets

_SAMPLE = """# DAVINCI Model Evaluation Brief

## What this run is
Synthetic O3 vs surface obs.

## Headline metrics
- Mean Bias +4.82 ppb
- RMSE 5.66 ppb
- R 0.849

## Caveats
- N=98; synthetic data
"""


def test_extract_bullets_returns_all_bullets() -> None:
    assert extract_bullets(_SAMPLE) == [
        "Mean Bias +4.82 ppb",
        "RMSE 5.66 ppb",
        "R 0.849",
        "N=98; synthetic data",
    ]


def test_extract_bullets_handles_star_and_unicode_markers() -> None:
    md = "* star item\n• dot item\n  - indented item\n"
    assert extract_bullets(md) == ["star item", "dot item", "indented item"]


def test_extract_bullets_falls_back_to_subheadings() -> None:
    md = "# Title\n## What this run is\nProse.\n## Caveats\nMore prose.\n"
    assert extract_bullets(md) == ["What this run is", "Caveats"]


def test_extract_bullets_empty_when_no_bullets_or_subheadings() -> None:
    assert extract_bullets("# Title only\nplain prose line\n") == []


def test_extract_bullets_caps_with_overflow() -> None:
    md = "\n".join(f"- item {i}" for i in range(20))
    out = extract_bullets(md, max_items=5)
    assert len(out) == 5
    assert out[:4] == ["item 0", "item 1", "item 2", "item 3"]
    assert out[4] == "… (full brief in AI_summary.md)"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/unit/ai/test_extract_bullets.py -v`
Expected: FAIL — `cannot import name 'extract_bullets'`.

- [ ] **Step 3: Implement `extract_bullets`**

In `davinci_monet/ai/summarizer.py`, add `import re` to the stdlib imports (with `logging`/`os`). Then add this function (place it after the `_fmt` helper, before `render_text`):

```python
_BULLET_RE = re.compile(r"^\s*[-*•]\s+(.*\S)\s*$")
_SUBHEADING_RE = re.compile(r"^\s*#{2,6}\s+(.*\S)\s*$")
_OVERFLOW_ITEM = "… (full brief in AI_summary.md)"


def extract_bullets(markdown: str, *, max_items: int = 12) -> list[str]:
    """Condense a brief to an itemized list for terminal display.

    Returns every markdown bullet line (-, *, or bullet char) across all
    sections, marker stripped. If the brief has no bullets, falls back to the
    level-2+ section headings. Returns an empty list if neither is present.
    Caps at ``max_items``; when truncated, the final item points to the file.
    """
    lines = markdown.splitlines()
    items = [m.group(1).strip() for line in lines if (m := _BULLET_RE.match(line))]
    if not items:
        items = [m.group(1).strip() for line in lines if (m := _SUBHEADING_RE.match(line))]
    if not items:
        return []
    if len(items) > max_items:
        items = items[: max_items - 1] + [_OVERFLOW_ITEM]
    return items
```

- [ ] **Step 4: Export from `ai/__init__.py`**

In `davinci_monet/ai/__init__.py`, add `extract_bullets` to the `summarizer` import and to `__all__`. The import line becomes:

```python
from davinci_monet.ai.summarizer import (
    SummaryError,
    SummaryResult,
    build_prompt,
    extract_bullets,
    generate_summary,
    resolve_api_key,
)
```

And add `"extract_bullets"` to the `__all__` list.

- [ ] **Step 5: Run test to verify it passes**

Run: `HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/unit/ai/test_extract_bullets.py -v`
Expected: PASS (5 tests).

- [ ] **Step 6: Commit**

```bash
git add davinci_monet/ai/summarizer.py davinci_monet/ai/__init__.py davinci_monet/tests/unit/ai/test_extract_bullets.py
git commit -m "feat(ai): extract_bullets — condense brief to itemized list"
```

---

## Task 2: `SummaryStage` carries bullets in stage data

**Files:**
- Modify: `davinci_monet/pipeline/stages.py` (`SummaryStage.execute`)
- Test: `davinci_monet/tests/unit/pipeline/test_summary_stage.py` (extend)

- [ ] **Step 1: Write the failing test**

Append to `davinci_monet/tests/unit/pipeline/test_summary_stage.py`:

```python
def test_summary_stage_includes_bullets_in_data(monkeypatch, tmp_path: Path) -> None:
    def _fake_client(cfg):
        class _Msgs:
            def create(self, **kwargs):
                class _Block:
                    text = (
                        "## Headline metrics\n- MB +4.8 ppb\n- R 0.85\n"
                        "## Caveats\n- N=98\n"
                    )

                class _Usage:
                    input_tokens = 1
                    output_tokens = 2

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
    assert result.data["bullets"] == ["MB +4.8 ppb", "R 0.85", "N=98"]
    # full brief still carried + written
    assert "## Caveats" in result.data["markdown"]
    assert "## Caveats" in Path(result.data["summary_file"]).read_text()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/unit/pipeline/test_summary_stage.py::test_summary_stage_includes_bullets_in_data -v`
Expected: FAIL — `KeyError: 'bullets'`.

- [ ] **Step 3: Add bullets to the stage result data**

In `davinci_monet/pipeline/stages.py`, in `SummaryStage.execute`:

First, add `extract_bullets` to the in-function import. Find:
```python
        from davinci_monet.ai import collect_payload, generate_summary
```
and change it to:
```python
        from davinci_monet.ai import collect_payload, extract_bullets, generate_summary
```

Then, in the success-path `return self._create_result(StageStatus.COMPLETED, data={...})`, add the `bullets` key. The data dict becomes:
```python
            data={
                "summary_file": str(out_path),
                "markdown": result.markdown,
                "bullets": extract_bullets(result.markdown),
                "model": result.model,
                "usage": result.usage,
                "images_sent": result.images_sent,
            },
```

- [ ] **Step 4: Run test to verify it passes**

Run: `HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/unit/pipeline/test_summary_stage.py -v`
Expected: PASS (all stage tests, including the existing ones).

- [ ] **Step 5: Commit**

```bash
git add davinci_monet/pipeline/stages.py davinci_monet/tests/unit/pipeline/test_summary_stage.py
git commit -m "feat(pipeline): carry itemized bullets in summary stage data"
```

---

## Task 3: `print_summary` renders the itemized list

**Files:**
- Modify: `davinci_monet/pipeline/runner.py` (`ProgressFormatter.print_summary` ~line 513; end-of-run call ~line 1698)
- Test: `davinci_monet/tests/unit/pipeline/test_summary_display.py` (replace the two `print_summary` tests)

- [ ] **Step 1: Update the failing tests**

In `davinci_monet/tests/unit/pipeline/test_summary_display.py`, replace the two existing tests `test_print_summary_outputs_markdown_when_enabled` and `test_print_summary_silent_when_disabled` with:

```python
def test_print_summary_lists_items_and_file() -> None:
    fmt, buf = _formatter_with_buffer(show_output=True)
    fmt.print_summary(["MB +4.82 ppb", "R 0.849"], "/out/AI_summary.md")
    out = buf.getvalue()
    assert "MB +4.82 ppb" in out
    assert "R 0.849" in out
    assert "AI_summary.md" in out


def test_print_summary_silent_when_disabled() -> None:
    fmt, buf = _formatter_with_buffer(show_output=False)
    fmt.print_summary(["hidden"], "/out/AI_summary.md")
    assert buf.getvalue() == ""


def test_print_summary_noop_when_no_items() -> None:
    fmt, buf = _formatter_with_buffer(show_output=True)
    fmt.print_summary([], "/out/AI_summary.md")
    assert buf.getvalue() == ""
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/unit/pipeline/test_summary_display.py -v`
Expected: FAIL — `print_summary` still takes `markdown` (TypeError on the list/2-arg calls, or wrong output).

- [ ] **Step 3: Change `print_summary` signature and body**

In `davinci_monet/pipeline/runner.py`, replace the entire `print_summary` method (currently rendering full markdown) with:

```python
    def print_summary(
        self, items: list[str], summary_file: str | None = None
    ) -> None:
        """Render an itemized AI summary to the terminal at end of run.

        Shows the condensed bullet list (derived from the full brief) plus a
        pointer to the full-brief file. No-op when output is disabled or there
        are no items.
        """
        if not self.show_output or not items:
            return
        from rich.panel import Panel

        body = "\n".join(f"• {item}" for item in items)
        if summary_file:
            body += f"\n\n[dim]Full brief → {summary_file}[/dim]"
        self._print()
        self._print(
            Panel(
                body,
                title="AI Summary",
                border_style=self.NCAR_AQUA,
                padding=(1, 2),
            )
        )
        self._print()
```

- [ ] **Step 4: Update the runner end-of-run call**

In `davinci_monet/pipeline/runner.py`, find the end-of-run summary block (it currently guards on `data.get("markdown")` and calls `print_summary(summary_result.data["markdown"])`) and replace it with:

```python
            summary_result = context.results.get("summary")
            if (
                summary_result is not None
                and summary_result.status == StageStatus.COMPLETED
                and isinstance(summary_result.data, dict)
                and summary_result.data.get("bullets")
            ):
                formatter.print_summary(
                    summary_result.data["bullets"],
                    summary_result.data.get("summary_file"),
                )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/unit/pipeline/test_summary_display.py -v`
Expected: PASS (3 print_summary tests + the unchanged validation-skip test).

- [ ] **Step 6: Commit**

```bash
git add davinci_monet/pipeline/runner.py davinci_monet/tests/unit/pipeline/test_summary_display.py
git commit -m "feat(display): render itemized AI summary + file pointer"
```

---

## Task 4: Integration — display receives bullets, file stays full

**Files:**
- Modify: `davinci_monet/tests/integration/test_ai_summary_pipeline.py` (update the display test)

- [ ] **Step 1: Update the integration display test**

In `davinci_monet/tests/integration/test_ai_summary_pipeline.py`, replace the body of `test_summary_displayed_to_terminal_at_end_of_run` (it currently monkeypatches `print_summary` with a `markdown` arg and asserts `"## Caveats" in displayed[0]`) with this version:

```python
def test_summary_displayed_to_terminal_at_end_of_run(monkeypatch, tmp_path: Path) -> None:
    """Terminal gets the itemized bullets; AI_summary.md keeps the full brief."""
    import davinci_monet.pipeline.runner as runner_mod
    from davinci_monet.pipeline.runner import PipelineRunner

    stub = _StubClient()
    monkeypatch.setattr(summarizer_mod, "_build_client", lambda cfg: stub)

    displayed: list[tuple] = []
    monkeypatch.setattr(
        runner_mod.ProgressFormatter,
        "print_summary",
        lambda self, items, summary_file=None: displayed.append((items, summary_file)),
    )

    config = _build_config(tmp_path)
    config["summary"] = {"enabled": True, "model": "claude-haiku-4-5"}

    runner = PipelineRunner(show_progress=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", LegacyConfigWarning)
        result = runner.run_from_config(config)

    assert result.success
    assert displayed, "summary was not displayed at end of run"
    items, summary_file = displayed[0]
    # the display got an itemized list (not the raw full markdown)
    assert isinstance(items, list) and items
    assert summary_file is not None and summary_file.endswith("AI_summary.md")
    # the file on disk still holds the full brief
    assert "## Caveats" in (tmp_path / "output" / "AI_summary.md").read_text()
```

Note: the `_StubClient` brief has no bullet lines, so `extract_bullets` falls back to the section headings — `items` is therefore the non-empty heading list, which is exactly the "itemized list (not the full markdown)" the test asserts.

- [ ] **Step 2: Run the integration test**

Run: `HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/integration/test_ai_summary_pipeline.py -v`
Expected: PASS (all 4 tests in the file).

- [ ] **Step 3: Commit**

```bash
git add davinci_monet/tests/integration/test_ai_summary_pipeline.py
git commit -m "test(ai): assert terminal gets itemized bullets, file stays full"
```

---

## Task 5: Full suite + gates

**Files:** none (verification only)

- [ ] **Step 1: Run the full suite**

Run:
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate davinci
DASK_NUM_WORKERS=1 HDF5_USE_FILE_LOCKING=FALSE python -m pytest -q -p no:cacheprovider
```
Expected: all tests pass (previous count + the new extractor tests), 1 skipped is fine.

- [ ] **Step 2: Format + import-sort**

Run:
```bash
black davinci_monet && isort davinci_monet
```
Expected: no changes (or trivial).

- [ ] **Step 3: Type-check**

Run:
```bash
mypy davinci_monet/ai davinci_monet/pipeline/runner.py davinci_monet/pipeline/stages.py
```
Expected: no new errors.

- [ ] **Step 4: Commit any formatting fixes**

```bash
git add -A
git commit -m "style: black/isort/mypy fixes for itemized summary display"
```
(Skip if nothing changed.)

---

## Self-Review Notes (for the implementer)

- **Spec coverage:** `extract_bullets` with bullet/heading fallback + cap (T1), bullets in stage data (T2), `print_summary` itemized render + runner call (T3), integration that display gets bullets and the file stays full (T4), suite + gates (T5).
- **Signature change is contained:** `print_summary(markdown)` → `print_summary(items, summary_file=None)` is updated in the method, its unit tests (T3), the runner call (T3), and the integration monkeypatch (T4) — all in lockstep.
- **No prompt / `.md` change:** `generate_summary` and `out_path.write_text(result.markdown)` are untouched; only `data["bullets"]` is added and the display layer changes.
- **`runner.py` stays free of `ai` imports:** extraction happens in the stage; the runner only renders the precomputed `bullets` list.
