# Summary Token Usage + OpenRouter Credits Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a tokens-used line (and, for OpenRouter, a credits-remaining line) to the terminal "AI Summary" panel, and strip markdown emphasis from the terminal bullets.

**Architecture:** `SummaryResult` gains `credits_remaining`; the OpenRouter provider fills it via a non-fatal `GET /api/v1/key`; `extract_bullets` strips emphasis; `ProgressFormatter.print_summary` gains `usage`/`credits_remaining` params and renders two extra lines. `AI_summary.md` and the prompt are unchanged.

**Tech Stack:** Python 3.11+, httpx (existing dep), Rich, pytest.

**Spec:** `docs/superpowers/specs/2026-06-06-summary-token-usage-design.md`

**Environment:** Run tests in the `davinci` conda env:
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate davinci
HDF5_USE_FILE_LOCKING=FALSE python -m pytest <path> -v
```
The **full** suite needs `DASK_NUM_WORKERS=1 HDF5_USE_FILE_LOCKING=FALSE` (HDF5 segfault, CLAUDE.md gotcha #8).

---

## Task 1: Strip markdown emphasis from terminal bullets

**Files:**
- Modify: `davinci_monet/ai/summarizer.py` (`extract_bullets` + a helper)
- Test: `davinci_monet/tests/unit/ai/test_extract_bullets.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `davinci_monet/tests/unit/ai/test_extract_bullets.py`:

```python
def test_extract_bullets_strips_markdown_emphasis() -> None:
    md = "## Headline metrics\n- **Mean Bias** +4.8 ppb\n- _RMSE_ 5.66\n- `R` 0.85\n"
    assert extract_bullets(md) == ["Mean Bias +4.8 ppb", "RMSE 5.66", "R 0.85"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/unit/ai/test_extract_bullets.py::test_extract_bullets_strips_markdown_emphasis -v`
Expected: FAIL — items still contain `**`, `_`, and backticks.

- [ ] **Step 3: Add the emphasis strip**

In `davinci_monet/ai/summarizer.py`, add an emphasis regex next to the existing `_BULLET_RE`/`_SUBHEADING_RE`/`_OVERFLOW_ITEM` definitions:

```python
_EMPHASIS_RE = re.compile(r"[*_`]")


def _strip_emphasis(text: str) -> str:
    return _EMPHASIS_RE.sub("", text).strip()
```

Then update `extract_bullets` to strip each item (bullet and heading paths), dropping any that become empty. The function body becomes:

```python
    lines = markdown.splitlines()
    items = [_strip_emphasis(m.group(1)) for line in lines if (m := _BULLET_RE.match(line))]
    if not items:
        items = [_strip_emphasis(m.group(1)) for line in lines if (m := _SUBHEADING_RE.match(line))]
    items = [item for item in items if item]
    if not items:
        return []
    if len(items) > max_items:
        items = items[: max_items - 1] + [_OVERFLOW_ITEM]
    return items
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/unit/ai/test_extract_bullets.py -v`
Expected: PASS (the new test plus all existing extract_bullets tests — the existing samples have no emphasis, so they are unchanged).

- [ ] **Step 5: Commit**

```bash
git add davinci_monet/ai/summarizer.py davinci_monet/tests/unit/ai/test_extract_bullets.py
git commit -m "feat(ai): strip markdown emphasis from terminal summary bullets"
```

---

## Task 2: `SummaryResult.credits_remaining` + OpenRouter credit fetch

**Files:**
- Modify: `davinci_monet/ai/summarizer.py` (`SummaryResult` field)
- Modify: `davinci_monet/ai/openrouter.py` (`_fetch_credits_remaining`, set on result)
- Test: `davinci_monet/tests/unit/ai/test_openrouter.py` (add + update one existing test)

- [ ] **Step 1: Write the failing tests**

Append to `davinci_monet/tests/unit/ai/test_openrouter.py`:

```python
def test_fetch_credits_remaining_parses(monkeypatch, tmp_path: Path) -> None:
    class _Resp:
        status_code = 200

        def json(self):
            return {"data": {"limit_remaining": 99.97}}

    monkeypatch.setattr("httpx.get", lambda *a, **k: _Resp())
    cfg = SummaryConfig.model_validate({"provider": "openrouter"})
    assert orouter._fetch_credits_remaining(cfg, "sk-or-test") == 99.97


def test_fetch_credits_remaining_none_on_non_200(monkeypatch) -> None:
    class _Resp:
        status_code = 402

        def json(self):
            return {}

    monkeypatch.setattr("httpx.get", lambda *a, **k: _Resp())
    cfg = SummaryConfig.model_validate({"provider": "openrouter"})
    assert orouter._fetch_credits_remaining(cfg, "k") is None


def test_fetch_credits_remaining_none_on_missing_field(monkeypatch) -> None:
    class _Resp:
        status_code = 200

        def json(self):
            return {"data": {}}

    monkeypatch.setattr("httpx.get", lambda *a, **k: _Resp())
    cfg = SummaryConfig.model_validate({"provider": "openrouter"})
    assert orouter._fetch_credits_remaining(cfg, "k") is None


def test_fetch_credits_remaining_none_on_error(monkeypatch) -> None:
    def _boom(*a, **k):
        raise RuntimeError("network down")

    monkeypatch.setattr("httpx.get", _boom)
    cfg = SummaryConfig.model_validate({"provider": "openrouter"})
    assert orouter._fetch_credits_remaining(cfg, "k") is None


def test_call_openrouter_sets_credits(monkeypatch, tmp_path: Path) -> None:
    keyfile = tmp_path / "k.api"
    keyfile.write_text("sk-or-test")
    cfg = SummaryConfig.model_validate(
        {"provider": "openrouter", "api_key_file": str(keyfile)}
    )
    monkeypatch.setattr(orouter, "_send_openrouter_request", lambda c, k, b: _canned())
    monkeypatch.setattr(orouter, "_fetch_credits_remaining", lambda c, k: 42.0)

    result = call_openrouter("SYS", "USER", [], cfg)
    assert result.credits_remaining == 42.0
```

Also update the existing `test_call_openrouter_maps_response` so it does not hit the network for credits: after its `monkeypatch.setattr(orouter, "_send_openrouter_request", _fake_send)` line, add:
```python
    monkeypatch.setattr(orouter, "_fetch_credits_remaining", lambda c, k: None)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/unit/ai/test_openrouter.py -v`
Expected: FAIL — `_fetch_credits_remaining` does not exist / `SummaryResult` has no `credits_remaining`.

- [ ] **Step 3: Add the `SummaryResult` field**

In `davinci_monet/ai/summarizer.py`, add a field to the `SummaryResult` dataclass (after `images_sent`):

```python
@dataclass
class SummaryResult:
    """Result of a successful summary generation."""

    markdown: str
    model: str
    usage: dict[str, Any]
    plots_used: list[str]
    images_sent: int
    credits_remaining: float | None = None
```

- [ ] **Step 4: Add the credit fetch in the OpenRouter provider**

In `davinci_monet/ai/openrouter.py`, add the key endpoint constant next to `OPENROUTER_URL`:

```python
OPENROUTER_KEY_URL = "https://openrouter.ai/api/v1/key"
```

Add this function (e.g. after `_send_openrouter_request`):

```python
def _fetch_credits_remaining(cfg: Any, key: str) -> float | None:
    """Best-effort remaining OpenRouter key credit ($). Never raises.

    Returns ``data.limit_remaining`` from GET /api/v1/key, or None on any error,
    non-200 response, or missing/null field. Credits are informational only and
    must never affect the summary.
    """
    import httpx

    try:
        resp = httpx.get(
            OPENROUTER_KEY_URL,
            headers={"Authorization": f"Bearer {key}"},
            timeout=15,
        )
        if resp.status_code != 200:
            return None
        remaining = (resp.json().get("data") or {}).get("limit_remaining")
        return float(remaining) if remaining is not None else None
    except Exception:  # noqa: BLE001 - credits are best-effort; never fail the summary
        return None
```

Then set it on the result in `call_openrouter` — change the `return SummaryResult(...)` to include `credits_remaining`:

```python
    return SummaryResult(
        markdown=markdown,
        model=model,
        usage=usage,
        plots_used=[caption for caption, _ in encoded_images],
        images_sent=len(encoded_images),
        credits_remaining=_fetch_credits_remaining(cfg, key),
    )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/unit/ai/test_openrouter.py davinci_monet/tests/unit/ai/test_generate_summary.py -v`
Expected: PASS (new credit tests + existing call/dispatch tests).

- [ ] **Step 6: Commit**

```bash
git add davinci_monet/ai/summarizer.py davinci_monet/ai/openrouter.py davinci_monet/tests/unit/ai/test_openrouter.py
git commit -m "feat(ai): fetch OpenRouter credits_remaining (non-fatal) onto SummaryResult"
```

---

## Task 3: Carry credits through the summary stage

**Files:**
- Modify: `davinci_monet/pipeline/stages.py` (`SummaryStage` data dict)
- Test: `davinci_monet/tests/unit/pipeline/test_summary_stage.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `davinci_monet/tests/unit/pipeline/test_summary_stage.py`:

```python
def test_summary_stage_includes_credits_key(monkeypatch, tmp_path: Path) -> None:
    def _fake_client(cfg):
        class _Msgs:
            def create(self, **kwargs):
                class _Block:
                    text = "## Caveats\n- only\n"

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
    # field is wired through; Anthropic path leaves it None
    assert "credits_remaining" in result.data
    assert result.data["credits_remaining"] is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/unit/pipeline/test_summary_stage.py::test_summary_stage_includes_credits_key -v`
Expected: FAIL — `KeyError: 'credits_remaining'`.

- [ ] **Step 3: Add `credits_remaining` to the stage result data**

In `davinci_monet/pipeline/stages.py`, in `SummaryStage.execute`'s success-path `data={...}` dict, add the key (next to `usage`):

```python
            data={
                "summary_file": str(out_path),
                "markdown": result.markdown,
                "bullets": extract_bullets(result.markdown),
                "model": result.model,
                "usage": result.usage,
                "credits_remaining": result.credits_remaining,
                "images_sent": result.images_sent,
            },
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/unit/pipeline/test_summary_stage.py -v`
Expected: PASS (all stage tests).

- [ ] **Step 5: Commit**

```bash
git add davinci_monet/pipeline/stages.py davinci_monet/tests/unit/pipeline/test_summary_stage.py
git commit -m "feat(pipeline): carry credits_remaining in summary stage data"
```

---

## Task 4: Render tokens + credits in the panel

**Files:**
- Modify: `davinci_monet/pipeline/runner.py` (`print_summary` ~line 513; end-of-run call ~line 1707)
- Test: `davinci_monet/tests/unit/pipeline/test_summary_display.py` (append)

- [ ] **Step 1: Write the failing tests**

Append to `davinci_monet/tests/unit/pipeline/test_summary_display.py`:

```python
def test_print_summary_shows_tokens() -> None:
    fmt, buf = _formatter_with_buffer(show_output=True)
    fmt.print_summary(
        ["x"], "/out/AI_summary.md", usage={"input_tokens": 1240, "output_tokens": 480}
    )
    out = buf.getvalue()
    assert "1,240 in" in out
    assert "480 out" in out
    assert "1,720 total" in out


def test_print_summary_shows_credits() -> None:
    fmt, buf = _formatter_with_buffer(show_output=True)
    fmt.print_summary(["x"], None, usage=None, credits_remaining=99.97)
    assert "$99.97 remaining" in buf.getvalue()


def test_print_summary_omits_credits_when_none() -> None:
    fmt, buf = _formatter_with_buffer(show_output=True)
    fmt.print_summary(
        ["x"], None, usage={"input_tokens": 1, "output_tokens": 2}, credits_remaining=None
    )
    assert "credits" not in buf.getvalue().lower()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/unit/pipeline/test_summary_display.py -v`
Expected: FAIL — `print_summary` does not accept `usage`/`credits_remaining`.

- [ ] **Step 3: Replace `print_summary` with the metadata-aware version**

In `davinci_monet/pipeline/runner.py`, replace the entire `print_summary` method with:

```python
    def print_summary(
        self,
        items: list[str],
        summary_file: str | None = None,
        usage: dict[str, Any] | None = None,
        credits_remaining: float | None = None,
    ) -> None:
        """Render an itemized AI summary to the terminal at end of run.

        Shows the condensed bullet list, then (dim) tokens used, OpenRouter
        credits remaining (when available), and a pointer to the full-brief
        file. No-op when output is disabled or there are no items.
        """
        if not self.show_output or not items:
            return
        from rich.panel import Panel

        body_lines = [f"• {item}" for item in items]
        meta: list[str] = []
        if (
            usage
            and usage.get("input_tokens") is not None
            and usage.get("output_tokens") is not None
        ):
            tin = int(usage["input_tokens"])
            tout = int(usage["output_tokens"])
            meta.append(f"Tokens: {tin:,} in / {tout:,} out ({tin + tout:,} total)")
        if credits_remaining is not None:
            meta.append(f"OpenRouter credits: ${credits_remaining:.2f} remaining")
        if summary_file:
            meta.append(f"Full brief → {summary_file}")
        if meta:
            body_lines.append("")
            body_lines.extend(f"[dim]{line}[/dim]" for line in meta)

        self._print()
        self._print(
            Panel(
                "\n".join(body_lines),
                title="AI Summary",
                border_style=self.NCAR_AQUA,
                padding=(1, 2),
            )
        )
        self._print()
```

- [ ] **Step 4: Update the runner end-of-run call**

In `davinci_monet/pipeline/runner.py`, find the end-of-run summary block that calls `formatter.print_summary(summary_result.data["bullets"], ...)` and update it to pass usage + credits:

```python
                formatter.print_summary(
                    summary_result.data["bullets"],
                    summary_result.data.get("summary_file"),
                    usage=summary_result.data.get("usage"),
                    credits_remaining=summary_result.data.get("credits_remaining"),
                )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/unit/pipeline/test_summary_display.py -v`
Expected: PASS (new tokens/credits tests + the existing list/silent/noop tests, which call `print_summary` with the original 2 args and still work).

- [ ] **Step 6: Commit**

```bash
git add davinci_monet/pipeline/runner.py davinci_monet/tests/unit/pipeline/test_summary_display.py
git commit -m "feat(display): show tokens used + OpenRouter credits in summary panel"
```

---

## Task 5: Integration — OpenRouter run shows tokens + credits

**Files:**
- Modify: `davinci_monet/tests/integration/test_ai_summary_openrouter_pipeline.py` (append)

- [ ] **Step 1: Write the integration test**

Append to `davinci_monet/tests/integration/test_ai_summary_openrouter_pipeline.py`:

```python
def test_openrouter_summary_displays_tokens_and_credits(monkeypatch, tmp_path: Path) -> None:
    import davinci_monet.pipeline.runner as runner_mod
    from davinci_monet.pipeline.runner import PipelineRunner

    def _fake_send(cfg, key, body):
        return {
            "model": body["model"],
            "choices": [{"message": {"content": "## Caveats\n- only point\n"}}],
            "usage": {"prompt_tokens": 123, "completion_tokens": 45},
        }

    monkeypatch.setattr(orouter, "_send_openrouter_request", _fake_send)
    monkeypatch.setattr(orouter, "_fetch_credits_remaining", lambda cfg, key: 88.5)

    captured: list[dict] = []
    monkeypatch.setattr(
        runner_mod.ProgressFormatter,
        "print_summary",
        lambda self, items, summary_file=None, usage=None, credits_remaining=None: captured.append(
            {"items": items, "usage": usage, "credits_remaining": credits_remaining}
        ),
    )

    keyfile = tmp_path / "OpenRouter.api"
    keyfile.write_text("sk-or-fake")
    config = _build_config(tmp_path)
    config["summary"] = {
        "enabled": True,
        "provider": "openrouter",
        "api_key_file": str(keyfile),
    }

    runner = PipelineRunner(show_progress=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", LegacyConfigWarning)
        result = runner.run_from_config(config)

    assert result.success
    assert captured, "summary was not displayed"
    call = captured[0]
    assert call["usage"] == {"input_tokens": 123, "output_tokens": 45}
    assert call["credits_remaining"] == 88.5
    # full brief still on disk
    assert "## Caveats" in (tmp_path / "output" / "AI_summary.md").read_text()
```

- [ ] **Step 2: Run the integration test**

Run: `HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/integration/test_ai_summary_openrouter_pipeline.py -v`
Expected: PASS (all tests in the file). Do not weaken assertions to pass; investigate failures via the pipeline log under `tmp_path/logs`.

- [ ] **Step 3: Commit**

```bash
git add davinci_monet/tests/integration/test_ai_summary_openrouter_pipeline.py
git commit -m "test(ai): OpenRouter run displays tokens + credits, file stays full"
```

---

## Task 6: Full suite + gates

**Files:** none (verification only)

- [ ] **Step 1: Run the full suite**

Run:
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate davinci
DASK_NUM_WORKERS=1 HDF5_USE_FILE_LOCKING=FALSE python -m pytest -q -p no:cacheprovider
```
Expected: all tests pass (previous count + the new ones), 1 skipped is fine.

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
git commit -m "style: black/isort/mypy fixes for token usage display"
```
(Skip if nothing changed.)

---

## Self-Review Notes (for the implementer)

- **Spec coverage:** emphasis strip (T1), `credits_remaining` field + non-fatal `_fetch_credits_remaining` + call_openrouter wiring (T2), credits through the stage (T3), `print_summary` tokens+credits + runner call (T4), OpenRouter integration showing both (T5), suite + gates (T6).
- **No network in tests:** `_fetch_credits_remaining` is stubbed (or its `httpx.get` is monkeypatched); the existing `test_call_openrouter_maps_response` is updated to stub it too so it never calls the real endpoint.
- **Backward-compatible signature:** `print_summary` gains `usage`/`credits_remaining` as optional kwargs; existing 2-arg test calls still pass.
- **Provider scope:** the Anthropic path leaves `credits_remaining=None`, so the credits line is omitted there; tokens line shows for both.
- **`AI_summary.md` and the prompt are unchanged** — only display + an additive result field + a best-effort credit query.
