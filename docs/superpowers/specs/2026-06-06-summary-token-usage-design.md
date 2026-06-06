# Summary Token Usage + OpenRouter Credits Display — Design Spec

**Date:** 2026-06-06
**Status:** Approved design, ready for implementation plan
**Branch:** develop

## Context

The AI summary stage already captures token usage from the API response
(`SummaryResult.usage = {input_tokens, output_tokens}`) and, after the itemized
display feature, renders a condensed "AI Summary" panel in the terminal. This
change adds two lines to that panel: tokens used for the call, and (for the
OpenRouter provider) the remaining account credit. It also cleans up literal
markdown emphasis (`**bold**`) that currently leaks into the terminal bullets.

### Decisions locked during brainstorming

| Decision | Choice |
|----------|--------|
| "Remaining" meaning | **OpenRouter credit balance** ($), not token-budget |
| Credits source | `GET https://openrouter.ai/api/v1/key` → `limit_remaining` (this key's remaining spend) |
| Token format | `<in> in / <out> out (<total> total)` |
| Provider scope | Tokens line: both providers. Credits line: **OpenRouter only** |
| Failure mode | Credits query is **non-fatal**: on error/null, omit the credits line; never fail the summary |
| Bundled polish | Strip markdown emphasis (`**`, `*`, `__`, `_`, `` ` ``) from terminal bullets |

Live-verified `GET /api/v1/key` response shape:
```json
{"data": {"limit": 100, "limit_remaining": 99.97, "usage": 0.0288, "is_free_tier": false, ...}}
```

## Goals

- Terminal "AI Summary" panel shows: bullets → `Tokens: …` → (OpenRouter)
  `OpenRouter credits: $… remaining` → `Full brief → …`.
- Terminal bullets are free of literal markdown emphasis markers.
- `AI_summary.md` (full brief) is unchanged.
- Credits query is one extra non-fatal HTTP call, only on the OpenRouter path.

## Non-Goals

- No credits/balance display for the Anthropic provider (no equivalent endpoint).
- No new config options; no change to the prompt, `generate_summary` contract
  (beyond the additive `credits_remaining` field), or `AI_summary.md`.
- No token-budget tracking across runs.

## Architecture

Four additive changes:

1. **`SummaryResult`** (`davinci_monet/ai/summarizer.py`) gains an optional
   field:
   ```python
   credits_remaining: float | None = None
   ```
   Anthropic leaves it `None`; OpenRouter sets it when the credit query succeeds.

2. **OpenRouter credit query** (`davinci_monet/ai/openrouter.py`):
   - `OPENROUTER_KEY_URL = "https://openrouter.ai/api/v1/key"`.
   - `_fetch_credits_remaining(cfg, key) -> float | None` — injectable seam
     (mirrors `_send_openrouter_request`): `GET` the key endpoint, return
     `data["limit_remaining"]` as a float, or `None` on any error / missing /
     null field. **Never raises.**
   - `call_openrouter` calls it after a successful chat response and sets
     `SummaryResult.credits_remaining`.

3. **Bullet emphasis strip** (`davinci_monet/ai/summarizer.py`): in
   `extract_bullets`, strip markdown emphasis markers from each item via a small
   `_strip_md = re.compile(r"[*_`]")` → `_strip_md.sub("", item).strip()`. Applies
   to both the bullet path and the heading-fallback path. (The full brief in
   `AI_summary.md` keeps its markdown — only the terminal items are cleaned.)

4. **Display** (`davinci_monet/pipeline/runner.py`): extend
   `ProgressFormatter.print_summary` signature to
   `(items, summary_file=None, usage=None, credits_remaining=None)`. After the
   bullets it appends:
   - `Tokens: {in:,} in / {out:,} out ({total:,} total)` when `usage` has both
     token counts (uses thousands separators).
   - `OpenRouter credits: ${credits_remaining:.2f} remaining` when
     `credits_remaining is not None`.
   - then the existing `Full brief → {summary_file}`.
   The `SummaryStage` already stores `usage`; it also stores
   `credits_remaining` (from the result). The runner passes both through.

`runner.py` stays free of `ai` imports — it only formats values from the
precomputed `data` dict.

## Data Flow

```
call_openrouter(...)                                   [ai/openrouter.py]
   data = _send_openrouter_request(...)                # chat completion
   result.usage = {input_tokens, output_tokens}
   result.credits_remaining = _fetch_credits_remaining(cfg, key)   # NEW, non-fatal
        │
SummaryStage.execute                                   [pipeline/stages.py]
   data["usage"]              = result.usage
   data["credits_remaining"]  = result.credits_remaining           # NEW
   data["bullets"]            = extract_bullets(result.markdown)    # now emphasis-stripped
        │
runner end-of-run                                      [pipeline/runner.py]
   formatter.print_summary(
       data["bullets"], data["summary_file"],
       usage=data["usage"], credits_remaining=data["credits_remaining"])
        │
ProgressFormatter.print_summary
   Panel: • items… / Tokens: … / OpenRouter credits: $… / Full brief → …
```

## Display Format

OpenRouter provider:
```
╭─ AI Summary ───────────────────────────────╮
│  • Mean Bias +4.82 ppb — overpredicts ~12% │
│  • RMSE 5.66 ppb; R 0.849; IOA 0.755       │
│                                            │
│  Tokens: 1,240 in / 480 out (1,720 total)  │
│  OpenRouter credits: $99.97 remaining      │
│  Full brief → …/output/AI_summary.md       │
╰────────────────────────────────────────────╯
```
Anthropic provider: identical minus the `OpenRouter credits:` line.

## Error Handling

- `_fetch_credits_remaining` catches all exceptions and returns `None` (no
  network failure, non-200, missing/`null` `limit_remaining`, or parse error can
  affect the summary). The chat completion already succeeded by then.
- `print_summary` only renders the tokens line when both token counts are
  present, and the credits line only when `credits_remaining is not None`.
- The summary stage's non-fatal guarantee is unchanged.

## Testing

Per repo rules (integration through `run_from_config`; no network in the suite).

**Unit:**
- `extract_bullets`: emphasis markers stripped (`**MB +4.8**` → `MB +4.8`);
  existing bullet/heading-fallback/cap behavior still holds.
- `_fetch_credits_remaining`: with a stubbed HTTP getter returning the live JSON
  shape → returns `99.97`; returns `None` on non-200, on missing
  `limit_remaining`, and on raised exception (monkeypatch the getter to raise).
- `call_openrouter`: with `_send_openrouter_request` and
  `_fetch_credits_remaining` stubbed → `SummaryResult.credits_remaining` is set;
  with the credit fetch stubbed to `None` → result still returned (non-fatal).
- `SummaryStage` (mocked client): `data["credits_remaining"]` present;
  `data["usage"]` present.
- `ProgressFormatter.print_summary`: renders the `Tokens:` line from a usage
  dict; renders the `OpenRouter credits:` line when `credits_remaining` is set
  and omits it when `None`; still silent when `show_output=False` / no items.

**Integration (through `run_from_config`):**
- OpenRouter path (extending `test_ai_summary_openrouter_pipeline.py`): stub the
  chat send and the credit fetch → `print_summary` receives a `usage` dict and
  `credits_remaining` value (assert via the monkeypatched `print_summary`
  recording its kwargs); `AI_summary.md` still full.

## Files Touched (anticipated)

- **Modified:** `davinci_monet/ai/summarizer.py` (`SummaryResult.credits_remaining`,
  emphasis strip in `extract_bullets`), `davinci_monet/ai/openrouter.py`
  (`_fetch_credits_remaining`, set on result), `davinci_monet/pipeline/stages.py`
  (`data["credits_remaining"]`), `davinci_monet/pipeline/runner.py`
  (`print_summary` signature/body + end-of-run call).
- **Tests:** extend `tests/unit/ai/test_extract_bullets.py`,
  `tests/unit/ai/test_openrouter.py`, `tests/unit/pipeline/test_summary_display.py`,
  `tests/unit/pipeline/test_summary_stage.py`,
  `tests/integration/test_ai_summary_openrouter_pipeline.py`.

## Open Items for the Plan

- Confirm the current `print_summary(items, summary_file=None)` signature (added
  by the itemized feature) so the new `usage`/`credits_remaining` params are
  appended without breaking existing callers/tests.
- Confirm `httpx` import style used in `openrouter.py` (`_send_openrouter_request`
  imports it lazily inside the function — mirror that).
