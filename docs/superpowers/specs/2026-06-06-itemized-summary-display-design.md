# Itemized AI Summary Display — Design Spec

**Date:** 2026-06-06
**Status:** Approved design, ready for implementation plan
**Branch:** develop

## Context

The AI summary stage writes the full brief to `AI_summary.md` and, after the
prior display fix (`66cf4ca`), renders the **full** markdown brief in an "AI
Summary" panel in the terminal at end of run. The full brief is long for a
terminal readout. This change makes the terminal show a concise **itemized
bullet list** derived from the brief, while `AI_summary.md` keeps the **full**
brief unchanged.

### Decisions locked during brainstorming

| Decision | Choice |
|----------|--------|
| List source | **Derive from the full brief** (no prompt change; keep the single full-brief prompt) |
| List content | **All bullet points** across all sections, heading-independent |
| Full brief | `AI_summary.md` keeps the **full** brief (unchanged) |
| No-bullets fallback | Show the `##` section headings as the items |
| Cap | ~12 items; overflow appends `… (full brief in AI_summary.md)` |

## Goals

- Terminal end-of-run display shows a short itemized list (bullets) plus a
  pointer to the full brief file.
- `AI_summary.md` is unchanged (full brief).
- The prompt and `generate_summary` are unchanged (derive, don't re-ask).
- Robust to brief variation (extract bullets regardless of heading wording;
  fall back to headings when there are no bullets).

## Non-Goals

- No prompt/system-message change; no second dataset call.
- No change to `AI_summary.md` contents or the OpenRouter/Anthropic paths.
- No new config options (the cap and fallback are fixed, sensible defaults).

## Architecture

Three small changes, each independently testable:

1. **Pure extractor** in `davinci_monet/ai/summarizer.py`:

   ```
   _BULLET_RE = re.compile(r"^\s*[-*•]\s+(.*\S)\s*$")
   _HEADING_RE = re.compile(r"^\s*#{1,6}\s+(.*\S)\s*$")

   def extract_bullets(markdown: str, *, max_items: int = 12) -> list[str]:
       """Condense a brief to an itemized list for terminal display.

       Returns every markdown bullet line (-, *, •) across all sections with the
       marker stripped. If the brief has no bullets, falls back to the section
       headings. Caps at max_items; when truncated, the final item is
       "… (full brief in AI_summary.md)".
       """
   ```
   - Bullets matched by `_BULLET_RE`; marker stripped, whitespace trimmed,
     blank items skipped.
   - If no bullets found, collect `_HEADING_RE` matches (heading text) as items.
   - If neither found, return `[]` (display shows the file pointer only).
   - Truncation: if more than `max_items` raw items, keep the first
     `max_items - 1` and append `"… (full brief in AI_summary.md)"`.

2. **`SummaryStage`** (`davinci_monet/pipeline/stages.py`): on success, add
   `"bullets": extract_bullets(result.markdown)` to `StageResult.data` (next to
   the existing `summary_file`, `markdown`, `dataset`, `usage`, `images_sent`).
   The `out_path.write_text(result.markdown)` (full brief) is unchanged.

3. **Display** (`davinci_monet/pipeline/runner.py`):
   - `ProgressFormatter.print_summary` changes signature from
     `(markdown: str)` to `(items: list[str], summary_file: str | None)`. It
     renders a panel titled "AI Summary" containing the items as a bullet list,
     then a dim footer line `Full brief → <summary_file>` (omitted if
     `summary_file` is None or no items). No-op when `show_output` is False.
   - The runner's end-of-run block passes
     `summary_result.data.get("bullets")` and
     `summary_result.data.get("summary_file")` instead of the markdown. Guard:
     only call when the summary stage COMPLETED and `bullets` is a non-empty
     list (else skip display; the file is still written).

This keeps `runner.py` free of any `ai` import — extraction happens in the stage
(which already imports `ai`), and the runner only renders the precomputed list.

## Data Flow

```
SummaryStage.execute
   result = generate_summary(...)              # full brief markdown (unchanged)
   out_path.write_text(result.markdown)        # full brief -> AI_summary.md (unchanged)
   data["markdown"]  = result.markdown
   data["bullets"]   = extract_bullets(result.markdown)   # NEW
   data["summary_file"] = str(out_path)
        │
        ▼
runner finally (end of run)
   if summary COMPLETED and data["bullets"]:
       formatter.print_summary(data["bullets"], data["summary_file"])
        │
        ▼
ProgressFormatter.print_summary(items, summary_file)
   Panel("AI Summary"): "• item" per item + "Full brief → <file>"
```

## Display Format

```
╭─ AI Summary ───────────────────────────────╮
│  • Mean Bias +4.82 ppb — overpredicts ~12% │
│  • RMSE 5.66 ppb; NME 12.4%                │
│  • R 0.849; IOA 0.755                      │
│  • N=98; synthetic data                    │
│                                            │
│  Full brief → …/output/AI_summary.md       │
╰────────────────────────────────────────────╯
```

## Error Handling

- The summary stage's non-fatal guarantee is unchanged. `extract_bullets` is a
  pure string operation that cannot raise on normal input; it runs inside the
  stage's existing `try` so any unexpected error still degrades to SKIPPED.
- If `bullets` is empty (brief had no bullets and no headings), the runner skips
  the terminal display; `AI_summary.md` is still written.

## Testing

Per repo rules (integration through `PipelineRunner.run_from_config`; no
network in the suite).

**Unit:**
- `extract_bullets`:
  - returns each bullet line (marker stripped) from a multi-section brief;
  - falls back to `##` heading text when there are no bullets;
  - returns `[]` when neither bullets nor headings exist;
  - caps at `max_items`, and the truncated final item is
    `"… (full brief in AI_summary.md)"`.
- `SummaryStage` (mocked client): `result.data["bullets"]` is the extracted list
  and `data["markdown"]` is still the full brief; `AI_summary.md` still holds the
  full brief.
- `ProgressFormatter.print_summary`: with output enabled and a captured console,
  renders each item and the `Full brief →` footer; silent when
  `show_output=False`.

**Integration (through `run_from_config`):**
- `show_progress=True`, summary enabled, client stubbed, with
  `ProgressFormatter.print_summary` monkeypatched to record its args → asserts it
  received the **bullets list** (not the full markdown) and the summary file
  path; and `AI_summary.md` on disk still contains the **full** brief.

## Files Touched (anticipated)

- **Modified:** `davinci_monet/ai/summarizer.py` (`extract_bullets`),
  `davinci_monet/pipeline/stages.py` (`data["bullets"]`),
  `davinci_monet/pipeline/runner.py` (`print_summary` signature/body + end-of-run
  call).
- **Tests:** extend `tests/unit/ai/` (extractor),
  `tests/unit/pipeline/test_summary_display.py` (print_summary),
  `tests/unit/pipeline/test_summary_stage.py` (bullets in data),
  `tests/integration/test_ai_summary_pipeline.py` (display receives bullets).

## Open Items for the Plan

- Confirm the existing `print_summary` unit tests (added in `66cf4ca`) are
  updated to the new `(items, summary_file)` signature.
- Confirm `re` is imported in `summarizer.py` (add if missing).
