# OpenRouter Provider for AI Summary — Design Spec

**Date:** 2026-06-05
**Status:** Approved design, ready for implementation plan
**Branch:** develop

## Context

The AI analysis summary feature (see
`2026-06-05-ai-analysis-summary-design.md`) calls the Anthropic API directly via
the `anthropic` SDK (Messages API, native image blocks, `sk-ant-…` keys). The
user's only available key is an **OpenRouter** key (`sk-or-…`), stored in the
gitignored `OpenRouter.api` file at the repo root. A live test confirmed the key
works and can route to Claude models, but OpenRouter speaks the **OpenAI Chat
Completions** format, which the `anthropic` SDK cannot target.

This feature adds OpenRouter as an alternative provider for the summary stage so
the user's existing key drives the feature, without disturbing the working
Anthropic path.

### Decisions locked during brainstorming

| Decision | Choice |
|----------|--------|
| Key source | **File path in config** (`api_key_file`), generalized to fall back to `api_key_env` |
| HTTP mechanism | **httpx directly** (already a dependency via `anthropic`; no new package) |
| Structure | **Provider dispatch**: shared prompt/encode/key-resolve in `summarizer.py`; new `ai/openrouter.py`; `generate_summary` branches on `cfg.provider` |
| Default provider | **`anthropic`** (preserves current behavior) |
| Default OpenRouter model | **`anthropic/claude-3.5-haiku`** (verified working in the live key test; cheapest) |
| Tests | All mocked (no network in the suite) + one **manual** live end-to-end run at the end |

## Goals

- Let `summary.provider: openrouter` drive the summary stage with the user's
  OpenRouter key read from `OpenRouter.api`.
- Preserve the Anthropic path and every existing test unchanged.
- Keep vision working (plot images attached) over OpenRouter's OpenAI-format
  `image_url` data-URL blocks.
- Keep the non-fatal guarantee: any provider/key/HTTP failure → `SKIPPED`.
- No new third-party dependency; no network calls in the automated test suite.

## Non-Goals

- No `openai` SDK dependency (use httpx directly).
- No provider-strategy framework / plugin registry — two providers, simple
  dispatch.
- No configurable `base_url`, OpenRouter ranking headers (`HTTP-Referer`,
  `X-Title`), streaming, or per-provider retry policy (YAGNI).
- No change to `SummaryStage`, `collect_payload`, `images.py`, or the
  `SummaryResult` shape.

## Architecture

Additive provider dispatch. Existing modules and their roles are unchanged
except `summarizer.py` (gains key resolution + dispatch) and `config/schema.py`
(`SummaryConfig` gains two fields). One new module: `ai/openrouter.py`.

```
SummaryStage ── generate_summary(payload, cfg, client=None)   [summarizer.py]
                     │  encode images (shared, ai/images.py)
                     │  dispatch on cfg.provider
        ┌────────────┴───────────────┐
        ▼                            ▼
 _call_anthropic(...)        call_openrouter(...)        [ai/openrouter.py]
 anthropic SDK Messages      httpx POST OpenAI Chat
 (existing path)             Completions (new path)
        │                            │
        └──────────► SummaryResult ◄─┘   (identical shape either way)
```

### Module responsibilities

- **`davinci_monet/config/schema.py`** — `SummaryConfig` gains `provider` and
  `api_key_file`, plus a validator for provider-aware defaults.
- **`davinci_monet/ai/summarizer.py`** — keeps `SummaryError`, `SummaryResult`,
  `SYSTEM_PROMPT`, `render_text`, `build_prompt` (Anthropic blocks), gains
  `resolve_api_key`, refactors the current Anthropic call body into
  `_call_anthropic`, and `generate_summary` becomes a dispatcher. `_build_client`
  uses `resolve_api_key`.
- **`davinci_monet/ai/openrouter.py`** (new) — `OPENROUTER_URL`,
  `build_openrouter_messages`, `_send_openrouter_request` (injectable seam),
  `call_openrouter`. Imports `SummaryError`, `SummaryResult`, `resolve_api_key`
  from `summarizer` (no circular import: `summarizer` imports `call_openrouter`
  lazily inside `generate_summary`).
- **`davinci_monet/ai/__init__.py`** — additionally export `call_openrouter` and
  `resolve_api_key`.

## Config Changes (`SummaryConfig`)

New fields (added to the existing model in `config/schema.py`):

| Field | Type | Default | Meaning |
|-------|------|---------|---------|
| `provider` | `Literal["anthropic","openrouter"]` | `"anthropic"` | Which backend to call |
| `api_key_file` | `str \| None` | `None` | Path to a file containing the API key; read (stripped) before `api_key_env` |

Existing fields unchanged: `enabled`, `model` (`"claude-haiku-4-5"`),
`max_tokens` (2000), `api_key_env` (`"ANTHROPIC_API_KEY"`), `plots`,
`max_images` (8), `output_filename` (`"AI_summary.md"`), `instructions`.

**Provider-aware default validator** (`model_validator(mode="after")`): when
`provider == "openrouter"` and the Anthropic-default sentinels are still in
place, flip them:
- if `model == "claude-haiku-4-5"` → `model = "anthropic/claude-3.5-haiku"`
- if `api_key_env == "ANTHROPIC_API_KEY"` → `api_key_env = "OPENROUTER_API_KEY"`

This keeps the existing `test_summary_config.py` default assertions
(`SummaryConfig()` with no provider) valid while giving OpenRouter sensible
defaults. Explicit user values for `model`/`api_key_env` are never overridden.

## Key Resolution (`resolve_api_key`)

Shared by both providers (in `summarizer.py`):

```
def resolve_api_key(cfg) -> str:
    if cfg.api_key_file:
        path = Path(os.path.expanduser(cfg.api_key_file))
        try:
            key = path.read_text().strip()
        except OSError as exc:
            raise SummaryError(f"could not read api_key_file '{cfg.api_key_file}': {exc}")
        if not key:
            raise SummaryError(f"api_key_file '{cfg.api_key_file}' is empty")
        return key
    key = os.environ.get(cfg.api_key_env, "")
    if not key:
        raise SummaryError(
            f"API key not found: set env '{cfg.api_key_env}' or summary.api_key_file"
        )
    return key
```

- File path is resolved relative to the process working directory (the analysis
  is run from the repo root, where `OpenRouter.api` lives) unless absolute;
  `~` is expanded.
- `_build_client` (Anthropic) switches from `os.environ.get(...)` to
  `resolve_api_key(cfg)`, so the Anthropic path also gains file-key support. Its
  existing missing-key test still raises `SummaryError`.

## OpenRouter Provider (`ai/openrouter.py`)

```
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

def build_openrouter_messages(system_text, user_text, encoded_images):
    user_content = [{"type": "text", "text": user_text}]
    for caption, enc in encoded_images:
        user_content.append({"type": "text", "text": f"Figure: {caption}"})
        user_content.append({
            "type": "image_url",
            "image_url": {"url": f"data:{enc.media_type};base64,{enc.data}"},
        })
    return [
        {"role": "system", "content": system_text},
        {"role": "user", "content": user_content},
    ]

def _send_openrouter_request(cfg, key, body) -> dict:   # injectable seam
    import httpx
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    try:
        resp = httpx.post(OPENROUTER_URL, headers=headers, json=body, timeout=60)
    except Exception as exc:
        raise SummaryError(f"OpenRouter request failed: {exc}") from exc
    if resp.status_code != 200:
        raise SummaryError(f"OpenRouter HTTP {resp.status_code}: {resp.text[:300]}")
    return resp.json()

def call_openrouter(system_text, user_text, encoded_images, cfg) -> SummaryResult:
    key = resolve_api_key(cfg)
    body = {
        "model": cfg.model,
        "messages": build_openrouter_messages(system_text, user_text, encoded_images),
        "max_tokens": cfg.max_tokens,
    }
    data = _send_openrouter_request(cfg, key, body)
    try:
        markdown = data["choices"][0]["message"]["content"]
        usage_raw = data.get("usage") or {}
        usage = {
            "input_tokens": usage_raw.get("prompt_tokens"),
            "output_tokens": usage_raw.get("completion_tokens"),
        }
        model = data.get("model", cfg.model)
    except (KeyError, IndexError, TypeError) as exc:
        raise SummaryError(f"Unexpected OpenRouter response shape: {exc}") from exc
    return SummaryResult(
        markdown=markdown,
        model=model,
        usage=usage,
        plots_used=[caption for caption, _ in encoded_images],
        images_sent=len(encoded_images),
    )
```

The system prompt is sent as a plain `system` message (no `cache_control`;
OpenRouter's OpenAI-format request does not use Anthropic cache-control blocks).

## Dispatch (`summarizer.generate_summary`)

`generate_summary` keeps its signature `(payload, *, cfg, client=None)`. It
encodes images once (shared), then dispatches:

```
provider = getattr(cfg, "provider", "anthropic")
if provider == "openrouter":
    from davinci_monet.ai.openrouter import call_openrouter   # lazy: avoids cycle
    return call_openrouter(SYSTEM_PROMPT, render_text(payload), encoded, cfg)
return _call_anthropic(payload, encoded, cfg, client=client)
```

`_call_anthropic` is the current Anthropic body (`build_prompt` →
`_build_client`/injected `client` → `messages.create` → parse). The injectable
`client` param remains Anthropic-specific (used by existing tests). The
OpenRouter path's injectable seam is `_send_openrouter_request`.

## Error Handling

Unchanged guarantee. Every failure mode raises `SummaryError`, which
`SummaryStage.execute`'s broad `except Exception` converts to `SKIPPED` (run
still succeeds):

| Condition | Result |
|-----------|--------|
| `api_key_file` missing/unreadable/empty | `SummaryError` → SKIPPED |
| Neither file nor env key present | `SummaryError` → SKIPPED |
| OpenRouter network error / non-200 | `SummaryError` → SKIPPED |
| Malformed OpenRouter response | `SummaryError` → SKIPPED |

## Testing

Per repo rules (integration through `PipelineRunner.run_from_config`; no
network in the suite; no shortcuts to green).

**Unit:**
- `resolve_api_key`: reads a tmp file (stripped), errors on missing file, errors
  on empty file, falls back to env var, errors when neither present.
- `SummaryConfig`: `provider` defaults to `"anthropic"`; `SummaryConfig()`
  defaults unchanged; `provider="openrouter"` flips `model` →
  `anthropic/claude-3.5-haiku` and `api_key_env` → `OPENROUTER_API_KEY`; explicit
  `model`/`api_key_env` are preserved.
- `build_openrouter_messages`: system message present; user content has one text
  block then `image_url` data-URL blocks (`data:image/png;base64,…`) matching the
  encoded images; caption text precedes each image.
- `call_openrouter`: with `_send_openrouter_request` monkeypatched to return
  canned JSON, returns a `SummaryResult` with mapped usage
  (`prompt_tokens`→`input_tokens`, `completion_tokens`→`output_tokens`),
  `images_sent`, and model; raises `SummaryError` on a malformed dict.
- `generate_summary` dispatch: `provider="openrouter"` calls the OpenRouter path
  (monkeypatch `_send_openrouter_request`); `provider="anthropic"` (default)
  still uses the injected Anthropic `client`.

**Integration (through `run_from_config`):**
- Synthetic-data config with `summary: {enabled: true, provider: openrouter,
  api_key_file: <tmp key file>}`, `_send_openrouter_request` monkeypatched to
  return canned JSON → `AI_summary.md` written into `output_dir`, run succeeds,
  stage `COMPLETED`.
- `provider: openrouter` with a non-existent `api_key_file` and no env key → run
  still succeeds, stage `SKIPPED`, no file written.

**Manual (not in the suite):** after implementation, run a real analysis with
`provider: openrouter`, `api_key_file: OpenRouter.api`, and `model:
anthropic/claude-3.5-haiku` against the user's real key to confirm a real
`AI_summary.md` is produced end-to-end.

## Files Touched (anticipated)

- **New:** `davinci_monet/ai/openrouter.py`,
  `davinci_monet/tests/unit/ai/test_openrouter.py`,
  `davinci_monet/tests/unit/ai/test_resolve_api_key.py`,
  `davinci_monet/tests/integration/test_ai_summary_openrouter_pipeline.py`.
- **Modified:** `davinci_monet/config/schema.py` (`SummaryConfig`),
  `davinci_monet/ai/summarizer.py` (`resolve_api_key`, `_call_anthropic`,
  dispatch, `_build_client`), `davinci_monet/ai/__init__.py` (exports),
  `davinci_monet/tests/unit/config/test_summary_config.py` (provider cases),
  README + CLAUDE.md (`provider`/`api_key_file` docs).

## Open Items for the Plan

- Confirm `Literal` import availability and existing validator style in
  `config/schema.py` (use `field_validator`/`model_validator` consistent with the
  file).
- Confirm `httpx` is importable in the `davinci` env (it is, via `anthropic`).
- Decide exact OpenRouter integration test location to match the existing
  `test_ai_summary_pipeline.py` layout and synthetic-config helper.
