# OpenRouter Provider Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `summary.provider: openrouter` so the AI summary stage can call OpenRouter (OpenAI Chat Completions format, vision via `image_url`) using a key read from a file, without disturbing the Anthropic path.

**Architecture:** Additive provider dispatch. `summarizer.py` gains a shared `resolve_api_key` and a `generate_summary` that branches on `cfg.provider`; a new `ai/openrouter.py` builds the OpenAI-format request and POSTs via httpx. `SummaryResult` and `SummaryStage` are unchanged.

**Tech Stack:** Python 3.11+, Pydantic v2, httpx (already a dependency via `anthropic`), pytest.

**Spec:** `docs/superpowers/specs/2026-06-05-openrouter-provider-design.md`

**Environment:** Run all tests in the `davinci` conda env:
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate davinci
HDF5_USE_FILE_LOCKING=FALSE python -m pytest <path> -v
```
Note: the **full** suite needs `DASK_NUM_WORKERS=1 HDF5_USE_FILE_LOCKING=FALSE` to avoid the known HDF5 segfault (CLAUDE.md gotcha #8). Per-file test runs in this plan do not.

---

## Task 1: `SummaryConfig` gains `provider` + `api_key_file`

**Files:**
- Modify: `davinci_monet/config/schema.py` (`SummaryConfig`, ~line 677)
- Test: `davinci_monet/tests/unit/config/test_summary_config.py` (append cases)

Context: `SummaryConfig` is a `FlexibleModel` (extra allowed, `validate_default=True`). `Literal` and `model_validator` are already imported at the top of `schema.py`.

- [ ] **Step 1: Write the failing tests**

Append to `davinci_monet/tests/unit/config/test_summary_config.py`:

```python
def test_summary_config_provider_defaults_to_anthropic() -> None:
    cfg = SummaryConfig()
    assert cfg.provider == "anthropic"
    assert cfg.api_key_file is None
    # anthropic defaults are untouched
    assert cfg.model == "claude-haiku-4-5"
    assert cfg.api_key_env == "ANTHROPIC_API_KEY"


def test_summary_config_openrouter_flips_defaults() -> None:
    cfg = SummaryConfig.model_validate({"provider": "openrouter"})
    assert cfg.provider == "openrouter"
    # sentinels flip to OpenRouter-appropriate defaults
    assert cfg.model == "anthropic/claude-3.5-haiku"
    assert cfg.api_key_env == "OPENROUTER_API_KEY"


def test_summary_config_openrouter_preserves_explicit_values() -> None:
    cfg = SummaryConfig.model_validate(
        {
            "provider": "openrouter",
            "model": "anthropic/claude-sonnet-4",
            "api_key_env": "MY_KEY",
            "api_key_file": "OpenRouter.api",
        }
    )
    assert cfg.model == "anthropic/claude-sonnet-4"
    assert cfg.api_key_env == "MY_KEY"
    assert cfg.api_key_file == "OpenRouter.api"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/unit/config/test_summary_config.py -v`
Expected: FAIL — `SummaryConfig` has no attribute `provider`.

- [ ] **Step 3: Add the fields + validator**

In `davinci_monet/config/schema.py`, edit `SummaryConfig` to add two fields and a validator. The class becomes:

```python
class SummaryConfig(FlexibleModel):
    """Configuration for the optional AI analysis summary stage.

    When ``enabled`` is true, a final pipeline stage sends the run's
    statistics, config metadata, and selected plot images to a Claude model
    (via the Anthropic API directly, or via OpenRouter) and writes a markdown
    brief into the analysis output directory.
    """

    enabled: bool = False
    provider: Literal["anthropic", "openrouter"] = "anthropic"
    model: str = "claude-haiku-4-5"
    max_tokens: int = 2000
    api_key_env: str = "ANTHROPIC_API_KEY"
    api_key_file: str | None = None
    plots: list[str] | None = None
    max_images: int = 8
    output_filename: str = "AI_summary.md"
    instructions: str | None = None

    @model_validator(mode="after")
    def _apply_provider_defaults(self) -> "SummaryConfig":
        """Flip Anthropic-default sentinels to OpenRouter equivalents.

        Only fields still holding the Anthropic default are changed, so an
        explicit user value is never overridden.
        """
        if self.provider == "openrouter":
            if self.model == "claude-haiku-4-5":
                self.model = "anthropic/claude-3.5-haiku"
            if self.api_key_env == "ANTHROPIC_API_KEY":
                self.api_key_env = "OPENROUTER_API_KEY"
        return self
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/unit/config/test_summary_config.py -v`
Expected: PASS (all, including the pre-existing default tests).

- [ ] **Step 5: Commit**

```bash
git add davinci_monet/config/schema.py davinci_monet/tests/unit/config/test_summary_config.py
git commit -m "feat(config): add summary provider + api_key_file"
```

---

## Task 2: `resolve_api_key` (shared key resolution)

**Files:**
- Modify: `davinci_monet/ai/summarizer.py` (add `resolve_api_key`; use it in `_build_client`)
- Test: `davinci_monet/tests/unit/ai/test_resolve_api_key.py`

Context: `summarizer.py` already imports `logging`, `os`, and defines `SummaryError`. Add `from pathlib import Path` if not present (it is not currently imported — add it).

- [ ] **Step 1: Write the failing test**

Create `davinci_monet/tests/unit/ai/test_resolve_api_key.py`:

```python
"""Unit tests for ai.summarizer.resolve_api_key."""

from __future__ import annotations

from pathlib import Path

import pytest

from davinci_monet.ai.summarizer import SummaryError, resolve_api_key
from davinci_monet.config.schema import SummaryConfig


def test_resolve_from_file_stripped(tmp_path: Path) -> None:
    p = tmp_path / "key.api"
    p.write_text("sk-or-secret\n")
    cfg = SummaryConfig.model_validate({"provider": "openrouter", "api_key_file": str(p)})
    assert resolve_api_key(cfg) == "sk-or-secret"


def test_resolve_missing_file_raises(tmp_path: Path) -> None:
    cfg = SummaryConfig.model_validate(
        {"provider": "openrouter", "api_key_file": str(tmp_path / "nope.api")}
    )
    with pytest.raises(SummaryError, match="api_key_file"):
        resolve_api_key(cfg)


def test_resolve_empty_file_raises(tmp_path: Path) -> None:
    p = tmp_path / "empty.api"
    p.write_text("   \n")
    cfg = SummaryConfig.model_validate({"provider": "openrouter", "api_key_file": str(p)})
    with pytest.raises(SummaryError, match="empty"):
        resolve_api_key(cfg)


def test_resolve_from_env(monkeypatch) -> None:
    monkeypatch.setenv("MY_TEST_KEY", "env-secret")
    cfg = SummaryConfig.model_validate({"api_key_env": "MY_TEST_KEY"})
    assert resolve_api_key(cfg) == "env-secret"


def test_resolve_none_raises(monkeypatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    cfg = SummaryConfig()  # no file, default env unset
    with pytest.raises(SummaryError, match="API key not found"):
        resolve_api_key(cfg)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/unit/ai/test_resolve_api_key.py -v`
Expected: FAIL — `cannot import name 'resolve_api_key'`.

- [ ] **Step 3: Add `resolve_api_key` and rewire `_build_client`**

In `davinci_monet/ai/summarizer.py`, add `from pathlib import Path` to the imports (with the other stdlib imports). Add this function just above `_build_client`:

```python
def resolve_api_key(cfg: Any) -> str:
    """Resolve the API key from ``api_key_file`` (if set) else ``api_key_env``.

    Raises SummaryError if the file is unreadable/empty or no key is found.
    """
    if cfg.api_key_file:
        path = Path(os.path.expanduser(cfg.api_key_file))
        try:
            key = path.read_text().strip()
        except OSError as exc:
            raise SummaryError(
                f"could not read api_key_file '{cfg.api_key_file}': {exc}"
            ) from exc
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

Then change `_build_client` to use it. Replace its body's key lookup:

```python
def _build_client(cfg: Any) -> Any:
    """Construct a real Anthropic client (lazy import). Raises SummaryError."""
    try:
        import anthropic
    except ImportError as exc:  # pragma: no cover - exercised via stage tests
        raise SummaryError(
            "anthropic package not installed; install with: pip install 'davinci-monet[ai]'"
        ) from exc

    key = resolve_api_key(cfg)
    return anthropic.Anthropic(api_key=key)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/unit/ai/test_resolve_api_key.py davinci_monet/tests/unit/ai/test_generate_summary.py -v`
Expected: PASS (the existing `test_generate_summary_missing_key_raises` still passes — it relies on `_build_client` raising when the env key is absent).

- [ ] **Step 5: Commit**

```bash
git add davinci_monet/ai/summarizer.py davinci_monet/tests/unit/ai/test_resolve_api_key.py
git commit -m "feat(ai): shared resolve_api_key (file or env) used by Anthropic path"
```

---

## Task 3: `ai/openrouter.py` — OpenRouter provider

**Files:**
- Create: `davinci_monet/ai/openrouter.py`
- Test: `davinci_monet/tests/unit/ai/test_openrouter.py`

- [ ] **Step 1: Write the failing test**

Create `davinci_monet/tests/unit/ai/test_openrouter.py`:

```python
"""Unit tests for the OpenRouter provider (no network)."""

from __future__ import annotations

from pathlib import Path

import pytest

import davinci_monet.ai.openrouter as orouter
from davinci_monet.ai.images import EncodedImage
from davinci_monet.ai.openrouter import build_openrouter_messages, call_openrouter
from davinci_monet.ai.summarizer import SummaryError, SummaryResult
from davinci_monet.config.schema import SummaryConfig


def test_build_openrouter_messages_shape() -> None:
    encoded = [("01_o3_scatter", EncodedImage(media_type="image/png", data="QUJD"))]
    messages = build_openrouter_messages("SYS", "USER TEXT", encoded)
    assert messages[0] == {"role": "system", "content": "SYS"}
    user = messages[1]
    assert user["role"] == "user"
    assert user["content"][0] == {"type": "text", "text": "USER TEXT"}
    assert user["content"][1] == {"type": "text", "text": "Figure: 01_o3_scatter"}
    assert user["content"][2] == {
        "type": "image_url",
        "image_url": {"url": "data:image/png;base64,QUJD"},
    }


def _canned() -> dict:
    return {
        "model": "anthropic/claude-3.5-haiku",
        "choices": [
            {"message": {"role": "assistant", "content": "## What this run is\nx\n## Caveats\n"}}
        ],
        "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
    }


def test_call_openrouter_maps_response(monkeypatch, tmp_path: Path) -> None:
    keyfile = tmp_path / "k.api"
    keyfile.write_text("sk-or-test")
    cfg = SummaryConfig.model_validate(
        {"provider": "openrouter", "api_key_file": str(keyfile)}
    )

    captured = {}

    def _fake_send(cfg_arg, key, body):
        captured["key"] = key
        captured["body"] = body
        return _canned()

    monkeypatch.setattr(orouter, "_send_openrouter_request", _fake_send)

    encoded = [("fig", EncodedImage(media_type="image/png", data="QUJD"))]
    result = call_openrouter("SYS", "USER", encoded, cfg)

    assert isinstance(result, SummaryResult)
    assert "## Caveats" in result.markdown
    assert result.model == "anthropic/claude-3.5-haiku"
    assert result.usage == {"input_tokens": 100, "output_tokens": 50}
    assert result.images_sent == 1
    assert captured["key"] == "sk-or-test"
    assert captured["body"]["model"] == "anthropic/claude-3.5-haiku"
    assert captured["body"]["max_tokens"] == 2000


def test_call_openrouter_malformed_response_raises(monkeypatch, tmp_path: Path) -> None:
    keyfile = tmp_path / "k.api"
    keyfile.write_text("sk-or-test")
    cfg = SummaryConfig.model_validate(
        {"provider": "openrouter", "api_key_file": str(keyfile)}
    )
    monkeypatch.setattr(orouter, "_send_openrouter_request", lambda c, k, b: {"oops": 1})

    with pytest.raises(SummaryError, match="Unexpected OpenRouter response shape"):
        call_openrouter("SYS", "USER", [], cfg)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/unit/ai/test_openrouter.py -v`
Expected: FAIL — `No module named 'davinci_monet.ai.openrouter'`.

- [ ] **Step 3: Implement the provider**

Create `davinci_monet/ai/openrouter.py`:

```python
"""OpenRouter provider for the AI summary (OpenAI Chat Completions format).

Builds an OpenAI-style request with image_url vision blocks and POSTs it to
OpenRouter via httpx. Used when ``summary.provider == "openrouter"``.
"""

from __future__ import annotations

from typing import Any

from davinci_monet.ai.images import EncodedImage
from davinci_monet.ai.summarizer import (
    SummaryError,
    SummaryResult,
    resolve_api_key,
)

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


def build_openrouter_messages(
    system_text: str,
    user_text: str,
    encoded_images: list[tuple[str, EncodedImage]],
) -> list[dict[str, Any]]:
    """Build OpenAI-format chat messages with data-URL image blocks."""
    user_content: list[dict[str, Any]] = [{"type": "text", "text": user_text}]
    for caption, enc in encoded_images:
        user_content.append({"type": "text", "text": f"Figure: {caption}"})
        user_content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:{enc.media_type};base64,{enc.data}"},
            }
        )
    return [
        {"role": "system", "content": system_text},
        {"role": "user", "content": user_content},
    ]


def _send_openrouter_request(cfg: Any, key: str, body: dict[str, Any]) -> dict[str, Any]:
    """POST to OpenRouter and return parsed JSON. Injectable seam for tests."""
    import httpx

    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    try:
        resp = httpx.post(OPENROUTER_URL, headers=headers, json=body, timeout=60)
    except Exception as exc:  # noqa: BLE001 - any network error degrades
        raise SummaryError(f"OpenRouter request failed: {exc}") from exc
    if resp.status_code != 200:
        raise SummaryError(f"OpenRouter HTTP {resp.status_code}: {resp.text[:300]}")
    return resp.json()


def call_openrouter(
    system_text: str,
    user_text: str,
    encoded_images: list[tuple[str, EncodedImage]],
    cfg: Any,
) -> SummaryResult:
    """Call OpenRouter and return a SummaryResult (same shape as the Anthropic path)."""
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

- [ ] **Step 4: Run test to verify it passes**

Run: `HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/unit/ai/test_openrouter.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add davinci_monet/ai/openrouter.py davinci_monet/tests/unit/ai/test_openrouter.py
git commit -m "feat(ai): OpenRouter provider via httpx (OpenAI chat format + vision)"
```

---

## Task 4: Provider dispatch in `generate_summary` + exports

**Files:**
- Modify: `davinci_monet/ai/summarizer.py` (`generate_summary` → dispatch; extract `_call_anthropic`)
- Modify: `davinci_monet/ai/__init__.py` (export `call_openrouter`, `resolve_api_key`)
- Test: `davinci_monet/tests/unit/ai/test_generate_summary.py` (append dispatch test)

- [ ] **Step 1: Write the failing test**

Append to `davinci_monet/tests/unit/ai/test_generate_summary.py`:

```python
def test_generate_summary_routes_to_openrouter(monkeypatch, tmp_path) -> None:
    import davinci_monet.ai.openrouter as orouter
    from davinci_monet.config.schema import SummaryConfig

    keyfile = tmp_path / "k.api"
    keyfile.write_text("sk-or-test")
    cfg = SummaryConfig.model_validate(
        {"provider": "openrouter", "api_key_file": str(keyfile)}
    )

    def _fake_send(cfg_arg, key, body):
        return {
            "model": body["model"],
            "choices": [{"message": {"content": "## What this run is\nok\n## Caveats\n"}}],
            "usage": {"prompt_tokens": 7, "completion_tokens": 8},
        }

    monkeypatch.setattr(orouter, "_send_openrouter_request", _fake_send)

    result = generate_summary(_payload(_png_path(tmp_path)), cfg=cfg)
    assert "## Caveats" in result.markdown
    assert result.usage == {"input_tokens": 7, "output_tokens": 8}
    assert result.model == "anthropic/claude-3.5-haiku"
```

(`_payload` and `_png_path` already exist in this test file from the original feature.)

- [ ] **Step 2: Run test to verify it fails**

Run: `HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/unit/ai/test_generate_summary.py::test_generate_summary_routes_to_openrouter -v`
Expected: FAIL — `generate_summary` ignores `provider` and tries the Anthropic path (`_build_client` → constructs a real client / errors), not the stubbed OpenRouter send.

- [ ] **Step 3: Refactor `generate_summary` into a dispatcher**

In `davinci_monet/ai/summarizer.py`, replace the current `generate_summary` function with a dispatcher plus an extracted `_call_anthropic`:

```python
def generate_summary(
    payload: SummaryPayload,
    *,
    cfg: Any,
    client: Any | None = None,
) -> SummaryResult:
    """Encode images, then dispatch to the configured provider.

    ``client`` is the injectable Anthropic client (used when
    ``cfg.provider == "anthropic"``). The OpenRouter path's injectable seam is
    ``openrouter._send_openrouter_request``.
    """
    from davinci_monet.ai.images import encode_image

    encoded: list[tuple[str, EncodedImage]] = []
    for img in payload.images:
        try:
            encoded.append((img.caption, encode_image(img.path)))
        except Exception as exc:  # noqa: BLE001 - bad figure must not abort summary
            logger.warning("Skipping figure %s: %s", img.path, exc)

    provider = getattr(cfg, "provider", "anthropic")
    if provider == "openrouter":
        from davinci_monet.ai.openrouter import call_openrouter  # lazy: avoid cycle

        return call_openrouter(SYSTEM_PROMPT, render_text(payload), encoded, cfg)

    return _call_anthropic(payload, encoded, cfg, client=client)


def _call_anthropic(
    payload: SummaryPayload,
    encoded: list[tuple[str, EncodedImage]],
    cfg: Any,
    *,
    client: Any | None = None,
) -> SummaryResult:
    """Call the Anthropic Messages API and return a SummaryResult."""
    if client is None:
        client = _build_client(cfg)

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

- [ ] **Step 4: Export the new public symbols**

In `davinci_monet/ai/__init__.py`, add `call_openrouter` and `resolve_api_key` to the imports and `__all__`:

```python
from davinci_monet.ai.openrouter import call_openrouter
from davinci_monet.ai.summarizer import (
    SummaryError,
    SummaryResult,
    build_prompt,
    generate_summary,
    resolve_api_key,
)
```

Add `"call_openrouter"` and `"resolve_api_key"` to the `__all__` list.

- [ ] **Step 5: Run tests to verify they pass**

Run: `HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/unit/ai -v`
Expected: PASS (all ai unit tests, including the existing Anthropic `generate_summary` tests and the new dispatch test).

- [ ] **Step 6: Commit**

```bash
git add davinci_monet/ai/summarizer.py davinci_monet/ai/__init__.py davinci_monet/tests/unit/ai/test_generate_summary.py
git commit -m "feat(ai): dispatch generate_summary on summary.provider"
```

---

## Task 5: Integration test through `run_from_config`

**Files:**
- Test: `davinci_monet/tests/integration/test_ai_summary_openrouter_pipeline.py`

Context: mirror `davinci_monet/tests/integration/test_ai_summary_pipeline.py` (created in the original feature), but drive the OpenRouter path and stub `_send_openrouter_request`.

- [ ] **Step 1: Write the integration test**

Create `davinci_monet/tests/integration/test_ai_summary_openrouter_pipeline.py`:

```python
"""Integration: OpenRouter summary path through PipelineRunner.run_from_config.

The pipeline runs for real on synthetic data; only the OpenRouter HTTP send is
stubbed so no network call is made.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np

import davinci_monet.ai.openrouter as orouter
from davinci_monet.config.parser import LegacyConfigWarning
from davinci_monet.core.protocols import DataGeometry
from davinci_monet.tests.synthetic.generators import Domain, TimeConfig
from davinci_monet.tests.synthetic.models import create_model_dataset
from davinci_monet.tests.synthetic.scenarios import PerfectMatchScenario


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


def test_openrouter_summary_writes_file(monkeypatch, tmp_path: Path) -> None:
    from davinci_monet.pipeline.runner import PipelineRunner

    calls = {}

    def _fake_send(cfg, key, body):
        calls["key"] = key
        calls["body"] = body
        return {
            "model": body["model"],
            "choices": [
                {"message": {"content": "## What this run is\nx\n## Caveats\nsynthetic\n"}}
            ],
            "usage": {"prompt_tokens": 100, "completion_tokens": 50},
        }

    monkeypatch.setattr(orouter, "_send_openrouter_request", _fake_send)

    keyfile = tmp_path / "OpenRouter.api"
    keyfile.write_text("sk-or-fake")

    config = _build_config(tmp_path)
    config["summary"] = {
        "enabled": True,
        "provider": "openrouter",
        "api_key_file": str(keyfile),
    }

    runner = PipelineRunner(show_progress=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", LegacyConfigWarning)
        result = runner.run_from_config(config)

    assert result.success
    summary_file = tmp_path / "output" / "AI_summary.md"
    assert summary_file.exists()
    assert "## Caveats" in summary_file.read_text()
    # key came from the file; the request carried an image_url vision block
    assert calls["key"] == "sk-or-fake"
    user_content = calls["body"]["messages"][1]["content"]
    assert any(block.get("type") == "image_url" for block in user_content)


def test_openrouter_summary_skips_without_key(monkeypatch, tmp_path: Path) -> None:
    from davinci_monet.pipeline.runner import PipelineRunner

    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    config = _build_config(tmp_path)
    config["summary"] = {
        "enabled": True,
        "provider": "openrouter",
        "api_key_file": str(tmp_path / "missing.api"),  # does not exist
    }

    runner = PipelineRunner(show_progress=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", LegacyConfigWarning)
        result = runner.run_from_config(config)

    assert result.success  # non-fatal: run still succeeds
    assert not (tmp_path / "output" / "AI_summary.md").exists()
```

- [ ] **Step 2: Verify synthetic import paths**

Run:
```bash
HDF5_USE_FILE_LOCKING=FALSE python -c "from davinci_monet.tests.synthetic.generators import Domain, TimeConfig; from davinci_monet.tests.synthetic.models import create_model_dataset; from davinci_monet.tests.synthetic.scenarios import PerfectMatchScenario; from davinci_monet.config.parser import LegacyConfigWarning; print('ok')"
```
Expected: `ok`. If an import fails, align it with `davinci_monet/tests/integration/test_ai_summary_pipeline.py` (do not change the test logic).

- [ ] **Step 3: Run the integration test**

Run: `HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/integration/test_ai_summary_openrouter_pipeline.py -v`
Expected: PASS (2 tests). Do not weaken assertions to pass — investigate any failure via the pipeline log under `tmp_path/logs`.

- [ ] **Step 4: Commit**

```bash
git add davinci_monet/tests/integration/test_ai_summary_openrouter_pipeline.py
git commit -m "test(ai): OpenRouter summary integration through run_from_config"
```

---

## Task 6: Documentation

**Files:**
- Modify: `README.md` (AI Summary section)
- Modify: `CLAUDE.md` (summary gotcha entry)

- [ ] **Step 1: Extend the README AI Summary section**

In `README.md`, in the existing "AI Summary (Visual Intelligence)" section, add an OpenRouter example after the existing YAML block:

````markdown
To use OpenRouter instead of the Anthropic API directly (e.g. with a key in a
file), set the provider and point at the key file:

```yaml
summary:
  enabled: true
  provider: openrouter
  api_key_file: OpenRouter.api          # gitignored; falls back to api_key_env
  model: anthropic/claude-3.5-haiku     # OpenRouter model id (default for this provider)
```
````

- [ ] **Step 2: Update the CLAUDE.md summary gotcha**

In `CLAUDE.md`, extend the AI summary Common Gotchas entry to mention the provider option. Append to that entry:

```markdown
   The provider can be `anthropic` (default, `ANTHROPIC_API_KEY`) or `openrouter`
   (`provider: openrouter`, key via `api_key_file:` or `OPENROUTER_API_KEY`,
   default model `anthropic/claude-3.5-haiku`).
```

- [ ] **Step 3: Commit**

```bash
git add README.md CLAUDE.md
git commit -m "docs: document the OpenRouter summary provider"
```

---

## Task 7: Full suite + formatting/type gates

**Files:** none (verification only)

- [ ] **Step 1: Run the full suite (Dask single worker)**

Run:
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate davinci
DASK_NUM_WORKERS=1 HDF5_USE_FILE_LOCKING=FALSE python -m pytest -q -p no:cacheprovider
```
Expected: all tests pass (previous count + the new OpenRouter tests), 1 skipped is fine.

- [ ] **Step 2: Format + import-sort the new/changed code**

Run:
```bash
black davinci_monet && isort davinci_monet
```
Expected: no changes needed (or trivial reformat).

- [ ] **Step 3: Type-check**

Run:
```bash
mypy davinci_monet/ai davinci_monet/config/schema.py
```
Expected: no new errors. If mypy flags `import httpx` as untyped, add `# type: ignore[import-untyped]` on that line only.

- [ ] **Step 4: Commit any formatting fixes**

```bash
git add -A
git commit -m "style: black/isort/mypy fixes for OpenRouter provider"
```

(Skip if nothing changed.)

---

## Self-Review Notes (for the implementer)

- **Spec coverage:** config fields + provider-default validator (T1), `resolve_api_key` file/env (T2), `ai/openrouter.py` messages/send/call (T3), `generate_summary` dispatch + exports (T4), `run_from_config` integration both happy + skip paths (T5), docs (T6), full suite + gates (T7).
- **Anthropic path untouched in behavior:** `_call_anthropic` is the original body verbatim; all original `generate_summary`/stage/config tests remain valid.
- **No circular import:** `openrouter.py` imports from `summarizer.py` at module top; `summarizer.py` imports `call_openrouter` lazily inside `generate_summary`.
- **No network in tests:** every test stubs `_send_openrouter_request` or the Anthropic `client`/`_build_client`.
- **Manual live run (after T7, not a code task):** run a real analysis with `provider: openrouter`, `api_key_file: OpenRouter.api`, `model: anthropic/claude-3.5-haiku` and confirm a real `AI_summary.md`. This is done by the controller, not the implementer subagents.
