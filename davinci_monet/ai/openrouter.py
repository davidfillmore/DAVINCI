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
OPENROUTER_CREDITS_URL = "https://openrouter.ai/api/v1/credits"


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


def _fetch_credits_remaining(cfg: Any, key: str) -> float | None:
    """Best-effort remaining OpenRouter account credit ($). Never raises.

    Returns ``data.total_credits - data.total_usage`` from GET /api/v1/credits,
    or None on any error, non-200 response, or missing/null field. Credits are
    informational only and must never affect the summary.
    """
    import httpx

    try:
        resp = httpx.get(
            OPENROUTER_CREDITS_URL,
            headers={"Authorization": f"Bearer {key}"},
            timeout=15,
        )
        if resp.status_code != 200:
            return None
        credit_data = resp.json().get("data") or {}
        total_credits = credit_data.get("total_credits")
        total_usage = credit_data.get("total_usage")
        if total_credits is None or total_usage is None:
            return None
        return float(total_credits) - float(total_usage)
    except Exception:  # noqa: BLE001 - credits are best-effort; never fail the summary
        return None


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
        credits_remaining=_fetch_credits_remaining(cfg, key),
    )
