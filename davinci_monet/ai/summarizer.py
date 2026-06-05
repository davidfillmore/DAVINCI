"""Build and run the single-prompt Claude summary for an analysis run."""

from __future__ import annotations

import logging
import os
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
            metric_str = ", ".join(f"{k}={_fmt(v)}" for k, v in row["metrics"].items())
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
        raise SummaryError(f"API key environment variable '{cfg.api_key_env}' is not set")
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
