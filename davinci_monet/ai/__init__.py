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
