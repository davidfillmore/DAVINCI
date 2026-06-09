"""AI summary comparison templates (built-in YAML library + resolution)."""

from __future__ import annotations

from davinci_monet.ai.templates.registry import (
    TemplateRegistry,
    UnknownTemplateError,
    get_template_registry,
    resolve_template_for,
)
from davinci_monet.ai.templates.schema import SummaryTemplate, TemplateSection

__all__ = [
    "SummaryTemplate",
    "TemplateSection",
    "TemplateRegistry",
    "UnknownTemplateError",
    "get_template_registry",
    "resolve_template_for",
]
