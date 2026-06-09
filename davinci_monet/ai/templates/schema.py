"""Pydantic schema for AI summary comparison templates.

A template is an ordered list of sections; each section declares a fixed
``format`` and a soft word ``budget`` that become an instruction to the model.
Mirrors the satellite-catalog schema style (``extra='forbid'``).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

SectionFormat = Literal["prose", "bullets", "headline", "table", "metric_callout"]

# Each format maps to a fixed instruction phrase; ``{words}`` is the soft budget.
_FORMAT_PHRASES: dict[str, str] = {
    "prose": "{words} words or fewer of prose",
    "bullets": "a short bullet list, {words} words or fewer total",
    "headline": "one line, {words} words or fewer",
    "table": "a compact markdown table, {words} words or fewer total",
    "metric_callout": "a few `key: value` metric lines, {words} words or fewer total",
}


class TemplateSection(BaseModel):
    """One section of a comparison template."""

    model_config = ConfigDict(extra="forbid")

    heading: str
    format: SectionFormat
    words: int = Field(gt=0, description="Soft per-section word budget (prompt-only).")
    instruction: str | None = None

    def format_instruction(self) -> str:
        """Render the model-facing instruction for this section."""
        phrase = _FORMAT_PHRASES[self.format].format(words=self.words)
        if self.instruction:
            return f"{phrase}. {self.instruction}"
        return phrase


class SummaryTemplate(BaseModel):
    """A named comparison template selected by the comparand variable."""

    model_config = ConfigDict(extra="forbid")

    name: str
    title: str | None = None
    description: str | None = None
    matches: list[str] = Field(default_factory=list)
    sections: list[TemplateSection] = Field(min_length=1)
