"""Unit tests for the AI summary template schema."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from davinci_monet.ai.templates.schema import SummaryTemplate, TemplateSection


def _valid_template() -> dict:
    return {
        "name": "demo",
        "title": "Demo",
        "matches": ["o3"],
        "sections": [
            {"heading": "Bottom line", "format": "headline", "words": 20},
            {
                "heading": "Detail",
                "format": "prose",
                "words": 80,
                "instruction": "Be quantitative.",
            },
        ],
    }


def test_valid_template_round_trips() -> None:
    tmpl = SummaryTemplate(**_valid_template())
    assert tmpl.name == "demo"
    assert tmpl.matches == ["o3"]
    assert len(tmpl.sections) == 2
    assert tmpl.sections[0].format == "headline"


def test_extra_key_is_rejected() -> None:
    bad = _valid_template()
    bad["unexpected"] = "x"
    with pytest.raises(ValidationError):
        SummaryTemplate(**bad)


def test_zero_or_negative_words_rejected() -> None:
    with pytest.raises(ValidationError):
        TemplateSection(heading="h", format="prose", words=0)
    with pytest.raises(ValidationError):
        TemplateSection(heading="h", format="prose", words=-5)


def test_empty_sections_rejected() -> None:
    bad = _valid_template()
    bad["sections"] = []
    with pytest.raises(ValidationError):
        SummaryTemplate(**bad)


def test_unknown_format_rejected() -> None:
    with pytest.raises(ValidationError):
        TemplateSection(heading="h", format="paragraph", words=10)  # type: ignore[arg-type]


def test_format_instruction_includes_budget_and_extra() -> None:
    plain = TemplateSection(heading="h", format="bullets", words=40)
    assert "40 words or fewer" in plain.format_instruction()
    assert "bullet" in plain.format_instruction().lower()

    extra = TemplateSection(heading="h", format="prose", words=30, instruction="Lead with bias.")
    assert "30 words or fewer" in extra.format_instruction()
    assert "Lead with bias." in extra.format_instruction()
