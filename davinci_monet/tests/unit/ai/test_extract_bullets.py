"""Unit tests for ai.summarizer.extract_bullets."""

from __future__ import annotations

from davinci_monet.ai.summarizer import extract_bullets

_SAMPLE = """# DAVINCI Model Evaluation Brief

## What this run is
Synthetic O3 vs surface obs.

## Headline metrics
- Mean Bias +4.82 ppb
- RMSE 5.66 ppb
- R 0.849

## Caveats
- N=98; synthetic data
"""


def test_extract_bullets_returns_all_bullets() -> None:
    assert extract_bullets(_SAMPLE) == [
        "Mean Bias +4.82 ppb",
        "RMSE 5.66 ppb",
        "R 0.849",
        "N=98; synthetic data",
    ]


def test_extract_bullets_handles_star_and_unicode_markers() -> None:
    md = "* star item\n• dot item\n  - indented item\n"
    assert extract_bullets(md) == ["star item", "dot item", "indented item"]


def test_extract_bullets_falls_back_to_subheadings() -> None:
    md = "# Title\n## What this run is\nProse.\n## Caveats\nMore prose.\n"
    assert extract_bullets(md) == ["What this run is", "Caveats"]


def test_extract_bullets_empty_when_no_bullets_or_subheadings() -> None:
    assert extract_bullets("# Title only\nplain prose line\n") == []


def test_extract_bullets_caps_with_overflow() -> None:
    md = "\n".join(f"- item {i}" for i in range(20))
    out = extract_bullets(md, max_items=5)
    assert len(out) == 5
    assert out[:4] == ["item 0", "item 1", "item 2", "item 3"]
    assert out[4] == "… (full brief in AI_summary.md)"


def test_extract_bullets_strips_markdown_emphasis() -> None:
    md = "## Headline metrics\n- **Mean Bias** +4.8 ppb\n- _RMSE_ 5.66\n- `R` 0.85\n"
    assert extract_bullets(md) == ["Mean Bias +4.8 ppb", "RMSE 5.66", "R 0.85"]
