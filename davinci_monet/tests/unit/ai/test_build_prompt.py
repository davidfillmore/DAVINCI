"""Unit tests for ai.summarizer.build_prompt and render_text (pure, no network)."""

from __future__ import annotations

from davinci_monet.ai.images import EncodedImage
from davinci_monet.ai.payload import ImageRef, SummaryPayload
from davinci_monet.ai.summarizer import SYSTEM_PROMPT, build_prompt, render_text


def _payload(instructions: str | None = None) -> SummaryPayload:
    return SummaryPayload(
        period={"start": "2024-02-01", "end": "2024-02-03"},
        sources_summary=["cam (cesm_fv)", "airnow (pt_sfc)"],
        pairs_summary=["cam_vs_airnow_o3"],
        stats_rows=[
            {
                "pair": "cam_vs_airnow_o3",
                "variable": "O3",
                "metrics": {"N": 120, "MB": -2.5, "R": 0.82},
            }
        ],
        images=[ImageRef(caption="01_o3_scatter", path="/x/01_o3_scatter.png")],
        instructions=instructions,
    )


def test_render_text_includes_period_sources_and_stats() -> None:
    text = render_text(_payload())
    assert "2024-02-01" in text and "2024-02-03" in text
    assert "cam (cesm_fv)" in text
    assert "cam_vs_airnow_o3" in text
    assert "O3" in text and "N=120" in text


def test_render_text_appends_instructions() -> None:
    text = render_text(_payload(instructions="Focus on coastal sites."))
    assert "Focus on coastal sites." in text


def test_system_prompt_is_template_driven() -> None:
    lowered = SYSTEM_PROMPT.lower()
    assert "section" in lowered
    assert "format" in lowered
    assert "word budget" in lowered
    assert "comparison" in lowered


def test_build_prompt_structure() -> None:
    encoded = [("01_o3_scatter", EncodedImage(media_type="image/png", data="QUJD"))]
    system, content = build_prompt(_payload(), encoded)

    # system is a cache-controlled text block
    assert system[0]["type"] == "text"
    assert system[0]["cache_control"] == {"type": "ephemeral"}

    # content: one text block, then caption + image per figure
    assert content[0]["type"] == "text"
    assert content[1]["type"] == "text" and "Figure: 01_o3_scatter" in content[1]["text"]
    assert content[2]["type"] == "image"
    assert content[2]["source"]["media_type"] == "image/png"
    assert content[2]["source"]["data"] == "QUJD"


def test_build_prompt_no_images() -> None:
    payload = _payload()
    payload.images = []
    system, content = build_prompt(payload, [])
    assert len(content) == 1
    assert content[0]["type"] == "text"
