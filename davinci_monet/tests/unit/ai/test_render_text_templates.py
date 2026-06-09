"""render_text emits per-comparison template section instructions."""

from __future__ import annotations

from davinci_monet.ai.payload import ImageRef, SummaryPayload
from davinci_monet.ai.summarizer import render_text
from davinci_monet.ai.templates import get_template_registry


def _payload_with_template() -> SummaryPayload:
    ozone = get_template_registry().get("ozone_eval")
    return SummaryPayload(
        period={"start": "2024-02-01", "end": "2024-02-03"},
        sources_summary=["cam (cesm_fv)"],
        pairs_summary=["cam_vs_airnow_o3"],
        stats_rows=[
            {
                "pair": "cam_vs_airnow_o3",
                "variable": "O3",
                "metrics": {"N": 120, "MB": -2.5},
                "template": ozone,
            }
        ],
        images=[ImageRef(caption="01_o3", path="/x/01_o3.png")],
        instructions=None,
    )


def test_render_emits_template_headings_and_budget() -> None:
    text = render_text(_payload_with_template())
    assert "## cam_vs_airnow_o3 — O3" in text
    assert "### Bottom line" in text
    assert "20 words or fewer" in text
    assert "N=120" in text


def test_render_falls_back_to_generic_without_attached_template() -> None:
    payload = _payload_with_template()
    del payload.stats_rows[0]["template"]
    text = render_text(payload)
    assert "### What this run is" in text
