"""Unit tests for ai.payload.collect_payload."""

from __future__ import annotations

from pathlib import Path

from davinci_monet.ai.payload import ImageRef, SummaryPayload, collect_payload
from davinci_monet.config.parser import validate_config
from davinci_monet.config.schema import SummaryConfig
from davinci_monet.pipeline.stages import (
    PipelineContext,
    StageResult,
    StageStatus,
)


def _context_with_results(plot_paths: list[str]) -> PipelineContext:
    ctx = PipelineContext(
        config={
            "analysis": {"start_time": "2024-02-01", "end_time": "2024-02-03"},
            "sources": {"cam": {"type": "cesm_fv"}, "airnow": {"type": "pt_sfc"}},
            "pairs": {
                "cam_vs_airnow_o3": {
                    "x": {"source": "airnow", "variable": "o3"},
                    "y": {"source": "cam", "variable": "O3"},
                }
            },
        }
    )
    ctx.results["statistics"] = StageResult(
        stage_name="statistics",
        status=StageStatus.COMPLETED,
        data={
            "cam_vs_airnow_o3": {
                "O3": {"N": 120, "MB": -2.5, "RMSE": 6.1, "R": 0.82, "_internal": 1},
                "_per_flight": [{"flight": "x"}],
            }
        },
    )
    ctx.results["plotting"] = StageResult(
        stage_name="plotting",
        status=StageStatus.COMPLETED,
        data={"plots_generated": plot_paths},
    )
    return ctx


def test_collect_payload_flattens_stats() -> None:
    ctx = _context_with_results(["00_o3_scatter.png"])
    payload = collect_payload(ctx, SummaryConfig(enabled=True))
    assert isinstance(payload, SummaryPayload)
    assert payload.period == {"start": "2024-02-01", "end": "2024-02-03"}
    assert any("cam" in s for s in payload.sources_summary)
    assert payload.pairs_summary == ["cam_vs_airnow_o3"]
    assert len(payload.stats_rows) == 1
    row = payload.stats_rows[0]
    assert row["pair"] == "cam_vs_airnow_o3"
    assert row["variable"] == "O3"
    assert row["metrics"]["N"] == 120
    # internal keys are dropped
    assert "_internal" not in row["metrics"]


def test_collect_payload_supports_typed_config() -> None:
    ctx = _context_with_results(["00_o3_scatter.png"])
    assert isinstance(ctx.config, dict)
    ctx.config = validate_config(ctx.config)

    payload = collect_payload(ctx, SummaryConfig(enabled=True))

    assert payload.period["start"] is not None
    assert payload.sources_summary == ["cam (cesm_fv)", "airnow (pt_sfc)"]
    assert payload.pairs_summary == ["cam_vs_airnow_o3"]


def test_collect_payload_caps_images_when_no_plots_list() -> None:
    paths = [f"{i:02d}_plot.png" for i in range(12)]
    ctx = _context_with_results(paths)
    payload = collect_payload(ctx, SummaryConfig(enabled=True, max_images=3))
    assert len(payload.images) == 3
    assert all(isinstance(i, ImageRef) for i in payload.images)


def test_collect_payload_selects_named_plots() -> None:
    paths = ["00_o3_scatter.png", "01_pm25_spatial_bias.png", "02_o3_timeseries.png"]
    ctx = _context_with_results(paths)
    payload = collect_payload(ctx, SummaryConfig(enabled=True, plots=["pm25_spatial_bias"]))
    assert len(payload.images) == 1
    assert "pm25_spatial_bias" in payload.images[0].path
    assert payload.images[0].caption == Path(payload.images[0].path).stem


def test_collect_payload_ignores_non_png() -> None:
    ctx = _context_with_results(["00_o3_scatter.png", "00_o3_scatter.pdf"])
    payload = collect_payload(ctx, SummaryConfig(enabled=True))
    assert len(payload.images) == 1
    assert payload.images[0].path.endswith(".png")
