"""Collect the data the summary prompt needs out of a PipelineContext."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from davinci_monet.config.schema import SummaryConfig
    from davinci_monet.pipeline.stages import PipelineContext


@dataclass
class ImageRef:
    """A plot image selected for the prompt."""

    caption: str
    path: str


@dataclass
class SummaryPayload:
    """Everything the summary prompt is built from."""

    period: dict[str, Any]
    sources_summary: list[str]
    pairs_summary: list[str]
    stats_rows: list[dict[str, Any]]
    images: list[ImageRef]
    instructions: str | None


_STATS_STAGES = ("statistics", "obs_statistics")
_PLOT_STAGES = ("plotting", "obs_plotting")


def collect_payload(context: "PipelineContext", cfg: "SummaryConfig") -> SummaryPayload:
    """Build a :class:`SummaryPayload` from the run's config and stage results."""
    config = context.config
    analysis = config.get("analysis", {}) or {}
    period = {"start": analysis.get("start_time"), "end": analysis.get("end_time")}

    sources_summary: list[str] = []
    for block in ("sources", "model", "obs"):
        for label, spec in (config.get(block) or {}).items():
            if isinstance(spec, dict):
                stype = spec.get("type") or spec.get("mod_type") or spec.get("obs_type") or "?"
                sources_summary.append(f"{label} ({stype})")

    pairs_summary = list((config.get("pairs") or {}).keys())

    stats_rows: list[dict[str, Any]] = []
    for stage_key in _STATS_STAGES:
        result = context.results.get(stage_key)
        data = getattr(result, "data", None)
        if not isinstance(data, dict):
            continue
        for pair_key, pair_stats in data.items():
            if not isinstance(pair_stats, dict):
                continue
            for var_name, var_stats in pair_stats.items():
                if var_name.startswith("_") or not isinstance(var_stats, dict):
                    continue
                metrics = {k: v for k, v in var_stats.items() if not k.startswith("_")}
                stats_rows.append({"pair": pair_key, "variable": var_name, "metrics": metrics})

    all_plots: list[str] = []
    for stage_key in _PLOT_STAGES:
        result = context.results.get(stage_key)
        data = getattr(result, "data", None)
        if not isinstance(data, dict):
            continue
        for path in data.get("plots_generated", []) or []:
            if str(path).lower().endswith(".png"):
                all_plots.append(str(path))

    if cfg.plots:
        selected = [p for p in all_plots if any(k in Path(p).stem for k in cfg.plots)]
    else:
        selected = all_plots[: cfg.max_images]

    images = [ImageRef(caption=Path(p).stem, path=p) for p in selected]

    return SummaryPayload(
        period=period,
        sources_summary=sources_summary,
        pairs_summary=pairs_summary,
        stats_rows=stats_rows,
        images=images,
        instructions=cfg.instructions,
    )
