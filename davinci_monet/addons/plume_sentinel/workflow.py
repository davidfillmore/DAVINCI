"""Workflow factory and runner for the Plume Sentinel pipeline.

`create_plume_sentinel_pipeline` returns the ordered stage list that
``PipelineRunner`` consumes. `run` is a higher-level entry point that
runs the full pipeline and, if requested, serializes a
``plumesentinel.metrics.v1`` JSON sidecar next to the PNG outputs.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from davinci_monet.addons.plume_sentinel.metrics_payload import (
    build_metrics_payload,
)
from davinci_monet.addons.plume_sentinel.stages import (
    PlumeSentinelLoadStage,
    PlumeSentinelPlotStage,
    PlumeSentinelPrepareStage,
)
from davinci_monet.pipeline.stages import BaseStage

ADDON_VERSION = "0.1.0"


def create_plume_sentinel_pipeline() -> list[BaseStage]:
    """Create the three-stage Plume Sentinel pipeline.

    Returns
    -------
    list[BaseStage]
        Ordered list: load_inputs -> prepare_geospatial -> plotting.
    """
    return [
        PlumeSentinelLoadStage(),
        PlumeSentinelPrepareStage(),
        PlumeSentinelPlotStage(),
    ]


# ---------------------------------------------------------------------------
# High-level entry point with optional metrics emission
# ---------------------------------------------------------------------------


def run(
    config: dict[str, Any] | str | Path,
    *,
    output_dir: str | Path | None = None,
    emit_metrics_json: str | Path | None = None,
    run_id: str | None = None,
    region: str | None = None,
    config_slug: str | None = None,
    event_date: str | None = None,
):
    """Run the Plume Sentinel pipeline; optionally write a metrics.json sidecar.

    Parameters
    ----------
    config
        Either a config dict (already loaded) or a path to a YAML control file.
    output_dir
        Override for ``analysis.output_dir`` from the config. If provided,
        plots and (by default) the metrics file are written here.
    emit_metrics_json
        If provided, write a ``plumesentinel.metrics.v1``-conformant JSON
        sidecar at this path after the pipeline succeeds.
    run_id
        Run identifier embedded in the metrics payload. If omitted, a default
        is synthesized from event_date + region + config_slug.
    region, config_slug
        Provenance fields included verbatim in the metrics payload.
    event_date
        Optional ``YYYY-MM-DD`` date that, when provided, overrides
        ``analysis.start_time`` and ``analysis.end_time`` in the loaded
        config to a full UTC day spanning the date
        (``<event_date>T00:00:00+00:00`` through ``<event_date>T23:59:59+00:00``).
        The mutated config drives the rest of the run, so the emitted
        metrics payload's ``event_date`` and ``valid_time`` reflect the
        requested date. When omitted, behavior is unchanged from prior
        Phase 3 semantics.

    Returns
    -------
    PipelineResult
        Result of the pipeline run (success flag, stage results, context).
    """
    from davinci_monet.pipeline.runner import PipelineRunner

    config_path: str | None = None
    if isinstance(config, (str, Path)):
        config_path = str(config)
        from davinci_monet.config import load_config

        config = load_config(config_path).model_dump()
    else:
        # Make a shallow copy so we can override output_dir without mutating
        # the caller's dict.
        config = dict(config)
        if "analysis" in config:
            config["analysis"] = dict(config["analysis"])

    if output_dir is not None:
        config.setdefault("analysis", {})
        config["analysis"]["output_dir"] = str(output_dir)

    if event_date is not None:
        _apply_event_date_override(config, event_date)

    runner = PipelineRunner(show_progress=False)

    wallclock_start = time.time()
    result = runner.run_from_config(config)
    wallclock_s = time.time() - wallclock_start

    if emit_metrics_json is not None:
        if not result.success:
            # Don't write a misleading "OK" metrics file for a failed run.
            return result

        payload = build_metrics_payload(
            context_metadata=result.context.metadata,
            config=config,
            config_path=config_path,
            run_id=run_id,
            region=region,
            config_slug=config_slug,
            wallclock_s=wallclock_s,
            stage_results=result.stage_results,
        )
        out_path = Path(emit_metrics_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2, default=str))

    return result


def _apply_event_date_override(config: dict[str, Any], event_date: str) -> None:
    """Mutate ``config['analysis']`` so its time window spans ``event_date``.

    Sets ``analysis.start_time`` to ``<event_date>T00:00:00+00:00`` and
    ``analysis.end_time`` to ``<event_date>T23:59:59+00:00`` (UTC day).
    The values are stored as timezone-aware ``datetime`` objects, which
    matches the shape produced by ``Config.model_dump()`` and is accepted
    by ``PipelineRunner.run_from_config`` and the metrics-payload helpers
    in ``metrics_payload`` (``_event_date_from_config`` / ``_to_iso``).

    Accepts a config whose existing ``start_time``/``end_time`` are bare
    YAML dates (``str``), pre-parsed ``datetime`` objects, or absent —
    the override always wins.
    """
    try:
        parsed = datetime.strptime(event_date, "%Y-%m-%d")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"event_date must be YYYY-MM-DD, got {event_date!r}: {exc}") from exc

    start = parsed.replace(hour=0, minute=0, second=0, tzinfo=timezone.utc)
    end = parsed.replace(hour=23, minute=59, second=59, tzinfo=timezone.utc)

    analysis = config.setdefault("analysis", {})
    analysis["start_time"] = start
    analysis["end_time"] = end
