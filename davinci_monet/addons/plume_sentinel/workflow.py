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

import numpy as np
import xarray as xr

from davinci_monet.addons.plume_sentinel import metrics as _metrics
from davinci_monet.addons.plume_sentinel.processing import GriddedAodResult
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

        payload = _build_metrics_payload(
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


# ---------------------------------------------------------------------------
# Internal helpers — payload assembly
# ---------------------------------------------------------------------------


def _gridded_aod_to_dataarray(g: GriddedAodResult) -> xr.DataArray:
    """Wrap a ``GriddedAodResult`` in an xarray.DataArray with lat/lon coords.

    The metrics module recognises the dim names ``lat``/``lon`` (or ``y``/``x``)
    and uses the coordinate spacing to compute cell areas. ``data_2d`` is laid
    out as ``(nlat, nlon)`` (see ``processing.prepare_modis_aod``).
    """
    return xr.DataArray(
        g.data_2d,
        dims=("lat", "lon"),
        coords={"lat": g.lat_centers, "lon": g.lon_centers},
        name="aod",
    )


def _to_iso(value: Any, default: str) -> str:
    """Best-effort ISO-8601 string for a datetime/str value, with default."""
    if value is None:
        return default
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.isoformat()
    if isinstance(value, str):
        # Already a string; assume caller knows what they're doing.
        return value
    return str(value)


def _apply_event_date_override(config: dict[str, Any], event_date: str) -> None:
    """Mutate ``config['analysis']`` so its time window spans ``event_date``.

    Sets ``analysis.start_time`` to ``<event_date>T00:00:00+00:00`` and
    ``analysis.end_time`` to ``<event_date>T23:59:59+00:00`` (UTC day).
    The values are stored as timezone-aware ``datetime`` objects, which
    matches the shape produced by ``Config.model_dump()`` and is accepted
    by ``PipelineRunner.run_from_config`` and the metrics-payload helpers
    (``_event_date_from_config`` / ``_to_iso``).

    Accepts a config whose existing ``start_time``/``end_time`` are bare
    YAML dates (``str``), pre-parsed ``datetime`` objects, or absent —
    the override always wins.
    """
    try:
        parsed = datetime.strptime(event_date, "%Y-%m-%d")
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"event_date must be YYYY-MM-DD, got {event_date!r}: {exc}"
        ) from exc

    start = parsed.replace(hour=0, minute=0, second=0, tzinfo=timezone.utc)
    end = parsed.replace(hour=23, minute=59, second=59, tzinfo=timezone.utc)

    analysis = config.setdefault("analysis", {})
    analysis["start_time"] = start
    analysis["end_time"] = end


def _event_date_from_config(analysis_cfg: dict[str, Any]) -> str:
    """Extract YYYY-MM-DD from analysis.start_time."""
    start = analysis_cfg.get("start_time")
    if isinstance(start, datetime):
        return start.date().isoformat()
    if isinstance(start, str):
        return start[:10]
    return "1970-01-01"


def _derive_input_datasets(plume_cfg: dict[str, Any], default_valid_time: str) -> list[dict[str, Any]]:
    """Derive a structured input_datasets list from the YAML config.

    The loaders/processing modules do not yet emit structured provenance,
    so we fall back to extracting dataset metadata from the config. This is
    flagged in ``quality_flags`` (see ``_provenance_quality_flag``).
    """
    inputs = plume_cfg.get("inputs", {}) or {}
    datasets: list[dict[str, Any]] = []
    name_map = {
        "modis_l2_aod": ("MODIS L2 AOD (MOD04)", "Collection 6.1", "NASA LAADS"),
        "goes_truecolor": ("GOES-16 ABI L2 MCMIP", None, "NOAA NESDIS"),
        "hms_smoke": ("NOAA NESDIS HMS Smoke", None, "NOAA NESDIS"),
    }
    for _key, spec in inputs.items():
        if not isinstance(spec, dict):
            continue
        type_ = spec.get("type", "unknown")
        name, version, agency = name_map.get(
            type_, (f"{type_} (unspecified)", None, "unknown"),
        )
        granules: list[str] = []
        if spec.get("file"):
            granules.append(str(spec["file"]))
        if spec.get("files"):
            granules.extend(str(f) for f in spec["files"])
        ds: dict[str, Any] = {
            "name": name,
            "valid_time": default_valid_time,
            "agency": agency,
            "granules": granules,
        }
        if version is not None:
            ds["version"] = version
        datasets.append(ds)
    return datasets


def _provenance_quality_flag() -> dict[str, str]:
    return {
        "category": "provenance",
        "severity": "info",
        "message": (
            "input_datasets derived from config; loaders do not yet emit "
            "structured provenance"
        ),
    }


def _build_metrics_payload(
    *,
    context_metadata: dict[str, Any],
    config: dict[str, Any],
    config_path: str | None,
    run_id: str | None,
    region: str | None,
    config_slug: str | None,
    wallclock_s: float,
    stage_results: list,
) -> dict[str, Any]:
    """Assemble a ``plumesentinel.metrics.v1`` payload from stage outputs."""
    analysis_cfg = config.get("analysis", {}) or {}
    plume_cfg = config.get("plume_sentinel", {}) or {}
    prepared: dict[str, Any] = context_metadata.get(
        "plume_sentinel_prepared", {}
    )

    metrics_block: dict[str, Any] = {}
    quality_flags: list[dict[str, Any]] = []

    # Walk prepared inputs; dispatch by Python type so we don't rely on
    # naming conventions in the user-supplied config.
    try:
        import geopandas as gpd
    except ImportError:  # pragma: no cover - geopandas always installed in env
        gpd = None  # type: ignore[assignment]

    for _name, value in prepared.items():
        if isinstance(value, GriddedAodResult):
            try:
                da = _gridded_aod_to_dataarray(value)
                metrics_block.update(_metrics.aod_metrics(da))
            except Exception as exc:  # noqa: BLE001 - record + continue
                quality_flags.append({
                    "category": "metrics",
                    "severity": "warning",
                    "message": f"AOD metrics extraction failed: {exc}",
                })
        elif gpd is not None and isinstance(value, gpd.GeoDataFrame):
            # Heuristic: any 'density'/'Density' column → treat as HMS smoke.
            has_density = any(
                str(c).lower() == "density" for c in value.columns
            )
            if has_density:
                try:
                    metrics_block.update(_metrics.hms_metrics(value))
                except Exception as exc:  # noqa: BLE001
                    quality_flags.append({
                        "category": "metrics",
                        "severity": "warning",
                        "message": f"HMS metrics extraction failed: {exc}",
                    })

    # Plot URLs — collect generated plot paths from the plotting stage result.
    plot_urls: dict[str, str] = {}
    base_run_id = run_id or "unknown-run"
    for sr in stage_results:
        if getattr(sr, "stage_name", None) == "plotting" and getattr(sr, "data", None):
            for path in sr.data.get("plots_generated", []) or []:
                p = Path(path)
                key = p.stem  # e.g. "modis_aod_truecolor"
                plot_urls[key] = (
                    f"http://localhost:8080/runs/{base_run_id}/{p.name}"
                )

    event_date = _event_date_from_config(analysis_cfg)
    valid_time_iso = _to_iso(
        analysis_cfg.get("start_time"),
        default=f"{event_date}T00:00:00+00:00",
    )

    # input_datasets: prefer loader-provided structured provenance if present;
    # otherwise fall back to config-derived list and flag it.
    loader_input_datasets = context_metadata.get(
        "plume_sentinel_input_datasets"
    )
    if loader_input_datasets:
        input_datasets = loader_input_datasets
    else:
        input_datasets = _derive_input_datasets(plume_cfg, valid_time_iso)
        quality_flags.append(_provenance_quality_flag())

    # quality_flags: merge any loader-provided flags with what we accumulated.
    loader_quality_flags = context_metadata.get(
        "plume_sentinel_quality_flags", []
    )
    if loader_quality_flags:
        quality_flags = list(loader_quality_flags) + quality_flags

    import davinci_monet

    payload = {
        "schema": "plumesentinel.metrics.v1",
        "run_id": run_id or f"{event_date}-{region or 'unknown'}-{config_slug or 'unknown'}",
        "region": region or "unknown",
        "config_slug": config_slug or "unknown",
        "event_date": event_date,
        "valid_time": valid_time_iso,
        "produced_at": datetime.now(timezone.utc).isoformat(),
        "metrics": metrics_block,
        "plot_urls": plot_urls,
        "wallclock_s": wallclock_s,
        "pipeline_version": {
            "davinci_monet": davinci_monet.__version__,
            "plume_sentinel_addon": ADDON_VERSION,
        },
        "config_files": [
            Path(config_path).name if config_path else "<inline-config>"
        ],
        "input_datasets": input_datasets,
        "quality_flags": quality_flags,
    }
    return payload
