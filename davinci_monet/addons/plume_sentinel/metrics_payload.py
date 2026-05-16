"""Assemble plumesentinel.metrics.v1 payloads from pipeline outputs.

Extracted from workflow.py so both ``run(..., emit_metrics_json=...)`` and
``PlumeSentinelBulletinStage`` can call the same payload builder.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import xarray as xr

from davinci_monet.addons.plume_sentinel import metrics as _metrics
from davinci_monet.addons.plume_sentinel.processing import GriddedAodResult


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


def _event_date_from_config(analysis_cfg: dict[str, Any]) -> str:
    """Extract YYYY-MM-DD from analysis.start_time."""
    start = analysis_cfg.get("start_time")
    if isinstance(start, datetime):
        return start.date().isoformat()
    if isinstance(start, str):
        return start[:10]
    return "1970-01-01"


def _derive_input_datasets(
    plume_cfg: dict[str, Any], default_valid_time: str
) -> list[dict[str, Any]]:
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
            type_,
            (f"{type_} (unspecified)", None, "unknown"),
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
            "input_datasets derived from config; loaders do not yet emit " "structured provenance"
        ),
    }


def build_metrics_payload(
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
    from davinci_monet.addons.plume_sentinel.workflow import ADDON_VERSION

    analysis_cfg = config.get("analysis", {}) or {}
    plume_cfg = config.get("plume_sentinel", {}) or {}
    prepared: dict[str, Any] = context_metadata.get("plume_sentinel_prepared", {})

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
                quality_flags.append(
                    {
                        "category": "metrics",
                        "severity": "warning",
                        "message": f"AOD metrics extraction failed: {exc}",
                    }
                )
        elif gpd is not None and isinstance(value, gpd.GeoDataFrame):
            # Heuristic: any 'density'/'Density' column → treat as HMS smoke.
            has_density = any(str(c).lower() == "density" for c in value.columns)
            if has_density:
                try:
                    metrics_block.update(_metrics.hms_metrics(value))
                except Exception as exc:  # noqa: BLE001
                    quality_flags.append(
                        {
                            "category": "metrics",
                            "severity": "warning",
                            "message": f"HMS metrics extraction failed: {exc}",
                        }
                    )

    # Plot URLs — collect generated plot paths from the plotting stage result.
    plot_urls: dict[str, str] = {}
    base_run_id = run_id or "unknown-run"
    for sr in stage_results:
        if getattr(sr, "stage_name", None) == "plotting" and getattr(sr, "data", None):
            for path in sr.data.get("plots_generated", []) or []:
                p = Path(path)
                key = p.stem  # e.g. "modis_aod_truecolor"
                plot_urls[key] = f"http://localhost:8080/runs/{base_run_id}/{p.name}"

    event_date = _event_date_from_config(analysis_cfg)
    valid_time_iso = _to_iso(
        analysis_cfg.get("start_time"),
        default=f"{event_date}T00:00:00+00:00",
    )

    # input_datasets: prefer loader-provided structured provenance if present;
    # otherwise fall back to config-derived list and flag it.
    loader_input_datasets = context_metadata.get("plume_sentinel_input_datasets")
    if loader_input_datasets:
        input_datasets = loader_input_datasets
    else:
        input_datasets = _derive_input_datasets(plume_cfg, valid_time_iso)
        quality_flags.append(_provenance_quality_flag())

    # quality_flags: merge any loader-provided flags with what we accumulated.
    loader_quality_flags = context_metadata.get("plume_sentinel_quality_flags", [])
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
        "config_files": [Path(config_path).name if config_path else "<inline-config>"],
        "input_datasets": input_datasets,
        "quality_flags": quality_flags,
    }
    bulletin_info = context_metadata.get("plume_sentinel_bulletin")
    if bulletin_info:
        payload["bulletin"] = bulletin_info
    return payload
