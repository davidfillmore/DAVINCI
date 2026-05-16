"""Pipeline stages for the PlumeSentinel add-on workflow.

Each stage loads, prepares, or plots PlumeSentinel data using the
add-on's own loaders, processors, and renderers.
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from davinci_monet.addons.plume_sentinel.bulletin import (
    BulletinResponse,
    build_prompt,
    generate_bulletin,
    publish_mqtt,
)
from davinci_monet.addons.plume_sentinel.loaders import load_input
from davinci_monet.addons.plume_sentinel.metrics_payload import build_metrics_payload
from davinci_monet.addons.plume_sentinel.processing import (
    prepare_goes,
    prepare_hms,
    prepare_modis_aod,
)
from davinci_monet.addons.plume_sentinel.renderers import render_plot
from davinci_monet.addons.plume_sentinel.schema import PlumeSentinelConfig
from davinci_monet.pipeline.stages import BaseStage, PipelineContext, StageResult, StageStatus
from davinci_monet.plots.style import apply_ncar_style


def _get_addon_config(context: PipelineContext) -> PlumeSentinelConfig:
    """Extract and validate PlumeSentinel config from pipeline context."""
    ps_dict = context.config["plume_sentinel"]
    return PlumeSentinelConfig(**ps_dict)


def _get_output_dir(context: PipelineContext) -> Path:
    """Get output directory from analysis config, defaulting to 'output'."""
    analysis = context.config.get("analysis", {})
    return Path(analysis.get("output_dir", "output"))


class PlumeSentinelLoadStage(BaseStage):
    """Load GOES, HMS, MODIS, and other input datasets."""

    def __init__(self) -> None:
        super().__init__(name="load_inputs")

    def validate(self, context: PipelineContext) -> bool:
        """Check that plume_sentinel config is present."""
        return "plume_sentinel" in context.config

    def execute(self, context: PipelineContext) -> StageResult:
        start = time.time()
        cfg = _get_addon_config(context)

        loaded: dict[str, Any] = {}
        items = list(cfg.inputs.items())
        total = len(items)

        for i, (name, spec) in enumerate(items, 1):
            context.log_progress(f"Loading input: {name} ({i}/{total})")
            loaded[name] = load_input(spec.model_dump())

        # Store in context metadata for downstream stages
        context.metadata["plume_sentinel_loaded"] = loaded
        context.metadata["plume_sentinel_config"] = cfg

        return self._create_result(
            StageStatus.COMPLETED,
            data={"inputs_loaded": list(loaded.keys())},
            duration=time.time() - start,
        )


class PlumeSentinelPrepareStage(BaseStage):
    """Prepare geospatial data (regridding, reprojection, RGB assembly)."""

    def __init__(self) -> None:
        super().__init__(name="prepare_geospatial")

    def execute(self, context: PipelineContext) -> StageResult:
        start = time.time()
        cfg: PlumeSentinelConfig = context.metadata["plume_sentinel_config"]
        loaded: dict[str, Any] = context.metadata["plume_sentinel_loaded"]

        prepared: dict[str, Any] = {}
        items = list(cfg.inputs.items())
        total = len(items)

        for i, (name, spec) in enumerate(items, 1):
            context.log_progress(f"GeoOp: {name} ({i}/{total})")
            input_type = spec.type
            raw = loaded[name]

            if input_type == "goes_truecolor":
                context.log_progress(f"step: assembling GOES RGB for {name}")
                prepared[name] = prepare_goes(raw, gamma=spec.gamma)

            elif input_type == "hms_smoke":
                context.log_progress(f"step: cleaning HMS polygons for {name}")
                prepared[name] = prepare_hms(raw)

            elif input_type == "modis_l2_aod":
                context.log_progress(f"step: binning MODIS AOD to grid for {name}")
                grid_spec = spec.grid.model_dump() if spec.grid is not None else {}
                prepared[name] = prepare_modis_aod(raw, grid_spec)

            else:
                # Pass through unknown types unchanged
                prepared[name] = raw

        context.metadata["plume_sentinel_prepared"] = prepared

        return self._create_result(
            StageStatus.COMPLETED,
            data={"inputs_prepared": list(prepared.keys())},
            duration=time.time() - start,
        )


class PlumeSentinelPlotStage(BaseStage):
    """Generate true-color overlay and gridded field plots."""

    def __init__(self) -> None:
        super().__init__(name="plotting")

    def execute(self, context: PipelineContext) -> StageResult:
        start = time.time()
        cfg: PlumeSentinelConfig = context.metadata["plume_sentinel_config"]
        prepared: dict[str, Any] = context.metadata["plume_sentinel_prepared"]
        output_dir = _get_output_dir(context)

        # Apply NCAR styling before rendering
        apply_ncar_style()

        all_paths: list[str] = []
        items = list(cfg.plots.items())
        total = len(items)

        for i, (name, plot_spec) in enumerate(items, 1):
            context.log_progress(f"Plot: {name} ({i}/{total})")
            paths = render_plot(name, plot_spec.model_dump(), prepared, output_dir)
            for p in paths:
                context.log_progress(f"done: saved to {p}")
            all_paths.extend(paths)

        context.metadata["plume_sentinel_plots_generated"] = all_paths

        return self._create_result(
            StageStatus.COMPLETED,
            data={"plots_generated": all_paths},
            duration=time.time() - start,
        )


class PlumeSentinelBulletinStage(BaseStage):
    """Generate a meteorological bulletin via the Claude API; optionally publish to MQTT."""

    DEFAULT_TEMPLATE_PACKAGE = "davinci_monet.addons.plume_sentinel.templates"
    DEFAULT_TEMPLATE_NAME = "bulletin.template"

    def __init__(self) -> None:
        super().__init__(name="bulletin")

    def execute(self, context: PipelineContext) -> StageResult:
        start = time.time()
        cfg: PlumeSentinelConfig = context.metadata["plume_sentinel_config"]

        if cfg.bulletin is None:
            return self._create_result(
                StageStatus.COMPLETED,
                data={"bulletin": "skipped (no config)"},
                duration=time.time() - start,
            )

        bcfg = cfg.bulletin

        # Canned-bulletin mode (demo): read pre-saved bulletin and skip API.
        demo_block = context.config.get("analysis", {}).get("_demo", {}) or {}
        canned_path = demo_block.get("canned_bulletin")
        if canned_path:
            canned = Path(canned_path)
            if not canned.is_file():
                _append_quality_flag(
                    context,
                    "warning",
                    f"Canned bulletin not found at {canned}; bulletin skipped",
                )
                return self._create_result(
                    StageStatus.COMPLETED,
                    data={"bulletin": "skipped (canned bulletin missing)"},
                    duration=time.time() - start,
                )
            output_dir = _get_output_dir(context)
            output_dir.mkdir(parents=True, exist_ok=True)
            bulletin_path = output_dir / bcfg.output_filename
            text = canned.read_text()
            try:
                bulletin_path.write_text(text)
            except OSError as exc:
                _append_quality_flag(
                    context,
                    "warning",
                    f"Bulletin file write failed: {exc}",
                )
                return self._create_result(
                    StageStatus.COMPLETED,
                    data={"bulletin": "skipped (file write failed)"},
                    duration=time.time() - start,
                )

            mqtt_published = False
            if bcfg.mqtt is not None:
                try:
                    publish_mqtt(
                        text=text,
                        broker=bcfg.mqtt.broker,
                        topic=bcfg.mqtt.topic,
                        port=bcfg.mqtt.port,
                        qos=bcfg.mqtt.qos,
                    )
                    mqtt_published = True
                except Exception as exc:  # noqa: BLE001
                    _append_quality_flag(
                        context,
                        "warning",
                        f"MQTT publish to {bcfg.mqtt.broker}:{bcfg.mqtt.port} failed: {exc}",
                    )

            context.metadata["plume_sentinel_bulletin"] = {
                "path": str(bulletin_path),
                "mode": "canned",
                "source": str(canned),
            }
            return self._create_result(
                StageStatus.COMPLETED,
                data={
                    "bulletin_path": str(bulletin_path),
                    "mqtt_published": mqtt_published,
                    "mode": "canned",
                    "source": str(canned),
                },
                duration=time.time() - start,
            )

        api_key = os.environ.get(bcfg.api_key_env)
        if not api_key:
            _append_quality_flag(
                context,
                "warning",
                f"API key env var {bcfg.api_key_env} not set; bulletin skipped",
            )
            return self._create_result(
                StageStatus.COMPLETED,
                data={"bulletin": "skipped (no api key)"},
                duration=time.time() - start,
            )

        # Build metrics payload from the same helper the CLI uses.
        try:
            payload = build_metrics_payload(
                context_metadata=context.metadata,
                config=context.config,
                config_path=None,
                run_id=None,
                region=None,
                config_slug=None,
                wallclock_s=0.0,
                stage_results=[],
            )
        except Exception as exc:  # noqa: BLE001
            _append_quality_flag(
                context,
                "warning",
                f"Metrics payload build failed: {exc}; bulletin skipped",
            )
            return self._create_result(
                StageStatus.COMPLETED,
                data={"bulletin": "skipped (metrics payload failed)"},
                duration=time.time() - start,
            )

        # Load template (packaged default unless overridden).
        try:
            template_text = _load_template(bcfg.template)
        except FileNotFoundError as exc:
            _append_quality_flag(
                context,
                "warning",
                f"Bulletin template not found at {exc}; bulletin skipped",
            )
            return self._create_result(
                StageStatus.COMPLETED,
                data={"bulletin": "skipped (template missing)"},
                duration=time.time() - start,
            )

        issued = datetime.now(timezone.utc).strftime("%B %-d, %Y")
        prompt = build_prompt(template_text, payload, issued_date=issued)

        plots_generated = context.metadata.get("plume_sentinel_plots_generated", []) or []
        image_paths = [Path(p) for p in plots_generated] if bcfg.include_images else []

        # Call Claude.
        try:
            resp: BulletinResponse = generate_bulletin(
                prompt=prompt,
                metrics_json=payload,
                image_paths=image_paths,
                model=bcfg.model,
                api_key=api_key,
            )
        except Exception as exc:  # noqa: BLE001 - record + continue
            _append_quality_flag(
                context,
                "warning",
                f"Claude API call failed: {exc}; bulletin skipped",
            )
            return self._create_result(
                StageStatus.COMPLETED,
                data={"bulletin": "skipped (api error)"},
                duration=time.time() - start,
            )

        for missing in resp.skipped_images:
            _append_quality_flag(
                context,
                "info",
                f"Bulletin image not found: {missing}",
            )

        # Write file.
        output_dir = _get_output_dir(context)
        output_dir.mkdir(parents=True, exist_ok=True)
        bulletin_path = output_dir / bcfg.output_filename
        try:
            bulletin_path.write_text(resp.text)
        except OSError as exc:
            _append_quality_flag(
                context,
                "warning",
                f"Bulletin file write failed: {exc}",
            )
            return self._create_result(
                StageStatus.COMPLETED,
                data={"bulletin": "skipped (file write failed)"},
                duration=time.time() - start,
            )

        # Publish to MQTT if configured.
        mqtt_published = False
        if bcfg.mqtt is not None:
            try:
                publish_mqtt(
                    text=resp.text,
                    broker=bcfg.mqtt.broker,
                    topic=bcfg.mqtt.topic,
                    port=bcfg.mqtt.port,
                    qos=bcfg.mqtt.qos,
                )
                mqtt_published = True
            except Exception as exc:  # noqa: BLE001
                _append_quality_flag(
                    context,
                    "warning",
                    f"MQTT publish to {bcfg.mqtt.broker}:{bcfg.mqtt.port} failed: {exc}",
                )

        # Surface a structured summary into context.metadata so the metrics
        # payload extension (Task 10) can include it.
        context.metadata["plume_sentinel_bulletin"] = {
            "path": str(bulletin_path),
            "mode": "live",
            "model": resp.model,
            "input_tokens": resp.input_tokens,
            "cache_read_tokens": resp.cache_read_tokens,
            "output_tokens": resp.output_tokens,
        }

        return self._create_result(
            StageStatus.COMPLETED,
            data={
                "bulletin_path": str(bulletin_path),
                "mqtt_published": mqtt_published,
                "mode": "live",
                "model": resp.model,
                "input_tokens": resp.input_tokens,
                "cache_read_tokens": resp.cache_read_tokens,
                "output_tokens": resp.output_tokens,
            },
            duration=time.time() - start,
        )


def _load_template(template_path: str | None) -> str:
    """Load the bulletin template; default to the packaged copy."""
    if template_path:
        p = Path(template_path)
        if not p.is_file():
            raise FileNotFoundError(str(p))
        return p.read_text()
    from importlib.resources import files

    return (
        files(PlumeSentinelBulletinStage.DEFAULT_TEMPLATE_PACKAGE)
        .joinpath(PlumeSentinelBulletinStage.DEFAULT_TEMPLATE_NAME)
        .read_text()
    )


def _append_quality_flag(context: PipelineContext, severity: str, message: str) -> None:
    """Append a bulletin-category quality_flag onto context.metadata."""
    flags = context.metadata.setdefault("plume_sentinel_quality_flags", [])
    flags.append({"category": "bulletin", "severity": severity, "message": message})
