"""Pipeline stages for the PlumeSentinel add-on workflow.

Each stage loads, prepares, or plots PlumeSentinel data using the
add-on's own loaders, processors, and renderers.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from davinci_monet.addons.plume_sentinel.loaders import load_input
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

        return self._create_result(
            StageStatus.COMPLETED,
            data={"plots_generated": all_paths},
            duration=time.time() - start,
        )
