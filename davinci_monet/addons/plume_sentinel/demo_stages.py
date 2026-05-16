"""Demo stages for the PlumeSentinel add-on.

These stages replace the real load/prepare/plot stages when ``--demo-mode``
is set. They sleep, emit realistic progress messages, and populate stub
metadata so the bulletin stage downstream sees a plausible pipeline state.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from davinci_monet.pipeline.stages import BaseStage, PipelineContext, StageResult, StageStatus

LOAD_SLEEP_SECONDS = 3.0
PREPARE_SLEEP_SECONDS = 4.0
PLOT_SLEEP_SECONDS = 3.0


def _get_output_dir(context: PipelineContext) -> Path:
    analysis = context.config.get("analysis", {})
    return Path(analysis.get("output_dir", "output"))


class PlumeSentinelDemoLoadStage(BaseStage):
    """Simulate loading inputs (~3 s) without touching disk or network."""

    def __init__(self) -> None:
        super().__init__(name="load_inputs")

    def execute(self, context: PipelineContext) -> StageResult:
        start = time.time()
        stub_inputs = {
            "modis_terra_aod_event": "<demo-stub>",
            "goes_event_image": "<demo-stub>",
            "hms_smoke_event": "<demo-stub>",
        }
        steps = list(stub_inputs.keys())
        per_step = LOAD_SLEEP_SECONDS / max(len(steps), 1)
        for i, name in enumerate(steps, 1):
            context.log_progress(f"Loading input: {name} ({i}/{len(steps)})")
            time.sleep(per_step)

        context.metadata["plume_sentinel_loaded"] = stub_inputs
        return self._create_result(
            StageStatus.COMPLETED,
            data={"inputs_loaded": list(stub_inputs.keys()), "demo": True},
            duration=time.time() - start,
        )


class PlumeSentinelDemoPrepareStage(BaseStage):
    """Simulate geospatial preparation (~4 s) and populate stub provenance."""

    def __init__(self) -> None:
        super().__init__(name="prepare_geospatial")

    def execute(self, context: PipelineContext) -> StageResult:
        start = time.time()
        analysis = context.config.get("analysis", {})
        valid_time = str(analysis.get("start_time", "1970-01-01"))
        if "T" not in valid_time:
            valid_time = f"{valid_time}T00:00:00+00:00"

        input_datasets: list[dict[str, Any]] = [
            {
                "name": "MODIS L2 AOD (MOD04)",
                "version": "Collection 6.1",
                "agency": "NASA LAADS",
                "valid_time": valid_time,
                "granules": [],
            },
            {
                "name": "GOES-16 ABI L2 MCMIP",
                "agency": "NOAA NESDIS",
                "valid_time": valid_time,
                "granules": [],
            },
            {
                "name": "NOAA NESDIS HMS Smoke",
                "agency": "NOAA NESDIS",
                "valid_time": valid_time,
                "granules": [],
            },
        ]

        steps = [
            "assembling GOES RGB for goes_event_image",
            "cleaning HMS polygons for hms_smoke_event",
            "binning MODIS AOD to grid for modis_terra_aod_event",
        ]
        per_step = PREPARE_SLEEP_SECONDS / len(steps)
        for s in steps:
            context.log_progress(f"step: {s}")
            time.sleep(per_step)

        context.metadata["plume_sentinel_prepared"] = {
            name: "<demo-stub>"
            for name in ("modis_terra_aod_event", "goes_event_image", "hms_smoke_event")
        }
        context.metadata["plume_sentinel_input_datasets"] = input_datasets

        return self._create_result(
            StageStatus.COMPLETED,
            data={
                "inputs_prepared": list(context.metadata["plume_sentinel_prepared"].keys()),
                "demo": True,
            },
            duration=time.time() - start,
        )


class PlumeSentinelDemoPlotStage(BaseStage):
    """Simulate plotting (~3 s) by scanning ``output_dir`` for existing PNGs."""

    def __init__(self) -> None:
        super().__init__(name="plotting")

    def execute(self, context: PipelineContext) -> StageResult:
        start = time.time()
        output_dir = _get_output_dir(context)
        pngs = sorted(p for p in output_dir.glob("*.png")) if output_dir.is_dir() else []

        if pngs:
            per_step = PLOT_SLEEP_SECONDS / len(pngs)
            for i, p in enumerate(pngs, 1):
                context.log_progress(f"Plot: {p.stem} ({i}/{len(pngs)})")
                time.sleep(per_step)
                context.log_progress(f"done: saved to {p}")
        else:
            time.sleep(PLOT_SLEEP_SECONDS)

        paths = [str(p) for p in pngs]
        context.metadata["plume_sentinel_plots_generated"] = paths
        return self._create_result(
            StageStatus.COMPLETED,
            data={"plots_generated": paths, "demo": True},
            duration=time.time() - start,
        )
