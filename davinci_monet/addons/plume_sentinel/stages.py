"""Pipeline stage skeletons for Plume Sentinel workflow.

Each stage returns COMPLETED with empty data. Full implementations
will be added in a later task.
"""

from __future__ import annotations

import time

from davinci_monet.pipeline.stages import BaseStage, PipelineContext, StageResult, StageStatus


class PlumeSentinelLoadStage(BaseStage):
    """Load GOES, HMS, MODIS, and other input datasets."""

    def __init__(self) -> None:
        super().__init__(name="load_inputs")

    def execute(self, context: PipelineContext) -> StageResult:
        start = time.time()
        return self._create_result(StageStatus.COMPLETED, duration=time.time() - start)


class PlumeSentinelPrepareStage(BaseStage):
    """Prepare geospatial data (regridding, reprojection, etc.)."""

    def __init__(self) -> None:
        super().__init__(name="prepare_geospatial")

    def execute(self, context: PipelineContext) -> StageResult:
        start = time.time()
        return self._create_result(StageStatus.COMPLETED, duration=time.time() - start)


class PlumeSentinelPlotStage(BaseStage):
    """Generate true-color overlay and gridded field plots."""

    def __init__(self) -> None:
        super().__init__(name="plotting")

    def execute(self, context: PipelineContext) -> StageResult:
        start = time.time()
        return self._create_result(
            StageStatus.COMPLETED,
            data={"plots_generated": []},
            duration=time.time() - start,
        )
