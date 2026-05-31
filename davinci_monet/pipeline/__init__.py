"""Pipeline module for orchestrating analysis workflows.

This module provides the components for building and executing
analysis pipelines that flow data through stages.
"""

from davinci_monet.pipeline.parallel import (
    ParallelExecutor,
    ParallelPairingExecutor,
    ParallelResult,
    parallel_process_files,
)
from davinci_monet.pipeline.runner import (
    PipelineBuilder,
    PipelineResult,
    PipelineRunner,
    run_analysis,
)
from davinci_monet.pipeline.stages import (
    BaseStage,
    LoadModelsStage,
    LoadObservationsStage,
    LoadSourcesStage,
    ObsPlottingStage,
    ObsStatisticsStage,
    PairingStage,
    PipelineContext,
    PlottingStage,
    SaveResultsStage,
    Stage,
    StageResult,
    StageStatus,
    StatisticsStage,
    create_obs_pipeline,
    create_standard_pipeline,
)

__all__ = [
    # Stages
    "Stage",
    "BaseStage",
    "StageResult",
    "StageStatus",
    "PipelineContext",
    "LoadModelsStage",
    "LoadObservationsStage",
    "LoadSourcesStage",
    "PairingStage",
    "StatisticsStage",
    "PlottingStage",
    "ObsPlottingStage",
    "ObsStatisticsStage",
    "SaveResultsStage",
    "create_obs_pipeline",
    "create_standard_pipeline",
    # Runner
    "PipelineRunner",
    "PipelineResult",
    "PipelineBuilder",
    "run_analysis",
    # Parallel
    "ParallelExecutor",
    "ParallelPairingExecutor",
    "ParallelResult",
    "parallel_process_files",
]
