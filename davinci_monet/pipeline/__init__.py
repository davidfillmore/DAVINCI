"""Pipeline module for orchestrating analysis workflows.

This module provides the components for building and executing
analysis pipelines that flow data through stages.
"""

from davinci_monet.pipeline.runner import (
    PipelineBuilder,
    PipelineResult,
    PipelineRunner,
    run_analysis,
)
from davinci_monet.pipeline.stages import (
    BaseStage,
    LoadSourcesStage,
    PairingStage,
    PipelineContext,
    PlottingStage,
    SaveResultsStage,
    Stage,
    StageResult,
    StageStatus,
    StatisticsStage,
    create_geometry_pipeline,
    create_standard_pipeline,
)

__all__ = [
    # Stages
    "Stage",
    "BaseStage",
    "StageResult",
    "StageStatus",
    "PipelineContext",
    "LoadSourcesStage",
    "PairingStage",
    "StatisticsStage",
    "PlottingStage",
    "SaveResultsStage",
    "create_geometry_pipeline",
    "create_standard_pipeline",
    # Runner
    "PipelineRunner",
    "PipelineResult",
    "PipelineBuilder",
    "run_analysis",
]
