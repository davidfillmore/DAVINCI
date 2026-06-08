"""Pipeline stage definitions.

This package provides the Stage protocol and concrete stage implementations
for the analysis pipeline. Each stage is a composable unit of work that
transforms data through the analysis workflow.

The implementation is split across submodules for maintainability; this
``__init__`` re-exports the full public surface so existing
``from davinci_monet.pipeline.stages import X`` imports keep working:

- :mod:`~davinci_monet.pipeline.stages.base` — framework primitives
  (:class:`StageStatus`, :class:`StageResult`, :class:`Stage`,
  :class:`PipelineContext`, :class:`SourceData`, :class:`SourcePairJob`,
  :class:`BaseStage`).
- :mod:`~davinci_monet.pipeline.stages.helpers` — module-level helpers
  (role tagging, variable resolution, formatters).
- :mod:`~davinci_monet.pipeline.stages.load` — :class:`LoadSourcesStage`.
- :mod:`~davinci_monet.pipeline.stages.pair` — :class:`PairingStage`.
- :mod:`~davinci_monet.pipeline.stages.stats` — :class:`StatisticsStage`.
- :mod:`~davinci_monet.pipeline.stages.plot` — :class:`PlottingStage`.
- :mod:`~davinci_monet.pipeline.stages.io` — :class:`SaveResultsStage`.
- :mod:`~davinci_monet.pipeline.stages.summary` — :class:`SummaryStage`.
- :mod:`~davinci_monet.pipeline.stages.factory` — pipeline constructors.
"""

from __future__ import annotations

from davinci_monet.pipeline.stages.base import (
    BaseStage,
    PipelineContext,
    SourceData,
    SourcePairJob,
    Stage,
    StageResult,
    StageStatus,
)
from davinci_monet.pipeline.stages.factory import (
    create_obs_pipeline,
    create_standard_pipeline,
)
from davinci_monet.pipeline.stages.helpers import (
    _format_duration,
    _format_size,
    iter_single_source_datasets,
    resolve_paired_var_names,
    tag_paired_roles,
    tag_source_roles,
)
from davinci_monet.pipeline.stages.io import SaveResultsStage
from davinci_monet.pipeline.stages.load import LoadSourcesStage
from davinci_monet.pipeline.stages.pair import PairingStage
from davinci_monet.pipeline.stages.plot import PlottingStage
from davinci_monet.pipeline.stages.stats import StatisticsStage
from davinci_monet.pipeline.stages.summary import SummaryStage

__all__ = [
    # Framework primitives
    "Stage",
    "BaseStage",
    "StageStatus",
    "StageResult",
    "PipelineContext",
    "SourceData",
    "SourcePairJob",
    # Module-level helpers
    "tag_paired_roles",
    "tag_source_roles",
    "iter_single_source_datasets",
    "resolve_paired_var_names",
    "_format_size",
    "_format_duration",
    # Stage classes
    "LoadSourcesStage",
    "PairingStage",
    "StatisticsStage",
    "PlottingStage",
    "SaveResultsStage",
    "SummaryStage",
    # Pipeline factories
    "create_standard_pipeline",
    "create_obs_pipeline",
]
