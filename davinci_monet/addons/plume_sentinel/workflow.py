"""Workflow factory for Plume Sentinel pipeline."""

from __future__ import annotations

from davinci_monet.addons.plume_sentinel.stages import (
    PlumeSentinelLoadStage,
    PlumeSentinelPlotStage,
    PlumeSentinelPrepareStage,
)
from davinci_monet.pipeline.stages import BaseStage


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
