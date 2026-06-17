"""AnalysesStage runs after LoadSources and before Pairing in the standard pipeline."""

from __future__ import annotations

from davinci_monet.pipeline.stages import (
    AnalysesStage,
    create_geometry_pipeline,
    create_standard_pipeline,
)


def _names(stages) -> list[str]:
    return [s.name for s in stages]


def test_analyses_in_standard_pipeline_order() -> None:
    names = _names(create_standard_pipeline())
    assert "analyses" in names
    assert names.index("analyses") == names.index("load_sources") + 1
    assert names.index("analyses") < names.index("pairing")


def test_analyses_in_geometry_pipeline() -> None:
    assert "analyses" in _names(create_geometry_pipeline())


def test_analyses_stage_exported() -> None:
    assert AnalysesStage().name == "analyses"
