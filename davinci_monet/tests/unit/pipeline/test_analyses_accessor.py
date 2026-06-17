"""PipelineContext.analyses_config() returns typed specs for typed and dict configs."""

from __future__ import annotations

from davinci_monet.config.schema import EOFSpec, MonetConfig
from davinci_monet.pipeline.stages.base import PipelineContext

_ANALYSES = {"cam_O3_eof": {"type": "eof", "source": "cam", "variable": "O3"}}
_SOURCES = {"cam": {"type": "generic", "files": "x.nc", "variables": {"O3": {"units": "ppb"}}}}


def test_accessor_from_typed_config() -> None:
    cfg = MonetConfig(sources=_SOURCES, analyses=_ANALYSES)
    ctx = PipelineContext(config=cfg)
    out = ctx.analyses_config()
    assert isinstance(out["cam_O3_eof"], EOFSpec)


def test_accessor_from_dict_config() -> None:
    ctx = PipelineContext(config={"sources": _SOURCES, "analyses": _ANALYSES})
    out = ctx.analyses_config()
    assert isinstance(out["cam_O3_eof"], EOFSpec)


def test_accessor_empty() -> None:
    assert PipelineContext(config={}).analyses_config() == {}
