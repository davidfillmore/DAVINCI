"""Analysis spec models parse and dispatch by type."""

from __future__ import annotations

import pytest

from davinci_monet.config.schema import (
    EOFSpec,
    PointReduce,
    WaveletSpec,
    build_analysis_spec,
)


def test_build_eof_spec() -> None:
    spec = build_analysis_spec({"type": "eof", "source": "cam", "variable": "O3", "n_modes": 6})
    assert isinstance(spec, EOFSpec)
    assert spec.n_modes == 6
    assert spec.standardize is False
    assert spec.rotation == "none"


def test_build_wavelet_spec_with_point_reduce() -> None:
    spec = build_analysis_spec(
        {"type": "wavelet", "source": "cam", "variable": "O3", "reduce": {"point": [40.0, -105.0]}}
    )
    assert isinstance(spec, WaveletSpec)
    assert isinstance(spec.reduce, PointReduce)
    assert spec.reduce.point == (40.0, -105.0)


def test_wavelet_default_reduce_is_area_mean() -> None:
    spec = build_analysis_spec({"type": "wavelet", "source": "cam", "variable": "O3"})
    assert spec.reduce == "area_mean"


def test_unknown_analysis_type_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown analysis type"):
        build_analysis_spec({"type": "bogus", "source": "cam", "variable": "O3"})
