"""SPECTRUM geometry exists for wavelet (time, period) outputs."""

from __future__ import annotations

from davinci_monet.core.protocols import DataGeometry


def test_spectrum_geometry_member() -> None:
    assert DataGeometry.SPECTRUM.name == "SPECTRUM"
    assert DataGeometry.SPECTRUM not in {
        DataGeometry.POINT,
        DataGeometry.TRACK,
        DataGeometry.PROFILE,
        DataGeometry.SWATH,
        DataGeometry.GRID,
    }
