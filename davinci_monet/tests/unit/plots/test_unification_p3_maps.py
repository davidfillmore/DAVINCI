"""P3 map renderers are available under canonical names."""

from __future__ import annotations


def test_flight_track_registered() -> None:
    from davinci_monet.plots.registry import get_plotter_class
    from davinci_monet.plots.renderers.flight_track import FlightTrackPlotter

    assert get_plotter_class("flight_track") is FlightTrackPlotter


def test_lma_density_registered() -> None:
    from davinci_monet.plots.registry import get_plotter_class
    from davinci_monet.plots.renderers.lma_density import LMADensityPlotter

    assert get_plotter_class("lma_density") is LMADensityPlotter
