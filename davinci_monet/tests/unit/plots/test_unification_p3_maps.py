"""P3 — obs_flight_track / obs_lma_density promoted to canonical renderers."""

from __future__ import annotations

import warnings


def test_flight_track_alias() -> None:
    from davinci_monet.plots.registry import get_plotter_class
    from davinci_monet.plots.renderers.flight_track import FlightTrackPlotter

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert get_plotter_class("obs_flight_track") is FlightTrackPlotter


def test_lma_density_alias() -> None:
    from davinci_monet.plots.registry import get_plotter_class
    from davinci_monet.plots.renderers.lma_density import LMADensityPlotter

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert get_plotter_class("obs_lma_density") is LMADensityPlotter
