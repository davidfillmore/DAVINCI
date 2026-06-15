"""Guard tests for single-series renderers.

Verifies that FlightTrackPlotter and LMADensityPlotter raise NotImplementedError
when called with != 1 series, as required by their single-source contracts.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from davinci_monet.plots import (
    FlightTrackPlotter,
    HistogramPlotter,
    LMADensityPlotter,
    SpatialPlotter,
    VerticalProfilePlotter,
)
from davinci_monet.plots.base import build_series
from davinci_monet.tests.synthetic.geometries import create_point_geometries


def _track_ds() -> xr.Dataset:
    n = 10
    return xr.Dataset(
        {"O3": (["time"], np.arange(n, dtype=float))},
        coords={
            "time": pd.to_datetime(["2012-05-29"] * n),
            "latitude": ("time", np.linspace(34, 36, n)),
            "longitude": ("time", np.linspace(-98, -96, n)),
            "altitude": ("time", np.linspace(500, 8000, n), {"units": "m"}),
        },
    )


def test_flight_track_requires_one_series() -> None:
    from davinci_monet.plots.renderers.flight_track import FlightTrackPlotter

    s = build_series(_track_ds(), "O3")
    with pytest.raises(NotImplementedError, match="1 series"):
        FlightTrackPlotter().render(s + s)


def test_lma_density_requires_one_series() -> None:
    from davinci_monet.plots.renderers.lma_density import LMADensityPlotter

    s = build_series(_track_ds(), "O3")
    with pytest.raises(NotImplementedError, match="1 series"):
        LMADensityPlotter().render(s + s)


@pytest.mark.parametrize(
    "plotter_cls",
    [
        SpatialPlotter,
        FlightTrackPlotter,
        HistogramPlotter,
        LMADensityPlotter,
        VerticalProfilePlotter,
    ],
)
def test_single_source_plotters_reject_multiple_series(plotter_cls) -> None:
    ds = create_point_geometries(variables=["O3", "NO2"])
    series = build_series(ds, ["O3", "NO2"])

    with pytest.raises(NotImplementedError, match="requires exactly 1 series"):
        plotter_cls().render(series)
