"""Guard tests for single-series renderers.

Verifies that FlightTrackPlotter and LMADensityPlotter raise NotImplementedError
when called with != 1 series, as required by their single-source contracts.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from davinci_monet.plots.base import build_series


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
