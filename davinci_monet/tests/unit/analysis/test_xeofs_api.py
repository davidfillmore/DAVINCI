"""Pin the xeofs API surface EOFAnalysis relies on (models.EOF + scores/ev).

xeofs 2.x uses ``xeofs.models.EOF``; 3.x reorganised to ``xeofs.single.EOF``
(and also requires pandas>=2).  We pin to 2.x (``xeofs>=2.3,<3``) so the
import path here is ``xeofs.models``.
"""

from __future__ import annotations

import numpy as np
import xarray as xr


def test_xeofs_minimal_api() -> None:
    from xeofs.models import EOF  # import path used by EOFAnalysis (xeofs 2.x)

    rng = np.random.default_rng(0)
    da = xr.DataArray(
        rng.normal(size=(40, 6, 5)),
        dims=("time", "lat", "lon"),
        coords={"time": np.arange(40), "lat": np.linspace(0, 10, 6), "lon": np.linspace(0, 8, 5)},
    )
    model = EOF(n_modes=3, use_coslat=False, standardize=False)
    model.fit(da, dim="time")
    scores = model.scores()
    ev = model.explained_variance_ratio()
    assert "time" in scores.dims and "mode" in scores.dims
    assert "mode" in ev.dims
    assert int(ev.sizes["mode"]) == 3
