"""DerivedAnalysis is an ABC: a concrete subclass implements analyze()."""

from __future__ import annotations

import numpy as np
import xarray as xr

from davinci_monet.analysis import DerivedAnalysis
from davinci_monet.core.protocols import DataGeometry
from davinci_monet.core.registry import analysis_registry


def test_concrete_analysis_runs_and_registers() -> None:
    @analysis_registry.register("identity_t3")
    class Identity(DerivedAnalysis):
        name = "identity_t3"
        long_name = "Identity"
        output_geometry = DataGeometry.GRID

        def analyze(self, data: xr.Dataset, spec: object) -> xr.Dataset:
            return data

    ds = xr.Dataset({"x": ("t", np.arange(3.0))})
    out = analysis_registry.get("identity_t3")().analyze(ds, None)
    assert out is ds
    analysis_registry.unregister("identity_t3")
