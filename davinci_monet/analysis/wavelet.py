"""Continuous wavelet transform (Torrence & Compo) of a 1-D series."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

from davinci_monet.analysis.base import DerivedAnalysis
from davinci_monet.analysis.reductions import (
    ar1_alpha,
    detrend_series,
    normalize_series,
    regularize,
    select_series,
)
from davinci_monet.core.protocols import DataGeometry
from davinci_monet.core.registry import analysis_registry

if TYPE_CHECKING:
    from davinci_monet.config.schema import WaveletSpec

logger = logging.getLogger(__name__)


@analysis_registry.register("wavelet")
class WaveletAnalysis(DerivedAnalysis):
    """Morlet CWT with AR(1) red-noise significance and cone of influence."""

    name = "wavelet"
    long_name = "Continuous Wavelet Transform"
    output_geometry = DataGeometry.SPECTRUM

    def analyze(self, data: xr.Dataset, spec: "WaveletSpec") -> xr.Dataset:
        import pycwt

        series = select_series(data, spec)
        regular, dt, unit, frac = regularize(series)
        if frac > 0.5:
            logger.warning(
                "wavelet input for '%s' was %.0f%% synthesized by time-axis "
                "regularization; AR(1) and power may be unreliable",
                spec.variable,
                100.0 * frac,
            )

        y = np.asarray(regular.values, dtype=float)
        y = detrend_series(y)
        alpha = ar1_alpha(y)            # estimate red noise BEFORE normalization
        y_norm, _std, _mean = normalize_series(y)

        mother = pycwt.Morlet(spec.omega0)
        s0 = spec.s0 if spec.s0 is not None else 2.0 * dt
        big_j = spec.j if spec.j is not None else -1
        wave, scales, freqs, coi, _, _ = pycwt.cwt(y_norm, dt, spec.dj, s0, big_j, mother)

        power = np.abs(wave) ** 2          # (scale, time)
        period = 1.0 / freqs               # (scale,)
        n = y_norm.size

        local_signif, _ = pycwt.significance(
            1.0, dt, scales, 0, alpha,
            significance_level=spec.significance_level, wavelet=mother,
        )
        power_sig = power / local_signif[:, None]

        global_power = power.mean(axis=1)  # (scale,)
        dof = n - scales
        global_signif, _ = pycwt.significance(
            1.0, dt, scales, 1, alpha,
            significance_level=spec.significance_level, dof=dof, wavelet=mother,
        )

        ds = xr.Dataset(
            {
                "power": (("time", "period"), power.T, {"kind": "power", "long_name": "Wavelet power"}),
                "power_significance": (("time", "period"), power_sig.T, {"kind": "power"}),
                "coi": (("time",), np.asarray(coi, dtype=float), {"kind": "coi", "units": unit}),
                "global_power": (("period",), global_power, {"kind": "global"}),
                "global_significance": (("period",), np.asarray(global_signif, dtype=float), {"kind": "global"}),
            },
            coords={
                "time": regular["time"].values,
                "period": ("period", period, {"units": unit, "long_name": "Period"}),
            },
        )
        ds.attrs["wavelet_quantity"] = spec.variable
        ds.attrs["dt"] = float(dt)
        ds.attrs["dt_units"] = unit
        return ds
