"""Pin the pycwt API surface WaveletAnalysis relies on."""

from __future__ import annotations

import numpy as np


def test_pycwt_minimal_api() -> None:
    import pycwt

    nt, dt = 256, 1.0
    t = np.arange(nt)
    sig = np.sin(2 * np.pi * t / 16.0)
    sig = (sig - sig.mean()) / sig.std()

    alpha, _, _ = pycwt.ar1(sig)
    alpha = float(alpha)
    mother = pycwt.Morlet(6)
    wave, scales, freqs, coi, _, _ = pycwt.cwt(sig, dt, 0.25, 2 * dt, -1, mother)
    power = np.abs(wave) ** 2
    assert power.shape == (scales.size, nt)
    assert coi.shape == (nt,)

    signif, _ = pycwt.significance(1.0, dt, scales, 0, alpha, significance_level=0.95, wavelet=mother)
    assert signif.shape == (scales.size,)

    dof = nt - scales
    glbl, _ = pycwt.significance(1.0, dt, scales, 1, alpha, significance_level=0.95, dof=dof, wavelet=mother)
    assert glbl.shape == (scales.size,)
