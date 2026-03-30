"""Progressive semi-empirical model levels for RT comparison.

Defines 5 levels of increasing physical complexity for estimating
surface shortwave dimming from smoke aerosol, from basic Beer-Lambert
to full spectral two-stream with Rayleigh scattering.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from davinci_monet.radiative.rt import (
    combine_aerosol_rayleigh,
    daily_mean_coszen,
    delta_eddington_2stream,
    rayleigh_tau,
    smoke_aod_at_wavelength,
)

SSA = 0.92
ASYM = 0.65

# (name, wavelength_um, f_solar, ssa_band, g_band)
BANDS = [
    ("UV+VIS", 0.50, 0.46, 0.88, 0.68),
    ("Near-IR", 1.00, 0.37, 0.95, 0.60),
    ("SW-IR", 2.00, 0.17, 0.98, 0.50),
]

LEVEL_NAMES = [
    "L0: Beer-Lambert",
    "L1: + true SZA",
    "L2: + \u03b4-Eddington",
    "L3: + albedo",
    "L4: + spectral + Rayleigh",
]


def level0_beer_lambert(
    tau: np.ndarray,
    s0: np.ndarray,
    **kw: np.ndarray,
) -> np.ndarray:
    """L0: Basic Beer-Lambert with fixed mu=0.5."""
    mu = 0.5
    extinct = 1.0 - np.exp(-tau / mu)
    fwd_frac = SSA * (1 + ASYM) / 2
    return -s0 * extinct * (1.0 - fwd_frac)


def level1_true_sza(
    tau: np.ndarray,
    s0: np.ndarray,
    mu_bar: np.ndarray | None = None,
    **kw: np.ndarray,
) -> np.ndarray:
    """L1: Beer-Lambert with true daily-mean solar zenith angle."""
    extinct = 1.0 - np.exp(-tau / mu_bar)
    fwd_frac = SSA * (1 + ASYM) / 2
    return -s0 * extinct * (1.0 - fwd_frac)


def level2_two_stream(
    tau: np.ndarray,
    s0: np.ndarray,
    mu_bar: np.ndarray | None = None,
    **kw: np.ndarray,
) -> np.ndarray:
    """L2: delta-Eddington two-stream solver, no surface albedo."""
    _, T_frac, _ = delta_eddington_2stream(tau, SSA, ASYM, mu_bar, albedo=0.0)
    return s0 * (T_frac - 1.0)


def level3_two_stream_albedo(
    tau: np.ndarray,
    s0: np.ndarray,
    mu_bar: np.ndarray | None = None,
    albedo: np.ndarray | None = None,
    **kw: np.ndarray,
) -> np.ndarray:
    """L3: delta-Eddington two-stream with surface albedo."""
    _, T_frac, _ = delta_eddington_2stream(tau, SSA, ASYM, mu_bar, albedo=albedo)
    return s0 * (T_frac - 1.0)


def level4_spectral(
    tau_550: np.ndarray,
    s0: np.ndarray,
    mu_bar: np.ndarray | None = None,
    albedo: np.ndarray | None = None,
    **kw: np.ndarray,
) -> np.ndarray:
    """L4: 3-band spectral integration with Rayleigh scattering."""
    dimming = np.zeros_like(tau_550)
    for _name, lam, f_solar, ssa_band, g_band in BANDS:
        tau_aer = smoke_aod_at_wavelength(tau_550, lam)
        tau_ray = rayleigh_tau(lam)
        tau_tot, ssa_tot, g_tot = combine_aerosol_rayleigh(tau_aer, ssa_band, g_band, tau_ray)
        _, T_frac, _ = delta_eddington_2stream(tau_tot, ssa_tot, g_tot, mu_bar, albedo)
        dimming += f_solar * s0 * (T_frac - 1.0)
    return dimming


# Type alias for the level functions
LevelFunc = Callable[..., np.ndarray]

ALL_LEVELS: list[LevelFunc] = [
    level0_beer_lambert,
    level1_true_sza,
    level2_two_stream,
    level3_two_stream_albedo,
    level4_spectral,
]


def compute_rt_levels(
    smoke_aod: np.ndarray,
    s0: np.ndarray,
    mu_bar: np.ndarray,
    albedo: np.ndarray,
) -> list[np.ndarray]:
    """Compute all 5 progressive model levels.

    Parameters
    ----------
    smoke_aod : np.ndarray
        Smoke AOD at 550 nm on the MERRA-2 grid.
    s0 : np.ndarray
        Clear-sky downwelling SW at surface (W/m2).
    mu_bar : np.ndarray
        Daily-mean cosine of solar zenith angle.
    albedo : np.ndarray
        Surface albedo (0-1).

    Returns
    -------
    list[np.ndarray]
        List of 5 arrays, one per model level, each giving the
        estimated surface SW dimming (W/m2, negative = darkening).
    """
    kw = dict(s0=s0, mu_bar=mu_bar, albedo=albedo)
    return [fn(smoke_aod, **kw) for fn in ALL_LEVELS]
