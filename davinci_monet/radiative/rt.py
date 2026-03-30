"""Shortwave radiative transfer for smoke plume surface dimming.

Provides a δ-Eddington two-stream solver (Meador & Weaver 1980),
spectral utilities (Rayleigh scattering, Angstrom scaling), and
solar geometry for semi-empirical smoke radiative impact estimation.

References
----------
Meador, W.E. and W.R. Weaver, 1980: Two-Stream Approximations to
  Radiative Transfer in Planetary Atmospheres: A Unified Description
  of Existing Methods and a New Improvement. J. Atmos. Sci., 37,
  630–643.
Toon, O.B., C.P. McKay, T.P. Ackerman, and K. Santhanam, 1989:
  Rapid Calculation of Radiative Heating Rates and Photodissociation
  Rates in Inhomogeneous Multiple Scattering Atmospheres. J. Geophys.
  Res., 94, 16287–16301.
"""

from __future__ import annotations

import numpy as np


# ── δ-Eddington two-stream solver ───────────────────────────────────────────


def delta_eddington_2stream(tau, omega, g, mu0, albedo=0.0):
    """Meador & Weaver (1980) δ-Eddington two-stream for a single layer.

    Solves the two-stream equations for diffuse fluxes F⁺ (upward) and
    F⁻ (downward) in a homogeneous layer illuminated by direct beam
    with cos(SZA) = μ₀.

    The two-stream equations (τ increasing downward)::

        dF⁺/dτ = γ₁ F⁺ − γ₂ F⁻ − ω' γ₃ exp(−τ/μ₀)
        dF⁻/dτ = γ₂ F⁺ − γ₁ F⁻ + ω' γ₄ exp(−τ/μ₀)

    Homogeneous solution eigenmodes (eigenvalues ±k)::

        F⁺(τ) = c₁ Γ exp(−kτ) + c₂ Γ' exp(kτ)
        F⁻(τ) = c₁ exp(−kτ) + c₂ exp(kτ)

    where Γ = γ₂/(γ₁+k) and Γ' = γ₂/(γ₁−k).

    Parameters
    ----------
    tau : array_like
        Total extinction optical depth at 550 nm (or band-effective).
    omega : array_like
        Single-scattering albedo (0–1).
    g : array_like
        Asymmetry parameter (−1 to 1, typically 0.5–0.7 for smoke).
    mu0 : array_like
        Cosine of solar zenith angle (0–1).
    albedo : array_like, optional
        Surface albedo (0–1). Default 0.

    Returns
    -------
    R_frac : ndarray
        Fraction of incident horizontal flux reflected upward.
    T_frac : ndarray
        Fraction of incident horizontal flux reaching the surface.
    A_frac : ndarray
        Fraction of incident horizontal flux absorbed in the layer.

    Notes
    -----
    Energy conservation: R_frac + T_frac + A_frac = 1 (to machine
    precision for all inputs).
    """
    tau = np.asarray(tau, dtype=np.float64)
    omega = np.asarray(omega, dtype=np.float64)
    g = np.asarray(g, dtype=np.float64)
    mu0 = np.asarray(mu0, dtype=np.float64)
    albedo = np.asarray(albedo, dtype=np.float64)

    # δ-Eddington rescaling
    f = g ** 2
    ts = np.maximum((1.0 - omega * f) * tau, 1e-10)
    ws = np.clip(
        omega * (1.0 - f) / np.maximum(1.0 - omega * f, 1e-10),
        0, 0.999999,
    )
    gs = g / np.maximum(1.0 + g, 1e-10)

    # Eddington two-stream coefficients (Toon et al. 1989 Table 1)
    g1 = (7.0 - ws * (4.0 + 3.0 * gs)) / 4.0
    g2 = -(1.0 - ws * (4.0 - 3.0 * gs)) / 4.0
    g3 = (2.0 - 3.0 * gs * mu0) / 4.0
    g4 = 1.0 - g3

    k = np.sqrt(np.maximum(g1 ** 2 - g2 ** 2, 1e-20))
    Gp = g2 / (g1 + k)  # Γ  for exp(−kτ) eigenmode
    Gm = g2 / np.where(  # Γ' for exp(+kτ) eigenmode
        np.abs(g1 - k) < 1e-10, 1e-10, g1 - k
    )

    # Clamp exponentials to avoid overflow
    kt = np.minimum(k * ts, 500.0)
    tm = np.minimum(ts / mu0, 500.0)
    ekt = np.exp(kt)
    emkt = np.exp(-kt)
    etm = np.exp(-tm)

    # ── Particular solution for direct beam source ──
    # (γ₁ + 1/μ₀) Z⁺ − γ₂ Z⁻ = ω' γ₃
    # −γ₂ Z⁺ + (γ₁ − 1/μ₀) Z⁻ = ω' γ₄
    # det = k² − 1/μ₀²
    det_ps = k ** 2 - 1.0 / mu0 ** 2
    det_ps = np.where(np.abs(det_ps) < 1e-10, -1e-10, det_ps)

    Zp = ws * ((g1 - 1.0 / mu0) * g3 + g2 * g4) / det_ps
    Zm = ws * ((g1 + 1.0 / mu0) * g4 + g2 * g3) / det_ps

    # ── Boundary value problem ──
    # F⁺(τ) = c₁ Γ exp(−kτ) + c₂ Γ' exp(kτ) + Z⁺ exp(−τ/μ₀)
    # F⁻(τ) = c₁ exp(−kτ) + c₂ exp(kτ) + Z⁻ exp(−τ/μ₀)
    #
    # Top:    F⁻(0) = 0  →  c₁ + c₂ = −Z⁻
    # Bottom: F⁺(τ*) = α [F⁻(τ*) + μ₀ exp(−τ*/μ₀)]
    #   →  c₁ emkt(Γ−α) + c₂ ekt(Γ'−α) = (α(Z⁻+μ₀)−Z⁺) etm
    a_A1 = np.ones_like(k)
    a_A2 = np.ones_like(k)
    b_A = -Zm

    a_B1 = emkt * (Gp - albedo)
    a_B2 = ekt * (Gm - albedo)
    b_B = (albedo * (Zm + mu0) - Zp) * etm

    det_bc = a_A1 * a_B2 - a_A2 * a_B1
    det_bc = np.where(np.abs(det_bc) < 1e-30, 1e-30, det_bc)

    c1 = (b_A * a_B2 - a_A2 * b_B) / det_bc
    c2 = (a_A1 * b_B - b_A * a_B1) / det_bc

    # ── Extract fluxes ──
    R = c1 * Gp + c2 * Gm + Zp                       # F⁺(0)
    F_diff_bot = c1 * emkt + c2 * ekt + Zm * etm      # F⁻(τ*)
    F_dir_bot = mu0 * etm                              # direct at bottom

    R_frac = np.clip(R / mu0, 0.0, 1.0)
    T_frac = np.clip((F_diff_bot + F_dir_bot) / mu0, 0.0, 1.5)
    A_frac = 1.0 - R_frac - T_frac

    return R_frac, T_frac, A_frac


# ── Spectral utilities ───────────────────────────────────────────────────────


def rayleigh_tau(lambda_um, p_hpa=1013.25):
    """Rayleigh scattering optical depth.

    Parameters
    ----------
    lambda_um : float
        Wavelength in micrometers.
    p_hpa : float, optional
        Surface pressure in hPa. Default 1013.25.

    Returns
    -------
    float
        Rayleigh optical depth (dimensionless).
    """
    return 0.008569 * lambda_um ** (-4) * (p_hpa / 1013.25)


def smoke_aod_at_wavelength(aod_550, lambda_um, angstrom=1.8):
    """Scale smoke AOD from 550 nm to another wavelength via Angstrom law.

    Parameters
    ----------
    aod_550 : array_like
        AOD at 550 nm.
    lambda_um : float
        Target wavelength in micrometers.
    angstrom : float, optional
        Angstrom exponent. Default 1.8 (typical aged smoke).

    Returns
    -------
    ndarray
        AOD at the target wavelength.
    """
    return np.asarray(aod_550) * (0.55 / lambda_um) ** angstrom


def combine_aerosol_rayleigh(tau_aer, ssa_aer, g_aer, tau_ray):
    """Combine aerosol and Rayleigh into bulk optical properties.

    Rayleigh scattering has SSA = 1 and g = 0.

    Parameters
    ----------
    tau_aer : array_like
        Aerosol extinction optical depth.
    ssa_aer : float or array_like
        Aerosol single-scattering albedo.
    g_aer : float or array_like
        Aerosol asymmetry parameter.
    tau_ray : float
        Rayleigh optical depth.

    Returns
    -------
    tau_tot, ssa_tot, g_tot : ndarrays
        Combined optical properties.
    """
    tau_aer = np.asarray(tau_aer, dtype=np.float64)
    tau_tot = tau_aer + tau_ray
    tau_tot = np.maximum(tau_tot, 1e-10)
    ssa_tot = (ssa_aer * tau_aer + tau_ray) / tau_tot
    ssa_tot = np.minimum(ssa_tot, 0.999999)
    g_tot = np.where(
        ssa_tot * tau_tot > 1e-20,
        g_aer * ssa_aer * tau_aer / (ssa_tot * tau_tot),
        0.0,
    )
    return tau_tot, ssa_tot, g_tot


# ── Solar geometry ───────────────────────────────────────────────────────────


def daily_mean_coszen(lat_deg, doy):
    """Daily-mean cosine of solar zenith angle.

    Integrates cos(SZA) over daylight hours using the standard
    solar declination formula.

    Parameters
    ----------
    lat_deg : array_like
        Latitude in degrees (positive north).
    doy : int or float
        Day of year (1 = Jan 1).

    Returns
    -------
    ndarray
        Daily-mean cos(SZA), floored at 0.05.
    """
    lat = np.deg2rad(np.asarray(lat_deg, dtype=np.float64))
    decl = np.deg2rad(23.45 * np.sin(np.deg2rad(360.0 / 365.0 * (doy - 81))))
    cos_ha0 = np.clip(-np.tan(lat) * np.tan(decl), -1.0, 1.0)
    ha0 = np.arccos(cos_ha0)
    mu_bar = (1.0 / np.pi) * (
        ha0 * np.sin(lat) * np.sin(decl)
        + np.cos(lat) * np.cos(decl) * np.sin(ha0)
    )
    return np.maximum(mu_bar, 0.05)
