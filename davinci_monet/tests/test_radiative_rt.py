"""Unit tests for the rt module (shortwave radiative transfer).

Usage
-----
>>> conda activate davinci-monet
>>> python -m pytest davinci_monet/tests/test_radiative_rt.py -v
"""

from __future__ import annotations

import numpy as np

from davinci_monet.radiative.rt import (
    combine_aerosol_rayleigh,
    daily_mean_coszen,
    delta_eddington_2stream,
    rayleigh_tau,
    smoke_aod_at_wavelength,
)

ATOL = 1e-6  # absolute tolerance for energy conservation


# ── Two-stream: energy conservation ─────────────────────────────────────────


def test_energy_conservation_basic():
    """R + T + A = 1 for a range of inputs."""
    cases = [
        (0.001, 0.92, 0.65, 0.5, 0.0),
        (0.5, 0.92, 0.65, 0.5, 0.0),
        (3.0, 0.92, 0.65, 0.5, 0.0),
        (10.0, 0.92, 0.65, 0.5, 0.0),
        (100.0, 0.0, 0.0, 0.5, 0.0),
        (0.5, 1.0, 0.0, 0.5, 0.0),
        (0.5, 0.92, 0.65, 0.8, 0.0),
        (0.5, 0.92, 0.65, 0.3, 0.0),
        (1.0, 0.85, 0.70, 0.6, 0.0),
    ]
    for tau, omega, g, mu0, alb in cases:
        R, T, A = delta_eddington_2stream(tau, omega, g, mu0, alb)
        total = R + T + A
        assert abs(total - 1.0) < ATOL, (
            f"Energy not conserved: τ={tau} ω={omega} g={g} μ₀={mu0} "
            f"R={R:.6f} T={T:.6f} A={A:.6f} sum={total:.6f}"
        )


def test_energy_conservation_with_albedo():
    """R + T + A = 1 with nonzero surface albedo."""
    cases = [
        (0.5, 0.92, 0.65, 0.5, 0.1),
        (0.5, 0.92, 0.65, 0.5, 0.3),
        (1.0, 0.92, 0.65, 0.5, 0.5),
        (2.0, 0.88, 0.68, 0.6, 0.2),
    ]
    for tau, omega, g, mu0, alb in cases:
        R, T, A = delta_eddington_2stream(tau, omega, g, mu0, alb)
        total = R + T + A
        assert abs(total - 1.0) < ATOL, (
            f"Energy not conserved with albedo: τ={tau} α={alb} " f"sum={total:.6f}"
        )


def test_energy_conservation_arrays():
    """Energy conservation for array inputs."""
    tau = np.array([0.1, 0.5, 1.0, 3.0, 10.0])
    R, T, A = delta_eddington_2stream(tau, 0.92, 0.65, 0.5)
    total = R + T + A
    assert np.allclose(total, 1.0, atol=ATOL), f"Array energy: {total}"


# ── Two-stream: physical limits ─────────────────────────────────────────────


def test_thin_layer_transmits():
    """Very thin layer should transmit nearly all light."""
    R, T, A = delta_eddington_2stream(0.001, 0.92, 0.65, 0.5)
    assert T > 0.99, f"Thin layer T={T:.4f}, expected > 0.99"
    assert R < 0.01, f"Thin layer R={R:.4f}, expected < 0.01"


def test_thick_absorber_absorbs():
    """Thick purely absorbing layer should absorb everything."""
    R, T, A = delta_eddington_2stream(100.0, 0.0, 0.0, 0.5)
    assert A > 0.99, f"Thick absorber A={A:.4f}, expected > 0.99"
    assert T < 0.01, f"Thick absorber T={T:.4f}, expected < 0.01"


def test_conservative_no_absorption():
    """Conservative scattering (ω=1) should have zero absorption."""
    R, T, A = delta_eddington_2stream(0.5, 0.999999, 0.0, 0.5)
    assert abs(A) < 0.01, f"Conservative A={A:.6f}, expected ≈ 0"


def test_thick_smoke_moderate_transmission():
    """τ=3 with SSA=0.92 should still transmit ~30%."""
    R, T, A = delta_eddington_2stream(3.0, 0.92, 0.65, 0.5)
    assert 0.15 < T < 0.50, f"Thick smoke T={T:.4f}, expected 0.15–0.50"
    assert 0.20 < R < 0.50, f"Thick smoke R={R:.4f}, expected 0.20–0.50"


def test_higher_mu_more_transmission():
    """Higher μ₀ (sun higher) → shorter path → more transmission."""
    _, T_low, _ = delta_eddington_2stream(1.0, 0.92, 0.65, 0.3)
    _, T_high, _ = delta_eddington_2stream(1.0, 0.92, 0.65, 0.8)
    assert T_high > T_low, f"T(μ₀=0.8)={T_high:.4f} should > T(μ₀=0.3)={T_low:.4f}"


def test_albedo_increases_reflection():
    """Surface albedo should increase total reflected flux."""
    R0, _, _ = delta_eddington_2stream(0.5, 0.92, 0.65, 0.5, albedo=0.0)
    R3, _, _ = delta_eddington_2stream(0.5, 0.92, 0.65, 0.5, albedo=0.3)
    assert R3 > R0, f"R(α=0.3)={R3:.4f} should > R(α=0)={R0:.4f}"


def test_zero_tau():
    """Zero optical depth → perfect transmission."""
    R, T, A = delta_eddington_2stream(0.0, 0.92, 0.65, 0.5)
    assert T > 0.999, f"Zero τ: T={T:.4f}"
    assert R < 0.001, f"Zero τ: R={R:.4f}"


def test_nonnegative_outputs():
    """R, T, A should all be non-negative for physical inputs."""
    taus = [0.01, 0.1, 0.5, 1.0, 3.0, 10.0]
    for tau in taus:
        R, T, A = delta_eddington_2stream(tau, 0.92, 0.65, 0.5)
        assert R >= -ATOL, f"R={R:.6f} < 0 at τ={tau}"
        assert T >= -ATOL, f"T={T:.6f} < 0 at τ={tau}"
        assert A >= -ATOL, f"A={A:.6f} < 0 at τ={tau}"


# ── Spectral utilities ───────────────────────────────────────────────────────


def test_rayleigh_wavelength_dependence():
    """Rayleigh τ should decrease strongly with wavelength."""
    t_uv = rayleigh_tau(0.35)
    t_vis = rayleigh_tau(0.55)
    t_nir = rayleigh_tau(1.0)
    assert t_uv > t_vis > t_nir, "Rayleigh should decrease with λ"
    assert t_uv / t_nir > 50, "UV/NIR Rayleigh ratio should be > 50"


def test_rayleigh_pressure_scaling():
    """Rayleigh τ should scale linearly with pressure."""
    t_full = rayleigh_tau(0.55, p_hpa=1013.25)
    t_half = rayleigh_tau(0.55, p_hpa=506.625)
    assert abs(t_half / t_full - 0.5) < 0.001


def test_smoke_aod_angstrom():
    """Smoke AOD should increase at shorter wavelengths."""
    aod_uv = smoke_aod_at_wavelength(1.0, 0.35)
    aod_vis = smoke_aod_at_wavelength(1.0, 0.55)
    aod_nir = smoke_aod_at_wavelength(1.0, 1.0)
    assert aod_uv > aod_vis > aod_nir
    assert abs(aod_vis - 1.0) < 0.01  # 550nm should give ~1.0


def test_combine_aerosol_rayleigh_conservative():
    """Adding Rayleigh (SSA=1, g=0) should increase SSA and decrease g."""
    tau_aer, ssa_aer, g_aer = 0.5, 0.88, 0.68
    tau_ray = rayleigh_tau(0.50)
    tau_t, ssa_t, g_t = combine_aerosol_rayleigh(tau_aer, ssa_aer, g_aer, tau_ray)
    assert tau_t > tau_aer, "Total τ should exceed aerosol τ"
    assert ssa_t > ssa_aer, "Adding conservative scatterer should raise SSA"
    assert g_t < g_aer, "Adding isotropic scatterer should lower g"


# ── Solar geometry ───────────────────────────────────────────────────────────


def test_coszen_equinox_equator():
    """At equinox, equator should have μ̄ ≈ 1/π ≈ 0.318."""
    mu = daily_mean_coszen(0.0, 81)  # Mar 22 ≈ equinox
    assert abs(mu - 1.0 / np.pi) < 0.02, f"Equator equinox μ̄={mu:.4f}"


def test_coszen_latitude_gradient():
    """Higher latitude → lower μ̄ in summer."""
    mu_low = daily_mean_coszen(30.0, 249)  # Sep 5
    mu_high = daily_mean_coszen(50.0, 249)
    assert mu_low > mu_high, "Lower latitude should have higher μ̄"


def test_coszen_positive():
    """μ̄ should always be positive for our domain."""
    lats = np.arange(30, 53, 1.0)
    for doy in range(249, 260):
        mu = daily_mean_coszen(lats, doy)
        assert np.all(mu > 0), f"Negative μ̄ at doy={doy}"
