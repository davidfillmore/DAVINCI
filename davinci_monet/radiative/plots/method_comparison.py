"""Scatter + radiative efficiency: MERRA-2 RT vs semi-empirical surface effect.

Reproduces the method comparison view from PlumeSentinelAI,
showing a density scatter of all grid cells and binned radiative
efficiency curves.
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from davinci_monet.plots.style import NCAR_COLORS


def plot_method_comparison(
    records: list[dict[str, Any]],
    event_name: str = "",
) -> Figure:
    """Scatter: MERRA-2 RT vs semi-empirical surface effect, + radiative efficiency.

    Parameters
    ----------
    records
        List of daily record dicts with keys: smoke_aod, m2_sfc_effect,
        semi_dimming (all 2-D arrays).
    event_name
        Optional event label for the figure title.

    Returns
    -------
    Figure
    """
    # Concatenate all days into flat arrays
    all_aod = np.concatenate([np.asarray(r["smoke_aod"]).ravel() for r in records])
    all_m2 = np.concatenate([np.asarray(r["m2_sfc_effect"]).ravel() for r in records])
    all_semi = np.concatenate([np.asarray(r["semi_dimming"]).ravel() for r in records])

    # Filter: valid and AOD > 0.05
    valid = np.isfinite(all_aod) & np.isfinite(all_m2) & np.isfinite(all_semi) & (all_aod > 0.05)
    aod_v = all_aod[valid]
    m2_v = all_m2[valid]
    semi_v = all_semi[valid]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # --- Left panel: scatter ---
    ax = axes[0]
    ax.scatter(semi_v, m2_v, s=2, alpha=0.1, color=NCAR_COLORS["ncar_blue"])
    lims = (-300, 50)
    ax.plot(lims, lims, "k--", linewidth=0.8, label="1:1")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("Semi-empirical \u0394SW (W/m\u00b2)")
    ax.set_ylabel("MERRA-2 RT \u0394SW (W/m\u00b2)")

    if len(m2_v) > 2:
        r_val = float(np.corrcoef(semi_v, m2_v)[0, 1])
        ax.set_title(f"All grid cells (r = {r_val:.3f})")
    else:
        ax.set_title("All grid cells")
    ax.legend(fontsize=9)

    # --- Right panel: radiative efficiency ---
    ax = axes[1]
    bin_edges = np.arange(0, 4.75, 0.25)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    m2_means: list[float] = []
    semi_means: list[float] = []
    valid_centers: list[float] = []

    for lo, hi, center in zip(bin_edges[:-1], bin_edges[1:], bin_centers):
        mask = (aod_v >= lo) & (aod_v < hi)
        if mask.sum() > 20:
            m2_means.append(float(np.mean(m2_v[mask])))
            semi_means.append(float(np.mean(semi_v[mask])))
            valid_centers.append(float(center))

    ax.plot(
        valid_centers,
        m2_means,
        "s-",
        color=NCAR_COLORS["ncar_blue"],
        lw=2,
        ms=5,
        label="MERRA-2 RT",
    )
    ax.plot(
        valid_centers,
        semi_means,
        "D--",
        color=NCAR_COLORS["red"],
        lw=1.5,
        ms=5,
        label="Semi-empirical",
    )
    ax.axhline(0, color="k", linewidth=0.5)
    ax.set_xlabel("MERRA-2 Smoke AOD bin center")
    ax.set_ylabel("Mean \u0394SW surface (W/m\u00b2)")
    ax.set_title("Radiative Efficiency")
    ax.legend(fontsize=9)

    suptitle = "Surface SW Dimming: MERRA-2 RT vs Semi-Empirical"
    if event_name:
        suptitle = f"{event_name} \u2014 {suptitle}"
    fig.suptitle(suptitle, fontsize=16)
    fig.tight_layout()
    return fig
