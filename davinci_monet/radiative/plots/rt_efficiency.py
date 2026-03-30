"""Radiative efficiency curves: progressive model levels vs MERRA-2 RT.

Bins surface dimming by smoke AOD and plots mean dimming per bin for
each model level alongside the MERRA-2 RT "truth".
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from davinci_monet.plots.style import NCAR_COLORS
from davinci_monet.radiative.processing_rt_levels import LEVEL_NAMES

LEVEL_COLORS = [
    NCAR_COLORS["gray"],
    NCAR_COLORS["aqua"],
    NCAR_COLORS["orange"],
    NCAR_COLORS["red"],
    NCAR_COLORS["ncar_blue"],
]


def plot_rt_efficiency(
    records: list[dict[str, Any]],
    event_name: str = "",
) -> Figure:
    """Plot radiative efficiency curves for all model levels vs MERRA-2 RT.

    Parameters
    ----------
    records
        List of daily record dicts, each with keys: ``smoke_aod_m2``,
        ``m2_truth``, ``levels`` (list of 5 arrays).
    event_name
        Optional event label for the figure title.

    Returns
    -------
    Figure
    """
    # Gather all grid cells across all days
    all_aod = np.concatenate([r["smoke_aod_m2"].ravel() for r in records if "levels" in r])
    all_truth = np.concatenate([r["m2_truth"].ravel() for r in records if "levels" in r])
    all_levels = [
        np.concatenate([r["levels"][j].ravel() for r in records if "levels" in r]) for j in range(5)
    ]

    # Bin by smoke AOD
    bin_edges = np.arange(0, 4.75, 0.25)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    min_count = 20

    truth_means = np.full(len(bin_centers), np.nan)
    level_means = [np.full(len(bin_centers), np.nan) for _ in range(5)]

    for b in range(len(bin_centers)):
        mask = (all_aod >= bin_edges[b]) & (all_aod < bin_edges[b + 1])
        if mask.sum() < min_count:
            continue
        truth_means[b] = np.nanmean(all_truth[mask])
        for j in range(5):
            level_means[j][b] = np.nanmean(all_levels[j][mask])

    fig, ax = plt.subplots(figsize=(10, 7))

    # MERRA-2 RT truth
    valid = np.isfinite(truth_means)
    ax.plot(
        bin_centers[valid],
        truth_means[valid],
        "o-",
        color="black",
        lw=3,
        ms=7,
        zorder=10,
        label="MERRA-2 RT",
    )

    # Model levels
    for j in range(5):
        v = np.isfinite(level_means[j])
        ax.plot(
            bin_centers[v],
            level_means[j][v],
            "o--",
            color=LEVEL_COLORS[j],
            lw=1.5,
            ms=5,
            label=LEVEL_NAMES[j],
        )

    ax.set_xlabel("MERRA-2 Smoke AOD (OC+BC)")
    ax.set_ylabel("Mean \u0394SW Surface (W/m\u00b2)")
    ax.axhline(0, color="gray", lw=0.5, ls=":")
    ax.legend(fontsize=9)

    title = "Radiative Efficiency: Progressive Semi-Empirical Models"
    if event_name:
        title += f" \u2014 {event_name}"
    ax.set_title(title, color=NCAR_COLORS["space"])

    fig.tight_layout()
    return fig
