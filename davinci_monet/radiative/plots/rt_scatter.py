"""5-panel scatter: each model level vs MERRA-2 RT surface dimming.

Produces a 2x3 grid (last cell hidden) comparing semi-empirical
estimates against the MERRA-2 radiative transfer "truth".
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


def plot_rt_scatter(
    records: list[dict[str, Any]],
    event_name: str = "",
) -> Figure:
    """Plot 5-panel scatter of model levels vs MERRA-2 RT.

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
    all_aod = np.concatenate([r["smoke_aod_m2"].ravel() for r in records if "levels" in r])
    all_truth = np.concatenate([r["m2_truth"].ravel() for r in records if "levels" in r])
    all_levels = [
        np.concatenate([r["levels"][j].ravel() for r in records if "levels" in r]) for j in range(5)
    ]

    # Filter to cells with meaningful smoke
    mask = all_aod > 0.05
    truth = all_truth[mask]

    fig, axes = plt.subplots(2, 3, figsize=(17, 11))

    for j in range(5):
        row, col = divmod(j, 3)
        ax = axes[row, col]
        pred = all_levels[j][mask]

        ax.scatter(truth, pred, s=1, alpha=0.08, color=LEVEL_COLORS[j], edgecolors="none")

        # 1:1 line
        lims = (-300, 50)
        ax.plot(lims, lims, "k--", lw=0.8, alpha=0.5)

        # Statistics
        valid = np.isfinite(truth) & np.isfinite(pred)
        if valid.sum() > 2:
            r = np.corrcoef(truth[valid], pred[valid])[0, 1]
            rmse = np.sqrt(np.nanmean((pred[valid] - truth[valid]) ** 2))
            bias = np.nanmean(pred[valid] - truth[valid])
        else:
            r, rmse, bias = 0.0, 0.0, 0.0

        ax.set_title(
            f"{LEVEL_NAMES[j]}\nr={r:.3f}, RMSE={rmse:.1f}, bias={bias:+.1f}",
            fontsize=10,
        )
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel("MERRA-2 RT (W/m\u00b2)")
        ax.set_ylabel(f"{LEVEL_NAMES[j]} (W/m\u00b2)")

    # Hide last subplot
    axes[1, 2].set_visible(False)

    suptitle = "Semi-Empirical vs MERRA-2 RT Surface Dimming"
    if event_name:
        suptitle += f" \u2014 {event_name}"
    fig.suptitle(suptitle, fontsize=16, color=NCAR_COLORS["space"])
    fig.tight_layout()
    return fig
