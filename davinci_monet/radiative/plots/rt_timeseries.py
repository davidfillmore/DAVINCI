"""6-site timeseries: all model levels vs MERRA-2 RT surface dimming.

Produces a 2x3 grid of per-site time series comparing the progressive
semi-empirical model levels against MERRA-2 radiative transfer.
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


def _nearest_idx(arr: np.ndarray, value: float) -> int:
    """Return the index of the nearest element in *arr* to *value*."""
    return int(np.argmin(np.abs(arr - value)))


def plot_rt_timeseries(
    lats: np.ndarray,
    lons: np.ndarray,
    records: list[dict[str, Any]],
    sites: list[tuple[str, float, float, str]],
    event_name: str = "",
) -> Figure:
    """Plot per-site time series of model levels vs MERRA-2 RT.

    Parameters
    ----------
    lats
        1-D latitude array (MERRA-2 native grid).
    lons
        1-D longitude array (MERRA-2 native grid).
    records
        List of daily record dicts with: ``date``, ``m2_truth``,
        ``levels`` (list of 5 arrays).
    sites
        List of ``(name, lat, lon, aeronet_id)`` tuples (up to 6).
    event_name
        Optional event label for the figure title.

    Returns
    -------
    Figure
    """
    n_sites = min(len(sites), 6)
    n_cols = 3
    n_rows = 2

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 9), sharex=True)
    dates = [r["date"] for r in records if "levels" in r]
    rt_records = [r for r in records if "levels" in r]

    for s_idx in range(n_sites):
        row, col = divmod(s_idx, n_cols)
        ax = axes[row, col]
        name, slat, slon, _ = sites[s_idx]

        ilat = _nearest_idx(lats, slat)
        ilon = _nearest_idx(lons, slon)

        truth_ts = [r["m2_truth"][ilat, ilon] for r in rt_records]
        level_ts = [[r["levels"][j][ilat, ilon] for r in rt_records] for j in range(5)]

        # MERRA-2 RT truth
        ax.plot(
            dates,
            truth_ts,
            "o-",
            color="black",
            lw=2.5,
            ms=5,
            zorder=10,
            label="MERRA-2 RT",
        )

        # Model levels
        for j in range(5):
            ax.plot(
                dates,
                level_ts[j],
                "o--",
                color=LEVEL_COLORS[j],
                lw=1.2,
                ms=3,
                alpha=0.8,
                label=LEVEL_NAMES[j],
            )

        ax.set_title(name, fontsize=10)
        ax.set_ylabel("\u0394SW Surface (W/m\u00b2)")
        ax.axhline(0, color="gray", lw=0.5, ls=":")

        if row == n_rows - 1:
            ax.tick_params(axis="x", rotation=30)

    # Hide unused panels
    for s_idx in range(n_sites, n_rows * n_cols):
        row, col = divmod(s_idx, n_cols)
        axes[row, col].set_visible(False)

    # Legend in last visible panel or last panel position
    legend_ax = axes[n_rows - 1, n_cols - 1]
    if not legend_ax.get_visible():
        legend_ax.set_visible(True)
        legend_ax.axis("off")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    legend_ax.legend(handles, labels, loc="center", fontsize=7)

    suptitle = "Surface SW Dimming: Model Levels vs MERRA-2 RT"
    if event_name:
        suptitle += f" \u2014 {event_name}"
    fig.suptitle(suptitle, fontsize=16, color=NCAR_COLORS["space"])
    fig.tight_layout()
    return fig
