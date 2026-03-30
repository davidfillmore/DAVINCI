"""Surface dimming time-series at sites: MERRA-2 RT vs semi-empirical.

Reproduces the per-site dimming comparison from PlumeSentinelAI,
showing smoke AOD bars alongside two independent surface SW
perturbation estimates.
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from davinci_monet.plots.style import NCAR_COLORS


def _nearest_idx(arr: np.ndarray, value: float) -> int:
    """Return the index of the nearest element in *arr* to *value*."""
    return int(np.argmin(np.abs(arr - value)))


def plot_surface_dimming_timeseries(
    lats: np.ndarray,
    lons: np.ndarray,
    records: list[dict[str, Any]],
    sites: list[tuple[str, float, float, str]],
    event_name: str = "",
) -> Figure:
    """Surface dimming time-series at sites: MERRA-2 RT vs semi-empirical.

    Parameters
    ----------
    lats
        1-D latitude array.
    lons
        1-D longitude array.
    records
        List of daily record dicts with keys: date, smoke_aod,
        m2_sfc_effect, semi_dimming (all 2-D arrays).
    sites
        List of (display_name, lat, lon, aeronet_name) tuples.
    event_name
        Optional event label for the figure title.

    Returns
    -------
    Figure
    """
    n_sites = len(sites)
    ncols = min(n_sites, 3)
    nrows = max(1, (n_sites + ncols - 1) // ncols)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(18, 9),
        sharex=True,
    )
    if n_sites == 1:
        axes = np.array([axes])
    axes = np.atleast_2d(axes)

    dates = [r["date"] for r in records]
    x = np.arange(len(dates))

    # Track handles for shared legend
    legend_handles: list[Any] = []
    legend_labels: list[str] = []

    for idx, (display_name, site_lat, site_lon, _aeronet_name) in enumerate(sites):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]

        ilat = _nearest_idx(lats, site_lat)
        ilon = _nearest_idx(lons, site_lon)

        smoke_vals = [r["smoke_aod"][ilat, ilon] for r in records]
        m2_vals = [r["m2_sfc_effect"][ilat, ilon] for r in records]
        semi_vals = [r["semi_dimming"][ilat, ilon] for r in records]

        # Smoke AOD bars (left y-axis)
        bar_h = ax.bar(
            x,
            smoke_vals,
            width=0.8,
            color=NCAR_COLORS["orange"],
            alpha=0.4,
            label="Smoke AOD",
        )
        ax.set_ylabel("Smoke AOD")
        ax.set_xticks(x)
        ax.set_xticklabels(dates, rotation=45, ha="right", fontsize=7)
        ax.set_title(display_name, fontsize=13, fontweight="bold")

        # SW dimming lines (right y-axis)
        ax2 = ax.twinx()
        (line_m2,) = ax2.plot(
            x,
            m2_vals,
            "s-",
            color=NCAR_COLORS["ncar_blue"],
            lw=2,
            ms=4,
            label="MERRA-2 RT",
        )
        (line_semi,) = ax2.plot(
            x,
            semi_vals,
            "D--",
            color=NCAR_COLORS["red"],
            lw=1.5,
            ms=4,
            label="Semi-empirical",
        )
        ax2.set_ylabel("\u0394SW surface (W/m\u00b2)")
        ax2.axhline(0, color="k", linewidth=0.5)

        # Collect legend handles from first panel only
        if idx == 0:
            legend_handles = [bar_h, line_m2, line_semi]
            legend_labels = ["Smoke AOD", "MERRA-2 RT", "Semi-empirical"]

    # Hide unused axes
    for idx in range(n_sites, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row, col].set_visible(False)

    suptitle = f"Surface SW Dimming from Smoke \u2014 {event_name}" if event_name else "Surface SW Dimming from Smoke"
    fig.suptitle(suptitle, fontsize=17)

    if legend_handles:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="lower center",
            ncol=3,
            fontsize=11,
            bbox_to_anchor=(0.5, -0.02),
        )
    fig.tight_layout()
    return fig
