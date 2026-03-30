"""Per-site time series of smoke AOD and SW anomaly.

Produces a grid of subplots (up to 2x3), one per site, showing
smoke AOD bars, CERES delta-SW on a twin axis, and optional AERONET
scatter overlay.
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from davinci_monet.plots.style import NCAR_COLORS


def _nearest_idx(arr: np.ndarray, value: float) -> int:
    """Return the index of the nearest element in *arr* to *value*."""
    return int(np.argmin(np.abs(arr - value)))


def plot_site_timeseries(
    lats: np.ndarray,
    lons: np.ndarray,
    records: list[dict[str, Any]],
    bg_sw: np.ndarray,
    sites: list[tuple[str, float, float, str]],
    aeronet: pd.DataFrame | None = None,
    event_name: str = "",
) -> Figure:
    """Plot per-site time series of smoke AOD and SW anomaly.

    Parameters
    ----------
    lats
        1-D latitude array.
    lons
        1-D longitude array.
    records
        List of daily record dicts with: date, smoke_aod, sw_all.
    bg_sw
        2-D background SW array (same shape as record fields).
    sites
        List of (name, lat, lon, aeronet_name) tuples.
    aeronet
        Optional DataFrame with columns: siteid, time, aod.
    event_name
        Optional event label for the figure title.

    Returns
    -------
    Figure
    """
    n_sites = len(sites)
    ncols = min(n_sites, 3)
    nrows = max(1, (n_sites + ncols - 1) // ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows))
    if n_sites == 1:
        axes = np.array([axes])
    axes = np.atleast_2d(axes)

    dates = [r["date"] for r in records]
    x = np.arange(len(dates))

    for idx, (name, site_lat, site_lon, aeronet_name) in enumerate(sites):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]

        ilat = _nearest_idx(lats, site_lat)
        ilon = _nearest_idx(lons, site_lon)

        smoke_vals = [r["smoke_aod"][ilat, ilon] for r in records]
        sw_vals = [r["sw_all"][ilat, ilon] for r in records]
        dsw = [sw - bg_sw[ilat, ilon] for sw in sw_vals]

        ax.bar(x, smoke_vals, color=NCAR_COLORS["orange"], alpha=0.7, label="Smoke AOD")
        ax.set_ylabel("Smoke AOD", color=NCAR_COLORS["orange"])
        ax.set_xticks(x)
        ax.set_xticklabels(dates, rotation=45, ha="right", fontsize=7)
        ax.set_title(name)

        ax2 = ax.twinx()
        ax2.plot(x, dsw, color=NCAR_COLORS["ncar_blue"], marker="o", label="\u0394SW")
        ax2.set_ylabel("\u0394SW (W/m\u00b2)", color=NCAR_COLORS["ncar_blue"])

        # AERONET overlay
        if aeronet is not None and aeronet_name in aeronet["siteid"].values:
            site_df = aeronet[aeronet["siteid"] == aeronet_name].copy()
            site_df["date_str"] = site_df["time"].dt.strftime("%Y-%m-%d")
            # Scatter individual points
            for di, d in enumerate(dates):
                day_pts = site_df[site_df["date_str"] == d]
                if not day_pts.empty:
                    ax.scatter(
                        [di] * len(day_pts),
                        day_pts["aod"].values,
                        color=NCAR_COLORS["red"],
                        s=10,
                        alpha=0.5,
                        zorder=5,
                    )
                    ax.scatter(
                        di,
                        day_pts["aod"].mean(),
                        color=NCAR_COLORS["red"],
                        marker="D",
                        s=40,
                        zorder=6,
                        edgecolors="k",
                        linewidths=0.5,
                    )

    # Hide unused axes
    for idx in range(n_sites, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row, col].set_visible(False)

    suptitle = f"{event_name} — Site Time Series" if event_name else "Site Time Series"
    fig.suptitle(suptitle, fontsize=16, color=NCAR_COLORS["space"])
    fig.tight_layout()
    return fig
