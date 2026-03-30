"""Anomaly maps (event minus background) for radiative fields.

Produces a 2x2 panel figure showing the departure from background
conditions for AOD, SW all-sky, SW clear-sky, and TOA net flux.
"""

from __future__ import annotations

from typing import Any

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from davinci_monet.plots.style import NCAR_COLORS


def plot_anomaly_maps(
    lats: np.ndarray,
    lons: np.ndarray,
    record: dict[str, Any],
    background: dict[str, Any],
    event_name: str = "",
    bg_description: str = "Pre-Event Mean",
) -> Figure:
    """Plot 2x2 anomaly maps (event - background).

    Parameters
    ----------
    lats
        1-D latitude array.
    lons
        1-D longitude array.
    record
        Dict with keys: aod, sw_all, sw_clr, toa_net.
    background
        Dict with the same keys, representing the background state.
    event_name
        Optional event label for the figure title.

    Returns
    -------
    Figure
    """
    proj = ccrs.PlateCarree()

    panels = [
        ("\u0394AOD 550 nm", "aod", "PuOr_r", -2, 2, "\u0394AOD"),
        ("\u0394TOA SW Reflected (W/m\u00b2)", "sw_all", "RdBu_r", -100, 100, "W/m\u00b2"),
        ("\u0394TOA SW Clear-Sky (W/m\u00b2)", "sw_clr", "RdBu_r", -100, 100, "W/m\u00b2"),
        ("\u0394TOA Net Flux (W/m\u00b2)", "toa_net", "RdBu_r", -100, 100, "W/m\u00b2"),
    ]

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(14, 10),
        subplot_kw={"projection": proj},
    )

    for ax, (title, key, cmap, vmin, vmax, cbar_label) in zip(axes.flat, panels):
        anomaly = record[key] - background[key]
        mesh = ax.pcolormesh(
            lons,
            lats,
            anomaly,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            shading="auto",
            transform=proj,
        )
        ax.add_feature(cfeature.STATES, linewidth=0.5)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
        ax.add_feature(cfeature.BORDERS, linewidth=0.5)
        ax.set_title(title)
        fig.colorbar(mesh, ax=ax, label=cbar_label, shrink=0.8)

    date_obj = record.get("date", "")
    if hasattr(date_obj, "strftime"):
        date_str = date_obj.strftime("%b %-d")
    else:
        date_str = str(date_obj)
    suptitle = f"CERES SYN1deg \u2014 {date_str} Anomaly vs Pre-Event ({bg_description})"
    fig.suptitle(suptitle, fontsize=16, color=NCAR_COLORS["space"])
    fig.tight_layout()
    return fig
