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
        ("\u0394AOD", "aod", "PuOr_r", -2, 2, "\u0394AOD"),
        ("\u0394SW All-Sky", "sw_all", "RdBu_r", -100, 100, "W/m\u00b2"),
        ("\u0394SW Clear-Sky", "sw_clr", "RdBu_r", -100, 100, "W/m\u00b2"),
        ("\u0394Net", "toa_net", "RdBu_r", -100, 100, "W/m\u00b2"),
    ]

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(14, 10),
        subplot_kw={"projection": proj},
    )

    for ax, (title, key, cmap, vmin, vmax, cbar_label) in zip(
        axes.flat, panels
    ):
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

    date_str = record.get("date", "")
    suptitle = f"{event_name} — Anomaly {date_str}" if event_name else f"Anomaly {date_str}"
    fig.suptitle(suptitle, fontsize=16, color=NCAR_COLORS["space"])
    fig.tight_layout()
    return fig
