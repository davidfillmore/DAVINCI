"""4-panel CERES surface flux fields.

Reproduces the surface flux impact view from PlumeSentinelAI,
showing absolute and delta surface SW/LW down fields.
"""

from __future__ import annotations

from typing import Any

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from davinci_monet.plots.style import NCAR_COLORS


def plot_surface_flux(
    lats: np.ndarray,
    lons: np.ndarray,
    record: dict[str, Any],
    background: dict[str, Any],
    event_name: str = "",
) -> Figure:
    """4-panel: Surface SW Down, delta-Surface SW Down, Surface LW Down, delta-Surface LW Down.

    Parameters
    ----------
    lats
        1-D latitude array.
    lons
        1-D longitude array.
    record
        Dict with keys: sfc_sw_dn, sfc_lw_dn, date.
    background
        Dict with keys: sfc_sw_dn, sfc_lw_dn (pre-event reference).
    event_name
        Optional event label for the figure title.

    Returns
    -------
    Figure
    """
    proj = ccrs.PlateCarree()
    date_str = record.get("date", "")

    panels = [
        (f"Surface SW Down {date_str} (W/m\u00b2)", record["sfc_sw_dn"], "YlOrRd", 0, 400),
        (
            "\u0394Surface SW Down (W/m\u00b2)",
            record["sfc_sw_dn"] - background["sfc_sw_dn"],
            "RdBu_r",
            -200,
            200,
        ),
        (f"Surface LW Down {date_str} (W/m\u00b2)", record["sfc_lw_dn"], "YlOrRd", 200, 400),
        (
            "\u0394Surface LW Down (W/m\u00b2)",
            record["sfc_lw_dn"] - background["sfc_lw_dn"],
            "RdBu_r",
            -50,
            50,
        ),
    ]

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(14, 10),
        subplot_kw={"projection": proj},
    )

    for ax, (title, data, cmap, vmin, vmax) in zip(axes.ravel(), panels):
        mesh = ax.pcolormesh(
            lons,
            lats,
            data,
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
        fig.colorbar(mesh, ax=ax, shrink=0.8)

    suptitle_parts = ["CERES SYN1deg \u2014 Surface Flux Impact"]
    if date_str:
        suptitle_parts.append(f"{date_str} vs Pre-Event")
    if event_name:
        suptitle_parts.insert(0, event_name)
    fig.suptitle(
        ", ".join(suptitle_parts) if event_name else " ".join(suptitle_parts),
        fontsize=18,
        fontweight="bold",
        color=NCAR_COLORS["space"],
    )
    fig.tight_layout()
    return fig
