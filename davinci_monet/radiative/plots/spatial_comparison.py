"""Side-by-side maps of MERRA-2 smoke AOD vs CERES delta-SW.

Reproduces the spatial comparison view from PlumeSentinelAI for the
peak day of a smoke event.
"""

from __future__ import annotations

from typing import Any

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from davinci_monet.plots.style import NCAR_COLORS


def plot_spatial_comparison(
    lats: np.ndarray,
    lons: np.ndarray,
    record: dict[str, Any],
    bg_sw: np.ndarray,
    event_name: str = "",
) -> Figure:
    """Side-by-side: MERRA-2 smoke AOD vs CERES delta-SW for peak day.

    Parameters
    ----------
    lats
        1-D latitude array.
    lons
        1-D longitude array.
    record
        Dict with keys: date, smoke_aod, sw_all (2-D arrays).
    bg_sw
        2-D background SW array (same shape as record fields).
    event_name
        Optional event label for the figure title.

    Returns
    -------
    Figure
    """
    proj = ccrs.PlateCarree()
    date_str = record.get("date", "")

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14, 6),
        subplot_kw={"projection": proj},
    )

    # Left panel: MERRA-2 Smoke AOD
    ax_left = axes[0]
    mesh_left = ax_left.pcolormesh(
        lons,
        lats,
        record["smoke_aod"],
        cmap="YlOrRd",
        vmin=0,
        vmax=3.0,
        shading="auto",
        transform=proj,
    )
    ax_left.add_feature(cfeature.STATES, linewidth=0.5)
    ax_left.add_feature(cfeature.COASTLINE, linewidth=0.8)
    ax_left.set_title("MERRA-2 Smoke AOD (OC+BC)")
    fig.colorbar(mesh_left, ax=ax_left, label="AOD", shrink=0.8)

    # Right panel: CERES delta-TOA SW Reflected
    ax_right = axes[1]
    delta_sw = record["sw_all"] - bg_sw
    mesh_right = ax_right.pcolormesh(
        lons,
        lats,
        delta_sw,
        cmap="RdBu_r",
        vmin=-150,
        vmax=150,
        shading="auto",
        transform=proj,
    )
    ax_right.add_feature(cfeature.STATES, linewidth=0.5)
    ax_right.add_feature(cfeature.COASTLINE, linewidth=0.8)
    ax_right.set_title("CERES \u0394TOA SW Reflected (vs background)")
    fig.colorbar(mesh_right, ax=ax_right, label="W/m\u00b2", shrink=0.8)

    suptitle = f"{date_str} \u2014 MERRA-2 Smoke AOD vs CERES \u0394SW Reflected"
    if event_name:
        suptitle = f"{event_name}: {suptitle}"
    fig.suptitle(
        suptitle,
        fontsize=16,
        fontweight="bold",
        color=NCAR_COLORS["space"],
    )
    fig.tight_layout()
    return fig
