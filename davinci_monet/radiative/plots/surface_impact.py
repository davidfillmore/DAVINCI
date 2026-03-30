"""Surface radiative impact maps (smoke AOD, MERRA-2, semi-empirical).

Produces a 1x3 panel figure comparing smoke AOD with two estimates
of surface SW flux perturbation.
"""

from __future__ import annotations

from typing import Any

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from davinci_monet.plots.style import NCAR_COLORS


def plot_surface_impact(
    lats: np.ndarray,
    lons: np.ndarray,
    record: dict[str, Any],
    event_name: str = "",
    ssa: float = 0.92,
) -> Figure:
    """Plot 1x3 surface radiative impact maps.

    Parameters
    ----------
    lats
        1-D latitude array.
    lons
        1-D longitude array.
    record
        Dict with keys: smoke_aod, m2_sfc_effect, semi_dimming, date.
    event_name
        Optional event label for the figure title.

    Returns
    -------
    Figure
    """
    proj = ccrs.PlateCarree()

    panels = [
        ("MERRA-2 Smoke AOD (OC+BC)", record["smoke_aod"], "YlOrRd", 0, 3, "AOD"),
        ("MERRA-2 \u0394SW_net Surface\n(aerosol \u2212 no aerosol)", record["m2_sfc_effect"], "RdBu_r", -250, 250, "W/m\u00b2"),
        (f"Semi-Empirical \u0394SW Surface\n(Beer-Lambert, SSA={ssa})", record["semi_dimming"], "RdBu_r", -250, 250, "W/m\u00b2"),
    ]

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(18, 5),
        subplot_kw={"projection": proj},
    )

    for ax, (title, data, cmap, vmin, vmax, cbar_label) in zip(axes, panels):
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
        fig.colorbar(mesh, ax=ax, label=cbar_label, shrink=0.8)

    date_obj = record.get("date", "")
    if hasattr(date_obj, "strftime"):
        date_str = date_obj.strftime("%B %-d, %Y")
    else:
        date_str = str(date_obj)
    suptitle = f"Surface SW Impact of Smoke \u2014 {date_str}"
    fig.suptitle(suptitle, fontsize=16, color=NCAR_COLORS["space"])
    fig.tight_layout()
    return fig
