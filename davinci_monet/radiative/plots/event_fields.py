"""Event-day field maps (AOD, SW, TOA net, cloud fraction).

Produces a 2x2 panel figure showing key radiative fields for a single day.
"""

from __future__ import annotations

from typing import Any

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from davinci_monet.plots.style import NCAR_COLORS


def plot_event_fields(
    lats: np.ndarray,
    lons: np.ndarray,
    record: dict[str, Any],
    event_name: str = "",
) -> Figure:
    """Plot 2x2 map panels of event-day radiative fields.

    Parameters
    ----------
    lats
        1-D latitude array.
    lons
        1-D longitude array.
    record
        Dict with keys: aod, sw_all, toa_net, cld_frac, date.
    event_name
        Optional event label for the figure title.

    Returns
    -------
    Figure
    """
    proj = ccrs.PlateCarree()

    panels = [
        ("AOD 550 nm (MATCH)", record["aod"], "YlOrRd", 0, 3, "AOD"),
        ("TOA SW Reflected (W/m\u00b2)", record["sw_all"], "YlOrRd", 0, 350, "W/m\u00b2"),
        ("TOA Net Flux (W/m\u00b2)", record["toa_net"], "RdBu_r", -150, 150, "W/m\u00b2"),
        ("Cloud Fraction (%)", record["cld_frac"], "Blues", 0, 100, "%"),
    ]

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(14, 10),
        subplot_kw={"projection": proj},
    )

    for ax, (title, data, cmap, vmin, vmax, cbar_label) in zip(axes.flat, panels):
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
    suptitle = f"CERES SYN1deg \u2014 {date_str} ({event_name})" if event_name else f"CERES SYN1deg \u2014 {date_str}"
    fig.suptitle(suptitle, fontsize=16, color=NCAR_COLORS["space"])
    fig.tight_layout()
    return fig
