"""Spatial comparison: MERRA-2 RT vs best model level on peak day.

Produces a 1x3 panel figure with MERRA-2 RT truth, the best
semi-empirical model level, and the residual.
"""

from __future__ import annotations

from typing import Any

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from davinci_monet.plots.style import NCAR_COLORS
from davinci_monet.radiative.processing_rt_levels import LEVEL_NAMES


def plot_rt_spatial(
    lats: np.ndarray,
    lons: np.ndarray,
    record: dict[str, Any],
    best_level_idx: int = 3,
    event_name: str = "",
) -> Figure:
    """Plot 3-panel spatial comparison on peak day.

    Parameters
    ----------
    lats
        1-D latitude array (MERRA-2 native grid).
    lons
        1-D longitude array (MERRA-2 native grid).
    record
        Dict with keys: ``m2_truth``, ``levels`` (list of 5 arrays), ``date``.
    best_level_idx
        Index of the model level to compare (default 3 = L3).
    event_name
        Optional event label for the figure title.

    Returns
    -------
    Figure
    """
    proj = ccrs.PlateCarree()

    m2_truth = record["m2_truth"]
    model_pred = record["levels"][best_level_idx]
    residual = model_pred - m2_truth

    level_label = LEVEL_NAMES[best_level_idx]
    level_short = f"L{best_level_idx}"

    panels = [
        ("MERRA-2 RT \u0394SW Surface", m2_truth, "RdBu_r", -250, 250, "W/m\u00b2"),
        (f"{level_label}", model_pred, "RdBu_r", -250, 250, "W/m\u00b2"),
        (f"Residual ({level_short} \u2212 RT)", residual, "RdBu_r", -100, 100, "W/m\u00b2"),
    ]

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(20, 6),
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
    suptitle = f"Spatial RT Comparison \u2014 {date_str}"
    if event_name:
        suptitle = f"Spatial RT Comparison \u2014 {event_name} \u2014 {date_str}"
    fig.suptitle(suptitle, fontsize=16, color=NCAR_COLORS["space"])
    fig.tight_layout()
    return fig
