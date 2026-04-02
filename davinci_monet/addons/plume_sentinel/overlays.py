"""HMS smoke polygon overlay renderer for PlumeSentinel add-on."""

from __future__ import annotations

from typing import Any

import cartopy.crs as ccrs
import matplotlib.axes
import matplotlib.patches as mpatches

SMOKE_DENSITY_STYLES: dict[str, dict[str, Any]] = {
    "Light": {"color": "#FFDD31", "linewidth": 1.0},
    "Medium": {"color": "#FF8C00", "linewidth": 1.5},
    "Heavy": {"color": "#D62839", "linewidth": 2.5},
}


def render_overlays(
    ax: matplotlib.axes.Axes,
    overlay_names: list[str],
    prepared_inputs: dict[str, Any],
) -> list[mpatches.Patch]:
    """Render HMS smoke overlays on a matplotlib axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to render on.
    overlay_names : list[str]
        Names of overlays to render (keys into prepared_inputs).
    prepared_inputs : dict[str, Any]
        Mapping of overlay name to GeoDataFrame with smoke polygons.

    Returns
    -------
    list[mpatches.Patch]
        Legend handles for each density level rendered.
    """
    legend_handles: list[mpatches.Patch] = []
    for name in overlay_names:
        gdf = prepared_inputs[name]
        handles = _render_hms_smoke(ax, gdf)
        legend_handles.extend(handles)
    return legend_handles


def _render_hms_smoke(ax: matplotlib.axes.Axes, gdf: Any) -> list[mpatches.Patch]:
    """Render HMS smoke polygons colored by density category.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to render on.
    gdf : geopandas.GeoDataFrame
        Smoke polygons with a "Density" column.

    Returns
    -------
    list[mpatches.Patch]
        Legend handles for each density level.
    """
    handles: list[mpatches.Patch] = []
    for density, style in SMOKE_DENSITY_STYLES.items():
        subset = gdf[gdf["Density"] == density]
        if len(subset) > 0:
            subset.plot(
                ax=ax,
                transform=ccrs.PlateCarree(),
                facecolor="none",
                edgecolor=style["color"],
                linewidth=style["linewidth"],
                alpha=0.9,
                zorder=4,
            )
        handles.append(
            mpatches.Patch(
                facecolor="none",
                edgecolor=style["color"],
                linewidth=style["linewidth"],
                label=f"{density} smoke",
            )
        )
    return handles
