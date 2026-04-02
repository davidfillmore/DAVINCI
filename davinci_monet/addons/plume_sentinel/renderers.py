"""Workflow-local plot renderers for PlumeSentinel add-on.

These renderers are NOT registered in the global DAVINCI plotter registry.
They are called directly by the PlumeSentinelPlotStage.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from davinci_monet.addons.plume_sentinel.backgrounds import render_background
from davinci_monet.addons.plume_sentinel.overlays import render_overlays
from davinci_monet.addons.plume_sentinel.processing import GriddedAodResult
from davinci_monet.plots.style import NCAR_COLORS


def render_plot(
    plot_name: str,
    plot_spec: dict[str, Any],
    prepared_inputs: dict[str, Any],
    output_dir: str | Path,
) -> list[str]:
    """Dispatch to the appropriate renderer based on plot type.

    Parameters
    ----------
    plot_name
        Name used for the output filename.
    plot_spec
        Plot specification dictionary (from PlotSpec.model_dump()).
    prepared_inputs
        Mapping of input name to prepared data objects.
    output_dir
        Directory for saving output files.

    Returns
    -------
    list[str]
        Paths to generated plot files.
    """
    plot_type = plot_spec["type"]
    if plot_type == "truecolor_contour":
        return _render_truecolor_contour(plot_name, plot_spec, prepared_inputs, output_dir)
    elif plot_type == "truecolor_aod":
        return _render_truecolor_aod(plot_name, plot_spec, prepared_inputs, output_dir)
    else:
        raise ValueError(f"Unknown plot type: {plot_type!r}")


def _make_projection(proj_spec: dict[str, Any] | None) -> ccrs.Projection:
    """Build a cartopy projection from a spec dict.

    Parameters
    ----------
    proj_spec
        Projection specification with keys: type, central_longitude,
        central_latitude.  If None, returns PlateCarree.
    """
    if proj_spec is None:
        return ccrs.PlateCarree()

    # Handle Pydantic model objects that slipped through without model_dump()
    if not isinstance(proj_spec, dict):
        proj_spec = {
            "type": getattr(proj_spec, "type", "plate_carree"),
            "central_longitude": getattr(proj_spec, "central_longitude", 0.0),
            "central_latitude": getattr(proj_spec, "central_latitude", 0.0),
        }

    proj_type = proj_spec.get("type", "plate_carree")
    if proj_type == "lambert_conformal":
        return ccrs.LambertConformal(
            central_longitude=proj_spec.get("central_longitude", 0.0),
            central_latitude=proj_spec.get("central_latitude", 0.0),
        )
    return ccrs.PlateCarree()


def _save_figure(fig: plt.Figure, output_dir: str | Path, plot_name: str) -> list[str]:
    """Save figure to PNG and close it.

    Returns
    -------
    list[str]
        Single-element list with the output path.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{plot_name}.png"
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return [str(png_path)]


def _render_truecolor_contour(
    plot_name: str,
    plot_spec: dict[str, Any],
    prepared_inputs: dict[str, Any],
    output_dir: str | Path,
) -> list[str]:
    """Render a true-color background with HMS smoke contour overlays.

    Creates a GOES true-color satellite image with HMS smoke polygon
    outlines colored by density category.
    """
    projection = _make_projection(plot_spec.get("projection"))
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(1, 1, 1, projection=projection)

    # Set map extent if provided
    extent = plot_spec.get("extent")
    if extent is not None:
        ax.set_extent(extent, crs=ccrs.PlateCarree())  # type: ignore[attr-defined]

    # Render GOES true-color background
    background = plot_spec.get("background")
    if background is not None:
        render_background(ax, background, prepared_inputs)

    # Render HMS smoke overlays
    overlay_names = plot_spec.get("overlays") or []
    legend_handles = render_overlays(ax, overlay_names, prepared_inputs)

    # Add political/geographic features with white edges for visibility
    ax.add_feature(  # type: ignore[attr-defined]
        cfeature.STATES.with_scale("50m"),
        edgecolor="white",
        linewidth=0.5,
        facecolor="none",
        zorder=5,
    )
    ax.add_feature(  # type: ignore[attr-defined]
        cfeature.COASTLINE.with_scale("50m"),
        edgecolor="white",
        linewidth=0.8,
        zorder=5,
    )

    # Add legend with black background for readability
    if legend_handles:
        ax.legend(
            handles=legend_handles,
            loc="lower left",
            fontsize=10,
            facecolor="black",
            edgecolor="white",
            labelcolor="white",
            framealpha=0.8,
        )

    # Title
    title = plot_spec.get("title", plot_name)
    fig.suptitle(title, fontsize=16, fontweight="bold")

    return _save_figure(fig, output_dir, plot_name)


def _render_truecolor_aod(
    plot_name: str,
    plot_spec: dict[str, Any],
    prepared_inputs: dict[str, Any],
    output_dir: str | Path,
) -> list[str]:
    """Render a GIBS background with gridded MODIS AOD overlay.

    Uses a NASA GIBS WMTS tile as background with semi-transparent
    gridded AOD field overlaid using a colormap.
    """
    projection = _make_projection(plot_spec.get("projection"))
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(1, 1, 1, projection=projection)

    # Set map extent if provided
    extent = plot_spec.get("extent")
    if extent is not None:
        ax.set_extent(extent, crs=ccrs.PlateCarree())  # type: ignore[attr-defined]

    # Render GIBS WMTS background
    background = plot_spec.get("background")
    if background is not None:
        render_background(ax, background, prepared_inputs)

    # Pull GriddedAodResult from prepared inputs
    field_name = plot_spec.get("field")
    if field_name is not None and field_name in prepared_inputs:
        aod_result: GriddedAodResult = prepared_inputs[field_name]
        data_2d = aod_result.data_2d

        # Build RGBA array with alpha masking (transparent where NaN)
        cmap_name = plot_spec.get("cmap", "YlOrRd")
        cmap = plt.get_cmap(cmap_name)
        alpha = plot_spec.get("alpha", 0.7)

        # Normalize data to [0, 1] for colormap
        vmin = np.nanmin(data_2d) if np.any(np.isfinite(data_2d)) else 0.0
        vmax = np.nanmax(data_2d) if np.any(np.isfinite(data_2d)) else 1.0
        if vmin == vmax:
            vmax = vmin + 1.0
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        data_normed = norm(data_2d)

        # Apply colormap to get RGBA
        rgba = cmap(data_normed)

        # Set alpha: transparent where NaN, semi-transparent elsewhere
        nan_mask = ~np.isfinite(data_2d)
        rgba[nan_mask, 3] = 0.0
        rgba[~nan_mask, 3] = alpha

        # Compute extent from grid coordinates
        lon_min = aod_result.lon_centers[0] - aod_result.resolution / 2
        lon_max = aod_result.lon_centers[-1] + aod_result.resolution / 2
        lat_min = aod_result.lat_centers[0] - aod_result.resolution / 2
        lat_max = aod_result.lat_centers[-1] + aod_result.resolution / 2
        img_extent = (lon_min, lon_max, lat_min, lat_max)

        im = ax.imshow(
            rgba,
            origin="lower",
            extent=img_extent,
            transform=ccrs.PlateCarree(),
            interpolation="nearest",
            zorder=3,
        )

        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar_label = plot_spec.get("colorbar_label", "AOD")
        fig.colorbar(sm, ax=ax, label=cbar_label, shrink=0.7, pad=0.02)

    # Add political/geographic features using NCAR brand colors
    ax.add_feature(  # type: ignore[attr-defined]
        cfeature.STATES.with_scale("50m"),
        edgecolor=NCAR_COLORS["gray"],
        linewidth=0.5,
        facecolor="none",
        zorder=5,
    )
    ax.add_feature(  # type: ignore[attr-defined]
        cfeature.COASTLINE.with_scale("50m"),
        edgecolor=NCAR_COLORS["space"],
        linewidth=0.8,
        zorder=5,
    )

    # Title
    title = plot_spec.get("title", plot_name)
    fig.suptitle(title, fontsize=16, fontweight="bold")

    return _save_figure(fig, output_dir, plot_name)
