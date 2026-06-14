"""Spatial plotting renderers for DAVINCI.

This subpackage provides map-based visualization:
- SpatialBiasPlotter: x-vs-y bias on maps
- SpatialOverlayPlotter: y contours with x overlays
- SpatialPlotter: Single-source spatial field map (shape-aware)
"""

from davinci_monet.plots.renderers.spatial.base import (
    BaseSpatialPlotter,
    MapConfig,
    draw_spatial_field,
    get_domain_extent,
    get_projection,
)
from davinci_monet.plots.renderers.spatial.bias import SpatialBiasPlotter, plot_spatial_bias
from davinci_monet.plots.renderers.spatial.field import SpatialPlotter
from davinci_monet.plots.renderers.spatial.overlay import (
    SpatialOverlayPlotter,
    plot_spatial_overlay,
)

__all__ = [
    # Base classes
    "BaseSpatialPlotter",
    "MapConfig",
    "draw_spatial_field",
    "get_domain_extent",
    "get_projection",
    # Plotters
    "SpatialBiasPlotter",
    "SpatialOverlayPlotter",
    "SpatialPlotter",
    # Convenience functions
    "plot_spatial_bias",
    "plot_spatial_overlay",
]
