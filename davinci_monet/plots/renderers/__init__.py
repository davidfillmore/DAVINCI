"""Plot renderers for DAVINCI.

This subpackage provides individual plot type implementations:

Temporal plots:
- TimeSeriesPlotter: Time series comparisons
- DiurnalPlotter: Diurnal cycle plots

Statistical plots:
- ScatterPlotter: Scatter plots with regression
- TaylorPlotter: Taylor diagrams
- BoxPlotter: Box plot comparisons

Spatial plots:
- SpatialBiasPlotter: Bias maps
- SpatialOverlayPlotter: y/x overlays
- SpatialPlotter: Single-source spatial field maps

Specialized plots:
- CurtainPlotter: Vertical cross-sections
- ScorecardPlotter: Multi-metric scorecards
"""

from davinci_monet.plots.renderers.boxplot import BoxPlotter, plot_boxplot

# Specialized plots
from davinci_monet.plots.renderers.curtain import CurtainPlotter, plot_curtain
from davinci_monet.plots.renderers.diurnal import DiurnalPlotter, plot_diurnal
from davinci_monet.plots.renderers.flight_track import FlightTrackPlotter
from davinci_monet.plots.renderers.histogram import HistogramPlotter
from davinci_monet.plots.renderers.lma_density import LMADensityPlotter

# Statistical plots
from davinci_monet.plots.renderers.scatter import ScatterPlotter, plot_scatter
from davinci_monet.plots.renderers.scorecard import ScorecardPlotter, plot_scorecard

# Spatial plots (imported from subpackage)
from davinci_monet.plots.renderers.spatial import (
    BaseSpatialPlotter,
    MapConfig,
    SpatialBiasPlotter,
    SpatialOverlayPlotter,
    SpatialPlotter,
    get_domain_extent,
    get_projection,
    plot_spatial_bias,
    plot_spatial_overlay,
)
from davinci_monet.plots.renderers.taylor import TaylorPlotter, plot_taylor

# Temporal plots
from davinci_monet.plots.renderers.timeseries import TimeSeriesPlotter, plot_timeseries
from davinci_monet.plots.renderers.track_map_3d import TrackMap3DPlotter, plot_track_map_3d
from davinci_monet.plots.renderers.vertical_profile import VerticalProfilePlotter

__all__ = [
    # Temporal
    "TimeSeriesPlotter",
    "plot_timeseries",
    "DiurnalPlotter",
    "plot_diurnal",
    # Statistical
    "ScatterPlotter",
    "plot_scatter",
    "TaylorPlotter",
    "plot_taylor",
    "BoxPlotter",
    "plot_boxplot",
    # Specialized
    "CurtainPlotter",
    "plot_curtain",
    "ScorecardPlotter",
    "plot_scorecard",
    "TrackMap3DPlotter",
    "plot_track_map_3d",
    # Spatial
    "BaseSpatialPlotter",
    "MapConfig",
    "SpatialBiasPlotter",
    "SpatialOverlayPlotter",
    "SpatialPlotter",
    "plot_spatial_bias",
    "plot_spatial_overlay",
    "get_domain_extent",
    "get_projection",
]
