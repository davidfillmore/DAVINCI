"""Plot renderers for DAVINCI.

This subpackage provides individual plot type implementations:

Temporal plots:
- TimeSeriesPlotter: Time series comparisons
- DiurnalPlotter: Diurnal cycle plots
- PerSiteTimeSeriesPlotter: Individual per-site time series

Statistical plots:
- ScatterPlotter: Scatter plots with regression
- TaylorPlotter: Taylor diagrams
- BoxPlotter: Box plot comparisons

Spatial plots:
- SpatialBiasPlotter: Bias maps
- SpatialOverlayPlotter: Dataset/geometry overlays
- SpatialDistributionPlotter: Value distribution maps

Specialized plots:
- CurtainPlotter: Vertical cross-sections
- ScorecardPlotter: Multi-metric scorecards
"""

from davinci_monet.plots.renderers.boxplot import BoxPlotter, plot_boxplot

# Specialized plots
from davinci_monet.plots.renderers.curtain import CurtainPlotter, plot_curtain
from davinci_monet.plots.renderers.diurnal import DiurnalPlotter, plot_diurnal
from davinci_monet.plots.renderers.flight_timeseries import (
    FlightTimeSeriesPlotter,
    plot_flight_timeseries,
)
from davinci_monet.plots.renderers.flight_track import FlightTrackPlotter
from davinci_monet.plots.renderers.histogram import HistogramPlotter
from davinci_monet.plots.renderers.lma_density import LMADensityPlotter
from davinci_monet.plots.renderers.per_site_timeseries import (
    PerSiteTimeSeriesPlotter,
    plot_per_site_timeseries,
)

# Statistical plots
from davinci_monet.plots.renderers.scatter import ScatterPlotter, plot_scatter
from davinci_monet.plots.renderers.scorecard import ScorecardPlotter, plot_scorecard
from davinci_monet.plots.renderers.site_timeseries import (
    SiteTimeSeriesPlotter,
    plot_site_timeseries,
)

# Spatial plots (imported from subpackage)
from davinci_monet.plots.renderers.spatial import (
    BaseSpatialPlotter,
    MapConfig,
    SpatialBiasPlotter,
    SpatialDistributionPlotter,
    SpatialOverlayPlotter,
    get_domain_extent,
    get_projection,
    plot_spatial_bias,
    plot_spatial_distribution,
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
    "SiteTimeSeriesPlotter",
    "plot_site_timeseries",
    "FlightTimeSeriesPlotter",
    "plot_flight_timeseries",
    "PerSiteTimeSeriesPlotter",
    "plot_per_site_timeseries",
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
    "SpatialDistributionPlotter",
    "plot_spatial_bias",
    "plot_spatial_overlay",
    "plot_spatial_distribution",
    "get_domain_extent",
    "get_projection",
]
