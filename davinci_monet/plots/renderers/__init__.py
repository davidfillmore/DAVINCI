"""Plot renderers for DAVINCI-MONET.

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
- SpatialOverlayPlotter: Model/obs overlays
- SpatialDistributionPlotter: Value distribution maps

Specialized plots:
- CurtainPlotter: Vertical cross-sections
- ScorecardPlotter: Multi-metric scorecards
"""

# Temporal plots
from davinci_monet.plots.renderers.timeseries import (
    TimeSeriesPlotter,
    plot_timeseries,
)
from davinci_monet.plots.renderers.diurnal import (
    DiurnalPlotter,
    plot_diurnal,
)
from davinci_monet.plots.renderers.site_timeseries import (
    SiteTimeSeriesPlotter,
    plot_site_timeseries,
)
from davinci_monet.plots.renderers.flight_timeseries import (
    FlightTimeSeriesPlotter,
    plot_flight_timeseries,
)

# Statistical plots
from davinci_monet.plots.renderers.scatter import (
    ScatterPlotter,
    plot_scatter,
)
from davinci_monet.plots.renderers.taylor import (
    TaylorPlotter,
    plot_taylor,
)
from davinci_monet.plots.renderers.boxplot import (
    BoxPlotter,
    plot_boxplot,
)

# Specialized plots
from davinci_monet.plots.renderers.curtain import (
    CurtainPlotter,
    plot_curtain,
)
from davinci_monet.plots.renderers.scorecard import (
    ScorecardPlotter,
    plot_scorecard,
)

# Spatial plots (imported from subpackage)
from davinci_monet.plots.renderers.spatial import (
    BaseSpatialPlotter,
    MapConfig,
    SpatialBiasPlotter,
    SpatialOverlayPlotter,
    SpatialDistributionPlotter,
    plot_spatial_bias,
    plot_spatial_overlay,
    plot_spatial_distribution,
    get_domain_extent,
    get_projection,
)

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
