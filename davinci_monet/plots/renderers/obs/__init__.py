"""Observation-only plot renderers.

These renderers work with raw observation datasets (not paired model-obs data).
They inherit from ObsPlotter instead of BasePlotter.

Available renderers:
- FlightTrackMapPlotter: Cartopy map colored by variable value
- VerticalProfilePlotter: Altitude vs. concentration (scatter or binned)
- ObsTimeSeriesPlotter: Variable vs. time with optional altitude overlay
- ObsHistogramPlotter: Distribution histogram with optional stats
- ObsLMADensityPlotter: Cartopy map of LMA gridded flash density
"""

from davinci_monet.plots.renderers.obs.flight_track_map import FlightTrackMapPlotter
from davinci_monet.plots.renderers.obs.vertical_profile import VerticalProfilePlotter
from davinci_monet.plots.renderers.obs.obs_timeseries import ObsTimeSeriesPlotter
from davinci_monet.plots.renderers.obs.obs_histogram import ObsHistogramPlotter
from davinci_monet.plots.renderers.obs.obs_lma_density import ObsLMADensityPlotter

__all__ = [
    "FlightTrackMapPlotter",
    "VerticalProfilePlotter",
    "ObsTimeSeriesPlotter",
    "ObsHistogramPlotter",
    "ObsLMADensityPlotter",
]
