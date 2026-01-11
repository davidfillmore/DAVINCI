"""Plotting module for DAVINCI-MONET.

This module provides a comprehensive plotting system for model-observation
comparison, including time series, spatial maps, Taylor diagrams, and more.

Quick Start
-----------
>>> from davinci_monet.plots import get_plotter, plot_timeseries
>>>
>>> # Using convenience function
>>> fig = plot_timeseries(paired_data, "obs_o3", "model_o3")
>>>
>>> # Using plotter instance
>>> plotter = get_plotter("scatter")
>>> fig = plotter.plot(paired_data, "obs_o3", "model_o3")

Available Plot Types
--------------------
Temporal:
    timeseries : Time series comparison
    diurnal : Diurnal cycle comparison

Statistical:
    scatter : Scatter plot with regression
    taylor : Taylor diagram
    boxplot : Box plot comparison

Spatial:
    spatial_bias : Bias map
    spatial_overlay : Model contour + obs scatter
    spatial_distribution : Value distribution map

Specialized:
    curtain : Vertical cross-section
    scorecard : Multi-metric heatmap
"""

# Base classes and utilities
from davinci_monet.plots.base import (
    BasePlotter,
    PlotConfig,
    FigureConfig,
    TextConfig,
    StyleConfig,
    DomainConfig,
    merge_config_dicts,
    get_variable_label,
    get_variable_units,
    format_label_with_units,
    calculate_symmetric_limits,
    calculate_data_limits,
)

# Registry and factory
from davinci_monet.plots.registry import (
    plotter_registry,
    register_plotter,
    get_plotter,
    get_plotter_class,
    list_plotters,
    has_plotter,
    get_plot_category,
    TEMPORAL_PLOTS,
    STATISTICAL_PLOTS,
    SPATIAL_PLOTS,
    SPECIALIZED_PLOTS,
    ALL_PLOT_TYPES,
)

# Import all renderers to ensure they register
from davinci_monet.plots.renderers import (
    # Temporal
    TimeSeriesPlotter,
    plot_timeseries,
    DiurnalPlotter,
    plot_diurnal,
    SiteTimeSeriesPlotter,
    plot_site_timeseries,
    FlightTimeSeriesPlotter,
    plot_flight_timeseries,
    # Statistical
    ScatterPlotter,
    plot_scatter,
    TaylorPlotter,
    plot_taylor,
    BoxPlotter,
    plot_boxplot,
    # Specialized
    CurtainPlotter,
    plot_curtain,
    ScorecardPlotter,
    plot_scorecard,
    TrackMap3DPlotter,
    plot_track_map_3d,
    # Spatial
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
    # Base classes
    "BasePlotter",
    "PlotConfig",
    "FigureConfig",
    "TextConfig",
    "StyleConfig",
    "DomainConfig",
    # Utilities
    "merge_config_dicts",
    "get_variable_label",
    "get_variable_units",
    "format_label_with_units",
    "calculate_symmetric_limits",
    "calculate_data_limits",
    # Registry
    "plotter_registry",
    "register_plotter",
    "get_plotter",
    "get_plotter_class",
    "list_plotters",
    "has_plotter",
    "get_plot_category",
    "TEMPORAL_PLOTS",
    "STATISTICAL_PLOTS",
    "SPATIAL_PLOTS",
    "SPECIALIZED_PLOTS",
    "ALL_PLOT_TYPES",
    # Temporal plotters
    "TimeSeriesPlotter",
    "plot_timeseries",
    "DiurnalPlotter",
    "plot_diurnal",
    "SiteTimeSeriesPlotter",
    "plot_site_timeseries",
    "FlightTimeSeriesPlotter",
    "plot_flight_timeseries",
    # Statistical plotters
    "ScatterPlotter",
    "plot_scatter",
    "TaylorPlotter",
    "plot_taylor",
    "BoxPlotter",
    "plot_boxplot",
    # Specialized plotters
    "CurtainPlotter",
    "plot_curtain",
    "ScorecardPlotter",
    "plot_scorecard",
    "TrackMap3DPlotter",
    "plot_track_map_3d",
    # Spatial plotters
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
