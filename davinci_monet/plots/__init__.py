"""Plotting module for DAVINCI.

This module provides a comprehensive plotting system for dataset-dataset
comparison, including time series, spatial maps, Taylor diagrams, and more.

Quick Start
-----------
>>> from davinci_monet.plots import get_plotter, plot_timeseries
>>>
>>> # Using convenience function
>>> fig = plot_timeseries(paired_data, "geometry_o3", "dataset_o3")
>>>
>>> # Using plotter instance
>>> plotter = get_plotter("scatter")
>>> fig = plotter.plot(paired_data, "geometry_o3", "dataset_o3")

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
    spatial_overlay : Dataset contour + geometry scatter
    spatial_distribution : Value distribution map

Specialized:
    curtain : Vertical cross-section
    scorecard : Multi-metric heatmap
"""

# Base classes and utilities
from davinci_monet.plots.base import (
    BasePlotter,
    DomainConfig,
    FigureConfig,
    PlotConfig,
    StyleConfig,
    TextConfig,
    calculate_data_limits,
    calculate_symmetric_limits,
    format_label_with_units,
    get_variable_label,
    get_variable_units,
    merge_config_dicts,
    resolve_source_variable,
)

# Registry and factory
from davinci_monet.plots.registry import (
    ALL_PLOT_TYPES,
    SPATIAL_PLOTS,
    SPECIALIZED_PLOTS,
    STATISTICAL_PLOTS,
    TEMPORAL_PLOTS,
    get_plot_category,
    get_plotter,
    get_plotter_class,
    has_plotter,
    list_plotters,
    plotter_registry,
    register_plotter,
)

# Import all renderers to ensure they register
from davinci_monet.plots.renderers import (  # Temporal; Statistical; Specialized; Spatial
    BaseSpatialPlotter,
    BoxPlotter,
    CurtainPlotter,
    DiurnalPlotter,
    FlightTimeSeriesPlotter,
    FlightTrackPlotter,
    HistogramPlotter,
    LMADensityPlotter,
    MapConfig,
    PerSiteTimeSeriesPlotter,
    ScatterPlotter,
    ScorecardPlotter,
    SiteTimeSeriesPlotter,
    SpatialBiasPlotter,
    SpatialDistributionPlotter,
    SpatialOverlayPlotter,
    TaylorPlotter,
    TimeSeriesPlotter,
    TrackMap3DPlotter,
    VerticalProfilePlotter,
    get_domain_extent,
    get_projection,
    plot_boxplot,
    plot_curtain,
    plot_diurnal,
    plot_flight_timeseries,
    plot_per_site_timeseries,
    plot_scatter,
    plot_scorecard,
    plot_site_timeseries,
    plot_spatial_bias,
    plot_spatial_distribution,
    plot_spatial_overlay,
    plot_taylor,
    plot_timeseries,
    plot_track_map_3d,
)

# Style configuration (NCAR branding)
from davinci_monet.plots.style import (
    DATASET_A_COLOR,
    DATASET_B_COLOR,
    FONT_SIZES_DEFAULT,
    FONT_SIZES_PRESENTATION,
    FONT_SIZES_PUBLICATION,
    NCAR_ACCENT,
    NCAR_COLORS,
    NCAR_PALETTE,
    NCAR_PRIMARY,
    NCAR_SECONDARY,
    FontSizes,
    apply_ncar_style,
    get_bias_cmap,
    get_color_for_variable,
    get_density_cmap,
    get_palette,
    get_sequential_cmap,
    reset_style,
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
    "resolve_source_variable",
    "get_variable_units",
    "format_label_with_units",
    "calculate_symmetric_limits",
    "calculate_data_limits",
    # Style configuration (NCAR branding)
    "NCAR_COLORS",
    "NCAR_PALETTE",
    "NCAR_PRIMARY",
    "NCAR_SECONDARY",
    "NCAR_ACCENT",
    "DATASET_A_COLOR",
    "DATASET_B_COLOR",
    "apply_ncar_style",
    "reset_style",
    "get_color_for_variable",
    "get_palette",
    "get_bias_cmap",
    "get_sequential_cmap",
    "get_density_cmap",
    "FontSizes",
    "FONT_SIZES_DEFAULT",
    "FONT_SIZES_PRESENTATION",
    "FONT_SIZES_PUBLICATION",
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
    "PerSiteTimeSeriesPlotter",
    "plot_per_site_timeseries",
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
    # Single-source / distribution plotters (unified onto BasePlotter)
    "HistogramPlotter",
    "VerticalProfilePlotter",
    "FlightTrackPlotter",
    "LMADensityPlotter",
]
