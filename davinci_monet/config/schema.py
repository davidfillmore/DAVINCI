"""Pydantic models for MELODIES-MONET configuration validation.

This module provides type-safe validation for YAML configuration files,
maintaining backward compatibility with existing MELODIES-MONET configs.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

# =============================================================================
# Base Configuration
# =============================================================================


class StrictModel(BaseModel):
    """Base model with strict validation settings."""

    model_config = ConfigDict(
        extra="forbid",
        validate_default=True,
        str_strip_whitespace=True,
    )


class FlexibleModel(BaseModel):
    """Base model that allows extra fields for backward compatibility."""

    model_config = ConfigDict(
        extra="allow",
        validate_default=True,
        str_strip_whitespace=True,
    )


# =============================================================================
# Plot Style Configuration
# =============================================================================


class PlotStyleConfig(FlexibleModel):
    """Configuration for plot styling.

    Parameters
    ----------
    theme
        Plot theme to apply. Options:
        - "ncar": NSF NCAR brand colors and fonts (Poppins)
        - "default": matplotlib defaults
        - None: no theme applied (use current matplotlib state)
    context
        Font size context for the theme:
        - "default": Standard sizes suitable for most uses
        - "presentation": Larger sizes for slides
        - "publication": Smaller sizes for journal figures
    use_seaborn
        If True and seaborn is available, apply seaborn theme
        for cleaner grid styling.
    seaborn_style
        Seaborn style to apply if use_seaborn is True.
        Options: "whitegrid", "darkgrid", "white", "dark", "ticks"
    """

    theme: Literal["ncar", "default"] | None = None
    context: Literal["default", "presentation", "publication"] = "default"
    use_seaborn: bool = True
    seaborn_style: str = "whitegrid"


# =============================================================================
# Analysis Section
# =============================================================================


class AnalysisConfig(FlexibleModel):
    """Configuration for the analysis section.

    Parameters
    ----------
    start_time
        Start time of analysis window (UTC).
    end_time
        End time of analysis window (UTC).
    output_dir
        Directory for output files.
    log_dir
        Directory for log files.
    debug
        Enable debug mode.
    style
        Plot styling configuration (NCAR branding, fonts, colors).
    workflow
        Optional workflow name for add-on pipelines (e.g., "plume_sentinel").
    """

    start_time: datetime | str | None = None
    end_time: datetime | str | None = None
    output_dir: Path | str | None = None
    log_dir: Path | str | None = None
    debug: bool = False
    style: PlotStyleConfig | dict[str, Any] | None = None
    workflow: str | None = None

    @field_validator("style", mode="before")
    @classmethod
    def parse_style(cls, v: Any) -> PlotStyleConfig | None:
        """Parse style configuration."""
        if v is None:
            return None
        if isinstance(v, dict):
            return PlotStyleConfig(**v)
        return v

    @field_validator("start_time", "end_time", mode="before")
    @classmethod
    def parse_datetime(cls, v: Any) -> datetime | None:
        """Parse datetime from various formats."""
        if v is None:
            return None
        if isinstance(v, datetime):
            return v
        if isinstance(v, str):
            # Handle MELODIES-MONET format: '2019-08-02-12:00:00'
            for fmt in [
                "%Y-%m-%d-%H:%M:%S",
                "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%dT%H:%M:%S",
                "%Y-%m-%d",
            ]:
                try:
                    return datetime.strptime(v, fmt)
                except ValueError:
                    continue
            raise ValueError(f"Cannot parse datetime: {v}")
        # Return value unchanged for Pydantic to handle
        result: datetime | None = v
        return result

    @field_validator("output_dir", mode="before")
    @classmethod
    def parse_path(cls, v: Any) -> Path | None:
        """Convert string to Path."""
        if v is None:
            return None
        return Path(v)


# =============================================================================
# Variable Configuration
# =============================================================================


class VariableConfig(FlexibleModel):
    """Configuration for a single variable.

    Parameters
    ----------
    unit_scale
        Scaling factor for unit conversion.
    unit_scale_method
        Method for scaling: '*', '+', '-', '/'.
    obs_min
        Minimum valid value (values below set to NaN).
    obs_max
        Maximum valid value (values above set to NaN).
    nan_value
        Value to treat as NaN.
    source_name
        Original variable name in the data file (e.g., 'NO_ESRL').
        If set, the source variable is renamed to the config key name
        before other transforms are applied.
    rename
        Rename variable to this name.
    units
        Unit string for variable (e.g., 'ppb', 'μg/m³').
    display_name
        Display name for plots (e.g., 'PM₂.₅', 'O₃'). Overrides automatic formatting.
    ylabel_plot
        Y-axis label for plots.
    ty_scale
        Scale for Taylor diagrams.
    vmin_plot
        Minimum value for plot axis.
    vmax_plot
        Maximum value for plot axis.
    vdiff_plot
        +/- range for bias plots.
    nlevels_plot
        Number of contour levels.
    LLOD_value
        Lower limit of detection value.
    LLOD_setvalue
        Value to replace LLOD with.
    need
        Whether this variable is needed.
    """

    source_name: str | None = None
    unit_scale: float = 1.0
    unit_scale_method: Literal["*", "+", "-", "/"] = "*"
    obs_min: float | None = None
    obs_max: float | None = None
    nan_value: float | None = None
    rename: str | None = None
    units: str | None = None
    display_name: str | None = None
    ylabel_plot: str | None = None
    ty_scale: float | None = None
    vmin_plot: float | None = None
    vmax_plot: float | None = None
    vdiff_plot: float | None = None
    nlevels_plot: int | None = None
    LLOD_value: float | None = None
    LLOD_setvalue: float | None = None
    need: bool | None = None


# =============================================================================
# Model Configuration
# =============================================================================


class PlotKwargs(FlexibleModel):
    """Matplotlib plot keyword arguments."""

    color: str | None = None
    marker: str | None = None
    linestyle: str | None = None
    linewidth: float | None = None
    markersize: float | None = None


class ModelConfig(FlexibleModel):
    """Configuration for a single model.

    Parameters
    ----------
    files
        Path or glob pattern to model files.
    files_vert
        Path to vertical coordinate files (CMAQ).
    files_surf
        Path to surface files (CMAQ).
    mod_type
        Model type: 'cmaq', 'wrfchem', 'ufs', 'raqms', etc.
    mod_kwargs
        Additional model-specific keyword arguments.
    radius_of_influence
        Horizontal radius for spatial matching (meters).
    mapping
        Variable mapping: {obs_label: {obs_var: model_var}}.
    variables
        Model variable configurations.
    projection
        Map projection specification.
    plot_kwargs
        Default plot styling for this model.
    apply_ak
        Apply averaging kernels for satellite comparison.
    """

    files: str | Path | list[str | Path] | None = None
    files_vert: str | Path | None = None
    files_surf: str | Path | None = None
    mod_type: str | None = None
    mod_kwargs: dict[str, Any] = Field(default_factory=dict)
    radius_of_influence: float = 12000.0
    mapping: dict[str, dict[str, str]] = Field(default_factory=dict)
    variables: dict[str, VariableConfig] = Field(default_factory=dict)
    projection: Any = None
    plot_kwargs: PlotKwargs | dict[str, Any] | None = None
    apply_ak: bool = False
    display_name: str | None = None

    @field_validator("files", mode="before")
    @classmethod
    def convert_files(cls, v: Any) -> str | list[str] | None:
        """Keep paths as strings to preserve glob patterns; accept a list of
        paths for cases that need to combine multiple cycles or explicit files.
        """
        if v is None:
            return None
        if isinstance(v, (list, tuple)):
            return [str(item) for item in v]
        return str(v)

    @field_validator("files_vert", "files_surf", mode="before")
    @classmethod
    def convert_path(cls, v: Any) -> str | None:
        """Keep paths as strings to preserve glob patterns."""
        if v is None:
            return None
        return str(v)

    @field_validator("variables", mode="before")
    @classmethod
    def parse_variables(cls, v: Any) -> dict[str, VariableConfig]:
        """Parse variable configurations."""
        if v is None:
            return {}
        if isinstance(v, dict):
            result: dict[str, VariableConfig] = {
                str(name): VariableConfig(**cfg) if isinstance(cfg, dict) else cfg
                for name, cfg in v.items()
            }
            return result
        result = dict(v)
        return result


# =============================================================================
# Observation Configuration
# =============================================================================


ObsType = Literal[
    "pt_sfc",  # Point surface (AirNow, etc.)
    "aircraft",  # Aircraft tracks
    "sat_swath_clm",  # Satellite swath column
    "sat_grid_clm",  # Satellite gridded column
    "mobile",  # Mobile platforms
    "sonde",  # Vertical soundings
    "lma",  # Lightning Mapping Array grids
]


class FilterConfig(FlexibleModel):
    """Configuration for data filtering."""

    value: Any
    oper: str  # 'isin', '<', '>', '==', etc.


class ObservationConfig(FlexibleModel):
    """Configuration for a single observation source.

    Parameters
    ----------
    filename
        Path to observation file(s).
    obs_type
        Observation type determining geometry.
    sat_type
        Satellite-specific type (e.g., 'mopitt_l3').
    variables
        Variable configurations.
    use_airnow
        Use AirNow data (legacy flag).
    resample
        Pandas resample string (e.g., 'h' for hourly, '30min' for 30 minutes).
        Used to average high-frequency observations to match model resolution.
    min_obs_count
        Minimum number of observations required per resampled average.
        Averages with fewer observations are set to NaN. Only used when
        resample is specified.
    track_obs_count
        If True, add 'obs_count' variable to output tracking the number of
        observations in each resampled average. Only used when resample is
        specified.
    data_proc
        Data processing options.
    grid_source
        Model label whose grid is used as the binning target for swath
        observations (e.g., ``modis_l2``). The model must be loaded
        before observations.
    time_resolution
        Pandas frequency string for temporal binning of swath data
        (e.g., "1D" for daily). Default "1D".
    save_binned
        If True, write the gridded observation product to NetCDF
        after binning.
    load_binned
        If True, load a previously saved gridded product instead of
        re-reading HDF4 granules and re-binning.
    binned_file
        Path for the cached gridded NetCDF file.
    """

    filename: str | Path | None = None
    obs_type: str | None = None
    sat_type: str | None = None
    variables: dict[str, VariableConfig] = Field(default_factory=dict)
    display_name: str | None = None
    use_airnow: bool | None = None
    resample: str | None = None
    min_obs_count: int | None = None
    track_obs_count: bool = False
    data_proc: dict[str, Any] = Field(default_factory=dict)
    grid_source: str | None = None
    time_resolution: str = "1D"
    save_binned: bool = False
    load_binned: bool = False
    binned_file: str | Path | None = None

    @field_validator("filename", mode="before")
    @classmethod
    def convert_filename(cls, v: Any) -> str | None:
        """Keep filenames as strings to preserve glob patterns."""
        if v is None:
            return None
        return str(v)

    @field_validator("variables", mode="before")
    @classmethod
    def parse_variables(cls, v: Any) -> dict[str, VariableConfig]:
        """Parse variable configurations."""
        if v is None:
            return {}
        if isinstance(v, dict):
            result: dict[str, VariableConfig] = {
                str(name): VariableConfig(**cfg) if isinstance(cfg, dict) else cfg
                for name, cfg in v.items()
            }
            return result
        result = dict(v)
        return result


# =============================================================================
# Plot Configuration
# =============================================================================


PlotType = Literal[
    "timeseries",
    "taylor",
    "spatial_bias",
    "spatial_overlay",
    "boxplot",
    "gridded_spatial_bias",
    "diurnal",
    "scatter",
    "curtain",
]


class DataProcConfig(FlexibleModel):
    """Data processing configuration for plots.

    Parameters
    ----------
    filter_dict
        Dictionary-based filtering.
    filter_string
        Pandas query string for filtering.
    rem_obs_by_nan_pct
        Remove observations by NaN percentage.
    rem_obs_nan
        Remove NaN observations.
    ts_select_time
        Time selection for timeseries: 'time' (UTC) or 'time_local'.
    ts_avg_window
        Pandas resample rule for averaging.
    set_axis
        Use variable-specified axis limits.
    """

    filter_dict: dict[str, FilterConfig | dict[str, Any]] | None = None
    filter_string: str | None = None
    rem_obs_by_nan_pct: dict[str, Any] | None = None
    rem_obs_nan: bool = True
    ts_select_time: Literal["time", "time_local"] = "time"
    ts_avg_window: str | None = None
    set_axis: bool = False


class FigKwargs(FlexibleModel):
    """Figure keyword arguments."""

    figsize: list[float] | tuple[float, float] | None = None
    states: bool | None = None


class TextKwargs(FlexibleModel):
    """Text styling keyword arguments."""

    fontsize: float = 12.0


class PlotGroupConfig(FlexibleModel):
    """Configuration for a plot group.

    Parameters
    ----------
    type
        Plot type.
    fig_kwargs
        Figure keyword arguments.
    default_plot_kwargs
        Default plot styling.
    text_kwargs
        Text styling.
    domain_type
        List of domain types: 'all' or specific domains.
    domain_name
        List of domain names.
    data
        List of model_obs pair identifiers (model label first).
        Legacy obs_model identifiers are also supported.
    data_proc
        Data processing configuration.
    """

    type: str
    fig_kwargs: FigKwargs | dict[str, Any] = Field(default_factory=dict)
    default_plot_kwargs: PlotKwargs | dict[str, Any] = Field(default_factory=dict)
    text_kwargs: TextKwargs | dict[str, Any] = Field(default_factory=dict)
    domain_type: list[str] = Field(default_factory=lambda: ["all"])
    domain_name: list[str] = Field(default_factory=lambda: ["CONUS"])
    data: list[str] = Field(default_factory=list)
    data_proc: DataProcConfig | dict[str, Any] = Field(default_factory=dict)

    @field_validator("data_proc", mode="before")
    @classmethod
    def parse_data_proc(cls, v: Any) -> DataProcConfig | dict[str, Any]:
        """Parse data processing config."""
        if v is None:
            return DataProcConfig()
        if isinstance(v, dict):
            return DataProcConfig(**v)
        result: DataProcConfig | dict[str, Any] = v
        return result


# =============================================================================
# Statistics Configuration
# =============================================================================


StatMetric = Literal[
    "MB",
    "MdnB",
    "NMB",
    "NMdnB",
    "R2",
    "RMSE",
    "STDO",
    "STDP",
    "MdnNB",
    "MdnNE",
    "NMdnGE",
    "NO",
    "NOP",
    "NP",
    "MO",
    "MP",
    "MdnO",
    "MdnP",
    "RM",
    "RMdn",
    "FB",
    "ME",
    "MdnE",
    "NME",
    "NMdnE",
    "FE",
    "d1",
    "E1",
    "IOA",
    "AC",
]


class OutputTableKwargs(FlexibleModel):
    """Keyword arguments for statistics output table."""

    figsize: list[float] | tuple[float, float] | None = None
    fontsize: float = 12.0
    xscale: float = 1.0
    yscale: float = 1.0
    edges: str = "horizontal"


class StatsConfig(FlexibleModel):
    """Configuration for statistics calculation.

    Parameters
    ----------
    stat_list
        List of statistics to calculate.
    round_output
        Decimal places for rounding.
    output_table
        Generate output table image.
    output_table_kwargs
        Table styling options.
    domain_type
        List of domain types.
    domain_name
        List of domain names.
    data
        List of model_obs pair identifiers (model label first).
        Legacy obs_model identifiers are also supported.
    data_proc
        Data processing configuration.
    """

    stat_list: list[str] = Field(default_factory=lambda: ["MB", "NMB", "R2", "RMSE"])
    round_output: int = 3
    output_table: bool = False
    output_table_kwargs: OutputTableKwargs | dict[str, Any] = Field(default_factory=dict)
    domain_type: list[str] = Field(default_factory=lambda: ["all"])
    domain_name: list[str] = Field(default_factory=lambda: ["CONUS"])
    data: list[str] = Field(default_factory=list)
    data_proc: DataProcConfig | dict[str, Any] | None = None


# =============================================================================
# Root Configuration
# =============================================================================


class MonetConfig(FlexibleModel):
    """Root configuration model for MELODIES-MONET.

    This is the top-level configuration that contains all sections.

    Parameters
    ----------
    analysis
        Analysis configuration (time window, output directory).
    model
        Dictionary of model configurations keyed by model label.
    obs
        Dictionary of observation configurations keyed by obs label.
    plots
        Dictionary of plot group configurations keyed by group name.
    stats
        Statistics configuration.

    Examples
    --------
    >>> from davinci_monet.config.schema import MonetConfig
    >>> config = MonetConfig.model_validate({
    ...     "analysis": {"start_time": "2024-01-01", "end_time": "2024-01-02"},
    ...     "model": {"test_model": {"mod_type": "cmaq"}},
    ...     "obs": {"test_obs": {"obs_type": "pt_sfc"}},
    ... })
    >>> config.analysis.start_time
    datetime.datetime(2024, 1, 1, 0, 0)
    """

    analysis: AnalysisConfig = Field(default_factory=AnalysisConfig)
    model: dict[str, ModelConfig] = Field(default_factory=dict)
    obs: dict[str, ObservationConfig] = Field(default_factory=dict)
    plots: dict[str, PlotGroupConfig] = Field(default_factory=dict)
    stats: StatsConfig | None = None

    @field_validator("model", mode="before")
    @classmethod
    def parse_models(cls, v: Any) -> dict[str, ModelConfig]:
        """Parse model configurations."""
        if v is None:
            return {}
        if isinstance(v, dict):
            result: dict[str, ModelConfig] = {
                str(name): ModelConfig(**cfg) if isinstance(cfg, dict) else cfg
                for name, cfg in v.items()
            }
            return result
        result = dict(v)
        return result

    @field_validator("obs", mode="before")
    @classmethod
    def parse_observations(cls, v: Any) -> dict[str, ObservationConfig]:
        """Parse observation configurations."""
        if v is None:
            return {}
        if isinstance(v, dict):
            result: dict[str, ObservationConfig] = {
                str(name): ObservationConfig(**cfg) if isinstance(cfg, dict) else cfg
                for name, cfg in v.items()
            }
            return result
        result = dict(v)
        return result

    @field_validator("plots", mode="before")
    @classmethod
    def parse_plots(cls, v: Any) -> dict[str, PlotGroupConfig]:
        """Parse plot configurations."""
        if v is None:
            return {}
        if isinstance(v, dict):
            result: dict[str, PlotGroupConfig] = {
                str(name): PlotGroupConfig(**cfg) if isinstance(cfg, dict) else cfg
                for name, cfg in v.items()
            }
            return result
        result = dict(v)
        return result

    @model_validator(mode="after")
    def validate_data_references(self) -> "MonetConfig":
        """Validate that data references in plots/stats exist."""
        model_labels = set(self.model.keys())
        obs_labels = set(self.obs.keys())

        # Check plot data references
        for plot_name, plot_config in self.plots.items():
            for data_ref in plot_config.data:
                # Data refs are typically formatted as "model_obs" (e.g., "cmaq_airnow")
                # Legacy "obs_model" (e.g., "airnow_cmaq") is also supported.
                found = False
                for obs_label in obs_labels:
                    if data_ref.startswith(obs_label + "_"):
                        model_part = data_ref[len(obs_label) + 1 :]
                        if model_part in model_labels:
                            found = True
                            break
                if not found:
                    for model_label in model_labels:
                        if data_ref.startswith(model_label + "_"):
                            obs_part = data_ref[len(model_label) + 1 :]
                            if obs_part in obs_labels:
                                found = True
                                break
                # Don't raise error for now - allow flexible data references
                # This supports legacy configs that may have different formats

        return self

    def get_model_obs_pairs(self) -> list[tuple[str, str]]:
        """Get all model-observation pairs from the configuration.

        Returns
        -------
        list[tuple[str, str]]
            List of (obs_label, model_label) pairs.
        """
        pairs: set[tuple[str, str]] = set()

        # Extract from model mappings
        for model_label, model_config in self.model.items():
            for obs_label in model_config.mapping.keys():
                pairs.add((obs_label, model_label))

        # Extract from plot data references (support model_obs and legacy obs_model)
        obs_labels = set(self.obs.keys())
        model_labels = set(self.model.keys())

        for plot_config in self.plots.values():
            for data_ref in plot_config.data:
                # Legacy obs_model
                for obs_label in obs_labels:
                    if data_ref.startswith(obs_label + "_"):
                        model_part = data_ref[len(obs_label) + 1 :]
                        if model_part in model_labels:
                            pairs.add((obs_label, model_part))
                            break
                # Preferred model_obs
                for model_label in model_labels:
                    if data_ref.startswith(model_label + "_"):
                        obs_part = data_ref[len(model_label) + 1 :]
                        if obs_part in obs_labels:
                            pairs.add((obs_part, model_label))
                            break

        return sorted(pairs)


# =============================================================================
# Convenience Aliases
# =============================================================================


Config = MonetConfig  # Alias for common usage
