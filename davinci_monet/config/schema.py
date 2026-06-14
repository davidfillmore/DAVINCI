"""Pydantic schemas for DAVINCI configuration validation."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

# =============================================================================
# Base Configuration
# =============================================================================


class StrictSchema(
    BaseModel,
    extra="forbid",
    validate_default=True,
    str_strip_whitespace=True,
):
    """Base dataset with strict validation settings."""


class FlexibleSchema(
    BaseModel,
    extra="allow",
    validate_default=True,
    str_strip_whitespace=True,
):
    """Base schema that allows reader-specific extra fields."""


# =============================================================================
# Plot Style Configuration
# =============================================================================


class PlotStyleConfig(FlexibleSchema):
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


class AnalysisConfig(FlexibleSchema):
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
    """

    start_time: datetime | str | None = None
    end_time: datetime | str | None = None
    output_dir: Path | str | None = None
    log_dir: Path | str | None = None
    debug: bool = False
    style: PlotStyleConfig | dict[str, Any] | None = None

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


class VariableConfig(FlexibleSchema):
    """Configuration for a single variable.

    Parameters
    ----------
    unit_scale
        Scaling factor for unit conversion.
    unit_scale_method
        Method for scaling: '*', '+', '-', '/'.
    valid_min
        Minimum valid value; values below this threshold are set to NaN.
    valid_max
        Maximum valid value; values above this threshold are set to NaN.
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
    valid_min: float | None = None
    valid_max: float | None = None
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
# Plot / Source Keyword Arguments
# =============================================================================


class PlotKwargs(FlexibleSchema):
    """Matplotlib plot keyword arguments."""

    color: str | None = None
    marker: str | None = None
    linestyle: str | None = None
    linewidth: float | None = None
    markersize: float | None = None


class FilterConfig(FlexibleSchema):
    """Configuration for data filtering."""

    value: Any
    oper: str  # 'isin', '<', '>', '==', etc.


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


class SourceConfig(FlexibleSchema):
    """Unified configuration for a single data source.

    A data source is data with a declared geometry. Pairing direction is chosen
    by each pair's ``geometry`` field and by geometry precedence when that field
    is omitted. Extra reader-specific keys are accepted and passed through.
    """

    type: str | None = None
    files: str | Path | list[str | Path] | None = None
    filename: str | Path | None = None
    variables: dict[str, VariableConfig] = Field(default_factory=dict)
    radius_of_influence: float = 12000.0
    display_name: str | None = None
    resample: str | None = None
    min_geometry_count: int | None = None
    track_geometry_count: bool = False

    @model_validator(mode="before")
    @classmethod
    def reject_source_mapping(cls, data: Any) -> Any:
        """Pair variables must live in ``pairs:``, not in source metadata."""
        if isinstance(data, dict) and "mapping" in data:
            raise ValueError("source-level mapping is not supported; use pairs variables")
        return data

    @field_validator("files", mode="before")
    @classmethod
    def _convert_files(cls, v: Any) -> str | list[str] | None:
        if v is None:
            return None
        if isinstance(v, (list, tuple)):
            return [str(item) for item in v]
        return str(v)

    @field_validator("filename", mode="before")
    @classmethod
    def _convert_filename(cls, v: Any) -> str | None:
        return None if v is None else str(v)

    @field_validator("variables", mode="before")
    @classmethod
    def _parse_variables(cls, v: Any) -> dict[str, VariableConfig]:
        if v is None:
            return {}
        if isinstance(v, dict):
            return {
                str(name): VariableConfig(**cfg) if isinstance(cfg, dict) else cfg
                for name, cfg in v.items()
            }
        return dict(v)


class SourcePairConfig(FlexibleSchema):
    """Binary pair definition.

    Pairs use ``sources``/``geometry``/``variables``.
    """

    sources: list[str] = Field(default_factory=list)
    geometry: str | None = None
    variables: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_pair_shape(self) -> "SourcePairConfig":
        if not self.sources:
            raise ValueError("pair must define 'sources' (exactly two source labels)")
        if len(self.sources) != 2:
            raise ValueError("pair 'sources' must contain exactly two labels")
        if self.geometry is not None and self.geometry not in self.sources:
            raise ValueError("'geometry' must be one of the pair sources")
        missing = [label for label in self.sources if label not in self.variables]
        if missing:
            raise ValueError("pair 'variables' missing source label(s): " + ", ".join(missing))
        return self


class DataProcConfig(FlexibleSchema):
    """Data processing configuration for plots.

    Parameters
    ----------
    filter_dict
        Dictionary-based filtering.
    filter_string
        Pandas query string for filtering.
    rem_geometry_by_nan_pct
        Remove datasets by NaN percentage.
    rem_geometry_nan
        Remove NaN datasets.
    ts_select_time
        Time selection for timeseries: 'time' (UTC) or 'time_local'.
    ts_avg_window
        Pandas resample rule for averaging.
    set_axis
        Use variable-specified axis limits.
    """

    filter_dict: dict[str, FilterConfig | dict[str, Any]] | None = None
    filter_string: str | None = None
    rem_geometry_by_nan_pct: dict[str, Any] | None = None
    rem_geometry_nan: bool = True
    ts_select_time: Literal["time", "time_local"] = "time"
    ts_avg_window: str | None = None
    set_axis: bool = False


class FigKwargs(FlexibleSchema):
    """Figure keyword arguments."""

    figsize: list[float] | tuple[float, float] | None = None
    states: bool | None = None


class TextKwargs(FlexibleSchema):
    """Text styling keyword arguments."""

    fontsize: float = 12.0


class PlotGroupConfig(FlexibleSchema):
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
        List of pair identifiers.
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
    "STDG",
    "STDD",
    "MG",
    "MD",
    "MdnG",
    "MdnD",
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


class OutputTableKwargs(FlexibleSchema):
    """Keyword arguments for statistics output table."""

    figsize: list[float] | tuple[float, float] | None = None
    fontsize: float = 12.0
    xscale: float = 1.0
    yscale: float = 1.0
    edges: str = "horizontal"


class StatsConfig(FlexibleSchema):
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
        List of pair identifiers.
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
# AI Summary Configuration
# =============================================================================


class SummaryConfig(FlexibleSchema):
    """Configuration for the optional AI analysis summary stage.

    When ``enabled`` is true, a final pipeline stage sends the run's
    statistics, config metadata, and selected plot images to an AI model
    (via the Anthropic API directly, or via OpenRouter) and writes a markdown
    brief into the analysis output directory.
    """

    enabled: bool = False
    provider: Literal["anthropic", "openrouter"] = "anthropic"
    model: str = "claude-haiku-4-5"
    max_tokens: int = 2000
    api_key_env: str = "ANTHROPIC_API_KEY"
    api_key_file: str | None = None
    plots: list[str] | None = None
    max_images: int = 8
    output_filename: str = "AI_summary.md"
    instructions: str | None = None
    templates: dict[str, dict] | None = None
    template_overrides: dict[str, str] | None = None

    @model_validator(mode="after")
    def _apply_provider_defaults(self) -> "SummaryConfig":
        """Flip Anthropic-default sentinels to OpenRouter equivalents.

        Only fields still holding the Anthropic default are changed, so an
        explicit user value is never overridden.
        """
        if self.provider == "openrouter":
            if self.model == "claude-haiku-4-5":
                self.model = "anthropic/claude-haiku-4.5"
            if self.api_key_env == "ANTHROPIC_API_KEY":
                self.api_key_env = "OPENROUTER_API_KEY"
        return self


# =============================================================================
# Root Configuration
# =============================================================================


class MonetConfig(StrictSchema):
    """Root configuration dataset for MELODIES-MONET.

    This is the top-level configuration that contains all sections.

    Parameters
    ----------
    analysis
        Analysis configuration (time window, output directory).
    sources
        Dictionary of unified data-source configurations keyed by source label.
    pairs
        Dictionary of binary pair definitions keyed by pair name.
    plots
        Dictionary of plot group configurations keyed by group name.
    stats
        Statistics configuration.

    Examples
    --------
    >>> from davinci_monet.config.schema import MonetConfig
    >>> config = MonetConfig(**{
    ...     "analysis": {"start_time": "2024-01-01", "end_time": "2024-01-02"},
    ...     "sources": {
    ...         "cam": {"type": "cesm_fv"},
    ...         "airnow": {"type": "pt_sfc"},
    ...     },
    ... })
    >>> config.analysis.start_time
    datetime.datetime(2024, 1, 1, 0, 0)
    """

    analysis: AnalysisConfig = Field(default_factory=AnalysisConfig)
    # Data sources keyed by dataset label.
    sources: dict[str, SourceConfig] = Field(default_factory=dict)
    pairs: dict[str, SourcePairConfig] = Field(default_factory=dict)
    plots: dict[str, PlotGroupConfig] = Field(default_factory=dict)
    stats: StatsConfig | None = None
    summary: SummaryConfig | None = None

    @field_validator("sources", mode="before")
    @classmethod
    def parse_sources(cls, v: Any) -> dict[str, SourceConfig]:
        """Parse unified data-source configurations."""
        if v is None:
            return {}
        if isinstance(v, dict):
            return {
                str(name): SourceConfig(**cfg) if isinstance(cfg, dict) else cfg
                for name, cfg in v.items()
            }
        return dict(v)

    @field_validator("pairs", mode="before")
    @classmethod
    def parse_pairs(cls, v: Any) -> dict[str, SourcePairConfig]:
        """Parse unified source-pair configurations."""
        if v is None:
            return {}
        if isinstance(v, dict):
            return {
                str(name): SourcePairConfig(**cfg) if isinstance(cfg, dict) else cfg
                for name, cfg in v.items()
            }
        return dict(v)

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
    def validate_data_names(self) -> "MonetConfig":
        """Validate that plot/stats data names resolve to defined pairs.

        Plot ``data`` entries normally name a pair (or a single source for
        single-source plots). Names are checked permissively; unknown names are
        tolerated rather than raised, since plot data refs can
        legitimately combine labels in flexible ways — but the hook is kept so
        future validation can hang off it.
        """
        return self


# =============================================================================
# Convenience Aliases
# =============================================================================


Config = MonetConfig  # Alias for common usage
