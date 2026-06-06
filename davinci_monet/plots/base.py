"""Base plotter classes and common utilities for DAVINCI.

This module provides:
- PlotConfig: Pydantic model for plot configuration
- BasePlotter: Abstract base class for all plotters
- Common utilities for figure creation, styling, and saving
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import matplotlib.pyplot as plt
import numpy as np

# Import NCAR style colors for defaults
from davinci_monet.core.base import (
    PlotSeries,
    paired_variable_pair_role,
    paired_variable_role,
)
from davinci_monet.plots.style import MODEL_COLOR, NCAR_PALETTE, NCAR_PRIMARY, OBS_COLOR

if TYPE_CHECKING:
    import matplotlib.axes
    import matplotlib.figure
    import xarray as xr


# =============================================================================
# Configuration Dataclasses
# =============================================================================


@dataclass
class FigureConfig:
    """Configuration for figure creation.

    Attributes
    ----------
    figsize : tuple[float, float]
        Figure size in inches (width, height).
    dpi : int
        Dots per inch for rendering.
    facecolor : str
        Figure background color.
    constrained_layout : bool
        Use constrained layout for automatic spacing.
    """

    figsize: tuple[float, float] = (8, 5)
    dpi: int = 300
    facecolor: str = "white"
    constrained_layout: bool = True


@dataclass
class TextConfig:
    """Configuration for text styling.

    All sizes are in points (1/72 inch). These are absolute measurements
    that do not scale with figure size.

    Attributes
    ----------
    fontsize : float
        Base font size for axis labels.
    title_fontsize : float
        Font size for figure/axes titles.
    tick_fontsize : float
        Font size for tick labels.
    legend : float
        Font size for primary legend text.
    legend_small : float
        Font size for crowded multi-panel legends.
    annotation : float
        Font size for text annotations (stats boxes).
    annotation_small : float
        Font size for dense multi-panel annotations.
    site_label : float
        Font size for map site markers and city labels.
    fontweight : str
        Font weight ('normal', 'bold').
    """

    fontsize: float = 14.0
    title_fontsize: float = 16.0
    tick_fontsize: float = 12.0
    legend: float = 12.0
    legend_small: float = 10.0
    annotation: float = 12.0
    annotation_small: float = 10.0
    site_label: float = 10.0
    fontweight: str = "normal"


@dataclass
class StyleConfig:
    """Configuration for plot styling.

    Attributes
    ----------
    obs_color : str
        Color for observation data (default: NCAR gray #58595B).
    model_color : str
        Color for model data (default: NCAR blue #0A5DDA).
    obs_marker : str
        Marker style for observations.
    model_marker : str
        Marker style for model.
    obs_linestyle : str
        Line style for observations.
    model_linestyle : str
        Line style for model.
    linewidth : float
        Line width.
    markersize : float
        Marker size.
    alpha : float
        Transparency (0-1).
    """

    obs_color: str = OBS_COLOR  # NCAR gray
    model_color: str = MODEL_COLOR  # NCAR blue
    obs_marker: str = "o"
    model_marker: str = "s"
    obs_linestyle: str = "-"
    model_linestyle: str = "--"
    linewidth: float = 1.5
    markersize: float = 6.0
    alpha: float = 1.0


@dataclass
class DomainConfig:
    """Configuration for spatial domain.

    Attributes
    ----------
    domain_type : str | None
        Type of domain ('all', 'epa_region', 'custom', etc.).
    domain_name : str | None
        Name of specific domain (e.g., 'R1' for EPA Region 1).
    extent : tuple[float, float, float, float] | None
        Custom extent (lon_min, lon_max, lat_min, lat_max).
    """

    domain_type: str | None = None
    domain_name: str | None = None
    extent: tuple[float, float, float, float] | None = None


@dataclass
class PlotConfig:
    """Complete configuration for a plot.

    Combines figure, text, style, and domain configurations.
    """

    figure: FigureConfig = field(default_factory=FigureConfig)
    text: TextConfig = field(default_factory=TextConfig)
    style: StyleConfig = field(default_factory=StyleConfig)
    domain: DomainConfig = field(default_factory=DomainConfig)

    # Variable configuration
    obs_var: str | None = None
    model_var: str | None = None
    obs_label: str | None = None
    model_label: str | None = None

    # Value limits
    vmin: float | None = None
    vmax: float | None = None

    # Axis labels
    xlabel: str | None = None
    ylabel: str | None = None
    title: str | None = None

    # Output
    output_dir: Path | None = None
    output_format: str = "png"

    # Debug mode
    debug: bool = False

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> PlotConfig:
        """Create PlotConfig from a dictionary.

        Parameters
        ----------
        config_dict
            Configuration dictionary, possibly from YAML.

        Returns
        -------
        PlotConfig
            Configured instance.
        """
        # Extract nested configs
        figure_dict = config_dict.get("figure", config_dict.get("fig_dict", {}))
        text_dict = config_dict.get("text", config_dict.get("text_dict", {}))
        style_dict = config_dict.get("style", config_dict.get("plot_dict", {}))
        domain_dict = config_dict.get("domain", {})

        # Handle legacy domain_type/domain_name at top level
        if "domain_type" in config_dict:
            domain_dict.setdefault("domain_type", config_dict["domain_type"])
        if "domain_name" in config_dict:
            domain_dict.setdefault("domain_name", config_dict["domain_name"])

        figure = FigureConfig(
            **{k: v for k, v in figure_dict.items() if k in FigureConfig.__dataclass_fields__}
        )
        text = TextConfig(
            **{k: v for k, v in text_dict.items() if k in TextConfig.__dataclass_fields__}
        )
        style = StyleConfig(
            **{k: v for k, v in style_dict.items() if k in StyleConfig.__dataclass_fields__}
        )
        domain = DomainConfig(
            **{k: v for k, v in domain_dict.items() if k in DomainConfig.__dataclass_fields__}
        )

        # Extract top-level fields
        top_level_fields = {
            "obs_var",
            "model_var",
            "obs_label",
            "model_label",
            "vmin",
            "vmax",
            "xlabel",
            "ylabel",
            "title",
            "output_dir",
            "output_format",
            "debug",
        }
        top_level = {k: v for k, v in config_dict.items() if k in top_level_fields}

        if "output_dir" in top_level and top_level["output_dir"] is not None:
            top_level["output_dir"] = Path(top_level["output_dir"])

        return cls(
            figure=figure,
            text=text,
            style=style,
            domain=domain,
            **top_level,
        )


# =============================================================================
# Base Plotter
# =============================================================================


class BasePlotter(ABC):
    """Abstract base class for all plotters.

    Provides common functionality for figure management, styling,
    and saving. Subclasses implement specific plot types.

    Parameters
    ----------
    config
        Plot configuration. If None, uses defaults.

    Attributes
    ----------
    config : PlotConfig
        The plot configuration.
    default_figsize : tuple[float, float]
        Default figure size for this plotter type. Subclasses can override.
    """

    # Class-level registry name (override in subclasses)
    name: str = "base"

    # Default figure size - subclasses can override for optimal sizing
    default_figsize: tuple[float, float] = (8, 5)

    def __init__(self, config: PlotConfig | None = None) -> None:
        self.config = config or PlotConfig()
        # Apply plotter-specific default figsize if not explicitly set
        if self.config.figure.figsize == (8, 5):  # Original default
            self.config.figure.figsize = self.default_figsize

    @abstractmethod
    def plot(
        self,
        paired_data: xr.Dataset,
        obs_var: str,
        model_var: str,
        ax: matplotlib.axes.Axes | None = None,
        **kwargs: Any,
    ) -> matplotlib.figure.Figure:
        """Generate the plot.

        Parameters
        ----------
        paired_data
            Paired dataset with model and observation variables.
        obs_var
            Name of observation variable.
        model_var
            Name of model variable.
        ax
            Optional axes to plot on. If None, creates new figure.
        **kwargs
            Additional plot-specific options.

        Returns
        -------
        matplotlib.figure.Figure
            The generated figure.
        """
        ...

    def render(
        self,
        series: list[PlotSeries],
        ax: matplotlib.axes.Axes | None = None,
        **kwargs: Any,
    ) -> matplotlib.figure.Figure:
        """Render a list of source series (unified renderer contract).

        ``len(series) == 1`` → single line; ``== 2`` → reference-vs-comparand
        (obs gray / model blue); ``>= 2`` → multi-source overlay. This default
        handles only the 2-series case by delegating to the legacy
        ``plot(paired_data, obs_var, model_var)`` hook, so unmigrated paired
        renderers keep working unchanged. Renderers that support 1 or N series
        override this method (unification P3).
        """
        if len(series) == 2:
            ref = next((s for s in series if s.pair_role == "reference"), series[0])
            comp = next((s for s in series if s.pair_role == "comparand"), series[1])
            return self.plot(ref.dataset, ref.var_name, comp.var_name, ax=ax, **kwargs)
        raise NotImplementedError(
            f"{type(self).__name__}.render does not support {len(series)} series; "
            "override render() for single-/N-source support (unification P3)."
        )

    def create_figure(
        self,
        config: FigureConfig | None = None,
        **kwargs: Any,
    ) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
        """Create a new figure and axes.

        Parameters
        ----------
        config
            Figure configuration. If None, uses self.config.figure.
        **kwargs
            Additional arguments passed to plt.subplots.

        Returns
        -------
        tuple[Figure, Axes]
            The created figure and axes.
        """
        cfg = config or self.config.figure
        fig_kwargs = {
            "figsize": cfg.figsize,
            "dpi": cfg.dpi,
            "facecolor": cfg.facecolor,
            "constrained_layout": cfg.constrained_layout,
            **kwargs,
        }
        fig, ax = plt.subplots(**fig_kwargs)
        return fig, ax

    def apply_text_style(
        self,
        ax: matplotlib.axes.Axes,
        config: TextConfig | None = None,
    ) -> None:
        """Apply text styling to axes.

        Parameters
        ----------
        ax
            Axes to style.
        config
            Text configuration. If None, uses self.config.text.
        """
        cfg = config or self.config.text
        ax.tick_params(axis="both", labelsize=cfg.tick_fontsize)

    def set_labels(
        self,
        ax: matplotlib.axes.Axes,
        xlabel: str | None = None,
        ylabel: str | None = None,
        title: str | None = None,
    ) -> None:
        """Set axis labels and title.

        Parameters
        ----------
        ax
            Axes to label.
        xlabel
            X-axis label. Uses config if None.
        ylabel
            Y-axis label. Uses config if None.
        title
            Plot title. Uses config if None.
        """
        cfg = self.config.text

        if xlabel or self.config.xlabel:
            ax.set_xlabel(
                xlabel or self.config.xlabel,  # type: ignore[arg-type]
                fontsize=cfg.fontsize,
                fontweight=cfg.fontweight,
            )

        if ylabel or self.config.ylabel:
            ax.set_ylabel(
                ylabel or self.config.ylabel,  # type: ignore[arg-type]
                fontsize=cfg.fontsize,
                fontweight=cfg.fontweight,
            )

        if title or self.config.title:
            formatted_title = format_plot_title(title or self.config.title)  # type: ignore[arg-type]
            ax.set_title(
                formatted_title,
                fontsize=cfg.title_fontsize,
                fontweight=cfg.fontweight,
            )

    def set_limits(
        self,
        ax: matplotlib.axes.Axes,
        vmin: float | None = None,
        vmax: float | None = None,
        axis: Literal["x", "y", "both"] = "y",
    ) -> None:
        """Set axis limits.

        Parameters
        ----------
        ax
            Axes to configure.
        vmin
            Minimum value. Uses config if None.
        vmax
            Maximum value. Uses config if None.
        axis
            Which axis to set ('x', 'y', or 'both').
        """
        vmin = vmin if vmin is not None else self.config.vmin
        vmax = vmax if vmax is not None else self.config.vmax

        if vmin is not None or vmax is not None:
            if axis in ("y", "both"):
                ax.set_ylim(vmin, vmax)
            if axis in ("x", "both"):
                ax.set_xlim(vmin, vmax)

    def add_legend(
        self,
        ax: matplotlib.axes.Axes,
        loc: str = "best",
        **kwargs: Any,
    ) -> None:
        """Add legend to axes.

        Parameters
        ----------
        ax
            Axes to add legend to.
        loc
            Legend location.
        **kwargs
            Additional legend arguments.
        """
        legend_kwargs = {
            "loc": loc,
            "fontsize": self.config.text.legend,
            "framealpha": 0.9,
            **kwargs,
        }
        ax.legend(**legend_kwargs)

    def save(
        self,
        fig: matplotlib.figure.Figure,
        output_path: str | Path,
        dpi: int | None = None,
        bbox_inches: str = "tight",
        **kwargs: Any,
    ) -> Path:
        """Save figure to file.

        Parameters
        ----------
        fig
            Figure to save.
        output_path
            Output file path.
        dpi
            Resolution. Uses config if None.
        bbox_inches
            Bounding box setting.
        **kwargs
            Additional savefig arguments.

        Returns
        -------
        Path
            Path to saved file.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        save_kwargs = {
            "dpi": dpi or self.config.figure.dpi,
            "bbox_inches": bbox_inches,
            "facecolor": fig.get_facecolor(),
            **kwargs,
        }
        fig.savefig(output_path, **save_kwargs)
        return output_path

    def close(self, fig: matplotlib.figure.Figure) -> None:
        """Close a figure to free memory.

        Parameters
        ----------
        fig
            Figure to close.
        """
        plt.close(fig)


# =============================================================================
# Utility Functions
# =============================================================================


def merge_config_dicts(
    defaults: dict[str, Any],
    overrides: dict[str, Any] | None,
) -> dict[str, Any]:
    """Merge configuration dictionaries with defaults.

    Parameters
    ----------
    defaults
        Default configuration values.
    overrides
        Override values (can be None).

    Returns
    -------
    dict[str, Any]
        Merged configuration.
    """
    if overrides is None:
        return defaults.copy()
    return {**defaults, **overrides}


# Lookup table for common atmospheric variable display names
# Maps lowercase variable names (or patterns) to display names with proper formatting
VARIABLE_DISPLAY_NAMES: dict[str, str] = {
    # Surface pollutants (using LaTeX math mode for subscripts)
    "pm25": r"PM$_{2.5}$",
    "pm2.5": r"PM$_{2.5}$",
    "pm10": r"PM$_{10}$",
    "o3": r"O$_3$",
    "ozone": r"O$_3$",
    "no2": r"NO$_2$",
    "no": "NO",
    "nox": r"NO$_x$",
    "co": "CO",
    "co2": r"CO$_2$",
    "so2": r"SO$_2$",
    "hcho": "HCHO",
    "ch2o": "CH$_2$O",
    "nh3": r"NH$_3$",
    "hno3": r"HNO$_3$",
    "n2o": r"N$_2$O",
    "n2o5": r"N$_2$O$_5$",
    "ch4": r"CH$_4$",
    # AOD variables
    "aod": "AOD",
    "aod_500nm": "AOD (500 nm)",
    "aod_550nm": "AOD (550 nm)",
    "aod_440nm": "AOD (440 nm)",
    "aodvisdn": "AOD",
    # Column variables
    "no2_trop_column": r"Tropospheric NO$_2$ Column",
    "no2_column": r"NO$_2$ Column",
    "o3_column": r"O$_3$ Column",
    "trop_no2": r"Tropospheric NO$_2$",
    # Model variables (uppercase)
    "PM25": r"PM$_{2.5}$",
    "O3": r"O$_3$",
    "NO2": r"NO$_2$",
    "CO": "CO",
    "SO2": r"SO$_2$",
    "AODVISdn": "AOD",
    "NO2_column": r"NO$_2$ Column",
    # ASIA-AQ DC-8 aircraft variables (ICARTT naming convention)
    "O3_ROZE_STCLAIR": r"O$_3$",
    "NO2_CANOE_STCLAIR": r"NO$_2$",
    "CO_DACOM_DISKIN": "CO",
}

# Patterns for title formatting (case-insensitive replacements)
# Order matters - longer patterns first to avoid partial matches
# Uses LaTeX math mode subscripts for matplotlib rendering
TITLE_FORMULA_REPLACEMENTS: list[tuple[str, str]] = [
    # Longer patterns first
    ("PM2.5", r"PM$_{2.5}$"),
    ("PM25", r"PM$_{2.5}$"),
    ("PM10", r"PM$_{10}$"),
    ("N2O5", r"N$_2$O$_5$"),
    ("HNO3", r"HNO$_3$"),
    ("N2O", r"N$_2$O"),
    ("NO2", r"NO$_2$"),
    ("SO2", r"SO$_2$"),
    ("CO2", r"CO$_2$"),
    ("NH3", r"NH$_3$"),
    ("CH4", r"CH$_4$"),
    ("CH2O", r"CH$_2$O"),
    ("NOx", r"NO$_x$"),
    ("NOX", r"NO$_x$"),
    ("O3", r"O$_3$"),
]


# Unit string replacements for plot labels. Longer/more-specific patterns
# first so e.g. "ug/m3" doesn't get partially rewritten by a bare "m3" rule.
UNIT_REPLACEMENTS: list[tuple[str, str]] = [
    ("ug/m3", r"$\mu$g/m$^3$"),
    ("ug m-3", r"$\mu$g m$^{-3}$"),
    ("ug m^-3", r"$\mu$g m$^{-3}$"),
    ("mg/m3", r"mg/m$^3$"),
    ("mg m-3", r"mg m$^{-3}$"),
    ("kg/m3", r"kg/m$^3$"),
    ("kg m-3", r"kg m$^{-3}$"),
    ("g/m3", r"g/m$^3$"),
    ("m/s2", r"m/s$^2$"),
    ("m s-2", r"m s$^{-2}$"),
    ("m/s", "m/s"),
    ("W/m2", r"W/m$^2$"),
    ("W m-2", r"W m$^{-2}$"),
]


def format_plot_title(title: str) -> str:
    """Format a plot title with proper chemical formula subscripts.

    Replaces common chemical formulas (NO2, O3, PM2.5, etc.) with
    LaTeX subscript versions for matplotlib rendering.

    Parameters
    ----------
    title
        Raw title string.

    Returns
    -------
    str
        Title with chemical formulas properly formatted with LaTeX.

    Examples
    --------
    >>> format_plot_title("PM2.5 Model vs Observations")
    'PM$_{2.5}$ Model vs Observations'
    >>> format_plot_title("NO2 Time Series")
    'NO$_2$ Time Series'
    """
    import re

    result = title
    for pattern, replacement in TITLE_FORMULA_REPLACEMENTS:
        # Case-insensitive replacement while preserving surrounding text
        result = re.sub(re.escape(pattern), replacement, result, flags=re.IGNORECASE)
    return result


def format_variable_display_name(var_name: str, include_prefix: bool = True) -> str:
    """Format a variable name for display.

    Uses lookup table for known variables, otherwise applies
    basic formatting (replace underscores, title case).

    Parameters
    ----------
    var_name
        Raw variable name.
    include_prefix
        If True, include "Observed"/"Modeled" prefix for obs_/model_ variables.
        Set to False for shared axes (e.g., time series y-axis).

    Returns
    -------
    str
        Formatted display name.
    """
    # Strip obs_/model_ prefixes for lookup
    base_name = var_name
    prefix = ""
    if var_name.startswith("obs_"):
        base_name = var_name[4:]
        if include_prefix:
            prefix = "Observed "
    elif var_name.startswith("model_"):
        base_name = var_name[6:]
        if include_prefix:
            prefix = "Modeled "

    # Check lookup table (try exact match first, then lowercase)
    if base_name in VARIABLE_DISPLAY_NAMES:
        return prefix + VARIABLE_DISPLAY_NAMES[base_name]
    if base_name.lower() in VARIABLE_DISPLAY_NAMES:
        return prefix + VARIABLE_DISPLAY_NAMES[base_name.lower()]

    # Basic formatting: replace underscores, apply title case
    formatted = base_name.replace("_", " ")
    if formatted.islower() or formatted.isupper():
        formatted = formatted.title()

    return prefix + formatted


def get_variable_label(
    dataset: xr.Dataset,
    var_name: str,
    custom_label: str | None = None,
    include_prefix: bool = True,
) -> str:
    """Get a display label for a variable.

    Uses custom label if provided, then checks dataset attributes
    (display_name, long_name, standard_name), then falls back to
    automatic formatting via lookup table.

    Parameters
    ----------
    dataset
        Dataset containing the variable.
    var_name
        Variable name.
    custom_label
        Custom label to use (overrides all other sources).
    include_prefix
        If True, include "Observed"/"Modeled" prefix for obs_/model_ variables.
        Set to False for shared axes (e.g., time series y-axis).

    Returns
    -------
    str
        Display label for the variable.
    """
    if custom_label:
        return custom_label

    if var_name in dataset:
        attrs = dataset[var_name].attrs
        # Check for display_name first (our custom attribute)
        if attrs.get("display_name"):
            return str(attrs["display_name"])
        if attrs.get("long_name"):
            return str(attrs["long_name"])
        if attrs.get("standard_name"):
            return str(attrs["standard_name"])
        # Source-label-named paired vars (e.g. ``cam_o3``) carry role/source_label
        # attrs; translate to the legacy ``obs_``/``model_`` form so the prefix-
        # based Observed/Modeled formatting + canonical lookup is preserved (R-2).
        role = attrs.get("role")
        if role in ("obs", "model"):
            canonical = canonical_variable_name(dataset, var_name)
            var_name = f"obs_{canonical}" if role == "obs" else f"model_{canonical}"

    # Fall back to automatic formatting
    return format_variable_display_name(var_name, include_prefix=include_prefix)


def canonical_variable_name(dataset: xr.Dataset, var_name: str) -> str:
    """Strip a paired variable's prefix to its canonical (unprefixed) name.

    Handles both the source-label naming (``<source_label>_<canonical>``, e.g.
    ``cam_o3`` -> ``o3``, derived from the variable's ``source_label`` attr) and
    the legacy ``obs_``/``model_`` prefixes (``obs_o3`` -> ``o3``). Names with no
    recognised prefix are returned unchanged.
    """
    if var_name in dataset:
        source_label = dataset[var_name].attrs.get("source_label")
        if source_label and var_name.startswith(f"{source_label}_"):
            return var_name[len(source_label) + 1 :]
    for prefix in ("obs_", "model_"):
        if var_name.startswith(prefix):
            return var_name[len(prefix) :]
    return var_name


def build_series(dataset: xr.Dataset, *var_args: Any) -> list[PlotSeries]:
    """Resolve facade var-args into an ordered list of :class:`PlotSeries`.

    Accepts the three call shapes the unified facade supports:

    - ``build_series(ds, obs_var, model_var)`` → 2 series
    - ``build_series(ds, variable)`` → 1 series
    - ``build_series(ds, [v1, ..., vN])`` → N series

    A trailing positional ``matplotlib`` Axes (legacy ``plot(ds, var, ax)``) is
    ignored for series building. ``role``/``pair_role``/``source_label``/
    ``canonical`` are read from the dataset's attrs, with the legacy
    ``obs_``/``model_`` prefix fallback.
    """
    import matplotlib.axes

    args = list(var_args)
    if args and isinstance(args[-1], matplotlib.axes.Axes):
        args = args[:-1]
    if len(args) == 1 and isinstance(args[0], (list, tuple)):
        names = [str(n) for n in args[0]]
    else:
        names = [a for a in args if isinstance(a, str)]

    series: list[PlotSeries] = []
    for i, name in enumerate(names):
        # Prefer the per-variable source_label (paired/tagged data); fall back to
        # the dataset-level label that single-source obs datasets carry.
        source_label = (
            dataset[name].attrs.get("source_label") if name in dataset.data_vars else None
        ) or dataset.attrs.get("source_label")
        series.append(
            PlotSeries(
                dataset=dataset,
                var_name=name,
                canonical=canonical_variable_name(dataset, name),
                role=paired_variable_role(dataset, name),
                pair_role=paired_variable_pair_role(dataset, name),
                source_label=str(source_label) if source_label else None,
                index=i,
            )
        )
    return series


def series_colors(
    series: list[PlotSeries],
    *,
    obs_color: str | None = None,
    model_color: str | None = None,
) -> list[str]:
    """Per-series colors under the unified, count-aware rule.

    - **1 series** → ``NCAR_PRIMARY`` (the single-source brand blue), or ``MODEL_COLOR``
      when the lone source is ``role == "model"``. This is what keeps a single
      source blue rather than the paired-reference gray that ``get_color_for_role``
      would assign.
    - **2 series** → reference in ``obs_color`` (gray) and comparand in
      ``model_color`` (blue), preserving today's comparison contrast.
    - **N > 2 series** → distinct ``NCAR_PALETTE`` colors cycled by ``index``.

    ``obs_color``/``model_color`` let a caller pass the active ``StyleConfig``
    colors; they default to the module ``OBS_COLOR``/``MODEL_COLOR``.
    """
    n = len(series)
    if n == 1:
        return [MODEL_COLOR if series[0].role == "model" else NCAR_PRIMARY]
    if n == 2:
        out: list[str] = []
        for s in series:
            is_model = s.role == "model" or s.pair_role == "comparand"
            out.append((model_color or MODEL_COLOR) if is_model else (obs_color or OBS_COLOR))
        return out
    return [NCAR_PALETTE[s.index % len(NCAR_PALETTE)] for s in series]


def get_role_color(
    dataset: xr.Dataset,
    var_name: str,
    index: int = 0,
    *,
    obs_color: str | None = None,
    model_color: str | None = None,
) -> str:
    """Plot color for a paired series, by its source role (renderer rewire R-3).

    Reads the variable's ``role`` attr (set by ``tag_paired_roles``): ``obs``
    renders in the neutral reference gray, ``model`` in NCAR blue (preserving
    the legacy model-vs-obs convention), and same-role / role-less series cycle the
    NCAR palette by ``index`` (their order in the plot).

    ``obs_color``/``model_color`` let a caller supply the active ``StyleConfig``
    colors so a customised style is honoured for the obs/model roles; when
    omitted the module's :func:`get_color_for_role` defaults are used.
    """
    from davinci_monet.plots.style import get_color_for_role

    role = dataset[var_name].attrs.get("role") if var_name in dataset else None
    # Fall back to the legacy prefix when no role attr is present, so renderers
    # called directly with model_/obs_ names (tests, examples, user scripts)
    # still get the obs gray / model blue convention rather than palette colors.
    if role is None:
        lname = str(var_name).lower()
        if lname.startswith("obs_"):
            role = "obs"
        elif lname.startswith("model_"):
            role = "model"
    if role == "obs" and obs_color is not None:
        return obs_color
    if role == "model" and model_color is not None:
        return model_color
    return get_color_for_role(role, index)


def dataset_source_label(dataset: xr.Dataset, default: str | None = None) -> str | None:
    """Source label for a single-source dataset.

    Single-source datasets carry their source label in the dataset-level ``attrs``
    (set by the loading stage), not per-variable. Returns it so a source plot
    can self-identify its source, or ``default`` when absent.
    """
    label = dataset.attrs.get("source_label")
    return str(label) if label else default


def get_series_label(
    dataset: xr.Dataset,
    var_name: str,
    custom_label: str | None = None,
) -> str:
    """Legend label for a paired series (renderer rewire R-3).

    Prefers an explicit ``custom_label``, then the variable's ``source_label``
    attr (the source's identity in a unified pair, e.g. ``airnow`` / ``cam``),
    and finally falls back to the standard variable label (the role-aware
    Observed/Modeled formatting). Use this for the *series* legend; axis labels
    that name the variable should keep using :func:`get_variable_label`.
    """
    if custom_label:
        return custom_label
    if var_name in dataset:
        source_label = dataset[var_name].attrs.get("source_label")
        if source_label:
            return str(source_label)
    return get_variable_label(dataset, var_name)


def resolve_source_variable(
    dataset: xr.Dataset,
    canonical_var: str,
    source_label: str,
) -> str | None:
    """Resolve a variable name by source label (Phase 5, additive).

    Supports the unified source-label naming (``<source_label>_<canonical>``,
    e.g. ``cam_o3``) while falling back to the bare canonical name. Returns the
    matching variable name present in the dataset, or ``None`` if neither is
    found. Does not alter the existing ``model_``/``obs_`` prefix handling.

    Parameters
    ----------
    dataset
        Dataset to search.
    canonical_var
        Canonical (unprefixed) variable name, e.g. ``"o3"``.
    source_label
        Source label used as a prefix, e.g. ``"cam"`` or ``"airnow"``.

    Returns
    -------
    str | None
        The resolved variable name, or ``None`` if absent.
    """
    for candidate in (f"{source_label}_{canonical_var}", canonical_var):
        if candidate in dataset.data_vars or candidate in dataset.coords:
            return candidate
    return None


def get_variable_units(
    dataset: xr.Dataset,
    var_name: str,
) -> str | None:
    """Get units for a variable.

    Parameters
    ----------
    dataset
        Dataset containing the variable.
    var_name
        Variable name.

    Returns
    -------
    str | None
        Units string, or None if not found.
    """
    if var_name in dataset:
        return dataset[var_name].attrs.get("units")
    return None


def format_units(units: str) -> str:
    """Rewrite raw unit strings to LaTeX-rendered form.

    Applies UNIT_REPLACEMENTS so e.g. ``"ug/m3"`` becomes the proper
    ``"$\\mu$g/m$^3$"`` with greek mu and superscripted exponent.
    """
    result = units
    for pattern, replacement in UNIT_REPLACEMENTS:
        if pattern in result:
            result = result.replace(pattern, replacement)
            break
    return result


def format_label_with_units(label: str, units: str | None) -> str:
    """Format a label with units.

    Parameters
    ----------
    label
        Base label.
    units
        Units string (can be None). Dimensionless units ("1") are omitted.

    Returns
    -------
    str
        Formatted label with units in parentheses if provided. The units
        string is passed through :func:`format_units` so common bare-ASCII
        forms (``ug/m3``, ``W m-2``, ...) render with proper LaTeX symbols.
    """
    if units and units != "1":
        return f"{label} ({format_units(units)})"
    return label


def calculate_symmetric_limits(
    data: np.ndarray,
    percentile: float = 98,
) -> tuple[float, float]:
    """Calculate symmetric limits around zero for bias plots.

    Parameters
    ----------
    data
        Data array.
    percentile
        Percentile to use for limit calculation.

    Returns
    -------
    tuple[float, float]
        Symmetric (vmin, vmax) limits.
    """
    data = np.asarray(data).flatten()
    data = data[np.isfinite(data)]
    if len(data) == 0:
        return (-1.0, 1.0)

    abs_max = np.percentile(np.abs(data), percentile)
    return (-abs_max, abs_max)


def calculate_data_limits(
    data: np.ndarray,
    percentile: float = 98,
    symmetric: bool = False,
) -> tuple[float, float]:
    """Calculate data limits for colorbar/axis scaling.

    Parameters
    ----------
    data
        Data array.
    percentile
        Percentile to use for limit calculation.
    symmetric
        If True, make limits symmetric around zero.

    Returns
    -------
    tuple[float, float]
        (vmin, vmax) limits.
    """
    if symmetric:
        return calculate_symmetric_limits(data, percentile)

    data = np.asarray(data).flatten()
    data = data[np.isfinite(data)]
    if len(data) == 0:
        return (0.0, 1.0)

    vmin = np.percentile(data, 100 - percentile)
    vmax = np.percentile(data, percentile)
    return (vmin, vmax)
