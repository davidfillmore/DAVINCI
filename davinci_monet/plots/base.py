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
from davinci_monet.plots.style import MODEL_COLOR, OBS_COLOR

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
    ("NOx", r"NO$_x$"),
    ("NOX", r"NO$_x$"),
    ("O3", r"O$_3$"),
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

    # Fall back to automatic formatting
    return format_variable_display_name(var_name, include_prefix=include_prefix)


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
        Formatted label with units in parentheses if provided.
    """
    if units and units != "1":
        return f"{label} ({units})"
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
