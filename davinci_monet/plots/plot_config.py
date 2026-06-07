"""Plot configuration dataclasses for DAVINCI.

Provides typed configuration containers for figure layout, text styling,
plot styling, spatial domain, and the composite PlotConfig that combines them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from davinci_monet.plots.style import MODEL_COLOR, OBS_COLOR

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


__all__ = [
    "FigureConfig",
    "TextConfig",
    "StyleConfig",
    "DomainConfig",
    "PlotConfig",
]
