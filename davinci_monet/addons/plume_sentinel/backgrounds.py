"""True-color background renderers for PlumeSentinel add-on."""

from __future__ import annotations

from typing import Any

import matplotlib.axes

GIBS_URL = "https://gibs.earthdata.nasa.gov/wmts/epsg4326/best/wmts.cgi"


def render_background(
    ax: matplotlib.axes.Axes,
    background_spec: str | dict[str, Any],
    prepared_inputs: dict[str, Any],
) -> None:
    """Render a true-color background on the given axes.

    background_spec is either:
    - A string naming a prepared input (GOES GoesRgbResult)
    - A dict with type: gibs_wmts for inline GIBS backgrounds
    - A GibsBackgroundConfig pydantic model
    """
    if isinstance(background_spec, str):
        goes_result = prepared_inputs[background_spec]
        ax.imshow(
            goes_result.rgb,
            origin="upper",
            extent=goes_result.extent,
            transform=goes_result.cartopy_crs,
            interpolation="nearest",
            zorder=1,
        )
    elif isinstance(background_spec, dict):
        bg_type = background_spec.get("type", "")
        if bg_type == "gibs_wmts":
            _render_gibs(ax, background_spec)
        else:
            raise ValueError(f"Unknown background type: {bg_type!r}")
    else:
        # Pydantic GibsBackgroundConfig model
        _render_gibs(
            ax,
            {
                "type": "gibs_wmts",
                "layer": background_spec.layer,
                "date": background_spec.date,
            },
        )


def _render_gibs(ax: matplotlib.axes.Axes, spec: dict[str, Any]) -> None:
    """Render a NASA GIBS WMTS tile layer on the axes."""
    layer = spec["layer"]
    date = spec.get("date")
    wmts_kwargs: dict[str, Any] = {}
    if date:
        wmts_kwargs["time"] = date
    ax.add_wmts(GIBS_URL, layer, wmts_kwargs=wmts_kwargs)
