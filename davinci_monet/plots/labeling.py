"""Central, publication-quality text composition for all DAVINCI plots.

Pure functions only — no matplotlib, no I/O. Renderers call these instead of
building label/title strings themselves.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from davinci_monet.plots.labels import (
    VARIABLE_DISPLAY_NAMES,
    format_plot_title,
    get_variable_label,
    get_variable_units,
)

if TYPE_CHECKING:
    import xarray as xr

_DIMENSIONLESS = {"", "1", "none", "dimensionless", "unitless", "fraction"}


def format_units(units: str | None) -> str:
    """Rewrite a raw unit string to negative-exponent SI LaTeX (D3)."""
    if units is None:
        return ""
    s = str(units).strip()
    if s.lower() in _DIMENSIONLESS:
        return ""
    s = s.replace("ug", r"$\mu$g")

    # division: "/unitN" -> " unit^{-N}" (N defaults to 1)
    def _div(m: re.Match[str]) -> str:
        unit, power = m.group(1), m.group(2) or "1"
        return f" {unit}$^{{-{power}}}$"

    s = re.sub(r"/([A-Za-z]+)(\d+)?", _div, s)
    # space/dash negative exponents: "unit-N" -> "unit^{-N}"
    s = re.sub(r"([A-Za-z])-(\d+)", lambda m: f"{m.group(1)}$^{{-{m.group(2)}}}$", s)
    # remaining positive exponents: "unitN" -> "unit^{N}"
    s = re.sub(r"([A-Za-z])(\d+)", lambda m: f"{m.group(1)}$^{{{m.group(2)}}}$", s)
    return s.strip()
