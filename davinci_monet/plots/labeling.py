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


_SOURCE_ACRONYMS = {
    "cesm": "CESM",
    "cam": "CAM",
    "wrf": "WRF",
    "wrfchem": "WRF-Chem",
    "merra2": "MERRA-2",
    "geoschem": "GEOS-Chem",
    "geos": "GEOS",
    "airnow": "AirNow",
    "aeronet": "AERONET",
    "pandora": "Pandora",
    "ceres": "CERES",
    "modis": "MODIS",
    "tropomi": "TROPOMI",
    "tempo": "TEMPO",
    "ufs": "UFS",
    "cmaq": "CMAQ",
    "ebaf": "EBAF",
    "ssf": "SSF",
}


def source_display_name(source_label: str | None) -> str:
    """Friendly source name from a raw config key (D5); never ALL-CAPS."""
    if not source_label:
        return ""
    out = []
    for tok in str(source_label).split("_"):
        low = tok.lower()
        if low in _SOURCE_ACRONYMS:
            out.append(_SOURCE_ACRONYMS[low])
        elif low in VARIABLE_DISPLAY_NAMES:
            out.append(VARIABLE_DISPLAY_NAMES[low])
        else:
            out.append(tok.capitalize())
    return " ".join(out)


def quantity_label(dataset: "xr.Dataset", var_name: str) -> str:
    """Quantity name only (no source, no units), chem-formatted."""
    return get_variable_label(dataset, var_name, include_prefix=False)


def axis_label(quantity: str, units: str | None, source: str | None = None) -> str:
    """Compose an axis label (D2 context-aware, with de-dup)."""
    label = quantity or ""
    if source:
        src = source_display_name(source)
        if quantity and quantity.lower() in src.lower():
            label = src
        else:
            label = f"{src} {quantity}".strip()
    u = format_units(units)
    return f"{label} ({u})" if u else label


def legend_label(source_label: str, uncertainty: str | None = None) -> str:
    """Compose a legend entry: friendly name, optionally with uncertainty note."""
    name = source_display_name(source_label)
    return f"{name} ({uncertainty})" if uncertainty else name


def bias_label(y_source: str, x_source: str, units: str | None) -> str:
    """'Bias, <Ysrc> − <Xsrc> (units)'; factor a shared trailing quantity."""
    yw = source_display_name(y_source).split()
    xw = source_display_name(x_source).split()
    while yw and xw and yw[-1] == xw[-1]:
        yw.pop()
        xw.pop()
    y = " ".join(yw) or source_display_name(y_source)
    x = " ".join(xw) or source_display_name(x_source)
    core = f"Bias, {y} − {x}"
    u = format_units(units)
    return f"{core} ({u})" if u else core
