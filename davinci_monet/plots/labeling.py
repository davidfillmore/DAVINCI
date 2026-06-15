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
    """Rewrite a raw unit string to negative-exponent SI LaTeX (D3).

    Strings that already contain non-ASCII characters or LaTeX (``$``) are
    returned unchanged — they are assumed to be pre-formatted.
    """
    if units is None:
        return ""
    s = str(units).strip()
    if s.lower() in _DIMENSIONLESS:
        return ""
    # Pass through pre-formatted strings (Unicode or LaTeX already present).
    if "$" in s or not s.isascii():
        return s
    s = re.sub(r"(?<![A-Za-z])ug(?![A-Za-z])", r"$\\mu$g", s)

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
        if not tok:
            continue
        low = tok.lower()
        if low in _SOURCE_ACRONYMS:
            out.append(_SOURCE_ACRONYMS[low])
        elif low in VARIABLE_DISPLAY_NAMES:
            out.append(VARIABLE_DISPLAY_NAMES[low])
        else:
            out.append(tok.capitalize())
    return " ".join(out)


def quantity_label(dataset: "xr.Dataset", var_name: str) -> str:
    """Quantity name only (no source, no units), chem-formatted.

    Chemical formulas are subscripted (``NO2`` -> ``NO$_2$``) even when the name
    comes from a raw ``long_name``/``standard_name`` attr, so axes and colorbars
    match the title. ``format_plot_title`` is idempotent on already-formatted
    lookup-table names (no bare ``NO2`` left to match).
    """
    return format_plot_title(get_variable_label(dataset, var_name, include_prefix=False))


def _distinctive_source_tokens(source: str, quantity: str) -> list[str]:
    """Source-name tokens that are NOT already part of the quantity.

    Lets a source key that redundantly embeds the quantity (e.g.
    ``cesm_no2_column``) contribute only its distinctive identifier (``CESM``)
    when placed next to that quantity, avoiding repetition.
    """
    q_tokens = set((quantity or "").split())
    return [t for t in source_display_name(source).split() if t not in q_tokens]


def axis_label(quantity: str, units: str | None, source: str | None = None) -> str:
    """Compose an axis label (D2 context-aware, with de-dup).

    When the source name overlaps the quantity, only the source's distinctive
    tokens are kept, so e.g. ``cesm_no2_column`` + ``Tropospheric NO2 Column``
    renders ``CESM Tropospheric NO2 Column`` (not ``CESM NO2 Column
    Tropospheric NO2 Column``).
    """
    q = quantity or ""
    if source:
        distinctive = _distinctive_source_tokens(source, q)
        if distinctive and q:
            label = f"{' '.join(distinctive)} {q}"
        elif distinctive:
            label = " ".join(distinctive)
        else:
            label = q
    else:
        label = q
    u = format_units(units)
    return f"{label} ({u})" if u else label


def legend_label(source_label: str, uncertainty: str | None = None) -> str:
    """Compose a legend entry: friendly name, optionally with uncertainty note."""
    name = source_display_name(source_label)
    return f"{name} ({uncertainty})" if uncertainty else name


def bias_label(
    y_source: str,
    x_source: str,
    units: str | None,
    quantity: str | None = None,
) -> str:
    """'Bias, <Ysrc> − <Xsrc> (units)' — viewer-facing, never x/y.

    When ``quantity`` is given (it already appears in the plot title), tokens
    each source shares with it are stripped so the label stays terse, e.g.
    ``cesm_no2_column`` vs ``pandora`` with quantity ``NO2 Column`` →
    ``Bias, CESM − Pandora``. Any remaining shared trailing tokens are factored.
    """
    if quantity:
        yw = _distinctive_source_tokens(y_source, quantity)
        xw = _distinctive_source_tokens(x_source, quantity)
    else:
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


def title_text(quantity: str, operation: str | None = None) -> str:
    """Terse plot title: chem-format the quantity, optionally append operation."""
    q = format_plot_title(quantity or "")
    return f"{q} {operation}".strip() if operation else q


def subtitle_text(start: Any, end: Any) -> str:
    """Date-range subtitle: 'YYYY-MM-DD – YYYY-MM-DD' or single date or ''."""
    if not start:
        return ""
    s = str(start).split(" ")[0].split("T")[0]
    e = str(end).split(" ")[0].split("T")[0] if end else s
    return s if s == e else f"{s} – {e}"


__all__ = [
    "format_units",
    "source_display_name",
    "quantity_label",
    "axis_label",
    "legend_label",
    "bias_label",
    "title_text",
    "subtitle_text",
]
