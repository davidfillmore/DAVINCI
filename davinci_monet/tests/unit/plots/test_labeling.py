"""Tests for davinci_monet.plots.labeling — TDD, task by task."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from davinci_monet.plots import labeling as L

# ---------------------------------------------------------------------------
# Task 1: format_units
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("mol/m2", "mol m$^{-2}$"),
        ("W m-2", "W m$^{-2}$"),
        ("mol/mol", "mol mol$^{-1}$"),
        ("kg/kg", "kg kg$^{-1}$"),
        ("m/s", "m s$^{-1}$"),
        ("ppb", "ppb"),
        ("K", "K"),
        ("1", ""),
        ("none", ""),
        ("", ""),
        (None, ""),
    ],
)
def test_format_units(raw, expected):
    assert L.format_units(raw) == expected


def test_format_units_micrograms():
    assert L.format_units("ug/m3") == r"$\mu$g m$^{-3}$"


# ---------------------------------------------------------------------------
# Task 2: source_display_name
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "key,expected",
    [
        ("cesm_no2_column", r"CESM NO$_2$ Column"),
        ("airnow", "AirNow"),
        ("pandora", "Pandora"),
        ("merra2", "MERRA-2"),
        ("ceres", "CERES"),
        ("cam", "CAM"),
        ("", ""),
        (None, ""),
    ],
)
def test_source_display_name(key, expected):
    assert L.source_display_name(key) == expected


# ---------------------------------------------------------------------------
# Task 3: quantity_label
# ---------------------------------------------------------------------------


def _ds(var: str, **attrs: object) -> xr.Dataset:
    d = xr.Dataset({var: ("t", np.arange(3.0))})
    d[var].attrs.update(attrs)
    return d


def test_quantity_label_from_lookup():
    ds = _ds("no2_column")
    assert L.quantity_label(ds, "no2_column") == r"NO$_2$ Column"


def test_quantity_label_prefers_long_name():
    ds = _ds("X", long_name="Tropospheric NO2 Column")
    assert "Tropospheric" in L.quantity_label(ds, "X")


# ---------------------------------------------------------------------------
# Task 4: axis_label
# ---------------------------------------------------------------------------


def test_axis_label_no_source():
    assert L.axis_label(r"NO$_2$ Column", "mol/m2") == r"NO$_2$ Column (mol m$^{-2}$)"


def test_axis_label_with_source():
    assert (
        L.axis_label(r"NO$_2$ Column", "mol/m2", source="pandora")
        == r"Pandora NO$_2$ Column (mol m$^{-2}$)"
    )


def test_axis_label_dedup():
    out = L.axis_label(r"NO$_2$ Column", "mol/m2", source="cesm_no2_column")
    assert out == r"CESM NO$_2$ Column (mol m$^{-2}$)"


def test_axis_label_no_units():
    assert L.axis_label("Altitude", "1") == "Altitude"


# ---------------------------------------------------------------------------
# Task 5: legend_label
# ---------------------------------------------------------------------------


def test_legend_label_plain():
    assert L.legend_label("cesm_no2_column") == r"CESM NO$_2$ Column"


def test_legend_label_uncertainty():
    assert L.legend_label("pandora", uncertainty="mean ± σ") == "Pandora (mean ± σ)"


# ---------------------------------------------------------------------------
# Task 6: bias_label
# ---------------------------------------------------------------------------


def test_bias_label_factors_shared_quantity():
    out = L.bias_label("merra2_olr", "ceres_olr", "W m-2")
    assert out == "Bias, MERRA-2 − CERES (W m$^{-2}$)"


def test_bias_label_no_shared_quantity():
    out = L.bias_label("cesm_no2_column", "pandora", "mol/m2")
    assert out == "Bias, CESM NO$_2$ Column − Pandora (mol m$^{-2}$)"


def test_bias_label_never_uses_xy():
    out = L.bias_label("cesm_no2_column", "pandora", None)
    assert " x" not in out.lower() and " y" not in out.lower()
