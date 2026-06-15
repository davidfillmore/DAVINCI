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
