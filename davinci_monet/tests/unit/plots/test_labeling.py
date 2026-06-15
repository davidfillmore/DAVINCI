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


def test_quantity_label_subscripts_formula_from_long_name():
    # A chemical formula coming from a raw long_name must be subscripted so the
    # colorbar/axis matches the title (NO2 -> NO$_2$), not left bare.
    ds = _ds("X", long_name="NO2 total column")
    assert L.quantity_label(ds, "X") == r"NO$_2$ total column"


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


def test_axis_label_dedup_partial_overlap():
    # source key embeds "NO2 Column" but the quantity is the longer
    # "Tropospheric NO2 Column" -> keep only the distinctive source token (CESM),
    # do NOT repeat "NO2 Column".
    out = L.axis_label(r"Tropospheric NO$_2$ Column", "mol/m2", source="cesm_no2_column")
    assert out == r"CESM Tropospheric NO$_2$ Column (mol m$^{-2}$)"


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


def test_bias_label_strips_quantity_when_given():
    # quantity is already in the title -> strip it from the source names so the
    # colorbar stays terse: "Bias, CESM − Pandora".
    out = L.bias_label("cesm_no2_column", "pandora", "mol/m2", quantity=r"NO$_2$ Column")
    assert out == "Bias, CESM − Pandora (mol m$^{-2}$)"


# ---------------------------------------------------------------------------
# Task 7: title_text + subtitle_text
# ---------------------------------------------------------------------------


def test_title_text_terse():
    assert L.title_text("NO2 Tropospheric Column") == r"NO$_2$ Tropospheric Column"


def test_title_text_operation():
    assert L.title_text("OLR", operation="Bias") == "OLR Bias"


def test_subtitle_range():
    assert L.subtitle_text("2024-02-01", "2024-02-29") == "2024-02-01 – 2024-02-29"


def test_subtitle_single():
    assert L.subtitle_text("2024-02-01", "2024-02-01") == "2024-02-01"


def test_subtitle_empty():
    assert L.subtitle_text(None, None) == ""


# ---------------------------------------------------------------------------
# Task 8: labels.format_units delegates to SI labeling
# ---------------------------------------------------------------------------


def test_labels_format_units_delegates_to_si():
    from davinci_monet.plots import labels

    assert labels.format_units("mol/m2") == "mol m$^{-2}$"


# ---------------------------------------------------------------------------
# Fix #5: source_display_name — skip empty tokens (double-underscore safety)
# ---------------------------------------------------------------------------


def test_source_display_name_double_underscore():
    """Double underscores (empty tokens) must not produce double spaces."""
    result = L.source_display_name("cesm__no2")
    assert "  " not in result, f"Double space found in: {result!r}"


def test_source_display_name_leading_underscore():
    """Leading underscore must not produce a leading space."""
    result = L.source_display_name("_cesm")
    assert not result.startswith(" "), f"Leading space found in: {result!r}"


def test_source_display_name_trailing_underscore():
    """Trailing underscore must not produce a trailing space."""
    result = L.source_display_name("cesm_")
    assert not result.endswith(" "), f"Trailing space found in: {result!r}"


# ---------------------------------------------------------------------------
# Fix #6: format_units ug replace — word-boundary safe
# ---------------------------------------------------------------------------


def test_format_units_ug_m3():
    """'ug/m3' must render as µg m^{-3}."""
    assert L.format_units("ug/m3") == r"$\mu$g m$^{-3}$"


def test_format_units_ug_space_m3():
    """'ug m-3' (space-separated) must render correctly."""
    assert L.format_units("ug m-3") == r"$\mu$g m$^{-3}$"


def test_format_units_bug_not_replaced():
    """'bug' must NOT have its 'ug' substring replaced."""
    result = L.format_units("bug")
    assert r"$\mu$g" not in result, f"False 'ug' substitution in 'bug': {result!r}"


def test_format_units_slug_not_replaced():
    """'slug' must NOT have its 'ug' substring replaced."""
    result = L.format_units("slug")
    assert r"$\mu$g" not in result, f"False 'ug' substitution in 'slug': {result!r}"
