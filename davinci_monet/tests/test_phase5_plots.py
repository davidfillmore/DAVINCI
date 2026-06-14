"""Phase 5 tests: variable color resolver + source-label variable resolver.

Timeseries site-aggregation (formerly the geometry_timeseries opt-in) is now the
unified renderer's default — covered by
tests/unit/plots/test_unification_p3_timeseries.py.
"""

from __future__ import annotations

import xarray as xr

from davinci_monet.plots.base import resolve_dataset_variable
from davinci_monet.plots.style import (
    DATASET_A_COLOR,
    DATASET_B_COLOR,
    NCAR_COLORS,
    get_color_for_variable,
)


class TestGetColorForVariable:
    def test_geometry_variable_is_geometry_color(self) -> None:
        assert get_color_for_variable("geometry_o3") == DATASET_A_COLOR

    def test_dataset_variable_is_dataset_color(self) -> None:
        assert get_color_for_variable("dataset_o3") == DATASET_B_COLOR

    def test_bias_variable_is_bias_color(self) -> None:
        assert get_color_for_variable("bias_o3") == NCAR_COLORS["red"]

    def test_unknown_variable_uses_primary_color(self) -> None:
        assert get_color_for_variable("o3") == NCAR_COLORS["ncar_blue"]


class TestResolveSourceVariable:
    def test_prefers_dataset_label_prefixed_name(self) -> None:
        ds = xr.Dataset({"cam_o3": ("x", [1.0]), "o3": ("x", [2.0])})
        assert resolve_dataset_variable(ds, "o3", "cam") == "cam_o3"

    def test_falls_back_to_canonical_name(self) -> None:
        ds = xr.Dataset({"o3": ("x", [2.0])})
        assert resolve_dataset_variable(ds, "o3", "airnow") == "o3"

    def test_returns_none_when_absent(self) -> None:
        ds = xr.Dataset({"pm25": ("x", [2.0])})
        assert resolve_dataset_variable(ds, "o3", "airnow") is None
