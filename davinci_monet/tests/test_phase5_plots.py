"""Phase 5 tests: role-based color resolver + source-label variable resolver.

Timeseries site-aggregation (formerly the obs_timeseries opt-in) is now the
unified renderer's default — covered by
tests/unit/plots/test_unification_p3_timeseries.py.
"""

from __future__ import annotations

import xarray as xr

from davinci_monet.plots.base import resolve_source_variable
from davinci_monet.plots.style import MODEL_COLOR, NCAR_PALETTE, OBS_COLOR, get_color_for_role


class TestGetColorForRole:
    def test_obs_role_is_obs_color(self) -> None:
        assert get_color_for_role("obs") == OBS_COLOR

    def test_model_role_is_model_color(self) -> None:
        assert get_color_for_role("model") == MODEL_COLOR

    def test_roleless_cycles_palette_by_index(self) -> None:
        assert get_color_for_role(None, index=0) == NCAR_PALETTE[0]
        assert get_color_for_role(None, index=1) == NCAR_PALETTE[1]

    def test_index_wraps_around_palette(self) -> None:
        assert get_color_for_role("", index=len(NCAR_PALETTE)) == NCAR_PALETTE[0]


class TestResolveSourceVariable:
    def test_prefers_source_label_prefixed_name(self) -> None:
        ds = xr.Dataset({"cam_o3": ("x", [1.0]), "o3": ("x", [2.0])})
        assert resolve_source_variable(ds, "o3", "cam") == "cam_o3"

    def test_falls_back_to_canonical_name(self) -> None:
        ds = xr.Dataset({"o3": ("x", [2.0])})
        assert resolve_source_variable(ds, "o3", "airnow") == "o3"

    def test_returns_none_when_absent(self) -> None:
        ds = xr.Dataset({"pm25": ("x", [2.0])})
        assert resolve_source_variable(ds, "o3", "airnow") is None
