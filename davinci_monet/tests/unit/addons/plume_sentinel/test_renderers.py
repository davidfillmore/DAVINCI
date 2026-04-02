"""Tests for PlumeSentinel renderers and overlays."""
from __future__ import annotations

from davinci_monet.addons.plume_sentinel.overlays import SMOKE_DENSITY_STYLES


class TestSmokeDensityStyles:
    def test_all_densities_defined(self):
        assert "Light" in SMOKE_DENSITY_STYLES
        assert "Medium" in SMOKE_DENSITY_STYLES
        assert "Heavy" in SMOKE_DENSITY_STYLES

    def test_light_color(self):
        assert SMOKE_DENSITY_STYLES["Light"]["color"] == "#FFDD31"

    def test_medium_color(self):
        assert SMOKE_DENSITY_STYLES["Medium"]["color"] == "#FF8C00"

    def test_heavy_color(self):
        assert SMOKE_DENSITY_STYLES["Heavy"]["color"] == "#D62839"

    def test_linewidths_increase(self):
        lw_light = SMOKE_DENSITY_STYLES["Light"]["linewidth"]
        lw_medium = SMOKE_DENSITY_STYLES["Medium"]["linewidth"]
        lw_heavy = SMOKE_DENSITY_STYLES["Heavy"]["linewidth"]
        assert lw_light < lw_medium < lw_heavy
