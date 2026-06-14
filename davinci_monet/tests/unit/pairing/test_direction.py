"""Tests for x/y pairing direction and dispatch."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import xarray as xr

from davinci_monet.core.exceptions import PairingError
from davinci_monet.core.protocols import DataGeometry
from davinci_monet.core.types import TimeDelta
from davinci_monet.pairing.direction import PairDirectionWarning, resolve_pair_direction
from davinci_monet.pairing.engine import PairingEngine
from davinci_monet.pairing.strategies.base import BasePairingStrategy

G = DataGeometry


class TestResolvePairDirection:
    @pytest.mark.parametrize(
        "irregular",
        [G.POINT, G.TRACK, G.PROFILE, G.SWATH],
    )
    def test_irregular_outranks_grid_either_order(self, irregular: DataGeometry) -> None:
        # GRID is sampled onto the irregular geometry regardless of arg order.
        assert resolve_pair_direction(irregular, G.GRID) == (irregular, G.GRID)
        assert resolve_pair_direction(G.GRID, irregular) == (irregular, G.GRID)

    def test_same_geometry_warns_and_defaults_to_first(self) -> None:
        with pytest.warns(PairDirectionWarning):
            x_geometry, y_geometry = resolve_pair_direction(G.GRID, G.GRID)
        assert (x_geometry, y_geometry) == (G.GRID, G.GRID)

    def test_two_different_irregular_warns_and_defaults_to_first(self) -> None:
        with pytest.warns(PairDirectionWarning):
            x_geometry, y_geometry = resolve_pair_direction(G.POINT, G.TRACK)
        assert (x_geometry, y_geometry) == (G.POINT, G.TRACK)

    def test_explicit_geometry_overrides_precedence(self) -> None:
        # Force GRID as the x source even though POINT would outrank it.
        x_geometry, y_geometry = resolve_pair_direction(G.POINT, G.GRID, explicit_geometry="b")
        assert (x_geometry, y_geometry) == (G.GRID, G.POINT)

    def test_explicit_geometry_a(self) -> None:
        x_geometry, y_geometry = resolve_pair_direction(G.GRID, G.POINT, explicit_geometry="a")
        assert (x_geometry, y_geometry) == (G.GRID, G.POINT)


class TestEnginePairDispatch:
    def test_supported_pairing_combinations_are_explicit(self) -> None:
        engine = PairingEngine()
        supported = engine.supported_pairing_combinations()

        assert (G.POINT, G.GRID) in supported
        assert (G.TRACK, G.GRID) in supported
        assert (G.PROFILE, G.GRID) in supported
        assert (G.SWATH, G.GRID) in supported
        assert (G.GRID, G.GRID) in supported
        assert (G.POINT, G.TRACK) not in supported

    def test_get_strategy_for_irregular_grid_matches_geometry(self) -> None:
        engine = PairingEngine()
        for geom in (G.POINT, G.TRACK, G.PROFILE, G.SWATH):
            assert engine.get_strategy_for(geom, G.GRID) is engine.get_strategy(geom)

    def test_get_strategy_for_grid_grid(self) -> None:
        engine = PairingEngine()
        assert engine.get_strategy_for(G.GRID, G.GRID) is engine.get_strategy(G.GRID)

    def test_unsupported_combo_raises(self) -> None:
        engine = PairingEngine()
        # y source must be GRID in the seeded combinations.
        with pytest.raises(PairingError):
            engine.get_strategy_for(G.POINT, G.TRACK)

    def test_get_strategy_returns_registered_strategy(self) -> None:
        engine = PairingEngine()
        assert engine.get_strategy(G.POINT).geometry is G.POINT


class _SpyStrategy(BasePairingStrategy):
    """Minimal concrete strategy that records what pair_sources() received."""

    def __init__(self) -> None:
        self.captured: dict[str, Any] = {}

    @property
    def geometry(self) -> DataGeometry:
        return DataGeometry.POINT

    def pair_sources(
        self,
        x_data: xr.Dataset,
        y_data: xr.Dataset,
        radius_of_influence: float | None = None,
        time_tolerance: TimeDelta | None = None,
        vertical_method: str = "nearest",
        horizontal_method: str = "nearest",
        **kwargs: Any,
    ) -> xr.Dataset:
        kwargs.update(
            {
                "radius_of_influence": radius_of_influence,
                "time_tolerance": time_tolerance,
                "vertical_method": vertical_method,
                "horizontal_method": horizontal_method,
            }
        )
        self.captured = {"y": y_data, "x": x_data, "kwargs": kwargs}
        return xr.Dataset(attrs={"ok": True})


def _coords_ds(lat: float, lon: float) -> xr.Dataset:
    return xr.Dataset(
        {"v": ("site", np.array([1.0]))},
        coords={"latitude": ("site", np.array([lat])), "longitude": ("site", np.array([lon]))},
    )


class TestPairSourcesWrapper:
    def test_pair_sources_maps_x_to_x_and_y_to_y(self) -> None:
        strat = _SpyStrategy()
        x = _coords_ds(10.0, 20.0)
        y = _coords_ds(30.0, 40.0)
        out = strat.pair_sources(x_data=x, y_data=y, radius_of_influence=1.0)
        assert out.attrs["ok"] is True
        assert strat.captured["x"] is x
        assert strat.captured["y"] is y
        assert strat.captured["kwargs"]["radius_of_influence"] == 1.0

    def test_coord_helpers_return_x_and_y_coordinates(self) -> None:
        strat = _SpyStrategy()
        x = _coords_ds(1.0, 2.0)
        y = _coords_ds(3.0, 4.0)

        x_lat, x_lon = strat._get_x_coords(x)
        y_lat, y_lon = strat._get_y_coords(y)

        assert x_lat.values == pytest.approx(np.array([1.0]))
        assert x_lon.values == pytest.approx(np.array([2.0]))
        assert y_lat.values == pytest.approx(np.array([3.0]))
        assert y_lon.values == pytest.approx(np.array([4.0]))
