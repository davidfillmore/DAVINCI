"""Tests for the role-neutral pairing scaffolding (Phase 4).

Phase 4 is additive: it introduces a precedence-based direction resolver, an
engine ``(reference_geometry, comparand_geometry)`` dispatch table, and a
role-neutral ``pair_sources`` entrypoint (plus reference/comparand coord-helper
aliases) — all alongside the existing ``pair(model, obs)`` API, which is left
untouched so the suite stays green.
"""

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
            ref, comp = resolve_pair_direction(G.GRID, G.GRID)
        assert (ref, comp) == (G.GRID, G.GRID)

    def test_two_different_irregular_warns_and_defaults_to_first(self) -> None:
        with pytest.warns(PairDirectionWarning):
            ref, comp = resolve_pair_direction(G.POINT, G.TRACK)
        assert (ref, comp) == (G.POINT, G.TRACK)

    def test_explicit_reference_overrides_precedence(self) -> None:
        # Force GRID as the reference even though POINT would outrank it.
        ref, comp = resolve_pair_direction(G.POINT, G.GRID, explicit_reference="b")
        assert (ref, comp) == (G.GRID, G.POINT)

    def test_explicit_reference_a(self) -> None:
        ref, comp = resolve_pair_direction(G.GRID, G.POINT, explicit_reference="a")
        assert (ref, comp) == (G.GRID, G.POINT)


class TestEngineRefCompDispatch:
    def test_supported_pairing_combinations_are_explicit(self) -> None:
        engine = PairingEngine()
        supported = engine.supported_pairing_combinations()

        assert (G.POINT, G.GRID) in supported
        assert (G.TRACK, G.GRID) in supported
        assert (G.PROFILE, G.GRID) in supported
        assert (G.SWATH, G.GRID) in supported
        assert (G.GRID, G.GRID) in supported
        assert (G.POINT, G.TRACK) not in supported

    def test_get_strategy_for_irregular_grid_matches_legacy(self) -> None:
        engine = PairingEngine()
        for geom in (G.POINT, G.TRACK, G.PROFILE, G.SWATH):
            assert engine.get_strategy_for(geom, G.GRID) is engine.get_strategy(geom)

    def test_get_strategy_for_grid_grid(self) -> None:
        engine = PairingEngine()
        assert engine.get_strategy_for(G.GRID, G.GRID) is engine.get_strategy(G.GRID)

    def test_unsupported_combo_raises(self) -> None:
        engine = PairingEngine()
        # comparand must be GRID in the seeded combinations.
        with pytest.raises(PairingError):
            engine.get_strategy_for(G.POINT, G.TRACK)

    def test_legacy_get_strategy_still_works(self) -> None:
        engine = PairingEngine()
        assert engine.get_strategy(G.POINT).geometry is G.POINT


class _SpyStrategy(BasePairingStrategy):
    """Minimal concrete strategy that records what pair() received."""

    def __init__(self) -> None:
        self.captured: dict[str, Any] = {}

    @property
    def geometry(self) -> DataGeometry:
        return DataGeometry.POINT

    def pair(
        self,
        model: xr.Dataset,
        obs: xr.Dataset,
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
        self.captured = {"model": model, "obs": obs, "kwargs": kwargs}
        return xr.Dataset(attrs={"ok": True})


def _coords_ds(lat: float, lon: float) -> xr.Dataset:
    return xr.Dataset(
        {"v": ("site", np.array([1.0]))},
        coords={"latitude": ("site", np.array([lat])), "longitude": ("site", np.array([lon]))},
    )


class TestRoleNeutralWrapper:
    def test_pair_sources_maps_reference_to_obs_and_comparand_to_model(self) -> None:
        strat = _SpyStrategy()
        reference = _coords_ds(10.0, 20.0)
        comparand = _coords_ds(30.0, 40.0)
        out = strat.pair_sources(reference=reference, comparand=comparand, radius_of_influence=1.0)
        assert out.attrs["ok"] is True
        # reference -> obs, comparand -> model (preserving today's model->obs sampling).
        assert strat.captured["obs"] is reference
        assert strat.captured["model"] is comparand
        assert strat.captured["kwargs"]["radius_of_influence"] == 1.0

    def test_coord_helper_aliases(self) -> None:
        strat = _SpyStrategy()
        ref = _coords_ds(1.0, 2.0)
        comp = _coords_ds(3.0, 4.0)
        # reference alias mirrors _get_obs_coords; comparand alias mirrors _get_model_coords.
        assert strat._get_reference_coords(ref)[0].values == strat._get_obs_coords(ref)[0].values
        assert (
            strat._get_comparand_coords(comp)[0].values == strat._get_model_coords(comp)[0].values
        )
