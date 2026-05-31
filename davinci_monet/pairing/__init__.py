"""Unified pairing engine for model-observation collocation."""

from davinci_monet.pairing.direction import (
    IRREGULAR_GEOMETRIES,
    PairDirectionWarning,
    resolve_pair_direction,
)
from davinci_monet.pairing.engine import PairingConfig, PairingEngine
from davinci_monet.pairing.strategies import (
    BasePairingStrategy,
    GridStrategy,
    PointStrategy,
    ProfileStrategy,
    SwathStrategy,
    TrackStrategy,
)

__all__ = [
    "IRREGULAR_GEOMETRIES",
    "PairDirectionWarning",
    "resolve_pair_direction",
    "BasePairingStrategy",
    "GridStrategy",
    "PairingConfig",
    "PairingEngine",
    "PointStrategy",
    "ProfileStrategy",
    "SwathStrategy",
    "TrackStrategy",
]
