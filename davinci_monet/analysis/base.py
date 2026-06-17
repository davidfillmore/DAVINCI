"""Base class for derived analyses (field/series-producing, not scalar)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import xarray as xr

    from davinci_monet.core.protocols import DataGeometry


class DerivedAnalysis(ABC):
    """An analysis that consumes ONE source dataset and emits a derived dataset.

    Concrete analyses register via ``@analysis_registry.register("<type>")`` and
    set ``output_geometry`` to the geometry of their principal output field.
    """

    name: str = "base"
    long_name: str = "Base Derived Analysis"
    output_geometry: "DataGeometry"

    @abstractmethod
    def analyze(self, data: "xr.Dataset", spec: Any) -> "xr.Dataset":
        """Return a derived dataset.

        ``data`` is the fully-built input dataset (a raw source or an
        already-built derived source). ``spec`` is the validated Pydantic
        params for this analysis entry.
        """
        ...
