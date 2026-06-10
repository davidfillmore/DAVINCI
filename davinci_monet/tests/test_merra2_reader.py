"""Unit tests for the MERRA-2 gridded reader."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from davinci_monet.core.protocols import DataGeometry
from davinci_monet.core.registry import source_registry
from davinci_monet.models.merra2 import MERRA2Reader


def test_reader_registered_and_grid_geometry() -> None:
    assert "merra2" in source_registry
    reader = MERRA2Reader()
    assert reader.name == "merra2"
    assert reader.geometry is DataGeometry.GRID
