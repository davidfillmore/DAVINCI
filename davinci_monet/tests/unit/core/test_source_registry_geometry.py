"""Reader geometry conformance and unified source_registry tests.

Every reader registration lives on the single ``source_registry`` and every
reader declares its output geometry so it satisfies the ``SourceReader``
protocol.
"""

from __future__ import annotations

import pytest

# Importing the package triggers every reader's registration side effect.
import davinci_monet.datasets  # noqa: F401
from davinci_monet.core.protocols import DataGeometry, SourceReader
from davinci_monet.core.registry import source_registry
from davinci_monet.datasets.cesm import CESMFVReader, CESMSEReader
from davinci_monet.datasets.cmaq import CMAQReader
from davinci_monet.datasets.generic import GenericReader
from davinci_monet.datasets.ufs import RRFSReader, UFSReader
from davinci_monet.datasets.wrfchem import WRFChemReader

GRID_READERS = [
    CMAQReader,
    GenericReader,
    UFSReader,
    RRFSReader,
    CESMFVReader,
    CESMSEReader,
    WRFChemReader,
]

GRID_TYPES = ["cmaq", "generic", "ufs", "rrfs", "cesm_fv", "cesm_se", "wrfchem"]
GEOMETRY_TYPES = [
    "lma",
    "icartt",
    "airnow",
    "aqs",
    "openaq",
    "aeronet",
    "pandora",
    "ozonesonde",
    "goes_l3_aod",
    "tempo_l2_no2",
    "modis_l2_aod",
    "omps_l3_o3",
    "satellite_l2",
    "tropomi",
    "mopitt_l3_co",
    "satellite_l3",
]

GEOMETRY_READER_TYPES = [
    "airnow",
    "aqs",
    "openaq",
    "aeronet",
    "pandora",
    "icartt",
    "ozonesonde",
    "lma",
    "goes_l3_aod",
    "tempo_l2_no2",
    "modis_l2_aod",
    "omps_l3_o3",
    "satellite_l2",
    "tropomi",
    "mopitt_l3_co",
    "satellite_l3",
]


class TestGridReaderGeometry:
    @pytest.mark.parametrize("reader_cls", GRID_READERS)
    def test_reader_declares_grid_geometry(self, reader_cls: type) -> None:
        assert reader_cls().geometry is DataGeometry.GRID

    @pytest.mark.parametrize("reader_cls", GRID_READERS)
    def test_reader_satisfies_source_reader(self, reader_cls: type) -> None:
        assert isinstance(reader_cls(), SourceReader)


class TestUnifiedSourceRegistry:
    @pytest.mark.parametrize("type_id", GRID_TYPES + GEOMETRY_TYPES)
    def test_type_registered_in_source_registry(self, type_id: str) -> None:
        assert type_id in source_registry

    def test_dataset_and_geometry_lookups_in_source_registry(self) -> None:
        # Both dataset and geometry readers are registered directly on source_registry.
        assert source_registry.get("cmaq") is not None
        assert source_registry.get("airnow") is not None

    @pytest.mark.parametrize("type_id", GEOMETRY_READER_TYPES)
    def test_geometry_reader_satisfies_source_reader(self, type_id: str) -> None:
        reader_cls = source_registry.get(type_id)
        assert isinstance(reader_cls(), SourceReader)
