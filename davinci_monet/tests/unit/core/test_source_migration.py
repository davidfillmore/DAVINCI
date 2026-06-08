"""Phase 2 tests: reader geometry conformance + unified source_registry.

Phase 2 of the model/obs unification migrates every reader registration onto
the single ``source_registry`` and requires every model reader to declare its
output geometry so it satisfies the ``SourceReader`` protocol (observation
readers already do, from Phase 1). ``model_registry`` and
``observation_registry`` become deprecated aliases of ``source_registry``.
"""

from __future__ import annotations

import pytest

# Importing the packages triggers every reader's registration side effect.
import davinci_monet.models  # noqa: F401
import davinci_monet.observations  # noqa: F401
from davinci_monet.core.protocols import DataGeometry, SourceReader
from davinci_monet.core.registry import source_registry
from davinci_monet.models.cesm import CESMFVReader, CESMSEReader
from davinci_monet.models.cmaq import CMAQReader
from davinci_monet.models.generic import GenericReader
from davinci_monet.models.ufs import RRFSReader, UFSReader
from davinci_monet.models.wrfchem import WRFChemReader

MODEL_READERS = [
    CMAQReader,
    GenericReader,
    UFSReader,
    RRFSReader,
    CESMFVReader,
    CESMSEReader,
    WRFChemReader,
]

MODEL_TYPES = ["cmaq", "generic", "ufs", "rrfs", "cesm_fv", "cesm_se", "wrfchem"]
OBS_TYPES = [
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

OBS_READER_TYPES = [
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


class TestModelReaderGeometry:
    @pytest.mark.parametrize("reader_cls", MODEL_READERS)
    def test_reader_declares_grid_geometry(self, reader_cls: type) -> None:
        assert reader_cls().geometry is DataGeometry.GRID

    @pytest.mark.parametrize("reader_cls", MODEL_READERS)
    def test_reader_satisfies_source_reader(self, reader_cls: type) -> None:
        assert isinstance(reader_cls(), SourceReader)


class TestUnifiedSourceRegistry:
    @pytest.mark.parametrize("type_id", MODEL_TYPES + OBS_TYPES)
    def test_type_registered_in_source_registry(self, type_id: str) -> None:
        assert type_id in source_registry

    def test_model_and_obs_lookups_in_source_registry(self) -> None:
        # Both model and obs readers are registered directly on source_registry.
        assert source_registry.get("cmaq") is not None
        assert source_registry.get("airnow") is not None

    @pytest.mark.parametrize("type_id", OBS_READER_TYPES)
    def test_observation_reader_satisfies_source_reader(self, type_id: str) -> None:
        reader_cls = source_registry.get(type_id)
        assert isinstance(reader_cls(), SourceReader)
