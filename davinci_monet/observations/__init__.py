"""Observation data handling module.

This module provides classes for loading and processing observational
data from various sources (surface stations, aircraft, satellites, etc.).
"""

from davinci_monet.core.protocols import DataGeometry

# Aircraft observation readers
from davinci_monet.observations.aircraft.icartt import ICARTTReader

# Generic NetCDF readers + dataset helpers
from davinci_monet.observations.base import (
    AircraftReader,
    GriddedObsReader,
    PointSurfaceReader,
    ProfileReader,
    resample_dataset,
)

# Lightning observation readers
from davinci_monet.observations.lightning.lma import LMAReader

# Satellite observation readers - L3 gridded
from davinci_monet.observations.satellite.ceres_l3 import CERESEBAFReader, CERESSYN1degReader

# Generic satellite readers
from davinci_monet.observations.satellite.generic_l2 import GenericL2Reader
from davinci_monet.observations.satellite.generic_l3 import GenericL3Reader
from davinci_monet.observations.satellite.goes_l3_aod import (
    GOESReader,  # Backward compatibility alias
)
from davinci_monet.observations.satellite.goes_l3_aod import (
    GOESL3AODReader,
)
from davinci_monet.observations.satellite.modis_l2_aod import MODISL2AODReader
from davinci_monet.observations.satellite.modis_viirs import MODISVIIRSReader
from davinci_monet.observations.satellite.mopitt_l3_co import MOPITTL3COReader
from davinci_monet.observations.satellite.omps_l3_o3 import OMPSL3O3Reader
from davinci_monet.observations.satellite.tempo_l2_no2 import TEMPOL2NO2Reader

# Satellite observation readers - L2 swath
from davinci_monet.observations.satellite.tropomi import TROPOMIReader

# Sonde observation readers
from davinci_monet.observations.sonde.ozonesonde import OzonesondeReader

# Surface observation readers
from davinci_monet.observations.surface.aeronet import AERONETReader
from davinci_monet.observations.surface.airnow import AirNowReader
from davinci_monet.observations.surface.aqs import AQSReader
from davinci_monet.observations.surface.openaq import OpenAQReader
from davinci_monet.observations.surface.pandora import PandoraReader


def _geometry_property(geometry: DataGeometry) -> property:
    """Create a SourceReader-compatible geometry property."""

    return property(lambda self: geometry)


for _reader_cls, _geometry in {
    AQSReader: DataGeometry.POINT,
    AirNowReader: DataGeometry.POINT,
    AERONETReader: DataGeometry.POINT,
    OpenAQReader: DataGeometry.POINT,
    PandoraReader: DataGeometry.POINT,
    ICARTTReader: DataGeometry.TRACK,
    OzonesondeReader: DataGeometry.PROFILE,
    LMAReader: DataGeometry.GRID,
    GOESL3AODReader: DataGeometry.GRID,
    MODISL2AODReader: DataGeometry.SWATH,
    MOPITTL3COReader: DataGeometry.GRID,
    OMPSL3O3Reader: DataGeometry.GRID,
    TEMPOL2NO2Reader: DataGeometry.SWATH,
    TROPOMIReader: DataGeometry.SWATH,
    GenericL2Reader: DataGeometry.SWATH,
    GenericL3Reader: DataGeometry.GRID,
}.items():
    if not hasattr(_reader_cls, "geometry"):
        setattr(_reader_cls, "geometry", _geometry_property(_geometry))

del _reader_cls, _geometry

__all__ = [
    # Generic NetCDF readers + helpers
    "PointSurfaceReader",
    "AircraftReader",
    "ProfileReader",
    "GriddedObsReader",
    "resample_dataset",
    # Surface readers
    "AQSReader",
    "AirNowReader",
    "AERONETReader",
    "OpenAQReader",
    "PandoraReader",
    # Aircraft readers
    "ICARTTReader",
    # Satellite L2 readers
    "TROPOMIReader",
    "TEMPOL2NO2Reader",
    "MODISL2AODReader",
    # Satellite L3 readers
    "CERESEBAFReader",
    "CERESSYN1degReader",
    "MODISVIIRSReader",
    "GOESL3AODReader",
    "GOESReader",  # Backward compatibility
    "MOPITTL3COReader",
    "OMPSL3O3Reader",
    # Generic satellite readers
    "GenericL2Reader",
    "GenericL3Reader",
    # Sonde readers
    "OzonesondeReader",
    # Lightning readers
    "LMAReader",
]
