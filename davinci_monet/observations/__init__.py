"""Observation data handling module.

This module provides classes for loading and processing observational
data from various sources (surface stations, aircraft, satellites, etc.).
"""

# Aircraft observation readers
from davinci_monet.observations.aircraft.icartt import ICARTTReader, open_icartt
from davinci_monet.observations.base import (
    GriddedObservation,
    ObservationData,
    PointObservation,
    ProfileObservation,
    SwathObservation,
    TrackObservation,
    create_observation_data,
)

# Lightning observation readers
from davinci_monet.observations.lightning.lma import LMAReader, open_lma

# Generic satellite readers
from davinci_monet.observations.satellite.generic_l2 import (
    GenericL2Reader,
    open_satellite_l2,
)
from davinci_monet.observations.satellite.generic_l3 import (
    GenericL3Reader,
    open_satellite_l3,
)

# Satellite observation readers - L3 gridded
from davinci_monet.observations.satellite.goes_l3_aod import (
    GOESReader,  # Backward compatibility alias
)
from davinci_monet.observations.satellite.goes_l3_aod import (
    open_goes,  # Backward compatibility alias (deprecated)
)
from davinci_monet.observations.satellite.goes_l3_aod import (
    GOESL3AODReader,
    open_goes_l3_aod,
)
from davinci_monet.observations.satellite.modis_l2_aod import (
    MODISL2AODReader,
    open_modis_l2_aod,
)
from davinci_monet.observations.satellite.mopitt_l3_co import (
    MOPITTL3COReader,
    open_mopitt_l3_co,
)
from davinci_monet.observations.satellite.omps_l3_o3 import (
    OMPSL3O3Reader,
    open_omps_l3_o3,
)
from davinci_monet.observations.satellite.tempo_l2_no2 import (
    TEMPOL2NO2Reader,
    open_tempo_l2_no2,
)

# Satellite observation readers - L2 swath
from davinci_monet.observations.satellite.tropomi import TROPOMIReader, open_tropomi

# Sonde observation readers
from davinci_monet.observations.sonde.ozonesonde import OzonesondeReader, open_ozonesonde

# Surface observation readers
from davinci_monet.observations.surface.aeronet import AERONETReader, open_aeronet
from davinci_monet.observations.surface.airnow import AirNowReader, open_airnow
from davinci_monet.observations.surface.aqs import AQSReader, open_aqs
from davinci_monet.observations.surface.openaq import OpenAQReader, open_openaq
from davinci_monet.observations.surface.pandora import PandoraReader, open_pandora

__all__ = [
    # Base classes
    "ObservationData",
    "PointObservation",
    "TrackObservation",
    "ProfileObservation",
    "SwathObservation",
    "GriddedObservation",
    "create_observation_data",
    # Surface readers
    "AQSReader",
    "AirNowReader",
    "AERONETReader",
    "OpenAQReader",
    "PandoraReader",
    "open_aqs",
    "open_airnow",
    "open_aeronet",
    "open_openaq",
    "open_pandora",
    # Aircraft readers
    "ICARTTReader",
    "open_icartt",
    # Satellite L2 readers
    "TROPOMIReader",
    "open_tropomi",
    "TEMPOL2NO2Reader",
    "open_tempo_l2_no2",
    "MODISL2AODReader",
    "open_modis_l2_aod",
    # Satellite L3 readers
    "GOESL3AODReader",
    "GOESReader",  # Backward compatibility
    "open_goes_l3_aod",
    "open_goes",  # Backward compatibility (deprecated)
    "MOPITTL3COReader",
    "open_mopitt_l3_co",
    "OMPSL3O3Reader",
    "open_omps_l3_o3",
    # Generic satellite readers
    "GenericL2Reader",
    "GenericL3Reader",
    "open_satellite_l2",
    "open_satellite_l3",
    # Sonde readers
    "OzonesondeReader",
    "open_ozonesonde",
    # Lightning readers
    "LMAReader",
    "open_lma",
]
