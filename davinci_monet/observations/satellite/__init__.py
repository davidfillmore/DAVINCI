"""Satellite observation readers.

This subpackage provides readers for satellite-based observations.

Satellite-Specific Readers
--------------------------
These readers are optimized for specific satellite products and rely on
monetio for full functionality:

- **TROPOMIReader** / **open_tropomi**: TROPOMI L2 products (NO2, O3, CO, HCHO, SO2)
  Requires: monetio.sat._tropomi_l2_no2_mm

- **TEMPOL2NO2Reader** / **open_tempo_l2_no2**: TEMPO L2 NO2 products
  Requires: monetio.sat._tempo_l2_no2_mm

- **MODISL2AODReader** / **open_modis_l2_aod**: MODIS L2 AOD products
  Requires: monetio.sat._modis_l2_mm

- **GOESL3AODReader** / **open_goes_l3_aod**: GOES-ABI L3 AOD products
  Requires: monetio.sat.goes

- **MOPITTL3COReader** / **open_mopitt_l3_co**: MOPITT L3 CO products
  Requires: monetio.sat._mopitt_l3_mm

- **OMPSL3O3Reader** / **open_omps_l3_o3**: OMPS L3 total ozone products
  Requires: monetio.sat._omps_l3_mm

Generic Readers
---------------
These readers work with any satellite product but lack satellite-specific
features like projection handling and specialized QA filtering:

- **GenericL2Reader** / **open_satellite_l2**: Generic L2 swath products
- **GenericL3Reader** / **open_satellite_l3**: Generic L3 gridded products

Note
----
The satellite-specific readers fall back to basic xarray reading if monetio
is not available, but some features (projections, swath geometry) may not
work correctly without monetio.
"""

# Generic readers (pure xarray, no monetio dependency)
from davinci_monet.observations.satellite.generic_l2 import GenericL2Reader, open_satellite_l2
from davinci_monet.observations.satellite.generic_l3 import GenericL3Reader, open_satellite_l3

# Satellite-specific L3 readers (require monetio for full functionality)
from davinci_monet.observations.satellite.goes_l3_aod import (
    GOES_VARIABLE_MAPPING,  # Backward compatibility alias
)
from davinci_monet.observations.satellite.goes_l3_aod import (
    GOESReader,  # Backward compatibility alias
)
from davinci_monet.observations.satellite.goes_l3_aod import (
    open_goes,  # Backward compatibility alias (deprecated)
)
from davinci_monet.observations.satellite.goes_l3_aod import (
    GOES_AOD_VARIABLE_MAPPING,
    GOESL3AODReader,
    open_goes_l3_aod,
)
from davinci_monet.observations.satellite.modis_l2_aod import (
    MODIS_AOD_VARIABLE_MAPPING,
    MODISL2AODReader,
    open_modis_l2_aod,
)
from davinci_monet.observations.satellite.mopitt_l3_co import (
    MOPITT_CO_VARIABLE_MAPPING,
    MOPITTL3COReader,
    open_mopitt_l3_co,
)
from davinci_monet.observations.satellite.omps_l3_o3 import (
    OMPS_O3_VARIABLE_MAPPING,
    OMPSL3O3Reader,
    open_omps_l3_o3,
)
from davinci_monet.observations.satellite.tempo_l2_no2 import (
    TEMPO_NO2_VARIABLE_MAPPING,
    TEMPOL2NO2Reader,
    open_tempo_l2_no2,
)

# Satellite-specific L2 readers (require monetio for full functionality)
from davinci_monet.observations.satellite.tropomi import (
    TROPOMI_VARIABLE_MAPPING,
    TROPOMIReader,
    open_tropomi,
)

__all__ = [
    # TROPOMI L2
    "TROPOMIReader",
    "open_tropomi",
    "TROPOMI_VARIABLE_MAPPING",
    # TEMPO L2 NO2
    "TEMPOL2NO2Reader",
    "open_tempo_l2_no2",
    "TEMPO_NO2_VARIABLE_MAPPING",
    # MODIS L2 AOD
    "MODISL2AODReader",
    "open_modis_l2_aod",
    "MODIS_AOD_VARIABLE_MAPPING",
    # GOES L3 AOD
    "GOESL3AODReader",
    "GOESReader",  # Backward compatibility
    "open_goes_l3_aod",
    "open_goes",  # Backward compatibility (deprecated)
    "GOES_AOD_VARIABLE_MAPPING",
    "GOES_VARIABLE_MAPPING",  # Backward compatibility
    # MOPITT L3 CO
    "MOPITTL3COReader",
    "open_mopitt_l3_co",
    "MOPITT_CO_VARIABLE_MAPPING",
    # OMPS L3 O3
    "OMPSL3O3Reader",
    "open_omps_l3_o3",
    "OMPS_O3_VARIABLE_MAPPING",
    # Generic L2
    "GenericL2Reader",
    "open_satellite_l2",
    # Generic L3
    "GenericL3Reader",
    "open_satellite_l3",
]
