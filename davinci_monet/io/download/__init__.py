"""Optional data-staging helpers for reanalysis and EOS data sources.

Public API is re-exported from the product modules (``merra2``, ``ceres``).
These helpers require optional extras (``pip install -e ".[reanalysis]"``)
that are imported lazily, so importing this package never requires the
network or ``earthaccess`` to be installed.
"""

from davinci_monet.io.download.ceres import CERES_COLLECTIONS, DryRunReport, stage_ceres
from davinci_monet.io.download.merra2 import MERRA2_COLLECTIONS, stage_merra2

__all__ = [
    "CERES_COLLECTIONS",
    "DryRunReport",
    "MERRA2_COLLECTIONS",
    "stage_ceres",
    "stage_merra2",
]
