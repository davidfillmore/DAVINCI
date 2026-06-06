"""Aircraft observation readers.

This subpackage provides readers for aircraft-based observations including
ICARTT format files from field campaigns.
"""

from davinci_monet.observations.aircraft.icartt import ICARTTReader

__all__ = [
    "ICARTTReader",
]
