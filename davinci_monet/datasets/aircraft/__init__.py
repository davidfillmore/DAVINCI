"""Aircraft dataset readers.

This subpackage provides readers for aircraft-based datasets including
ICARTT format files from field campaigns.
"""

from davinci_monet.datasets.aircraft.icartt import ICARTTReader

__all__ = [
    "ICARTTReader",
]
