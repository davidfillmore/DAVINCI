"""Sonde dataset readers.

This subpackage provides readers for vertical profile datasets from
balloon-borne instruments including ozonesondes.
"""

from davinci_monet.datasets.sonde.ozonesonde import OzonesondeReader

__all__ = [
    "OzonesondeReader",
]
