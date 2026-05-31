"""Model data handling module.

This module provides classes for loading and processing atmospheric
model output from various sources (CMAQ, WRF-Chem, UFS, etc.).
"""

from davinci_monet.models.base import ModelData, create_model_data
from davinci_monet.models.cesm import CESMFVReader, CESMSEReader, open_cesm
from davinci_monet.models.cmaq import CMAQReader, open_cmaq
from davinci_monet.models.generic import GenericReader, open_model
from davinci_monet.models.ufs import RRFSReader, UFSReader, open_ufs
from davinci_monet.models.wrfchem import WRFChemReader, open_wrfchem

__all__ = [
    # Base classes
    "ModelData",
    "create_model_data",
    # Readers
    "CESMFVReader",
    "CESMSEReader",
    "CMAQReader",
    "GenericReader",
    "RRFSReader",
    "UFSReader",
    "WRFChemReader",
    # Convenience functions
    "open_cesm",
    "open_cmaq",
    "open_model",
    "open_ufs",
    "open_wrfchem",
]
