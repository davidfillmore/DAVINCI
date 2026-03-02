"""Lightning observation readers.

This subpackage provides readers for lightning mapping array (LMA) data
from networks such as OKLMA, COLMA, and NALMA.
"""

from davinci_monet.observations.lightning.lma import LMAReader, open_lma

__all__ = ["LMAReader", "open_lma"]
