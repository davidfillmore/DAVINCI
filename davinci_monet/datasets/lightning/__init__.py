"""Lightning dataset readers.

This subpackage provides readers for lightning mapping array (LMA) data
from networks such as OKLMA, COLMA, and NALMA.
"""

from davinci_monet.datasets.lightning.lma import LMAReader

__all__ = ["LMAReader"]
