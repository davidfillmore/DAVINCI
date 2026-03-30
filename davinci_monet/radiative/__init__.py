"""Smoke radiative analysis subpackage.

Provides tools for analyzing wildfire smoke radiative effects using
CERES, MERRA-2, and AERONET data. This is a standalone analysis
workflow that uses DAVINCI's config, styling, and CLI infrastructure.

Usage
-----
Via CLI::

    davinci-monet radiative run config.yaml
    davinci-monet radiative fetch-ceres --product syn1deg --start 2020-09-05 --end 2020-09-15

Via Python::

    from davinci_monet.radiative.runner import run_radiative_analysis
    result = run_radiative_analysis("config.yaml")
"""

from davinci_monet.radiative.config import RadiativeConfig
from davinci_monet.radiative.runner import run_radiative_analysis

__all__ = ["RadiativeConfig", "run_radiative_analysis"]
