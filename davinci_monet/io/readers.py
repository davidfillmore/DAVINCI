"""File readers for various data formats.

This module previously provided generic readers for NetCDF, pickle, CSV,
and ICARTT formats.  Those functions had no production callers and have been
removed.  Source-specific readers live in ``davinci_monet.models`` and
``davinci_monet.observations``; shared I/O utilities are in
``davinci_monet.io.reader_utils``.
"""
