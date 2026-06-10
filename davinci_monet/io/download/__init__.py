"""Optional data-staging helpers for reanalysis sources.

Public API is re-exported once the staging functions are defined (see
``merra2``). These helpers require optional extras (``pip install -e
".[reanalysis]"``) that are imported lazily, so importing this package never
requires the network or ``earthaccess`` to be installed.
"""
