"""Source reader registration bootstrap helpers."""

from __future__ import annotations

import importlib

BUILTIN_SOURCE_READER_MODULES = ("davinci_monet.datasets",)


def ensure_builtin_source_readers_registered() -> None:
    """Import built-in reader packages so their registry decorators run."""
    for module_name in BUILTIN_SOURCE_READER_MODULES:
        importlib.import_module(module_name)
