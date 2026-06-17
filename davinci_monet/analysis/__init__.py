"""Derived-analysis package: EOF, wavelet, and the shared base/registry.

Importing this package registers all concrete analyses as an import
side-effect (added in later plans). The registry itself lives in
``davinci_monet.core.registry`` to avoid circular imports.
"""

from __future__ import annotations

from davinci_monet.analysis.base import DerivedAnalysis

__all__ = ["DerivedAnalysis"]
