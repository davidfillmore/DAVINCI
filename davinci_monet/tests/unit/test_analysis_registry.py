"""The analysis registry registers and retrieves analysis classes."""

from __future__ import annotations

from davinci_monet.core.registry import analysis_registry


def test_analysis_registry_register_and_get() -> None:
    @analysis_registry.register("dummy_t1")
    class Dummy:
        name = "dummy_t1"

    assert analysis_registry.get("dummy_t1") is Dummy
    assert "dummy_t1" in analysis_registry.list()
    analysis_registry.unregister("dummy_t1")
