"""Tests for source reader registration bootstrap helpers."""

from __future__ import annotations

from davinci_monet.io import source_registration


def test_ensure_builtin_source_readers_registered_imports_builtin_packages(
    monkeypatch,
) -> None:
    imported: list[str] = []

    def fake_import_module(module_name: str) -> object:
        imported.append(module_name)
        return object()

    monkeypatch.setattr(
        source_registration.importlib,
        "import_module",
        fake_import_module,
    )

    source_registration.ensure_builtin_source_readers_registered()

    assert imported == list(source_registration.BUILTIN_SOURCE_READER_MODULES)
