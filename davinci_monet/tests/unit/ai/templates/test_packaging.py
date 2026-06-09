"""Template API is exported and the YAML data is declared as package data."""

from __future__ import annotations

import tomllib
from pathlib import Path


def test_template_api_exported_from_ai() -> None:
    import davinci_monet.ai as ai

    assert hasattr(ai, "get_template_registry")
    assert hasattr(ai, "resolve_template_for")
    assert hasattr(ai, "SummaryTemplate")


def test_template_yaml_declared_as_package_data() -> None:
    repo_root = Path(__file__).resolve().parents[5]
    data = tomllib.loads((repo_root / "pyproject.toml").read_text())
    pkg_data = data["tool"]["setuptools"]["package-data"]["davinci_monet"]
    assert any("ai/templates/data/*.yaml" in entry for entry in pkg_data)
