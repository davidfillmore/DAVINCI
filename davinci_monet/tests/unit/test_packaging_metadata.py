"""Tests for package metadata needed by optional features."""

from __future__ import annotations

import tomllib
from pathlib import Path


def test_ai_extra_declares_openrouter_http_client() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    data = tomllib.loads((repo_root / "pyproject.toml").read_text())

    ai_deps = data["project"]["optional-dependencies"]["ai"]
    assert any(dep.lower().startswith("httpx") for dep in ai_deps)
