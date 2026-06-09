"""Unit tests for template loading and resolution."""

from __future__ import annotations

import pytest

from davinci_monet.ai.templates.registry import (
    UnknownTemplateError,
    get_template_registry,
    resolve_template_for,
)


def test_builtin_library_loads() -> None:
    reg = get_template_registry()
    assert {"generic_eval", "ozone_eval", "aerosol_aod_eval", "pm_eval", "trace_gas_eval"} <= set(
        reg.names()
    )


def test_resolve_by_variable_match() -> None:
    assert resolve_template_for("O3").name == "ozone_eval"
    assert resolve_template_for("aod_550nm").name == "aerosol_aod_eval"
    assert resolve_template_for("PM25").name == "pm_eval"
    assert resolve_template_for("NO2").name == "trace_gas_eval"


def test_unmatched_variable_falls_back_to_generic() -> None:
    assert resolve_template_for("relative_humidity").name == "generic_eval"


def test_explicit_override_wins() -> None:
    assert resolve_template_for("O3", override="pm_eval").name == "pm_eval"


def test_unknown_override_raises_with_hint() -> None:
    with pytest.raises(UnknownTemplateError) as exc:
        resolve_template_for("O3", override="ozon_eval")
    assert "ozone_eval" in str(exc.value)


def test_inline_template_merges_and_extends_index() -> None:
    inline: dict[str, dict] = {
        "ozone_eval": {
            "title": "Custom O3",
            "matches": ["o3"],
            "sections": [{"heading": "Just one", "format": "headline", "words": 10}],
        },
        "custom_scn": {
            "matches": ["mytracer"],
            "sections": [{"heading": "X", "format": "prose", "words": 10}],
        },
    }
    assert resolve_template_for("O3", inline=inline).title == "Custom O3"
    assert resolve_template_for("mytracer", inline=inline).name == "custom_scn"
