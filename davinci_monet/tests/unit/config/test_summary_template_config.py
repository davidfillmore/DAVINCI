"""SummaryConfig gains optional template fields."""

from __future__ import annotations

from davinci_monet.config.schema import SummaryConfig
from davinci_monet.core.schema_utils import validate_schema


def test_template_fields_default_none() -> None:
    cfg = SummaryConfig()
    assert cfg.templates is None
    assert cfg.template_overrides is None


def test_inline_templates_and_overrides_parse() -> None:
    cfg = validate_schema(
        SummaryConfig,
        {
            "enabled": True,
            "templates": {
                "my_o3": {
                    "matches": ["o3"],
                    "sections": [{"heading": "h", "format": "headline", "words": 12}],
                }
            },
            "template_overrides": {"pair_a": "my_o3"},
        },
    )
    assert cfg.templates is not None and "my_o3" in cfg.templates
    assert cfg.template_overrides == {"pair_a": "my_o3"}
