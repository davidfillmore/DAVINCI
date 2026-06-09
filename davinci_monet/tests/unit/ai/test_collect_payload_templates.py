"""collect_payload attaches a resolved template to each stats row."""

from __future__ import annotations

from types import SimpleNamespace

from davinci_monet.ai.payload import collect_payload
from davinci_monet.config.schema import SummaryConfig


def _context(stats: dict) -> SimpleNamespace:
    return SimpleNamespace(
        config={
            "analysis": {"start_time": "2024-02-01", "end_time": "2024-02-03"},
            "sources": {"cam": {"type": "cesm_fv", "role": "model"}},
            "pairs": {"p_o3": {}, "p_pm": {}},
        },
        results={"statistics": SimpleNamespace(data=stats)},
    )


def test_each_row_gets_template_by_variable() -> None:
    ctx = _context(
        {
            "p_o3": {"O3": {"N": 10, "MB": 1.0}},
            "p_pm": {"PM25": {"N": 12, "MB": -2.0}},
        }
    )
    payload = collect_payload(ctx, SummaryConfig())
    by_var = {row["variable"]: row["template"].name for row in payload.stats_rows}
    assert by_var["O3"] == "ozone_eval"
    assert by_var["PM25"] == "pm_eval"


def test_override_forces_template_for_pair() -> None:
    ctx = _context({"p_o3": {"O3": {"N": 10}}})
    cfg = SummaryConfig.model_validate({"template_overrides": {"p_o3": "trace_gas_eval"}})
    payload = collect_payload(ctx, cfg)
    assert payload.stats_rows[0]["template"].name == "trace_gas_eval"
