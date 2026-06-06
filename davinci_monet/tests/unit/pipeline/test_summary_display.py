"""Tests for AI summary terminal display and benign-skip log level.

Covers two bugs found via a live CLI run:
- the summary brief was never printed to the terminal (swallowed by the
  prefix-matching progress callback), violating the spec's "file + terminal".
- obs-only stages idle in a paired run logged "validation failed, skipping" at
  WARNING, reading as failures.
"""

from __future__ import annotations

import logging
from io import StringIO

from rich.console import Console

from davinci_monet.pipeline.runner import PipelineRunner, ProgressFormatter
from davinci_monet.pipeline.stages import (
    BaseStage,
    PipelineContext,
    StageResult,
    StageStatus,
)


def _formatter_with_buffer(show_output: bool) -> tuple[ProgressFormatter, StringIO]:
    fmt = ProgressFormatter(show_output=show_output)
    buf = StringIO()
    fmt.console = Console(file=buf, force_terminal=False, no_color=True, width=80)
    return fmt, buf


def test_print_summary_lists_items_and_file() -> None:
    fmt, buf = _formatter_with_buffer(show_output=True)
    fmt.print_summary(["MB +4.82 ppb", "R 0.849"], "/out/AI_summary.md")
    out = buf.getvalue()
    assert "MB +4.82 ppb" in out
    assert "R 0.849" in out
    assert "AI_summary.md" in out


def test_print_summary_silent_when_disabled() -> None:
    fmt, buf = _formatter_with_buffer(show_output=False)
    fmt.print_summary(["hidden"], "/out/AI_summary.md")
    assert buf.getvalue() == ""


def test_print_summary_noop_when_no_items() -> None:
    fmt, buf = _formatter_with_buffer(show_output=True)
    fmt.print_summary([], "/out/AI_summary.md")
    assert buf.getvalue() == ""


class _AlwaysInvalidStage(BaseStage):
    def __init__(self) -> None:
        super().__init__("always_invalid")

    def validate(self, context: PipelineContext) -> bool:
        return False

    def execute(self, context: PipelineContext) -> StageResult:
        raise AssertionError("execute must not run when validate() is False")


def test_validation_skip_is_not_a_warning(caplog) -> None:
    runner = PipelineRunner(stages=[_AlwaysInvalidStage()], show_progress=False)
    ctx = PipelineContext(config={})
    with caplog.at_level(logging.DEBUG, logger="davinci_monet.pipeline.runner"):
        result = runner._execute_stage(_AlwaysInvalidStage(), ctx)
    assert result.status == StageStatus.SKIPPED
    offending = [
        r
        for r in caplog.records
        if r.levelno >= logging.WARNING and "valid" in r.getMessage().lower()
    ]
    assert not offending, (
        "a benign not-applicable skip must not be logged at WARNING: "
        f"{[r.getMessage() for r in offending]}"
    )
