"""Tests for HDF5-safe-default fix in PipelineRunner (WS5-A item 3).

Verifies that PipelineRunner.run() sets HDF5_USE_FILE_LOCKING=FALSE when
the environment variable is absent, and leaves an explicit value untouched.
"""

from __future__ import annotations

import os

import pytest

from davinci_monet.pipeline.runner import PipelineRunner
from davinci_monet.pipeline.stages import (
    BaseStage,
    PipelineContext,
    StageResult,
    StageStatus,
)

# ---------------------------------------------------------------------------
# Minimal stage that captures env at execution time
# ---------------------------------------------------------------------------


class _CapturingStage(BaseStage):
    """Records HDF5_USE_FILE_LOCKING at execution time then stops the pipeline."""

    name = "_capturing_stage"

    def __init__(self) -> None:
        self.captured_value: str | None = None

    def validate(self, context: PipelineContext) -> bool:
        return True

    def execute(self, context: PipelineContext) -> StageResult:
        self.captured_value = os.environ.get("HDF5_USE_FILE_LOCKING")
        # Raise to short-circuit; the runner will catch and mark FAILED but
        # we only care about the env value captured before the exception.
        raise SystemExit(0)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestHdf5SafeDefault:
    """PipelineRunner sets HDF5_USE_FILE_LOCKING safely before any stage runs."""

    def _run_and_capture(
        self,
        monkeypatch: pytest.MonkeyPatch,
        preset_value: str | None,
    ) -> str | None:
        """Run a one-stage pipeline and return the env value seen by the stage."""
        # Start from a clean slate
        monkeypatch.delenv("HDF5_USE_FILE_LOCKING", raising=False)
        if preset_value is not None:
            monkeypatch.setenv("HDF5_USE_FILE_LOCKING", preset_value)

        capturing = _CapturingStage()
        runner = PipelineRunner(stages=[capturing], show_progress=False)
        ctx = PipelineContext(config={})

        # The stage raises SystemExit; run() catches it internally as a
        # stage failure.  We don't care about the result, only the captured value.
        try:
            runner.run(ctx)
        except SystemExit:
            pass

        return capturing.captured_value

    def test_sets_false_when_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When HDF5_USE_FILE_LOCKING is absent the runner sets it to FALSE."""
        value = self._run_and_capture(monkeypatch, preset_value=None)
        assert value == "FALSE", f"Expected HDF5_USE_FILE_LOCKING='FALSE' when unset, got {value!r}"

    def test_leaves_explicit_false_untouched(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """An existing 'FALSE' value is preserved."""
        value = self._run_and_capture(monkeypatch, preset_value="FALSE")
        assert value == "FALSE"

    def test_leaves_explicit_true_untouched(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """An explicit 'TRUE' value is NOT overridden by the runner."""
        value = self._run_and_capture(monkeypatch, preset_value="TRUE")
        assert (
            value == "TRUE"
        ), f"Runner must not override an explicit HDF5_USE_FILE_LOCKING=TRUE, got {value!r}"

    def test_leaves_custom_value_untouched(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Any pre-existing value other than FALSE is preserved."""
        value = self._run_and_capture(monkeypatch, preset_value="CUSTOM")
        assert value == "CUSTOM"

    def test_env_set_in_os_environ_after_run(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """After run(), os.environ reflects HDF5_USE_FILE_LOCKING=FALSE if it was unset."""
        monkeypatch.delenv("HDF5_USE_FILE_LOCKING", raising=False)

        runner = PipelineRunner(stages=[], show_progress=False)
        ctx = PipelineContext(config={})
        runner.run(ctx)

        assert os.environ.get("HDF5_USE_FILE_LOCKING") == "FALSE"
