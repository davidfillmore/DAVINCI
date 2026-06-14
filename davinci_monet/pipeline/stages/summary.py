"""AI summary stage.

Optional final stage that sends stats + plot images to the Claude API. Always
non-fatal: any failure (missing key/dep, network/API error) degrades to SKIPPED.
"""

from __future__ import annotations

from davinci_monet.pipeline.stages.base import (
    BaseStage,
    PipelineContext,
    StageResult,
    StageStatus,
)


class SummaryStage(BaseStage):
    """Optional final stage: AI summary of the analysis run via the Claude API.

    Always non-fatal. When ``summary.enabled`` is false the stage is skipped.
    Any failure (missing dependency/key, network/API error) logs a warning and
    returns SKIPPED so an otherwise-complete run is still reported successful.
    """

    def __init__(self) -> None:
        super().__init__("summary")

    def execute(self, context: PipelineContext) -> StageResult:
        import logging
        import time
        from pathlib import Path

        from davinci_monet.ai import collect_payload, extract_bullets, generate_summary
        from davinci_monet.config.schema import SummaryConfig
        from davinci_monet.core.schema_utils import validate_schema

        start = time.time()
        logger = logging.getLogger(__name__)

        # The summary is a bonus produced after the analysis is already complete,
        # so this stage must never fail the run. Any error (config validation,
        # payload collection, the API call, or writing the file) degrades to
        # SKIPPED with a warning rather than propagating to the runner, which
        # would otherwise mark the whole run FAILED.
        try:
            cfg = validate_schema(SummaryConfig, context.config.get("summary") or {})
            if not cfg.enabled:
                return self._create_result(
                    StageStatus.SKIPPED,
                    data={"skipped": "summary disabled"},
                    duration=time.time() - start,
                )

            payload = collect_payload(context, cfg)
            result = generate_summary(payload, cfg=cfg)

            output_dir = Path(context.config.get("analysis", {}).get("output_dir") or ".")
            output_dir.mkdir(parents=True, exist_ok=True)
            out_path = output_dir / cfg.output_filename
            out_path.write_text(result.markdown)
        except Exception as exc:  # noqa: BLE001 - summary must never fail the run
            logger.warning("AI summary skipped: %s: %s", type(exc).__name__, exc)
            return self._create_result(
                StageStatus.SKIPPED,
                data={"skipped": f"{type(exc).__name__}: {exc}"},
                duration=time.time() - start,
            )

        # The brief is displayed by the runner at end of run (via
        # ProgressFormatter.print_summary, reading data["markdown"]). A raw
        # log_progress(markdown) here is swallowed by the prefix-matching
        # progress callback, so it is not used for display.
        context.log_progress(f"done: AI summary written ({result.images_sent} figures)")

        return self._create_result(
            StageStatus.COMPLETED,
            data={
                "summary_file": str(out_path),
                "markdown": result.markdown,
                "bullets": extract_bullets(result.markdown),
                "model": result.model,
                "usage": result.usage,
                "credits_remaining": result.credits_remaining,
                "images_sent": result.images_sent,
            },
            duration=time.time() - start,
        )
