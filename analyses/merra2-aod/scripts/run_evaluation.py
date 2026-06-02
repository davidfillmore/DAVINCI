#!/usr/bin/env python
"""Run the MERRA2 vs MODIS AOD evaluation through the DAVINCI pipeline."""
from __future__ import annotations

import sys
from pathlib import Path

from davinci_monet.pipeline.runner import run_analysis

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "merra2-modis-aod.example.yaml"


def main() -> int:
    config = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CONFIG
    result = run_analysis(str(config))
    if result.success:
        print(f"Completed in {result.total_duration_seconds:.1f}s")
        return 0
    print("Analysis failed", file=sys.stderr)
    for failed in result.failed_stages:
        print(f"  {failed.stage_name}: {failed.error}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
