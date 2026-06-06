"""P2 of the renderer unification: collapse the obs/paired stage fork.

The unified PlottingStage / StatisticsStage handle obs-only runs (early-dispatch
when there are observations but no pairs), the obs stages leave the pipeline
builds, and obs descriptive stats are written to a separate
``statistics_descriptive.csv`` (Q3) while the paired ``statistics_summary.csv``
stays byte-identical. These are unit tests: they call stage ``execute`` directly
(consistent with the existing obs-stage tests), not through the pipeline.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from davinci_monet.core.protocols import DataGeometry
from davinci_monet.pipeline.stages import (
    PipelineContext,
    PlottingStage,
    SaveResultsStage,
    SourceData,
    StageStatus,
    StatisticsStage,
    create_standard_pipeline,
)


def _obs_ctx(tmp_path: Any) -> PipelineContext:
    n = 100
    rng = np.random.default_rng(0)
    ds = xr.Dataset(
        {
            "O3": ("time", rng.uniform(20, 120, n), {"units": "ppbv"}),
            "CO": ("time", rng.uniform(50, 300, n), {"units": "ppbv"}),
        },
        coords={"time": np.datetime64("2024-02-01") + np.arange(n) * np.timedelta64(1, "h")},
    )
    obs = SourceData(
        data=ds,
        label="airnow",
        source_type="pt_sfc",
        geometry=DataGeometry.POINT,
        role="obs",
    )
    return PipelineContext(
        config={
            "analysis": {"output_dir": str(tmp_path / "out")},
            "plots": {
                "o3_hist": {
                    "type": "obs_histogram",
                    "obs": "airnow",
                    "variable": "O3",
                    "title": "O3",
                }
            },
            "stats": {"metrics": ["N", "mean", "median", "std", "min", "max", "p10", "p90"]},
        },
        observations={"airnow": obs},
    )


class TestUnifiedStatisticsStage:
    def test_validate_true_for_obs_only(self, tmp_path: Any) -> None:
        assert StatisticsStage().validate(_obs_ctx(tmp_path)) is True

    def test_descriptive_stats_for_obs_only(self, tmp_path: Any) -> None:
        ctx = _obs_ctx(tmp_path)
        res = StatisticsStage().execute(ctx)
        assert res.status == StageStatus.COMPLETED
        assert "airnow" in res.data
        assert "O3" in res.data["airnow"]
        for m in ["N", "mean", "median", "std", "min", "max", "p10", "p25", "p75", "p90"]:
            assert m in res.data["airnow"]["O3"]
        assert res.data["airnow"]["O3"]["N"] == 100
        assert ctx.metadata.get("statistics_kind") == "descriptive"


class TestUnifiedPlottingStage:
    def test_validate_true_for_obs_only(self, tmp_path: Any) -> None:
        assert PlottingStage().validate(_obs_ctx(tmp_path)) is True

    def test_execute_creates_obs_plots(self, tmp_path: Any) -> None:
        ctx = _obs_ctx(tmp_path)
        res = PlottingStage().execute(ctx)
        assert res.status == StageStatus.COMPLETED
        pngs = list((tmp_path / "out").glob("*.png"))
        assert any("o3_hist" in p.name for p in pngs)


class TestSaveResultsDescriptive:
    def test_descriptive_writes_separate_csv_not_summary(self, tmp_path: Any) -> None:
        ctx = _obs_ctx(tmp_path)
        ctx.results["statistics"] = StatisticsStage().execute(ctx)
        SaveResultsStage().execute(ctx)
        out = tmp_path / "out"
        # Descriptive stats go to their own file; the comparison summary is NOT written.
        assert (out / "statistics_descriptive.csv").exists()
        assert not (out / "statistics_summary.csv").exists()
        df = pd.read_csv(out / "statistics_descriptive.csv")
        assert {"mean", "median", "p90"}.issubset(df.columns)
        assert {"O3", "CO"}.issubset(set(df["Variable"]))


class TestUnifiedPipelineComposition:
    def test_obs_stages_dropped_from_standard_pipeline(self) -> None:
        names = [s.name for s in create_standard_pipeline()]
        assert "obs_statistics" not in names
        assert "obs_plotting" not in names
        assert "statistics" in names
        assert "plotting" in names
