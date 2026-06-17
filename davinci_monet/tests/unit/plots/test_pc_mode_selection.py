"""A plot-time mode: selector picks one PC for the timeseries renderer."""

from __future__ import annotations

import numpy as np
import xarray as xr

from davinci_monet.config.schema import PlotGroupConfig
from davinci_monet.pipeline.stages.plot_options import single_source_plot_kwargs


def test_plotgroup_accepts_mode_and_display_level() -> None:
    cfg = PlotGroupConfig(type="timeseries", source="cam_O3_eof", variable="pc", mode=1)
    assert cfg.mode == 1
    cfg2 = PlotGroupConfig(
        type="eof_pattern", source="cam_O3_eof", variable="eofs", display_level=-1
    )
    assert cfg2.display_level == -1


def test_display_level_forwarded_to_render_kwargs() -> None:
    spec = {"type": "eof_pattern", "source": "s", "variable": "eofs", "display_level": -1}
    kwargs = single_source_plot_kwargs(spec, analysis_config=None)
    assert kwargs.get("display_level") == -1


def test_mode_selection_picks_single_pc() -> None:
    pc = xr.Dataset(
        {"pc": (("time", "mode"), np.array([[1.0, 9.0], [2.0, 9.0], [3.0, 9.0]]))},
        coords={"time": np.arange(3), "mode": [1, 2]},
    )
    selected = pc.sel(mode=1)["pc"].values
    assert list(selected) == [1.0, 2.0, 3.0]
