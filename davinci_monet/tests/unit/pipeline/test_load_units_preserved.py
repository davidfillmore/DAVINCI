"""Reader/file-provided units must survive the load stage's variable processing.

Regression: ``VariableConfig.unit_scale`` defaults to 1.0, so the scaling block
runs for every variable; xarray arithmetic drops attrs, which silently stripped
reader-provided units (e.g. ICARTT ``ppbv`` / altitude ``m``). A no-op scale must
preserve the original units; an explicit config ``units`` always wins; a real
conversion must not keep the now-stale original unit.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from davinci_monet.pipeline.stages.load import LoadSourcesStage


def _ds_with_units(var: str, units: str) -> xr.Dataset:
    ds = xr.Dataset({var: ("time", np.arange(3.0))})
    ds[var].attrs["units"] = units
    return ds


def test_noop_scale_preserves_reader_units() -> None:
    # Default unit_scale=1.0 (*) is a no-op: the reader's "ppbv" must survive.
    ds = _ds_with_units("O3_raw", "ppbv")
    variables = {"O3": {"source_name": "O3_raw", "unit_scale": 1.0, "unit_scale_method": "*"}}
    out = LoadSourcesStage._apply_variable_config(ds, variables)
    assert out["O3"].attrs.get("units") == "ppbv"


def test_no_scale_key_preserves_reader_units() -> None:
    ds = _ds_with_units("x", "m")
    out = LoadSourcesStage._apply_variable_config(ds, {"x": {}})
    assert out["x"].attrs.get("units") == "m"


def test_explicit_config_units_win() -> None:
    ds = _ds_with_units("x", "ppbv")
    out = LoadSourcesStage._apply_variable_config(ds, {"x": {"unit_scale": 1.0, "units": "ppb"}})
    assert out["x"].attrs.get("units") == "ppb"


def test_real_scale_drops_stale_units() -> None:
    # mol/mol -> ppb via *1e9: the stale "mol/mol" must NOT persist when no
    # explicit units are configured.
    ds = _ds_with_units("x", "mol/mol")
    out = LoadSourcesStage._apply_variable_config(
        ds, {"x": {"unit_scale": 1.0e9, "unit_scale_method": "*"}}
    )
    assert out["x"].attrs.get("units") != "mol/mol"
