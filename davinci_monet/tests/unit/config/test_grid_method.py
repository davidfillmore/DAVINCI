import pytest
from pydantic import ValidationError

from davinci_monet.config.schema import SourcePairConfig


def test_method_grid_parses_with_grid_block():
    p = SourcePairConfig(
        x={"source": "a", "variable": "v"},  # type: ignore[arg-type]
        y={"source": "b", "variable": "V"},  # type: ignore[arg-type]
        method="grid",
        grid={"horizontal_res": 0.5, "time_resolution": "1D", "min_sample_count": 1},  # type: ignore[arg-type]
    )
    assert p.method == "grid"
    assert p.grid is not None and p.grid.horizontal_res == 0.5


def test_method_defaults_to_auto():
    p = SourcePairConfig(
        x={"source": "a", "variable": "v"},  # type: ignore[arg-type]
        y={"source": "b", "variable": "V"},  # type: ignore[arg-type]
    )
    assert p.method == "auto" and p.grid is None


def test_method_grid_requires_grid_block():
    with pytest.raises(ValidationError, match="grid"):
        SourcePairConfig(
            x={"source": "a", "variable": "v"},  # type: ignore[arg-type]
            y={"source": "b", "variable": "V"},  # type: ignore[arg-type]
            method="grid",
        )


def test_auto_with_grid_block_is_rejected():
    with pytest.raises(ValidationError, match="auto"):
        SourcePairConfig(
            x={"source": "a", "variable": "v"},  # type: ignore[arg-type]
            y={"source": "b", "variable": "V"},  # type: ignore[arg-type]
            method="auto",
            grid={"horizontal_res": 0.5},  # type: ignore[arg-type]
        )


def test_grid_vertical_block_parses():
    from davinci_monet.config.schema import SourcePairConfig

    p = SourcePairConfig(
        x={"source": "a", "variable": "v"},  # type: ignore[arg-type]
        y={"source": "b", "variable": "V"},  # type: ignore[arg-type]
        method="grid",
        grid={"horizontal_res": 0.5, "vertical": {"res": 500, "units": "m", "extent": [0, 12000]}},  # type: ignore[arg-type]
    )
    assert p.grid is not None and p.grid.vertical is not None
    assert p.grid.vertical.res == 500.0 and p.grid.vertical.units == "m"
    assert p.grid.vertical.extent == (0.0, 12000.0)


def test_grid_vertical_defaults_units_m_and_optional():
    from davinci_monet.config.schema import SourcePairConfig

    p = SourcePairConfig(
        x={"source": "a", "variable": "v"},  # type: ignore[arg-type]
        y={"source": "b", "variable": "V"},  # type: ignore[arg-type]
        method="grid",
        grid={"horizontal_res": 0.5},  # type: ignore[arg-type]
    )
    assert p.grid is not None and p.grid.vertical is None  # 2-D when omitted
    p2 = SourcePairConfig(
        x={"source": "a", "variable": "v"},  # type: ignore[arg-type]
        y={"source": "b", "variable": "V"},  # type: ignore[arg-type]
        method="grid",
        grid={"horizontal_res": 0.5, "vertical": {"res": 1.0}},  # type: ignore[arg-type]
    )
    assert p2.grid is not None and p2.grid.vertical is not None
    assert p2.grid.vertical.units == "m"
