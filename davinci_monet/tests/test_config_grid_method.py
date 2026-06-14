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
