"""Cross-reference and dependency rules for the analyses: block."""

from __future__ import annotations

import pytest

from davinci_monet.config.schema import MonetConfig

_SOURCES = {"cam": {"type": "generic", "files": "x.nc", "variables": {"O3": {"units": "ppb"}}}}


def test_analysis_unknown_source_rejected() -> None:
    with pytest.raises(ValueError, match="references unknown source"):
        MonetConfig(
            sources=_SOURCES,  # type: ignore[arg-type]
            analyses={"a": {"type": "eof", "source": "nope", "variable": "O3"}},  # type: ignore[dict-item]
        )


def test_analysis_cycle_rejected() -> None:
    with pytest.raises(ValueError, match="cycle"):
        MonetConfig(
            sources=_SOURCES,  # type: ignore[arg-type]
            analyses={
                "a": {"type": "wavelet", "source": "b", "variable": "pc"},  # type: ignore[dict-item]
                "b": {"type": "wavelet", "source": "a", "variable": "pc"},  # type: ignore[dict-item]
            },
        )


def test_analysis_key_collides_with_source_rejected() -> None:
    with pytest.raises(ValueError, match="collides"):
        MonetConfig(
            sources=_SOURCES,  # type: ignore[arg-type]
            analyses={"cam": {"type": "eof", "source": "cam", "variable": "O3"}},  # type: ignore[dict-item]
        )


def test_pair_referencing_derived_source_rejected() -> None:
    with pytest.raises(ValueError, match="derived sources are not pairable") as excinfo:
        MonetConfig(
            sources=_SOURCES,  # type: ignore[arg-type]
            analyses={"cam_eof": {"type": "eof", "source": "cam", "variable": "O3"}},  # type: ignore[dict-item]
            pairs={
                "p": {
                    "x": {"source": "cam", "variable": "O3"},  # type: ignore[dict-item]
                    "y": {"source": "cam_eof", "variable": "O3"},
                }
            },
        )
    # The misleading duplicate "references unknown source" message must NOT fire
    # for this pair — only the specific not-pairable message.
    assert "references unknown source" not in str(excinfo.value)


@pytest.mark.skip(reason="eof_pattern registered in Plan B")
def test_plot_may_reference_derived_source() -> None:
    cfg = MonetConfig(
        sources=_SOURCES,  # type: ignore[arg-type]
        analyses={"cam_O3_eof": {"type": "eof", "source": "cam", "variable": "O3"}},  # type: ignore[dict-item]
        plots={"m": {"type": "eof_pattern", "source": "cam_O3_eof", "variable": "mode"}},  # type: ignore[dict-item]
    )
    assert "m" in cfg.plots
