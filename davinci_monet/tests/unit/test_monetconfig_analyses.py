"""MonetConfig parses the analyses: block into typed specs."""

from __future__ import annotations

from davinci_monet.config.schema import EOFSpec, MonetConfig, WaveletSpec


def test_analyses_block_parsed() -> None:
    cfg = MonetConfig(
        sources={
            "cam": {"type": "generic", "files": "x.nc", "variables": {"O3": {"units": "ppb"}}}  # type: ignore[dict-item]
        },
        analyses={
            "cam_O3_eof": {"type": "eof", "source": "cam", "variable": "O3", "n_modes": 4},  # type: ignore[dict-item]
            "pc1_wav": {"type": "wavelet", "source": "cam_O3_eof", "variable": "pc", "mode": 1},  # type: ignore[dict-item]
        },
    )
    assert isinstance(cfg.analyses["cam_O3_eof"], EOFSpec)
    assert isinstance(cfg.analyses["pc1_wav"], WaveletSpec)
    assert cfg.analyses["cam_O3_eof"].n_modes == 4


def test_analyses_defaults_empty() -> None:
    cfg = MonetConfig()
    assert cfg.analyses == {}
