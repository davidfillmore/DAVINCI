"""Tests that the --demo-mode and --demo-bulletin flags populate config correctly."""

from __future__ import annotations

import pytest

from davinci_monet.cli.commands.run import _apply_demo_flags


def test_apply_demo_flags_default_off():
    cfg = {"analysis": {"output_dir": "out"}}
    _apply_demo_flags(cfg, demo_mode=False, demo_bulletin=None)
    assert "_demo" not in cfg["analysis"]


def test_apply_demo_flags_enabled_no_canned():
    cfg = {"analysis": {"output_dir": "out"}}
    _apply_demo_flags(cfg, demo_mode=True, demo_bulletin=None)
    assert cfg["analysis"]["_demo"] == {"enabled": True, "canned_bulletin": None}


def test_apply_demo_flags_enabled_with_canned():
    cfg = {"analysis": {"output_dir": "out"}}
    _apply_demo_flags(cfg, demo_mode=True, demo_bulletin="/tmp/saved.txt")
    assert cfg["analysis"]["_demo"] == {
        "enabled": True,
        "canned_bulletin": "/tmp/saved.txt",
    }


def test_apply_demo_flags_bulletin_without_demo_mode_raises():
    cfg = {"analysis": {"output_dir": "out"}}
    with pytest.raises(ValueError, match="--demo-bulletin requires --demo-mode"):
        _apply_demo_flags(cfg, demo_mode=False, demo_bulletin="/tmp/x.txt")


def test_apply_demo_flags_initialises_analysis_block():
    cfg: dict = {}
    _apply_demo_flags(cfg, demo_mode=True, demo_bulletin=None)
    assert "analysis" in cfg
    assert cfg["analysis"]["_demo"] == {"enabled": True, "canned_bulletin": None}
