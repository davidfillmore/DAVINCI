"""Offline unit tests for the MERRA-2 staging helper."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from davinci_monet.io.download import merra2


def test_known_collection_resolves_shortname_and_destdir() -> None:
    spec = merra2.resolve_collection("tavgM_2d_aer_Nx")
    assert spec.short_name == "M2TMNXAER"
    assert spec.subpath == Path("MERRA2_tavgM/aer_Nx")


def test_destdir_joins_root_and_subpath() -> None:
    dest = merra2.dest_dir("tavgM_2d_aer_Nx", root="/Volumes/Io")
    assert dest == Path("/Volumes/Io/MERRA2_tavgM/aer_Nx")


def test_unknown_collection_raises_with_helpful_message() -> None:
    with pytest.raises(KeyError) as exc:
        merra2.resolve_collection("not_a_collection")
    assert "tavgM_2d_aer_Nx" in str(exc.value)
