"""Offline unit tests for the shared Earthdata staging core."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from davinci_monet.io.download import earthdata


class _FakeGranule:
    """Mimics earthaccess.DataGranule's size() -> MB accessor."""

    def __init__(self, mb: float) -> None:
        self._mb = mb

    def size(self) -> float:
        return self._mb


def test_collection_spec_version_defaults_to_none() -> None:
    spec = earthdata.CollectionSpec("M2TMNXAER", Path("MERRA2_tavgM/aer_Nx"))
    assert spec.version is None


def test_granule_size_mb_uses_size_method() -> None:
    assert earthdata.granule_size_mb(_FakeGranule(42.5)) == 42.5


def test_granule_size_mb_zero_when_unavailable() -> None:
    assert earthdata.granule_size_mb("just-a-string") == 0.0


class _FakeUmmGranule(dict):
    """Dict-like granule carrying UMM archive metadata, like DataGranule."""

    def __init__(self, infos: list[dict[str, Any]]) -> None:
        super().__init__({"umm": {"DataGranule": {"ArchiveAndDistributionInformation": infos}}})

    def size(self) -> float:  # pragma: no cover - must not be reached
        raise AssertionError("size() fallback used despite UMM metadata")


def test_granule_size_mb_converts_umm_units() -> None:
    granule = _FakeUmmGranule(
        [
            {"Size": 1.9072410818189383, "SizeUnit": "GB"},
            {"Size": 16.345703125, "SizeUnit": "KB"},
        ]
    )
    assert earthdata.granule_size_mb(granule) == pytest.approx(1953.0, abs=0.1)


def test_granule_size_mb_skips_unknown_units_and_falls_back() -> None:
    class _UnknownUnitGranule(dict):
        def __init__(self) -> None:
            super().__init__(
                {
                    "umm": {
                        "DataGranule": {
                            "ArchiveAndDistributionInformation": [{"Size": 5.0, "SizeUnit": "NA"}]
                        }
                    }
                }
            )

        def size(self) -> float:
            return 7.5

    assert earthdata.granule_size_mb(_UnknownUnitGranule()) == 7.5


def test_stage_collection_dry_run_returns_results_without_download(tmp_path: Path) -> None:
    spec = earthdata.CollectionSpec("X", Path("X/sub"), version="V1")
    calls: dict[str, Any] = {}

    results = earthdata.stage_collection(
        spec,
        ("2023-01", "2023-02"),
        tmp_path / "X/sub",
        dry_run=True,
        login=lambda: calls.setdefault("login", True),
        search=lambda s, t: ["g1", "g2"],
        download=lambda r, d: pytest.fail("download called in dry run"),
    )

    assert calls["login"] is True
    assert list(results) == ["g1", "g2"]
    assert not (tmp_path / "X/sub").exists()  # dry run creates nothing


def test_stage_collection_creates_dest_and_downloads(tmp_path: Path) -> None:
    spec = earthdata.CollectionSpec("X", Path("X/sub"))
    dest = tmp_path / "X/sub"
    seen: dict[str, Any] = {}

    def _fake_download(results: Any, d: str) -> list[Path]:
        seen["results"] = list(results)
        seen["dest"] = d
        return [Path(d) / "f.nc"]

    out = earthdata.stage_collection(
        spec,
        None,
        dest,
        login=lambda: None,
        search=lambda s, t: ["g1"],
        download=_fake_download,
    )

    assert dest.is_dir()  # created before download
    assert seen["results"] == ["g1"]
    assert Path(seen["dest"]) == dest
    assert out == [dest / "f.nc"]


def test_stage_collection_default_search_passes_spec_fields(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Default hooks resolve earthdata module functions at call time."""
    spec = earthdata.CollectionSpec("CERES_EBAF", Path("CERES/EBAF"), version="Edition4.2.1")
    calls: dict[str, Any] = {}

    monkeypatch.setattr(earthdata, "_login", lambda: calls.setdefault("login", True))

    def _fake_search(
        short_name: str,
        temporal: tuple[str, str] | None = None,
        version: str | None = None,
    ) -> list[str]:
        calls["search"] = (short_name, temporal, version)
        return ["g1"]

    monkeypatch.setattr(earthdata, "_search", _fake_search)

    results = earthdata.stage_collection(spec, None, tmp_path / "CERES/EBAF", dry_run=True)

    assert calls["login"] is True
    assert calls["search"] == ("CERES_EBAF", None, "Edition4.2.1")
    assert list(results) == ["g1"]
