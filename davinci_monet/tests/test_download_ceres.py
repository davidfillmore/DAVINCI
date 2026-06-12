"""Offline unit tests for the CERES staging helper."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from davinci_monet.io.download import ceres, earthdata


@pytest.mark.parametrize(
    "collection, short_name, version, subpath",
    [
        ("ssf_terra-fm1", "CER_SSF_Terra-FM1-MODIS", "Edition4A", "CERES/SSF/Terra-FM1"),
        ("ssf_aqua-fm3", "CER_SSF_Aqua-FM3-MODIS", "Edition4A", "CERES/SSF/Aqua-FM3"),
        ("ssf_npp-fm5", "CER_SSF_NPP-FM5-VIIRS", "Edition2A", "CERES/SSF/NPP-FM5"),
        ("ssf_noaa20-fm6", "CER_SSF_NOAA20-FM6-VIIRS", "Edition1C", "CERES/SSF/NOAA20-FM6"),
        ("ebaf", "CERES_EBAF", "Edition4.2.1", "CERES/EBAF"),
        (
            "syn1deg_month",
            "CER_SYN1deg-Month_Terra-Aqua-NOAA20",
            "Edition4B",
            "CERES/SYN1deg/month",
        ),
        ("syn1deg_day", "CER_SYN1deg-Day_Terra-Aqua-NOAA20", "Edition4B", "CERES/SYN1deg/day"),
        ("syn1deg_hour", "CER_SYN1deg-1Hour_Terra-Aqua-NOAA20", "Edition4B", "CERES/SYN1deg/hour"),
    ],
)
def test_all_collections_resolve(
    collection: str, short_name: str, version: str, subpath: str
) -> None:
    spec = ceres.resolve_collection(collection)
    assert spec.short_name == short_name
    assert spec.version == version
    assert spec.subpath == Path(subpath)


def test_collection_table_is_exactly_the_documented_set() -> None:
    assert set(ceres.CERES_COLLECTIONS) == {
        "ssf_terra-fm1",
        "ssf_aqua-fm3",
        "ssf_npp-fm5",
        "ssf_noaa20-fm6",
        "ebaf",
        "syn1deg_month",
        "syn1deg_day",
        "syn1deg_hour",
    }


def test_every_collection_is_under_ceres_root() -> None:
    for spec in ceres.CERES_COLLECTIONS.values():
        assert spec.subpath.parts[0] == "CERES"
        assert spec.version is not None  # editions share short-names in CMR


def test_unknown_collection_raises_with_helpful_message() -> None:
    with pytest.raises(KeyError) as exc:
        ceres.resolve_collection("not_a_collection")
    assert "ebaf" in str(exc.value)


def test_destdir_joins_root_and_subpath() -> None:
    dest = ceres.dest_dir("ebaf", root="/Volumes/Io")
    assert dest == Path("/Volumes/Io/CERES/EBAF")


class _FakeGranule:
    """Mimics earthaccess.DataGranule's size() -> MB accessor."""

    def __init__(self, mb: float) -> None:
        self._mb = mb

    def size(self) -> float:
        return self._mb


def test_stage_ceres_dry_run_reports_count_and_size(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: dict[str, Any] = {}

    monkeypatch.setattr(earthdata, "_login", lambda: calls.setdefault("login", True))

    def _fake_search(
        short_name: str,
        temporal: tuple[str, str] | None = None,
        version: str | None = None,
    ) -> list[_FakeGranule]:
        calls["search"] = (short_name, temporal, version)
        return [_FakeGranule(60.0), _FakeGranule(70.0)]

    monkeypatch.setattr(earthdata, "_search", _fake_search)
    monkeypatch.setattr(
        earthdata, "_download", lambda *a: pytest.fail("download called in dry run")
    )

    report = ceres.stage_ceres(
        "ssf_aqua-fm3", "2023-07-01", "2023-07-02", root=tmp_path, dry_run=True
    )

    assert calls["login"] is True
    assert calls["search"] == (
        "CER_SSF_Aqua-FM3-MODIS",
        ("2023-07-01", "2023-07-02"),
        "Edition4A",
    )
    assert isinstance(report, ceres.DryRunReport)
    assert report.granules == 2
    assert report.total_mb == pytest.approx(130.0)


def test_stage_ceres_ebaf_searches_without_temporal(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(earthdata, "_login", lambda: None)
    seen: dict[str, Any] = {}

    def _fake_search(
        short_name: str,
        temporal: tuple[str, str] | None = None,
        version: str | None = None,
    ) -> list[_FakeGranule]:
        seen["args"] = (short_name, temporal, version)
        return [_FakeGranule(2000.0)]

    monkeypatch.setattr(earthdata, "_search", _fake_search)

    report = ceres.stage_ceres("ebaf", root=tmp_path, dry_run=True)

    assert seen["args"] == ("CERES_EBAF", None, "Edition4.2.1")
    assert isinstance(report, ceres.DryRunReport)
    assert report.granules == 1


def test_stage_ceres_requires_temporal_for_non_ebaf(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(earthdata, "_login", lambda: pytest.fail("network touched"))

    with pytest.raises(ValueError, match="start and end are required"):
        ceres.stage_ceres("ssf_aqua-fm3", root=tmp_path, dry_run=True)


def test_stage_ceres_rejects_half_open_temporal(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(earthdata, "_login", lambda: pytest.fail("network touched"))

    with pytest.raises(ValueError, match="both start and end"):
        ceres.stage_ceres("ebaf", start="2023-07-01", root=tmp_path, dry_run=True)


def test_stage_ceres_downloads_into_dest_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(earthdata, "_login", lambda: None)
    monkeypatch.setattr(
        earthdata,
        "_search",
        lambda short_name, temporal=None, version=None: ["g1"],
    )

    seen: dict[str, Any] = {}

    def _fake_download(results: Any, dest: str) -> list[Path]:
        seen["results"] = list(results)
        seen["dest"] = dest
        return [Path(dest) / "CER_SSF_Aqua-FM3-MODIS_Edition4A_407405.2023070100.hdf"]

    monkeypatch.setattr(earthdata, "_download", _fake_download)

    out = ceres.stage_ceres("ssf_aqua-fm3", "2023-07-01", "2023-07-01", root=tmp_path)

    expected_dir = tmp_path / "CERES/SSF/Aqua-FM3"
    assert expected_dir.is_dir()  # created before download
    assert seen["results"] == ["g1"]
    assert Path(seen["dest"]) == expected_dir
    assert isinstance(out, list)
    assert out[0].name == "CER_SSF_Aqua-FM3-MODIS_Edition4A_407405.2023070100.hdf"
