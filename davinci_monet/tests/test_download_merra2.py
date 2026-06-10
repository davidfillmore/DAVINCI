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


def test_stage_merra2_dry_run_searches_but_does_not_download(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: dict[str, Any] = {}

    monkeypatch.setattr(merra2, "_login", lambda: calls.setdefault("login", True))

    def _fake_search(short_name: str, temporal: tuple[str, str]) -> list[str]:
        calls["search"] = (short_name, temporal)
        return ["granule-1", "granule-2"]

    def _fake_download(results: Any, dest: str) -> list[Path]:  # pragma: no cover
        calls["download"] = True
        return []

    monkeypatch.setattr(merra2, "_search", _fake_search)
    monkeypatch.setattr(merra2, "_download", _fake_download)

    planned = merra2.stage_merra2(
        "tavgM_2d_aer_Nx", "2003-01", "2003-03", root=tmp_path, dry_run=True
    )

    assert calls["login"] is True
    assert calls["search"] == ("M2TMNXAER", ("2003-01", "2003-03"))
    assert "download" not in calls
    assert planned == 2  # number of granules found


def test_stage_merra2_downloads_into_dest_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(merra2, "_login", lambda: None)
    monkeypatch.setattr(merra2, "_search", lambda short_name, temporal: ["g1"])

    seen: dict[str, Any] = {}

    def _fake_download(results: Any, dest: str) -> list[Path]:
        seen["results"] = results
        seen["dest"] = dest
        return [Path(dest) / "MERRA2_400.tavgM_2d_aer_Nx.200301.nc4"]

    monkeypatch.setattr(merra2, "_download", _fake_download)

    out = merra2.stage_merra2("tavgM_2d_aer_Nx", "2003-01", "2003-01", root=tmp_path)

    expected_dir = tmp_path / "MERRA2_tavgM/aer_Nx"
    assert expected_dir.is_dir()  # created before download
    assert seen["results"] == ["g1"]
    assert Path(seen["dest"]) == expected_dir
    assert isinstance(out, list)
    assert out[0].name == "MERRA2_400.tavgM_2d_aer_Nx.200301.nc4"


def test_main_dry_run_invokes_stage(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    captured: dict[str, Any] = {}

    def _fake_stage(
        collection: str, start: str, end: str, *, root: Any, dry_run: bool
    ) -> int:
        captured.update(
            collection=collection, start=start, end=end, root=root, dry_run=dry_run
        )
        return 7

    monkeypatch.setattr(merra2, "stage_merra2", _fake_stage)

    rc = merra2.main(
        [
            "--collection", "tavgM_2d_aer_Nx",
            "--start", "2003-01",
            "--end", "2003-03",
            "--root", str(tmp_path),
            "--dry-run",
        ]
    )

    assert rc == 0
    assert captured["collection"] == "tavgM_2d_aer_Nx"
    assert captured["dry_run"] is True
    assert captured["root"] == str(tmp_path)
    assert "7" in capsys.readouterr().out
