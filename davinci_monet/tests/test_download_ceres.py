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
