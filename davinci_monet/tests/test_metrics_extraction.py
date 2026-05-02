"""Tests for plume_sentinel.metrics extraction from synthetic stage outputs."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from davinci_monet.addons.plume_sentinel import metrics


@pytest.fixture
def synthetic_aod():
    """A 100x100 AOD field with known peak=4.5, p50≈1.0, retrieval failures ~5%."""
    rng = np.random.default_rng(42)
    arr = rng.lognormal(mean=0, sigma=0.5, size=(100, 100)).clip(0, 4.5)
    arr[0:5, 0:10] = np.nan  # 0.5% nan; pad more
    arr.flat[:500] = np.nan  # 5% nan total = retrieval failures
    arr[10, 10] = 4.5  # ensure peak
    da = xr.DataArray(
        arr,
        dims=("y", "x"),
        coords={"y": np.linspace(40, 50, 100), "x": np.linspace(-130, -120, 100)},
    )
    return da


def test_aod_peak(synthetic_aod):
    m = metrics.aod_metrics(synthetic_aod)
    assert m["aod_peak"] == pytest.approx(4.5, abs=0.01)


def test_aod_percentiles(synthetic_aod):
    m = metrics.aod_metrics(synthetic_aod)
    # Lognormal sigma=0.5 → p50≈1.0
    assert 0.7 < m["aod_p50"] < 1.3
    assert 1.5 < m["aod_p95"] < 3.5


def test_aod_retrieval_failure_pct(synthetic_aod):
    m = metrics.aod_metrics(synthetic_aod)
    # 500 nan / 10000 = 5%
    assert m["retrieval_failure_pct"] == pytest.approx(5.0, abs=0.5)


def test_aod_area_gt_threshold(synthetic_aod):
    """Area of pixels with AOD > 2 in km²; depends on lat-lon grid spacing."""
    m = metrics.aod_metrics(synthetic_aod)
    assert m["area_aod_gt_2_km2"] > 0


def test_hms_density_class_areas():
    """HMS contours given as a GeoDataFrame; areas in km² per density class."""
    import geopandas as gpd
    from shapely.geometry import Polygon

    poly = Polygon([(-125, 40), (-120, 40), (-120, 45), (-125, 45)])
    gdf = gpd.GeoDataFrame(
        {"density": ["Heavy", "Medium", "Light"]},
        geometry=[poly, poly, poly],
        crs="EPSG:4326",
    )
    m = metrics.hms_metrics(gdf)
    assert m["hms_heavy_km2"] > 0
    assert m["hms_medium_km2"] > 0
    assert m["hms_light_km2"] > 0
