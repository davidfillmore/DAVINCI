"""Tests for synthetic dataset generators."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from davinci_monet.tests.synthetic.datasets import (
    create_gridded_geometries,
    create_point_geometries,
    create_profile_geometries,
    create_swath_geometries,
    create_track_geometries,
)
from davinci_monet.tests.synthetic.generators import Domain, TimeConfig


class TestPointGeometries:
    """Tests for point dataset generator."""

    def test_basic_creation(self) -> None:
        """Test basic point datasets creation."""
        ds = create_point_geometries(n_sites=10)

        assert isinstance(ds, xr.Dataset)
        assert "time" in ds.dims
        assert "site" in ds.dims
        assert len(ds.site) == 10

    def test_variables_present(self) -> None:
        """Test requested variables are present."""
        ds = create_point_geometries(variables=["O3", "PM25", "NO2"])

        assert "O3" in ds
        assert "PM25" in ds
        assert "NO2" in ds

    def test_coordinates_present(self) -> None:
        """Test location coordinates are present."""
        ds = create_point_geometries(n_sites=5)

        assert "latitude" in ds.coords
        assert "longitude" in ds.coords
        assert "site_id" in ds.coords

    def test_coordinates_in_domain(self) -> None:
        """Test site locations are within domain."""
        domain = Domain(lon_min=-100, lon_max=-90, lat_min=30, lat_max=40)
        ds = create_point_geometries(n_sites=20, domain=domain)

        lons = ds.longitude.values
        lats = ds.latitude.values

        assert np.all(lons >= domain.lon_min)
        assert np.all(lons <= domain.lon_max)
        assert np.all(lats >= domain.lat_min)
        assert np.all(lats <= domain.lat_max)

    def test_geometry_attribute(self) -> None:
        """Test geometry attribute is set."""
        ds = create_point_geometries()
        assert ds.attrs.get("geometry") == "point"

    def test_site_prefix(self) -> None:
        """Test custom site ID prefix."""
        ds = create_point_geometries(n_sites=3, site_prefix="AQS")
        site_ids = ds.site_id.values

        assert all(str(sid).startswith("AQS") for sid in site_ids)

    def test_reproducibility(self) -> None:
        """Test reproducibility with seed."""
        ds1 = create_point_geometries(n_sites=5, seed=42)
        ds2 = create_point_geometries(n_sites=5, seed=42)

        xr.testing.assert_equal(ds1, ds2)


class TestTrackGeometries:
    """Tests for track dataset generator."""

    def test_basic_creation(self) -> None:
        """Test basic track datasets creation."""
        ds = create_track_geometries(n_points=100)

        assert isinstance(ds, xr.Dataset)
        assert "time" in ds.dims
        assert len(ds.time) == 100

    def test_coordinates_present(self) -> None:
        """Test 3D coordinates are present."""
        ds = create_track_geometries()

        assert "latitude" in ds.coords
        assert "longitude" in ds.coords
        assert "altitude" in ds.coords

    def test_altitude_profile(self) -> None:
        """Test altitude has realistic profile."""
        ds = create_track_geometries(n_points=100, altitude_range=(1000.0, 8000.0))

        alt = ds.altitude.values

        # Should have takeoff and landing (lower values at ends)
        assert alt[0] < alt[len(alt) // 2]  # Start lower than middle
        assert alt[-1] < alt[len(alt) // 2]  # End lower than middle

    def test_geometry_attribute(self) -> None:
        """Test geometry attribute is set."""
        ds = create_track_geometries()
        assert ds.attrs.get("geometry") == "track"

    def test_track_stays_in_domain(self) -> None:
        """Test track path stays mostly within domain."""
        domain = Domain(lon_min=-100, lon_max=-90, lat_min=30, lat_max=40)
        ds = create_track_geometries(n_points=50, domain=domain)

        # Most points should be within domain (some noise may push outside)
        lons = ds.longitude.values
        lats = ds.latitude.values

        in_lon = (lons >= domain.lon_min - 1) & (lons <= domain.lon_max + 1)
        in_lat = (lats >= domain.lat_min - 1) & (lats <= domain.lat_max + 1)

        assert np.mean(in_lon & in_lat) > 0.8  # At least 80% inside


class TestProfileGeometries:
    """Tests for profile dataset generator."""

    def test_basic_creation(self) -> None:
        """Test basic profile datasets creation."""
        ds = create_profile_geometries(n_profiles=5, n_levels=30)

        assert isinstance(ds, xr.Dataset)
        assert "time" in ds.dims
        assert "level" in ds.dims
        assert len(ds.time) == 5
        assert len(ds.level) == 30

    def test_coordinates_present(self) -> None:
        """Test profile coordinates are present."""
        ds = create_profile_geometries()

        assert "latitude" in ds.coords
        assert "longitude" in ds.coords

    def test_geometry_attribute(self) -> None:
        """Test geometry attribute is set."""
        ds = create_profile_geometries()
        assert ds.attrs.get("geometry") == "profile"

    def test_vertical_structure(self) -> None:
        """Test O3 has vertical structure."""
        ds = create_profile_geometries(variables=["O3"], n_profiles=1, n_levels=20)

        # O3 should generally increase with altitude (decreasing pressure)
        o3_profile = ds["O3"].isel(time=0).values

        # Compare lower levels (high pressure) to upper levels (low pressure)
        lower_mean = o3_profile[:5].mean()
        upper_mean = o3_profile[-5:].mean()

        assert upper_mean > lower_mean  # Stratospheric enhancement


class TestSwathGeometries:
    """Tests for swath dataset generator."""

    def test_basic_creation(self) -> None:
        """Test basic swath datasets creation."""
        ds = create_swath_geometries(n_scans=50, n_pixels=30)

        assert isinstance(ds, xr.Dataset)
        assert "scanline" in ds.dims
        assert "pixel" in ds.dims
        assert len(ds.scanline) == 50
        assert len(ds.pixel) == 30

    def test_2d_coordinates(self) -> None:
        """Test 2D lat/lon coordinates."""
        ds = create_swath_geometries(n_scans=20, n_pixels=15)

        assert ds.latitude.dims == ("scanline", "pixel")
        assert ds.longitude.dims == ("scanline", "pixel")

    def test_quality_flag_present(self) -> None:
        """Test quality flag is present."""
        ds = create_swath_geometries()

        assert "qa_flag" in ds
        # Should have mostly good quality
        good_fraction = (ds.qa_flag == 0).mean()
        assert float(good_fraction) > 0.7

    def test_geometry_attribute(self) -> None:
        """Test geometry attribute is set."""
        ds = create_swath_geometries()
        assert ds.attrs.get("geometry") == "swath"


class TestGriddedGeometries:
    """Tests for gridded dataset generator."""

    def test_basic_creation(self) -> None:
        """Test basic gridded datasets creation."""
        ds = create_gridded_geometries()

        assert isinstance(ds, xr.Dataset)
        assert "time" in ds.dims
        assert "lat" in ds.dims
        assert "lon" in ds.dims

    def test_regular_grid(self) -> None:
        """Test data is on regular grid."""
        ds = create_gridded_geometries()

        # Lon/lat should be 1D coordinates
        assert ds.lon.ndim == 1
        assert ds.lat.ndim == 1

    def test_geometry_attribute(self) -> None:
        """Test geometry attribute is set."""
        ds = create_gridded_geometries()
        assert ds.attrs.get("geometry") == "grid"


class TestGeometryDataQuality:
    """Tests for dataset data quality."""

    def test_point_values_reasonable(self) -> None:
        """Test point dataset values are reasonable."""
        ds = create_point_geometries(variables=["O3", "PM25"])

        assert float(ds["O3"].min()) >= 0
        assert float(ds["O3"].max()) < 300
        assert float(ds["PM25"].min()) >= 0

    def test_track_values_reasonable(self) -> None:
        """Test track dataset values are reasonable."""
        ds = create_track_geometries(variables=["O3", "CO"])

        assert float(ds["O3"].min()) >= 0
        assert float(ds["CO"].min()) >= 0

    def test_no_nans_in_data(self) -> None:
        """Test generated data has no NaN values."""
        point = create_point_geometries()
        track = create_track_geometries()
        profile = create_profile_geometries()

        assert not np.any(np.isnan(point["O3"].values))
        assert not np.any(np.isnan(track["O3"].values))
        assert not np.any(np.isnan(profile["O3"].values))
