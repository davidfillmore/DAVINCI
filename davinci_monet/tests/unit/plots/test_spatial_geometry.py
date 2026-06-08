"""Unit tests for detect_spatial_geometry helper.

Covers the three canonical geometry cases that the spatial renderers
must handle:  point/site data, regular rectilinear grids, and
curvilinear (2-D coord) grids.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from davinci_monet.plots.renderers.spatial.base import detect_spatial_geometry


class TestDetectSpatialGeometry:
    """Tests for detect_spatial_geometry."""

    def test_point_geometry_1d_shared_dim(self) -> None:
        """1-D lat/lon sharing a single site dim that is in the field → 'point'."""
        n_sites = 8
        lat_da = xr.DataArray(
            np.linspace(20.0, 50.0, n_sites),
            dims=("site",),
        )
        lon_da = xr.DataArray(
            np.linspace(-120.0, -70.0, n_sites),
            dims=("site",),
        )
        # Field also has the site dim (e.g. already time-averaged)
        field_da = xr.DataArray(
            np.random.default_rng(0).uniform(0, 1, n_sites),
            dims=("site",),
        )

        result = detect_spatial_geometry(lat_da, lon_da, field_da)

        assert result == "point"

    def test_point_geometry_2d_field_same_site_dim(self) -> None:
        """1-D lat/lon sharing the site dim, with a (time, site) field → 'point'."""
        n_sites = 5
        n_times = 3
        lat_da = xr.DataArray(np.linspace(30.0, 50.0, n_sites), dims=("site",))
        lon_da = xr.DataArray(np.linspace(-110.0, -80.0, n_sites), dims=("site",))
        field_da = xr.DataArray(
            np.ones((n_times, n_sites)),
            dims=("time", "site"),
        )

        result = detect_spatial_geometry(lat_da, lon_da, field_da)

        assert result == "point"

    def test_regular_grid_1d_axes_2d_field(self) -> None:
        """1-D lat + 1-D lon as independent axes with a 2-D field → 'regular_grid'."""
        nlat, nlon = 10, 12
        lat_da = xr.DataArray(np.linspace(-89.5, 89.5, nlat), dims=("lat",))
        lon_da = xr.DataArray(np.linspace(-179.5, 179.5, nlon), dims=("lon",))
        field_da = xr.DataArray(
            np.random.default_rng(1).uniform(0, 1, (nlat, nlon)),
            dims=("lat", "lon"),
        )

        result = detect_spatial_geometry(lat_da, lon_da, field_da)

        assert result == "regular_grid"

    def test_curvilinear_grid_2d_lat_lon(self) -> None:
        """2-D lat/lon arrays → 'curvilinear_grid'."""
        nlat, nlon = 6, 8
        lat_2d = xr.DataArray(
            np.broadcast_to(np.linspace(30.0, 50.0, nlat)[:, None], (nlat, nlon)),
            dims=("y", "x"),
        )
        lon_2d = xr.DataArray(
            np.broadcast_to(np.linspace(-120.0, -80.0, nlon)[None, :], (nlat, nlon)),
            dims=("y", "x"),
        )
        field_da = xr.DataArray(
            np.ones((nlat, nlon)),
            dims=("y", "x"),
        )

        result = detect_spatial_geometry(lat_2d, lon_2d, field_da)

        assert result == "curvilinear_grid"

    def test_point_site_geometry_airnow_style(self) -> None:
        """AirNow-style data: lat/lon on x dim, field on (time, y=1, x) → 'point'.

        The key here is that lat_da.dims[0] == 'x' appears in field_da.dims
        even though field_da also has time and y dims.
        """
        n_sites = 5
        lat_da = xr.DataArray(np.linspace(30.0, 50.0, n_sites), dims=("x",))
        lon_da = xr.DataArray(np.linspace(-110.0, -70.0, n_sites), dims=("x",))
        field_da = xr.DataArray(
            np.ones((3, 1, n_sites)),
            dims=("time", "y", "x"),
        )

        result = detect_spatial_geometry(lat_da, lon_da, field_da)

        assert result == "point"
