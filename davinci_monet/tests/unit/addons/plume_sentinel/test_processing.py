"""Tests for PlumeSentinel geospatial processing."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from davinci_monet.addons.plume_sentinel.processing import prepare_goes, prepare_modis_aod


class TestPrepareGoes:
    def _make_goes_dataset(self, shape=(10, 10)):
        h, w = shape
        return xr.Dataset(
            {
                "CMI_C01": (["y", "x"], np.full((h, w), 0.3, dtype=np.float32)),
                "CMI_C02": (["y", "x"], np.full((h, w), 0.5, dtype=np.float32)),
                "CMI_C03": (["y", "x"], np.full((h, w), 0.4, dtype=np.float32)),
                "goes_imager_projection": (
                    [],
                    0,
                    {
                        "perspective_point_height": 35786023.0,
                        "longitude_of_projection_origin": -75.0,
                        "sweep_angle_axis": "x",
                        "semi_major_axis": 6378137.0,
                        "semi_minor_axis": 6356752.31414,
                    },
                ),
            },
            coords={
                "x": np.linspace(-0.1, 0.1, w),
                "y": np.linspace(-0.1, 0.1, h),
            },
        )

    def test_rgb_shape(self):
        ds = self._make_goes_dataset(shape=(10, 10))
        result = prepare_goes(ds, gamma=1.8)
        assert result.rgb.shape == (10, 10, 3)

    def test_rgb_clipped_to_unit(self):
        ds = self._make_goes_dataset()
        result = prepare_goes(ds, gamma=1.8)
        assert result.rgb.min() >= 0.0
        assert result.rgb.max() <= 1.0

    def test_synthetic_green_formula(self):
        """Green = 0.45*red + 0.10*blue + 0.45*veggie."""
        ds = self._make_goes_dataset()
        result = prepare_goes(ds, gamma=1.0)  # no gamma
        red, blue, veggie = 0.5, 0.3, 0.4
        expected_green = 0.45 * red + 0.10 * blue + 0.45 * veggie
        np.testing.assert_allclose(result.rgb[0, 0, 1], expected_green, atol=1e-5)

    def test_extent_from_projection(self):
        ds = self._make_goes_dataset()
        result = prepare_goes(ds, gamma=1.8)
        h = 35786023.0
        expected_xmin = -0.1 * h
        expected_xmax = 0.1 * h
        assert result.extent[0] == pytest.approx(expected_xmin, rel=1e-3)
        assert result.extent[1] == pytest.approx(expected_xmax, rel=1e-3)

    def test_cartopy_crs_is_geostationary(self):
        import cartopy.crs as ccrs

        ds = self._make_goes_dataset()
        result = prepare_goes(ds, gamma=1.8)
        assert isinstance(result.cartopy_crs, ccrs.Geostationary)


class TestPrepareModisAod:
    def test_bins_synthetic_swath(self):
        rng = np.random.default_rng(42)
        n = 1000
        loaded = {
            "latitude": rng.uniform(30, 50, n),
            "longitude": rng.uniform(-130, -110, n),
            "data": rng.uniform(0, 2, n),
            "granule_files": ["test.hdf"],
        }
        grid_spec = {
            "resolution": 1.0,
            "lon_range": [-180, 180],
            "lat_range": [-90, 90],
            "min_obs_count": 1,
        }
        result = prepare_modis_aod(loaded, grid_spec)
        assert result.data_2d.ndim == 2
        assert len(result.lon_centers) > 0
        assert len(result.lat_centers) > 0
        assert np.isfinite(result.data_2d).sum() > 0

    def test_empty_swath_gives_all_nan(self):
        loaded = {
            "latitude": np.array([]),
            "longitude": np.array([]),
            "data": np.array([]),
            "granule_files": [],
        }
        grid_spec = {
            "resolution": 1.0,
            "lon_range": [-180, 180],
            "lat_range": [-90, 90],
            "min_obs_count": 1,
        }
        result = prepare_modis_aod(loaded, grid_spec)
        assert np.all(np.isnan(result.data_2d))
