"""Unit tests for the shared HDF4 scale/mask helper."""

from __future__ import annotations

import numpy as np

from davinci_monet.io.reader_utils import apply_hdf4_scale


def test_scale_and_offset_applied() -> None:
    raw = np.array([0, 10, 20], dtype=np.int16)
    out = apply_hdf4_scale(raw, {"scale_factor": 0.5, "add_offset": 1.0})
    np.testing.assert_allclose(out, [1.0, 6.0, 11.0])
    assert out.dtype == np.float64


def test_fill_value_masked_to_nan() -> None:
    fill = 3.4028234663852886e38
    raw = np.array([100.0, fill, 200.0], dtype=np.float32)
    out = apply_hdf4_scale(raw, {"_FillValue": fill})
    assert np.isnan(out[1])
    np.testing.assert_allclose(out[[0, 2]], [100.0, 200.0])


def test_valid_range_masked_to_nan() -> None:
    raw = np.array([-5.0, 250.0, 600.0])
    out = apply_hdf4_scale(raw, {"valid_range": [0.0, 500.0]})
    assert np.isnan(out[0]) and np.isnan(out[2])
    assert out[1] == 250.0


def test_no_attrs_is_identity_in_float64() -> None:
    raw = np.array([1, 2, 3], dtype=np.int32)
    out = apply_hdf4_scale(raw, {})
    np.testing.assert_allclose(out, [1.0, 2.0, 3.0])
    assert out.dtype == np.float64


def test_modis_viirs_staticmethod_delegates() -> None:
    from davinci_monet.datasets.satellite.modis_viirs import MODISVIIRSReader

    raw = np.array([10], dtype=np.int16)
    out = MODISVIIRSReader._apply_hdf4_scale(raw, {"scale_factor": 2.0})
    np.testing.assert_allclose(out, [20.0])
