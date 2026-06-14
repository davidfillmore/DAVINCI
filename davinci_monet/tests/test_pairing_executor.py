"""Unit tests for the HDF5-safe bounded pairing executor.

These exercise the pure partition/worker-count helpers on ``PairingStage``
directly (no pipeline), so they are unit tests, not integration tests. The
end-to-end "all jobs run" behaviour is covered by the ``run_from_config`` tests
in ``test_unified_sources_runtime.py``.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from davinci_monet.pipeline.stages import PairingStage


def _eager_dataset() -> xr.Dataset:
    """A numpy-backed (unchunked) dataset — does a no-op ``.compute()``."""
    return xr.Dataset({"o3": ("time", np.arange(4.0))})


def _dask_dataset() -> xr.Dataset:
    """A dask-chunked dataset — triggers the per-strategy dask thread pool."""
    return _eager_dataset().chunk({"time": 2})


def test_is_dask_backed_false_for_eager_dataset() -> None:
    assert PairingStage._is_dask_backed(_eager_dataset()) is False


def test_is_dask_backed_true_for_chunked_dataset() -> None:
    assert PairingStage._is_dask_backed(_dask_dataset()) is True


def test_is_dask_backed_true_if_any_var_chunked() -> None:
    """Mixed dataset: one eager var, one chunked var -> dask-backed (any())."""
    ds = xr.Dataset(
        {
            "a": ("time", np.arange(4.0)),
            "b": ("time", np.arange(4.0)),
        }
    )
    ds["b"] = ds["b"].chunk({"time": 2})
    assert PairingStage._is_dask_backed(ds) is True


def test_is_dask_backed_none_is_false() -> None:
    assert PairingStage._is_dask_backed(None) is False


def test_worker_counts_default_half_cpu_clamped_by_job_count() -> None:
    """Default eager workers = min(n_eager, cpu // 2); dask serial by default."""
    eager, dask = PairingStage._pair_worker_counts({}, n_eager=3, cpu=8)
    assert eager == 3  # min(3, 8 // 2 = 4)
    assert dask == 1  # serial default


def test_worker_counts_default_capped_by_half_cpu() -> None:
    eager, _ = PairingStage._pair_worker_counts({}, n_eager=10, cpu=8)
    assert eager == 4  # min(10, 8 // 2 = 4)


def test_worker_counts_explicit_max_pair_workers_honored() -> None:
    eager, _ = PairingStage._pair_worker_counts({"max_pair_workers": 2}, n_eager=10, cpu=64)
    assert eager == 2


def test_worker_counts_eager_capped_at_8() -> None:
    """Even an explicit huge value is clamped to the hard cap of 8."""
    eager, _ = PairingStage._pair_worker_counts({"max_pair_workers": 100}, n_eager=100, cpu=128)
    assert eager == 8


def test_worker_counts_eager_floor_is_1() -> None:
    """A zero/negative request floors to 1 (never a 0-worker pool)."""
    eager, _ = PairingStage._pair_worker_counts({"max_pair_workers": 0}, n_eager=5, cpu=8)
    assert eager == 1


def test_worker_counts_no_eager_jgeometry_floors_to_1() -> None:
    """Default formula with n_eager=0 still floors to >=1 workers."""
    eager, _ = PairingStage._pair_worker_counts({}, n_eager=0, cpu=8)
    assert eager == 1


def test_worker_counts_dask_workers_configurable() -> None:
    _, dask = PairingStage._pair_worker_counts({"dask_pair_workers": 3}, n_eager=2, cpu=8)
    assert dask == 3


def test_worker_counts_dask_workers_floor_is_1() -> None:
    _, dask = PairingStage._pair_worker_counts({"dask_pair_workers": 0}, n_eager=2, cpu=8)
    assert dask == 1
