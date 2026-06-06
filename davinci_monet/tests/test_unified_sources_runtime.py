"""Runtime regressions for the unified data-source refactor."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr


def _write_swath_source(path: Path, *, n_scan: int = 30, n_pix: int = 8) -> None:
    """Write a raw L2 swath NetCDF (2-D lat/lon on scanline/pixel dims).

    Mirrors a satellite L2 product: along-track scanlines, cross-track pixels,
    a single retrieved variable, and a per-scanline time coordinate. Lat/lon
    fall inside the GRID source's footprint so binning lands on real cells.
    """
    scan_lats = np.linspace(-48.0, -2.0, n_scan)
    scan_lons = np.linspace(102.0, 158.0, n_scan)
    pix_offsets = np.linspace(-3.0, 3.0, n_pix)
    lat2d = scan_lats[:, None] + pix_offsets[None, :] * 0.1
    lon2d = scan_lons[:, None] + pix_offsets[None, :]
    # Deterministic retrieved field: smooth gradient so binning is reproducible.
    aod = 0.1 + 0.002 * (lat2d - lat2d.min()) + 0.001 * (lon2d - lon2d.min())
    scan_times = np.array(["2019-12-21T12:00"], dtype="datetime64[m]").repeat(n_scan)
    ds = xr.Dataset(
        {"aod_550nm": (["scanline", "pixel"], aod.astype("float32"))},
        coords={
            "scanline": np.arange(n_scan),
            "pixel": np.arange(n_pix),
            "latitude": (["scanline", "pixel"], lat2d),
            "longitude": (["scanline", "pixel"], lon2d),
            "time": (["scanline"], scan_times),
        },
        attrs={"geometry": "swath"},
    )
    ds.to_netcdf(path)


def _write_grid_aod_source(path: Path, *, n_lat: int = 12, n_lon: int = 12) -> None:
    """Write a GRID NetCDF (1-D lat/lon) covering the swath footprint."""
    lat = np.linspace(-50.0, 0.0, n_lat)
    lon = np.linspace(100.0, 160.0, n_lon)
    times = np.array(["2019-12-21T00:00"], dtype="datetime64[m]")
    rng = np.random.default_rng(0)
    data = rng.uniform(0.05, 0.4, size=(1, n_lat, n_lon)).astype("float32")
    ds = xr.Dataset(
        {"AOD": (("time", "lat", "lon"), data)},
        coords={"time": times, "lat": lat, "lon": lon},
        attrs={"geometry": "grid"},
    )
    ds.to_netcdf(path)


def _write_grid_source(path: Path, *, offset: float = 0.0) -> None:
    times = np.array(["2024-01-01T00:00", "2024-01-01T01:00"], dtype="datetime64[m]")
    lat = np.array([40.0, 41.0])
    lon = np.array([-105.0, -104.0])
    values = np.arange(8, dtype=float).reshape(2, 2, 2) + offset
    ds = xr.Dataset(
        {"O3": (("time", "lat", "lon"), values)},
        coords={"time": times, "lat": lat, "lon": lon},
        attrs={"geometry": "grid"},
    )
    ds.to_netcdf(path)


def _write_point_source(path: Path) -> None:
    times = np.array(["2024-01-01T00:00", "2024-01-01T01:00"], dtype="datetime64[m]")
    ds = xr.Dataset(
        {"o3": (("time", "site"), np.array([[1.0, 2.0], [3.0, 4.0]]))},
        coords={
            "time": times,
            "site": np.array([0, 1]),
            "latitude": ("site", np.array([40.0, 41.0])),
            "longitude": ("site", np.array([-105.0, -104.0])),
        },
        attrs={"geometry": "point"},
    )
    ds.to_netcdf(path)


def test_sources_config_pairs_from_pair_variables(tmp_path: Path) -> None:
    from davinci_monet.pipeline.runner import PipelineRunner

    model_path = tmp_path / "model.nc"
    obs_path = tmp_path / "obs.nc"
    _write_grid_source(model_path)
    _write_point_source(obs_path)

    config = {
        "analysis": {"output_dir": str(tmp_path / "out")},
        "sources": {
            "cam": {
                "type": "generic",
                "role": "model",
                "files": str(model_path),
                "radius_of_influence": 200000,
                "variables": {"O3": {"units": "ppb"}},
            },
            "airnow": {
                "type": "pt_sfc",
                "role": "obs",
                "filename": str(obs_path),
                "variables": {"o3": {"units": "ppb"}},
            },
        },
        "pairs": {
            "cam_airnow_o3": {
                "sources": ["cam", "airnow"],
                "reference": "airnow",
                "variables": {"cam": "O3", "airnow": "o3"},
            }
        },
        "stats": {"metrics": ["N", "MB"]},
    }

    result = PipelineRunner(show_progress=False).run_from_config(config)

    assert result.success
    assert result.context is not None
    assert set(result.context.sources) == {"cam", "airnow"}
    assert set(result.context.paired) == {"cam_airnow_o3"}
    paired = result.context.paired["cam_airnow_o3"].data
    assert set(paired.data_vars) == {"cam_o3", "airnow_o3"}
    assert paired["cam_o3"].attrs["source_label"] == "cam"
    assert paired["airnow_o3"].attrs["source_label"] == "airnow"


def test_implicit_auto_pairing_from_mapping_without_pairs(tmp_path: Path) -> None:
    """A sources config with a model ``mapping:`` and no ``pairs:`` auto-pairs.

    Runs the real user path (``PipelineRunner.run_from_config``). With no
    explicit ``pairs:`` block, PairingStage synthesizes jobs from the model
    source's ``mapping:`` and routes them through ``engine.pair_sources`` — the
    same unified executor as explicit pairs. The pair key keeps the historical
    ``<model>_<obs>`` form and paired vars are renamed to source labels and
    role-tagged, proving the unified (not the removed legacy) path produced them.
    """
    from davinci_monet.pipeline.runner import PipelineRunner

    model_path = tmp_path / "model.nc"
    obs_path = tmp_path / "obs.nc"
    _write_grid_source(model_path)
    _write_point_source(obs_path)

    config = {
        "analysis": {"output_dir": str(tmp_path / "out")},
        "sources": {
            "cam": {
                "type": "generic",
                "role": "model",
                "files": str(model_path),
                "radius_of_influence": 200000,
                "mapping": {"airnow": {"o3": "O3"}},
                "variables": {"O3": {"units": "ppb"}},
            },
            "airnow": {
                "type": "pt_sfc",
                "role": "obs",
                "filename": str(obs_path),
                "variables": {"o3": {"units": "ppb"}},
            },
        },
        "stats": {"metrics": ["N", "MB"]},
    }

    result = PipelineRunner(show_progress=False).run_from_config(config)

    assert result.success
    assert result.context is not None
    # Implicit pair key is <model>_<obs>, matching the historical loop.
    assert set(result.context.paired) == {"cam_airnow"}
    paired = result.context.paired["cam_airnow"].data
    # pair_sources renamed both sides to source-label names and tagged roles.
    assert set(paired.data_vars) == {"cam_o3", "airnow_o3"}
    assert paired["cam_o3"].attrs["role"] == "model"
    assert paired["airnow_o3"].attrs["role"] == "obs"
    assert paired["cam_o3"].attrs["pair_role"] == "comparand"
    assert paired["airnow_o3"].attrs["pair_role"] == "reference"


def test_sources_config_supports_model_model_pair(tmp_path: Path) -> None:
    from davinci_monet.pipeline.runner import PipelineRunner

    ref_path = tmp_path / "ref.nc"
    comp_path = tmp_path / "comp.nc"
    _write_grid_source(ref_path, offset=0.0)
    _write_grid_source(comp_path, offset=1.0)

    config = {
        "analysis": {"output_dir": str(tmp_path / "out")},
        "sources": {
            "cam_ref": {"type": "generic", "role": "model", "files": str(ref_path)},
            "cam_cmp": {"type": "generic", "role": "model", "files": str(comp_path)},
        },
        "pairs": {
            "cam_ref_cam_cmp_o3": {
                "sources": ["cam_ref", "cam_cmp"],
                "reference": "cam_ref",
                "variables": {"cam_ref": "O3", "cam_cmp": "O3"},
            }
        },
        "stats": {"metrics": ["N", "MB"]},
    }

    result = PipelineRunner(show_progress=False).run_from_config(config)

    assert result.success
    assert result.context is not None
    paired = result.context.paired["cam_ref_cam_cmp_o3"].data
    assert set(paired.data_vars) == {"cam_ref_O3", "cam_cmp_O3"}
    assert paired["cam_ref_O3"].attrs["pair_role"] == "reference"
    assert paired["cam_cmp_O3"].attrs["pair_role"] == "comparand"
    assert paired["cam_ref_O3"].attrs["role"] == "model"
    assert paired["cam_cmp_O3"].attrs["role"] == "model"


def test_plot_data_reference_without_pair_spec_uses_paired_dataset(
    tmp_path: Path,
) -> None:
    from davinci_monet.pipeline.stages import PipelineContext, PlottingStage, StageStatus

    times = np.array(["2024-01-01T00:00", "2024-01-01T01:00"], dtype="datetime64[m]")
    paired = xr.Dataset(
        {
            "airnow_o3": ("time", np.array([1.0, 2.0])),
            "cam_o3": ("time", np.array([1.1, 2.2])),
        },
        coords={"time": times},
    )
    paired["airnow_o3"].attrs.update(
        {"role": "obs", "pair_role": "reference", "source_label": "airnow"}
    )
    paired["cam_o3"].attrs.update(
        {"role": "model", "pair_role": "comparand", "source_label": "cam"}
    )
    ctx = PipelineContext(
        config={
            "analysis": {"output_dir": str(tmp_path)},
            "plots": {"scatter_o3": {"type": "scatter", "data": ["cam_airnow"]}},
        }
    )
    ctx.paired["cam_airnow"] = paired

    result = PlottingStage().execute(ctx)

    assert result.status is StageStatus.COMPLETED
    assert len(result.data["plots_generated"]) == 2


def test_plot_sources_pair_spec_uses_reference_and_comparand(
    tmp_path: Path,
) -> None:
    from davinci_monet.pipeline.stages import PipelineContext, PlottingStage, StageStatus

    times = np.array(["2024-01-01T00:00", "2024-01-01T01:00"], dtype="datetime64[m]")
    paired = xr.Dataset(
        {
            "airnow_o3": ("time", np.array([1.0, 2.0])),
            "cam_o3": ("time", np.array([1.1, 2.2])),
        },
        coords={"time": times},
    )
    paired["airnow_o3"].attrs.update(
        {"role": "obs", "pair_role": "reference", "source_label": "airnow"}
    )
    paired["cam_o3"].attrs.update(
        {"role": "model", "pair_role": "comparand", "source_label": "cam"}
    )
    ctx = PipelineContext(
        config={
            "analysis": {"output_dir": str(tmp_path)},
            "pairs": {
                "cam_airnow_o3": {
                    "sources": ["cam", "airnow"],
                    "reference": "airnow",
                    "variables": {"cam": "O3", "airnow": "o3"},
                }
            },
            "plots": {"scatter_o3": {"type": "scatter", "data": ["cam_airnow_o3"]}},
        }
    )
    ctx.paired["cam_airnow_o3"] = paired

    result = PlottingStage().execute(ctx)

    assert result.status is StageStatus.COMPLETED
    assert len(result.data["plots_generated"]) == 2


def test_plot_legacy_pair_spec_falls_back_to_configured_pair_name(
    tmp_path: Path,
) -> None:
    from davinci_monet.pipeline.stages import PipelineContext, PlottingStage, StageStatus

    times = np.array(["2024-01-01T00:00", "2024-01-01T01:00"], dtype="datetime64[m]")
    paired = xr.Dataset(
        {
            "airnow_o3": ("time", np.array([1.0, 2.0])),
            "cam_o3": ("time", np.array([1.1, 2.2])),
        },
        coords={"time": times},
    )
    paired["airnow_o3"].attrs.update(
        {"role": "obs", "pair_role": "reference", "source_label": "airnow"}
    )
    paired["cam_o3"].attrs.update(
        {"role": "model", "pair_role": "comparand", "source_label": "cam"}
    )
    ctx = PipelineContext(
        config={
            "analysis": {"output_dir": str(tmp_path)},
            "pairs": {
                "cam_airnow_o3": {
                    "model": "cam",
                    "obs": "airnow",
                    "variable": {"model_var": "O3", "obs_var": "o3"},
                }
            },
            "plots": {"scatter_o3": {"type": "scatter", "data": ["cam_airnow_o3"]}},
        }
    )
    ctx.paired["cam_airnow_o3"] = paired

    result = PlottingStage().execute(ctx)

    assert result.status is StageStatus.COMPLETED
    assert len(result.data["plots_generated"]) == 2


def test_invalid_sources_pair_missing_variable_fails() -> None:
    from davinci_monet.core.protocols import DataGeometry
    from davinci_monet.pipeline.stages import (
        PairingStage,
        PipelineContext,
        SourceData,
        StageStatus,
    )

    ctx = PipelineContext(
        config={
            "pairs": {
                "cam_airnow_o3": {
                    "sources": ["cam", "airnow"],
                    "reference": "airnow",
                    "variables": {"cam": "O3"},
                }
            }
        },
        sources={
            "cam": SourceData(
                data=xr.Dataset({"O3": ("time", np.array([1.0]))}),
                label="cam",
                source_type="generic",
                geometry=DataGeometry.GRID,
                role="model",
            ),
            "airnow": SourceData(
                data=xr.Dataset({"o3": ("time", np.array([1.0]))}),
                label="airnow",
                source_type="pt_sfc",
                geometry=DataGeometry.POINT,
                role="obs",
            ),
        },
    )

    result = PairingStage().execute(ctx)

    assert result.status is StageStatus.FAILED
    assert "cam_airnow_o3" in str(result.error)
    assert "missing variable mapping" in str(result.error)
    assert ctx.paired == {}


def test_invalid_sources_pair_unknown_source_fails() -> None:
    from davinci_monet.core.protocols import DataGeometry
    from davinci_monet.pipeline.stages import (
        PairingStage,
        PipelineContext,
        SourceData,
        StageStatus,
    )

    ctx = PipelineContext(
        config={
            "pairs": {
                "cam_missing_o3": {
                    "sources": ["cam", "missing_obs"],
                    "reference": "missing_obs",
                    "variables": {"cam": "O3", "missing_obs": "o3"},
                }
            }
        },
        sources={
            "cam": SourceData(
                data=xr.Dataset({"O3": ("time", np.array([1.0]))}),
                label="cam",
                source_type="generic",
                geometry=DataGeometry.GRID,
                role="model",
            )
        },
    )

    stage = PairingStage()
    assert stage.validate(ctx)
    result = stage.execute(ctx)

    assert result.status is StageStatus.FAILED
    assert "cam_missing_o3" in str(result.error)
    assert "unknown source" in str(result.error)
    assert ctx.paired == {}


def test_invalid_legacy_pair_missing_source_fails_when_sources_loaded() -> None:
    """A legacy model/obs/variable pair is migrated then validated.

    The real pipeline auto-converts legacy ``pairs:`` to the unified
    ``sources:`` form via ``migrate_to_sources`` before pairing runs;
    PairingStage no longer accepts the legacy pair shape directly. Here the
    migrated pair still fails because ``airnow`` is not a loaded source.
    """
    from davinci_monet.config.migration import migrate_to_sources
    from davinci_monet.core.protocols import DataGeometry
    from davinci_monet.pipeline.stages import (
        PairingStage,
        PipelineContext,
        SourceData,
        StageStatus,
    )

    ctx = PipelineContext(
        config=migrate_to_sources(
            {
                "pairs": {
                    "cam_airnow_o3": {
                        "model": "cam",
                        "obs": "airnow",
                        "variable": {"model_var": "O3", "obs_var": "o3"},
                    }
                }
            }
        ),
        sources={
            "cam": SourceData(
                data=xr.Dataset({"O3": ("time", np.array([1.0]))}),
                label="cam",
                source_type="generic",
                geometry=DataGeometry.GRID,
                role="model",
            )
        },
    )

    result = PairingStage().execute(ctx)

    assert result.status is StageStatus.FAILED
    assert "cam_airnow_o3" in str(result.error)
    assert "unknown source" in str(result.error)
    assert ctx.paired == {}


def test_invalid_legacy_pair_missing_variable_fails_when_sources_loaded() -> None:
    """A migrated legacy pair missing a variable mapping fails validation."""
    from davinci_monet.config.migration import migrate_to_sources
    from davinci_monet.core.protocols import DataGeometry
    from davinci_monet.pipeline.stages import (
        PairingStage,
        PipelineContext,
        SourceData,
        StageStatus,
    )

    ctx = PipelineContext(
        config=migrate_to_sources(
            {
                "pairs": {
                    "cam_airnow_o3": {
                        "model": "cam",
                        "obs": "airnow",
                        "variable": {"model_var": "O3"},
                    }
                }
            }
        ),
        sources={
            "cam": SourceData(
                data=xr.Dataset({"O3": ("time", np.array([1.0]))}),
                label="cam",
                source_type="generic",
                geometry=DataGeometry.GRID,
                role="model",
            ),
            "airnow": SourceData(
                data=xr.Dataset({"o3": ("time", np.array([1.0]))}),
                label="airnow",
                source_type="pt_sfc",
                geometry=DataGeometry.POINT,
                role="obs",
            ),
        },
    )

    result = PairingStage().execute(ctx)

    assert result.status is StageStatus.FAILED
    assert "cam_airnow_o3" in str(result.error)
    assert "missing variable mapping" in str(result.error)
    assert ctx.paired == {}


def test_single_model_source_gets_descriptive_stats(tmp_path: Path) -> None:
    from davinci_monet.pipeline.runner import PipelineRunner

    source_path = tmp_path / "cam.nc"
    _write_grid_source(source_path)

    config = {
        "analysis": {"output_dir": str(tmp_path / "out")},
        "sources": {
            "cam": {
                "type": "generic",
                "role": "model",
                "files": str(source_path),
                "variables": {"O3": {"units": "ppb"}},
            }
        },
        "stats": {"metrics": ["N"]},
    }

    result = PipelineRunner(show_progress=False).run_from_config(config)

    assert result.success
    assert result.context is not None
    stats = result.context.results["statistics"].data
    assert "cam" in stats
    assert "O3" in stats["cam"]
    assert stats["cam"]["O3"]["N"] == 8


def test_single_source_plot_uses_source_key_not_obs_key(tmp_path: Path) -> None:
    from davinci_monet.pipeline.runner import PipelineRunner

    source_path = tmp_path / "cam.nc"
    _write_grid_source(source_path)

    config = {
        "analysis": {"output_dir": str(tmp_path / "out")},
        "sources": {
            "cam": {
                "type": "generic",
                "role": "model",
                "files": str(source_path),
                "variables": {"O3": {"units": "ppb"}},
            }
        },
        "plots": {
            "hist_o3": {
                "type": "histogram",
                "source": "cam",
                "variable": "O3",
            }
        },
    }

    result = PipelineRunner(show_progress=False).run_from_config(config)

    assert result.success
    assert result.context is not None
    plots = result.context.results["plotting"].data["plots_generated"]
    assert len([p for p in plots if p.endswith(".png")]) == 1


def test_unsupported_source_pair_fails_pairing_stage() -> None:
    from davinci_monet.core.protocols import DataGeometry
    from davinci_monet.pipeline.stages import (
        PairingStage,
        PipelineContext,
        SourceData,
        StageStatus,
    )

    point_a = xr.Dataset(
        {"o3": ("site", np.array([1.0]))},
        coords={
            "site": [0],
            "latitude": ("site", [40.0]),
            "longitude": ("site", [-105.0]),
        },
        attrs={"geometry": "point"},
    )
    track_b = xr.Dataset(
        {"o3": ("time", np.array([1.2]))},
        coords={
            "time": np.array(["2024-01-01T00:00"], dtype="datetime64[m]"),
            "latitude": ("time", [40.0]),
            "longitude": ("time", [-105.0]),
        },
        attrs={"geometry": "track"},
    )
    ctx = PipelineContext(
        config={
            "pairs": {
                "a_b_o3": {
                    "sources": ["a", "b"],
                    "reference": "a",
                    "variables": {"a": "o3", "b": "o3"},
                }
            }
        },
        sources={
            "a": SourceData(point_a, "a", "pt_sfc", DataGeometry.POINT, role="obs"),
            "b": SourceData(track_b, "b", "icartt", DataGeometry.TRACK, role="obs"),
        },
    )

    result = PairingStage().execute(ctx)

    assert result.status is StageStatus.FAILED
    assert "a_b_o3" in str(result.error)
    assert "Unsupported pairing combination" in str(result.error)


def test_sources_config_supports_obs_obs_grid_pair(tmp_path: Path) -> None:
    from davinci_monet.pipeline.runner import PipelineRunner

    ref_path = tmp_path / "sat_ref.nc"
    comp_path = tmp_path / "sat_cmp.nc"
    _write_grid_source(ref_path, offset=0.0)
    _write_grid_source(comp_path, offset=1.0)

    config = {
        "analysis": {"output_dir": str(tmp_path / "out")},
        "sources": {
            "modis": {"type": "generic", "role": "obs", "files": str(ref_path)},
            "viirs": {"type": "generic", "role": "obs", "files": str(comp_path)},
        },
        "pairs": {
            "modis_viirs_o3": {
                "sources": ["modis", "viirs"],
                "reference": "modis",
                "variables": {"modis": "O3", "viirs": "O3"},
            }
        },
        "stats": {"metrics": ["N", "MB"]},
    }

    result = PipelineRunner(show_progress=False).run_from_config(config)

    assert result.success
    assert result.context is not None
    paired = result.context.paired["modis_viirs_o3"].data
    assert set(paired.data_vars) == {"modis_O3", "viirs_O3"}
    assert paired["modis_O3"].attrs["pair_role"] == "reference"
    assert paired["viirs_O3"].attrs["pair_role"] == "comparand"
    assert paired["modis_O3"].attrs["role"] == "obs"
    assert paired["viirs_O3"].attrs["role"] == "obs"


def test_unified_source_applies_resample(tmp_path: Path) -> None:
    """A `sources:` obs with `resample` is averaged to the target frequency at load."""
    import pandas as pd

    from davinci_monet.pipeline.runner import PipelineRunner

    src = tmp_path / "hf.nc"
    times = pd.date_range("2024-01-01T00:00", periods=4, freq="15min")
    ds = xr.Dataset(
        {"o3": (("time", "site"), np.array([[10.0], [20.0], [30.0], [40.0]]))},
        coords={
            "time": times,
            "site": [0],
            "latitude": ("site", [40.0]),
            "longitude": ("site", [-105.0]),
        },
        attrs={"geometry": "point"},
    )
    ds.to_netcdf(src)

    config = {
        "analysis": {"output_dir": str(tmp_path / "out")},
        "sources": {
            "hf": {
                "type": "generic",
                "role": "obs",
                "files": str(src),
                "resample": "h",
                "track_obs_count": True,
                "variables": {"o3": {"units": "ppb"}},
            }
        },
    }

    result = PipelineRunner(show_progress=False).run_from_config(config)

    assert result.success
    assert result.context is not None
    loaded = result.context.sources["hf"].data
    assert loaded.sizes["time"] == 1
    assert float(loaded["o3"].isel(time=0, site=0)) == 25.0
    assert "obs_count" in loaded
    assert int(loaded["obs_count"].isel(time=0, site=0)) == 4


def test_sources_config_pairs_swath_onto_grid(tmp_path: Path) -> None:
    """A SWATH source pairs onto a GRID reference end-to-end via run_from_config.

    Proves the production ``SwathGridStrategy`` (numba binning) is what the
    engine routes a swath-vs-grid pair through: the swath pixels are binned onto
    the grid, so the paired output is GRID-geometry ``(time, lon, lat)`` with
    reference/comparand variables and role tags. Before SwathGridStrategy was
    registered, the engine used the non-production per-pixel SwathStrategy, whose
    pair() emits ``(y, x)``/``(pixel,)`` model output and obs-named (not
    ``obs_``-prefixed) vars, so this assertion set fails (no binned grid).
    """
    from davinci_monet.pipeline.runner import PipelineRunner

    swath_path = tmp_path / "modis_l2.nc"
    grid_path = tmp_path / "cam.nc"
    _write_swath_source(swath_path)
    _write_grid_aod_source(grid_path)

    config = {
        "analysis": {
            "output_dir": str(tmp_path / "out"),
        },
        "sources": {
            "cam": {
                "type": "generic",
                "role": "model",
                "files": str(grid_path),
                "variables": {"AOD": {"units": "1"}},
            },
            "modis": {
                "type": "satellite_l2",
                "role": "obs",
                "files": str(swath_path),
                "variables": {"aod_550nm": {"units": "1"}},
            },
        },
        "pairs": {
            "cam_modis_aod": {
                "sources": ["cam", "modis"],
                "reference": "modis",
                "variables": {"cam": "AOD", "modis": "aod_550nm"},
            }
        },
        "stats": {"metrics": ["N", "MB"]},
    }

    result = PipelineRunner(show_progress=False).run_from_config(config)

    failed = [
        f"{s.stage_name}: {s.error}" for s in result.stage_results if s.status.name == "FAILED"
    ]
    assert result.success, f"Pipeline failed. Errors: {failed}"
    assert result.context is not None
    assert set(result.context.sources) == {"cam", "modis"}
    assert set(result.context.paired) == {"cam_modis_aod"}

    paired = result.context.paired["cam_modis_aod"].data
    # SwathGridStrategy binned onto the grid: GRID-geometry output, NOT the
    # per-pixel (scanline, pixel)/(y, x) output the non-production SwathStrategy
    # would emit.
    assert set(paired.dims) >= {"time", "lon", "lat"}
    assert not ({"scanline", "pixel", "y", "x"} & set(paired.dims))
    # Reference and comparand share the canonical stem (aod_550nm) under their
    # source-label prefixes, with role + pair_role tags.
    assert "modis_aod_550nm" in paired.data_vars
    assert "cam_aod_550nm" in paired.data_vars
    assert paired["modis_aod_550nm"].attrs["pair_role"] == "reference"
    assert paired["cam_aod_550nm"].attrs["pair_role"] == "comparand"
    assert paired["modis_aod_550nm"].attrs["role"] == "obs"
    assert paired["cam_aod_550nm"].attrs["role"] == "model"
    # At least one grid cell received binned swath pixels (non-NaN), proving the
    # numba binning actually ran end-to-end.
    assert bool(np.isfinite(paired["modis_aod_550nm"].values).any())
