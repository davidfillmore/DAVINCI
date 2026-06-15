"""Plot gallery: one PDF per plot type, every figure rendered THROUGH the pipeline.

This is the pipeline-driven gallery.  Per the project rule "ALL analysis
scripts MUST use DAVINCI pipelines", nothing here calls renderer ``.render()``
directly.  Instead the script:

  1. Writes synthetic *source* NetCDF files to ``analyses/_gallery/data/``.
  2. Writes YAML configs to ``analyses/_gallery/configs/``.
  3. Runs each config via :func:`davinci_monet.pipeline.runner.run_analysis`,
     exercising the full load -> pair -> stats -> plot path.
  4. Collects the resulting PDFs into a clean ``analyses/_gallery/output/``.

Plot types covered (each produced through the pipeline):
  scatter, timeseries, spatial (single-source map), spatial_bias, curtain,
  vertical_profile, histogram, flight_track, track_map_3d.

Data is internally consistent (source name matches the physical quantity):
  - O3 plots use ppbv "Ozone" sources (cam / wrf / airnow / cesm).
  - NO2-column plots use mol/m2 sources (cesm_no2_column model, tropomi obs).
  - OLR bias uses W m-2 sources (merra2 model, ceres obs).

Usage::

    source ~/miniconda3/etc/profile.d/conda.sh && conda activate davinci
    HDF5_USE_FILE_LOCKING=FALSE python analyses/_gallery/run_gallery.py

Output: analyses/_gallery/output/*.pdf
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # non-interactive backend; must precede pyplot import

import numpy as np
import pandas as pd
import xarray as xr
import yaml

# Resolve project root so the script runs from any working directory.
GALLERY_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = GALLERY_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from davinci_monet.pipeline.runner import run_analysis  # noqa: E402

DATA_DIR = GALLERY_DIR / "data"
CONFIG_DIR = GALLERY_DIR / "configs"
OUTPUT_DIR = GALLERY_DIR / "output"
# Each config writes its plots under a private run directory so the comparison
# and single-source naming schemes don't collide; PDFs are then copied into the
# flat OUTPUT_DIR with stable gallery names.
RUNS_DIR = GALLERY_DIR / "_runs"

START = "2024-02-01"
END = "2024-02-03"


# ---------------------------------------------------------------------------
# Synthetic source builders (each returns a plain xr.Dataset to NetCDF)
# ---------------------------------------------------------------------------


def _time_axis(n: int, freq: str = "1h", start: str = START) -> pd.DatetimeIndex:
    return pd.date_range(start, periods=n, freq=freq)


def _gridded_o3(seed: int, mean: float = 42.0) -> xr.Dataset:
    """2-D (time, lat, lon) O3 grid [ppbv].  Read by the ``generic`` reader."""
    rng = np.random.default_rng(seed)
    times = _time_axis(48, "1h")
    lats = np.linspace(35.0, 45.0, 12)
    lons = np.linspace(-105.0, -95.0, 12)
    lat_norm = (lats - lats.min()) / (lats.max() - lats.min())
    # base is (n_lat, n_lon): a latitudinal O3 gradient, uniform across lon.
    base = mean + 15.0 * np.broadcast_to(lat_norm[:, None], (len(lats), len(lons)))
    field = np.stack(
        [base + rng.normal(0, 3.0, base.shape) for _ in range(len(times))], axis=0
    ).clip(min=0)
    return xr.Dataset(
        {"O3": (["time", "lat", "lon"], field, {"units": "ppbv", "long_name": "Ozone"})},
        coords={
            "time": times,
            "lat": ("lat", lats, {"units": "degrees_north"}),
            "lon": ("lon", lons, {"units": "degrees_east"}),
        },
        attrs={"geometry": "grid"},
    )


def _point_o3(seed: int, mean: float = 40.0, n_site: int = 25) -> xr.Dataset:
    """(time, site) surface O3 point data [ppbv].  Read by the ``pt_sfc`` reader."""
    rng = np.random.default_rng(seed)
    times = _time_axis(48, "1h")
    lats = rng.uniform(35.0, 45.0, n_site)
    lons = rng.uniform(-105.0, -95.0, n_site)
    vals = rng.normal(mean, mean * 0.18, (len(times), n_site)).clip(min=0)
    return xr.Dataset(
        {"O3": (["time", "site"], vals, {"units": "ppbv", "long_name": "Ozone"})},
        coords={
            "time": times,
            "site": np.arange(n_site),
            "latitude": ("site", lats, {"units": "degrees_north"}),
            "longitude": ("site", lons, {"units": "degrees_east"}),
        },
        attrs={"geometry": "point"},
    )


def _track_o3(seed: int, n: int = 300, two_flights: bool = True) -> xr.Dataset:
    """(time,) aircraft track O3 [ppbv] with lat/lon/altitude coords.

    Read by the ``aircraft`` reader.  A ``flight`` coord splits the data into
    two flights for the geometry-only renderers and per-flight 3-D tracks.
    """
    rng = np.random.default_rng(seed)
    times = np.datetime64(f"{START}T14:00") + np.arange(n) * np.timedelta64(30, "s")
    t = np.linspace(0, 4 * np.pi, n)
    lats = 38.0 + 3.0 * np.sin(t / 2)
    lons = -100.0 + 4.0 * np.cos(t / 3)
    alts = 1000.0 + 8000.0 * (0.5 + 0.5 * np.sin(t / 2))
    o3 = (30.0 + 6.0 * (alts / 1000.0) + rng.normal(0, 3, n)).clip(min=0)
    coords = {
        "time": times,
        "latitude": ("time", lats, {"units": "degrees_north"}),
        "longitude": ("time", lons, {"units": "degrees_east"}),
        "altitude": ("time", alts, {"units": "m", "long_name": "Altitude ASL"}),
    }
    if two_flights:
        coords["flight"] = ("time", np.where(np.arange(n) < n // 2, "F01", "F02"))
    return xr.Dataset(
        {"O3": (["time"], o3, {"units": "ppbv", "long_name": "Ozone"})},
        coords=coords,
        attrs={"geometry": "track"},
    )


def _gridded_no2_column(seed: int, scale: float = 1.0) -> xr.Dataset:
    """2-D (time, lat, lon) NO2 *column* [mol/m2].  Read by the ``generic`` reader.

    Legitimately an NO2 column, so the source key may embed the species word.
    """
    rng = np.random.default_rng(seed)
    times = _time_axis(8, "1D")
    lats = np.linspace(20.0, 55.0, 18)
    lons = np.linspace(-130.0, -60.0, 36)
    lat_field = (2.0e-5 + 1.5e-5 * np.cos(np.radians(lats))) * scale
    base = np.broadcast_to(lat_field[:, None], (len(lats), len(lons)))
    field = np.stack(
        [base + rng.normal(0, 1e-6, base.shape) for _ in range(len(times))], axis=0
    ).clip(min=0)
    return xr.Dataset(
        {
            "NO2": (
                ["time", "lat", "lon"],
                field,
                {"units": "mol/m2", "long_name": "NO2 total column"},
            )
        },
        coords={
            "time": times,
            "lat": ("lat", lats, {"units": "degrees_north"}),
            "lon": ("lon", lons, {"units": "degrees_east"}),
        },
        attrs={"geometry": "grid"},
    )


def _gridded_olr(seed: int, bias: float = 0.0) -> xr.Dataset:
    """2-D (time, lat, lon) outgoing longwave radiation [W m-2].

    Read by the ``generic`` reader.  Used for the OLR spatial-bias map.
    """
    rng = np.random.default_rng(seed)
    times = _time_axis(3, "1D")
    lats = np.linspace(20.0, 55.0, 15)
    lons = np.linspace(-130.0, -60.0, 30)
    lat_field = 220.0 + 80.0 * np.cos(np.radians(lats)) + bias
    base = np.broadcast_to(lat_field[:, None], (len(lats), len(lons)))
    field = np.stack(
        [base + rng.normal(0, 8.0, base.shape) for _ in range(len(times))], axis=0
    ).clip(min=0)
    return xr.Dataset(
        {
            "OLR": (
                ["time", "lat", "lon"],
                field,
                {"units": "W m-2", "long_name": "Outgoing longwave radiation"},
            )
        },
        coords={
            "time": times,
            "lat": ("lat", lats, {"units": "degrees_north"}),
            "lon": ("lon", lons, {"units": "degrees_east"}),
        },
        attrs={"geometry": "grid"},
    )


# ---------------------------------------------------------------------------
# Config builders.  Each returns (config_name, config_dict) and writes the
# source NetCDFs it needs into DATA_DIR.
# ---------------------------------------------------------------------------


def _analysis_block(run_dir: Path) -> dict:
    return {
        "start_time": START,
        "end_time": END,
        "output_dir": str(run_dir),
        "log_dir": str(run_dir / "logs"),
        "style": {"theme": "ncar"},
    }


def _write_nc(ds: xr.Dataset, name: str) -> str:
    path = DATA_DIR / name
    if path.exists():
        path.unlink()
    ds.to_netcdf(path)
    return str(path)


def config_o3_surface(run_dir: Path) -> dict:
    """Paired surface O3: cesm grid (y) vs airnow points (x).

    Drives: scatter, timeseries, spatial_bias (paired); spatial (single-source
    map of the cesm grid); histogram (single-source of the airnow points).
    """
    cesm = _write_nc(_gridded_o3(seed=42, mean=44.0), "o3_cesm_grid.nc")
    airnow = _write_nc(_point_o3(seed=7, mean=40.0), "o3_airnow_point.nc")
    return {
        "analysis": _analysis_block(run_dir),
        "sources": {
            "cesm": {
                "type": "generic",
                "files": cesm,
                "radius_of_influence": 100000,
                "variables": {
                    "O3": {
                        "units": "ppbv",
                        "vmin_plot": 30,
                        "vmax_plot": 70,
                        "vdiff_plot": 15,
                    }
                },
            },
            "airnow": {
                "type": "pt_sfc",
                "filename": airnow,
                "variables": {"O3": {"valid_min": 0, "valid_max": 200, "units": "ppbv"}},
            },
        },
        "pairs": {
            "cesm_vs_airnow_o3": {
                "x": {"source": "airnow", "variable": "O3"},
                "y": {"source": "cesm", "variable": "O3"},
            }
        },
        "plots": {
            "scatter": {
                "type": "scatter",
                "pairs": ["cesm_vs_airnow_o3"],
                "title": "O3",
                "show_density": True,
            },
            "timeseries": {
                "type": "timeseries",
                "pairs": ["cesm_vs_airnow_o3"],
                "title": "O3",
                "aggregate_dim": "site",
                "show_uncertainty": True,
            },
            "spatial_bias": {
                "type": "spatial_bias",
                "pairs": ["cesm_vs_airnow_o3"],
                "title": "O3 Bias",
            },
            "spatial": {
                "type": "spatial",
                "source": "cesm",
                "variable": "O3",
                "title": "O3",
            },
            "histogram": {
                "type": "histogram",
                "source": "airnow",
                "variable": "O3",
                "title": "O3",
            },
        },
        "stats": {"metrics": ["N", "MB", "RMSE", "R", "NMB", "NME", "IOA"]},
    }


def config_o3_track(run_dir: Path) -> dict:
    """Paired O3 track: cam grid (y) sampled onto an aircraft track (x).

    Drives: curtain, track_map_3d (both pairwise track plots).
    """
    cam = _write_nc(_gridded_o3(seed=11, mean=46.0), "o3_cam_grid.nc")
    track = _write_nc(_track_o3(seed=55), "o3_aircraft_track.nc")
    return {
        "analysis": _analysis_block(run_dir),
        "sources": {
            "cam": {
                "type": "generic",
                "files": cam,
                "radius_of_influence": 100000,
                "variables": {"O3": {"units": "ppbv"}},
            },
            "aircraft": {
                "type": "aircraft",
                "filename": track,
                "variables": {"O3": {"units": "ppbv"}},
            },
        },
        "pairs": {
            "cam_vs_aircraft_o3": {
                "x": {"source": "aircraft", "variable": "O3"},
                "y": {"source": "cam", "variable": "O3"},
            }
        },
        "plots": {
            "curtain": {
                "type": "curtain",
                "pairs": ["cam_vs_aircraft_o3"],
                "title": "O3",
                "show_var": "x",
            },
            "track_map_3d": {
                "type": "track_map_3d",
                "pairs": ["cam_vs_aircraft_o3"],
                "title": "O3",
                "show_var": "x",
                "show_coastlines": False,
            },
        },
        "stats": {"metrics": ["N", "MB", "RMSE", "R"]},
    }


def config_o3_geometry(run_dir: Path) -> dict:
    """Geometry-only O3 aircraft track (single source, no pairs).

    Drives: vertical_profile, flight_track (single-source specialised plots).
    """
    track = _write_nc(_track_o3(seed=88), "o3_dc8_track.nc")
    return {
        "analysis": _analysis_block(run_dir),
        "sources": {
            "cam": {
                "type": "aircraft",
                "filename": track,
                "variables": {"O3": {"units": "ppbv"}},
            }
        },
        "plots": {
            "vertical_profile": {
                "type": "vertical_profile",
                "source": "cam",
                "variable": "O3",
                "title": "O3",
            },
            "flight_track": {
                "type": "flight_track",
                "source": "cam",
                "variable": "O3",
                "title": "O3",
            },
        },
    }


def config_no2_column(run_dir: Path) -> dict:
    """NO2 column [mol/m2]: cesm_no2_column model (y) vs tropomi obs (x).

    Drives: scatter (paired, here overwriting the O3 scatter is avoided by a
    distinct gallery name); spatial (single-source NO2-column map of the model).
    Both sources are gridded NO2 columns, so the species word in the model key
    is legitimate.
    """
    model = _write_nc(_gridded_no2_column(seed=22, scale=1.1), "no2_cesm_column.nc")
    obs = _write_nc(_gridded_no2_column(seed=23, scale=1.0), "no2_tropomi_column.nc")
    return {
        "analysis": _analysis_block(run_dir),
        "sources": {
            "cesm_no2_column": {
                "type": "generic",
                "files": model,
                "radius_of_influence": 100000,
                "variables": {"NO2": {"units": "mol/m2"}},
            },
            "tropomi": {
                "type": "gridded",
                "files": obs,
                "variables": {"NO2": {"units": "mol/m2"}},
            },
        },
        "pairs": {
            "cesm_vs_tropomi_no2": {
                "x": {"source": "tropomi", "variable": "NO2"},
                "y": {"source": "cesm_no2_column", "variable": "NO2"},
            }
        },
        "plots": {
            "scatter_no2": {
                "type": "scatter",
                "pairs": ["cesm_vs_tropomi_no2"],
                "title": "NO2 Column",
                "show_density": True,
            },
            "spatial_no2": {
                "type": "spatial",
                "source": "cesm_no2_column",
                "variable": "NO2",
                "title": "NO2 Column",
            },
        },
        "stats": {"metrics": ["N", "MB", "RMSE", "R"]},
    }


def config_olr_bias(run_dir: Path) -> dict:
    """OLR spatial bias [W m-2]: merra2 model (y) vs ceres obs (x)."""
    merra2 = _write_nc(_gridded_olr(seed=66, bias=12.0), "olr_merra2_grid.nc")
    ceres = _write_nc(_gridded_olr(seed=67, bias=0.0), "olr_ceres_grid.nc")
    return {
        "analysis": _analysis_block(run_dir),
        "sources": {
            "merra2": {
                "type": "generic",
                "files": merra2,
                "radius_of_influence": 100000,
                "variables": {"OLR": {"units": "W m-2", "vdiff_plot": 30}},
            },
            "ceres": {
                "type": "gridded",
                "files": ceres,
                "variables": {"OLR": {"units": "W m-2"}},
            },
        },
        "pairs": {
            "merra2_vs_ceres_olr": {
                "x": {"source": "ceres", "variable": "OLR"},
                "y": {"source": "merra2", "variable": "OLR"},
            }
        },
        "plots": {
            "spatial_bias_olr": {
                "type": "spatial_bias",
                "pairs": ["merra2_vs_ceres_olr"],
                "title": "TOA OLR Bias",
            }
        },
        "stats": {"metrics": ["N", "MB", "RMSE", "R", "NMB", "NME", "IOA"]},
    }


# ---------------------------------------------------------------------------
# Gallery driver
# ---------------------------------------------------------------------------

# (config_name, builder, run_subdir, {gallery_pdf_name: plot_name})
# The plot_name is the key in the config's ``plots:`` block; the pipeline
# writes ``<NN>_<plot_name>.pdf`` (paired) or ``<plot_name>.pdf`` (single
# source).  We resolve the actual file by globbing for ``*<plot_name>.pdf``.
_CONFIGS = [
    (
        "gallery-o3-surface",
        config_o3_surface,
        "o3_surface",
        {
            "scatter": "scatter",
            "timeseries": "timeseries",
            "spatial_bias": "spatial_bias",
            "spatial": "spatial",
            "histogram": "histogram",
        },
    ),
    (
        "gallery-o3-track",
        config_o3_track,
        "o3_track",
        {"curtain": "curtain", "track_map_3d": "track_map_3d"},
    ),
    (
        "gallery-o3-geometry",
        config_o3_geometry,
        "o3_geometry",
        {"vertical_profile": "vertical_profile", "flight_track": "flight_track"},
    ),
    (
        "gallery-no2-column",
        config_no2_column,
        "no2_column",
        {"scatter_no2": "scatter_no2", "spatial_no2": "spatial_no2"},
    ),
    (
        "gallery-olr-bias",
        config_olr_bias,
        "olr_bias",
        {"spatial_bias_olr": "spatial_bias_olr"},
    ),
]


def _collect_pdf(run_dir: Path, plot_name: str, gallery_name: str) -> Path | None:
    """Find the pipeline-produced PDF for ``plot_name`` and copy it to OUTPUT_DIR.

    Single-source plots may emit per-flight PDFs (``<plot>_F01.pdf``); we prefer
    the un-suffixed file and otherwise take the first per-flight figure.
    """
    candidates = sorted(run_dir.rglob(f"*{plot_name}.pdf"))
    if not candidates:
        # Per-flight / per-figure suffixes (e.g. track_map_3d_F01.pdf).
        candidates = sorted(run_dir.rglob(f"*{plot_name}*.pdf"))
    if not candidates:
        return None
    src = candidates[0]
    dst = OUTPUT_DIR / f"{gallery_name}.pdf"
    shutil.copy2(src, dst)
    return dst


def main() -> int:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Clean prior gallery artifacts for a fresh, reproducible run.
    for old in OUTPUT_DIR.glob("*.pdf"):
        old.unlink()
    for old in OUTPUT_DIR.glob("*.png"):
        old.unlink()
    if RUNS_DIR.exists():
        shutil.rmtree(RUNS_DIR)

    print(f"\nPipeline-driven plot gallery -> {OUTPUT_DIR}\n")

    failures: list[str] = []
    produced: dict[str, Path] = {}

    for config_name, builder, subdir, plot_map in _CONFIGS:
        run_dir = RUNS_DIR / subdir
        run_dir.mkdir(parents=True, exist_ok=True)

        config = builder(run_dir)
        config_path = CONFIG_DIR / f"{config_name}.yaml"
        with open(config_path, "w") as f:
            yaml.safe_dump(config, f, sort_keys=False)

        print(f"[{config_name}] running pipeline ...")
        result = run_analysis(str(config_path), show_progress=False)

        if not result.success:
            failed = [
                f"{s.stage_name}: {s.error}"
                for s in result.failed_stages  # type: ignore[attr-defined]
            ]
            failures.append(f"{config_name}: pipeline failed ({'; '.join(failed)})")
            print(f"  PIPELINE FAILED: {'; '.join(failed)}")
            continue

        for gallery_name, plot_name in plot_map.items():
            pdf = _collect_pdf(run_dir, plot_name, gallery_name)
            if pdf is None:
                failures.append(
                    f"{config_name}: no PDF found for plot '{plot_name}' "
                    f"(gallery '{gallery_name}')"
                )
                print(f"  MISSING PDF: {plot_name}")
            else:
                produced[gallery_name] = pdf
                print(f"  wrote {pdf.relative_to(GALLERY_DIR)}")

    print(f"\nProduced {len(produced)} gallery PDFs:")
    for name in sorted(produced):
        print(f"  {name}.pdf")

    if failures:
        print(f"\nFAILED ({len(failures)}):")
        for f in failures:
            print(f"  - {f}")
        return 1

    print("\nAll plot types rendered through run_analysis successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
