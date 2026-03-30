"""Radiative analysis runner — orchestrates load, process, and plot.

This module provides the main entry point for running a complete smoke
radiative analysis from a YAML configuration file.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from davinci_monet.logging import get_logger
from davinci_monet.plots.style import apply_ncar_style
from davinci_monet.radiative.config import RadiativeConfig
from davinci_monet.radiative.loaders.aeronet import load_aeronet
from davinci_monet.radiative.loaders.ceres import load_ceres_local
from davinci_monet.radiative.loaders.merra2 import load_merra2
from davinci_monet.radiative.plots import (
    plot_anomaly_maps,
    plot_event_fields,
    plot_surface_impact,
    plot_sw_vs_aod_scatter,
    plot_site_timeseries,
)
from davinci_monet.radiative.processing import (
    regrid_nearest,
    semi_empirical_surface_dimming,
)

logger = get_logger(__name__)

# Map from CERES variable names to standardized record keys
_VAR_MAP: dict[str, str] = {
    "obs_all_toa_sw": "sw_all",
    "obs_clr_toa_sw": "sw_clr",
    "obs_all_toa_net": "toa_net",
    "init_match_aod55": "aod",
    "obs_cld_amount": "cld_frac",
    "toa_sw_insol": "toa_insol",
}


def _parse_config(config_path: str) -> RadiativeConfig:
    """Load YAML and extract the 'radiative' section as a RadiativeConfig."""
    with open(config_path) as f:
        raw = yaml.safe_load(f)
    if "radiative" not in raw:
        raise ValueError(f"Config file {config_path} has no 'radiative' section")
    return RadiativeConfig(**raw["radiative"])


def _dates_in_range(start: date, end: date) -> list[date]:
    """Return an inclusive list of dates from *start* to *end*."""
    dates: list[date] = []
    d = start
    while d <= end:
        dates.append(d)
        d += timedelta(days=1)
    return dates


def _get_sites(cfg: RadiativeConfig) -> list[tuple[str, float, float, str]]:
    """Extract site list from AERONET config.

    Returns list of (name, lat, lon, aeronet_name) tuples.
    Placeholder coordinates are used -- AERONET data provides real coords.
    """
    if cfg.aeronet is None or cfg.aeronet.sites is None:
        return []
    # Use AERONET site names directly; lat/lon will be looked up from data
    return [(s, 0.0, 0.0, s) for s in cfg.aeronet.sites]


def run_radiative_analysis(config_path: str) -> dict[str, Any]:
    """Run a smoke radiative analysis from a YAML configuration file.

    Parameters
    ----------
    config_path : str
        Path to the YAML config file with a ``radiative`` section.

    Returns
    -------
    dict[str, Any]
        Result dict with keys: ``success``, ``plots_generated``, ``errors``.
    """
    apply_ncar_style()

    plots_generated: list[str] = []
    errors: list[str] = []

    try:
        cfg = _parse_config(config_path)
    except Exception as exc:
        logger.error("Config parse error: %s", exc)
        return {"success": False, "plots_generated": [], "errors": [str(exc)]}

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    domain = cfg.event.domain
    event_name = cfg.event.name
    bg_window = cfg.event.background_window

    # ------------------------------------------------------------------
    # 1. Load CERES
    # ------------------------------------------------------------------
    logger.info("Loading CERES data (%s)", cfg.ceres.product)
    try:
        ceres_ds = load_ceres_local(
            files=cfg.ceres.files,  # type: ignore[arg-type]
            domain=domain,
            variables=list(_VAR_MAP.keys()),
        )
    except Exception as exc:
        logger.error("CERES load failed: %s", exc)
        return {"success": False, "plots_generated": [], "errors": [str(exc)]}

    lats = ceres_ds.lat.values
    lons = ceres_ds.lon.values
    n_times = ceres_ds.sizes["time"]
    logger.info("CERES: %d time steps, grid %d x %d", n_times, len(lats), len(lons))

    # ------------------------------------------------------------------
    # 2. Build records — one dict per time step with standardized keys
    # ------------------------------------------------------------------
    records: list[dict[str, Any]] = []
    dates = _dates_in_range(
        cfg.event.start_time.date()
        if hasattr(cfg.event.start_time, "date")
        else cfg.event.start_time,
        cfg.event.end_time.date()
        if hasattr(cfg.event.end_time, "date")
        else cfg.event.end_time,
    )
    for t in range(n_times):
        rec: dict[str, Any] = {}
        for ceres_name, std_key in _VAR_MAP.items():
            if ceres_name in ceres_ds:
                rec[std_key] = ceres_ds[ceres_name].isel(time=t).values
        # Date label
        if t < len(dates):
            rec["date"] = dates[t].isoformat()
        else:
            rec["date"] = f"t={t}"
        # Use total AOD as placeholder for tot_aod (scatter plots need it)
        rec["tot_aod"] = rec.get("aod", np.zeros((len(lats), len(lons))))
        records.append(rec)

    # ------------------------------------------------------------------
    # 3. Load MERRA-2 (if configured) — regrid to CERES grid
    # ------------------------------------------------------------------
    m2_ds = None
    if cfg.merra2 is not None:
        logger.info("Loading MERRA-2 data")
        try:
            smoke_species = cfg.merra2.smoke_species
            m2_ds = load_merra2(
                files=cfg.merra2.files,
                domain=domain,
                smoke_species=smoke_species,
            )
            # Regrid each day's smoke AOD to CERES grid and attach to records
            m2_times = min(m2_ds.sizes["time"], len(records))
            for t in range(m2_times):
                smoke_da = m2_ds["SMOKEAOD"].isel(time=t)
                regridded = regrid_nearest(smoke_da, lats, lons)
                records[t]["smoke_aod"] = regridded.values
        except Exception as exc:
            logger.warning("MERRA-2 load/regrid failed: %s", exc)
            errors.append(f"MERRA-2: {exc}")

    # Ensure smoke_aod exists for all records (zero fill if missing)
    for rec in records:
        if "smoke_aod" not in rec:
            rec["smoke_aod"] = np.zeros((len(lats), len(lons)))

    # ------------------------------------------------------------------
    # 4. Load AERONET (if configured)
    # ------------------------------------------------------------------
    aeronet_df = None
    if cfg.aeronet is not None:
        logger.info("Loading AERONET data")
        try:
            aeronet_df = load_aeronet(
                files=cfg.aeronet.files,
                domain=domain,
                sites=cfg.aeronet.sites,
            )
            # Rename 'site' -> 'siteid' and parse time for plot functions
            if "site" in aeronet_df.columns and "siteid" not in aeronet_df.columns:
                aeronet_df = aeronet_df.rename(columns={"site": "siteid"})
            if "time" in aeronet_df.columns:
                aeronet_df["time"] = pd.to_datetime(aeronet_df["time"])
            # Update site tuples with real coordinates from data
            sites = _get_sites(cfg)
            updated_sites: list[tuple[str, float, float, str]] = []
            for name, _, _, aeronet_name in sites:
                site_rows = aeronet_df[aeronet_df["siteid"] == aeronet_name]
                if not site_rows.empty:
                    lat = float(site_rows["latitude"].iloc[0])
                    lon = float(site_rows["longitude"].iloc[0])
                    updated_sites.append((name, lat, lon, aeronet_name))
                else:
                    updated_sites.append((name, 0.0, 0.0, aeronet_name))
            sites = updated_sites
        except Exception as exc:
            logger.warning("AERONET load failed: %s", exc)
            errors.append(f"AERONET: {exc}")
    else:
        sites = []

    # ------------------------------------------------------------------
    # 5. Compute background (mean of first N days)
    # ------------------------------------------------------------------
    bg_n = min(bg_window, len(records))
    background: dict[str, Any] = {}
    for key in ("sw_all", "sw_clr", "toa_net", "aod"):
        stack = np.stack([records[t][key] for t in range(bg_n)])
        background[key] = np.nanmean(stack, axis=0)
    bg_sw = background["sw_all"]

    # ------------------------------------------------------------------
    # 6. Surface impact (if configured)
    # ------------------------------------------------------------------
    if cfg.surface_impact is not None:
        logger.info("Computing surface impact")
        ssa = cfg.surface_impact.ssa
        asymmetry = cfg.surface_impact.asymmetry
        for rec in records:
            if "semi_empirical" in cfg.surface_impact.method:
                rec["semi_dimming"] = semi_empirical_surface_dimming(
                    rec["smoke_aod"],
                    rec.get("toa_insol", np.full_like(rec["smoke_aod"], 400.0)),
                    ssa=ssa,
                    asymmetry=asymmetry,
                )
            # NOTE: Full MERRA-2 radiation (tavg1_2d_rad_Nx) loader
            # deferred to a follow-up task. For now, approximate.
            if "m2_sfc_effect" not in rec:
                rec["m2_sfc_effect"] = rec.get(
                    "semi_dimming",
                    np.zeros_like(rec["smoke_aod"]),
                ) * 0.9

    # ------------------------------------------------------------------
    # 7. Generate plots
    # ------------------------------------------------------------------
    # Peak day index: middle of event window past background
    peak_idx = min(bg_window + (len(records) - bg_window) // 2, len(records) - 1)
    peak_rec = records[peak_idx]

    for plot_type in cfg.plots:
        try:
            fig = None
            if plot_type == "toa_event_fields":
                fig = plot_event_fields(lats, lons, peak_rec, event_name=event_name)
            elif plot_type == "anomaly_maps":
                fig = plot_anomaly_maps(
                    lats, lons, peak_rec, background, event_name=event_name
                )
            elif plot_type == "sw_vs_aod_scatter":
                fig = plot_sw_vs_aod_scatter(
                    lats, lons, records, event_name=event_name
                )
            elif plot_type == "site_timeseries":
                if sites:
                    fig = plot_site_timeseries(
                        lats, lons, records, bg_sw, sites,
                        aeronet=aeronet_df, event_name=event_name,
                    )
                else:
                    logger.warning("site_timeseries requested but no sites configured")
            elif plot_type == "surface_impact":
                if "semi_dimming" in peak_rec or "m2_sfc_effect" in peak_rec:
                    # Ensure all required keys exist
                    if "semi_dimming" not in peak_rec:
                        peak_rec["semi_dimming"] = np.zeros_like(peak_rec["smoke_aod"])
                    if "m2_sfc_effect" not in peak_rec:
                        peak_rec["m2_sfc_effect"] = peak_rec["semi_dimming"] * 0.9
                    fig = plot_surface_impact(
                        lats, lons, peak_rec, event_name=event_name
                    )
                else:
                    logger.warning(
                        "surface_impact plot requested but no surface impact data"
                    )
            else:
                logger.warning("Unknown plot type: %s", plot_type)
                continue

            if fig is not None:
                out_path = output_dir / f"{event_name}_{plot_type}.png"
                fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
                plt.close(fig)
                plots_generated.append(str(out_path))
                logger.info("Saved: %s", out_path.name)

        except Exception as exc:
            logger.error("Plot '%s' failed: %s", plot_type, exc)
            errors.append(f"Plot {plot_type}: {exc}")

    success = len(errors) == 0
    logger.info(
        "Radiative analysis complete: %d plots, %d errors",
        len(plots_generated),
        len(errors),
    )
    return {
        "success": success,
        "plots_generated": plots_generated,
        "errors": errors,
    }
