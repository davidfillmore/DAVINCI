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
from davinci_monet.radiative.plots.anomaly_maps import plot_anomaly_maps
from davinci_monet.radiative.plots.daily_correlation import plot_daily_correlation
from davinci_monet.radiative.plots.event_fields import plot_event_fields
from davinci_monet.radiative.plots.method_comparison import plot_method_comparison

# RT level comparison imports (lazy-loaded in runner body)
from davinci_monet.radiative.plots.rt_efficiency import plot_rt_efficiency
from davinci_monet.radiative.plots.rt_scatter import plot_rt_scatter
from davinci_monet.radiative.plots.rt_spatial import plot_rt_spatial
from davinci_monet.radiative.plots.rt_timeseries import plot_rt_timeseries
from davinci_monet.radiative.plots.scatter import plot_sw_vs_aod_scatter
from davinci_monet.radiative.plots.site_timeseries import plot_site_timeseries
from davinci_monet.radiative.plots.spatial_comparison import plot_spatial_comparison
from davinci_monet.radiative.plots.surface_dimming_timeseries import (
    plot_surface_dimming_timeseries,
)
from davinci_monet.radiative.plots.surface_flux import plot_surface_flux
from davinci_monet.radiative.plots.surface_impact import plot_surface_impact
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
    "init_all_sfc_sw_dn": "sfc_sw_dn",
    "init_all_sfc_lw_dn": "sfc_lw_dn",
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
    """Build site list from config.

    Uses SiteConfig entries if available (with display names and coordinates).
    Falls back to AERONET site names with placeholder coordinates.
    """
    if cfg.sites is not None:
        return [(s.name, s.latitude, s.longitude, s.aeronet_id) for s in cfg.sites]
    if cfg.aeronet is not None and cfg.aeronet.sites is not None:
        return [(s, 0.0, 0.0, s) for s in cfg.aeronet.sites]
    return []


def _find_peak_index(cfg: RadiativeConfig, records: list[dict[str, Any]]) -> int:
    """Determine the peak day index from config or auto-select."""
    if cfg.event.peak_date is not None:
        peak = cfg.event.peak_date
        for i, rec in enumerate(records):
            if rec["date"] == peak:
                return i
        logger.warning("peak_date %s not found in records, auto-selecting", peak)

    # Auto: pick day with highest mean AOD
    best_idx = 0
    best_aod = -1.0
    for i, rec in enumerate(records):
        mean_aod = float(np.nanmean(rec.get("aod", np.zeros(1))))
        if mean_aod > best_aod:
            best_aod = mean_aod
            best_idx = i
    return best_idx


def _bg_description(cfg: RadiativeConfig, dates: list[date]) -> str:
    """Build a human-readable background window description."""
    n = min(cfg.event.background_window, len(dates))
    if n == 0:
        return "Pre-Event Mean"
    first = dates[0]
    last = dates[n - 1]
    if first.month == last.month:
        return f"{first.strftime('%b')} {first.day}\u2013{last.day} Mean"
    return f"{first.strftime('%b %-d')}\u2013{last.strftime('%b %-d')} Mean"


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
    start_dt = cfg.event.start_time
    end_dt = cfg.event.end_time
    start_d = start_dt.date() if hasattr(start_dt, "date") else start_dt
    end_d = end_dt.date() if hasattr(end_dt, "date") else end_dt
    dates = _dates_in_range(start_d, end_d)

    records: list[dict[str, Any]] = []
    for t in range(n_times):
        rec: dict[str, Any] = {}
        for ceres_name, std_key in _VAR_MAP.items():
            if ceres_name in ceres_ds:
                rec[std_key] = ceres_ds[ceres_name].isel(time=t).values
        rec["date"] = dates[t] if t < len(dates) else date(2000, 1, 1)
        rec["tot_aod"] = rec.get("aod", np.zeros((len(lats), len(lons))))
        records.append(rec)

    # ------------------------------------------------------------------
    # 3. Load MERRA-2 (if configured) — regrid to CERES grid
    # ------------------------------------------------------------------
    if cfg.merra2 is not None:
        logger.info("Loading MERRA-2 data")
        try:
            m2_ds = load_merra2(
                files=cfg.merra2.files,
                domain=domain,
                smoke_species=cfg.merra2.smoke_species,
            )
            m2_times = min(m2_ds.sizes["time"], len(records))
            for t in range(m2_times):
                day_m2 = m2_ds.isel(time=t)
                records[t]["smoke_aod"] = regrid_nearest(day_m2["SMOKEAOD"], lats, lons).values
                if "TOTEXTTAU" in day_m2:
                    records[t]["tot_aod"] = regrid_nearest(day_m2["TOTEXTTAU"], lats, lons).values
        except Exception as exc:
            logger.warning("MERRA-2 load/regrid failed: %s", exc)
            errors.append(f"MERRA-2: {exc}")

    for rec in records:
        if "smoke_aod" not in rec:
            rec["smoke_aod"] = np.zeros((len(lats), len(lons)))

    # ------------------------------------------------------------------
    # 4. Load AERONET (if configured)
    # ------------------------------------------------------------------
    aeronet_df = None
    sites = _get_sites(cfg)

    if cfg.aeronet is not None:
        logger.info("Loading AERONET data")
        try:
            aeronet_df = load_aeronet(
                files=cfg.aeronet.files,
                domain=domain,
                sites=cfg.aeronet.sites,
            )
            if "time" in aeronet_df.columns:
                aeronet_df["time"] = pd.to_datetime(aeronet_df["time"])
            # If sites have placeholder coords (0,0), fill from AERONET data
            if sites and sites[0][1] == 0.0:
                site_col = "site" if "site" in aeronet_df.columns else "siteid"
                updated: list[tuple[str, float, float, str]] = []
                for name, _, _, aeronet_id in sites:
                    rows = aeronet_df[aeronet_df[site_col] == aeronet_id]
                    if not rows.empty:
                        lat = float(rows["latitude"].iloc[0])
                        lon = float(rows["longitude"].iloc[0])
                        updated.append((name, lat, lon, aeronet_id))
                    else:
                        updated.append((name, 0.0, 0.0, aeronet_id))
                sites = updated
        except Exception as exc:
            logger.warning("AERONET load failed: %s", exc)
            errors.append(f"AERONET: {exc}")

    # ------------------------------------------------------------------
    # 5. Compute background (mean of first N days)
    # ------------------------------------------------------------------
    bg_n = min(bg_window, len(records))
    background: dict[str, Any] = {}
    for key in ("sw_all", "sw_clr", "toa_net", "aod", "sfc_sw_dn", "sfc_lw_dn"):
        if key in records[0]:
            background[key] = np.nanmean(np.stack([records[t][key] for t in range(bg_n)]), axis=0)
    bg_sw = background.get("sw_all")
    bg_desc = _bg_description(cfg, dates)

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
            # NOTE: Full MERRA-2 radiation (tavg1_2d_rad_Nx) loader deferred.
            if "m2_sfc_effect" not in rec:
                rec["m2_sfc_effect"] = (
                    rec.get("semi_dimming", np.zeros_like(rec["smoke_aod"])) * 0.9
                )

    # ------------------------------------------------------------------
    # 6b. MERRA-2 radiation + RT level comparison (if configured)
    # ------------------------------------------------------------------
    if cfg.merra2_rad is not None:
        logger.info("Loading MERRA-2 radiation data for RT level comparison")
        try:
            from davinci_monet.radiative.loaders.merra2_rad import load_merra2_rad
            from davinci_monet.radiative.processing_rt_levels import compute_rt_levels
            from davinci_monet.radiative.rt import daily_mean_coszen

            m2_rad_ds = load_merra2_rad(
                files=cfg.merra2_rad.files,
                domain=domain,
            )
            m2_lats = m2_rad_ds.lat.values
            m2_lons = m2_rad_ds.lon.values

            # Also reload MERRA-2 aerosol on native grid for RT levels
            assert cfg.merra2 is not None  # validated by config
            m2_aer_ds = load_merra2(
                files=cfg.merra2.files,
                domain=domain,
                smoke_species=cfg.merra2.smoke_species,
            )

            import xarray as xr

            doy_start = dates[0].timetuple().tm_yday
            n_rt = min(
                m2_rad_ds.sizes["time"],
                m2_aer_ds.sizes["time"],
                len(records),
            )
            for t in range(n_rt):
                rec = records[t]
                day_rad = m2_rad_ds.isel(time=t)
                day_aer = m2_aer_ds.isel(time=t)

                smoke_aod = day_aer["SMOKEAOD"].values
                s0 = day_rad["SWGDNCLR"].values
                albedo = np.clip(day_rad["ALBEDO"].values, 0.01, 0.99)

                lat2d = np.broadcast_to(m2_lats[:, None], smoke_aod.shape)
                mu_bar = daily_mean_coszen(lat2d, doy_start + t)

                rec["m2_truth"] = day_rad["m2_sfc_effect"].values
                rec["m2_lats"] = m2_lats
                rec["m2_lons"] = m2_lons
                rec["smoke_aod_m2"] = smoke_aod
                rec["levels"] = compute_rt_levels(
                    smoke_aod,
                    s0,
                    mu_bar,
                    albedo,
                )

                # Update CERES-grid m2_sfc_effect with real radiation data
                rec["m2_sfc_effect"] = regrid_nearest(
                    xr.DataArray(
                        day_rad["m2_sfc_effect"].values,
                        dims=["lat", "lon"],
                        coords={"lat": m2_lats, "lon": m2_lons},
                    ),
                    lats,
                    lons,
                ).values
        except Exception as exc:
            logger.warning("MERRA-2 radiation / RT levels failed: %s", exc)
            errors.append(f"MERRA-2 radiation: {exc}")

    # ------------------------------------------------------------------
    # 7. Determine peak day
    # ------------------------------------------------------------------
    peak_idx = _find_peak_index(cfg, records)
    peak_rec = records[peak_idx]
    logger.info("Peak day: %s (index %d)", peak_rec["date"], peak_idx)

    # ------------------------------------------------------------------
    # 8. Generate plots
    # ------------------------------------------------------------------
    ssa = cfg.surface_impact.ssa if cfg.surface_impact else 0.92

    for plot_type in cfg.plots:
        try:
            fig = _dispatch_plot(
                plot_type,
                cfg,
                lats,
                lons,
                records,
                peak_rec,
                background,
                bg_sw,
                bg_desc,
                sites,
                aeronet_df,
                event_name,
                ssa,
            )
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


def _dispatch_plot(
    plot_type: str,
    cfg: RadiativeConfig,
    lats: np.ndarray,
    lons: np.ndarray,
    records: list[dict[str, Any]],
    peak_rec: dict[str, Any],
    background: dict[str, Any],
    bg_sw: np.ndarray | None,
    bg_desc: str,
    sites: list[tuple[str, float, float, str]],
    aeronet_df: pd.DataFrame | None,
    event_name: str,
    ssa: float,
) -> plt.Figure | None:
    """Dispatch to the correct plot function."""
    if plot_type == "toa_event_fields":
        return plot_event_fields(lats, lons, peak_rec, event_name=event_name)

    if plot_type == "anomaly_maps":
        return plot_anomaly_maps(
            lats,
            lons,
            peak_rec,
            background,
            event_name=event_name,
            bg_description=bg_desc,
        )

    if plot_type == "surface_flux":
        return plot_surface_flux(lats, lons, peak_rec, background, event_name=event_name)

    if plot_type == "sw_vs_aod_scatter":
        return plot_sw_vs_aod_scatter(lats, lons, records, event_name=event_name)

    if plot_type == "daily_correlation":
        return plot_daily_correlation(records, event_name=event_name)

    if plot_type == "spatial_comparison":
        if bg_sw is not None:
            return plot_spatial_comparison(
                lats,
                lons,
                peak_rec,
                bg_sw,
                event_name=event_name,
            )
        logger.warning("spatial_comparison needs background SW")
        return None

    if plot_type == "site_timeseries":
        if sites and bg_sw is not None:
            return plot_site_timeseries(
                lats,
                lons,
                records,
                bg_sw,
                sites,
                aeronet=aeronet_df,
                event_name=event_name,
            )
        logger.warning("site_timeseries needs sites and background SW")
        return None

    if plot_type == "surface_impact":
        return plot_surface_impact(
            lats,
            lons,
            peak_rec,
            event_name=event_name,
            ssa=ssa,
        )

    if plot_type == "surface_dimming_timeseries":
        if sites:
            return plot_surface_dimming_timeseries(
                lats,
                lons,
                records,
                sites,
                event_name=event_name,
            )
        logger.warning("surface_dimming_timeseries needs sites")
        return None

    if plot_type == "method_comparison":
        return plot_method_comparison(records, event_name=event_name)

    if plot_type == "rt_efficiency":
        if "levels" in peak_rec:
            return plot_rt_efficiency(records, event_name=event_name)
        logger.warning("rt_efficiency needs merra2_rad config")
        return None

    if plot_type == "rt_scatter":
        if "levels" in peak_rec:
            return plot_rt_scatter(records, event_name=event_name)
        logger.warning("rt_scatter needs merra2_rad config")
        return None

    if plot_type == "rt_timeseries":
        if "levels" in peak_rec and sites:
            m2_lats = peak_rec.get("m2_lats", lats)
            m2_lons = peak_rec.get("m2_lons", lons)
            return plot_rt_timeseries(
                m2_lats,
                m2_lons,
                records,
                sites,
                event_name=event_name,
            )
        logger.warning("rt_timeseries needs merra2_rad config and sites")
        return None

    if plot_type == "rt_spatial":
        if "levels" in peak_rec:
            m2_lats = peak_rec.get("m2_lats", lats)
            m2_lons = peak_rec.get("m2_lons", lons)
            return plot_rt_spatial(
                m2_lats,
                m2_lons,
                peak_rec,
                event_name=event_name,
            )
        logger.warning("rt_spatial needs merra2_rad config")
        return None

    logger.warning("Unknown plot type: %s", plot_type)
    return None
