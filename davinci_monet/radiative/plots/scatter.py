"""SW flux vs AOD scatter plots with shared date colorbar.

Produces a 1x2 panel figure: total AOD scatter and smoke AOD scatter,
both color-coded by date with a shared colorbar.
"""

from __future__ import annotations

from typing import Any

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.figure import Figure

from davinci_monet.plots.style import NCAR_COLORS


def plot_sw_vs_aod_scatter(
    lats: np.ndarray,
    lons: np.ndarray,
    records: list[dict[str, Any]],
    event_name: str = "",
) -> Figure:
    """Plot SW vs AOD scatter (2-panel) with shared date colorbar.

    Parameters
    ----------
    lats
        1-D latitude array.
    lons
        1-D longitude array.
    records
        List of dicts, each with: date, tot_aod, smoke_aod, sw_all.
    event_name
        Optional event label for the figure title.

    Returns
    -------
    Figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    n_days = len(records)
    cmap = plt.get_cmap("coolwarm")
    norm = Normalize(vmin=0, vmax=max(n_days - 1, 1))
    dates = [r["date"] for r in records]

    # --- Panel 1: Total AOD vs SW ---
    ax = axes[0]
    all_tot = []
    all_sw_tot = []
    for i, rec in enumerate(records):
        tot_flat = rec["tot_aod"].ravel()
        sw_flat = rec["sw_all"].ravel()
        ax.scatter(
            tot_flat,
            sw_flat,
            s=4,
            alpha=0.3,
            color=cmap(norm(i)),
            edgecolors="none",
        )
        mask = np.isfinite(tot_flat) & np.isfinite(sw_flat)
        all_tot.append(tot_flat[mask])
        all_sw_tot.append(sw_flat[mask])

    all_tot_arr = np.concatenate(all_tot)
    all_sw_tot_arr = np.concatenate(all_sw_tot)
    valid = np.isfinite(all_tot_arr) & np.isfinite(all_sw_tot_arr)
    if valid.sum() > 2 and np.std(all_tot_arr[valid]) > 0:
        r_val = np.corrcoef(all_tot_arr[valid], all_sw_tot_arr[valid])[0, 1]
    else:
        r_val = 0.0
    ax.set_title(f"r = {r_val:.3f}, n = {valid.sum():,}")
    ax.set_xlabel("MERRA-2 Total AOD 550 nm")
    ax.set_ylabel("CERES TOA SW Reflected (W/m\u00b2)")
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 400)

    # --- Panel 2: Smoke AOD vs SW ---
    ax = axes[1]
    all_smoke = []
    all_sw_smoke = []
    for i, rec in enumerate(records):
        smoke_flat = rec["smoke_aod"].ravel()
        sw_flat = rec["sw_all"].ravel()
        ax.scatter(
            smoke_flat,
            sw_flat,
            s=4,
            alpha=0.3,
            color=cmap(norm(i)),
            edgecolors="none",
        )
        mask = np.isfinite(smoke_flat) & np.isfinite(sw_flat)
        all_smoke.append(smoke_flat[mask])
        all_sw_smoke.append(sw_flat[mask])

    all_smoke_arr = np.concatenate(all_smoke)
    all_sw_smoke_arr = np.concatenate(all_sw_smoke)
    valid_s = np.isfinite(all_smoke_arr) & np.isfinite(all_sw_smoke_arr)
    if valid_s.sum() > 2 and np.std(all_smoke_arr[valid_s]) > 0:
        r_val_s = np.corrcoef(all_smoke_arr[valid_s], all_sw_smoke_arr[valid_s])[0, 1]
    else:
        r_val_s = 0.0
    ax.set_title(f"r = {r_val_s:.3f}, n = {valid_s.sum():,}")
    ax.set_xlabel("MERRA-2 Smoke AOD (OC+BC) 550 nm")
    ax.set_ylabel("CERES TOA SW Reflected (W/m\u00b2)")
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 400)

    # --- Shared colorbar ---
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes.tolist(), shrink=0.8, pad=0.02)
    tick_indices = list(range(0, n_days, 2)) if n_days > 2 else list(range(n_days))
    cbar.set_ticks(tick_indices)
    tick_labels = []
    for idx in tick_indices:
        d = dates[idx]
        if hasattr(d, "strftime"):
            tick_labels.append(d.strftime("%b %d"))
        else:
            tick_labels.append(str(d))
    cbar.set_ticklabels(tick_labels)

    suptitle = (
        f"CERES TOA SW vs MERRA-2 AOD \u2014 {event_name}"
        if event_name
        else "CERES TOA SW vs MERRA-2 AOD"
    )
    fig.suptitle(suptitle, fontsize=16, color=NCAR_COLORS["space"])
    fig.subplots_adjust(left=0.07, right=0.88, top=0.90, bottom=0.12, wspace=0.25)
    return fig
