#!/usr/bin/env python
"""Single-Source Spatial Field Map Example.

Demonstrates the single-source ``spatial`` plotter (SpatialPlotter) for showing
the geographic field of one source's variable. The renderer is shape-aware:
point/site/track data render as scatter, grid/swath data as pcolormesh.

Data: Surface point datasets (O3)
"""

import matplotlib.pyplot as plt
import xarray as xr
from _helpers import create_paired_surface_data, save_figure

from davinci_monet.plots import PlotConfig, SpatialPlotter, build_series


def main():
    """Generate single-source spatial field map example."""
    print("Creating single-source spatial field map...")

    # Create synthetic paired data, then keep a SINGLE source (the x / geometry
    # side) — the single-source spatial plot renders one source's field.
    paired = create_paired_surface_data(n_sites=50, variables=["O3"])
    ds = xr.Dataset(
        {"O3": paired["x_o3"]},
        coords={
            "time": paired["time"],
            "site": paired["site"],
            "latitude": paired["latitude"],
            "longitude": paired["longitude"],
        },
        attrs={"geometry": "point", "source_label": "surface"},
    )
    ds["O3"].attrs["units"] = "ppbv"

    # Render via the unified contract: SpatialPlotter().render(build_series(ds, var)).
    plotter = SpatialPlotter(config=PlotConfig(title="Spatial Field: Surface O3"))
    fig = plotter.render(build_series(ds, "O3"))

    save_figure(fig, "08_spatial_field")
    plt.close(fig)

    print("Done!")


if __name__ == "__main__":
    main()
