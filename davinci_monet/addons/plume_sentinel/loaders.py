from __future__ import annotations
from pathlib import Path
from typing import Any
import numpy as np
import xarray as xr


def load_input(spec: dict[str, Any]) -> Any:
    """Dispatch to an add-on input loader by spec["type"]."""
    input_type = spec["type"]
    if input_type == "goes_truecolor":
        return _load_goes(spec)
    elif input_type == "hms_smoke":
        return _load_hms(spec)
    elif input_type == "modis_l2_aod":
        return _load_modis_aod(spec)
    else:
        raise ValueError(f"Unknown input type: {input_type!r}")


def _load_goes(spec: dict[str, Any]) -> xr.Dataset:
    path = Path(spec["file"])
    if not path.exists():
        raise FileNotFoundError(f"GOES file not found: {path}")
    return xr.open_dataset(path)


def _load_hms(spec: dict[str, Any]) -> Any:
    import geopandas as gpd

    path = Path(spec["file"])
    if not path.exists():
        raise FileNotFoundError(f"HMS shapefile not found: {path}")
    return gpd.read_file(path)


def _load_modis_aod(spec: dict[str, Any]) -> dict[str, np.ndarray]:
    from pyhdf.SD import SD, SDC

    files = spec.get("files", [])
    variable = spec["variable"]
    valid_range = spec.get("valid_range", [0.0, 5.0])
    all_lat, all_lon, all_data = [], [], []
    for fpath in files:
        p = Path(fpath)
        if not p.exists():
            raise FileNotFoundError(f"MODIS granule not found: {p}")
        f = SD(str(p), SDC.READ)
        lat = f.select("Latitude")[:]
        lon = f.select("Longitude")[:]
        sds = f.select(variable)
        raw = sds[:].astype(np.float64)
        attrs = sds.attributes()
        scale = attrs.get("scale_factor", 1.0)
        offset = attrs.get("add_offset", 0.0)
        fill = attrs.get("_FillValue", -9999)
        data = np.where(raw != fill, raw * scale + offset, np.nan)
        f.end()
        vmin, vmax = valid_range
        data = np.where((data >= vmin) & (data <= vmax), data, np.nan)
        all_lat.append(lat.ravel())
        all_lon.append(lon.ravel())
        all_data.append(data.ravel())
    return {
        "latitude": np.concatenate(all_lat),
        "longitude": np.concatenate(all_lon),
        "data": np.concatenate(all_data),
        "granule_files": files,
    }
