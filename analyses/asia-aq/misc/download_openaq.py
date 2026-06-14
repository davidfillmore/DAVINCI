#!/usr/bin/env python
"""
Download OpenAQ data for the ASIA-AQ domain.

Domain: 0-45°N, 90-140°E
Period: February 1-3, 2024
Countries: TH, PH, KR, TW, VN, MY, ID, JP, CN, HK, SG, MM, LA, KH

REQUIREMENTS:
    OpenAQ API v3 requires an API key.
    1. Register at: https://docs.openaq.org/using-the-api/api-key
    2. Set environment variable: export OPENAQ_API_KEY=your_key

Usage:
    export OPENAQ_API_KEY=your_key
    python download_openaq.py
"""

from pathlib import Path
import os
import pandas as pd
import xarray as xr
from datetime import datetime, timedelta

# Output directory
DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

# ASIA-AQ domain
BBOX = {
    "lat_min": 0,
    "lat_max": 45,
    "lon_min": 90,
    "lon_max": 140,
}

# Countries in ASIA-AQ domain
COUNTRIES = [
    "TH",  # Thailand
    "PH",  # Philippines
    "KR",  # South Korea
    "TW",  # Taiwan
    "VN",  # Vietnam
    "MY",  # Malaysia
    "ID",  # Indonesia
    "JP",  # Japan
    "CN",  # China
    "HK",  # Hong Kong
    "SG",  # Singapore
    "MM",  # Myanmar
    "LA",  # Laos
    "KH",  # Cambodia
]

# Parameters to download
PARAMETERS = ["o3", "pm25", "no2", "co", "so2"]

# Date range (ASIA-AQ campaign period sample)
START_DATE = "2024-02-01"
END_DATE = "2024-02-03"


def download_openaq_monetio():
    """Download OpenAQ data using monetio openaq_v3."""
    try:
        from monetio.geometry import openaq_v3 as openaq
    except ImportError:
        print("ERROR: monetio not installed. Install with: pip install monetio")
        return None

    print(f"Downloading OpenAQ data for {START_DATE} to {END_DATE}")
    print(f"Domain: {BBOX['lat_min']}-{BBOX['lat_max']}N, {BBOX['lon_min']}-{BBOX['lon_max']}E")
    print(f"Parameters: {', '.join(PARAMETERS)}")
    print(f"Countries: {', '.join(COUNTRIES)}")
    print()

    all_data = []

    # Download by country to avoid rate limits
    for country in COUNTRIES:
        print(f"  Fetching {country}...", end=" ", flush=True)
        try:
            df = openaq.add_data(
                dates=pd.date_range(START_DATE, END_DATE, freq="D"),
                country=country,
                parameters=PARAMETERS,
                wide_fmt=True,
            )
            if df is not None and not df.empty:
                # Filter to bbox
                lat_col = "latitude" if "latitude" in df.columns else "lat"
                lon_col = "longitude" if "longitude" in df.columns else "lon"

                if lat_col in df.columns and lon_col in df.columns:
                    mask = (
                        (df[lat_col] >= BBOX["lat_min"]) &
                        (df[lat_col] <= BBOX["lat_max"]) &
                        (df[lon_col] >= BBOX["lon_min"]) &
                        (df[lon_col] <= BBOX["lon_max"])
                    )
                    df = df[mask]

                if not df.empty:
                    all_data.append(df)
                    print(f"{len(df)} records")
                else:
                    print("0 records (outside bbox)")
            else:
                print("no data")
        except Exception as e:
            print(f"error: {e}")

    if not all_data:
        print("\nNo data retrieved!")
        return None

    # Combine all data
    print("\nCombining data...")
    df = pd.concat(all_data, ignore_index=True)
    print(f"Total records: {len(df)}")

    # Show summary
    lat_col = "latitude" if "latitude" in df.columns else "lat"
    lon_col = "longitude" if "longitude" in df.columns else "lon"

    print("\nData summary:")
    if "time" in df.columns:
        print(f"  Time range: {df['time'].min()} to {df['time'].max()}")
    print(f"  Lat range: {df[lat_col].min():.2f} to {df[lat_col].max():.2f}")
    print(f"  Lon range: {df[lon_col].min():.2f} to {df[lon_col].max():.2f}")

    # Show available parameters (in wide format, parameters are columns)
    param_cols = [c for c in PARAMETERS if c in df.columns]
    print(f"\nAvailable parameters: {param_cols}")
    for col in param_cols:
        valid = df[col].notna().sum()
        print(f"  {col}: {valid} valid values")

    if "country" in df.columns:
        print("\nRecords by country:")
        print(df["country"].value_counts())

    return df


def dataframe_to_dataset(df: pd.DataFrame) -> xr.Dataset:
    """Convert OpenAQ DataFrame to xarray Dataset suitable for pairing."""
    if df is None or df.empty:
        return None

    # Pivot to get parameters as variables
    # Expected columns: time, location, latitude, longitude, parameter, value, unit

    # Get unique locations with their coordinates
    if "location" not in df.columns:
        if "siteid" in df.columns:
            df = df.rename(columns={"siteid": "location"})
        elif "location_id" in df.columns:
            df = df.rename(columns={"location_id": "location"})

    # Ensure we have required columns
    required = ["time", "latitude", "longitude"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        # Try alternate names
        renames = {
            "datetime": "time",
            "date": "time",
            "lat": "latitude",
            "lon": "longitude",
        }
        for source_name, target_name in renames.items():
            if source_name in df.columns and target_name not in df.columns:
                df = df.rename(columns={source_name: target_name})

    # Check again
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"WARNING: Missing required columns: {missing}")
        print(f"Available columns: {list(df.columns)}")
        return None

    # Convert time
    df["time"] = pd.to_datetime(df["time"])

    # Create site identifier if not present
    if "location" not in df.columns:
        df["location"] = df.apply(
            lambda r: f"{r['latitude']:.3f}_{r['longitude']:.3f}", axis=1
        )

    # Get unique sites
    sites = df.groupby("location").first()[["latitude", "longitude"]].reset_index()

    # Pivot parameters to columns
    if "parameter" in df.columns and "value" in df.columns:
        # Pivot each parameter to a column
        pivoted = df.pivot_table(
            index=["time", "location"],
            columns="parameter",
            values="value",
            aggfunc="mean",
        ).reset_index()
    else:
        pivoted = df

    # Create xarray dataset
    pivoted = pivoted.set_index(["time", "location"])
    ds = pivoted.to_xarray()

    # Add site coordinates
    site_lats = sites.set_index("location")["latitude"]
    site_lons = sites.set_index("location")["longitude"]

    ds = ds.assign_coords(
        latitude=("location", [site_lats.get(loc, float("nan")) for loc in ds.location.values]),
        longitude=("location", [site_lons.get(loc, float("nan")) for loc in ds.location.values]),
    )

    # Add attributes
    ds.attrs["source"] = "OpenAQ"
    ds.attrs["domain"] = f"{BBOX['lat_min']}-{BBOX['lat_max']}N, {BBOX['lon_min']}-{BBOX['lon_max']}E"
    ds.attrs["geometry"] = "point"

    return ds


def main():
    """Main download routine."""
    print("=" * 60)
    print("OpenAQ Data Download for ASIA-AQ")
    print("=" * 60)
    print()

    # Check for API key
    api_key = os.environ.get("OPENAQ_API_KEY")
    if not api_key:
        print("ERROR: OPENAQ_API_KEY environment variable not set.")
        print()
        print("OpenAQ API v3 requires authentication. To get an API key:")
        print("  1. Register at: https://docs.openaq.org/using-the-api/api-key")
        print("  2. Set the environment variable:")
        print("     export OPENAQ_API_KEY=your_key_here")
        print()
        print("Then re-run this script.")
        return

    print(f"API key found: {api_key[:8]}...")
    print()

    # Download data
    df = download_openaq_monetio()

    if df is None or df.empty:
        print("\nNo data to save.")
        return

    # Save raw CSV
    csv_file = DATA_DIR / f"openaq_asiaq_{START_DATE}_{END_DATE}.csv"
    df.to_csv(csv_file, index=False)
    print(f"\nSaved CSV: {csv_file}")

    # Convert to xarray and save NetCDF
    print("\nConverting to xarray Dataset...")
    ds = dataframe_to_dataset(df)

    if ds is not None:
        nc_file = DATA_DIR / f"openaq_asiaq_{START_DATE}_{END_DATE}.nc"
        ds.to_netcdf(nc_file)
        print(f"Saved NetCDF: {nc_file}")

        print("\nDataset summary:")
        print(ds)
    else:
        print("Could not convert to xarray format.")

    print("\n" + "=" * 60)
    print("Download complete.")


if __name__ == "__main__":
    main()
