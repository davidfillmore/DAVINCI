# DAVINCI-MONET Validation Plan

This document tracks the human validation of DAVINCI-MONET using real observational datasets. Model data is internal to our institutions; observational data sources are documented below.

## Observation Validation Status

| Category | Reader | Status | Validated By | Date | Notes |
|----------|--------|--------|--------------|------|-------|
| **Surface** | AirNow | Complete | D. Fillmore | 2026-01-10 | ASIA-AQ analysis (36 sites, Feb 2024) |
| | AQS | Not Started | | | |
| | AERONET | Complete | D. Fillmore | 2026-01-10 | ASIA-AQ analysis (68 sites, Feb 2024) |
| | OpenAQ | Not Started | | | |
| **Column** | Pandora | Complete | D. Fillmore | 2026-01-10 | ASIA-AQ analysis (13 sites, Feb 2024) |
| **Sonde** | Ozonesonde | Not Started | | | |
| **Aircraft** | ICARTT | Complete | D. Fillmore | 2026-01-10 | ASIA-AQ DC-8 (O3, NO2, CO; 17 flights, Feb 2024) |
| **Satellite L2** | TROPOMI | Not Started | | | Polar; NO2, O3, CO, HCHO |
| | MODIS | Not Started | | | Polar; AOD |
| | TEMPO | Not Started | | | Geo; NO2, O3, HCHO |
| **Satellite L3** | MOPITT | Not Started | | | Polar; CO |
| | OMPS | Not Started | | | Polar; O3 |
| | GOES | Not Started | | | Geo; AOD |

### Observation Acronyms

| Acronym | Full Name |
|---------|-----------|
| AirNow | EPA Real-Time Air Quality Index |
| AQS | EPA Air Quality System |
| AERONET | Aerosol Robotic Network |
| OpenAQ | Open Air Quality (global platform) |
| Pandora | Pandora Global Network (ground-based spectrometers) |
| ICARTT | International Consortium for Atmospheric Research on Transport and Transformation |
| TROPOMI | TROPOspheric Monitoring Instrument (Sentinel-5P) |
| MODIS | Moderate Resolution Imaging Spectroradiometer (Terra/Aqua) |
| TEMPO | Tropospheric Emissions: Monitoring of Pollution |
| MOPITT | Measurements of Pollution in the Troposphere (Terra) |
| OMPS | Ozone Mapping and Profiler Suite (Suomi-NPP) |
| GOES | Geostationary Operational Environmental Satellite |

---

## Model Validation Status

| Model | Reader | Status | Validated By | Date | Notes |
|-------|--------|--------|--------------|------|-------|
| CESM/CAM-chem | cesm_fv | Complete | D. Fillmore | 2026-01-10 | ASIA-AQ analysis (0.1° FV grid, Feb 2024) |
| CESM (SE grid) | cesm_se | Not Started | | | |
| WRF-Chem | wrfchem | Not Started | | | |
| CMAQ | cmaq | Not Started | | | |
| UFS-AQM | ufs | Not Started | | | |
| Generic NetCDF | generic | Complete | D. Fillmore | 2026-01-10 | Used for precomputed NO2 column |

### Model Acronyms

| Acronym | Full Name |
|---------|-----------|
| CESM | Community Earth System Model |
| CAM-chem | Community Atmosphere Model with Chemistry |
| FV | Finite Volume (dynamical core) |
| SE | Spectral Element (dynamical core) |
| WRF-Chem | Weather Research and Forecasting model with Chemistry |
| CMAQ | Community Multiscale Air Quality model |
| UFS-AQM | Unified Forecast System - Air Quality Model |

---

**Status Key:**
- Not Started
- In Progress
- Blocked (with reason)
- Complete

---

## Observational Data Sources

### Surface Observations

#### AirNow (EPA Real-Time Air Quality)
- **Variables:** O3, PM2.5, NO2, CO
- **Coverage:** United States, Canada, Mexico
- **Temporal:** Hourly (real-time), daily (historical)
- **Data Portal:** [AirNow API](https://docs.airnowapi.org/)
- **Alternative:** [EPA Outdoor Air Quality Data](https://www.epa.gov/outdoor-air-quality-data)
- **Notes:** Real-time data is preliminary; use AQS for verified historical data

#### AQS (EPA Air Quality System)
- **Variables:** O3, PM2.5, NO2, SO2, CO
- **Coverage:** United States
- **Temporal:** Hourly, daily, annual summaries (1990-present)
- **Data Portal:** [AQS Download Files](https://aqs.epa.gov/aqsweb/airdata/download_files.html)
- **API:** [AQS Data Mart](https://aqs.epa.gov/aqsweb/documents/data_mart_welcome.html)
- **Notes:** Quality-assured data; 6+ month lag from collection

#### AERONET (Aerosol Robotic Network)
- **Variables:** AOD, Angstrom exponent, precipitable water
- **Coverage:** Global network (500+ stations)
- **Temporal:** ~15-minute intervals
- **Data Portal:** [AERONET Data Download Tool](https://aeronet.gsfc.nasa.gov/cgi-bin/webtool_aod_v3)
- **Main Site:** [AERONET Homepage](https://aeronet.gsfc.nasa.gov/)
- **Notes:** Level 2.0 is quality-assured (12+ month delay); Level 1.5 available in near real-time

#### OpenAQ (Global Air Quality Platform)
- **Variables:** O3, PM2.5, PM10, NO2, SO2, CO
- **Coverage:** Global (180+ countries)
- **Temporal:** Varies by source (typically hourly)
- **Data Portal:** [OpenAQ Explorer](https://explore.openaq.org/)
- **API:** [OpenAQ API v3](https://docs.openaq.org/)
- **Notes:** Aggregates data from government and research sources worldwide; API key required

---

### Column Observations

#### Pandora (Ground-Based Spectrometers)
- **Variables:** Tropospheric NO2 column, total O3 column, HCHO
- **Coverage:** Global network (150+ instruments)
- **Temporal:** ~1-2 minute intervals during daylight
- **Data Portal:** [Pandora Global Network](https://data.pandonia-global-network.org/)
- **Product Info:** [PGN Data Products](https://www.pandonia-global-network.org/home/documents/products/)
- **Notes:** L2 files contain quality flags (0=high, 1=medium, 2=low); use quality_flag ≤ 1 for research

---

### Sonde Observations

#### Ozonesonde (Balloon Profiles)
- **Variables:** O3 vertical profiles, temperature, humidity
- **Coverage:** Global network
- **Temporal:** Weekly launches (varies by station)

**Data Sources:**
- **WOUDC:** [World Ozone and UV Data Centre](https://woudc.org/data/explore.php?dataset=ozonesonde)
- **NOAA GML:** [Boulder Ozonesondes](https://gml.noaa.gov/ozwv/ozsondes/)
- **SHADOZ:** Southern Hemisphere ADditional OZonesondes

**Notes:** WOUDC is the WMO archive; NOAA GML provides U.S. stations

---

### Aircraft Observations

#### ICARTT (NASA/NOAA Flight Campaigns)
- **Variables:** Multiple trace gases (O3, CO, NO2, VOCs, aerosols)
- **Coverage:** Campaign-specific regions
- **Temporal:** Campaign periods

**Data Sources:**
- **ESPO Archive:** [NASA Earth Science Project Office](https://espoarchive.nasa.gov/)
- **Format Specification:** [ICARTT File Format](https://www.earthdata.nasa.gov/esdis/esco/standards-and-practices/icartt-file-format)

**Recent Campaigns:**
- ATom (Atmospheric Tomography)
- FIREX-AQ (Fire Influence on Regional to Global Environments)
- DISCOVER-AQ

**Notes:** Earthdata login required; data in ASCII ICARTT format (.ict files)

---

### Satellite L2 (Swath) Observations

#### TROPOMI (Sentinel-5P)
- **Variables:** NO2, O3, CO, HCHO, SO2, aerosol index
- **Coverage:** Global daily
- **Resolution:** ~5.5 km x 3.5 km (nadir)
- **Data Portal:** [Copernicus Data Space](https://dataspace.copernicus.eu/explore-data/data-collections/sentinel-data/sentinel-5p)
- **Alternative:** [S5P-PAL Data Portal](https://data-portal.s5p-pal.com/)
- **AWS:** [Sentinel-5P on AWS](https://registry.opendata.aws/sentinel5p/)
- **Notes:** Operational since 2018; use OFFL (offline) products for research

#### TEMPO (Tropospheric Emissions Monitoring of Pollution)
- **Variables:** NO2, O3, HCHO
- **Coverage:** North America (hourly daylight)
- **Resolution:** ~10 km² at center of field
- **Data Portal:** [NASA ASDC TEMPO](https://asdc.larc.nasa.gov/project/TEMPO)
- **L2 NO2:** [TEMPO_NO2_L2_V03](https://asdc.larc.nasa.gov/project/TEMPO/TEMPO_NO2_L2_V03)
- **L3 NO2:** [TEMPO_NO2_L3_V01](https://asdc.larc.nasa.gov/project/TEMPO/TEMPO_NO2_L3_V01)
- **Notes:** First geostationary air quality mission; operational since 2023

#### MODIS (Terra/Aqua AOD)
- **Variables:** AOD, Angstrom exponent
- **Coverage:** Global daily
- **Resolution:** 10 km (MOD04_L2), 3 km (MOD04_3K), 1 km (MCD19A2)
- **Data Portal:** [LAADS DAAC](https://ladsweb.modaps.eosdis.nasa.gov/search/)
- **Product Info:** [MODIS Aerosol Product](https://modis.gsfc.nasa.gov/data/dataprod/mod04.php)
- **Notes:** Terra (morning), Aqua (afternoon); requires pyhdf for HDF4 files

---

### Satellite L3 (Gridded) Observations

#### MOPITT (Terra CO)
- **Variables:** CO total column and profiles
- **Coverage:** Global daily
- **Resolution:** 22 km x 22 km
- **Data Portal:** [NASA ASDC MOPITT](https://asdc.larc.nasa.gov/project/MOPITT)
- **L3 Daily:** [MOP03J_9](https://asdc.larc.nasa.gov/project/MOPITT/MOP03J_9)
- **Visualization:** [NASA Earth Observations](https://neo.gsfc.nasa.gov/view.php?datasetId=MOP_CO_M)
- **Notes:** Operational since 2000; Version 9 is current

#### OMPS (Suomi-NPP Total Ozone)
- **Variables:** Total column O3, UV aerosol index
- **Coverage:** Global daily
- **Resolution:** 50 km (nadir), 1° gridded
- **Data Portal:** [NASA Ozone Science Team](https://ozoneaq.gsfc.nasa.gov/data/omps/)
- **Earthdata:** [OMPS NRT Data](https://www.earthdata.nasa.gov/data/instruments/omps)
- **L3 Gridded:** [OMPS-NPP L3 Daily](https://catalog.data.gov/dataset/omps-npp-l3-nm-ozone-o3-total-column-1-0-deg-grid-daily-v2)
- **Notes:** Continues TOMS/OMI record; NRT available within 3 hours

#### GOES (GOES-R/S AOD)
- **Variables:** AOD at 550 nm
- **Coverage:** Western Hemisphere
- **Resolution:** 2 km
- **Temporal:** Full disk every 10 min, CONUS every 5 min
- **Data Portal:** [NOAA Data Catalog](https://data.noaa.gov/dataset/dataset/noaa-goes-r-series-advanced-baseline-imager-abi-level-2-aerosol-optical-depth-aod1)
- **Product Info:** [GOES-R AOD](https://www.goes-r.gov/products/baseline-aerosol-opt-depth.html)
- **Notes:** GOES-16 (East), GOES-18 (West); clear-sky only

---

## Validation Procedures

### For Each Reader

1. **Data Acquisition**
   - Download sample dataset from source
   - Document file format and size
   - Note any access requirements (accounts, API keys)

2. **Reader Testing**
   - Load data using DAVINCI-MONET reader
   - Verify coordinates (lat, lon, time)
   - Check variable names and units
   - Validate against source metadata

3. **Pairing Verification**
   - Pair with appropriate model output
   - Check spatial/temporal alignment
   - Verify paired dataset structure

4. **Statistics and Plots**
   - Generate basic statistics (MB, RMSE, R)
   - Create standard plots (scatter, time series)
   - Compare with published evaluation results if available

5. **Documentation**
   - Update status in table above
   - Note any issues or limitations
   - Record software versions used

---

## Known Limitations

### Reader Dependencies

| Reader | Required Package | Notes |
|--------|-----------------|-------|
| MODIS L2 | pyhdf | HDF4/HDF-EOS format |
| All satellite | monetio | Full functionality requires monetio.sat modules |
| TROPOMI | monetio.sat._tropomi_l2_no2_mm | Falls back to basic xarray without |

### Data Access Requirements

| Source | Authentication |
|--------|---------------|
| AirNow API | API key (free registration) |
| OpenAQ | API key (free registration) |
| NASA Earthdata | Earthdata login |
| Copernicus | Copernicus account |

---

## Validation Notes

### ASIA-AQ Analysis (January 2026)

**Analysis Period:** February 1-29, 2024 (full month, leap year)
**Domain:** 0-45°N, 90-140°E (East and Southeast Asia)
**Model:** CESM/CAM-chem at 0.1° resolution (f.e3b06m.FCnudged.t6s.01x01.01)

#### Observations Validated

**AirNow (Surface)**
- 36 sites in Asia (US Embassy/Consulate monitors)
- Variables: PM2.5 (1,008 pairs), O3 (152), NO2 (54), CO (133)
- Reader: `davinci_monet.observations.surface.airnow`
- Download: monetio integration works correctly
- Pairing: Point-to-grid strategy successful
- Issues: None

**AERONET (Surface AOD)**
- 68 sites in domain
- Variables: AOD at 500nm (8,150 pairs)
- Reader: `davinci_monet.observations.surface.aeronet`
- Download: monetio integration, latlonbox order is [lat_min, lon_min, lat_max, lon_max]
- Product: Level 1.5 (near real-time)
- Issues: None

**Pandora (Column NO2)**
- 13 sites in domain (Korea, Thailand, Laos, Malaysia, Singapore, Palau)
- Variables: Tropospheric NO2 column (8,886 pairs)
- Reader: `davinci_monet.observations.surface.pandora`
- L2 file parsing with quality filtering (flags 0-1 accepted)
- Solar zenith angle filtering (< 80°)
- Issues: None; reader handles varied L2 file formats correctly

#### Model Validated

**CESM/CAM-chem (cesm_fv reader)**
- 0.1° finite volume grid
- Hourly output (696 files for Feb 2024)
- Variables tested: PM25, O3, NO2, CO, AODVISdn
- Unit conversions verified: mol/mol → ppb (×1e9), kg/kg → μg/m³ (×1.2e9)
- Vertical coordinate handling: hybrid sigma-pressure, surface at lev index 0 after processing
- NO2 column integration: `compute_tropospheric_column()` function validated against Pandora

#### Statistics Summary

| Variable | N | R | NMB |
|----------|---|---|-----|
| PM2.5 | 1,008 | 0.21 | +29% |
| O3 | 152 | 0.48 | -45% |
| NO2 (sfc) | 54 | 0.43 | +150% |
| CO | 133 | 0.07 | -27% |
| AOD | 8,150 | 0.51 | -46% |
| NO2 column | 8,886 | 0.57 | +86% |

#### DC-8 Aircraft Observations Validated

**ICARTT (DC-8 Aircraft)**
- 17 flights in February 2024
- Variables: O3 (ROZE), NO2 (CANOE), CO (DACOM)
- Reader: `davinci_monet.observations.aircraft.icartt`
- Data: 60-second merged files from NASA ESPO archive
- Pairing: Track-to-grid strategy (vectorized extraction)
- Flight coordinate added automatically based on date
- Issues: None; reader handles campaign-specific variable naming

| Variable | N | Mean Obs | Mean Model | R | NMB |
|----------|---|----------|------------|-----|------|
| O3 (ROZE) | 3,248 | 37.7 ppb | 50.9 ppb | 0.28 | +35% |
| NO2 (CANOE) | 3,255 | 0.98 ppb | 6.15 ppb | 0.43 | +529% |
| CO (DACOM) | 3,244 | 169 ppb | 187 ppb | 0.32 | +10% |

#### Pipeline Components Validated

- `load_models` stage: CESM reader, unit scaling
- `load_observations` stage: AirNow, AERONET, Pandora, ICARTT readers
- `pairing` stage: Point-to-grid strategy, Track-to-grid strategy
- `statistics` stage: All standard metrics (N, MB, RMSE, R, NMB, NME, IOA)
- `plotting` stage: scatter, timeseries, spatial_bias, site_timeseries, flight_timeseries
- `save_results` stage: CSV output

---

## References

- EPA Air Quality Data: https://www.epa.gov/outdoor-air-quality-data
- NASA Earthdata: https://www.earthdata.nasa.gov/
- Copernicus Data Space: https://dataspace.copernicus.eu/
- WOUDC: https://woudc.org/
- NOAA GML: https://gml.noaa.gov/
- Pandora Global Network: https://www.pandonia-global-network.org/
