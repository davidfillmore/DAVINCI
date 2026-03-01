# ASIA-AQ: Target Application for DAVINCI

The **Airborne and Satellite Investigation of Asian Air Quality (ASIA-AQ)** is the target field campaign for DAVINCI's first real-world application. This document summarizes the campaign and its relevance to our modeling goals.

See `ACRONYMS.md` for acronym definitions used in this document.

## Campaign Overview

ASIA-AQ was an international cooperative field study jointly sponsored by NASA and the National Institute of Environmental Research (NIER) of South Korea ([ASDC data release announcement](https://asdc.larc.nasa.gov/news/asia-aq-data-release-announcement)). It deployed multiple aircraft, ground networks, and satellite assets to investigate air quality across Asia.

- **Dates:** 2024-01-29 to 2024-04-01 (study dates; [CASEI campaign summary](https://impact.earthdata.nasa.gov/casei/campaign/ASIA-AQ)). Planning documents proposed a January-March 2024 window to target wintertime PM2.5 ([ASIA-AQ white paper](https://espo.nasa.gov/document/ASIA-AQ_White_Paper)).
- **Regions:** Philippines, South Korea, Taiwan, Thailand ([ASDC data release announcement](https://asdc.larc.nasa.gov/news/asia-aq-data-release-announcement)). Planning scope included Japan, Vietnam, Malaysia, Singapore, Bangladesh, India; as of July 2023 negotiations focused on South Korea, Taiwan, Philippines, Thailand, Malaysia ([ASIA-AQ white paper](https://espo.nasa.gov/document/ASIA-AQ_White_Paper)).
- **PI:** James Crawford (PI), Laura Judd (Co-PI), Barry Lefer (Program Scientist) ([ASIA-AQ Code of Conduct](https://espo.nasa.gov/asia-aq/content/ASIA-AQ_ASIA-AQ_Code_of_Conduct)).
- **Heritage:** Builds on KORUS-AQ (Korea, 2016) and DISCOVER-AQ campaigns ([ASIA-AQ white paper](https://espo.nasa.gov/document/ASIA-AQ_White_Paper))

## Science Objectives

Five primary science themes ([ASIA-AQ white paper](https://espo.nasa.gov/document/ASIA-AQ_White_Paper)):

1. **Satellite Validation and Interpretation** -- Validate retrievals from GEMS (South Korea's Geostationary Environment Monitoring Spectrometer), TEMPO (NASA, North America), and Sentinel-4 (ESA, Europe). GEMS provides hourly UV-vis retrievals of O3, NO2, SO2, HCHO, CHOCHO, and aerosols with near-Seoul resolution of ~7 x 7.7 km for gases and ~3.5 x 7.7 km for aerosols ([GEMS product and resolution overview](https://acp.copernicus.org/articles/24/8943/2024/)). TEMPO provides hourly daytime observations with ~2 x 4.75 km pixels at field center ([TEMPO instrument specs](https://tempo.si.edu/instrument.html)). Sentinel-4 provides hourly observations with L2 products including O3, NO2, SO2, HCHO, CHOCHO, clouds, and aerosol properties ([Sentinel-4 data products](https://sentinels.copernicus.eu/missions/sentinel-4/data-products)). ASIA-AQ also provides validation opportunities for LEO sensors (TROPOMI, IASI, CrIS, OMPS, VIIRS) ([ASIA-AQ white paper](https://espo.nasa.gov/document/ASIA-AQ_White_Paper)).

2. **Emissions Quantification and Verification** -- Use aircraft and satellite observations to constrain bottom-up emission inventories for NOx, VOCs, CO, SO2, and particulate matter across diverse Asian source regions.

3. **Model Evaluation** -- Compare chemical transport model forecasts and analyses against the multi-platform observational dataset. Models deployed include WRF-Chem, MUSICA/CAM-chem, GEOS-Chem, WRF-CMAQ, and WACCM.

4. **Aerosol Chemistry** -- Characterize aerosol composition, secondary organic aerosol (SOA) formation, and aerosol-radiation interactions in polluted Asian environments.

5. **Ozone Chemistry** -- Process-level understanding of tropospheric ozone production, the role of NOx vs VOC sensitivity (via HCHO/NO2 ratios), and ozone-aerosol interactions.

## Deployment Sites and Planning Status (July 2023 Draft)

Source: [ASIA-AQ white paper](https://espo.nasa.gov/document/ASIA-AQ_White_Paper).

- **Proposed window:** January-March 2024 (targeting wintertime PM2.5 peaks).
- **Candidate locations:** Seoul (South Korea), Tokyo (Japan), Taiwan, Manila (Philippines), Hanoi and Ho Chi Minh City (Vietnam), Bangkok (Thailand), Kuala Lumpur (Malaysia), Singapore, Dhaka (Bangladesh), Kolkata and Delhi (India).
- **Locations under negotiation (July 2023):** South Korea, Taiwan, Philippines, Thailand, Malaysia.
Country status updates (July 2023):
- Bangladesh/India: no plans to fly in 2024; Pandora instruments deployed/planned to support GEMS validation.
- Vietnam: no plans to fly in 2024; Pandora instruments assigned pending approvals.
- Thailand: flight negotiations underway; Pandora record in Bangkok already established.
- Malaysia: proposal received provisional acceptance (June 2023) pending security/safety responses.
- Philippines: agreed in principle pending a NASA-DENR MOU; flight planning underway.
- Taiwan: discussions underway for up to ~4 hours of loiter time during Korea-Philippines transits.

## Aircraft and Key Instruments

### NASA DC-8

Large research aircraft (final deployment; [CASEI campaign summary](https://impact.earthdata.nasa.gov/casei/campaign/ASIA-AQ)) carrying a comprehensive payload.

**In-situ trace gases (ASDC DC-8 trace gas dataset):**
| Instrument | Notes |
|-----------|-------------|
| TOGA | Trace Organic Gas Analyzer |
| WAS | Whole Air Sampler |
| QC-TILDAS | Quantum Cascade Tunable Infrared Differential Absorption Spectrometer |
| CIT-ToF-CIMS | Chemical Ionization Time-of-Flight Mass Spectrometer |
| DACOM | Differential Absorption CO, CH4, N2O Measurements |
| LI-7000 | Closed Path CO2/H2O Gas Analyzer |
| LGR | CO/CO2/H2O Analyzer |
| ACES | Airborne Cavity Enhanced Spectrometer |
| MIRO MGA | MIRO Multi-compound Gas Analyzer |
| CANOE | Compact Airborne Nitrogen diOxide Experiment |
| ROZE | Rapid Ozone Experiment |
| PTR-MS | Proton Transfer Mass Spectrometer |
| ISAF | In Situ Airborne Formaldehyde |
| OPALS | Open-Path Ammonia Laser Spectrometer |
Source: [ASDC DC-8 Trace Gas dataset](https://asdc.larc.nasa.gov/project/ASIA-AQ/ASIA-AQ_TraceGas_AircraftInSitu_DC8_Data_1).

**Aerosol in-situ (ASDC DC-8 aerosol dataset):**
| Instrument | Notes |
|-----------|-------------|
| TEM | Transmission Electron Microscopy |
| AMS | Aerosol Mass Spectrometer |
| DMT SP2 | Single Particle Soot Photometer |
| DMT UHSAS | Ultra-High Sensitivity Aerosol Spectrometer |
| SMPS | Scanning Mobility Particle Sizer |
| TSI-3563 Nephelometer | Nephelometer |
Source: [ASDC DC-8 Aerosol dataset](https://asdc.larc.nasa.gov/project/ASIA-AQ/ASIA-AQ_Aerosol_AircraftInSitu_DC8_Data_1).

**Cloud in-situ (ASDC DC-8 cloud dataset):**
| Instrument | Notes |
|-----------|-------------|
| CCN | Condensation Nuclei Counter |
| CDP | Cloud Droplet Probe |
| CPSPD | Cloud Particle Spectrometer with Polarized Detection |
| CPC | Cloud Particle Counter |
Source: [ASDC DC-8 Cloud dataset](https://asdc.larc.nasa.gov/project/ASIA-AQ/ASIA-AQ_Cloud_AircraftInSitu_DC8_Data_1).

**Photolysis frequencies (ASDC DC-8 j-values dataset):**
| Instrument | Notes |
|-----------|-------------|
| CAFS | CCD-based Actinic Flux Spectroradiometer |
Source: [ASDC DC-8 j-values dataset](https://asdc.larc.nasa.gov/project/ASIA-AQ/ASIA-AQ_PhotolysisFreq_AircraftInSitu_DC8_Data_1).

**Meteorology and navigation (ASDC DC-8 MetNav dataset):**
| Instrument | Notes |
|-----------|-------------|
| DLH | Diode Laser Hygrometer |
| MMS | Meteorological Measurement System |
Source: [NASA ASIA-AQ MetNav dataset](https://catalog.data.gov/dataset/asia-aq-meteorology-navigation-aircraft-in-situ-dc8-data-1).

### NASA GV (planned in July 2023 draft)

High-altitude remote sensing platform planned to map trace-gas columns (GEMS proxy), lidar ozone/aerosol profiles, and possible aerosol polarimetry ([ASIA-AQ white paper](https://espo.nasa.gov/document/ASIA-AQ_White_Paper)).

### NASA G-III

Remote sensing and greenhouse gas platform:
- GCAS (Geo-CAPE Airborne Simulator) -- mapping NO2, O3, HCHO columns ([ASDC G-III GCAS dataset](https://asdc.larc.nasa.gov/project/ASIA-AQ/ASIA-AQ_AircraftRemoteSensing_LaRC-G3_GCAS_Data_1))
- HSRL-2 -- aerosol and ozone profiles ([ASDC G-III HSRL-2 dataset](https://asdc.larc.nasa.gov/project/ASIA-AQ/ASIA-AQ_AircraftRemoteSensing_LaRC-G3_HSRL2_Data_1))
- Greenhouse gas remote sensing for CO2 and CH4 with lidar sampling of methane/particulate pollution (planning concept; [ASIA-AQ white paper](https://espo.nasa.gov/document/ASIA-AQ_White_Paper))

## Satellite Context

ASIA-AQ was timed to coincide with a new era of geostationary air quality monitoring:

| Satellite | Coverage | Launch | Cadence | Key Products |
|-----------|----------|--------|---------|-------------|
| GEMS | Asia | Feb 2020 ([GEMS overview](https://acp.copernicus.org/articles/24/8943/2024/)) | Hourly | O3, NO2, SO2, HCHO, CHOCHO, aerosols ([GEMS overview](https://acp.copernicus.org/articles/24/8943/2024/)) |
| TEMPO | N. America | Apr 2023 ([ASDC TEMPO project](https://asdc.larc.nasa.gov/project/TEMPO)) | Hourly | O3, NO2, SO2, HCHO, CHOCHO, aerosols, clouds ([TEMPO expected products](https://weather.ndc.nasa.gov/tempo/expected_products.html)) |
| Sentinel-4 | Europe | 1 Jul 2025 ([ESA Sentinel-4 mission](https://www.esa.int/Applications/Observing_the_Earth/Copernicus/Sentinel-4/Introducing_the_Sentinel-4_mission)) | Hourly | O3, NO2, SO2, HCHO, CHOCHO, aerosols, clouds ([Sentinel-4 data products](https://sentinels.copernicus.eu/missions/sentinel-4/data-products)) |

Typical nadir pixel sizes: GEMS near Seoul is ~7 x 7.7 km for gases and ~3.5 x 7.7 km for aerosols ([GEMS overview](https://acp.copernicus.org/articles/24/8943/2024/)); TEMPO is ~2 x 4.75 km at field center ([TEMPO instrument specs](https://tempo.si.edu/instrument.html)).

Together these form the first geostationary constellation for atmospheric composition, enabling diurnal cycle studies of pollutants for the first time from space.

## Modeling Support

### ACOM/NCAR Models

| Model | Resolution | Chemistry |
|-------|-----------|-----------|
| NCAR WRF-Tracer | 5 km | Inert tracers (CO, NO from ASIA-AQv2 + FINNv2.5 emissions) |
| MUSICAv0 | 14 km (Asia), 100 km (global) | CAM aerosols with specified oxidants; post-campaign: full chemistry |
| WACCM | Global | Full stratosphere-troposphere chemistry |

### Other Forecasting Models

- UCLA WRF-Chem (RACM MADE-VBS mechanism)
- U. Iowa WRF-CMAQ (CB6r5 mechanism)
- Yonsei University WRF-Chem
- GEOS-Chem variants

## Key Species for DAVINCI

The ASIA-AQ measurement suite directly maps to DAVINCI's phase roadmap:

| DAVINCI Phase | Species | ASIA-AQ Measurement |
|---------------|---------|-------------------|
| 5 (Chapman + NOx) | NO, NO2 | QC-TILDAS, GCAS columns |
| 6 (CO + CH4) | CO, CH4, HCHO | DACOM, LGR, ISAF |
| 7 (SO2) | SO2 | GEMS columns |
| 8 (Trop O3) | O3, VOCs, HOx | ROZE, TOGA, PTR-MS, CIT-ToF-CIMS |
| 9 (MOZART) | Full suite | All instruments |

### HCHO/NO2 as Ozone Sensitivity Indicator

A key ASIA-AQ analysis uses the HCHO/NO2 column ratio to diagnose whether ozone production is NOx-limited or VOC-limited (see e.g., [ACP 2022](https://acp.copernicus.org/articles/22/15035/2022/)). This ratio is directly observable from GEMS and GCAS, and is a natural validation target for DAVINCI's tropospheric ozone mechanism (Phase 8).

## Data Access

- **NASA ASDC:** [asdc.larc.nasa.gov/project/ASIA-AQ](https://asdc.larc.nasa.gov/project/ASIA-AQ)
- **NASA ESPO:** [espo.nasa.gov/asia-aq](https://espo.nasa.gov/asia-aq)
- **ASDC data release (2025-03-18):** [asdc.larc.nasa.gov/news/asia-aq-data-release-announcement](https://asdc.larc.nasa.gov/news/asia-aq-data-release-announcement)
- **Data format:** ICARTT (merge files available per aircraft)
- **DOI:** 10.5067/SUBORBITAL/ASIA-AQ/DATA001 ([CASEI campaign summary](https://impact.earthdata.nasa.gov/casei/campaign/ASIA-AQ))
- **Mission page:** [www-air.larc.nasa.gov/missions/asia-aq](https://www-air.larc.nasa.gov/missions/asia-aq/)

## Relevance to DAVINCI

ASIA-AQ provides:

1. **Validation data** -- Aircraft in-situ measurements of the species DAVINCI will simulate (O3, NO, NO2, CO, CH4, HCHO, SO2, VOCs, HOx)
2. **Satellite context** -- GEMS hourly column data for NO2, O3, HCHO, SO2 over the simulation domain
3. **Emission inventories** -- ASIA-AQv2 anthropogenic emissions + FINNv2.5 fire emissions, already gridded for the campaign domain
4. **Model intercomparison** -- Multiple CTM results to benchmark DAVINCI against
5. **Science questions** -- Ozone sensitivity regimes, diurnal NOx cycles, secondary aerosol formation -- all testable with DAVINCI's mechanism progression

The long-term goal is to run DAVINCI-MPAS over an ASIA-AQ domain (e.g., Seoul metropolitan area or Bangkok) and evaluate against the campaign observations, using GEMS satellite data for broader spatial context.

## References

- Draft Planning Document for ASIA-AQ (2023-07-20). Local file: /Users/fillmore/Downloads/Draft Planning Document for ASIA-AQ_20230720.pdf
- Crawford, J. H. & Lefer, B. (2023). "The Airborne and Satellite Investigation of Asian Air Quality (ASIA-AQ)." AGU Fall Meeting 2023, A12A-02.
- Kim, J., et al. (2020). "New Era of Air Quality Monitoring from Space: Geostationary Environment Monitoring Spectrometer (GEMS)." Bull. Amer. Meteor. Soc., 101(1).
- Choi, Y., et al. (2024). "Quantifying the diurnal variation in atmospheric NO2 from GEMS observations." Atmos. Chem. Phys., 24, 8943.
