# ASIA-AQ Research Survey (as of 2026-02-05)

**Scope**
This survey covers public, citable ASIA-AQ literature and datasets available as of 2026-02-05. It emphasizes peer-reviewed papers, preprints, conference outputs, and NASA technical reports that explicitly use ASIA-AQ observations, plus key satellite literature that underpins ASIA-AQ science objectives.

See `ACRONYMS.md` for acronym definitions used in this document.

**Campaign Overview (validated)**
ASIA-AQ is an international cooperative field campaign led by NASA in partnership with South Korea's National Institute of Environmental Research (NIER) and other regional agencies. The campaign deployed multiple aircraft and ground networks to study air quality and to improve the integration of satellite observations with in-country monitoring and models. Public campaign documentation indicates the main deployment window from 2024-01-29 to 2024-04-01, with country operations over the Philippines, South Korea, Taiwan, and Thailand. The Atmospheric Science Data Center (ASDC) released ASIA-AQ data publicly on 2025-03-18, and the campaign DOI is 10.5067/SUBORBITAL/ASIA-AQ/DATA001.

**Public Data Products (ASDC)**
The ASDC currently hosts multiple Level-2 ASIA-AQ datasets, including:
- DC-8 in-situ trace gases (TOGA, WAS, QC-TILDAS, CIT-ToF-CIMS, DACOM, LI-7000, LGR, ACES, MIRO MGA, CANOE, ROZE, PTR-MS, ISAF, OPALS).
- DC-8 in-situ aerosols (TEM, AMS, DMT SP2, DMT UHSAS, SMPS, APS, TSI-3563 nephelometer).
- DC-8 in-situ clouds (CCN, CDP, CPSPD, CPC).
- DC-8 meteorology/navigation (DLH, MMS).
- LaRC G-III remote sensing (GCAS UV-Vis trace-gas columns; HSRL-2 aerosol/ozone profiles).
Most DC-8 in-situ products are distributed in ICARTT format.

**Aircraft Platform Note (G-III)**
The LaRC G-III (Gulfstream III) served as the campaign's remote sensing aircraft. Planning documents assign it greenhouse gas remote sensing (CO2, CH4) with lidar sampling of methane and particulate pollution (Draft Planning Document for ASIA-AQ, 2023-07-20), and ASDC hosts the G-III [GCAS](https://asdc.larc.nasa.gov/project/ASIA-AQ/ASIA-AQ_AircraftRemoteSensing_LaRC-G3_GCAS_Data_1) and [HSRL-2](https://asdc.larc.nasa.gov/project/ASIA-AQ/ASIA-AQ_AircraftRemoteSensing_LaRC-G3_HSRL2_Data_1) datasets.

**ASIA-AQ Literature and Outputs**

**Peer-Reviewed (ASIA-AQ-specific)**
- Miech et al. (2025, ACP) "Identifying biomass burning emissions during ASIA-AQ using greenhouse gas enhancement ratios." Uses airborne enhancement ratios and particle scattering to identify biomass-burning-impacted air masses during Feb-Mar 2024, with transport modeling and satellite hotspot products to interpret transport histories over Thailand.

**Preprints and Conference Outputs**
- Cho et al. (2026, EGUsphere preprint; under review in ACP) "Insights on Ozone Formation Sensitivity in Southeast and East Asian Megacities during ASIA-AQ."
- Hong et al. (2025, EGU General Assembly abstract) "GEMS L2 data validation during ASIA-AQ campaign." Reports NO2 correlations >0.6 between GEMS and ground-based remote sensing in Korea and improvements of GEMS NO2 from v2 to v4; includes comparisons with GeoTASO.
- NASA NTRS technical reports/presentations (2025) on GEOS aerosol assimilation and GEOS simulations evaluated with ASIA-AQ observations.

**Mission Planning and Overview (Pre-deployment)**
- Crawford et al. (2022, IGARSS) "ASIA-AQ: An Opportunity for International Collaboration." Outlines objectives (satellite validation, emissions quantification, model evaluation, aerosol and ozone chemistry), the multi-platform observing strategy, and the planned Jan-Mar 2024 window.

**White Paper (Draft Planning Document, 2023-07-20)**
Key points from the draft planning white paper (pre-deployment, July 2023):
- Proposed deployment window: January-March 2024, chosen to target wintertime PM2.5 peaks; exact dates to be finalized.
- Five science goals are explicitly listed: satellite validation and interpretation, emissions quantification and verification, model evaluation, aerosol chemistry, and ozone chemistry.
- Candidate locations include: Seoul (South Korea), Tokyo (Japan), Taiwan, Manila (Philippines), Hanoi and Ho Chi Minh City (Vietnam), Bangkok (Thailand), Kuala Lumpur (Malaysia), Singapore, Dhaka (Bangladesh), Kolkata and Delhi (India).
- Locations under negotiation for flight sampling as of July 2023: South Korea, Taiwan, Philippines, Thailand, Malaysia.
- Aircraft concept: DC-8 for in situ chemistry; NASA GV (or substitute) for high-altitude trace-gas remote sensing plus lidar for ozone/aerosols (and possible polarimetry); NASA G-III for greenhouse-gas remote sensing (CO2, CH4) and lidar sampling of methane/particulate pollution.

Country updates as of July 2023:
- No plans to fly in Bangladesh or India in 2024; Pandora instruments deployed/planned for Dhaka and India to support GEMS validation.
- No plans to fly in Vietnam; Pandora instruments assigned to Hanoi and Ho Chi Minh City pending approvals.
- Thailand: flight negotiations still underway; Pandora record in Bangkok already established.
- Malaysia: proposal received provisional acceptance (June 2023) pending security/safety responses.
- Philippines: agreed in principle pending a NASA-DENR MOU; flight planning underway.
- Taiwan: discussions underway for up to ~4 hours of loiter time during Korea-Philippines transits.

**Satellite and Constellation Context (Support Literature)**
ASIA-AQ was designed around the new geostationary constellation and related LEO sensors.
- GEMS (Asia): Launched 2020; hourly daytime observations over East/Southeast Asia; ~3.5 x 7.7 km2 resolution at Seoul. Peer-reviewed evaluations now exist for GEMS HCHO and CHOCHO products.
- TEMPO (North America): Launched 2023; hourly daytime measurements at ~2.0 x 4.75 km2; retrieves NO2, SO2, O3, HCHO, CHOCHO, aerosols, and other constituents.
- Sentinel-4 (Europe): Launched 2025 on MTG-S1; L2 products include O3, NO2, SO2, HCHO, CHOCHO, and aerosol properties.

**Ozone Sensitivity Diagnostics (Relevant Method Literature)**
Multiple satellite studies use the formaldehyde-to-NO2 ratio (HCHO/NO2) to diagnose NOx- vs VOC-limited ozone regimes, and this diagnostic is directly applicable to ASIA-AQ's ozone-chemistry objectives.

**Key Findings to Date (Synthesis)**
- Biomass burning impacts during Feb-Mar 2024 were identified using airborne greenhouse-gas enhancement ratios, trace gases, and aerosol scattering, with transport analyses indicating plume histories over Thailand.
- GEMS L2 validation during ASIA-AQ shows meaningful agreement with ground-based instruments (NO2 correlation >0.6 in Korea) and improved NO2 performance in later GEMS versions (v4 vs v2).
- GEMS product evaluations (HCHO and CHOCHO) demonstrate retrieval capability for VOC-related products that are central to ASIA-AQ's satellite-validation and ozone-chemistry goals.

**Gaps and Near-Term Opportunities**
- The peer-reviewed ASIA-AQ literature is still emerging. Most ASIA-AQ-specific results are in technical notes, preprints, or conference abstracts.
- High-priority near-term analyses include diurnal ozone-production sensitivity over megacities, emissions evaluation using multi-platform constraints, and aerosol chemical evolution in biomass-burning outflow.

**References and Links**
1. ASDC Data Release Announcement (2025-03-18): https://asdc.larc.nasa.gov/news/asia-aq-data-release-announcement
2. ASDC Project Page (ASIA-AQ): https://asdc.larc.nasa.gov/project/ASIA-AQ
3. CASEI Campaign Summary (DOI, study dates): https://impact.earthdata.nasa.gov/casei/campaign/ASIA-AQ
4. NASA Health & Air Quality Newsletter (Apr-Jun 2024) ASIA-AQ summary: https://appliedsciences.nasa.gov/sites/default/files/2024-09/HAQ_NL_V39_Apr-Jun2024.pdf
5. DC-8 Trace Gas Dataset (ASDC): https://asdc.larc.nasa.gov/project/ASIA-AQ/ASIA-AQ_TraceGas_AircraftInSitu_DC8_Data_1
6. DC-8 Aerosol Dataset (ASDC): https://asdc.larc.nasa.gov/project/ASIA-AQ/ASIA-AQ_Aerosol_AircraftInSitu_DC8_Data_1
7. DC-8 Cloud Dataset (ASDC): https://asdc.larc.nasa.gov/project/ASIA-AQ/ASIA-AQ_Cloud_AircraftInSitu_DC8_Data_1
8. DC-8 Met/Nav Dataset (ASDC): https://asdc.larc.nasa.gov/project/ASIA-AQ/ASIA-AQ_MetNav_AircraftInSitu_DC8_Data_1
9. G-III GCAS Dataset (ASDC): https://asdc.larc.nasa.gov/project/ASIA-AQ/ASIA-AQ_AircraftRemoteSensing_LaRC-G3_GCAS_Data_1
10. G-III HSRL-2 Dataset (ASDC): https://asdc.larc.nasa.gov/project/ASIA-AQ/ASIA-AQ_AircraftRemoteSensing_LaRC-G3_HSRL2_Data_1
11. Miech et al. (2025, ACP) ASIA-AQ biomass burning technical note: https://doi.org/10.5194/acp-25-15701-2025
12. Cho et al. (2026, EGUsphere preprint) ozone sensitivity during ASIA-AQ: https://doi.org/10.5194/egusphere-2025-6434
13. Hong et al. (2025, EGU abstract) GEMS validation during ASIA-AQ: https://doi.org/10.5194/egusphere-egu25-15190
14. Crawford et al. (2022, IGARSS) ASIA-AQ planning paper: https://doi.org/10.1109/IGARSS46834.2022.9883819
15. GEMS aerosol results (AMT 2024) with instrument specs: https://amt.copernicus.org/articles/17/4369/2024/
16. GEMS HCHO evaluation (ACP 2024): https://doi.org/10.5194/acp-24-4733-2024
17. GEMS CHOCHO evaluation (AMT 2024): https://doi.org/10.5194/amt-17-6369-2024
18. TEMPO mission overview and launch: https://asdc.larc.nasa.gov/project/TEMPO
19. TEMPO product/resolution overview (NASA/SAO): https://weather.ndc.nasa.gov/tempo/
20. Sentinel-4 launch (ESA): https://www.esa.int/ESA_Multimedia/Images/2025/07/MTG-S1_and_Sentinel-4_launch
21. Sentinel-4 L2 product list: https://sentinels.copernicus.eu/missions/sentinel-4/data-products
22. HCHO/NO2 ozone-sensitivity diagnostics (ACP 2022): https://doi.org/10.5194/acp-22-15035-2022
23. Draft Planning Document for ASIA-AQ (2023-07-20) [local file]: /Users/fillmore/Downloads/Draft Planning Document for ASIA-AQ_20230720.pdf

**Appendix: Acronyms**
See `ACRONYMS.md`.
