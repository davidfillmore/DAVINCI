# DC3: Deep Convective Clouds and Chemistry

The **Deep Convective Clouds and Chemistry (DC3)** field campaign is the primary observational reference for DAVINCI's lightning NOx implementation (Phase 3) and future convective chemistry development. This document summarizes the campaign and its relevance to our modeling goals.

See `ACRONYMS.md` for acronym definitions used in this document.

## Campaign Overview

DC3 was a multi-agency field experiment designed to study the impact of deep midlatitude continental convective clouds on upper tropospheric composition and chemistry. It combined three research aircraft, three ground-based radar/lightning networks, and extensive radiosonde support to sample thunderstorms from inflow to outflow and through next-day chemical aging ([Barth et al., 2015](https://journals.ametsoc.org/view/journals/bams/96/8/bams-d-13-00290.1.xml)).

- **Dates:** 15 May -- 30 June 2012 (field phase); intensive operations period 18 May -- 22 June ([Barth et al., 2015](https://journals.ametsoc.org/view/journals/bams/96/8/bams-d-13-00290.1.xml))
- **Operations Base:** Salina, Kansas (all three aircraft, forecasting center)
- **Ground Regions:** Northeast Colorado (CSU), northern Alabama (UAH/NSSTC), and Oklahoma/west Texas (OU/NSSL)
- **PIs:** Mary C. Barth (NCAR, lead PI), Christopher A. Cantrell (CU Boulder), William H. Brune (Penn State), Steven A. Rutledge (CSU), James H. Crawford (NASA Langley), Heidi Huntrieser (DLR)
- **Sponsors:** NSF, NASA, NOAA, DLR (German Aerospace Center) ([NSF Award #0921488](https://www.nsf.gov/awardsearch/showAward?AWD_ID=0921488))
- **Heritage:** Builds on STERAO (1996, Colorado), CRYSTAL-FACE (2002, Florida), and STEPS (2000, Kansas) campaigns for convective chemistry

## Science Objectives

Four primary science themes ([Barth et al., 2015](https://journals.ametsoc.org/view/journals/bams/96/8/bams-d-13-00290.1.xml); [Barth et al., 2019](https://agupubs.onlinelibrary.wiley.com/doi/abs/10.1029/2019JD030944)):

1. **Lightning NOx Production** -- Quantify NOx production per lightning flash using coordinated aircraft (inflow/outflow sampling) and ground-based lightning mapping arrays. Determine the vertical distribution of lightning-produced NOx and its dependence on storm type and flash characteristics.

2. **Convective Transport of Boundary Layer Pollutants** -- Measure the efficiency with which deep convection lofts boundary layer trace gases (CO, VOCs, aerosols) to the upper troposphere. Characterize how storm dynamics (updraft strength, entrainment) control transport efficiency across different convective regimes.

3. **Wet Scavenging** -- Determine scavenging efficiencies of soluble trace gases (H2O2, CH3OOH, HNO3, CH2O) by thunderstorms, including the role of ice retention during drop freezing. Constrain the partitioning between convective transport and removal for species of intermediate solubility.

4. **Post-Convective Chemistry** -- Characterize the photochemical evolution of convective outflow on timescales of hours to one day. Quantify upper tropospheric ozone production from lightning NOx in the presence of lofted VOCs and HOx precursors.

## Ground Regions and Assets

DC3's three-region design enabled sampling diverse convective regimes:

### Northeast Colorado (CSU)

- **Convective regime:** Airmass thunderstorms, multicells, occasional supercells; relatively clean continental boundary layer
- **Radars:** CSU-CHILL (dual-frequency S/X-band, dual-polarization), CSU-Pawnee (S-band Doppler), NEXRAD WSR-88D network
- **Lightning:** COLMA (Colorado Lightning Mapping Array) -- 15-station VHF network installed for DC3 in spring 2012, ~100 km diameter, detection range ~350 km ([Basarab et al., 2015](https://doi.org/10.1002/2015JD023470))
- **Storms sampled:** 8 Colorado storms

### Northern Alabama (UAH/NSSTC)

- **Convective regime:** Warm-season air mass convection with high moisture; maritime-influenced boundary layer
- **Radars:** ARMOR (C-band dual-polarization, 0.9 deg beam, 125/250 m range gates), MAX (mobile X-band dual-polarization, 0.95 deg beam, 125 m range gates), NEXRAD KHTX
- **Lightning:** NALMA (North Alabama Lightning Mapping Array) -- pre-existing VHF network near Huntsville
- **Storms sampled:** 2 Alabama storms

### Oklahoma / West Texas (OU/NSSL)

- **Convective regime:** Severe supercells, squall lines, mesoscale convective systems; polluted boundary layer with urban/industrial and agricultural sources
- **Radars:** SMART-R (two mobile C-band Doppler radars, 1.5 deg beam), NEXRAD KOUN (Norman, OK)
- **Lightning:** OKLMA (Oklahoma Lightning Mapping Array) -- pre-existing network at NSSL ([NOAA NSSL](https://www.nssl.noaa.gov/tools/oklma/))
- **Storms sampled:** 6 Oklahoma/Texas storms

### Sounding Sites

- **ARM SGP** (Southern Great Plains Central Facility, Lamont, OK) -- 238 high-resolution radiosondes during DC3; 4/day at 00, 06, 12, 18 UTC
- **NSSL MGAUS** (Mobile GPS Advanced Upper-Air Sounding System) -- deployed around Oklahoma/west Texas, 19 May -- 21 June 2012
- Additional mobile sounding teams in Colorado and Alabama

## Aircraft and Key Instruments

### NSF/NCAR GV (Gulfstream V, HIAPER)

High-altitude platform for sampling convective outflow in the upper troposphere.

- **Operator:** NCAR Earth Observing Laboratory (EOL)
- **Ceiling:** ~15.5 km (FL510 certified); typical DC3 sampling at FL400--FL450 (~12--14 km)
- **Role:** Primary outflow sampling (anvil penetration, aged outflow tracking)

**Key instruments:**

| Instrument | Measurement | PI / Institution | Notes |
|------------|-------------|------------------|-------|
| NOxyO3 | NO, NO2, NOy, O3 | Andrew Weinheimer, NCAR | 4-channel chemiluminescence; 1 s time resolution ([NASA Airborne Science](https://airbornescience.nasa.gov/instrument/NOxyO3)) |
| TOGA | ~60+ VOC species | Eric Apel, NCAR | GC-MS; NMHCs, OVOCs, nitriles, halogenated VOCs |
| CAMS | CH2O (formaldehyde) | Alan Fried, CU Boulder/INSTAAR | DFG-based spectrometer |
| PAN-CIMS | PAN, PPN, peroxyacyl nitrates | Frank Flocke, NCAR | Chemical ionization mass spectrometry |
| P-CIMS | H2O2, CH3OOH | Brian Heikes, URI | Negative-ion CIMS for peroxides |
| ACOM CO | CO | Teresa Campos, NCAR | Aero-Laser VUV fluorescence |
| Picarro | CO2, CH4, H2O | NCAR | Cavity ring-down spectroscopy |
| CAFS | Actinic flux, j-values | NCAR | CCD Actinic Flux Spectroradiometer |
| Cloud probes | Droplet/ice size distributions | Various | Multiple probes for hydrometeor characterization |

### NASA DC-8

Large research platform for boundary layer and inflow sampling, plus remote sensing.

- **Operator:** NASA Armstrong/Dryden Flight Research Center
- **Ceiling:** ~12 km (FL400)
- **Role:** Inflow characterization (low-level sampling near storms), DIAL lidar profiling

**Key instruments:**

| Instrument | Measurement | PI / Institution | Notes |
|------------|-------------|------------------|-------|
| DIAL | O3, H2O profiles (remote) | NASA Langley | UV/vis/IR differential absorption lidar; vertical curtain data |
| DACOM | CO, N2O, CH4 | Glenn Diskin, NASA Langley | Three tunable diode lasers (4.7, 4.5, 3.3 um) |
| ATHOS | OH, HO2 | William Brune, Penn State | UV laser-induced fluorescence; low-pressure detection |
| NOAA NOyO3 | NO, NO2, NOy, O3 | Tom Ryerson / Jeff Peischl, NOAA | Chemiluminescence |
| TD-LIF | NO2, sum PNs, sum ANs, HNO3 | Ronald Cohen, UC Berkeley | 30 pptv NO2 at 1 Hz (S/N=2); 5% NO2, 10% PNs, 15% ANs/HNO3 ([NASA Airborne Science](https://airbornescience.nasa.gov/instrument/TD-LIF)) |
| GT-CIMS | HNO3, SO2, pernitric acid | L. Greg Huey, Georgia Tech | Chemical ionization mass spectrometry |
| DFG | CH2O (formaldehyde) | Alan Fried, CU Boulder | Complementary to CAMS on GV |
| WAS | VOCs, halocarbons | Donald Blake, UC Irvine | Canister samples analyzed post-flight by GC-MS/FID |
| HR-ToF-AMS | Non-refractory submicron aerosol | Jose Jimenez / Pedro Campuzano-Jost, CU Boulder | Aerosol Mass Spectrometer |
| LARGE suite | Aerosol size, number, optics | Bruce Anderson, NASA Langley | Multiple instruments |

### DLR Falcon 20

German research aircraft for fresh anvil outflow sampling.

- **Operator:** Deutsches Zentrum fur Luft- und Raumfahrt (DLR), Oberpfaffenhofen
- **Deployed:** 29 May -- 14 June 2012 (subset of campaign); 13 research flights
- **Role:** Trace gas measurements in fresh anvil outflow, complementing the GV

**Key instruments:**

| Instrument | Measurement | PI / Institution |
|------------|-------------|------------------|
| In-situ trace gas suite | CO, O3, SO2, CH4, NO, NOx, NOy | Heidi Huntrieser / Hans Schlager, DLR |
| PTR-MS | VOCs | Armin Wisthaler, U. Innsbruck |
| CN counter | Condensation nuclei | DLR |
| SP2 | Refractory black carbon | DLR |

### Sampling Strategy

DC3 employed a coordinated multi-aircraft approach: the DC-8 characterized inflow composition at low altitude near storms while the GV and Falcon sampled outflow in the anvil region at 8--14 km. The aircraft sampled ~16 thunderstorms total. Five times, the GV and DC-8 returned the following day to track the aged outflow and measure post-convective chemical evolution ([Barth et al., 2015](https://journals.ametsoc.org/view/journals/bams/96/8/bams-d-13-00290.1.xml)).

## Lightning NOx: Key Findings

DC3 provides the most comprehensive observational dataset for lightning NOx (LNOx) production. This section summarizes findings directly relevant to DAVINCI's Phase 3 implementation.

### Measurement Approach

Three synergistic strategies:

1. **Lightning Mapping Arrays (LMAs)** -- 3D flash structure, flash rates, IC/CG ratios, flash channel segment altitude distributions
2. **Aircraft in-situ NOx** -- Inflow (DC-8) vs. outflow (GV, Falcon) NOx concentrations; the difference yields LNOx signal
3. **Ground-based radars** -- Storm structure, volume, updraft properties, ice content for relating flash rates to dynamics

### NOx Production Per Flash

The range of DC3-derived LNOx estimates spans 80--550 mol NO per flash, reflecting storm-to-storm variability and methodological differences:

| Study | Method | mol NOx flash^-1 | Notes |
|-------|--------|-------------------|-------|
| Pollack et al. (2016) | Volume-based, 3 storms | 142--291 (avg 194) | Oklahoma and Colorado storms; uncertainty range 117--332 |
| Nault et al. (2017) | In-situ + OMI satellite | 510--550 | Updated NOx chemistry (organic nitrate formation); implies ~9 Tg N yr^-1 globally |
| Pickering et al. (2024) | WRF-Chem, constrained | 80--110 (best 82) | 29--30 May OK supercell; very high flash rate, small flash extents |
| Cummings et al. (2024) | WRF-Chem, 18 FRPSs | 82 (best fit) | Same storm as Pickering; 13/18 flash rate schemes overestimate by >100% |

**Key insight (Nault et al., 2017):** When updated NOx chemistry is applied -- accounting for the ~3 hour effective NOx lifetime in convective outflow due to rapid conversion to methyl peroxy nitrate and organic nitrates -- measurement-based and model-based estimates converge toward ~500 mol flash^-1, consistent with pre-DC3 modeling estimates ([Ott et al., 2010](https://agupubs.onlinelibrary.wiley.com/doi/abs/10.1029/2009JD011880)).

### Ozone Production in Outflow

- Downwind O3 production of **10--11 ppbv per day** in the 9--11 km outflow layer, measured over 24 hours after the 29--30 May storm ([Pickering et al., 2024](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2023JD039439))
- Colorado 22 June 2012 case: upper tropospheric O3 enhancement from LNOx, complicated by smoke ingestion from the High Park wildfire ([Apel et al., 2015](https://agupubs.onlinelibrary.wiley.com/doi/abs/10.1002/2014JD022121))

### Wet Scavenging Efficiencies

| Species | Scavenging Efficiency | Source |
|---------|----------------------|--------|
| H2O2 | 79--97% | Barth et al. (2016) |
| CH3OOH | 12--84% | Barth et al. (2016) |
| CH2O | Variable, highly sensitive to ice retention | Fried et al. (2016) |

Scavenging efficiency is strongly dependent on ice retention factors during cloud drop freezing ([Bela et al., 2016](https://www.pnnl.gov/publications/wet-scavenging-soluble-gases-dc3-deep-convective-storms-using-wrf-chem-simulations-and)).

## Modeling Context

### Models Used for DC3 Analysis

| Model | Configuration | Application |
|-------|---------------|-------------|
| WRF-Chem | Cloud-resolving (1 km), 480 x 420 x 89 grid | Flash rate parameterization, LNOx, scavenging, transport |
| WRF-Chem | Parameterized convection | Convective transport evaluation |
| GEOS-Chem | Global | LNOx emission constraints with OMI satellite data |
| CAM-chem / CESM | Global | Convective transport parameterization evaluation |
| NASA LNOM | Lightning model | Flash-to-NOx production relationships |

### Key Modeling Findings

- 13 of 18 flash rate parameterization schemes overestimate observed flash counts by >100%; upward cloud ice flux and updraft volume schemes perform best ([Cummings et al., 2024](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2023JD039492))
- Convective transport in parameterized models can be evaluated against cloud-resolved simulations using inflow/outflow CO ([Li et al., 2017](https://agupubs.onlinelibrary.wiley.com/doi/abs/10.1002/2017JD026461), [2018](https://pmc.ncbi.nlm.nih.gov/articles/PMC6999733/))
- Stratospheric O3 entrainment around MCS anvils produces pronounced O3 gradients in outflow; DeltaO3/DeltaCO ratios from +1.4 to -3.9 ([Huntrieser et al., 2016](https://agupubs.onlinelibrary.wiley.com/doi/abs/10.1002/2015JD024279))

## Key Species for DAVINCI

DC3 measurements map directly to DAVINCI's phase roadmap:

| DAVINCI Phase | Species | DC3 Measurement | Instrument |
|---------------|---------|-----------------|------------|
| 3 (Lightning NOx) | NO, NO2, NOy | Inflow + outflow NOx | NOxyO3 (GV), NOAA NOyO3 (DC-8), TD-LIF (DC-8) |
| 5 (Chapman) | O3 | Profiles + in-situ | DIAL (DC-8), NOxyO3 (GV) |
| 6 (NOx coupling) | NO, NO2, PAN, HNO3 | Speciated nitrogen budget | PAN-CIMS (GV), TD-LIF (DC-8), GT-CIMS (DC-8) |
| 7 (CO + CH4) | CO, CH4, CH2O | Inflow/outflow differences | DACOM (DC-8), ACOM CO (GV), CAMS/DFG (CH2O) |
| 9 (Trop O3) | O3, VOCs, OH, HO2 | Outflow chemistry | ATHOS (DC-8), TOGA (GV), WAS (DC-8) |

## Data Access

- **NASA ASDC:** [asdc.larc.nasa.gov/project/DC3](https://asdc.larc.nasa.gov/project/DC3) -- all airborne data (DC-8, GV, Falcon), merged datasets
- **NCAR EOL:** [data.eol.ucar.edu/project/DC3](https://data.eol.ucar.edu/project/DC3) -- ground-based data (radar, LMA, radiosondes), GV data, field catalog
- **NASA ESPO:** [espo.nasa.gov/dc3](https://espo.nasa.gov/dc3) -- mission information and links
- **NCAR ACOM:** [www2.acom.ucar.edu/dc3](https://www2.acom.ucar.edu/dc3) -- project page, publications list, science highlights
- **Data format:** ICARTT (aircraft merge files); NetCDF (radar, some derived products)
- **NASA CASEI:** [impact.earthdata.nasa.gov/casei/campaign/DC3](https://impact.earthdata.nasa.gov/casei/campaign/DC3/)

## Relevance to DAVINCI

DC3 provides validation targets for multiple DAVINCI development phases:

1. **Lightning NOx parameterization** -- DC3-derived flash rate parameterizations (based on updraft mass flux, ice flux, updraft volume) can validate DAVINCI's w-proportional source (`S = rate * max(0, w - w_thr) / w_ref`). The DC3-constrained production range (82--550 mol flash^-1, converging to ~500 with updated chemistry) directly informs the source magnitude.

2. **Convective transport efficiency** -- Inflow/outflow CO differences provide benchmarks for MPAS convection at convection-allowing resolution. MPAS's variable-resolution Voronoi mesh could run DC3 cases with locally refined grids over the storm region.

3. **Wet scavenging** -- H2O2 (79--97%) and CH3OOH (12--84%) scavenging efficiencies provide direct constraints if DAVINCI implements aqueous/ice-phase chemistry.

4. **Post-convective O3 production** -- Day-after outflow observations (10--11 ppbv O3/day) validate photochemistry in Phases 6+.

5. **NOx lifetime in outflow** -- The ~3 hour effective NOx lifetime (Nault et al., 2017) constrains how rapidly the model should convert fresh lightning NOx to reservoir species (PAN, organic nitrates).

### Benchmark Case: 29--30 May 2012 Oklahoma Supercell

The most extensively modeled DC3 storm, with the most complete observational coverage across all three aircraft, OKLMA flash data, and multiple radar systems. This is the primary candidate for a future MPAS convection-allowing simulation with DAVINCI chemistry.

## References

### Campaign Overview

- Barth, M. C., C. A. Cantrell, W. H. Brune, S. A. Rutledge, J. H. Crawford, H. Huntrieser, L. D. Carey, D. MacGorman, M. Weisman, K. E. Pickering, E. Bruning, B. Anderson, E. Apel, M. Biggerstaff, T. Campos, P. Campuzano-Jost, R. Cohen, J. Crounse, D. A. Day, G. Diskin, F. Flocke, A. Fried, C. Garland, B. Heikes, S. Honomichl, R. Hornbrook, L. G. Huey, J. L. Jimenez, T. Lang, M. Lichtenstern, T. Mikoviny, B. Nault, D. O'Sullivan, L. L. Pan, and others (2015). "The Deep Convective Clouds and Chemistry (DC3) Field Campaign." *Bull. Amer. Meteor. Soc.*, 96, 1281--1309. doi:[10.1175/BAMS-D-13-00290.1](https://doi.org/10.1175/BAMS-D-13-00290.1)
- Barth, M. C., and co-authors (2019). "Introduction to the Deep Convective Clouds and Chemistry (DC3) 2012 Studies." *J. Geophys. Res. Atmos.*, 124. doi:[10.1029/2019JD030944](https://doi.org/10.1029/2019JD030944)

### Lightning NOx

- Pollack, I. B., E. Bruning, D. MacGorman, K. A. Cummings, K. E. Pickering, H. Huntrieser, M. Lichtenstern, H. Schlager, and M. C. Barth (2016). "Airborne quantification of upper tropospheric NOx production from lightning in deep convective storms over the United States Great Plains." *J. Geophys. Res. Atmos.*, 121. doi:[10.1002/2015JD023941](https://doi.org/10.1002/2015JD023941)
- Nault, B. A., J. L. Laughner, P. J. Wooldridge, J. D. Crounse, J. Dibb, G. Diskin, J. Peischl, J. R. Podolske, I. B. Pollack, T. B. Ryerson, E. Scheuer, P. O. Wennberg, and R. C. Cohen (2017). "Lightning NOx Emissions: Reconciling Measured and Modeled Estimates With Updated NOx Chemistry." *Geophys. Res. Lett.*, 44, 9479--9488. doi:[10.1002/2017GL074436](https://doi.org/10.1002/2017GL074436)
- Pickering, K. E., and co-authors (2024). "Lightning NOx in the 29--30 May 2012 Deep Convective Clouds and Chemistry (DC3) Severe Storm and Its Downwind Chemical Consequences." *J. Geophys. Res. Atmos.*, 129. doi:[10.1029/2023JD039439](https://doi.org/10.1029/2023JD039439)
- Cummings, K. A., and co-authors (2024). "Evaluation of Lightning Flash Rate Parameterizations in a Cloud-Resolved WRF-Chem Simulation of the 29--30 May 2012 Oklahoma Severe Supercell System Observed During DC3." *J. Geophys. Res. Atmos.*, 129. doi:[10.1029/2023JD039492](https://doi.org/10.1029/2023JD039492)
- Ott, L. E., K. E. Pickering, G. L. Stenchikov, D. J. Allen, A. J. DeCaria, B. Ridley, R.-F. Lin, S. Lang, and W.-K. Tao (2010). "Production of lightning NOx and its vertical distribution calculated from three-dimensional cloud-scale chemical transport model simulations." *J. Geophys. Res.*, 115, D04301. doi:[10.1029/2009JD011880](https://doi.org/10.1029/2009JD011880)

### Convective Transport and Wet Scavenging

- Barth, M. C., and co-authors (2016). "Convective transport and scavenging of peroxides by thunderstorms observed over the central U.S. during DC3." *J. Geophys. Res. Atmos.*, 121, 4272--4295. doi:[10.1002/2015JD024570](https://doi.org/10.1002/2015JD024570)
- Fried, A., and co-authors (2016). "Convective transport of formaldehyde to the upper troposphere and lower stratosphere and associated scavenging in thunderstorms over the central United States during the 2012 DC3 study." *J. Geophys. Res. Atmos.*, 121, 7430--7460. doi:[10.1002/2015JD024477](https://doi.org/10.1002/2015JD024477)
- Bela, M. M., and co-authors (2016). "Wet scavenging of soluble gases in DC3 deep convective storms using WRF-Chem simulations and aircraft observations." *J. Geophys. Res. Atmos.*, 121. doi:[10.1002/2015JD024623](https://doi.org/10.1002/2015JD024623)
- Li, Y., K. E. Pickering, M. C. Barth, M. M. Bela, K. A. Cummings, and D. Allen (2017). "Evaluation of deep convective transport in storms from different convective regimes during the DC3 field campaign using WRF-Chem with lightning data assimilation." *J. Geophys. Res. Atmos.*, 122. doi:[10.1002/2017JD026461](https://doi.org/10.1002/2017JD026461)
- Li, Y., K. E. Pickering, M. C. Barth, M. M. Bela, K. A. Cummings, and D. Allen (2018). "Evaluation of parameterized convective transport of trace gases in simulation of storms observed during the DC3 field campaign." *J. Geophys. Res. Atmos.* doi:[10.1029/2018JD028779](https://doi.org/10.1029/2018JD028779)

### Chemical Aging, Oxidation, and Ozone

- Apel, E. C., R. S. Hornbrook, and co-authors (2015). "Upper tropospheric ozone production from lightning NOx-impacted convection: Smoke ingestion case study from the DC3 campaign." *J. Geophys. Res. Atmos.*, 120, 2505--2523. doi:[10.1002/2014JD022121](https://doi.org/10.1002/2014JD022121)
- Brune, W. H., X. Ren, L. Zhang, J. Mao, D. O. Miller, B. E. Anderson, D. R. Blake, R. C. Cohen, G. S. Diskin, S. R. Hall, T. F. Hanisco, L. G. Huey, B. A. Nault, J. W. Peischl, I. B. Pollack, T. B. Ryerson, T. Shingler, A. Sorooshian, K. L. Ullmann, A. Wisthaler, and P. J. Wooldridge (2018). "Atmospheric oxidation in the presence of clouds during the Deep Convective Clouds and Chemistry (DC3) study." *Atmos. Chem. Phys.*, 18, 14493--14510. doi:[10.5194/acp-18-14493-2018](https://doi.org/10.5194/acp-18-14493-2018)
- Huntrieser, H., M. Lichtenstern, M. Scheibe, H. Aufmhoff, H. Schlager, T. Pucik, and M. C. Barth (2016). "On the origin of pronounced O3 gradients in the thunderstorm outflow region during DC3." *J. Geophys. Res. Atmos.*, 121, 6600--6637. doi:[10.1002/2015JD024279](https://doi.org/10.1002/2015JD024279)

### Stratosphere-Troposphere Exchange

- Pan, L. L., C. R. Homeyer, S. Honomichl, B. A. Ridley, M. Weisman, M. C. Barth, J. W. Hair, M. A. Fenn, C. Butler, G. S. Diskin, J. H. Crawford, T. B. Ryerson, I. Pollack, J. Peischl, and H. Huntrieser (2014). "Thunderstorms enhance tropospheric ozone by wrapping and shedding stratospheric air." *Geophys. Res. Lett.*, 41, 7785--7790. doi:[10.1002/2014GL061921](https://doi.org/10.1002/2014GL061921)
- Phoenix, D. B., and co-authors (2020). "Mechanisms Responsible for Stratosphere-to-Troposphere Transport Around a Mesoscale Convective System Anvil." *J. Geophys. Res. Atmos.*, 125, e2019JD032016. doi:[10.1029/2019JD032016](https://doi.org/10.1029/2019JD032016)

### Lightning Meteorology

- Basarab, B. M., S. A. Rutledge, and B. R. Fuchs (2015). "An improved lightning flash rate parameterization developed from Colorado DC3 thunderstorm data for use in cloud-resolving chemical transport models." *J. Geophys. Res. Atmos.*, 120, 9481--9499. doi:[10.1002/2015JD023470](https://doi.org/10.1002/2015JD023470)

**Joint Special Section:** Most DC3 results are published in a joint special section *"Deep Convective Clouds and Chemistry 2012 Studies (DC3)"* in [JGR Atmospheres and Geophysical Research Letters](https://agupubs.onlinelibrary.wiley.com/doi/toc/10.1002/(ISSN)2169-8996.DEEPCON1).
