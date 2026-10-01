# LUMO tower — Observation-Channel Time Histories (paper figure)

178 distinct Sentinel-1 overpasses (de-duplicated from 356 artifact rows: each overpass appears once per polarisation and some under two burst ids ~3 s apart), 2020-08 .. 2021-07. DAM 3/4/6 campaigns shaded on every panel.

| channel | n | median |
|---------|--:|-------:|
| brightness_ratio | 178 | 0.0175 |
| raw_phase_coherence | 178 | 0.1314 |
| gamma2 | 178 | 0.04489 |
| modulation_depth | 178 | 0.8404 |

## Caveats

* **Panel 5** is data availability, not a measurement channel: de-duplicated overpasses per month, stacked by orbit. It shows the revisit cadence the other four panels' monthly medians are drawn from.
* Counts and medians are per **distinct overpass** (acquisition minute + orbit, VV preferred); the raw stacks hold 356 rows. The backend collapses the same pseudo-replicates via `insar_monitor::acquisition_key`, so these numbers match the DB.
* Single-look SLC phase is backscatter/atmosphere dominated (circular sigma ~ 2 rad); phase coherence is near the clutter floor.
* Amplitude/brightness does not separate the damage states (all Welch p > 0.10); vv is ~2.1x brighter than vh (pol confound).
* Orbit stratification dominates the amplitude scale (ASC ~ 2x DESC).

Build: `python3 fig_lumo_channels.py`.
