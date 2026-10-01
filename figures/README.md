# `figures/` — shell wrappers + `.md`/`.png` reports

The plotting scripts themselves live in `../code/` (they share that
directory's Python conventions and imports); each has a `.sh` wrapper here
that runs it with the right working directory. Every script reads only
`../data/*.csv` / `../data/*.json`, writes its `.json` artifact to `../data/`
and its `.md` report / `.png` figure here, next to its `.sh` wrapper. No
network, no database, no Rust pipeline — matplotlib is optional (without it
the `.png` is skipped, the `.json`/`.md` are still written).

### `fig_espoo_channels.sh` → `../code/fig_espoo_channels.py`

Five stacked time-history panels for the Espoo Kurttila mast over the 2-year
Sentinel-1 backfill: whole-target coherence (`gamma^2`, with LUMO reference
medians for scale), echo mask size (the clutter discriminator), tower phase
coherence, tower phase SNR and tower phase RMS activity. Reads
`../data/espoo_channels.csv`.

```bash
./figures/fig_espoo_channels.sh
./figures/fig_espoo_channels.sh --no-lumo   # drop the LUMO reference lines
```

### `fig_lumo_channels.sh` → `../code/fig_lumo_channels.py`

Five stacked time-history panels for the LUMO lattice tower over its full
Sentinel-1 record (2020-08 .. 2021-07), with the DAM3/DAM4/DAM6 campaign
windows shaded: SAR brightness, raw SAR phase, whole-structure `gamma^2`,
dwell/sub-aperture modulation and data availability (de-duplicated overpasses
per month). Reads `../data/lumo_channels.csv`.

```bash
./figures/fig_lumo_channels.sh
```

### `lumo_damping_gamma2_modulation.sh` → `../code/lumo_damping_gamma2_modulation.py`

Tests whether SAR sees LUMO's modal damping indirectly: combines the
half-power damping ratio (ζ, from the raw SHM recordings) with the whole-tower
coherence (`gamma^2`), intra-dwell brightness modulation and SHM wind-response
curves for the DAM3/DAM4/DAM6 states, each against its own healthy campaign.
Reads `../data/lumo_damping_ratio.json`, `../data/lumo_tower_coherence_states.json`,
`../data/lumo_tower_monthly_amp_phase.json` and `../data/wind_response_shm.json`.

```bash
./figures/lumo_damping_gamma2_modulation.sh
```

### `fig_burst_cache_audit.sh` → `../code/fig_burst_cache_audit.py`

Five stacked panels documenting the burst-cache audit rather than a
measurement: containment (distance to nearest geolocation-grid node),
window-match fraction per anchor rule and vintage, ground error of both rules,
anchor drift the `fad0e46` fix removed, and per-month availability of the
audited sample. Reads only `../data/audit.csv` — fully offline, no cache
directory or network needed.

```bash
./figures/fig_burst_cache_audit.sh
```
