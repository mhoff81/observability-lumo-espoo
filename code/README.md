# `code/` — analysis scripts

Run every command below from the repository root (`observability-lumo-espoo/`).

### `espoo_mast_observability.py`

Computes the Espoo Kurttila mast's Sentinel-1 observability from its 2-year
burst backfill: per-acquisition whole-target echo coherence (`gamma^2`) and the
echo mask size, compared against the LUMO healthy/ASC/DESC reference medians, to
judge whether the mast is a usable scatterer (not a damage verdict — this site
has no damage-state label). Writes `../data/espoo_mast_observability.json`,
`ESPOO_MAST_OBSERVABILITY.md` and a `.png` next to itself.

```bash
python3 code/espoo_mast_observability.py
python3 code/espoo_mast_observability.py --asset-id <uuid> --request-id <uuid>
```

### `figures_data_csv.py`

Materialises the per-record Sentinel-1 burst/weather data of both sites into one
flat, 78-column CSV each — `data/espoo_channels.csv` and `data/lumo_channels.csv`
— so the two channel figures run from a single file instead of the underlying
JSON artifacts. It reads only `data/*.json` in this repository and writes only
`data/*.csv`; it never reaches a database or a live API. The weather columns
(`wind_speed_ms`, `wind_gust_ms`, `temperature_c`, `precipitation_mm`,
`relative_humidity_pct`, `snow_depth_m`, `snowfall_cm`, `weather_ts`) are frozen
values already baked into the `data/*.json` artifacts at acquisition time — they
were originally sourced from the Open-Meteo historical weather API by the
upstream campaign pipeline, not fetched by anything in this repository.

`espoo_mast_observability.json` and `lumo_tower_monthly_timeseries.json` (two of
the four source JSONs this script used to read) have since been removed — no
figure read either directly, and the two channel CSVs already carry everything
those two files fed into them (including, as of the 78-column schema, the full
`sub_aperture_brightness` block and the per-acquisition `segments` list that
used to be collapsed or dropped). That means `--site espoo` and the
`acquisition`/`campaign` half of `--site lumo` can no longer be *regenerated*
from scratch this way — the committed `data/*.csv` files are now the canonical
artifact for those record classes. `--selftest` detects the missing JSONs and
skips its artifact-level checks rather than failing.

```bash
python3 code/figures_data_csv.py             # both CSVs (needs all four source JSONs)
python3 code/figures_data_csv.py --site lumo # one site
python3 code/figures_data_csv.py --selftest  # offline: contract, codec, artifacts
```

### `recompute_lumo_coherence_states.py`

Recomputes `lumo_tower_coherence_states.json`'s aggregate statistics
(`per_state`, `tests`, `per_orbit`, `desc_tests`, `wind_control`) from
`data/lumo_channels.csv`'s `coherence` rows alone — a from-scratch port of the
external, not-included `analyze_lumo_tower_coherence.py` generator that
needs no raw binary SLC cache, because every field that script's statistics
need is already a flat CSV column. Two metadata fields that describe the raw
cache this repository does not hold (`n_bursts_cached`, `n_skipped`) are
carried as documented constants rather than recomputed. Its committed JSON has
since been removed from `data/` — `lumo_damping_gamma2_modulation.py` (its last
direct reader) now calls `recompute()` here in-memory instead, so `--check`
needs the original restored from git history to diff against.

```bash
python3 code/recompute_lumo_coherence_states.py --check        # diff vs a restored data/*.json
python3 code/recompute_lumo_coherence_states.py --out /tmp/x.json
```

### `recompute_lumo_monthly_amp_phase.py`

Recomputes `lumo_tower_monthly_amp_phase.json` in full (`state_analysis`,
`modulation_analysis`, `wind_direction_analysis`, `falsification_tests`,
`phase_interferometry_tests`, `shm_correlation_test`, `filtered_analysis`,
`wind_slope_state_test`, `polarization_analysis`, `bessel_aperture_analysis` and
the `monthly` strongest-candidate rollup) from `data/lumo_channels.csv`'s
`burst` rows alone — a from-scratch port of the external, not-included
`build_lumo_tower_monthly_amp_phase.py` generator. That
script's `analyze_burst()` needs the raw binary SLC cache this repository does
not hold, but every field its ten analysis functions read (including the full
`sub_aperture_brightness` block with its per-0.1-s-block `blocks` list, the
`filtered` amplitude variants, and the wind-direction/phase-RMS/displacement
fields the original attaches in place from an Open-Meteo fetch and
`lumo_tower_monthly_timeseries.json`) is already a flat CSV column, so none of
that needs to be re-run. `data/wind_response_shm.json` is still read directly
for the SHM-fit lookups. `source.cached_burst_count` (482, the pre-dedup raw
cache size) is carried as a documented constant rather than recomputed. Its
committed JSON has since been removed from `data/` — `lumo_damping_gamma2_
modulation.py` (its last direct reader) now calls `recompute()` here
in-memory instead, so `--check` needs the original restored from git history
to diff against.

```bash
python3 code/recompute_lumo_monthly_amp_phase.py --check        # diff vs a restored data/*.json
python3 code/recompute_lumo_monthly_amp_phase.py --out /tmp/x.json
```

### `lumo_damping_frequencies.py`

Step 1 of the LUMO modal analysis: computes the measured natural frequencies and
half-power damping ratios (ζ) of the LUMO lattice tower in its healthy, DAM3,
DAM4 and DAM6 states directly from the public Uni-Hannover SHM `.mat` recordings
(not included in this repository, see the script's docstring for the dataset DOI
and download instructions). Writes `../data/lumo_damping_frequencies.json`.

```bash
python3 code/lumo_damping_frequencies.py --root /path/to/lumo_data --verify
```

### `lumo_sds.py`

Step 2: turns the frequencies and ζ values from `lumo_damping_frequencies.json`
into the three 0–10 state-discrimination scores (`LUMO_H1_FREQUENCY`,
`LUMO_H1_DAMPING`, `LUMO_H2_DAMPING`) via Cliff's delta between healthy and
damaged recordings, paired within each field campaign. Also computes two
whole-tower channels from `../data/lumo_channels.csv`, both using the same
max-across-states aggregate (not the modal mean): the gamma2 coherence SDS
(via `recompute_lumo_coherence_states.recompute()`; paper's "1.39" max pairwise
/ Table 4's "1.4", pooled "0.10") and the phase coherence SDS (via
`recompute_lumo_monthly_amp_phase.recompute()`'s `phase_coherence` field; the
reproducible pooled Healthy-vs-all-damaged value is 0.91 - this was found to
disagree with the paper's current "1.04" and is pending a paper correction).
All three channel families are verified against the single
`../data/lumo_sds_expected.json` reference (`channels` block for the modal
scores, `gamma2` and `phase_coherence` blocks for the two coherence channels).

```bash
python3 code/lumo_sds.py --verify --explain
python3 code/lumo_sds.py --selftest
```

### `lumo_slc_ranged_read.py`

Reproduces **one** cached LUMO burst window/strip by ranged-reading only the
bytes it needs directly out of the Sentinel-1 SLC on CDSE, and compares the
result sample-by-sample against the cache the Rust pipeline (`tower/backend`)
wrote — without ever downloading the full measurement TIFF. Needs network access,
CDSE credentials (see `../secrets/README.md`) **and** a local burst cache
directory (`--cache-dir`) written by the Rust pipeline. That cache is
machine-local binary pipeline output, not part of this repository and **not
needed to review this work** — `../data/audit.csv` already carries the result of
running this check over the whole cache (see `../data/README.md`); this script
is included only so the method that produced that CSV is fully documented and
re-runnable against a live cache.

```bash
python3 code/lumo_slc_ranged_read.py --cache-dir DIR --entry 2021-04/2021-04-03_1e17aa73 --verify
python3 code/lumo_slc_ranged_read.py --selftest
```

### `lumo_burst_cache_audit.py`

Runs the same two anchoring rules as `lumo_slc_ranged_read.py` over an **entire**
cache directory, scoring every entry against both the pre- and post-fix anchor
behaviour, classifying it by vintage, and checking whether the burst that hosts
it actually contains the mast. Also needs network access, CDSE credentials and a
local burst cache directory (`--cache-dir`) from the Rust pipeline — same caveat
as above: that cache is not in this repository and not needed to review this
work, since its output is already committed as `../data/audit.csv` /
`../data/audit.json`.

```bash
python3 code/lumo_burst_cache_audit.py --cache-dir DIR --limit 2
python3 code/lumo_burst_cache_audit.py --cache-dir DIR --stride 10 --csv data/audit.csv
python3 code/lumo_burst_cache_audit.py --selftest
```

### `fig_espoo_channels.py`, `fig_lumo_channels.py`, `fig_burst_cache_audit.py`, `lumo_damping_gamma2_modulation.py`

The four plotting scripts behind the paper figures. Each reads only
`data/*.csv` / `data/*.json`, writes its `.json` artifact back to `data/` and
its `.md` report / `.png` figure to `../figures/` — fully offline, no network,
database or Rust pipeline. They're normally run through the matching `.sh`
wrapper in `figures/` (see [figures/README.md](../figures/README.md)); each
docstring documents its own `--csv`/`--out-dir` flags for ad hoc runs.

```bash
../figures/fig_espoo_channels.sh
../figures/fig_lumo_channels.sh
../figures/fig_burst_cache_audit.sh
../figures/lumo_damping_gamma2_modulation.sh
```
