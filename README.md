# LUMO — Modal Frequencies and Damping Scores Recomputed from the Raw Recordings

## Objective

Derive the **measured natural frequencies** and the **half-power damping ratios**
of the LUMO lattice tower in the **healthy**, **DAM3**, **DAM4** and **DAM6**
states directly from the public Uni-Hannover SHM recordings, in two
self-contained scripts, so that the three quoted results can be reproduced — or
contradicted — from source data alone:

| Channel | Value | What it is |
|---|---|---|
| `LUMO_H1_FREQUENCY` | 10.0 | h1 peak frequency over 30 recordings |
| `LUMO_H1_DAMPING` | 9.7333333333 | h1 half-power ζ over 30 recordings |
| `LUMO_H2_DAMPING` | 3.2 | h2 half-power ζ over 20 recordings (see the exclusion) |

Both quantities come out of one and the same evaluation of the spectrum: the peak
frequency is the **vertex of the parabola through the log-PSD** found inside the
half-power (ζ) routine, aggregated per state as the **median over the 5
recordings**. Step 1 stores the frequencies, every per-recording ζ and the
resolution floor below which a ζ is meaningless; step 2 turns ζ into the three
0–10 discrimination scores above.

## Files

| File | Role |
|------|------|
| `lumo_damping_frequencies.py` | step 1: raw `.mat` → frequencies + ζ (numpy + scipy only) |
| `lumo_damping_frequencies.json` | **generated** by step 1: 6 states × 3 modes + all 30 per-recording rows |
| `lumo_sds.py` | step 2: those rows → the three scores above (standard library only) |
| `lumo_frequencies.json` | **hand-curated** cross-reference: paper Table 2 + local medians + mode mapping |
| `lumo_sds_expected.json` | **optional** pinned transcription of the three stored scores; the `--verify` reference only |

Neither script **reads** `lumo_frequencies.json` or `lumo_sds_expected.json` to
compute anything. Those two are consulted only by `--verify`, which compares the
freshly computed values against them and exits non-zero on a mismatch. Delete
both reference files and the analysis still produces exactly the same numbers.

Both scripts are self-contained: Python 3 plus numpy/scipy for step 1, the
standard library alone for step 2 — no other project code, no database, no
network, no Rust. Everything needed to recompute the three scores is in this
folder.

## Quick start

```bash
python3 lumo_damping_frequencies.py --root /path/to/lumo_data --verify   # ~71 s
python3 lumo_sds.py --verify --explain                                   # < 1 s
```

`--root` defaults to `$LUMO_DATA_DIR`, so `export LUMO_DATA_DIR=...` once and
then just run the script. Step 2 reads the JSON step 1 wrote (`--from`), so it
needs no dataset at all — and `lumo_sds.py --selftest` checks the score's
arithmetic on built-in fixtures with no data and no reference file.

* Dependencies: Python 3, numpy, scipy for step 1; the standard library alone for
  step 2. No matplotlib, no network, and nothing written outside `--out` /
  `--json` (no figures, no git operations).
* Runtime (8 cores, single process, warm page cache, measured 2026-09-30):
  **71 s for all 30 recordings**, ≈ 2.4 s each, almost all of it the
  channel-mean Welch PSD; step 2 is instant.
* Exit codes, step 1: `0` ok, `1` unusable `--reference` (or an import error),
  `2` dataset missing/empty **or** bad usage (argparse exits with 2 as well),
  `3` verification mismatch.
* Exit codes, step 2: `0` ok, `1` unusable `--expected`, `2` input JSON missing
  or incompatible, `3` verification mismatch, `4` selftest failure.

## Data

Not in this repository — 3 archives, ≈ 1.9 GB zipped / **3.8 GB extracted**,
licensed **CC BY 3.0**.

Dataset: *"LUMO — Leibniz University Test Structure for Monitoring"*, Wernitz,
Hofmeister, Jonscher, Gießmann & Rolfes, Institut für Statik und Dynamik,
Leibniz Universität Hannover — DOI
[10.25835/0027803](https://doi.org/10.25835/0027803).

| Archive | Extracts to | `.mat` | Bytes | Resource id |
|---------|-------------|--------|-------|-------------|
| `exemplary_datasets_dam6_111.zip` | `01_Healthy`, `02_DAM6_111` | 10 | 680 154 659 | `d9661b47-1f25-4194-99e8-6805fcd5810e` |
| `exemplary_datasets_dam4_111.zip` | `03_Healthy`, `04_DAM4_111` | 10 | 665 801 542 | `83ee4ee4-d86b-49ce-92dd-743bd779d845` |
| `exemplary_datasets_dam3_111.zip` | `05_Healthy`, `06_DAM3_111` | 10 | 638 712 761 | `78da8221-a6bb-4ad9-9f53-e3d9164b3c52` |

Landing page:
<https://data.uni-hannover.de/dataset/93b52576-6a5a-4ce9-8c27-a0372590f7b0>
(dataset `readme.pdf` 1 853 104 B, resource `bd0a6d0a…`; the landing page also
carries `lumo_fem_healthy.inp` and the unused `_010` archives; meteorological
data on request from `public.data@isd.uni-hannover.de`).
The URLs, byte sizes and `accept-ranges: bytes` support below were verified
against the live server on 2026-09-30.

```bash
B=https://data.uni-hannover.de/dataset/93b52576-6a5a-4ce9-8c27-a0372590f7b0/resource
mkdir -p lumo_data && cd lumo_data
curl -L -C - -O $B/d9661b47-1f25-4194-99e8-6805fcd5810e/download/exemplary_datasets_dam6_111.zip
curl -L -C - -O $B/83ee4ee4-d86b-49ce-92dd-743bd779d845/download/exemplary_datasets_dam4_111.zip
curl -L -C - -O $B/78da8221-a6bb-4ad9-9f53-e3d9164b3c52/download/exemplary_datasets_dam3_111.zip
wc -c *.zip            # 680154659 665801542 638712761
for z in *.zip; do python3 -m zipfile -e "$z" .; done   # `unzip` may not exist
```

All three responses carry `accept-ranges: bytes`, so `curl -C -` resumes an
interrupted download. The archives already ship the `01…06` numbering, so the
extraction order does not matter, nothing collides, and **no renaming is
needed** — the archive directory names are exactly the `NN_State` names this
script scans for.

```bash
python3 lumo_damping_frequencies.py --root lumo_data --verify
```

Recordings are found with `root/**/<state>/*.mat`, which covers both the flat
layout produced by extraction and the nested layout of the LUIS bulk vault.

```
lumo_data/
├── 01_Healthy/      5 × SHMTS_2020100*.mat    (healthy, Oct-2020, DAM6 campaign)
├── 02_DAM6_111/     5 × SHMTS_2020101*.mat    (all struts at level 6 removed)
├── 03_Healthy/      5 × SHMTS_2020110*.mat    (healthy, Nov-2020, DAM4 campaign)
├── 04_DAM4_111/     5 × SHMTS_2020111*.mat    (all struts at level 4 removed)
├── 05_Healthy/      5 × SHMTS_2021031*.mat    (healthy, Mar-2021, DAM3 campaign)
└── 06_DAM3_111/     5 × SHMTS_2021032*.mat    (all struts at level 3 removed)
```

`_111` = *all* struts of that level removed (paper cases A/B/C); the `_010`
single-strut archives are cases D/E/F and are **not** used here. Each `.mat`
holds a struct `Dat` with `Fs` ≈ 1651.6 Hz and `Data` of shape (990600 × 22);
columns 0–17 are the 18 accelerometer channels in **g**, columns 18–20 strain
and column 21 temperature.
## Method

Per recording and mode — the constants are all listed in the JSON's `method`
block, and the estimator is implemented **verbatim and in full inside this
folder** (step 1 imports nothing beyond numpy/scipy), so the ζ values below are
computed here and not taken over from anywhere:

1. Load the 18 accelerometer channels `Dat.Data[:, :18]`, convert g → m/s² and
   detrend them.
2. Welch PSD with `nperseg = 32 s` (Hann window — the scipy default — 50 %
   overlap, `scaling="density"`): bin width 0.0313 Hz. The 18 channel spectra
   are averaged into one channel-mean spectrum.
3. Noise floor = median of that spectrum over 1–50 Hz.
4. Fundamental: the dominant bin inside **2.0–3.5 Hz**. Higher modes: the
   dominant bin inside a **state-adaptive** ±0.9 Hz window around the expected
   frequency, because the modes move with damage — a fixed 13.5 Hz band would sit
   on noise for DAM6.
5. SNR gate: the peak must exceed **3×** the noise floor, otherwise the mode
   counts as *not excited* for that recording and is excluded from the median
   (never zero-filled).
6. `f_peak` = peak bin **+ the sub-bin offset of the parabola through the
   log-PSD** of that bin and its two neighbours. This is the value produced
   inside `zeta_half_power()`; it is *not* a plain `argmax`. Do not simplify it —
   the third decimal moves.
7. State value = `np.median` over the recordings of the state, with `p25`/`p75`
   and min/max kept as the spread. Every per-recording `f_peak` and `snr` is in
   the JSON, so any median can be re-derived by hand.
8. `zeta` = half-power bandwidth at the peak bin, `(f_hi − f_lo) / (2 f_peak)`,
   with the −3 dB crossings interpolated linearly. Stored with it:
   `resolution_floor = df / (2 f_peak)`, the half-power width that one 0.0313 Hz
   bin would produce on its own, and `resolvable = zeta > 3 × resolution_floor`.
   The floor is not bookkeeping: of the 30 recordings it is cleared by 2 for the
   fundamental, 7 for h1 and 11 for h2 (all 10 DAM3 h2 plus one DAM4 healthy h2),
   and a ζ below it is a bin artefact, not a damping estimate.

The adaptive windows are
`(13.5, 16.1)` for every healthy state, `(11.75, 14.0)` for DAM6,
`(12.5, 14.9)` for DAM3 and `(12.5, 16.1)` for DAM4. DAM4 keeps the healthy
16.1 Hz window because its h2 is the one mode that does **not** soften.

## Results

Medians of `f_peak` over the 5 recordings of each state, with `[p25, p75]`:

| State | Dir | n | fundamental | h1 | h2 |
|---|---|---|---|---|---|
| healthy (Oct-2020, DAM6 campaign) | `01_Healthy` | 5 | 2.8017 [2.7973, 2.8030] | 13.3932 [13.3881, 13.4195] | 16.0008 [16.0004, 16.2958] |
| DAM6 (all struts @ L6) | `02_DAM6_111` | 5 | 2.7685 [2.7566, 2.7693] | 11.6444 [11.6411, 11.6456] | 14.6968 [14.3001, 14.6981] |
| healthy (Nov-2020, DAM4 campaign) | `03_Healthy` | 5 | 2.7988 [2.7978, 2.8019] | 13.5215 [13.5141, 13.5290] | 16.0103 [16.0070, 16.0258] |
| DAM4 (all struts @ L4) | `04_DAM4_111` | 5 | 2.7688 [2.7654, 2.7740] | 12.3947 [12.3847, 12.3960] | 16.2215 [16.2109, 16.2217] |
| healthy (Mar-2021, DAM3 campaign) | `05_Healthy` | 5 | 2.7917 [2.7900, 2.8005] | 13.5525 [13.5414, 13.5537] | 15.9887 [15.9820, 16.0262] |
| DAM3 (all struts @ L3) | `06_DAM3_111` | 5 | 2.7924 [2.7766, 2.7941] | 12.5243 [12.5183, 12.5246] | 15.7621 [15.7619, 15.7623] |

Paired **within** each archive (its own healthy set vs its own damaged set —
the three healthy sets are not interchangeable, see the caveats):

| Campaign | Mode | Healthy (Hz) | Damaged (Hz) | Shift (Hz) | Shift (%) |
|---|---|---|---|---|---|
| DAM6 | fundamental | 2.8017 | 2.7685 | −0.0333 | −1.19 |
| DAM6 | h1 | 13.3932 | 11.6444 | −1.7487 | **−13.06** |
| DAM6 | h2 | 16.0008 | 14.6968 | −1.3039 | **−8.15** |
| DAM4 | fundamental | 2.7988 | 2.7688 | −0.0301 | −1.07 |
| DAM4 | h1 | 13.5215 | 12.3947 | −1.1268 | **−8.33** |
| DAM4 | h2 | 16.0103 | 16.2215 | +0.2112 | +1.32 |
| DAM3 | fundamental | 2.7917 | 2.7924 | +0.0006 | +0.02 |
| DAM3 | h1 | 13.5525 | 12.5243 | −1.0282 | **−7.59** |
| DAM3 | h2 | 15.9887 | 15.7621 | −0.2267 | −1.42 |

What stands out:

* **h1 is the strong damage indicator**: −13.1 % (DAM6), −8.3 % (DAM4), −7.6 %
  (DAM3). It has no counterpart in the paper's Table 2, which lists only the
  B2-y…B5-y family.
* **h2 tracks the paper's B2-y** (16.00 Hz vs 15.94 Hz healthy, +0.4 %) and drops
  8.15 % for DAM6, −10.3 % in the paper — the one clean one-to-one pair. For
  DAM3/DAM4 its shift (+/−1.3 %) is inside the bin quantisation, see caveats.
* **The fundamental is nearly damage-insensitive** (−1.2 % max, +0.02 % for
  DAM3); its per-state medians also differ by 1 % between the three campaigns,
  so it behaves more like a slow season/temperature indicator than a damage one.
* **The DAM6 h2 median hides a bimodal window**: 3 of 5 recordings peak at
  14.697 Hz, 2 at 14.300 Hz (hence `p25 = 14.300`). The median reproduces the
  curated 14.697 Hz value, but a damage decision built on a single DAM6
  recording would be a coin flip. The same bimodality appears in `01_Healthy`'s
  h2 (4 recordings at 16.000 Hz, one at 16.315 Hz).
* **DAM4's h2 is the exception that matters for observability**: it moves *up*
  (+1.3 %), so a damage index that assumes "damage → frequencies fall" would
  score DAM4 h2 as healthy while h1 already fell 8.3 %.
## Damping: the half-power ζ values

Same estimator call, same peak bin — ζ is the *width* of that peak. Medians over
the 5 recordings of each state, with `n_resolvable/5` in brackets:

| State | Dir | ζ fundamental | ζ h1 | ζ h2 |
|---|---|---|---|---|
| healthy (Oct-2020) | `01_Healthy` | 0.013881 [0/5] | 0.004031 [4/5] | 0.002522 [0/5] |
| DAM6 | `02_DAM6_111` | 0.015216 [1/5] | 0.002902 [0/5] | 0.002752 [0/5] |
| healthy (Nov-2020) | `03_Healthy` | 0.014900 [0/5] | 0.003280 [1/5] | 0.002613 [1/5] |
| DAM4 | `04_DAM4_111` | 0.016546 [1/5] | 0.002453 [0/5] | 0.002120 [0/5] |
| healthy (Mar-2021) | `05_Healthy` | 0.014699 [0/5] | 0.003374 [2/5] | 0.003665 [5/5] |
| DAM3 | `06_DAM3_111` | 0.014364 [0/5] | 0.002575 [0/5] | 0.049599 [5/5] |

* **h1 damping falls in every campaign** — 0.004031 → 0.002902 (DAM6),
  0.003280 → 0.002453 (DAM4), 0.003374 → 0.002575 (DAM3): 5 vs 5 completely
  separated, all three times (Cliff's δ = −1.0). A frequency channel and a
  damping channel agree on h1, and only on h1.
* **h2 damping does not**: it rises slightly for DAM6 (0.002522 → 0.002752,
  δ = +0.20, i.e. 5 net pairs of 25) and falls for DAM4 (0.002613 → 0.002120,
  δ = −0.44), while **DAM3's h2 jumps by a factor of ~14** (0.003665 → 0.049599).
  That jump is a resolution artefact, not a modal property: in the healthy DAM3
  state ζ sits only ~1.25 × above the resolvability threshold (0.003665 vs
  3 × floor = 0.00293), and the damaged peak is so broad that its ζ is 50 × the
  floor. It is the one block excluded from the score below.
* The **fundamental's ζ is unresolvable in every state** (0–1 of 5 recordings
  clear the floor): no damping estimate for a 2.8 Hz mode from a 32 s segment
  exists here, which is why the fundamental is not an evidence channel.

## State-discrimination scores (SDS)

`lumo_sds.py` (step 2) turns only the h1 frequency and the h1/h2 ζ above into the
three quoted scores. For one channel:

```
SDS = 10 × mean over campaigns of |δ_campaign|,  clamped to 0 … 10
δ   = Cliff's delta between the 5 healthy and the 5 damaged recordings of one
      campaign: (#y > x − #y < x) / (n_healthy × n_damaged), strict comparisons,
      ties counting for neither side
```

Pairing is **within** a campaign, never across the three healthy sets, which are
not interchangeable. `python3 lumo_sds.py --explain`, run 2026-09-30:

```
channel              mode  value        n  delta per campaign                           excluded             SDS
LUMO_H1_FREQUENCY    h1    f_peak_hz   30  dam3 -1.0000  dam4 -1.0000  dam6 -1.0000                10.0000000000
LUMO_H1_DAMPING      h1    zeta        30  dam3 -0.9200  dam4 -1.0000  dam6 -1.0000                 9.7333333333
LUMO_H2_DAMPING      h2    zeta        20  dam3 excluded  dam4 -0.4400  dam6 +0.2000    dam3/h2     3.2000000000
(informational)      fundamental f_peak_hz   30  dam3 -0.2000  dam4 -1.0000  dam6 -1.0000                 7.3333333333  no evidence channel

modal exclusions:
LUMO_H1_FREQUENCY (h1 f_peak_hz): no exclusion - SDS 10.0000 (n=30)
LUMO_H1_DAMPING (h1 zeta): no exclusion - SDS 9.7333 (n=30)
LUMO_H2_DAMPING (h2 zeta): exclusion ON  SDS 3.2000 (n=20)  |  OFF SDS 5.4667 (n=30)
```

* **h1 is perfect on both readings** — 10.0 from the peak frequency (all 15
  healthy/damaged pairs separated in all three campaigns) and 9.7333 from ζ, the
  score stopping one campaign short of 10.0 because DAM3's h1 loses 2 of its 25
  pairs (δ = −0.92). Frequency *and* damping carry the same damage information
  for h1, which is what makes it the strongest mode here.
* **h2 damping is weak** — 3.2, and for a structural reason: h2 ζ moves in
  *opposite* directions in the two regular campaigns (up for DAM6, down for
  DAM4), so two of the three campaigns average out instead of adding up.
* **The one rule that changes a number**: `("dam3", "h2")` is dropped from both
  sides of every channel, which is why `LUMO_H2_DAMPING` rests on 20 recordings
  and reads 3.2 — with the pair kept it reads **5.4667** on 30. `--explain`
  prints the pair; `lumo_sds.py --selftest` asserts it, together with the tie
  rule, the 5-vs-5 quantisation (one net pair = 0.04, one recording out of 5
  moving fully = 0.20) and the clamp, on fixtures with no data at all.
## Verification

`--verify` never feeds the reference into the computation: once the medians are
built, it compares all 18 of them with `lumo_frequencies.json →
local_measured_hz` and exits `3` on any mismatch. The tolerance is 0.5 mHz, half
of the reference's own 1 mHz precision. Run of 2026-09-30:

```
18/18 medians match lumo_frequencies.json -> local_measured_hz (tol 0.0005 Hz)
$ echo $?   # 0
```

Every delta is ≤ 0.49 mHz, i.e. exactly what rounding the recomputed median to
three decimals would produce — the curated values *are* the medians of this
estimator, not an independent measurement. The comparison exists to notice a
drift in the estimator first; if a change is deliberate, update
`lumo_frequencies.json` by hand in the same commit.

### The two score checks (step 2)

`lumo_sds.py --verify` compares the three scores, their `n_samples` and every
per-campaign δ against `lumo_sds_expected.json`, the transcription of the values
held in the external evidence-channel store:

```
  MATCH    LUMO_H1_DAMPING      sds=  9.7333333333  n=30
  MATCH    LUMO_H1_FREQUENCY    sds= 10.0000000000  n=30
  MATCH    LUMO_H2_DAMPING      sds=  3.2000000000  n=20
  3/3 channels match lumo_sds_expected.json (tol 1e-09)
$ echo $?   # 0
```

That file is optional and never on the compute path: move it away and the scores
are identical — the script then says that verification was skipped and still
exits `0`. The second check needs neither data nor reference:
`lumo_sds.py --selftest` recomputes the metric on built-in fixtures (the tie rule,
complete separation, the 5-vs-5 quantisation, the exclusion at 20 vs 30 rows, and
the three score values) and exits `4` if any assertion fails.

## Determinism, portability, footprint

* **Reproducible**: sorted iteration everywhere, `np.median`, no RNG, no
  threads, no timestamps inside the data. Output is byte-identical except for
  `generated`, which `--date` pins, so two machines can `diff` the JSON directly.
* **Portable**: recordings are found with `root/**/<state>/*.mat`, which covers
  both a flat extraction directory and the nested LUIS vault. The only paths are
  `--root`/`$LUMO_DATA_DIR`, `--out`, `--reference`; nothing is hard-coded and
  the script runs from any working directory.
* **Clean**: no absolute paths, host names or machine state in the JSON; writes
  only `--out` / `--json`; no matplotlib, no PNG, no `__pycache__` (a
  `.gitignore` keeps bytecode out anyway). Both scripts perform **no git
  operations** — review and commit by hand. (`python3 -m py_compile` still writes
  a `__pycache__` directory even with `PYTHONDONTWRITEBYTECODE=1`; use
  `python3 -c "import ast,pathlib; ast.parse(pathlib.Path('lumo_damping_frequencies.py').read_text())"`
  — and the same with `lumo_sds.py` — for a bytecode-free syntax check.)

## Caveats

The JSON's `caveats` array is the authoritative list; in short:

* **5 recordings per state, one field campaign per damage state.** Seasonal and
  wind confounds apply, and the three healthy sets are *not* interchangeable —
  their fundamentals differ by up to 1 % (2.7917 / 2.7988 / 2.8017 Hz). Only
  paired within-campaign shifts are meaningful.
* A mode below the 3× SNR gate contributes **no** frequency for that recording
  (`n_excited < n_recordings`), never a zero. All 30 recordings here cleared the
  gate for all three modes.
* The p25/p75 spreads are dominated by ambient excitation and by *which* of two
  neighbouring peaks wins the window in a given recording (DAM6 h2: 14.300 vs
  14.697 Hz), not by estimator noise — a single recording is not a state.
* DAM3/DAM4 h2 shifts are inside the 0.25 Hz frequency grid of the paper's own
  forced-excitation tests (±0.8 % at 16.1 Hz), so only DAM6's h2 shift is clearly
  resolvable; the paper's own B2-y rows agree with these numbers only for DAM6
  (−10.3 % published vs −8.15 % here; +1.3 %/−1.4 % here vs −0.3 %/+0.8 %
  published for DAM4/DAM3).
* The paper reports **no** damping ratios for LUMO (its "damping rate of 0.01"
  belongs to the simulated 3-DOF example), so there is no published ζ to compare
  against; ζ also cannot be derived from frequencies (6e-5 Hz effect vs a
  0.031 Hz bin), and it is only resolvable for a mode whose half-power width
  clears the one-bin floor — never for the fundamental.
* **The scores are quantised by 5 recordings per side.** With 25 pairs, |δ| moves
  in steps of 0.04, and one recording out of five moving fully out of the healthy
  range is already 0.20 — 2.0 on the 0–10 scale before any averaging.
  `LUMO_H2_DAMPING` = 3.2 therefore rests on δ = −0.44 and +0.20 built from 5-vs-5
  comparisons, and re-recording one state could move it by roughly ±1.5. The
  endpoints (h1, where δ = −1.0 three times over) are robust; the middle of the
  scale is not.

## Folder contents, and what is deliberately absent

Everything needed to check the three scores is inside this folder:

| File | Contents |
|---|---|
| `lumo_damping_frequencies.py` | step 1: loading, Welch PSD, peak and ζ estimation, windows, exit codes |
| `lumo_damping_frequencies.json` | `generated`/`method`/dataset provenance; per state the frequency **and** ζ blocks; all 30 per-recording rows with `f_peak_hz`, `snr`, `zeta`, `resolution_floor`, `resolvable` |
| `lumo_sds.py` | step 2: Cliff's δ, the exclusion rule, `--selftest`, `--verify`, `--explain` |
| `lumo_sds_expected.json` | the pinned transcription of the three stored scores — reference only |
| `lumo_frequencies.json` | paper Table 2 as transcribed, the 18 curated medians, the mode mapping, the caveats |
| `.gitignore` | keeps `__pycache__` out of the history |

Nothing here points at a second codebase, a database, a host or a working
directory: step 1 takes `--root` and writes one JSON, step 2 reads that JSON and
prints; both run from any `cwd`, and every number in this note can be traced back
to a per-recording row in the folder. The only external references are to the
source data itself — the dataset DOI, its landing page and the three download
URLs above — and to the paper whose Table 2 is transcribed; where those paper
rows agree with these recordings and where they do not is stated in
`lumo_frequencies.json → shift_check_B2_y_vs_h2_pct` and in the caveats above.


