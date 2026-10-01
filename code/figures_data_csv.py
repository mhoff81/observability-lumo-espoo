#!/usr/bin/env python3
"""Write the per-site channel CSVs the two channel figures run from.

The `figures/` scripts take different inputs: these two CSVs for the channel
figures, its own JSON for `lumo_damping_gamma2_modulation.py`, the audit sweep's
CSV for `fig_burst_cache_audit.py`. This script materialises the *per-record*
half of the channel figures' input - the Sentinel-1 burst channels plus the
weather that comes with them - as one flat CSV per site, so
`fig_espoo_channels.py` and `fig_lumo_channels.py` run from a single file
instead of the analysis JSONs directly:

  * `data/espoo_channels.csv` - 150 `measurement` rows (the Espoo backfill);
  * `data/lumo_channels.csv`  - 356 `burst` + 178 `coherence` + 482 `acquisition`
    + 13 `campaign` rows (the LUMO lattice mast campaign).

Those two files are the **only** input those two figures read.

THE COLUMN CONTRACT (78 columns, fixed order, identical in both files)

  `site`, `record_class`, `record_key`      identity: what this row is
  `ESPOO_COLUMNS`   (27)                    the Espoo measurement vocabulary
  `LUMO_COLUMNS`    (31)                    the LUMO burst/coherence additions
  `EXTRA_COLUMNS`   (17)                    weather, campaign window, DB-only

The middle 58 columns are the *union* of the three per-record artifacts with
four name unifications, which is what the "58-column" schema always was:

  * `lumo_tower_coherence_states.json`  `gamma2`    -> `coherence_gamma2`
  *                                     `n_masked`  -> `coherence_masked_pixels`
  *                                     `date`      -> `acquisition_date`
  * both LUMO artifacts                 `orbit`     -> `orbit_direction`

so that 27 (Espoo) + 30 (the LUMO keys not already among them) + 1
(`peak_intensity`, the one coherence-only key) = 58. The two blocks are disjoint:
a name both artifacts use is listed once, in the Espoo block.
The fields no figure reads per record (`filtered`, `weather`,
`sub_aperture_brightness`, `segments`) stay single columns holding their full
JSON - nothing a source artifact carries is collapsed or dropped any more -
and the weather scalars that block carries are *also* lifted into
`EXTRA_COLUMNS` so no weather value needs decoding JSON to be read.

WHY THE EXTRA COLUMNS ARE NOT OPTIONAL

  * `phase_snr_db`, `amplitude_measured_m`, `amplitude_expected_m` are plotted /
    loaded by `fig_espoo_channels.py` but exist in **no JSON** read by this
    project - only in the live `onboarder.insar_measurements` table, which this
    script never reads. Those cells stay empty, and the figure plots that
    channel as "no data in this channel".
  * `wind_gust_ms`, `precipitation_mm`, `relative_humidity_pct`, `snow_depth_m`,
    `snowfall_cm`, `weather_ts` are the rest of the weather dicts (the LUMO
    artifacts carry them per burst/acquisition, the Espoo table per measurement).
  * `start_date` / `end_date` / `period_index` are `campaign_timeline`, which
    `fig_lumo_channels.py` shades its panels with; `dwell_s`, `scatterers`,
    `is_damaged`, `campaign_period_index` are the per-acquisition fields of
    `lumo_tower_monthly_timeseries.json`. `segments` (that artifact's nested
    per-acquisition list, 3 per acquisition) is carried too, as its own JSON
    column - no figure reads it, but a reconstruction script needs it.
  * Those fourteen DB-only columns stay `null`/empty in the frozen Espoo
    artifact and in the CSV built from it - this script and the figures that
    read its CSV never touch a database or the live API; only `data/*.json`
    and `data/*.csv` in this repository are read.

READING IT BACK

`read_channels_csv(path)` returns `{record_class: [row, ...]}` where every row is
in its **source shape** again (`orbit`, `date`, `gamma2`, `n_masked`,
`sub_aperture_brightness` back as the full artifact dict), so the channel
figures keep running the loaders they ran on the JSON artifacts, one flat file
instead of three. At the switch the two figures were checked against the previous input:
`.json` and `.png` byte-identical, `.md` identical apart from the `Source:` line.
Floats use `repr()` (shortest round-trip), booleans are `true`/`false`, an unknown
value is an empty cell, and the columns in `TEXT_COLUMNS` are never parsed as
numbers.

USAGE
  python3 code/figures_data_csv.py                 # both CSVs into data/
  python3 code/figures_data_csv.py --site lumo     # one site
  python3 code/figures_data_csv.py --out-dir /tmp/x
  python3 code/figures_data_csv.py --selftest      # offline, no DB

OUTPUT: LF line endings, no BOM, no timestamp and no comment line, so two runs
on unchanged artifacts are byte-identical and `diff`-able. This script reads
only `data/*.json` in this repository and writes only `data/*.csv` - no
database, no live API, ever. CRLF, a BOM or a changed column order are
selftest failures, not cosmetics.

EXIT CODES: 0 wrote the CSVs (or the selftest passed); 2 a missing input
artifact or an output directory that does not exist; 4 the selftest failed
(`argparse` exits 2 on bad usage).
"""
import argparse
import csv
import json
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.normpath(os.path.join(HERE, os.pardir, "data"))

ESPOO_JSON = os.path.join(DATA_DIR, "espoo_mast_observability.json")
LUMO_AMP = os.path.join(DATA_DIR, "lumo_tower_monthly_amp_phase.json")
LUMO_COH = os.path.join(DATA_DIR, "lumo_tower_coherence_states.json")
LUMO_TS = os.path.join(DATA_DIR, "lumo_tower_monthly_timeseries.json")

# `--site` -> file name; the identity column carries the same word.
SITE_FILES = (("espoo", "espoo_channels.csv"), ("lumo", "lumo_channels.csv"))

IDENTITY_COLUMNS = ("site", "record_class", "record_key")

# The Espoo measurement vocabulary, in artifact order: `measurement` rows carry
# all 27. Five of these names are also keys of the LUMO artifacts
# (`acquisition_ts`, `brightness_ratio`, `phase_coherence`, `phase_rms_rad`,
# `wind_speed_ms`), so they are listed once, here, and the LUMO block below holds
# only what is new - the two blocks are disjoint.
ESPOO_COLUMNS = (
    "id", "acquisition_ts", "pass_label", "orbit_direction",
    "displacement_los_m", "brightness_ratio", "measured_frequency_hz",
    "baseline_frequency_hz", "frequency_drop_pct", "wind_speed_ms",
    "temperature_c", "traffic_load_label", "status",
    "phase_detected_frequency_hz", "phase_coherence", "phase_status",
    "coherence_gamma2", "coherence_masked_pixels", "ndvi", "ndvi_scene_date",
    "ndvi_confound_flagged", "sub_aperture_modulation",
    "coherent_sum_amplitude", "registration_status", "phase_rms_rad",
    "condition_label", "campaign_label")

# Everything the LUMO artifacts add on top of that vocabulary; `peak_intensity`
# is the only coherence-only column, the other four keys of that artifact are
# unifications of columns already listed above.
LUMO_COLUMNS = (
    "month", "burst_dir", "burst_id", "acquisition_date", "subswath",
    "polarisation", "damage_label", "structural_state", "campaign_range",
    "incidence_angle_deg", "amplitude", "phase_rad", "phase_deg",
    "mast_peak_row", "mast_peak_col", "mast_mean_amplitude",
    "mast_coherent_amplitude", "window_peak_amplitude",
    "window_peak_phase_deg", "strip_peak_amplitude", "strip_brightness_ratio",
    "filtered", "weather", "wind_direction_from_deg", "wind_direction_to_deg",
    "look_azimuth_deg", "wind_along_los_ms", "wind_cross_los_ms",
    "disp_middle_m", "peak_intensity", "sub_aperture_brightness")

# The weather/campaign/DB additions, present because the figures need them (see
# the module docstring) - not one of them is derivable from the 58 columns.
EXTRA_COLUMNS = (
    "phase_snr_db", "amplitude_measured_m", "amplitude_expected_m",
    "wind_gust_ms", "precipitation_mm", "relative_humidity_pct",
    "snow_depth_m", "snowfall_cm", "weather_ts", "dwell_s", "scatterers",
    "is_damaged", "campaign_period_index", "period_index", "start_date",
    "end_date", "segments")

# The 58 columns in the middle: the union of the three per-record artifacts,
# after the four unifications in the module docstring. Named, because the checks
# assert both halves against it.
CORE_COLUMNS = ESPOO_COLUMNS + LUMO_COLUMNS

COLUMNS = IDENTITY_COLUMNS + CORE_COLUMNS + EXTRA_COLUMNS

# `record_class` -> the artifact the rows come from. `measurement` is Espoo,
# the other four are LUMO (the coherence artifact is the de-duplicated
# whole-tower gamma2 stack, `acquisition` the timeseries the per-acquisition
# weather comes from, `campaign` the damage/healthy windows the figure shades).
RECORD_CLASSES = ("measurement", "burst", "coherence", "acquisition", "campaign")

# Never parsed as a number on the way back in: an all-digit label (or a date
# like `2024-09-01`) must stay a string.
TEXT_COLUMNS = frozenset({
    "site", "record_class", "record_key", "id", "acquisition_ts",
    "acquisition_date", "month", "pass_label", "orbit_direction", "status",
    "phase_status", "traffic_load_label", "registration_status",
    "condition_label", "campaign_label", "ndvi_scene_date", "burst_id",
    "burst_dir", "subswath", "polarisation", "damage_label",
    "structural_state", "campaign_range", "weather_ts", "start_date",
    "end_date"})

# Booleans: `_cell` writes `true`/`false`, the psql reader turns `t`/`f` into the
# same, `_value` turns both back into `bool`.
BOOL_COLUMNS = frozenset({"is_damaged", "ndvi_confound_flagged"})

# The nested source keys no figure reads field-by-field: they travel as one JSON
# cell each, so the file stays lossless without inventing dozens more columns.
# `segments` (acquisition, 3 per row) and `sub_aperture_brightness` (burst, the
# full block incl. `blocks`/`bessel_summary`) used to be dropped/collapsed; both
# are now carried whole, so a reconstruction script can recompute the aggregate
# analysis blocks the figures themselves never read.
JSON_COLUMNS = frozenset({"filtered", "weather", "segments",
                         "sub_aperture_brightness"})

# --- The cell codec ----------------------------------------------------------
def _cell(value):
    """One CSV cell. `None` is an empty cell (the file's only missing marker),
    a float is `repr()` so it round-trips bit-exactly, a bool is `true`/`false`,
    and a nested `dict`/`list` is compact JSON on one line."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, (dict, list)):
        return json.dumps(value, separators=(",", ":"), sort_keys=True)
    return str(value)


def _value(cell, column):
    """The inverse of `_cell`, `column`-aware: an empty cell is `None`, embedded
    JSON is decoded, a `TEXT_COLUMNS` cell stays the string it was written from,
    `true`/`false` (and the `t`/`f` the psql reader produces) are booleans, and
    anything else that is an integer/float comes back as `int`/`float`."""
    if cell == "":
        return None
    if column in JSON_COLUMNS or (column not in TEXT_COLUMNS
                                  and cell[0] in "{["):
        try:
            return json.loads(cell)
        except ValueError:
            pass
    if column in TEXT_COLUMNS:
        return cell
    if column in BOOL_COLUMNS or cell in ("true", "false", "t", "f"):
        return cell in ("true", "t")
    try:
        return int(cell)
    except ValueError:
        pass
    try:
        return float(cell)
    except ValueError:
        return cell


# --- Source rows -> cells ---------------------------------------------------
def _weather_cells(weather, cells):
    """Lift the scalars of a weather block into the flat columns and keep the
    block itself as its JSON cell, so nothing the artifacts recorded is lost."""
    if not isinstance(weather, dict):
        return
    cells["weather"] = weather
    if cells.get("wind_speed_ms") is None:
        cells["wind_speed_ms"] = weather.get("wind_speed_ms")
    if cells.get("temperature_c") is None:
        cells["temperature_c"] = weather.get("temperature_celsius")
    cells["wind_gust_ms"] = weather.get("wind_gust_ms")
    cells["precipitation_mm"] = weather.get("precipitation_mm")
    cells["relative_humidity_pct"] = weather.get("relative_humidity_pct")
    cells["snow_depth_m"] = weather.get("snow_depth_m")
    cells["snowfall_cm"] = weather.get("snowfall_cm")
    cells["weather_ts"] = weather.get("timestamp")


def _measurement_cells(row):
    """Espoo measurement: the artifact already speaks the column vocabulary, so
    this is a copy - the DB-only extras stay empty (this project never reads a
    database)."""
    return {c: row.get(c) for c in ESPOO_COLUMNS + EXTRA_COLUMNS}


def _burst_cells(row):
    """LUMO amp/phase burst: `orbit` and the intra-dwell block are renamed, the
    weather block is flattened, and `LUMO_COLUMNS` is copied verbatim."""
    cells = {c: row.get(c) for c in COLUMNS if c in row}
    if "orbit" in row:
        cells["orbit_direction"] = row.get("orbit")
    aperture = row.get("sub_aperture_brightness") or {}
    cells["sub_aperture_modulation"] = aperture.get("modulation_depth")
    _weather_cells(row.get("weather"), cells)
    return cells


def _coherence_cells(row):
    """LUMO coherence stack: the four renamed keys are the whole difference."""
    cells = {c: row.get(c) for c in COLUMNS if c in row}
    cells["orbit_direction"] = row.get("orbit")
    cells["acquisition_date"] = row.get("date")
    cells["coherence_gamma2"] = row.get("gamma2")
    cells["coherence_masked_pixels"] = row.get("n_masked")
    return cells


def _acquisition_cells(row):
    """LUMO per-acquisition timeseries: `orbit` renamed, weather flattened, and
    the `segments` list dropped - no figure reads it (see the docstring)."""
    cells = {c: row.get(c) for c in COLUMNS if c in row}
    cells["orbit_direction"] = row.get("orbit")
    _weather_cells(row.get("weather"), cells)
    return cells


def _campaign_cells(row):
    """LUMO campaign window: the five keys `fig_lumo_channels.py` shades with."""
    return {c: row.get(c) for c in
            ("damage_label", "structural_state", "period_index", "start_date",
             "end_date")}


def _record(site, record_class, record_key, cells):
    row = {"site": site, "record_class": record_class, "record_key": record_key}
    row.update({c: _cell(cells.get(c)) for c in COLUMNS
                if c not in IDENTITY_COLUMNS})
    return row


# `record_class` -> its builder, so the checks can drive any class without
# naming its artifact; and -> the site that writes it.
BUILDERS = {"measurement": _measurement_cells, "burst": _burst_cells,
            "coherence": _coherence_cells, "acquisition": _acquisition_cells,
            "campaign": _campaign_cells}
CLASS_SITES = {"measurement": "espoo", "burst": "lumo", "coherence": "lumo",
               "acquisition": "lumo", "campaign": "lumo"}


# --- The record identity ----------------------------------------------------
def record_key(record_class, row):
    """The `record_key` of one source row: the artifact's own identity column
    (`id` for Espoo, `burst_id` for the three LUMO burst stacks) or, for a
    campaign window, its period index - the only unique key it has."""
    if record_class == "measurement":
        return row.get("id")
    if record_class == "campaign":
        return f"period-{row.get('period_index')}"
    return row.get("burst_id")


# --- Artifacts -> records ---------------------------------------------------
def espoo_records(doc):
    """`espoo_mast_observability.json` -> one `measurement` row per acquisition
    (150 for the frozen 2-year backfill), keyed by the artifact's own `id`."""
    return [_record("espoo", "measurement", record_key("measurement", r),
                    _measurement_cells(r))
            for r in doc.get("acquisitions", [])]


def lumo_records(amp, coh, ts):
    """The three LUMO artifacts -> 356 `burst` + 178 `coherence` + 482
    `acquisition` + 13 `campaign` rows. Order is artifact order, so the file is
    stable across runs."""
    rows = [_record("lumo", "burst", record_key("burst", r), _burst_cells(r))
            for r in amp.get("bursts", [])]
    rows += [_record("lumo", "coherence", record_key("coherence", r),
                     _coherence_cells(r))
             for r in coh.get("bursts", [])]
    rows += [_record("lumo", "acquisition", record_key("acquisition", r),
                     _acquisition_cells(r))
             for r in ts.get("acquisitions", [])]
    rows += [_record("lumo", "campaign", record_key("campaign", r),
                     _campaign_cells(r))
             for r in ts.get("campaign_timeline", [])]
    return rows


def write_csv(path, rows):
    """One header line of `COLUMNS` plus one line per record. LF line endings and
    no `newline` translation (`newline=""` keeps csv from doubling the `\\n`),
    no timestamp, no comment line: two runs over unchanged artifacts are
    byte-identical and `diff`-able. Returns the number of rows written."""
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(COLUMNS),
                                lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


# --- CSV -> source-shaped rows (what the figures read) ----------------------
def _native(record_class, cells):
    """Inverse of the builders: the flat cells back in the **source shape**, so
    the figures run unchanged. Every column is present (missing ones as `None`,
    which is what the figures' `.get()` calls already see), `orbit_direction`
    goes back to `orbit` for the LUMO classes, and the four coherence-only
    renames are undone."""
    row = {c: cells.get(c) for c in COLUMNS if c not in IDENTITY_COLUMNS}
    if record_class == "measurement":
        return row
    row["orbit"] = cells.get("orbit_direction")
    if record_class == "coherence":
        row["date"] = cells.get("acquisition_date")
        row["gamma2"] = cells.get("coherence_gamma2")
        row["n_masked"] = cells.get("coherence_masked_pixels")
    if record_class == "burst":
        if cells.get("sub_aperture_brightness") is None:
            depth = cells.get("sub_aperture_modulation")
            row["sub_aperture_brightness"] = (
                None if depth is None else {"modulation_depth": depth})
    return row


# --- Build ------------------------------------------------------------------
def load_json(path):
    if not os.path.exists(path):
        print(f"error: missing input artifact: {path}", file=sys.stderr)
        raise SystemExit(2)
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def build_rows(site):
    """The records of one site, in file order - read from `data/*.json` only."""
    if site == "espoo":
        return espoo_records(load_json(ESPOO_JSON))
    return lumo_records(load_json(LUMO_AMP), load_json(LUMO_COH),
                        load_json(LUMO_TS))


# --- Offline checks: no DB, no network, no figures --------------------------
def _diff(record_class, source, rebuilt):
    """How many source cells the file failed to carry, plus one example. Every
    source key is now expected to survive, including `segments` and the full
    `sub_aperture_brightness` block."""
    bad = 0
    detail = ""
    for key, want in source.items():
        got = rebuilt.get(key)
        if got != want or type(got) is not type(want):
            bad += 1
            if not detail:
                detail = f"{key}: file {got!r} != artifact {want!r}"
    return bad, detail


def _report(fails, quiet, label, cond, detail=""):
    """Record and print one check (`ok  `/`FAIL`), the way the sibling scripts
    do it."""
    if not cond:
        fails.append(f"{label}{': ' + detail if detail else ''}")
    if not quiet:
        print(f"  {'ok  ' if cond else 'FAIL'} {label}"
              f"{' -> ' + detail if detail else ''}")


def _same(fails, quiet, label, got, want):
    _report(fails, quiet, label, got == want,
            "" if got == want else f"got {got!r}, want {want!r}")


def _selftest(quiet=False):
    """Check the contract, the codec and the artifacts. Returns the list of
    failures. The artifact half is skipped (with a note) when `data/` is
    incomplete, so the structural half always runs."""
    fails = []

    def ok(label, cond, detail=""):
        _report(fails, quiet, label, cond, detail)

    def eq(label, got, want):
        _same(fails, quiet, label, got, want)

    # 1. The column contract: 3 + 58 + 17 = 78, one name each, fixed blocks.
    print("contract")
    eq("column count", len(COLUMNS), 78)
    eq("column names are unique", len(set(COLUMNS)), len(COLUMNS))
    eq("identity block first", list(COLUMNS[:3]), list(IDENTITY_COLUMNS))
    eq("Espoo block", len(ESPOO_COLUMNS), 27)
    eq("LUMO block", len(LUMO_COLUMNS), 31)
    eq("core block is 58 columns", len(CORE_COLUMNS), 58)
    eq("core block is the middle block", list(COLUMNS[3:61]), list(CORE_COLUMNS))
    eq("the two blocks are disjoint",
       sorted(set(ESPOO_COLUMNS) & set(LUMO_COLUMNS)), [])
    ok("the five names both artifacts use sit in the Espoo block",
       all(c in ESPOO_COLUMNS for c in
           ("acquisition_ts", "brightness_ratio", "phase_coherence",
            "phase_rms_rad", "wind_speed_ms")))
    eq("extra block is the tail", list(COLUMNS[-17:]), list(EXTRA_COLUMNS))
    eq("text columns are real columns", sorted(TEXT_COLUMNS - set(COLUMNS)), [])
    eq("bool columns are real columns", sorted(BOOL_COLUMNS - set(COLUMNS)), [])
    eq("json columns are real columns", sorted(JSON_COLUMNS - set(COLUMNS)), [])
    eq("record classes", sorted(RECORD_CLASSES),
       ["acquisition", "burst", "campaign", "coherence", "measurement"])
    ok("no output collides with the audit sweep",
       all(name != "audit.csv" for _, name in SITE_FILES))

    # 2. The cell codec: exact round trip, one missing marker, no type drift.
    print("codec")
    for column, value in (("peak_intensity", 0.43104165509818504),
                          ("dwell_s", 1.0), ("dwell_s", 0.8),
                          ("mast_peak_row", 7), ("wind_direction_from_deg", 120),
                          ("ndvi", None), ("phase_coherence", -0.0),
                          ("is_damaged", True), ("is_damaged", False),
                          ("ndvi_confound_flagged", False),
                          ("id", "0ab1c2d3-0000-1111-2222-333344445555"),
                          ("campaign_label", "37"),
                          ("ndvi_scene_date", "2024-09-01"),
                          ("acquisition_ts", "2024-09-05T05:31:12Z"),
                          ("weather", {"relative_humidity_pct": 88,
                                       "timestamp": "2024-09-05T05:00:00Z"}),
                          ("filtered", {"strict": [0.5, None, 2]})):
        eq(f"round trip {column}={value!r}", _value(_cell(value), column), value)
    eq("a float keeps all its digits", _cell(0.43104165509818504),
       "0.43104165509818504")
    eq("missing is an empty cell", _cell(None), "")
    eq("an empty cell is missing", _value("", "ndvi"), None)
    eq("an all-digit text value stays text",
       _value("37", "campaign_label"), "37")
    eq("a date stays text", _value("2024-09-01", "ndvi_scene_date"),
       "2024-09-01")
    with tempfile.TemporaryDirectory() as tmp:
        bad = os.path.join(tmp, "bad.csv")
        with open(bad, "w") as fh:
            fh.write("a,b\n1,2\n")
        try:
            read_channels_csv(bad)
            ok("a foreign header is rejected", False, "no ValueError raised")
        except ValueError:
            ok("a foreign header is rejected", True)

    # 3. Every class through a real file: builder -> writer -> reader, cell by
    #    cell, on fixtures small enough to be hand-checked - this is the half
    #    that still runs when `data/` is absent.
    print("synthetic records")
    synth = {
        "measurement": {"id": "0ab1c2d3", "acquisition_ts": "2024-09-05T05:31:12Z",
                        "pass_label": "morning", "orbit_direction": "ASCENDING",
                        "coherence_gamma2": 0.42, "coherence_masked_pixels": 22,
                        "phase_coherence": 0.31, "phase_snr_db": 1.25,
                        "phase_rms_rad": 3.1, "ndvi": None,
                        "traffic_load_label": "37"},
        "burst": {"burst_id": "S1A-IW2-slc-vv-20210403T055142-037189-001",
                  "acquisition_ts": "2020-08-01T05:31:12Z",
                  "acquisition_date": "2020-08-01", "orbit": "DESCENDING",
                  "polarisation": "vv", "month": "2020-08",
                  "strip_brightness_ratio": 2.105, "phase_coherence": 0.02,
                  "sub_aperture_brightness": {"modulation_depth": 0.0612,
                                              "blocks": 8},
                  "weather": {"wind_gust_ms": 4.5, "precipitation_mm": 0.0,
                              "temperature_celsius": 18.25, "wind_speed_ms": 2.5,
                              "relative_humidity_pct": 88, "snow_depth_m": 0.0,
                              "snowfall_cm": 0.0,
                              "timestamp": "2020-08-01T05:00:00Z"},
                  "filtered": {"strict": True}},
        "coherence": {"burst_id": "S1A-IW2-slc-vv-20210403T055142-037189-001",
                      "date": "2020-08-01", "month": "2020-08",
                      "orbit": "DESCENDING", "polarisation": "vv",
                      "damage_label": "healthy", "gamma2": 0.0617,
                      "n_masked": 22, "peak_intensity": 0.43104165509818504,
                      "wind_speed_ms": 2.5},
        "acquisition": {"burst_id": "S1A-IW2-slc-vv-20210403T055142-037189-001",
                        "acquisition_ts": "2020-08-01T05:31:12Z",
                        "acquisition_date": "2020-08-01", "month": "2020-08",
                        "orbit": "DESCENDING", "polarisation": "vv",
                        "brightness_ratio": 1.5, "incidence_angle_deg": 32.75,
                        "dwell_s": 0.8, "scatterers": 4, "is_damaged": False,
                        "campaign_period_index": 0, "damage_label": "healthy",
                        "structural_state": "intact", "segments": [{"x": 1}],
                        "campaign_range": "2020-08 .. 2020-09",
                        "weather": {"wind_gust_ms": 4.5,
                                    "timestamp": "2020-08-01T05:00:00Z"}},
        "campaign": {"damage_label": "DAM 6", "structural_state": "damaged",
                     "period_index": 2, "start_date": "2021-01-15",
                     "end_date": "2021-02-20"},
    }
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "synthetic.csv")
        rows = [_record(CLASS_SITES[cls], cls, f"key-{cls}", BUILDERS[cls](row))
                for cls, row in synth.items()]
        write_csv(path, rows)
        blocks = read_channels_csv(path)
        for cls, source in synth.items():
            eq(f"a synthetic {cls} row keeps its key",
               blocks[cls][0]["record_key"], f"key-{cls}")
            bad, detail = _diff(cls, source, blocks[cls][0])
            ok(f"a synthetic {cls} row survives the file", bad == 0, detail)
        ok("the builders never invent a column",
           all(set(BUILDERS[cls](row)) <= set(COLUMNS)
               for cls, row in synth.items()))

    fails += _artifact_checks(quiet)
    return fails


# --- Reading the file back (the figures' entry point) -----------------------
def read_channels_csv(path, site=None):
    """The file back as `{record_class: [row, ...]}` with the rows in source
    shape (see `_native`), which is what makes the figures CSV-agnostic: they
    keep running the code path they run on the JSON artifacts. `site` filters
    (a file is normally single-site, so it is a guard rather than a feature);
    the header is checked against `COLUMNS`, so a stale or hand-edited file is
    rejected instead of silently mis-read."""
    out = {cls: [] for cls in RECORD_CLASSES}
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames != list(COLUMNS):
            raise ValueError(f"{path}: unexpected columns {reader.fieldnames!r}")
        for raw in reader:
            if site is not None and raw["site"] != site:
                continue
            cells = {c: _value(raw[c], c) for c in COLUMNS
                     if c not in IDENTITY_COLUMNS}
            row = _native(raw["record_class"], cells)
            row["record_key"] = raw["record_key"]
            out.setdefault(raw["record_class"], []).append(row)
    return out


def _artifact_checks(quiet=False):
    """The half that needs `data/`: build both files out of the artifacts, read
    them back and compare every recorded value cell by cell, then check the byte
    contract (LF, no BOM, header, one line per row, a rebuild is identical)."""
    fails = []
    if not all(os.path.exists(p) for p in
               (ESPOO_JSON, LUMO_AMP, LUMO_COH, LUMO_TS)):
        if not quiet:
            print("artifacts\n  -- data/ is incomplete, the per-cell checks "
                  "are skipped")
        return fails
    print("artifacts")
    espoo_doc = load_json(ESPOO_JSON)
    amp, coh, ts = load_json(LUMO_AMP), load_json(LUMO_COH), load_json(LUMO_TS)
    sources = {
        "espoo": [("measurement", espoo_doc.get("acquisitions", []))],
        "lumo": [("burst", amp.get("bursts", [])),
                 ("coherence", coh.get("bursts", [])),
                 ("acquisition", ts.get("acquisitions", [])),
                 ("campaign", ts.get("campaign_timeline", []))],
    }
    covered = set()
    for site, name in SITE_FILES:
        rows = build_rows(site)
        for cls, src in sources[site]:
            for src_row in src:
                covered |= set(BUILDERS[cls](src_row)) & set(CORE_COLUMNS)
        n_src = sum(len(src) for _, src in sources[site])
        _same(fails, quiet, f"{name}: one row per artifact row", len(rows), n_src)
        _same(fails, quiet, f"{name}: every row carries all 78 columns",
              sorted({len(r) for r in rows}), [len(COLUMNS)])
        _same(fails, quiet, f"{name}: (record_class, record_key) is unique",
              len({(r["record_class"], r["record_key"]) for r in rows}), len(rows))
        db_only = ("phase_snr_db", "amplitude_measured_m", "amplitude_expected_m")
        if site == "espoo":
            _same(fails, quiet, f"{name}: the DB-only columns start empty",
                  sorted({r[c] for r in rows for c in db_only}), [""])
        for cls, src in sources[site]:
            n = sum(1 for r in rows if r["record_class"] == cls)
            _report(fails, quiet, f"{name}: {len(src)} {cls} rows", n == len(src),
                    f"{n} rows")
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, name)
            write_csv(path, rows)
            with open(path, "rb") as fh:
                data = fh.read()
            _report(fails, quiet, f"{name}: LF only, no BOM, final newline",
                    b"\r" not in data and not data.startswith(b"\xef\xbb\xbf")
                    and data.endswith(b"\n"))
            _same(fails, quiet, f"{name}: header",
                  data.split(b"\n")[0].decode("utf-8"), ",".join(COLUMNS))
            _same(fails, quiet, f"{name}: one line per record",
                  data.count(b"\n"), len(rows) + 1)
            again = os.path.join(tmp, "again-" + name)
            write_csv(again, build_rows(site))
            with open(again, "rb") as fh:
                _report(fails, quiet, f"{name}: a rebuild is byte-identical",
                        fh.read() == data)
            blocks = read_channels_csv(path, site=site)
            for cls, src in sources[site]:
                got = blocks[cls]
                _same(fails, quiet, f"{name}: {cls} rows come back",
                      len(got), len(src))
                _same(fails, quiet, f"{name}: {cls} keys are in artifact order",
                      [g["record_key"] for g in got],
                      [record_key(cls, s) for s in src])
                bad, detail = 0, ""
                for src_row, got_row in zip(src, got):
                    n, d = _diff(cls, src_row, got_row)
                    bad += n
                    detail = detail or d
                _report(fails, quiet,
                        f"{name}: every {cls} artifact value survives the file",
                        bad == 0, detail)
    _same(fails, quiet, "every core column is fed by an artifact row",
          sorted(covered), sorted(CORE_COLUMNS))
    return fails


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--site", choices=["espoo", "lumo", "all"], default="all",
                    help="which CSV to write (default: %(default)s)")
    ap.add_argument("--out-dir", default=DATA_DIR, metavar="DIR",
                    help="directory for the CSVs (default: %(default)s)")
    ap.add_argument("--selftest", action="store_true",
                    help="check the column contract, the cell codec and the "
                         "artifacts - needs no database")
    args = ap.parse_args(argv)

    if args.selftest:
        print("figures_data_csv selftest - the file contract without a database")
        fails = _selftest()
        if fails:
            for line in fails:
                print(f"  !! {line}", file=sys.stderr)
            print(f"  {len(fails)} check(s) FAILED", file=sys.stderr)
            return 4
        print("  all checks passed")
        return 0

    if not os.path.isdir(args.out_dir):
        print(f"error: no such directory: {args.out_dir}", file=sys.stderr)
        return 2
    sites = [s for s, _ in SITE_FILES] if args.site == "all" else [args.site]
    for site in sites:
        name = dict(SITE_FILES)[site]
        rows = build_rows(site)
        write_csv(os.path.join(args.out_dir, name), rows)
        counts = ", ".join(f"{sum(1 for r in rows if r['record_class'] == c)} {c}"
                           for c in RECORD_CLASSES
                           if any(r["record_class"] == c for r in rows))
        print(f"Wrote: {name} - {len(rows)} rows ({counts}), "
              f"{len(COLUMNS)} columns")
    print("Read them back with `figures/fig_espoo_channels.py` and "
          "`figures/fig_lumo_channels.py` - each defaults to its own file.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

