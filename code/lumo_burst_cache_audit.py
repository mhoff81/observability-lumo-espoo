#!/usr/bin/env python3
"""Audit a whole LUMO burst cache against the CDSE SLCs it was cut from.

The sibling `lumo_slc_ranged_read.py` reproduces *one* cache entry by ranged
reads over the Sentinel-1 SLC on CDSE. This script runs the same rule set over
every `<YYYY-MM>/<entry>/` of a cache directory written by the Rust pipeline, and
reports what a single entry cannot show - whether the cached windows hold
together as a body of evidence:

  * every entry is scored against both anchor rules, `legacy` (before commit
    fad0e46, "location to insar pixel mapping fixed", 2026-09-10T19:40:13Z) and
    `current` (after it), and matched to the rule that reproduces its
    `window.bin` bit-exactly;
  * every entry is classified into a vintage by the mtime of its own window.bin,
    because the only evidence of which commit wrote an entry is when it was
    written, and the cache spans the fix;
  * the burst that hosts an entry is checked for *containment* of the mast: the
    footprint of its geolocation grid, whether the coordinate is inside that
    box, and the distance to the nearest grid node. Both anchor rules clamp, so
    a burst that does not cover the mast still yields an anchor hundreds of
    pixels away; only this check separates "this window is the tower" from
    "this window is a clamped corner of the wrong burst".

WHAT THE EXIT CODE MEANS
  0 every audited entry was reproduced exactly by the rule its vintage
    predicts, and every entry could be read and resolved;
  2 at least one entry could not be read or resolved (listed in the summary and
    in the per-entry table; by itself not a verdict on the cache);
  3 at least one entry was reproduced by no rule at all, or by the rule the
    *other* vintage predicts - the sweep contradicted the cache's own history;
  4 the offline selftest failed (argparse exits 2 on bad usage);
  5 at least one entry was reproduced exactly, but its burst is farther than
    `--max-node-km` from the mast: the window is bit-for-bit the right bytes of
    the wrong place, which is the defect only this sweep can name. 3 beats 5,
    5 beats 2.

NETWORK: like `lumo_slc_ranged_read.py`, this script needs network access and
CDSE credentials. The four secrets come from the environment first and then from
`--env-file` / `$LUMO_SLC_ENV_FILE`, and are never printed. One token is fetched
per run and reused, with a single refresh when a burst lookup fails after it has
aged past five minutes. No measurement TIFF is ever downloaded: each entry costs
two 41x41 window decodes (~41 ranged GETs each), `--strip-every N` adds a 400x11
strip (~400 GETs) to every N-th entry, and each burst is resolved once for all
the entries that share it. `--selftest` is fully offline.

INPUT: a cache directory written by the Rust pipeline - one sub-directory per
entry, each holding `meta.json`, `window.bin` and `strip.bin`:

  <cache>/<YYYY-MM>/<date>_<id>/meta.json

USAGE
  python3 code/lumo_burst_cache_audit.py --cache-dir DIR --limit 2
  python3 code/lumo_burst_cache_audit.py --cache-dir DIR --month 2021-04
  python3 code/lumo_burst_cache_audit.py --cache-dir DIR --limit 2 --json audit.json
  python3 code/lumo_burst_cache_audit.py --cache-dir DIR --stride 10 --csv audit.csv
  python3 code/lumo_burst_cache_audit.py --cache-dir DIR --entry 2021-03/2021-03-01_22752f50 --safe
  python3 code/lumo_burst_cache_audit.py --selftest

OUTPUT: `--json` carries every per-entry number the table prints, plus the ones
it does not; `--csv` writes the same rows as one row each of 54 columns (the
names are in `csv_columns` in `--json`), with LF line endings and no timestamp,
so two sweeps diff cleanly and a spreadsheet can sort or filter them.
"""
import argparse
import csv
import datetime
import json
import os
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:         # also lets `python3 -m code.<name>` find it
    sys.path.insert(0, HERE)
import lumo_slc_ranged_read as slc  # noqa: E402  (needs the path above)

SCHEMA = "lumo_burst_cache_audit/v1"
VINTAGE_PRE = "pre-fad0e46"
VINTAGE_POST = "post-fad0e46"
# --max-node-km default: one grid step in range is ~10 km, so a burst that does
# cover the mast sits well inside this and a burst that does not sits far out.
OFF_TARGET_KM = 10.0
# A sweep of a full cache runs for hours, so it can outlive the CDSE access
# token: a burst lookup that fails while the token is at least this old is
# retried once with a fresh one.
TOKEN_REFRESH_S = 300.0

# The `--csv` columns, in file order: one row per audited entry, so a sweep can
# be sorted, filtered or diffed without a JSON parser. The derived columns
# (epoch, km, fractions, byte count, pixel deltas, margin) are computed by
# `entry_csv_row`; seven verdict booleans keep the conjunctions a single
# `verdict` word has to collapse, and an unknown number is an empty cell.
ENTRY_COLUMNS = (
    "entry", "month", "name", "burst_id", "subswath", "polarisation",
    "acquisition_ts", "incidence_angle_deg",
    "window_mtime", "window_mtime_epoch", "vintage", "expected_rule",
    "verdict", "unusable", "no_rule", "ambiguous", "wrong_rule", "off_target",
    "outside_grid", "matched_expected", "matched_rules", "match_count",
    "flags", "error",
    "product", "grid_points", "grid_lat_min", "grid_lat_max", "grid_lon_min",
    "grid_lon_max", "mast_inside_grid", "nearest_node_line",
    "nearest_node_pixel", "nearest_node_m", "nearest_node_km",
    "off_target_margin_km",
    "window_samples", "window_bytes",
    "legacy_exact", "legacy_fraction", "legacy_center_line",
    "legacy_center_pixel", "legacy_ground_error_m",
    "current_exact", "current_fraction", "current_center_line",
    "current_center_pixel", "current_ground_error_m",
    "center_delta_line", "center_delta_pixel",
    "strip_rule", "strip_samples", "strip_exact", "strip_fraction")

# `classify` returns exactly these, in this order of precedence.
VERDICTS = ("unusable", "no-rule", "ambiguous", "wrong-rule", "off-target",
            "consistent")


def _mtime(entry_dir):
    """mtime of an entry's `window.bin`, or None when the entry is incomplete."""
    try:
        return os.path.getmtime(os.path.join(entry_dir, "window.bin"))
    except OSError:
        return None


def entries_of(cache_dir, months=None, only=None, limit=0, stride=1):
    """The cache entries to audit, sorted, as `[(month, name, dir, mtime)]`.

    `only` holds explicit `MONTH/NAME` entries and is taken literally - a name
    that is not there comes back with `mtime=None` so that the audit reports it
    as unusable instead of skipping it silently. Otherwise every month (all of
    them, or the `months` given) is walked in name order, keeping every
    `stride`-th entry and at most `limit` of them per month: the sample that
    keeps a sweep over a thousand entries inside a coffee break."""
    if only:
        out = []
        for entry in only:
            month, _, name = entry.partition("/")
            path = os.path.join(cache_dir, month, name)
            out.append((month, name, path, _mtime(path)))
        return out
    out = []
    for month in sorted(os.listdir(cache_dir)):
        month_dir = os.path.join(cache_dir, month)
        if not os.path.isdir(month_dir) or (months and month not in months):
            continue
        names = sorted(n for n in os.listdir(month_dir)
                       if os.path.isdir(os.path.join(month_dir, n)))
        if stride > 1:
            names = names[::stride]
        if limit:
            names = names[:limit]
        for name in names:
            path = os.path.join(month_dir, name)
            out.append((month, name, path, _mtime(path)))
    return out


def vintage_of(mtime):
    """`VINTAGE_PRE`/`VINTAGE_POST`/None for a `window.bin` mtime: the audit
    reads the cache's own history, which is the only record of which commit cut
    a given window."""
    if mtime is None:
        return None
    return VINTAGE_POST if mtime >= slc.FIX_TS else VINTAGE_PRE


def expected_rule(vintage):
    """The anchor rule a cache entry of this vintage should reproduce under -
    the cache and the Rust code that wrote it are the same commit."""
    if vintage == VINTAGE_PRE:
        return "legacy"
    if vintage == VINTAGE_POST:
        return "current"
    return None


def subswath_pol(name):
    """`s1a-iw1-slc-vv-...-001.xml` -> `('iw1', 'vv')`, for the `--safe` table
    that has to name the six annotations of one SAFE."""
    parts = name.lower().split("-")
    sw = next((p for p in parts if p.startswith("iw")), "?")
    pol = next((p for p in parts if p in ("vv", "vh", "hh", "hv")), "?")
    return sw, pol


def _iso(mtime):
    """A `window.bin` mtime as an ISO-8601 UTC string: the cache's own file
    time, which is the evidence for the vintage - not a report timestamp."""
    if not mtime:
        return None
    return datetime.datetime.fromtimestamp(
        mtime, datetime.timezone.utc).replace(microsecond=0).isoformat()


class Store:
    """One CDSE session plus the caches that keep a sweep thrifty: resolved
    bursts, their annotation grids and those grids' footprints are keyed by
    burst id, so the entries that share a burst cost one lookup between them.

    `token` is normally omitted and fetched here; the `S3` object ignores it,
    the catalogue (OData) calls do not."""

    def __init__(self, creds, s3, lat, lon, max_node_m=OFF_TARGET_KM * 1000.0,
                 token=None):
        self.creds, self.s3 = creds, s3
        self.lat, self.lon = lat, lon
        self.max_node_m = max_node_m
        self.products = {}
        self.grids = {}
        if token is None:
            self._fetch_token()
        else:
            self.token, self.token_ts = token, time.time()

    def _fetch_token(self):
        self.token = slc.fetch_token(self.creds["COPERNICUS_CLIENT_ID"],
                                     self.creds["COPERNICUS_CLIENT_SECRET"])
        self.token_ts = time.time()

    def stale_token(self):
        """Is the cached token old enough that a failure may be its fault? A
        young token is not re-fetched, so a genuine 404 stays a 404."""
        return time.time() - self.token_ts >= TOKEN_REFRESH_S

    def burst(self, meta):
        """`(product, measurement href, grid, bbox, nearest)` of the burst a
        cached `meta.json` names; `nearest` is `(metres, grid node)`. A lookup
        that fails after the token has aged is retried once against a fresh
        token, which is what keeps a multi-hour sweep from turning its last
        hours into "unusable" entries."""
        bid = meta["burst_id"]
        try:
            found = slc.burst_product(meta, self.token, self.s3, self.products)
        except Exception:
            if not self.stale_token():
                raise
            self._fetch_token()
            self.products.pop(bid, None)
            found = slc.burst_product(meta, self.token, self.s3, self.products)
        product, root, ann_rel, meas_rel = found
        if bid not in self.grids:
            _, grid = slc.annotation_grid(self.s3, root, ann_rel)
            self.grids[bid] = (grid,) + slc.grid_stats(grid, self.lat, self.lon)
        grid, bbox, nearest = self.grids[bid]
        return product, f"{root}/{meas_rel}", grid, bbox, nearest

    def footprints(self, meta):
        """Every annotation of the SAFE that hosts `meta`, as a list of
        `(sub-swath, polarisation, bbox, metres to the nearest node, inside)` -
        the table that tells a burst of the wrong *sub-swath* of the right SAFE
        from a burst of the wrong SAFE altogether. Calibration and noise files
        are skipped: they carry the same sub-swath and polarisation words."""
        _, root, _, _ = slc.burst_product(meta, self.token, self.s3,
                                         self.products)
        xml = self.s3.get(f"{root}/manifest.safe").decode("utf-8", "replace")
        ann_paths, _ = slc.parse_manifest(xml)
        out = []
        for path in ann_paths:
            name = path.rsplit("/", 1)[-1]
            low = name.lower()
            if "calibration" in low or "noise" in low:
                continue
            sw, pol = subswath_pol(name)
            _, grid = slc.annotation_grid(self.s3, root, path)
            bbox, nearest = slc.grid_stats(grid, self.lat, self.lon)
            out.append((sw, pol, bbox, nearest[0],
                        slc.tower_inside(bbox, self.lat, self.lon)))
        return out


def audit_entry(store, month, name, entry_dir, mtime=None, strip_lines=0):
    """One entry: vintage, containment, both anchors, both scores.

    Returns the report row; failures are recorded in `row["error"]` and never
    raised, because a sweep over a thousand entries must not die on one."""
    vintage = vintage_of(mtime)
    row = {"entry": f"{month}/{name}", "burst_id": None, "subswath": None,
           "polarisation": None, "acquisition_ts": None,
           "incidence_angle_deg": None, "window_mtime": _iso(mtime),
           "window_mtime_epoch": int(mtime) if mtime is not None else None,
           "vintage": vintage, "expected_rule": expected_rule(vintage),
           "error": None, "product": None, "grid_points": None,
           "grid_bbox": None, "mast_inside_grid": None, "nearest_node": None,
           "nearest_node_m": None, "off_target": None,
           "center": {}, "anchor_ground_error_m": {}, "window_samples": None,
           "matched": [], "rules": {}, "strip": None}
    try:
        with open(os.path.join(entry_dir, "meta.json")) as fh:
            meta = json.load(fh)
    except (OSError, ValueError) as exc:
        row["error"] = f"meta.json: {type(exc).__name__}: {exc}"
        return row
    for key in ("burst_id", "subswath", "polarisation", "acquisition_ts",
                "incidence_angle_deg"):
        row[key] = meta.get(key)
    try:
        product, tiff, grid, bbox, nearest = store.burst(meta)
        hdr = slc.parse_tiff_header(store.s3, tiff)
    except Exception as exc:
        row["error"] = f"resolve: {type(exc).__name__}: {exc}"
        return row

    cell = slc.select_cell(grid, store.lat, store.lon)
    row["product"] = product
    row["grid_points"] = len(grid)
    row["grid_bbox"] = [round(v, 6) for v in bbox]
    row["mast_inside_grid"] = slc.tower_inside(bbox, store.lat, store.lon)
    row["nearest_node"] = [nearest[1]["line"], nearest[1]["pixel"]]
    row["nearest_node_m"] = round(nearest[0], 1)
    row["off_target"] = nearest[0] > store.max_node_m
    for rule, anchor in slc.anchors(grid, store.lat, store.lon).items():
        if anchor is None:
            row["center"][rule] = row["anchor_ground_error_m"][rule] = None
            continue
        row["center"][rule] = [slc.rust_round(anchor[0]),
                               slc.rust_round(anchor[1])]
        row["anchor_ground_error_m"][rule] = None if cell is None else round(
            slc.ground_error_m(cell, anchor[0], anchor[1], store.lat,
                               store.lon), 1)
    try:
        res = slc.verify_entry(store.s3, meta, entry_dir, grid, tiff, hdr,
                               store.lat, store.lon, strip_lines=strip_lines)
    except Exception as exc:
        row["error"] = f"decode: {type(exc).__name__}: {exc}"
        return row
    row["window_samples"] = res["window_samples"]
    row["matched"] = list(res["matched"])
    row["rules"] = {rule: {"exact": info["exact"], "center": info["center"]}
                    for rule, info in res["rules"].items()}
    if res["strip"] is not None:
        row["strip"] = {"rule": res["strip"]["rule"],
                        "samples": res["strip"]["samples"],
                        "exact": res["strip"]["exact"]}
    return row


def classify(row):
    """One word per audited entry, in the precedence `audit_exit` uses: an entry
    that could not be read is `unusable`, one no rule reproduced is `no-rule`,
    one several rules reproduced is `ambiguous`, one only the rule the *other*
    vintage predicts reproduced is `wrong-rule`, one that is off-target even on
    its own rule is `off-target`, and everything else is `consistent`. The word
    never hides a boolean, it only orders them: an off-target entry that no rule
    reproduced is `no-rule` with `off_target` set in the same CSV row."""
    if row.get("error"):
        return "unusable"
    if not row["matched"]:
        return "no-rule"
    if len(row["matched"]) > 1:
        return "ambiguous"
    if row["matched"][0] != row["expected_rule"]:
        return "wrong-rule"
    if row["off_target"]:
        return "off-target"
    return "consistent"


def summarise(rows):
    """The counts the headline and the exit code are built from."""
    out = {"entries": len(rows), "unusable": 0, "no_rule": 0, "ambiguous": 0,
           "unexpected": 0, "off_target": 0, "mast_outside_grid": 0,
           "matched": {}, "strips": 0, "strips_exact": 0, "vintages": {},
           "verdicts": {verdict: 0 for verdict in VERDICTS},
           "unusable_entries": [], "mismatched_entries": [],
           "unexpected_entries": [], "off_target_entries": []}
    for row in rows:
        out["verdicts"][classify(row)] += 1
        if row["error"]:
            out["unusable"] += 1
            out["unusable_entries"].append(row["entry"])
            continue
        vintage = row["vintage"] or "unknown"
        slot = out["vintages"].setdefault(
            vintage, {"entries": 0, "matched": {}, "off_target": 0,
                      "unexpected": 0})
        slot["entries"] += 1
        for rule in row["matched"]:
            slot["matched"][rule] = slot["matched"].get(rule, 0) + 1
            out["matched"][rule] = out["matched"].get(rule, 0) + 1
        if not row["matched"]:
            out["no_rule"] += 1
            out["mismatched_entries"].append(row["entry"])
        elif len(row["matched"]) > 1:
            out["ambiguous"] += 1
        elif row["matched"][0] != row["expected_rule"]:
            out["unexpected"] += 1
            slot["unexpected"] += 1
            out["unexpected_entries"].append(row["entry"])
        if row["off_target"]:
            out["off_target"] += 1
            slot["off_target"] += 1
            out["off_target_entries"].append(row["entry"])
        if row["mast_inside_grid"] is False:
            out["mast_outside_grid"] += 1
        if row["strip"] is not None:
            out["strips"] += 1
            if row["strip"]["exact"] == row["strip"]["samples"]:
                out["strips_exact"] += 1
    return out


def audit_exit(summary):
    """Exit code of a finished audit: 3 beats 5 beats 2 beats 0 (docstring)."""
    if summary["no_rule"] or summary["unexpected"]:
        return 3
    if summary["off_target"]:
        return 5
    if summary["unusable"]:
        return 2
    return 0


def _abbrev(names, limit=6):
    """The first few names and a count of the rest; the full list stays in the
    `--json` report, where a reader can filter it."""
    shown = ", ".join(names[:limit])
    if len(names) > limit:
        shown += f", +{len(names) - limit} more"
    return shown


def _flags(row):
    """The flags of a row as one string, for `render_row` and `--csv` alike:
    nothing matched the entry, its burst is off-target, its grid excludes the
    mast, how much of the window survived the strip - or that it is unusable."""
    if row.get("error"):
        return "unusable"
    flags = []
    if not row["matched"]:
        flags.append("NO-MATCH")
    if row["off_target"]:
        flags.append("OFF-TARGET")
    if row["mast_inside_grid"] is False:
        flags.append("OUTSIDE-GRID")
    if row["strip"] is not None:
        flags.append(f"strip {row['strip']['exact']}/{row['strip']['samples']}")
    return " ".join(flags)


def render_row(row):
    """One per-entry line: entry, vintage, the score of each rule (`l` = legacy,
    `c` = current), the rule that reproduced the window, and the distance to the
    nearest grid node. Off-target and unmatched entries say so here, which makes
    `grep OFF-TARGET` a usable filter over a sweep."""
    if row["error"]:
        return f"  {row['entry']:30s} unusable: {row['error']}"
    rules = " ".join(f"{rule[0]}:{row['rules'][rule]['exact']}"
                     for rule in sorted(row["rules"]))
    return (f"  {row['entry']:30s} {(row['subswath'] or '?'):4s} "
            f"{(row['polarisation'] or '?'):3s} "
            f"{(row['vintage'] or 'unknown'):12s} [{rules}] "
            f"matched={','.join(row['matched']) or '-':15s} "
            f"nearest {row['nearest_node_m'] / 1000:7.1f} km  "
            f"{_flags(row)}")


def render_summary(summary, cache_name):
    """The headline: what was audited, vintage by vintage, plus the counts the
    exit code comes from."""
    lines = [f"audited {summary['entries']} entries of {cache_name}; "
             f"l = legacy rule, c = current rule"]
    for vintage, slot in sorted(summary["vintages"].items()):
        matched = ", ".join(f"{rule} {n}"
                            for rule, n in sorted(slot["matched"].items()))
        lines.append(f"  {vintage:12s} {slot['entries']:4d} entries, expected "
                     f"{expected_rule(vintage) or '?':7s} rule, matched by "
                     f"{matched or '(no rule)':18s} off-target "
                     f"{slot['off_target']:3d}, reproduced by the other rule "
                     f"{slot['unexpected']}")
    lines.append(f"  totals: legacy {summary['matched'].get('legacy', 0)}, "
                 f"current {summary['matched'].get('current', 0)}, both rules "
                 f"{summary['ambiguous']}, no rule {summary['no_rule']}, wrong "
                 f"rule {summary['unexpected']}, unusable {summary['unusable']}, "
                 f"off-target {summary['off_target']} (of which outside the grid "
                 f"box {summary['mast_outside_grid']}), strips "
                 f"{summary['strips_exact']}/{summary['strips']}")
    for label, key in (("off-target entries", "off_target_entries"),
                       ("entries no rule reproduced", "mismatched_entries"),
                       ("entries reproduced by the other rule",
                        "unexpected_entries"),
                       ("unusable entries", "unusable_entries")):
        if summary[key]:
            lines.append(f"  {label}: {_abbrev(summary[key])}")
    return lines


def report_block(summary, rows, cache_dir, generated, lat, lon, max_node_km,
                 csv_path=None):
    """The JSON report: the same numbers as the table, per entry. The cache
    directory is recorded by its base name only - like `csv`, and only when a
    CSV was written - so the file carries no absolute path of the machine that
    produced it. `csv_columns` names the columns of that CSV in file order, so
    the report documents its own sibling format even without `--csv`."""
    return {"schema": SCHEMA, "generated": generated,
            "csv_columns": list(ENTRY_COLUMNS),
            "csv": ({"file": os.path.basename(csv_path), "rows": len(rows)}
                    if csv_path else None),
            "cache_dir": os.path.basename(os.path.normpath(cache_dir)),
            "mast_latitude": lat, "mast_longitude": lon,
            "max_node_km": max_node_km,
            "fix": {"commit": "fad0e46",
                    "message": "location to insar pixel mapping fixed",
                    "epoch": slc.FIX_TS,
                    "pre": VINTAGE_PRE, "post": VINTAGE_POST},
            "summary": summary, "entries": rows}


def entry_csv_row(row, max_node_m=None):
    """One report row as one flat dict keyed by `ENTRY_COLUMNS`.

    The measured values go in as they are, the derived columns (epoch, km,
    fractions, byte count, pixel deltas, off-target margin) are computed here,
    and every unknown number becomes an empty cell: an entry that could not be
    read keeps its `entry`, `flags`, `verdict` and `error` and leaves the rest
    blank, so an export covers every entry the sweep walked."""
    rules = row.get("rules") or {}
    bbox = row.get("grid_bbox") or []
    node = row.get("nearest_node") or []
    matched = row.get("matched") or []
    errors = row.get("anchor_ground_error_m") or {}
    expected = row.get("expected_rule")
    samples = row.get("window_samples")
    strip = row.get("strip")
    node_m = row.get("nearest_node_m")
    month, _, name = row["entry"].partition("/")

    def cell(value):               # a measured value, or an empty cell
        return "" if value is None else value

    def num(value, fmt):           # a formatted float, or an empty cell
        return "" if value is None else format(value, fmt)

    def bit(value):                # a measured flag: 1, 0 or unknown
        return "" if value is None else int(bool(value))

    def outside(inside):           # mast_inside_grid, inverted
        return "" if inside is None else int(not inside)

    def rule_cell(rule, key):
        return cell((rules.get(rule) or {}).get(key))

    def fraction(rule):
        exact = (rules.get(rule) or {}).get("exact")
        if not samples or exact is None:
            return ""
        return f"{exact / samples:.6f}"

    def center_at(rule, i):
        """`(line, pixel)` of a rule's clamped anchor, when it has one."""
        center = (row.get("center") or {}).get(rule) or []
        return cell(center[i]) if len(center) > i else ""

    def delta(i):
        """The pixel drift the fix removed: legacy center minus current."""
        legacy = (row.get("center") or {}).get("legacy") or []
        current = (row.get("center") or {}).get("current") or []
        if len(legacy) < 2 or len(current) < 2:
            return ""
        return legacy[i] - current[i]

    return {"entry": row["entry"], "month": month, "name": name,
            "burst_id": cell(row.get("burst_id")),
            "subswath": cell(row.get("subswath")),
            "polarisation": cell(row.get("polarisation")),
            "acquisition_ts": cell(row.get("acquisition_ts")),
            "incidence_angle_deg": cell(row.get("incidence_angle_deg")),
            "window_mtime": cell(row.get("window_mtime")),
            "window_mtime_epoch": cell(row.get("window_mtime_epoch")),
            "vintage": cell(row.get("vintage")),
            "expected_rule": cell(expected),
            "verdict": classify(row),
            "unusable": int(bool(row.get("error"))),
            "no_rule": int(not row.get("error") and not matched),
            "ambiguous": int(len(matched) > 1),
            "wrong_rule": int(len(matched) == 1 and matched[0] != expected),
            "off_target": bit(row.get("off_target")),
            "outside_grid": outside(row.get("mast_inside_grid")),
            "matched_expected": int(expected is not None
                                    and matched == [expected]),
            "matched_rules": ";".join(matched),
            "match_count": len(matched),
            "flags": _flags(row),
            "error": cell(row.get("error")),
            "product": cell(row.get("product")),
            "grid_points": cell(row.get("grid_points")),
            "grid_lat_min": cell(bbox[0]) if len(bbox) > 0 else "",
            "grid_lat_max": cell(bbox[1]) if len(bbox) > 1 else "",
            "grid_lon_min": cell(bbox[2]) if len(bbox) > 2 else "",
            "grid_lon_max": cell(bbox[3]) if len(bbox) > 3 else "",
            "mast_inside_grid": bit(row.get("mast_inside_grid")),
            "nearest_node_line": cell(node[0]) if len(node) > 0 else "",
            "nearest_node_pixel": cell(node[1]) if len(node) > 1 else "",
            "nearest_node_m": num(node_m, ".1f"),
            "nearest_node_km": num(None if node_m is None else node_m / 1000.0,
                                   ".3f"),
            "off_target_margin_km": num(
                None if node_m is None or max_node_m is None
                else (node_m - max_node_m) / 1000.0, ".3f"),
            "window_samples": cell(samples),
            "window_bytes": "" if not samples else samples * 16,
            "legacy_exact": rule_cell("legacy", "exact"),
            "legacy_fraction": fraction("legacy"),
            "legacy_center_line": center_at("legacy", 0),
            "legacy_center_pixel": center_at("legacy", 1),
            "legacy_ground_error_m": num(errors.get("legacy"), ".1f"),
            "current_exact": rule_cell("current", "exact"),
            "current_fraction": fraction("current"),
            "current_center_line": center_at("current", 0),
            "current_center_pixel": center_at("current", 1),
            "current_ground_error_m": num(errors.get("current"), ".1f"),
            "center_delta_line": delta(0),
            "center_delta_pixel": delta(1),
            "strip_rule": cell(strip["rule"]) if strip else "",
            "strip_samples": cell(strip["samples"]) if strip else "",
            "strip_exact": cell(strip["exact"]) if strip else "",
            "strip_fraction": (f"{strip['exact'] / strip['samples']:.6f}"
                               if strip and strip["samples"] else "")}


def write_entries_csv(path, rows, max_node_m=None):
    """`--csv`: every audited entry as one row of `ENTRY_COLUMNS`, in the order
    the sweep walked them. LF line endings, no timestamp and no comment line, so
    two sweeps of one cache are byte-identical and `diff`-able; a spreadsheet
    can sort or filter the file directly. Returns the number of rows written."""
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(ENTRY_COLUMNS),
                                lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(entry_csv_row(row, max_node_m))
    return len(rows)


# --- Offline checks: no network, no cache, no credentials --------------------
def _row(entry, vintage, matched, **kw):
    """A synthetic report row, so the cache walk, the aggregation, the rendering
    and the CSV export can be checked without a cache or a network. It carries
    every key `audit_entry` puts in a row that scored, so `entry_csv_row` fills
    every column but `error`, `flags` and `strip_*`."""
    row = {"entry": entry, "vintage": vintage, "expected_rule":
           expected_rule(vintage), "matched": list(matched), "error": None,
           "burst_id": "S1A-IW2-slc-vv-20210403T055142-037189-001",
           "subswath": "iw2", "polarisation": "vv",
           "acquisition_ts": "2021-04-03T05:51:42.000000Z",
           "incidence_angle_deg": 33.0, "window_mtime": _iso(slc.FIX_TS),
           "window_mtime_epoch": int(slc.FIX_TS), "window_samples": 1681,
           "product": "S1A_IW_SLC__1SDV_20210403T055142_x",
           "grid_points": 1089, "grid_bbox": [51.9, 52.9, 9.2, 10.2],
           "mast_inside_grid": True, "nearest_node": [240, 380],
           "nearest_node_m": 9000.0, "off_target": False,
           "center": {"legacy": [1, 2], "current": [2, 3]},
           "anchor_ground_error_m": {"legacy": 12.3, "current": 12.3},
           "strip": None,
           "rules": {"legacy": {"exact": 0, "center": [1, 2]},
                     "current": {"exact": 0, "center": [2, 3]}}}
    row.update(kw)
    return row


def _selftest():
    """Every offline check; returns the failures, empty when all passed."""
    fails = []

    def check(label, got, want):
        if got != want:
            fails.append(f"{label}: got {got!r}, want {want!r}")

    def ok(label, cond):
        if not cond:
            fails.append(label)

    # the vintage boundary: the commit's own timestamp is the first post-fix
    # second, and an entry with no window.bin has no vintage at all
    check("mtime before the fix", vintage_of(slc.FIX_TS - 1), VINTAGE_PRE)
    check("mtime at the fix", vintage_of(slc.FIX_TS), VINTAGE_POST)
    check("mtime after the fix", vintage_of(slc.FIX_TS + 86400), VINTAGE_POST)
    check("missing window.bin has no vintage", vintage_of(None), None)
    check("pre-fix expected rule", expected_rule(VINTAGE_PRE), "legacy")
    check("post-fix expected rule", expected_rule(VINTAGE_POST), "current")
    check("unknown vintage expects no rule", expected_rule("?"), None)
    check("annotation name",
          subswath_pol("s1a-iw1-slc-vh-20200724t055214-001.xml"),
          ("iw1", "vh"))
    check("annotation name without a swath", subswath_pol("white.png"),
          ("?", "?"))
    check("an epoch is no evidence of a vintage", _iso(0), None)
    check("a known stamp in ISO UTC",
          _iso(datetime.datetime(2021, 4, 3,
                                 tzinfo=datetime.timezone.utc).timestamp()),
          "2021-04-03T00:00:00+00:00")

    # the cache walk: order, sampling, month filter, explicit entries
    with tempfile.TemporaryDirectory() as tmp:
        for month, names in (("2021-04", ("a", "b", "c", "e")),
                             ("2021-05", ("d",))):
            for name in names:
                entry = os.path.join(tmp, month, name)
                os.makedirs(entry)
                open(os.path.join(entry, "meta.json"), "w").close()
                if name != "e":            # `e` is the incomplete entry
                    open(os.path.join(entry, "window.bin"), "w").close()
        check("entries in order", [n for _, n, _, _ in entries_of(tmp)],
              ["a", "b", "c", "e", "d"])
        check("--limit caps each month",
              [n for _, n, _, _ in entries_of(tmp, limit=2)], ["a", "b", "d"])
        check("--stride keeps every n-th",
              [n for _, n, _, _ in entries_of(tmp, stride=2)], ["a", "c", "d"])
        check("--month keeps one month",
              [n for _, n, _, _ in entries_of(tmp, months=["2021-05"])], ["d"])
        check("--entry is taken literally",
              [f"{m}/{n}" for m, n, _, _
               in entries_of(tmp, only=["2021-04/b"])], ["2021-04/b"])
        check("an entry that is not there has no mtime",
              [mt for _, _, _, mt in entries_of(tmp, only=["2021-04/zz"])],
              [None])
        check("an incomplete entry has no mtime",
              [mt for _, n, _, mt in entries_of(tmp) if n == "e"], [None])
        ok("a complete entry has an mtime",
           [mt for _, n, _, mt in entries_of(tmp) if n == "a"][0] > 0)
        check("a missing entry directory has no mtime",
              [_mtime(os.path.join(tmp, "nope"))], [None])

    row = audit_entry(None, "2021-04", "zz", "/nonexistent/zz")
    ok("a missing entry is reported, not raised",
       bool(row["error"]) and row["error"].startswith("meta.json:"))
    check("a missing entry has no vintage", row["vintage"], None)

    # the token policy, offline: a young token is never re-fetched (so a real
    # 404 stays a 404), an aged one is
    store = Store({"COPERNICUS_CLIENT_ID": "id",
                   "COPERNICUS_CLIENT_SECRET": "secret"}, None, 52.38, 9.72,
                  token="fixture")
    ok("a just-fetched token is not stale", not store.stale_token())
    store.token_ts -= TOKEN_REFRESH_S + 1
    ok("a token older than the refresh window is stale", store.stale_token())

    # aggregation and the exit code
    summary = summarise([
        _row("2021-04/a", VINTAGE_PRE, ["legacy"]),
        _row("2021-04/b", VINTAGE_PRE, ["legacy"], off_target=True,
             mast_inside_grid=False, nearest_node_m=65500.0),
        _row("2021-05/c", VINTAGE_POST, ["current"],
             strip={"rule": "current", "samples": 4400, "exact": 4400}),
        _row("2021-05/d", VINTAGE_POST, []),
        _row("2021-05/e", VINTAGE_POST, ["legacy"]),
        _row("2021-05/f", VINTAGE_POST, ["legacy", "current"]),
        _row("2021-05/g", VINTAGE_POST, [], error="meta.json: OSError: gone"),
    ])
    check("entries counted", summary["entries"], 7)
    check("unusable counted", summary["unusable"], 1)
    check("no-rule counted", summary["no_rule"], 1)
    check("both-rules counted", summary["ambiguous"], 1)
    check("wrong-rule counted", summary["unexpected"], 1)
    check("off-target counted", summary["off_target"], 1)
    check("outside-grid counted", summary["mast_outside_grid"], 1)
    check("strips counted", (summary["strips"], summary["strips_exact"]),
          (1, 1))
    check("legacy matches counted", summary["matched"]["legacy"], 4)
    check("current matches counted", summary["matched"]["current"], 2)
    check("pre-fix slot entries",
          summary["vintages"][VINTAGE_PRE]["entries"], 2)
    check("pre-fix slot off-target",
          summary["vintages"][VINTAGE_PRE]["off_target"], 1)
    check("post-fix slot wrong rule",
          summary["vintages"][VINTAGE_POST]["unexpected"], 1)
    check("exit code of a contradicted sweep", audit_exit(summary), 3)
    check("a contradiction outranks an off-target entry",
          audit_exit({"no_rule": 0, "unexpected": 1, "off_target": 4,
                      "unusable": 1}), 3)
    check("exit code of an off-target sweep",
          audit_exit({"no_rule": 0, "unexpected": 0, "off_target": 3,
                      "unusable": 0}), 5)
    check("exit code with only unusable entries",
          audit_exit({"no_rule": 0, "unexpected": 0, "off_target": 0,
                      "unusable": 2}), 2)
    check("exit code of a clean sweep",
          audit_exit({"no_rule": 0, "unexpected": 0, "off_target": 0,
                      "unusable": 0}), 0)

    # the rendering
    line = render_row(_row("2021-04/b", VINTAGE_PRE, ["legacy"],
                           off_target=True, mast_inside_grid=False,
                           nearest_node_m=65500.0,
                           rules={"legacy": {"exact": 1681},
                                  "current": {"exact": 0}}))
    ok("render_row names the entry", "2021-04/b" in line)
    ok("render_row flags an off-target entry",
       "OFF-TARGET" in line and "OUTSIDE-GRID" in line)
    ok("render_row shows both scores", "[c:0 l:1681]" in line)
    ok("render_row shows the nearest node distance", "65.5 km" in line)
    bad = render_row({"entry": "2021-04/zz", "error": "meta.json: OSError: x"})
    ok("render_row reports an unusable entry",
       "unusable" in bad and "2021-04/zz" in bad)
    striped = _row("2021-04/b", VINTAGE_PRE, ["legacy"], off_target=True,
                   strip={"rule": "legacy", "samples": 4400, "exact": 4400})
    check("the table and the CSV share one flags string",
          (render_row(striped).endswith(_flags(striped)),
           entry_csv_row(striped)["flags"]),
          (True, "OFF-TARGET strip 4400/4400"))
    text = "\n".join(render_summary(summary, "monthly_bursts"))
    ok("summary names the cache", "monthly_bursts" in text)
    ok("summary names both vintages",
       VINTAGE_PRE in text and VINTAGE_POST in text)
    ok("summary lists the no-rule entry", "2021-05/d" in text)
    ok("summary lists the wrong-rule entry", "2021-05/e" in text)
    ok("summary lists the unusable entry", "2021-05/g" in text)
    ok("summary lists the off-target entry", "2021-04/b" in text)

    block = report_block(summary, [_row("2021-04/a", VINTAGE_PRE, ["legacy"])],
                         "/home/someone/monthly_bursts", "2026-09-30", 52.38,
                         9.72, OFF_TARGET_KM)
    check("report schema", block["schema"], SCHEMA)
    check("report keeps the cache directory relative", block["cache_dir"],
          "monthly_bursts")
    ok("report carries no absolute path", "/home/" not in json.dumps(block))
    check("report names the fix", (block["fix"]["commit"], block["fix"]["pre"]),
          ("fad0e46", VINTAGE_PRE))
    check("report carries the rows", len(block["entries"]), 1)
    check("report names the csv columns even without --csv",
          len(block["csv_columns"]), len(ENTRY_COLUMNS))
    check("report has no csv block without --csv", block["csv"], None)
    written = report_block(summary, [], "/home/someone/monthly_bursts",
                           "2026-09-30", 52.38, 9.72, OFF_TARGET_KM,
                           "/tmp/somewhere/audit.csv")
    check("report names the csv it wrote",
          (written["csv"]["file"], written["csv"]["rows"]), ("audit.csv", 0))

    # the CSV column contract: 54 columns, none named twice, and a source for
    # every one of them in a row that scored
    check("csv column count", len(ENTRY_COLUMNS), 54)
    check("csv column names are unique", len(set(ENTRY_COLUMNS)),
          len(ENTRY_COLUMNS))
    full = entry_csv_row(_row("2021-04/a", VINTAGE_PRE, ["legacy"]),
                         OFF_TARGET_KM * 1000.0)
    check("csv row has every column", sorted(full), sorted(ENTRY_COLUMNS))
    by_design = {"error", "flags", "strip_rule", "strip_samples", "strip_exact",
                 "strip_fraction"}
    ok("csv leaves no column empty in a row that scored",
       all(full[key] != "" for key in ENTRY_COLUMNS if key not in by_design))
    check("csv derives epoch, km, drift and bytes",
          (full["window_mtime_epoch"], full["nearest_node_km"],
           full["off_target_margin_km"], full["center_delta_line"],
           full["center_delta_pixel"], full["window_bytes"]),
          (int(slc.FIX_TS), "9.000", "-1.000", -1, -1, 1681 * 16))
    bare = {"entry": "2021-05/g", "burst_id": None, "subswath": None,
            "polarisation": None, "acquisition_ts": None,
            "incidence_angle_deg": None, "window_mtime": None,
            "window_mtime_epoch": None, "vintage": None, "expected_rule": None,
            "error": "meta.json: gone", "product": None, "grid_points": None,
            "grid_bbox": None, "mast_inside_grid": None, "nearest_node": None,
            "nearest_node_m": None, "off_target": None, "center": {},
            "anchor_ground_error_m": {}, "window_samples": None, "matched": [],
            "rules": {}, "strip": None}
    check("csv row of a bare unusable entry has every column",
          sorted(entry_csv_row(bare)), sorted(ENTRY_COLUMNS))
    check("csv keeps only the known cells of a bare unusable entry",
          sorted(k for k, v in entry_csv_row(bare).items() if v != ""),
          ["ambiguous", "entry", "error", "flags", "match_count",
           "matched_expected", "month", "name", "no_rule", "unusable",
           "verdict", "wrong_rule"])

    # the verdicts: one per row, in the precedence `audit_exit` uses
    check("verdict of a clean entry",
          classify(_row("2021-04/a", VINTAGE_PRE, ["legacy"])), "consistent")
    check("verdict of an unusable entry",
          classify(_row("2021-05/g", VINTAGE_PRE, [], error="gone")),
          "unusable")
    check("verdict of an entry no rule reproduced",
          classify(_row("2021-04/c", VINTAGE_PRE, [])), "no-rule")
    check("verdict of an ambiguous entry",
          classify(_row("2021-04/d", VINTAGE_PRE, ["legacy", "current"])),
          "ambiguous")
    check("verdict of an entry only the other rule reproduced",
          classify(_row("2021-04/e", VINTAGE_PRE, ["current"])), "wrong-rule")
    check("verdict of an off-target entry on its own rule",
          classify(_row("2021-04/b", VINTAGE_PRE, ["legacy"],
                        off_target=True)), "off-target")
    check("verdicts agree with the counters",
          (summary["verdicts"]["unusable"], summary["verdicts"]["no-rule"],
           summary["verdicts"]["ambiguous"], summary["verdicts"]["wrong-rule"]),
          (summary["unusable"], summary["no_rule"], summary["ambiguous"],
           summary["unexpected"]))
    check("every entry has exactly one verdict",
          sum(summary["verdicts"].values()), summary["entries"])
    ok("verdicts never over-count the flags",
       summary["verdicts"]["off-target"] <= summary["off_target"])

    # the CSV itself: header, one line per row, no CR, and repeatable
    csv_rows = [_row("2021-04/a", VINTAGE_PRE, ["legacy"]),
                _row("2021-04/b", VINTAGE_PRE, ["legacy"], off_target=True,
                     strip={"rule": "legacy", "samples": 4400, "exact": 4400}),
                _row("2021-05/g", VINTAGE_PRE, [], error="meta.json: gone")]
    with tempfile.TemporaryDirectory() as tmp:
        first = os.path.join(tmp, "audit.csv")
        again = os.path.join(tmp, "audit-again.csv")
        check("csv row count", write_entries_csv(first, csv_rows, 10000.0), 3)
        write_entries_csv(again, csv_rows, 10000.0)
        with open(first, "rb") as fh:
            raw = fh.read()
        with open(again, "rb") as fh:
            twice = fh.read()
        with open(first, newline="") as fh:
            parsed = list(csv.reader(fh))
        check("csv header", parsed[0], list(ENTRY_COLUMNS))
        check("csv line count", len(parsed), 4)
        check("csv field count per row", {len(cells) for cells in parsed},
              {len(ENTRY_COLUMNS)})
        ok("csv has no carriage return", b"\r" not in raw)
        ok("csv ends with exactly one newline",
           raw.endswith(b"\n") and not raw.endswith(b"\n\n"))
        ok("two csv writes are byte-identical", raw == twice)
        check("csv records the strip it decoded",
              (parsed[2][ENTRY_COLUMNS.index("strip_rule")],
               parsed[2][ENTRY_COLUMNS.index("strip_samples")],
               parsed[2][ENTRY_COLUMNS.index("strip_fraction")]),
              ("legacy", "4400", "1.000000"))
    return fails


def _print_footprints(store, entry_dir, label):
    """The `--safe` table: what the SAFE that hosts an entry actually covers.
    A failure here is a diagnostic, not a verdict, so it never raises."""
    try:
        with open(os.path.join(entry_dir, "meta.json")) as fh:
            meta = json.load(fh)
        for sw, pol, bbox, near_m, inside in store.footprints(meta):
            print(f"    {label} {sw} {pol}: lat {bbox[0]:.4f}..{bbox[1]:.4f} "
                  f"lon {bbox[2]:.4f}..{bbox[3]:.4f}, nearest grid node "
                  f"{near_m / 1000:7.1f} km, mast inside="
                  f"{'yes' if inside else 'NO'}")
    except Exception as exc:
        print(f"    {label} footprints unavailable: {type(exc).__name__}: {exc}",
              file=sys.stderr)


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog=os.path.basename(__file__),
        description="Audit a cache directory written by the Rust pipeline "
                    "against the Sentinel-1 SLCs it was cut from: score both "
                    "anchor rules per entry, classify each entry by the vintage "
                    "that wrote it, and check that its burst covers the mast.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="exit codes: 0 ok, 1 unusable --cache-dir/--env-file or missing "
               "credentials, 2 cache entries unusable, 3 entries reproduced "
               "by no rule or by the rule the other vintage predicts, 4 "
               "selftest failure, 5 entries reproduced exactly on a burst that "
               "does not cover the mast (3 beats 5, 5 beats 2)")
    ap.add_argument("--cache-dir", metavar="DIR",
                    default=os.environ.get(slc.CACHE_VAR),
                    help="cache written by the Rust pipeline, one "
                         "<YYYY-MM>/<entry>/ sub-directory each "
                         "(default: $%s)" % slc.CACHE_VAR)
    ap.add_argument("--entry", metavar="MONTH/NAME", action="append",
                    help="audit exactly this entry; repeatable")
    ap.add_argument("--month", metavar="YYYY-MM", action="append",
                    help="audit only these months; repeatable")
    ap.add_argument("--limit", type=int, default=0, metavar="N",
                    help="at most N entries per month (default: all)")
    ap.add_argument("--stride", type=int, default=1, metavar="N",
                    help="keep every N-th entry of each month (default: 1)")
    ap.add_argument("--strip-every", dest="strip_every", type=int, default=0,
                    metavar="N",
                    help="also decode the 400x11 strip of every N-th audited "
                         "entry (default: none)")
    ap.add_argument("--max-node-km", type=float, default=OFF_TARGET_KM,
                    metavar="KM",
                    help="flag a burst whose nearest grid node is farther than "
                         "this from the mast, i.e. one that cannot cover it "
                         "(default: %(default)g)")
    ap.add_argument("--env-file", metavar="FILE",
                    default=os.environ.get(slc.ENV_FILE_VAR),
                    help="key=value file holding the CDSE secrets; the process "
                         "environment wins (default: $%s)" % slc.ENV_FILE_VAR)
    # the sibling's env helper, so both CLIs read the mast the same way
    ap.add_argument("--lat", type=float,
                    default=slc._env_float(slc.LAT_VAR, slc.MAST_LAT),
                    help="latitude of the mast (default: %(default)s)")
    ap.add_argument("--lon", type=float,
                    default=slc._env_float(slc.LON_VAR, slc.MAST_LON),
                    help="longitude of the mast (default: %(default)s)")
    ap.add_argument("--safe", action="store_true",
                    help="print the footprint of all six sub-swaths of the SAFE "
                         "that hosts each audited entry, before scoring it")
    ap.add_argument("--json", dest="json_out", metavar="FILE",
                    help="write the full result of the sweep to FILE")
    ap.add_argument("--csv", dest="csv_out", metavar="FILE",
                    help="write every audited entry to FILE as one row of 54 "
                         "columns, in the order the sweep walked them (the "
                         "column names are in `csv_columns` in --json)")
    ap.add_argument("--date", metavar="YYYY-MM-DD",
                    default=datetime.date.today().isoformat(),
                    help="value of `generated` in --json (default: today)")
    ap.add_argument("--quiet", action="store_true",
                    help="only the summary, and the entries that did not match")
    ap.add_argument("--selftest", action="store_true",
                    help="check the cache walk, the aggregation and the "
                         "rendering on fixtures - no network, no cache, no "
                         "credentials")
    args = ap.parse_args(argv)

    if args.selftest:
        print("lumo_burst_cache_audit selftest - no network, no cache, "
              "no credentials")
        fails = _selftest()
        if fails:
            for line in fails:
                print(f"  !! {line}", file=sys.stderr)
            print(f"  {len(fails)} check(s) FAILED", file=sys.stderr)
            return 4
        print("  all checks passed")
        return 0

    if not args.cache_dir:
        print(f"error: no cache directory - pass --cache-dir or set "
              f"${slc.CACHE_VAR}", file=sys.stderr)
        return 1
    if not os.path.isdir(args.cache_dir):
        print(f"error: not a cache directory: {args.cache_dir}",
              file=sys.stderr)
        return 1
    entries = entries_of(args.cache_dir, months=args.month, only=args.entry,
                         limit=args.limit, stride=max(1, args.stride))
    if not entries:
        print("error: no cache entries selected", file=sys.stderr)
        return 1
    try:
        creds = slc.credentials(args.env_file)
    except KeyError as exc:
        print(f"error: credentials {exc.args[0]} are in neither the "
              f"environment nor {args.env_file or '$' + slc.ENV_FILE_VAR}",
              file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"error: cannot read --env-file: {exc}", file=sys.stderr)
        return 1

    name = os.path.basename(os.path.normpath(args.cache_dir))
    print(f"auditing {len(entries)} entries of {name} at "
          f"({args.lat}, {args.lon}), off-target beyond "
          f"{args.max_node_km:g} km")
    store = Store(creds, slc.S3(creds["AWS_ACCESS_KEY_ID"],
                                creds["AWS_SECRET_ACCESS_KEY"]),
                  args.lat, args.lon, args.max_node_km * 1000.0)

    rows = []
    for i, (month, entry_name, entry_dir, mtime) in enumerate(entries, 1):
        if args.safe:
            _print_footprints(store, entry_dir, f"{month}/{entry_name}")
        strip_lines = 1 if args.strip_every and i % args.strip_every == 0 else 0
        row = audit_entry(store, month, entry_name, entry_dir, mtime,
                          strip_lines)
        rows.append(row)
        if not args.quiet or row["error"] or not row["matched"]:
            print(render_row(row), flush=True)

    summary = summarise(rows)
    for line in render_summary(summary, name):
        print(line)
    if args.json_out:
        block = report_block(summary, rows, args.cache_dir, args.date,
                             args.lat, args.lon, args.max_node_km,
                             args.csv_out)
        with open(args.json_out, "w") as fh:
            json.dump(block, fh, indent=2, sort_keys=False)
            fh.write("\n")
        print(f"wrote {args.json_out}")
    if args.csv_out:
        try:
            written = write_entries_csv(args.csv_out, rows,
                                        args.max_node_km * 1000.0)
        except OSError as exc:
            print(f"error: cannot write --csv: {exc}", file=sys.stderr)
            return 1
        print(f"wrote {written} entries to {args.csv_out}")
    code = audit_exit(summary)
    if code == 3:
        print(f"RESULT: MISMATCH - {summary['no_rule']} entry/entries "
              f"reproduced by no rule, {summary['unexpected']} by the rule the "
              f"other vintage predicts")
    elif code == 5:
        print(f"RESULT: OFF-TARGET - {summary['off_target']} entry/entries "
              f"reproduced their vintage rule exactly but sit on a burst whose "
              f"nearest grid node is more than {args.max_node_km:g} km from the "
              f"mast")
    elif code == 2:
        print(f"RESULT: INCOMPLETE - {summary['unusable']} entry/entries could "
              f"not be read or resolved")
    else:
        print("RESULT: CONSISTENT - every audited entry reproduced exactly by "
              "the rule its vintage predicts")
    return code


if __name__ == "__main__":
    sys.exit(main())
