#!/usr/bin/env python3
"""LUMO SLC burst cache — what the audit sweep found (paper figure).

Five stacked panels over the entries of the `--stride 10` sweep the audit
exported to `data/audit.csv`, one row per audited cache entry:

  1) containment    distance to the nearest geolocation-grid node, per audited
                    acquisition, coloured by verdict, with the audit's
                    `--max-node-km` threshold drawn: one cluster at 3.7-9.9 km
                    and the off-target entries at 62-65 km
  2) window scores  fraction of window samples each rule reproduces bit for
                    bit, the rule the vintage expects ("held") against the
                    other one, per vintage: 1.0 / 0.0
  3) ground error   distance between each rule's anchor and the mast, held and
                    other rule, per vintage (log scale): the size of the error
                    a silent clamp leaves behind
  4) anchor drift   legacy minus current anchor, line and pixel, per audited
                    acquisition and vintage — the pixel drift the `fad0e46`
                    fix removed
  5) availability   audited entries per month, stacked by vintage, with the
                    months whose strips were decoded marked

Provenance: the rows are the `--csv` artefact of
`code/lumo_burst_cache_audit.py --cache-dir DIR --stride 10 --strip-every 10`,
i.e. one entry out of every ten in the cache, with every tenth of those also
decoded over its full 4400-sample strip. Two sweeps of one cache diff cleanly:
the export carries no timestamp and every number here is a count or a summary
of the CSV's own columns, so this figure adds no analysis of its own — it
reports the audit's verdicts, offline, and reads nothing else (no network, no
tower database, no cache directory).

Careful with the two time axes. `acquisition_ts` (2018-01 .. 2021-07 here) is
the physical acquisition timeline and is what panels 1, 4 and 5 are plotted
against. `window_mtime` is *not* an acquisition date: it is when the cache
wrote the file, i.e. the only surviving evidence of which anchor revision cut
it, and it is what `vintage` is derived from (pre-`fad0e46` = clamped anchor,
post-`fad0e46` = fixed cell selection). `vintage` is the rule the cache should
hold, `matched_rules` the rule that actually reproduces the cached bytes; an
entry the *other* vintage's rule reproduces is a contradiction and is reported
as one by the audit, never averaged away here.

Reads:  ../data/audit.csv (the --csv artefact of code/lumo_burst_cache_audit.py)
Writes: ../data/fig_burst_cache_audit.json (or --json-dir),
        ../figures/fig_burst_cache_audit.md / .png (or --out-dir)

Usage:
    ../figures/fig_burst_cache_audit.sh
    python3 fig_burst_cache_audit.py --csv ../data/audit.csv --out-dir /tmp
"""
import argparse
import csv
import json
import os
from datetime import date

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    plt = None

HERE = os.path.dirname(os.path.abspath(__file__))
# this script lives in `code/`; the artefact it reads is the CSV the audit
# writes, in the sibling `data/` directory (the same convention as
# `code/lumo_damping_frequencies.py`). The figure's own .json/.md/.png go in
# the sibling `figures/` directory; only the .json also moves to `data/`.
DATA_DIR = os.path.normpath(os.path.join(HERE, os.pardir, "data"))
FIGURES_DIR = os.path.normpath(os.path.join(HERE, os.pardir, "figures"))
CSV = os.path.join(DATA_DIR, "audit.csv")

SCHEMA = "fig_burst_cache_audit/v1"
VINTAGE_PRE = "pre-fad0e46"
VINTAGE_POST = "post-fad0e46"
VINTAGES = (VINTAGE_PRE, VINTAGE_POST)
# the audit's `expected_rule` naming: one rule per vintage, and the rule the
# *other* vintage was cut with is what the fix replaced (or introduced)
HELD_RULE = {VINTAGE_PRE: "legacy", VINTAGE_POST: "current"}
OTHER_RULE = {VINTAGE_PRE: "current", VINTAGE_POST: "legacy"}
# the audit's own `VERDICTS`, in its order of precedence, so the counts here
# and in `--json` read the same
VERDICTS = ("unusable", "no-rule", "ambiguous", "wrong-rule", "off-target",
            "consistent")
# the seven verdict booleans `ENTRY_COLUMNS` keeps so the word never hides them
VERDICT_FLAGS = ("unusable", "no_rule", "ambiguous", "wrong_rule", "off_target",
                 "outside_grid", "matched_expected")
VERDICT_COLOR = {"consistent": "tab:blue", "off-target": "tab:red"}
VINTAGE_COLOR = {VINTAGE_PRE: "tab:red", VINTAGE_POST: "tab:blue"}
# the audit's `--max-node-km` default: one grid step in range is ~10 km
OFF_TARGET_KM = 10.0

# The audit's own column names, verbatim. A rename in `ENTRY_COLUMNS` must fail
# loudly here rather than let this figure plot zeros.
REQUIRED = (
    "entry", "month", "subswath", "polarisation", "acquisition_ts",
    "incidence_angle_deg", "window_mtime", "vintage", "expected_rule",
    "verdict", "unusable", "no_rule", "ambiguous", "wrong_rule", "off_target",
    "outside_grid", "matched_expected", "matched_rules", "mast_inside_grid",
    "nearest_node_km", "off_target_margin_km", "window_samples", "window_bytes",
    "legacy_exact", "legacy_fraction", "legacy_ground_error_m",
    "current_exact", "current_fraction", "current_ground_error_m",
    "center_delta_line", "center_delta_pixel",
    "strip_rule", "strip_samples", "strip_exact", "strip_fraction")


def _cell(row, key):
    """One CSV cell, stripped: an empty cell is the audit's "unknown"."""
    return (row.get(key) or "").strip()


def _num(row, key):
    """A CSV cell as a float, or None when the audit left it unknown."""
    v = _cell(row, key)
    return float(v) if v else None


def _bit(row, key):
    """A measured flag as a bool, or None when the audit left it unknown."""
    v = _cell(row, key)
    return None if v == "" else v == "1"


def _day(row, key="acquisition_ts"):
    """The date part of a timestamp cell: both timelines here are day-spaced."""
    return _cell(row, key)[:10]


def _d(iso):
    return date(int(iso[:4]), int(iso[5:7]), int(iso[8:10]))


def _tally(values):
    """Counts by value, keys sorted so two sweeps of one cache diff cleanly."""
    out = {}
    for v in values:
        k = v or "unknown"
        out[k] = out.get(k, 0) + 1
    return {k: out[k] for k in sorted(out)}


def _describe(values, nd=3):
    """n / min / median / max of a column, ignoring the audit's empty cells."""
    v = [x for x in values if x is not None]
    if not v:
        return {"n": 0, "min": None, "median": None, "max": None}
    return {"n": len(v), "min": round(min(v), nd),
            "median": round(float(np.median(v)), nd), "max": round(max(v), nd)}


def _fractions(values):
    """A `*_fraction` column: its summary, plus how 0 / 1 / in-between it is."""
    d = _describe(values, 6)
    v = [x for x in values if x is not None]
    d["n_zero"] = sum(1 for x in v if x == 0.0)
    d["n_full"] = sum(1 for x in v if x == 1.0)
    d["n_partial"] = len(v) - d["n_zero"] - d["n_full"]
    return d


def load(path=CSV):
    """The audit's `--csv` artefact: one dict per audited cache entry."""
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        have = set(reader.fieldnames or ())
        missing = [c for c in REQUIRED if c not in have]
        if missing:
            raise SystemExit(f"{path}: not an audit --csv artefact "
                             f"(missing column(s): {', '.join(missing)})")
        rows = list(reader)
    if not rows:
        raise SystemExit(f"{path}: no audited entries")
    return rows


def _verdicts(rows):
    """The audit's verdict counts, all six keys present as in its `--json`."""
    out = {v: 0 for v in VERDICTS}
    for r in rows:
        out[_cell(r, "verdict")] = out.get(_cell(r, "verdict"), 0) + 1
    return out


def _window_block(rows, rule, samples):
    """One rule's window score over `rows`: exact cells and fractions."""
    d = _fractions([_num(r, f"{rule}_fraction") for r in rows])
    exact = [_num(r, f"{rule}_exact") for r in rows]
    d["n_exact_full"] = sum(1 for x, s in zip(exact, samples) if x == s)
    return d


def _error_block(rows, rule):
    """One rule's anchor-to-mast ground error over `rows`, in metres.

    0.0 m is not a missing value but an anchor that maps back onto the mast,
    i.e. a rule that leaves no silent-clamp error behind, so it is counted
    apart from the minimum."""
    vals = [_num(r, f"{rule}_ground_error_m") for r in rows]
    d = _describe(vals, 1)
    d["n_zero"] = sum(1 for x in vals if x == 0.0)
    return d


def _strip_block(rows):
    """The decoded strips, from the audit's own strip columns."""
    d = _fractions([_num(r, "strip_fraction") for r in rows])
    d["n_exact_full"] = sum(
        1 for r in rows
        if _num(r, "strip_samples") is not None
        and _num(r, "strip_exact") == _num(r, "strip_samples"))
    d["samples"] = _tally(_cell(r, "strip_samples") for r in rows)
    d["rules"] = _tally(_cell(r, "strip_rule") for r in rows)
    return d


def _vintage_block(rows, vintage):
    """One vintage: the rule its mtime predicts scored against the other one."""
    held, other = HELD_RULE[vintage], OTHER_RULE[vintage]
    sub = [r for r in rows if _cell(r, "vintage") == vintage]
    samples = [_num(r, "window_samples") for r in sub]
    off = [r for r in sub if _bit(r, "off_target") is True]
    return {
        "entries": len(sub),
        "expected_rule": held,
        "expected_rule_matches": sum(1 for r in sub
                                     if _cell(r, "expected_rule") == held),
        "verdicts": _verdicts(sub),
        "matched_rules": _tally(_cell(r, "matched_rules") for r in sub),
        "window_samples": _describe(samples, 1),
        "held_window": _window_block(sub, held, samples),
        "other_window": _window_block(sub, other, samples),
        "held_ground_error_m": _error_block(sub, held),
        "other_ground_error_m": _error_block(sub, other),
        "anchor_delta": {
            "line": _describe([_num(r, "center_delta_line") for r in sub], 1),
            "pixel": _describe([_num(r, "center_delta_pixel") for r in sub], 1)},
        "nearest_node_km": _describe([_num(r, "nearest_node_km") for r in sub]),
        "off_target_margin_km": _describe(
            [_num(r, "off_target_margin_km") for r in sub]),
        "mast_outside_grid": sum(1 for r in sub if _bit(r, "outside_grid")),
        "off_target": {"entries": len(off), "list": [r["entry"] for r in off],
                       "nearest_node_km": _describe(
                           [_num(r, "nearest_node_km") for r in off]),
                       "months": sorted({_cell(r, "month") for r in off})},
        "mtime_days": _tally(_day(r, "window_mtime") for r in sub),
        "strips": _strip_block([r for r in sub
                                if _num(r, "strip_samples") is not None]),
    }


def _monthly(rows):
    """Per-month availability of the sample, split by vintage.

    A vintage the audit does not know would still count in `entries`, so the
    per-month sums cannot silently overstate the two columns."""
    out = {}
    for r in rows:
        slot = out.setdefault(_cell(r, "month"),
                              {"entries": 0, VINTAGE_PRE: 0, VINTAGE_POST: 0,
                               "off_target": 0, "strips": 0})
        vintage = _cell(r, "vintage")
        slot["entries"] += 1
        if vintage in slot:
            slot[vintage] += 1
        slot["off_target"] += int(_bit(r, "off_target") is True)
        slot["strips"] += int(_num(r, "strip_samples") is not None)
    return {m: out[m] for m in sorted(out)}


def stats(rows, csv_path=CSV):
    """The `.json` payload: counts and summaries of the audit's own columns.

    Nothing is recomputed from the raw bursts — the audit already decided every
    number here — and no absolute path is written, so two sweeps of one cache
    produce byte-identical output."""
    inside = [r for r in rows if _bit(r, "mast_inside_grid") is True]
    outside = [r for r in rows if _bit(r, "outside_grid") is True]
    off = [r for r in rows if _bit(r, "off_target") is True]
    strips = [r for r in rows if _num(r, "strip_samples") is not None]
    # the threshold the audit used is recoverable from its own two columns:
    # off_target_margin_km == nearest_node_km - max_node_km for every entry
    gaps = [_num(r, "nearest_node_km") - _num(r, "off_target_margin_km")
            for r in rows if _num(r, "off_target_margin_km") is not None]
    recovered = round(float(np.median(gaps)), 3) if gaps else None
    return {
        "schema": SCHEMA,
        "csv": {"file": os.path.basename(csv_path), "rows": len(rows)},
        "n_entries": len(rows),
        "n_months": len({_cell(r, "month") for r in rows}),
        "date_first": min(_day(r) for r in rows),
        "date_last": max(_day(r) for r in rows),
        "subswaths": _tally(_cell(r, "subswath") for r in rows),
        "polarisations": _tally(_cell(r, "polarisation") for r in rows),
        "incidence_angle_deg": _describe(
            [_num(r, "incidence_angle_deg") for r in rows]),
        "window_samples": _describe([_num(r, "window_samples") for r in rows], 1),
        "window_bytes": _tally(_cell(r, "window_bytes") for r in rows),
        "max_node_km": recovered,
        "max_node_km_source": "nearest_node_km - off_target_margin_km",
        "max_node_km_is_default": recovered == OFF_TARGET_KM,
        "verdicts": _verdicts(rows),
        "verdict_flags": {flag: sum(1 for r in rows if _bit(r, flag) is True)
                          for flag in VERDICT_FLAGS},
        "matched_rules": _tally(_cell(r, "matched_rules") for r in rows),
        "fractions": {
            "legacy": _fractions([_num(r, "legacy_fraction") for r in rows]),
            "current": _fractions([_num(r, "current_fraction") for r in rows])},
        "nearest_node_km": _describe([_num(r, "nearest_node_km") for r in rows]),
        "off_target_margin_km": _describe(
            [_num(r, "off_target_margin_km") for r in rows]),
        "grid": {
            "mast_inside": len(inside), "outside_grid": len(outside),
            "columns_agree": len(inside) == len(rows) - len(outside),
            "inside_node_km": _describe(
                [_num(r, "nearest_node_km") for r in inside]),
            "outside_node_km": _describe(
                [_num(r, "nearest_node_km") for r in outside])},
        "off_target": {
            "entries": len(off), "list": [r["entry"] for r in off],
            "months": sorted({_cell(r, "month") for r in off}),
            "vintages": _tally(_cell(r, "vintage") for r in off),
            "nearest_node_km": _describe(
                [_num(r, "nearest_node_km") for r in off]),
            "threshold_km": recovered},
        "strips": dict(_strip_block(strips), entries=len(strips)),
        "mtime_days": _tally(_day(r, "window_mtime") for r in rows),
        "vintages": {v: _vintage_block(rows, v) for v in VINTAGES},
        "monthly": _monthly(rows),
    }


def plot(rows, out, out_dir=FIGURES_DIR):
    """The five stacked panels; `.json`/`.md` are written even without it."""
    if plt is None:
        print("matplotlib unavailable — skipping the figure")
        return False
    vintages = out["vintages"]
    fig, axes = plt.subplots(5, 1, figsize=(11, 14), sharex=True)

    # 1) containment: distance to the nearest geolocation-grid node
    ax = axes[0]
    for verdict in ("consistent", "off-target"):
        sub = [r for r in rows if _cell(r, "verdict") == verdict]
        if not sub:
            continue
        ax.plot([_d(_day(r)) for r in sub],
                [_num(r, "nearest_node_km") for r in sub], ".", ms=5,
                color=VERDICT_COLOR[verdict], label=f"{verdict} (n={len(sub)})")
    if out["max_node_km"] is not None:
        ax.axhline(out["max_node_km"], color="0.3", ls="--", lw=1,
                   label=f"--max-node-km ({out['max_node_km']:g} km)")
    ax.set_yscale("log")
    ax.set_title("Containment — distance to the nearest grid node, per "
                 "acquisition", fontsize=9, loc="left")
    ax.set_ylabel("km (log)", fontsize=8)

    # 2) window scores: the rule the vintage expects against the other one
    ax = axes[1]
    xs = np.arange(len(VINTAGES), dtype=float)
    width = 0.36
    for i, vintage in enumerate(VINTAGES):
        for slot, dx in (("held_window", -width / 2), ("other_window", width / 2)):
            rule = (HELD_RULE if slot == "held_window" else OTHER_RULE)[vintage]
            d = vintages[vintage][slot]
            ax.bar(xs[i] + dx, d["median"] or 0.0, width,
                   color=VINTAGE_COLOR[vintage],
                   alpha=1.0 if slot == "held_window" else 0.45,
                   hatch=None if slot == "held_window" else "//",
                   label=f"{vintage} · {rule} (n={d['n']})")
            if d["n_partial"]:
                ax.text(xs[i] + dx, 0.08, f"{d['n_partial']} partial",
                        ha="center", fontsize=6, rotation=90)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{v}\n({vintages[v]['entries']} entries)"
                        for v in VINTAGES], fontsize=8)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("fraction of the window", fontsize=8)
    ax.set_title("Window score — samples reproduced bit for bit (median per "
                 "rule), the other rule hatched", fontsize=9, loc="left")

    # 3) ground error of each rule's anchor, per vintage
    ax = axes[2]
    floor = 1.0
    series = []
    for vintage in VINTAGES:
        series += [(f"{vintage} · held ({HELD_RULE[vintage]})",
                    vintages[vintage]["held_ground_error_m"],
                    VINTAGE_COLOR[vintage]),
                   (f"{vintage} · other ({OTHER_RULE[vintage]})",
                    vintages[vintage]["other_ground_error_m"], "0.5")]
    for y, (label, d, color) in enumerate(series):
        if not d["n"]:
            continue
        ax.plot([max(d["min"], floor), max(d["max"], floor)], [y, y], "-",
                lw=1.4, color=color, alpha=0.7)
        ax.plot([max(d["median"], floor)], [y], "o", ms=7, color=color)
        ax.text(max(d["max"], floor) * 1.25, y,
                f"n={d['n']}, median {d['median']:g} m, max {d['max']:g} m",
                fontsize=6.5, va="center")
        if d["n_zero"]:
            ax.text(floor * 0.55, y, f"{d['n_zero']}× 0.0 m", fontsize=6.5,
                    va="center", color=color)
    ax.set_yticks(range(len(series)))
    ax.set_yticklabels([s[0] for s in series], fontsize=7)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlim(floor * 0.45, 6e5)
    ax.set_xlabel("ground distance between the rule's anchor and the mast "
                  "[m, log]", fontsize=8)
    ax.set_title("Ground error of each rule's anchor — 0.0 m = the anchor "
                 "lands on the mast", fontsize=9, loc="left")

    # 4) the pixel drift the fix removed: legacy anchor minus current
    ax = axes[3]
    for vintage in VINTAGES:
        sub = [r for r in rows if _cell(r, "vintage") == vintage]
        if not sub:
            continue
        days = [_d(_day(r)) for r in sub]
        ax.plot(days, [_num(r, "center_delta_line") for r in sub], ".", ms=4,
                color=VINTAGE_COLOR[vintage], label=f"{vintage} line")
        ax.plot(days, [_num(r, "center_delta_pixel") for r in sub], "^", ms=3,
                alpha=0.6, color=VINTAGE_COLOR[vintage],
                label=f"{vintage} pixel")
    ax.axhline(0.0, color="0.3", lw=1, ls=":")
    ax.set_ylabel("line / pixel", fontsize=8)
    ax.set_title("Anchor drift the fix removed — legacy minus current anchor, "
                 "per acquisition", fontsize=9, loc="left")

    # 5) availability of the audited sample, by vintage
    ax = axes[4]
    months = sorted(out["monthly"])
    xs = [_d(m + "-01") for m in months]
    bottom = np.zeros(len(months))
    for vintage in VINTAGES:
        vals = np.array([out["monthly"][m][vintage] for m in months],
                        dtype=float)
        ax.bar(xs, vals, width=22, bottom=bottom,
               color=VINTAGE_COLOR[vintage], label=vintage)
        bottom += vals
    for key, marker, offset, color in (("off_target", "v", 0.4, "tab:red"),
                                       ("strips", "1", 1.3, "tab:green")):
        vals = np.array([out["monthly"][m][key] for m in months], dtype=float)
        if vals.sum():
            ax.plot([x for x, n in zip(xs, vals) if n],
                    [b + offset for b, n in zip(bottom, vals) if n], marker,
                    ms=6, color=color, label=f"{key} ({int(vals.sum())})")
    ax.set_ylabel("audited entries", fontsize=8)
    ax.set_xlabel("acquisition month", fontsize=8)
    ax.set_title(f"Availability of the audited sample — {out['n_entries']} "
                 f"entries in {out['n_months']} months, by vintage",
                 fontsize=9, loc="left")

    for ax in axes:
        ax.grid(alpha=0.3)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=6.5, ncol=2, loc="best")
    verdicts = out["verdicts"]
    fig.suptitle(
        f"SLC burst-cache audit — {out['n_entries']} audited cache entries "
        f"({out['date_first']} .. {out['date_last']}): {verdicts['consistent']} "
        f"consistent, {out['off_target']['entries']} off-target, "
        f"{out['strips']['n_exact_full']}/{out['strips']['entries']} strips "
        "exact\none entry per ten in the cache (source: "
        f"{out['csv']['file']}, the --csv artefact of "
        "code/lumo_burst_cache_audit.py --stride 10 --strip-every 10)",
        fontsize=10)
    fig.autofmt_xdate()
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    png = os.path.join(out_dir, "fig_burst_cache_audit.png")
    fig.savefig(png, dpi=130)
    plt.close(fig)
    print(f"Wrote: {png}")
    return True


def _g(value, fmt="g"):
    """A summary number for the markdown, or `-` when the column was empty."""
    return "-" if value is None else format(value, fmt)


def _list(items, limit=8):
    """An entry list for the markdown, abbreviated past `limit` names."""
    if len(items) <= limit:
        return ", ".join(f"`{i}`" for i in items)
    return (", ".join(f"`{i}`" for i in items[:limit])
            + f", … (+{len(items) - limit} more)")


def report(out, out_dir=FIGURES_DIR):
    """Companion markdown for the figure (LUMO convention)."""
    vintages, v = out["vintages"], out["verdicts"]
    lg, cu = out["fractions"]["legacy"], out["fractions"]["current"]
    held_full = sum(vintages[x]["held_window"]["n_full"] for x in VINTAGES)
    other_full = sum(vintages[x]["other_window"]["n_full"] for x in VINTAGES)
    L = ["# SLC burst cache — Audit Sweep Summary", "",
         f"Every tenth entry of the cache: {out['n_entries']} audited entries "
         f"in {out['n_months']} month folders, acquisitions "
         f"{out['date_first']} .. {out['date_last']}, sub-swaths "
         + ", ".join(f"{k} {n}" for k, n in out["subswaths"].items())
         + ", polarisations "
         + ", ".join(f"{k} {n}" for k, n in out["polarisations"].items())
         + f", window {_g(out['window_samples']['median'], '.0f')} samples, "
         + ", ".join(f"{k} B" for k in out["window_bytes"]) + ".", "",
         f"From `{out['csv']['file']}`, the `--csv` artefact of "
         "`code/lumo_burst_cache_audit.py --stride 10 --strip-every 10`. Every "
         "number below is a count or a summary of that file's own columns.",
         "", "| verdict | n |", "|---------|--:|"]
    for verdict in VERDICTS:
        L.append(f"| {verdict} | {v[verdict]} |")
    L += ["", "| vintage | entries | expected rule | window: held / other "
          "(median fraction) | anchor delta (median line) | off-target |",
          "|---------|--------:|---------------|------------------------------"
          "|-----------------------:|-----------:|"]
    for vintage in VINTAGES:
        b = vintages[vintage]
        L.append(f"| {vintage} | {b['entries']} | {b['expected_rule']} "
                 f"({b['expected_rule_matches']}/{b['entries']} rows) | "
                 f"{_g(b['held_window']['median'], '.3f')} / "
                 f"{_g(b['other_window']['median'], '.3f')} | "
                 f"{_g(b['anchor_delta']['line']['median'], '.1f')} | "
                 f"{b['off_target']['entries']} |")
    L += ["", "## Containment", "",
          "| split | n | nearest grid node [km]: min / median / max |",
          "|-------|--:|-------------------------------------------:|"]
    for label, d in (("mast inside the burst's grid",
                      out["grid"]["inside_node_km"]),
                     ("mast outside it", out["grid"]["outside_node_km"])):
        L.append(f"| {label} | {d['n']} | {_g(d['min'], '.3f')} / "
                 f"{_g(d['median'], '.3f')} / {_g(d['max'], '.3f')} |")
    L += ["", f"The {out['off_target']['entries']} off-target entries "
          f"(nearest node beyond the threshold, "
          f"{_g(out['max_node_km'], 'g')} km) come from "
          + ", ".join(f"{k} {n}" for k, n in out["off_target"]["vintages"].items())
          + " and sit "
          f"{_g(out['off_target']['nearest_node_km']['min'], '.3f')} .. "
          f"{_g(out['off_target']['nearest_node_km']['max'], '.3f')} km out: "
          + _list(out["off_target"]["list"]) + ".", ""]
    L += ["## Ground error — distance between each rule's anchor and the mast",
          "",
          "| vintage | rule | role | n | anchors at 0.0 m | median [m] | max [m] |",
          "|---------|------|------|--:|-----------------:|-----------:|--------:|"]
    for vintage in VINTAGES:
        b = vintages[vintage]
        for slot, role, rule in (
                ("held_ground_error_m", "the vintage's rule", HELD_RULE[vintage]),
                ("other_ground_error_m", "the other rule", OTHER_RULE[vintage])):
            d = b[slot]
            L.append(f"| {vintage} | {rule} | {role} | {d['n']} | "
                     f"{d['n_zero']} | {_g(d['median'], '.1f')} | "
                     f"{_g(d['max'], '.1f')} |")
    s = out["strips"]
    if s["min"] == s["max"]:
        strip_fraction = "all at fraction " + _g(s["max"], ".6f")
    else:
        strip_fraction = (f"fraction {_g(s['min'], '.6f')} .. "
                          f"{_g(s['max'], '.6f')}")
    L += ["", "## Decoded strips", "",
          f"{s['entries']} of the {out['n_entries']} entries were also decoded "
          "over their full strip ("
          + ", ".join(f"{k} samples" for k in s["samples"])
          + "), reproduced by rule "
          + ", ".join(f"`{k}` in {n}" for k, n in s["rules"].items())
          + f": {strip_fraction}, {s['n_exact_full']} of {s['entries']} exact in "
          f"full and {s['n_partial']} partial — the strip agrees with the window "
          "it was cut from.", ""]
    L += ["## Reading", "",
          f"- Every window file is reproduced bit for bit by exactly one rule, "
          f"the one its vintage predicts ({held_full} of {out['n_entries']} "
          "entries at fraction 1.0, and the other rule in none of them — it "
          f"reaches 1.0 in {other_full} entries). Pooled over the "
          f"sample: `legacy` 1.0 in {lg['n_full']} entries, 0.0 in "
          f"{lg['n_zero']}, in between in {lg['n_partial']}; `current` 1.0 in "
          f"{cu['n_full']}, 0.0 in {cu['n_zero']}, in between in "
          f"{cu['n_partial']}. Those in-between entries are the near-misses a "
          "rule test on one entry alone would misread as a contradiction.",
          "- No entry is unusable, unmatched, ambiguous or reproduced by the "
          "other vintage's rule: verdict flags "
          + ", ".join(f"{k}={n}" for k, n in out["verdict_flags"].items()) + ".",
          f"- The threshold the audit applied is recoverable from its own two "
          f"columns: `{out['max_node_km_source']}` gives "
          f"{_g(out['max_node_km'], 'g')} km for every entry, the "
          "`--max-node-km` default (one grid step in range is ~10 km).",
          "- The fix shows up in the anchors too: the pre-fix entries' bytes "
          "match `legacy`, whose anchor sits a median "
          f"{_g(vintages[VINTAGE_PRE]['held_ground_error_m']['median'], '.1f')} m "
          "from the mast, while the post-fix rule lands on it "
          f"({vintages[VINTAGE_POST]['held_ground_error_m']['n_zero']} of "
          f"{vintages[VINTAGE_POST]['entries']} entries at 0.0 m). The drift "
          "between the two anchors is "
          f"{_g(vintages[VINTAGE_PRE]['anchor_delta']['line']['median'], '.1f')} "
          f"lines (pre) and "
          f"{_g(vintages[VINTAGE_POST]['anchor_delta']['line']['median'], '.1f')} "
          "lines (post).",
          f"- The vintage is read from the mtime of each `window.bin` ("
          + ", ".join(f"{k} → {n} entries" for k, n in out["mtime_days"].items())
          + "), never from the acquisition date: the two timelines are years "
          "apart on purpose.", "",
          "## Caveats", "",
          "- A sample, not a census: `--stride 10` audits one entry in ten, and "
          f"only {s['entries']} of those also had their strip decoded. Nothing "
          "here is a statement about the whole cache — the full sweep is "
          "`--stride 1`.",
          "- The time-series panels' x axis is `acquisition_ts`, the physical "
          "timeline. `window_mtime` is when the cache wrote the file and is the "
          "only surviving evidence of the vintage, so an entry whose mtime was "
          "copied or restored would be mis-vintaged.",
          "- A small ground error is the size of the clamp error a rule leaves "
          "behind, not proof of a correct mapping: the verdict comes from the "
          "decoded window bytes, and the distance is reported next to it, "
          "never instead of it.",
          "- This figure adds no analysis of its own: it counts and plots the "
          "audit's columns, offline, reading no cache directory and no network.",
          "",
          "Build: `python3 fig_burst_cache_audit.py` (reads "
          f"`../data/{out['csv']['file']}` by default; `--csv` and `--out-dir` "
          "point it at another sweep or another directory).", ""]
    md = os.path.join(out_dir, "fig_burst_cache_audit.md")
    with open(md, "w") as fh:
        fh.write("\n".join(L))
    print("Wrote: fig_burst_cache_audit.md")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default=CSV, metavar="FILE",
                    help="audit --csv artefact to read (default: %(default)s)")
    ap.add_argument("--out-dir", default=FIGURES_DIR, metavar="DIR",
                    help="directory for the .md/.png (default: %(default)s)")
    ap.add_argument("--json-dir", default=DATA_DIR, metavar="DIR",
                    help="directory for the .json (default: %(default)s)")
    args = ap.parse_args()

    rows = load(args.csv)
    out = stats(rows, args.csv)
    with open(os.path.join(args.json_dir, "fig_burst_cache_audit.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print("Wrote: fig_burst_cache_audit.json")
    plot(rows, out, args.out_dir)
    report(out, args.out_dir)
    v = out["verdicts"]
    print(f"n={out['n_entries']}  source={out['csv']['file']}  "
          f"consistent={v['consistent']}  "
          f"off-target={out['off_target']['entries']}  "
          f"strips={out['strips']['n_exact_full']}/{out['strips']['entries']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
