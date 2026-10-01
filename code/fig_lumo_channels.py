#!/usr/bin/env python3
"""LUMO tower — observation-channel time histories (paper figure).

Five stacked time-history panels over the full Sentinel-1 burst record
(2020-08 .. 2021-07). LUMO has no collapse; the DAM 3 / DAM 4 / DAM 6
campaign windows are shaded instead.

  1) SAR brightness        (strip_brightness_ratio, ASC / DESC)
  2) raw SAR phase         (phase_coherence; per overpass + monthly median)
  3) whole-structure gamma2 (per overpass + monthly median)
  4) dwell/sub-aperture    (sub_aperture_brightness.modulation_depth)
  5) data availability     (de-duplicated overpasses per month, stacked by orbit)

De-duplication (important): the analysis artifacts carry each physical overpass
several times — once per polarisation (VV+VH) and, for some overpasses, under
two burst ids a few seconds apart (Sentinel-1 bursts repeat every ~2.7 s). The
raw stacks therefore hold 482 rows (356 after the upstream builder has collapsed
the sub-minute repeats) but only 178 distinct overpasses. Plotting the
pseudo-replicates overplots the scatter and inflates every count, so all channels
are collapsed to one row per (acquisition minute, orbit), preferring VV — the
same identity the backend's `insar_monitor::acquisition_key` uses.
Both the raw row count and the de-duplicated overpass count are reported.

Reads:  data/lumo_channels.csv (the sibling `data/` directory) — the flat
        per-record CSV that `code/figures_data_csv.py` builds from
        data/lumo_tower_monthly_amp_phase.json,
        data/lumo_tower_coherence_states.json and
        data/lumo_tower_monthly_timeseries.json. That CSV is the **only**
        input: this figure never reads those three artifacts itself.
Writes: ../data/fig_lumo_channels.json, ../figures/fig_lumo_channels.md / .png

Usage:
    ../figures/fig_lumo_channels.sh                   # data/lumo_channels.csv
    python3 fig_lumo_channels.py --csv OTHER.csv      # an explicit CSV path

Exit codes: 0 wrote the figure; 2 the CSV is missing or is not a channel CSV.
"""
import argparse
import collections
import json
import os
import sys
from datetime import date

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    plt = None

HERE = os.path.dirname(os.path.abspath(__file__))
# this script lives in `code/`; the CSV it reads lives in the sibling `data/`
# directory, its .json output also goes to `data/`, and its .md/.png go to the
# sibling `figures/` directory (the same convention as
# `code/lumo_damping_frequencies.py`)
DATA_DIR = os.path.normpath(os.path.join(HERE, os.pardir, "data"))
FIGURES_DIR = os.path.normpath(os.path.join(HERE, os.pardir, "figures"))
CODE_DIR = os.path.normpath(os.path.join(HERE, os.pardir, "code"))
if CODE_DIR not in sys.path:         # also lets `python3 -m figures.<name>` work
    sys.path.insert(0, CODE_DIR)

from figures_data_csv import read_channels_csv  # noqa: E402  (needs the path above)

CHANNELS_CSV = os.path.join(DATA_DIR, "lumo_channels.csv")
DAM_COLORS = {"DAM 3": "tab:orange", "DAM 4": "tab:red", "DAM 6": "tab:purple"}


def _d(iso):
    return date(int(iso[:4]), int(iso[5:7]), int(iso[8:10]))


def _overpass_key_amp(x):
    """Overpass identity of one amplitude/phase row: acquisition minute + orbit."""
    ts = x.get("acquisition_ts") or (str(x.get("acquisition_date", "")) + "T00:00:00Z")
    return (ts[:16], x.get("orbit"))


def _dedup_overpasses(rows, key_fn, better):
    """Collapse pseudo-replicate bursts to one row per physical overpass.

    The CDSE burst catalogue lists one overpass under several burst ids a few
    seconds apart (Sentinel-1 bursts repeat every ~2.7 s) and the analysis
    artifacts carry each acquisition once per polarisation, so the 482-row
    stacks hold only 178 distinct overpasses. Duplicates overplot the scatter
    and inflate every count/median, so one representative per bucket is kept:
    the first row of a bucket wins, or `better(new, current)` if it returns True.
    """
    chosen = {}
    order = []
    for x in rows:
        k = key_fn(x)
        if k not in chosen:
            chosen[k] = x
            order.append(k)
        elif better(x, chosen[k]):
            chosen[k] = x
    return [chosen[k] for k in order]


def load_bursts(gamma2_ids, bursts):
    """De-duplicated amplitude/phase/modulation channels + availability counts.

    Returns (bright, phase, mod, preferred_ids, n_raw, n_overpasses, monthly,
    monthly_by_orbit). Within one overpass VV is preferred (primary polarisation
    of the coherence analysis), falling back to whichever burst carries a
    gamma2 entry. `bursts` is the `burst` block of the input CSV, in artifact
    shape.
    """
    raw = bursts

    def rank(x):
        return (str(x.get("polarisation", "")).lower() == "vv",
                x.get("burst_id") in gamma2_ids)

    dedup = _dedup_overpasses(raw, _overpass_key_amp,
                              lambda x, cur: rank(x) > rank(cur))
    bright = {}
    phase = []
    mod = []
    monthly = collections.Counter()
    monthly_by_orbit = {}
    for x in dedup:
        iso, orbit = x["acquisition_date"], x["orbit"]
        br = x.get("strip_brightness_ratio")
        if br is not None:
            bright.setdefault((iso, orbit), []).append(float(br))
        pc = x.get("phase_coherence")
        if pc is not None:
            phase.append((iso, float(pc)))
        md = (x.get("sub_aperture_brightness") or {}).get("modulation_depth")
        if md is not None:
            mod.append((iso, float(md)))
        month = x.get("month") or iso[:7]
        monthly[month] += 1
        slot = monthly_by_orbit.setdefault(month, {"ASCENDING": 0, "DESCENDING": 0})
        if orbit in slot:
            slot[orbit] += 1
    preferred = [x["burst_id"] for x in dedup]
    return (bright, phase, mod, preferred, len(raw), len(dedup), monthly,
            monthly_by_orbit)


def load_gamma2(preferred_ids, coh_rows):
    """De-duplicated gamma2 series; keep the canonical (VV) burst per overpass.

    The coherence channel carries no polarisation column and only a date, so an
    overpass is identified by (date, orbit) — a safe key because Sentinel-1
    revisits are 6+ days apart — and the burst id selected by `load_bursts` is
    preferred inside each bucket. `coh_rows` is the `coherence` block of the
    input CSV, in artifact shape.
    """
    raw = [x for x in coh_rows if x.get("gamma2") is not None]
    preferred = set(preferred_ids)
    dedup = _dedup_overpasses(
        raw,
        lambda x: (x.get("date"), x.get("orbit")),
        lambda x, cur: (x.get("burst_id") in preferred)
        and (cur.get("burst_id") not in preferred),
    )
    return [(x["date"], float(x["gamma2"])) for x in dedup]


def _med_series(store):
    res = {}
    for orbit in ("ASCENDING", "DESCENDING"):
        pts = sorted((iso, float(np.median(v))) for (iso, o), v in store.items()
                     if o == orbit and v)
        res[orbit] = ([_d(i) for i, _ in pts], [m for _, m in pts])
    return res


def _monthly_median(series):
    mm = {}
    for iso, v in series:
        mm.setdefault(iso[:7], []).append(v)
    months = sorted(mm)
    return ([date(int(m[:4]), int(m[5:7]), 15) for m in months],
            [float(np.median(mm[m])) for m in months])


def stats(bright, phase, mod, gamma2, n_raw, n_overpasses, monthly, monthly_by_orbit):
    def summ(vals):
        vals = [v for v in vals if v is not None]
        return {"n": len(vals), "median": float(np.median(vals)) if vals else None}
    return {
        # Raw artifact rows vs distinct physical overpasses (see module docstring).
        "n_bursts_raw": n_raw,
        "n_overpasses": n_overpasses,
        "n_bursts": n_overpasses,
        "brightness_ratio": summ([v for vs in bright.values() for v in vs]),
        "raw_phase_coherence": summ([v for _, v in phase]),
        "gamma2": summ([v for _, v in gamma2]),
        "modulation_depth": summ([v for _, v in mod]),
        "monthly_overpasses": dict(sorted(monthly.items())),
        "monthly_overpasses_by_orbit": {m: monthly_by_orbit[m]
                                        for m in sorted(monthly_by_orbit)},
    }


def _shade(ax, timeline):
    used = set()
    for p in timeline:
        lab = p["damage_label"]
        if lab == "healthy":
            continue
        key = lab if lab not in used else None
        used.add(lab)
        ax.axvspan(_d(p["start_date"]), _d(p["end_date"]),
                   color=DAM_COLORS.get(lab, "0.8"), alpha=0.15,
                   label=key)


def plot(bright, phase, mod, gamma2, timeline, monthly_by_orbit):
    if plt is None:
        print("matplotlib unavailable — skip plot")
        return False
    fig, axes = plt.subplots(5, 1, figsize=(12, 15), sharex=True)
    bs = _med_series(bright)
    pairs = (("ASCENDING", "o", "tab:blue", "ASC"), ("DESCENDING", "s", "tab:orange", "DESC"))
    # 1) brightness
    ax = axes[0]
    for orbit, mk, col, lab in pairs:
        xs, ys = bs[orbit]
        if xs:
            ax.plot(xs, ys, mk, ms=3, color=col, label=lab)
    ax.set_title("SAR brightness (strip brightness ratio)", fontsize=9)
    # 2) raw phase coherence
    ax = axes[1]
    ax.plot([_d(i) for i, _ in phase], [v for _, v in phase], ".", ms=3,
            color="0.6", label="per overpass")
    mx, my = _monthly_median(phase)
    ax.plot(mx, my, "-o", ms=4, color="tab:red", label="monthly median")
    ax.set_title("Raw SAR phase (phase coherence)", fontsize=9)
    # 3) gamma2
    ax = axes[2]
    ax.plot([_d(i) for i, _ in gamma2], [v for _, v in gamma2], ".", ms=3,
            color="0.6", label="per overpass")
    mx, my = _monthly_median(gamma2)
    ax.plot(mx, my, "-o", ms=4, color="tab:purple", label="monthly median")
    ax.set_title("Whole-structure coherence gamma2", fontsize=9)
    ax.set_ylabel("γ²", fontsize=8)
    # 4) modulation depth
    ax = axes[3]
    ax.plot([_d(i) for i, _ in mod], [v for _, v in mod], ".", ms=3,
            color="0.6", label="per overpass")
    mx, my = _monthly_median(mod)
    ax.plot(mx, my, "-o", ms=4, color="tab:green", label="monthly median")
    ax.set_title("Dwell / sub-aperture modulation depth", fontsize=9)
    # 5) data availability, by orbit, monthly
    ax = axes[4]
    months = sorted(monthly_by_orbit)
    xs = [_d(m + "-15") for m in months]
    bottom = np.zeros(len(months))
    for orbit, col in (("ASCENDING", "tab:blue"), ("DESCENDING", "tab:orange")):
        vals = np.array([monthly_by_orbit[m].get(orbit, 0) for m in months],
                        dtype=float)
        ax.bar(xs, vals, width=22, bottom=bottom, color=col, label=orbit)
        bottom += vals
    n_overpasses = sum(sum(c.values()) for c in monthly_by_orbit.values())
    ax.set_ylabel("overpasses", fontsize=8)
    ax.set_title(f"Data availability — {n_overpasses} overpasses in "
                f"{len(months)} months, by orbit", fontsize=9, loc="left")
    for ax in axes:
        _shade(ax, timeline)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc="best")
    fig.suptitle("LUMO tower — observation-channel time histories "
                 "(de-duplicated to distinct overpasses; shaded = DAM 3/4/6 campaigns)",
                 fontsize=11)
    fig.autofmt_xdate()
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    p = os.path.join(FIGURES_DIR, "fig_lumo_channels.png")
    fig.savefig(p, dpi=130)
    print(f"Wrote: {p}")
    return True


def report(out):
    n_raw, n_ded = out["n_bursts_raw"], out["n_overpasses"]
    L = ["# LUMO tower — Observation-Channel Time Histories (paper figure)", "",
         f"{n_ded} distinct Sentinel-1 overpasses (de-duplicated from {n_raw} "
         "artifact rows: each overpass appears once per polarisation and some "
         "under two burst ids ~3 s apart), 2020-08 .. 2021-07. DAM 3/4/6 "
         "campaigns shaded on every panel.", "",
         "| channel | n | median |", "|---------|--:|-------:|"]
    for k in ("brightness_ratio", "raw_phase_coherence", "gamma2", "modulation_depth"):
        s = out[k]
        med = f"{s['median']:.4g}" if s["median"] is not None else "-"
        L.append(f"| {k} | {s['n']} | {med} |")
    L += ["", "## Caveats", "",
          "* **Panel 5** is data availability, not a measurement channel: "
          "de-duplicated overpasses per month, stacked by orbit. It shows the "
          "revisit cadence the other four panels' monthly medians are drawn from.",
          "* Counts and medians are per **distinct overpass** (acquisition minute + "
          "orbit, VV preferred); the raw stacks hold "
          f"{n_raw} rows. The backend collapses the same pseudo-replicates via "
          "`insar_monitor::acquisition_key`, so these numbers match the DB.",
          "* Single-look SLC phase is backscatter/atmosphere dominated "
          "(circular sigma ~ 2 rad); phase coherence is near the clutter floor.",
          "* Amplitude/brightness does not separate the damage states "
          "(all Welch p > 0.10); vv is ~2.1x brighter than vh (pol confound).",
          "* Orbit stratification dominates the amplitude scale (ASC ~ 2x DESC).",
          "", "Build: `python3 fig_lumo_channels.py`.", ""]
    open(os.path.join(FIGURES_DIR, "fig_lumo_channels.md"), "w").write("\n".join(L))
    print("Wrote: fig_lumo_channels.md")


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", metavar="PATH", default=CHANNELS_CSV,
                    help="the flat per-record channel CSV written by "
                         "code/figures_data_csv.py (default: %(default)s)")
    args = ap.parse_args(argv)

    if not os.path.exists(args.csv):
        print(f"error: missing input CSV: {args.csv}\n"
              "  build it with: python3 code/figures_data_csv.py --site lumo",
              file=sys.stderr)
        return 2
    try:
        blocks = read_channels_csv(args.csv, site="lumo")
    except ValueError as exc:
        print(f"error: {exc}\n"
              "  rebuild it with: python3 code/figures_data_csv.py "
              "--site lumo", file=sys.stderr)
        return 2
    burst_rows, coh_rows = blocks["burst"], blocks["coherence"]
    if not burst_rows or not coh_rows:
        print(f"error: no lumo burst/coherence rows in {args.csv}",
              file=sys.stderr)
        return 2

    gamma2_ids = {x["burst_id"] for x in coh_rows}
    bright, phase, mod, preferred, n_raw, n_dedup, monthly, monthly_by_orbit = (
        load_bursts(gamma2_ids, burst_rows))
    gamma2 = load_gamma2(preferred, coh_rows)
    timeline = blocks["campaign"]          # the windows the panels shade
    out = stats(bright, phase, mod, gamma2, n_raw, n_dedup, monthly, monthly_by_orbit)
    json.dump(out, open(os.path.join(DATA_DIR, "fig_lumo_channels.json"), "w"), indent=2)
    print("Wrote: fig_lumo_channels.json")
    plot(bright, phase, mod, gamma2, timeline, monthly_by_orbit)
    report(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
