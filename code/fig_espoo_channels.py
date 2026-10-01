#!/usr/bin/env python3
"""Espoo Kurttila mast — observation-channel time histories (paper figure).

Five stacked time-history panels over the decoded Sentinel-1 burst record of the
2-year backfill (2024-09 .. ), by acquisition, with monthly medians per orbit:

  1) whole-target coherence gamma^2   (coherence_gamma2) + the LUMO reference
                                      medians (ASC 0.0617 / DESC 0.0261) for scale
  2) echo mask size                   (coherence_masked_pixels) — the clutter
                                      discriminator; a big mask with low gamma^2
                                      means the mask latched onto buildings
  3) tower phase coherence            (phase_coherence, adaptive sub-apertures)
  4) tower phase SNR                  (phase_snr_db) — 0 dB reference
  5) tower phase RMS activity         (phase_rms_rad)

What is populated here and why the rest is not (verified against the DB):

* **populated** — whole-target coherence, echo mask size and the tower-phase
  family (`phase_coherence`, `phase_snr_db`, `phase_rms_rad`).
* **empty by construction** — `sub_aperture_modulation` (the LUMO intra-dwell
  brightness modulation). The stored window is **7 x 7 pixels**, and
  `sub_aperture_modulation_for_samples` returns `None` whenever `height < n_sub`
  (`src/asset_onboarder/insar_monitor.rs:243`): 7 azimuth lines cannot be split
  into the 8 intra-dwell blocks of the LUMO definition. The LUMO modulation
  values come from the LUMO campaign ingest, not from this window table.
* **empty for this asset type** — `brightness_ratio`, `displacement_los_m`.
* **own figure, not part of this repository** — the dwell channel (Bessel
  brightness frequency vs the healthy FEM baseline, the measured/expected
  amplitudes and the LUMO-style view). For this Espoo mast, the phase-domain
  dwell frequency (`phase_detected_frequency_hz`) is empty too, because the
  channel is `phase_observable = false` (reason `low_coherence`) although the
  full-dwell phase capture ran: 150 rows in `onboarder.insar_phase_sequences`,
  11-column strip, 0.8 s dwell, 49 adaptive sub-apertures
  (`ceil(15.18 Hz * 0.8 s * 4) = 49`, i.e. driven by the FEM fundamental).

The HTTP measurement response deliberately omits some internal columns, so the
database-only values this figure would plot (`phase_snr_db`) and the dwell
family (`measured_frequency_hz`, `baseline_frequency_hz`, `frequency_drop_pct`,
`amplitude_measured_m`, `amplitude_expected_m`) stay empty in `data/*.json` and
in the CSV built from it: this project reads no database and no live API,
only `data/*.json` and `data/*.csv`.

This site has **no damage states**, so nothing is shaded. The mid-2025 Sentinel-1C
step-up in availability is marked instead — pass/orbit balance is not constant
across the window, which is a real confound for any trend reading.

Reads:  data/espoo_channels.csv — the flat per-record CSV that
        `code/figures_data_csv.py` builds from
        `data/espoo_mast_observability.json`. That CSV is the **only** input:
        this figure never calls the live API, never reads the JSON artifact
        directly and never queries a database.
Writes: ../data/fig_espoo_channels.json (or --json-dir),
        ../figures/fig_espoo_channels.md / .png (or --out-dir)

Usage:
    ../figures/fig_espoo_channels.sh                # data/espoo_channels.csv
    python3 fig_espoo_channels.py --no-lumo         # site-only (no comparison)
    python3 fig_espoo_channels.py --csv OTHER.csv   # an explicit CSV path

Exit codes: 0 wrote the figure; 2 the CSV is missing or is not a channel CSV.

`--no-lumo` drops the LUMO reference medians from panel 1 and the comparison
prose from the companion markdown, leaving a pure Espoo figure. The LUMO-
referenced reading stays in `ESPOO_MAST_OBSERVABILITY.md` either way.
"""
import argparse
import json
import os
import sys
from datetime import date, datetime

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    plt = None

# This script lives in `code/`: its input CSV comes from the sibling `data/`
# directory, the .json output also goes to `data/`, and the .md/.png go to the
# sibling `figures/` directory (the same `DATA_DIR` convention as
# `code/lumo_damping_frequencies.py`).
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.normpath(os.path.join(HERE, os.pardir, "data"))
FIGURES_DIR = os.path.normpath(os.path.join(HERE, os.pardir, "figures"))
CODE_DIR = os.path.normpath(os.path.join(HERE, os.pardir, "code"))
if CODE_DIR not in sys.path:         # also lets `python3 -m figures.<name>` work
    sys.path.insert(0, CODE_DIR)

from espoo_mast_observability import _num  # noqa: E402  (needs the path above)
from figures_data_csv import read_channels_csv  # noqa: E402  (same path)

CHANNELS_CSV = os.path.join(DATA_DIR, "espoo_channels.csv")

ORBIT_STYLE = {"ASCENDING": ("tab:blue", "o"), "DESCENDING": ("tab:orange", "s")}
LUMO_ASC, LUMO_DESC = 0.0617, 0.0261
MASK_CLUTTER_PX = 25
FIELDS = ("coherence_gamma2", "coherence_masked_pixels", "phase_coherence",
          "phase_snr_db", "phase_rms_rad")

# `phase_snr_db` is DB-enriched (onboarder.insar_measurements) and is never
# empty the way `phase_coherence`/`phase_rms_rad` are: when phase goes
# unobservable the backend still writes `snr_db_from_coherence(gamma)` with
# gamma clamped to a 1e-6 floor (`fusion::snr_db_from_coherence`), which is a
# fixed ~-60 dB value, not a measurement. Gate it on `phase_coherence` — the
# field that actually goes empty when the phase channel is unobservable — so
# the floor-clamped rows read as "no data" instead of a flat tail.
REQUIRES = {"phase_snr_db": "phase_coherence"}


def _d(iso):
    return date(int(iso[:4]), int(iso[5:7]), int(iso[8:10]))


def _valid(r, field):
    """The numeric value of `field` on row `r`, or `None` if the field is a
    clamp-floor artifact rather than a real measurement (see `REQUIRES`)."""
    gate = REQUIRES.get(field)
    if gate is not None and _num(r.get(gate)) is None:
        return None
    return _num(r.get(field))


def series(rows, field):
    """(date, value) pairs per orbit for one measurement field."""
    out = {"ASCENDING": [], "DESCENDING": []}
    for r in rows:
        v = _valid(r, field)
        if v is None:
            continue
        orbit = r.get("orbit_direction") or ""
        iso = str(r.get("acquisition_ts"))[:10]
        if orbit in out:
            out[orbit].append((iso, v))
    for k in out:
        out[k].sort()
    return out


def monthly_median(pairs):
    mm = {}
    for iso, v in pairs:
        mm.setdefault(iso[:7], []).append(v)
    months = sorted(mm)
    return ([date(int(m[:4]), int(m[5:7]), 15) for m in months],
            [float(np.median(mm[m])) for m in months])


def monthly_counts(rows):
    """Per-month acquisition counts, split by orbit — the availability panel.

    Every decoded row has an `acquisition_ts` and an `orbit_direction`, so this
    counts the whole record (not one channel), unlike the five channel panels
    above which only count rows where that one field is populated."""
    out = {}
    for r in rows:
        iso = str(r.get("acquisition_ts"))[:10]
        slot = out.setdefault(iso[:7], {"ASCENDING": 0, "DESCENDING": 0})
        orbit = r.get("orbit_direction") or ""
        if orbit in slot:
            slot[orbit] += 1
    return {m: out[m] for m in sorted(out)}


def stats(rows, show_lumo=True):
    def summ(field):
        vals = [v for v in (_valid(r, field) for r in rows) if v is not None]
        return {"n": len(vals), "median": float(np.median(vals)) if vals else None}
    out = {
        "n_acquisitions": len(rows),
        "date_first": min((str(r.get("acquisition_ts"))[:10] for r in rows), default=None),
        "date_last": max((str(r.get("acquisition_ts"))[:10] for r in rows), default=None),
        "lumo_reference": {"ascending_gamma2": LUMO_ASC, "descending_gamma2": LUMO_DESC},
        "monthly": monthly_counts(rows),
    }
    if not show_lumo:
        # Site-only figure: the payload carries no LUMO reference either, so the
        # JSON cannot disagree with the image.
        out.pop("lumo_reference", None)
    for field in FIELDS:
        out[field] = summ(field)
        for orbit in ("ASCENDING", "DESCENDING"):
            sub = [r for r in rows if r.get("orbit_direction") == orbit]
            out[f"{field}_{orbit.lower()}"] = summ_of(sub, field)
    return out


def summ_of(rows, field):
    vals = [v for v in (_valid(r, field) for r in rows) if v is not None]
    return {"n": len(vals), "median": float(np.median(vals)) if vals else None}


def plot(rows, out, source, show_lumo=True, out_dir=FIGURES_DIR):
    if plt is None:
        print("matplotlib unavailable — skipping the figure")
        return False
    fig, axes = plt.subplots(6, 1, figsize=(11.5, 14), sharex=True)
    channels = [
        ("coherence_gamma2", "Whole-target coherence γ²", "γ²", "lumo"),
        ("coherence_masked_pixels", "Echo mask size (clutter discriminator)", "px", "mask"),
        ("phase_coherence", "Tower phase coherence (adaptive sub-apertures)", "coherence", None),
        ("phase_snr_db", "Tower phase SNR", "dB", "snr"),
        ("phase_rms_rad", "Tower phase RMS activity", "rad", None),
    ]
    for ax, (field, title, ylab, extra) in zip(axes[:5], channels):
        s = series(rows, field)
        plotted = False
        for orbit, pairs in s.items():
            if not pairs:
                continue
            col, mk = ORBIT_STYLE[orbit]
            ax.plot([_d(i) for i, _ in pairs], [v for _, v in pairs], mk, ms=4,
                    color=col, alpha=0.75, label=f"{orbit} (n={len(pairs)})")
            mx, my = monthly_median(pairs)
            if len(mx) > 1:
                ax.plot(mx, my, "-", lw=1.2, color=col, alpha=0.55)
            plotted = True
        if extra == "lumo" and show_lumo:
            ax.axhline(LUMO_ASC, color="tab:blue", ls="--", lw=1,
                       label=f"LUMO ASC {LUMO_ASC}")
            ax.axhline(LUMO_DESC, color="tab:orange", ls="--", lw=1,
                       label=f"LUMO DESC {LUMO_DESC}")
        if extra == "mask":
            ax.axhline(MASK_CLUTTER_PX, color="red", ls="--", lw=1,
                       label=f"mask > {MASK_CLUTTER_PX} px = clutter-ish")
            ax.set_yscale("log")
        if extra == "snr":
            ax.axhline(0.0, color="red", ls="--", lw=1, label="0 dB (noise floor)")
        if not plotted:
            ax.text(0.5, 0.5, "no data in this channel", ha="center", va="center",
                    transform=ax.transAxes, fontsize=8, color="0.5")
        ax.set_title(title, fontsize=9, loc="left")
        ax.set_ylabel(ylab, fontsize=8)
        ax.grid(alpha=0.3)
        if plotted:
            ax.legend(fontsize=6.5, ncol=2, loc="best")

    # 6) data availability by month, stacked by orbit
    ax = axes[5]
    months = sorted(out["monthly"])
    xs = [date(int(m[:4]), int(m[5:7]), 15) for m in months]
    bottom = np.zeros(len(months))
    for orbit in ("ASCENDING", "DESCENDING"):
        col, _ = ORBIT_STYLE[orbit]
        vals = np.array([out["monthly"][m][orbit] for m in months], dtype=float)
        ax.bar(xs, vals, width=22, bottom=bottom, color=col, label=orbit)
        bottom += vals
    ax.set_ylabel("acquisitions", fontsize=8)
    ax.set_title(f"Data availability — {out['n_acquisitions']} acquisitions in "
                f"{len(months)} months, by orbit", fontsize=9, loc="left")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=6.5, ncol=2, loc="best")

    # S1C step-up marker (6-day repeat restored / extra coverage from mid-2025)
    for ax in axes:
        ax.axvline(date(2025, 4, 1), color="green", ls=":", lw=1, alpha=0.7)
    axes[0].annotate("Sentinel-1C coverage step-up (approx.)", xy=(date(2025, 4, 1), 1),
                     xycoords=("data", "axes fraction"), xytext=(6, -14),
                     textcoords="offset points", fontsize=7, color="green")

    n = out["n_acquisitions"]
    fig.suptitle(
        f"Espoo Kurttila mast — observation-channel time histories "
        f"({out['date_first']} .. {out['date_last']}, n={n} decoded acquisitions)\n"
        "no damage states exist for this site — observability only",
        fontsize=10)
    fig.autofmt_xdate()
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    png = os.path.join(out_dir, "fig_espoo_channels.png")
    fig.savefig(png, dpi=130)
    plt.close(fig)
    print(f"Wrote: {png}  (source: {source})")
    return True


def report(out, source, show_lumo=True, out_dir=FIGURES_DIR):
    """Companion markdown for the figure (LUMO convention)."""
    L = ["# Espoo Kurttila mast — Observation-Channel Time Histories", "",
         f"{out['n_acquisitions']} decoded acquisitions, "
         f"{out['date_first']} .. {out['date_last']}. No damage states exist for "
         "this site, so no campaign windows are shaded; the mid-2025 S1C "
         "availability step-up is marked instead.", "",
         f"Source: {source}", "",
         "| channel | n | median | DESCENDING (median) | ASCENDING (median) |",
         "|---------|--:|-------:|--------------------:|-------------------:|"]
    for field in FIELDS:
        s = out[field]
        med = f"{s['median']:.4g}" if s["median"] is not None else "-"
        d = out.get(f"{field}_descending") or {}
        a = out.get(f"{field}_ascending") or {}
        dm = f"{d['median']:.4g}" if d.get("median") is not None else "-"
        am = f"{a['median']:.4g}" if a.get("median") is not None else "-"
        L.append(f"| {field} | {s['n']} | {med} | {dm} | {am} |")
    reading = []
    if show_lumo:
        reading.append(
            "* **Panel 1 vs the dashed lines** is the scale check: the LUMO reference "
            f"medians are {LUMO_ASC} (ASC) and {LUMO_DESC} (DESC). Espoo ASC sits well "
            "above its reference; Espoo DESC sits below its reference.")
    else:
        reading.append(
            "* **Panel 1** is the whole-target coherence γ² per acquisition. This figure "
            "is deliberately **site-only**: the LUMO reference comparison is not drawn. "
            "The LUMO-referenced reading "
            f"(ASC {LUMO_ASC} / DESC {LUMO_DESC}) lives in `ESPOO_MAST_OBSERVABILITY.md`.")
    reading += [
        "* **Panel 2** carries the diagnostic weight. Low γ² with a large mask is "
        "clutter; high γ² with a small mask is a real point scatterer. Do not read "
        "panel 1 without panel 2.",
        "* **Panels 3–5** are the tower phase family. Negative SNR with phase RMS "
        "near π (≈3.14 rad = fully decorrelated) is the signature of a "
        "non-observable phase channel — which is what the pipeline concluded here "
        "(`phase_observable = false`, reason `low_coherence`).",
        "* **Panel 4 stops where panels 3 and 5 stop.** `phase_snr_db` is DB-enriched "
        "and is never empty the way `phase_coherence`/`phase_rms_rad` are: once the "
        "phase channel goes unobservable the backend still writes "
        "`snr_db_from_coherence(gamma)` with gamma clamped to a 1e-6 floor "
        "(`fusion::snr_db_from_coherence`), a fixed ≈-60 dB value, not a "
        "measurement. This figure drops those floor-clamped rows (gated on "
        "`phase_coherence` being populated) instead of drawing a flat tail.",
        "* **Panel 6** is data availability, not a measurement channel: decoded "
        "acquisitions per month, stacked by orbit. It is what panel 1's reading "
        "depends on — the pass/orbit balance it shows is the same step-up marked "
        "on every panel.",
    ]
    L += ["", "## Reading the figure", ""] + reading + [
        "", "## Caveats", "",
        "* No ground-truth state label → observability evidence only, never a "
        "condition assessment.",
        "* 37 OSM building ways lie within 100 m of the mast — the built-up "
        "Kurttila/Saunalahti surroundings, not open ground.",
        "* Pass/orbit balance changes mid-2025 (S1C); the step-up is drawn as a "
        "guide line on every panel, and it is a real confound for any trend "
        "reading.",
        "* The record is still growing while the 2-year backfill runs; re-run to "
        "refresh.",
        "* `phase_snr_db` rows where the phase channel is unobservable (empty "
        "`phase_coherence`) are excluded from panel 4 and its stats — they are a "
        "fixed clamp-floor value (≈-60 dB), not a varying measurement.",
        "* **The LUMO intra-dwell brightness modulation "
        "(`sub_aperture_modulation`) is absent for this site by construction**: the "
        "stored window is 7 x 7 px and the helper returns `None` for "
        "`height < n_sub` (`src/asset_onboarder/insar_monitor.rs:243`), so 8 "
        "intra-dwell blocks cannot be formed. Reproducing the LUMO dwell-modulation "
        "channel here would need a full-dwell strip capture (11 columns, ~0.8 s), "
        "not a re-run of the window backfill.", "",
        "Build: `python3 fig_espoo_channels.py"
        + ("" if show_lumo else " --no-lumo") + "`.", ""]
    md = os.path.join(out_dir, "fig_espoo_channels.md")
    with open(md, "w") as fh:
        fh.write("\n".join(L))
    print("Wrote: fig_espoo_channels.md")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", metavar="PATH", default=CHANNELS_CSV,
                    help="the flat per-record channel CSV written by "
                         "code/figures_data_csv.py (default: %(default)s)")
    ap.add_argument("--no-lumo", action="store_true",
                    help="omit the LUMO reference lines and comparison (site-only figure)")
    ap.add_argument("--out-dir", default=FIGURES_DIR, metavar="DIR",
                    help="directory for the .md/.png (default: %(default)s)")
    ap.add_argument("--json-dir", default=DATA_DIR, metavar="DIR",
                    help="directory for the .json (default: %(default)s)")
    args = ap.parse_args()
    show_lumo = not args.no_lumo

    if not os.path.exists(args.csv):
        print(f"error: missing input CSV: {args.csv}\n"
              "  build it with: python3 code/figures_data_csv.py "
              "--site espoo", file=sys.stderr)
        return 2
    try:
        rows = read_channels_csv(args.csv, site="espoo")["measurement"]
    except ValueError as exc:
        print(f"error: {exc}\n"
              "  rebuild it with: python3 code/figures_data_csv.py "
              "--site espoo", file=sys.stderr)
        return 2
    if not rows:
        print(f"error: no espoo measurement rows in {args.csv}", file=sys.stderr)
        return 2
    source = os.path.basename(args.csv)
    out = stats(rows, show_lumo)
    with open(os.path.join(args.json_dir, "fig_espoo_channels.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print("Wrote: fig_espoo_channels.json")
    plot(rows, out, source, show_lumo, args.out_dir)
    report(out, source, show_lumo, args.out_dir)
    print(f"n={out['n_acquisitions']}  γ² ASC median="
          f"{out['coherence_gamma2_ascending']['median']}  DESC median="
          f"{out['coherence_gamma2_descending']['median']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())