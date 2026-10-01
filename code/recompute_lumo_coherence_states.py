#!/usr/bin/env python3
"""Recompute `lumo_tower_coherence_states.json` from `data/lumo_channels.csv`.

This is a from-scratch port of `analyze_lumo_tower_coherence.py` (the original
generator, kept outside this repository), with one
change of input: the original reads the raw per-burst binary SLC cache
(`monthly_bursts/<month>/<key>/{meta.json,strip.bin}`) and computes `gamma2` /
`n_masked` / `peak_intensity` itself via `coherence_mast_echo()`. That cache is
not part of this repository. Instead, this script reads the *already computed*
per-burst `coherence` rows out of `data/lumo_channels.csv` — `figures_data_csv.py`
flattened them straight out of the committed `lumo_tower_coherence_states.json`,
so every field `per_state`/`tests`/`per_orbit`/`desc_tests`/`wind_control` need
(`damage_label`, `gamma2`, `n_masked`, `wind_speed_ms`, `orbit`, `date`) is
already a plain CSV column. The aggregate statistics below are therefore an
exact re-derivation, not an approximation: `_stats`/`_test`/the five analysis
blocks are ported near-verbatim from the original script.

Two top-level fields are metadata about the raw cache this repository does not
hold, and are carried as documented constants rather than recomputed:
  * `n_bursts_cached` (482) - the pre-deduplication burst-row count; the CSV
    only carries the 178 post-dedup rows, so the discarded pseudo-replicate
    count cannot be recovered from it.
  * `n_skipped` (0) - bursts the original script could not compute `gamma2` for
    (zero denominator, too few masked pixels); not observable once a burst
    already has no coherence row.
`params` (`peak_frac`, `median_mult`, `min_n_masked`) are the original script's
CLI defaults, also constants (they parameterise the raw-cache step this script
does not re-run).

Usage:
    python3 recompute_lumo_coherence_states.py [--csv data/lumo_channels.csv]
                                               [--out /tmp/recomputed.json]
    python3 recompute_lumo_coherence_states.py --check   # diff vs data/*.json
"""
import argparse
import json
import os
import sys

import numpy as np

try:
    from scipy import stats as sps
except ImportError:  # pragma: no cover - statistical tests are optional
    sps = None

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figures_data_csv import read_channels_csv  # noqa: E402

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
DEFAULT_CSV = os.path.join(DATA_DIR, "lumo_channels.csv")
DEFAULT_JSON = os.path.join(DATA_DIR, "lumo_tower_coherence_states.json")

STATE_ORDER = ["healthy", "DAM 3", "DAM 4", "DAM 6"]
STATE_SHORT = {"healthy": "DAM0(healthy)", "DAM 3": "DAM3",
               "DAM 4": "DAM4", "DAM 6": "DAM6"}

# Metadata about the raw binary cache this repository does not hold (see the
# module docstring) - documented constants, not recomputed.
PARAMS = {"peak_frac": 0.3, "median_mult": 5.0, "min_n_masked": 2}
N_BURSTS_CACHED = 482
N_SKIPPED = 0

METHOD = (
    "whole-tower coherence gamma2 = |sum(z)|^2/(N*sum(|z|^2)) over the masked "
    "mast echo (bright cluster: intensity >= peak_frac*peak AND >= "
    "median_mult*strip median). Single scatterer -> no segment split. One row "
    "per distinct overpass; pseudo-replicate burst rows (same overpass under "
    "several burst ids a few seconds apart / per polarisation) are collapsed "
    "first."
)


def _stats(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    a = np.asarray(vals, dtype=float)
    return {
        "n": int(len(vals)),
        "median": float(np.median(a)),
        "mean": float(np.mean(a)),
        "std": float(np.std(a, ddof=1)) if len(vals) > 1 else 0.0,
        "p5": float(np.percentile(a, 5)),
        "p25": float(np.percentile(a, 25)),
        "p75": float(np.percentile(a, 75)),
        "p95": float(np.percentile(a, 95)),
    }


def _test(a, b):
    """Welch + Mann-Whitney U on two numeric lists."""
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    out = {"n": [int(len(x)), int(len(y))],
           "median": [float(np.median(x)), float(np.median(y))]}
    if sps is not None and len(x) >= 3 and len(y) >= 3:
        t, p = sps.ttest_ind(x, y, equal_var=False)
        out["welch_t"] = float(t)
        out["welch_p"] = float(p)
        try:
            u, pu = sps.mannwhitneyu(x, y, alternative="two-sided")
            out["mannwhitney_u"] = float(u)
            out["mannwhitney_p"] = float(pu)
        except ValueError:
            pass
    return out


def recompute(csv_path):
    blocks = read_channels_csv(csv_path, site="lumo")
    bursts = [{
        "burst_id": r["burst_id"],
        "date": r["date"],
        "month": r["month"],
        "orbit": r["orbit"],
        "polarisation": r["polarisation"],
        "damage_label": r["damage_label"],
        "wind_speed_ms": r["wind_speed_ms"],
        "gamma2": r["gamma2"],
        "n_masked": r["n_masked"],
        "peak_intensity": r["peak_intensity"],
    } for r in blocks["coherence"]]
    bursts.sort(key=lambda b: (b["month"], b["date"], b["burst_id"]))

    out = {
        "method": METHOD,
        "params": dict(PARAMS),
        "n_bursts_cached": N_BURSTS_CACHED,
        "n_bursts_total": len(bursts),
        "n_skipped": N_SKIPPED,
        "bursts": bursts,
    }

    by_state = {lab: [b for b in bursts if b["damage_label"] == lab]
                for lab in STATE_ORDER}

    per_state = {}
    for lab, recs in by_state.items():
        row = {"n": len(recs)}
        if recs:
            g2 = [r["gamma2"] for r in recs]
            npx = [r["n_masked"] for r in recs]
            wind = [r["wind_speed_ms"] for r in recs if r["wind_speed_ms"] is not None]
            row["gamma2"] = _stats(g2)
            row["n_masked"] = _stats(npx)
            row["wind_speed_ms"] = _stats(wind)
            dates = sorted({r["date"] for r in recs if r.get("date")})
            row["date_range"] = [dates[0], dates[-1]] if dates else None
        per_state[STATE_SHORT[lab]] = row
    out["per_state"] = per_state

    tests = {}
    for other in ("DAM 3", "DAM 4", "DAM 6"):
        x = [r["gamma2"] for r in by_state["healthy"]]
        y = [r["gamma2"] for r in by_state[other]]
        if len(x) >= 3 and len(y) >= 3:
            tests[f"healthy_vs_{STATE_SHORT[other]}"] = _test(x, y)
    out["tests"] = tests

    orbit_stats = {}
    for orb in ("ASCENDING", "DESCENDING"):
        orbit_stats[orb] = {}
        for lab in STATE_ORDER:
            sub = [r["gamma2"] for r in by_state[lab] if r["orbit"] == orb]
            orbit_stats[orb][STATE_SHORT[lab]] = _stats(sub)
    out["per_orbit"] = orbit_stats

    desc_tests = {}
    for other in ("DAM 3", "DAM 4", "DAM 6"):
        x = [r["gamma2"] for r in by_state["healthy"] if r["orbit"] == "DESCENDING"]
        y = [r["gamma2"] for r in by_state[other] if r["orbit"] == "DESCENDING"]
        if len(x) >= 3 and len(y) >= 3:
            desc_tests[f"desc_healthy_vs_{STATE_SHORT[other]}"] = _test(x, y)
    out["desc_tests"] = desc_tests

    wind_control = {}
    if sps is not None:
        hw = [(r["wind_speed_ms"], r["gamma2"]) for r in by_state["healthy"]
              if r["wind_speed_ms"] is not None]
        if len(hw) >= 10:
            wx = np.asarray([p[0] for p in hw])
            gy = np.asarray([p[1] for p in hw])
            rho, p = sps.spearmanr(wx, gy)
            wind_control["healthy_gamma2_vs_wind"] = {
                "n": len(hw), "spearman_rho": float(rho), "p": float(p)}
        for other in ("DAM 3", "DAM 6"):
            x = [r["gamma2"] for r in by_state["healthy"]
                 if r["wind_speed_ms"] is not None and r["wind_speed_ms"] <= 4.0]
            y = [r["gamma2"] for r in by_state[other]
                 if r["wind_speed_ms"] is not None and r["wind_speed_ms"] <= 4.0]
            if len(x) >= 3 and len(y) >= 3:
                wind_control[f"windmatched_le4_healthy_vs_{STATE_SHORT[other]}"] = \
                    _test(x, y)
    out["wind_control"] = wind_control

    return out


def _check(rebuilt, original_path):
    """Compare the recomputed payload against the committed artifact field by
    field (order-independent for `bursts`, since the CSV's row order need not
    match the original cache-walk order)."""
    if not os.path.isfile(original_path):
        print(f"error: no reference JSON at {original_path} to check against "
              "(it was removed from data/ once lumo_damping_gamma2_modulation.py "
              "switched to reading this script's recompute() directly; restore "
              "it from git history if you need to re-verify)", file=sys.stderr)
        raise SystemExit(3)
    with open(original_path) as fh:
        original = json.load(fh)
    fails = []

    def eq(label, got, want):
        if got != want:
            fails.append(f"{label}: got {got!r}, want {want!r}")

    for key in ("method", "params", "n_bursts_cached", "n_bursts_total",
                "n_skipped", "per_state", "tests", "per_orbit", "desc_tests",
                "wind_control"):
        eq(key, rebuilt.get(key), original.get(key))
    by_id = {b["burst_id"]: b for b in original["bursts"]}
    eq("burst count", len(rebuilt["bursts"]), len(original["bursts"]))
    for b in rebuilt["bursts"]:
        want = by_id.get(b["burst_id"])
        if want != b:
            fails.append(f"bursts[{b['burst_id']}]: got {b!r}, want {want!r}")
    return fails


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default=DEFAULT_CSV)
    ap.add_argument("--out", default=None,
                    help="write the recomputed JSON here (default: do not write)")
    ap.add_argument("--check", action="store_true",
                    help=f"diff against {DEFAULT_JSON} instead of writing")
    args = ap.parse_args(argv)

    rebuilt = recompute(args.csv)

    if args.check:
        fails = _check(rebuilt, DEFAULT_JSON)
        if fails:
            for f in fails:
                print(f"  !! {f}", file=sys.stderr)
            print(f"  {len(fails)} check(s) FAILED", file=sys.stderr)
            return 4
        print("  recomputed JSON matches data/lumo_tower_coherence_states.json")
        return 0

    if args.out:
        with open(args.out, "w") as fh:
            json.dump(rebuilt, fh, indent=2)
        print(f"Wrote: {args.out}")
    else:
        json.dump(rebuilt, sys.stdout, indent=2)
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
