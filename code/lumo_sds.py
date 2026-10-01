#!/usr/bin/env python3
"""LUMO - damage-detection scores (SDS): modal (SHM) channels + gamma2 coherence.

Computes the four 0-10 discrimination scores the observability study for the
LUMO lattice tower derives from these channels:

    LUMO_H1_FREQUENCY   10.0        (h1 peak frequency,   n = 30 recordings)
    LUMO_H1_DAMPING      9.7333333  (h1 half-power zeta,  n = 30 recordings)
    LUMO_H2_DAMPING      3.2        (h2 half-power zeta,  n = 20 recordings)
    LUMO_GAMMA2_COHERENCE 1.3917355 (whole-tower InSAR gamma2, max pairwise)

The first three are the SHM modal channels, below. The gamma2 coherence channel
(Table 4's "Whole-Tower gamma2 Coherence", SDS = 1.4; Section 4.1's "the maximum
pairwise SDS was 1.39 for Healthy versus DAM3, whereas the pooled
Healthy-versus-all-damaged comparison yielded SDS = 0.10") is documented in its
own section further down, since it uses a different aggregate (max across
comparisons against one pooled healthy population, not a per-campaign mean) and a
different input (`data/lumo_channels.csv` instead of the modal JSON).

THE SCORE (SDS, "state-discrimination score")

    SDS = 10 * mean over campaigns of |delta_campaign|,  clamped to 0 ... 10

where `delta_campaign` is Cliff's delta between the healthy and the damaged
recordings of that one campaign. The estimate is paired *within* a campaign
because the three healthy sets come from three different field campaigns and are
not interchangeable.

CLIFF'S DELTA (no scipy, no numpy - plain Python)

    delta(a, b) = (#{(x, y) : y > x} - #{(x, y) : y < x}) / (n_a * n_b),  x in a, y in b

with a = the healthy values and b = the damaged ones, so a *drop* from healthy to
damaged gives a negative delta. The comparisons are strict: a tie contributes to
neither count and therefore pulls the delta towards 0. With 5 vs 5 recordings
(25 pairs) delta can only take multiples of 0.04, and |delta| = 1.0 requires
complete separation.

MODAL EXCLUSIONS

    ("dam3", "h2") is excluded from every channel.

The h2 damping of the DAM3 campaign is not admissible evidence at the estimator's
resolution: the healthy half-power width sits only ~1.25 x above the
resolvability threshold (zeta 0.0037 vs 3 x floor = 0.0029) while the damaged
width is ~14 x larger (zeta 0.0496), so that campaign's near-complete separation
(delta = -0.92) measures the resolution floor, not the structure. The pair is
dropped from *both* sides, so LUMO_H2_DAMPING is scored on 20 recordings instead
of 30 - and reads 3.2 instead of 5.4667.

INPUT: `../data/lumo_damping_frequencies.json`, the output of the sibling script
`lumo_damping_frequencies.py` (schema `lumo_damping_frequencies/v1`), which
supplies, per recording, `f_peak_hz` and `zeta` for the fundamental, h1 and h2.

SELF-CONTAINED: this script uses the Python 3 standard library only. It imports
no other project code, reads no database, performs no network access and needs no
Rust. The input JSON is the only file it reads; delete `lumo_sds_expected.json`
and the numbers below do not move - that file is a pinned transcription used by
`--verify`, never an input to the computation.

GAMMA2 COHERENCE (same `cliffs_delta`, a different aggregate and input)

INPUT: `../data/lumo_channels.csv`'s `coherence` rows, read via
`recompute_lumo_coherence_states.recompute()` - one row per overpass
(burst-id/polarisation pseudo-replicates already collapsed), labelled healthy /
DAM3 / DAM4 / DAM6. Unlike the SHM channels above, gamma2 has one pooled healthy
population to compare against each damage state, not one healthy baseline per
campaign, so:

    SDS_gamma2 = 10 * max(|delta_DAM3|, |delta_DAM4|, |delta_DAM6|)

with the pooled Healthy-versus-all-damaged score (b = DAM3 + DAM4 + DAM6 pooled)
reported alongside it, not folded into the max.

USAGE
  python3 lumo_sds.py                       # both tables on stdout
  python3 lumo_sds.py --explain             # also show the modal exclusion ON/OFF pair
  python3 lumo_sds.py --json lumo_sds.json  # write the combined JSON block to a file
  python3 lumo_sds.py --verify              # compare both channels with the pinned values
  python3 lumo_sds.py --selftest            # check the metrics, needs no data

Exit codes: 0 ok, 1 unusable --expected, 2 input JSON/CSV missing or
incompatible, 3 verification mismatch, 4 selftest failure (argparse exits 2 on
bad usage).
"""
import argparse
import datetime
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
# The scripts live in `code/`, the committed data (the JSON/CSV this script
# consumes and the pinned references it verifies against) in the sibling
# `data/` directory.
DATA_DIR = os.path.normpath(os.path.join(HERE, os.pardir, "data"))
DEFAULT_CSV = os.path.join(DATA_DIR, "lumo_channels.csv")

sys.path.insert(0, HERE)
from recompute_lumo_coherence_states import recompute, STATE_ORDER, STATE_SHORT  # noqa: E402

# --- The score: constants of the 0-10 discrimination scale -------------------
SDS_SCALE = 10.0
SDS_MAX = 10.0
GAMMA2_DAMAGE_STATES = ("DAM 3", "DAM 4", "DAM 6")
BANDS = ((2.0, "very weak"), (4.0, "weak"), (6.0, "moderate"), (8.0, "strong"),
         (float("inf"), "very strong"))

# (campaign, mode) blocks that are not admissible evidence for any channel.
# The healthy DAM3 h2 width sits only ~1.25 x above the resolvability threshold
# (3 x the one-bin floor) while the damaged one is ~14 x larger, so that
# campaign's near-complete separation measures the floor, not the structure.
MODAL_EXCLUSIONS = (("dam3", "h2"),)

# channel -> (mode key in the input JSON, per-recording field)
CHANNELS = (
    ("LUMO_H1_FREQUENCY", "h1", "f_peak_hz"),
    ("LUMO_H1_DAMPING", "h1", "zeta"),
    ("LUMO_H2_DAMPING", "h2", "zeta"),
)
# Reported for context only: the fundamental is not an evidence channel.
INFORMATIONAL = ("fundamental", "fundamental", "f_peak_hz")


def cliffs_delta(a, b):
    """Cliff's delta between the healthy values `a` and the damaged values `b`.

    Strict comparisons (ties count for neither side) over the full n_a * n_b
    pair set. Input order does not matter; the sign does - a decrease from
    healthy to damaged is negative.
    """
    if not a or not b:
        return None
    greater = less = 0
    for x in a:
        for y in b:
            if y > x:
                greater += 1
            elif y < x:
                less += 1
    return (greater - less) / float(len(a) * len(b))


def band_label(sds):
    """Verbal band of a 0-10 score."""
    if sds is None:
        return None
    for edge, label in BANDS:
        if sds < edge:
            return label
    return BANDS[-1][1]


# ----------------------------------------------------------------------------
# Input
# ----------------------------------------------------------------------------
def sds_of_deltas(deltas):
    """The score itself: 10 x mean |delta| over campaigns, clamped to 0 ... 10."""
    if not deltas:
        return None
    return min(SDS_MAX, SDS_SCALE * sum(abs(d) for d in deltas) / float(len(deltas)))
def load_source(path):
    """Read and sanity-check the input JSON. Returns (document, problem)."""
    if not os.path.isfile(path):
        return None, (f"input JSON not found: {path}\n"
                      "  run first: python3 lumo_damping_frequencies.py "
                      "--root <dataset root>")
    try:
        with open(path) as fh:
            doc = json.load(fh)
    except (OSError, ValueError) as exc:
        return None, f"input JSON unreadable: {path}: {exc}"
    if doc.get("schema") != "lumo_damping_frequencies/v1":
        return None, (f"input JSON has schema {doc.get('schema')!r}, expected "
                      "'lumo_damping_frequencies/v1'")
    if not doc.get("campaigns") or not doc.get("states"):
        return None, "input JSON has no `campaigns`/`states` block"
    zeta_modes = tuple(sorted({mode for _, mode, field in CHANNELS
                               if field == "zeta"}))
    found = any("zeta" in (row.get(mode) or {})
                for entry in doc["states"].values()
                for row in entry.get("recordings", [])
                for mode in zeta_modes)
    if not found:
        return None, ("input JSON carries no `zeta` fields - it predates the "
                      "damping output.\n  re-run: python3 "
                      "lumo_damping_frequencies.py --root <dataset root>")
    return doc, None


def values_of(entry, mode, field):
    """Every non-null value of `field` under `entry['recordings'][*][mode]`."""
    out = []
    for row in (entry or {}).get("recordings", []):
        v = (row.get(mode) or {}).get(field)
        if v is not None:
            out.append(float(v))
    return out


def channel_stats(doc, mode, field, exclusions=MODAL_EXCLUSIONS):
    """Per-campaign cliff's deltas and the resulting SDS for one channel."""
    per_campaign, raw_deltas = {}, []
    n_healthy = n_damaged = 0
    states = doc.get("states") or {}
    for name in sorted(doc.get("campaigns") or {}):
        blk = doc["campaigns"][name]
        if (name, mode) in exclusions:
            per_campaign[name] = {"delta": None, "n_healthy": 0, "n_damaged": 0,
                                  "excluded": True}
            continue
        healthy = values_of(states.get(blk.get("healthy")), mode, field)
        damaged = values_of(states.get(blk.get("damaged")), mode, field)
        n_healthy += len(healthy)
        n_damaged += len(damaged)
        delta = cliffs_delta(healthy, damaged)
        if delta is not None:
            raw_deltas.append(delta)
        per_campaign[name] = {
            "delta": None if delta is None else round(delta, 9),
            "n_healthy": len(healthy), "n_damaged": len(damaged), "excluded": False}
    sds = sds_of_deltas(raw_deltas)
    return {
        "mode": mode,
        "field": field,
        "n_samples": n_healthy + n_damaged,
        "n_healthy": n_healthy,
        "n_damaged": n_damaged,
        "n_campaigns": len(raw_deltas),
        "sds": sds,
        "band": band_label(sds),
        "per_campaign": per_campaign,
        "excluded_pairs": [list(p) for p in exclusions if p[1] == mode],
    }


def score(doc, exclusions=MODAL_EXCLUSIONS):
    """All channels + the informational fundamental row."""
    channels = {name: channel_stats(doc, mode, field, exclusions)
                for name, mode, field in CHANNELS}
    info = channel_stats(doc, INFORMATIONAL[1], INFORMATIONAL[2], exclusions)
    del info["excluded_pairs"]
    return {"channels": channels, "informational": {INFORMATIONAL[0]: info}}


# ----------------------------------------------------------------------------
# Gamma2 coherence (max-based aggregate, pooled healthy population)
# ----------------------------------------------------------------------------
def _gamma2_by_state(csv_path):
    out = recompute(csv_path)
    bursts = out["bursts"]
    by_state = {lab: [b for b in bursts if b["damage_label"] == lab]
                for lab in STATE_ORDER}
    return {lab: [r["gamma2"] for r in recs] for lab, recs in by_state.items()}


def score_gamma2(csv_path=DEFAULT_CSV):
    """Per-state deltas/SDS, the reported max-pairwise SDS, and the pooled SDS."""
    g2 = _gamma2_by_state(csv_path)
    healthy = g2["healthy"]

    per_state = {}
    for lab in GAMMA2_DAMAGE_STATES:
        damaged = g2[lab]
        delta = cliffs_delta(healthy, damaged)
        per_state[STATE_SHORT[lab]] = {
            "n_healthy": len(healthy), "n_damaged": len(damaged),
            "delta": None if delta is None else round(delta, 9),
            "sds": None if delta is None else round(SDS_SCALE * abs(delta), 9),
        }

    valid = [row for row in per_state.values() if row["delta"] is not None]
    best = max(valid, key=lambda row: abs(row["delta"])) if valid else None
    max_pairwise_sds = best["sds"] if best else None
    max_pairwise_state = next((name for name, row in per_state.items()
                               if row is best), None)

    pooled_damaged = [v for lab in GAMMA2_DAMAGE_STATES for v in g2[lab]]
    pooled_delta = cliffs_delta(healthy, pooled_damaged)
    pooled_sds = None if pooled_delta is None else round(SDS_SCALE * abs(pooled_delta), 9)

    return {
        "n_healthy": len(healthy),
        "per_state": per_state,
        "max_pairwise_sds": max_pairwise_sds,
        "max_pairwise_state": max_pairwise_state,
        "pooled": {
            "n_damaged": len(pooled_damaged),
            "delta": None if pooled_delta is None else round(pooled_delta, 9),
            "sds": pooled_sds,
        },
    }


def render_gamma2_table(result):
    out = ["", "gamma2 coherence (whole tower, data/lumo_channels.csv):",
          f"{'state':<14} {'n_healthy':>9} {'n_damaged':>9} {'delta':>10} {'sds':>8}"]
    for name, row in result["per_state"].items():
        out.append(f"{name:<14} {row['n_healthy']:>9} {row['n_damaged']:>9} "
                   f"{row['delta']:>10.4f} {row['sds']:>8.4f}")
    out.append("")
    out.append(f"max pairwise SDS = {result['max_pairwise_sds']:.4f} "
               f"(Healthy vs {result['max_pairwise_state']})")
    pooled = result["pooled"]
    out.append(f"pooled Healthy-vs-all-damaged SDS = {pooled['sds']:.4f} "
               f"(n_damaged={pooled['n_damaged']})")
    return "\n".join(out) + "\n"


# ----------------------------------------------------------------------------
# Rendering
# ----------------------------------------------------------------------------
def _per_campaign_cell(stat):
    parts = []
    for name in sorted(stat["per_campaign"]):
        blk = stat["per_campaign"][name]
        if blk["excluded"]:
            parts.append(f"{name} excluded")
        elif blk["delta"] is None:
            parts.append(f"{name} n/a")
        else:
            parts.append(f"{name} {blk['delta']:+.4f}")
    return "  ".join(parts)


def render_table(result):
    """The reviewer-facing table: one line per channel and the fundamental."""
    out = [f"{'channel':<20} {'mode':<5} {'value':<10} {'n':>3}  "
           f"{'delta per campaign':<44} {'excluded':<9} {'SDS':>14}"]
    rows = [(name, result["channels"][name], "") for name, _, _ in CHANNELS
            if name in result["channels"]]
    for name in sorted(result.get("informational", {})):
        rows.append(("(informational)", result["informational"][name],
                     "  no evidence channel"))
    for name, stat, note in rows:
        excluded = ", ".join("/".join(p) for p in stat.get("excluded_pairs", []))
        out.append(f"{name:<20} {stat['mode']:<5} {stat['field']:<10} "
                   f"{stat['n_samples']:>3}  {_per_campaign_cell(stat):<44} "
                   f"{excluded:<9} {stat['sds']:>14.10f}{note}")
    return "\n".join(out) + "\n"


def render_explain(result, doc):
    """The exclusion ON/OFF pair - the one rule a reader would otherwise miss."""
    out = []
    for name, mode, field in CHANNELS:
        on = result["channels"][name]
        off = channel_stats(doc, mode, field, exclusions=())
        if on["excluded_pairs"]:
            out.append(f"{name} ({mode} {field}): exclusion ON  SDS "
                       f"{on['sds']:.4f} (n={on['n_samples']})  |  OFF SDS "
                       f"{off['sds']:.4f} (n={off['n_samples']})")
        else:
            out.append(f"{name} ({mode} {field}): no exclusion - SDS "
                       f"{on['sds']:.4f} (n={on['n_samples']})")
    return "\n".join(out) + "\n"


# ----------------------------------------------------------------------------
# Selftest - the metric, with no data and no reference file
# ----------------------------------------------------------------------------
def _synthetic(pairs):
    """Minimal input document with the real schema: 5 healthy + 5 damaged rows
    per campaign, all `h2` zeta values equal within one side of a campaign.

    `pairs` maps campaign -> (healthy zeta, damaged zeta). Used by --selftest
    only; never written to disk.
    """
    dirs = {"dam6": ("01_Healthy", "02_DAM6_111"),
            "dam4": ("03_Healthy", "04_DAM4_111"),
            "dam3": ("05_Healthy", "06_DAM3_111")}
    doc = {"schema": "lumo_damping_frequencies/v1", "states": {}, "campaigns": {}}
    for name, (healthy, damaged) in sorted(pairs.items()):
        hdir, ddir = dirs[name]
        doc["states"][hdir] = {"recordings": [{"h2": {"zeta": healthy}}
                                              for _ in range(5)]}
        doc["states"][ddir] = {"recordings": [{"h2": {"zeta": damaged}}
                                              for _ in range(5)]}
        doc["campaigns"][name] = {"healthy": hdir, "damaged": ddir, "modes": {}}
    return doc


def _selftest(quiet=False):
    """Check the metric on hand-verifiable fixtures. Returns the list of failures."""
    fails = []

    def num(label, got, want, tol=1e-9):
        ok = got is not None and abs(got - want) <= tol
        if not ok:
            fails.append(f"{label}: got {got!r}, want {want!r}")
        if not quiet:
            print(f"  {'ok  ' if ok else 'FAIL'} {label}: {got!r}")

    def eq(label, got, want):
        ok = got == want
        if not ok:
            fails.append(f"{label}: got {got!r}, want {want!r}")
        if not quiet:
            print(f"  {'ok  ' if ok else 'FAIL'} {label}: {got!r}")

    # 1. Cliff's delta: complete separation, both signs, ties, 5v5 quantisation.
    num("cliffs_delta separation (healthy below damaged)", cliffs_delta([1, 2, 3], [4, 5, 6]), 1.0)
    num("cliffs_delta separation (healthy above damaged)", cliffs_delta([4, 5, 6], [1, 2, 3]), -1.0)
    num("cliffs_delta all pairs tied", cliffs_delta([2, 2], [2, 2]), 0.0)
    num("cliffs_delta mixed, one pair tied", cliffs_delta([0, 1], [1, 2]), 0.75)
    num("cliffs_delta 5v5: one net pair = 0.04",
        cliffs_delta([1, 1, 1, 1, 3], [1, 1, 1, 1, 2]), -0.04)
    num("cliffs_delta 5v5: one recording above all = 0.20",
        cliffs_delta([1] * 5, [1, 1, 1, 1, 2]), 0.2)

    # 2. The mean -> score mapping, the clamp and the bands.
    num("SDS of three complete-separation campaigns", sds_of_deltas([-1.0, -1.0, -1.0]), 10.0)
    num("SDS of the h1 damping deltas  -1, -1, -0.92", sds_of_deltas([-1.0, -1.0, -0.92]),
        9.733333333333333)
    num("SDS of the h2 damping deltas  +0.20, -0.44", sds_of_deltas([0.2, -0.44]), 3.2)
    num("SDS is clamped at the scale maximum", sds_of_deltas([1.0, 1.0]), 10.0)
    eq("band of 10.0", band_label(10.0), "very strong")
    eq("band of 3.2", band_label(3.2), "weak")
    eq("band of 1.0", band_label(1.0), "very weak")

    # 3. Exclusion semantics: dam3/h2 is dropped from both sides, so it shrinks
    #    n and the campaign count, and it changes the score.
    syn = _synthetic({"dam6": (0.005, 0.010),   # zeta rises  -> delta +1
                      "dam4": (0.0075, 0.0075),  # all ties    -> delta  0
                      "dam3": (0.005, 0.010)})   # the excluded pair
    with_ex = channel_stats(syn, "h2", "zeta", MODAL_EXCLUSIONS)
    no_ex = channel_stats(syn, "h2", "zeta", ())
    num("exclusion ON: n_samples", with_ex["n_samples"], 20)
    num("exclusion OFF: n_samples", no_ex["n_samples"], 30)
    num("exclusion ON: campaigns scored", with_ex["n_campaigns"], 2)
    num("exclusion OFF: campaigns scored", no_ex["n_campaigns"], 3)
    num("exclusion ON: SDS = mean(|+1|, |0|) * 10", with_ex["sds"], 5.0)
    num("exclusion OFF: SDS = mean(|+1|, |0|, |+1|) * 10", no_ex["sds"], 6.666666666666667)
    eq("exclusion lists the dam3/h2 pair", with_ex["excluded_pairs"], [["dam3", "h2"]])
    eq("nothing is excluded without the rule", no_ex["excluded_pairs"], [])
    return fails


# ----------------------------------------------------------------------------
# Verification against the pinned reference (never an input to the score)
# ----------------------------------------------------------------------------
def verify(result, expected, path, tol=1e-9):
    """Compare the computed scores with the pinned transcription.

    The reference holds values from the external evidence-channel store for this
    asset, transcribed here by hand on 2026-09-25. It is read and nowhere else and
    never feeds the computation: delete it and the scores above do not move, so a
    mismatch means this pipeline and the store disagree, not that a value was
    borrowed.
    """
    ref = expected.get("channels") or {}
    if not ref:
        print("  !! the reference file has no `channels` block", file=sys.stderr)
        return None
    bad, checked = [], 0
    for name in sorted(ref):
        got, want = result["channels"].get(name), ref[name]
        if got is None:
            print(f"  MISSING  {name} (not computed here)")
            bad.append(name)
            continue
        checked += 1
        problems = []
        if abs(got["sds"] - float(want["sds"])) > tol:
            problems.append(f"sds {got['sds']:.10f} vs {float(want['sds']):.10f}")
        if want.get("n_samples") is not None and got["n_samples"] != want["n_samples"]:
            problems.append(f"n_samples {got['n_samples']} vs {want['n_samples']}")
        for campaign in sorted(want.get("per_campaign") or {}):
            g = (got["per_campaign"].get(campaign) or {}).get("delta")
            w = (want["per_campaign"][campaign] or {}).get("delta")
            if g is None or w is None:
                if g != w:
                    problems.append(f"{campaign} delta {g!r} vs {w!r}")
            elif abs(g - float(w)) > tol:
                problems.append(f"{campaign} delta {g:+.4f} vs {float(w):+.4f}")
        if problems:
            bad.append(name)
        print(f"  {'MATCH   ' if not problems else 'MISMATCH'} {name:<20} "
              f"sds={got['sds']:>14.10f}  n={got['n_samples']:<3}  "
              + "; ".join(problems))
    print(f"  {checked - len(bad)}/{checked} channels match "
          f"{os.path.basename(path)} (tol {tol:g})")
    for name in bad:
        print(f"  !! {name} does not match the pinned values", file=sys.stderr)
    return not bad


def verify_gamma2(result, expected, path, tol=1e-6):
    """Compare the computed gamma2 scores with the pinned `gamma2` block."""
    ref = expected.get("gamma2") or {}
    if not ref:
        print("  !! the reference file has no `gamma2` block", file=sys.stderr)
        return None
    bad = []

    def check(label, got, want):
        ok = got is not None and want is not None and abs(got - want) <= tol
        if not ok:
            bad.append(label)
        print(f"  {'MATCH   ' if ok else 'MISMATCH'} {label}: got {got!r}, "
              f"want {want!r}")

    for name, row in (ref.get("per_state") or {}).items():
        got = result["per_state"].get(name, {})
        check(f"gamma2 {name} delta", got.get("delta"), row.get("delta"))
        check(f"gamma2 {name} sds", got.get("sds"), row.get("sds"))
    check("gamma2 max_pairwise_sds", result["max_pairwise_sds"],
          ref.get("max_pairwise_sds"))
    check("gamma2 pooled sds", result["pooled"]["sds"],
          (ref.get("pooled") or {}).get("sds"))
    print(f"  {len(bad)} mismatch(es) in gamma2 ({os.path.basename(path)})")
    return not bad


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog=os.path.basename(__file__),
        description="Compute the LUMO state-discrimination scores (SDS): the modal "
                    "SHM channels and the whole-tower gamma2 coherence channel.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="exit codes: 0 ok, 1 unusable --expected, 2 input JSON/CSV missing "
               "or incompatible, 3 verification mismatch, 4 selftest failure")
    ap.add_argument("--from", dest="source", metavar="FILE",
                    default=os.path.join(DATA_DIR, "lumo_damping_frequencies.json"),
                    help="input JSON written by lumo_damping_frequencies.py "
                         "(default: %(default)s)")
    ap.add_argument("--csv", metavar="FILE", default=DEFAULT_CSV,
                    help="input CSV for the gamma2 coherence channel "
                         "(default: %(default)s)")
    ap.add_argument("--expected", metavar="FILE",
                    help="pinned reference for --verify, covering both the modal "
                         "and the gamma2 channels (default: "
                         "../data/lumo_sds_expected.json); optional")
    ap.add_argument("--json", dest="json_out", metavar="FILE",
                    help="also write the score block to FILE")
    ap.add_argument("--explain", action="store_true",
                    help="print the exclusion ON/OFF pair for every modal channel")
    ap.add_argument("--verify", action="store_true",
                    help="compare the scores with --expected")
    ap.add_argument("--selftest", action="store_true",
                    help="check the metric on fixtures - needs no data and no "
                         "reference file")
    ap.add_argument("--date",
                    default=datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d"),
                    help="value for the `generated` field (default: today, UTC)")
    args = ap.parse_args(argv)

    if args.selftest:
        print("lumo_sds selftest - the score computed without any data")
        fails = _selftest()
        if fails:
            for line in fails:
                print(f"  !! {line}", file=sys.stderr)
            print(f"  {len(fails)} check(s) FAILED", file=sys.stderr)
            return 4
        print("  all checks passed")
        return 0

    doc, problem = load_source(args.source)
    if problem:
        print(f"error: {problem}", file=sys.stderr)
        return 2

    if not os.path.isfile(args.csv):
        print(f"error: input CSV not found: {args.csv}", file=sys.stderr)
        return 2

    res = score(doc)
    res_gamma2 = score_gamma2(args.csv)
    block = {
        "schema": "lumo_sds/v1",
        "generated": args.date,
        "unit": "score 0-10",
        "source_json": os.path.basename(args.source),
        "source_schema": doc.get("schema"),
        "source_generated": doc.get("generated"),
        "sds_scale": SDS_SCALE,
        "method": {
            "metric": "Cliff's delta between the healthy and the damaged "
                      "recordings of one campaign; strict comparisons, ties count "
                      "for neither side, denominator n_healthy * n_damaged",
            "aggregate": "SDS = 10 x mean over campaigns of |delta|, clamped to "
                         "0 ... 10",
            "pairing": "within campaign only - the three healthy sets come from "
                       "three different field campaigns and are not "
                       "interchangeable",
            "excluded_pairs": [list(p) for p in MODAL_EXCLUSIONS],
            "exclusion_reason": "the DAM3 h2 damping is not admissible evidence at "
                                "the estimator's resolution: the healthy half-power "
                                "width sits only ~1.25x above the resolvability "
                                "threshold (zeta 0.0037 vs 3 x floor = 0.0029) while "
                                "the damaged width is ~14x larger (zeta 0.0496), so "
                                "that campaign's near-complete separation measures "
                                "the resolution floor, not the structure",
            "channels": {name: f"{mode} / {field}" for name, mode, field in CHANNELS},
        },
        "channels": res["channels"],
        "informational": res["informational"],
        "gamma2": {
            "source_csv": os.path.basename(args.csv),
            "method": {
                "metric": "Cliff's delta between the pooled healthy gamma2 values "
                          "and one damage state's gamma2 values; strict "
                          "comparisons, ties count for neither side",
                "aggregate": "SDS = 10 * max(|delta|) across the three "
                             "Healthy-to-damage comparisons; the pooled "
                             "Healthy-vs-all-damaged SDS is reported separately, "
                             "not folded into the max",
            },
            **res_gamma2,
        },
    }

    print(f"LUMO modal state-discrimination scores - source "
          f"{os.path.basename(args.source)} ({block['generated']})")
    print(render_table(res))
    if args.explain:
        print("modal exclusions:")
        print(render_explain(res, doc))
    print(render_gamma2_table(res_gamma2))

    ok = True
    if args.verify:
        expected_path = args.expected or os.path.join(DATA_DIR, "lumo_sds_expected.json")
        if os.path.isfile(expected_path):
            with open(expected_path) as fh:
                expected = json.load(fh)
            ok_modal = verify(res, expected, expected_path) is True
            ok_gamma2 = verify_gamma2(res_gamma2, expected, expected_path) is True
            ok = ok_modal and ok_gamma2
        elif args.expected:
            print(f"error: --expected file not found: {expected_path}", file=sys.stderr)
            return 1
        else:
            print(f"note: no pinned reference at {os.path.basename(expected_path)} - "
                  "verification skipped (the scores above come from the data alone)")

    if args.json_out:
        with open(args.json_out, "w") as fh:
            json.dump(block, fh, indent=2, sort_keys=False)
            fh.write("\n")
        print(f"wrote {args.json_out}")
    return 0 if ok else 3


if __name__ == "__main__":
    sys.exit(main())

