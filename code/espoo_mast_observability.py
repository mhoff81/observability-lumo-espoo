#!/usr/bin/env python3
"""
Espoo Kurttila telecom lattice mast — Sentinel-1 observability from the real
2-year burst backfill.

# Locality (verified, not assumed)

The mast is in **Kurttila, Espoo** (suburb of the Suur-Kauklahti district,
postcode 02780). OSM/Nominatim reverse-geocoding of the supplied operator
coordinate (60.173433838534095, 24.61160264667601) returns
`25, Brinkinmäentie, Taltrikinmäki, Kurttila, Saunalahti, Suur-Kauklahti, Espoo`.
Note that the mast **node** itself (OSM 2233589804, 286 m south of that
coordinate) falls in the adjacent *Saunaniemi / Saunalahti* neighbourhood, so
both labels are recorded here rather than one being silently preferred.

Earlier revisions of this analysis called the site "Otaniemi" — that came from
the repository's *synthetic* demo case (`Site 1155 Espoo/Otaniemi`), not from
any site data, and was wrong.

# What this measures

The mast (OSM node 2233589804, 60.1708674 N / 24.6111439 E, 36 m,
`tower:construction=lattice`, `tower:type=communication`) was onboarded as a
tower asset and a 2-year Sentinel-1 burst backfill was requested
(2024-09-01 .. 2026-09-14, monthly slots, both passes). Every acquired burst is
decoded at the mast and the pipeline records the **whole-target mast echo
coherence**

    gamma^2 = |sum(z)|^2 / (N * sum(|z|^2))     over the masked mast echo

with the mask `intensity >= 0.30 * peak AND intensity >= 5.0 * strip median`
and a minimum of 2 masked pixels (identical constants to the Rust pipeline:
`sarsolve::coherence::COHERENCE_PEAK_FRAC_DEFAULT` = 0.30,
`COHERENCE_MEDIAN_MULT_DEFAULT` = 5.0, `COHERENCE_MIN_MASKED_DEFAULT` = 2).

For a stationary point scatterer the phases align and gamma^2 -> 1; a vibrating
scatterer smears them (Bessel J0 channel), and **clutter does the opposite of
what one wants**: it inflates the mask with incoherent pixels and drags
gamma^2 toward 0. So the pair (gamma^2, n_masked) — not gamma^2 alone — is the
discriminator this analysis keys on:

* **compact high-coherence echo**  = small n_masked, high gamma^2  -> a real
  point scatterer (the mast itself is a plausible candidate).
* **large incoherent mask**        = large n_masked, low gamma^2   -> the mask
  latched onto built-up clutter, not the mast.

# Reference: LUMO (the project's only state-labelled tower)

Taken from `lumo_dam6_analysis/lumo_tower_coherence_states.md` (identical
estimator, 482 real bursts, 9 m mast, open ground):

    healthy  gamma^2 median 0.0379 [0.0085, 0.1410], n_masked median 19  (n=322)
    ASC 0.0617 (n=80)  vs  DESC 0.0261 (n=242)
    single-look coherence floor ~0.04 is clutter-dominated

# Honest scope

* This site has **no damage state label**. The output is an observability /
  baseline verdict, never a damage detection. Everything below is a statement
  about whether the mast is a usable scatterer, not about its condition.
* `PRED_MIN_COHERENCE = 0.15` (`input_proxy::prediction_pipeline`) is the
  pipeline's own floor for using a sample in prediction.
* The mast footprint (base width) is **not mapped in OSM**; the FEM geometry is
  an assumption recorded at onboarding (base radius 2.25 m -> 4.5 m base).

# Outputs (in this directory)

    espoo_mast_observability.json   per-acquisition records + aggregates
    ESPOO_MAST_OBSERVABILITY.md     human-readable report
    espoo_mast_observability.png    gamma^2 series + mask-size discriminator

# Usage

    python3 espoo_mast_observability.py
    python3 espoo_mast_observability.py --asset-id <uuid> --request-id <uuid>
"""
import argparse
import json
import os
import statistics
import urllib.request
from collections import defaultdict

import numpy as np

try:
    from scipy import stats as sps
except ImportError:  # pragma: no cover - statistics optional
    sps = None

HERE = os.path.dirname(os.path.abspath(__file__))

# Onboarded asset + its 2-year backfill request (see the report for provenance).
DEFAULT_ASSET_ID = "c4ac1eb8-63e6-481d-997e-4bfe90405d69"
DEFAULT_REQUEST_ID = "1231623a-7e1e-4366-b46d-fd171b72871f"
DEFAULT_API = "http://localhost:8080"

# Mast coordinates (OSM node 2233589804) and the 2-year catalogue window.
MAST_LAT, MAST_LON = 60.1708674, 24.6111439

# Locality, verified rather than assumed (see the module docstring): the operator
# coordinate reverse-geocodes to Kurttila; the mast node itself sits 286 m south,
# across a neighbourhood boundary. Both are recorded.
SITE_ASSET_NAME = "Espoo Kurttila Mast (OSM 2233589804)"
SITE_LOCALITY = ("Kurttila, Suur-Kauklahti, Espoo 02780 — reverse-geocoded from the "
                 "supplied operator coordinate; OSM places the mast node itself in the "
                 "adjacent Saunaniemi/Saunalahti neighbourhood")
CATALOG_START, CATALOG_END = "2024-09-14T00:00:00Z", "2026-09-14T23:59:59Z"

# Rust pipeline constants — must stay identical to the backend.
COHERENCE_PEAK_FRAC = 0.30
COHERENCE_MEDIAN_MULT = 5.0
COHERENCE_MIN_MASKED = 2
PRED_MIN_COHERENCE = 0.15

# LUMO reference medians (documentation, not recomputed here).
LUMO_HEALTHY_G2_MEDIAN = 0.0379
LUMO_HEALTHY_NMASKED_MEDIAN = 19
LUMO_ASC_G2_MEDIAN = 0.0617
LUMO_DESC_G2_MEDIAN = 0.0261
LUMO_CLUTTER_FLOOR = 0.04

# Clutter context measured live from OSM for the report.
OSM_BUILDINGS_WITHIN_100M = 37


def fetch_measurements(api, asset_id, request_id, timeout=120):
    """GET the per-acquisition records for one monitoring request."""
    url = f"{api}/api/v1/assets/{asset_id}/insar/measurements/{request_id}"
    req = urllib.request.Request(url, headers={"User-Agent": "espoo-mast-analysis/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as fh:
        payload = json.load(fh)
    return payload.get("measurements", payload if isinstance(payload, list) else [])


def fetch_s1_inventory(lat, lon, start, end, timeout=240):
    """2-year Sentinel-1 IW SLC inventory over the mast from the ASF catalogue.

    The ASF record is the availability view: how many scenes exist, on which
    relative orbits / frames, in which direction. It does not say what the
    pipeline can decode.
    """
    from urllib.parse import urlencode

    q = urlencode({
        "intersectsWith": f"POINT({lon} {lat})",
        "platform": "Sentinel-1",
        "processingLevel": "SLC",
        "beamMode": "IW",
        "start": start,
        "end": end,
        "output": "json",
    })
    url = "https://api.daac.asf.alaska.edu/services/search/param?" + q
    req = urllib.request.Request(url, headers={"User-Agent": "espoo-mast-analysis/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as fh:
        data = json.load(fh)
    rows = data[0] if isinstance(data, list) else data
    tracks = defaultdict(int)
    months = defaultdict(int)
    for r in rows:
        tracks[(str(r.get("relativeOrbit")), str(r.get("flightDirection")),
                str(r.get("frameNumber")))] += 1
        months[str(r.get("startTime"))[:7]] += 1
    return {
        "total": len(rows),
        "by_track": {"|".join(k): v for k, v in sorted(tracks.items(), key=lambda p: -p[1])},
        "by_month": dict(sorted(months.items())),
    }


def _num(v):
    """Coerce a JSON value to float, or None when absent/non-numeric."""
    try:
        return None if v is None else float(v)
    except (TypeError, ValueError):
        return None


def _pct(vals, q):
    return float(np.percentile(vals, q)) if vals else None


def group_stats(rows):
    """gamma^2 / mask-size statistics for one group of acquisitions."""
    g2 = [_num(r.get("coherence_gamma2")) for r in rows]
    g2 = [v for v in g2 if v is not None]
    px = [_num(r.get("coherence_masked_pixels")) for r in rows]
    px = [v for v in px if v is not None]
    if not g2:
        return None
    above_floor = sum(1 for v in g2 if v >= PRED_MIN_COHERENCE)
    return {
        "n": len(rows),
        "n_with_gamma2": len(g2),
        "gamma2_median": float(np.median(g2)),
        "gamma2_p25": _pct(g2, 25),
        "gamma2_p75": _pct(g2, 75),
        "gamma2_min": float(min(g2)),
        "gamma2_max": float(max(g2)),
        "n_masked_median": float(np.median(px)) if px else None,
        "n_masked_max": float(max(px)) if px else None,
        "share_above_pred_floor": above_floor / len(g2),
        "share_mask_below_min": sum(1 for v in px if v < COHERENCE_MIN_MASKED) / len(px) if px else None,
    }


def classify(stat):
    """Observability class from the (gamma^2, mask size) pair — see the docstring.

    Reference-relative, because `PRED_MIN_COHERENCE` (0.15) is the pipeline's bar
    for *using* a sample, not a bar for whether the mast is a scatterer at all.
    Each group carries `reference_g2` — the matching LUMO median for its orbit.

    Returns one of:
      * ``compact_echo_strong``            — compact mask and γ² at/above both the
                                             LUMO reference and the 0.15 floor
      * ``compact_echo_moderate``          — compact mask, at/above the LUMO
                                             reference but below the 0.15 floor
      * ``coherent_but_clutter_masked``    — coherent, but the mask is large
      * ``below_reference``                — below the LUMO reference scale
      * ``insufficient_data``
    """
    g2m = stat.get("gamma2_median")
    if not stat or g2m is None:
        return "insufficient_data"
    pxm = stat.get("n_masked_median")
    ref = stat.get("reference_g2", LUMO_HEALTHY_G2_MEDIAN)
    compact = pxm is not None and pxm <= 25.0
    if g2m < ref:
        return "below_reference"
    if not compact:
        return "coherent_but_clutter_masked"
    return "compact_echo_strong" if g2m >= PRED_MIN_COHERENCE else "compact_echo_moderate"


CLASS_TEXT = {
    "compact_echo_strong": "compact coherent echo, above the LUMO reference scale and the 0.15 pipeline floor — usable",
    "compact_echo_moderate": "compact coherent echo, above the LUMO reference scale but below the 0.15 pipeline floor",
    "coherent_but_clutter_masked": "coherent, but the mask is large — coherence plausibly carried by built-up clutter",
    "below_reference": "below the LUMO reference scale for this geometry",
    "insufficient_data": "insufficient data",
}


def _reference_for(key, stat):
    """Attach the matching LUMO reference median to a group's statistics."""
    if not stat:
        return stat
    k = key.upper()
    if k.startswith("DESC") or k.startswith("MORNING"):
        stat["reference_g2"] = LUMO_DESC_G2_MEDIAN
    elif k.startswith("ASC") or k.startswith("AFTERNOON"):
        stat["reference_g2"] = LUMO_ASC_G2_MEDIAN
    else:
        stat["reference_g2"] = LUMO_HEALTHY_G2_MEDIAN
    return stat


def per_group(rows):
    """Group the acquisitions by pass label and orbit direction."""
    by_pass = defaultdict(list)
    by_orbit = defaultdict(list)
    for r in rows:
        by_pass[r.get("pass_label") or "?"].append(r)
        by_orbit[r.get("orbit_direction") or "?"].append(r)
    return (
        {k: _reference_for(k, group_stats(v)) for k, v in sorted(by_pass.items())},
        {k: _reference_for(k, group_stats(v)) for k, v in sorted(by_orbit.items())},
    )


def compare_orbit(a, b):
    """Welch t + Mann-Whitney between two groups (gamma^2), when scipy exists."""
    if sps is None or not a or not b:
        return None
    if len(a) < 3 or len(b) < 3:
        return None
    try:
        t, p_t = sps.ttest_ind(a, b, equal_var=False)
        _, p_mw = sps.mannwhitneyu(a, b, alternative="two-sided")
    except Exception:  # pragma: no cover - degenerate inputs
        return None
    return {"n_a": len(a), "n_b": len(b), "welch_t": float(t),
            "welch_p": float(p_t), "mannwhitney_p": float(p_mw)}


def write_png(rows, path):
    """Two panels: gamma^2 over time (by pass) and the mask-size discriminator."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dated = []
    for r in rows:
        g2 = _num(r.get("coherence_gamma2"))
        if g2 is None:
            continue
        dated.append((str(r.get("acquisition_ts"))[:10], g2,
                      _num(r.get("coherence_masked_pixels")) or 0.0,
                      r.get("pass_label") or "?"))
    dated.sort()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    colors = {"morning": "#1f77b4", "afternoon": "#d62728", "?": "#7f7f7f"}

    if dated:
        dates = [d[0] for d in dated]
        for label in sorted({d[3] for d in dated}):
            sel = [d for d in dated if d[3] == label]
            ax1.plot([d[0] for d in sel], [d[1] for d in sel], "o-",
                     color=colors.get(label, "#7f7f7f"), label=f"{label} (n={len(sel)})")
        ax1.axhline(PRED_MIN_COHERENCE, color="grey", ls=":", lw=1,
                    label=f"prediction floor {PRED_MIN_COHERENCE}")
        ax1.axhline(LUMO_ASC_G2_MEDIAN, color="green", ls="--", lw=1,
                    label=f"LUMO ASC median {LUMO_ASC_G2_MEDIAN}")
        ax1.axhline(LUMO_DESC_G2_MEDIAN, color="purple", ls="--", lw=1,
                    label=f"LUMO DESC median {LUMO_DESC_G2_MEDIAN}")
        ax1.set_title("Mast-echo coherence $\\gamma^2$ per acquisition")
        ax1.set_ylabel("$\\gamma^2$")
        step = max(1, len(dates) // 8)
        idx = list(range(0, len(dates), step))
        ax1.set_xticks(idx)
        ax1.set_xticklabels([dates[i] for i in idx], rotation=45, ha="right", fontsize=8)
        ax1.legend(fontsize=7)
        ax1.grid(alpha=0.3)

        for label in sorted({d[3] for d in dated}):
            sel = [d for d in dated if d[3] == label]
            ax2.scatter([d[2] for d in sel], [d[1] for d in sel], s=55,
                        color=colors.get(label, "#7f7f7f"), label=label,
                        edgecolor="black", linewidth=0.4)
        ax2.axvline(25, color="orange", ls="--", lw=1, label="mask > 25 px = clutter-ish")
        ax2.axhline(PRED_MIN_COHERENCE, color="grey", ls=":", lw=1)
        ax2.set_xlabel("masked pixels in mast echo")
        ax2.set_ylabel("$\\gamma^2$")
        ax2.set_title("Discriminator: compact echo vs clutter mask")
        ax2.legend(fontsize=7)
        ax2.grid(alpha=0.3)
    else:
        for ax in (ax1, ax2):
            ax.text(0.5, 0.5, "no decoded acquisitions yet", ha="center", va="center")
            ax.set_axis_off()

    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def build_markdown(prov, catalog, by_pass, by_orbit, orbit_test, verdict, rows=None):
    """Human-readable report — same shape of reasoning as the LUMO write-ups."""
    L = []
    L.append("# Espoo Kurttila mast — Sentinel-1 observability (2-year backfill)\n")
    L.append("## Provenance\n")
    L.append(f"* **Site** — OSM node `2233589804`, `{MAST_LAT}, {MAST_LON}`, "
             "36 m, `tower:construction=lattice`, `tower:type=communication`.")
    L.append(f"* **Locality** — {SITE_LOCALITY}.")
    L.append("* **Distance to the operator coordinate supplied** "
             "(60.173433838534095, 24.61160264667601): **286 m** — the supplied "
             "point is *not* the mast.")
    L.append(f"* **Asset** — `{prov['asset_id']}` (`{prov['asset_name']}`), onboarded "
             f"tower, fundamental **{prov['fundamental_hz']:.2f} Hz**, "
             f"{prov['node_count']} nodes / {prov['element_count']} elements.")
    L.append("* **Geometry assumption** — base radius 2.25 m (4.5 m base), "
             "L 100x100x10 legs / L 60x60x6 bracing, 3 legs. OSM carries the mast "
             "**height only**; the footprint is not mapped.")
    L.append(f"* **Backfill request** — `{prov['request_id']}`, {prov['time_start']} .. "
             f"{prov['time_end']}, {prov['cadence']}, passes {prov['passes']}, "
             f"{prov['total_acquisitions']} slots, status `{prov['status']}`.")
    L.append(f"* **Catalogue view (ASF, IW SLC, 2 years)** — **{catalog['total']} "
             "scenes** over the mast.\n")
    if catalog.get("by_track"):
        L.append("| track (rel.orbit \\| dir \\| frame) | scenes |")
        L.append("|---|---:|")
        for k, v in list(catalog["by_track"].items())[:8]:
            L.append(f"| `{k}` | {v} |")
        L.append("")
    L.append("## Result — mast echo coherence and mask size\n")
    L.append("| group | n | γ² median | γ² [p25, p75] | masked px (median) | "
             "share γ² ≥ 0.15 | class |")
    L.append("|---|---:|---:|---|---:|---:|---|")
    for name, s in list(by_pass.items()) + list(by_orbit.items()):
        if not s:
            continue
        L.append(f"| **{name}** | {s['n']} | {s['gamma2_median']:.4f} | "
                 f"[{s['gamma2_p25']:.4f}, {s['gamma2_p75']:.4f}] | "
                 f"{s['n_masked_median']:.0f} | "
                 f"{s['share_above_pred_floor'] * 100:.0f}% | "
                 f"{CLASS_TEXT[classify(s)]} |")
    L.append("")
    L.append("## Comparison against LUMO (the only state-labelled tower)\n")
    L.append("| reference | γ² median | masked px |")
    L.append("|---|---:|---:|")
    L.append(f"| LUMO healthy (9 m mast, open ground) | {LUMO_HEALTHY_G2_MEDIAN} | "
             f"{LUMO_HEALTHY_NMASKED_MEDIAN} |")
    L.append(f"| LUMO ASCENDING | {LUMO_ASC_G2_MEDIAN} | — |")
    L.append(f"| LUMO DESCENDING | {LUMO_DESC_G2_MEDIAN} | — |")
    for name, s in by_orbit.items():
        if s:
            L.append(f"| **Espoo {name}** | {s['gamma2_median']:.4f} | "
                     f"{s['n_masked_median']:.0f} |")
    L.append("")
    if orbit_test:
        L.append(f"*Orbit contrast (Welch t={orbit_test['welch_t']:+.2f}, "
                 f"p={orbit_test['welch_p']:.4f}; Mann-Whitney "
                 f"p={orbit_test['mannwhitney_p']:.4f}; "
                 f"n={orbit_test['n_a']} vs {orbit_test['n_b']}).*\n")
    if rows:
        L.append("## Raw acquisitions\n")
        L.append("| acquisition (UTC) | pass | orbit | γ² | masked px |")
        L.append("|---|---|---|---:|---:|")
        for r in sorted(rows, key=lambda x: str(x.get("acquisition_ts"))):
            g2 = _num(r.get("coherence_gamma2"))
            px = _num(r.get("coherence_masked_pixels"))
            g2s = "" if g2 is None else f"{g2:.4f}"
            pxs = "" if px is None else f"{px:.0f}"
            L.append(f"| {str(r.get('acquisition_ts'))[:19]} | {r.get('pass_label')} | "
                     f"{r.get('orbit_direction')} | {g2s} | {pxs} |")
        L.append("")

    L.append("## Verdict\n")
    L.append(f"**{verdict['headline']}**\n")
    for line in verdict["notes"]:
        L.append(f"* {line}")
    L.append("")
    L.append("## Caveats (honest scope)\n")
    L.append("* **No damage state label exists for this site.** The output is an "
             "observability/baseline statement, never a condition assessment.")
    L.append(f"* The site is **clutter-heavy**: {OSM_BUILDINGS_WITHIN_100M} OSM building "
             "ways lie within 100 m of the mast, so the echo mask can latch onto "
             "buildings instead of the mast.")
    L.append("* Large mask + low coherence and small mask + high coherence are "
             "**different physical situations** and are reported separately rather "
             "than averaged into one misleading number.")
    L.append("* The catalogue count is SLC *availability*; the pipeline decodes "
             "**bursts**, so not every catalogue scene becomes a measurement.")
    L.append("* Orbit files can be missing for recent dates (`POEORB 404`), which "
             "disables orbit correction for that acquisition.")
    L.append("* Sentinel-1 coverage improves mid-2025 (S1C), so the pass/orbit "
             "balance is not constant across the 2-year window.\n")
    return "\n".join(L) + "\n"


def build_verdict(by_orbit, by_pass):
    """Classify each geometry and phrase the honest overall verdict."""
    classes = {k: classify(s) for k, s in by_orbit.items() if s}
    notes = []
    for k, s in sorted(by_orbit.items()):
        if not s:
            continue
        px = s["n_masked_median"]
        notes.append(
            f"**{k}** (n={s['n']}): γ² median {s['gamma2_median']:.4f} "
            f"[{s['gamma2_p25']:.4f}, {s['gamma2_p75']:.4f}], mask median "
            f"{px:.0f} px → {CLASS_TEXT[classify(s)]}."
        )
        if k.upper().startswith("ASC") and s["gamma2_median"] > LUMO_ASC_G2_MEDIAN:
            notes.append(
                f"ASC coherence exceeds the LUMO ASC reference "
                f"({LUMO_ASC_G2_MEDIAN}) with a "
                f"{'smaller' if (px or 0) < LUMO_HEALTHY_NMASKED_MEDIAN else 'larger'} "
                f"mask ({px:.0f} vs {LUMO_HEALTHY_NMASKED_MEDIAN} px) — the signature "
                "of a dominant compact scatterer rather than clutter.")
        if k.upper().startswith("DESC") and s["gamma2_median"] < LUMO_CLUTTER_FLOOR:
            notes.append(
                "DESC coherence sits at or below the documented LUMO single-look "
                "clutter floor (~0.04), so those samples carry no usable structural "
                "signal even though the mask is large.")
    if "compact_echo_strong" in classes.values():
        best = [k for k, c in classes.items() if c == "compact_echo_strong"]
        headline = ("Observable on " + ", ".join(best) +
                    " — a compact, coherent mast echo sits above both the LUMO "
                    "reference scale and the pipeline floor, so the site is usable on "
                    "that geometry. Geometry-dependent: use must be orbit-stratified.")
    elif "compact_echo_moderate" in classes.values():
        best = [k for k, c in classes.items() if c == "compact_echo_moderate"]
        headline = ("Partially observable on " + ", ".join(best) +
                    " — compact echo above the LUMO reference scale, but the median "
                    "sits below the 0.15 coherence the pipeline requires for inference.")
    elif "coherent_but_clutter_masked" in classes.values():
        headline = ("Not attributable — coherence exists but is carried by a "
                    "clutter-dominated mask and cannot be assigned to the mast.")
    elif classes:
        headline = ("Not observable — every geometry sits below the LUMO reference "
                    "scale for its orbit.")
    else:
        headline = "Insufficient data — no decoded acquisitions with coherence yet."
    notes.append("This site has **no ground-truth state label**; treat the result "
                 "only as observability evidence.")
    return {"headline": headline, "classes": classes, "notes": notes}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--asset-id", default=DEFAULT_ASSET_ID)
    ap.add_argument("--request-id", default=DEFAULT_REQUEST_ID)
    ap.add_argument("--api", default=DEFAULT_API)
    ap.add_argument("--lat", type=float, default=MAST_LAT)
    ap.add_argument("--lon", type=float, default=MAST_LON)
    ap.add_argument("--skip-catalogue", action="store_true",
                    help="reuse the cached catalogue inventory (offline run)")
    ap.add_argument("--out-dir", default=HERE)
    args = ap.parse_args()

    rows = fetch_measurements(args.api, args.asset_id, args.request_id)
    print(f"decoded acquisitions: {len(rows)}")

    cache = os.path.join(args.out_dir, "espoo_mast_observability.json")
    catalog = None
    if not args.skip_catalogue:
        try:
            catalog = fetch_s1_inventory(args.lat, args.lon, CATALOG_START, CATALOG_END)
            print(f"catalogue scenes over the mast (2 y): {catalog['total']}")
        except Exception as exc:  # a network hiccup must not kill the report
            print(f"catalogue query failed ({exc}); falling back to cache")
    if catalog is None and os.path.exists(cache):
        try:
            with open(cache) as fh:
                catalog = json.load(fh).get("catalog")
        except Exception:
            catalog = None
    catalog = catalog or {"total": 0, "by_track": {}, "by_month": {}}

    by_pass, by_orbit = per_group(rows)
    g2_by_orbit = {}
    for r in rows:
        k = r.get("orbit_direction") or "?"
        v = _num(r.get("coherence_gamma2"))
        if v is not None:
            g2_by_orbit.setdefault(k, []).append(v)
    orbit_keys = sorted(g2_by_orbit)
    orbit_test = None
    if len(orbit_keys) == 2:
        orbit_test = compare_orbit(g2_by_orbit[orbit_keys[0]], g2_by_orbit[orbit_keys[1]])

    verdict = build_verdict(by_orbit, by_pass)

    prov = {
        "asset_id": args.asset_id,
        "asset_name": SITE_ASSET_NAME,
        "request_id": args.request_id,
        "time_start": None, "time_end": None, "cadence": "monthly",
        "passes": [], "total_acquisitions": None, "status": "unknown",
        "fundamental_hz": 15.18239714298073, "node_count": 6, "element_count": 9,
    }
    try:  # prefer live request metadata when the local DB is reachable
        import subprocess
        sql = ("select time_start, time_end, cadence, array_to_string(passes,','), "
               "total_acquisitions, status from onboarder.insar_monitoring_requests "
               f"where id='{args.request_id}';")
        out = subprocess.run(
            ["docker", "exec", "tower-postgres", "psql", "-U", "tower", "-d", "tower",
             "-t", "-A", "-F", "|", "-c", sql],
            capture_output=True, text=True, timeout=30)
        if out.returncode == 0 and out.stdout.strip():
            f = out.stdout.strip().splitlines()[0].split("|")
            prov.update({"time_start": f[0], "time_end": f[1], "cadence": f[2],
                         "passes": f[3].split(",") if len(f) > 3 and f[3] else [],
                         "total_acquisitions": int(f[4]) if len(f) > 4 and f[4].isdigit() else None,
                         "status": f[5] if len(f) > 5 else "unknown"})
    except Exception as exc:
        print(f"request metadata lookup skipped: {exc}")

    n_total = len(rows)
    if prov.get("status") == "running" or n_total < 20:
        verdict["notes"].append(
            f"**Provisional** — the 2-year backfill is still running "
            f"({n_total} decoded acquisitions of ~{prov.get('total_acquisitions') or 25} "
            "slots). Re-run this script as slots complete; per-geometry n is too small "
            "for inference.")

    payload = {
        "provenance": prov,
        "constants": {"coherence_peak_frac": COHERENCE_PEAK_FRAC,
                      "coherence_median_mult": COHERENCE_MEDIAN_MULT,
                      "coherence_min_masked": COHERENCE_MIN_MASKED,
                      "pred_min_coherence": PRED_MIN_COHERENCE},
        "lumo_reference": {"healthy_g2_median": LUMO_HEALTHY_G2_MEDIAN,
                           "healthy_nmasked_median": LUMO_HEALTHY_NMASKED_MEDIAN,
                           "asc_g2_median": LUMO_ASC_G2_MEDIAN,
                           "desc_g2_median": LUMO_DESC_G2_MEDIAN,
                           "clutter_floor": LUMO_CLUTTER_FLOOR},
        "site_context": {"osm_node": 2233589804, "height_m": 36,
                         "construction": "lattice", "tower_type": "communication",
                         "locality": SITE_LOCALITY,
                         "buildings_within_100m": OSM_BUILDINGS_WITHIN_100M,
                         "coordinate_offset_to_supplied_point_m": 286},
        "catalog": catalog,
        "per_pass": by_pass,
        "per_orbit": by_orbit,
        "orbit_test": orbit_test,
        "verdict": verdict,
        "acquisitions": rows,
    }
    json_path = os.path.join(args.out_dir, "espoo_mast_observability.json")
    md_path = os.path.join(args.out_dir, "ESPOO_MAST_OBSERVABILITY.md")
    png_path = os.path.join(args.out_dir, "espoo_mast_observability.png")
    with open(json_path, "w") as fh:
        json.dump(payload, fh, indent=2)
    with open(md_path, "w") as fh:
        fh.write(build_markdown(prov, catalog, by_pass, by_orbit, orbit_test, verdict,
                                rows))
    write_png(rows, png_path)
    print(f"wrote {json_path}\nwrote {md_path}\nwrote {png_path}")
    print("VERDICT:", verdict["headline"])


if __name__ == "__main__":
    main()
