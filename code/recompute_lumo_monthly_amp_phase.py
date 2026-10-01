#!/usr/bin/env python3
"""Recompute `lumo_tower_monthly_amp_phase.json` from `data/lumo_channels.csv`.

A from-scratch port of `build_lumo_tower_monthly_amp_phase.py` (the original
generator, kept outside this repository in `lumo_dam6_analysis/`), with one
change of input: the original reads the raw per-burst binary SLC cache
(`monthly_bursts/<month>/<key>/{meta.json,window.bin,strip.bin}`) via
`analyze_burst()` to compute `amplitude`, `phase_rad`, `sub_aperture_brightness`
(including the per-block `blocks`/`bessel_summary`), `filtered` variants, etc.
That cache is not part of this repository - but `read_channels_csv()` returns
`burst` rows in the *exact source shape* of `amp["bursts"]`, because
`figures_data_csv.py` flattened them straight out of the committed
`lumo_tower_monthly_amp_phase.json` (confirmed key-for-key: the CSV-reconstructed
row has every key the original burst record has, after the 78-column schema
widening added the full `sub_aperture_brightness` block). So the ten analysis
functions below (`state_analysis` ... `polarization_analysis`,
`bessel_aperture_analysis`) are ported almost verbatim and run directly on
`blocks["burst"]` - no raw cache, no network call, no `attach_wind_direction()`
re-run, because:
  * `wind_direction_from_deg`/`wind_along_los_ms`/`wind_cross_los_ms`/etc. are
    already flat columns (the original attaches them in place from an
    Open-Meteo archive fetch; the committed JSON - and therefore the CSV - has
    already paid that cost).
  * `phase_rms_rad`/`disp_middle_m` are already flat columns too (the original
    `phase_interferometry_tests`/`shm_correlation_test` attach them in place
    from `lumo_tower_monthly_timeseries.json`, which this repository no longer
    carries - but the attached values already survived into every burst row).
  * `dedup_overpasses()` is a no-op here: the CSV's 356 `burst` rows are already
    the de-duplicated set (one row per distinct overpass-polarisation).
`data/wind_response_shm.json` is still read directly (present, used by
`falsification_tests`/`shm_correlation_test`'s SHM-fit lookups).

Usage:
    python3 recompute_lumo_monthly_amp_phase.py [--csv data/lumo_channels.csv]
                                                [--out /tmp/recomputed.json]
    python3 recompute_lumo_monthly_amp_phase.py --check   # diff vs data/*.json
"""
import argparse
import datetime as dt
import itertools
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np

try:
    from scipy import stats as sp_stats
except ImportError:  # pragma: no cover - statistical tests are optional
    sp_stats = None

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figures_data_csv import read_channels_csv  # noqa: E402

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
DEFAULT_CSV = os.path.join(DATA_DIR, "lumo_channels.csv")
DEFAULT_JSON = os.path.join(DATA_DIR, "lumo_tower_monthly_amp_phase.json")
SHM_PATH = os.path.join(DATA_DIR, "wind_response_shm.json")

DAMAGE_LABELS = ["healthy", "DAM 3", "DAM 4", "DAM 6"]
BESSEL_R_MIN_DEFAULT = 0.10

# The original burst record schema (from `analyze_burst()`). The CSV carries
# extra derived/native columns beyond this (record_key, sub_aperture_modulation,
# temperature_c, wind_gust_ms, precipitation_mm, orbit_direction, ...); trim
# down to this set when assembling the final `bursts`/`strongest_candidate`
# output so it matches the original JSON schema exactly.
ORIGINAL_BURST_KEYS = {
    "month", "burst_dir", "burst_id", "acquisition_date", "acquisition_ts",
    "orbit", "subswath", "polarisation", "damage_label", "structural_state",
    "campaign_range", "incidence_angle_deg", "amplitude", "phase_rad",
    "phase_deg", "mast_peak_row", "mast_peak_col", "mast_mean_amplitude",
    "mast_coherent_amplitude", "window_peak_amplitude", "window_peak_phase_deg",
    "strip_peak_amplitude", "strip_brightness_ratio", "sub_aperture_brightness",
    "filtered", "weather", "wind_direction_from_deg", "wind_direction_to_deg",
    "look_azimuth_deg", "wind_speed_ms", "wind_along_los_ms", "wind_cross_los_ms",
    "phase_rms_rad", "phase_coherence", "disp_middle_m",
}


def _native_burst(r):
    return {k: r[k] for k in ORIGINAL_BURST_KEYS if k in r}

# Source metadata the original records in `source` (raw-cache paths this
# repository does not hold) - documented constants, not recomputed.
SOURCE_META = {
    "database": "/home/projects/lumo_dam6_analysis/monthly_bursts",
    "cached_burst_count": 482,
    "dedup": "record-level identity = (acquisition minute, orbit, "
             "polarisation); pseudo-replicate bursts (same overpass listed "
             "under several burst ids a few seconds apart) are collapsed "
             "before any statistic is computed",
    "window_format": "(u32 width, u32 height) + 2*w*h interleaved f64 (real, imag)",
    "selection": "per month, the burst with the largest mast-echo amplitude "
                 "(peak |z| in the central 11x11 mast template) is the "
                 "strongest monthly candidate; its amplitude and phase are the "
                 "table values.",
}


# --- stats helpers (ported verbatim) ----------------------------------------
def _stats(vals):
    vals = list(vals)
    if not vals:
        return None
    a = np.asarray(vals)
    return {
        "n": len(vals),
        "mean": float(a.mean()),
        "median": float(np.median(a)),
        "std": float(a.std(ddof=1)),
    }


def _welch(a, b):
    if sp_stats is None:
        return None
    a, b = list(a), list(b)
    if not a or not b:
        return None
    t, p = sp_stats.ttest_ind(a, b, equal_var=False)
    return {"t": float(t), "p": float(p)}


def _corr(x, y):
    pr = sp_stats.pearsonr(x, y)
    sp_ = sp_stats.spearmanr(x, y)
    return {"r": float(pr.statistic), "p": float(pr.pvalue),
            "rho": float(sp_.statistic), "rho_p": float(sp_.pvalue)}


def load_shm_fits(with_a4=False):
    fits = {}
    if not os.path.isfile(SHM_PATH):
        return fits, None
    with open(SHM_PATH) as fh:
        shm = json.load(fh)
    for v in shm["verdicts"]:
        rc = v["response_curve"]
        for state, val in rc.items():
            if not isinstance(val, dict):
                continue
            k = val.get("loglog_exponent_k")
            c = val.get("coefficient_c")
            a4 = val.get("predicted_amp_at_4_ms")
            if k is not None and c is not None and k > 0:
                fits[(v["campaign"], state)] = (k, c, a4) if with_a4 else (k, c)
    return fits, shm


def campaign_window(r):
    """Which campaign window (dam3/dam6) and normalised label a burst falls in."""
    def inw(lo, hi):
        try:
            t = dt.date.fromisoformat(r["acquisition_date"])
        except (ValueError, TypeError):
            return False
        return lo <= t <= hi

    if inw(dt.date(2021, 3, 1), dt.date(2021, 4, 20)):
        camp = "dam3"
    elif inw(dt.date(2020, 10, 5), dt.date(2020, 10, 31)):
        camp = "dam6"
    else:
        return None, None
    lab = ("dam3" if r["damage_label"] == "DAM 3"
           else "dam6" if r["damage_label"] == "DAM 6" else "healthy")
    return camp, lab


# --- the ten analysis blocks (ported near-verbatim from build_lumo_tower_
#     monthly_amp_phase.py; signatures/field access unchanged) -------------
def state_analysis(records):
    groups = {lab: [r for r in records if r["damage_label"] == lab]
              for lab in DAMAGE_LABELS}
    per_state = {}
    for lab, recs in groups.items():
        row = {"n": len(recs)}
        if recs:
            row["amplitude"] = _stats([r["amplitude"] for r in recs])
            ph = np.asarray([r["phase_rad"] for r in recs])
            if sp_stats is not None:
                row["phase_circular_std_rad"] = float(
                    sp_stats.circstd(ph, high=np.pi, low=-np.pi))
        per_state[lab] = row

    tests = {}
    healthy_amp = [r["amplitude"] for r in groups["healthy"]]
    healthy_ph = [r["phase_rad"] for r in groups["healthy"]]
    all_damaged = [r["amplitude"] for r in records if r["damage_label"] != "healthy"]
    for lab in DAMAGE_LABELS[1:]:
        amp = [r["amplitude"] for r in groups[lab]]
        ph = [r["phase_rad"] for r in groups[lab]]
        tests[lab] = {
            "amplitude_vs_healthy": _welch(healthy_amp, amp),
            "phase_rad_vs_healthy": _welch(healthy_ph, ph),
        }
    tests["all_damaged"] = {"amplitude_vs_healthy": _welch(healthy_amp, all_damaged)}

    per_orbit = {}
    for orb in sorted({r["orbit"] for r in records}):
        per_orbit[orb] = {
            lab: _stats([r["amplitude"] for r in groups[lab] if r["orbit"] == orb])
            for lab in DAMAGE_LABELS
        }
    return {"per_state": per_state, "tests_vs_healthy": tests,
            "per_orbit_amplitude": per_orbit}


def modulation_analysis(records):
    def mod(r):
        return r.get("sub_aperture_brightness", {}).get("modulation_depth")

    def wind(r):
        return r.get("weather", {}).get("wind_speed_ms")

    def temp(r):
        return r.get("weather", {}).get("temperature_celsius")

    groups = {lab: [r for r in records if r["damage_label"] == lab]
              for lab in DAMAGE_LABELS}
    per_state = {lab: _stats([mod(r) for r in recs]) for lab, recs in groups.items()}

    tests = {}
    healthy = [mod(r) for r in groups["healthy"]]
    all_damaged = [mod(r) for r in records if r["damage_label"] != "healthy"]
    for lab in DAMAGE_LABELS[1:]:
        tests[lab] = {"modulation_vs_healthy": _welch(healthy, [mod(r) for r in groups[lab]])}
    tests["all_damaged"] = {"modulation_vs_healthy": _welch(healthy, all_damaged)}

    out = {"per_state": per_state, "tests_vs_healthy": tests}
    if sp_stats is None:
        return out

    valid = [(mod(r), wind(r), temp(r), r["month"], r["damage_label"])
             for r in records
             if mod(r) is not None and wind(r) is not None and temp(r) is not None]
    if len(valid) >= 10:
        m_ = np.asarray([v[0] for v in valid], dtype=float)
        w_ = np.asarray([v[1] for v in valid], dtype=float)
        t_ = np.asarray([v[2] for v in valid], dtype=float)

        def _pearson(x, y):
            r_, p_ = sp_stats.pearsonr(x, y)
            return {"r": float(r_), "p": float(p_)}

        def _partial(x, y, ctrl):
            xm = x - np.column_stack([np.ones(len(x)), ctrl]) @ np.linalg.lstsq(
                np.column_stack([np.ones(len(x)), ctrl]), x, rcond=None)[0]
            ym = y - np.column_stack([np.ones(len(y)), ctrl]) @ np.linalg.lstsq(
                np.column_stack([np.ones(len(y)), ctrl]), y, rcond=None)[0]
            return _pearson(xm, ym)

        out["weather_correlation"] = {
            "wind_speed_ms": {
                "n": len(valid),
                "pearson": _pearson(w_, m_),
                "spearman": {"rho": float(sp_stats.spearmanr(w_, m_).statistic),
                             "p": float(sp_stats.spearmanr(w_, m_).pvalue)},
            },
            "temperature_celsius": {
                "n": len(valid),
                "pearson": _pearson(t_, m_),
                "spearman": {"rho": float(sp_stats.spearmanr(t_, m_).statistic),
                             "p": float(sp_stats.spearmanr(t_, m_).pvalue)},
            },
            "confound_wind_vs_temp": _pearson(w_, t_),
            "partial_mod_wind_given_temp": _partial(m_, w_, t_),
            "partial_mod_temp_given_wind": _partial(m_, t_, w_),
        }

        by_month = defaultdict(list)
        for v in valid:
            by_month[v[3]].append(v)
        within_wind = []
        for mon in sorted(by_month):
            v = by_month[mon]
            if len(v) < 8:
                continue
            x = np.asarray([e[1] for e in v])
            y = np.asarray([e[0] for e in v])
            rho, p = sp_stats.spearmanr(x, y)
            within_wind.append({"month": mon, "rho": float(rho), "p": float(p), "n": len(v)})
        out["within_month_mod_vs_wind"] = within_wind
        neg = sum(1 for e in within_wind if e["rho"] < 0)
        out["within_month_negative_fraction"] = neg / len(within_wind) if within_wind else None

        bym = defaultdict(lambda: defaultdict(list))
        for v in valid:
            bym[v[3]][v[4]].append(v[0])
        within_state = []
        for mon in sorted(bym):
            g = bym[mon]
            if "healthy" not in g:
                continue
            for lab in DAMAGE_LABELS[1:]:
                if lab not in g:
                    continue
                h = np.asarray(g["healthy"])
                d = np.asarray(g[lab])
                if len(h) < 5 or len(d) < 5:
                    continue
                tst, pst = sp_stats.ttest_ind(h, d, equal_var=False)
                within_state.append({
                    "month": mon, "state": lab, "n_healthy": int(len(h)),
                    "n_state": int(len(d)), "t": float(tst), "p": float(pst),
                })
        out["within_month_state_tests"] = within_state
    return out


def wind_direction_analysis(records):
    sub = [r for r in records if r.get("wind_direction_from_deg") is not None]
    out = {"n": len(sub)}
    if not sub or sp_stats is None:
        return out

    def _corr(x, y):
        pr = sp_stats.pearsonr(x, y)
        sp_ = sp_stats.spearmanr(x, y)
        return {
            "pearson": {"r": float(pr.statistic), "p": float(pr.pvalue)},
            "spearman": {"rho": float(sp_.statistic), "p": float(sp_.pvalue)},
        }

    groups = {"ALL": sub}
    for o in ("ASCENDING", "DESCENDING"):
        g = [r for r in sub if r["orbit"] == o]
        if len(g) >= 30:
            groups[o] = g

    variants = ["wind_speed_ms", "wind_along_los_ms", "wind_cross_los_ms"]
    targets = {
        "modulation": lambda r: r["sub_aperture_brightness"]["modulation_depth"],
        "amplitude": lambda r: r["amplitude"],
    }
    out["correlations"] = {}
    for gname, g in groups.items():
        out["correlations"][gname] = {}
        for tname, get in targets.items():
            y = np.asarray([get(r) for r in g])
            out["correlations"][gname][tname] = {}
            for v in variants:
                x = np.asarray([r[v] for r in g])
                out["correlations"][gname][tname][v] = _corr(x, y)

    out["angle_summary"] = {}
    for o in ("ASCENDING", "DESCENDING"):
        g = [r for r in sub if r["orbit"] == o]
        if not g:
            continue
        ang = np.asarray([
            min(abs((r["wind_direction_to_deg"] - r["look_azimuth_deg"]) % 360),
                360 - abs((r["wind_direction_to_deg"] - r["look_azimuth_deg"]) % 360))
            for r in g
        ])
        out["angle_summary"][o] = {
            "n": len(g),
            "look_azimuth_deg": g[0]["look_azimuth_deg"],
            "median_flow_los_angle_deg": float(np.median(ang)),
            "median_along_cos": float(np.median(np.abs(np.cos(np.radians(ang))))),
            "median_cross_sin": float(np.median(np.abs(np.sin(np.radians(ang))))),
        }

    def _partial(y, x, ctrl):
        X = np.column_stack([np.ones(len(ctrl)), ctrl])
        bx = np.linalg.lstsq(X, x, rcond=None)[0]
        by = np.linalg.lstsq(X, y, rcond=None)[0]
        r_, p_ = sp_stats.pearsonr(x - X @ bx, y - X @ by)
        return {"r": float(r_), "p": float(p_)}

    desc = [r for r in sub if r["orbit"] == "DESCENDING"]
    if len(desc) >= 30:
        mod = np.asarray([r["sub_aperture_brightness"]["modulation_depth"] for r in desc])
        raw = np.asarray([r["wind_speed_ms"] for r in desc])
        cr = np.asarray([r["wind_cross_los_ms"] for r in desc])
        al = np.asarray([r["wind_along_los_ms"] for r in desc])
        out["desc_partial"] = {
            "cross_los_given_raw": _partial(mod, cr, raw),
            "along_los_given_raw": _partial(mod, al, raw),
        }
    return out


def falsification_tests(records):
    sub = [r for r in records if r.get("wind_direction_from_deg") is not None]
    out = {"n": len(sub)}
    if not sub or sp_stats is None:
        return out

    def mod(r):
        return r["sub_aperture_brightness"]["modulation_depth"]

    def wind(r):
        return r["wind_speed_ms"]

    def flow(r):
        return r["wind_direction_to_deg"]

    desc = [r for r in sub if r["orbit"] == "DESCENDING"]
    if len(desc) >= 30:
        y = np.asarray([mod(r) for r in desc])
        w = np.asarray([wind(r) for r in desc])
        raw = _corr(w, y)
        best_theta, best_r = None, 0.0
        for theta in range(0, 180, 5):
            x = np.asarray([
                wind(r) * abs(math.sin(math.radians(flow(r) - theta))) for r in desc
            ])
            rr = sp_stats.pearsonr(x, y).statistic
            if best_theta is None or abs(rr) > abs(best_r):
                best_theta, best_r = theta, rr
        z = (0.5 * math.log((1 + raw["r"]) / (1 - raw["r"]))
             - 0.5 * math.log((1 + best_r) / (1 - best_r))) / math.sqrt(2.0 / (len(desc) - 3))
        p = 2.0 * (1.0 - sp_stats.norm.cdf(abs(z)))
        out["test1_crosswind_axis"] = {
            "orbit": "DESCENDING",
            "n": len(desc),
            "raw_wind": raw,
            "best_crosswind_axis_deg": best_theta,
            "best_crosswind_r": float(best_r),
            "fisher_raw_vs_best": {"z": float(z), "p": float(p)},
        }
        out["test2_wind_squared"] = {
            "orbit": "DESCENDING",
            "n": len(desc),
            "v": raw,
            "v_squared": _corr(w ** 2, y),
        }

    fits, shm = load_shm_fits(with_a4=True)
    out["test3_shm_response"] = {
        "available_fits": [f"{c}/{s}" for (c, s) in fits],
    }
    rows = []
    for r in sub:
        camp, lab = campaign_window(r)
        if camp is None:
            continue
        fit = fits.get((camp, lab))
        if not fit:
            continue
        k, c, _a4 = fit
        rows.append({
            "date": r["acquisition_date"], "campaign": camp, "state": lab,
            "wind_ms": wind(r), "a_pred_ms2": c * wind(r) ** k,
            "modulation": mod(r),
        })
    t3 = out["test3_shm_response"]
    t3["n_paired"] = len(rows)
    if len(rows) >= 10:
        A = np.asarray([x["a_pred_ms2"] for x in rows])
        M = np.asarray([x["modulation"] for x in rows])
        W = np.asarray([x["wind_ms"] for x in rows])
        t3["mod_vs_wind"] = _corr(W, M)
        t3["mod_vs_a_pred"] = _corr(A, M)
        per_state = {}
        for camp in ("dam3", "dam6"):
            for lab_ in ("healthy", "dam3", "dam6"):
                sel = [x for x in rows if x["campaign"] == camp and x["state"] == lab_]
                if len(sel) < 5:
                    continue
                Aa = np.asarray([x["a_pred_ms2"] for x in sel])
                Mm = np.asarray([x["modulation"] for x in sel])
                per_state[f"{camp}/{lab_}"] = {
                    "n": len(sel), "mod_vs_a_pred": _corr(Aa, Mm),
                }
        t3["per_state"] = per_state
        state_level = []
        for (camp, lab_), (k, c, a4) in fits.items():
            sel = [x for x in rows if x["campaign"] == camp and x["state"] == lab_]
            if len(sel) < 5 or a4 is None:
                continue
            state_level.append({
                "state": f"{camp}/{lab_}", "n": len(sel),
                "shm_a4_ms2": a4,
                "sentinel_mod_median": float(np.median([x["modulation"] for x in sel])),
            })
        t3["state_level"] = state_level

    out["test4_frequency_nyquist"] = {
        "sub_aperture_rate_hz": 10.0,
        "nyquist_hz": 5.0,
        "fundamental_mode_hz": 2.75,
        "higher_modes_hz": [13.5, 16.1],
        "conclusion": (
            "The 10 Hz sub-aperture series (Nyquist 5 Hz) can only couple to the "
            "2.75 Hz fundamental (wind/excitation channel); the 13.5/16.1 Hz "
            "higher modes alias into [0,5] Hz and are not resolvable, so the SAR "
            "modulation cannot carry the damage-state signature."
        ),
    }
    return out


def phase_interferometry_tests(records):
    """Phase/displacement fields (`phase_rms_rad`, `disp_middle_m`) are already
    flat on every record - the original's load of
    `lumo_tower_monthly_timeseries.json` to attach them is skipped."""
    sub = [r for r in records if r.get("wind_direction_from_deg") is not None]
    out = {"n": len(sub)}
    if not sub or sp_stats is None:
        return out

    def mod(r):
        return r["sub_aperture_brightness"]["modulation_depth"]

    def wind(r):
        return r["wind_speed_ms"]

    desc = [r for r in sub if r["orbit"] == "DESCENDING" and r.get("phase_rms_rad") is not None]
    if len(desc) >= 30:
        y = np.asarray([r["phase_rms_rad"] for r in desc])
        w = np.asarray([wind(r) for r in desc])
        out["phase_rms_vs_wind"] = {
            "DESCENDING": {"n": len(desc), "wind": _corr(w, y)},
        }
        rp, rw = [], []
        gr = defaultdict(list)
        for r in desc:
            gr[r["acquisition_ts"][11:16]].append(r)
        for v in gr.values():
            if len(v) < 5:
                continue
            yy = np.asarray([r["phase_rms_rad"] for r in v])
            ww = np.asarray([wind(r) for r in v])
            rp.extend(yy - yy.mean())
            rw.extend(ww - ww.mean())
        if len(rp) > 10:
            out["phase_rms_vs_wind"]["DESCENDING_within_track"] = _corr(np.asarray(rw), np.asarray(rp))
        bins = []
        for lo, hi in ((0, 2), (2, 3.5), (3.5, 5), (5, 9)):
            sel = [r for r in desc if lo <= wind(r) < hi]
            if len(sel) >= 5:
                bins.append({
                    "wind_range": [lo, hi], "n": len(sel),
                    "median_phase_rms_rad": float(np.median([r["phase_rms_rad"] for r in sel])),
                })
        out["phase_rms_vs_wind"]["wind_bins"] = bins
    asc = [r for r in sub if r["orbit"] == "ASCENDING" and r.get("phase_rms_rad") is not None]
    if len(asc) >= 30:
        y = np.asarray([r["phase_rms_rad"] for r in asc])
        w = np.asarray([wind(r) for r in asc])
        out["phase_rms_vs_wind"]["ASCENDING"] = {"n": len(asc), "wind": _corr(w, y)}

    fits, _shm = load_shm_fits(with_a4=False)
    rows = []
    for r in sub:
        camp, lab = campaign_window(r)
        if camp is None:
            continue
        fit = fits.get((camp, lab))
        if not fit or r.get("phase_rms_rad") is None:
            continue
        k, c = fit
        rows.append((wind(r), c * wind(r) ** k, r["phase_rms_rad"]))
    if len(rows) >= 10:
        A = np.asarray([x[1] for x in rows])
        P = np.asarray([x[2] for x in rows])
        out["phase_rms_vs_shm"] = {"n": len(rows), "phase_rms_vs_a_pred": _corr(A, P)}

    dvalid = [r for r in sub if r["orbit"] == "DESCENDING" and r.get("disp_middle_m") is not None]
    if len(dvalid) >= 30:
        dv = np.asarray([r["disp_middle_m"] for r in dvalid])
        out["interferometric_disp"] = {
            "segment": "middle",
            "n": len(dvalid),
            "disp_m_distribution": {
                "median": float(np.median(dv)), "p5": float(np.percentile(dv, 5)),
                "p95": float(np.percentile(dv, 95)), "min": float(dv.min()),
                "max": float(dv.max()),
            },
            "vs": {},
        }
        for key, get in (("wind", wind),
                         ("temperature", lambda r: r["weather"].get("temperature_celsius")),
                         ("modulation", mod),
                         ("amplitude", lambda r: r["amplitude"]),
                         ("phase_rms", lambda r: r.get("phase_rms_rad"))):
            sel = [r for r in dvalid if get(r) is not None]
            if len(sel) < 30:
                continue
            x = np.asarray([get(r) for r in sel])
            yy = np.asarray([r["disp_middle_m"] for r in sel])
            out["interferometric_disp"]["vs"][key] = _corr(x, yy)
        sel = [r for r in dvalid if wind(r) is not None]
        if len(sel) >= 30:
            x = np.asarray([mod(r) for r in sel])
            ctrl = np.asarray([wind(r) for r in sel])
            yy = np.asarray([r["disp_middle_m"] for r in sel])
            X = np.column_stack([np.ones(len(ctrl)), ctrl])
            bx = np.linalg.lstsq(X, x, rcond=None)[0]
            by = np.linalg.lstsq(X, yy, rcond=None)[0]
            pr = sp_stats.pearsonr(x - X @ bx, yy - X @ by)
            out["interferometric_disp"]["disp_vs_modulation_given_wind"] = {
                "r": float(pr.statistic), "p": float(pr.pvalue)}
        thr = float(np.percentile(dv, 90))
        strong = [r for r in dvalid if r["disp_middle_m"] >= thr]
        weak = [r for r in dvalid if r["disp_middle_m"] < thr]
        sc = [r for r in strong if r.get("phase_coherence") is not None]
        wc = [r for r in weak if r.get("phase_coherence") is not None]
        out["interferometric_disp"]["strong_disp_p90"] = {
            "threshold_m": thr,
            "n_strong": len(strong), "n_weak": len(weak),
            "modulation_median_strong": float(np.median([mod(r) for r in strong])),
            "modulation_median_weak": float(np.median([mod(r) for r in weak])),
            "phase_coherence_median_strong": float(np.median([r["phase_coherence"] for r in sc])),
            "phase_coherence_median_weak": float(np.median([r["phase_coherence"] for r in wc])),
            "wind_median_strong": float(np.median([wind(r) for r in strong])),
            "wind_median_weak": float(np.median([wind(r) for r in weak])),
        }
    return out


def shm_correlation_test(records):
    """As `phase_interferometry_tests`: `phase_rms_rad`/`disp_middle_m` are
    already flat, so the timeseries-JSON attach step is skipped."""
    sub = [r for r in records if r.get("wind_direction_from_deg") is not None]
    out = {"n": len(sub)}
    if not sub or sp_stats is None:
        return out

    def wind(r):
        return r["wind_speed_ms"]

    channels = {
        "amplitude": lambda r: r["amplitude"],
        "mast_mean": lambda r: r["mast_mean_amplitude"],
        "modulation": lambda r: r["sub_aperture_brightness"]["modulation_depth"],
        "brightness_ratio": lambda r: r["strip_brightness_ratio"],
        "phase_rms": lambda r: r.get("phase_rms_rad"),
        "disp_middle": lambda r: r.get("disp_middle_m"),
    }

    fits, shm = load_shm_fits(with_a4=True)

    rows = []
    for r in sub:
        camp, lab = campaign_window(r)
        if camp is None:
            continue
        fit = fits.get((camp, lab))
        if not fit:
            continue
        k, c, _a4 = fit
        rows.append({"camp": camp, "lab": lab, "r": r, "a_pred": c * wind(r) ** k})
    out["per_burst"] = {"n": len(rows)}
    if len(rows) >= 10:
        per_burst = {}
        for name, get in channels.items():
            sel = [x for x in rows if get(x["r"]) is not None]
            if len(sel) < 10:
                continue
            A = np.asarray([x["a_pred"] for x in sel])
            Y = np.asarray([get(x["r"]) for x in sel])
            per_burst[name] = _corr(A, Y)
        out["per_burst"]["channels"] = per_burst

    a4_by_state = {}
    if shm is not None:
        for v in shm["verdicts"]:
            rc = v["response_curve"]
            for state, val in rc.items():
                if not isinstance(val, dict):
                    continue
                a4 = val.get("predicted_amp_at_4_ms")
                k = val.get("loglog_exponent_k")
                if a4 is not None:
                    a4_by_state[(v["campaign"], state)] = (
                        a4, bool(k is not None and k > 0))

    def exact_spearman_p(x, y):
        rx = sp_stats.rankdata(x)
        obs = abs(float(np.corrcoef(rx, sp_stats.rankdata(y))[0, 1]))
        cnt, tot = 0, 0
        for perm in itertools.permutations(range(1, len(y) + 1)):
            rho = np.corrcoef(rx, np.asarray(perm, dtype=float))[0, 1]
            tot += 1
            if abs(rho) >= obs:
                cnt += 1
        return obs, cnt / tot

    windows = {
        "dam3": (dt.date(2021, 3, 1), dt.date(2021, 4, 20)),
        "dam6": (dt.date(2020, 10, 5), dt.date(2020, 10, 31)),
    }
    state_level = {}
    for name, get in channels.items():
        pairs = []
        for (camp, lab), (a4, fit_ok) in a4_by_state.items():
            lab_s = "DAM 3" if lab == "dam3" else ("DAM 6" if lab == "dam6" else "healthy")
            sel = [r for r in sub
                   if windows[camp][0] <= dt.date.fromisoformat(r["acquisition_date"]) <= windows[camp][1]
                   and r["damage_label"] == lab_s]
            sel = [r for r in sel if get(r) is not None]
            if len(sel) < 5:
                continue
            x = np.asarray([wind(r) for r in sel])
            y = np.asarray([get(r) for r in sel])
            b, a = np.polyfit(x, y, 1)
            pairs.append({
                "state": f"{camp}/{lab}", "n": len(sel),
                "shm_a4_ms2": a4, "fit_reliable": fit_ok,
                "channel_at_4ms": float(a + b * 4.0),
            })
        if len(pairs) >= 3:
            A = np.asarray([p_["shm_a4_ms2"] for p_ in pairs])
            C = np.asarray([p_["channel_at_4ms"] for p_ in pairs])
            rho, p_exact = exact_spearman_p(A, C)
            state_level[name] = {
                "n_states": len(pairs),
                "states": pairs,
                "spearman_rho": rho,
                "exact_permutation_p": p_exact,
            }
    out["state_level"] = state_level
    return out


def filtered_analysis(records):
    sub = [r for r in records if "filtered" in r and r["filtered"]]
    out = {"n": len(sub)}
    if not sub or sp_stats is None:
        return out

    def vget(name):
        if name == "amplitude":
            return lambda r: r["amplitude"]
        return lambda r: r.get("filtered", {}).get(name)

    variants = ["amplitude", "coherent_amplitude", "multilook_amplitude",
                "med3_amplitude", "box3_amplitude"]

    out["state_separation"] = {}
    for v in variants:
        get = vget(v)
        row = {}
        for gname, group in (("ALL", sub),
                             ("DESCENDING", [r for r in sub if r["orbit"] == "DESCENDING"])):
            healthy = [get(r) for r in group
                       if r["damage_label"] == "healthy" and get(r) is not None]
            if len(healthy) < 20:
                continue
            row[gname] = {}
            for lab in DAMAGE_LABELS[1:]:
                d = [get(r) for r in group
                     if r["damage_label"] == lab and get(r) is not None]
                if len(d) < 5:
                    continue
                t, p = sp_stats.ttest_ind(healthy, d, equal_var=False)
                row[gname][lab] = {"t": float(t), "p": float(p)}
        out["state_separation"][v] = row

    out["wind_coupling"] = {}
    for v in variants:
        get = vget(v)
        desc = [r for r in sub
                if r["orbit"] == "DESCENDING" and get(r) is not None
                and r.get("wind_speed_ms") is not None]
        if len(desc) < 30:
            continue
        x = np.asarray([r["wind_speed_ms"] for r in desc])
        y = np.asarray([get(r) for r in desc])
        pr = sp_stats.pearsonr(x, y)
        sp_ = sp_stats.spearmanr(x, y)
        out["wind_coupling"][v] = {
            "n": len(desc),
            "pearson": {"r": float(pr.statistic), "p": float(pr.pvalue)},
            "spearman": {"rho": float(sp_.statistic), "p": float(sp_.pvalue)},
        }
    if "coherent_amplitude" in out.get("wind_coupling", {}):
        gr = defaultdict(list)
        for r in sub:
            if r["orbit"] == "DESCENDING" and r.get("wind_speed_ms") is not None \
                    and r.get("filtered", {}).get("coherent_amplitude") is not None:
                gr[r["acquisition_ts"][11:16]].append(r)
        rp, rw = [], []
        for v in gr.values():
            if len(v) < 5:
                continue
            yy = np.asarray([r["filtered"]["coherent_amplitude"] for r in v])
            ww = np.asarray([r["wind_speed_ms"] for r in v])
            rp.extend(yy - yy.mean())
            rw.extend(ww - ww.mean())
        if len(rp) > 10:
            pr = sp_stats.pearsonr(np.asarray(rw), np.asarray(rp))
            sp_ = sp_stats.spearmanr(np.asarray(rw), np.asarray(rp))
            out["wind_coupling"]["coherent_amplitude_within_track"] = {
                "n": len(rp), "r": float(pr.statistic), "p": float(pr.pvalue),
                "rho": float(sp_.statistic), "rho_p": float(sp_.pvalue),
            }

    def watson_williams(a, b):
        a, b = np.asarray(a), np.asarray(b)
        n1, n2 = len(a), len(b)
        r1 = float(np.abs(np.exp(1j * a).mean()))
        r2 = float(np.abs(np.exp(1j * b).mean()))
        R = float(np.abs(np.exp(1j * a).sum() + np.exp(1j * b).sum()) / (n1 + n2))
        k = 1.0 / (1.0 - R * R) * np.sqrt((R * R - r1 * r1 - r2 * r2) ** 2
                                          + 4 * r1 * r2 * R * R)
        denom = 1.0 - r1 - r2 + R
        F = (n1 + n2 - 2) * k * (r1 + r2 - R) / denom if denom > 0 else 0.0
        p = 1.0 - sp_stats.f.cdf(F, 1, n1 + n2 - 2)
        return {"F": float(F), "p": float(p)}

    out["phase_circular"] = {}
    for pname, get in (("raw", lambda r: r["phase_rad"]),
                       ("coherent", lambda r: r.get("filtered", {}).get("coherent_phase_rad"))):
        row = {}
        for lab in DAMAGE_LABELS[1:]:
            h = [get(r) for r in sub
                 if r["damage_label"] == "healthy" and get(r) is not None]
            d = [get(r) for r in sub
                 if r["damage_label"] == lab and get(r) is not None]
            if len(h) >= 20 and len(d) >= 5:
                row[lab] = watson_williams(h, d)
        out["phase_circular"][pname] = row
    return out


def wind_slope_state_test(records):
    sub = [r for r in records if r["orbit"] == "DESCENDING"
           and r.get("wind_speed_ms") is not None]
    out = {"n": len(sub)}
    if not sub or sp_stats is None:
        return out

    channels = {
        "modulation": lambda r: r["sub_aperture_brightness"]["modulation_depth"],
        "coherent_amplitude": lambda r: r.get("filtered", {}).get("coherent_amplitude"),
    }

    def linfit(x, y):
        b, a = np.polyfit(x, y, 1)
        pred = a + b * x
        ss = 1.0 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)
        se_b = float(np.sqrt(np.sum((y - pred) ** 2) / (len(y) - 2)
                             / np.sum((x - x.mean()) ** 2)))
        return float(b), se_b, float(ss)

    out["per_state_slopes"] = {}
    for ch_name, get in channels.items():
        rows = {}
        for lab in DAMAGE_LABELS:
            v = [r for r in sub if r["damage_label"] == lab and get(r) is not None]
            if len(v) < 10:
                continue
            x = np.asarray([r["wind_speed_ms"] for r in v])
            y = np.asarray([get(r) for r in v])
            b, se, ss = linfit(x, y)
            pr = sp_stats.pearsonr(x, y)
            rows[lab] = {
                "n": len(v), "slope": b, "slope_se": se, "r2": ss,
                "pearson_r": float(pr.statistic), "pearson_p": float(pr.pvalue),
            }
        out["per_state_slopes"][ch_name] = rows

    def interaction(x, y, s):
        X = np.column_stack([np.ones(len(x)), x, s, x * s])
        beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        dof = len(x) - 4
        sigma2 = np.sum((y - X @ beta) ** 2) / dof
        cov = sigma2 * np.linalg.inv(X.T @ X)
        se_d = float(np.sqrt(cov[3, 3]))
        t_d = beta[3] / se_d
        p_d = 2.0 * (1.0 - sp_stats.t.cdf(abs(t_d), dof))
        return float(beta[3]), se_d, float(t_d), float(p_d)

    out["interactions"] = {}
    for ch_name, get in channels.items():
        rows = {}
        for lab in DAMAGE_LABELS[1:]:
            sub2 = [r for r in sub
                    if r["damage_label"] in ("healthy", lab) and get(r) is not None]
            if len(sub2) < 20:
                continue
            x = np.asarray([r["wind_speed_ms"] for r in sub2])
            y = np.asarray([get(r) for r in sub2])
            s = np.asarray([1.0 if r["damage_label"] == lab else 0.0 for r in sub2])
            d, se_d, t_d, p_d = interaction(x, y, s)
            rows[lab] = {"n": len(sub2), "slope_diff_d": d, "se": se_d,
                         "t": t_d, "p": p_d}
        out["interactions"][ch_name] = rows

    out["within_month"] = {}
    for ch_name, get in channels.items():
        rows = {}
        for lab in DAMAGE_LABELS[1:]:
            per_month = []
            for mon in sorted({r["month"] for r in sub}):
                mh = [r for r in sub if r["month"] == mon
                      and r["damage_label"] == "healthy" and get(r) is not None]
                md = [r for r in sub if r["month"] == mon
                      and r["damage_label"] == lab and get(r) is not None]
                if len(mh) < 5 or len(md) < 5:
                    continue
                x = np.concatenate([[r["wind_speed_ms"] for r in mh],
                                    [r["wind_speed_ms"] for r in md]])
                y = np.concatenate([[get(r) for r in mh], [get(r) for r in md]])
                s = np.concatenate([np.zeros(len(mh)), np.ones(len(md))])
                d, se_d, t_d, p_d = interaction(x, y, s)
                per_month.append({"month": mon, "n_healthy": len(mh), "n_state": len(md),
                                  "slope_diff_d": d, "p": p_d})
            rows[lab] = per_month
        out["within_month"][ch_name] = rows

    out["power_analysis"] = {}
    for ch_name, get in channels.items():
        h = [r for r in sub if r["damage_label"] == "healthy" and get(r) is not None]
        if len(h) < 20:
            continue
        x = np.asarray([r["wind_speed_ms"] for r in h])
        y = np.asarray([get(r) for r in h])
        b, a = np.polyfit(x, y, 1)
        sigma = float(np.std(y - (a + b * x), ddof=2))
        varx = float(np.var(x))
        delta = abs(b)
        n_req = None
        if delta > 0:
            n_req = int(np.ceil(2.0 * sigma ** 2 / (varx * (delta / 2.8) ** 2)))
        out["power_analysis"][ch_name] = {
            "healthy_slope": float(b), "sigma": sigma, "var_wind": varx,
            "required_n_per_state_80pct": n_req,
        }
    return out


def polarization_analysis(records):
    sub = [r for r in records if r.get("polarisation") in ("vv", "vh")]
    out = {"n": len(sub),
           "counts": {p: sum(1 for r in sub if r["polarisation"] == p)
                      for p in ("vv", "vh")}}
    if not sub or sp_stats is None:
        return out

    out["vv_vs_vh_amplitude"] = {}
    for gname, group in (("ALL", sub),
                         ("ASCENDING", [r for r in sub if r["orbit"] == "ASCENDING"]),
                         ("DESCENDING", [r for r in sub if r["orbit"] == "DESCENDING"])):
        vv = [r["amplitude"] for r in group if r["polarisation"] == "vv"]
        vh = [r["amplitude"] for r in group if r["polarisation"] == "vh"]
        if len(vv) < 10 or len(vh) < 10:
            continue
        t, p = sp_stats.ttest_ind(vv, vh, equal_var=False)
        out["vv_vs_vh_amplitude"][gname] = {
            "n_vv": len(vv), "n_vh": len(vh),
            "median_vv": float(np.median(vv)), "median_vh": float(np.median(vh)),
            "ratio": float(np.median(vv) / np.median(vh)),
            "t": float(t), "p": float(p),
        }

    out["state_separation"] = {}
    for pol in ("vv", "vh"):
        g = [r for r in sub if r["polarisation"] == pol]
        row = {}
        for lab in DAMAGE_LABELS[1:]:
            h = [r["amplitude"] for r in g if r["damage_label"] == "healthy"]
            dd = [r["amplitude"] for r in g if r["damage_label"] == lab]
            if len(h) >= 10 and len(dd) >= 5:
                t, p = sp_stats.ttest_ind(h, dd, equal_var=False)
                row[lab] = {"t": float(t), "p": float(p),
                            "n_healthy": len(h), "n_state": len(dd)}
        out["state_separation"][pol] = row

    out["wind_coupling"] = {}
    for pol in ("vv", "vh"):
        g = [r for r in sub if r["polarisation"] == pol and r["orbit"] == "DESCENDING"
             and r.get("wind_speed_ms") is not None]
        row = {}
        for ch_name, get in (
                ("modulation",
                 lambda r: r["sub_aperture_brightness"]["modulation_depth"]),
                ("coherent_amplitude",
                 lambda r: r.get("filtered", {}).get("coherent_amplitude"))):
            sel = [r for r in g if get(r) is not None]
            if len(sel) < 20:
                continue
            x = np.asarray([r["wind_speed_ms"] for r in sel])
            y = np.asarray([get(r) for r in sel])
            pr = sp_stats.pearsonr(x, y)
            sp_ = sp_stats.spearmanr(x, y)
            row[ch_name] = {
                "n": len(sel),
                "pearson": {"r": float(pr.statistic), "p": float(pr.pvalue)},
                "spearman": {"rho": float(sp_.statistic), "p": float(sp_.pvalue)},
            }
        out["wind_coupling"][pol] = row

    out["orbit_scale"] = {}
    for pol in ("vv", "vh"):
        asc = [r["amplitude"] for r in sub
               if r["polarisation"] == pol and r["orbit"] == "ASCENDING"]
        desc = [r["amplitude"] for r in sub
                if r["polarisation"] == pol and r["orbit"] == "DESCENDING"]
        if asc and desc:
            out["orbit_scale"][pol] = {
                "n_asc": len(asc), "n_desc": len(desc),
                "median_asc": float(np.median(asc)),
                "median_desc": float(np.median(desc)),
                "ratio_asc_desc": float(np.median(asc) / np.median(desc)),
            }
    return out


def bessel_aperture_analysis(records, r_min=BESSEL_R_MIN_DEFAULT):
    rows = []
    for r in records:
        sab = r.get("sub_aperture_brightness")
        if not sab:
            continue
        blocks = sab.get("blocks", [])
        for b in blocks:
            fl = b.get("flags", {})
            rows.append({
                "burst_id": r.get("burst_id"),
                "orbit": r.get("orbit"),
                "damage_label": r.get("damage_label"),
                "date": r.get("acquisition_date"),
                "wind": r.get("wind_speed_ms"),
                "peak_ratio": b.get("peak_ratio", 0.0),
                "amp": b.get("bessel_amplitude_m"),
                "peak": b.get("peak_intensity", 0.0),
                "edge": fl.get("edge_block", False),
                "noise": fl.get("noise_dominated", False),
            })
    n = len(rows)
    out = {"n_blocks": n, "r_min": r_min}
    if not n:
        return out

    peak_ratio = np.asarray([x["peak_ratio"] for x in rows])
    amp = np.asarray([x["amp"] if x["amp"] is not None else float("nan")
                      for x in rows])
    peak = np.asarray([x["peak"] for x in rows])
    edge = np.asarray([x["edge"] for x in rows])
    noise = np.asarray([x["noise"] for x in rows])

    nonz = peak[peak > 0]
    floor = float(np.percentile(nonz, 5)) if nonz.size else 0.0
    out["noise_floor"] = floor

    stages = []
    m = np.ones(n, dtype=bool)
    for label, cond in (
        ("baseline (all blocks)", m),
        ("peak_intensity >= noise floor", peak >= floor),
        ("0 < peak_ratio <= 1", (peak_ratio > 0.0) & (peak_ratio <= 1.0)),
        (f"peak_ratio >= r_min ({r_min:.2f})", peak_ratio >= r_min),
        ("exclude edge / noise-dominated blocks", ~edge & ~noise),
    ):
        m = m & cond
        keep = int(m.sum())
        stages.append({
            "stage": label,
            "keep_blocks": keep,
            "keep_pct": round(100.0 * keep / n, 2),
            "filtered_pct": round(100.0 * (1.0 - keep / n), 2),
        })
    out["filter_stages"] = stages

    valid_idx = np.flatnonzero(m)

    per_burst = defaultdict(int)
    vb = defaultdict(int)
    for x in rows:
        per_burst[x["burst_id"]] += 1
    for i in valid_idx:
        vb[rows[i]["burst_id"]] += 1
    n_bursts = len(per_burst)
    out["dwell_usability"] = {}
    for thr in (1, 3, 4):
        usable = sum(1 for v in vb.values() if v >= thr)
        out["dwell_usability"][f"bursts_with_ge{thr}_valid_blocks"] = {
            "n": usable,
            "of": n_bursts,
            "pct": round(100.0 * usable / n_bursts, 2) if n_bursts else 0.0,
        }

    valid_amps = amp[valid_idx] * 1000.0
    if valid_amps.size:
        out["valid_amplitude_mm"] = {
            "n": int(valid_amps.size),
            "median": float(np.nanmedian(valid_amps)),
            "mean": float(np.nanmean(valid_amps)),
            "p5": float(np.nanpercentile(valid_amps, 5)),
            "p95": float(np.nanpercentile(valid_amps, 95)),
        }

    out["per_orbit"] = {}
    for o in sorted({x["orbit"] for x in rows}):
        idx = [i for i in valid_idx if rows[i]["orbit"] == o]
        a = amp[idx] * 1000.0
        out["per_orbit"][o] = {
            "n": int(len(idx)),
            "median_mm": float(np.nanmedian(a)) if idx else None,
        }

    if sp_stats is not None:
        by_burst = defaultdict(list)
        for i in valid_idx:
            x = rows[i]
            if x["orbit"] == "DESCENDING" and x["wind"] is not None:
                by_burst[x["burst_id"]].append((x["wind"], amp[i] * 1000.0))
        pairs = [(float(np.median([w for w, _ in v])),
                  float(np.median([a for _, a in v])))
                 for v in by_burst.values() if len(v) >= 2]
        if len(pairs) >= 10:
            wx = np.asarray([p[0] for p in pairs])
            ay = np.asarray([p[1] for p in pairs])
            out["wind_coupling_desc"] = {
                "n_bursts": int(len(pairs)),
                "spearman": {"rho": float(sp_stats.spearmanr(wx, ay).statistic),
                             "p": float(sp_stats.spearmanr(wx, ay).pvalue)},
                "pearson": {"r": float(sp_stats.pearsonr(wx, ay).statistic),
                            "p": float(sp_stats.pearsonr(wx, ay).pvalue)},
            }
    return out


# --- assembly ----------------------------------------------------------------
def recompute(csv_path):
    blocks = read_channels_csv(csv_path, site="lumo")
    records = list(blocks["burst"])

    by_month = defaultdict(list)
    for rec in records:
        by_month[rec["month"]].append(rec)
    months = sorted(by_month)

    monthly = []
    for month in months:
        recs = by_month[month]
        best = sorted(recs, key=lambda r: (-r["amplitude"], r["acquisition_ts"]))[0]
        labels = sorted({r["damage_label"] for r in recs})
        monthly.append({
            "month": month,
            "n_bursts": len(recs),
            "damage_labels": labels,
            "states_active": " + ".join(labels),
            "flags": {lab: (1 if lab in labels else 0) for lab in DAMAGE_LABELS},
            "median_mast_amplitude": float(np.median([r["amplitude"] for r in recs])),
            "mean_mast_amplitude": float(np.mean([r["amplitude"] for r in recs])),
            "strongest_candidate": _native_burst(best),
        })

    return {
        "tower": {"name": "LUMO lattice mast", "latitude": 52.38, "longitude": 9.72},
        "source": dict(SOURCE_META, burst_count=len(records)),
        "state_analysis": state_analysis(records),
        "modulation_analysis": modulation_analysis(records),
        "wind_direction_analysis": wind_direction_analysis(records),
        "falsification_tests": falsification_tests(records),
        "phase_interferometry_tests": phase_interferometry_tests(records),
        "shm_correlation_test": shm_correlation_test(records),
        "filtered_analysis": filtered_analysis(records),
        "wind_slope_state_test": wind_slope_state_test(records),
        "polarization_analysis": polarization_analysis(records),
        "bessel_aperture_analysis": bessel_aperture_analysis(records),
        "monthly": monthly,
        "bursts": [_native_burst(r) for r in records],
    }


def _close(a, b, rel=1e-9, abs_tol=1e-9):
    if isinstance(a, float) and isinstance(b, float):
        return math.isclose(a, b, rel_tol=rel, abs_tol=abs_tol)
    if isinstance(a, dict) and isinstance(b, dict):
        return set(a) == set(b) and all(_close(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_close(x, y) for x, y in zip(a, b))
    return a == b


def _check(rebuilt, original_path):
    if not os.path.isfile(original_path):
        print(f"error: no reference JSON at {original_path} to check against "
              "(it was removed from data/ once lumo_damping_gamma2_modulation.py "
              "switched to reading this script's recompute() directly; restore "
              "it from git history if you need to re-verify)", file=sys.stderr)
        raise SystemExit(3)
    with open(original_path) as fh:
        original = json.load(fh)
    fails = []
    for key in ("tower", "state_analysis", "modulation_analysis",
                "wind_direction_analysis", "falsification_tests",
                "phase_interferometry_tests", "shm_correlation_test",
                "filtered_analysis", "wind_slope_state_test",
                "polarization_analysis", "bessel_aperture_analysis", "monthly"):
        if not _close(rebuilt.get(key), original.get(key)):
            fails.append(f"{key}: mismatch")
    by_id = {b["burst_id"]: b for b in original["bursts"]}
    if len(rebuilt["bursts"]) != len(original["bursts"]):
        fails.append(f"burst count: got {len(rebuilt['bursts'])}, "
                     f"want {len(original['bursts'])}")
    for b in rebuilt["bursts"]:
        want = by_id.get(b["burst_id"])
        if want is None:
            fails.append(f"bursts[{b['burst_id']}]: not in original")
            continue
        # The CSV carries extra derived/native columns beyond the original
        # burst record's keys (e.g. record_key, sub_aperture_modulation) -
        # only compare the keys the original record actually has.
        got = {k: b[k] for k in want}
        if not _close(got, want):
            fails.append(f"bursts[{b['burst_id']}]: mismatch")
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
        print("  recomputed JSON matches data/lumo_tower_monthly_amp_phase.json")
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
