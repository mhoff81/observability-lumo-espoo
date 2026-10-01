#!/usr/bin/env python3
"""LUMO — does SAR see damping indirectly? zeta vs gamma2 vs sub-aperture modulation.

Combines the NEW damping-ratio measurements (half-power bandwidth from the raw
Uni-Hannover SHM recordings, `data/lumo_damping_ratio.json`) with the existing
SAR channels of the same states, both recomputed from `data/lumo_channels.csv`:

  * whole-tower coherence gamma2      (recompute_lumo_coherence_states.recompute)
  * intra-dwell brightness modulation (recompute_lumo_monthly_amp_phase.recompute)
  * SHM wind-response suppression     (data/wind_response_shm.json)

Tests the two proposed causal chains:
  M1  damage -> damping -> vibration response -> gamma2  (gamma2 as damping proxy)
  M2  damage -> damping -> wind reaction -> sub-aperture modulation
Physical prior: higher damping -> smaller amplitude at same wind -> less phase
smearing -> HIGHER gamma2 / SHALLOWER modulation. Confounded by stiffness loss
(lower stiffness -> MORE amplitude -> LOWER gamma2 / deeper modulation), so the
SHM amplitude-at-common-wind is the discriminator.

Usage: ../figures/lumo_damping_gamma2_modulation.sh
Writes: ../data/lumo_damping_gamma2_modulation.json,
        ../figures/lumo_damping_gamma2_modulation.md / .png
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
# this script lives in `code/`; the caches it reads live in the sibling
# `data/` directory (the same convention as `code/lumo_damping_frequencies.py`).
# Its .json output also goes to `data/`; its .md/.png go to the sibling
# `figures/` directory.
DATA_DIR = os.path.normpath(os.path.join(HERE, os.pardir, "data"))
FIGURES_DIR = os.path.normpath(os.path.join(HERE, os.pardir, "figures"))
LUMO6 = DATA_DIR  # the LUMO monthly/coherence/wind caches
ZETA = os.path.join(DATA_DIR, "lumo_damping_ratio.json")
LUMO_CSV = os.path.join(DATA_DIR, "lumo_channels.csv")

sys.path.insert(0, HERE)
import recompute_lumo_coherence_states as _coh_mod  # noqa: E402
import recompute_lumo_monthly_amp_phase as _amp_mod  # noqa: E402

MODULATION_LABELS = {"healthy": "healthy", "dam3": "DAM 3", "dam4": "DAM 4", "dam6": "DAM 6"}
COH_LABELS = {"healthy": "DAM0(healthy)", "dam3": "DAM3", "dam4": "DAM4", "dam6": "DAM6"}

# One row per damage state, in label order. Each state is compared against its OWN healthy
# campaign (see `campaign_layout` in lumo_damping_ratio.json), so the healthy ζ baseline
# differs per state; the figure plots Δζ to make the three rows comparable.
STATES = (("dam3", "dam3"), ("dam4", "dam4"), ("dam6", "dam6"))


def load():
    zeta = json.load(open(ZETA))
    coh = _coh_mod.recompute(LUMO_CSV)
    mod = _amp_mod.recompute(LUMO_CSV)
    wind = json.load(open(os.path.join(LUMO6, "wind_response_shm.json")))
    return zeta, coh, mod, wind


def zeta_state(zeta, campaign, state, label):
    s = zeta["state_stats"][label][campaign]
    med = s["median_zeta"][state]
    cd = s.get("cliffs_delta_damaged_vs_healthy")
    return med, cd, s


def main():
    zeta, coh, mod, wind = load()
    campaign_ids = {c["name"]: dict(c["states"]) for c in zeta["campaign_layout"]}

    rows = []
    for state, windcamp in STATES:
        zH1, cdH1, _ = zeta_state(zeta, state, state, "H1")
        zH1h, _, _ = zeta_state(zeta, state, "healthy", "H1")
        zF, cdF, _ = zeta_state(zeta, state, state, "fundamental")
        zFh, _, _ = zeta_state(zeta, state, "healthy", "fundamental")
        g_coh = coh["per_state"][COH_LABELS[state]]["gamma2"]["median"]
        g_coh_h = coh["per_state"][COH_LABELS["healthy"]]["gamma2"]["median"]
        g_mod = mod["modulation_analysis"]["per_state"][MODULATION_LABELS[state]]["median"]
        g_mod_h = mod["modulation_analysis"]["per_state"][MODULATION_LABELS["healthy"]]["median"]
        t = coh["tests"].get(f"healthy_vs_{state.upper()}")
        t_mod = mod["modulation_analysis"]["tests_vs_healthy"][MODULATION_LABELS[state]]
        # DAM4 has ζ / γ² / modulation data but NO SHM wind-response recordings
        # (wind_response_shm.json covers dam3 + dam6 only), so keep this lookup optional.
        wvs = [v for v in wind["verdicts"] if v["campaign"] == windcamp]
        wv = wvs[0] if wvs else {}
        wratio = (wv.get("response_curve") or {}).get("damaged_vs_healthy_at_4ms")
        wsignal = wv.get("signal_present")
        rows.append({
            "state": state,
            "healthy_campaign": campaign_ids[state]["healthy"],
            "damaged_campaign": campaign_ids[state][state],
            "zeta_H1_healthy": zH1h, "zeta_H1_damaged": zH1,
            "zeta_H1_cliffs_delta": cdH1,
            "zeta_fund_healthy": zFh, "zeta_fund_damaged": zF,
            "zeta_fund_cliffs_delta": cdF,
            "gamma2_healthy": g_coh_h, "gamma2_damaged": g_coh,
            "gamma2_welch_p": t.get("welch_p") if t else None,
            "modulation_healthy": g_mod_h, "modulation_damaged": g_mod,
            "modulation_welch_p": t_mod.get("modulation_vs_healthy", {}).get("p"),
            "shm_wind_resp_ratio_4ms": wratio, "shm_wind_signal_present": wsignal,
            "zeta_H1_change": zH1 - zH1h,
            "zeta_fund_change": zF - zFh,
            "gamma2_change": g_coh - g_coh_h,
            "modulation_change": g_mod - g_mod_h,
        })

    result = {"rows": rows,
              "mechanism1": mechanism1(rows),
              "mechanism2": mechanism2(rows),
              "verdict": verdict(rows)}
    json_path = os.path.join(DATA_DIR, "lumo_damping_gamma2_modulation.json")
    with open(json_path, "w") as f:
        json.dump(result, f, indent=1)
    base = os.path.join(FIGURES_DIR, "lumo_damping_gamma2_modulation")
    report(result, base)
    plot(rows, base)
    print("Wrote: data/lumo_damping_gamma2_modulation.json, "
          "figures/lumo_damping_gamma2_modulation.md/.png")


def mechanism1(rows):
    out = {}
    for r in rows:
        dz = r["zeta_H1_damaged"] - r["zeta_H1_healthy"]
        dg = r["gamma2_damaged"] - r["gamma2_healthy"]
        out[r["state"]] = {"zeta_H1_change": dz, "gamma2_change": dg,
                           "consistent_with_damping_drive": bool(dz < 0 and dg > 0)}
    return out


def mechanism2(rows):
    out = {}
    for r in rows:
        st = r["state"]
        dm = r["modulation_damaged"] - r["modulation_healthy"]
        w = r["shm_wind_resp_ratio_4ms"]
        out[st] = {"modulation_change": dm, "shm_wind_resp_ratio_4ms": w,
                   "modulation_tracks_amplitude": bool(
                       (dm < 0 and w is not None and w < 1.0)
                       or (dm > 0 and w is not None and w > 1.0))}
    return out


def verdict(rows):
    L = []
    for r in rows:
        st = r["state"]
        w = r["shm_wind_resp_ratio_4ms"]
        L.append(f"* **{st}** (`{r['healthy_campaign']}` → `{r['damaged_campaign']}`): "
                 f"ζ(H1) {r['zeta_H1_healthy']:.5f} → {r['zeta_H1_damaged']:.5f} "
                 f"(Δ {r['zeta_H1_change']:+.5f}, Cliff's δ {r['zeta_H1_cliffs_delta']:+.2f}); "
                 f"ζ(fund) Δ {r['zeta_fund_change']:+.5f} "
                 f"(δ {r['zeta_fund_cliffs_delta']:+.2f}); γ² {r['gamma2_healthy']:.4f} → "
                 f"{r['gamma2_damaged']:.4f} (Welch p={r['gamma2_welch_p']:.4f}); modulation "
                 f"{r['modulation_healthy']:.3f} → {r['modulation_damaged']:.3f} (p="
                 f"{r['modulation_welch_p']:.3f}); SHM wind-resp ratio @4 m/s "
                 f"{w if w is not None else 'open (no SHM wind recordings for this campaign)'}.")
    return L
def report(result, base):
    L = []
    L.append("# LUMO — Does SAR see damping indirectly? ζ vs γ² vs sub-aperture modulation\n")
    L.append("New: modal damping ratio ζ (half-power bandwidth, raw Uni-Hannover SHM "
             "recordings, `lumo_damping_ratio.json`, state-adaptive mode windows, SNR gate). "
             "Existing: whole-tower γ² and intra-dwell brightness modulation from the same "
             "Sentinel-1 stack — one row per distinct overpass (acquisition minute + orbit, "
             "VV preferred; the artifact's burst rows are de-duplicated); SHM wind-response "
             "curves (`wind_response_shm.json`).\n")
    L.append("\nThree damage states (DAM3, DAM4, DAM6) are compared, each against its OWN "
             "healthy campaign (`campaign_layout`: 05_Healthy / 03_Healthy / 01_Healthy). Those "
             "healthy ζ(H1) baselines differ by ~2e-4, so panel 1 of the figure plots **Δζ** "
             "rather than absolute ζ — an absolute-ζ line would show a healthy \"trend\" that is "
             "pure baseline drift between campaigns. DAM4 has no SHM wind-response recordings "
             "(`wind_response_shm.json` covers dam3 + dam6 only), so its wind column is `open` "
             "by construction.\n")
    L.append("\n## State table\n")
    L.append("| state | healthy campaign | ζ H1 healthy | ζ H1 damaged | Δζ H1 | Cliff's δ | ζ fund Δ | γ² healthy | γ² damaged | γ² Welch p | mod healthy | mod damaged | mod p | SHM wind resp @4m/s |\n")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")
    for r in result["rows"]:
        wr = f"{r['shm_wind_resp_ratio_4ms']:.2f}" if r["shm_wind_resp_ratio_4ms"] is not None else "open"
        L.append(f"| {r['state']} | {r['healthy_campaign']} | {r['zeta_H1_healthy']:.5f} | "
                 f"{r['zeta_H1_damaged']:.5f} | {r['zeta_H1_change']:+.5f} | "
                 f"{r['zeta_H1_cliffs_delta']:+.2f} | {r['zeta_fund_change']:+.5f} | "
                 f"{r['gamma2_healthy']:.4f} | {r['gamma2_damaged']:.4f} | "
                 f"{r['gamma2_welch_p']:.4f} | {r['modulation_healthy']:.3f} | "
                 f"{r['modulation_damaged']:.3f} | {r['modulation_welch_p']:.3f} | {wr} |\n")
    L.append("\n## Mechanism 1 — γ² as a damping proxy\n")
    for st, m in result["mechanism1"].items():
        L.append(f"* {st}: ζ(H1) {m['zeta_H1_change']:+.4f}, γ² {m['gamma2_change']:+.4f} → "
                 f"damping-driven consistency: **{m['consistent_with_damping_drive']}**.\n")
    L.append("* Interpretation: γ² is the NET vibration channel. DAM6 γ² drops sharply "
             "(stiffness loss dominates); DAM3 γ² is slightly UP (suppression); DAM4 sits "
             "between them, DOWN but only at ~40 % of DAM6's Δ (and with the largest median "
             "`n_masked`, 14 vs DAM6's 12 — see `lumo_gamma2_confound_diagnosis.md`). A pure "
             "damping proxy would predict γ² UP in every state, because ζ(H1) falls in all "
             "three — only DAM3 is consistent. **γ² alone is NOT a damping proxy; it is a "
             "vibration-amplitude channel.**\n")
    L.append("\n## Mechanism 2 — sub-aperture modulation as damping/wind-response sensor\n")
    for st, m in result["mechanism2"].items():
        w = m["shm_wind_resp_ratio_4ms"]
        L.append(f"* {st}: modulation {m['modulation_change']:+.4f} vs SHM wind-resp ratio "
                 f"{w if w is not None else 'open'} → modulation tracks the SHM amplitude: "
                 f"**{m['modulation_tracks_amplitude']}**.\n")
    L.append("* `open` means there are NO SHM wind-response recordings for that campaign "
             "(true of DAM4), so its `tracks` flag is False by construction — absence of "
             "evidence, not evidence against the wind-response mapping.\n")
    L.append("* Interpretation: DAM3's SHM suppression (ratio 0.3782) maps onto shallower "
             "modulation (0.8213, p=0.088) and DAM6's higher wind-normalized response (2.0473) "
             "onto deeper modulation (0.8652, p=0.134) — the two states whose SHM amplitude is "
             "measured both agree in sign. DAM4 shows the DEEPEST modulation of the three "
             "(0.8832, p=0.047, the only p < 0.05) but has no wind-response recordings, so its "
             "amplitude is unknown and that leg cannot be closed for it. **The sub-aperture "
             "modulation tracks the wind-normalized vibration amplitude, which is the net of "
             "damping + stiffness.**\n")
    r4 = next((r for r in result["rows"] if r["state"] == "dam4"), None)
    if r4 is not None:
        L.append("\n## DAM4 — SAR-weak, damping-strong\n")
        L.append(f"* **ζ (SHM)**: H1 {r4['zeta_H1_healthy']:.5f} → {r4['zeta_H1_damaged']:.5f} "
                 f"(Δ {r4['zeta_H1_change']:+.5f}, δ {r4['zeta_H1_cliffs_delta']:+.2f}) — the "
                 f"lowest damaged H1 median and complete separation, i.e. the *relatively* "
                 f"largest H1 damping loss of the three states; fundamental "
                 f"{r4['zeta_fund_healthy']:.5f} → {r4['zeta_fund_damaged']:.5f} "
                 f"(Δ {r4['zeta_fund_change']:+.5f}, δ {r4['zeta_fund_cliffs_delta']:+.2f}) — "
                 f"the largest fundamental ζ increase. So on the SHM side DAM4 is not a mild "
                 f"case.\n")
        L.append(f"* **γ² (SAR)**: {r4['gamma2_healthy']:.4f} → {r4['gamma2_damaged']:.4f} "
                 f"(Δ {r4['gamma2_change']:+.4f}, Welch p={r4['gamma2_welch_p']:.3f}) — the "
                 f"weakest contrast of the three states (which in the committed "
                 f"`LUMO_GAMMA2_MASKED` units stand at +0.1392 / -0.0527 / -0.1281 for "
                 f"DAM3 / DAM4 / DAM6) *and* the least separable from the N/brightness "
                 f"bookkeeping: the N-mechanism surrogate (band a, "
                 f"`lumo_gamma2_confound_diagnosis.md`) puts its raw δ at percentile +12.75 "
                 f"(inside the band, vs +51.05 for DAM3 and +4.5 for DAM6, which is the only "
                 f"state outside it). 87.5 % (14/16) of its rows lie inside the healthy "
                 f"N/brightness box — the best-supported positivity of the three (DAM3 84.0 %, "
                 f"DAM6 50.0 %) — with a median `n_masked` of 14 against the healthy 15, so its "
                 f"raw γ² deficit is what the healthy γ² = -0.0090 + 1.3886/N law already "
                 f"predicts.\n")
        L.append("* **N controls**: fixed-N (k = 10) δ = -0.2295, between DAM3 (-0.1096) and "
                 "DAM6 (-0.3375); but the closest-N matched control flips its sign to +0.0306 on "
                 "14 pairs (DAM3 -0.0350 on 20, DAM6 -0.2200 on 10). It is the only state with "
                 "the same number of overpasses in each season (8 summer, 8 winter), so this "
                 "contrast is not a season artifact — unlike DAM3, whose contrast is summer-only "
                 "and whose sign flips with orbit.\n")
        L.append(f"* **Modulation (SAR)**: {r4['modulation_healthy']:.3f} → "
                 f"{r4['modulation_damaged']:.3f} (Δ "
                 f"{r4['modulation_damaged'] - r4['modulation_healthy']:+.4f}, "
                 f"p={r4['modulation_welch_p']:.3f}) — the DEEPEST modulation of the three and "
                 f"the only state reaching p < 0.05. With no wind-response recordings for "
                 f"`04_DAM4_111` there is no common-wind amplitude to explain it, so this is the "
                 f"one channel where DAM4 leads and the one that cannot yet be closed.\n")
        L.append("* **Sampling**: 16 SAR overpasses (8 summer Mar-Jul, 8 winter Oct-Feb), like "
                 "DAM6 and unlike DAM3 (25, summer only); ζ from 5 SHM recordings per campaign. "
                 "Net: DAM4 is the state where the two SAR channels disagree — deepest "
                 "modulation, weakest and least separable γ² — which is the signature of a "
                 "response channel dominated by amplitude/mask bookkeeping rather than by ζ.\n")
    L.append("\n## Verdict\n")
    for v in result["verdict"]:
        L.append(v + "\n")
    zv = ", ".join(f"{r['state']} {r['zeta_H1_healthy']:.5f}→{r['zeta_H1_damaged']:.5f} "
                   f"(Δ {r['zeta_H1_change']:+.5f}, δ {r['zeta_H1_cliffs_delta']:+.2f})"
                   for r in result["rows"])
    fv = ", ".join(f"{r['state']} Δ {r['zeta_fund_change']:+.5f} "
                   f"(δ {r['zeta_fund_cliffs_delta']:+.2f})" for r in result["rows"])
    L.append(f"* **Direct damping evidence is limited**: the half-power ζ of the 1st higher "
             f"mode (H1) DECREASES in ALL THREE damaged states — {zv} — i.e. narrower PSD peaks "
             "in the damaged recordings, at a similar relative loss in DAM3/DAM4 and a somewhat "
             "larger one in DAM6 even though the three states differ strongly in stiffness "
             f"(fundamental mode: {fv}). Under ambient wind excitation the half-power width is "
             "biased by the excitation spectrum, and the SHM shows a strongly SUPPRESSED wind "
             "response in DAM3, so the apparent ζ decrease may be excitation-driven rather than "
             "a true damping change — and it cannot rank damage severity either, because the "
             "state with the lowest damaged ζ(H1) (DAM4) is also the one with the weakest and "
             "least separable SAR response.\n")
    L.append("* **SAR sees the VIBRATION RESPONSE, not ζ itself**: γ² (net amplitude) and "
             "sub-aperture modulation separate the states in the direction predicted by the SHM "
             "amplitude at common wind wherever that amplitude is measured — DAM3 suppressed "
             "(ratio 0.38) → shallower modulation, DAM6 amplified (2.05) → deeper modulation "
             "plus the only γ² drop that survives a fixed-N control — while DAM4 has no wind "
             "recordings at all and therefore cannot be checked on this axis. This makes the SAR "
             "channels indirect *response/damping-coupled* sensors, but they cannot isolate ζ "
             "without a wind-referenced amplitude model (the `A=c·v^k` suppression ratio is that "
             "model's output and is the only channel separating both directions).\n")
    L.append("* **Honest limits**: n=5 SHM recordings per state; one campaign per state (season "
             "confound — DAM3 is a single-season contrast and the only state whose γ² sign flips "
             "between seasons); single-look γ² floor ≈ 0.04 limits DAM3 sensitivity; H2 mode in "
             "DAM3 not cleanly identifiable (weak/scattered peaks) — excluded from the "
             "comparison; DAM4 has no wind-response recordings and n=16 SAR overpasses (same as "
             "DAM6), so its γ² Welch test has little power (p=0.714 is weak evidence of no "
             "effect, not proof of it); DAM4 is drawn from the same ζ/γ²/modulation sources as "
             "DAM3/DAM6 but its campaign (`03_Healthy`/`04_DAM4_111`) covers a different season "
             "window.\n")
    with open(base + ".md", "w") as f:
        f.writelines(L)


def plot(rows, base):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    states = [r["state"] for r in rows]
    # Panel 1 is Δζ: each state has its OWN healthy campaign (05_/03_/01_Healthy), so absolute
    # ζ would draw a healthy "trend" that is nothing but baseline drift between campaigns.
    x = np.arange(len(rows))
    axes[0].axhline(0.0, color="tab:blue", ls="--", lw=1.2, label="healthy (own campaign)")
    axes[0].plot(x, [0.0] * len(rows), "o", color="tab:blue")
    axes[0].plot(x, [r["zeta_H1_change"] for r in rows], "s-", color="tab:orange", label="damaged")
    for xi, r in zip(x, rows):
        axes[0].annotate(r["healthy_campaign"], (xi, 0.0), textcoords="offset points",
                         xytext=(0, 8), ha="center", fontsize=7, color="tab:blue")
        axes[0].annotate(f"{r['zeta_H1_damaged']:.5f}  (δ {r['zeta_H1_cliffs_delta']:+.2f})",
                         (xi, r["zeta_H1_change"]), textcoords="offset points",
                         xytext=(0, 8), ha="center", fontsize=7, color="tab:orange")
    lo = min(r["zeta_H1_change"] for r in rows)
    axes[0].set_ylim(lo - 0.00040, 0.00030)   # headroom for labels + clear legend strip
    axes[0].set_title("Δζ H1 (half-power, vs own healthy)")
    axes[0].set_xticks(x); axes[0].set_xticklabels(states)
    axes[0].legend(loc="lower center", fontsize=8, framealpha=0.9)
    axes[1].plot(x, [r["gamma2_healthy"] for r in rows], "o-", label="healthy")
    axes[1].plot(x, [r["gamma2_damaged"] for r in rows], "s-", label="damaged")
    axes[1].set_title("γ² (whole-tower coherence)")
    axes[1].set_xticks(x); axes[1].set_xticklabels(states)
    axes[1].legend()
    axes[2].plot(x, [r["modulation_healthy"] for r in rows], "o-", label="healthy")
    axes[2].plot(x, [r["modulation_damaged"] for r in rows], "s-", label="damaged")
    axes[2].set_title("sub-aperture modulation")
    axes[2].set_xticks(x); axes[2].set_xticklabels(states)
    axes[2].legend()
    fig.suptitle("LUMO — damping ζ vs SAR γ² vs modulation by damage state")
    fig.tight_layout()
    fig.savefig(base + ".png", dpi=120)


if __name__ == "__main__":
    main()

