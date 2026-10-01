# Espoo Kurttila mast — Observation-Channel Time Histories

150 decoded acquisitions, 2024-09-05 .. 2026-09-07. No damage states exist for this site, so no campaign windows are shaded; the mid-2025 S1C availability step-up is marked instead.

Source: espoo_channels.csv

| channel | n | median | DESCENDING (median) | ASCENDING (median) |
|---------|--:|-------:|--------------------:|-------------------:|
| coherence_gamma2 | 150 | 0.06244 | 0.02575 | 0.1746 |
| coherence_masked_pixels | 150 | 8 | 11 | 6 |
| phase_coherence | 80 | 0.1412 | 0.1407 | 0.1459 |
| phase_snr_db | 0 | - | - | - |
| phase_rms_rad | 80 | 5.327 | 5.436 | 5.24 |

## Reading the figure

* **Panel 1 vs the dashed lines** is the scale check: the LUMO reference medians are 0.0617 (ASC) and 0.0261 (DESC). Espoo ASC sits well above its reference; Espoo DESC sits below its reference.
* **Panel 2** carries the diagnostic weight. Low γ² with a large mask is clutter; high γ² with a small mask is a real point scatterer. Do not read panel 1 without panel 2.
* **Panels 3–5** are the tower phase family. Negative SNR with phase RMS near π (≈3.14 rad = fully decorrelated) is the signature of a non-observable phase channel — which is what the pipeline concluded here (`phase_observable = false`, reason `low_coherence`).
* **Panel 4 stops where panels 3 and 5 stop.** `phase_snr_db` is DB-enriched and is never empty the way `phase_coherence`/`phase_rms_rad` are: once the phase channel goes unobservable the backend still writes `snr_db_from_coherence(gamma)` with gamma clamped to a 1e-6 floor (`fusion::snr_db_from_coherence`), a fixed ≈-60 dB value, not a measurement. This figure drops those floor-clamped rows (gated on `phase_coherence` being populated) instead of drawing a flat tail.
* **Panel 6** is data availability, not a measurement channel: decoded acquisitions per month, stacked by orbit. It is what panel 1's reading depends on — the pass/orbit balance it shows is the same step-up marked on every panel.

## Caveats

* No ground-truth state label → observability evidence only, never a condition assessment.
* 37 OSM building ways lie within 100 m of the mast — the built-up Kurttila/Saunalahti surroundings, not open ground.
* Pass/orbit balance changes mid-2025 (S1C); the step-up is drawn as a guide line on every panel, and it is a real confound for any trend reading.
* The record is still growing while the 2-year backfill runs; re-run to refresh.
* `phase_snr_db` rows where the phase channel is unobservable (empty `phase_coherence`) are excluded from panel 4 and its stats — they are a fixed clamp-floor value (≈-60 dB), not a varying measurement.
* **The LUMO intra-dwell brightness modulation (`sub_aperture_modulation`) is absent for this site by construction**: the stored window is 7 x 7 px and the helper returns `None` for `height < n_sub` (`src/asset_onboarder/insar_monitor.rs:243`), so 8 intra-dwell blocks cannot be formed. Reproducing the LUMO dwell-modulation channel here would need a full-dwell strip capture (11 columns, ~0.8 s), not a re-run of the window backfill.

Build: `python3 fig_espoo_channels.py`.
