# SLC burst cache — Audit Sweep Summary

Every tenth entry of the cache: 149 audited entries in 35 month folders, acquisitions 2018-01-01 .. 2021-07-28, sub-swaths iw1 79, iw2 70, polarisations vh 64, vv 85, window 1681 samples, 26896 B.

From `audit.csv`, the `--csv` artefact of `code/lumo_burst_cache_audit.py --stride 10 --strip-every 10`. Every number below is a count or a summary of that file's own columns.

| verdict | n |
|---------|--:|
| unusable | 0 |
| no-rule | 0 |
| ambiguous | 0 |
| wrong-rule | 0 |
| off-target | 8 |
| consistent | 141 |

| vintage | entries | expected rule | window: held / other (median fraction) | anchor delta (median line) | off-target |
|---------|--------:|---------------|------------------------------|-----------------------:|-----------:|
| pre-fad0e46 | 52 | legacy (52/52 rows) | 1.000 / 0.000 | 85.5 | 8 |
| post-fad0e46 | 97 | current (97/97 rows) | 1.000 / 0.000 | 853.0 | 0 |

## Containment

| split | n | nearest grid node [km]: min / median / max |
|-------|--:|-------------------------------------------:|
| mast inside the burst's grid | 141 | 3.747 / 4.611 / 9.880 |
| mast outside it | 8 | 62.249 / 63.960 / 65.519 |

The 8 off-target entries (nearest node beyond the threshold, 10 km) come from pre-fad0e46 8 and sit 62.249 .. 65.519 km out: `2020-09/2020-09-14_60f0f0cf`, `2020-11/2020-11-01_c0eb272f`, `2020-12/2020-12-01_43b203ee`, `2020-12/2020-12-31_cb7e4907`, `2021-01/2021-01-24_2a6f1722`, `2021-03/2021-03-01_22752f50`, `2021-03/2021-03-25_1d955b37`, `2021-06/2021-06-17_4766a775`.

## Ground error — distance between each rule's anchor and the mast

| vintage | rule | role | n | anchors at 0.0 m | median [m] | max [m] |
|---------|------|------|--:|-----------------:|-----------:|--------:|
| pre-fad0e46 | legacy | the vintage's rule | 52 | 0 | 11372.6 | 69510.7 |
| pre-fad0e46 | current | the other rule | 52 | 44 | 0.0 | 65470.4 |
| post-fad0e46 | current | the vintage's rule | 97 | 97 | 0.0 | 0.0 |
| post-fad0e46 | legacy | the other rule | 97 | 0 | 11361.6 | 23074.2 |

## Decoded strips

14 of the 149 entries were also decoded over their full strip (4400 samples), reproduced by rule `current` in 9, `legacy` in 5: all at fraction 1.000000, 14 of 14 exact in full and 0 partial — the strip agrees with the window it was cut from.

## Reading

- Every window file is reproduced bit for bit by exactly one rule, the one its vintage predicts (149 of 149 entries at fraction 1.0, and the other rule in none of them — it reaches 1.0 in 0 entries). Pooled over the sample: `legacy` 1.0 in 52 entries, 0.0 in 94, in between in 3; `current` 1.0 in 97, 0.0 in 44, in between in 8. Those in-between entries are the near-misses a rule test on one entry alone would misread as a contradiction.
- No entry is unusable, unmatched, ambiguous or reproduced by the other vintage's rule: verdict flags unusable=0, no_rule=0, ambiguous=0, wrong_rule=0, off_target=8, outside_grid=8, matched_expected=149.
- The threshold the audit applied is recoverable from its own two columns: `nearest_node_km - off_target_margin_km` gives 10 km for every entry, the `--max-node-km` default (one grid step in range is ~10 km).
- The fix shows up in the anchors too: the pre-fix entries' bytes match `legacy`, whose anchor sits a median 11372.6 m from the mast, while the post-fix rule lands on it (97 of 97 entries at 0.0 m). The drift between the two anchors is 85.5 lines (pre) and 853.0 lines (post).
- The vintage is read from the mtime of each `window.bin` (2026-08-24 → 52 entries, 2026-09-28 → 97 entries), never from the acquisition date: the two timelines are years apart on purpose.

## Caveats

- A sample, not a census: `--stride 10` audits one entry in ten, and only 14 of those also had their strip decoded. Nothing here is a statement about the whole cache — the full sweep is `--stride 1`.
- The time-series panels' x axis is `acquisition_ts`, the physical timeline. `window_mtime` is when the cache wrote the file and is the only surviving evidence of the vintage, so an entry whose mtime was copied or restored would be mis-vintaged.
- A small ground error is the size of the clamp error a rule leaves behind, not proof of a correct mapping: the verdict comes from the decoded window bytes, and the distance is reported next to it, never instead of it.
- This figure adds no analysis of its own: it counts and plots the audit's columns, offline, reading no cache directory and no network.

Build: `python3 fig_burst_cache_audit.py` (reads `../data/audit.csv` by default; `--csv` and `--out-dir` point it at another sweep or another directory).
