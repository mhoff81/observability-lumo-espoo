#!/usr/bin/env bash
# Runs code/fig_burst_cache_audit.py from the repository root, so its
# defaults (../data/audit.csv in, ../data/fig_burst_cache_audit.json and
# ../figures/fig_burst_cache_audit.md/.png out) resolve correctly.
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec python3 code/fig_burst_cache_audit.py "$@"
