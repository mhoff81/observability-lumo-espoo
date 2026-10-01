#!/usr/bin/env bash
# Runs code/fig_espoo_channels.py from the repository root, so its defaults
# (../data/espoo_channels.csv in, ../data/fig_espoo_channels.json and
# ../figures/fig_espoo_channels.md/.png out) resolve correctly.
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec python3 code/fig_espoo_channels.py "$@"
