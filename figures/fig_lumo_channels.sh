#!/usr/bin/env bash
# Runs code/fig_lumo_channels.py from the repository root, so its defaults
# (../data/lumo_channels.csv in, ../data/fig_lumo_channels.json and
# ../figures/fig_lumo_channels.md/.png out) resolve correctly.
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec python3 code/fig_lumo_channels.py "$@"
