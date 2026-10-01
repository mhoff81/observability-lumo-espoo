#!/usr/bin/env bash
# Runs code/lumo_damping_gamma2_modulation.py from the repository root, so its
# defaults (../data/*.json in and out, ../figures/lumo_damping_gamma2_modulation.md/.png
# out) resolve correctly.
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec python3 code/lumo_damping_gamma2_modulation.py "$@"
