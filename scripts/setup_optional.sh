#!/usr/bin/env bash
set -euo pipefail

# Install optional packages used by the reinforcement learning
# components and the metrics dashboard.
offline="${MENACE_OFFLINE_INSTALL:-0}"
wheel_dir="${MENACE_WHEEL_DIR:-}"

pip_opts=()
if [[ "$offline" = "1" && -n "$wheel_dir" ]]; then
    pip_opts+=(--no-index --find-links "$wheel_dir")
fi

python -m pip install --upgrade pip
pip install --no-cache-dir "${pip_opts[@]}" torch statsmodels scipy matplotlib
