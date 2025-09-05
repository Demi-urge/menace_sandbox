#!/bin/bash
set -e

# Ensure repository root on PYTHONPATH so dynamic_path_router is importable
script_dir="$(cd "$(dirname "$0")" && pwd -P)"
repo_root="$(cd "$script_dir/.." && pwd -P)"
export PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}"

# Install pinned Python dependencies for Menace Sandbox.
# Locate requirements.txt dynamically so the script works from any directory.

req_file=$(python - <<'PY'
from dynamic_path_router import resolve_path
print(resolve_path('requirements.txt'))
PY
)

python -m pip install --upgrade pip
python -m pip install -r "$req_file"
