#!/bin/bash
set -e

# Ensure repository root on PYTHONPATH so dynamic_path_router is importable
script_dir="$(cd "$(dirname "$0")" && pwd -P)"
repo_root="$(cd "$script_dir/.." && pwd -P)"
export PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}"

# Install system packages unless offline mode is enabled
offline="${MENACE_OFFLINE_INSTALL:-0}"
wheel_dir="${MENACE_WHEEL_DIR:-}"

if [ "$offline" != "1" ] && command -v apt-get >/dev/null 2>&1; then
    sudo apt-get update
    sudo apt-get install -y docker.io qemu-system-x86 git ffmpeg tesseract-ocr
fi

pip_opts=()
if [ "$offline" = "1" ] && [ -n "$wheel_dir" ]; then
    pip_opts+=(--no-index --find-links "$wheel_dir")
fi

python -m pip install --upgrade pip
pip install "${pip_opts[@]}" -e "$repo_root"
# Install pytest as run_autonomous.py expects
pip install "${pip_opts[@]}" pytest

