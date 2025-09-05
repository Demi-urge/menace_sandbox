#!/usr/bin/env bash
set -euo pipefail

# Ensure repository root on PYTHONPATH so dynamic_path_router is importable
script_dir="$(cd "$(dirname "$0")" && pwd -P)"
repo_root="$(cd "$script_dir/.." && pwd -P)"
export PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}"

# Run base Python environment setup
setup_env=$(python - <<'PY'
from dynamic_path_router import resolve_path
print(resolve_path('setup_env.sh'))
PY
)

(cd "$repo_root" && bash "$setup_env")

# Install and verify project dependencies
(cd "$repo_root" && python setup_dependencies.py)

# Create a .env file with sensible defaults
python - <<'PY'
import auto_env_setup as a
a.ensure_env()
PY

# Verify presence of optional system tools
missing=()
for t in ffmpeg tesseract docker; do
    command -v "$t" >/dev/null 2>&1 || missing+=("$t")
done
# qemu may use different binary names
if ! command -v qemu-system-x86_64 >/dev/null 2>&1 && \
   ! command -v qemu-system-x86 >/dev/null 2>&1; then
    missing+=("qemu-system-x86_64")
fi
if [[ ${#missing[@]} -ne 0 ]]; then
    echo "Missing optional tools: ${missing[*]}" >&2
    exit 1
fi

echo "Setup completed successfully"

