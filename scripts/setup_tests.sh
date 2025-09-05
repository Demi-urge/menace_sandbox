#!/usr/bin/env bash
set -euo pipefail

# Ensure repository root on PYTHONPATH so dynamic_path_router is importable
script_dir="$(cd "$(dirname "$0")" && pwd -P)"
repo_root="$(cd "$script_dir/.." && pwd -P)"
export PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}"

# Ensure pip is up to date
python -m pip install --upgrade pip

# Install pinned runtime dependencies
req_file=$(python - <<'PY'
from dynamic_path_router import resolve_path
print(resolve_path('requirements.txt'))
PY
)
pip install --no-cache-dir -r "$req_file"

# Install the package itself in editable mode
pip install --no-cache-dir -e "$repo_root"

# Install core testing utilities
pip install --no-cache-dir pytest hypothesis

# Explicitly install packages that tests stub
pip install --no-cache-dir jinja2==3.1.6 sqlalchemy==2.0.41

# Verify and install additional test requirements
for pkg in requests boto3 pandas; do
    if ! python - "$pkg" >/dev/null 2>&1 <<'EOF'; then
import importlib, sys
pkg = sys.argv[1]
sys.exit(0 if importlib.util.find_spec(pkg) else 1)
EOF
        echo "Installing missing package: $pkg" >&2
        pip install --no-cache-dir "$pkg" || {
            echo "Failed to install required package: $pkg" >&2
            exit 1
        }
    fi
done
