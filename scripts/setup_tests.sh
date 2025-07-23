#!/usr/bin/env bash
set -euo pipefail

# Ensure pip is up to date
python -m pip install --upgrade pip

# Install pinned runtime dependencies
pip install --no-cache-dir -r requirements.txt

# Install the package itself in editable mode
pip install --no-cache-dir -e .

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
