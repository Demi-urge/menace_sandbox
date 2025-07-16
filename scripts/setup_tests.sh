#!/bin/bash
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
