#!/bin/bash
set -e

# Install pinned Python dependencies for Menace Sandbox.
# Execute this script from the repository root.

if [ ! -f requirements.txt ]; then
    echo "requirements.txt not found. Run this script from the repository root." >&2
    exit 1
fi

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
