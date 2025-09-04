#!/bin/sh
# Build the Docker image for the Menace sandbox
repo_root=$(python - <<'PY'
from dynamic_path_router import repo_root
print(repo_root())
PY
)
image_name="menace_sandbox"
exec docker build -t "$image_name" "$repo_root"
