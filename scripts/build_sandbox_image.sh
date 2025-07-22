#!/bin/sh
# Build the Docker image for the Menace sandbox
repo_root="$(cd "$(dirname "$0")/.." && pwd)"
image_name="menace_sandbox"
exec docker build -t "$image_name" "$repo_root"
