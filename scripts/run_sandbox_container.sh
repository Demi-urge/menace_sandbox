#!/bin/sh
# Run the Menace sandbox Docker image with persistent data
repo_root="$(cd "$(dirname "$0")/.." && pwd)"
image_name="${IMAGE:-menace_sandbox}"

docker run --rm -it \
  --env-file "$repo_root/.env" \
  -v "$repo_root/sandbox_data:/app/sandbox_data" \
  "$image_name" "$@"
