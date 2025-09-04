#!/bin/sh
# Run the Menace sandbox Docker image with persistent data
env_file=$(python - <<'PY'
from dynamic_path_router import resolve_path
print(resolve_path('.env'))
PY
)
data_dir=$(python - <<'PY'
from dynamic_path_router import resolve_path
print(resolve_path('sandbox_data'))
PY
)
image_name="${IMAGE:-menace_sandbox}"

docker run --rm -it \
  --env-file "$env_file" \
  -v "$data_dir:/app/sandbox_data" \
  "$image_name" "$@"
