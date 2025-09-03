#!/bin/sh
# Ensure dependencies are installed and environment is set up
"$(dirname "$0")/../setup_env.sh"

# Launch sandbox runner with metrics dashboard
python $(python -c 'from dynamic_path_router import resolve_path; print(resolve_path("sandbox_runner.py"))') full-autonomous-run --preset-count 3 --dashboard-port 8002 "$@"

