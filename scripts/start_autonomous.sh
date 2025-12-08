#!/bin/sh
# Bootstrap PYTHONPATH so the script works from any directory
# Resolve repository root so imports work regardless of the CWD
repo_root=$(python - <<'PY'
from dynamic_path_router import repo_root
print(repo_root())
PY
)
export PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}"
export REPO_ROOT="$repo_root"

# Ensure bootstrap timeouts are exported before any Python entry point executes
# Policy floors: 720s for standard bootstrap, 900s for vector paths
: "${MENACE_BOOTSTRAP_WAIT_SECS:=720}"
: "${MENACE_BOOTSTRAP_VECTOR_WAIT_SECS:=900}"
: "${BOOTSTRAP_STEP_TIMEOUT:=720}"
: "${BOOTSTRAP_VECTOR_STEP_TIMEOUT:=900}"
: "${MENACE_BOOTSTRAP_STAGGER_SECS:=30}"
: "${MENACE_BOOTSTRAP_STAGGER_JITTER_SECS:=30}"
export MENACE_BOOTSTRAP_WAIT_SECS
export MENACE_BOOTSTRAP_VECTOR_WAIT_SECS
export BOOTSTRAP_STEP_TIMEOUT
export BOOTSTRAP_VECTOR_STEP_TIMEOUT
export MENACE_BOOTSTRAP_STAGGER_SECS
export MENACE_BOOTSTRAP_STAGGER_JITTER_SECS

# Execute run_autonomous after ensuring the environment is configured
python - <<'PYCODE'
import logging

from bootstrap_conflict_check import (
    enforce_conflict_free_environment,
    enforce_timeout_floor_envs,
)
from menace_sandbox.auto_env_setup import ensure_env


logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
conflict_logger = logging.getLogger("bootstrap.conflict-check")
enforce_timeout_floor_envs(logger=conflict_logger)
enforce_conflict_free_environment(logger=conflict_logger)
ensure_env()
PYCODE

exec python -m menace_sandbox.run_autonomous "$@"
