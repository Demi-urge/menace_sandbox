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
: "${MENACE_BOOTSTRAP_WAIT_SECS:=240}"
: "${MENACE_BOOTSTRAP_VECTOR_WAIT_SECS:=240}"
: "${BOOTSTRAP_STEP_TIMEOUT:=240}"
: "${BOOTSTRAP_VECTOR_STEP_TIMEOUT:=240}"
export MENACE_BOOTSTRAP_WAIT_SECS
export MENACE_BOOTSTRAP_VECTOR_WAIT_SECS
export BOOTSTRAP_STEP_TIMEOUT
export BOOTSTRAP_VECTOR_STEP_TIMEOUT

# Execute run_autonomous after ensuring the environment is configured
exec python - "$@" <<'PYCODE'
import importlib
import importlib.util
import os
import pathlib
import sys

repo_root = pathlib.Path(os.environ["REPO_ROOT"])

# Load the menace package from the repository root
spec = importlib.util.spec_from_file_location("menace", repo_root / "__init__.py")
menace_pkg = importlib.util.module_from_spec(spec)
sys.modules["menace"] = menace_pkg
spec.loader.exec_module(menace_pkg)

auto_env_setup = importlib.import_module("menace.auto_env_setup")
run_autonomous = importlib.import_module("run_autonomous")

auto_env_setup.ensure_env()
run_autonomous.main(sys.argv[1:])
PYCODE
