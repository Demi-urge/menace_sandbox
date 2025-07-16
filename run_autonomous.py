from __future__ import annotations

"""Wrapper for running the autonomous sandbox loop after dependency checks."""

import argparse
import json
import logging
import os
import shutil
from pathlib import Path
from typing import List
import sys

# allow execution directly from the package directory
_pkg_dir = Path(__file__).resolve().parent
if _pkg_dir.name == "menace" and str(_pkg_dir.parent) not in sys.path:
    sys.path.insert(0, str(_pkg_dir.parent))
elif "menace" not in sys.modules:
    import types

    menace_pkg = types.ModuleType("menace")
    menace_pkg.__path__ = [str(_pkg_dir)]
    sys.modules["menace"] = menace_pkg

from menace.environment_generator import generate_presets
from menace.startup_checks import verify_project_dependencies
from sandbox_runner.cli import full_autonomous_run
from sandbox_recovery_manager import SandboxRecoveryManager
import sandbox_runner

if not hasattr(sandbox_runner, "_sandbox_main"):
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "sandbox_runner", _pkg_dir / "sandbox_runner.py"
    )
    sr_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sr_mod)
    sandbox_runner = sys.modules["sandbox_runner"] = sr_mod

logger = logging.getLogger(__name__)


def _check_dependencies() -> None:
    """Log missing optional runtime dependencies."""
    missing: List[str] = []

    if shutil.which("docker") is None:
        missing.append("docker")
    try:  # pragma: no cover - optional
        import docker  # type: ignore
    except Exception:
        missing.append("docker python package")

    if shutil.which("qemu-system-x86_64") is None:
        missing.append("qemu-system-x86_64")

    missing_pkgs = verify_project_dependencies()
    if missing_pkgs:
        missing.extend(missing_pkgs)

    if missing:
        logger.warning("Missing dependencies: %s", ", ".join(missing))
    else:
        logger.info("All dependencies satisfied")


def main(argv: List[str] | None = None) -> None:
    """Entry point for the autonomous runner."""
    parser = argparse.ArgumentParser(
        description="Run full autonomous sandbox with environment presets",
    )
    parser.add_argument(
        "--preset-count",
        type=int,
        default=3,
        help="number of presets per iteration",
    )
    parser.add_argument("--max-iterations", type=int, help="maximum iterations")
    parser.add_argument("--sandbox-data-dir", help="override sandbox data directory")
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="number of full sandbox runs to execute sequentially",
    )
    parser.add_argument(
        "--dashboard-port",
        type=int,
        help="start MetricsDashboard on this port for each run",
    )
    parser.add_argument(
        "--roi-cycles",
        type=int,
        default=3,
        help="cycles below threshold before module convergence",
    )
    parser.add_argument(
        "--synergy-cycles",
        type=int,
        default=3,
        help="cycles below threshold before synergy convergence",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO)
    _check_dependencies()

    if args.dashboard_port:
        from metrics_dashboard import MetricsDashboard
        from threading import Thread

        history_file = (
            Path(args.sandbox_data_dir or "sandbox_data") / "roi_history.json"
        )
        dash = MetricsDashboard(str(history_file))
        Thread(
            target=dash.run,
            kwargs={"port": args.dashboard_port},
            daemon=True,
        ).start()

    for idx in range(args.runs):
        logger.info("Starting autonomous run %d/%d", idx + 1, args.runs)
        presets = generate_presets(args.preset_count)
        os.environ["SANDBOX_ENV_PRESETS"] = json.dumps(presets)

        recovery = SandboxRecoveryManager(sandbox_runner._sandbox_main)
        sandbox_runner._sandbox_main = recovery.run
        try:
            full_autonomous_run(args)
        finally:
            sandbox_runner._sandbox_main = recovery.sandbox_main


if __name__ == "__main__":
    main()
