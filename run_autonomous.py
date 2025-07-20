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
import subprocess
import time
import importlib

# Default to test mode when using the bundled SQLite database.
if os.getenv("MENACE_MODE", "test").lower() == "production" and os.getenv(
    "DATABASE_URL", ""
).startswith("sqlite"):
    logging.warning(
        "MENACE_MODE=production with SQLite database; switching to test mode"
    )
    os.environ["MENACE_MODE"] = "test"

# allow execution directly from the package directory
_pkg_dir = Path(__file__).resolve().parent
if _pkg_dir.name == "menace" and str(_pkg_dir.parent) not in sys.path:
    sys.path.insert(0, str(_pkg_dir.parent))
elif "menace" not in sys.modules:
    import importlib.util

    spec = importlib.util.spec_from_file_location("menace", _pkg_dir / "__init__.py")
    menace_pkg = importlib.util.module_from_spec(spec)
    sys.modules["menace"] = menace_pkg
    spec.loader.exec_module(menace_pkg)

from menace.auto_env_setup import ensure_env

from menace.environment_generator import generate_presets
from menace.startup_checks import verify_project_dependencies
from menace.dependency_installer import install_packages
import sandbox_runner.cli as cli
from sandbox_runner.cli import full_autonomous_run
from menace.roi_tracker import ROITracker
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
    """Install missing project dependencies and warn about binaries."""
    missing: List[str] = []

    if sys.version_info[:2] < (3, 10):
        msg = (
            f"Python >=3.10 required, found {sys.version_info.major}.{sys.version_info.minor}"
        )
        logger.error(msg)
        raise RuntimeError(msg)

    if shutil.which("docker") is None:
        logger.warning(
            "Docker not found. Install using: sudo apt-get install docker.io"
        )
        missing.append("docker")
    try:  # pragma: no cover - optional
        import docker  # type: ignore
    except Exception:
        missing.append("docker python package")

    if shutil.which("qemu-system-x86_64") is None:
        logger.warning(
            "qemu-system-x86_64 not found. Install using: sudo apt-get install qemu-system-x86"
        )
        missing.append("qemu-system-x86_64")

    if shutil.which("git") is None:
        logger.warning("git not found. Install using your system package manager")
        missing.append("git")

    if shutil.which("pytest") is None:
        logger.warning(
            "pytest not found. Install using: pip install pytest or system package"
        )
        missing.append("pytest")

    # verify docker group membership if docker command exists
    if shutil.which("docker") is not None:
        try:  # pragma: no cover - optional
            import grp
            import getpass

            user = getpass.getuser()
            docker_grp = grp.getgrnam("docker")
            if user not in docker_grp.gr_mem and os.getgid() != docker_grp.gr_gid:
                logger.warning("User %s not in docker group", user)
                missing.append("docker group")
        except Exception:  # pragma: no cover - platform dependent
            logger.warning("Unable to verify docker group membership")
            missing.append("docker group")

    missing_pkgs = verify_project_dependencies()
    if missing_pkgs:
        offline = os.getenv("MENACE_OFFLINE_INSTALL", "0") == "1"
        errors = install_packages(missing_pkgs, offline=offline)
        if offline and missing_pkgs:
            logger.info(
                "offline install mode enabled; skipping installation for: %s",
                ", ".join(missing_pkgs),
            )
        for pkg, err in errors.items():
            logger.error("failed installing %s: %s", pkg, err)
        failed_pkgs = [p for p in missing_pkgs if p in errors]
        if failed_pkgs:
            missing.extend(failed_pkgs)

    if missing:
        logger.error("Missing dependencies: %s", ", ".join(missing))
        raise RuntimeError("dependency installation failed")
    else:
        logger.info("All dependencies satisfied")


def _get_env_override(name: str, current):
    """Return parsed environment variable when ``current`` is ``None``."""
    env_val = os.getenv(name)
    if current is not None or env_val is None:
        return current
    try:
        if isinstance(current, int):
            return int(env_val)
        if isinstance(current, float):
            return float(env_val)
    except Exception:
        return None
    for cast in (int, float):
        try:
            return cast(env_val)
        except Exception:
            continue
    return None


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
        help="maximum number of full sandbox runs to execute",
    )
    parser.add_argument(
        "--dashboard-port",
        type=int,
        help=(
            "start MetricsDashboard on this port for each run"
            " (overrides AUTO_DASHBOARD_PORT)"
        ),
    )
    parser.add_argument(
        "--roi-cycles",
        type=int,
        default=3,
        help="cycles below threshold before module convergence",
    )
    parser.add_argument(
        "--roi-threshold",
        type=float,
        help="override ROI delta threshold",
    )
    parser.add_argument(
        "--roi-confidence",
        type=float,
        help="confidence level for ROI convergence",
    )
    parser.add_argument(
        "--synergy-cycles",
        type=int,
        default=3,
        help="cycles below threshold before synergy convergence",
    )
    parser.add_argument(
        "--synergy-threshold",
        type=float,
        help="override synergy threshold",
    )
    parser.add_argument(
        "--synergy-threshold-window",
        type=int,
        help="window size for adaptive synergy threshold",
    )
    parser.add_argument(
        "--synergy-threshold-weight",
        type=float,
        help="exponential weight for adaptive synergy threshold",
    )
    parser.add_argument(
        "--synergy-confidence",
        type=float,
        help="confidence level for synergy convergence",
    )
    parser.add_argument(
        "--synergy-ma-window",
        type=int,
        help="window size for synergy moving average",
    )
    parser.add_argument(
        "--synergy-stationarity-confidence",
        type=float,
        help="confidence level for synergy stationarity test",
    )
    parser.add_argument(
        "--synergy-std-threshold",
        type=float,
        help="standard deviation threshold for synergy convergence",
    )
    parser.add_argument(
        "--synergy-variance-confidence",
        type=float,
        help="confidence level for variance change test",
    )
    parser.add_argument(
        "--auto-thresholds",
        action="store_true",
        help="compute convergence thresholds adaptively",
    )
    parser.add_argument(
        "--preset-file",
        action="append",
        dest="preset_files",
        help="JSON file defining environment presets; can be repeated",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO)

    env_file = Path(os.getenv("MENACE_ENV_FILE", ".env"))
    created_env = not env_file.exists()
    ensure_env(str(env_file))
    if created_env:
        logger.info("created env file at %s", env_file)

    if args.preset_files is None:
        data_dir = Path(
            args.sandbox_data_dir
            or os.getenv("SANDBOX_DATA_DIR", "sandbox_data")
        )
        preset_file = data_dir / "presets.json"
        created_preset = False
        env_val = os.getenv("SANDBOX_ENV_PRESETS")
        if env_val:
            try:
                presets = json.loads(env_val)
                if isinstance(presets, dict):
                    presets = [presets]
            except Exception:
                presets = generate_presets(args.preset_count)
                os.environ["SANDBOX_ENV_PRESETS"] = json.dumps(presets)
        elif preset_file.exists():
            try:
                presets = json.loads(preset_file.read_text())
                if isinstance(presets, dict):
                    presets = [presets]
            except Exception:
                presets = generate_presets(args.preset_count)
        else:
            presets = generate_presets(args.preset_count)
        os.environ["SANDBOX_ENV_PRESETS"] = json.dumps(presets)
        if not preset_file.exists():
            data_dir.mkdir(parents=True, exist_ok=True)
            preset_file.write_text(json.dumps(presets))
            created_preset = True
        args.preset_files = [str(preset_file)]
        if created_preset:
            logger.info("created preset file at %s", preset_file)

    _check_dependencies()

    dash_port = args.dashboard_port
    if dash_port is None:
        env_val = os.getenv("AUTO_DASHBOARD_PORT")
        if env_val is not None:
            try:
                dash_port = int(env_val)
            except Exception:
                dash_port = None

    if dash_port:
        from menace.metrics_dashboard import MetricsDashboard
        from threading import Thread

        history_file = (
            Path(args.sandbox_data_dir or "sandbox_data") / "roi_history.json"
        )
        dash = MetricsDashboard(str(history_file))
        Thread(
            target=dash.run,
            kwargs={"port": dash_port},
            daemon=True,
        ).start()

    agent_proc = None
    autostart = os.getenv("VISUAL_AGENT_AUTOSTART", "1") != "0"
    if autostart:
        try:
            import requests  # type: ignore

            base = (os.getenv("VISUAL_AGENT_URLS", "http://127.0.0.1:8001")).split(";")[
                0
            ]
            resp = requests.get(f"{base}/status", timeout=3)
            if resp.status_code != 200:
                raise RuntimeError(resp.text)
        except Exception:
            cmd = [sys.executable, str(_pkg_dir / "menace_visual_agent_2.py")]
            agent_proc = subprocess.Popen(cmd)
            time.sleep(2)

    module_history: dict[str, list[float]] = {}
    flagged: set[str] = set()
    synergy_history: list[dict[str, float]] = []
    roi_ma_history: list[float] = []
    synergy_ma_history: list[dict[str, float]] = []
    roi_threshold = _get_env_override("ROI_THRESHOLD", args.roi_threshold)
    synergy_threshold = _get_env_override("SYNERGY_THRESHOLD", args.synergy_threshold)
    roi_confidence = _get_env_override("ROI_CONFIDENCE", args.roi_confidence)
    synergy_confidence = _get_env_override("SYNERGY_CONFIDENCE", args.synergy_confidence)
    synergy_threshold_window = _get_env_override(
        "SYNERGY_THRESHOLD_WINDOW", args.synergy_threshold_window
    )
    synergy_threshold_weight = _get_env_override(
        "SYNERGY_THRESHOLD_WEIGHT", args.synergy_threshold_weight
    )
    synergy_ma_window = _get_env_override("SYNERGY_MA_WINDOW", args.synergy_ma_window)
    synergy_stationarity_confidence = _get_env_override(
        "SYNERGY_STATIONARITY_CONFIDENCE", args.synergy_stationarity_confidence
    )
    synergy_std_threshold = _get_env_override(
        "SYNERGY_STD_THRESHOLD", args.synergy_std_threshold
    )
    synergy_variance_confidence = _get_env_override(
        "SYNERGY_VARIANCE_CONFIDENCE", args.synergy_variance_confidence
    )
    if synergy_threshold_window is None:
        synergy_threshold_window = args.synergy_cycles
    if synergy_threshold_weight is None:
        synergy_threshold_weight = 1.0
    if synergy_ma_window is None:
        synergy_ma_window = args.synergy_cycles
    if synergy_stationarity_confidence is None:
        synergy_stationarity_confidence = synergy_confidence or 0.95
    if synergy_std_threshold is None:
        synergy_std_threshold = 1e-3
    if synergy_variance_confidence is None:
        synergy_variance_confidence = synergy_confidence or 0.95
    last_tracker = None

    run_idx = 0
    while args.runs is None or run_idx < args.runs:
        run_idx += 1
        logger.info(
            "Starting autonomous run %d/%s",
            run_idx,
            args.runs if args.runs is not None else "?",
        )
        if args.preset_files:
            pf = Path(args.preset_files[(run_idx - 1) % len(args.preset_files)])
            with open(pf, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            presets = [data] if isinstance(data, dict) else list(data)
        else:
            presets = generate_presets(args.preset_count)
        os.environ["SANDBOX_ENV_PRESETS"] = json.dumps(presets)

        recovery = SandboxRecoveryManager(sandbox_runner._sandbox_main)
        sandbox_runner._sandbox_main = recovery.run
        try:
            full_autonomous_run(args)
        finally:
            sandbox_runner._sandbox_main = recovery.sandbox_main

        hist_file = Path(args.sandbox_data_dir or "sandbox_data") / "roi_history.json"
        tracker = ROITracker()
        try:
            tracker.load_history(str(hist_file))
            last_tracker = tracker
        except Exception:
            logger.exception("failed to load tracker history: %s", hist_file)
            continue

        for mod, vals in tracker.module_deltas.items():
            module_history.setdefault(mod, []).extend(vals)

        syn_vals = {
            k: v[-1]
            for k, v in tracker.metrics_history.items()
            if k.startswith("synergy_") and v
        }
        if syn_vals:
            synergy_history.append(syn_vals)
            ma_entry: dict[str, float] = {}
            for k in syn_vals:
                vals = [h.get(k, 0.0) for h in synergy_history[-args.synergy_cycles :]]
                ema, _ = cli._ema(vals) if vals else (0.0, 0.0)
                ma_entry[k] = ema
            synergy_ma_history.append(ma_entry)
        history = getattr(tracker, "roi_history", [])
        if history:
            ema, _ = cli._ema(history[-args.roi_cycles :])
            roi_ma_history.append(ema)

        if getattr(args, "auto_thresholds", False):
            roi_threshold = cli._adaptive_threshold(
                tracker.roi_history, args.roi_cycles
            )
        elif roi_threshold is None:
            roi_threshold = tracker.diminishing()
        new_flags, _ = cli._diminishing_modules(
            module_history,
            flagged,
            roi_threshold,
            consecutive=args.roi_cycles,
            confidence=roi_confidence or 0.95,
        )
        flagged.update(new_flags)

        if getattr(args, "auto_thresholds", False):
            synergy_threshold = cli._adaptive_synergy_threshold(
                synergy_history,
                synergy_threshold_window,
                weight=synergy_threshold_weight,
                confidence=synergy_confidence or 0.95,
            )
        elif synergy_threshold is None:
            synergy_threshold = tracker.diminishing()
        converged, ema_val, _ = cli._synergy_converged(
            synergy_history,
            args.synergy_cycles,
            synergy_threshold,
            std_threshold=synergy_std_threshold,
            ma_window=(
                synergy_ma_window
                if synergy_ma_window is not None
                else args.synergy_cycles
            ),
            confidence=synergy_confidence or 0.95,
            stationarity_confidence=synergy_stationarity_confidence
            or (synergy_confidence or 0.95),
            variance_confidence=synergy_variance_confidence
            or (synergy_confidence or 0.95),
        )

        if module_history and set(module_history) <= flagged and converged:
            logger.info("convergence reached", extra={"run": run_idx, "ema": ema_val})
            break

    if agent_proc and agent_proc.poll() is None:
        try:
            agent_proc.terminate()
            try:
                agent_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                agent_proc.kill()
        except Exception:
            logger.exception("failed to shutdown visual agent")
        agent_proc = None


if __name__ == "__main__":
    main()
