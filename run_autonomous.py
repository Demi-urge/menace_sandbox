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
if (
    os.getenv("MENACE_MODE", "test").lower() == "production"
    and os.getenv("DATABASE_URL", "").startswith("sqlite")
):
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

from menace.environment_generator import generate_presets
from menace.startup_checks import verify_project_dependencies, _parse_requirement
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

    missing_pkgs = verify_project_dependencies()
    if missing_pkgs:
        lock = None
        for name in ("uv.lock", "requirements.txt"):
            path = Path(name)
            if path.exists():
                lock = path
                break
        if lock:
            try:
                subprocess.check_call(
                    [sys.executable, "-m", "pip", "install", "-r", str(lock)]
                )
            except Exception as exc:
                logger.error("failed installing %s: %s", lock, exc)
        for pkg in missing_pkgs:
            mod = _parse_requirement(pkg)
            success = False
            for i in range(2):
                try:
                    subprocess.check_call(
                        [sys.executable, "-m", "pip", "install", pkg]
                    )
                    importlib.import_module(mod)
                    success = True
                    break
                except Exception as exc:  # pragma: no cover - retry
                    logger.warning(
                        "install attempt %s for %s failed: %s", i + 1, pkg, exc
                    )
                    time.sleep(1)
            if not success:
                missing.append(pkg)
    if missing_pkgs:
        missing.extend([m for m in missing_pkgs if m not in missing])

    if missing:
        logger.error("Missing dependencies: %s", ", ".join(missing))
        raise RuntimeError("dependency installation failed")
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
        help="maximum number of full sandbox runs to execute",
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
    _check_dependencies()

    if args.dashboard_port:
        from menace.metrics_dashboard import MetricsDashboard
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

    agent_proc = None
    autostart = os.getenv("VISUAL_AGENT_AUTOSTART", "1") != "0"
    if autostart:
        try:
            import requests  # type: ignore
            base = (
                os.getenv("VISUAL_AGENT_URLS", "http://127.0.0.1:8001")
            ).split(";")[0]
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
    roi_threshold = args.roi_threshold
    env_val = os.getenv("ROI_THRESHOLD")
    if roi_threshold is None and env_val is not None:
        try:
            roi_threshold = float(env_val)
        except Exception:
            roi_threshold = None
    synergy_threshold = args.synergy_threshold
    env_val = os.getenv("SYNERGY_THRESHOLD")
    if synergy_threshold is None and env_val is not None:
        try:
            synergy_threshold = float(env_val)
        except Exception:
            synergy_threshold = None
    roi_confidence = args.roi_confidence
    env_val = os.getenv("ROI_CONFIDENCE")
    if roi_confidence is None and env_val is not None:
        try:
            roi_confidence = float(env_val)
        except Exception:
            roi_confidence = None
    synergy_confidence = args.synergy_confidence
    env_val = os.getenv("SYNERGY_CONFIDENCE")
    if synergy_confidence is None and env_val is not None:
        try:
            synergy_confidence = float(env_val)
        except Exception:
            synergy_confidence = None
    synergy_ma_window = args.synergy_ma_window
    env_val = os.getenv("SYNERGY_MA_WINDOW")
    if synergy_ma_window is None and env_val is not None:
        try:
            synergy_ma_window = int(env_val)
        except Exception:
            synergy_ma_window = None
    synergy_stationarity_confidence = args.synergy_stationarity_confidence
    env_val = os.getenv("SYNERGY_STATIONARITY_CONFIDENCE")
    if synergy_stationarity_confidence is None and env_val is not None:
        try:
            synergy_stationarity_confidence = float(env_val)
        except Exception:
            synergy_stationarity_confidence = None
    if synergy_ma_window is None:
        synergy_ma_window = args.synergy_cycles
    if synergy_stationarity_confidence is None:
        synergy_stationarity_confidence = synergy_confidence or 0.95
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
                vals = [h.get(k, 0.0) for h in synergy_history[-args.synergy_cycles:]]
                ema, _ = cli._ema(vals) if vals else (0.0, 0.0)
                ma_entry[k] = ema
            synergy_ma_history.append(ma_entry)
        history = getattr(tracker, "roi_history", [])
        if history:
            ema, _ = cli._ema(history[-args.roi_cycles:])
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
                synergy_history, args.synergy_cycles
            )
        elif synergy_threshold is None:
            synergy_threshold = tracker.diminishing()
        converged, ema_val, _ = cli._synergy_converged(
            synergy_history,
            args.synergy_cycles,
            synergy_threshold,
            ma_window=synergy_ma_window if synergy_ma_window is not None else args.synergy_cycles,
            confidence=synergy_confidence or 0.95,
            stationarity_confidence=synergy_stationarity_confidence or (synergy_confidence or 0.95),
        )

        if module_history and set(module_history) <= flagged and converged:
            logger.info(
                "convergence reached", extra={"run": run_idx, "ema": ema_val}
            )
            break

    if agent_proc:
        agent_proc.terminate()
        try:
            agent_proc.wait(timeout=5)
        except Exception:
            agent_proc.kill()
        agent_proc = None


if __name__ == "__main__":
    main()
