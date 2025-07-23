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
from datetime import datetime
import importlib
import importlib.util
import socket
import contextlib


REQUIRED_SYSTEM_TOOLS = ["ffmpeg", "tesseract", "qemu-system-x86_64"]
REQUIRED_PYTHON_PKGS = ["filelock", "pydantic", "dotenv"]


def _verify_required_dependencies() -> None:
    missing_sys = [t for t in REQUIRED_SYSTEM_TOOLS if shutil.which(t) is None]
    missing_py = [p for p in REQUIRED_PYTHON_PKGS if importlib.util.find_spec(p) is None]
    if missing_sys or missing_py:
        messages: list[str] = []
        if missing_sys:
            messages.append(
                "Missing system packages: "
                + ", ".join(missing_sys)
                + ". Install them using your package manager."
            )
        if missing_py:
            messages.append(
                "Missing Python packages: "
                + ", ".join(missing_py)
                + ". Install them with 'pip install <package>'."
            )
        raise SystemExit("\n".join(messages))


_verify_required_dependencies()

from filelock import FileLock
from pydantic import BaseModel, ValidationError, validator

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
from sandbox_settings import SandboxSettings

import menace.environment_generator as environment_generator
from menace.environment_generator import generate_presets
from menace.synergy_exporter import SynergyExporter
from menace.audit_trail import AuditTrail
import sandbox_runner.cli as cli
from sandbox_runner.cli import full_autonomous_run
from menace.roi_tracker import ROITracker
from logging_utils import get_logger, setup_logging
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

logger = get_logger(__name__)


def _visual_agent_running(urls: str) -> bool:
    """Return ``True`` if the visual agent responds to ``/status``."""
    try:
        import requests  # type: ignore

        base = urls.split(";")[0]
        resp = requests.get(f"{base}/status", timeout=3)
        return resp.status_code == 200
    except Exception:
        return False


def _port_available(port: int, host: str = "0.0.0.0") -> bool:
    """Return True if the TCP ``port`` is free on ``host``."""
    with contextlib.closing(socket.socket()) as sock:
        try:
            sock.bind((host, port))
            return True
        except OSError:
            return False


def _free_port() -> int:
    """Return an available TCP port."""
    with contextlib.closing(socket.socket()) as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]


class PresetModel(BaseModel):
    """Schema for environment presets."""

    CPU_LIMIT: str
    MEMORY_LIMIT: str

    class Config:
        extra = "allow"

    @validator("CPU_LIMIT", pre=True, allow_reuse=True)
    def _cpu_numeric(cls, v):
        try:
            float(v)
        except Exception as e:
            raise ValueError("CPU_LIMIT must be numeric") from e
        return str(v)

    @validator("MEMORY_LIMIT", pre=True, allow_reuse=True)
    def _mem_numeric(cls, v):
        val = str(v)
        digits = "".join(ch for ch in val if ch.isdigit() or ch == ".")
        if not digits:
            raise ValueError("MEMORY_LIMIT must contain a numeric value")
        try:
            float(digits)
        except Exception as e:
            raise ValueError("MEMORY_LIMIT must contain a numeric value") from e
        return val


class SynergyEntry(BaseModel):
    """Schema for synergy history entries."""

    __root__: dict[str, float]

    @validator("__root__", pre=True, allow_reuse=True)
    def _check_values(cls, v):
        if not isinstance(v, dict):
            raise ValueError("entry must be a dict")
        out: dict[str, float] = {}
        for k, val in v.items():
            try:
                out[str(k)] = float(val)
            except Exception as e:
                raise ValueError("synergy values must be numeric") from e
        return out


def validate_presets(presets: list[dict]) -> list[dict]:
    """Validate preset dictionaries using :class:`PresetModel`."""
    validated: list[dict] = []
    for idx, p in enumerate(presets):
        try:
            validated.append(PresetModel.parse_obj(p).dict())
        except ValidationError as exc:
            sys.exit(f"Invalid preset at index {idx}: {exc}")
    return validated


def validate_synergy_history(hist: list[dict]) -> list[dict[str, float]]:
    """Validate synergy history entries using :class:`SynergyEntry`."""
    validated: list[dict[str, float]] = []
    for idx, entry in enumerate(hist):
        try:
            validated.append(SynergyEntry.parse_obj(entry).__root__)
        except ValidationError as exc:
            sys.exit(f"Invalid synergy history entry at index {idx}: {exc}")
    return validated


_SETUP_MARKER = Path(".autonomous_setup_complete")


def _check_dependencies(settings: SandboxSettings) -> bool:
    """Return ``True`` and warn if the setup script has not been executed."""
    if not _SETUP_MARKER.exists():
        logger.warning(
            "Dependencies may be missing. Run 'python setup_dependencies.py' first"
        )
    return True


def _get_env_override(name: str, current, settings: SandboxSettings):
    """Return parsed environment variable when ``current`` is ``None``."""
    env_val = getattr(settings, name.lower())
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


def load_previous_synergy(
    data_dir: str | Path,
) -> tuple[list[dict[str, float]], list[dict[str, float]]]:
    """Return synergy history and moving averages from ``synergy_history.json``."""

    path = Path(data_dir) / "synergy_history.json"
    if not path.exists():
        return [], []
    history: list[dict[str, float]] = []
    try:
        loaded = json.loads(path.read_text())
        if isinstance(loaded, list):
            history = validate_synergy_history(
                [entry for entry in loaded if isinstance(entry, dict)]
            )
        else:
            raise ValueError("synergy history must be a list")
    except (ValidationError, ValueError, json.JSONDecodeError) as exc:
        logger.warning("could not parse synergy history %s: %s", path, exc)
        history = []
    except Exception as exc:  # pragma: no cover - unexpected errors
        logger.warning("failed to load synergy history %s: %s", path, exc)
        history = []

    ma_history: list[dict[str, float]] = []
    for idx, entry in enumerate(history):
        ma_entry: dict[str, float] = {}
        for k in entry:
            vals = [h.get(k, 0.0) for h in history[: idx + 1]]
            ema, _ = cli._ema(vals) if vals else (0.0, 0.0)
            ma_entry[k] = ema
        ma_history.append(ma_entry)

    return history, ma_history


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
    parser.add_argument(
        "--no-preset-evolution",
        action="store_true",
        dest="disable_preset_evolution",
        help="disable adapting presets from previous run history",
    )
    parser.add_argument(
        "--save-synergy-history",
        dest="save_synergy_history",
        action="store_true",
        default=None,
        help="persist synergy metrics across runs",
    )
    parser.add_argument(
        "--no-save-synergy-history",
        dest="save_synergy_history",
        action="store_false",
        default=None,
        help="do not persist synergy metrics",
    )
    parser.add_argument(
        "--recover",
        action="store_true",
        help="reload last ROI and synergy histories before running",
    )
    parser.add_argument(
        "--check-settings",
        action="store_true",
        help="validate environment settings and exit",
    )
    args = parser.parse_args(argv)


    setup_logging()

    env_file = Path(os.getenv("MENACE_ENV_FILE", ".env"))
    created_env = not env_file.exists()
    ensure_env(str(env_file))
    if created_env:
        logger.info("created env file at %s", env_file)

    try:
        settings = SandboxSettings()
    except ValidationError as exc:
        if args.check_settings:
            print(exc)
            return
        raise
    if args.check_settings:
        print("Environment settings valid")
        return

    if settings.roi_cycles is not None:
        args.roi_cycles = settings.roi_cycles
    if settings.synergy_cycles is not None:
        args.synergy_cycles = settings.synergy_cycles
    if settings.save_synergy_history is not None:
        args.save_synergy_history = settings.save_synergy_history
    elif args.save_synergy_history is None:
        args.save_synergy_history = True

    data_dir = Path(args.sandbox_data_dir or settings.sandbox_data_dir)
    synergy_history: list[dict[str, float]] = []
    synergy_ma_prev: list[dict[str, float]] = []
    if args.save_synergy_history or args.recover:
        synergy_history, synergy_ma_prev = load_previous_synergy(data_dir)
    if args.synergy_cycles is None:
        args.synergy_cycles = max(3, len(synergy_history))


    if args.preset_files is None:
        data_dir = Path(args.sandbox_data_dir or settings.sandbox_data_dir)
        preset_file = data_dir / "presets.json"
        created_preset = False
        env_val = settings.sandbox_env_presets
        if env_val:
            try:
                presets_raw = json.loads(env_val)
                if isinstance(presets_raw, dict):
                    presets_raw = [presets_raw]
                presets = validate_presets(presets_raw)
            except Exception:
                presets = validate_presets(generate_presets(args.preset_count))
                os.environ["SANDBOX_ENV_PRESETS"] = json.dumps(presets)
        elif preset_file.exists():
            try:
                presets_raw = json.loads(preset_file.read_text())
                if isinstance(presets_raw, dict):
                    presets_raw = [presets_raw]
                presets = validate_presets(presets_raw)
            except Exception:
                presets = validate_presets(generate_presets(args.preset_count))
        else:
            if getattr(args, "disable_preset_evolution", False):
                presets = validate_presets(generate_presets(args.preset_count))
            else:
                gen_func = getattr(
                    environment_generator,
                    "generate_presets_from_history",
                    generate_presets,
                )
                if gen_func is generate_presets:
                    presets = validate_presets(generate_presets(args.preset_count))
                else:
                    presets = validate_presets(
                        gen_func(str(data_dir), args.preset_count)
                    )
        os.environ["SANDBOX_ENV_PRESETS"] = json.dumps(presets)
        if not preset_file.exists():
            data_dir.mkdir(parents=True, exist_ok=True)
            preset_file.write_text(json.dumps(presets))
            created_preset = True
        args.preset_files = [str(preset_file)]
        if created_preset:
            logger.info("created preset file at %s", preset_file)

    _check_dependencies(settings)

    dash_port = args.dashboard_port
    dash_env = settings.auto_dashboard_port
    if dash_port is None and dash_env is not None:
        dash_port = dash_env

    synergy_dash_port = None
    if args.save_synergy_history and dash_env is not None:
        synergy_dash_port = dash_env + 1

    meta_log_path = Path(args.sandbox_data_dir or settings.sandbox_data_dir) / "sandbox_meta.log"
    exporter_log = AuditTrail(str(meta_log_path))

    synergy_exporter: SynergyExporter | None = None
    if os.getenv("EXPORT_SYNERGY_METRICS") == "1":
        port = int(os.getenv("SYNERGY_METRICS_PORT", "8003"))
        if not _port_available(port):
            logger.error("synergy exporter port %d in use", port)
            port = _free_port()
            logger.info("using port %d for synergy exporter", port)
        history_file = Path(args.sandbox_data_dir or settings.sandbox_data_dir) / "synergy_history.json"
        synergy_exporter = SynergyExporter(
            history_file=str(history_file),
            port=port,
        )
        try:
            synergy_exporter.start()
            exporter_log.record({"timestamp": int(time.time()), "event": "exporter_started"})
        except Exception as exc:  # pragma: no cover - runtime issues
            logger.warning("failed to start synergy exporter: %s", exc)
            exporter_log.record({"timestamp": int(time.time()), "event": "exporter_start_failed", "error": str(exc)})

    if dash_port:
        if not _port_available(dash_port):
            logger.error("metrics dashboard port %d in use", dash_port)
            dash_port = _free_port()
            logger.info("using port %d for MetricsDashboard", dash_port)
        from menace.metrics_dashboard import MetricsDashboard
        from threading import Thread

        history_file = Path(args.sandbox_data_dir or settings.sandbox_data_dir) / "roi_history.json"
        dash = MetricsDashboard(str(history_file))
        Thread(
            target=dash.run,
            kwargs={"port": dash_port},
            daemon=True,
        ).start()

    if synergy_dash_port:
        if not _port_available(synergy_dash_port):
            logger.error("synergy dashboard port %d in use", synergy_dash_port)
            synergy_dash_port = _free_port()
            logger.info("using port %d for SynergyDashboard", synergy_dash_port)
        from menace.self_improvement_engine import SynergyDashboard
        from threading import Thread

        synergy_file = Path(args.sandbox_data_dir or settings.sandbox_data_dir) / "synergy_history.json"
        s_dash = SynergyDashboard(str(synergy_file))
        Thread(
            target=s_dash.run,
            kwargs={"port": synergy_dash_port},
            daemon=True,
        ).start()

    agent_proc = None
    autostart = settings.visual_agent_autostart
    if autostart and not _visual_agent_running(settings.visual_agent_urls):
        cmd = [sys.executable, str(_pkg_dir / "menace_visual_agent_2.py")]
        if os.getenv("VISUAL_AGENT_AUTO_RECOVER", "0") == "1":
            cmd.append("--auto-recover")
        try:
            agent_proc = subprocess.Popen(cmd)
        except Exception:  # pragma: no cover - runtime dependent
            logger.exception("failed to launch visual agent")
            sys.exit(1)

        started = False
        for _ in range(5):
            time.sleep(1)
            if agent_proc.poll() is not None:
                logger.error(
                    "visual agent exited with code %s", agent_proc.returncode
                )
                sys.exit(1)
            if _visual_agent_running(settings.visual_agent_urls):
                started = True
                break
        if not started:
            logger.error("visual agent failed to start at %s", settings.visual_agent_urls)
            try:
                agent_proc.terminate()
                agent_proc.wait(timeout=5)
            except Exception:
                pass
            sys.exit(1)

    module_history: dict[str, list[float]] = {}
    flagged: set[str] = set()
    roi_ma_history: list[float] = []
    synergy_ma_history: list[dict[str, float]] = list(synergy_ma_prev)
    roi_threshold = _get_env_override("ROI_THRESHOLD", args.roi_threshold, settings)
    synergy_threshold = _get_env_override("SYNERGY_THRESHOLD", args.synergy_threshold, settings)
    roi_confidence = _get_env_override("ROI_CONFIDENCE", args.roi_confidence, settings)
    synergy_confidence = _get_env_override(
        "SYNERGY_CONFIDENCE", args.synergy_confidence, settings
    )
    synergy_threshold_window = _get_env_override(
        "SYNERGY_THRESHOLD_WINDOW", args.synergy_threshold_window, settings
    )
    synergy_threshold_weight = _get_env_override(
        "SYNERGY_THRESHOLD_WEIGHT", args.synergy_threshold_weight, settings
    )
    synergy_ma_window = _get_env_override("SYNERGY_MA_WINDOW", args.synergy_ma_window, settings)
    synergy_stationarity_confidence = _get_env_override(
        "SYNERGY_STATIONARITY_CONFIDENCE", args.synergy_stationarity_confidence, settings
    )
    synergy_std_threshold = _get_env_override(
        "SYNERGY_STD_THRESHOLD", args.synergy_std_threshold, settings
    )
    synergy_variance_confidence = _get_env_override(
        "SYNERGY_VARIANCE_CONFIDENCE", args.synergy_variance_confidence, settings
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

    if args.recover:
        tracker = SandboxRecoveryManager.load_last_tracker(data_dir)
        if tracker:
            last_tracker = tracker
            for mod, vals in tracker.module_deltas.items():
                module_history.setdefault(mod, []).extend(vals)
            missing = tracker.synergy_history[len(synergy_history) :]
            for entry in missing:
                synergy_history.append(entry)
                ma_entry: dict[str, float] = {}
                for k in entry:
                    vals = [h.get(k, 0.0) for h in synergy_history[-args.synergy_cycles :]]
                    ema, _ = cli._ema(vals) if vals else (0.0, 0.0)
                    ma_entry[k] = ema
                synergy_ma_history.append(ma_entry)
            if missing:
                synergy_ma_prev = synergy_ma_history
            if missing and args.save_synergy_history:
                try:
                    data_dir.mkdir(parents=True, exist_ok=True)
                    sy_path = data_dir / "synergy_history.json"
                    with FileLock(str(sy_path) + ".lock"):
                        sy_path.write_text(json.dumps(synergy_history))
                except Exception:
                    logger.exception("failed to save synergy history")
            if tracker.roi_history:
                ema, _ = cli._ema(tracker.roi_history[-args.roi_cycles :])
                roi_ma_history.append(ema)
    else:
        last_tracker = None

    run_idx = 0
    while args.runs is None or run_idx < args.runs:
        run_idx += 1
        logger.info(
            "Starting autonomous run %d/%s",
            run_idx,
            args.runs if args.runs is not None else "?",
        )
        if (
            synergy_exporter is not None
            and synergy_exporter._thread is not None
            and not synergy_exporter._thread.is_alive()
        ):
            exporter_log.record({"timestamp": int(time.time()), "event": "exporter_restarted"})
            try:
                synergy_exporter.start()
            except Exception as exc:
                logger.warning("failed to restart synergy exporter: %s", exc)
                exporter_log.record({"timestamp": int(time.time()), "event": "exporter_restart_failed", "error": str(exc)})
        if agent_proc and agent_proc.poll() is not None:
            logger.error(
                "visual agent process terminated with code %s",
                agent_proc.returncode,
            )
            sys.exit(1)
        if args.preset_files:
            pf = Path(args.preset_files[(run_idx - 1) % len(args.preset_files)])
            with open(pf, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            presets_raw = [data] if isinstance(data, dict) else list(data)
            presets = validate_presets(presets_raw)
        else:
            if getattr(args, "disable_preset_evolution", False):
                presets = validate_presets(generate_presets(args.preset_count))
            else:
                gen_func = getattr(
                    environment_generator,
                    "generate_presets_from_history",
                    generate_presets,
                )
                if gen_func is generate_presets:
                    presets = validate_presets(generate_presets(args.preset_count))
                else:
                    data_dir = args.sandbox_data_dir or settings.sandbox_data_dir
                    presets = validate_presets(
                        gen_func(str(data_dir), args.preset_count)
                    )
        os.environ["SANDBOX_ENV_PRESETS"] = json.dumps(presets)

        recovery = SandboxRecoveryManager(sandbox_runner._sandbox_main)
        sandbox_runner._sandbox_main = recovery.run
        try:
            full_autonomous_run(
                args,
                synergy_history=synergy_history,
                synergy_ma_history=synergy_ma_history,
            )
        finally:
            sandbox_runner._sandbox_main = recovery.sandbox_main

        data_dir = Path(args.sandbox_data_dir or settings.sandbox_data_dir)
        hist_file = data_dir / "roi_history.json"
        tracker = ROITracker()
        try:
            tracker.load_history(str(hist_file))
            last_tracker = tracker
        except Exception:
            logger.exception("failed to load tracker history: %s", hist_file)
            continue

        if args.save_synergy_history:
            synergy_file = data_dir / "synergy_history.json"
            if synergy_file.exists():
                try:
                    loaded = json.loads(synergy_file.read_text())
                    if isinstance(loaded, list):
                        synergy_history = validate_synergy_history(
                            [entry for entry in loaded if isinstance(entry, dict)]
                        )
                    else:
                        raise ValueError("synergy history must be a list")
                except (ValidationError, ValueError, json.JSONDecodeError) as exc:
                    logger.warning(
                        "could not parse synergy history %s: %s", synergy_file, exc
                    )
                    synergy_history = []
                except Exception as exc:  # pragma: no cover - unexpected errors
                    logger.warning(
                        "failed to load synergy history %s: %s", synergy_file, exc
                    )
                    synergy_history = []
            else:
                synergy_history = []

        for mod, vals in tracker.module_deltas.items():
            module_history.setdefault(mod, []).extend(vals)

        syn_vals = {
            k: v[-1]
            for k, v in tracker.metrics_history.items()
            if k.startswith("synergy_") and v
        }
        if syn_vals:
            synergy_history.append(syn_vals)
            if args.save_synergy_history:
                try:
                    data_dir.mkdir(parents=True, exist_ok=True)
                    sy_path = data_dir / "synergy_history.json"
                    with FileLock(str(sy_path) + ".lock"):
                        sy_path.write_text(json.dumps(synergy_history))
                except Exception:
                    logger.exception("failed to save synergy history")
            ma_entry: dict[str, float] = {}
            for k in syn_vals:
                vals = [h.get(k, 0.0) for h in synergy_history[-args.synergy_cycles :]]
                ema, _ = cli._ema(vals) if vals else (0.0, 0.0)
                ma_entry[k] = ema
            synergy_ma_history.append(ma_entry)
            synergy_ma_prev = synergy_ma_history
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

        thr = args.synergy_threshold
        if getattr(args, "auto_thresholds", False) or thr is None:
            thr = None
        converged, ema_val, _ = cli.adaptive_synergy_convergence(
            synergy_history,
            args.synergy_cycles,
            threshold=thr,
            threshold_window=synergy_threshold_window,
            weight=synergy_threshold_weight,
            confidence=synergy_confidence or 0.95,
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

    if synergy_exporter is not None:
        try:
            synergy_exporter.stop()
            exporter_log.record({"timestamp": int(time.time()), "event": "exporter_stopped"})
        except Exception:
            logger.exception("failed to stop synergy exporter")


if __name__ == "__main__":
    main()
