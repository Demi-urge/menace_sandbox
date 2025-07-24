from __future__ import annotations

"""Wrapper for running the autonomous sandbox loop after dependency checks."""

import argparse
import atexit
import contextlib
import importlib
import importlib.util
import json
import logging
import os
import shutil
import signal
import socket
import sqlite3
import subprocess
import sys
import threading
import time
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Callable, List

EXPORTER_CHECK_INTERVAL = float(os.getenv("SYNERGY_EXPORTER_CHECK_INTERVAL", "10"))

AGENT_MONITOR_INTERVAL = float(os.getenv("VISUAL_AGENT_MONITOR_INTERVAL", "30"))


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
from pydantic import BaseModel, RootModel, ValidationError, validator

# Default to test mode when using the bundled SQLite database.
if os.getenv("MENACE_MODE", "test").lower() == "production" and os.getenv(
    "DATABASE_URL", ""
).startswith("sqlite"):
    logging.warning("MENACE_MODE=production with SQLite database; switching to test mode")
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

import menace.environment_generator as environment_generator
import sandbox_runner
import sandbox_runner.cli as cli
from logging_utils import get_logger, setup_logging
from menace.audit_trail import AuditTrail
from menace.auto_env_setup import ensure_env
from menace.environment_generator import generate_presets
from menace.roi_tracker import ROITracker
from menace.synergy_exporter import SynergyExporter
from menace.synergy_history_db import migrate_json_to_db
from sandbox_recovery_manager import SandboxRecoveryManager
from sandbox_runner.cli import full_autonomous_run
from sandbox_settings import SandboxSettings

if not hasattr(sandbox_runner, "_sandbox_main"):
    import importlib.util

    spec = importlib.util.spec_from_file_location("sandbox_runner", _pkg_dir / "sandbox_runner.py")
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


def _exporter_health_ok(exp: SynergyExporter, *, max_age: float = 30.0) -> bool:
    """Return ``True`` if the exporter health endpoint responds and is fresh."""
    if exp._thread is None or not exp._thread.is_alive():
        return False
    if exp.health_port is None:
        return False
    try:
        with urllib.request.urlopen(
            f"http://localhost:{exp.health_port}/health", timeout=3
        ) as resp:
            if resp.status != 200:
                return False
            data = json.loads(resp.read().decode())
            updated = data.get("updated")
            if updated is None:
                return False
            if time.time() - float(updated) > max_age:
                return False
    except Exception:
        return False
    return True


class ExporterMonitor:
    """Background monitor to keep the exporter running."""

    def __init__(
        self,
        exporter: SynergyExporter,
        log: AuditTrail,
        *,
        interval: float = EXPORTER_CHECK_INTERVAL,
    ) -> None:
        self.exporter = exporter
        self.log = log
        self.interval = float(interval)
        self.restart_count = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)
        try:
            self.exporter.stop()
        except Exception:
            logger.exception("failed to stop synergy exporter")

    # ------------------------------------------------------------------
    def _restart(self) -> None:
        try:
            self.exporter.stop()
        except Exception:
            logger.exception("failed to stop synergy exporter")
        try:
            self.exporter = SynergyExporter(
                history_file=str(self.exporter.history_file),
                interval=self.exporter.interval,
                port=self.exporter.port,
            )
            self.exporter.start()
            self.restart_count += 1
            self.log.record(
                {
                    "timestamp": int(time.time()),
                    "event": "exporter_restarted",
                    "restart_count": self.restart_count,
                }
            )
        except Exception as exc:
            logger.warning("failed to restart synergy exporter: %s", exc)
            self.log.record(
                {
                    "timestamp": int(time.time()),
                    "event": "exporter_restart_failed",
                    "error": str(exc),
                }
            )

    def _loop(self) -> None:
        while not self._stop.is_set():
            if not _exporter_health_ok(self.exporter):
                self._restart()
            self._stop.wait(self.interval)


class VisualAgentMonitor:
    """Background monitor that restarts the visual agent if it stops responding."""

    def __init__(self, manager, urls: str, *, interval: float = AGENT_MONITOR_INTERVAL) -> None:
        self.manager = manager
        self.urls = urls
        self.interval = float(interval)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)
        try:
            self.manager.shutdown()
        except Exception:
            logger.exception("failed to shutdown visual agent")

    def _loop(self) -> None:
        while not self._stop.is_set():
            if not _visual_agent_running(self.urls):
                try:
                    tok = os.getenv("VISUAL_AGENT_TOKEN", "")
                    self.manager.restart_with_token(tok)
                except Exception:
                    logger.exception("failed to restart visual agent")
            self._stop.wait(self.interval)


class PresetModel(BaseModel):
    """Schema for environment presets."""

    CPU_LIMIT: str
    MEMORY_LIMIT: str
    DISK_LIMIT: str | None = None
    NETWORK_LATENCY_MS: float | None = None
    NETWORK_JITTER_MS: float | None = None
    MIN_BANDWIDTH: str | None = None
    MAX_BANDWIDTH: str | None = None
    BANDWIDTH_LIMIT: str | None = None
    PACKET_LOSS: float | None = None
    PACKET_DUPLICATION: float | None = None
    SECURITY_LEVEL: int | None = None
    THREAT_INTENSITY: int | None = None
    GPU_LIMIT: int | None = None
    OS_TYPE: str | None = None
    CONTAINER_IMAGE: str | None = None
    VM_SETTINGS: dict | None = None
    FAILURE_MODES: list[str] | str | None = None

    class Config:
        extra = "forbid"

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

    @validator(
        "NETWORK_LATENCY_MS",
        "NETWORK_JITTER_MS",
        "PACKET_LOSS",
        "PACKET_DUPLICATION",
        pre=True,
        allow_reuse=True,
    )
    def _float_fields(cls, v):
        if v is None:
            return v
        try:
            return float(v)
        except Exception as e:
            raise ValueError("value must be numeric") from e

    @validator("SECURITY_LEVEL", "THREAT_INTENSITY", "GPU_LIMIT", pre=True, allow_reuse=True)
    def _int_fields(cls, v):
        if v is None:
            return v
        try:
            return int(v)
        except Exception as e:
            raise ValueError("value must be an integer") from e

    @validator("FAILURE_MODES", pre=True, allow_reuse=True)
    def _fm_list(cls, v):
        if v is None:
            return v
        if isinstance(v, str):
            return [m.strip() for m in v.split(",") if m.strip()]
        if isinstance(v, list):
            return [str(m) for m in v]
        raise ValueError("FAILURE_MODES must be a list or comma separated string")


class SynergyEntry(RootModel[dict[str, float]]):
    """Schema for synergy history entries."""

    @validator("root", pre=True, allow_reuse=True)
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
    errors: list[dict] = []
    for idx, p in enumerate(presets):
        try:
            model = PresetModel.parse_obj(p)
            validated.append(model.dict(exclude_none=True))
        except ValidationError as exc:
            for err in exc.errors():
                new_err = err.copy()
                new_err["loc"] = ("preset", idx) + tuple(err.get("loc", ()))
                errors.append(new_err)
    if errors:
        raise ValidationError.from_exception_data("PresetModel", errors)
    return validated


def validate_synergy_history(hist: list[dict]) -> list[dict[str, float]]:
    """Validate synergy history entries using :class:`SynergyEntry`."""
    validated: list[dict[str, float]] = []
    for idx, entry in enumerate(hist):
        try:
            validated.append(SynergyEntry.parse_obj(entry).root)
        except ValidationError as exc:
            sys.exit(f"Invalid synergy history entry at index {idx}: {exc}")
    return validated


_SETUP_MARKER = Path(".autonomous_setup_complete")


def _check_dependencies(settings: SandboxSettings) -> bool:
    """Return ``True`` and warn if the setup script has not been executed."""
    if not _SETUP_MARKER.exists():
        logger.warning("Dependencies may be missing. Run 'python setup_dependencies.py' first")
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
    """Return synergy history and moving averages from ``synergy_history.db``."""

    path = Path(data_dir) / "synergy_history.db"
    if not path.exists():
        return [], []
    history: list[dict[str, float]] = []
    try:
        with sqlite3.connect(path) as conn:
            rows = conn.execute("SELECT entry FROM synergy_history ORDER BY id").fetchall()
        for (text,) in rows:
            data = json.loads(text)
            if isinstance(data, dict):
                history.append({str(k): float(v) for k, v in data.items()})
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
            "start MetricsDashboard on this port for each run" " (overrides AUTO_DASHBOARD_PORT)"
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

    data_dir = Path(args.sandbox_data_dir or settings.sandbox_data_dir)
    legacy_json = data_dir / "synergy_history.json"
    db_file = data_dir / "synergy_history.db"
    if not db_file.exists() and legacy_json.exists():
        logger.info("migrating %s to SQLite", legacy_json)
        migrate_json_to_db(legacy_json, db_file)

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

    synergy_history: list[dict[str, float]] = []
    synergy_ma_prev: list[dict[str, float]] = []
    history_conn: sqlite3.Connection | None = None
    if args.save_synergy_history or args.recover:
        synergy_history, synergy_ma_prev = load_previous_synergy(data_dir)
        history_conn = sqlite3.connect(str(data_dir / "synergy_history.db"))
        history_conn.execute(
            "CREATE TABLE IF NOT EXISTS synergy_history (id INTEGER PRIMARY KEY AUTOINCREMENT, entry TEXT NOT NULL)"
        )
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
            except ValidationError as exc:
                sys.exit(f"Invalid preset from SANDBOX_ENV_PRESETS: {exc}")
            except Exception:
                presets = validate_presets(generate_presets(args.preset_count))
                os.environ["SANDBOX_ENV_PRESETS"] = json.dumps(presets)
        elif preset_file.exists():
            try:
                presets_raw = json.loads(preset_file.read_text())
                if isinstance(presets_raw, dict):
                    presets_raw = [presets_raw]
                presets = validate_presets(presets_raw)
            except ValidationError as exc:
                sys.exit(f"Invalid preset file {preset_file}: {exc}")
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
                    presets = validate_presets(gen_func(str(data_dir), args.preset_count))
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

    cleanup_funcs: list[Callable[[], None]] = []

    def _cleanup() -> None:
        for func in cleanup_funcs:
            try:
                func()
            except Exception:
                logger.exception("cleanup failed")

    atexit.register(_cleanup)
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda _s, _f: (_cleanup(), sys.exit(0)))

    meta_log_path = Path(args.sandbox_data_dir or settings.sandbox_data_dir) / "sandbox_meta.log"
    exporter_log = AuditTrail(str(meta_log_path))

    synergy_exporter: SynergyExporter | None = None
    exporter_monitor: ExporterMonitor | None = None
    if os.getenv("EXPORT_SYNERGY_METRICS") == "1":
        port = int(os.getenv("SYNERGY_METRICS_PORT", "8003"))
        if not _port_available(port):
            logger.error("synergy exporter port %d in use", port)
            port = _free_port()
            logger.info("using port %d for synergy exporter", port)
        history_file = (
            Path(args.sandbox_data_dir or settings.sandbox_data_dir) / "synergy_history.db"
        )
        synergy_exporter = SynergyExporter(
            history_file=str(history_file),
            port=port,
        )
        try:
            synergy_exporter.start()
            exporter_log.record({"timestamp": int(time.time()), "event": "exporter_started"})
            exporter_monitor = ExporterMonitor(synergy_exporter, exporter_log)
            exporter_monitor.start()
            cleanup_funcs.append(exporter_monitor.stop)
        except Exception as exc:  # pragma: no cover - runtime issues
            logger.warning("failed to start synergy exporter: %s", exc)
            exporter_log.record(
                {"timestamp": int(time.time()), "event": "exporter_start_failed", "error": str(exc)}
            )

    auto_trainer = None
    if os.getenv("AUTO_TRAIN_SYNERGY") == "1":
        from menace.synergy_auto_trainer import SynergyAutoTrainer

        try:
            interval = float(os.getenv("AUTO_TRAIN_INTERVAL", "600"))
        except Exception:
            interval = 600.0
        history_file = (
            Path(args.sandbox_data_dir or settings.sandbox_data_dir) / "synergy_history.db"
        )
        weights_file = (
            Path(args.sandbox_data_dir or settings.sandbox_data_dir) / "synergy_weights.json"
        )
        auto_trainer = SynergyAutoTrainer(
            history_file=str(history_file),
            weights_file=str(weights_file),
            interval=interval,
        )
        try:
            auto_trainer.start()
            cleanup_funcs.append(auto_trainer.stop)
        except Exception as exc:  # pragma: no cover - runtime issues
            logger.warning("failed to start synergy auto trainer: %s", exc)

    dash_thread = None
    if dash_port:
        if not _port_available(dash_port):
            logger.error("metrics dashboard port %d in use", dash_port)
            dash_port = _free_port()
            logger.info("using port %d for MetricsDashboard", dash_port)
        from threading import Thread

        from menace.metrics_dashboard import MetricsDashboard

        history_file = Path(args.sandbox_data_dir or settings.sandbox_data_dir) / "roi_history.json"
        dash = MetricsDashboard(str(history_file))
        dash_thread = Thread(
            target=dash.run,
            kwargs={"port": dash_port},
            daemon=True,
        )
        dash_thread.start()
        cleanup_funcs.append(
            lambda: dash_thread and dash_thread.is_alive() and dash_thread.join(0.1)
        )

    s_dash = None
    if synergy_dash_port:
        if not _port_available(synergy_dash_port):
            logger.error("synergy dashboard port %d in use", synergy_dash_port)
            synergy_dash_port = _free_port()
            logger.info("using port %d for SynergyDashboard", synergy_dash_port)
        from threading import Thread

        from menace.self_improvement_engine import SynergyDashboard

        synergy_file = (
            Path(args.sandbox_data_dir or settings.sandbox_data_dir) / "synergy_history.db"
        )
        s_dash = SynergyDashboard(str(synergy_file))
        dash_t = Thread(
            target=s_dash.run,
            kwargs={"port": synergy_dash_port},
            daemon=True,
        )
        dash_t.start()
        cleanup_funcs.append(s_dash.stop)
        cleanup_funcs.append(lambda: dash_t.is_alive() and dash_t.join(0.1))

    agent_proc = None
    agent_mgr = None
    agent_monitor = None
    autostart = settings.visual_agent_autostart
    if autostart:
        from visual_agent_manager import VisualAgentManager

        agent_mgr = VisualAgentManager(str(_pkg_dir / "menace_visual_agent_2.py"))
        if not _visual_agent_running(settings.visual_agent_urls):
            try:
                agent_mgr.start(os.getenv("VISUAL_AGENT_TOKEN", ""))
                agent_proc = agent_mgr.process
            except Exception:  # pragma: no cover - runtime dependent
                logger.exception("failed to launch visual agent")
                sys.exit(1)

            started = False
            for _ in range(5):
                time.sleep(1)
                if agent_mgr.process and agent_mgr.process.poll() is not None:
                    logger.error("visual agent exited with code %s", agent_mgr.process.returncode)
                    sys.exit(1)
                if _visual_agent_running(settings.visual_agent_urls):
                    started = True
                    break
            if not started:
                logger.error("visual agent failed to start at %s", settings.visual_agent_urls)
                try:
                    agent_mgr.shutdown()
                except Exception:
                    pass
                sys.exit(1)

        agent_monitor = VisualAgentMonitor(agent_mgr, settings.visual_agent_urls)
        agent_monitor.start()
        cleanup_funcs.append(agent_monitor.stop)

    module_history: dict[str, list[float]] = {}
    flagged: set[str] = set()
    roi_ma_history: list[float] = []
    synergy_ma_history: list[dict[str, float]] = list(synergy_ma_prev)
    roi_threshold = _get_env_override("ROI_THRESHOLD", args.roi_threshold, settings)
    synergy_threshold = _get_env_override("SYNERGY_THRESHOLD", args.synergy_threshold, settings)
    roi_confidence = _get_env_override("ROI_CONFIDENCE", args.roi_confidence, settings)
    synergy_confidence = _get_env_override("SYNERGY_CONFIDENCE", args.synergy_confidence, settings)
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
            if missing and args.save_synergy_history and history_conn is not None:
                try:
                    for entry in missing:
                        history_conn.execute(
                            "INSERT INTO synergy_history(entry) VALUES (?)",
                            (json.dumps(entry),),
                        )
                    history_conn.commit()
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
        if run_idx > 1:
            new_tok = os.getenv("VISUAL_AGENT_TOKEN_ROTATE")
            if new_tok and agent_mgr and _visual_agent_running(settings.visual_agent_urls):
                try:
                    agent_mgr.restart_with_token(new_tok)
                    os.environ["VISUAL_AGENT_TOKEN"] = new_tok
                    os.environ.pop("VISUAL_AGENT_TOKEN_ROTATE", None)
                    logger.info("visual agent token rotated")
                except Exception:
                    logger.exception("failed to rotate visual agent token")

        logger.info(
            "Starting autonomous run %d/%s",
            run_idx,
            args.runs if args.runs is not None else "?",
        )
        # exporter health is monitored in background
        if agent_mgr and agent_mgr.process and agent_mgr.process.poll() is not None:
            logger.error(
                "visual agent process terminated with code %s",
                agent_mgr.process.returncode,
            )
            sys.exit(1)
        if args.preset_files:
            pf = Path(args.preset_files[(run_idx - 1) % len(args.preset_files)])
            with open(pf, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            presets_raw = [data] if isinstance(data, dict) else list(data)
            try:
                presets = validate_presets(presets_raw)
            except ValidationError as exc:
                sys.exit(f"Invalid preset file {pf}: {exc}")
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
                    presets = validate_presets(gen_func(str(data_dir), args.preset_count))
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

        if args.save_synergy_history and history_conn is not None:
            try:
                rows = history_conn.execute(
                    "SELECT entry FROM synergy_history ORDER BY id"
                ).fetchall()
                synergy_history = [
                    {str(k): float(v) for k, v in json.loads(text).items()}
                    for (text,) in rows
                    if isinstance(json.loads(text), dict)
                ]
            except Exception as exc:  # pragma: no cover - unexpected errors
                logger.warning(
                    "failed to load synergy history %s: %s",
                    data_dir / "synergy_history.db",
                    exc,
                )
                synergy_history = []

        for mod, vals in tracker.module_deltas.items():
            module_history.setdefault(mod, []).extend(vals)

        syn_vals = {
            k: v[-1] for k, v in tracker.metrics_history.items() if k.startswith("synergy_") and v
        }
        if syn_vals:
            synergy_history.append(syn_vals)
            if args.save_synergy_history and history_conn is not None:
                try:
                    history_conn.execute(
                        "INSERT INTO synergy_history(entry) VALUES (?)",
                        (json.dumps(syn_vals),),
                    )
                    history_conn.commit()
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
            roi_threshold = cli._adaptive_threshold(tracker.roi_history, args.roi_cycles)
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

    if agent_monitor is not None:
        try:
            agent_monitor.stop()
        except Exception:
            logger.exception("failed to shutdown visual agent")

    if exporter_monitor is not None:
        try:
            exporter_monitor.stop()
            exporter_log.record(
                {
                    "timestamp": int(time.time()),
                    "event": "exporter_stopped",
                    "restart_count": exporter_monitor.restart_count,
                }
            )
        except Exception:
            logger.exception("failed to stop synergy exporter")

    if history_conn is not None:
        history_conn.close()


if __name__ == "__main__":
    main()
