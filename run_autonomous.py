from __future__ import annotations

# flake8: noqa

"""Wrapper for running the autonomous sandbox loop after dependency checks.

Initialises :data:`GLOBAL_ROUTER` via :func:`init_db_router` before importing
modules that touch the database.
"""

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
import _thread
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, List
import math
import uuid
from scipy.stats import t
from db_router import init_db_router
from dynamic_path_router import resolve_path, get_project_root, path_for_prompt
from sandbox_settings import SandboxSettings
from sandbox_runner.bootstrap import (
    bootstrap_environment,
    _verify_required_dependencies,
)

logger = logging.getLogger(__name__)

settings = SandboxSettings()
settings = bootstrap_environment(settings, _verify_required_dependencies)
os.environ["SANDBOX_CENTRAL_LOGGING"] = "1" if settings.sandbox_central_logging else "0"
LOCAL_KNOWLEDGE_REFRESH_INTERVAL = settings.local_knowledge_refresh_interval
_LKM_REFRESH_STOP = threading.Event()
_LKM_REFRESH_THREAD: threading.Thread | None = None


# Initialise database router with a unique menace_id. All DB access must go
# through the router.  Import modules requiring database access afterwards so
# they can rely on ``GLOBAL_ROUTER``.
MENACE_ID = uuid.uuid4().hex
LOCAL_DB_PATH = settings.menace_local_db_path or str(
    resolve_path(f"menace_{MENACE_ID}_local.db")
)
SHARED_DB_PATH = settings.menace_shared_db_path or str(resolve_path("shared/global.db"))
GLOBAL_ROUTER = init_db_router(MENACE_ID, LOCAL_DB_PATH, SHARED_DB_PATH)

from gpt_memory import GPTMemoryManager
from memory_maintenance import MemoryMaintenance, _load_retention_rules
from gpt_knowledge_service import GPTKnowledgeService
from local_knowledge_module import LocalKnowledgeModule, init_local_knowledge
from filelock import FileLock
from pydantic import BaseModel, RootModel, ValidationError, validator

# Default to test mode when using the bundled SQLite database.
if settings.menace_mode.lower() == "production" and settings.database_url.startswith(
    "sqlite"
):
    logger.warning(
        "MENACE_MODE=production with SQLite database; switching to test mode"
    )
    os.environ["MENACE_MODE"] = "test"
    settings = SandboxSettings()

# Ensure repository root on sys.path when running as a script
if "menace" not in sys.modules:
    sys.path.insert(0, str(get_project_root()))

# Repository root used by background services like the RelevancyRadarService.
# Default to ``SANDBOX_REPO_PATH`` when provided, otherwise fall back to the
# directory containing this file.  ``SANDBOX_REPO_PATH`` is required in most
# environments but this fallback keeps unit tests and ad-hoc scripts working.
REPO_ROOT = resolve_path(settings.sandbox_repo_path or ".")


spec = importlib.util.spec_from_file_location("menace", resolve_path("__init__.py"))
menace_pkg = importlib.util.module_from_spec(spec)
sys.modules["menace"] = menace_pkg
spec.loader.exec_module(menace_pkg)

import menace.environment_generator as environment_generator
import sandbox_runner
import sandbox_runner.cli as cli
from logging_utils import (
    get_logger,
    setup_logging,
    log_record,
    set_correlation_id,
)
from menace.audit_trail import AuditTrail
from menace.environment_generator import generate_presets
from menace.roi_tracker import ROITracker
from foresight_tracker import ForesightTracker
from menace.synergy_exporter import SynergyExporter
from menace.synergy_history_db import migrate_json_to_db, insert_entry, connect_locked
import menace.synergy_history_db as shd

try:  # pragma: no cover - executed when run as a script
    from metrics_exporter import (
        start_metrics_server,
        roi_threshold_gauge,
        synergy_threshold_gauge,
        roi_forecast_gauge,
        synergy_forecast_gauge,
        synergy_adaptation_actions_total,
    )
    import metrics_exporter
except ImportError:  # pragma: no cover - executed when run as a module
    from .metrics_exporter import (
        start_metrics_server,
        roi_threshold_gauge,
        synergy_threshold_gauge,
        roi_forecast_gauge,
        synergy_forecast_gauge,
        synergy_adaptation_actions_total,
    )
    from . import metrics_exporter

# ``synergy_monitor`` is normally imported as a top-level module so tests can
# replace it via ``sys.modules['synergy_monitor']``.  When executing
# ``run_autonomous`` as a package (e.g. with ``python -m menace_sandbox.run_autonomous``)
# that import may fail because the package directory isn't on ``sys.path``.
# Fallback to a relative import in that case.
try:  # pragma: no cover - exercised implicitly in integration tests
    synergy_monitor = importlib.import_module("synergy_monitor")
except ModuleNotFoundError:  # pragma: no cover - executed when not installed
    from . import synergy_monitor

ExporterMonitor = synergy_monitor.ExporterMonitor
AutoTrainerMonitor = synergy_monitor.AutoTrainerMonitor
try:  # pragma: no cover - exercised implicitly in integration tests
    from sandbox_recovery_manager import SandboxRecoveryManager
except ImportError:  # pragma: no cover - executed when not installed or run directly
    from .sandbox_recovery_manager import SandboxRecoveryManager
from sandbox_runner.cli import full_autonomous_run
from sandbox_settings import SandboxSettings
from threshold_logger import ThresholdLogger
from forecast_logger import ForecastLogger
from preset_logger import PresetLogger

# ``relevancy_radar_service`` relies on package-relative imports.  When running
# this module as a script the package may not be installed, so fall back to a
# relative import to keep tests and direct execution working.
try:  # pragma: no cover - simple import shim
    from relevancy_radar_service import RelevancyRadarService
except Exception:  # pragma: no cover - executed when run via package
    from .relevancy_radar_service import RelevancyRadarService

if not hasattr(sandbox_runner, "_sandbox_main"):
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "sandbox_runner", path_for_prompt("sandbox_runner.py")
    )
    sr_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sr_mod)
    sandbox_runner = sys.modules["sandbox_runner"] = sr_mod

logger = get_logger(__name__)

GPT_MEMORY_MANAGER: GPTMemoryManager | None = None
GPT_KNOWLEDGE_SERVICE: GPTKnowledgeService | None = None
LOCAL_KNOWLEDGE_MODULE: LocalKnowledgeModule | None = None


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


def _start_local_knowledge_refresh(cleanup_funcs: List[Callable[[], None]]) -> None:
    """Start background thread to periodically refresh local knowledge."""

    def _loop() -> None:
        run = 0
        while not _LKM_REFRESH_STOP.wait(LOCAL_KNOWLEDGE_REFRESH_INTERVAL):
            run += 1
            if LOCAL_KNOWLEDGE_MODULE is not None:
                try:
                    LOCAL_KNOWLEDGE_MODULE.refresh()
                    LOCAL_KNOWLEDGE_MODULE.memory.conn.commit()
                except Exception:
                    logger.exception(
                        "failed to refresh local knowledge module",
                        extra=log_record(run=run),
                    )

    global _LKM_REFRESH_THREAD
    if _LKM_REFRESH_THREAD is None:
        _LKM_REFRESH_THREAD = threading.Thread(target=_loop, daemon=True)
        _LKM_REFRESH_THREAD.start()

        def _stop() -> None:
            global _LKM_REFRESH_THREAD
            _LKM_REFRESH_STOP.set()
            if _LKM_REFRESH_THREAD is not None:
                _LKM_REFRESH_THREAD.join(timeout=1.0)
                if _LKM_REFRESH_THREAD.is_alive():
                    logger.warning(
                        "local knowledge refresh thread did not exit within timeout",
                        extra=log_record(timeout=1.0),
                    )
                _LKM_REFRESH_THREAD = None
            _LKM_REFRESH_STOP.clear()

        cleanup_funcs.append(_stop)


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

    @validator(
        "SECURITY_LEVEL", "THREAT_INTENSITY", "GPU_LIMIT", pre=True, allow_reuse=True
    )
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


# Resolve the setup marker path via the dynamic router to avoid
# reliance on relative paths when the module is imported from different
# working directories.
_SETUP_MARKER = resolve_path(".autonomous_setup_complete")


def _check_dependencies(settings: SandboxSettings) -> bool:
    """Return ``True`` and warn if the setup script has not been executed."""
    if not _SETUP_MARKER.exists():
        logger.warning(
            "Dependencies may be missing. Run 'python setup_dependencies.py' first"
        )
    return True


def check_env() -> None:
    """Exit if critical environment variables are unset."""
    missing = [
        name
        for name, val in (
            ("SANDBOX_REPO_PATH", settings.sandbox_repo_path),
        )
        if not val
    ]
    if missing:
        raise SystemExit(
            "Missing required environment variables: " + ", ".join(missing)
        )


def _get_env_override(name: str, current, settings: SandboxSettings):
    """Return parsed environment variable when ``current`` is ``None``."""
    env_val = getattr(settings, name.lower())
    if current is not None or env_val is None:
        return current

    result = None
    try:
        if isinstance(current, int):
            result = int(env_val)
        elif isinstance(current, float):
            result = float(env_val)
    except Exception:
        result = None

    if result is None:
        for cast in (int, float):
            try:
                result = cast(env_val)
                break
            except Exception:
                continue

    logger.debug(
        "environment variable %s overrides CLI value: %s",
        name,
        result,
        extra=log_record(variable=name, value=result),
    )

    return result


def load_previous_synergy(
    data_dir: str | Path,
) -> tuple[list[dict[str, float]], list[dict[str, float]]]:
    """Return synergy history and moving averages from ``synergy_history.db``."""

    data_dir = Path(resolve_path(data_dir))
    path = data_dir / "synergy_history.db"
    if not path.exists():
        return [], []
    history: list[dict[str, float]] = []
    try:
        with connect_locked(path) as conn:
            rows = conn.execute(
                "SELECT entry FROM synergy_history ORDER BY id"
            ).fetchall()
        for (text,) in rows:
            try:
                data = json.loads(text)
            except json.JSONDecodeError as exc:
                logger.warning("invalid synergy history entry ignored: %s", exc)
                continue
            if isinstance(data, dict):
                history.append({str(k): float(v) for k, v in data.items()})
    except Exception as exc:  # pragma: no cover - unexpected errors
        logger.warning(
            "failed to load synergy history %s: %s",
            path_for_prompt(path),
            exc,
        )
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


def prepare_presets(
    run_idx: int,
    args: argparse.Namespace,
    settings: SandboxSettings,
    preset_log: PresetLogger | None = None,
) -> tuple[list[dict], str]:
    """Return presets for ``run_idx`` and their source."""

    preset_source = "static file"
    if args.preset_files:
        pf = Path(args.preset_files[(run_idx - 1) % len(args.preset_files)])
        try:
            data = json.loads(pf.read_text())
        except json.JSONDecodeError as exc:
            logger.warning("preset file %s is corrupted: %s", pf, exc)
            data = generate_presets(args.preset_count)
            pf.write_text(json.dumps(data))
        presets_raw = [data] if isinstance(data, dict) else list(data)
        presets = validate_presets(presets_raw)
        logger.info(
            "loaded presets from file", extra=log_record(run=run_idx, presets=presets)
        )
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
                data_dir = resolve_path(
                    args.sandbox_data_dir or settings.sandbox_data_dir
                )
                presets = validate_presets(gen_func(str(data_dir), args.preset_count))
                preset_source = "history adaptation"
                if getattr(
                    getattr(environment_generator, "adapt_presets", object),
                    "_rl_agent",
                    None,
                ):
                    preset_source = "RL agent"
        logger.info("generated presets", extra=log_record(run=run_idx, presets=presets))
        actions = getattr(environment_generator.adapt_presets, "last_actions", [])
        for act in actions:
            try:
                synergy_adaptation_actions_total.labels(action=act).inc()
            except Exception:
                logger.exception("failed to update adaptation actions gauge")
        logger.debug(
            "preset source=%s last_actions=%s",
            preset_source,
            actions,
            extra=log_record(
                run=run_idx,
                preset_source=preset_source,
                last_actions=actions,
            ),
        )
    os.environ["SANDBOX_ENV_PRESETS"] = json.dumps(presets)
    prepare_presets.last_source = preset_source  # type: ignore[attr-defined]
    if preset_log is not None:
        try:
            preset_log.log(run_idx, preset_source, actions)
        except Exception:
            logger.exception("failed to log preset details")
    return presets, preset_source


def execute_iteration(
    args: argparse.Namespace,
    settings: SandboxSettings,
    presets: list[dict],
    synergy_history: list[dict[str, float]],
    synergy_ma_history: list[dict[str, float]],
) -> tuple[ROITracker | None, ForesightTracker | None]:
    """Run one autonomous iteration using ``presets`` and return trackers."""

    recovery = SandboxRecoveryManager(sandbox_runner._sandbox_main)
    sandbox_runner._sandbox_main = recovery.run
    volatility_threshold = settings.sandbox_volatility_threshold
    foresight_tracker = ForesightTracker(
        max_cycles=10, volatility_threshold=volatility_threshold
    )
    setattr(args, "foresight_tracker", foresight_tracker)
    try:
        full_autonomous_run(
            args,
            synergy_history=synergy_history,
            synergy_ma_history=synergy_ma_history,
        )
    finally:
        sandbox_runner._sandbox_main = recovery.sandbox_main
        if hasattr(args, "foresight_tracker"):
            delattr(args, "foresight_tracker")

    data_dir = Path(resolve_path(args.sandbox_data_dir or settings.sandbox_data_dir))
    hist_file = data_dir / "roi_history.json"
    tracker = ROITracker()
    try:
        tracker.load_history(str(hist_file))
    except Exception:
        logger.exception("failed to load tracker history: %s", hist_file)
        return None, foresight_tracker
    return tracker, foresight_tracker


def update_metrics(
    tracker: ROITracker,
    args: argparse.Namespace,
    run_idx: int,
    module_history: dict[str, list[float]],
    entropy_history: dict[str, list[float]],
    flagged: set[str],
    synergy_history: list[dict[str, float]],
    synergy_ma_history: list[dict[str, float]],
    roi_ma_history: list[float],
    history_conn: sqlite3.Connection | None,
    roi_threshold: float | None,
    roi_confidence: float | None,
    entropy_threshold: float | None,
    entropy_consecutive: int | None,
    synergy_threshold_window: int,
    synergy_threshold_weight: float,
    synergy_confidence: float | None,
    threshold_log: ThresholdLogger,
    forecast_log: ForecastLogger | None = None,
) -> tuple[bool, float, float]:
    """Update histories and return convergence status and EMA."""

    for mod, vals in tracker.module_deltas.items():
        module_history.setdefault(mod, []).extend(vals)
    for mod, vals in tracker.module_entropy_deltas.items():
        entropy_history.setdefault(mod, []).extend(vals)

    syn_vals = {
        k: v[-1]
        for k, v in tracker.metrics_history.items()
        if k.startswith("synergy_") and v
    }
    synergy_ema = None
    if syn_vals:
        synergy_history.append(syn_vals)
        if args.save_synergy_history and history_conn is not None:
            try:
                insert_entry(history_conn, syn_vals)
            except Exception:
                logger.exception("failed to save synergy history")
        ma_entry: dict[str, float] = {}
        for k in syn_vals:
            vals = [h.get(k, 0.0) for h in synergy_history[-args.synergy_cycles :]]
            ema, _ = cli._ema(vals) if vals else (0.0, 0.0)
            ma_entry[k] = ema
        synergy_ma_history.append(ma_entry)
        synergy_ema = ma_entry

    history = getattr(tracker, "roi_history", [])
    roi_ema = None
    if history:
        roi_ema, _ = cli._ema(history[-args.roi_cycles :])
        roi_ma_history.append(roi_ema)

    roi_pred = None
    ci_lo = None
    ci_hi = None
    try:
        pred, (lo, hi) = tracker.forecast()
        roi_pred = float(pred)
        ci_lo = float(lo)
        ci_hi = float(hi)
        roi_forecast_gauge.set(roi_pred)
        logger.debug(
            "roi forecast=%.3f CI=(%.3f, %.3f)",
            roi_pred,
            ci_lo,
            ci_hi,
            extra=log_record(
                run=run_idx,
                roi_prediction=roi_pred,
                ci_lower=ci_lo,
                ci_upper=ci_hi,
            ),
        )
    except Exception:
        logger.exception("ROI forecast failed")
        metrics_exporter.roi_forecast_failures_total.inc()

    try:
        syn_pred = tracker.predict_synergy()
        synergy_forecast_gauge.set(float(syn_pred))
        logger.debug(
            "synergy forecast=%.3f",
            syn_pred,
            extra=log_record(run=run_idx, synergy_prediction=syn_pred),
        )
    except Exception:
        logger.exception("synergy forecast failed")
        metrics_exporter.synergy_forecast_failures_total.inc()

    if getattr(args, "auto_thresholds", False):
        roi_threshold = cli._adaptive_threshold(tracker.roi_history, args.roi_cycles)
        thr_method = "adaptive"
    elif roi_threshold is None:
        roi_threshold = tracker.diminishing()
        thr_method = "diminishing"
    else:
        thr_method = "fixed"
    roi_threshold_gauge.set(float(roi_threshold))
    logger.debug(
        "roi threshold=%.3f method=%s",
        roi_threshold,
        thr_method,
        extra=log_record(
            run=run_idx,
            roi_threshold=roi_threshold,
            method=thr_method,
        ),
    )
    new_flags, _ = cli._diminishing_modules(
        module_history,
        flagged,
        roi_threshold,
        consecutive=args.roi_cycles,
        confidence=roi_confidence,
        entropy_history=entropy_history,
        entropy_threshold=entropy_threshold,
        entropy_consecutive=entropy_consecutive,
    )
    flagged.update(new_flags)

    thr = args.synergy_threshold
    if getattr(args, "auto_thresholds", False) or thr is None:
        thr = None
    syn_thr_val = (
        thr
        if thr is not None
        else cli._adaptive_synergy_threshold(
            synergy_history, synergy_threshold_window, weight=synergy_threshold_weight
        )
    )
    synergy_threshold_gauge.set(float(syn_thr_val))
    logger.debug(
        "synergy threshold=%.3f fixed=%s",
        syn_thr_val,
        thr is not None,
        extra=log_record(
            run=run_idx,
            synergy_threshold=syn_thr_val,
            fixed=thr is not None,
        ),
    )
    converged, ema_val, conf = cli.adaptive_synergy_convergence(
        synergy_history,
        args.synergy_cycles,
        threshold=thr,
        threshold_window=synergy_threshold_window,
        weight=synergy_threshold_weight,
        confidence=synergy_confidence,
    )
    threshold_log.log(run_idx, roi_threshold, syn_thr_val, converged)
    logger.debug(
        "synergy convergence=%s max|ema|=%.3f conf=%.3f thr=%.3f",
        converged,
        ema_val,
        conf,
        syn_thr_val,
        extra=log_record(
            run=run_idx,
            converged=converged,
            ema_value=ema_val,
            confidence=conf,
            threshold=syn_thr_val,
        ),
    )
    logger.debug(
        "forecast %.3f CI=(%.3f, %.3f) roi_thr=%.3f(%s) syn_thr=%.3f",
        roi_pred if roi_pred is not None else float("nan"),
        ci_lo if ci_lo is not None else float("nan"),
        ci_hi if ci_hi is not None else float("nan"),
        roi_threshold,
        thr_method,
        syn_thr_val,
        extra=log_record(
            run=run_idx,
            roi_prediction=roi_pred,
            ci_lower=ci_lo,
            ci_upper=ci_hi,
            roi_threshold=roi_threshold,
            threshold_method=thr_method,
            synergy_threshold=syn_thr_val,
        ),
    )
    # log threshold calculation details for debugging
    roi_vals = tracker.roi_history[-args.roi_cycles :] if tracker.roi_history else []
    roi_ema_val, roi_std_val = cli._ema(roi_vals) if roi_vals else (0.0, 0.0)

    synergy_details: dict[str, dict[str, float]] = {}
    if synergy_history:
        metrics: dict[str, list[float]] = {}
        for entry in synergy_history[-args.synergy_cycles :]:
            for k, v in entry.items():
                if k.startswith("synergy_"):
                    metrics.setdefault(k, []).append(float(v))
        for k, vals in metrics.items():
            ema_m, std_m = cli._ema(vals)
            n = len(vals)
            if n < 2 or std_m == 0:
                conf_m = 1.0 if abs(ema_m) <= syn_thr_val else 0.0
            else:
                se = std_m / math.sqrt(n)
                t_stat = abs(ema_m) / se
                p = 2 * (1 - t.cdf(t_stat, n - 1))
                conf_m = 1 - p
            synergy_details[k] = {
                "ema": ema_m,
                "std": std_m,
                "confidence": conf_m,
            }

    if synergy_details:
        logger.debug(
            "synergy metric stats: %s",
            synergy_details,
            extra=log_record(run=run_idx, synergy_metric_stats=synergy_details),
        )

    logger.debug(
        "metrics window sizes roi=%d synergy=%d sy_win=%d w=%.3f",
        args.roi_cycles,
        args.synergy_cycles,
        synergy_threshold_window,
        synergy_threshold_weight,
        extra=log_record(
            run=run_idx,
            roi_ema=roi_ema_val,
            roi_std=roi_std_val,
            synergy_metrics=synergy_details,
            roi_window=args.roi_cycles,
            synergy_window=args.synergy_cycles,
            synergy_threshold_window=synergy_threshold_window,
            synergy_threshold_weight=synergy_threshold_weight,
        ),
    )
    if forecast_log is not None:
        forecast_log.log(
            {
                "run": run_idx,
                "roi_forecast": roi_pred,
                "ci_lower": ci_lo,
                "ci_upper": ci_hi,
                "roi_threshold": roi_threshold,
                "threshold_method": thr_method,
                "synergy_threshold": syn_thr_val,
                "synergy_converged": converged,
                "synergy_confidence": conf,
                "synergy_metrics": synergy_details,
                "roi_ema": roi_ema_val,
                "roi_std": roi_std_val,
            }
        )
    return converged, ema_val, roi_threshold


def main(argv: List[str] | None = None) -> None:
    """Entry point for the autonomous runner."""
    set_correlation_id(str(uuid.uuid4()))
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
        "--memory-db",
        dest="memory_db",
        help="path to GPT memory database (overrides GPT_MEMORY_DB)",
    )
    parser.add_argument(
        "--memory-compact-interval",
        type=float,
        help=(
            "seconds between GPT memory compaction cycles "
            "(overrides GPT_MEMORY_COMPACT_INTERVAL)"
        ),
    )
    parser.add_argument(
        "--memory-retention",
        help=(
            "comma separated tag=limit pairs controlling memory retention "
            "(overrides GPT_MEMORY_RETENTION)"
        ),
    )
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
        "--metrics-port",
        type=int,
        help="start Prometheus metrics server on this port",
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
        "--entropy-plateau-threshold",
        type=float,
        help="threshold for entropy delta plateau detection",
    )
    parser.add_argument(
        "--entropy-plateau-consecutive",
        type=int,
        help="entropy delta samples below threshold before module convergence",
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
        "--preset-debug",
        action="store_true",
        help="enable verbose preset adaptation logs",
    )
    parser.add_argument(
        "--debug-log-file",
        help="write verbose logs to this file when --preset-debug is enabled",
    )
    parser.add_argument(
        "--forecast-log",
        help="write ROI forecast and threshold details to this file",
    )
    parser.add_argument(
        "--preset-log-file",
        help="write preset source and actions to this JSONL file",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="enable debug logging (overrides --log-level)",
    )
    parser.add_argument(
        "--log-level",
        default=settings.sandbox_log_level or settings.log_level,
        help="logging level for console output",
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
        "--recursive-orphans",
        "--recursive-include",
        action="store_true",
        dest="recursive_orphans",
        default=None,
        help=(
            "recursively integrate orphan dependency chains (sets "
            "SANDBOX_RECURSIVE_ORPHANS=1; alias: --recursive-include)"
        ),
    )
    parser.add_argument(
        "--no-recursive-orphans",
        "--no-recursive-include",
        action="store_false",
        dest="recursive_orphans",
        help=(
            "disable recursive integration of orphan dependency chains (sets "
            "SANDBOX_RECURSIVE_ORPHANS=0)"
        ),
    )
    parser.add_argument(
        "--include-orphans",
        action="store_false",
        dest="include_orphans",
        default=None,
        help="disable running orphan modules during sandbox runs",
    )
    parser.add_argument(
        "--discover-orphans",
        action="store_false",
        dest="discover_orphans",
        default=None,
        help="disable automatic orphan discovery",
    )
    parser.add_argument(
        "--discover-isolated",
        action="store_true",
        dest="discover_isolated",
        default=None,
        help="automatically run discover_isolated_modules before the orphan scan",
    )
    parser.add_argument(
        "--no-discover-isolated",
        action="store_false",
        dest="discover_isolated",
        help="disable discover_isolated_modules during the orphan scan",
    )
    parser.add_argument(
        "--recursive-isolated",
        action="store_true",
        dest="recursive_isolated",
        default=None,
        help="recurse through dependencies of isolated modules (default)",
    )
    parser.add_argument(
        "--no-recursive-isolated",
        action="store_false",
        dest="recursive_isolated",
        help="disable recursively processing modules from discover_isolated_modules",
    )
    parser.add_argument(
        "--auto-include-isolated",
        action="store_true",
        help=(
            "automatically include isolated modules recursively (sets "
            "SANDBOX_AUTO_INCLUDE_ISOLATED=1 and SANDBOX_RECURSIVE_ISOLATED=1)"
        ),
    )
    parser.add_argument(
        "--foresight-trend",
        nargs=2,
        metavar=("FILE", "WORKFLOW_ID"),
        help="show ROI trend metrics from foresight history file",
    )
    parser.add_argument(
        "--foresight-stable",
        nargs=2,
        metavar=("FILE", "WORKFLOW_ID"),
        help="check workflow stability from foresight history file",
    )
    parser.add_argument(
        "--check-settings",
        action="store_true",
        help="validate environment settings and exit",
    )
    args = parser.parse_args(argv)

    setup_logging(level="DEBUG" if args.verbose else args.log_level)

    if args.foresight_trend:
        file, workflow_id = args.foresight_trend
        cli.foresight_trend(file, workflow_id)
        return
    if args.foresight_stable:
        file, workflow_id = args.foresight_stable
        cli.foresight_stability(file, workflow_id)
        return

    mem_db = args.memory_db or settings.gpt_memory_db
    global LOCAL_KNOWLEDGE_MODULE, GPT_MEMORY_MANAGER, GPT_KNOWLEDGE_SERVICE
    LOCAL_KNOWLEDGE_MODULE = init_local_knowledge(mem_db)
    GPT_MEMORY_MANAGER = LOCAL_KNOWLEDGE_MODULE.memory
    GPT_KNOWLEDGE_SERVICE = LOCAL_KNOWLEDGE_MODULE.knowledge

    if args.preset_debug:
        os.environ["PRESET_DEBUG"] = "1"
        log_path = args.debug_log_file
        if not log_path:
            data_dir = Path(
                resolve_path(args.sandbox_data_dir or settings.sandbox_data_dir)
            )
            data_dir.mkdir(parents=True, exist_ok=True)
            log_path = data_dir / "preset_debug.log"
        else:
            log_path = resolve_path(log_path)
            Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_path)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        )
        logging.getLogger().addHandler(fh)

    port = args.metrics_port
    if port is None:
        env_val = settings.metrics_port
        if env_val is not None:
            port = env_val
    if port is not None:
        try:
            logger.info("starting metrics server on port %d", port)
            start_metrics_server(int(port))
            logger.info("metrics server running on port %d", port)
        except Exception:
            logger.exception("failed to start metrics server")

    logger.info("validating environment variables")
    check_env()
    logger.info("environment validation complete")

    try:
        settings = SandboxSettings()
    except ValidationError as exc:
        if args.check_settings:
            logger.warning("%s", exc)
            return
        raise

    auto_include_isolated = bool(
        getattr(settings, "auto_include_isolated", True)
        or getattr(args, "auto_include_isolated", False)
    )
    recursive_orphans = getattr(settings, "recursive_orphan_scan", True)
    if args.recursive_orphans is not None:
        recursive_orphans = args.recursive_orphans
    recursive_isolated = getattr(settings, "recursive_isolated", True)
    if args.recursive_isolated is not None:
        recursive_isolated = args.recursive_isolated
    # ``auto_include_isolated`` forces recursive isolated scans only when the
    # user didn't explicitly request otherwise via ``--recursive-isolated``.
    if auto_include_isolated and args.recursive_isolated is None:
        recursive_isolated = True

    args.auto_include_isolated = auto_include_isolated
    args.recursive_orphans = recursive_orphans
    args.recursive_isolated = recursive_isolated

    os.environ["SANDBOX_AUTO_INCLUDE_ISOLATED"] = "1" if auto_include_isolated else "0"
    os.environ["SELF_TEST_AUTO_INCLUDE_ISOLATED"] = (
        "1" if auto_include_isolated else "0"
    )
    val = "1" if recursive_orphans else "0"
    os.environ["SANDBOX_RECURSIVE_ORPHANS"] = val
    os.environ["SELF_TEST_RECURSIVE_ORPHANS"] = val
    val_iso = "1" if recursive_isolated else "0"
    os.environ["SANDBOX_RECURSIVE_ISOLATED"] = val_iso
    os.environ["SELF_TEST_RECURSIVE_ISOLATED"] = val_iso
    os.environ["SANDBOX_DISCOVER_ISOLATED"] = "1"
    os.environ["SELF_TEST_DISCOVER_ISOLATED"] = "1"
    include_orphans = True
    if getattr(args, "include_orphans") is False:
        include_orphans = False
    args.include_orphans = include_orphans
    os.environ["SANDBOX_INCLUDE_ORPHANS"] = "1" if include_orphans else "0"
    os.environ["SELF_TEST_INCLUDE_ORPHANS"] = "1" if include_orphans else "0"
    if not include_orphans:
        os.environ["SANDBOX_DISABLE_ORPHANS"] = "1"
    discover_orphans = True
    if getattr(args, "discover_orphans") is False:
        discover_orphans = False
    args.discover_orphans = discover_orphans
    os.environ["SANDBOX_DISABLE_ORPHAN_SCAN"] = "1" if not discover_orphans else "0"
    os.environ["SELF_TEST_DISCOVER_ORPHANS"] = "1" if discover_orphans else "0"
    if getattr(args, "discover_isolated") is not None:
        val_di = "1" if args.discover_isolated else "0"
        os.environ["SANDBOX_DISCOVER_ISOLATED"] = val_di
        os.environ["SELF_TEST_DISCOVER_ISOLATED"] = val_di

    logger.info(
        "run_autonomous starting with data_dir=%s runs=%s metrics_port=%s",
        resolve_path(args.sandbox_data_dir or settings.sandbox_data_dir),
        args.runs,
        port,
    )

    data_dir = Path(resolve_path(args.sandbox_data_dir or settings.sandbox_data_dir))
    legacy_json = data_dir / "synergy_history.json"
    db_file = data_dir / "synergy_history.db"
    if not db_file.exists() and legacy_json.exists():
        logger.info("migrating %s to SQLite", legacy_json)
        migrate_json_to_db(legacy_json, db_file)

    if args.check_settings:
        logger.info("Environment settings valid")
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
        history_conn = shd.connect_locked(data_dir / "synergy_history.db")
    if args.synergy_cycles is None:
        args.synergy_cycles = max(3, len(synergy_history))

    if args.preset_files is None:
        data_dir = Path(
            resolve_path(args.sandbox_data_dir or settings.sandbox_data_dir)
        )
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
            logger.info(
                "generated presets from environment",
                extra=log_record(presets=presets),
            )
        elif preset_file.exists():
            try:
                presets_raw = json.loads(preset_file.read_text())
                if isinstance(presets_raw, dict):
                    presets_raw = [presets_raw]
                presets = validate_presets(presets_raw)
            except ValidationError as exc:
                sys.exit(f"Invalid preset file {preset_file}: {exc}")
            except json.JSONDecodeError as exc:
                logger.warning("preset file %s is corrupted: %s", preset_file, exc)
                presets = validate_presets(generate_presets(args.preset_count))
                preset_file.write_text(json.dumps(presets))
            except Exception:
                presets = validate_presets(generate_presets(args.preset_count))
            logger.info(
                "loaded presets from file",
                extra=log_record(presets=presets, source=str(preset_file)),
            )
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
                    actions = getattr(
                        environment_generator.adapt_presets, "last_actions", []
                    )
                    for act in actions:
                        try:
                            synergy_adaptation_actions_total.labels(action=act).inc()
                        except Exception:
                            logger.exception(
                                "failed to update adaptation actions gauge"
                            )
            logger.info("generated presets", extra=log_record(presets=presets))
        os.environ["SANDBOX_ENV_PRESETS"] = json.dumps(presets)
        if not preset_file.exists():
            data_dir.mkdir(parents=True, exist_ok=True)
            preset_file.write_text(json.dumps(presets))
            created_preset = True
        args.preset_files = [str(preset_file)]
        if created_preset:
            logger.info("created preset file at %s", preset_file)

    logger.info("performing dependency check")
    _check_dependencies(settings)
    logger.info("dependency check complete")

    dash_port = args.dashboard_port
    dash_env = settings.auto_dashboard_port
    if dash_port is None and dash_env is not None:
        dash_port = dash_env

    synergy_dash_port = None
    if args.save_synergy_history and dash_env is not None:
        synergy_dash_port = dash_env + 1

    cleanup_funcs: list[Callable[[], None]] = []
    _start_local_knowledge_refresh(cleanup_funcs)

    def _cleanup() -> None:
        for func in cleanup_funcs:
            try:
                func()
            except Exception:
                logger.exception("cleanup failed")

    atexit.register(_cleanup)
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda _s, _f: (_cleanup(), sys.exit(0)))

    mem_maint = None
    if GPT_MEMORY_MANAGER is not None:
        retention_rules = _load_retention_rules()
        if args.memory_retention:
            os.environ["GPT_MEMORY_RETENTION"] = args.memory_retention
            retention_rules = _load_retention_rules()
        interval = args.memory_compact_interval
        if interval is None and settings.gpt_memory_compact_interval is not None:
            interval = settings.gpt_memory_compact_interval
        mem_maint = MemoryMaintenance(
            GPT_MEMORY_MANAGER,
            interval=interval,
            retention=retention_rules,
            knowledge_service=GPT_KNOWLEDGE_SERVICE,
        )
        mem_maint.start()
        cleanup_funcs.append(mem_maint.stop)
    if GPT_KNOWLEDGE_SERVICE is not None:
        cleanup_funcs.append(getattr(GPT_KNOWLEDGE_SERVICE, "stop", lambda: None))

    meta_log_path = (
        Path(resolve_path(args.sandbox_data_dir or settings.sandbox_data_dir))
        / "sandbox_meta.log"
    )
    exporter_log = AuditTrail(str(meta_log_path))
    threshold_log_path = (
        Path(resolve_path(args.sandbox_data_dir or settings.sandbox_data_dir))
        / "threshold_log.jsonl"
    )
    threshold_log = ThresholdLogger(str(threshold_log_path))
    cleanup_funcs.append(threshold_log.close)
    preset_log_path = (
        Path(resolve_path(args.preset_log_file))
        if args.preset_log_file
        else Path(resolve_path(args.sandbox_data_dir or settings.sandbox_data_dir))
        / "preset_log.jsonl"
    )
    preset_log = PresetLogger(str(preset_log_path))
    cleanup_funcs.append(preset_log.close)
    forecast_log = None
    if args.forecast_log:
        forecast_log = ForecastLogger(str(Path(resolve_path(args.forecast_log))))
        cleanup_funcs.append(forecast_log.close)

    synergy_exporter: SynergyExporter | None = None
    exporter_monitor: ExporterMonitor | None = None
    trainer_monitor: AutoTrainerMonitor | None = None
    if settings.export_synergy_metrics:
        port = settings.synergy_metrics_port
        if not _port_available(port):
            logger.error("synergy exporter port %d in use", port)
            port = _free_port()
            logger.info("using port %d for synergy exporter", port)
        history_file = (
            Path(resolve_path(args.sandbox_data_dir or settings.sandbox_data_dir))
            / "synergy_history.db"
        )
        synergy_exporter = SynergyExporter(
            history_file=str(history_file),
            port=port,
        )
        try:
            logger.info(
                "starting synergy exporter on port %d using history %s",
                port,
                history_file,
            )
            synergy_exporter.start()
            logger.info(
                "synergy exporter running on port %d serving %s",
                port,
                history_file,
            )
            exporter_log.record(
                {"timestamp": int(time.time()), "event": "exporter_started"}
            )
            exporter_monitor = ExporterMonitor(synergy_exporter, exporter_log)
            exporter_monitor.start()
            cleanup_funcs.append(exporter_monitor.stop)
        except Exception as exc:  # pragma: no cover - runtime issues
            logger.warning("failed to start synergy exporter: %s", exc)
            exporter_log.record(
                {
                    "timestamp": int(time.time()),
                    "event": "exporter_start_failed",
                    "error": str(exc),
                }
            )

    auto_trainer = None
    if settings.auto_train_synergy:
        from menace.synergy_auto_trainer import SynergyAutoTrainer

        interval = settings.auto_train_interval
        history_file = (
            Path(resolve_path(args.sandbox_data_dir or settings.sandbox_data_dir))
            / "synergy_history.db"
        )
        weights_file = (
            Path(resolve_path(args.sandbox_data_dir or settings.sandbox_data_dir))
            / "synergy_weights.json"
        )
        auto_trainer = SynergyAutoTrainer(
            history_file=str(history_file),
            weights_file=str(weights_file),
            interval=interval,
        )
        try:
            logger.info(
                "starting synergy auto trainer with history %s weights %s interval %.1fs",
                history_file,
                weights_file,
                interval,
            )
            auto_trainer.start()
            logger.info(
                "synergy auto trainer running with history %s weights %s",
                history_file,
                weights_file,
            )
            trainer_monitor = AutoTrainerMonitor(auto_trainer, exporter_log)
            trainer_monitor.start()
            cleanup_funcs.append(trainer_monitor.stop)
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

        history_file = (
            Path(resolve_path(args.sandbox_data_dir or settings.sandbox_data_dir))
            / "roi_history.json"
        )
        dash = MetricsDashboard(str(history_file))
        dash_thread = Thread(
            target=dash.run,
            kwargs={"port": dash_port},
            daemon=True,
        )
        logger.info("starting MetricsDashboard on port %d", dash_port)
        dash_thread.start()
        logger.info("MetricsDashboard running on port %d", dash_port)
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

        try:
            from menace.self_improvement.engine import SynergyDashboard
        except RuntimeError as exc:
            logger.warning("SynergyDashboard unavailable: %s", exc)
        else:
            synergy_file = (
                Path(resolve_path(args.sandbox_data_dir or settings.sandbox_data_dir))
                / "synergy_history.db"
            )
            s_dash = SynergyDashboard(str(synergy_file))
            dash_t = Thread(
                target=s_dash.run,
                kwargs={"port": synergy_dash_port},
                daemon=True,
            )
            logger.info("starting SynergyDashboard on port %d", synergy_dash_port)
            dash_t.start()
            logger.info("SynergyDashboard running on port %d", synergy_dash_port)
            cleanup_funcs.append(s_dash.stop)
            cleanup_funcs.append(lambda: dash_t.is_alive() and dash_t.join(0.1))

    relevancy_radar = None
    if (
        settings.enable_relevancy_radar
        and settings.relevancy_radar_interval is not None
    ):
        relevancy_radar = RelevancyRadarService(
            REPO_ROOT, float(settings.relevancy_radar_interval)
        )
        relevancy_radar.start()
        atexit.register(relevancy_radar.stop)
        cleanup_funcs.append(relevancy_radar.stop)

    module_history: dict[str, list[float]] = {}
    entropy_history: dict[str, list[float]] = {}
    flagged: set[str] = set()
    roi_ma_history: list[float] = []
    synergy_ma_history: list[dict[str, float]] = list(synergy_ma_prev)
    roi_threshold = _get_env_override("ROI_THRESHOLD", args.roi_threshold, settings)
    synergy_threshold = _get_env_override(
        "SYNERGY_THRESHOLD", args.synergy_threshold, settings
    )
    roi_confidence = _get_env_override("ROI_CONFIDENCE", args.roi_confidence, settings)
    synergy_confidence = _get_env_override(
        "SYNERGY_CONFIDENCE", args.synergy_confidence, settings
    )
    if roi_confidence is None:
        roi_confidence = settings.roi.confidence or 0.95
    if synergy_confidence is None:
        synergy_confidence = settings.synergy.confidence or 0.95
    entropy_threshold = _get_env_override(
        "ENTROPY_PLATEAU_THRESHOLD", args.entropy_plateau_threshold, settings
    )
    entropy_consecutive = _get_env_override(
        "ENTROPY_PLATEAU_CONSECUTIVE", args.entropy_plateau_consecutive, settings
    )
    synergy_threshold_window = _get_env_override(
        "SYNERGY_THRESHOLD_WINDOW", args.synergy_threshold_window, settings
    )
    synergy_threshold_weight = _get_env_override(
        "SYNERGY_THRESHOLD_WEIGHT", args.synergy_threshold_weight, settings
    )
    synergy_ma_window = _get_env_override(
        "SYNERGY_MA_WINDOW", args.synergy_ma_window, settings
    )
    synergy_stationarity_confidence = _get_env_override(
        "SYNERGY_STATIONARITY_CONFIDENCE",
        args.synergy_stationarity_confidence,
        settings,
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
        synergy_stationarity_confidence = (
            settings.synergy.stationarity_confidence or synergy_confidence or 0.95
        )
    if synergy_std_threshold is None:
        synergy_std_threshold = 1e-3
    if synergy_variance_confidence is None:
        synergy_variance_confidence = (
            settings.synergy.variance_confidence or synergy_confidence or 0.95
        )

    if args.recover:
        tracker = SandboxRecoveryManager.load_last_tracker(data_dir)
        if tracker:
            last_tracker = tracker
            for mod, vals in tracker.module_deltas.items():
                module_history.setdefault(mod, []).extend(vals)
            for mod, vals in tracker.module_entropy_deltas.items():
                entropy_history.setdefault(mod, []).extend(vals)
            missing = tracker.synergy_history[len(synergy_history) :]
            for entry in missing:
                synergy_history.append(entry)
                ma_entry: dict[str, float] = {}
                for k in entry:
                    vals = [
                        h.get(k, 0.0) for h in synergy_history[-args.synergy_cycles :]
                    ]
                    ema, _ = cli._ema(vals) if vals else (0.0, 0.0)
                    ma_entry[k] = ema
                synergy_ma_history.append(ma_entry)
            if missing:
                synergy_ma_prev = synergy_ma_history
            if missing and args.save_synergy_history and history_conn is not None:
                try:
                    for entry in missing:
                        insert_entry(history_conn, entry)
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
        set_correlation_id(f"run-{run_idx}")
        logger.info(
            "Starting autonomous run %d/%s",
            run_idx,
            args.runs if args.runs is not None else "?",
        )
        presets, preset_source = prepare_presets(run_idx, args, settings, preset_log)
        logger.info(
            "using presets from %s",
            preset_source,
            extra=log_record(run=run_idx, preset_source=preset_source),
        )
        logger.debug(
            "loaded presets from %s: %s",
            preset_source,
            presets,
            extra=log_record(run=run_idx, preset_source=preset_source, presets=presets),
        )

        tracker, foresight_tracker = execute_iteration(
            args,
            settings,
            presets,
            synergy_history,
            synergy_ma_history,
        )
        if tracker is None:
            logger.info("completed autonomous run %d", run_idx)
            set_correlation_id(None)
            continue
        last_tracker = tracker

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

        converged, ema_val, roi_threshold = update_metrics(
            tracker,
            args,
            run_idx,
            module_history,
            entropy_history,
            flagged,
            synergy_history,
            synergy_ma_history,
            roi_ma_history,
            history_conn,
            roi_threshold,
            roi_confidence,
            entropy_threshold,
            entropy_consecutive,
            synergy_threshold_window,
            synergy_threshold_weight,
            synergy_confidence,
            threshold_log,
            forecast_log,
        )

        if foresight_tracker is not None:
            try:
                slope, second_derivative, avg_stability = (
                    foresight_tracker.get_trend_curve("_global")
                )
                logger.info(
                    "foresight trend: slope=%.3f curve=%.3f avg_stability=%.3f",
                    slope,
                    second_derivative,
                    avg_stability,
                    extra=log_record(
                        run=run_idx,
                        foresight_slope=slope,
                        foresight_curve=second_derivative,
                        foresight_avg_stability=avg_stability,
                    ),
                )
            except Exception:
                logger.exception("failed to compute foresight trend")

        logger.info(
            "run %d summary: roi_threshold=%.3f ema=%.3f converged=%s flagged_modules=%d",
            run_idx,
            roi_threshold,
            ema_val,
            converged,
            len(flagged),
            extra=log_record(
                run=run_idx,
                roi_threshold=roi_threshold,
                ema_value=ema_val,
                converged=converged,
                flagged_count=len(flagged),
            ),
        )

        logger.info("completed autonomous run %d", run_idx)
        set_correlation_id(None)

        all_mods = set(module_history) | set(entropy_history)
        if all_mods and all_mods <= flagged and converged:
            logger.info(
                "convergence reached",
                extra=log_record(
                    run=run_idx,
                    ema_value=ema_val,
                    flagged_modules=sorted(flagged),
                ),
            )
            break

    _cleanup()
    try:
        atexit.unregister(_cleanup)
    except Exception:
        pass
    cleanup_funcs.clear()

    if LOCAL_KNOWLEDGE_MODULE is not None:
        try:
            LOCAL_KNOWLEDGE_MODULE.refresh()
            LOCAL_KNOWLEDGE_MODULE.memory.conn.commit()
        except Exception:
            logger.exception("failed to refresh local knowledge module")

    if exporter_monitor is not None:
        try:
            logger.info(
                "synergy exporter stopped after %d restarts",
                exporter_monitor.restart_count,
            )
            exporter_log.record(
                {
                    "timestamp": int(time.time()),
                    "event": "exporter_stopped",
                    "restart_count": exporter_monitor.restart_count,
                }
            )
        except Exception:
            logger.exception("failed to stop synergy exporter")

    if trainer_monitor is not None:
        try:
            logger.info(
                "synergy auto trainer stopped after %d restarts",
                trainer_monitor.restart_count,
            )
            exporter_log.record(
                {
                    "timestamp": int(time.time()),
                    "event": "auto_trainer_stopped",
                    "restart_count": trainer_monitor.restart_count,
                }
            )
        except Exception:
            logger.exception("failed to stop synergy auto trainer")

    if history_conn is not None:
        history_conn.close()

    if GPT_MEMORY_MANAGER is not None:
        try:
            GPT_MEMORY_MANAGER.close()
        except Exception:
            logger.exception("failed to close GPT memory")

    logger.info("run_autonomous exiting")


def bootstrap(
    config_path: str | Path = get_project_root() / "config" / "bootstrap.yaml",
) -> None:
    """Bootstrap the autonomous sandbox using configuration from ``config_path``.

    The helper loads :class:`SandboxSettings`, initialises core databases and the
    optional event bus, starts the self improvement cycle in a background
    daemon thread and finally launches the sandbox runner.
    """
    from pydantic import ValidationError
    from sandbox_settings import load_sandbox_settings
    from self_improvement.api import init_self_improvement
    from self_improvement.orchestration import (
        start_self_improvement_cycle,
        stop_self_improvement_cycle,
    )
    from unified_event_bus import UnifiedEventBus
    from roi_results_db import ROIResultsDB
    from workflow_stability_db import WorkflowStabilityDB
    from sandbox_runner import launch_sandbox
    from sandbox_runner.bootstrap import bootstrap_environment
    from self_learning_service import run_background as run_learning_background
    from self_test_service import SelfTestService
    import asyncio
    import threading

    try:
        settings = load_sandbox_settings(resolve_path(config_path))
    except ValidationError as exc:
        raise SystemExit(f"Invalid bootstrap configuration: {exc}") from exc

    bootstrap_environment(settings, _verify_required_dependencies)
    os.environ.setdefault("SANDBOX_REPO_PATH", settings.sandbox_repo_path)
    os.environ.setdefault("SANDBOX_DATA_DIR", resolve_path(settings.sandbox_data_dir))

    init_self_improvement(settings)

    try:
        ROIResultsDB()
        WorkflowStabilityDB()
    except Exception:
        logger.warning("database initialisation failed", exc_info=True)

    try:
        bus = UnifiedEventBus()
    except Exception:
        bus = None
        logger.warning("UnifiedEventBus unavailable")

    cleanup_funcs: list[Callable[[], None]] = []

    class _ExceptionThread(threading.Thread):
        """Thread subclass that stores exceptions from its target."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.exception: BaseException | None = None

        def run(self) -> None:  # pragma: no cover - defensive
            try:
                super().run()
            except BaseException as exc:  # pragma: no cover - runtime safety
                self.exception = exc
                raise

    learn_start, learn_stop = run_learning_background()

    _orig_thread = threading.Thread
    learn_thread: _ExceptionThread | None = None
    try:
        threading.Thread = _ExceptionThread  # type: ignore[assignment]
        learn_start()
        if learn_start.__closure__ is not None:
            for cell in learn_start.__closure__:
                obj = cell.cell_contents
                if isinstance(obj, _ExceptionThread):
                    learn_thread = obj
                    break
    finally:
        threading.Thread = _orig_thread

    monitor_stop = threading.Event()

    def _monitor_learning() -> None:
        if learn_thread is None:
            return
        while not monitor_stop.wait(5.0):
            if learn_thread.exception is not None:
                logger.critical(
                    "self-learning service failed", exc_info=learn_thread.exception
                )
                _thread.interrupt_main()
                return
            if not learn_thread.is_alive():
                logger.critical("self-learning service exited unexpectedly")
                _thread.interrupt_main()
                return

    monitor_thread = threading.Thread(target=_monitor_learning, daemon=True)
    monitor_thread.start()

    def _stop_monitor() -> None:
        monitor_stop.set()
        monitor_thread.join(timeout=1.0)

    cleanup_funcs.append(_stop_monitor)
    cleanup_funcs.append(learn_stop)

    from vector_service.context_builder import ContextBuilder
    from context_builder_util import ensure_fresh_weights

    builder = ContextBuilder("bots.db", "code.db", "errors.db", "workflows.db")
    try:
        ensure_fresh_weights(builder)
    except Exception:  # pragma: no cover - log and skip self-test when init fails
        logger.exception(
            "ContextBuilder initialisation failed; self-test loop disabled"
        )
    else:
        tester = SelfTestService(context_builder=builder)
        test_loop = asyncio.new_event_loop()

        def _tester_thread() -> None:
            asyncio.set_event_loop(test_loop)
            tester.run_continuous(loop=test_loop)
            test_loop.run_forever()

        t = threading.Thread(target=_tester_thread, daemon=True)
        t.start()

        def _stop_tests() -> None:
            test_loop.call_soon_threadsafe(lambda: asyncio.create_task(tester.stop()))
            test_loop.call_soon_threadsafe(test_loop.stop)
            t.join(timeout=1.0)

        cleanup_funcs.append(_stop_tests)

    def _noop():
        return None

    cycle_thread = start_self_improvement_cycle({"bootstrap": _noop}, event_bus=bus)
    cycle_thread.start()
    cleanup_funcs.append(stop_self_improvement_cycle)

    def _cleanup() -> None:
        if learn_thread is not None and getattr(learn_thread, "exception", None):
            logger.critical(
                "self-learning service failed", exc_info=learn_thread.exception
            )
        for func in cleanup_funcs:
            try:
                func()
            except Exception:
                logger.exception("cleanup failed")

    atexit.register(_cleanup)

    try:
        launch_sandbox(settings=settings)
    finally:
        _cleanup()


if __name__ == "__main__":
    main()
