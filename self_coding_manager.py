from __future__ import annotations

"""Manage self-coding patches and deployment cycles.

Many operations require a provenance token issued by the active
``EvolutionOrchestrator``.  Call :func:`validate_provenance` to verify that
requests originate from the orchestrator before proceeding.
"""

from pathlib import Path
import sys
import concurrent.futures
from collections import deque

try:  # pragma: no cover - allow flat imports
    from .dynamic_path_router import resolve_path, path_for_prompt
except Exception:  # pragma: no cover - fallback for flat layout
    from dynamic_path_router import resolve_path, path_for_prompt  # type: ignore
try:  # pragma: no cover - import module for cache management
    from . import dynamic_path_router as _path_router
except Exception:  # pragma: no cover - fallback for flat layout
    import dynamic_path_router as _path_router  # type: ignore
try:  # pragma: no cover - objective integrity guard
    from .objective_guard import ObjectiveGuard, ObjectiveGuardViolation
except Exception:  # pragma: no cover - fallback for flat layout
    from objective_guard import ObjectiveGuard, ObjectiveGuardViolation  # type: ignore
try:  # pragma: no cover - shared objective hash-lock verifier
    from .objective_hash_lock import verify_objective_hash_lock
except Exception:  # pragma: no cover - fallback for flat layout
    from objective_hash_lock import verify_objective_hash_lock  # type: ignore
try:  # pragma: no cover - policy import for unsafe paths
    from .self_coding_policy import (
        ensure_self_coding_unsafe_paths_env,
        get_patch_promotion_policy,
        is_self_coding_unsafe_path,
    )
except Exception:  # pragma: no cover - fallback for flat layout
    from self_coding_policy import (  # type: ignore
        ensure_self_coding_unsafe_paths_env,
        get_patch_promotion_policy,
        is_self_coding_unsafe_path,
    )
try:  # pragma: no cover - allow package/flat imports
    from .self_coding_divergence_detector import (
        CycleMetricsRecord,
        SelfCodingDivergenceDetector,
        load_divergence_detector_config,
    )
except Exception:  # pragma: no cover - fallback for flat layout
    from self_coding_divergence_detector import (  # type: ignore
        CycleMetricsRecord,
        SelfCodingDivergenceDetector,
        load_divergence_detector_config,
    )
import logging
import subprocess
import tempfile
import threading
import time
import traceback
import re
import json
import uuid
import os
import importlib
import shlex
import hashlib
from datetime import datetime, timezone
from dataclasses import asdict, dataclass
from typing import Dict, Any, TYPE_CHECKING, Callable, Iterator, Iterable

ensure_self_coding_unsafe_paths_env()

# Delay between successive internalisation attempts to avoid overwhelming the
# manager bootstrap logic when multiple bots start simultaneously.
_INTERNALIZE_THROTTLE_SECONDS = 1.5
_INTERNALIZE_THROTTLE_LOCK = threading.Lock()
_LAST_INTERNALIZE_AT = 0.0
_INTERNALIZE_IN_FLIGHT_LOCK = threading.Lock()
_INTERNALIZE_IN_FLIGHT: dict[str, float] = {}
_INTERNALIZE_STALE_TIMEOUT_SECONDS = float(
    os.getenv("SELF_CODING_INTERNALIZE_STALE_TIMEOUT_SECONDS", "1200")
)
_INTERNALIZE_IN_FLIGHT_WARN_THRESHOLD_SECONDS = float(
    os.getenv("SELF_CODING_INTERNALIZE_IN_FLIGHT_WARN_THRESHOLD_SECONDS", "300")
)
_RAW_INTERNALIZE_HARD_TIMEOUT_SECONDS = os.getenv(
    "SELF_CODING_INTERNALIZE_HARD_TIMEOUT_SECONDS", ""
).strip()
if _RAW_INTERNALIZE_HARD_TIMEOUT_SECONDS:
    try:
        _INTERNALIZE_HARD_TIMEOUT_SECONDS = float(_RAW_INTERNALIZE_HARD_TIMEOUT_SECONDS)
    except ValueError:
        _INTERNALIZE_HARD_TIMEOUT_SECONDS = max(
            _INTERNALIZE_IN_FLIGHT_WARN_THRESHOLD_SECONDS * 2,
            _INTERNALIZE_IN_FLIGHT_WARN_THRESHOLD_SECONDS,
        )
else:
    _INTERNALIZE_HARD_TIMEOUT_SECONDS = max(
        _INTERNALIZE_IN_FLIGHT_WARN_THRESHOLD_SECONDS * 2,
        _INTERNALIZE_IN_FLIGHT_WARN_THRESHOLD_SECONDS,
    )
_INTERNALIZE_MONITOR_INTERVAL_SECONDS = float(
    os.getenv("SELF_CODING_INTERNALIZE_MONITOR_INTERVAL_SECONDS", "30")
)
_RAW_INTERNALIZE_FORCE_CLEAR_MAX_AGE_SECONDS = os.getenv(
    "SELF_CODING_INTERNALIZE_FORCE_CLEAR_MAX_AGE_SECONDS", ""
).strip()
if _RAW_INTERNALIZE_FORCE_CLEAR_MAX_AGE_SECONDS:
    try:
        _INTERNALIZE_FORCE_CLEAR_MAX_AGE_SECONDS = max(
            float(_RAW_INTERNALIZE_FORCE_CLEAR_MAX_AGE_SECONDS),
            0.0,
        )
    except ValueError:
        _INTERNALIZE_FORCE_CLEAR_MAX_AGE_SECONDS = max(
            _INTERNALIZE_IN_FLIGHT_WARN_THRESHOLD_SECONDS * 2,
            _INTERNALIZE_IN_FLIGHT_WARN_THRESHOLD_SECONDS,
        )
else:
    _INTERNALIZE_FORCE_CLEAR_MAX_AGE_SECONDS = max(
        _INTERNALIZE_IN_FLIGHT_WARN_THRESHOLD_SECONDS * 2,
        _INTERNALIZE_IN_FLIGHT_WARN_THRESHOLD_SECONDS,
    )
_INTERNALIZE_BOT_LOCKS_LOCK = threading.Lock()
_INTERNALIZE_BOT_LOCKS: dict[str, threading.Lock] = {}
_INTERNALIZE_REUSE_WINDOW_SECONDS = float(
    os.getenv("SELF_CODING_INTERNALIZE_REUSE_WINDOW_SECONDS", "90")
)
_INTERNALIZE_FAILURE_THRESHOLD = int(
    os.getenv("SELF_CODING_INTERNALIZE_FAILURE_THRESHOLD", "3")
)
_INTERNALIZE_FAILURE_COOLDOWN_SECONDS = float(
    os.getenv("SELF_CODING_INTERNALIZE_FAILURE_COOLDOWN_SECONDS", "300")
)
_INTERNALIZE_FAILURE_LOCK = threading.Lock()
_INTERNALIZE_FAILURE_STATE: dict[str, dict[str, float | int | str | None]] = {}
_INTERNALIZE_MONITOR_THREAD: threading.Thread | None = None
_INTERNALIZE_MONITOR_STARTED = False
_INTERNALIZE_MONITOR_START_LOCK = threading.Lock()
_INTERNALIZE_MONITOR_LAST_LOGGED_AT: dict[str, float] = {}
_RAW_INTERNALIZE_DEBOUNCE_SECONDS = os.getenv(
    "SELF_CODING_INTERNALIZE_DEBOUNCE_SECONDS", "60"
).strip()
try:
    _INTERNALIZE_DEBOUNCE_SECONDS = float(_RAW_INTERNALIZE_DEBOUNCE_SECONDS)
except ValueError:
    _INTERNALIZE_DEBOUNCE_SECONDS = 60.0
_INTERNALIZE_DEBOUNCE_MIN_SECONDS = 30.0
_INTERNALIZE_DEBOUNCE_MAX_SECONDS = 120.0
_INTERNALIZE_DEBOUNCE_LOCK = threading.Lock()
_INTERNALIZE_LAST_ATTEMPT_STARTED_AT: dict[str, float] = {}
_SELF_DEBUG_BACKOFF_SECONDS = float(
    os.getenv("SELF_CODING_SELF_DEBUG_BACKOFF_SECONDS", "600")
)
_INTERNALIZE_TIMEOUT_RETRY_BACKOFF_SECONDS = float(
    os.getenv("SELF_CODING_INTERNALIZE_TIMEOUT_RETRY_BACKOFF_SECONDS", "15")
)
_RAW_MANAGER_CONSTRUCTION_TIMEOUT_SECONDS = os.getenv(
    "SELF_CODING_MANAGER_CONSTRUCTION_TIMEOUT_SECONDS", "45"
).strip()
try:
    _MANAGER_CONSTRUCTION_TIMEOUT_SECONDS = float(
        _RAW_MANAGER_CONSTRUCTION_TIMEOUT_SECONDS
    )
except ValueError:
    _MANAGER_CONSTRUCTION_TIMEOUT_SECONDS = 45.0
_RAW_BOTPLANNINGBOT_TIMEOUT_FALLBACK_SECONDS = os.getenv(
    "SELF_CODING_MANAGER_CONSTRUCTION_TIMEOUT_SECONDS_BOTPLANNINGBOT", "105"
).strip()
try:
    _BOTPLANNINGBOT_MANAGER_CONSTRUCTION_TIMEOUT_FALLBACK_SECONDS = float(
        _RAW_BOTPLANNINGBOT_TIMEOUT_FALLBACK_SECONDS
    )
except ValueError:
    _BOTPLANNINGBOT_MANAGER_CONSTRUCTION_TIMEOUT_FALLBACK_SECONDS = 105.0
_HEAVY_MANAGER_TIMEOUT_BOT_KEYS = {"BOTPLANNINGBOT"}
_OBJECTIVE_INTEGRITY_LOCK_PATH = Path("config/objective_integrity_lock.json")
_RAW_HEAVY_MANAGER_TIMEOUT_MIN_SECONDS = os.getenv(
    "SELF_CODING_MANAGER_CONSTRUCTION_TIMEOUT_MIN_SECONDS_HEAVY_BOTS", "90"
).strip()
try:
    _HEAVY_MANAGER_TIMEOUT_MIN_SECONDS = float(_RAW_HEAVY_MANAGER_TIMEOUT_MIN_SECONDS)
except ValueError:
    _HEAVY_MANAGER_TIMEOUT_MIN_SECONDS = 90.0


def _warn_on_low_heavy_bot_timeout(
    *, bot_name: str, bot_key: str, timeout_seconds: float, source: str
) -> None:
    """Warn when heavy bots are configured with fragile timeout budgets."""

    if (
        bot_key in _HEAVY_MANAGER_TIMEOUT_BOT_KEYS
        and timeout_seconds < _HEAVY_MANAGER_TIMEOUT_MIN_SECONDS
    ):
        logging.getLogger(__name__).warning(
            "manager timeout %.2fs for %s from %s is below recommended %.2fs "
            "for heavy bot startup",
            timeout_seconds,
            bot_name,
            source,
            _HEAVY_MANAGER_TIMEOUT_MIN_SECONDS,
        )


def _normalize_env_bot_name(bot_name: str) -> str:
    """Return an env-var-safe key for a bot name."""

    normalized = re.sub(r"[^A-Za-z0-9]+", "_", str(bot_name)).strip("_")
    return normalized.upper() or "UNKNOWN"


def _resolve_manager_timeout_seconds(bot_name: str) -> float:
    """Resolve manager construction timeout with optional per-bot overrides."""

    logger = logging.getLogger(__name__)

    bot_key = _normalize_env_bot_name(bot_name)
    candidate_vars: list[str] = []
    if bot_key == "BOTPLANNINGBOT":
        candidate_vars.append(
            "SELF_CODING_MANAGER_CONSTRUCTION_TIMEOUT_SECONDS_BOTPLANNINGBOT"
        )
    candidate_vars.extend(
        [
            f"SELF_CODING_MANAGER_CONSTRUCTION_TIMEOUT_SECONDS_{bot_key}",
            f"SELF_CODING_MANAGER_TIMEOUT_SECONDS_{bot_key}",
        ]
    )
    for env_var in candidate_vars:
        raw_value = os.getenv(env_var)
        if raw_value is None:
            continue
        try:
            resolved_timeout = max(float(raw_value.strip()), 0.0)
            _warn_on_low_heavy_bot_timeout(
                bot_name=bot_name,
                bot_key=bot_key,
                timeout_seconds=resolved_timeout,
                source=env_var,
            )
            logger.info(
                "resolved manager construction timeout for %s (%s): %.2fs via %s",
                bot_name,
                bot_key,
                resolved_timeout,
                env_var,
                extra={
                    "event": "manager_construction_timeout_resolved",
                    "bot_name": bot_name,
                    "bot_key": bot_key,
                    "manager_timeout_seconds": resolved_timeout,
                    "source": env_var,
                },
            )
            return resolved_timeout
        except ValueError:
            logger.warning(
                "invalid manager timeout override %s=%r for %s; using default %.2fs",
                env_var,
                raw_value,
                bot_name,
                _MANAGER_CONSTRUCTION_TIMEOUT_SECONDS,
            )
            continue
    if bot_key == "BOTPLANNINGBOT":
        resolved_timeout = max(
            _BOTPLANNINGBOT_MANAGER_CONSTRUCTION_TIMEOUT_FALLBACK_SECONDS,
            _MANAGER_CONSTRUCTION_TIMEOUT_SECONDS,
            0.0,
        )
        source = "botplanningbot_default_fallback"
        _warn_on_low_heavy_bot_timeout(
            bot_name=bot_name,
            bot_key=bot_key,
            timeout_seconds=resolved_timeout,
            source=source,
        )
        logger.info(
            "resolved manager construction timeout for %s (%s): %.2fs via %s",
            bot_name,
            bot_key,
            resolved_timeout,
            source,
            extra={
                "event": "manager_construction_timeout_resolved",
                "bot_name": bot_name,
                "bot_key": bot_key,
                "manager_timeout_seconds": resolved_timeout,
                "source": source,
            },
        )
        return resolved_timeout
    resolved_timeout = max(_MANAGER_CONSTRUCTION_TIMEOUT_SECONDS, 0.0)
    logger.info(
        "resolved manager construction timeout for %s (%s): %.2fs via %s",
        bot_name,
        bot_key,
        resolved_timeout,
        "global_default",
        extra={
            "event": "manager_construction_timeout_resolved",
            "bot_name": bot_name,
            "bot_key": bot_key,
            "manager_timeout_seconds": resolved_timeout,
            "source": "global_default",
        },
    )
    _warn_on_low_heavy_bot_timeout(
        bot_name=bot_name,
        bot_key=bot_key,
        timeout_seconds=resolved_timeout,
        source="global_default",
    )
    return resolved_timeout


def _resolve_internalize_debounce_seconds(bot_name: str) -> float:
    """Resolve per-bot debounce window for internalization attempts."""

    bot_key = _normalize_env_bot_name(bot_name)
    candidate_vars = [
        f"SELF_CODING_INTERNALIZE_DEBOUNCE_SECONDS_{bot_key}",
        f"SELF_CODING_INTERNALIZATION_DEBOUNCE_SECONDS_{bot_key}",
    ]
    for env_var in candidate_vars:
        raw_value = os.getenv(env_var)
        if raw_value is None:
            continue
        try:
            value = float(raw_value.strip())
            return max(
                _INTERNALIZE_DEBOUNCE_MIN_SECONDS,
                min(_INTERNALIZE_DEBOUNCE_MAX_SECONDS, value),
            )
        except ValueError:
            logging.getLogger(__name__).warning(
                "invalid internalize debounce override %s=%r for %s; using default",
                env_var,
                raw_value,
                bot_name,
            )
            break
    return max(
        _INTERNALIZE_DEBOUNCE_MIN_SECONDS,
        min(_INTERNALIZE_DEBOUNCE_MAX_SECONDS, _INTERNALIZE_DEBOUNCE_SECONDS),
    )

def _resolve_manager_retry_timeout_seconds(bot_name: str, *, primary_timeout: float) -> float:
    """Resolve manager construction retry timeout for bounded retries."""

    bot_key = _normalize_env_bot_name(bot_name)
    candidate_vars: list[str] = []
    if bot_key == "BOTPLANNINGBOT":
        candidate_vars.append(
            "SELF_CODING_MANAGER_CONSTRUCTION_RETRY_TIMEOUT_SECONDS_BOTPLANNINGBOT"
        )
    candidate_vars.append(
        f"SELF_CODING_MANAGER_CONSTRUCTION_RETRY_TIMEOUT_SECONDS_{bot_key}"
    )
    for env_var in candidate_vars:
        raw_value = os.getenv(env_var)
        if raw_value is None:
            continue
        try:
            return max(float(raw_value.strip()), primary_timeout, 0.0)
        except ValueError:
            logging.getLogger(__name__).warning(
                "invalid manager retry timeout override %s=%r for %s; using default",
                env_var,
                raw_value,
                bot_name,
            )
            break

    default_retry_timeout = max(primary_timeout + 15.0, primary_timeout * 1.5, 60.0)
    return max(default_retry_timeout, 0.0)




_MANAGER_PHASE_DURATION_ROLLING_LIMIT = max(1, int(os.getenv("SELF_CODING_MANAGER_PHASE_DURATION_ROLLING_LIMIT", "8")))


def _compute_adaptive_manager_retry_timeout_seconds(
    bot_name: str,
    *,
    primary_timeout: float,
    timeout_phase: str,
    timeout_phase_elapsed_seconds: float,
    timeout_history: list[tuple[str, float]],
    phase_metrics: dict[str, Any] | None,
) -> float:
    """Return a bounded adaptive timeout using successful phase timing history."""

    configured_retry = _resolve_manager_retry_timeout_seconds(
        bot_name, primary_timeout=primary_timeout
    )
    hard_cap = max(primary_timeout * 2.0, primary_timeout, 0.0)
    retry_timeout = min(max(configured_retry, primary_timeout), hard_cap)

    completed_manager_init = any(phase == "manager_init:return" for phase, _ in timeout_history)
    if completed_manager_init:
        return retry_timeout

    phase_key = str(timeout_phase or "unknown")
    metrics = phase_metrics or {}
    phase_entry = metrics.get(phase_key) if isinstance(metrics, dict) else None
    successful_samples = []
    if isinstance(phase_entry, dict):
        raw_samples = phase_entry.get("successful_elapsed_seconds", [])
        if isinstance(raw_samples, list):
            successful_samples = [
                float(sample)
                for sample in raw_samples
                if isinstance(sample, (int, float)) and float(sample) > 0.0
            ]

    if not successful_samples:
        return retry_timeout

    longest_success = max(successful_samples)
    if longest_success <= timeout_phase_elapsed_seconds:
        return retry_timeout

    adaptive_candidate = max(retry_timeout, longest_success * 1.1)
    return min(max(adaptive_candidate, primary_timeout), hard_cap)

_SELF_DEBUG_BACKOFF_LOCK = threading.Lock()
_SELF_DEBUG_LAST_TRIGGER: dict[str, float] = {}
_INTERNALIZE_TIMEOUT_RETRY_LOCK = threading.Lock()
_INTERNALIZE_TIMEOUT_RETRY_STATE: dict[str, float] = {}
_RAW_BROKER_OWNER_READY_TIMEOUT = os.getenv(
    "BROKER_OWNER_READY_TIMEOUT_SECS", "3.0"
).strip()
try:
    _BROKER_OWNER_READY_TIMEOUT_SECS = float(_RAW_BROKER_OWNER_READY_TIMEOUT)
except ValueError:
    _BROKER_OWNER_READY_TIMEOUT_SECS = 3.0
from contextlib import contextmanager

try:  # pragma: no cover - optional dependency in stripped environments
    from .failure_guard import FailureGuard
except Exception:  # pragma: no cover - fallback for flat layouts
    try:
        from failure_guard import FailureGuard  # type: ignore
    except Exception:  # pragma: no cover - guard is best-effort
        FailureGuard = None  # type: ignore

from .error_parser import FailureCache, ErrorReport, ErrorParser
from .failure_fingerprint_store import (
    FailureFingerprint,
    FailureFingerprintStore,
)
from .failure_retry_utils import check_similarity_and_warn, record_failure
from vector_service.context_builder import (
    record_failed_tags,
    load_failed_tags,
    ContextBuilder,
)

from .sandbox_runner.test_harness import run_tests, TestHarnessResult

from .self_coding_engine import SelfCodingEngine
from .data_bot import DataBot, persist_sc_thresholds
try:  # pragma: no cover - optional dependency
    from .advanced_error_management import FormalVerifier, AutomatedRollbackManager
except Exception:  # pragma: no cover - provide stubs in minimal environments
    FormalVerifier = AutomatedRollbackManager = None  # type: ignore
try:  # pragma: no cover - optional dependency
    from . import mutation_logger as MutationLogger
except Exception:  # pragma: no cover - provide stub when unavailable
    MutationLogger = None  # type: ignore
try:  # pragma: no cover - optional dependency
    from .rollback_manager import RollbackManager
except Exception:  # pragma: no cover - provide stub when unavailable
    RollbackManager = None  # type: ignore
from .sandbox_settings import SandboxSettings, normalize_workflow_tests
from .patch_attempt_tracker import PatchAttemptTracker
from .threshold_service import (
    ThresholdService,
    threshold_service as _DEFAULT_THRESHOLD_SERVICE,
)

try:  # pragma: no cover - tolerate deployments without the guard helper
    from .shared.self_coding_import_guard import self_coding_import_depth as _import_depth
except Exception:  # pragma: no cover - fallback for flat layout or minimal bundles
    try:
        from shared.self_coding_import_guard import self_coding_import_depth as _import_depth  # type: ignore
    except Exception:  # pragma: no cover - guard unavailable in stripped environments
        _import_depth = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from .quick_fix_engine import (
        QuickFixEngine,
        QuickFixEngineError,
        generate_patch,
    )
except Exception as exc:  # pragma: no cover - fail fast when unavailable
    raise ImportError(
        "QuickFixEngine is required but could not be imported"
    ) from exc

from context_builder_util import ensure_fresh_weights, create_context_builder

if TYPE_CHECKING:  # pragma: no cover - typing only import avoids circular dependency
    from .error_bot import ErrorDB
    from .model_automation_pipeline import AutomationResult, ModelAutomationPipeline

_MODEL_AUTOMATION_PIPELINE_CLS: type["ModelAutomationPipeline"] | None = None
_AUTOMATION_RESULT_CLS: type["AutomationResult"] | None = None

_INTERNALIZE_SHUTDOWN_RACE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"cannot\s+schedule\s+new\s+futures\s+after\s+interpreter\s+shutdown", re.IGNORECASE),
    re.compile(r"cannot\s+schedule\s+new\s+futures\s+after\s+shutdown", re.IGNORECASE),
    re.compile(r"event\s+loop\s+is\s+closed", re.IGNORECASE),
    re.compile(r"sys\.is_finalizing", re.IGNORECASE),
    re.compile(r"interpreter\s+shutdown", re.IGNORECASE),
    re.compile(r"python\s+finaliz", re.IGNORECASE),
)


def _is_internalize_shutdown_race(reason: str) -> bool:
    """Return ``True`` when *reason* matches known shutdown race messages."""

    reason_text = str(reason or "")
    return any(pattern.search(reason_text) for pattern in _INTERNALIZE_SHUTDOWN_RACE_PATTERNS)


def _load_pipeline_components() -> tuple[type["ModelAutomationPipeline"], type["AutomationResult"]]:
    """Import ``ModelAutomationPipeline`` and ``AutomationResult`` lazily."""

    global _MODEL_AUTOMATION_PIPELINE_CLS, _AUTOMATION_RESULT_CLS
    if _MODEL_AUTOMATION_PIPELINE_CLS is None or _AUTOMATION_RESULT_CLS is None:
        from .model_automation_pipeline import (  # Local import avoids circular dependency
            ModelAutomationPipeline as _Pipeline,
            AutomationResult as _AutomationResult,
        )

        _MODEL_AUTOMATION_PIPELINE_CLS = _Pipeline
        _AUTOMATION_RESULT_CLS = _AutomationResult
    return _MODEL_AUTOMATION_PIPELINE_CLS, _AUTOMATION_RESULT_CLS


def _automation_result(*args: Any, **kwargs: Any) -> "AutomationResult":
    """Return an ``AutomationResult`` instance importing lazily when required."""

    _, result_cls = _load_pipeline_components()
    return result_cls(*args, **kwargs)


def _emit_timing_marker(
    logger: logging.Logger,
    bot_name: str,
    stage: str,
    *,
    event: str,
    elapsed_seconds: float | None = None,
) -> None:
    """Emit structured timing markers for manager construction diagnostics."""

    payload: dict[str, Any] = {
        "event": "self_coding_manager_timing",
        "bot": bot_name,
        "stage": stage,
        "marker": event,
    }
    if elapsed_seconds is not None:
        payload["elapsed_seconds"] = elapsed_seconds
    logger.info(
        "self_coding_manager timing marker %s stage=%s bot=%s",
        event,
        stage,
        bot_name,
        extra=payload,
    )


@contextmanager
def _timed_marker(logger: logging.Logger, bot_name: str, stage: str) -> Iterator[None]:
    """Emit begin/end timing markers around expensive manager stages."""

    started_at = time.monotonic()
    _emit_timing_marker(logger, bot_name, stage, event="begin")
    try:
        yield
    finally:
        elapsed = max(0.0, time.monotonic() - started_at)
        _emit_timing_marker(
            logger,
            bot_name,
            stage,
            event="end",
            elapsed_seconds=elapsed,
        )

if TYPE_CHECKING:  # pragma: no cover - typing only import avoids circular dependency
    from .coding_bot_interface import manager_generate_helper as _ManagerGenerateHelperProto
else:  # pragma: no cover - runtime fallback type to avoid import cycle
    _ManagerGenerateHelperProto = Callable[..., str]

_BASE_MANAGER_GENERATE_HELPER: _ManagerGenerateHelperProto | None = None


def _get_base_manager_generate_helper() -> _ManagerGenerateHelperProto:
    """Lazily import ``manager_generate_helper`` to avoid circular imports."""

    global _BASE_MANAGER_GENERATE_HELPER
    if _BASE_MANAGER_GENERATE_HELPER is not None:
        return _BASE_MANAGER_GENERATE_HELPER

    try:  # pragma: no cover - prefer relative import when packaged
        from .coding_bot_interface import manager_generate_helper as _imported_helper
    except Exception:  # pragma: no cover - support flat execution layouts
        from coding_bot_interface import (  # type: ignore
            manager_generate_helper as _imported_helper,
        )

    _BASE_MANAGER_GENERATE_HELPER = _imported_helper
    return _imported_helper


_DEFAULT_CREATE_CONTEXT_BUILDER = create_context_builder
_DEFAULT_CONTEXT_BUILDER_CLS = ContextBuilder


def _current_self_coding_import_depth() -> int:
    """Return the active self-coding import depth if the guard is available."""

    depth_source = _import_depth
    if depth_source is None:
        return 0

    getter = getattr(depth_source, "get", None)
    if callable(getter):
        try:
            return int(getter(0))
        except Exception:  # pragma: no cover - guard errors should not block bootstrap
            return 0

    if callable(depth_source):
        try:
            return int(depth_source())
        except Exception:  # pragma: no cover - tolerate guard failures gracefully
            return 0

    return 0


def _get_internalize_lock(bot_name: str) -> threading.Lock:
    with _INTERNALIZE_BOT_LOCKS_LOCK:
        lock = _INTERNALIZE_BOT_LOCKS.get(bot_name)
        if lock is None:
            lock = threading.Lock()
            _INTERNALIZE_BOT_LOCKS[bot_name] = lock
    return lock


def _consume_stale_internalize_in_flight(
    *,
    now: float,
) -> list[tuple[str, float]]:
    """Return and remove stale in-flight entries older than the configured timeout."""

    timeout = _INTERNALIZE_STALE_TIMEOUT_SECONDS
    if timeout <= 0:
        return []

    stale: list[tuple[str, float]] = []
    with _INTERNALIZE_IN_FLIGHT_LOCK:
        for candidate, started_at in list(_INTERNALIZE_IN_FLIGHT.items()):
            if now - started_at <= timeout:
                continue
            stale.append((candidate, started_at))
            _INTERNALIZE_IN_FLIGHT.pop(candidate, None)
    return stale


def _replace_internalize_lock(bot_name: str) -> None:
    """Replace the per-bot lock so fresh attempts are not blocked by stale state."""

    with _INTERNALIZE_BOT_LOCKS_LOCK:
        previous = _INTERNALIZE_BOT_LOCKS.get(bot_name)
        if previous is not None and not previous.locked():
            return
        _INTERNALIZE_BOT_LOCKS[bot_name] = threading.Lock()


def _force_clear_in_flight_entry(
    *,
    bot_name: str,
    started_at: float,
    age_seconds: float,
    reason: str,
    logger: logging.Logger,
    bot_registry: Any = None,
) -> None:
    """Force-clear stale in-flight state and emit structured diagnostics."""

    node = None
    with _INTERNALIZE_IN_FLIGHT_LOCK:
        removed_at = _INTERNALIZE_IN_FLIGHT.pop(bot_name, None)
        _INTERNALIZE_MONITOR_LAST_LOGGED_AT.pop(bot_name, None)
    if removed_at is None:
        return

    if bot_registry is not None:
        try:
            node = bot_registry.graph.nodes.get(bot_name)
        except Exception:
            node = None

    last_step = None
    started_epoch = None
    if node is not None:
        last_step = node.get("internalization_last_step")
        started_epoch = node.pop("internalization_in_progress", None)
        node["internalization_last_step"] = f"forced stale cleanup ({reason})"

    _replace_internalize_lock(bot_name)
    logger.warning(
        "force-cleared internalize in-flight entry for %s after %.1fs (%s)",
        bot_name,
        age_seconds,
        reason,
        extra={
            "event": "internalize_in_flight_force_cleared",
            "bot": bot_name,
            "reason": reason,
            "started_at_monotonic": started_at,
            "in_flight_seconds": age_seconds,
            "internalization_last_step": last_step,
            "internalization_in_progress": started_epoch,
            "force_clear_max_age_seconds": _INTERNALIZE_FORCE_CLEAR_MAX_AGE_SECONDS,
        },
    )

    _record_internalize_failure(
        bot_name,
        module_path=None,
        reason=reason,
        logger=logger,
    )
    _record_stale_internalization_failure_event(
        bot_name=bot_name,
        started_at=started_at,
        age_seconds=age_seconds,
        threshold_seconds=_INTERNALIZE_FORCE_CLEAR_MAX_AGE_SECONDS,
        logger=logger,
        bot_registry=bot_registry,
        node=node,
    )
    _schedule_internalization_timeout_retry(
        bot_name=bot_name,
        logger=logger,
        bot_registry=bot_registry,
    )


def _record_stale_internalization_failure_event(
    *,
    bot_name: str,
    started_at: float,
    age_seconds: float,
    threshold_seconds: float,
    logger: logging.Logger,
    bot_registry: Any = None,
    node: dict[str, Any] | None = None,
) -> None:
    payload = {
        "bot": bot_name,
        "reason": "stale_internalization_timeout",
        "started_at_monotonic": started_at,
        "in_flight_seconds": age_seconds,
        "hard_timeout_seconds": threshold_seconds,
        "timestamp": time.time(),
    }
    event_bus = None
    if node is not None:
        manager = node.get("selfcoding_manager") or node.get("manager")
        if manager is not None:
            event_bus = getattr(manager, "event_bus", None)
    if event_bus is None and bot_registry is not None:
        event_bus = getattr(bot_registry, "event_bus", None)
    if event_bus is None:
        return
    try:
        event_bus.publish("self_coding:internalization_failure", payload)
    except Exception:
        logger.exception(
            "failed to publish stale internalization timeout failure for %s",
            bot_name,
        )


def _launch_internalize_timeout_self_debug(
    *,
    bot_name: str,
    age_seconds: float,
    timeout_seconds: float,
    started_at: float,
    logger: logging.Logger,
) -> None:
    if not _should_trigger_self_debug(bot_name):
        return

    payload = {
        "event": "stale_internalization_timeout",
        "bot": bot_name,
        "in_flight_seconds": age_seconds,
        "hard_timeout_seconds": timeout_seconds,
        "started_at_monotonic": started_at,
        "timestamp": time.time(),
    }
    try:
        context_dir = Path(tempfile.mkdtemp(prefix="stale_internalize_self_debug_"))
        context_path = context_dir / "failure_context.json"
        context_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    except Exception:
        logger.exception(
            "failed to persist stale internalization timeout failure context for %s",
            bot_name,
        )
        return

    def _run_self_debug() -> None:
        try:
            from menace_sandbox import menace_workflow_self_debug
        except Exception:
            logger.exception(
                "self-debug import failed while handling stale internalization timeout for %s",
                bot_name,
                extra={"bot": bot_name, "context_path": str(context_path)},
            )
            return
        args = [
            "--repo-root",
            str(Path(".").resolve()),
            "--metrics-source",
            "stale_internalization_timeout",
            "--source-menace-id",
            "stale_internalization_timeout",
            "--failure-context-path",
            str(context_path),
        ]
        try:
            exit_code = menace_workflow_self_debug.main(args)
        except Exception:
            logger.exception(
                "self-debug run failed for stale internalization timeout on %s",
                bot_name,
                extra={"bot": bot_name},
            )
            return
        logger.info(
            "self-debug run completed for stale internalization timeout on %s",
            bot_name,
            extra={"bot": bot_name, "exit_code": exit_code},
        )

    thread = threading.Thread(
        target=_run_self_debug,
        name=f"stale_internalization_self_debug:{bot_name}",
        daemon=True,
    )
    thread.start()


def _schedule_internalization_timeout_retry(
    *,
    bot_name: str,
    logger: logging.Logger,
    bot_registry: Any = None,
) -> None:
    if bot_registry is None or not hasattr(bot_registry, "force_internalization_retry"):
        return
    now = time.monotonic()
    with _INTERNALIZE_TIMEOUT_RETRY_LOCK:
        retry_after = _INTERNALIZE_TIMEOUT_RETRY_STATE.get(bot_name, 0.0)
        if retry_after and now < retry_after:
            return
        _INTERNALIZE_TIMEOUT_RETRY_STATE[bot_name] = now + _INTERNALIZE_TIMEOUT_RETRY_BACKOFF_SECONDS
    try:
        bot_registry.force_internalization_retry(
            bot_name,
            delay=max(0.0, _INTERNALIZE_TIMEOUT_RETRY_BACKOFF_SECONDS),
        )
    except Exception:
        logger.exception(
            "failed to schedule stale internalization timeout retry for %s",
            bot_name,
        )


def _start_internalize_monitor(bot_registry: Any) -> None:
    """Start a lightweight monitor that warns about long-running internalization."""

    global _INTERNALIZE_MONITOR_STARTED, _INTERNALIZE_MONITOR_THREAD

    if (
        _INTERNALIZE_IN_FLIGHT_WARN_THRESHOLD_SECONDS <= 0
        or _INTERNALIZE_MONITOR_INTERVAL_SECONDS <= 0
    ):
        return

    with _INTERNALIZE_MONITOR_START_LOCK:
        if _INTERNALIZE_MONITOR_STARTED:
            return
        _INTERNALIZE_MONITOR_STARTED = True

        def _monitor() -> None:
            logger = logging.getLogger(__name__)

            while True:
                now = time.monotonic()
                with _INTERNALIZE_IN_FLIGHT_LOCK:
                    inflight_entries = list(_INTERNALIZE_IN_FLIGHT.items())
                for candidate, started_at in inflight_entries:
                    age = max(0.0, now - started_at)
                    if age < _INTERNALIZE_IN_FLIGHT_WARN_THRESHOLD_SECONDS:
                        continue

                    with _INTERNALIZE_IN_FLIGHT_LOCK:
                        last_logged_at = _INTERNALIZE_MONITOR_LAST_LOGGED_AT.get(
                            candidate, 0.0
                        )
                        if (
                            last_logged_at
                            and now - last_logged_at
                            < _INTERNALIZE_MONITOR_INTERVAL_SECONDS
                        ):
                            continue
                        _INTERNALIZE_MONITOR_LAST_LOGGED_AT[candidate] = now

                    node = None
                    if bot_registry is not None:
                        try:
                            node = bot_registry.graph.nodes.get(candidate)
                        except Exception:
                            node = None

                    last_step = None
                    started_epoch = None
                    if node is not None:
                        last_step = node.get("internalization_last_step")
                        started_epoch = node.get("internalization_in_progress")

                    logger.warning(
                        "internalize_coding_bot in-flight for %s beyond threshold "
                        "(%.1fs >= %.1fs); last_step=%s started_at=%s",
                        candidate,
                        age,
                        _INTERNALIZE_IN_FLIGHT_WARN_THRESHOLD_SECONDS,
                        last_step,
                        started_epoch,
                        extra={
                            "bot": candidate,
                            "in_flight_seconds": age,
                            "in_flight_warn_threshold_seconds": (
                                _INTERNALIZE_IN_FLIGHT_WARN_THRESHOLD_SECONDS
                            ),
                            "internalization_last_step": last_step,
                            "internalization_in_progress": started_epoch,
                        },
                    )

                    if (
                        _INTERNALIZE_FORCE_CLEAR_MAX_AGE_SECONDS > 0
                        and age >= _INTERNALIZE_FORCE_CLEAR_MAX_AGE_SECONDS
                    ):
                        _force_clear_in_flight_entry(
                            bot_name=candidate,
                            started_at=started_at,
                            age_seconds=age,
                            reason="stale_watchdog_force_clear",
                            logger=logger,
                            bot_registry=bot_registry,
                        )

                time.sleep(_INTERNALIZE_MONITOR_INTERVAL_SECONDS)

        monitor_thread = threading.Thread(
            target=_monitor,
            name="self-coding-internalize-monitor",
            daemon=True,
        )
        monitor_thread.start()
        _INTERNALIZE_MONITOR_THREAD = monitor_thread


def _internalize_in_cooldown(bot_name: str) -> bool:
    """Return True when *bot_name* is in a cooldown window."""

    if _INTERNALIZE_FAILURE_THRESHOLD <= 0:
        return False
    now = time.monotonic()
    with _INTERNALIZE_FAILURE_LOCK:
        state = _INTERNALIZE_FAILURE_STATE.get(bot_name)
        if not state:
            return False
        cooldown_until = float(state.get("cooldown_until", 0.0) or 0.0)
        if cooldown_until and now < cooldown_until:
            return True
        if cooldown_until and now >= cooldown_until:
            state["cooldown_until"] = 0.0
            state["count"] = 0
            _INTERNALIZE_FAILURE_STATE[bot_name] = state
    return False


def _should_trigger_self_debug(bot_name: str) -> bool:
    """Return True when a self-debug run should be triggered for *bot_name*."""

    if _SELF_DEBUG_BACKOFF_SECONDS <= 0:
        return True
    now = time.monotonic()
    with _SELF_DEBUG_BACKOFF_LOCK:
        last_trigger = _SELF_DEBUG_LAST_TRIGGER.get(bot_name)
        if last_trigger is not None and now - last_trigger < _SELF_DEBUG_BACKOFF_SECONDS:
            return False
        _SELF_DEBUG_LAST_TRIGGER[bot_name] = now
    return True


def _launch_module_path_self_debug(
    bot_name: str,
    module_candidates: dict[str, str | None],
    module_hint: str | None,
    manager: "SelfCodingManager",
) -> None:
    if not _should_trigger_self_debug(bot_name):
        if hasattr(manager, "logger"):
            try:
                manager.logger.info(
                    "self-debug backoff active for %s; skipping module_path_missing trigger",
                    bot_name,
                    extra={"bot": bot_name},
                )
            except Exception:
                pass
        return

    payload = {
        "event": "module_path_missing",
        "bot": bot_name,
        "module_hint": module_hint,
        "module_candidates": dict(module_candidates),
        "timestamp": time.time(),
    }
    try:
        context_dir = Path(tempfile.mkdtemp(prefix="module_path_self_debug_"))
        context_path = context_dir / "failure_context.json"
        context_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    except Exception:
        if hasattr(manager, "logger"):
            manager.logger.exception(
                "failed to persist module_path_missing failure context for %s",
                bot_name,
            )
        return

    def _run_self_debug() -> None:
        try:
            from menace_sandbox import menace_workflow_self_debug
        except Exception:
            if hasattr(manager, "logger"):
                manager.logger.exception(
                    "self-debug import failed while handling module_path_missing for %s",
                    bot_name,
                    extra={"bot": bot_name, "context_path": str(context_path)},
                )
            return

        args = [
            "--repo-root",
            str(Path(".").resolve()),
            "--metrics-source",
            "module_path_missing",
            "--source-menace-id",
            "module_path_missing",
            "--failure-context-path",
            str(context_path),
        ]
        try:
            exit_code = menace_workflow_self_debug.main(args)
        except Exception:
            if hasattr(manager, "logger"):
                manager.logger.exception(
                    "self-debug run failed for module_path_missing on %s",
                    bot_name,
                    extra={"bot": bot_name},
                )
            return
        if hasattr(manager, "logger"):
            try:
                manager.logger.info(
                    "self-debug run completed for module_path_missing on %s",
                    bot_name,
                    extra={"bot": bot_name, "exit_code": exit_code},
                )
            except Exception:
                pass

    thread = threading.Thread(
        target=_run_self_debug,
        name=f"module_path_self_debug:{bot_name}",
        daemon=True,
    )
    thread.start()


def _record_internalize_failure(
    bot_name: str,
    *,
    module_path: str | None,
    reason: str,
    logger: logging.Logger | None = None,
) -> None:
    """Track a failure and enter cooldown after the configured threshold."""

    reason_text = str(reason or "")
    reason_category = "shutdown_race" if _is_internalize_shutdown_race(reason_text) else reason_text
    now = time.monotonic()
    should_log = False
    should_warn_shutdown_race = False

    with _INTERNALIZE_FAILURE_LOCK:
        state = _INTERNALIZE_FAILURE_STATE.get(
            bot_name,
            {
                "count": 0,
                "cooldown_until": 0.0,
                "module_path": None,
                "reason": None,
                "shutdown_race_warned": 0,
            },
        )

        state["module_path"] = module_path
        state["reason"] = reason_category

        if reason_category == "shutdown_race":
            state["count"] = 0
            state["cooldown_until"] = 0.0
            if not bool(state.get("shutdown_race_warned", 0)):
                state["shutdown_race_warned"] = 1
                should_warn_shutdown_race = True
            _INTERNALIZE_FAILURE_STATE[bot_name] = state
        else:
            state["shutdown_race_warned"] = 0
            if _INTERNALIZE_FAILURE_THRESHOLD <= 0:
                _INTERNALIZE_FAILURE_STATE[bot_name] = state
                return
            cooldown_until = float(state.get("cooldown_until", 0.0) or 0.0)
            if cooldown_until and now < cooldown_until:
                _INTERNALIZE_FAILURE_STATE[bot_name] = state
                return
            state["count"] = int(state.get("count", 0) or 0) + 1
            if state["count"] >= _INTERNALIZE_FAILURE_THRESHOLD:
                state["count"] = 0
                state["cooldown_until"] = now + _INTERNALIZE_FAILURE_COOLDOWN_SECONDS
                should_log = True
            _INTERNALIZE_FAILURE_STATE[bot_name] = state

    log = logger or logging.getLogger(__name__)
    if should_warn_shutdown_race:
        log.warning(
            "internalize_coding_bot observed shutdown race for %s; skipping cooldown escalation "
            "(module_path=%s). This commonly occurs during interpreter/event-loop shutdown; "
            "allow shutdown to complete before re-internalizing.",
            bot_name,
            module_path or "unknown",
            extra={
                "bot": bot_name,
                "module_path": module_path or "unknown",
                "reason": "shutdown_race",
                "original_reason": reason_text,
                "remediation": "wait_for_shutdown_and_retry_internalize",
            },
        )
    if should_log:
        log.error(
            "internalize_coding_bot entering cooldown for %s after repeated failures "
            "(module_path=%s, reason=%s, cooldown=%.1fs)",
            bot_name,
            module_path or "unknown",
            reason_category,
            _INTERNALIZE_FAILURE_COOLDOWN_SECONDS,
            extra={
                "bot": bot_name,
                "module_path": module_path or "unknown",
                "reason": reason_category,
                "cooldown_seconds": _INTERNALIZE_FAILURE_COOLDOWN_SECONDS,
            },
        )


def _record_internalize_success(bot_name: str) -> None:
    """Reset failure tracking after a successful internalization."""

    if _INTERNALIZE_FAILURE_THRESHOLD <= 0:
        return
    with _INTERNALIZE_FAILURE_LOCK:
        state = _INTERNALIZE_FAILURE_STATE.get(bot_name)
        if not state:
            return
        state["count"] = 0
        state["cooldown_until"] = 0.0
        _INTERNALIZE_FAILURE_STATE[bot_name] = state


def _cooldown_disabled_manager(bot_registry: Any, data_bot: Any) -> Any:
    """Return a disabled manager placeholder when internalize is in cooldown."""
    module_name = (
        f"{__package__}.coding_bot_interface" if __package__ else "coding_bot_interface"
    )
    module = importlib.import_module(module_name)
    disabled_cls = getattr(module, "_DisabledSelfCodingManager")
    return disabled_cls(
        bot_registry=bot_registry,
        data_bot=data_bot,
        bootstrap_placeholder=True,
    )


def _manager_generate_helper_with_builder(
    manager,
    description: str,
    *,
    context_builder: ContextBuilder,
    **kwargs: Any,
) -> str:
    """Invoke the base helper ensuring a usable ``ContextBuilder`` is supplied."""

    if context_builder is None:  # pragma: no cover - defensive
        raise TypeError("context_builder is required")

    builder = context_builder
    ensure_fresh_weights(builder)
    base_helper = _get_base_manager_generate_helper()

    try:
        return base_helper(
            manager,
            description,
            context_builder=builder,
            **kwargs,
        )
    except TypeError:  # pragma: no cover - backwards compatibility for stubs
        return base_helper(manager, description, **kwargs)


for _mod_name in ("quick_fix_engine", "menace_sandbox.quick_fix_engine", "menace.quick_fix_engine"):
    _module = sys.modules.get(_mod_name)
    if _module is not None:
        try:
            setattr(
                _module,
                "manager_generate_helper",
                _manager_generate_helper_with_builder,
            )
        except Exception:
            pass


try:  # pragma: no cover - allow package/flat imports
    from .patch_suggestion_db import PatchSuggestionDB
except Exception:  # pragma: no cover - fallback for flat layout
    from patch_suggestion_db import PatchSuggestionDB  # type: ignore

try:  # pragma: no cover - allow package/flat imports
    from .patch_provenance import record_patch_metadata, get_patch_by_commit
except Exception:  # pragma: no cover - fallback for flat layout
    from patch_provenance import (
        record_patch_metadata,  # type: ignore
        get_patch_by_commit,  # type: ignore
    )

try:  # pragma: no cover - optional dependency
    from .unified_event_bus import UnifiedEventBus
except Exception:  # pragma: no cover - fallback for flat layout
    from unified_event_bus import UnifiedEventBus  # type: ignore
try:  # pragma: no cover - allow package/flat imports
    from .shared_event_bus import event_bus as _SHARED_EVENT_BUS
except Exception:  # pragma: no cover - flat layout fallback
    from shared_event_bus import event_bus as _SHARED_EVENT_BUS  # type: ignore

try:  # pragma: no cover - allow package/flat imports
    from .code_database import PatchRecord
except Exception:  # pragma: no cover - fallback for flat layout
    from code_database import PatchRecord  # type: ignore

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from .bot_registry import BotRegistry
    from .enhancement_classifier import EnhancementClassifier
    from .evolution_orchestrator import EvolutionOrchestrator
    from .self_improvement.baseline_tracker import BaselineTracker as _BaselineTracker
    from .self_improvement.target_region import TargetRegion as _TargetRegion
else:  # pragma: no cover - runtime stubs avoid circular imports
    BotRegistry = Any  # type: ignore[misc, assignment]
    _BaselineTracker = Any  # type: ignore[misc, assignment]
    _TargetRegion = Any  # type: ignore[misc, assignment]


def _get_bot_workflow_tests(*args: Any, **kwargs: Any) -> list[str]:
    """Lazily import :func:`get_bot_workflow_tests` to avoid circular imports."""

    try:
        from .bot_registry import get_bot_workflow_tests as _inner
    except Exception as exc:  # pragma: no cover - best effort fallback
        raise RuntimeError("bot workflow lookup is unavailable") from exc
    result = _inner(*args, **kwargs)
    return list(result or [])


_BASELINE_TRACKER_CLS: type[_BaselineTracker] | None = None
_TARGET_REGION_CLS: type[_TargetRegion] | None = None


def _get_baseline_tracker_cls() -> type[_BaselineTracker]:
    """Import ``BaselineTracker`` lazily to avoid circular imports."""

    global _BASELINE_TRACKER_CLS
    if _BASELINE_TRACKER_CLS is None:
        from .self_improvement.baseline_tracker import BaselineTracker as _LoadedBaselineTracker

        _BASELINE_TRACKER_CLS = _LoadedBaselineTracker
    return _BASELINE_TRACKER_CLS


def _get_target_region_cls() -> type[_TargetRegion]:
    """Import ``TargetRegion`` lazily to avoid circular imports."""

    global _TARGET_REGION_CLS
    if _TARGET_REGION_CLS is None:
        from .self_improvement.target_region import TargetRegion as _LoadedTargetRegion

        _TARGET_REGION_CLS = _LoadedTargetRegion
    return _TARGET_REGION_CLS


def _get_error_db_cls():
    """Import ``ErrorDB`` lazily so module import does not require ``error_bot``."""

    from .error_bot import ErrorDB as _ErrorDB

    return _ErrorDB


class ObjectiveAdjacentPathClassifier:
    """Classify patch targets using the shared objective-surface manifest."""

    def __init__(self, *, repo_root: Path | None = None) -> None:
        self.repo_root = (repo_root or Path.cwd()).resolve()

    def classify(self, path: Path) -> str:
        if is_self_coding_unsafe_path(path, repo_root=self.repo_root):
            return "objective_adjacent"
        return "standard"


class PatchApprovalPolicy:
    """Run formal verification and tests before patching.

    The test runner command can be customised via ``test_command``.
    When omitted the command is loaded from the ``threshold_service`` or
    per-bot configuration.
    """

    REASON_MANUAL_APPROVAL_MISSING = "manual_approval_missing"
    REASON_VERIFICATION_FAILED = "verification_failed"
    REASON_TESTS_FAILED = "tests_failed"

    def __init__(
        self,
        *,
        verifier: FormalVerifier | None = None,
        rollback_mgr: AutomatedRollbackManager | None = None,
        bot_name: str = "menace",
        test_command: list[str] | None = None,
        threshold_service: ThresholdService | None = None,
        path_classifier: ObjectiveAdjacentPathClassifier | None = None,
    ) -> None:
        self.verifier = verifier or FormalVerifier()
        self.rollback_mgr = rollback_mgr
        self.bot_name = bot_name
        self.logger = logging.getLogger(self.__class__.__name__)
        svc = threshold_service or _DEFAULT_THRESHOLD_SERVICE
        if test_command is None:
            settings = SandboxSettings()
            try:
                test_command = svc.load(bot_name, settings).test_command
            except Exception:  # pragma: no cover - service issues
                test_command = None
            if not test_command:
                try:
                    bt = settings.bot_thresholds.get(bot_name)
                    if bt and bt.test_command:
                        test_command = list(bt.test_command)
                except Exception:  # pragma: no cover - settings issues
                    test_command = None
        self.test_command = list(test_command) if test_command else ["pytest", "-q"]
        self._path_classifier = path_classifier or ObjectiveAdjacentPathClassifier(
            repo_root=Path.cwd().resolve()
        )
        self.last_decision: dict[str, Any] = {
            "approved": False,
            "reason_codes": (),
            "target_classification": "standard",
            "approval_source": None,
            "approval_actor": None,
            "approval_timestamp": None,
            "approval_rationale": None,
        }

    # ------------------------------------------------------------------
    def update_test_command(self, new_cmd: list[str]) -> None:
        """Refresh the command used for running tests."""
        self.test_command = list(new_cmd)

    def classify_target(self, path: Path) -> str:
        """Classify *path* for approval handling."""
        return self._path_classifier.classify(path)

    def _record_decision(
        self,
        *,
        approved: bool,
        path: Path,
        classification: str,
        approval_source: str | None,
        reason_codes: list[str],
    ) -> None:
        rationale = "checks_passed" if approved else ",".join(reason_codes) or "approval_denied"
        actor = os.getenv("MENACE_APPROVAL_ACTOR", "automation")
        if approval_source is not None:
            actor = f"{actor}:{approval_source}"
        timestamp = datetime.now(timezone.utc).isoformat()
        self.last_decision = {
            "approved": approved,
            "reason_codes": tuple(reason_codes),
            "target_classification": classification,
            "approval_source": approval_source,
            "approval_actor": actor,
            "approval_timestamp": timestamp,
            "approval_rationale": rationale,
        }
        level = logging.INFO if approved else logging.WARNING
        self.logger.log(
            level,
            "patch approval decision for %s",
            path_for_prompt(path),
            extra={
                "event": "self_coding_approval_decision",
                "actor": actor,
                "timestamp": timestamp,
                "file": path_for_prompt(path),
                "rationale": rationale,
            },
        )

    @staticmethod
    def _path_matches_rule(path: Path, rule: str) -> bool:
        cleaned_rule = str(rule).replace("\\", "/").strip().strip("/")
        if not cleaned_rule:
            return False
        target = str(path).replace("\\", "/").strip().strip("/")
        return target == cleaned_rule or target.startswith(f"{cleaned_rule}/")

    @classmethod
    def resolve_manual_approval_source(
        cls,
        path: Path,
        *,
        manual_approval_token: str | None = None,
    ) -> str | None:
        token = (manual_approval_token or "").strip()
        if token:
            return "token"
        token_file = os.getenv("MENACE_MANUAL_APPROVAL_FILE", "").strip()
        if token_file:
            candidate = Path(token_file).expanduser()
            try:
                raw_payload = json.loads(candidate.read_text(encoding="utf-8"))
            except Exception:
                raw_payload = None
            if isinstance(raw_payload, dict):
                approved_paths = raw_payload.get("approved_paths", [])
                if isinstance(approved_paths, list):
                    rel_path = path_for_prompt(path)
                    if any(cls._path_matches_rule(Path(rel_path), str(item)) for item in approved_paths):
                        return "record"
            elif isinstance(raw_payload, list):
                rel_path = path_for_prompt(path)
                if any(cls._path_matches_rule(Path(rel_path), str(item)) for item in raw_payload):
                    return "record"
        queue_file = os.getenv("MENACE_MANUAL_APPROVAL_QUEUE", "").strip()
        if queue_file:
            candidate = Path(queue_file).expanduser()
            if candidate.exists():
                rel_path = path_for_prompt(path)
                try:
                    for line in candidate.read_text(encoding="utf-8").splitlines():
                        line = line.strip()
                        if not line:
                            continue
                        entry = json.loads(line)
                        if not isinstance(entry, dict):
                            continue
                        if not bool(entry.get("approved", False)):
                            continue
                        if cls._path_matches_rule(Path(rel_path), str(entry.get("path", ""))):
                            return "queue"
                except Exception:
                    return None
        return None

    def approve(self, path: Path, *, manual_approval_token: str | None = None) -> bool:
        ok = True
        classification = self.classify_target(path)
        approval_source: str | None = None
        reason_codes: list[str] = []
        if classification == "objective_adjacent":
            approval_source = self.resolve_manual_approval_source(
                path,
                manual_approval_token=manual_approval_token,
            )
            if approval_source is None:
                reason_codes.append(self.REASON_MANUAL_APPROVAL_MISSING)
                self._record_decision(
                    approved=False,
                    path=path,
                    classification=classification,
                    approval_source=None,
                    reason_codes=reason_codes,
                )
                self.logger.warning(
                    "manual approval required for objective-adjacent target: %s",
                    path_for_prompt(path),
                    extra={
                        "event": "self_coding_approval_denied",
                        "target_classification": classification,
                        "approval_required": True,
                        "approval_source": None,
                        "reason_codes": tuple(reason_codes),
                    },
                )
                return False
        try:
            if self.verifier and not self.verifier.verify(path):
                ok = False
                reason_codes.append(self.REASON_VERIFICATION_FAILED)
        except Exception as exc:  # pragma: no cover - verification issues
            self.logger.error("verification failed: %s", exc)
            ok = False
            reason_codes.append(self.REASON_VERIFICATION_FAILED)
        try:
            subprocess.run(self.test_command, check=True)
        except Exception as exc:  # pragma: no cover - test runner issues
            self.logger.error("self tests failed: %s", exc)
            ok = False
            reason_codes.append(self.REASON_TESTS_FAILED)
        if ok and self.rollback_mgr:
            try:
                self.rollback_mgr.log_healing_action(
                    self.bot_name, "patch_checks", path_for_prompt(path)
                )
            except Exception as exc:  # pragma: no cover - audit logging issues
                self.logger.exception("failed to log healing action: %s", exc)
        if ok:
            self.logger.info(
                "patch approval succeeded for %s",
                path_for_prompt(path),
                extra={
                    "event": "self_coding_approval_approved",
                    "target_classification": classification,
                    "approval_required": classification == "objective_adjacent",
                    "approval_source": approval_source,
                },
            )
        self._record_decision(
            approved=ok,
            path=path,
            classification=classification,
            approval_source=approval_source,
            reason_codes=reason_codes,
        )
        return ok


class ObjectiveApprovalPolicy:
    """Enforce explicit human approval for objective-adjacent targets."""

    REASON_MANUAL_APPROVAL_MISSING = PatchApprovalPolicy.REASON_MANUAL_APPROVAL_MISSING

    def __init__(self, *, repo_root: Path | None = None) -> None:
        self.repo_root = (repo_root or Path.cwd()).resolve()
        self.logger = logging.getLogger(self.__class__.__name__)
        self._path_classifier = ObjectiveAdjacentPathClassifier(repo_root=self.repo_root)
        self.last_decision: dict[str, Any] = {
            "approved": False,
            "reason_codes": (),
            "target_classification": "standard",
            "approval_source": None,
        }

    def classify_target(self, path: Path) -> str:
        return self._path_classifier.classify(path)

    def approve(self, path: Path, *, manual_approval_token: str | None = None) -> bool:
        classification = self.classify_target(path)
        approval_source: str | None = None
        reason_codes: list[str] = []
        if classification == "objective_adjacent":
            approval_source = PatchApprovalPolicy.resolve_manual_approval_source(
                path,
                manual_approval_token=manual_approval_token,
            )
            if approval_source is None:
                reason_codes.append(self.REASON_MANUAL_APPROVAL_MISSING)
                self.last_decision = {
                    "approved": False,
                    "reason_codes": tuple(reason_codes),
                    "target_classification": classification,
                    "approval_source": None,
                }
                self.logger.warning(
                    "manual approval required for objective-adjacent target: %s",
                    path_for_prompt(path),
                )
                return False
        self.last_decision = {
            "approved": True,
            "reason_codes": tuple(reason_codes),
            "target_classification": classification,
            "approval_source": approval_source,
        }
        return True


class HelperGenerationError(RuntimeError):
    """Raised when helper generation fails before patching."""


class SelfCodingManager:
    """Apply code patches and redeploy bots.

    ``data_bot`` and ``bot_registry`` must be provided; a
    :class:`ValueError` is raised otherwise. A functioning
    :class:`EvolutionOrchestrator` is mandatory and failure to
    construct one results in a :class:`RuntimeError`.
    """

    def __init__(
        self,
        self_coding_engine: SelfCodingEngine,
        pipeline: ModelAutomationPipeline,
        *,
        bot_name: str = "menace",
        data_bot: DataBot | None = None,
        approval_policy: "PatchApprovalPolicy | None" = None,
        suggestion_db: PatchSuggestionDB | None = None,
        enhancement_classifier: "EnhancementClassifier" | None = None,
        failure_store: FailureFingerprintStore | None = None,
        skip_similarity: float | None = None,
        baseline_window: int | None = None,
        bot_registry: BotRegistry | None = None,
        quick_fix: QuickFixEngine | None = None,
        error_db: "ErrorDB | None" = None,
        event_bus: UnifiedEventBus | None = None,
        evolution_orchestrator: "EvolutionOrchestrator | None" = None,
        threshold_service: ThresholdService | None = None,
        roi_drop_threshold: float | None = None,
        error_rate_threshold: float | None = None,
        bootstrap_mode: bool | None = None,
        bootstrap_register_timeout: float | None = None,
        bootstrap_fast: bool | None = None,
        defer_orchestrator_init: bool = False,
        construction_phase_callback: Callable[[str], None] | None = None,
    ) -> None:
        if data_bot is None or bot_registry is None:
            raise ValueError("data_bot and bot_registry are required")
        self.engine = self_coding_engine
        self.pipeline = pipeline
        self.bot_name = bot_name
        self._construction_phase_callback = construction_phase_callback
        self.data_bot = data_bot
        self.threshold_service = threshold_service or _DEFAULT_THRESHOLD_SERVICE
        self.approval_policy = approval_policy
        self.logger = logging.getLogger(self.__class__.__name__)
        self.objective_guard = ObjectiveGuard(repo_root=Path.cwd().resolve())
        self._mark_construction_phase("init:start")
        registry_bootstrap = bool(getattr(bot_registry, "bootstrap", False))
        resolved_bootstrap = registry_bootstrap if bootstrap_mode is None else bootstrap_mode
        self.bootstrap = bool(resolved_bootstrap)
        self.bootstrap_mode = self.bootstrap
        self.bootstrap_register_timeout = bootstrap_register_timeout
        if bootstrap_fast is None:
            self.bootstrap_fast = bool(
                bootstrap_mode is None or self.bootstrap_mode
            )
        else:
            self.bootstrap_fast = bool(bootstrap_fast)
        self._settings: SandboxSettings | None = None
        settings = self._get_settings()
        if self.bootstrap_fast:
            self.logger.info(
                "bootstrap_fast requested for %s (bootstrap_mode=%s)",
                bot_name,
                self.bootstrap_mode,
            )
            try:
                settings_mod = importlib.import_module(
                    settings.__class__.__module__ if settings else "sandbox_settings"
                )
                validation_check = False
                validation_fn = getattr(
                    settings_mod, "_bootstrap_validation_enabled", None
                )
                if callable(validation_fn):
                    validation_check = bool(validation_fn())
                if validation_check:
                    self.logger.info(
                        "bootstrap_fast validation active; parent directory creation is skipped",
                    )
                else:
                    self.logger.warning(
                        "bootstrap_fast requested but sandbox validation flag is missing or disabled; falling back to full validation",
                    )
                    self.bootstrap_fast = False
            except Exception:  # pragma: no cover - defensive verification
                self.logger.exception(
                    "unable to verify bootstrap_fast validation state during init; falling back to full validation",
                )
                self.bootstrap_fast = False
            if not self.bootstrap_fast:
                self._settings = None
                settings = self._get_settings()
        self._last_patch_id: int | None = None
        self._last_event_id: int | None = None
        self._last_commit_hash: str | None = None
        self._last_validation_summary: Dict[str, Any] | None = None
        self._objective_breach_lock = threading.Lock()
        self._objective_breach_handled = False
        self._active_objective_checkpoint: Dict[str, Any] | None = None
        self._objective_lock_manifest_sha_at_breach: str | None = None
        self._objective_lock_requires_manifest_refresh = False
        self._self_coding_paused = False
        self._self_coding_disabled_reason: str | None = None
        self._self_coding_pause_source: str | None = None
        divergence_config = load_divergence_detector_config()
        self._divergence_window = divergence_config.window_size
        self._divergence_detector = SelfCodingDivergenceDetector(divergence_config)
        self._divergence_threshold_cycles = divergence_config.divergence_threshold_cycles
        self._divergence_recovery_cycles = divergence_config.recovery_threshold_cycles
        self._divergence_fail_closed_on_missing_metrics = (
            divergence_config.fail_closed_on_missing_metrics
        )
        self._missing_metric_pause_cycles = divergence_config.missing_metric_pause_cycles
        self._divergence_streak = 0
        self._divergence_recovery_streak = 0
        self._missing_real_metric_streak = 0
        self._cycle_metrics_window: deque[CycleMetricsRecord] = deque(
            maxlen=self._divergence_window
        )
        thresholds = self.threshold_service.get(
            bot_name, bootstrap_mode=self.bootstrap_mode
        )
        print(
            f"[debug] SelfCodingManager init for: {bot_name}, thresholds={thresholds}, orchestrator={evolution_orchestrator}"
        )
        self.roi_drop_threshold = (
            roi_drop_threshold
            if roi_drop_threshold is not None
            else thresholds.roi_drop
        )
        self.error_rate_threshold = (
            error_rate_threshold
            if error_rate_threshold is not None
            else thresholds.error_threshold
        )
        self.test_failure_threshold = thresholds.test_failure_threshold
        self._refresh_thresholds()
        self._failure_cache = FailureCache()
        self.suggestion_db = suggestion_db or getattr(
            self.engine, "patch_suggestion_db", None
        )
        self.enhancement_classifier = enhancement_classifier or getattr(
            self.engine, "enhancement_classifier", None
        )
        self.failure_store = failure_store
        self.skip_similarity = skip_similarity
        self.quick_fix = quick_fix
        self.error_db = error_db
        if baseline_window is None:
            try:
                settings = self._get_settings()
                baseline_window = getattr(settings, "baseline_window", 5)
            except Exception:
                baseline_window = 5
        baseline_tracker_cls = _get_baseline_tracker_cls()
        self.baseline_tracker = baseline_tracker_cls(
            window=int(baseline_window), metrics=["confidence"]
        )
        try:
            settings = self._get_settings()
            configured_retries = getattr(
                settings, "self_test_repair_retries", None
            )
            if configured_retries is None:
                configured_retries = getattr(
                    settings, "post_patch_repair_attempts", None
            )
        except Exception:
            configured_retries = None
        env_retries = os.getenv("SELF_TEST_REPAIR_RETRIES")
        retry_candidate: int | None = None
        for candidate in (env_retries, configured_retries):
            if candidate is None:
                continue
            try:
                retry_candidate = int(candidate)
                break
            except (TypeError, ValueError):
                continue
        self.post_patch_repair_retries = max(int(retry_candidate or 0), 0)
        # ``_forecast_history`` stores predicted metrics so threshold updates
        # can adapt based on recent trends.
        self._forecast_history: Dict[str, list[float]] = {
            "roi": [],
            "errors": [],
            "tests_failed": [],
        }
        if enhancement_classifier and not getattr(
            self.engine, "enhancement_classifier", None
        ):
            try:
                self.engine.enhancement_classifier = enhancement_classifier
            except Exception as exc:
                self.logger.warning(
                    "Failed to attach enhancement classifier to engine; "
                    "enhancement classification disabled: %s",
                    exc,
                )
                self.enhancement_classifier = None
        self.bot_registry = bot_registry
        # Ensure all managers use the shared event bus unless a specific one
        # is supplied.
        self.event_bus = event_bus or _SHARED_EVENT_BUS
        if self.event_bus:
            try:  # pragma: no cover - best effort
                self.event_bus.subscribe("thresholds:updated", self._on_thresholds_updated)
            except Exception:
                self.logger.exception("threshold update subscription failed")
        self.evolution_orchestrator = evolution_orchestrator
        if self.bot_registry:
            try:
                register_kwargs: dict[str, Any] = {}
                register_timeout = self.bootstrap_register_timeout
                if register_timeout is None and self.bootstrap_mode:
                    register_timeout = 0.0
                if register_timeout is not None:
                    register_kwargs["lock_timeout"] = register_timeout
                if self.bootstrap_mode:
                    register_kwargs["bootstrap_mode"] = True
                self.bot_registry.register_bot(
                    self.bot_name,
                    manager=self,
                    data_bot=self.data_bot,
                    is_coding_bot=True,
                    **register_kwargs,
                )
            except Exception:  # pragma: no cover - best effort
                self.logger.exception("failed to register bot in registry")
        self._hydrate_self_coding_disabled_state()

        clayer = getattr(self.engine, "cognition_layer", None)
        builder = getattr(clayer, "context_builder", None)
        if builder is None:
            builder = getattr(self.engine, "context_builder", None)
        if builder is None:
            raise RuntimeError(
                "engine.cognition_layer must provide a context_builder"
            )
        self._context_builder = builder
        self._mark_construction_phase("context_builder:prepare")
        with _timed_marker(self.logger, self.bot_name, "context_builder:prepare"):
            self._prepare_context_builder(builder)
        self._mark_construction_phase("quick_fix:init")
        with _timed_marker(self.logger, self.bot_name, "quick_fix:init"):
            self._init_quick_fix_engine(builder)

        if defer_orchestrator_init:
            self._mark_construction_phase("orchestrator:init:deferred")
        else:
            self.initialize_deferred_components()
        self._mark_construction_phase("init:complete")

    def _mark_construction_phase(self, phase: str) -> None:
        """Record manager construction phase transitions for diagnostics."""

        callback = getattr(self, "_construction_phase_callback", None)
        if callable(callback):
            try:
                callback(phase)
            except Exception:  # pragma: no cover - diagnostics are best effort
                self.logger.debug("construction phase callback failed", exc_info=True)
        self.logger.debug("%s: manager construction phase=%s", self.bot_name, phase)

    def initialize_deferred_components(self, *, skip_non_critical: bool = False) -> None:
        """Initialize expensive orchestration components when deferred."""

        stage = (
            "orchestrator:init:begin:reduced"
            if skip_non_critical
            else "orchestrator:init:begin"
        )
        self._mark_construction_phase(stage)
        if self.evolution_orchestrator is None:
            try:  # pragma: no cover - optional dependencies
                from .capital_management_bot import CapitalManagementBot
                from .self_improvement.engine import SelfImprovementEngine
                from .system_evolution_manager import SystemEvolutionManager
                from .evolution_orchestrator import EvolutionOrchestrator

                with _timed_marker(self.logger, self.bot_name, "orchestrator:init:capital"):
                    capital = CapitalManagementBot(data_bot=self.data_bot)
                pipeline_promoter = getattr(self.pipeline, "_pipeline_promoter", None)
                bootstrap_owner = getattr(self, "_bootstrap_owner_token", None)
                if bootstrap_owner is None:
                    bootstrap_owner = getattr(self, "_bootstrap_provenance_token", None)
                if bootstrap_owner is None:
                    try:
                        from .coding_bot_interface import get_structural_bootstrap_owner
                    except Exception:  # pragma: no cover - optional bootstrap context
                        bootstrap_owner = None
                    else:
                        bootstrap_owner = get_structural_bootstrap_owner()
                # Startup order: ensure broker owner is active -> SelfImprovementEngine
                # (which constructs ResearchAggregatorBot) -> EvolutionOrchestrator.
                self._mark_construction_phase("orchestrator:init:broker_owner_ready")
                with _timed_marker(
                    self.logger,
                    self.bot_name,
                    "orchestrator:init:broker_owner_ready",
                ):
                    self._ensure_broker_owner_ready(bootstrap_owner=bootstrap_owner)
                if skip_non_critical:
                    self.logger.info(
                        "%s: deferred orchestrator initialization reduced scope active",
                        self.bot_name,
                    )
                    self._mark_construction_phase("orchestrator:init:reduced_complete")
                    return
                self._mark_construction_phase("orchestrator:init:self_improvement")
                with _timed_marker(
                    self.logger,
                    self.bot_name,
                    "orchestrator:init:self_improvement",
                ):
                    improv = SelfImprovementEngine(
                        context_builder=self._context_builder,
                        data_bot=self.data_bot,
                        bot_name=self.bot_name,
                        manager=self,
                        pipeline=self.pipeline,
                        pipeline_promoter=pipeline_promoter,
                        bootstrap_owner=bootstrap_owner,
                    )
                with _timed_marker(
                    self.logger,
                    self.bot_name,
                    "orchestrator:init:system_evolution_manager",
                ):
                    bots = list(getattr(self.bot_registry, "graph", {}).keys())
                    evol_mgr = SystemEvolutionManager(bots)
                self._mark_construction_phase("orchestrator:init:evolution_orchestrator")
                with _timed_marker(
                    self.logger,
                    self.bot_name,
                    "orchestrator:init:evolution_orchestrator",
                ):
                    self.evolution_orchestrator = EvolutionOrchestrator(
                        data_bot=self.data_bot,
                        capital_bot=capital,
                        improvement_engine=improv,
                        evolution_manager=evol_mgr,
                        selfcoding_manager=self,
                        event_bus=self.event_bus,
                    )
                self.logger.debug(
                    "%s: EvolutionOrchestrator initialized successfully", self.bot_name
                )
            except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
                self.logger.warning(
                    "%s: EvolutionOrchestrator dependencies unavailable: %s",
                    self.bot_name,
                    exc,
                )
                self.evolution_orchestrator = None
            except Exception as exc:  # pragma: no cover - optional dependency
                self.logger.warning(
                    "%s: EvolutionOrchestrator could not be instantiated; disabling orchestration",
                    self.bot_name,
                    exc_info=exc,
                )
                self.evolution_orchestrator = None

        if not self.evolution_orchestrator:
            self.logger.debug(
                "%s: EvolutionOrchestrator not available; continuing without orchestration",
                self.bot_name,
            )
        else:
            try:  # pragma: no cover - best effort
                self.evolution_orchestrator.register_bot(self.bot_name)
            except Exception:
                self.logger.exception(
                    "failed to register bot with evolution orchestrator",
                )
        self._mark_construction_phase("orchestrator:init:complete")

    def register_bot(
        self,
        name: str,
        module_path: str | os.PathLike[str] | None = None,
        *,
        roi_threshold: float | None = None,
        error_threshold: float | None = None,
        test_failure_threshold: float | None = None,
        patch_id: int | str | None = None,
        commit: str | None = None,
        provenance: Any | None = None,
        **kwargs: Any,
    ) -> None:
        """Register *name* with the underlying :class:`BotRegistry`."""
        if not self.bot_registry:
            return
        try:
            register_kwargs: dict[str, Any] = {
                "manager": self,
                "data_bot": self.data_bot,
                "is_coding_bot": True,
            }
            if roi_threshold is not None:
                register_kwargs["roi_threshold"] = roi_threshold
            if error_threshold is not None:
                register_kwargs["error_threshold"] = error_threshold
            if test_failure_threshold is not None:
                register_kwargs["test_failure_threshold"] = test_failure_threshold
            if patch_id is not None:
                register_kwargs["patch_id"] = patch_id
            if commit is not None:
                register_kwargs["commit"] = commit
            if provenance is not None:
                register_kwargs["provenance"] = provenance
            register_kwargs.update(kwargs)
            self.bot_registry.register_bot(name, module_path, **register_kwargs)
            if self.data_bot:
                try:
                    self.threshold_service.reload(name)
                    self.data_bot.check_degradation(
                        name, roi=0.0, errors=0.0, test_failures=0.0
                    )
                    self.logger.info("seeded thresholds for %s", name)
                except Exception as exc:  # pragma: no cover - best effort
                    self.logger.exception(
                        "failed to seed thresholds for %s: %s", name, exc
                    )
                    bus = self.event_bus or getattr(self.data_bot, "event_bus", None)
                    if bus:
                        try:  # pragma: no cover - best effort
                            bus.publish(
                                "data:threshold_update_failed",
                                {"bot": name, "error": str(exc)},
                            )
                        except Exception:
                            self.logger.exception(
                                "failed to publish threshold update failed event"
                            )
            if self.evolution_orchestrator:
                try:  # pragma: no cover - best effort
                    self.evolution_orchestrator.register_bot(name)
                except Exception:
                    self.logger.exception(
                        "failed to register bot with evolution orchestrator"
                    )
            if self.data_bot and self.evolution_orchestrator:
                bus = getattr(self.data_bot, "event_bus", None)
                if bus:
                    try:
                        bus.subscribe(
                            "degradation:detected",
                            lambda _t, e: self.evolution_orchestrator.register_patch_cycle(e),
                        )
                    except Exception:  # pragma: no cover - best effort
                        self.logger.exception(
                            "failed to subscribe to degradation events"
                        )
        except Exception:  # pragma: no cover - best effort
            self.logger.exception("failed to register bot in registry")

    def _get_settings(self) -> SandboxSettings | None:
        """Return cached ``SandboxSettings`` initialised for the bootstrap mode."""

        if self._settings is not None:
            return self._settings

        try:
            self._settings = SandboxSettings(
                bootstrap_fast=self.bootstrap_fast,
                build_groups=False,
            )
        except Exception:  # pragma: no cover - best effort cache
            self._settings = None
        return self._settings

    def _refresh_thresholds(
        self, *, bootstrap_fast: bool | None = None, bootstrap_mode: bool | None = None
    ) -> None:
        """Fetch ROI, error and test-failure thresholds via ``ThresholdService``.

        When adaptive thresholding is enabled via :class:`SandboxSettings`,
        rolling metrics from :class:`BaselineTracker` are analysed for long-term
        drift.  Sustained shifts tighten or relax the ROI drop and error rate
        limits which are then persisted through the shared service.
        """

        if not self.data_bot:
            return
        try:
            getattr(self, "_last_thresholds", None)
            bootstrap_active = (
                bool(bootstrap_mode)
                if bootstrap_mode is not None
                else bool(getattr(self, "bootstrap_mode", False))
            )
            t = self.threshold_service.reload(
                self.bot_name,
                bootstrap_mode=(
                    True
                    if bootstrap_active
                    else (
                        bootstrap_fast
                        if bootstrap_fast is not None
                        else self.bootstrap_mode
                    )
                ),
            )

            adaptive = False
            try:
                adaptive = getattr(self._get_settings(), "adaptive_thresholds", False)
            except Exception:
                adaptive = False
            if adaptive and hasattr(self, "baseline_tracker"):
                try:
                    roi_deltas = self.baseline_tracker.delta_history("roi")
                    err_deltas = self.baseline_tracker.delta_history("errors")
                    new_roi = t.roi_drop
                    new_err = t.error_threshold
                    updated = False
                    if roi_deltas and len(roi_deltas) >= self.baseline_tracker.window:
                        roi_drift = sum(roi_deltas) / len(roi_deltas)
                        if abs(roi_drift) > 0.01:
                            new_roi = max(min(t.roi_drop + roi_drift, 0.0), -1.0)
                            updated = updated or new_roi != t.roi_drop
                    if err_deltas and len(err_deltas) >= self.baseline_tracker.window:
                        err_drift = sum(err_deltas) / len(err_deltas)
                        if abs(err_drift) > 0.01:
                            new_err = max(t.error_threshold + err_drift, 0.0)
                            updated = updated or new_err != t.error_threshold
                    success_deltas = self.baseline_tracker.delta_history("patch_success")
                    if success_deltas and len(success_deltas) >= self.baseline_tracker.window:
                        success_drift = sum(success_deltas) / len(success_deltas)
                        if abs(success_drift) > 0.01:
                            new_roi = max(min(new_roi + success_drift, 0.0), -1.0)
                            new_err = max(new_err - success_drift, 0.0)
                            updated = True
                    if updated:
                        self.threshold_service.update(
                            self.bot_name,
                            roi_drop=new_roi if new_roi != t.roi_drop else None,
                            error_threshold=(
                                new_err if new_err != t.error_threshold else None
                            ),
                            bootstrap_mode=self.bootstrap_mode,
                        )
                        try:  # pragma: no cover - best effort persistence
                            persist_sc_thresholds(
                                self.bot_name,
                                roi_drop=new_roi if new_roi != t.roi_drop else None,
                                error_increase=(
                                    new_err if new_err != t.error_threshold else None
                                ),
                                event_bus=self.event_bus,
                            )
                        except Exception:
                            self.logger.exception(
                                "failed to persist thresholds for %s",
                                self.bot_name,
                            )
                        t = self.threshold_service.reload(self.bot_name)
                except Exception:  # pragma: no cover - adaptive failures
                    self.logger.exception("adaptive threshold update failed")

            self.roi_drop_threshold = t.roi_drop
            self.error_rate_threshold = t.error_threshold
            self.test_failure_threshold = t.test_failure_threshold
            self._last_thresholds = t
        except Exception:  # pragma: no cover - best effort
            self.logger.exception("failed to load thresholds for %s", self.bot_name)

    def _on_thresholds_updated(self, _topic: str, event: object) -> None:
        """Refresh cached thresholds when configuration changes."""
        if not isinstance(event, dict):
            return
        bot = event.get("bot")
        if bot and bot != self.bot_name:
            return
        try:
            self._refresh_thresholds()
        except Exception:
            self.logger.exception("failed to refresh thresholds after update")

    def _prepare_context_builder(self, builder: ContextBuilder) -> None:
        """Refresh *builder* weights and log its session id."""

        ensure_fresh_weights(builder)
        patch_db = getattr(self.engine, "patch_db", None)
        session_id = getattr(builder, "session_id", "") or uuid.uuid4().hex
        setattr(builder, "session_id", session_id)
        if patch_db:
            try:
                conn = patch_db.router.get_connection("patch_history")
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS patch_contributors(
                        patch_id INTEGER,
                        vector_id TEXT,
                        influence REAL,
                        session_id TEXT,
                        FOREIGN KEY(patch_id) REFERENCES patch_history(id)
                    )
                    """
                )
                conn.execute(
                    (
                        "INSERT INTO patch_contributors("
                        "patch_id, vector_id, influence, session_id"
                        ") VALUES(?,?,?,?)"
                    ),
                    (None, "", 0.0, session_id),
                )
                conn.commit()
            except Exception:
                self.logger.exception("failed to record context builder session")

    def _init_quick_fix_engine(self, builder: ContextBuilder) -> None:
        """Instantiate :class:`QuickFixEngine` if missing."""

        if self.quick_fix is not None:
            return
        if QuickFixEngine is None:
            msg = (
                "QuickFixEngine is required but could not be imported; "
                "pip install menace[quickfix]"
            )
            self.logger.error(msg)
            raise ImportError(msg)
        error_db_cls = _get_error_db_cls()
        db = self.error_db or error_db_cls()
        self.error_db = db
        try:
            self.quick_fix = QuickFixEngine(
                db,
                self,
                context_builder=builder,
                helper_fn=_manager_generate_helper_with_builder,
            )
        except Exception as exc:  # pragma: no cover - instantiation errors
            raise RuntimeError(
                "failed to initialise QuickFixEngine",
            ) from exc

    def _ensure_broker_owner_ready(self, *, bootstrap_owner: object | None) -> bool:
        """Ensure the bootstrap dependency broker owner is active before bootstrapping."""
        module_name = f"{__package__}.bootstrap_placeholder" if __package__ else "bootstrap_placeholder"
        spec = importlib.util.find_spec(module_name)
        if spec is None and module_name != "bootstrap_placeholder":
            module_name = "bootstrap_placeholder"
            spec = importlib.util.find_spec(module_name)
        if spec is None:  # pragma: no cover - dependency unavailable
            self.logger.warning(
                "Bootstrap dependency broker placeholder utilities unavailable; "
                "broker owner may remain inactive",
            )
            return False
        module = importlib.import_module(module_name)
        advertise_broker_placeholder = getattr(module, "advertise_broker_placeholder")
        bootstrap_broker = getattr(module, "bootstrap_broker")

        broker = bootstrap_broker()
        if not getattr(broker, "active_owner", False):
            advertise_broker_placeholder(dependency_broker=broker)

        timeout = max(_BROKER_OWNER_READY_TIMEOUT_SECS, 0.0)
        deadline = time.monotonic() + timeout
        while not getattr(broker, "active_owner", False) and time.monotonic() < deadline:
            time.sleep(0.05)

        owner_ready = bool(getattr(broker, "active_owner", False))
        if not owner_ready:
            self.logger.warning(
                "Bootstrap dependency broker owner not active after %.2fs; "
                "continuing with degraded bootstrap (bootstrap_owner=%s).",
                timeout,
                bootstrap_owner,
                extra={"event": "broker-owner-not-ready", "timeout": timeout},
            )
        else:
            self.logger.debug(
                "Bootstrap dependency broker owner active (bootstrap_owner=%s).",
                bootstrap_owner,
                extra={"event": "broker-owner-ready"},
            )
        return owner_ready

    def _ensure_quick_fix_engine(self, builder: ContextBuilder) -> QuickFixEngine:
        """Return an initialised :class:`QuickFixEngine`.

        *builder* must be supplied by the caller and is attached to the
        underlying :class:`QuickFixEngine` instance. When no engine is present a
        new instance is created so patches always undergo validation.
        """

        if builder is None:  # pragma: no cover - defensive
            raise ValueError("ContextBuilder is required")

        try:
            self._init_quick_fix_engine(builder)
        except Exception as exc:
            raise QuickFixEngineError(
                "quick_fix_init_error", "failed to initialise QuickFixEngine"
            ) from exc

        try:
            self._prepare_context_builder(builder)
            self.quick_fix.context_builder = builder
        except Exception as exc:
            self.logger.exception(
                "failed to update QuickFixEngine context builder",
            )
            raise QuickFixEngineError(
                "quick_fix_validation_error",
                "QuickFixEngine context validation failed",
            ) from exc
        return self.quick_fix

    def refresh_quick_fix_context(self) -> ContextBuilder:
        """Attach a fresh ``ContextBuilder`` to :class:`QuickFixEngine`."""

        builder = create_context_builder()
        ensure_fresh_weights(builder)
        if self.quick_fix is None:
            self._init_quick_fix_engine(builder)
        else:
            try:
                self.quick_fix.context_builder = builder
            except Exception:
                self.logger.exception(
                    "failed to update QuickFixEngine context builder",
                )
                raise
        return builder

    def _resolve_repo_root(self, module: Path) -> Path:
        """Return the git repository root for the current module.

        ``Path.cwd()`` can point outside the repository when the manager is invoked
        from a different working directory. This helper attempts to anchor the
        repository root using ``git rev-parse`` first and then falls back to
        discovering a ``.git`` directory while walking up the filesystem tree.
        """

        candidate_bases = [
            module if module.is_absolute() else Path(__file__).resolve().parent,
            Path(__file__).resolve().parent,
        ]

        for base in candidate_bases:
            try:
                result = subprocess.run(
                    ["git", "rev-parse", "--show-toplevel"],
                    cwd=base if base.is_dir() else base.parent,
                    capture_output=True,
                    text=True,
                    check=True,
                )
                root = Path(result.stdout.strip()).resolve()
                if root.exists():
                    return root
            except Exception:
                continue

        for base in candidate_bases:
            current = base if base.is_dir() else base.parent
            for parent in [current] + list(current.parents):
                if (parent / ".git").exists():
                    return parent.resolve()

        return Path.cwd().resolve()

    @contextmanager
    def _temporary_repo_root(self, root: Path) -> Iterator[None]:
        """Temporarily redirect dynamic path resolution to *root*."""

        cache_lock = getattr(_path_router, "_CACHE_LOCK", None)
        path_cache = getattr(_path_router, "_PATH_CACHE", None)
        prev_root = getattr(_path_router, "_PROJECT_ROOT", None)
        prev_roots = getattr(_path_router, "_PROJECT_ROOTS", None)
        prev_cache: dict[str, Path] = {}
        env_keys = (
            "MENACE_ROOT",
            "MENACE_ROOTS",
            "SANDBOX_REPO_PATH",
            "SANDBOX_REPO_PATHS",
        )
        prev_env = {key: os.environ.get(key) for key in env_keys}
        if cache_lock and path_cache is not None:
            with cache_lock:
                try:
                    prev_cache = dict(path_cache)
                except Exception:
                    prev_cache = {}
                setattr(_path_router, "_PROJECT_ROOT", root)
                setattr(_path_router, "_PROJECT_ROOTS", [root])
                try:
                    path_cache.clear()
                except Exception:
                    pass
        for key in env_keys:
            os.environ[key] = str(root)
        try:
            yield
        finally:
            for key, value in prev_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
            if cache_lock and path_cache is not None:
                with cache_lock:
                    setattr(_path_router, "_PROJECT_ROOT", prev_root)
                    setattr(_path_router, "_PROJECT_ROOTS", prev_roots)
                    try:
                        path_cache.clear()
                        path_cache.update(prev_cache)
                    except Exception:
                        pass

    def _workflow_test_service_args(
        self,
    ) -> tuple[str | None, dict[str, Any], list[str], dict[str, list[str]]]:
        """Resolve pytest arguments, kwargs and selected workflow tests."""

        def _resolve(source: Any) -> Any:
            if source is None:
                return None
            if callable(source):
                try:
                    return source(self.bot_name)
                except TypeError:
                    return source()
                except Exception:
                    self.logger.exception("workflow test args callable failed")
                    return None
            if isinstance(source, dict):
                return source.get(self.bot_name) or source.get("default")
            return source

        def _normalise_tokens(candidate: Any) -> list[str]:
            if candidate is None:
                return []
            if isinstance(candidate, str):
                candidate = candidate.strip()
                if not candidate:
                    return []
                try:
                    return [token for token in shlex.split(candidate) if token]
                except ValueError:
                    return [candidate]
            if isinstance(candidate, (list, tuple, set)):
                tokens: list[str] = []
                for item in candidate:
                    if item is None:
                        continue
                    if isinstance(item, str) and " " in item.strip():
                        try:
                            tokens.extend(token for token in shlex.split(item) if token)
                        except ValueError:
                            tokens.append(item.strip())
                    else:
                        text = str(item).strip()
                        if text:
                            tokens.append(text)
                return tokens
            text = str(candidate).strip()
            return [text] if text else []

        def _is_selector(token: str) -> bool:
            if not token or token.startswith("-"):
                return False
            lowered = token.lower()
            if lowered in {"python", "pytest", "py.test", sys.executable.lower()}:
                return False
            return True

        workflow_tests: list[str] = []
        workflow_sources: dict[str, list[str]] = {}
        seen: set[str] = set()
        pytest_tokens: list[str] | None = None

        def _record(source: str, tokens: Iterable[str]) -> list[str]:
            added: list[str] = []
            for token in tokens:
                tok = str(token).strip()
                if not _is_selector(tok):
                    continue
                if tok not in seen:
                    seen.add(tok)
                    workflow_tests.append(tok)
                    added.append(tok)
            if added:
                dest = workflow_sources.setdefault(source, [])
                for tok in added:
                    if tok not in dest:
                        dest.append(tok)
            return added

        def _extend_pytest(tokens: Iterable[str]) -> None:
            nonlocal pytest_tokens
            if pytest_tokens is None:
                pytest_tokens = []
            for token in tokens:
                tok = str(token).strip()
                if not tok:
                    continue
                if tok not in pytest_tokens:
                    pytest_tokens.append(tok)

        provider_sources = (
            ("pipeline", getattr(self.pipeline, "workflow_test_args", None)),
            ("engine", getattr(self.engine, "workflow_test_args", None)),
            ("data_bot", getattr(self.data_bot, "workflow_test_args", None)),
        )
        for source_name, provider in provider_sources:
            candidate = _resolve(provider)
            if not candidate:
                continue
            tokens = _normalise_tokens(candidate)
            if not tokens:
                continue
            _extend_pytest(tokens)
            _record(source_name, tokens)

        def _registry_workflow_tests() -> list[str]:
            tests: list[str] = []
            if not self.bot_registry:
                return tests
            try:
                tests = _get_bot_workflow_tests(
                    self.bot_name, registry=self.bot_registry
                )
            except Exception:
                self.logger.exception("failed to resolve default workflow tests")
                return []
            return tests

        def _summary_workflow_tests() -> list[str]:
            summary_tests: list[str] = []
            overrides = getattr(self, "_historical_workflow_tests", None)
            if overrides:
                summary_tests.extend(normalize_workflow_tests(overrides))
            summary_dirs: list[Path] = []
            try:
                from . import workflow_run_summary as _wrs

                store = getattr(_wrs, "_SUMMARY_STORE", None)
                if store:
                    summary_dirs.append(Path(store))
            except Exception:
                self.logger.debug("workflow summary store unavailable", exc_info=True)
            try:
                data_root = Path(resolve_path("sandbox_data"))
                summary_dirs.extend([data_root, data_root / "workflows"])
            except Exception:
                summary_dirs.extend([Path("sandbox_data"), Path("sandbox_data") / "workflows"])

            seen_dirs: set[Path] = set()
            for directory in summary_dirs:
                directory = Path(directory)
                if not directory.exists() or directory in seen_dirs:
                    continue
                seen_dirs.add(directory)
                for summary_path in directory.glob("*.summary.json"):
                    try:
                        data = json.loads(summary_path.read_text())
                    except Exception:
                        continue
                    metadata = data.get("metadata")
                    if not isinstance(metadata, dict):
                        metadata = {}
                    targeted = False
                    for key in ("bot", "bot_name", "target_bot", "owner"):
                        value = metadata.get(key)
                        if value and str(value) == self.bot_name:
                            targeted = True
                            break
                    if not targeted:
                        bots = normalize_workflow_tests(metadata.get("bots"))
                        if bots and self.bot_name in bots:
                            targeted = True
                    if not targeted and metadata:
                        continue
                    for key in (
                        "workflow_tests",
                        "pytest_args",
                        "tests",
                        "selectors",
                        "test_paths",
                    ):
                        summary_tests.extend(normalize_workflow_tests(metadata.get(key)))
                        summary_tests.extend(normalize_workflow_tests(data.get(key)))
            return summary_tests

        def _heuristic_workflow_tests() -> list[str]:
            selectors: list[str] = []
            module_path: Path | None = None
            if self.bot_registry:
                try:
                    graph = getattr(self.bot_registry, "graph", None)
                    if graph is not None and self.bot_name in getattr(graph, "nodes", {}):
                        node = graph.nodes[self.bot_name]
                        module_val = node.get("module")
                        if module_val:
                            module_path = Path(module_val)
                except Exception:
                    module_path = None
                if (module_path is None or not module_path.exists()) and hasattr(
                    self.bot_registry, "modules"
                ):
                    try:
                        module_entry = self.bot_registry.modules.get(self.bot_name)  # type: ignore[arg-type]
                        if module_entry:
                            module_path = Path(module_entry)
                    except Exception:
                        module_path = None
            if (module_path is None) or not module_path.exists():
                try:
                    module = importlib.import_module(self.bot_name)
                except Exception:
                    module = None
                if module is not None:
                    module_file = getattr(module, "__file__", "")
                    if module_file:
                        candidate = Path(module_file)
                        if candidate.exists():
                            module_path = candidate
            identifiers: set[str] = {self.bot_name}
            if module_path and module_path.exists():
                identifiers.add(module_path.stem)
                if module_path.parent.name:
                    identifiers.add(module_path.parent.name)
            test_roots = [Path("tests"), Path("tests") / "integration", Path("unit_tests")]
            candidates: list[Path] = []
            for ident in {slug.replace("-", "_") for slug in identifiers if slug}:
                for root in test_roots:
                    base = root / f"test_{ident}.py"
                    if base.exists():
                        candidates.append(base)
                    workflow_variant = root / f"test_{ident}_workflow.py"
                    if workflow_variant.exists():
                        candidates.append(workflow_variant)
                    dir_candidate = root / ident
                    if dir_candidate.exists():
                        candidates.append(dir_candidate)
            if module_path and module_path.exists():
                local_dir = module_path.parent
                local_candidate = local_dir / f"test_{module_path.stem}.py"
                if local_candidate.exists():
                    candidates.append(local_candidate)
            selectors.extend(str(path.resolve()) for path in candidates if path.exists())
            return selectors

        if not workflow_tests:
            registry_tokens = _registry_workflow_tests()
            added = _record("registry", registry_tokens)
            if added:
                _extend_pytest(added)

        if not workflow_tests:
            summary_tokens = _summary_workflow_tests()
            added = _record("summary", summary_tokens)
            if added:
                _extend_pytest(added)

        if not workflow_tests:
            heuristic_tokens = _heuristic_workflow_tests()
            added = _record("heuristic", heuristic_tokens)
            if added:
                _extend_pytest(added)

        if not workflow_tests:
            self.logger.warning(
                "no workflow tests resolved for bot %s; skipping validation",
                self.bot_name,
            )
            return None, {}, [], workflow_sources

        args: str | None = None
        if pytest_tokens:
            try:
                args = shlex.join(pytest_tokens)
            except AttributeError:
                args = " ".join(pytest_tokens)

        kwargs: dict[str, Any] = {}
        worker_src = _resolve(getattr(self.pipeline, "workflow_test_workers", None))
        if worker_src is not None:
            try:
                kwargs["workers"] = int(worker_src)
            except Exception:
                self.logger.debug("invalid workflow_test_workers value: %s", worker_src)
        extra_opts = _resolve(getattr(self.pipeline, "workflow_test_kwargs", None))
        if isinstance(extra_opts, dict):
            kwargs.update(extra_opts)
        return args, kwargs, workflow_tests, workflow_sources

    @staticmethod
    def _truncate(text: str, *, limit: int = 2000) -> str:
        """Return ``text`` truncated to ``limit`` characters."""

        if text is None:
            return ""
        if len(text) <= limit:
            return text
        return text[:limit] + f"… ({len(text) - limit} bytes truncated)"

    @staticmethod
    def _pytest_failures(stdout: str) -> list[str]:
        """Extract pytest node ids from ``stdout``."""

        node_ids: list[str] = []
        if not stdout:
            return node_ids
        patterns = [
            re.compile(r"^(FAILED|ERROR)\s+([\w./:-]+::[^\s]+)", re.MULTILINE),
            re.compile(r"^([\w./:-]+::[^\s]+)\s+(FAILED|ERROR)$", re.MULTILINE),
        ]
        seen: set[str] = set()
        for pattern in patterns:
            for match in pattern.finditer(stdout):
                node = match.group(2 if pattern is patterns[0] else 1)
                if node and node not in seen:
                    seen.add(node)
                    node_ids.append(node)
        if seen:
            return node_ids
        summary_line = re.compile(r"^(FAILED|ERROR)\s+([\w./:-]+::[^\s]+)")
        for line in stdout.splitlines():
            line = line.strip()
            if "::" not in line:
                continue
            match = summary_line.match(line)
            if match:
                node = match.group(2)
            elif line.endswith("FAILED") or line.endswith("ERROR"):
                node = line.split()[0]
            else:
                continue
            if node and node not in seen:
                seen.add(node)
                node_ids.append(node)
        return node_ids

    def _collect_test_diagnostics(self, results: dict[str, Any]) -> dict[str, Any]:
        """Return structured diagnostics extracted from ``results``."""

        diagnostics: dict[str, Any] = {}
        stdout = str(results.get("stdout", "") or "")
        stderr = str(results.get("stderr", "") or "")
        logs = str(results.get("logs", "") or "")
        if stdout:
            diagnostics["stdout"] = self._truncate(stdout)
        if stderr:
            diagnostics["stderr"] = self._truncate(stderr)
        if logs:
            diagnostics["logs"] = self._truncate(logs)
        combined = stdout or stderr
        if combined:
            failure = ErrorParser.parse_failure(combined)
            trace = failure.get("stack") or combined
            diagnostics["trace"] = self._truncate(trace, limit=4000)
            if failure.get("strategy_tag"):
                diagnostics["failure_tag"] = failure.get("strategy_tag")
            if failure.get("signature"):
                diagnostics["failure_signature"] = failure.get("signature")
            if failure.get("file"):
                diagnostics["failure_file"] = failure.get("file")
        node_ids = self._pytest_failures(stdout)
        if node_ids:
            diagnostics["node_ids"] = node_ids
        modules: list[str] = []
        metrics = results.get("module_metrics") or {}
        if isinstance(metrics, dict):
            for module, info in metrics.items():
                categories = {str(cat) for cat in info.get("categories", [])}
                if categories.intersection({"failed", "error"}):
                    modules.append(str(module))
        if modules:
            diagnostics["failed_modules"] = modules
        retry_errors = results.get("retry_errors")
        if retry_errors:
            diagnostics["retry_errors"] = retry_errors
        return diagnostics

    def _select_repair_pytest_args(
        self,
        base_args: str | None,
        diagnostics: dict[str, Any],
    ) -> str | None:
        """Return pytest arguments targeting the failing subset of tests."""

        node_ids = diagnostics.get("node_ids") or []
        failed_modules = diagnostics.get("failed_modules") or []
        selectors: list[str] = []
        sources = node_ids if node_ids else failed_modules
        for item in sources:
            if not item:
                continue
            if item not in selectors:
                selectors.append(item)
        if not selectors:
            return base_args
        base_parts = shlex.split(base_args) if base_args else []
        options = [part for part in base_parts if part.startswith("-")]
        new_args = options + selectors
        return " ".join(new_args) if new_args else base_args

    def _synthesise_repair_description(
        self,
        base_description: str,
        diagnostics: dict[str, Any],
        *,
        attempt: int,
        failed_tests: int,
    ) -> str:
        """Create a repair prompt that includes failing test context."""

        lines = [base_description.strip() or "Self-test repair"]
        lines.append(
            f"Repair attempt {attempt} addressing {failed_tests} failing test(s)."
        )
        node_ids = diagnostics.get("node_ids") or []
        if node_ids:
            lines.append("Failing tests:")
            lines.extend(f"- {node}" for node in node_ids[:10])
        trace = diagnostics.get("trace") or diagnostics.get("stdout") or ""
        if trace:
            lines.append("Failure context:")
            lines.append(self._truncate(str(trace), limit=1200))
        return "\n".join(lines)

    def _record_repair_outcome(
        self,
        module: Path,
        *,
        attempt: int,
        success: bool,
        patch_id: int | None = None,
        flags: list[str] | None = None,
        error: str | None = None,
        diagnostics: dict[str, Any] | None = None,
    ) -> None:
        """Emit telemetry for a repair attempt."""

        payload: dict[str, Any] = {
            "bot": self.bot_name,
            "module": str(module),
            "attempt": attempt,
            "success": bool(success),
        }
        if patch_id is not None:
            payload["patch_id"] = patch_id
        if flags:
            payload["flags"] = list(flags)
        if error:
            payload["error"] = error
        if diagnostics and diagnostics.get("node_ids"):
            payload["node_ids"] = list(diagnostics["node_ids"])
        if diagnostics and diagnostics.get("failed_modules"):
            payload["failed_modules"] = list(diagnostics["failed_modules"])
        if self.event_bus:
            try:
                self.event_bus.publish("self_coding:repair_attempt", payload)
            except Exception:
                self.logger.exception("failed to publish repair attempt event")
        if self.data_bot and hasattr(self.data_bot, "record_validation"):
            try:
                self.data_bot.record_validation(
                    self.bot_name, str(module), bool(success), list(flags or [])
                )
            except Exception:
                self.logger.exception("failed to record repair validation")

    def run_post_patch_cycle(
        self,
        module_path: Path | str,
        description: str,
        *,
        provenance_token: str,
        context_meta: Dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Validate the updated module and execute workflow self tests."""

        print(f"[debug] run_post_patch_cycle starting for: {self.bot_name}")
        self.validate_provenance(provenance_token)
        if self.quick_fix is None:
            raise RuntimeError("QuickFixEngine validation unavailable")
        module = Path(module_path)
        repo_root = self._resolve_repo_root(module)
        if not module.is_absolute():
            module = (repo_root / module).resolve()
        if not module.exists():
            raise FileNotFoundError(f"module path not found: {module}")
        self.refresh_quick_fix_context()
        summary: dict[str, Any] = {}
        ctx_meta = dict(context_meta or {})
        default_timeout = 900.0
        raw_timeout = os.getenv("POST_PATCH_TIMEOUT_SECS")
        if raw_timeout:
            try:
                post_patch_timeout_secs = float(raw_timeout)
            except ValueError:
                self.logger.warning(
                    "invalid POST_PATCH_TIMEOUT_SECS value; using default",
                    extra={"value": raw_timeout, "default": default_timeout},
                )
                post_patch_timeout_secs = default_timeout
        else:
            post_patch_timeout_secs = default_timeout
        if post_patch_timeout_secs <= 0:
            post_patch_timeout_secs = default_timeout
        default_heartbeat = 45.0
        raw_heartbeat = os.getenv("POST_PATCH_HEARTBEAT_SECS")
        if raw_heartbeat:
            try:
                heartbeat_interval = float(raw_heartbeat)
            except ValueError:
                self.logger.warning(
                    "invalid POST_PATCH_HEARTBEAT_SECS value; using default",
                    extra={"value": raw_heartbeat, "default": default_heartbeat},
                )
                heartbeat_interval = default_heartbeat
        else:
            heartbeat_interval = default_heartbeat
        if heartbeat_interval < 30.0 or heartbeat_interval > 60.0:
            self.logger.warning(
                "POST_PATCH_HEARTBEAT_SECS outside 30-60s range; using default",
                extra={"value": heartbeat_interval, "default": default_heartbeat},
            )
            heartbeat_interval = default_heartbeat

        def _run_step_with_timeout(
            step_name: str,
            func: Callable[..., Any],
            *args: Any,
            **kwargs: Any,
        ) -> tuple[bool, Any, float]:
            done_event = threading.Event()
            start_time = time.monotonic()

            def _heartbeat() -> None:
                while not done_event.wait(heartbeat_interval):
                    self.logger.info(
                        "post patch cycle heartbeat",
                        extra={
                            "step": step_name,
                            "bot_name": self.bot_name,
                            "timeout": post_patch_timeout_secs,
                        },
                    )

            thread = threading.Thread(target=_heartbeat, daemon=True)
            thread.start()
            try:
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(func, *args, **kwargs)
                    try:
                        result = future.result(timeout=post_patch_timeout_secs)
                        return False, result, time.monotonic() - start_time
                    except concurrent.futures.TimeoutError:
                        cancelled = future.cancel()
                        elapsed = time.monotonic() - start_time
                        self.logger.error(
                            "post patch cycle step timed out",
                            extra={
                                "step": step_name,
                                "timeout": post_patch_timeout_secs,
                                "cancelled": cancelled,
                                "elapsed_secs": elapsed,
                            },
                        )
                        return True, None, elapsed
            finally:
                done_event.set()
                thread.join(timeout=1.0)

        def _finish_timeout(step_name: str, elapsed_secs: float) -> dict[str, Any]:
            summary["timed_out"] = {
                "step": step_name,
                "timeout_secs": post_patch_timeout_secs,
                "elapsed_secs": elapsed_secs,
            }
            self._last_validation_summary = summary
            if self.data_bot:
                try:
                    self.data_bot.collect(
                        self.bot_name,
                        post_patch_cycle_success=0.0,
                        post_patch_cycle_error=f"timeout:{step_name}",
                    )
                except Exception:
                    self.logger.exception(
                        "failed to record post patch timeout metrics"
                    )
            return summary
        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                subprocess.run(["git", "clone", str(repo_root), tmp_dir], check=True)
                clone_root = Path(tmp_dir).resolve()
                try:
                    rel = module.relative_to(repo_root)
                except ValueError:
                    rel = module.name
                cloned_module = clone_root / rel
                if not cloned_module.exists():
                    raise FileNotFoundError(
                        f"cloned module path not found: {cloned_module}"
                    )
                with self._temporary_repo_root(clone_root):
                    prev_cwd = os.getcwd()
                    os.chdir(str(clone_root))
                    try:
                        timed_out, result, elapsed_secs = _run_step_with_timeout(
                            "validate_patch",
                            self.quick_fix.validate_patch,
                            str(cloned_module),
                            description,
                            repo_root=clone_root,
                            provenance_token=provenance_token,
                        )
                        if timed_out:
                            summary["quick_fix"] = {
                                "timed_out": True,
                                "step": "validate_patch",
                                "elapsed_secs": elapsed_secs,
                            }
                            return _finish_timeout("validate_patch", elapsed_secs)
                        valid, flags = result
                        summary["quick_fix"] = {
                            "validation_flags": list(flags),
                        }
                        skip_apply = False
                        if not valid or flags:
                            skip_flags = {
                                "index out of range in self",
                                "goal must supply a non-empty query",
                            }
                            if set(flags).issubset(skip_flags):
                                skip_apply = True
                                summary["quick_fix"]["skipped_apply"] = True
                                self.logger.warning(
                                    "quick fix validation encountered non-fatal flags; "
                                    "continuing without apply",
                                    extra={
                                        "module_name": str(cloned_module),
                                        "flags": list(flags),
                                    },
                                )
                            else:
                                raise RuntimeError(
                                    "quick fix validation failed: "
                                    f"valid={valid}, flags={list(flags)}"
                                )
                        if not skip_apply:
                            self._run_integrity_check_hook(
                                stage="run_post_patch_cycle:pre_apply",
                                path=module,
                            )
                            timed_out, result, elapsed_secs = _run_step_with_timeout(
                                "apply_validated_patch",
                                self.quick_fix.apply_validated_patch,
                                str(cloned_module),
                                description,
                                ctx_meta,
                                provenance_token=provenance_token,
                            )
                            if timed_out:
                                summary["quick_fix"]["timed_out"] = True
                                summary["quick_fix"]["step"] = "apply_validated_patch"
                                summary["quick_fix"]["elapsed_secs"] = elapsed_secs
                                return _finish_timeout(
                                    "apply_validated_patch", elapsed_secs
                                )
                            passed, _pid, apply_flags = result
                            self._run_integrity_check_hook(
                                stage="run_post_patch_cycle:post_apply",
                                path=module,
                            )
                            summary["quick_fix"].update(
                                {
                                    "apply_flags": list(apply_flags),
                                    "passed": bool(passed),
                                }
                            )
                            if not passed or apply_flags:
                                raise RuntimeError("quick fix application failed")
                    finally:
                        os.chdir(prev_cwd)
            builder = create_context_builder()
            ensure_fresh_weights(builder)
            ( 
                pytest_args,
                svc_kwargs,
                workflow_tests,
                workflow_sources,
            ) = self._workflow_test_service_args()
            svc_kwargs = dict(svc_kwargs)
            if pytest_args is not None:
                svc_kwargs["pytest_args"] = pytest_args
            svc_kwargs.setdefault("data_bot", self.data_bot)
            svc_kwargs.setdefault("context_builder", builder)
            try:
                from .self_test_service import SelfTestService as _SelfTestService
            except Exception:
                try:
                    from self_test_service import SelfTestService as _SelfTestService  # type: ignore
                except Exception as exc:
                    raise RuntimeError("SelfTestService unavailable") from exc
            if not workflow_tests:
                summary["self_tests"] = {
                    "skipped": True,
                    "reason": "no workflow tests resolved",
                }
                if workflow_sources:
                    summary["self_tests"]["workflow_sources"] = {
                        key: list(values) for key, values in workflow_sources.items()
                    }
            else:
                base_kwargs = dict(svc_kwargs)
                base_pytest_args = base_kwargs.get("pytest_args")
                attempt_records: list[dict[str, Any]] = []
                attempt_count = 0
                current_pytest_args = base_pytest_args
                results: dict[str, Any] = {}
                passed_modules: list[str] = []
                while True:
                    self._run_integrity_check_hook(
                        stage="run_post_patch_cycle:pre_cycle",
                        path=module,
                    )
                    run_kwargs = dict(base_kwargs)
                    run_kwargs.setdefault("manager", self)
                    if current_pytest_args is None:
                        run_kwargs.pop("pytest_args", None)
                    else:
                        run_kwargs["pytest_args"] = current_pytest_args
                    try:
                        service = _SelfTestService(**run_kwargs)
                    except FileNotFoundError as exc:
                        raise RuntimeError("SelfTestService initialization failed") from exc
                    timed_out, result, elapsed_secs = _run_step_with_timeout(
                        "self_tests",
                        service.run_once,
                    )
                    if timed_out:
                        summary["self_tests"] = {
                            "timed_out": True,
                            "timeout_secs": post_patch_timeout_secs,
                            "elapsed_secs": elapsed_secs,
                            "attempts": attempt_count + 1,
                        }
                        if workflow_tests:
                            summary["self_tests"]["workflow_tests"] = list(
                                workflow_tests
                            )
                        if workflow_sources:
                            summary["self_tests"]["workflow_sources"] = {
                                key: list(values)
                                for key, values in workflow_sources.items()
                            }
                        return _finish_timeout("self_tests", elapsed_secs)
                    results, passed_modules = result
                    failed_count = int(results.get("failed", 0))
                    summary["self_tests"] = {
                        "passed": int(results.get("passed", 0)),
                        "failed": failed_count,
                        "coverage": float(results.get("coverage", 0.0)),
                        "runtime": float(results.get("runtime", 0.0)),
                        "pytest_args": current_pytest_args,
                        "passed_modules": passed_modules,
                    }
                    if workflow_tests:
                        summary["self_tests"]["workflow_tests"] = list(workflow_tests)
                    if workflow_sources:
                        summary["self_tests"]["workflow_sources"] = {
                            key: list(values)
                            for key, values in workflow_sources.items()
                        }
                    executed = results.get("workflow_tests")
                    if executed:
                        summary["self_tests"]["executed_workflows"] = list(executed)
                    diagnostics = self._collect_test_diagnostics(results)
                    if diagnostics:
                        summary["self_tests"]["diagnostics"] = diagnostics
                    else:
                        summary["self_tests"].pop("diagnostics", None)
                    summary["self_tests"]["attempts"] = attempt_count + 1
                    summary_attempts = [dict(record) for record in attempt_records]
                    summary["self_tests"]["repair_attempts"] = summary_attempts
                    summary.setdefault("quick_fix", {})["repair_attempts"] = list(
                        summary_attempts
                    )
                    if failed_count == 0:
                        break
                    if attempt_count >= self.post_patch_repair_retries:
                        self._last_validation_summary = summary
                        raise RuntimeError(
                            f"self tests failed ({failed_count}) after {attempt_count} repair attempts"
                        )
                    attempt_index = attempt_count + 1
                    repair_desc = self._synthesise_repair_description(
                        description,
                        diagnostics,
                        attempt=attempt_index,
                        failed_tests=failed_count,
                    )
                    ctx_meta_attempt = dict(ctx_meta)
                    ctx_meta_attempt.update(
                        {
                            "repair_attempt": attempt_index,
                            "repair_failed_tests": failed_count,
                        }
                    )
                    if diagnostics.get("node_ids"):
                        ctx_meta_attempt["repair_node_ids"] = list(
                            diagnostics["node_ids"]
                        )
                    if diagnostics.get("failed_modules"):
                        ctx_meta_attempt["repair_failed_modules"] = list(
                            diagnostics["failed_modules"]
                        )
                    next_pytest_args = self._select_repair_pytest_args(
                        base_pytest_args, diagnostics
                    )
                    attempt_record: dict[str, Any] = {
                        "attempt": attempt_index,
                        "failed_tests": failed_count,
                        "pytest_args": current_pytest_args,
                        "description": self._truncate(repair_desc, limit=800),
                    }
                    if diagnostics.get("node_ids"):
                        attempt_record["node_ids"] = list(diagnostics["node_ids"])
                    if diagnostics.get("failed_modules"):
                        attempt_record["failed_modules"] = list(
                            diagnostics["failed_modules"]
                        )
                    if next_pytest_args is not None:
                        attempt_record["next_pytest_args"] = next_pytest_args
                    try:
                        self.refresh_quick_fix_context()
                        self._run_integrity_check_hook(
                            stage="run_post_patch_cycle:repair_pre_apply",
                            path=module,
                        )
                        timed_out, result, elapsed_secs = _run_step_with_timeout(
                            "apply_validated_patch",
                            self.quick_fix.apply_validated_patch,
                            str(module),
                            repair_desc,
                            ctx_meta_attempt,
                            provenance_token=provenance_token,
                        )
                        if timed_out:
                            attempt_record["timed_out"] = True
                            attempt_record["elapsed_secs"] = elapsed_secs
                            attempt_records.append(attempt_record)
                            summary_attempts = [
                                dict(record) for record in attempt_records
                            ]
                            summary["self_tests"]["repair_attempts"] = summary_attempts
                            summary.setdefault("quick_fix", {})[
                                "repair_attempts"
                            ] = list(summary_attempts)
                            summary["self_tests"]["timed_out"] = True
                            summary["self_tests"]["timeout_secs"] = (
                                post_patch_timeout_secs
                            )
                            summary["self_tests"]["elapsed_secs"] = elapsed_secs
                            return _finish_timeout(
                                "apply_validated_patch", elapsed_secs
                            )
                        passed, patch_id, apply_flags = result
                        self._run_integrity_check_hook(
                            stage="run_post_patch_cycle:repair_post_apply",
                            path=module,
                        )
                    except Exception as exc:
                        attempt_record["error"] = str(exc)
                        attempt_records.append(attempt_record)
                        summary_attempts = [dict(record) for record in attempt_records]
                        summary["self_tests"]["repair_attempts"] = summary_attempts
                        summary.setdefault("quick_fix", {})["repair_attempts"] = list(
                            summary_attempts
                        )
                        self._record_repair_outcome(
                            module,
                            attempt=attempt_index,
                            success=False,
                            error=str(exc),
                            diagnostics=diagnostics,
                        )
                        self._last_validation_summary = summary
                        raise
                    attempt_record.update(
                        {
                            "patch_id": patch_id,
                            "apply_flags": list(apply_flags),
                            "patch_passed": bool(passed) and not apply_flags,
                        }
                    )
                    attempt_records.append(attempt_record)
                    summary_attempts = [dict(record) for record in attempt_records]
                    summary["self_tests"]["repair_attempts"] = summary_attempts
                    summary.setdefault("quick_fix", {})["repair_attempts"] = list(
                        summary_attempts
                    )
                    success = bool(passed) and not apply_flags
                    self._record_repair_outcome(
                        module,
                        attempt=attempt_index,
                        success=success,
                        patch_id=patch_id,
                        flags=list(apply_flags),
                        diagnostics=diagnostics,
                    )
                    if not success:
                        self._last_validation_summary = summary
                        raise RuntimeError("quick fix repair failed")
                    current_pytest_args = (
                        next_pytest_args
                        if next_pytest_args is not None
                        else base_pytest_args
                    )
                    attempt_count = attempt_index
        except Exception as exc:
            print(
                f"[debug] run_post_patch_cycle encountered error for {self.bot_name}: {exc}"
            )
            if self.data_bot:
                try:
                    self.data_bot.collect(
                        self.bot_name,
                        post_patch_cycle_success=0.0,
                        post_patch_cycle_error=str(exc),
                    )
                except Exception:
                    self.logger.exception(
                        "failed to record post patch failure metrics"
                    )
            raise
        else:
            self._last_validation_summary = summary
            if self.data_bot:
                try:
                    failed_tests = float(summary.get("self_tests", {}).get("failed", 0))
                    self.data_bot.collect(
                        self.bot_name,
                        post_patch_cycle_success=1.0,
                        post_patch_cycle_failed_tests=failed_tests,
                    )
                except Exception:
                    self.logger.exception(
                        "failed to record post patch success metrics"
                    )
            success = True
            print(f"[debug] run_post_patch_cycle success: {success}")
            return summary

    def generate_patch(
        self,
        module: str,
        description: str = "",
        *,
        helper_fn: Callable[..., str] | None = None,
        context_builder: ContextBuilder,
        provenance_token: str,
        **kwargs: Any,
    ):
        """Generate a quick fix patch for ``module``.

        ``context_builder`` must be provided by the caller and will be used for
        validation via :class:`QuickFixEngine`. ``helper_fn`` defaults to
        :func:`manager_generate_helper`.
        """

        if context_builder is None:  # pragma: no cover - defensive
            raise ValueError("ContextBuilder is required")
        if generate_patch is None:
            raise ImportError(
                "QuickFixEngine is required but generate_patch is unavailable"
            )
        module_path = Path(module)
        self._enforce_objective_guard(module_path)
        self._ensure_quick_fix_engine(context_builder)
        helper = helper_fn or _manager_generate_helper_with_builder
        return generate_patch(
            module,
            self,
            engine=getattr(self, "engine", None),
            context_builder=context_builder,
            description=description,
            helper_fn=helper,
            provenance_token=provenance_token,
            **kwargs,
        )

    # ------------------------------------------------------------------
    def scan_repo(self) -> None:
        """Invoke the enhancement classifier and check for manual commits."""

        if self.enhancement_classifier:
            try:
                suggestions = list(self.enhancement_classifier.scan_repo())
                db = self.suggestion_db or getattr(self.engine, "patch_suggestion_db", None)
                if db:
                    db.queue_suggestions(suggestions)
                event_bus = getattr(self.engine, "event_bus", None)
                if event_bus:
                    try:
                        top_scores = [
                            getattr(s, "score", 0.0)
                            for s in sorted(
                                suggestions,
                                key=lambda s: getattr(s, "score", 0.0),
                                reverse=True,
                            )[:5]
                        ]
                        event_bus.publish(
                            "enhancement:suggestions",
                            {"count": len(suggestions), "top_scores": top_scores},
                        )
                    except Exception:
                        self.logger.exception(
                            "failed to publish enhancement suggestions"
                        )
            except Exception:
                self.logger.exception("repo scan failed")

        try:
            revs = subprocess.check_output(
                ["git", "rev-list", "--max-count=10", "HEAD"], text=True
            ).splitlines()
            for commit in revs:
                meta = get_patch_by_commit(commit) if get_patch_by_commit else None
                if not meta or not meta.get("provenance_token"):
                    bus = getattr(self, "event_bus", None) or getattr(
                        self.engine, "event_bus", None
                    )
                    if bus:
                        try:
                            bus.publish(
                                "self_coding:unauthorised_commit", {"commit": commit}
                            )
                        except Exception:
                            self.logger.exception(
                                "failed to publish unauthorised commit"
                            )
                    try:
                        RollbackManager().rollback(commit)
                    except Exception:
                        self.logger.exception("rollback failed")
        except Exception:
            self.logger.exception("unauthorised commit scan failed")

    def schedule_repo_scan(self, interval: float = 3600.0) -> None:
        """Run :meth:`scan_repo` on a background scheduler."""
        if not self.enhancement_classifier:
            return

        def _loop() -> None:
            while True:
                time.sleep(interval)
                try:
                    self.scan_repo()
                    db = self.suggestion_db or getattr(
                        self.engine, "patch_suggestion_db", None
                    )
                    if db:
                        db.log_repo_scan()
                except Exception:
                    self.logger.exception("scheduled repo scan failed")

        threading.Thread(target=_loop, daemon=True).start()

    # ------------------------------------------------------------------
    def should_refactor(self) -> bool:
        """Return ``True`` when ROI, error or test metrics breach thresholds."""

        if not self.data_bot:
            return False

        self._refresh_thresholds()
        roi = self.data_bot.roi(self.bot_name)
        errors = self.data_bot.average_errors(self.bot_name)
        failures = self.data_bot.average_test_failures(self.bot_name)

        # Record metrics so rolling statistics can inform future predictions.
        self.baseline_tracker.update(roi=roi, errors=errors, tests_failed=failures)

        result = self.data_bot.check_degradation(self.bot_name, roi, errors, failures)

        # ``check_degradation`` adapts thresholds based on the latest metrics;
        # refresh the local cache so subsequent decisions reflect the new
        # values.
        self._refresh_thresholds()
        return result

    # ------------------------------------------------------------------
    def validate_provenance(self, token: str | None) -> None:
        """Ensure calls originate from the registered ``EvolutionOrchestrator``.

        A configured orchestrator is required and ``token`` must match its
        ``provenance_token``. Otherwise a :class:`PermissionError` is raised.
        """

        orchestrator = getattr(self, "evolution_orchestrator", None)
        if not orchestrator:
            self.logger.warning(
                "provenance validation skipped because EvolutionOrchestrator is unavailable",
                extra={"bot": self.bot_name},
            )
            return
        expected = getattr(orchestrator, "provenance_token", None)
        if not token or token != expected:
            self.logger.warning(
                "patch cycle without valid EvolutionOrchestrator token",
            )
            raise PermissionError("invalid provenance token")

    def _ensure_self_coding_active(self) -> None:
        if getattr(self, "_self_coding_paused", False):
            if getattr(self, "_objective_lock_requires_manifest_refresh", False):
                raise RuntimeError(
                    "self-coding paused: objective_integrity_breach "
                    "(operator reset + objective hash baseline refresh required)"
                )
            raise RuntimeError(
                f"self-coding paused: {getattr(self, '_self_coding_disabled_reason', 'paused')}"
            )

    def _objective_integrity_lock_path(self) -> Path:
        guard = getattr(self, "objective_guard", None)
        repo_root = Path.cwd().resolve() if guard is None else guard.repo_root
        return (repo_root / _OBJECTIVE_INTEGRITY_LOCK_PATH).resolve()

    def _read_manifest_sha(self) -> str | None:
        guard = getattr(self, "objective_guard", None)
        manifest_path = (
            Path("config/objective_hash_lock.json").resolve()
            if guard is None
            else guard.manifest_path.resolve()
        )
        if not manifest_path.exists():
            return None
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            return None
        if isinstance(payload, dict):
            manifest_sha = payload.get("manifest_sha256")
            if isinstance(manifest_sha, str) and manifest_sha.strip():
                return manifest_sha.strip()
        return None

    def _persist_objective_integrity_lock(self, details: Dict[str, Any]) -> None:
        target = self._objective_integrity_lock_path()
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "bot": self.bot_name,
            "locked": True,
            "reason": "objective_integrity_breach",
            "requires_operator_reset": True,
            "requires_manifest_refresh": True,
            "manifest_sha_at_breach": self._objective_lock_manifest_sha_at_breach,
            "details": details,
            "locked_at": datetime.now(timezone.utc).isoformat(),
        }
        target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def _clear_objective_integrity_lock(self) -> None:
        target = self._objective_integrity_lock_path()
        if target.exists():
            try:
                target.unlink()
            except Exception:
                self.logger.exception("failed to clear objective integrity lock file")

    def _record_audit_event(self, payload: Dict[str, Any]) -> None:
        engine = getattr(self, "engine", None)
        audit = getattr(engine, "audit_trail", None)
        if audit:
            try:
                audit.record(payload)
            except Exception:
                self.logger.exception("audit trail logging failed")

    def _persist_registry_state(self) -> None:
        registry = getattr(self, "bot_registry", None)
        target = getattr(registry, "persist_path", None)
        save = getattr(registry, "save", None)
        if target and callable(save):
            try:
                save(target)
            except Exception:
                self.logger.exception("failed to persist bot registry state")

    def _hydrate_self_coding_disabled_state(self) -> None:
        """Restore paused self-coding state from persisted registry metadata."""

        registry = getattr(self, "bot_registry", None)
        graph = getattr(registry, "graph", None)
        nodes = getattr(graph, "nodes", None)
        if not isinstance(nodes, dict):
            return
        node_state = nodes.get(self.bot_name)
        if not isinstance(node_state, dict):
            return
        disabled_state = node_state.get("self_coding_disabled")
        if not isinstance(disabled_state, dict):
            return
        reason = disabled_state.get("reason")
        if not reason:
            reason = None
        lock_path = self._objective_integrity_lock_path()
        lock_payload: Dict[str, Any] | None = None
        if lock_path.exists():
            try:
                raw = json.loads(lock_path.read_text(encoding="utf-8"))
                if isinstance(raw, dict):
                    lock_payload = raw
            except Exception:
                self.logger.exception("failed to read objective integrity lock file")
        if reason is None and lock_payload is None:
            return
        if isinstance(lock_payload, dict):
            self._objective_lock_manifest_sha_at_breach = (
                str(lock_payload.get("manifest_sha_at_breach"))
                if lock_payload.get("manifest_sha_at_breach")
                else None
            )
            self._objective_lock_requires_manifest_refresh = bool(
                lock_payload.get("requires_manifest_refresh", False)
            )
        elif str(reason) == "objective_integrity_breach":
            # If no explicit integrity lock file exists, treat the breach as
            # operator-cleared so autonomy can resume after restart.
            return
        self._self_coding_paused = True
        self._self_coding_disabled_reason = str(reason or "objective_integrity_breach")

    @staticmethod
    def _metric_from_context(context_meta: Dict[str, Any] | None, *keys: str) -> float | None:
        if not isinstance(context_meta, dict):
            return None
        for key in keys:
            if key not in context_meta:
                continue
            try:
                return float(context_meta[key])
            except (TypeError, ValueError):
                continue
        return None

    def _load_business_metric_from_data_bot(
        self,
        metric: str,
        *,
        context_meta: Dict[str, Any] | None = None,
    ) -> float | None:
        data_bot = getattr(self, "data_bot", None)
        if not data_bot:
            return None
        try:
            fetch = getattr(getattr(data_bot, "db", None), "fetch", None)
            if not callable(fetch):
                return None
            raw = fetch(limit=self._divergence_window)
            rows: list[dict[str, Any]] = []
            if isinstance(raw, list):
                rows = [dict(item) for item in raw if isinstance(item, dict)]
            elif hasattr(raw, "to_dict"):
                rows = [dict(item) for item in raw.to_dict("records") if isinstance(item, dict)]
            for row in rows:
                row_bot = str(row.get("bot") or "")
                if row_bot and row_bot != str(self.bot_name):
                    continue
                value = row.get(metric)
                if value is None:
                    continue
                try:
                    return float(value)
                except (TypeError, ValueError):
                    continue
        except Exception:
            self.logger.exception("failed to fetch %s trend from data bot", metric)
        return None

    def _evaluate_divergence_guard(self, context_meta: Dict[str, Any] | None) -> None:
        reward = self._metric_from_context(
            context_meta,
            "reward",
            "reward_score",
            "rl_reward",
            "reward_ledger_score",
            "reward_dispatcher_score",
        )
        profit = self._metric_from_context(context_meta, "profit", "profitability")
        revenue = self._metric_from_context(context_meta, "revenue")
        if profit is None:
            profit = self._load_business_metric_from_data_bot("profitability", context_meta=context_meta)
        if revenue is None:
            revenue = self._load_business_metric_from_data_bot("revenue", context_meta=context_meta)
        if reward is None:
            self._divergence_streak = 0
            self._divergence_recovery_streak = 0
            self._missing_real_metric_streak = 0
            return

        if profit is None and revenue is None:
            self._divergence_streak = 0
            self._divergence_recovery_streak = 0
            self._missing_real_metric_streak += 1
            missing_payload: Dict[str, Any] = {
                "bot": self.bot_name,
                "window": self._divergence_window,
                "reward": float(reward),
                "profit": None,
                "revenue": None,
                "streak": self._missing_real_metric_streak,
                "threshold": self._missing_metric_pause_cycles,
                "policy": (
                    "fail_closed"
                    if self._divergence_fail_closed_on_missing_metrics
                    else "allow"
                ),
                "reason": "real_metrics_unavailable",
                "check_status": "data_unavailable",
                "context_meta": dict(context_meta or {}),
            }
            self._record_audit_event(
                {
                    "action": "self_coding_divergence_data_unavailable",
                    "source": "divergence_monitor",
                    **missing_payload,
                }
            )
            self.logger.warning(
                "divergence guard skipped due to unavailable real metrics",
                extra={"event": "self_coding_divergence_data_unavailable", **missing_payload},
            )
            if (
                self._divergence_fail_closed_on_missing_metrics
                and self._missing_real_metric_streak >= self._missing_metric_pause_cycles
            ):
                reason = "reward_real_metrics_unavailable"
                telemetry_payload: Dict[str, Any] = {
                    "severity": "high",
                    "event": "self_coding_divergence_kill_switch",
                    "reason": reason,
                    **missing_payload,
                }
                self._self_coding_paused = True
                self._self_coding_disabled_reason = reason
                self._self_coding_pause_source = "divergence_monitor"
                self._record_audit_event(
                    {
                        "action": "self_coding_auto_pause",
                        "source": "divergence_monitor",
                        **telemetry_payload,
                    }
                )
                self.logger.critical(
                    "self-coding auto-paused: reward present while real metrics remain unavailable",
                    extra=telemetry_payload,
                )
                if self.event_bus:
                    try:
                        self.event_bus.publish("self_coding:divergence_kill_switch", telemetry_payload)
                        self.event_bus.publish("self_coding:critical_divergence", telemetry_payload)
                        self.event_bus.publish("self_coding:high_severity_alert", telemetry_payload)
                    except Exception:
                        self.logger.exception("failed to publish missing-metrics divergence telemetry")
            return

        self._missing_real_metric_streak = 0

        workflow_id = self.bot_name
        if isinstance(context_meta, dict):
            workflow_id = str(context_meta.get("workflow_id") or self.bot_name)
        cycle_index = int(self._last_event_id or 0) + 1
        record = CycleMetricsRecord(
            cycle_index=cycle_index,
            bot_id=str(self.bot_name),
            workflow_id=workflow_id,
            reward_score=float(reward),
            revenue=float(revenue) if revenue is not None else None,
            profit=float(profit) if profit is not None else None,
        )
        self._cycle_metrics_window.append(record)
        detection = self._divergence_detector.evaluate(list(self._cycle_metrics_window))
        cycle_window = list(self._cycle_metrics_window)
        reward_values = [item.reward_score for item in cycle_window]
        profit_values = [
            float(item.profit) for item in cycle_window if item.profit is not None
        ]
        revenue_values = [
            float(item.revenue) for item in cycle_window if item.revenue is not None
        ]
        diagnostic_payload: Dict[str, Any] = {
            "bot": self.bot_name,
            "window": self._divergence_window,
            "reward_window": reward_values,
            "profit_window": profit_values,
            "revenue_window": revenue_values,
            "cycle_ids": [item.cycle_index for item in cycle_window],
            "cycle_metrics": [asdict(item) for item in cycle_window],
            "detected_metric": detection.metric_name,
            "reward_delta": detection.reward_delta,
            "real_metric_delta": detection.real_metric_delta,
            "reward_trend": detection.reward_trend,
            "real_metric_trend": detection.real_metric_trend,
            "min_confidence": self._divergence_detector.config.minimum_confidence,
            "confidence": detection.confidence,
            "streak": self._divergence_streak,
            "threshold": self._divergence_threshold_cycles,
            "context_meta": dict(context_meta or {}),
            "check_status": "healthy",
        }

        if detection.triggered:
            self._divergence_streak += 1
            self._divergence_recovery_streak = 0
            diagnostic_payload["streak"] = self._divergence_streak
            self.logger.warning(
                "divergence candidate cycle detected",
                extra={"event": "self_coding_divergence_candidate", **diagnostic_payload},
            )
            if self._divergence_streak < self._divergence_threshold_cycles:
                return

            reason = "reward_profit_revenue_divergence"
            telemetry_payload: Dict[str, Any] = {
                "severity": "high",
                "event": "self_coding_divergence_kill_switch",
                "reason": reason,
                **diagnostic_payload,
            }
            self._self_coding_paused = True
            self._self_coding_disabled_reason = reason
            self._self_coding_pause_source = "divergence_monitor"
            self._record_audit_event({
                "action": "self_coding_auto_pause",
                "source": "divergence_monitor",
                **telemetry_payload,
            })
            self.logger.critical(
                "self-coding auto-paused: reward diverged from business outcomes",
                extra=telemetry_payload,
            )
            if self.event_bus:
                try:
                    self.event_bus.publish("self_coding:divergence_kill_switch", telemetry_payload)
                    self.event_bus.publish("self_coding:critical_divergence", telemetry_payload)
                    self.event_bus.publish("self_coding:high_severity_alert", telemetry_payload)
                except Exception:
                    self.logger.exception("failed to publish divergence telemetry")
            return

        self._divergence_streak = 0
        if self._self_coding_pause_source != "divergence_monitor" or not self._self_coding_paused:
            self._divergence_recovery_streak = 0
            return

        self._divergence_recovery_streak += 1
        if self._divergence_recovery_streak < self._divergence_recovery_cycles:
            return

        self._self_coding_paused = False
        self._self_coding_disabled_reason = None
        self._self_coding_pause_source = None
        self._divergence_recovery_streak = 0
        recovery_payload = {
            "event": "self_coding_divergence_recovered",
            "action": "self_coding_auto_resume",
            "source": "divergence_monitor",
            "bot": self.bot_name,
            "recovery_cycles": self._divergence_recovery_cycles,
            "window": self._divergence_window,
            "cycle_ids": [item.cycle_index for item in cycle_window],
            "reward_window": reward_values,
            "profit_window": profit_values,
            "revenue_window": revenue_values,
        }
        self._record_audit_event(recovery_payload)
        self.logger.warning("self-coding auto-resumed after divergence recovery", extra=recovery_payload)
        if self.event_bus:
            try:
                self.event_bus.publish("self_coding:divergence_recovered", recovery_payload)
            except Exception:
                self.logger.exception("failed to publish divergence recovery telemetry")

    def reset_self_coding_pause(self, *, operator_id: str, reason: str) -> None:
        """Operator-controlled reset for paused self-coding state."""

        if not operator_id.strip():
            raise ValueError("operator_id is required")
        if not reason.strip():
            raise ValueError("reason is required")
        previous_reason = self._self_coding_disabled_reason
        previous_source = self._self_coding_pause_source
        self._self_coding_paused = False
        self._self_coding_disabled_reason = None
        self._self_coding_pause_source = None
        self._divergence_streak = 0
        self._divergence_recovery_streak = 0
        self._missing_real_metric_streak = 0
        details = {
            "action": "self_coding_manual_reset",
            "bot": self.bot_name,
            "operator_id": operator_id,
            "reason": reason,
            "previous_reason": previous_reason,
            "previous_source": previous_source,
        }
        self._record_audit_event(details)
        self.logger.warning("self-coding pause reset by operator", extra=details)
        if self.event_bus:
            try:
                self.event_bus.publish("self_coding:manual_reset", details)
            except Exception:
                self.logger.exception("failed to publish self-coding reset event")

    # ------------------------------------------------------------------
    def register_patch_cycle(
        self,
        description: str,
        context_meta: Dict[str, Any] | None = None,
        *,
        patch_id: int | None = None,
        commit: str | None = None,
        provenance_token: str | None = None,
    ) -> tuple[int | None, str | None]:
        """Log baseline metrics for an upcoming patch cycle.

        Returns the ``(patch_id, commit)`` pair used for provenance
        verification.  The baseline ROI and error rates for ``bot_name`` are
        stored in :class:`PatchHistoryDB` and a ``self_coding:cycle_registered``
        event is emitted on the configured event bus.  The generated record and
        event identifiers are stored for linking with subsequent patch events.
        """

        self.validate_provenance(provenance_token)
        self._evaluate_divergence_guard(context_meta)
        self._ensure_self_coding_active()
        self._enforce_objective_manifest(stage="cycle_registration")

        roi = self.data_bot.roi(self.bot_name) if self.data_bot else 0.0
        errors = self.data_bot.average_errors(self.bot_name) if self.data_bot else 0.0
        failures = (
            self.data_bot.average_test_failures(self.bot_name) if self.data_bot else 0.0
        )
        patch_db = getattr(self.engine, "patch_db", None)
        if patch_db and patch_id is None:
            try:
                rec = PatchRecord(
                    filename=f"{self.bot_name}.cycle",
                    description=description,
                    roi_before=roi,
                    roi_after=roi,
                    errors_before=int(errors),
                    errors_after=int(errors),
                    tests_failed_before=int(failures),
                    tests_failed_after=int(failures),
                    source_bot=self.bot_name,
                    reason=context_meta.get("reason") if context_meta else None,
                    trigger=context_meta.get("trigger") if context_meta else None,
                )
                patch_id = patch_db.add(rec)
            except Exception:
                self.logger.exception("failed to log patch cycle to DB")
        elif patch_db and patch_id is not None:
            try:
                fail_after = (
                    self.data_bot.average_test_failures(self.bot_name)
                    if self.data_bot
                    else failures
                )
                conn = patch_db.router.get_connection("patch_history")
                conn.execute(
                    "UPDATE patch_history SET tests_failed_after=? WHERE id=?",
                    (int(fail_after), patch_id),
                )
                conn.commit()
            except Exception:
                self.logger.exception("failed to update test failure counts")
        self._last_patch_id = patch_id
        if commit is None:
            try:
                commit = (
                    subprocess.check_output(["git", "rev-parse", "HEAD"])
                    .decode()
                    .strip()
                )
            except Exception:
                commit = None
        self._last_commit_hash = commit
        event_id: int | None = None
        try:
            trigger = context_meta.get("trigger") if context_meta else "degradation"
            event_id = MutationLogger.log_mutation(
                change="patch_cycle_start",
                reason=description,
                trigger=trigger,
                performance=0.0,
                workflow_id=0,
                before_metric=roi,
                after_metric=roi,
                parent_id=self._last_event_id,
            )
            self._last_event_id = event_id
        except Exception:
            self.logger.exception("failed to log patch cycle event")
        if self.event_bus:
            try:
                payload = {
                    "bot": self.bot_name,
                    "patch_id": patch_id,
                    "roi_before": roi,
                    "errors_before": errors,
                    "tests_failed_before": failures,
                    "tests_failed_after": failures,
                    "description": description,
                }
                if context_meta:
                    payload.update(context_meta)
                self.event_bus.publish("self_coding:cycle_registered", payload)
            except Exception:
                self.logger.exception("failed to publish cycle_registered event")
        return patch_id, commit

    # ------------------------------------------------------------------
    def generate_and_patch(
        self,
        path: Path,
        description: str,
        *,
        context_meta: Dict[str, Any] | None = None,
        context_builder: ContextBuilder,
        provenance_token: str,
        **kwargs: Any,
    ) -> tuple[AutomationResult, str | None]:
        """Patch ``path`` using :meth:`run_patch` with the supplied context."""
        self.validate_provenance(provenance_token)

        if context_builder is None:  # pragma: no cover - defensive
            raise ValueError("ContextBuilder is required")
        self._enforce_objective_guard(path)
        builder = context_builder
        try:
            ensure_fresh_weights(builder)
        except Exception as exc:
            raise RuntimeError("failed to refresh context builder weights") from exc

        clayer = getattr(self.engine, "cognition_layer", None)
        if clayer is None:
            raise AttributeError(
                "engine.cognition_layer must provide a context_builder"
            )
        clayer.context_builder = builder
        try:
            self._ensure_quick_fix_engine(builder)
        except QuickFixEngineError as exc:
            if self.event_bus:
                try:
                    self.event_bus.publish(
                        "bot:patch_failed",
                        {"bot": self.bot_name, "reason": exc.code},
                    )
                except Exception:  # pragma: no cover - best effort
                    self.logger.exception("failed to publish patch_failed event")
            if self.data_bot:
                try:
                    self.data_bot.collect(
                        self.bot_name,
                        patch_success=0.0,
                        patch_failure_reason=exc.code,
                    )
                except Exception:  # pragma: no cover - best effort
                    self.logger.exception("failed to report patch outcome")
            self.baseline_tracker.update(patch_success=0.0)
            self._last_commit_hash = None
            raise
        except Exception as exc:
            if self.event_bus:
                try:
                    self.event_bus.publish(
                        "bot:patch_failed",
                        {"bot": self.bot_name, "reason": str(exc)},
                    )
                except Exception:  # pragma: no cover - best effort
                    self.logger.exception("failed to publish patch_failed event")
            if self.data_bot:
                try:
                    self.data_bot.collect(
                        self.bot_name,
                        patch_success=0.0,
                        patch_failure_reason=str(exc),
                    )
                except Exception:  # pragma: no cover - best effort
                    self.logger.exception("failed to report patch outcome")
            self.baseline_tracker.update(patch_success=0.0)
            self._last_commit_hash = None
            raise RuntimeError("QuickFixEngine validation unavailable") from exc
        self._last_commit_hash = None
        success = False
        failure_reason = ""
        try:
            result = self.run_patch(
                path,
                description,
                provenance_token=provenance_token,
                context_meta=context_meta,
                context_builder=builder,
                **kwargs,
            )
            commit = getattr(self, "_last_commit_hash", None)
            success = bool(commit)
            if not success:
                failure_reason = "no_commit"
        except Exception as exc:
            commit = None
            failure_reason = str(exc)
            if self.data_bot:
                try:
                    self.data_bot.collect(
                        self.bot_name,
                        patch_success=0.0,
                        patch_failure_reason=failure_reason,
                    )
                except Exception:  # pragma: no cover - best effort
                    self.logger.exception("failed to report patch outcome")
            self.baseline_tracker.update(patch_success=0.0)
            self._last_commit_hash = None
            raise
        patch_id = getattr(self, "_last_patch_id", None)
        if commit and patch_id and self.event_bus:
            try:
                self.event_bus.publish(
                    "self_coding:patch_attempt",
                    {
                        "bot": self.bot_name,
                        "path": str(path),
                        "patch_id": patch_id,
                        "commit": commit,
                    },
                )
            except Exception:  # pragma: no cover - best effort
                self.logger.exception("failed to publish patch_attempt event")
        if self.data_bot:
            try:
                self.data_bot.collect(
                    self.bot_name,
                    patch_success=1.0 if success else 0.0,
                    patch_failure_reason=None if success else failure_reason,
                )
            except Exception:  # pragma: no cover - best effort
                self.logger.exception("failed to report patch outcome")
        self.baseline_tracker.update(patch_success=1.0 if success else 0.0)
        return result, commit

    # ------------------------------------------------------------------
    def auto_run_patch(
        self,
        path: Path,
        description: str,
        *,
        run_post_validation: bool = True,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Run :meth:`run_patch` using the orchestrator's provenance token.

        ``run_post_validation`` controls whether :meth:`run_post_patch_cycle`
        executes automatically after a successful commit.  When enabled the
        resulting validation summary is returned and also published via the
        event bus and metrics collectors.  Callers that perform their own
        orchestration can disable the automatic validation to avoid duplicate
        runs.

        The returned dictionary always includes the automation ``result``
        alongside the generated ``commit`` hash, ``patch_id`` and any post
        validation ``summary`` gathered from :class:`SelfTestService`.
        """

        orchestrator = getattr(self, "evolution_orchestrator", None)
        token = getattr(orchestrator, "provenance_token", None)
        if not token:
            raise PermissionError("missing provenance token")

        context_meta: Dict[str, Any] | None = kwargs.get("context_meta")
        summary: Dict[str, Any] | None = None
        try:
            result = self.run_patch(path, description, provenance_token=token, **kwargs)
        except ObjectiveGuardViolation as exc:
            if getattr(exc, "reason", "") == "objective_integrity_breach":
                self._handle_objective_integrity_breach(
                    violation=exc,
                    path=path,
                    stage="auto_run_patch",
                    objective_checkpoint=getattr(self, "_active_objective_checkpoint", None),
                )
            raise
        commit = getattr(self, "_last_commit_hash", None)
        patch_id = getattr(self, "_last_patch_id", None)
        if run_post_validation and commit:
            summary = self.run_post_patch_cycle(
                path,
                description,
                provenance_token=token,
                context_meta=context_meta,
            )

        outcome: Dict[str, Any] = {
            "result": result,
            "commit": commit,
            "patch_id": patch_id,
            "summary": summary,
        }

        self._last_validation_summary = summary
        if summary is not None:
            if self.event_bus:
                try:
                    payload: Dict[str, Any] = {
                        "bot": self.bot_name,
                        "path": str(path),
                        "patch_id": patch_id,
                        "commit": commit,
                        "summary": summary,
                    }
                    if context_meta:
                        payload["context_meta"] = context_meta
                    self.event_bus.publish("self_coding:post_validation", payload)
                except Exception:
                    self.logger.exception("failed to publish post_validation event")
            if self.data_bot:
                try:
                    failed_tests = float(summary.get("self_tests", {}).get("failed", 0) or 0.0)
                    coverage = float(summary.get("self_tests", {}).get("coverage", 0.0) or 0.0)
                    self.data_bot.collect(
                        self.bot_name,
                        patch_validation_failed_tests=failed_tests,
                        patch_validation_coverage=coverage,
                    )
                except Exception:
                    self.logger.exception("failed to record post validation metrics")
        return outcome

    def _enforce_objective_manifest(self, *, stage: str, path: Path | None = None) -> None:
        guard = getattr(self, "objective_guard", None)
        if guard is None:
            return
        target_path = path or Path("self_coding_manager.py")
        try:
            verify_objective_hash_lock(guard=guard)
        except ObjectiveGuardViolation as exc:
            details = dict(getattr(exc, "details", {}) or {})
            details.setdefault("bot", self.bot_name)
            details.setdefault("path", path_for_prompt(target_path))
            self.logger.critical("objective guard blocked self-coding action", extra=details)
            if self.event_bus:
                try:
                    payload = {"bot": self.bot_name, "reason": exc.reason, "details": details}
                    self.event_bus.publish("self_coding:objective_guard_block", payload)
                except Exception:
                    self.logger.exception("failed to publish objective guard block event")
            if getattr(exc, "reason", "") in {
                "objective_integrity_breach",
                "manifest_hash_mismatch",
                "manifest_missing",
                "manifest_invalid",
            }:
                self._trigger_objective_circuit_breaker(
                    violation=exc,
                    path=target_path,
                    stage=stage,
                )
            raise

    def _enforce_objective_guard(self, path: Path) -> None:
        self._enforce_objective_manifest(stage="pre_patch_guard_check", path=path)
        guard = getattr(self, "objective_guard", None)
        if guard is not None:
            try:
                guard.assert_patch_target_safe(path)
            except ObjectiveGuardViolation as exc:
                details = dict(getattr(exc, "details", {}) or {})
                details.setdefault("bot", self.bot_name)
                details.setdefault("path", path_for_prompt(path))
                self.logger.critical("objective guard blocked self-coding action", extra=details)
                if self.event_bus:
                    try:
                        payload = {"bot": self.bot_name, "reason": exc.reason, "details": details}
                        self.event_bus.publish("self_coding:objective_guard_block", payload)
                    except Exception:
                        self.logger.exception("failed to publish objective guard block event")
                raise

        policy = get_patch_promotion_policy(repo_root=Path.cwd().resolve())
        if not policy.is_safe_target(path):
            details = {
                "bot": self.bot_name,
                "path": path_for_prompt(path),
                "policy": "self_coding_patch_promotion",
                "classification": "objective_adjacent",
            }
            self.logger.warning(
                "self-coding target classified as objective-adjacent and requires manual approval",
                extra=details,
            )

    def _set_self_coding_disabled_state(self, reason: str, *, details: Dict[str, Any]) -> None:
        """Persist disabled state in-memory and in registry metadata when possible."""

        self._self_coding_paused = True
        self._self_coding_disabled_reason = reason
        if reason == "objective_integrity_breach":
            self._objective_lock_requires_manifest_refresh = True
            self._objective_lock_manifest_sha_at_breach = self._read_manifest_sha()
            self._persist_objective_integrity_lock(details)
        registry = getattr(self, "bot_registry", None)
        graph = getattr(registry, "graph", None)
        nodes = getattr(graph, "nodes", None)
        if isinstance(nodes, dict):
            node_state = nodes.setdefault(self.bot_name, {})
            if isinstance(node_state, dict):
                node_state["self_coding_disabled"] = {
                    "source": "objective_integrity_breach",
                    "reason": reason,
                    "details": details,
                    "requires_manifest_refresh": self._objective_lock_requires_manifest_refresh,
                    "manifest_sha_at_breach": self._objective_lock_manifest_sha_at_breach,
                }
        self._persist_registry_state()

    def reset_objective_integrity_lock(self, *, operator_id: str, reason: str) -> None:
        """Allow operators to explicitly resume self-coding after an objective breach."""

        if not operator_id.strip():
            raise ValueError("operator_id is required")
        if not reason.strip():
            raise ValueError("reason is required")
        verify_objective_hash_lock(guard=self.objective_guard)
        current_manifest_sha = self._read_manifest_sha()
        if (
            self._objective_lock_requires_manifest_refresh
            and self._objective_lock_manifest_sha_at_breach
            and current_manifest_sha == self._objective_lock_manifest_sha_at_breach
        ):
            raise RuntimeError(
                "objective integrity lock reset denied: refresh objective hash baseline first"
            )

        with self._objective_breach_lock:
            self._objective_breach_handled = False
            self._self_coding_paused = False
            self._self_coding_disabled_reason = None
            self._objective_lock_requires_manifest_refresh = False
            self._objective_lock_manifest_sha_at_breach = None
        registry = getattr(self, "bot_registry", None)
        graph = getattr(registry, "graph", None)
        nodes = getattr(graph, "nodes", None)
        if isinstance(nodes, dict):
            node_state = nodes.setdefault(self.bot_name, {})
            if isinstance(node_state, dict):
                node_state.pop("self_coding_disabled", None)
        self._record_audit_event(
            {
                "event": "objective_integrity_lock_reset",
                "severity": "high",
                "bot": self.bot_name,
                "operator_id": operator_id,
                "reason": reason,
                "manifest_refreshed": True,
            }
        )
        self._clear_objective_integrity_lock()
        self._persist_registry_state()

    def _run_integrity_check_hook(self, *, stage: str, path: Path) -> None:
        guard = getattr(self, "objective_guard", None)
        if guard is None:
            return
        try:
            guard.assert_integrity()
        except ObjectiveGuardViolation as exc:
            self._trigger_objective_circuit_breaker(
                violation=exc,
                path=path,
                stage=stage,
            )

    def _trigger_objective_circuit_breaker(
        self,
        *,
        violation: ObjectiveGuardViolation,
        path: Path,
        stage: str,
    ) -> None:
        """Circuit-break on objective integrity violations during autonomous cycles."""

        details = dict(getattr(violation, "details", {}) or {})
        details.setdefault("bot", self.bot_name)
        details.setdefault("path", path_for_prompt(path))
        details.setdefault("stage", stage)
        commit, commit_meta = self._resolve_current_or_last_patch_commit()
        self._set_self_coding_disabled_state("objective_integrity_breach", details=details)

        rollback_attempted = bool(commit)
        rollback_ok = False
        rollback_error: str | None = None
        if commit and RollbackManager is not None:
            try:
                RollbackManager().rollback(commit, requesting_bot=self.bot_name)
                rollback_ok = True
            except Exception as exc:
                rollback_error = str(exc)
                self.logger.critical("circuit breaker rollback failed", exc_info=True)
        elif commit and RollbackManager is None:
            rollback_error = "rollback_manager_unavailable"
        else:
            rollback_error = "no_commit_available"

        payload: Dict[str, Any] = {
            "severity": "critical",
            "bot": self.bot_name,
            "reason": violation.reason,
            "stage": stage,
            "details": details,
            "reverted_commit": commit,
            "reverted_commit_metadata": commit_meta,
            "rollback_attempted": rollback_attempted,
            "rollback_ok": rollback_ok,
            "rollback_error": rollback_error,
        }
        if self.event_bus:
            try:
                self.event_bus.publish("self_coding:circuit_breaker_triggered", payload)
            except Exception:
                self.logger.exception("failed to publish circuit breaker event")
        self._active_objective_checkpoint = None
        raise RuntimeError("objective integrity circuit breaker triggered")

    def _resolve_current_or_last_patch_commit(self) -> tuple[str | None, Dict[str, Any]]:
        """Resolve the best-known patch commit and associated provenance metadata."""

        commit = getattr(self, "_last_commit_hash", None)
        metadata: Dict[str, Any] = {}
        if commit and get_patch_by_commit:
            try:
                metadata = dict(get_patch_by_commit(commit) or {})
            except Exception:
                self.logger.exception("failed to resolve patch provenance metadata")
        patch_id = getattr(self, "_last_patch_id", None)
        if patch_id is not None:
            metadata.setdefault("patch_id", patch_id)
        return commit, metadata

    def _handle_objective_integrity_breach(
        self,
        *,
        violation: ObjectiveGuardViolation,
        path: Path,
        stage: str,
        objective_checkpoint: Dict[str, Any] | None = None,
    ) -> None:
        """Roll back latest patch state and halt self-coding after objective drift."""

        details = dict(getattr(violation, "details", {}) or {})
        details.setdefault("bot", self.bot_name)
        details.setdefault("path", path_for_prompt(path))
        changed_files = sorted(
            str(item) for item in (details.get("changed_files") or []) if item
        )
        details["changed_files"] = changed_files
        commit, commit_meta = self._resolve_current_or_last_patch_commit()
        patch_id = getattr(self, "_last_patch_id", None)
        rollback_target = commit if commit else (str(patch_id) if patch_id is not None else None)
        telemetry_payload: Dict[str, Any] = {
            "severity": "critical",
            "priority": "high",
            "bot": self.bot_name,
            "reason": violation.reason,
            "stage": stage,
            "details": details,
            "changed_objective_files": changed_files,
            "reverted_commit": commit,
            "reverted_commit_metadata": commit_meta,
            "already_handled": False,
        }
        with self._objective_breach_lock:
            if self._objective_breach_handled:
                telemetry_payload["already_handled"] = True
                self._set_self_coding_disabled_state(
                    "objective_integrity_breach", details=details
                )
                if self.event_bus:
                    try:
                        self.event_bus.publish(
                            "self_coding:objective_integrity_breach",
                            telemetry_payload,
                        )
                        self.event_bus.publish(
                            "self_coding:objective_integrity_trip",
                            telemetry_payload,
                        )
                        self.event_bus.publish(
                            "self_coding:objective_circuit_breaker_trip",
                            telemetry_payload,
                        )
                    except Exception:
                        self.logger.exception(
                            "failed to publish objective integrity breach duplicate event"
                        )
                return
            self._objective_breach_handled = True

        self._record_audit_event(
            {
                "event": "objective_integrity_breach",
                "severity": "critical",
                "priority": "high",
                "bot": self.bot_name,
                "stage": stage,
                "reason": violation.reason,
                "changed_files": changed_files,
                "rollback_target": rollback_target,
            }
        )

        rollback_attempted = bool(rollback_target)
        rollback_ok = False
        rollback_error: str | None = None
        if rollback_target and RollbackManager is not None:
            try:
                RollbackManager().rollback(rollback_target, requesting_bot=self.bot_name)
                rollback_ok = True
            except Exception as exc:
                rollback_error = str(exc)
                self.logger.critical(
                    "rollback failed during objective integrity breach handling",
                    exc_info=True,
                )
        elif rollback_target and RollbackManager is None:
            rollback_error = "rollback_manager_unavailable"
        else:
            rollback_error = "no_patch_available"

        restore_ok = True
        restore_method = "not_required"
        restore_errors: list[str] = []
        if changed_files:
            restore_ok, restore_method, restore_errors = self._restore_objective_files(
                changed_files=changed_files,
                checkpoint=(objective_checkpoint or getattr(self, "_active_objective_checkpoint", None)),
            )

        telemetry_payload.update(
            {
                "rollback_attempted": rollback_attempted,
                "rollback_ok": rollback_ok,
                "rollback_error": rollback_error,
                "restoration_attempted": bool(changed_files),
                "restoration_ok": restore_ok,
                "restoration_method": restore_method,
                "restoration_errors": restore_errors,
            }
        )
        self._set_self_coding_disabled_state("objective_integrity_breach", details=details)
        if self.event_bus:
            try:
                self.event_bus.publish(
                    "self_coding:objective_integrity_breach",
                    telemetry_payload,
                )
                self.event_bus.publish(
                    "self_coding:objective_integrity_trip",
                    telemetry_payload,
                )
                self.event_bus.publish(
                    "self_coding:objective_circuit_breaker_trip",
                    telemetry_payload,
                )
                self.event_bus.publish(
                    "self_coding:objective_file_mutation_restoration",
                    telemetry_payload,
                )
            except Exception:
                self.logger.exception("failed to publish objective integrity breach event")
        if rollback_attempted and not rollback_ok and self.event_bus:
            try:
                self.event_bus.publish(
                    "self_coding:objective_integrity_breach_rollback_failed",
                    telemetry_payload,
                )
            except Exception:
                self.logger.exception(
                    "failed to publish objective integrity rollback failure telemetry"
                )
        self._active_objective_checkpoint = None
        raise RuntimeError("objective integrity breach triggered self-coding halt")

    def _objective_checkpoint_paths(self) -> list[str]:
        guard = getattr(self, "objective_guard", None)
        if guard is None or not getattr(guard, "enabled", True):
            return []
        paths: list[str] = []
        for spec in getattr(guard, "hash_specs", ()):
            if getattr(spec, "prefix", False):
                continue
            normalized = str(getattr(spec, "normalized", "")).strip()
            if normalized:
                paths.append(normalized)
        return sorted(set(paths))

    def _capture_objective_checkpoint(self, *, stage: str) -> Dict[str, Any]:
        guard = getattr(self, "objective_guard", None)
        repo_root = Path.cwd().resolve() if guard is None else guard.repo_root
        files: Dict[str, Dict[str, Any]] = {}
        tracked_paths = self._objective_checkpoint_paths()
        for rel in tracked_paths:
            target = (repo_root / rel).resolve()
            exists = target.exists() and target.is_file()
            content: bytes | None = None
            sha256: str | None = None
            if exists:
                content = target.read_bytes()
                sha256 = hashlib.sha256(content).hexdigest()
            files[rel] = {
                "exists": exists,
                "sha256": sha256,
                "content": content,
            }
        git_index_state: Dict[str, str] = {}
        if tracked_paths:
            try:
                output = subprocess.check_output(
                    ["git", "-C", str(repo_root), "ls-files", "-s", "--", *tracked_paths],
                    text=True,
                )
                for line in output.splitlines():
                    if not line.strip() or "\t" not in line:
                        continue
                    meta, rel = line.split("\t", 1)
                    git_index_state[rel.strip()] = meta.strip()
            except Exception:
                pass
        return {
            "stage": stage,
            "captured_at": datetime.now(timezone.utc).isoformat(),
            "files": files,
            "git_index_state": git_index_state,
        }

    def _changed_objective_files_from_checkpoint(
        self,
        *,
        before: Dict[str, Any],
        after: Dict[str, Any],
    ) -> list[str]:
        changed: list[str] = []
        before_files = before.get("files", {}) if isinstance(before, dict) else {}
        after_files = after.get("files", {}) if isinstance(after, dict) else {}
        candidate_paths = sorted(set(before_files) | set(after_files))
        before_index = before.get("git_index_state", {}) if isinstance(before, dict) else {}
        after_index = after.get("git_index_state", {}) if isinstance(after, dict) else {}
        for rel in candidate_paths:
            pre = before_files.get(rel, {})
            post = after_files.get(rel, {})
            if pre.get("exists") != post.get("exists") or pre.get("sha256") != post.get("sha256"):
                changed.append(rel)
                continue
            if before_index.get(rel) != after_index.get(rel):
                changed.append(rel)
        return changed

    def _restore_objective_files(
        self,
        *,
        changed_files: list[str],
        checkpoint: Dict[str, Any] | None,
    ) -> tuple[bool, str, list[str]]:
        if not changed_files:
            return True, "not_required", []
        repo_root = getattr(getattr(self, "objective_guard", None), "repo_root", Path.cwd().resolve())
        errors: list[str] = []
        used_baseline_snapshot = False
        used_git_checkout = False
        files_snapshot = checkpoint.get("files", {}) if isinstance(checkpoint, dict) else {}
        for rel in changed_files:
            restored = False
            snap = files_snapshot.get(rel, {}) if isinstance(files_snapshot, dict) else {}
            target = (repo_root / rel).resolve()
            try:
                if snap:
                    used_baseline_snapshot = True
                    if snap.get("exists") and isinstance(snap.get("content"), (bytes, bytearray)):
                        target.parent.mkdir(parents=True, exist_ok=True)
                        target.write_bytes(bytes(snap["content"]))
                        restored = True
                    elif not snap.get("exists"):
                        if target.exists():
                            target.unlink()
                        restored = True
            except Exception as exc:
                errors.append(f"{rel}:baseline_restore_failed:{exc}")
            if restored:
                continue
            try:
                subprocess.run(
                    ["git", "-C", str(repo_root), "checkout", "--", rel],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                used_git_checkout = True
                restored = True
            except Exception as exc:
                errors.append(f"{rel}:git_checkout_failed:{exc}")
            if not restored:
                continue
        if errors:
            method = "baseline_snapshot+git_checkout" if used_baseline_snapshot and used_git_checkout else (
                "baseline_snapshot" if used_baseline_snapshot else "git_checkout"
            )
            return False, method, errors
        if used_baseline_snapshot and used_git_checkout:
            method = "baseline_snapshot+git_checkout"
        elif used_baseline_snapshot:
            method = "baseline_snapshot"
        elif used_git_checkout:
            method = "git_checkout"
        else:
            method = "none"
        return True, method, []

    # ------------------------------------------------------------------
    def run_patch(
        self,
        path: Path,
        description: str,
        energy: int = 1,
        *,
        provenance_token: str,
        context_meta: Dict[str, Any] | None = None,
        max_attempts: int = 3,
        confidence_threshold: float = 0.5,
        review_branch: str | None = None,
        auto_merge: bool = False,
        backend: str = "venv",
        clone_command: list[str] | None = None,
        manual_approval_token: str | None = None,
    ) -> AutomationResult:
        """Patch *path* then deploy using the automation pipeline.

        ``max_attempts`` controls how many times the patch is retried when tests
        fail.  Context will be rebuilt for each retry excluding tags extracted
        from the failing traceback.  After a successful patch the change is
        committed in a sandbox clone, pushed to ``review_branch`` and merged
        into ``main`` when ``auto_merge`` is ``True`` and the confidence score
        exceeds ``confidence_threshold``.  ``backend`` selects the test
        execution environment; ``"venv"`` uses a virtual environment while
        ``"docker"`` runs tests inside a Docker container. ``clone_command``
        customises the VCS command used to clone the repository. A new
        :class:`ContextBuilder` is created for each attempt.
        """
        self.validate_provenance(provenance_token)
        self._ensure_self_coding_active()
        objective_checkpoint = self._capture_objective_checkpoint(stage="run_patch:pre_mutation")
        self._active_objective_checkpoint = objective_checkpoint
        try:
            self._enforce_objective_guard(path)
        except ObjectiveGuardViolation as exc:
            if getattr(exc, "reason", "") == "objective_integrity_breach":
                self._handle_objective_integrity_breach(
                    violation=exc,
                    path=path,
                    stage="pre_patch_guard_check",
                    objective_checkpoint=objective_checkpoint,
                )
            raise
        self.refresh_quick_fix_context()
        if self.approval_policy:
            target_classification = "standard"
            classify = getattr(self.approval_policy, "classify_target", None)
            if callable(classify):
                try:
                    target_classification = str(classify(path))
                except Exception:
                    target_classification = "standard"
            approval_required = target_classification == "objective_adjacent"
            approval_event = {
                "bot": self.bot_name,
                "path": path_for_prompt(path),
                "classification": target_classification,
                "approval_required": approval_required,
            }
            if approval_required:
                self.logger.info(
                    "manual approval required before patching objective-adjacent target %s",
                    path_for_prompt(path),
                    extra={"event": "self_coding_approval_required", **approval_event},
                )
                if self.event_bus:
                    try:
                        self.event_bus.publish("self_coding:approval_required", approval_event)
                    except Exception:
                        self.logger.exception("failed to publish approval_required event")
            approved = self.approval_policy.approve(
                path,
                manual_approval_token=manual_approval_token,
            )
            if not approved:
                decision = getattr(self.approval_policy, "last_decision", {}) or {}
                reason_codes = tuple(decision.get("reason_codes", ()) or ())
                primary_reason = reason_codes[0] if reason_codes else "approval_denied"
                deny_payload = {
                    **approval_event,
                    "outcome": "denied",
                    "reason_codes": reason_codes,
                    "reason_code": primary_reason,
                    "approval_source": decision.get("approval_source"),
                }
                self.logger.warning(
                    "patch approval denied for %s",
                    path_for_prompt(path),
                    extra={"event": "self_coding_approval_denied", **deny_payload},
                )
                if self.event_bus:
                    try:
                        self.event_bus.publish("self_coding:approval_denied", deny_payload)
                        if primary_reason == PatchApprovalPolicy.REASON_MANUAL_APPROVAL_MISSING:
                            self.event_bus.publish("self_coding:manual_approval_required", deny_payload)
                    except Exception:
                        self.logger.exception("failed to publish approval_denied event")
                raise RuntimeError("patch approval failed")
            approved_payload = {
                **approval_event,
                "outcome": "approved",
                "manual_approval": approval_required,
            }
            self.logger.info(
                "patch approval granted for %s",
                path_for_prompt(path),
                extra={"event": "self_coding_approval_approved", **approved_payload},
            )
            if self.event_bus:
                try:
                    self.event_bus.publish("self_coding:approval_approved", approved_payload)
                except Exception:
                    self.logger.exception("failed to publish approval_approved event")
        if self.data_bot:
            self._refresh_thresholds()
        roi = self.data_bot.roi(self.bot_name) if self.data_bot else 0.0
        errors = self.data_bot.average_errors(self.bot_name) if self.data_bot else 0.0
        failures = (
            self.data_bot.average_test_failures(self.bot_name) if self.data_bot else 0.0
        )
        if self.data_bot and not self.data_bot.check_degradation(
            self.bot_name, roi, errors, failures
        ):
            self.logger.info("ROI and error thresholds not met; skipping patch")
            return _automation_result(None, None)
        before_roi = roi
        err_before = errors
        repo_root = Path.cwd().resolve()
        result: AutomationResult | None = None
        after_roi = before_roi
        roi_delta = 0.0
        with tempfile.TemporaryDirectory() as tmp:
            cmd = (clone_command or ["git", "clone"]) + [str(repo_root), tmp]
            subprocess.run(cmd, check=True)
            clone_root = resolve_path(tmp)
            cloned_path = clone_root / path.resolve().relative_to(repo_root)
            prompt_path = path_for_prompt(path)
            attempt = 0
            patch_id: int | None = None
            commit_hash: str | None = None
            reverted = False
            ctx_meta = context_meta or {}
            clayer = self.engine.cognition_layer
            if clayer is None:
                raise AttributeError(
                    "engine.cognition_layer must provide a context_builder",
                )
            desc = description
            last_fp: FailureFingerprint | None = None
            target_region: TargetRegion | None = None
            func_region: TargetRegion | None = None
            tracker = PatchAttemptTracker(self.logger)

            def _coverage_ratio(output: str, success: bool) -> float:
                try:
                    passed_match = re.search(r"(\d+)\s+passed", output)
                    failed_match = re.search(r"(\d+)\s+failed", output)
                    passed = int(passed_match.group(1)) if passed_match else 0
                    failed = int(failed_match.group(1)) if failed_match else 0
                    total = passed + failed
                    return passed / total if total else (1.0 if success else 0.0)
                except Exception:
                    return 1.0 if success else 0.0

            def _failed_tests(output: str) -> int:
                try:
                    m = re.search(r"(\d+)\s+failed", output)
                    return int(m.group(1)) if m else 0
                except Exception:
                    return 0

            def _tests_run(output: str) -> int:
                try:
                    passed_match = re.search(r"(\d+)\s+passed", output)
                    failed_match = re.search(r"(\d+)\s+failed", output)
                    passed = int(passed_match.group(1)) if passed_match else 0
                    failed = int(failed_match.group(1)) if failed_match else 0
                    return passed + failed
                except Exception:
                    return 0

            def _run(repo: Path, changed: Path | None) -> TestHarnessResult:
                try:
                    res = run_tests(repo, changed, backend=backend)
                except TypeError:
                    res = run_tests(repo, changed)
                if isinstance(res, list):
                    return res[0]
                return res

            baseline = _run(clone_root, cloned_path)
            if (
                self.data_bot
                and hasattr(self.data_bot, "record_test_failure")
                and not baseline.success
            ):
                try:
                    self.data_bot.record_test_failure(
                        self.bot_name, _failed_tests(baseline.stdout)
                    )
                except Exception:
                    self.logger.exception("failed to record baseline test failures")
            coverage_before = _coverage_ratio(baseline.stdout, baseline.success)
            runtime_before = baseline.duration
            coverage_after = coverage_before
            runtime_after = runtime_before

            while attempt < max_attempts:
                attempt += 1
                self._run_integrity_check_hook(
                    stage="run_patch:pre_cycle",
                    path=path,
                )
                self.logger.info("patch attempt %s", attempt)
                # Create a fresh ContextBuilder for each attempt so validation
                # always runs on a clean context.
                builder = create_context_builder()
                try:
                    ensure_fresh_weights(builder)
                except Exception as exc:
                    raise RuntimeError(
                        "failed to refresh context builder weights"
                    ) from exc
                clayer.context_builder = builder
                try:
                    self.engine.context_builder = builder
                except Exception:
                    self.logger.exception("failed to refresh engine context builder")
                try:
                    self._ensure_quick_fix_engine(builder)
                except Exception as exc:
                    if self.event_bus:
                        try:
                            self.event_bus.publish(
                                "bot:patch_failed",
                                {"bot": self.bot_name, "reason": str(exc)},
                            )
                        except Exception:  # pragma: no cover - best effort
                            self.logger.exception(
                                "failed to publish patch_failed event",
                            )
                    if QuickFixEngine is None or self.quick_fix is None:
                        raise ImportError(
                            "QuickFixEngine is required but not installed",
                        ) from exc
                    raise RuntimeError(
                        "QuickFixEngine validation unavailable",
                    ) from exc
                provisional_fp: FailureFingerprint | None = None
                if self.failure_store:
                    try:
                        latest_fp: FailureFingerprint | None = None
                        for fp in getattr(self.failure_store, "_cache", {}).values():
                            if fp.filename != prompt_path:
                                continue
                            if latest_fp is None or fp.timestamp > latest_fp.timestamp:
                                latest_fp = fp
                        if latest_fp is not None:
                            provisional_fp = FailureFingerprint.from_failure(
                                prompt_path,
                                getattr(
                                    latest_fp,
                                    "function_name",
                                    getattr(latest_fp, "function", ""),
                                ),
                                latest_fp.stack_trace,
                                latest_fp.error_message,
                                desc,
                            )
                        else:
                            diff = subprocess.run(
                                ["git", "diff", "--unified=0", str(path)],
                                capture_output=True,
                                text=True,
                                check=False,
                            ).stdout
                            provisional_fp = FailureFingerprint.from_failure(
                                prompt_path,
                                "",
                                diff,
                                "",
                                desc,
                            )
                        desc, skip, best, matches, _ = check_similarity_and_warn(
                            provisional_fp,
                            self.failure_store,
                            self.skip_similarity or 0.95,
                            desc,
                        )
                    except Exception:
                        matches = []
                        best = 0.0
                        skip = False
                    if matches:
                        action = "abort" if skip else "warn"
                        self.logger.info(
                            "failure fingerprint decision",
                            extra={"action": action, "similarity": best},
                        )
                        if skip:
                            details = {
                                "fingerprint_hash": getattr(provisional_fp, "hash", ""),
                                "similarity": best,
                                "cluster_id": (
                                    getattr(matches[0], "cluster_id", None)
                                    if matches
                                    else None
                                ),
                                "reason": "retry_skipped_due_to_similarity",
                            }
                            audit = getattr(self.engine, "audit_trail", None)
                            if audit:
                                try:
                                    audit.record(details)
                                except Exception:
                                    self.logger.exception("audit trail logging failed")
                            pdb = getattr(self.engine, "patch_db", None)
                            if pdb:
                                try:
                                    conn = pdb.router.get_connection("patch_history")
                                    conn.execute(
                                        (
                                            "INSERT INTO patch_history("
                                            "filename, description, outcome"
                                            ") VALUES(?,?,?)"
                                        ),
                                        (
                                            prompt_path,
                                            json.dumps(details),
                                            "retry_skipped",
                                        ),
                                    )
                                    conn.commit()
                                except Exception:
                                    self.logger.exception(
                                        "failed to record retry status"
                                    )
                            raise RuntimeError("similar failure detected")
                if last_fp and self.failure_store:
                    threshold = getattr(
                        self.engine, "failure_similarity_threshold", None
                    )
                    if threshold is None and self.failure_store is not None:
                        try:
                            threshold = self.failure_store.adaptive_threshold()
                        except Exception:
                            threshold = None
                    if threshold is None:
                        threshold = self.skip_similarity or 0.95
                    if self.skip_similarity is not None:
                        threshold = max(threshold, self.skip_similarity)
                    try:
                        desc, skip, best, matches, _ = check_similarity_and_warn(
                            last_fp,
                            self.failure_store,
                            threshold,
                            desc,
                        )
                    except Exception:
                        matches = []
                        best = 0.0
                        skip = False
                    action = "skip" if skip else "warning"
                    if matches:
                        self.logger.info(
                            "failure fingerprint decision",
                            extra={"action": action, "similarity": best},
                        )
                    if skip:
                        details = {
                            "fingerprint_hash": getattr(last_fp, "hash", ""),
                            "similarity": best,
                            "cluster_id": (
                                getattr(matches[0], "cluster_id", None)
                                if matches
                                else None
                            ),
                            "reason": "retry_skipped_due_to_similarity",
                        }
                        audit = getattr(self.engine, "audit_trail", None)
                        if audit:
                            try:
                                audit.record(details)
                            except Exception:
                                self.logger.exception("audit trail logging failed")
                        pdb = getattr(self.engine, "patch_db", None)
                        if pdb:
                            try:
                                conn = pdb.router.get_connection("patch_history")
                                conn.execute(
                                    (
                                        "INSERT INTO patch_history("
                                        "filename, description, outcome"
                                        ") VALUES(?,?,?)"
                                    ),
                                    (
                                        prompt_path,
                                        json.dumps(details),
                                        "retry_skipped",
                                    ),
                                )
                                conn.commit()
                            except Exception:
                                self.logger.exception("failed to record retry status")
                        raise RuntimeError("similar failure detected")
                if target_region is not None:
                    target_region_cls = _get_target_region_cls()
                    func_region = func_region or target_region_cls(
                        file=target_region.file,
                        start_line=0,
                        end_line=0,
                        function=target_region.function,
                    )
                    level, patch_region = tracker.level_for(target_region, func_region)
                else:
                    level, patch_region = "module", None
                ctx_meta["escalation_level"] = level
                if patch_region is not None:
                    ctx_meta["target_region"] = asdict(patch_region)
                else:
                    ctx_meta.pop("target_region", None)

                module_path = str(cloned_path)
                module_name = path_for_prompt(cloned_path)
                predicted_gain = 0.0
                if self.data_bot and hasattr(self.data_bot, "forecast_roi_drop"):
                    try:
                        predicted_gain = float(self.data_bot.forecast_roi_drop())
                    except Exception:
                        self.logger.exception("roi prediction failed")
                else:
                    evo = getattr(self.pipeline, "forecast_roi_drop", None)
                    if evo:
                        try:
                            predicted_gain = float(evo())
                        except Exception:
                            self.logger.exception("roi prediction failed")
                if predicted_gain < self.roi_drop_threshold:
                    self.logger.info(
                        "patch_skip_low_roi_prediction",
                        extra={"bot": self.bot_name, "predicted_gain": predicted_gain},
                    )
                    if self.event_bus:
                        try:
                            self.event_bus.publish(
                                "bot:patch_skipped",
                                {"bot": self.bot_name, "reason": "roi_prediction"},
                            )
                        except Exception:
                            self.logger.exception(
                                "failed to publish patch_skipped event",
                            )
                    return _automation_result(None, None)
                if self.quick_fix is None:
                    raise RuntimeError("QuickFixEngine validation unavailable")
                try:
                    valid, _flags = self.quick_fix.validate_patch(
                        module_path,
                        desc,
                        repo_root=clone_root,
                        provenance_token=provenance_token,
                    )
                except ObjectiveGuardViolation as exc:
                    if getattr(exc, "reason", "") == "objective_integrity_breach":
                        self._handle_objective_integrity_breach(
                            violation=exc,
                            path=path,
                            stage="quick_fix_validate_patch",
                            objective_checkpoint=objective_checkpoint,
                        )
                    raise
                except Exception as exc:
                    try:
                        RollbackManager().rollback(
                            "pre_commit_validation",
                            requesting_bot=self.bot_name,
                        )
                    except Exception:
                        self.logger.exception("rollback failed")
                    MutationLogger.log_mutation(
                        change="quick_fix_validation_error",
                        reason=description,
                        trigger=module_name,
                        workflow_id=0,
                        parent_id=self._last_event_id,
                    )
                    raise RuntimeError("quick fix validation failed") from exc
                if not valid:
                    try:
                        RollbackManager().rollback(
                            "quick_fix_validation_failed",
                            requesting_bot=self.bot_name,
                        )
                    except Exception:
                        self.logger.exception("rollback failed")
                    MutationLogger.log_mutation(
                        change="quick_fix_validation_failed",
                        reason=description,
                        trigger=module_name,
                        workflow_id=0,
                        parent_id=self._last_event_id,
                    )
                    raise RuntimeError("quick fix validation failed")
                try:
                    passed, patch_id, flags = self.quick_fix.apply_validated_patch(
                        module_path,
                        desc,
                        ctx_meta,
                        provenance_token=provenance_token,
                    )
                except ObjectiveGuardViolation as exc:
                    if getattr(exc, "reason", "") == "objective_integrity_breach":
                        self._handle_objective_integrity_breach(
                            violation=exc,
                            path=path,
                            stage="quick_fix_apply_patch",
                            objective_checkpoint=objective_checkpoint,
                        )
                    raise
                except Exception as exc:
                    if self.event_bus:
                        try:
                            self.event_bus.publish(
                                "bot:patch_failed",
                                {
                                    "bot": self.bot_name,
                                    "stage": "generate_patch",
                                    "reason": str(exc),
                                },
                            )
                        except Exception:  # pragma: no cover - best effort
                            self.logger.exception(
                                "failed to publish patch_failed event",
                            )
                    raise RuntimeError("quick fix generation failed") from exc
                flags = list(flags or [])
                self._run_integrity_check_hook(
                    stage="run_patch:post_apply",
                    path=path,
                )
                post_apply_checkpoint = self._capture_objective_checkpoint(
                    stage="run_patch:post_mutation",
                )
                changed_objective_files = self._changed_objective_files_from_checkpoint(
                    before=objective_checkpoint,
                    after=post_apply_checkpoint,
                )
                if changed_objective_files:
                    raise ObjectiveGuardViolation(
                        "objective_integrity_breach",
                        details={
                            "changed_files": changed_objective_files,
                            "checkpoint_stage": "run_patch:post_mutation",
                        },
                    )
                self._last_risk_flags = flags
                ctx_meta["risk_flags"] = flags
                reverted = not passed
                if not passed:
                    if target_region is not None and func_region is not None:
                        tracker.record_failure(level, target_region, func_region)
                    raise RuntimeError("quick fix validation failed")
                harness_result: TestHarnessResult = _run(clone_root, cloned_path)
                if (
                    self.data_bot
                    and hasattr(self.data_bot, "record_test_failure")
                    and not harness_result.success
                ):
                    try:
                        self.data_bot.record_test_failure(
                            self.bot_name, _failed_tests(harness_result.stdout)
                        )
                    except Exception:
                        self.logger.exception("failed to record test failures")
                if harness_result.success:
                    coverage_after = _coverage_ratio(
                        harness_result.stdout, harness_result.success
                    )
                    runtime_after = harness_result.duration
                    if target_region is not None:
                        tracker.reset(target_region)
                    break

                if attempt >= max_attempts:
                    raise RuntimeError("patch tests failed")

                failure = harness_result.failure or {}
                trace = (
                    failure.get("stack")
                    or harness_result.stderr
                    or harness_result.stdout
                    or ""
                )
                if self._failure_cache.seen(trace):
                    raise RuntimeError("patch tests failed")
                if not failure:
                    failure = ErrorParser.parse_failure(trace)
                tag = failure.get("strategy_tag", "")
                tags = [tag] if tag else []
                self.logger.error(
                    "patch tests failed",
                    extra={
                        "stdout": harness_result.stdout,
                        "stderr": harness_result.stderr,
                        "tags": tags,
                    },
                )
                self._failure_cache.add(ErrorReport(trace=trace, tags=tags))
                try:
                    record_failed_tags(list(tags))
                except Exception:  # pragma: no cover - best effort
                    self.logger.exception("failed to record failed tags")
                if getattr(self.engine, "patch_suggestion_db", None):
                    for tag in tags:
                        try:
                            self.engine.patch_suggestion_db.add_failed_strategy(tag)
                        except Exception:  # pragma: no cover - best effort
                            self.logger.exception("failed to store failed strategy tag")

                parsed = ErrorParser.parse(trace)
                stack_trace = parsed.get("trace", trace)
                region_obj = parsed.get("target_region")
                if target_region is None and region_obj is not None:
                    try:
                        target_region_cls = _get_target_region_cls()
                        target_region = target_region_cls(
                            file=getattr(
                                region_obj,
                                "file",
                                getattr(region_obj, "filename", ""),
                            ),
                            start_line=getattr(region_obj, "start_line", 0),
                            end_line=getattr(region_obj, "end_line", 0),
                            function=getattr(
                                region_obj,
                                "function",
                                getattr(region_obj, "func_name", ""),
                            ),
                        )
                    except Exception:
                        target_region = None
                function_name = ""
                error_msg = ""
                m = re.findall(r'File "[^"]+", line \d+, in ([^\n]+)', stack_trace)
                if m:
                    function_name = m[-1]
                m_err = re.findall(r"([\w.]+(?:Error|Exception):.*)", stack_trace)
                if m_err:
                    error_msg = m_err[-1]
                fingerprint = FailureFingerprint.from_failure(
                    prompt_path,
                    function_name,
                    stack_trace,
                    error_msg,
                    self.engine.last_prompt_text,
                )
                last_fp = fingerprint
                record_failure(fingerprint, self.failure_store)

                if target_region is not None and func_region is None:
                    target_region_cls = _get_target_region_cls()
                    func_region = target_region_cls(
                        file=target_region.file,
                        start_line=0,
                        end_line=0,
                        function=target_region.function,
                    )
                if target_region is not None and func_region is not None:
                    tracker.record_failure(level, target_region, func_region)
                    level, patch_region = tracker.level_for(target_region, func_region)
                else:
                    level, patch_region = "module", None

                self.logger.info(
                    "rebuilding context",
                    extra={"tags": tags, "attempt": attempt},
                )
                if not builder or not tags:
                    raise RuntimeError("patch tests failed")
                try:
                    ctx_result = builder.query(
                        desc,
                        exclude_tags=tags,
                        include_vectors=True,
                        return_metadata=True,
                    )
                    if isinstance(ctx_result, (list, tuple)):
                        ctx = ctx_result[0]
                        sid = ctx_result[1] if len(ctx_result) > 1 else ""
                        vectors = ctx_result[2] if len(ctx_result) > 2 else []
                        meta = ctx_result[3] if len(ctx_result) > 3 else {}
                    else:
                        ctx, sid, vectors, meta = ctx_result, "", [], {}
                    ctx_meta = {
                        "retrieval_context": ctx,
                        "retrieval_session_id": sid,
                        "escalation_level": level,
                    }
                    if vectors:
                        ctx_meta["vectors"] = vectors
                    if meta:
                        ctx_meta["retrieval_metadata"] = meta
                    if patch_region is not None:
                        ctx_meta["target_region"] = asdict(patch_region)
                except Exception as exc:  # pragma: no cover - best effort
                    self.logger.error("context rebuild failed: %s", exc)
                    raise RuntimeError("patch tests failed")

                # failure fingerprint logged above

            description = desc
            path.write_text(cloned_path.read_text(encoding="utf-8"), encoding="utf-8")
            branch_name = review_branch or f"review/{patch_id}"
            try:
                subprocess.run(
                    ["git", "config", "user.email", "bot@example.com"],
                    check=True,
                    cwd=str(clone_root),
                )
                subprocess.run(
                    ["git", "config", "user.name", "bot"],
                    check=True,
                    cwd=str(clone_root),
                )
                subprocess.run(
                    ["git", "checkout", "-b", branch_name],
                    check=True,
                    cwd=str(clone_root),
                )
                subprocess.run(["git", "add", "-A"], check=True, cwd=str(clone_root))
                prov_path = Path(tmp) / "patch_provenance.json"
                try:
                    prov_path.write_text(json.dumps({"patch_id": patch_id}))
                except Exception:
                    self.logger.error("failed to write provenance file")
                env = os.environ.copy()
                env["PATCH_PROVENANCE_FILE"] = str(prov_path)
                subprocess.run(
                    ["git", "commit", "-m", f"patch {patch_id}: {description}"],
                    check=True,
                    cwd=str(clone_root),
                    env=env,
                )
                try:
                    prov_path.unlink()
                except Exception:
                    self.logger.exception(
                        "failed to remove patch provenance file %s", prov_path
                    )
                commit_hash = (
                    subprocess.check_output(
                        ["git", "rev-parse", "HEAD"], cwd=str(clone_root)
                    )
                    .decode()
                    .strip()
                )
                if not commit_hash:
                    raise RuntimeError("failed to retrieve commit hash")
                self._last_patch_id = patch_id
                self._last_commit_hash = commit_hash
            except Exception as exc:  # pragma: no cover - best effort
                self.logger.error("git commit failed: %s", exc)
                try:
                    RollbackManager().rollback(
                        str(patch_id), requesting_bot=self.bot_name
                    )
                except Exception:
                    self.logger.exception("rollback failed")
                raise

            result = self.pipeline.run(self.bot_name, energy=energy)
            after_roi = self.data_bot.roi(self.bot_name) if self.data_bot else 0.0
            err_after = (
                self.data_bot.average_errors(self.bot_name) if self.data_bot else 0.0
            )
            roi_delta = after_roi - before_roi
            err_delta = err_after - err_before
            coverage_delta = coverage_after - coverage_before
            runtime_improvement = runtime_before - runtime_after
            self.logger.info(
                "roi metrics",
                extra={
                    "coverage_before": coverage_before,
                    "coverage_after": coverage_after,
                    "coverage_delta": coverage_delta,
                    "runtime_before": runtime_before,
                    "runtime_after": runtime_after,
                    "runtime_improvement": runtime_improvement,
                },
            )
            if self.data_bot:
                try:
                    self.data_bot.collect(
                        self.bot_name,
                        revenue=after_roi,
                        errors=int(err_after),
                        tests_failed=_failed_tests(harness_result.stdout),
                        tests_run=_tests_run(harness_result.stdout),
                    )
                except Exception:
                    self.logger.exception("data_bot.collect failed")
            patch_logger = getattr(self.engine, "patch_logger", None)
            vectors = ctx_meta.get("vectors", []) if ctx_meta else []
            retrieval_metadata = (
                ctx_meta.get("retrieval_metadata", {}) if ctx_meta else {}
            )
            if patch_logger is not None:
                try:
                    patch_logger.track_contributors(
                        vectors,
                        True,
                        patch_id=str(patch_id or ""),
                        contribution=roi_delta,
                        retrieval_metadata=retrieval_metadata,
                    )
                except Exception:
                    self.logger.exception("track_contributors failed")
            if commit_hash and patch_id:
                try:
                    record_patch_metadata(
                        int(patch_id),
                        {
                            "commit": commit_hash,
                            "vectors": list(vectors),
                        },
                    )
                except Exception:
                    self.logger.exception("failed to record patch metadata")
            session_id = ""
            if ctx_meta:
                session_id = ctx_meta.get("retrieval_session_id", "")
            clayer = getattr(self.engine, "cognition_layer", None)
            if clayer is not None and session_id:
                try:
                    clayer.record_patch_outcome(
                        session_id, True, contribution=roi_delta
                    )
                except Exception:
                    self.logger.exception("failed to record patch outcome")
            if self.quick_fix is None:
                raise RuntimeError("QuickFixEngine validation unavailable")
            try:
                _src = path.read_text(encoding="utf-8")
                valid_post, _flags_post = self.quick_fix.validate_patch(
                    str(path),
                    description,
                    provenance_token=provenance_token,
                )
                path.write_text(_src, encoding="utf-8")
                if not valid_post:
                    raise RuntimeError("quick fix validation failed")
            except Exception as exc:
                raise RuntimeError("QuickFixEngine validation unavailable") from exc
            conf = 1.0
            if result is not None and getattr(result, "roi", None) is not None:
                conf = getattr(result.roi, "confidence", None)  # type: ignore[attr-defined]
                if conf is None:
                    risk = getattr(result.roi, "risk", None)  # type: ignore[attr-defined]
                    if risk is not None:
                        try:
                            conf = 1.0 - float(risk)
                        except Exception:
                            conf = 1.0
                if conf is None:
                    conf = 1.0
            patch_db = getattr(self.engine, "patch_db", None)
            try:
                subprocess.run(
                    ["git", "push", "origin", branch_name],
                    check=True,
                    cwd=str(clone_root),
                )
                MutationLogger.log_mutation(
                    change="patch_branch",
                    reason=description,
                    trigger=prompt_path,
                    performance=0.0,
                    workflow_id=0,
                    parent_id=self._last_event_id,
                )
                if patch_db is not None:
                    try:
                        patch_db.log_branch_event(str(patch_id), branch_name, "pushed")
                    except Exception:
                        self.logger.exception("failed to log branch event")
            except Exception as exc:  # pragma: no cover - best effort
                self.logger.error("git push failed: %s", exc)
                try:
                    RollbackManager().rollback(
                        str(patch_id), requesting_bot=self.bot_name
                    )
                except Exception:
                    self.logger.exception("rollback failed")
                raise

            conf_avg = self.baseline_tracker.get("confidence")
            conf_std = self.baseline_tracker.std("confidence")
            dynamic_conf = max(confidence_threshold, conf_avg + conf_std)
            if auto_merge and conf >= dynamic_conf:
                try:
                    subprocess.run(
                        ["git", "checkout", "main"],
                        check=True,
                        cwd=str(clone_root),
                    )
                    subprocess.run(
                        ["git", "merge", "--no-ff", branch_name],
                        check=True,
                        cwd=str(clone_root),
                    )
                    subprocess.run(
                        ["git", "push", "origin", "main"],
                        check=True,
                        cwd=str(clone_root),
                    )
                    MutationLogger.log_mutation(
                        change="patch_merge",
                        reason=description,
                        trigger=prompt_path,
                        performance=roi_delta,
                        workflow_id=0,
                        parent_id=self._last_event_id,
                    )
                    if patch_db is not None:
                        try:
                            patch_db.log_branch_event(str(patch_id), "main", "merged")
                        except Exception:
                            self.logger.exception("failed to log merge event")
                except Exception as exc:  # pragma: no cover - best effort
                    self.logger.error("merge to main failed: %s", exc)
                    try:
                        RollbackManager().rollback(
                            str(patch_id), requesting_bot=self.bot_name
                        )
                    except Exception:
                        self.logger.exception("rollback failed")
            self.baseline_tracker.update(confidence=conf)
        event_id = MutationLogger.log_mutation(
            change=f"self_coding_patch_{patch_id}",
            reason=description,
            trigger=prompt_path,
            performance=roi_delta,
            workflow_id=0,
            parent_id=self._last_event_id,
        )
        MutationLogger.record_mutation_outcome(
            event_id,
            after_metric=after_roi,
            roi=after_roi,
            performance=roi_delta,
        )
        self._last_event_id = event_id
        self._last_patch_id = patch_id
        self._last_commit_hash = commit_hash
        if self.data_bot:
            try:
                roi_val = result.roi.roi if result.roi else 0.0
            except Exception:
                roi_val = 0.0
            patch_rate = 0.0
            patch_db = getattr(self.engine, "patch_db", None)
            if patch_db:
                try:
                    patch_rate = patch_db.success_rate()
                except Exception:
                    patch_rate = 0.0
            try:
                self.data_bot.log_evolution_cycle(
                    "self_coding",
                    before_roi,
                    after_roi,
                    roi_val,
                    0.0,
                    patch_success=patch_rate,
                    roi_delta=roi_delta,
                    patch_id=patch_id,
                    reverted=reverted,
                    reason=description,
                    trigger=prompt_path,
                    parent_event_id=self._last_event_id,
                )
            except Exception as exc:
                self.logger.exception("failed to log evolution cycle: %s", exc)
        if self.bot_registry:
            module_path = path_for_prompt(path)
            try:
                self.bot_registry.record_heartbeat(self.bot_name)
                self.bot_registry.register_interaction(self.bot_name, "patched")
                self.bot_registry.record_interaction_metadata(
                    self.bot_name,
                    "patched",
                    duration=runtime_after,
                    success=True,
                    resources=(f"hot_swap:{int(time.time())},patch_id:{patch_id}"),
                )
                self.bot_registry.register_bot(
                    self.bot_name,
                    manager=self,
                    data_bot=self.data_bot,
                    is_coding_bot=True,
                )
                self.bot_registry.record_interaction_metadata(
                    self.bot_name,
                    "evolution",
                    duration=runtime_after,
                    success=True,
                    resources=f"patch_id:{patch_id}",
                )
            except Exception:  # pragma: no cover - best effort
                self.logger.exception(
                    "failed to update bot registry",
                    extra={"bot": self.bot_name, "module_path": module_path},
                )

            prev_state: dict[str, object] | None = None
            if not commit_hash or patch_id is None:
                if self.event_bus:
                    try:
                        self.event_bus.publish(
                            "bot:update_blocked",
                            {
                                "bot": self.bot_name,
                                "path": module_path,
                                "patch_id": patch_id,
                                "commit": commit_hash,
                                "reason": "missing_provenance",
                            },
                        )
                    except Exception:  # pragma: no cover - best effort
                        self.logger.exception(
                            "failed to publish update_blocked event",
                            extra={"bot": self.bot_name},
                        )
            else:
                if self.bot_name in self.bot_registry.graph:
                    try:
                        prev_state = dict(self.bot_registry.graph.nodes[self.bot_name])
                    except Exception:  # pragma: no cover - best effort
                        prev_state = None
                try:
                    self.bot_registry.update_bot(
                        self.bot_name,
                        module_path,
                        patch_id=patch_id,
                        commit=commit_hash,
                    )
                    version = None
                    try:
                        version = self.bot_registry.graph.nodes[self.bot_name].get(
                            "version"
                        )
                    except Exception:
                        version = None
                    self.logger.info(
                        "bot registry updated",
                        extra={
                            "bot": self.bot_name,
                            "module_path": module_path,
                            "version": version,
                        },
                    )
                    try:
                        record_patch_metadata(
                            int(patch_id),
                            {"commit": commit_hash, "module": str(module_path)},
                        )
                    except Exception:  # pragma: no cover - best effort
                        self.logger.exception("failed to record patch metadata")
                except Exception:
                    self.logger.exception(
                        "failed to update bot registry",
                        extra={"bot": self.bot_name, "module_path": module_path},
                    )
                    if prev_state is not None:
                        try:
                            current = self.bot_registry.graph.nodes[self.bot_name]
                            current.clear()
                            current.update(prev_state)
                            target = getattr(self.bot_registry, "persist_path", None)
                            if target:
                                self.bot_registry.save(target)
                        except Exception:  # pragma: no cover - best effort
                            self.logger.exception(
                                "failed to revert bot registry",
                                extra={"bot": self.bot_name},
                            )
                    raise
                hot_swap_snapshot: dict[str, object] | None = None
                if self.bot_name in self.bot_registry.graph:
                    try:
                        hot_swap_snapshot = dict(
                            self.bot_registry.graph.nodes[self.bot_name]
                        )
                    except Exception:  # pragma: no cover - best effort
                        hot_swap_snapshot = None
                try:
                    self.bot_registry.hot_swap(self.bot_name, module_path)
                    self.bot_registry.health_check_bot(self.bot_name, prev_state)
                    MutationLogger.log_mutation(
                        change="hot_swap_success",
                        reason=description,
                        trigger=prompt_path,
                        performance=0.0,
                        workflow_id=0,
                        parent_id=self._last_event_id,
                    )
                except Exception:  # pragma: no cover - best effort
                    self.logger.exception(
                        "failed to hot swap bot",
                        extra={"bot": self.bot_name, "module_path": module_path},
                    )
                    if hot_swap_snapshot is not None:
                        try:
                            current = self.bot_registry.graph.nodes[self.bot_name]
                            current.clear()
                            current.update(hot_swap_snapshot)
                            target = getattr(self.bot_registry, "persist_path", None)
                            if target:
                                self.bot_registry.save(target)
                        except Exception:  # pragma: no cover - best effort
                            self.logger.exception(
                                "failed to restore bot registry",
                                extra={"bot": self.bot_name},
                            )
                    if self.event_bus:
                        try:
                            self.event_bus.publish(
                                "self_coding:hot_swap_failed",
                                {
                                    "bot": self.bot_name,
                                    "path": module_path,
                                    "patch_id": patch_id,
                                    "commit": commit_hash,
                                },
                            )
                        except Exception:  # pragma: no cover - best effort
                            self.logger.exception(
                                "failed to publish hot_swap_failed event",
                                extra={"bot": self.bot_name},
                            )
                    MutationLogger.log_mutation(
                        change="hot_swap_failed",
                        reason=description,
                        trigger=prompt_path,
                        performance=0.0,
                        workflow_id=0,
                        parent_id=self._last_event_id,
                    )
                    raise
                if self.event_bus:
                    try:
                        self.event_bus.publish(
                            "bot:updated",
                            {
                                "bot": self.bot_name,
                                "path": module_path,
                                "patch_id": patch_id,
                                "commit": commit_hash,
                            },
                        )
                    except Exception:  # pragma: no cover - best effort
                        self.logger.exception(
                            "failed to publish bot updated event",
                            extra={"bot": self.bot_name},
                        )
                target = getattr(self.bot_registry, "persist_path", None)
                if target:
                    try:
                        self.bot_registry.save(target)
                    except Exception:  # pragma: no cover - best effort
                        self.logger.exception(
                            "failed to persist bot registry",
                            extra={"path": str(target)},
                        )
        if self.event_bus:
            try:
                payload = {
                    "bot": self.bot_name,
                    "patch_id": patch_id,
                    "commit": commit_hash,
                    "path": prompt_path,
                    "description": description,
                    "roi_before": before_roi,
                    "roi_after": after_roi,
                    "roi_delta": roi_delta,
                    "errors_before": err_before,
                    "errors_after": err_after,
                    "error_delta": err_delta,
                }
                self.event_bus.publish("self_coding:patch_applied", payload)
            except Exception:  # pragma: no cover - best effort
                self.logger.exception("failed to publish patch_applied event")
        self.scan_repo()
        try:
            load_failed_tags()
        except Exception:  # pragma: no cover - best effort
            self.logger.exception("failed to refresh failed tags")
        if self.data_bot:
            # Refresh thresholds post-patch so subsequent decisions use the
            # latest configuration.
            self._refresh_thresholds()
        self._active_objective_checkpoint = None
        return result

    # ------------------------------------------------------------------
    def idle_cycle(self) -> None:
        """Poll suggestion DB and schedule queued enhancements."""
        if not self.suggestion_db:
            return
        try:
            rows = self.suggestion_db.conn.execute(
                "SELECT id, module, description FROM suggestions ORDER BY id"
            ).fetchall()
        except Exception:  # pragma: no cover - best effort
            self.logger.exception("failed to fetch suggestions")
            return
        for sid, module, description in rows:
            if getattr(self, "_self_coding_paused", False):
                self.logger.warning(
                    "self-coding paused; halting idle cycle loop",
                    extra={"bot": self.bot_name, "reason": self._self_coding_disabled_reason},
                )
                break
            path = resolve_path(module)
            prompt_module = path_for_prompt(module)
            try:
                if getattr(self.engine, "audit_trail", None):
                    try:
                        score_part, rationale = description.split(" - ", 1)
                        score = float(score_part.split("=", 1)[1])
                    except Exception:
                        score = 0.0
                        rationale = description
                    try:
                        self.engine.audit_trail.record(
                            {
                                "event": "queued_enhancement",
                                "module": prompt_module,
                                "score": round(score, 2),
                                "rationale": rationale,
                            }
                        )
                    except Exception:  # pragma: no cover - best effort
                        self.logger.exception(
                            "failed to record audit log for %s", prompt_module
                        )
                self.auto_run_patch(path, description)
            except ObjectiveGuardViolation as exc:
                if getattr(exc, "reason", "") == "objective_integrity_breach":
                    self._handle_objective_integrity_breach(
                        violation=exc,
                        path=path,
                        stage="idle_cycle",
                        objective_checkpoint=getattr(self, "_active_objective_checkpoint", None),
                    )
                raise
            except Exception:  # pragma: no cover - best effort
                self.logger.exception(
                    "failed to apply suggestion for %s", prompt_module
                )
                if getattr(self, "_self_coding_paused", False):
                    break
            finally:
                try:
                    self.suggestion_db.conn.execute(
                        "DELETE FROM suggestions WHERE id=?", (sid,)
                    )
                    self.suggestion_db.conn.commit()
                except Exception:  # pragma: no cover - best effort
                    self.logger.exception("failed to delete suggestion %s", sid)


def internalize_coding_bot(
    bot_name: str,
    engine: SelfCodingEngine,
    pipeline: ModelAutomationPipeline,
    *,
    data_bot: DataBot,
    bot_registry: BotRegistry,
    evolution_orchestrator: "EvolutionOrchestrator | None" = None,
    provenance_token: str | None = None,
    roi_threshold: float | None = None,
    error_threshold: float | None = None,
    test_failure_threshold: float | None = None,
    **manager_kwargs: Any,
) -> SelfCodingManager:
    """Wire ``bot_name`` into the self‑coding system.

    The helper constructs a :class:`SelfCodingManager`, registers ROI/error/test
    failure thresholds with :class:`BotRegistry` and ensures
    ``EvolutionOrchestrator`` reacts to ``degradation:detected`` events.

    Parameters mirror :class:`SelfCodingManager` while providing explicit
    ``roi_threshold``, ``error_threshold`` and ``test_failure_threshold`` values.
    Additional keyword arguments are forwarded to ``SelfCodingManager``.
    """
    global _LAST_INTERNALIZE_AT

    module_path: Path | None = None
    module_hint: str | None = None
    manager: SelfCodingManager | None = None
    failure_recorded = False
    logger_ref = logging.getLogger(__name__)
    node: dict[str, Any] | None = None
    added_in_flight = False
    logged_internalize_stack = False
    attempt_result = "failed"
    attempt_started = False

    def _shutdown_guard_state() -> dict[str, Any] | None:
        reasons: list[str] = []
        checks: dict[str, Any] = {}

        finalizing = bool(sys.is_finalizing())
        checks["sys_finalizing"] = finalizing
        if finalizing:
            reasons.append("sys_finalizing")

        shutdown_events = {
            "shutdown_event": manager_kwargs.get("shutdown_event"),
            "stop_event": manager_kwargs.get("stop_event"),
            "global_shutdown_event": manager_kwargs.get("global_shutdown_event"),
            "pipeline_shutdown_event": getattr(pipeline, "shutdown_event", None),
            "engine_shutdown_event": getattr(engine, "shutdown_event", None),
            "data_bot_shutdown_event": getattr(data_bot, "shutdown_event", None),
        }
        for name, event in shutdown_events.items():
            is_set = bool(getattr(event, "is_set", lambda: False)()) if event is not None else False
            checks[name] = is_set
            if is_set:
                reasons.append(name)

        loop = getattr(pipeline, "loop", None) or getattr(engine, "loop", None)
        loop_closed = bool(getattr(loop, "is_closed", lambda: False)()) if loop is not None else False
        checks["loop_closed"] = loop_closed
        if loop_closed:
            reasons.append("loop_closed")

        executor = (
            getattr(pipeline, "executor", None)
            or getattr(engine, "executor", None)
            or manager_kwargs.get("executor")
        )
        executor_shutdown = bool(getattr(executor, "_shutdown", False)) if executor is not None else False
        checks["executor_shutdown"] = executor_shutdown
        if executor_shutdown:
            reasons.append("executor_shutdown")

        if not reasons:
            return None

        return {
            "event": "internalize_skipped_shutdown",
            "bot": bot_name,
            "reasons": sorted(set(reasons)),
            "checks": checks,
            "timestamp": time.time(),
        }

    def _caller_metadata() -> Any | None:
        for key in (
            "caller_metadata",
            "caller",
            "caller_info",
            "origin",
            "request_id",
            "trace_id",
        ):
            if key in manager_kwargs:
                return manager_kwargs.get(key)
        return None

    def _stack_trace_limit() -> int | None:
        if os.getenv("SELF_CODING_VERBOSE_INTERNALIZE_STACK"):
            return None
        raw_limit = os.getenv("SELF_CODING_INTERNALIZE_STACK_LIMIT", "6")
        try:
            return max(1, int(raw_limit))
        except (TypeError, ValueError):
            return 6

    def _log_internalize_stack(reason: str) -> None:
        if not logger_ref.isEnabledFor(logging.DEBUG):
            return
        stack_entries = traceback.format_stack()
        stack_limit = _stack_trace_limit()
        if stack_limit is not None:
            stack_entries = stack_entries[-stack_limit:]
        stack_trace = "".join(stack_entries)
        call_source = None
        stack_summary = traceback.extract_stack(limit=3)
        if len(stack_summary) > 1:
            caller = stack_summary[-2]
            call_source = f"{caller.filename}:{caller.lineno}:{caller.name}"
        extra = {"bot": bot_name, "reason": reason}
        if call_source:
            extra["call_source"] = call_source
        metadata = _caller_metadata()
        if metadata is not None:
            extra["caller_metadata"] = metadata
        logger_ref.debug(
            "internalize_coding_bot stack trace (%s):\n%s",
            reason,
            stack_trace,
            extra=extra,
        )

    def _mark_last_internalized() -> None:
        if node is None:
            return
        node["last_internalized"] = time.monotonic()

    def _recent_internalization() -> bool:
        if node is None or _INTERNALIZE_REUSE_WINDOW_SECONDS <= 0:
            return False
        last_value = node.get("last_internalized")
        if not isinstance(last_value, (int, float)):
            return False
        if node.get("internalization_errors"):
            return False
        state = _INTERNALIZE_FAILURE_STATE.get(bot_name)
        if state:
            if int(state.get("count", 0) or 0) > 0:
                return False
            if float(state.get("cooldown_until", 0.0) or 0.0) > 0.0:
                return False
        elapsed = time.monotonic() - float(last_value)
        return elapsed < _INTERNALIZE_REUSE_WINDOW_SECONDS

    def _manager_healthy(candidate: Any) -> bool:
        if candidate is None:
            return False
        if getattr(candidate, "quick_fix", None) is None:
            return False
        if getattr(candidate, "event_bus", None) is None:
            return False
        return True

    def _inflight_manager_fallback() -> SelfCodingManager:
        inflight_node = bot_registry.graph.nodes.get(bot_name) if bot_registry else None
        existing_manager = None
        if inflight_node is not None:
            existing_manager = inflight_node.get("selfcoding_manager") or inflight_node.get("manager")
        if existing_manager is not None and _manager_healthy(existing_manager):
            return existing_manager
        return _cooldown_disabled_manager(bot_registry, data_bot)

    def _record_attempt_start() -> None:
        nonlocal attempt_started
        attempt_started = True
        if node is None:
            return
        node["attempt_started_at"] = time.time()
        node["attempt_finished_at"] = None
        node["attempt_result"] = "in_progress"
        node["internalization_last_step"] = "attempt started"

    def _record_attempt_finish(result: str) -> None:
        nonlocal attempt_result
        attempt_result = result
        if node is None:
            return
        node["attempt_finished_at"] = time.time()
        node["attempt_result"] = result

    if bot_registry is not None:
        node = bot_registry.graph.nodes.get(bot_name)

    _start_internalize_monitor(bot_registry)

    stale_in_flight = _consume_stale_internalize_in_flight(now=time.monotonic())
    for stale_bot, started_at in stale_in_flight:
        stale_age = max(0.0, time.monotonic() - started_at)
        logger_ref.warning(
            "forcibly cleared stale internalize in-flight lock for %s after %.1fs",
            stale_bot,
            stale_age,
            extra={
                "bot": stale_bot,
                "stale_lock_seconds": stale_age,
                "stale_lock_timeout_seconds": _INTERNALIZE_STALE_TIMEOUT_SECONDS,
            },
        )
        stale_node = None
        if bot_registry is not None:
            stale_node = bot_registry.graph.nodes.get(stale_bot)
        if stale_node is not None:
            stale_node.pop("internalization_in_progress", None)
            stale_node["internalization_last_step"] = "stale in-flight cleanup"
        _replace_internalize_lock(stale_bot)

    with _INTERNALIZE_IN_FLIGHT_LOCK:
        started_at = _INTERNALIZE_IN_FLIGHT.get(bot_name)
        if started_at is not None:
            age = max(0.0, time.monotonic() - started_at)
            hard_timeout = _INTERNALIZE_HARD_TIMEOUT_SECONDS
            if hard_timeout > 0 and age >= hard_timeout:
                _INTERNALIZE_IN_FLIGHT.pop(bot_name, None)
                _INTERNALIZE_MONITOR_LAST_LOGGED_AT.pop(bot_name, None)
                last_step = None
                started_epoch = None
                if node is not None:
                    last_step = node.get("internalization_last_step")
                    started_epoch = node.pop("internalization_in_progress", None)
                    node["internalization_last_step"] = "stale hard-timeout recovery"
                _replace_internalize_lock(bot_name)
                logger_ref.error(
                    "internalize_coding_bot hard timeout exceeded for %s; forcing recovery",
                    bot_name,
                    extra={
                        "bot": bot_name,
                        "in_flight_seconds": age,
                        "internalization_last_step": last_step,
                        "internalization_in_progress": started_epoch,
                        "in_flight_hard_timeout_seconds": hard_timeout,
                        "reason": "stale_internalization_timeout",
                    },
                )
                _record_internalize_failure(
                    bot_name,
                    module_path=None,
                    reason="stale_internalization_timeout",
                    logger=logger_ref,
                )
                _record_stale_internalization_failure_event(
                    bot_name=bot_name,
                    started_at=started_at,
                    age_seconds=age,
                    threshold_seconds=hard_timeout,
                    logger=logger_ref,
                    bot_registry=bot_registry,
                    node=node,
                )
                _launch_internalize_timeout_self_debug(
                    bot_name=bot_name,
                    age_seconds=age,
                    timeout_seconds=hard_timeout,
                    started_at=started_at,
                    logger=logger_ref,
                )
                _schedule_internalization_timeout_retry(
                    bot_name=bot_name,
                    logger=logger_ref,
                    bot_registry=bot_registry,
                )
                _record_attempt_finish("stale_internalization_timeout")
                return _inflight_manager_fallback()
            _log_internalize_stack("in-flight")
            logged_internalize_stack = True
            last_step = None
            if node is not None:
                last_step = node.get("internalization_last_step")
            logger_ref.info(
                "internalize_coding_bot already in-flight for %s; skipping manager construction",
                bot_name,
                extra={
                    "bot": bot_name,
                    "in_flight_seconds": age,
                    "internalization_last_step": last_step,
                },
            )
            _record_attempt_finish("skipped_in_flight")
            return _inflight_manager_fallback()

    now_monotonic = time.monotonic()
    debounce_window_seconds = _resolve_internalize_debounce_seconds(bot_name)
    with _INTERNALIZE_DEBOUNCE_LOCK:
        previous_attempt_started = _INTERNALIZE_LAST_ATTEMPT_STARTED_AT.get(bot_name)
        within_debounce = (
            isinstance(previous_attempt_started, (int, float))
            and debounce_window_seconds > 0
            and (now_monotonic - float(previous_attempt_started)) < debounce_window_seconds
        )
        if not within_debounce:
            _INTERNALIZE_LAST_ATTEMPT_STARTED_AT[bot_name] = now_monotonic

    if within_debounce:
        if not logged_internalize_stack:
            _log_internalize_stack("debounce")
            logged_internalize_stack = True
        _record_attempt_finish("skipped_debounce")
        logger_ref.info(
            "internalize_coding_bot debounce active for %s; skipping manager construction",
            bot_name,
            extra={
                "bot": bot_name,
                "debounce_window_seconds": debounce_window_seconds,
            },
        )
        return _inflight_manager_fallback()

    with _INTERNALIZE_IN_FLIGHT_LOCK:
        _INTERNALIZE_IN_FLIGHT[bot_name] = now_monotonic
        added_in_flight = True
        _INTERNALIZE_MONITOR_LAST_LOGGED_AT.pop(bot_name, None)

    _record_attempt_start()
    if node is not None:
        node["internalization_in_progress"] = time.time()
        node["internalization_last_step"] = "internalization start"

    try:
        if _recent_internalization():
            if not logged_internalize_stack:
                _log_internalize_stack("recent-internalization")
                logged_internalize_stack = True
            logger_ref.info(
                "internalize_coding_bot skipping reinternalization for %s; recent internalization detected",
                bot_name,
            )
            return _inflight_manager_fallback()
        internalize_lock = _get_internalize_lock(bot_name)
        if not internalize_lock.acquire(blocking=False):
            if not logged_internalize_stack:
                _log_internalize_stack("lock-in-progress")
                logged_internalize_stack = True
            logger_ref.info(
                "internalize_coding_bot already in progress for %s; skipping",
                bot_name,
            )
            return _inflight_manager_fallback()

        try:
            def _track_failure(reason: str) -> None:
                nonlocal failure_recorded
                if failure_recorded:
                    return
                failure_recorded = True
                reason_text = str(reason or "")
                reason_category = (
                    "shutdown_race"
                    if _is_internalize_shutdown_race(reason_text)
                    else reason_text
                )
                path_value = (
                    str(module_path)
                    if module_path is not None
                    else module_hint
                )
                _record_internalize_failure(
                    bot_name,
                    module_path=path_value,
                    reason=reason_category,
                    logger=getattr(manager, "logger", None) or logger_ref,
                )

            def _start_step(step: str) -> float:
                print(f"[debug] {bot_name}: starting {step}")
                if node is not None:
                    node["internalization_last_step"] = step
                return time.monotonic()

            def _end_step(step: str, started_at: float) -> float:
                elapsed = time.monotonic() - started_at
                print(f"[debug] {bot_name}: finished {step} in {elapsed:.3f}s")
                return elapsed

            if _internalize_in_cooldown(bot_name):
                _record_attempt_finish("cooldown")
                return _cooldown_disabled_manager(bot_registry, data_bot)

            shutdown_guard = _shutdown_guard_state()
            if shutdown_guard is not None:
                if node is not None:
                    node["internalization_last_step"] = "internalization skipped: shutdown"
                logger_ref.info(
                    "internalize_coding_bot skipped for %s because shutdown is active",
                    bot_name,
                    extra=shutdown_guard,
                )
                event_bus = (
                    getattr(evolution_orchestrator, "event_bus", None)
                    or getattr(data_bot, "event_bus", None)
                )
                if event_bus is not None:
                    try:
                        event_bus.publish(
                            "self_coding:internalize_skipped_shutdown",
                            dict(shutdown_guard),
                        )
                    except Exception:
                        logger_ref.exception(
                            "failed to publish internalize shutdown skip event for %s",
                            bot_name,
                        )
                _record_attempt_finish("skipped_shutdown")
                return _inflight_manager_fallback()

            if _current_self_coding_import_depth() > 0:
                print("Forcing manager despite depth lock")

            if node is not None:
                disabled_state = node.get("self_coding_disabled")
                if (
                    isinstance(disabled_state, dict)
                    and disabled_state.get("source") == "module_path_resolution"
                ):
                    _record_attempt_finish("disabled_module_path_resolution")
                    return _cooldown_disabled_manager(bot_registry, data_bot)
            existing_manager = None
            if node is not None:
                existing_manager = node.get("selfcoding_manager") or node.get("manager")
            if existing_manager is not None and _manager_healthy(existing_manager):
                _mark_last_internalized()
                logger_ref.info(
                    "internalize_coding_bot reusing existing manager for %s", bot_name
                )
                _record_attempt_finish("reused_existing_manager")
                return existing_manager
            if existing_manager is not None and _recent_internalization():
                if not logged_internalize_stack:
                    _log_internalize_stack("recent-internalization")
                _mark_last_internalized()
                logger_ref.info(
                    "internalize_coding_bot skipping reinternalization for %s; "
                    "recent internalization detected",
                    bot_name,
                )
                _record_attempt_finish("reused_recent_manager")
                return existing_manager

            delay = 0.0
            if _INTERNALIZE_THROTTLE_SECONDS > 0:
                with _INTERNALIZE_THROTTLE_LOCK:
                    now = time.monotonic()
                    elapsed = now - _LAST_INTERNALIZE_AT
                    if elapsed < _INTERNALIZE_THROTTLE_SECONDS:
                        delay = _INTERNALIZE_THROTTLE_SECONDS - elapsed
                        target = now + delay
                    else:
                        target = now
                    _LAST_INTERNALIZE_AT = target
            if delay > 0:
                time.sleep(delay)

            print(f"[debug] internalize_coding_bot invoked for bot: {bot_name}")

            def _clear_internalize_state(*, last_step: str) -> None:
                nonlocal added_in_flight
                with _INTERNALIZE_IN_FLIGHT_LOCK:
                    _INTERNALIZE_IN_FLIGHT.pop(bot_name, None)
                    _INTERNALIZE_MONITOR_LAST_LOGGED_AT.pop(bot_name, None)
                added_in_flight = False
                if node is not None:
                    node.pop("internalization_in_progress", None)
                    node["internalization_last_step"] = last_step

            def _emit_manager_construction_timeout_failure(
                *,
                attempt_index: int,
                timeout_seconds: float,
                elapsed_seconds: float,
                phase: str | None = None,
                phase_elapsed_seconds: float | None = None,
                phase_history: list[tuple[str, float]] | None = None,
                retry_timeout_seconds: float | None = None,
                fallback_used: bool = False,
            ) -> None:
                _track_failure("manager_construction_timeout")
                event_bus = getattr(bot_registry, "event_bus", None)
                payload = {
                    "bot": bot_name,
                    "description": f"internalize:{bot_name}",
                    "path": str(module_path) if module_path else module_hint,
                    "severity": 0.0,
                    "success": False,
                    "post_validation_success": False,
                    "post_validation_error": "manager_construction_timeout",
                    "timeout_seconds": timeout_seconds,
                    "attempt_index": attempt_index,
                    "elapsed_seconds": elapsed_seconds,
                    "phase": phase,
                    "phase_elapsed_seconds": phase_elapsed_seconds,
                    "phase_history": phase_history or [],
                    "retry_timeout_seconds": retry_timeout_seconds,
                    "fallback_used": fallback_used,
                }
                if event_bus is not None:
                    try:
                        event_bus.publish("self_coding:patch_attempt", payload)
                    except Exception:
                        logger_ref.exception(
                            "failed to publish manager construction timeout failure for %s",
                            bot_name,
                        )

                internalize_failure_payload = {
                    "bot": bot_name,
                    "reason": "manager_construction_timeout",
                    "timeout_seconds": timeout_seconds,
                    "attempt_index": attempt_index,
                    "elapsed_seconds": elapsed_seconds,
                    "phase": phase,
                    "phase_elapsed_seconds": phase_elapsed_seconds,
                    "phase_history": phase_history or [],
                    "retry_timeout_seconds": retry_timeout_seconds,
                    "fallback_used": fallback_used,
                    "timestamp": time.time(),
                }
                if event_bus is not None:
                    try:
                        event_bus.publish(
                            "self_coding:internalization_failure",
                            internalize_failure_payload,
                        )
                    except Exception:
                        logger_ref.exception(
                            "failed to publish internalization timeout failure for %s",
                            bot_name,
                        )

            manager_timer = _start_step("manager construction")
            reduced_scope_after_timeout_retry = False
            manager_timeout = _resolve_manager_timeout_seconds(bot_name)
            normalized_bot_name = _normalize_env_bot_name(bot_name)
            manager_phase_lock = threading.Lock()
            manager_phase_state: dict[str, Any] = {
                "phase": "queued",
                "phase_started_at": time.monotonic(),
                "history": [("queued", time.monotonic())],
            }

            def _phase_metrics_store() -> dict[str, Any]:
                cache = getattr(bot_registry, "_manager_phase_duration_metrics", None)
                if not isinstance(cache, dict):
                    cache = {}
                    setattr(bot_registry, "_manager_phase_duration_metrics", cache)
                return cache

            def _phase_metrics_for_bot() -> dict[str, Any]:
                cache = _phase_metrics_store()
                entry = cache.get(normalized_bot_name)
                if not isinstance(entry, dict):
                    entry = {}
                    cache[normalized_bot_name] = entry
                if node is not None:
                    node.setdefault("manager_phase_duration_metrics", entry)
                return entry

            def _snapshot_phase_state() -> tuple[str, float, list[tuple[str, float]]]:
                with manager_phase_lock:
                    phase = str(manager_phase_state.get("phase", "unknown"))
                    phase_started_at = float(
                        manager_phase_state.get("phase_started_at", manager_timer)
                    )
                    raw_history = manager_phase_state.get("history", [])
                    history = list(raw_history) if isinstance(raw_history, list) else []
                return phase, phase_started_at, history

            def _record_successful_phase_durations() -> None:
                _, _, history = _snapshot_phase_state()
                if len(history) < 2:
                    return
                metrics = _phase_metrics_for_bot()
                for idx in range(1, len(history)):
                    phase_name, phase_started = history[idx - 1]
                    _, next_started = history[idx]
                    elapsed = max(0.0, float(next_started) - float(phase_started))
                    phase_key = str(phase_name or "unknown")
                    phase_entry = metrics.setdefault(
                        phase_key,
                        {"successful_elapsed_seconds": [], "last_successful_elapsed_seconds": 0.0},
                    )
                    samples = phase_entry.get("successful_elapsed_seconds", [])
                    if not isinstance(samples, list):
                        samples = []
                        phase_entry["successful_elapsed_seconds"] = samples
                    samples.append(elapsed)
                    if len(samples) > _MANAGER_PHASE_DURATION_ROLLING_LIMIT:
                        del samples[:-_MANAGER_PHASE_DURATION_ROLLING_LIMIT]
                    phase_entry["last_successful_elapsed_seconds"] = elapsed
                    phase_entry["updated_at"] = time.time()

            def _record_manager_phase(phase: str) -> None:
                now = time.monotonic()
                with manager_phase_lock:
                    manager_phase_state["phase"] = phase
                    manager_phase_state["phase_started_at"] = now
                    history = manager_phase_state.setdefault("history", [])
                    if isinstance(history, list):
                        history.append((phase, now))
                logger_ref.info(
                    "internalize_coding_bot manager construction phase for %s: %s",
                    bot_name,
                    phase,
                )

            def _build_manager() -> SelfCodingManager:
                _record_manager_phase("manager_init:enter")
                manager = SelfCodingManager(
                    engine,
                    pipeline,
                    bot_name=bot_name,
                    data_bot=data_bot,
                    bot_registry=bot_registry,
                    roi_drop_threshold=roi_threshold,
                    error_rate_threshold=error_threshold,
                    defer_orchestrator_init=True,
                    construction_phase_callback=_record_manager_phase,
                    **manager_kwargs,
                )
                _record_manager_phase("manager_init:return")
                return manager

            executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            manager_future = executor.submit(_build_manager)
            try:
                logger_ref.info(
                    "internalize_coding_bot waiting on manager construction future",
                    extra={
                        "event": "internalize_manager_wait_start",
                        "bot_name": bot_name,
                        "bot_key": _normalize_env_bot_name(bot_name),
                        "manager_timeout_seconds": manager_timeout,
                    },
                )
                manager = manager_future.result(
                    timeout=None if manager_timeout == 0.0 else manager_timeout
                )
                _record_successful_phase_durations()
            except concurrent.futures.TimeoutError:
                    elapsed = max(0.0, time.monotonic() - manager_timer)
                    timeout_phase, phase_started_at, timeout_history = _snapshot_phase_state()
                    phase_elapsed = max(0.0, time.monotonic() - phase_started_at)
                    _record_internalize_failure(
                        bot_name,
                        module_path=module_path,
                        reason="manager_construction_timeout",
                        logger=logger_ref,
                    )
                    is_bot_planning = _normalize_env_bot_name(bot_name) == "BOTPLANNINGBOT"
                    phase_metrics = _phase_metrics_for_bot()
                    retry_timeout = None
                    if is_bot_planning:
                        retry_timeout = _compute_adaptive_manager_retry_timeout_seconds(
                            bot_name,
                            primary_timeout=manager_timeout,
                            timeout_phase=timeout_phase,
                            timeout_phase_elapsed_seconds=phase_elapsed,
                            timeout_history=timeout_history,
                            phase_metrics=phase_metrics,
                        )
                    _emit_manager_construction_timeout_failure(
                        attempt_index=1,
                        timeout_seconds=manager_timeout,
                        elapsed_seconds=elapsed,
                        phase=timeout_phase,
                        phase_elapsed_seconds=phase_elapsed,
                        phase_history=timeout_history,
                        retry_timeout_seconds=retry_timeout,
                        fallback_used=False,
                    )
                    _clear_internalize_state(last_step=f"manager construction timeout ({timeout_phase})")
                    if node is not None:
                        node["internalization_deferred_retry_at"] = time.time()
                    _schedule_internalization_timeout_retry(
                        bot_name=bot_name,
                        logger=logger_ref,
                        bot_registry=bot_registry,
                    )

                    if is_bot_planning:
                        logger_ref.warning(
                            "internalize_coding_bot timeout for %s; retrying once with bounded timeout",
                            bot_name,
                            extra={
                                "event": "internalize_manager_construction_timeout_retry",
                                "bot": bot_name,
                                "attempt_index": 2,
                                "timeout_seconds": retry_timeout,
                                "phase": timeout_phase,
                                "phase_elapsed_seconds": phase_elapsed,
                                "phase_history": timeout_history,
                                "retry_timeout_seconds": retry_timeout,
                                "fallback_used": False,
                            },
                        )
                        retry_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
                        retry_future = retry_executor.submit(_build_manager)
                        try:
                            manager = retry_future.result(timeout=retry_timeout)
                            _record_successful_phase_durations()
                            reduced_scope_after_timeout_retry = True
                            logger_ref.info(
                                "internalize_coding_bot retry succeeded for %s",
                                bot_name,
                                extra={
                                    "event": "internalize_manager_construction_timeout_retry_succeeded",
                                    "bot": bot_name,
                                    "attempt_index": 2,
                                    "timeout_seconds": retry_timeout,
                                    "phase": timeout_phase,
                                    "phase_elapsed_seconds": phase_elapsed,
                                    "phase_history": timeout_history,
                                    "retry_timeout_seconds": retry_timeout,
                                    "fallback_used": False,
                                },
                            )
                        except (concurrent.futures.TimeoutError, Exception) as retry_exc:
                            retry_elapsed = max(0.0, time.monotonic() - manager_timer)
                            retry_phase, retry_phase_started_at, retry_phase_history = _snapshot_phase_state()
                            retry_phase_elapsed = max(
                                0.0, time.monotonic() - retry_phase_started_at
                            )
                            retry_timed_out = isinstance(
                                retry_exc,
                                concurrent.futures.TimeoutError,
                            )
                            fallback_manager: Any | None = None
                            fallback_error: Exception | None = None
                            try:
                                fallback_manager = _cooldown_disabled_manager(bot_registry, data_bot)
                            except Exception as exc:
                                fallback_error = exc
                            fallback_available = fallback_manager is not None
                            _emit_manager_construction_timeout_failure(
                                attempt_index=2,
                                timeout_seconds=retry_timeout,
                                elapsed_seconds=retry_elapsed,
                                phase=retry_phase,
                                phase_elapsed_seconds=retry_phase_elapsed,
                                phase_history=retry_phase_history,
                                retry_timeout_seconds=retry_timeout,
                                fallback_used=fallback_available,
                            )
                            retry_event = {
                                "event": "internalize_manager_construction_timeout_retry_failed",
                                "bot": bot_name,
                                "attempt_index": 2,
                                "phase": retry_phase,
                                "phase_elapsed_seconds": retry_phase_elapsed,
                                "phase_history": retry_phase_history,
                                "timeout_seconds": manager_timeout,
                                "retry_timeout_seconds": retry_timeout,
                                "fallback_available": fallback_available,
                                "retry_failure_kind": (
                                    "timeout" if retry_timed_out else "exception"
                                ),
                            }
                            if fallback_manager is not None:
                                _schedule_internalization_timeout_retry(
                                    bot_name=bot_name,
                                    logger=logger_ref,
                                    bot_registry=bot_registry,
                                )
                                logger_ref.warning(
                                    "internalize_coding_bot retry failed for %s; returning degraded fallback manager and deferring completion",
                                    bot_name,
                                    extra={
                                        **retry_event,
                                        "deferred_completion_scheduled": True,
                                        "fallback_used": True,
                                    },
                                )
                                return fallback_manager

                            logger_ref.error(
                                "internalize_coding_bot retry failed for %s with no degraded fallback",
                                bot_name,
                                extra={
                                    **retry_event,
                                    "retry_error": repr(retry_exc),
                                    "fallback_error": repr(fallback_error),
                                    "failure_kind": "hard_timeout_without_fallback",
                                },
                            )
                            if fallback_error is not None:
                                raise fallback_error
                            if retry_timed_out:
                                raise TimeoutError(
                                    "manager construction timed out for "
                                    f"{bot_name} after {manager_timeout:.2f}s "
                                    f"and retry timed out after {retry_timeout:.2f}s"
                                )
                            raise retry_exc
                        finally:
                            retry_executor.shutdown(wait=False, cancel_futures=True)
                    else:
                        fallback_manager: Any | None = None
                        fallback_error: Exception | None = None
                        try:
                            fallback_manager = _cooldown_disabled_manager(bot_registry, data_bot)
                        except Exception as exc:
                            fallback_error = exc

                        timeout_event = {
                            "event": "internalize_manager_construction_timeout",
                            "bot": bot_name,
                            "attempt_index": 1,
                            "timeout_seconds": manager_timeout,
                            "elapsed_seconds": elapsed,
                            "phase": timeout_phase,
                            "phase_elapsed_seconds": phase_elapsed,
                            "phase_history": timeout_history,
                            "retry_backoff_seconds": max(
                                0.0, _INTERNALIZE_TIMEOUT_RETRY_BACKOFF_SECONDS
                            ),
                            "retry_timeout_seconds": retry_timeout,
                            "fallback_used": fallback_manager is not None,
                        }
                        if fallback_manager is not None:
                            logger_ref.warning(
                                "internalize_coding_bot transient timeout for %s; returning degraded fallback manager",
                                bot_name,
                                extra=timeout_event,
                            )
                            if FailureGuard is not None:
                                with FailureGuard(
                                    stage="internalize_manager_construction",
                                    metadata={
                                        "bot": bot_name,
                                        "reason": "manager_construction_timeout",
                                        "severity": "warning",
                                        "timeout_seconds": manager_timeout,
                                        "elapsed_seconds": elapsed,
                                        "fallback": "cooldown_disabled_manager",
                                    },
                                    logger=logger_ref,
                                    suppress=True,
                                ):
                                    raise TimeoutError(
                                        f"manager construction timed out for {bot_name} after {manager_timeout:.2f}s"
                                    )
                            return fallback_manager

                        logger_ref.error(
                            "internalize_coding_bot hard failure for %s after manager construction timeout; degraded fallback unavailable",
                            bot_name,
                            extra={
                                **timeout_event,
                                "fallback_error": repr(fallback_error),
                                "failure_kind": "hard_timeout_without_fallback",
                            },
                        )
                        if fallback_error is not None:
                            raise fallback_error
                        raise TimeoutError(
                            f"manager construction timed out for {bot_name} after {manager_timeout:.2f}s"
                        )
            finally:
                executor.shutdown(wait=False, cancel_futures=True)
            _end_step("manager construction", manager_timer)
            deferred_orchestrator_timer = _start_step("deferred orchestrator initialization")
            try:
                if reduced_scope_after_timeout_retry:
                    manager.initialize_deferred_components(skip_non_critical=True)
                else:
                    manager.initialize_deferred_components()
            finally:
                _end_step("deferred orchestrator initialization", deferred_orchestrator_timer)
            if reduced_scope_after_timeout_retry:
                def _finish_deferred_setup() -> None:
                    try:
                        manager.initialize_deferred_components()
                    except Exception:
                        logger_ref.exception(
                            "internalize_coding_bot deferred completion failed for %s",
                            bot_name,
                        )

                threading.Thread(
                    target=_finish_deferred_setup,
                    name=f"internalize-deferred-complete-{bot_name}",
                    daemon=True,
                ).start()
            if provenance_token:
                try:
                    setattr(manager, "_bootstrap_provenance_token", provenance_token)
                except Exception:
                    manager.logger.debug(
                        "failed to persist bootstrap provenance token", exc_info=True
                    )
            if manager.quick_fix is None:
                raise ImportError("QuickFixEngine failed to initialise")
            _mark_last_internalized()
            manager.evolution_orchestrator = evolution_orchestrator
            register_timer = _start_step("bot_registry.register_bot")
            bot_registry.register_bot(
                bot_name,
                roi_threshold=roi_threshold,
                error_threshold=error_threshold,
                test_failure_threshold=test_failure_threshold,
                manager=manager,
                data_bot=data_bot,
                is_coding_bot=True,
            )
            _end_step("bot_registry.register_bot", register_timer)
            schedule_timer = _start_step("data_bot.schedule_monitoring")
            if hasattr(data_bot, "schedule_monitoring"):
                try:
                    data_bot.schedule_monitoring(bot_name)
                except Exception:  # pragma: no cover - best effort
                    manager.logger.exception(
                        "failed to schedule monitoring for %s", bot_name
                    )
                finally:
                    _end_step("data_bot.schedule_monitoring", schedule_timer)
            else:
                manager.logger.warning(
                    "data bot lacks schedule_monitoring for %s (data_bot=%s)",
                    bot_name,
                    type(data_bot).__name__,
                )
                _end_step("data_bot.schedule_monitoring", schedule_timer)
            settings = getattr(data_bot, "settings", None)
            thresholds = getattr(settings, "bot_thresholds", {}) if settings else {}
            if bot_name not in thresholds:
                threshold_timer = _start_step("threshold persistence")
                try:
                    persist_sc_thresholds(
                        bot_name,
                        roi_drop=(
                            roi_threshold
                            if roi_threshold is not None
                            else getattr(settings, "self_coding_roi_drop", None)
                        ),
                        error_increase=(
                            error_threshold
                            if error_threshold is not None
                            else getattr(settings, "self_coding_error_increase", None)
                        ),
                        test_failure_increase=(
                            test_failure_threshold
                            if test_failure_threshold is not None
                            else getattr(
                                settings, "self_coding_test_failure_increase", None
                            )
                        ),
                        event_bus=getattr(data_bot, "event_bus", None),
                    )
                except Exception:  # pragma: no cover - best effort
                    manager.logger.exception(
                        "failed to persist thresholds for %s", bot_name
                    )
                finally:
                    _end_step("threshold persistence", threshold_timer)
            if evolution_orchestrator is not None:
                orchestrator_timer = _start_step("evolution orchestrator registration")
                evolution_orchestrator.selfcoding_manager = manager
                try:
                    evolution_orchestrator.register_bot(bot_name)
                except Exception:  # pragma: no cover - best effort
                    manager.logger.exception(
                        "failed to register %s with EvolutionOrchestrator", bot_name
                    )
                bus = getattr(evolution_orchestrator, "event_bus", None)
                if bus:
                    try:
                        bus.subscribe(
                            "degradation:detected",
                            lambda _t, e: evolution_orchestrator.register_patch_cycle(e),
                        )
                    except Exception:  # pragma: no cover - best effort
                        manager.logger.exception(
                            "failed to subscribe degradation events for %s", bot_name
                        )
                _end_step("evolution orchestrator registration", orchestrator_timer)
            else:
                print(
                    f"[debug] {bot_name}: skipping evolution orchestrator registration (not provided)"
                )
            event_bus = (
                getattr(manager, "event_bus", None)
                or getattr(evolution_orchestrator, "event_bus", None)
                or getattr(data_bot, "event_bus", None)
            )
            module_resolution_timer = _start_step("module path resolution")
            module_candidates: dict[str, str | None] = {
                "graph_node": None,
                "registry_known": None,
                "modules_cache": None,
                "importlib_spec": None,
                "imported_module": None,
                "dynamic_router": None,
            }
            try:
                node = bot_registry.graph.nodes.get(bot_name) if bot_registry else None
                if node:
                    module_str = node.get("module")
                    if module_str:
                        module_path = Path(module_str)
                        module_hint = str(module_str)
                        module_candidates["graph_node"] = str(module_str)
            except Exception:
                module_path = None
            if module_path is None or not module_path.exists():
                module_entry: str | os.PathLike[str] | None = None
                try:
                    if bot_registry is not None and hasattr(
                        bot_registry, "get_known_module_path"
                    ):
                        module_entry = bot_registry.get_known_module_path(bot_name)
                except Exception:
                    module_entry = None
                if module_entry is not None:
                    module_candidates["registry_known"] = str(module_entry)
                if module_entry is None and bot_name in getattr(
                    bot_registry, "modules", {}
                ):
                    try:
                        module_entry = bot_registry.modules.get(bot_name)
                    except Exception:
                        module_entry = None
                    else:
                        if module_entry is not None:
                            module_candidates["modules_cache"] = str(module_entry)
                if module_entry:
                    if module_hint is None:
                        module_hint = str(module_entry)
                    dotted_module_path: Path | None = None
                    if isinstance(module_entry, (str, os.PathLike)):
                        module_entry_str = str(module_entry)
                        dotted_module_path = Path(module_entry)
                        is_explicit_path = module_entry_str.endswith(".py")
                        if not dotted_module_path.exists() and not is_explicit_path:
                            module_file = None
                            try:
                                spec = importlib.util.find_spec(module_entry_str)
                            except Exception:
                                spec = None
                            if spec and getattr(spec, "origin", None):
                                module_file = spec.origin
                                module_candidates["importlib_spec"] = spec.origin
                            if module_file is None:
                                try:
                                    imported_module = importlib.import_module(
                                        module_entry_str
                                    )
                                except Exception:
                                    imported_module = None
                                if imported_module is not None:
                                    module_file = getattr(
                                        imported_module, "__file__", None
                                    )
                                    module_candidates["imported_module"] = module_file
                            if module_file:
                                dotted_module_path = Path(module_file)
                            if module_file is None and hasattr(
                                _path_router, "resolve_module_path"
                            ):
                                try:
                                    resolved = _path_router.resolve_module_path(
                                        str(module_entry)
                                    )
                                except FileNotFoundError:
                                    resolved = None
                                except Exception:
                                    resolved = None
                                    manager.logger.debug(
                                        "dynamic_path_router failed to resolve %s",
                                        module_entry,
                                        exc_info=True,
                                    )
                                if resolved is not None:
                                    module_candidates["dynamic_router"] = str(resolved)
                                    dotted_module_path = Path(resolved)
                    module_path = dotted_module_path or Path(module_entry)
            if module_path is None or not (module_path and module_path.exists()):
                try:
                    module = importlib.import_module(bot_name)
                except Exception:
                    module = None
                if module is not None:
                    module_file = getattr(module, "__file__", "")
                    if module_file:
                        candidate = Path(module_file)
                        if candidate.exists():
                            module_path = candidate
            if module_path is not None and module_path.exists():
                module_path = module_path.resolve()
                if node is not None:
                    try:
                        node["module"] = str(module_path)
                    except Exception:
                        pass
                if bot_registry is not None and hasattr(bot_registry, "modules"):
                    try:
                        bot_registry.modules[bot_name] = str(module_path)
                    except Exception:  # pragma: no cover - best effort
                        pass
                if bot_registry is not None and hasattr(
                    bot_registry, "clear_module_path_failures"
                ):
                    try:
                        bot_registry.clear_module_path_failures(bot_name)
                    except Exception:  # pragma: no cover - best effort
                        pass
            _end_step("module path resolution", module_resolution_timer)
            provenance_timer = _start_step("provenance token acquisition")
            provenance_token = getattr(manager, "_bootstrap_provenance_token", None)
            if getattr(manager, "evolution_orchestrator", None) is not None:
                provenance_token = (
                    getattr(manager.evolution_orchestrator, "provenance_token", None)
                    or provenance_token
                )
            if provenance_token is None and evolution_orchestrator is not None:
                provenance_token = getattr(evolution_orchestrator, "provenance_token", None)
            _end_step("provenance token acquisition", provenance_timer)
            description = f"internalize:{bot_name}"
    
            def _emit_failure(reason: str) -> None:
                _track_failure(reason)
                data_bot_ref = getattr(manager, "data_bot", None)
                if data_bot_ref:
                    try:
                        data_bot_ref.collect(
                            bot_name,
                            patch_success=0.0,
                            patch_failure_reason=reason,
                        )
                    except Exception:
                        manager.logger.exception(
                            "failed to record post patch failure metrics"
                        )
                if event_bus:
                    payload = {
                        "bot": bot_name,
                        "description": description,
                        "path": str(module_path) if module_path else None,
                        "severity": 0.0,
                        "success": False,
                        "post_validation_success": False,
                        "post_validation_error": reason,
                    }
                    try:
                        event_bus.publish("self_coding:patch_attempt", payload)
                    except Exception:
                        manager.logger.exception(
                            "failed to publish internalize patch_attempt for %s",
                            bot_name,
                        )
    
            if module_path is None or not module_path.exists():
                print(
                    f"[debug] Bootstrap failed at module_path resolution due to missing path for: {bot_name}"
                )
                if hasattr(manager, "logger"):
                    try:
                        manager.logger.error(
                            "module_path resolution candidates for %s: graph_node=%s registry=%s modules_cache=%s importlib_spec=%s imported_module=%s dynamic_router=%s",
                            bot_name,
                            module_candidates.get("graph_node"),
                            module_candidates.get("registry_known"),
                            module_candidates.get("modules_cache"),
                            module_candidates.get("importlib_spec"),
                            module_candidates.get("imported_module"),
                            module_candidates.get("dynamic_router"),
                            extra={
                                "bot": bot_name,
                                "module_path": module_hint,
                                "module_candidates": module_candidates,
                            },
                        )
                        manager.logger.error(
                            "module_path resolution failed for %s; missing module path: %s",
                            bot_name,
                            module_hint,
                            extra={"bot": bot_name, "module_path": module_hint},
                        )
                    except Exception:
                        print(
                            f"[debug] failed to log module_path resolution error for {bot_name}",
                            file=sys.stderr,
                        )
                _launch_module_path_self_debug(
                    bot_name=bot_name,
                    module_candidates=module_candidates,
                    module_hint=module_hint,
                    manager=manager,
                )
                _emit_failure("module_path_missing")
                if bot_registry is not None and hasattr(
                    bot_registry, "register_module_path_failure"
                ):
                    try:
                        _attempts, delay, disabled = (
                            bot_registry.register_module_path_failure(
                                bot_name,
                                module_hint=module_hint,
                                resolved_path=str(module_path) if module_path else None,
                            )
                        )
                    except Exception:
                        manager.logger.exception(
                            "failed to register module_path resolution failure for %s",
                            bot_name,
                        )
                    else:
                        if disabled:
                            try:
                                manager.logger.error(
                                    "self-coding disabled for %s after repeated module_path resolution failures",
                                    bot_name,
                                )
                            except Exception:
                                pass
                        elif delay is not None:
                            try:
                                manager.logger.warning(
                                    "retrying module_path resolution for %s in %.1fs",
                                    bot_name,
                                    delay,
                                )
                            except Exception:
                                pass
                if hasattr(manager, "logger"):
                    try:
                        manager.logger.warning(
                            "skipping post patch cycle because module path is unavailable",
                            extra={"bot": bot_name},
                        )
                    except Exception:
                        # Fall back to stdout if structured logging fails during bootstrap
                        print(
                            f"[debug] failed to log missing module path for {bot_name}",
                            file=sys.stderr,
                        )
                print(
                    f"[debug] internalize_coding_bot returning early without post patch cycle for bot: {bot_name}"
                )
                _record_attempt_finish("module_path_missing")
                return manager
            if provenance_token is None:
                _emit_failure("missing_provenance")
                raise PermissionError("missing provenance token for post patch validation")
            post_cycle_timer = _start_step("run_post_patch_cycle")
            try:
                post_details = manager.run_post_patch_cycle(
                    module_path,
                    description,
                    provenance_token=provenance_token,
                    context_meta={"reason": "internalize"},
                )
            except Exception as exc:
                if RollbackManager is not None:
                    try:
                        RollbackManager().rollback("internalize", requesting_bot=bot_name)
                    except Exception:
                        manager.logger.exception("rollback failed for %s", bot_name)
                _emit_failure(str(exc))
                raise
            else:
                if event_bus:
                    payload = {
                        "bot": bot_name,
                        "description": description,
                        "path": str(module_path),
                        "severity": 0.0,
                        "success": True,
                        "post_validation_success": True,
                        "post_validation_details": post_details,
                    }
                    tests_failed = post_details.get("self_tests", {}).get("failed")
                    if tests_failed is not None:
                        payload["post_validation_tests_failed"] = tests_failed
                    try:
                        event_bus.publish("self_coding:patch_attempt", payload)
                    except Exception:
                        manager.logger.exception(
                            "failed to publish internalize patch_attempt for %s",
                            bot_name,
                        )
            finally:
                _end_step("run_post_patch_cycle", post_cycle_timer)
            _record_internalize_success(bot_name)
            _mark_last_internalized()
            _record_attempt_finish("success")
            print(f"[debug] internalize_coding_bot returning manager for bot: {bot_name}")
            return manager
        except Exception as exc:
            _track_failure(str(exc))
            _record_attempt_finish(str(exc) or "error")
            raise
        finally:
            internalize_lock.release()
    finally:
        if added_in_flight:
            with _INTERNALIZE_IN_FLIGHT_LOCK:
                _INTERNALIZE_IN_FLIGHT.pop(bot_name, None)
                _INTERNALIZE_MONITOR_LAST_LOGGED_AT.pop(bot_name, None)
        if attempt_started and node is not None and node.get("attempt_finished_at") is None:
            _record_attempt_finish(attempt_result)
        if node is not None:
            node.pop("internalization_in_progress", None)
            node["internalization_last_step"] = "internalization finished"
__all__ = [
    "SelfCodingManager",
    "PatchApprovalPolicy",
    "HelperGenerationError",
    "internalize_coding_bot",
]
