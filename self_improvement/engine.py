from __future__ import annotations

"""Periodic self-improvement engine for the Menace system.

Configuration is loaded via :class:`sandbox_settings.SandboxSettings` which
validates environment variables or values supplied through a configuration
file.  This surfaces misconfiguration errors early and provides typed access to
all tunable parameters.

The module relies on a few optional helpers that may not be installed in
minimal environments:

* ``sandbox_runner`` – integrates orphaned modules discovered during sandbox
  runs.
* ``quick_fix_engine`` – generates small corrective patches for helper code.

When these dependencies are missing the corresponding public functions now
raise :class:`RuntimeError` to surface configuration problems instead of
silently degrading behaviour.  Calls are wrapped with a small retry loop so
that transient failures are retried while permanent issues propagate errors.
"""

# flake8: noqa


def _qfe_log(message: str) -> None:
    print(f"[QFE:engine] {message}", flush=True)


_qfe_log("engine.py top-level import started")
_qfe_log(f"__name__ resolved to {__name__}")

import logging
_qfe_log("logging imported")

from pathlib import Path
_qfe_log("pathlib.Path imported")

try:
    from logging_utils import log_record, get_logger, setup_logging, set_correlation_id
except ImportError:  # pragma: no cover - allow package-relative import
    try:
        from menace_sandbox.logging_utils import (  # type: ignore
            log_record,
            get_logger,
            setup_logging,
            set_correlation_id,
        )
        _qfe_log("logging_utils imported (menace_sandbox variant)")
    except ImportError:  # pragma: no cover - simplified environments

        def log_record(**fields: object) -> dict[str, object]:  # type: ignore
            return fields

        def get_logger(name: str) -> logging.Logger:  # type: ignore
            return logging.getLogger(name)

        def setup_logging() -> None:  # type: ignore
            return

        def set_correlation_id(_: str | None) -> None:  # type: ignore
            return

        _qfe_log("logging_utils fallback shims defined")
else:
    _qfe_log("logging_utils imported")


import time
_qfe_log("time imported")

import threading
_qfe_log("threading imported")

import asyncio
_qfe_log("asyncio imported")

import os
_qfe_log("os imported")

import importlib
_qfe_log("importlib imported")

_MANUAL_LAUNCH_TRIGGERED = False


def _start_engine_heartbeat(interval: float = 5.0) -> None:
    """Emit periodic heartbeats to confirm module-level progress."""

    def _log_heartbeat() -> None:
        while True:
            print(
                f"[HEARTBEAT] engine alive at {time.time()}",
                flush=True,
            )
            time.sleep(interval)

    thread = threading.Thread(target=_log_heartbeat, daemon=True)
    thread.start()


try:
    _start_engine_heartbeat()
    print("[QFE:engine] heartbeat thread started", flush=True)
except Exception as exc:  # pragma: no cover - best effort diagnostics
    print(f"[QFE:engine] heartbeat thread failed: {exc}", flush=True)

from db_router import GLOBAL_ROUTER, init_db_router
_qfe_log("db_router imported")

from sandbox_settings import SandboxSettings, load_sandbox_settings
_qfe_log("sandbox_settings imported")
from .init import (
    init_self_improvement,
    settings,
    _repo_path,
    _data_dir,
    _atomic_write,
    get_default_synergy_weights,
)
_qfe_log("self_improvement.init imported")
try:  # pragma: no cover - prefer absolute imports when running from repo root
    from dynamic_path_router import resolve_path, resolve_module_path
except Exception:  # pragma: no cover - fallback for package-relative layout
    from menace_sandbox.dynamic_path_router import resolve_path, resolve_module_path
    _qfe_log("dynamic_path_router imported (menace_sandbox fallback)")
else:
    _qfe_log("dynamic_path_router imported")
try:  # pragma: no cover - prefer absolute imports when running from repo root
    from metrics_exporter import (
        synergy_weight_updates_total,
        synergy_weight_update_failures_total,
        synergy_weight_update_alerts_total,
        orphan_modules_reintroduced_total,
        orphan_modules_passed_total,
        orphan_modules_tested_total,
        orphan_modules_failed_total,
        orphan_modules_reclassified_total,
        orphan_modules_redundant_total,
        orphan_modules_legacy_total,
        prediction_mae,
        prediction_reliability,
        self_improvement_failure_total,
    )
except ImportError:  # pragma: no cover - fallback for package-relative layout
    from menace_sandbox.metrics_exporter import (
        synergy_weight_updates_total,
        synergy_weight_update_failures_total,
        synergy_weight_update_alerts_total,
        orphan_modules_reintroduced_total,
        orphan_modules_passed_total,
    orphan_modules_tested_total,
    orphan_modules_failed_total,
    orphan_modules_reclassified_total,
    orphan_modules_redundant_total,
    orphan_modules_legacy_total,
    prediction_mae,
    prediction_reliability,
    self_improvement_failure_total,
    )
    _qfe_log("metrics_exporter imported (menace_sandbox fallback)")
else:
    _qfe_log("metrics_exporter imported")

try:  # pragma: no cover - prefer absolute imports when running from repo root
    from composite_workflow_scorer import CompositeWorkflowScorer
    _COMPOSITE_SCORER_ERROR: ImportError | None = None
except ImportError as exc:  # pragma: no cover - fallback for package-relative layout
    try:
        from menace_sandbox.composite_workflow_scorer import CompositeWorkflowScorer
    except ImportError as exc2:
        CompositeWorkflowScorer = None  # type: ignore[assignment]
        _COMPOSITE_SCORER_ERROR = exc2
    else:
        _COMPOSITE_SCORER_ERROR = None
        _qfe_log("composite_workflow_scorer imported (menace_sandbox fallback)")
else:
    _COMPOSITE_SCORER_ERROR = None
    _qfe_log("composite_workflow_scorer imported")
try:  # pragma: no cover - prefer absolute imports when running from repo root
    from neuroplasticity import PathwayDB
except ImportError:  # pragma: no cover - fallback for package-relative layout
    from menace_sandbox.neuroplasticity import PathwayDB
    _qfe_log("neuroplasticity imported (menace_sandbox fallback)")
else:
    _qfe_log("neuroplasticity imported")
try:  # pragma: no cover - prefer absolute imports when running from repo root
    from data_bot import MetricsDB
except ImportError:  # pragma: no cover - fallback for package-relative layout
    from menace_sandbox.data_bot import MetricsDB
    _qfe_log("data_bot imported (menace_sandbox fallback)")
else:
    _qfe_log("data_bot imported")
try:  # pragma: no cover - prefer absolute imports when running from repo root
    from roi_results_db import ROIResultsDB
except ImportError:  # pragma: no cover - fallback for package-relative layout
    from menace_sandbox.roi_results_db import ROIResultsDB
    _qfe_log("roi_results_db imported (menace_sandbox fallback)")
else:
    _qfe_log("roi_results_db imported")
try:  # pragma: no cover - prefer absolute imports when running from repo root
    from workflow_stability_db import WorkflowStabilityDB
except ImportError:  # pragma: no cover - fallback for package-relative layout
    from menace_sandbox.workflow_stability_db import WorkflowStabilityDB
    _qfe_log("workflow_stability_db imported (menace_sandbox fallback)")
else:
    _qfe_log("workflow_stability_db imported")
try:  # pragma: no cover - optional dependency
    from task_handoff_bot import WorkflowDB, WorkflowRecord
except ImportError:  # pragma: no cover - best effort fallback
    WorkflowDB = WorkflowRecord = None  # type: ignore
    _qfe_log("task_handoff_bot unavailable - fallback to None")
else:
    _qfe_log("task_handoff_bot imported")
try:  # pragma: no cover - optional dependency
    from menace_sandbox.workflow_summary_db import WorkflowSummaryDB
except ImportError:  # pragma: no cover - fallback for flat layout
    from workflow_summary_db import WorkflowSummaryDB  # type: ignore
    _qfe_log("workflow_summary_db imported (flat layout fallback)")
else:
    _qfe_log("workflow_summary_db imported")
try:  # pragma: no cover - optional dependency
    from menace_sandbox.workflow_synergy_comparator import WorkflowSynergyComparator
    _WORKFLOW_SYNERGY_ERROR: ImportError | None = None
except ImportError as exc:  # pragma: no cover - fallback for flat layout
    try:
        from workflow_synergy_comparator import WorkflowSynergyComparator  # type: ignore
    except ImportError as exc2:  # pragma: no cover - best effort
        WorkflowSynergyComparator = None  # type: ignore[assignment]
        _WORKFLOW_SYNERGY_ERROR = exc2
    else:
        _WORKFLOW_SYNERGY_ERROR = None
        _qfe_log("workflow_synergy_comparator imported (flat layout fallback)")
else:
    _WORKFLOW_SYNERGY_ERROR = None
    _qfe_log("workflow_synergy_comparator imported")

_LOCAL_ROUTER: DBRouter | None = None


class _RouterProxy:
    """Lazy loader for :class:`DBRouter` used during module import."""

    def _resolve(self) -> DBRouter:
        global _LOCAL_ROUTER
        if GLOBAL_ROUTER is not None:
            return GLOBAL_ROUTER
        if _LOCAL_ROUTER is None:
            _LOCAL_ROUTER = init_db_router("self_improvement")
        return _LOCAL_ROUTER

    def __getattr__(self, name: str):
        return getattr(self._resolve(), name)

    def __bool__(self) -> bool:  # pragma: no cover - convenience only
        try:
            self._resolve()
        except Exception:
            return False
        return True


router = _RouterProxy()


class _WorkflowStabilityProxy:
    """Delay :class:`WorkflowStabilityDB` initialisation until first use."""

    def __init__(self) -> None:
        self._instance: WorkflowStabilityDB | None = None

    def _resolve(self) -> WorkflowStabilityDB:
        if self._instance is None:
            self._instance = WorkflowStabilityDB()
        return self._instance

    def __getattr__(self, name: str):
        return getattr(self._resolve(), name)

    def __bool__(self) -> bool:  # pragma: no cover - convenience only
        return self._instance is not None


STABLE_WORKFLOWS = _WorkflowStabilityProxy()
try:  # pragma: no cover - optional meta-planning helpers
    from .meta_planning import PLANNER_INTERVAL
    from . import meta_planning
    _META_PLANNING_ERROR: Exception | None = None
    _qfe_log("meta_planning imported")
except Exception as exc:  # pragma: no cover - simplified environments
    import sys
    import types

    _qfe_log("meta_planning import failed; using fallback stubs")

    PLANNER_INTERVAL = 1  # type: ignore[assignment]

    meta_planning = types.ModuleType("self_improvement.meta_planning")
    meta_planning.MetaWorkflowPlanner = None  # type: ignore[attr-defined]

    class _FallbackPlanner:
        roi_db = None

        def __init__(self, *args: object, **kwargs: object) -> None:
            return

        def schedule(
            self,
            *_args: object,
            **_kwargs: object,
        ) -> list[dict[str, object]]:
            return []

    def _recent_error_entropy(
        *_args: object,
        **_kwargs: object,
    ) -> tuple[list[object], float, float, float, float]:
        return ([], 0.0, 0.0, 0.0, 0.0)

    async def _self_improvement_cycle(*_args: object, **_kwargs: object) -> None:
        return None

    def _start_cycle(*_args: object, **_kwargs: object) -> None:
        return None

    def _stop_cycle(*_args: object, **_kwargs: object) -> None:
        return None

    meta_planning._FallbackPlanner = _FallbackPlanner  # type: ignore[attr-defined]
    meta_planning._recent_error_entropy = _recent_error_entropy  # type: ignore[attr-defined]
    meta_planning.PLANNER_INTERVAL = PLANNER_INTERVAL  # type: ignore[attr-defined]
    meta_planning.self_improvement_cycle = _self_improvement_cycle  # type: ignore[attr-defined]
    meta_planning.start_self_improvement_cycle = _start_cycle  # type: ignore[attr-defined]
    meta_planning.stop_self_improvement_cycle = _stop_cycle  # type: ignore[attr-defined]
    sys.modules.setdefault("self_improvement.meta_planning", meta_planning)
    sys.modules.setdefault("menace_sandbox.self_improvement.meta_planning", meta_planning)
    _META_PLANNING_ERROR = exc
# Time based interval (in seconds) used to periodically trigger the meta
# workflow planner in a background thread.  Keeping the configuration separate
# from the cycle based ``PLANNER_INTERVAL`` allows the planner to run even when
# no explicit self-improvement cycles are executed.
META_PLANNING_PERIOD = getattr(settings, "meta_planning_period", 0)
META_PLANNING_LOOP = getattr(settings, "meta_planning_loop", 0)
META_IMPROVEMENT_THRESHOLD = getattr(settings, "meta_improvement_threshold", 0)
try:  # pragma: no cover - neurosales provides advanced sales modelling
    import neurosales  # noqa: F401
    _NEUROSALES_ERROR: ImportError | None = None
    _qfe_log("neurosales imported")
except ImportError as exc:  # pragma: no cover - record for later use
    neurosales = None  # type: ignore[assignment]
    _NEUROSALES_ERROR = exc
    _qfe_log("neurosales import failed")

logger = get_logger(__name__)

if _NEUROSALES_ERROR is not None:
    logger.warning(  # noqa: TRY300
        "neurosales unavailable; advanced sales modelling disabled",
        extra=log_record(module=__name__, dependency="neurosales"),
        exc_info=_NEUROSALES_ERROR,
    )
if '_META_PLANNING_ERROR' in globals() and _META_PLANNING_ERROR is not None:
    logger.warning(  # noqa: TRY300
        "meta_planning unavailable; disabling planner integrations",
        extra=log_record(module=__name__, dependency="meta_planning"),
        exc_info=_META_PLANNING_ERROR,
    )
if '_MODULE_MAPPER_ERROR' in globals() and _MODULE_MAPPER_ERROR is not None:
    logger.warning(  # noqa: TRY300
        "dynamic_module_mapper unavailable; module grouping disabled",
        extra=log_record(module=__name__, dependency="dynamic_module_mapper"),
        exc_info=_MODULE_MAPPER_ERROR,
    )
if '_SYNERGY_GRAPH_ERROR' in globals() and _SYNERGY_GRAPH_ERROR is not None:
    logger.warning(  # noqa: TRY300
        "module_synergy_grapher unavailable; synergy visualisation disabled",
        extra=log_record(module=__name__, dependency="module_synergy_grapher"),
        exc_info=_SYNERGY_GRAPH_ERROR,
    )
if '_ENVIRONMENT_ERROR' in globals() and _ENVIRONMENT_ERROR is not None:
    logger.warning(  # noqa: TRY300
        "sandbox_runner environment unavailable; sandbox helpers disabled",
        extra=log_record(module=__name__, dependency="sandbox_runner.environment"),
        exc_info=_ENVIRONMENT_ERROR,
    )
if '_ORCHESTRATION_ERROR' in globals() and _ORCHESTRATION_ERROR is not None:
    logger.warning(  # noqa: TRY300
        "orchestration helpers unavailable; cycles disabled",
        extra=log_record(module=__name__, dependency="self_improvement.orchestration"),
        exc_info=_ORCHESTRATION_ERROR,
    )
if '_ROI_TRACKING_ERROR' in globals() and _ROI_TRACKING_ERROR is not None:
    logger.warning(  # noqa: TRY300
        "roi_tracking unavailable; alignment baseline updates disabled",
        extra=log_record(module=__name__, dependency="self_improvement.roi_tracking"),
        exc_info=_ROI_TRACKING_ERROR,
    )
if '_PATCH_APP_ERROR' in globals() and _PATCH_APP_ERROR is not None:
    logger.warning(  # noqa: TRY300
        "patch_application unavailable; patch generation disabled",
        extra=log_record(module=__name__, dependency="self_improvement.patch_application"),
        exc_info=_PATCH_APP_ERROR,
    )


TelemetryEvent = None  # type: ignore[assignment]
_TELEMETRY_EVENT_ERROR: BaseException | None = None
_TELEMETRY_EVENT_WARNED = False


def _load_telemetry_event() -> type:
    """Resolve :class:`TelemetryEvent` lazily."""

    global TelemetryEvent, _TELEMETRY_EVENT_ERROR
    if TelemetryEvent is not None:
        return TelemetryEvent
    if _TELEMETRY_EVENT_ERROR is not None:
        raise RuntimeError("TelemetryEvent previously failed to import") from _TELEMETRY_EVENT_ERROR

    try:  # pragma: no cover - prefer absolute imports when running from repo root
        module = importlib.import_module("menace_sandbox.error_logger")
    except Exception as exc:
        try:  # pragma: no cover - fallback for flat layout
            module = importlib.import_module("error_logger")
        except Exception:
            _TELEMETRY_EVENT_ERROR = exc
            raise RuntimeError("error_logger module unavailable") from exc

    try:
        TelemetryEvent = getattr(module, "TelemetryEvent")  # type: ignore[assignment]
    except AttributeError as exc:  # pragma: no cover - unexpected layout
        _TELEMETRY_EVENT_ERROR = exc
        raise RuntimeError("TelemetryEvent missing from error_logger") from exc

    return TelemetryEvent


def _get_telemetry_event() -> type | None:
    """Best-effort accessor for :class:`TelemetryEvent`."""

    global _TELEMETRY_EVENT_WARNED
    try:
        return _load_telemetry_event()
    except RuntimeError as exc:
        if not _TELEMETRY_EVENT_WARNED:
            _TELEMETRY_EVENT_WARNED = True
            logger.warning(  # noqa: TRY300
                "telemetry event unavailable",
                extra=log_record(module=__name__, dependency="TelemetryEvent"),
                exc_info=exc,
            )
        return None

try:
    DEFAULT_RELEVANCY_METRICS_DB = resolve_path("sandbox_data/relevancy_metrics.db")
except FileNotFoundError:
    data_dir = Path(resolve_path("sandbox_data"))
    DEFAULT_RELEVANCY_METRICS_DB = data_dir / "relevancy_metrics.db"
    try:
        DEFAULT_RELEVANCY_METRICS_DB.parent.mkdir(parents=True, exist_ok=True)
        DEFAULT_RELEVANCY_METRICS_DB.touch(exist_ok=True)
        logger.debug(
            "created default relevancy metrics database",
            extra=log_record(path=str(DEFAULT_RELEVANCY_METRICS_DB)),
        )
    except OSError as exc:  # pragma: no cover - surface creation failure
        raise RuntimeError(
            "unable to prepare relevancy metrics database"
        ) from exc
from alert_dispatcher import dispatch_alert
_qfe_log("alert_dispatcher imported")

import json
_qfe_log("json imported")

import inspect
_qfe_log("inspect imported")

import sqlite3
_qfe_log("sqlite3 imported")

import pickle
_qfe_log("pickle imported")

import io
_qfe_log("io imported")

import tempfile
_qfe_log("tempfile imported")

import math
_qfe_log("math imported")

import typing
_qfe_log("typing imported")

import shutil
_qfe_log("shutil imported")

import ast
_qfe_log("ast imported")

from yaml_fallback import get_yaml
_qfe_log("yaml_fallback imported")

import traceback
_qfe_log("traceback imported")

from typing import Mapping, Callable, Iterable, Dict, Any, Sequence, List, TYPE_CHECKING
_qfe_log("typing aliases imported")

from datetime import datetime
_qfe_log("datetime imported")
try:  # pragma: no cover - optional dependency
    from dynamic_module_mapper import build_module_map, discover_module_groups
    _MODULE_MAPPER_ERROR: ImportError | None = None
    _qfe_log("dynamic_module_mapper imported")
except ImportError as exc:  # pragma: no cover - graceful degradation
    def build_module_map(*_args: object, **_kwargs: object) -> dict[str, object]:
        return {}

    def discover_module_groups(*_args: object, **_kwargs: object) -> list[list[str]]:
        return []

    _MODULE_MAPPER_ERROR = exc
    _qfe_log("dynamic_module_mapper import failed")
try:  # pragma: no cover - allow flat imports
    from menace_sandbox.module_synergy_grapher import get_synergy_cluster
    _SYNERGY_GRAPH_ERROR: ImportError | None = None
except ImportError as exc:  # pragma: no cover - fallback for flat layout
    try:
        from module_synergy_grapher import get_synergy_cluster  # type: ignore
    except ImportError as exc2:
        def get_synergy_cluster(*_args: object, **_kwargs: object) -> list[str]:
            return []

        _SYNERGY_GRAPH_ERROR = exc2
        _qfe_log("module_synergy_grapher import failed")
    else:
        _SYNERGY_GRAPH_ERROR = None
        _qfe_log("module_synergy_grapher imported (flat layout)")
else:
    _SYNERGY_GRAPH_ERROR = None
    _qfe_log("module_synergy_grapher imported")
yaml = get_yaml("self_improvement.engine")
_qfe_log("yaml configuration loaded")

try:
    from menace_sandbox import security_auditor
    _qfe_log("security_auditor imported (menace_sandbox)")
except ImportError:  # pragma: no cover - fallback for flat layout
    import security_auditor  # type: ignore
    _qfe_log("security_auditor imported (flat layout)")
try:  # pragma: no cover - optional dependency
    import sandbox_runner.environment as environment
    _ENVIRONMENT_ERROR: Exception | None = None
    _qfe_log("sandbox_runner.environment imported")
except Exception as exc:  # pragma: no cover - explicit guidance for users
    class _EnvironmentStub:
        current_context = None

        @staticmethod
        def auto_include_modules(*_args: object, **_kwargs: object) -> tuple[None, list[object]]:
            return None, []

        @staticmethod
        def run_workflow_simulations(*_args: object, **_kwargs: object) -> tuple[None, dict[str, object]]:
            return None, {}

        @staticmethod
        def try_integrate_into_workflows(*_args: object, **_kwargs: object) -> bool:
            return False

    environment = _EnvironmentStub()  # type: ignore[assignment]
    _ENVIRONMENT_ERROR = exc
    _qfe_log("sandbox_runner.environment import failed; using stub")


try:
    from .orchestration import (
        integrate_orphans,
        post_round_orphan_scan,
        self_improvement_cycle,
        start_self_improvement_cycle,
        stop_self_improvement_cycle,
    )
    _ORCHESTRATION_ERROR: ImportError | None = None
    _qfe_log("self_improvement.orchestration imported")
except ImportError as exc:
    _ORCHESTRATION_ERROR = exc
    _qfe_log("self_improvement.orchestration import failed; using stubs")

    async def self_improvement_cycle(*_a: object, **_k: object) -> None:
        return None

    def start_self_improvement_cycle(*_a: object, **_k: object) -> None:
        return None

    def stop_self_improvement_cycle(*_a: object, **_k: object) -> None:
        return None

    def integrate_orphans(*_a: object, **_k: object) -> None:
        return None

    def post_round_orphan_scan(*_a: object, **_k: object) -> None:
        return None

try:
    from .roi_tracking import update_alignment_baseline
    _ROI_TRACKING_ERROR = None
    _qfe_log("self_improvement.roi_tracking imported")
except ImportError as exc:
    _ROI_TRACKING_ERROR = exc

    _qfe_log("self_improvement.roi_tracking import failed")

    def update_alignment_baseline(*_a: object, **_k: object) -> None:
        return None

try:
    from .patch_application import generate_patch, apply_patch
    _PATCH_APP_ERROR = None
    _qfe_log("self_improvement.patch_application imported")
except ImportError as exc:
    _PATCH_APP_ERROR = exc

    _qfe_log("self_improvement.patch_application import failed")

    def generate_patch(*_a: object, **_k: object) -> dict[str, object]:
        return {}

    def apply_patch(*_a: object, **_k: object) -> bool:
        return False
from .prompt_memory import log_prompt_attempt
_qfe_log("self_improvement.prompt_memory imported")
from .prompt_strategies import PromptStrategy
_qfe_log("self_improvement.prompt_strategies imported")
from .prompt_strategy_manager import PromptStrategyManager
_qfe_log("self_improvement.prompt_strategy_manager imported")
from .strategy_analytics import StrategyAnalytics
_qfe_log("self_improvement.strategy_analytics imported")
from .snapshot_tracker import (
    capture as capture_snapshot,
    compute_delta as snapshot_delta,
    SnapshotTracker,
)
_qfe_log("self_improvement.snapshot_tracker imported")
from . import snapshot_tracker
_qfe_log("self_improvement.snapshot_tracker module imported")
from .sandbox_score import get_latest_sandbox_score
_qfe_log("self_improvement.sandbox_score imported")
from db_router import DBRouter
_qfe_log("db_router.DBRouter imported")


from menace_sandbox.self_test_service import SelfTestService
_qfe_log("menace_sandbox.self_test_service imported")
try:
    from menace_sandbox import self_test_service as sts
    _qfe_log("self_test_service module imported (menace_sandbox)")
except ImportError:  # pragma: no cover - fallback for flat layout
    import self_test_service as sts  # type: ignore
    _qfe_log("self_test_service module imported (flat layout)")
from orphan_analyzer import classify_module, analyze_redundancy
_qfe_log("orphan_analyzer imported")

try:  # pragma: no cover - numpy is optional in lightweight environments
    import numpy as np
    _NUMPY_ERROR: ImportError | None = None
    _qfe_log("numpy imported")
    print("[QFE:engine] post-numpy checkpoint reached", flush=True)
except ImportError as exc:  # pragma: no cover - provide stub accessor
    _NUMPY_ERROR = exc

    _qfe_log("numpy import failed")

    class _MissingNumpy:
        """Proxy that raises a helpful error when numpy-backed code is used."""

        def __getattr__(self, attr: str) -> typing.Any:  # type: ignore[name-defined]
            raise RuntimeError(
                "numpy is required for self_improvement.engine functionality"
            ) from _NUMPY_ERROR

        def __call__(self, *args: object, **kwargs: object) -> typing.NoReturn:
            raise RuntimeError(
                "numpy is required for self_improvement.engine functionality"
            ) from _NUMPY_ERROR

    np = _MissingNumpy()  # type: ignore[assignment]
import socket
print("[QFE:engine] socket import reached", flush=True)
import contextlib
print("[QFE:engine] contextlib import reached", flush=True)
import subprocess
print("[QFE:engine] subprocess import reached", flush=True)
from collections import deque
print("[QFE:engine] collections.deque import reached", flush=True)
from menace_sandbox.error_cluster_predictor import ErrorClusterPredictor
print("[QFE:engine] ErrorClusterPredictor import reached", flush=True)
from menace_sandbox import mutation_logger as MutationLogger
print("[QFE:engine] mutation_logger import reached", flush=True)
from menace_sandbox.gpt_memory import GPTMemoryManager
print("[QFE:engine] GPTMemoryManager import reached", flush=True)
from menace_sandbox.local_knowledge_module import init_local_knowledge
print("[QFE:engine] init_local_knowledge import reached", flush=True)
from gpt_memory_interface import GPTMemoryInterface
print("[QFE:engine] GPTMemoryInterface import reached", flush=True)
try:
    from menace_sandbox.gpt_knowledge_service import GPTKnowledgeService
except ImportError:  # pragma: no cover - fallback for flat layout
    from gpt_knowledge_service import GPTKnowledgeService  # type: ignore
try:  # canonical tag constants
    from menace_sandbox.log_tags import FEEDBACK, IMPROVEMENT_PATH, ERROR_FIX, INSIGHT
except ImportError:  # pragma: no cover - fallback for flat layout
    from log_tags import FEEDBACK, IMPROVEMENT_PATH, ERROR_FIX, INSIGHT  # type: ignore
try:  # helper for standardised GPT memory logging
    from menace_sandbox.memory_logging import log_with_tags, ensure_tags
except ImportError:  # pragma: no cover - fallback for flat layout
    from memory_logging import log_with_tags, ensure_tags  # type: ignore
try:  # pragma: no cover - allow flat imports
    from menace_sandbox.local_knowledge_module import LocalKnowledgeModule
except ImportError:  # pragma: no cover - fallback for flat layout
    from local_knowledge_module import LocalKnowledgeModule  # type: ignore
try:  # pragma: no cover - allow flat imports
    from menace_sandbox.knowledge_retriever import (
        get_feedback,
        get_error_fixes,
        recent_feedback,
        recent_improvement_path,
        recent_error_fix,
    )
except ImportError:  # pragma: no cover - fallback for flat layout
    from knowledge_retriever import (  # type: ignore
        get_feedback,
        get_error_fixes,
        recent_feedback,
        recent_improvement_path,
        recent_error_fix,
    )
try:  # pragma: no cover - allow flat imports
    from menace_sandbox.relevancy_radar import RelevancyRadar, scan as radar_scan, radar
except ImportError:  # pragma: no cover - fallback for flat layout
    from relevancy_radar import RelevancyRadar, scan as radar_scan, radar  # type: ignore
try:  # pragma: no cover - allow flat imports
    from menace_sandbox.module_retirement_service import ModuleRetirementService
except ImportError:  # pragma: no cover - fallback for flat layout
    try:
        from module_retirement_service import ModuleRetirementService  # type: ignore
    except ImportError:  # pragma: no cover - last resort
        ModuleRetirementService = object  # type: ignore
try:  # pragma: no cover - allow flat imports
    from menace_sandbox.relevancy_metrics_db import RelevancyMetricsDB
except ImportError:  # pragma: no cover - fallback for flat layout
    from relevancy_metrics_db import RelevancyMetricsDB  # type: ignore
try:  # pragma: no cover - allow flat imports
    from menace_sandbox.intent_clusterer import IntentClusterer
except ImportError:  # pragma: no cover - fallback for flat layout
    from intent_clusterer import IntentClusterer  # type: ignore
try:  # pragma: no cover - allow flat imports
    from menace_sandbox.universal_retriever import UniversalRetriever
except ImportError:  # pragma: no cover - fallback for flat layout
    from universal_retriever import UniversalRetriever  # type: ignore
try:  # pragma: no cover - optional planner integration
    from menace_sandbox.workflow_chain_suggester import WorkflowChainSuggester
except ImportError:  # pragma: no cover - fallback for flat layout
    try:
        from workflow_chain_suggester import WorkflowChainSuggester  # type: ignore
    except ImportError:  # pragma: no cover - best effort
        WorkflowChainSuggester = None  # type: ignore
try:  # pragma: no cover - optional planner integration
    from menace_sandbox.meta_workflow_planner import MetaWorkflowPlanner
except ImportError:  # pragma: no cover - fallback for flat layout
    try:
        from meta_workflow_planner import MetaWorkflowPlanner  # type: ignore
    except ImportError:  # pragma: no cover - best effort
        MetaWorkflowPlanner = None  # type: ignore
try:  # pragma: no cover - optional consumer
    from menace_sandbox.workflow_synthesizer import consume_planner_suggestions
except ImportError:  # pragma: no cover - fallback for flat layout
    try:
        from workflow_synthesizer import consume_planner_suggestions  # type: ignore
    except ImportError:  # pragma: no cover - best effort
        consume_planner_suggestions = None  # type: ignore
try:  # pragma: no cover - optional dependency
    from sandbox_runner.orphan_discovery import (
        append_orphan_classifications,
        append_orphan_cache,
        append_orphan_traces,
    )
except ImportError:  # pragma: no cover - best effort fallback
    append_orphan_classifications = append_orphan_cache = append_orphan_traces = None  # type: ignore
from menace_sandbox.human_alignment_flagger import (
    HumanAlignmentFlagger,
    flag_improvement,
    flag_alignment_issues,
    _collect_diff_data,
)
from menace_sandbox.human_alignment_agent import HumanAlignmentAgent
from menace_sandbox.audit_logger import log_event as audit_log_event, get_recent_events
from menace_sandbox.violation_logger import log_violation
from menace_sandbox.alignment_review_agent import AlignmentReviewAgent
from menace_sandbox.governance import check_veto, load_rules
try:  # pragma: no cover - allow flat imports
    from menace_sandbox.evaluation_dashboard import append_governance_result
except ImportError:  # pragma: no cover - fallback for flat layout or missing deps
    try:
        from evaluation_dashboard import append_governance_result  # type: ignore
    except ImportError:  # pragma: no cover - best effort fallback
        append_governance_result = lambda *a, **k: None  # type: ignore
try:  # pragma: no cover - allow flat imports
    from menace_sandbox.deployment_governance import evaluate as deployment_evaluate
except ImportError:  # pragma: no cover - fallback for flat layout or missing deps
    try:
        from deployment_governance import evaluate as deployment_evaluate  # type: ignore
    except ImportError:  # pragma: no cover - best effort fallback
        def deployment_evaluate(*_: object, **__: object) -> dict[str, object] | None:
            """Fallback deployment evaluation used when governance tools are missing."""
            return None
try:
    from menace_sandbox.borderline_bucket import BorderlineBucket
except ImportError:  # pragma: no cover - fallback for flat layout
    from borderline_bucket import BorderlineBucket  # type: ignore
try:  # pragma: no cover - allow flat imports
    from menace_sandbox.foresight_gate import ForesightDecision, is_foresight_safe_to_promote
except ImportError:  # pragma: no cover - fallback for flat layout
    from foresight_gate import ForesightDecision, is_foresight_safe_to_promote  # type: ignore
try:  # pragma: no cover - allow flat imports
    from menace_sandbox.upgrade_forecaster import UpgradeForecaster
except ImportError:  # pragma: no cover - fallback for flat layout
    from upgrade_forecaster import UpgradeForecaster  # type: ignore
try:  # pragma: no cover - allow flat imports
    from menace_sandbox.workflow_graph import WorkflowGraph
except ImportError:  # pragma: no cover - fallback for flat layout
    from workflow_graph import WorkflowGraph  # type: ignore
try:  # pragma: no cover - allow flat imports
    from menace_sandbox.forecast_logger import ForecastLogger, log_forecast_record
except ImportError:  # pragma: no cover - fallback for flat layout
    from forecast_logger import ForecastLogger, log_forecast_record  # type: ignore
try:  # pragma: no cover - allow flat imports
    from menace_sandbox.workflow_evolution_manager import WorkflowEvolutionManager
except ImportError:  # pragma: no cover - fallback for flat layout
    from workflow_evolution_manager import WorkflowEvolutionManager  # type: ignore

logger = get_logger(__name__)



_BOOTSTRAP_FN: Callable[..., int] | None = None


def _load_bootstrap() -> Callable[..., int]:
    """Resolve the optional :func:`self_model_bootstrap.bootstrap` helper."""

    global _BOOTSTRAP_FN
    if _BOOTSTRAP_FN is not None:
        return _BOOTSTRAP_FN

    try:  # pragma: no cover - optional dependency
        module = importlib.import_module("menace_sandbox.self_model_bootstrap")
    except ImportError as exc:
        try:  # pragma: no cover - fallback for flat layout
            module = importlib.import_module("self_model_bootstrap")
        except ImportError:
            logger.warning(
                "self_model_bootstrap unavailable",  # noqa: TRY300
                extra=log_record(
                    module=__name__, dependency="self_model_bootstrap"
                ),
                exc_info=exc,
            )

            def _missing(*_a: object, **_k: object) -> int:  # type: ignore
                raise RuntimeError(
                    "self_model_bootstrap module is required for bootstrapping"
                ) from exc

            _BOOTSTRAP_FN = _missing
            return _BOOTSTRAP_FN

    _BOOTSTRAP_FN = getattr(module, "bootstrap")
    return _BOOTSTRAP_FN


def bootstrap(*args: object, **kwargs: object) -> int:  # type: ignore[override]
    """Proxy that lazily imports :mod:`self_model_bootstrap` when required."""

    bootstrap_fn = _load_bootstrap()
    return bootstrap_fn(*args, **kwargs)
from menace_sandbox.research_aggregator_bot import (
    ResearchAggregatorBot,
    ResearchItem,
    InfoDB,
)
from menace_sandbox.model_automation_pipeline import (
    ModelAutomationPipeline,
    AutomationResult,
)
from context_builder_util import create_context_builder

if TYPE_CHECKING:  # pragma: no cover - imported for type checking only
    from vector_service.context_builder import ContextBuilder
from menace_sandbox.diagnostic_manager import DiagnosticManager
from menace_sandbox.error_bot import ErrorBot, ErrorDB
from menace_sandbox.data_bot import MetricsDB, DataBot
from menace_sandbox.code_database import PatchHistoryDB
from menace_sandbox.patch_score_backend import PatchScoreBackend, backend_from_url
from menace_sandbox.capital_management_bot import CapitalManagementBot
from menace_sandbox.learning_engine import LearningEngine
from menace_sandbox.unified_event_bus import UnifiedEventBus
from menace_sandbox.neuroplasticity import PathwayRecord, Outcome
from menace_sandbox.self_coding_engine import SelfCodingEngine
from .target_region import TargetRegion
from menace_sandbox.action_planner import ActionPlanner
from menace_sandbox.evolution_history_db import EvolutionHistoryDB
from menace_sandbox import synergy_weight_cli
from menace_sandbox import synergy_history_db as shd
try:  # pragma: no cover - optional dependency
    from menace_sandbox.adaptive_roi_predictor import AdaptiveROIPredictor, load_training_data
    _HAS_ADAPTIVE_ROI_PREDICTOR = True
except ImportError as exc:  # pragma: no cover - fallback for tests
    get_logger(__name__).warning(
        "adaptive_roi_predictor unavailable",  # noqa: TRY300
        extra=log_record(module=__name__, dependency="adaptive_roi_predictor"),
        exc_info=exc,
    )
    AdaptiveROIPredictor = object  # type: ignore

    def load_training_data(*a, **k):  # type: ignore
        return []

    _HAS_ADAPTIVE_ROI_PREDICTOR = False
from menace_sandbox.adaptive_roi_dataset import build_dataset
try:  # pragma: no cover - optional dependency
    from menace_sandbox.roi_tracker import ROITracker
    _ROI_TRACKER_ERROR: ImportError | None = None
except ImportError as exc:  # pragma: no cover - best effort fallback
    ROITracker = None  # type: ignore[assignment]
    _ROI_TRACKER_ERROR = exc
from menace_sandbox.foresight_tracker import ForesightTracker
from menace_sandbox.truth_adapter import TruthAdapter
from menace_sandbox.evaluation_history_db import EvaluationHistoryDB
from menace_sandbox.self_improvement_policy import (
    SelfImprovementPolicy,
    ConfigurableSelfImprovementPolicy,
    DQNStrategy,
    DoubleDQNStrategy,
    ActorCriticStrategy,
    torch as sip_torch,
)
from menace_sandbox.pre_execution_roi_bot import PreExecutionROIBot, BuildTask, ROIResult
from menace_sandbox.env_config import PRE_ROI_SCALE, PRE_ROI_BIAS, PRE_ROI_CAP
from .dashboards import SynergyDashboard, load_synergy_history
from . import metrics as _si_metrics
from .baseline_tracker import BaselineTracker, TRACKER as GLOBAL_BASELINE_TRACKER


from .learners import (
    SynergyWeightLearner,
    DQNSynergyLearner,
    DoubleDQNSynergyLearner,
    SACSynergyLearner,
    TD3SynergyLearner,
)


POLICY_STATE_LEN = 21


class MovingBaseline:
    """Lightweight moving-average tracker for recent ROI scores.

    The helper maintains a fixed-size window of the most recent values and
    exposes the current moving average.  It is intentionally simple to avoid
    pulling in the full :mod:`baseline_tracker` machinery when only a single
    metric needs to be tracked.
    """

    def __init__(self, window: int = 5) -> None:
        self.window = max(1, int(window))
        self._scores: deque[float] = deque(maxlen=self.window)

    def append(self, value: float) -> None:
        """Record a new ROI *value* for baseline calculations."""

        self._scores.append(float(value))

    def average(self) -> float:
        """Return the current moving average of recorded values."""

        if not self._scores:
            return 0.0
        return sum(self._scores) / len(self._scores)

    def to_list(self) -> list[float]:
        """Expose recorded values for persistence."""

        return list(self._scores)

__all__ = [
    "SelfImprovementEngine",
    "SACSynergyLearner",
    "TD3SynergyLearner",
    "BaselineTracker",
]
class SelfImprovementEngine:
    """Run the automation pipeline on a configurable bot."""

    def __init__(
        self,
        *,
        context_builder: ContextBuilder,
        interval: int = 3600,
        pipeline: ModelAutomationPipeline | None = None,
        bot_name: str = "menace",
        diagnostics: DiagnosticManager | None = None,
        info_db: InfoDB | None = None,
        capital_bot: "CapitalManagementBot" | None = None,
        energy_threshold: float | None = None,
        baseline_window: int | None = None,
        roi_baseline_window: int | None = None,
        baseline_margin: float = 0.0,
        delta_score_dev_multiplier: float | None = None,
        recovery_threshold: float = 0.0,
        learning_engine: LearningEngine | None = None,
        self_coding_engine: SelfCodingEngine | None = None,
        action_planner: "ActionPlanner" | None = None,
        event_bus: UnifiedEventBus | None = None,
        evolution_history: EvolutionHistoryDB | None = None,
        data_bot: DataBot | None = None,
        patch_db: PatchHistoryDB | None = None,
        policy: SelfImprovementPolicy | None = None,
        policy_strategy: str | None = None,
        optimize_self: bool = False,
        meta_logger: object | None = None,
        module_index: "ModuleIndexDB" | None = None,
        module_clusters: dict[str, int] | None = None,
        module_groups: dict[str, str] | None = None,
        auto_refresh_map: bool = False,
        pre_roi_bot: PreExecutionROIBot | None = None,
        pre_roi_scale: float | None = None,
        pre_roi_bias: float | None = None,
        pre_roi_cap: float | None = None,
        synergy_weight_roi: float | None = None,
        synergy_weight_efficiency: float | None = None,
        synergy_weight_resilience: float | None = None,
        synergy_weight_antifragility: float | None = None,
        synergy_weight_reliability: float | None = None,
        synergy_weight_maintainability: float | None = None,
        synergy_weight_throughput: float | None = None,
        state_path: Path | str | None = None,
        roi_ema_alpha: float | None = None,
        roi_compounding_weight: float | None = None,
        entropy_window: int | None = None,
        entropy_weight: float | None = None,
        roi_weight: float | None = None,
        pass_rate_weight: float | None = None,
        momentum_weight: float | None = None,
        synergy_weights_path: Path | str | None = None,
        synergy_weights_lr: float | None = None,
        synergy_learner_cls: Type[SynergyWeightLearner] = SynergyWeightLearner,
        score_backend: PatchScoreBackend | None = None,
        error_predictor: ErrorClusterPredictor | None = None,
        roi_predictor: AdaptiveROIPredictor | None = None,
        roi_tracker: ROITracker | None = None,
        roi_db: ROIResultsDB | None = None,
        foresight_tracker: ForesightTracker | None = None,
        gpt_memory: GPTMemoryInterface | None = None,
        knowledge_service: GPTKnowledgeService | None = None,
        relevancy_radar: RelevancyRadar | None = None,
        intent_clusterer: IntentClusterer | None = None,
        workflow_evolver: WorkflowEvolutionManager | None = None,
        sandbox_integrate: Callable[..., Any] | None = None,
        orphan_scan: Callable[..., Any] | None = None,
        patch_generator: Callable[..., Any] | None = None,
        tau: float = 0.5,
        runner_config: Dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        if gpt_memory is None:
            gpt_memory = kwargs.get("gpt_memory_manager")
        self.interval = interval
        self.bot_name = bot_name
        self.info_db = info_db or InfoDB()
        self.context_builder = context_builder
        try:
            context_builder.refresh_db_weights()
        except Exception:
            pass
        self.aggregator = ResearchAggregatorBot(
            [bot_name], info_db=self.info_db, context_builder=context_builder
        )
        self.pipeline = pipeline or ModelAutomationPipeline(
            aggregator=self.aggregator,
            action_planner=action_planner,
            context_builder=context_builder,
        )
        self.action_planner = action_planner
        err_bot = ErrorBot(ErrorDB(), MetricsDB(), context_builder=context_builder)
        self.error_bot = err_bot
        self.diagnostics = diagnostics or DiagnosticManager(
            MetricsDB(), err_bot, context_builder=context_builder
        )
        self._alignment_agent: AlignmentReviewAgent | None = None
        self.last_run = 0.0
        self.capital_bot = capital_bot
        cfg = SandboxSettings()
        self.energy_threshold = (
            energy_threshold
            if energy_threshold is not None
            else getattr(cfg, "energy_deviation", 1.0)
        )
        self.baseline_window = (
            baseline_window
            if baseline_window is not None
            else getattr(cfg, "baseline_window", 10)
        )
        if not 5 <= self.baseline_window <= 10:
            raise ValueError("baseline_window must be between 5 and 10")
        self.baseline_tracker = GLOBAL_BASELINE_TRACKER
        self.baseline_tracker.window = self.baseline_window
        self.roi_weight = (
            roi_weight if roi_weight is not None else getattr(settings, "roi_weight", 1.0)
        )
        self.momentum_weight = (
            momentum_weight
            if momentum_weight is not None
            else getattr(settings, "momentum_weight", 1.0)
        )
        self.pass_rate_weight = (
            pass_rate_weight
            if pass_rate_weight is not None
            else getattr(settings, "pass_rate_weight", 1.0)
        )
        self.entropy_weight_scale = getattr(
            settings, "entropy_weight_scale", 0.0
        )
        self.momentum_weight_scale = getattr(
            settings, "momentum_weight_scale", 0.0
        )
        self.momentum_window = getattr(
            getattr(cfg, "roi", None), "momentum_window", self.baseline_window
        )
        self.momentum_dev_multiplier = getattr(
            getattr(cfg, "roi", None), "momentum_dev_multiplier", 1.0
        )
        self.roi_baseline_window = (
            roi_baseline_window
            if roi_baseline_window is not None
            else getattr(getattr(cfg, "roi", None), "baseline_window", 5)
        )
        self.roi_baseline = MovingBaseline(self.roi_baseline_window)
        self.baseline_margin = baseline_margin
        self.delta_score_dev_multiplier = (
            delta_score_dev_multiplier
            if delta_score_dev_multiplier is not None
            else getattr(cfg, "delta_score_deviation", 0.0)
        )
        self.urgency_recovery_threshold = recovery_threshold
        self.mae_dev_multiplier = getattr(cfg, "mae_deviation", 1.0)
        self.acc_dev_multiplier = getattr(cfg, "acc_deviation", 1.0)
        self.borderline_dev_multiplier = getattr(cfg, "roi_deviation", 1.0)
        self.entropy_dev_multiplier = getattr(cfg, "entropy_deviation", 1.0)
        self.pass_rate_dev_multiplier = getattr(cfg, "pass_rate_deviation", 1.0)
        self.learning_engine = learning_engine
        self.self_coding_engine = self_coding_engine
        self.event_bus = event_bus
        self.evolution_history = evolution_history
        self.data_bot = data_bot
        self.patch_db = patch_db or (data_bot.patch_db if data_bot else None)
        try:
            self.roi_db = roi_db or ROIResultsDB()
        except Exception:
            self.roi_db = None
        if policy is None:
            policy = ConfigurableSelfImprovementPolicy(strategy=policy_strategy)
        self.policy = policy
        if self.policy and getattr(self.policy, "path", None):
            try:
                self.policy.load(self.policy.path)
            except Exception as exc:  # pragma: no cover - best effort
                self.logger.exception("policy load failed: %s", exc)
        self.optimize_self_flag = optimize_self
        self.meta_logger = meta_logger
        self.metrics_db = getattr(meta_logger, "metrics_db", None) if meta_logger else None
        if self.metrics_db is None:
            try:
                data_dir = _data_dir()
                self.metrics_db = RelevancyMetricsDB(data_dir / "relevancy_metrics.db")
            except Exception:
                self.metrics_db = None
        self.auto_refresh_map = bool(auto_refresh_map)
        self.intent_clusterer = intent_clusterer
        if self.intent_clusterer is None:
            try:
                self.intent_clusterer = IntentClusterer(UniversalRetriever())
            except Exception:
                self.intent_clusterer = None
        self.workflow_evolver = workflow_evolver or WorkflowEvolutionManager()
        self._sandbox_integrate = sandbox_integrate or integrate_orphans
        self._post_round_scan = orphan_scan or post_round_orphan_scan
        self._patch_generator = patch_generator or generate_patch
        self.runner_config = runner_config
        self.pre_roi_bot = pre_roi_bot
        self.pre_roi_scale = (
            pre_roi_scale if pre_roi_scale is not None else PRE_ROI_SCALE
        )
        self.pre_roi_bias = pre_roi_bias if pre_roi_bias is not None else PRE_ROI_BIAS
        self.pre_roi_cap = pre_roi_cap if pre_roi_cap is not None else PRE_ROI_CAP
        self.borderline_bucket = BorderlineBucket()
        self.workflow_ready = False
        self.workflow_high_risk = False
        self.workflow_risk: dict[str, object] | None = None
        self.tau = tau if tau is not None else getattr(
            settings, "borderline_confidence_threshold", 0.0
        )
        self.use_adaptive_roi = getattr(settings, "adaptive_roi_prioritization", True)
        if self.use_adaptive_roi and ROITracker is None:
            logger.warning(  # noqa: TRY300
                "roi_tracker unavailable; disabling adaptive ROI",
                extra=log_record(module=__name__, dependency="roi_tracker"),
                exc_info=_ROI_TRACKER_ERROR,
            )
            self.use_adaptive_roi = False
        if self.use_adaptive_roi:
            if _HAS_ADAPTIVE_ROI_PREDICTOR:
                self.roi_predictor = roi_predictor or AdaptiveROIPredictor()
                self.roi_tracker = roi_tracker or ROITracker(
                    confidence_threshold=self.tau,
                    raroi_borderline_threshold=self._raroi_threshold(),
                    borderline_bucket=self.borderline_bucket,
                )
                self._adaptive_roi_last_train = time.time()
            else:
                logger.warning(
                    "adaptive_roi_predictor missing - disabling adaptive ROI",
                    extra=log_record(module=__name__),
                )
                self.use_adaptive_roi = False
                self.roi_predictor = None
                self.roi_tracker = None
                self._adaptive_roi_last_train = 0.0
        else:
            self.roi_predictor = None
            self.roi_tracker = None
            self._adaptive_roi_last_train = 0.0
        self.adaptive_roi_train_interval = getattr(
            settings, "adaptive_roi_train_interval", 3600
        )
        metrics_db = data_bot.db if data_bot else MetricsDB()
        self.pathway_db = PathwayDB()
        if CompositeWorkflowScorer is None:
            raise RuntimeError(
                "CompositeWorkflowScorer unavailable; install ROI dependencies"
            ) from _COMPOSITE_SCORER_ERROR
        self.workflow_scorer = CompositeWorkflowScorer(
            metrics_db, self.pathway_db, tracker=self.roi_tracker
        )
        self.foresight_tracker = foresight_tracker or ForesightTracker()
        self.truth_adapter = TruthAdapter()
        self._truth_adapter_needs_retrain = False
        self.synergy_weight_roi = (
            synergy_weight_roi
            if synergy_weight_roi is not None
            else settings.synergy_weight_roi
        )
        self.synergy_weight_efficiency = (
            synergy_weight_efficiency
            if synergy_weight_efficiency is not None
            else settings.synergy_weight_efficiency
        )
        self.synergy_weight_resilience = (
            synergy_weight_resilience
            if synergy_weight_resilience is not None
            else settings.synergy_weight_resilience
        )
        self.synergy_weight_antifragility = (
            synergy_weight_antifragility
            if synergy_weight_antifragility is not None
            else settings.synergy_weight_antifragility
        )
        self.synergy_weight_reliability = (
            synergy_weight_reliability
            if synergy_weight_reliability is not None
            else getattr(settings, "synergy_weight_reliability", 1.0)
        )
        self.synergy_weight_maintainability = (
            synergy_weight_maintainability
            if synergy_weight_maintainability is not None
            else getattr(settings, "synergy_weight_maintainability", 1.0)
        )
        self.synergy_weight_throughput = (
            synergy_weight_throughput
            if synergy_weight_throughput is not None
            else getattr(settings, "synergy_weight_throughput", 1.0)
        )
        self.roi_ema_alpha = (
            roi_ema_alpha if roi_ema_alpha is not None else settings.roi_ema_alpha
        )
        self.roi_compounding_weight = (
            roi_compounding_weight
            if roi_compounding_weight is not None
            else getattr(settings, "roi_compounding_weight", 0.0)
        )
        self.entropy_weight = (
            entropy_weight
            if entropy_weight is not None
            else getattr(settings, "entropy_weight", 0.1)
        )
        self.entropy_window = (
            entropy_window
            if entropy_window is not None
            else getattr(settings, "entropy_window", 5)
        )
        self.entropy_baseline = MovingBaseline(self.entropy_window)
        self.entropy_delta_ema = 0.0
        self.growth_weighting = getattr(settings, "roi_growth_weighting", True)
        self.growth_multipliers = {
            "exponential": getattr(settings, "growth_multiplier_exponential", 3.0),
            "linear": getattr(settings, "growth_multiplier_linear", 2.0),
            "marginal": getattr(settings, "growth_multiplier_marginal", 1.0),
        }
        default_path = Path(settings.synergy_weight_file)
        self.synergy_weights_path = (
            Path(synergy_weights_path)
            if synergy_weights_path is not None
            else default_path
        )
        self.synergy_weights_lr = (
            synergy_weights_lr
            if synergy_weights_lr is not None
            else settings.synergy_weights_lr
        )
        self.auto_patch_high_risk = getattr(settings, "auto_patch_high_risk", True)
        if getattr(settings, "menace_mode", "").lower() == "autonomous":
            self.auto_patch_high_risk = True

        self.gpt_memory = (
            gpt_memory
            or getattr(self_coding_engine, "gpt_memory", None)
            or getattr(self_coding_engine, "gpt_memory_manager", None)
            or init_local_knowledge(
                SandboxSettings().gpt_memory_db
            ).memory
        )
        self.gpt_memory_manager = self.gpt_memory  # backward compatibility
        self.local_knowledge = LocalKnowledgeModule(
            manager=self.gpt_memory, service=knowledge_service
        )
        self.knowledge_service = self.local_knowledge.knowledge
        self.relevancy_radar = relevancy_radar or RelevancyRadar()
        self.relevancy_flags: dict[str, str] = {}
        self.entropy_ceiling_modules: set[str] = set()
        # Track when the relevancy radar last ran so we can evaluate at intervals
        self._last_relevancy_eval = 0.0
        try:
            # Allow settings to override the default cadence
            self.relevancy_eval_interval = getattr(
                settings, "relevancy_eval_interval", 3600.0
            )
        except Exception:  # pragma: no cover - fallback for minimal settings
            self.relevancy_eval_interval = 3600.0

        if synergy_learner_cls is SynergyWeightLearner:
            env_name = SandboxSettings().synergy_learner.lower()
            mapping = {
                "dqn": DQNSynergyLearner,
                "double": DoubleDQNSynergyLearner,
                "double_dqn": DoubleDQNSynergyLearner,
                "ddqn": DoubleDQNSynergyLearner,
                "sac": SACSynergyLearner,
                "td3": TD3SynergyLearner,
            }
            synergy_learner_cls = mapping.get(env_name, synergy_learner_cls)

        self.synergy_learner = synergy_learner_cls(
            self.synergy_weights_path,
            lr=self.synergy_weights_lr,
            settings=settings,
        )
        if synergy_weight_roi is None:
            self.synergy_weight_roi = self.synergy_learner.weights["roi"]
        else:
            self.synergy_learner.weights["roi"] = self.synergy_weight_roi
        if synergy_weight_efficiency is None:
            self.synergy_weight_efficiency = self.synergy_learner.weights["efficiency"]
        else:
            self.synergy_learner.weights["efficiency"] = self.synergy_weight_efficiency
        if synergy_weight_resilience is None:
            self.synergy_weight_resilience = self.synergy_learner.weights["resilience"]
        else:
            self.synergy_learner.weights["resilience"] = self.synergy_weight_resilience
        if synergy_weight_antifragility is None:
            self.synergy_weight_antifragility = self.synergy_learner.weights[
                "antifragility"
            ]
        else:
            self.synergy_learner.weights["antifragility"] = (
                self.synergy_weight_antifragility
            )
        if synergy_weight_reliability is None:
            self.synergy_weight_reliability = self.synergy_learner.weights[
                "reliability"
            ]
        else:
            self.synergy_learner.weights["reliability"] = (
                self.synergy_weight_reliability
            )
        if synergy_weight_maintainability is None:
            self.synergy_weight_maintainability = self.synergy_learner.weights[
                "maintainability"
            ]
        else:
            self.synergy_learner.weights["maintainability"] = (
                self.synergy_weight_maintainability
            )
        if synergy_weight_throughput is None:
            self.synergy_weight_throughput = self.synergy_learner.weights["throughput"]
        else:
            self.synergy_learner.weights["throughput"] = self.synergy_weight_throughput
        self.state_path = Path(state_path) if state_path else None
        if error_predictor is None:
            graph = getattr(getattr(self.error_bot, "error_logger", None), "graph", None)
            if graph is None:
                graph = getattr(self.error_bot, "graph", None)
            if graph is not None and hasattr(self.error_bot, "db"):
                try:
                    self.error_predictor = ErrorClusterPredictor(graph, self.error_bot.db)
                except Exception:
                    logger.exception(
                        "error predictor init failed",
                        extra=log_record(component="ErrorClusterPredictor"),
                    )
                    self.error_predictor = None
            else:
                self.error_predictor = None
        else:
            self.error_predictor = error_predictor
        self.roi_history: list[float] = []
        self.raroi_history: list[float] = []
        self.roi_group_history: dict[int, list[float]] = {}
        self.roi_delta_ema: float = 0.0
        self._last_momentum: float = 0.0
        self.urgency_tier: int = 0
        self.stagnation_cycles: int = getattr(cfg.roi, "stagnation_cycles", 3)
        self.roi_stagnation_dev_multiplier: float = getattr(
            cfg.roi, "roi_stagnation_dev_multiplier", 1.0
        )
        # Tracks consecutive cycles without positive ROI delta
        self._roi_stagnation_count: int = 0
        self._momentum_streak: int = 0
        self.delta_score_history: deque[float] = deque(maxlen=self.baseline_window)
        self._delta_score_streak: int = 0
        self._last_growth_type: str | None = None
        self._synergy_cache: dict | None = None
        self.alignment_flagger = HumanAlignmentFlagger()
        self.cycle_logs: list[dict[str, Any]] = []
        self.warning_summary: list[dict[str, Any]] = []
        self.strategy_confidence: Dict[str, int] = {}
        self.pending_strategy: str | None = None
        self.strategy_manager = PromptStrategyManager()
        self.strategy_analytics = StrategyAnalytics(manager=self.strategy_manager)
        self._snapshot_tracker = SnapshotTracker()
        self._last_delta: dict[str, float] | None = None
        self.logger = get_logger("SelfImprovementEngine")
        self._load_state()
        self._load_synergy_weights()
        from menace_sandbox.module_index_db import ModuleIndexDB

        auto_map = SandboxSettings().sandbox_auto_map
        if not auto_map and SandboxSettings().sandbox_autodiscover_modules:
            self.logger.warning(
                "SANDBOX_AUTODISCOVER_MODULES is deprecated; use SANDBOX_AUTO_MAP"
            )
            auto_map = True
        self.module_index = module_index or ModuleIndexDB(auto_map=auto_map)
        map_path = getattr(
            self.module_index,
            "path",
            _data_dir() / "module_map.json",
        )
        try:
            self._last_map_refresh = map_path.stat().st_mtime
        except Exception:
            self._last_map_refresh = 0.0
        if self.module_index and self.patch_db:
            try:
                repo = _repo_path()
                with self.patch_db._connect() as conn:
                    rows = conn.execute(
                        "SELECT DISTINCT filename FROM patch_history"
                    ).fetchall()
                mods: list[str] = []
                for r in rows:
                    p = Path(r[0])
                    if not p.is_absolute():
                        p = repo / p
                    try:
                        rel = p.relative_to(repo).as_posix()
                    except Exception:
                        rel = p.as_posix()
                    mods.append(rel)
                self.module_index.refresh(mods)
            except Exception:
                self.logger.exception("module map refresh failed during init")
        if module_clusters is None and self.module_index is not None:
            try:
                module_clusters = dict(getattr(self.module_index, "_map", {}))
            except Exception:
                module_clusters = None
        self.module_clusters: dict[str, int] = module_clusters or {}
        # Filled by ``_update_orphan_modules`` when recursive orphan discovery
        # finds new modules. Maps module paths to metadata such as parents and
        # redundancy classification.
        self.orphan_traces: dict[str, dict[str, Any]] = {}
        # queue of modules needing preventative fixes
        self._preventative_queue: list[str] = []

        if module_groups is None:
            try:
                repo_path = _repo_path()
                discovered = discover_module_groups(repo_path)
                module_groups = {
                    str(resolve_path(m if m.endswith(".py") else f"{m}.py")): grp
                    for grp, mods in discovered.items()
                    for m in mods
                }
            except Exception:
                module_groups = None

        if module_groups:
            grp_map: dict[str, int] = {}
            for mod, grp in module_groups.items():
                try:
                    idx = (
                        self.module_index.group_id(str(grp))
                        if self.module_index
                        else abs(hash(grp)) % 1000
                    )
                except Exception:
                    idx = abs(hash(grp)) % 1000
                grp_map[mod] = idx
            if self.module_index:
                try:
                    self.module_index.merge_groups(grp_map)
                    grp_map = {m: self.module_index.get(m) for m in grp_map}
                except Exception:
                    logger.exception("Unhandled exception in self_improvement")
            self.module_clusters.update(grp_map)
        logging.basicConfig(level=logging.INFO)
        self._score_backend: PatchScoreBackend | None = None
        if score_backend is not None:
            self._score_backend = score_backend
        else:
            backend_url = SandboxSettings().patch_score_backend_url
            if backend_url:
                try:
                    self._score_backend = backend_from_url(backend_url)
                except Exception:
                    self.logger.exception("patch score backend init failed")
        self._cycle_running = False
        self._schedule_task: asyncio.Task | None = None
        self._stop_event: asyncio.Event | None = None
        self._trainer_stop: threading.Event | None = None
        self._trainer_thread: threading.Thread | None = None
        self._cycle_count = 0
        self._last_mutation_id: int | None = None
        self._last_patch_id: int | None = None
        self._last_scenario_metrics: dict[str, float] = {}
        self._last_scenario_trend: dict[str, float] = {}
        self._scenario_pass_rate: float = 0.0
        self._pass_rate_delta: float = 0.0
        self._force_rerun = False
        if self.event_bus:
            if self.learning_engine:
                try:
                    self.event_bus.subscribe("pathway:new", self._on_new_pathway)
                except Exception as exc:
                    self.logger.exception(
                        "failed to subscribe to pathway events: %s", exc
                    )
            try:
                self.event_bus.subscribe(
                    "evolve:self_improve", lambda *_: self.run_cycle()
                )
            except Exception as exc:
                self.logger.exception(
                    "failed to subscribe to self_improve events: %s", exc
                )

        if SandboxSettings().auto_train_synergy:
            interval = SandboxSettings().auto_train_interval
            hist_file = resolve_path(settings.sandbox_data_dir) / "synergy_history.db"
            self._start_synergy_trainer(hist_file, interval)

        # Schedule periodic meta-planning runs on a background thread.  This
        # keeps the meta planner active even when the main self-improvement
        # cycle is idle.
        self._meta_planner_thread: threading.Thread | None = None
        self._meta_planner_stop: threading.Event | None = None
        if META_PLANNING_PERIOD > 0 and MetaWorkflowPlanner is not None:
            self._start_meta_planner_thread(float(META_PLANNING_PERIOD))

        # Optional background evolution loop using the meta planner
        self._meta_loop_thread: threading.Thread | None = None
        self._meta_loop_stop: threading.Event | None = None
        if (
            META_PLANNING_LOOP
            and MetaWorkflowPlanner is not None
            and self.workflow_evolver is not None
        ):
            self._start_meta_planning_loop(
                float(PLANNER_INTERVAL), float(META_IMPROVEMENT_THRESHOLD)
            )

    def _plan_cross_domain_chains(self, top_k: int = 3) -> list[list[str]]:
        """Use meta planner to suggest new cross-domain workflow chains."""
        if MetaWorkflowPlanner is None or WorkflowChainSuggester is None:
            self.logger.debug(
                "cross-domain chain planning skipped: planner or suggester missing"
            )
            return []
        try:
            builder = create_context_builder()
            planner = MetaWorkflowPlanner(context_builder=builder)
            schedule_specs: dict[str, dict[str, Any]] = {}
            candidate_ids: list[str] = []
            for sched in Path.cwd().glob("*_schedule.json"):
                domain = sched.stem.replace("_schedule", "")
                try:
                    data = json.loads(sched.read_text())
                except Exception:
                    continue
                if not isinstance(data, list):
                    continue
                for spec in data:
                    if not isinstance(spec, dict):
                        continue
                    wid = str(
                        spec.get("id")
                        or spec.get("wid")
                        or spec.get("workflow_id")
                        or spec.get("name")
                        or ""
                    ).strip()
                    if not wid:
                        continue
                    seq = spec.get("workflow") or spec.get("task_sequence") or []
                    spec["workflow"] = seq
                    tags = list(spec.get("tags", []))
                    if domain not in tags:
                        tags.append(domain)
                    spec["tags"] = tags
                    spec["domain"] = domain
                    try:
                        planner.encode(wid, spec)
                    except Exception:
                        continue
                    schedule_specs[wid] = spec
                    candidate_ids.append(wid)
            target = planner.encode(
                "self_improvement", {"workflow": [], "domain": "self_improvement"}
            )
            suggester = WorkflowChainSuggester()
            chains = suggester.suggest_chains(target, top_k)
            if candidate_ids:
                chains.append(candidate_ids)
            try:
                from sandbox_runner.workflow_sandbox_runner import WorkflowSandboxRunner

                runner = WorkflowSandboxRunner()
                roi_db = ROIResultsDB()
                for chain in chains:
                    for wid in chain:
                        spec = schedule_specs.get(wid)
                        if not spec:
                            continue
                        seq = spec.get("workflow") or []
                        fn: Callable[[], Any] | None = None
                        if self.workflow_evolver:
                            try:
                                seq_str = "-".join(seq) if isinstance(seq, list) else str(seq)
                                fn = self.workflow_evolver.build_callable(seq_str)
                            except Exception:
                                fn = None
                        if fn is None:
                            fn = lambda: None
                        run_id = f"plan-{wid}"
                        try:
                            metrics = runner.run(fn, safe_mode=True)
                            runtime = float(getattr(metrics, "runtime", 0.0))
                            success = float(getattr(metrics, "success_rate", 1.0))
                            roi_db.log_result(
                                workflow_id=str(wid),
                                run_id=run_id,
                                runtime=runtime,
                                success_rate=success,
                                roi_gain=0.0,
                                workflow_synergy_score=0.0,
                                bottleneck_index=0.0,
                                patchability_score=0.0,
                            )
                        except Exception as exc:
                            roi_db.log_result(
                                workflow_id=str(wid),
                                run_id=run_id,
                                runtime=0.0,
                                success_rate=0.0,
                                roi_gain=0.0,
                                workflow_synergy_score=0.0,
                                bottleneck_index=0.0,
                                patchability_score=0.0,
                                failure_reason=str(exc),
                            )
            except Exception:
                logger.exception("Unhandled exception in self_improvement")
            return chains
        except Exception:
            return []

    def _discover_meta_workflows(self) -> list[dict[str, Any]]:
        """Run meta planner, evaluate chains and persist winners."""
        if MetaWorkflowPlanner is None or self.workflow_evolver is None:
            self.logger.debug(
                "meta workflow discovery skipped: planner or evolver missing"
            )
            return []
        try:
            builder = create_context_builder()
            planner = MetaWorkflowPlanner(context_builder=builder)
        except Exception:
            self.logger.exception("meta workflow planner instantiation failed")
            return []
        workflows: dict[str, Callable[[], Any]] = {}
        db: WorkflowDB | None = None
        if WorkflowDB is not None and WorkflowRecord is not None:
            try:
                from collections import defaultdict

                db = WorkflowDB(Path(SandboxSettings().workflows_db))
                domain_groups: dict[str, list[tuple[str, Callable[[], Any]]]] = defaultdict(list)
                for rec in db.fetch_workflows(limit=200):
                    seq = rec.get("workflow") or []
                    seq_str = "-".join(seq) if isinstance(seq, list) else str(seq)
                    wid = str(rec.get("id") or rec.get("wid") or "")
                    domain = rec.get("category") or ""
                    tags = rec.get("tags") or []
                    if not domain and isinstance(tags, list) and tags:
                        domain = str(tags[0])
                    domain_groups[domain or "uncategorized"].append(
                        (wid, self.workflow_evolver.build_callable(seq_str))
                    )
                for group in domain_groups.values():
                    if group:
                        wid, fn = group[0]
                        workflows[wid] = fn
            except Exception:
                self.logger.exception("failed loading workflows for meta discovery")
        records = planner.discover_and_persist(workflows, metrics_db=self.metrics_db)
        if not records:
            self.logger.debug("meta workflow discovery produced no records")
            return []
        if db is None and WorkflowDB is not None and WorkflowRecord is not None:
            try:
                db = WorkflowDB(Path(SandboxSettings().workflows_db))
            except Exception:
                db = None
        summary_db = None
        try:
            summary_db = WorkflowSummaryDB()
        except Exception:
            summary_db = None
        for rec in records:
            chain = rec.get("chain") or []
            roi = float(rec.get("roi_gain", 0.0))
            chain_id = "->".join(chain)
            if db is not None and chain:
                try:
                    db.add(WorkflowRecord(workflow=chain, status="meta"))
                except Exception:
                    self.logger.exception(
                        "meta workflow persistence failed",
                        extra=log_record(workflow_id=chain_id),
                    )
            if summary_db is not None and chain:
                try:
                    summary = f"{chain_id} (roi={roi:.3f})"
                    summary_db.set_summary(
                        abs(hash(chain_id)) % (10**9), summary
                    )
                except Exception:
                    self.logger.exception(
                        "meta workflow summary logging failed",
                        extra=log_record(workflow_id=chain_id),
                    )
            if self.evolution_history:
                try:
                    from menace_sandbox.evolution_history_db import EvolutionEvent

                    self.evolution_history.add(
                        EvolutionEvent(
                            action="meta_workflow",
                            before_metric=0.0,
                            after_metric=roi,
                            roi=roi,
                            reason="meta workflow discovery",
                            trigger="run_cycle",
                            workflow_id=abs(hash(chain_id)) % (10**9),
                        )
                    )
                except Exception:
                    self.logger.exception(
                        "meta workflow history logging failed",
                        extra=log_record(workflow_id=chain_id),
                    )
            if self.data_bot:
                baseline_roi = 0.0
                try:
                    baseline_roi = self.data_bot.roi(self.bot_name)
                    self.logger.info(
                        "baseline ROI",
                        extra=log_record(workflow_id=chain_id, roi=baseline_roi),
                    )
                except Exception:
                    self.logger.exception(
                        "baseline ROI lookup failed",
                        extra=log_record(workflow_id=chain_id),
                    )
                try:
                    self.data_bot.log_workflow_evolution(
                        workflow_id=abs(hash(chain_id)) % (10**9),
                        variant="meta",
                        baseline_roi=baseline_roi,
                        variant_roi=roi,
                    )
                except Exception:
                    self.logger.exception(
                        "meta workflow metrics logging failed",
                        extra=log_record(workflow_id=chain_id),
                    )
        return records

    def _execute_meta_planner(self) -> None:
        """Instantiate planner and run scheduled meta-pipelines."""
        if self.workflow_evolver is None:
            self.logger.debug(
                "meta planner execution skipped: workflow_evolver missing"
            )
            raise RuntimeError("workflow_evolver not initialised")
        planner_cls = (
            meta_planning.MetaWorkflowPlanner or meta_planning._FallbackPlanner
        )
        if planner_cls is meta_planning._FallbackPlanner:
            self.logger.debug(
                "MetaWorkflowPlanner unavailable; using fallback planner"
            )
        try:
            planner = (
                planner_cls(context_builder=create_context_builder())
                if planner_cls is meta_planning.MetaWorkflowPlanner
                else planner_cls()
            )
            workflows: dict[str, Callable[[], Any]] = {}
            if WorkflowDB is not None and WorkflowRecord is not None:
                try:
                    db = WorkflowDB(Path(SandboxSettings().workflows_db))
                    for rec in db.fetch_workflows(limit=200):
                        seq = rec.get("workflow") or []
                        seq_str = "-".join(seq) if isinstance(seq, list) else str(seq)
                        wid = str(rec.get("id") or rec.get("wid") or "")
                        workflows[wid] = self.workflow_evolver.build_callable(seq_str)
                except Exception:
                    self.logger.exception("failed loading workflows for meta planner")
            records = planner.schedule(workflows, metrics_db=self.metrics_db)
            for rec in records:
                chain = rec.get("chain") or []
                if not chain:
                    continue
                chain_id = "->".join(chain)
                roi = float(rec.get("roi_gain", 0.0))
                failures = int(rec.get("failures", 0))
                entropy = float(rec.get("entropy", 0.0))
                if roi > 0:
                    try:
                        with shd.connect_locked() as conn:
                            shd.insert_entry(
                                conn,
                                {"chain": chain_id, "roi_gain": roi, "failures": failures},
                            )
                    except Exception:
                        self.logger.exception(
                            "failed to save synergy history",
                            extra=log_record(workflow_id=chain_id),
                        )
                if planner.roi_db is not None:
                    try:
                        planner.roi_db.log_result(
                            workflow_id=chain_id,
                            run_id="bg",
                            runtime=0.0,
                            success_rate=1.0,
                            roi_gain=roi,
                            workflow_synergy_score=max(0.0, 1.0 - entropy),
                            bottleneck_index=0.0,
                            patchability_score=0.0,
                            module_deltas={},
                        )
                    except Exception:  # pragma: no cover - best effort
                        self.logger.exception(
                            "ROI logging failed", extra=log_record(workflow_id=chain_id)
                        )
                if planner.stability_db is not None:
                    try:
                        planner.stability_db.record_metrics(
                            chain_id, roi, failures, entropy, roi_delta=roi
                        )
                    except Exception:  # pragma: no cover - best effort
                        self.logger.exception(
                            "stability logging failed",
                            extra=log_record(workflow_id=chain_id),
                        )
        except Exception:
            self.logger.exception("meta planner execution failed")

    def _memory_summaries(self, key: str) -> str:
        """Return a summary of similar past actions from memory."""
        summaries: list[str] = []
        try:
            entries = get_feedback(self.gpt_memory, key, limit=5)
        except Exception:
            entries = []
        for ent in entries:
            resp = (getattr(ent, "response", "") or "").strip()
            tag = "success" if "success" in resp.lower() else "failure"
            snippet = resp.splitlines()[0]
            summaries.append(f"{tag}: {snippet}")
        try:
            fixes = get_error_fixes(self.gpt_memory, key, limit=3)
        except Exception:
            fixes = []
        for fix in fixes:
            resp = (getattr(fix, "response", "") or "").strip()
            if resp:
                summaries.append(f"fix: {resp.splitlines()[0]}")
        if getattr(self, "knowledge_service", None):
            try:
                insight = recent_feedback(self.knowledge_service)  # type: ignore[attr-defined]
                if insight:
                    summaries.append(f"{FEEDBACK} insight: {insight}")
            except Exception:
                logger.exception("Unhandled exception in self_improvement")
            try:
                insight = recent_improvement_path(self.knowledge_service)  # type: ignore[attr-defined]
                if insight:
                    summaries.append(f"{IMPROVEMENT_PATH} insight: {insight}")
            except Exception:
                logger.exception("Unhandled exception in self_improvement")
            try:
                insight = recent_error_fix(self.knowledge_service)  # type: ignore[attr-defined]
                if insight:
                    summaries.append(f"{ERROR_FIX} insight: {insight}")
            except Exception:
                logger.exception("Unhandled exception in self_improvement")
        return "\n".join(summaries)

    def _record_memory_outcome(
        self,
        module: str,
        action: str,
        success: bool,
        *,
        tags: Sequence[str] | None = None,
    ) -> None:
        try:
            outcome_tags = [
                f"self_improvement.{action}",
                FEEDBACK,
                IMPROVEMENT_PATH,
                ERROR_FIX,
                INSIGHT,
            ]
            if tags:
                outcome_tags.extend(tags)
            log_with_tags(
                self.gpt_memory,
                f"{action}:{module}",
                "success" if success else "failure",
                tags=outcome_tags,
            )
        except Exception:
            self.logger.exception("memory logging failed", extra=log_record(module=module))

    @radar.track
    def _generate_patch_with_memory(
        self,
        module: str,
        action: str,
        *,
        tags: Sequence[str] | None = None,
        retries: int | None = None,
        delay: float | None = None,
        target_region: "TargetRegion | None" = None,
    ) -> int | None:
        start = time.perf_counter()
        error_trace: str | None = None
        failure_reason: str | None = None
        sandbox_metrics: dict[str, float] | None = None
        validation_summary: Dict[str, Any] = {}
        validation_failed = False
        tests_failed = False
        self._last_validation_summary = {}
        patch_record = None
        history = self._memory_summaries(module)
        if history:
            self.logger.info(
                "patch memory context",
                extra=log_record(module=module, history=history, tags=[INSIGHT]),
            )
        client = getattr(self.self_coding_engine, "llm_client", None)
        if client is not None:
            try:
                ask_tags = [ERROR_FIX, IMPROVEMENT_PATH]
                if tags:
                    ask_tags.extend(tags)
                key = f"self_improvement.{action}"
                full_tags = ensure_tags(key, ask_tags)
                mem_ctx = ""
                try:
                    mem_ctx = self.local_knowledge.build_context(key, limit=5)
                except Exception:
                    pass
                intent_meta: dict[str, object] = {"module": module, "user_query": action}
                if mem_ctx:
                    intent_meta["memory_context"] = mem_ctx
                intent_meta.setdefault("intent_tags", list(full_tags))
                prompt_obj = self.self_coding_engine.build_enriched_prompt(
                    action,
                    intent=intent_meta,
                    context_builder=self.context_builder,
                )
                result = client.generate(
                    prompt_obj,
                    context_builder=self.context_builder,
                    tags=full_tags,
                )
                text = (result.text or "").strip()
                try:
                    log_prompt = "\n\n".join(
                        getattr(prompt_obj, "examples", []) + [prompt_obj.user]
                    )
                    if getattr(prompt_obj, "system", ""):
                        log_prompt = f"{prompt_obj.system}\n\n{log_prompt}"
                    log_with_tags(self.local_knowledge, log_prompt, text, full_tags)
                except Exception:
                    pass
                if text:
                    self.logger.info(
                        "gpt_suggestion",
                        extra=log_record(
                            module=module, suggestion=text, tags=[ERROR_FIX]
                        ),
                    )
            except Exception:
                self.logger.exception(
                    "gpt suggestion failed", extra=log_record(module=module)
                )
        gen_kwargs: dict[str, object] = {}
        if retries is not None:
            gen_kwargs["retries"] = retries
        if delay is not None:
            gen_kwargs["delay"] = delay
        if target_region is None:
            target_region = getattr(self, "_cycle_target_region", None)
        # Select the prompt strategy for this patch attempt
        try:
            strat_name = self.next_prompt_strategy()
        except Exception:
            strat_name = None

        def _last_prompt_with_strategy():
            prompt_obj = getattr(self.self_coding_engine, "_last_prompt", None)
            if prompt_obj and strat_name:
                meta = getattr(prompt_obj, "metadata", None)
                if isinstance(meta, dict) and "strategy" not in meta:
                    meta["strategy"] = strat_name
            return prompt_obj

        if strat_name:
            try:
                gen_kwargs["strategy"] = PromptStrategy(strat_name)
            except Exception:
                pass
        builder = self.self_coding_engine.context_builder
        patch_logger = (
            getattr(self.self_coding_engine, "patch_logger", None)
            or getattr(
                getattr(self.self_coding_engine, "cognition_layer", None),
                "patch_logger",
                None,
            )
        )
        gen_kwargs["context_builder"] = builder
        if patch_logger is not None:
            gen_kwargs["patch_logger"] = patch_logger
        patch_gen = getattr(self, "_patch_generator", generate_patch)
        try:
            sig = inspect.signature(patch_gen)
            params = sig.parameters
            has_varkw = any(
                p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()
            )
            if not has_varkw:
                gen_kwargs = {k: v for k, v in gen_kwargs.items() if k in params}
            call_kwargs = dict(gen_kwargs)
            if has_varkw or "target_region" in params:
                call_kwargs["target_region"] = target_region
            patch_id = patch_gen(
                module,
                self.self_coding_engine,
                **call_kwargs,
            )
        except RuntimeError as exc:
            self.logger.error("quick_fix_engine unavailable: %s", exc)
            raise
        except Exception:
            error_trace = traceback.format_exc()
            self.logger.exception(
                "patch generation failed", extra=log_record(module=module)
            )
            failure_reason = "generation_error"
            try:
                prompt_obj = _last_prompt_with_strategy()
                log_prompt_attempt(
                    prompt_obj,
                    False,
                    {"error": error_trace},
                    failure_reason=failure_reason,
                    sandbox_metrics=sandbox_metrics,
                )
            except Exception:
                self.logger.exception("log_prompt_attempt failed")
            patch_id = None
        else:
            if patch_id is not None:
                pre_snap = capture_snapshot(
                    stage="pre",
                    files=list(Path(_repo_path()).rglob("*.py")),
                    roi=self.baseline_tracker.current("roi"),
                    sandbox_score=get_latest_sandbox_score(
                        SandboxSettings().sandbox_score_db
                    ),
                )
                commit_hash = ""
                previous_head = ""
                try:
                    previous_head = (
                        subprocess.run(
                            ["git", "rev-parse", "HEAD"],
                            cwd=str(_repo_path()),
                            capture_output=True,
                            text=True,
                            check=True,
                        ).stdout.strip()
                    )
                except Exception:
                    previous_head = ""
                try:
                    commit_hash, patch_diff = apply_patch(patch_id, _repo_path())
                except RuntimeError:
                    error_trace = traceback.format_exc()
                    self.logger.exception(
                        "patch application failed", extra=log_record(module=module)
                    )
                    failure_reason = "apply_error"
                    try:
                        prompt_obj = _last_prompt_with_strategy()
                        log_prompt_attempt(
                            prompt_obj,
                            False,
                            {"patch_id": patch_id, "error": error_trace},
                            failure_reason=failure_reason,
                            sandbox_metrics=sandbox_metrics,
                        )
                    except Exception:
                        self.logger.exception("log_prompt_attempt failed")
                    patch_id = None
                else:
                    post_snap = capture_snapshot(
                        stage="post",
                        files=list(Path(_repo_path()).rglob("*.py")),
                        roi=self.baseline_tracker.current("roi"),
                        sandbox_score=get_latest_sandbox_score(
                            SandboxSettings().sandbox_score_db
                        ),
                    )
                    delta_vals = snapshot_delta(pre_snap, post_snap)
                    metrics = (
                        "roi",
                        "sandbox_score",
                        "entropy",
                        "call_graph_complexity",
                        "token_diversity",
                    )
                    sandbox_metrics = {k: delta_vals.get(k, 0.0) for k in metrics}
                    improved = all(sandbox_metrics.get(k, 0.0) > 0 for k in metrics)
                    if improved:
                        try:
                            module_path = Path(
                                resolve_path(
                                    f"{module}.py" if Path(module).suffix == "" else module
                                )
                            )
                            snapshot_tracker.save_checkpoint(module_path, commit_hash)
                        except Exception:
                            self.logger.exception(
                                "checkpoint save failed", extra=log_record(module=module)
                            )
                        self.strategy_confidence[action] = (
                            self.strategy_confidence.get(action, 0) + 1
                        )
                        self.logger.info(
                            "strategy_confidence_increase",
                            extra=log_record(
                                strategy=action,
                                confidence=self.strategy_confidence[action],
                                module=module,
                            ),
                        )
                        self._save_state()
                    regressed_metric = next(
                        (k for k, v in sandbox_metrics.items() if v < 0), None
                    )
                    if regressed_metric:
                        failure_reason = (
                            "roi_drop"
                            if regressed_metric == "roi"
                            else f"{regressed_metric}_regression"
                        )
                        try:
                            prompt_obj = _last_prompt_with_strategy()
                            log_prompt_attempt(
                                prompt_obj,
                                False,
                                {"delta": delta_vals, "patch_diff": patch_diff},
                                failure_reason=failure_reason,
                                sandbox_metrics=sandbox_metrics,
                                commit_hash=commit_hash or None,
                            )
                        except Exception:
                            self.logger.exception("log_prompt_attempt failed")
                        self.logger.warning(
                            "patch regression",
                            extra=log_record(
                                module=module,
                                patch_id=patch_id,
                                delta=delta_vals,
                                patch_diff=patch_diff,
                            ),
                        )
                    repo_root = Path(_repo_path()).resolve()
                    if self.patch_db is not None and patch_record is None:
                        try:
                            patch_record = self.patch_db.get(patch_id)
                        except Exception:
                            patch_record = None
                    module_rel_path: Path | None = None
                    description_text = action
                    if patch_record is not None:
                        try:
                            if getattr(patch_record, "filename", None):
                                module_rel_path = Path(patch_record.filename)
                                if module_rel_path.is_absolute():
                                    try:
                                        module_rel_path = module_rel_path.relative_to(repo_root)
                                    except ValueError:
                                        module_rel_path = Path(module_rel_path.name)
                        except Exception:
                            module_rel_path = None
                        description_text = patch_record.description or description_text
                    if module_rel_path is None:
                        try:
                            candidate = Path(module)
                            if candidate.suffix:
                                if candidate.is_absolute():
                                    try:
                                        module_rel_path = candidate.relative_to(repo_root)
                                    except ValueError:
                                        module_rel_path = Path(candidate.name)
                                else:
                                    module_rel_path = candidate
                            else:
                                module_rel_path = Path(module.replace(".", "/") + ".py")
                        except Exception:
                            module_rel_path = None
                    quick_fix_summary: Dict[str, Any] | None = None
                    quick_fix_engine = getattr(self, "quick_fix", None) or getattr(
                        self.self_coding_engine, "quick_fix", None
                    )
                    if quick_fix_engine is None:
                        validation_summary.setdefault(
                            "quick_fix",
                            {"skipped": True, "reason": "unavailable"},
                        )
                    else:
                        cloned_summary: Dict[str, Any] = {
                            "module": str(module_rel_path) if module_rel_path else module,
                            "description": description_text,
                        }
                        if module_rel_path is None:
                            cloned_summary["skipped"] = True
                            cloned_summary["reason"] = "module_unresolved"
                        else:
                            try:
                                with tempfile.TemporaryDirectory() as tmp_dir:
                                    try:
                                        subprocess.run(
                                            ["git", "clone", str(repo_root), tmp_dir],
                                            check=True,
                                            capture_output=True,
                                            text=True,
                                        )
                                    except Exception:
                                        raise RuntimeError("clone_failed")
                                    clone_root = Path(tmp_dir).resolve()
                                    cloned_module = clone_root / module_rel_path
                                    if not cloned_module.exists():
                                        cloned_summary["skipped"] = True
                                        cloned_summary["reason"] = "module_missing"
                                    else:
                                        prev_cwd = os.getcwd()
                                        os.chdir(str(clone_root))
                                        try:
                                            apply_kwargs: dict[str, Any] = {}
                                            try:
                                                sig = inspect.signature(
                                                    quick_fix_engine.apply_validated_patch
                                                )
                                                if "provenance_token" in sig.parameters:
                                                    provenance_token = (
                                                        getattr(self, "_provenance_token", None)
                                                        or getattr(
                                                            self.self_coding_engine,
                                                            "_provenance_token",
                                                            None,
                                                        )
                                                        or getattr(
                                                            self.self_coding_engine,
                                                            "provenance_token",
                                                            None,
                                                        )
                                                        or getattr(self, "bot_name", "self_improvement")
                                                    )
                                                    apply_kwargs["provenance_token"] = provenance_token
                                            except Exception:
                                                pass
                                            passed, validated_patch_id, flags = (
                                                quick_fix_engine.apply_validated_patch(
                                                    str(cloned_module),
                                                    description_text,
                                                    {
                                                        "trigger": action,
                                                        "module": str(module_rel_path),
                                                    },
                                                    **apply_kwargs,
                                                )
                                            )
                                        finally:
                                            os.chdir(prev_cwd)
                                        cloned_summary.update(
                                            {
                                                "passed": bool(passed),
                                                "flags": list(flags),
                                                "patch_id": validated_patch_id,
                                            }
                                        )
                                        if not passed or flags:
                                            validation_failed = True
                                            failure_reason = failure_reason or (
                                                "quick_fix_generation_error"
                                                if not passed
                                                else "quick_fix_flags"
                                            )
                            except RuntimeError:
                                validation_failed = True
                                cloned_summary["error"] = "clone_failed"
                                failure_reason = failure_reason or "quick_fix_error"
                            except Exception:
                                validation_failed = True
                                cloned_summary["error"] = traceback.format_exc()
                                failure_reason = failure_reason or "quick_fix_error"
                                self.logger.exception(
                                    "quick fix validation failed",
                                    extra=log_record(module=module, patch_id=patch_id),
                                )
                        quick_fix_summary = cloned_summary
                        validation_summary["quick_fix"] = cloned_summary
                    passed_modules: list[str] = []
                    if not validation_failed:
                        pytest_args: str | None = None
                        svc_kwargs: dict[str, Any] = {}
                        workflow_tests: list[str] = []
                        workflow_sources: dict[str, list[str]] = {}
                        resolver = None
                        if self.self_coding_engine is not None:
                            resolver = getattr(
                                self.self_coding_engine, "_workflow_test_service_args", None
                            )
                        if resolver is None:
                            resolver = getattr(self, "_workflow_test_service_args", None)
                        if callable(resolver):
                            try:
                                (
                                    pytest_args,
                                    svc_kwargs,
                                    workflow_tests,
                                    workflow_sources,
                                ) = resolver()
                            except Exception:
                                self.logger.exception(
                                    "workflow test args resolution failed",
                                    extra=log_record(module=module, patch_id=patch_id),
                                )
                                pytest_args, svc_kwargs, workflow_tests, workflow_sources = (
                                    None,
                                    {},
                                    [],
                                    {},
                                )
                        svc_kwargs = dict(svc_kwargs or {})
                        if pytest_args is not None:
                            svc_kwargs["pytest_args"] = pytest_args
                        builder = getattr(self.self_coding_engine, "context_builder", None)
                        if builder is not None:
                            svc_kwargs.setdefault("context_builder", builder)
                        if "context_builder" not in svc_kwargs:
                            svc_kwargs["context_builder"] = getattr(
                                self, "context_builder", None
                            )
                        data_bot_candidate = getattr(
                            self.self_coding_engine, "data_bot", None
                        ) or self.data_bot
                        if data_bot_candidate is not None:
                            svc_kwargs.setdefault("data_bot", data_bot_candidate)
                        try:
                            service = SelfTestService(**svc_kwargs)
                        except Exception:
                            validation_failed = True
                            failure_reason = failure_reason or "self_test_init_failed"
                            validation_summary["self_tests"] = {
                                "error": "initialization_failed",
                                "pytest_args": svc_kwargs.get("pytest_args"),
                            }
                        else:
                            results, passed_modules = service.run_once()
                            failed_count = int(results.get("failed", 0))
                            tests_summary = {
                                "passed": int(results.get("passed", 0)),
                                "failed": failed_count,
                                "coverage": float(results.get("coverage", 0.0)),
                                "runtime": float(results.get("runtime", 0.0)),
                                "pytest_args": svc_kwargs.get("pytest_args"),
                                "passed_modules": passed_modules,
                            }
                            if workflow_tests:
                                tests_summary["workflow_tests"] = list(workflow_tests)
                            if workflow_sources:
                                tests_summary["workflow_sources"] = {
                                    key: list(values)
                                    for key, values in workflow_sources.items()
                                }
                            executed = results.get("workflow_tests")
                            if executed:
                                tests_summary["executed_workflows"] = list(executed)
                            if results.get("retry_errors"):
                                tests_summary["retry_errors"] = results["retry_errors"]
                            if results.get("module_metrics"):
                                tests_summary["module_metrics"] = results["module_metrics"]
                            failed_modules = results.get("orphan_failed_modules") or []
                            if failed_modules:
                                tests_summary["failed_modules"] = list(failed_modules)
                            validation_summary["self_tests"] = tests_summary
                            tests_failed = failed_count > 0
                            if tests_failed:
                                validation_failed = True
                                failure_reason = failure_reason or "tests_failed"
                    if validation_summary:
                        self._last_validation_summary = dict(validation_summary)
                        if sandbox_metrics is None:
                            sandbox_metrics = {}
                        sandbox_metrics.setdefault("validation", validation_summary)
                        if "self_tests" in validation_summary:
                            sandbox_metrics["tests_passed"] = not tests_failed
                        elif validation_failed:
                            sandbox_metrics["tests_passed"] = False
                    if validation_failed:
                        if sandbox_metrics is None:
                            sandbox_metrics = {}
                        sandbox_metrics["reverted"] = True
                        if previous_head:
                            try:
                                subprocess.run(
                                    ["git", "reset", "--hard", previous_head],
                                    cwd=str(_repo_path()),
                                    check=True,
                                    capture_output=True,
                                    text=True,
                                )
                                commit_hash = previous_head
                            except Exception:
                                self.logger.exception(
                                    "failed to revert patch after validation",
                                    extra=log_record(module=module, patch_id=patch_id),
                                )
                        patch_id = None
        elapsed = time.perf_counter() - start
        if self.metrics_db:
            try:
                self.metrics_db.record(
                    module, elapsed, self.module_index, tags, roi_delta=0.0
                )
            except Exception:
                self.logger.exception(
                    "relevancy metrics record failed", extra=log_record(module=module)
                )
        try:
            log_tags = [
                f"self_improvement.{action}",
                FEEDBACK,
                IMPROVEMENT_PATH,
                ERROR_FIX,
                INSIGHT,
            ]
            if tags:
                log_tags.extend(tags)
            log_with_tags(
                self.gpt_memory,
                f"{action}:{module}",
                f"patch_id={patch_id}",
                tags=log_tags,
            )
        except Exception:
            self.logger.exception(
                "memory logging failed", extra=log_record(module=module)
            )
        self._record_memory_outcome(
            module, action, patch_id is not None, tags=tags
        )

        success = False
        roi_meta: Dict[str, Any] = {"roi_delta": 0.0}
        exec_res: Dict[str, Any] = {"patch_id": patch_id}
        if validation_summary:
            exec_res["validation"] = dict(validation_summary)
            roi_meta.setdefault("validation", dict(validation_summary))
            if "self_tests" in validation_summary:
                tests_meta = validation_summary["self_tests"]
                tests_passed_flag = bool(tests_meta.get("failed", 0) == 0)
                roi_meta["tests_passed"] = tests_passed_flag
                if not tests_passed_flag:
                    failed_modules = tests_meta.get("failed_modules")
                    if failed_modules:
                        roi_meta["failed_tests"] = list(failed_modules)
            elif validation_failed:
                roi_meta.setdefault("tests_passed", False)
        if validation_failed:
            exec_res.setdefault("reverted", True)
        if patch_id is not None and self.patch_db is not None:
            try:
                rec = self.patch_db.get(patch_id)
            except Exception:
                rec = None
            if rec is not None:
                success = not rec.reverted
                exec_res["reverted"] = rec.reverted
                roi_meta["roi_delta"] = rec.roi_delta
                try:
                    conn = self.patch_db.router.get_connection("patch_history")
                    row = conn.execute(
                        "SELECT tests_passed FROM patch_history WHERE id=?",
                        (patch_id,),
                    ).fetchone()
                    if row:
                        roi_meta.setdefault("tests_passed", bool(row[0]))
                except Exception:
                    self.logger.exception(
                        "failed to fetch patch test result",
                        extra=log_record(module=module, patch_id=patch_id),
                    )
                    raise
        if not success:
            fails: List[str] = []
            if error_trace:
                fails.append(error_trace)
            trace = getattr(self.self_coding_engine, "_last_retry_trace", None)
            if trace:
                fails.append(trace)
            if fails:
                exec_res["failures"] = fails
            if roi_meta.get("tests_passed") is False:
                failure_reason = "tests_failed"
            elif failure_reason is None and roi_meta.get("roi_delta", 0.0) < 0:
                failure_reason = "roi_drop"
        try:
            prompt_obj = _last_prompt_with_strategy()
            meta = dict(roi_meta)
            if "roi" not in meta and "roi_delta" in meta:
                meta["roi"] = meta.get("roi_delta", 0.0)
            log_success = success and failure_reason is None
            metrics_to_log = None
            if not log_success:
                metrics_to_log = dict(sandbox_metrics or {})
                if "tests_passed" in roi_meta:
                    metrics_to_log.setdefault("tests_passed", roi_meta["tests_passed"])
            log_prompt_attempt(
                prompt_obj,
                log_success,
                exec_res,
                roi_meta,
                failure_reason=None if log_success else failure_reason,
                sandbox_metrics=metrics_to_log,
                commit_hash=commit_hash or None,
            )
        except Exception:
            self.logger.exception("log_prompt_attempt failed")
        return patch_id

    # ------------------------------------------------------------------
    def recent_scores(self, limit: int = 20) -> list[tuple]:
        """Return recently stored patch scores."""
        if self._score_backend:
            try:
                rows = self._score_backend.fetch_recent(limit)
                if rows:
                    return rows
            except Exception:
                self.logger.exception("patch score backend fetch failed")
        return []

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    def _raroi_threshold(self) -> float:
        """Dynamic borderline ROI threshold based on baseline statistics."""
        base = self.roi_baseline.average()
        std = self.baseline_tracker.std("roi")
        momentum = self.baseline_tracker.momentum
        scale = 1.0 + (0.5 - momentum)
        return base - self.borderline_dev_multiplier * scale * std

    def _load_state(self) -> None:
        if not self.state_path or not self.state_path.exists():
            self.logger.debug("state file missing; skipping load")
            return
        try:
            with open(self.state_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            self.roi_history = [float(x) for x in data.get("roi_history", [])]
            self.raroi_history = [float(x) for x in data.get("raroi_history", [])]
            self.roi_group_history = {
                int(k): [float(vv) for vv in v]
                for k, v in data.get("roi_group_history", {}).items()
            }
            self.last_run = float(data.get("last_run", self.last_run))
            self.roi_delta_ema = float(data.get("roi_delta_ema", self.roi_delta_ema))
            self._pass_rate_delta = float(
                data.get("pass_rate_delta", self._pass_rate_delta)
            )
            self.entropy_delta_ema = float(
                data.get("entropy_delta_ema", self.entropy_delta_ema)
            )
            self.roi_weight = float(data.get("roi_weight", self.roi_weight))
            self.pass_rate_weight = float(
                data.get("pass_rate_weight", self.pass_rate_weight)
            )
            self.entropy_weight = float(
                data.get("entropy_weight", self.entropy_weight)
            )
            self.momentum_weight = float(
                data.get("momentum_weight", self.momentum_weight)
            )
            self.strategy_confidence = {
                str(k): int(v) for k, v in data.get("strategy_confidence", {}).items()
            }
            self.baseline_tracker = GLOBAL_BASELINE_TRACKER
            self.baseline_tracker.window = self.baseline_window
            bt_state = data.get("baseline_tracker")
            if isinstance(bt_state, dict):
                self.baseline_tracker.load_state(bt_state)
            else:
                # legacy sandbox_scores support
                scores = [float(x) for x in data.get("sandbox_scores", [])]
                for s in scores:
                    self.baseline_tracker.update(score=s)
            roi_scores = [float(x) for x in data.get("roi_baseline", [])]
            self.roi_baseline = MovingBaseline(self.roi_baseline_window)
            for v in roi_scores:
                self.roi_baseline.append(v)
        except Exception as exc:
            self.logger.exception("failed to load state: %s", exc)

    # ------------------------------------------------------------------
    def _save_state(self) -> None:
        if not self.state_path:
            self.logger.debug("state path not configured; skipping save")
            return
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                "w", delete=False, dir=self.state_path.parent, encoding="utf-8"
            ) as fh:
                json.dump(
                    {
                        "roi_history": self.roi_history,
                        "raroi_history": self.raroi_history,
                        "roi_group_history": self.roi_group_history,
                        "last_run": self.last_run,
                        "roi_delta_ema": self.roi_delta_ema,
                        "pass_rate_delta": self._pass_rate_delta,
                        "entropy_delta_ema": self.entropy_delta_ema,
                        "roi_weight": self.roi_weight,
                        "pass_rate_weight": self.pass_rate_weight,
                        "entropy_weight": self.entropy_weight,
                        "momentum_weight": self.momentum_weight,
                        "strategy_confidence": self.strategy_confidence,
                        "baseline_tracker": self.baseline_tracker.to_state(),
                        "roi_baseline": self.roi_baseline.to_list(),
                    },
                    fh,
                )
                tmp = Path(fh.name)
            os.replace(tmp, self.state_path)
        except Exception as exc:
            self.logger.exception("failed to save state: %s", exc)

    # ------------------------------------------------------------------
    def _load_synergy_weights(self) -> None:
        """Load persisted synergy weights from JSON file."""
        self.synergy_learner.load()
        self.synergy_weight_roi = self.synergy_learner.weights["roi"]
        self.synergy_weight_efficiency = self.synergy_learner.weights["efficiency"]
        self.synergy_weight_resilience = self.synergy_learner.weights["resilience"]
        self.synergy_weight_antifragility = self.synergy_learner.weights[
            "antifragility"
        ]
        self.synergy_weight_reliability = self.synergy_learner.weights["reliability"]
        self.synergy_weight_maintainability = self.synergy_learner.weights[
            "maintainability"
        ]
        self.synergy_weight_throughput = self.synergy_learner.weights["throughput"]
        self.logger.info(
            "synergy weights loaded",
            extra=log_record(weights=self.synergy_learner.weights),
        )

    # ------------------------------------------------------------------
    def _save_synergy_weights(self) -> None:
        """Persist synergy weights to JSON file."""
        self.synergy_learner.weights["roi"] = self.synergy_weight_roi
        self.synergy_learner.weights["efficiency"] = self.synergy_weight_efficiency
        self.synergy_learner.weights["resilience"] = self.synergy_weight_resilience
        self.synergy_learner.weights["antifragility"] = (
            self.synergy_weight_antifragility
        )
        self.synergy_learner.weights["reliability"] = self.synergy_weight_reliability
        self.synergy_learner.weights["maintainability"] = (
            self.synergy_weight_maintainability
        )
        self.synergy_learner.weights["throughput"] = self.synergy_weight_throughput
        self.synergy_learner.save()
        self.logger.info(
            "synergy weights saved",
            extra=log_record(weights=self.synergy_learner.weights),
        )

    # ------------------------------------------------------------------
    def _train_synergy_weights_once(self, history_file: Path) -> None:
        if not history_file.exists():
            return
        try:
            conn = router.get_connection("synergy_history")
            hist = shd.fetch_all(conn)
        except Exception as exc:  # pragma: no cover - runtime issues
            self.logger.exception("failed to load history: %s", exc)
            return
        if not hist:
            return
        try:
            synergy_weight_cli.train_from_history(hist, self.synergy_weights_path)
        except Exception as exc:  # pragma: no cover - runtime issues
            self.logger.exception("training failed: %s", exc)

    def _synergy_trainer_loop(self, history_file: Path, interval: float) -> None:
        assert self._trainer_stop is not None
        while not self._trainer_stop.is_set():
            self._train_synergy_weights_once(history_file)
            self._trainer_stop.wait(interval)

    def _start_synergy_trainer(self, history_file: Path, interval: float) -> None:
        if self._trainer_thread:
            return
        self.logger.info(
            "starting synergy trainer thread",
            extra=log_record(history_file=str(history_file), interval=float(interval)),
        )
        self._trainer_stop = threading.Event()
        self._trainer_thread = threading.Thread(
            target=self._synergy_trainer_loop,
            args=(history_file, interval),
            daemon=True,
        )
        self._trainer_thread.start()

    def stop_synergy_trainer(self) -> None:
        if self._trainer_thread and self._trainer_stop:
            self.logger.info("stopping synergy trainer thread")
            self._trainer_stop.set()
            self._trainer_thread.join(timeout=1.0)
            self._trainer_thread = None
            self._trainer_stop = None

    def _meta_planner_loop(self, interval: float) -> None:
        """Background loop that periodically executes the meta planner."""
        assert self._meta_planner_stop is not None
        while not self._meta_planner_stop.is_set():
            try:
                self._execute_meta_planner()
            except Exception:  # pragma: no cover - keep running on failure
                self.logger.exception("periodic meta planner run failed")
            self._meta_planner_stop.wait(interval)

    def _start_meta_planner_thread(self, interval: float) -> None:
        if self._meta_planner_thread:
            self.logger.debug("meta planner thread already running; skip start")
            return
        self._meta_planner_stop = threading.Event()
        self._meta_planner_thread = threading.Thread(
            target=self._meta_planner_loop,
            args=(interval,),
            daemon=True,
        )
        self._meta_planner_thread.start()

    def stop_meta_planner_thread(self) -> None:
        if self._meta_planner_thread and self._meta_planner_stop:
            self._meta_planner_stop.set()
            self._meta_planner_thread.join(timeout=1.0)
            self._meta_planner_thread = None
            self._meta_planner_stop = None

    # ------------------------------------------------------------------
    def _meta_planning_loop(
        self, interval: float, improvement_threshold: float
    ) -> None:
        """Background evolution loop driven by :class:`MetaWorkflowPlanner`."""

        assert self._meta_loop_stop is not None
        if self.workflow_evolver is None:
            self.logger.debug(
                "meta planning loop skipped: workflow_evolver missing"
            )
            return
        planner_cls = (
            meta_planning.MetaWorkflowPlanner or meta_planning._FallbackPlanner
        )
        if planner_cls is meta_planning._FallbackPlanner:
            self.logger.debug(
                "MetaWorkflowPlanner unavailable; using fallback planner"
            )
        planner = (
            planner_cls(context_builder=create_context_builder())
            if planner_cls is meta_planning.MetaWorkflowPlanner
            else planner_cls()
        )

        while not self._meta_loop_stop.is_set():
            try:
                workflows: dict[str, Callable[[], Any]] = {}
                if WorkflowDB is not None and WorkflowRecord is not None:
                    try:
                        db = WorkflowDB(Path(SandboxSettings().workflows_db))
                        for rec in db.fetch_workflows(limit=200):
                            seq = rec.get("workflow") or []
                            seq_str = "-".join(seq) if isinstance(seq, list) else str(seq)
                            wid = str(rec.get("id") or rec.get("wid") or "")
                            workflows[wid] = self.workflow_evolver.build_callable(seq_str)
                    except Exception:
                        self.logger.exception(
                            "failed loading workflows for meta planning loop"
                        )

                records = planner.discover_and_persist(
                    workflows, metrics_db=self.metrics_db
                )
                active = [r.get("chain", []) for r in records if r.get("chain")]

                while active and not self._meta_loop_stop.is_set():
                    mutated = planner.mutate_chains(active, workflows)
                    refined = planner.refine_chains(mutated, workflows)
                    chains = [r.get("chain", []) for r in refined if r.get("chain")]
                    if len(chains) > 1:
                        refined += planner.remerge_pipelines(chains, workflows)
                        chains = [r.get("chain", []) for r in refined if r.get("chain")]

                    improved = any(
                        (
                            abs(
                                float(
                                    planner.cluster_map.get(tuple(c), {}).get(
                                        "delta_roi", 0.0
                                    )
                                )
                            )
                            > improvement_threshold
                            or abs(
                                float(
                                    planner.cluster_map.get(tuple(c), {}).get(
                                        "delta_failures", 0.0
                                    )
                                )
                            )
                            > improvement_threshold
                            or abs(
                                float(
                                    planner.cluster_map.get(tuple(c), {}).get(
                                        "delta_entropy", 0.0
                                    )
                                )
                            )
                            > improvement_threshold
                        )
                        for c in chains
                    )

                    if not improved:
                        try:
                            dispatch_alert(
                                f"Meta planning stagnation: no chain improved beyond {improvement_threshold}"
                            )
                        except Exception:
                            self.logger.exception(
                                "failed to dispatch meta planning alert"
                            )

                    if all(
                        planner.cluster_map.get(tuple(c), {}).get("converged")
                        for c in chains
                    ):
                        break
                    active = chains
            except Exception:  # pragma: no cover - keep loop alive
                self.logger.exception("meta planning loop iteration failed")

            self._meta_loop_stop.wait(interval)

    def _start_meta_planning_loop(
        self, interval: float, improvement_threshold: float
    ) -> None:
        if self._meta_loop_thread:
            self.logger.debug("meta planning loop already running; skip start")
            return
        self._meta_loop_stop = threading.Event()
        self._meta_loop_thread = threading.Thread(
            target=self._meta_planning_loop,
            args=(interval, improvement_threshold),
            daemon=True,
        )
        self._meta_loop_thread.start()

    def stop_meta_planning_loop(self) -> None:
        if self._meta_loop_thread and self._meta_loop_stop:
            self._meta_loop_stop.set()
            self._meta_loop_thread.join(timeout=1.0)
            self._meta_loop_thread = None
            self._meta_loop_stop = None

    # ------------------------------------------------------------------
    def _metric_delta(self, name: str, window: int = 3) -> float:
        """Return rolling average delta for *name* metric."""
        tracker = getattr(self, "tracker", None) or getattr(self, "roi_tracker", None)
        if tracker is None:
            return 0.0
        try:
            vals = tracker.metrics_history.get(name, [])
        except Exception:
            return 0.0
        if not vals:
            return 0.0
        w = min(window, len(vals))
        current_avg = sum(vals[-w:]) / w
        if len(vals) > w:
            prev_w = min(w, len(vals) - w)
            prev_avg = sum(vals[-w - prev_w : -w]) / prev_w
        elif len(vals) >= 2:
            prev_avg = vals[-2]
        else:
            return float(vals[-1])
        return float(current_avg - prev_avg)

    # ------------------------------------------------------------------
    def _baseline_metric_delta(self, name: str, window: int = 3) -> float:
        """Return rolling average delta for a baseline tracker metric."""

        hist = self.baseline_tracker.to_dict().get(name, [])
        if not hist:
            return 0.0
        w = min(window, len(hist))
        current_avg = sum(hist[-w:]) / w
        if len(hist) > w:
            prev_w = min(w, len(hist) - w)
            prev_avg = sum(hist[-w - prev_w : -w]) / prev_w
        elif len(hist) >= 2:
            prev_avg = hist[-2]
        else:
            return float(hist[-1])
        return float(current_avg - prev_avg)

    # ------------------------------------------------------------------
    def _compute_delta_score(self) -> tuple[float, dict[str, float]]:
        """Return combined delta score and its components.

        The score combines ROI, pass rate, momentum, and entropy deltas with
        configurable weights. Positive ROI, pass rate, and momentum changes
        increase the score while entropy increases decrease it. Individual
        components are returned for audit logging.
        """

        roi_hist = self.baseline_tracker.delta_history("roi")
        roi_delta = roi_hist[-1] if roi_hist else 0.0
        pr_hist = self.baseline_tracker.delta_history("pass_rate")
        pass_rate_delta = pr_hist[-1] if pr_hist else 0.0
        pass_rate_std = self.baseline_tracker.std("pass_rate")
        pass_rate_threshold = self.pass_rate_dev_multiplier * pass_rate_std
        if abs(pass_rate_delta) <= pass_rate_threshold:
            pass_rate_delta = 0.0
        else:
            pass_rate_delta -= math.copysign(pass_rate_threshold, pass_rate_delta)
        ent_hist = self.baseline_tracker.delta_history("entropy")
        entropy_delta = ent_hist[-1] if ent_hist else 0.0
        entropy_std = self.baseline_tracker.std("entropy")
        entropy_threshold = self.entropy_dev_multiplier * entropy_std
        if abs(entropy_delta) <= entropy_threshold:
            entropy_delta = 0.0
        else:
            entropy_delta -= math.copysign(entropy_threshold, entropy_delta)
        momentum_delta = (
            self.baseline_tracker.momentum - self.baseline_tracker.get("momentum")
        )
        # Adapt weights based on recent variability and success trends.
        entropy_weight = self.entropy_weight * (
            1 + self.entropy_weight_scale * entropy_std
        )
        momentum_weight = self.momentum_weight * (
            1 + self.momentum_weight_scale * self.baseline_tracker.momentum
        )
        # Persist adaptive weights for historical analysis without advancing
        # momentum twice in a single cycle.
        if hasattr(self.baseline_tracker, "update"):
            self.baseline_tracker.update(
                record_momentum=False,
                entropy_weight=entropy_weight,
                momentum_weight=momentum_weight,
            )
        score = (
            self.roi_weight * roi_delta
            + self.pass_rate_weight * pass_rate_delta
            + momentum_weight * momentum_delta
            - entropy_weight * entropy_delta
        )
        components = {
            "roi_delta": roi_delta,
            "pass_rate_delta": pass_rate_delta,
            "entropy_delta": entropy_delta,
            "momentum_delta": momentum_delta,
            "roi_component": self.roi_weight * roi_delta,
            "pass_rate_component": self.pass_rate_weight * pass_rate_delta,
            "momentum_component": momentum_weight * momentum_delta,
            "entropy_component": -entropy_weight * entropy_delta,
        }
        return score, components

    def _evaluate_scenario_metrics(self, metrics: dict[str, float]) -> float:
        """Evaluate scenario metrics and trigger remediation based on deviations.

        Returns
        -------
        float
            Fraction of metrics that passed. The caller must update the
            :class:`~self_improvement.baseline_tracker.BaselineTracker` with this
            value and the provided ``metrics`` so future evaluations use the
            expanded history.
        """

        prev = getattr(self, "_last_scenario_metrics", {})
        trend = {k: float(v) - float(prev.get(k, 0.0)) for k, v in metrics.items()}
        failing: list[str] = []
        histories = self.baseline_tracker.to_dict()
        alert_mult = getattr(settings, "scenario_alert_dev_multiplier", 1.0)
        patch_mult = getattr(settings, "scenario_patch_dev_multiplier", 2.0)
        rerun_mult = getattr(settings, "scenario_rerun_dev_multiplier", 3.0)
        for name, val in metrics.items():
            avg = self.baseline_tracker.get(name)
            std = self.baseline_tracker.std(name)
            hist_len = len(histories.get(name, []))
            if hist_len == 0:
                continue
            delta = float(val) - avg
            ratio = abs(delta) / std if std else (float("inf") if delta else 0.0)
            action: str | None = None
            if ratio > rerun_mult:
                action = "rerun"
            elif ratio > patch_mult:
                action = "patch"
            elif ratio > alert_mult:
                action = "alert"
            if action:
                failing.append(name)
                self.logger.info(
                    "scenario_metric_action",
                    extra=log_record(
                        metric=name,
                        moving_avg=avg,
                        delta=delta,
                        action=action,
                    ),
                )
                if action == "alert":
                    try:
                        dispatch_alert(
                            "scenario_degradation",
                            2,
                            f"{name} degraded",
                            {"value": float(val), "moving_avg": avg, "delta": delta},
                        )
                    except Exception:
                        self.logger.exception("alert dispatch failed for %s", name)
                elif action == "patch":
                    try:
                        with tempfile.TemporaryDirectory() as before_dir, tempfile.TemporaryDirectory() as after_dir:
                            orig = Path(name)
                            rel = orig.name if orig.is_absolute() else orig
                            src = resolve_path(
                                f"{orig}.py" if orig.suffix == "" else str(orig)
                            )
                            before_target = Path(before_dir) / rel
                            before_target.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copy2(src, before_target)
                            self.logger.info(
                                "gpt_suggestion",
                                extra=log_record(
                                    module=name,
                                    suggestion="scenario_patch",
                                    tags=[ERROR_FIX],
                                ),
                            )
                            try:
                                log_with_tags(
                                    self.gpt_memory,
                                    f"scenario_patch:{name}",
                                    "suggested",
                                    tags=[f"self_improvement.scenario_patch", FEEDBACK, IMPROVEMENT_PATH, ERROR_FIX, INSIGHT],
                                )
                            except Exception:
                                self.logger.exception(
                                    "memory logging failed", extra=log_record(module=name)
                                )
                            patch_id = self._generate_patch_with_memory(
                                name, "scenario_patch"
                            )
                            self.logger.info(
                                "patch result",
                                extra=log_record(
                                    module=name,
                                    patch_id=patch_id,
                                    success=patch_id is not None,
                                    tags=["fix_result"],
                                ),
                            )
                            if patch_id is not None:
                                after_target = Path(after_dir) / rel
                                after_target.parent.mkdir(parents=True, exist_ok=True)
                                shutil.copy2(src, after_target)
                                diff_data = _collect_diff_data(Path(before_dir), Path(after_dir))
                                self._pre_commit_alignment_check(diff_data)
                                try:
                                    self._alignment_review_last_commit(
                                        f"scenario_patch_{patch_id}"
                                    )
                                    try:
                                        repo = resolve_path(".")
                                        self._sandbox_integrate(
                                            repo,
                                            router=GLOBAL_ROUTER,
                                            context_builder=self.self_coding_engine.context_builder,
                                        )
                                    except Exception:
                                        self.logger.exception(
                                            "post_patch_orphan_integration_failed"
                                        )
                                    try:
                                        self._post_round_orphan_scan()
                                    except Exception:
                                        self.logger.exception(
                                            "recursive orphan inclusion failed",
                                            extra=log_record(module=name),
                                        )
                                except Exception:
                                    self.logger.exception(
                                        "alignment review failed for %s",
                                        name,
                                        extra=log_record(module=name),
                                    )
                    except Exception:
                        self.logger.exception("patch generation failed for %s", name)
                elif action == "rerun":
                    self._force_rerun = True
        total = len(metrics)
        passed = total - len(failing)
        frac = passed / total if total else 1.0
        try:
            self._pass_rate_delta = frac - self.baseline_tracker.get("pass_rate")
        except Exception:
            self._pass_rate_delta = 0.0
        # store negative value when scenarios fail so reward is penalised
        self._scenario_pass_rate = frac - 1.0
        self._last_scenario_trend = trend
        self._last_scenario_metrics = dict(metrics)
        return frac

    # ------------------------------------------------------------------
    def _check_chain_stagnation(self, min_streak: int = 3) -> None:
        """Escalate urgency for chains with stagnant ROI."""

        if self.roi_db is None:
            return
        try:
            flagged = self.roi_db.fetch_stagnant_chains(min_streak)
        except Exception:
            self.logger.exception("failed to fetch stagnant chains")
            return
        if not flagged:
            return
        self.urgency_tier += 1
        for wid, streak in flagged.items():
            try:
                dispatch_alert(
                    "workflow_chain_stagnation",
                    2,
                    "workflow chain ROI non-positive trend",
                    {"workflow_id": wid, "streak": streak},
                )
            except Exception:
                self.logger.exception("failed to dispatch ROI chain alert")

    # ------------------------------------------------------------------
    def _check_roi_stagnation(self) -> None:
        """Escalate urgency when ROI fails to improve over consecutive cycles."""

        last_delta = self.baseline_tracker.delta("roi")
        roi_threshold = (
            self.baseline_tracker.std("roi") * self.roi_stagnation_dev_multiplier
        )
        if last_delta < -roi_threshold:
            self._roi_stagnation_count += 1
            if self._roi_stagnation_count > self.stagnation_cycles:
                self.urgency_tier += 1
                self.logger.warning(
                    "ROI momentum below threshold; increasing urgency tier",
                    extra=log_record(
                        tier=self.urgency_tier, streak=self._roi_stagnation_count
                    ),
                )
                try:
                    dispatch_alert(
                        "roi_negative_trend",
                        2,
                        "ROI momentum below threshold; increasing urgency tier",
                        {
                            "tier": self.urgency_tier,
                            "streak": self._roi_stagnation_count,
                        },
                    )
                except Exception:
                    self.logger.exception("failed to dispatch ROI trend alert")
        else:
            self._roi_stagnation_count = 0
            if (
                self.urgency_tier > 0
                and last_delta > self.urgency_recovery_threshold
            ):
                self.urgency_tier = 0

    # ------------------------------------------------------------------
    def _check_momentum(self) -> None:
        """Escalate urgency when momentum fails to improve."""

        delta = self.baseline_tracker.delta("momentum")
        momentum_threshold = (
            self.baseline_tracker.std("momentum")
            * settings.momentum_stagnation_dev_multiplier
        )
        if delta < -momentum_threshold:
            self._momentum_streak += 1
            if self._momentum_streak >= self.stagnation_cycles:
                self.urgency_tier += 1
                self.logger.warning(
                    "momentum below threshold; increasing urgency tier",
                    extra=log_record(
                        tier=self.urgency_tier,
                        delta=delta,
                        threshold=momentum_threshold,
                        streak=self._momentum_streak,
                    ),
                )
        else:
            self._momentum_streak = 0
            if (
                self.urgency_tier > 0
                and delta > self.urgency_recovery_threshold
            ):
                self.urgency_tier = 0

    # ------------------------------------------------------------------
    def _check_delta_score(self) -> None:
        """Escalate urgency when combined delta score trends negative.

        The combined score includes ROI, entropy, and momentum deltas, each
        scaled by a configurable weight.
        """

        score, components = self._compute_delta_score()
        self.baseline_tracker.update(score=score)
        score_threshold = (
            self.baseline_tracker.std("score") * settings.delta_score_dev_multiplier
        )
        self.delta_score_history.append(score)
        avg = sum(self.delta_score_history) / len(self.delta_score_history)
        self.logger.debug(
            "delta score update",
            extra=log_record(delta_score=score, delta_score_avg=avg, **components),
        )
        if (
            len(self.delta_score_history) >= self.baseline_window
            and avg < self.baseline_tracker.get("score") - score_threshold
        ):
            self._delta_score_streak += 1
            if self._delta_score_streak >= self.stagnation_cycles:
                self.urgency_tier += 1
                self.logger.warning(
                    "delta_score negative trend; increasing urgency tier",
                    extra=log_record(
                        tier=self.urgency_tier,
                        delta_score_avg=avg,
                        **components,
                    ),
                )
        else:
            self._delta_score_streak = 0

    # ------------------------------------------------------------------
    def _alignment_review_last_commit(self, description: str) -> None:
        """Run alignment flagger on the most recent commit."""
        settings = SandboxSettings()
        if not getattr(settings, "enable_alignment_flagger", True):
            return
        try:
            diff_proc = subprocess.run(
                ["git", "show", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
                timeout=60,
            )
            patch = diff_proc.stdout
            files_proc = subprocess.run(
                ["git", "show", "--pretty=", "--name-only", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
                timeout=60,
            )
            files = [ln.strip() for ln in files_proc.stdout.splitlines() if ln.strip()]
        except subprocess.CalledProcessError as exc:
            self.logger.error(
                "git command failed",
                extra=log_record(cmd=exc.cmd, rc=exc.returncode, output=exc.stderr),
            )
            return
        except subprocess.TimeoutExpired as exc:
            self.logger.error(
                "git command timed out",
                extra=log_record(cmd=exc.cmd, timeout=exc.timeout, output=exc.stderr),
            )
            return
        except Exception:
            self.logger.exception("git command unexpected failure")
            return

        try:
            report = self.alignment_flagger.flag_patch(patch, {"files": files})
        except Exception:
            self.logger.exception("alignment flagger failed")
            return
        issues = report.get("issues", [])
        max_severity = max((i.get("severity", 0) for i in issues), default=0) / 4.0
        warn_th = settings.alignment_warning_threshold
        fail_th = settings.alignment_failure_threshold
        if max_severity >= warn_th:
            warnings = [i.get("message", "") for i in issues]
            try:
                audit_log_event(
                    "alignment_flag",
                    {"description": description, "warnings": warnings, "files": files},
                )
            except Exception:
                self.logger.exception("alignment audit log failed")
            if max_severity >= fail_th:
                try:
                    dispatch_alert(
                        "alignment_warning",
                        5,
                        "alignment failure detected",
                        {"description": description, "severity": max_severity},
                    )
                except Exception:
                    self.logger.exception("alignment warning dispatch failed")
            else:
                self.logger.warning(
                    "alignment warnings detected",
                    extra={"description": description, "warnings": warnings},
                )

    # ------------------------------------------------------------------
    def _flag_patch_alignment(
        self, patch_id: int | None, context: dict[str, Any]
    ) -> None:
        """Analyse the latest commit for alignment concerns and log findings."""
        if patch_id is None:
            return
        settings = SandboxSettings()
        if not getattr(settings, "enable_alignment_flagger", True):
            return
        try:
            diff_proc = subprocess.run(
                ["git", "show", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
                timeout=60,
            )
            diff = diff_proc.stdout
        except subprocess.CalledProcessError as exc:
            self.logger.error(
                "git command failed",
                extra=log_record(cmd=exc.cmd, rc=exc.returncode, output=exc.stderr),
            )
            return
        except subprocess.TimeoutExpired as exc:
            self.logger.error(
                "git command timed out",
                extra=log_record(cmd=exc.cmd, timeout=exc.timeout, output=exc.stderr),
            )
            return
        except Exception:
            self.logger.exception("git command unexpected failure")
            return
        try:
            commit_hash = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
                timeout=60,
            ).stdout.strip()
        except subprocess.CalledProcessError as exc:
            self.logger.error(
                "git rev-parse failed",
                extra=log_record(cmd=exc.cmd, rc=exc.returncode, output=exc.stderr),
            )
            commit_hash = "unknown"
        except subprocess.TimeoutExpired as exc:
            self.logger.error(
                "git rev-parse timed out",
                extra=log_record(cmd=exc.cmd, timeout=exc.timeout, output=exc.stderr),
            )
            commit_hash = "unknown"
        except Exception:
            commit_hash = "unknown"
        try:
            report = self.alignment_flagger.flag_patch(diff, context)
            context["alignment_report"] = report
            score = report.get("score", 0)
            issues = report.get("issues", [])
            max_severity = max((i.get("severity", 0) for i in issues), default=0) / 4.0
            context["alignment_blocked"] = max_severity >= settings.alignment_failure_threshold
            self.logger.info(
                "alignment score computed",
                extra=log_record(patch_id=patch_id, score=score, severity=max_severity),
            )
            warn_th = settings.alignment_warning_threshold
            fail_th = settings.alignment_failure_threshold
            if max_severity < warn_th:
                self.logger.info(
                    "alignment severity below warning threshold",
                    extra=log_record(patch_id=patch_id, severity=max_severity),
                )
                update_alignment_baseline(settings)
                return
            for idx, issue in enumerate(issues):
                msg = issue.get("message", "")
                sev = int(issue.get("severity", 1))
                file = msg.rsplit(" in ", 1)[-1] if " in " in msg else None
                evidence = {"file": file, "snippet": msg}
                log_violation(
                    f"patch_{patch_id}_{idx}",
                    "alignment_warning",
                    sev,
                    evidence,
                    alignment_warning=True,
                )
            record = {
                "patch_id": patch_id,
                "commit": commit_hash,
                "score": score,
                "severity": max_severity,
                "report": report,
            }
            escalated = False
            if max_severity >= fail_th:
                escalated = True
                try:
                    dispatch_alert(
                        "alignment_review",
                        5,
                        "alignment severity exceeded threshold",
                        {"patch_id": patch_id, "severity": max_severity},
                    )
                except Exception:
                    self.logger.exception("alignment review dispatch failed")
            record["escalated"] = escalated
            try:
                security_auditor.dispatch_alignment_warning(record)
            except Exception:
                self.logger.exception("alignment warning dispatch failed")
            self.cycle_logs.append({"cycle": self._cycle_count, **record})
            if self.event_bus:
                try:
                    self.event_bus.publish("alignment:flag", record)
                except Exception:
                    self.logger.exception("alignment event publish failed")
            try:
                path = Path(settings.alignment_flags_path)
                with path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(record) + "\n")
            except Exception:
                self.logger.exception("alignment flag persistence failed")
            if not escalated:
                update_alignment_baseline(settings)
        except Exception:
            self.logger.exception(
                "alignment flagging failed", extra=log_record(patch_id=patch_id)
            )

    # ------------------------------------------------------------------
    def _weighted_synergy_adjustment(self, window: int = 3) -> float:
        """Compute weighted synergy adjustment factor.

        The weights for each synergy metric are derived from recent patch
        history when available.  Moving averages and variance normalise the
        metric deltas so that the adjustment adapts to historical trends.
        """

        pdb = self.patch_db or (self.data_bot.patch_db if self.data_bot else None)

        learner_weights = getattr(self, "synergy_learner", None)
        if learner_weights is not None:
            lw = learner_weights.weights
            base_weights = {
                "synergy_roi": lw.get("roi", self.synergy_weight_roi),
                "synergy_efficiency": lw.get(
                    "efficiency", self.synergy_weight_efficiency
                ),
                "synergy_resilience": lw.get(
                    "resilience", self.synergy_weight_resilience
                ),
                "synergy_antifragility": lw.get(
                    "antifragility", self.synergy_weight_antifragility
                ),
                "synergy_reliability": lw.get(
                    "reliability", self.synergy_weight_reliability
                ),
                "synergy_maintainability": lw.get(
                    "maintainability", self.synergy_weight_maintainability
                ),
                "synergy_throughput": lw.get(
                    "throughput", self.synergy_weight_throughput
                ),
            }
        else:
            base_weights = {
                "synergy_roi": self.synergy_weight_roi,
                "synergy_efficiency": self.synergy_weight_efficiency,
                "synergy_resilience": self.synergy_weight_resilience,
                "synergy_antifragility": self.synergy_weight_antifragility,
                "synergy_reliability": self.synergy_weight_reliability,
                "synergy_maintainability": self.synergy_weight_maintainability,
                "synergy_throughput": self.synergy_weight_throughput,
            }

        weights: dict[str, float]
        stats: dict[str, tuple[float, float]]
        weights = dict(base_weights)
        stats = {}

        cache = getattr(self, "_synergy_cache", None)

        if pdb:
            try:
                records = pdb.filter()
                records.sort(key=lambda r: getattr(r, "ts", ""))
                patch_count = len(records)
                if not cache or cache.get("count") != patch_count:
                    recent = records[-20:]
                    roi_vals: list[float] = []
                    data: list[list[float]] = []
                    for rec in recent:
                        roi_vals.append(float(getattr(rec, "roi_delta", 0.0)))
                        data.append(
                            [float(getattr(rec, name, 0.0)) for name in base_weights]
                        )

                    if len(data) >= 2 and any(any(row) for row in data):
                        import numpy as np

                        X = np.array(data, dtype=float)
                        y = np.array(roi_vals, dtype=float)
                        coefs, *_ = np.linalg.lstsq(X, y, rcond=None)
                        coef_abs = np.abs(coefs)
                        total = float(coef_abs.sum())
                        if total > 0:
                            for i, name in enumerate(base_weights):
                                w = coef_abs[i] / total
                                weights[name] = w * base_weights[name]
                        else:
                            weights = {
                                k: base_weights[k] / len(base_weights)
                                for k in base_weights
                            }
                        for i, name in enumerate(base_weights):
                            col = X[:, i]
                            stats[name] = (float(col.mean()), float(col.std() or 1.0))
                    cache = {"count": patch_count, "weights": weights, "stats": stats}
                    self._synergy_cache = cache
                else:
                    weights = cache.get("weights", weights)
                    stats = cache.get("stats", stats)
            except Exception as exc:  # pragma: no cover - best effort
                self.logger.exception(
                    "synergy weight history processing failed: %s", exc
                )
                raise shd.HistoryParseError(str(exc)) from exc
        if cache is None:
            self._synergy_cache = {"count": 0, "weights": weights, "stats": stats}

        def norm_delta(name: str) -> float:
            val = self._metric_delta(name, window)
            mean, std = stats.get(name, (0.0, 1.0))
            return (val - mean) / (std + 1e-6)

        try:
            syn_adj = sum(norm_delta(name) * weights.get(name, 0.0) for name in weights)
        except Exception:
            syn_adj = 0.0
        self.logger.info(
            "weighted synergy adjustment",
            extra=log_record(factor=float(syn_adj), weights=weights),
        )
        self.logger.debug(
            "synergy weights used for adjustment",
            extra=log_record(weights=weights),
        )
        return float(syn_adj)

    # ------------------------------------------------------------------
    def _update_synergy_weights(self, roi_delta: float) -> None:
        """Adjust synergy weights using reinforcement learning."""
        names = [
            "synergy_roi",
            "synergy_efficiency",
            "synergy_resilience",
            "synergy_antifragility",
            "synergy_reliability",
            "synergy_maintainability",
            "synergy_throughput",
        ]
        deltas = {n: self._metric_delta(n) for n in names}
        before_weight = self.synergy_weight_roi
        # Create a mutation event up-front so we can record the outcome even if
        # the update step fails.
        event_id = MutationLogger.log_mutation(
            change="synergy_weights_updated",
            reason="roi_delta adjustment",
            trigger="roi_delta",
            performance=0.0,
            workflow_id=0,
            before_metric=before_weight,
            after_metric=before_weight,
            parent_id=self._last_mutation_id,
        )

        after_metric = before_weight
        perf = 0.0
        roi_val = 0.0
        try:
            extra = dict(getattr(self, "_last_orphan_metrics", {}) or {})
            scen = getattr(self, "_last_scenario_trend", None)
            if scen:
                try:
                    extra["avg_roi"] = float(sum(scen.values()) / len(scen))
                except Exception:
                    extra["avg_roi"] = 0.0
            pr = getattr(self, "_scenario_pass_rate", None)
            if pr is not None:
                extra["pass_rate"] = float(pr)
            self.synergy_learner.update(roi_delta, deltas, extra)
            self.logger.info(
                "synergy weights updated",
                extra=log_record(
                    weights=self.synergy_learner.weights,
                    roi_delta=roi_delta,
                    state=self.synergy_learner._state,
                ),
            )
            self.synergy_weight_roi = self.synergy_learner.weights["roi"]
            self.synergy_weight_efficiency = self.synergy_learner.weights["efficiency"]
            self.synergy_weight_resilience = self.synergy_learner.weights["resilience"]
            self.synergy_weight_antifragility = self.synergy_learner.weights[
                "antifragility"
            ]
            self.synergy_weight_reliability = self.synergy_learner.weights["reliability"]
            self.synergy_weight_maintainability = self.synergy_learner.weights[
                "maintainability"
            ]
            self.synergy_weight_throughput = self.synergy_learner.weights["throughput"]
            self.logger.info(
                "synergy weights after update",
                extra=log_record(weights=self.synergy_learner.weights, roi_delta=roi_delta),
            )
            after_metric = self.synergy_weight_roi
            perf = roi_delta
            roi_val = roi_delta
        except Exception as exc:  # pragma: no cover - runtime issues
            try:
                synergy_weight_update_failures_total.inc()
            except Exception:
                logger.exception("Unhandled exception in self_improvement")
            try:
                dispatch_alert(
                    "synergy_weight_update_failure",
                    2,
                    "Weight update failed",
                    {"roi_delta": roi_delta},
                )
                synergy_weight_update_alerts_total.inc()
            except Exception:
                logger.exception("Unhandled exception in self_improvement")
            self.logger.exception("synergy weight update failed: %s", exc)
        finally:
            MutationLogger.record_mutation_outcome(
                event_id,
                after_metric=after_metric,
                roi=roi_val,
                performance=perf,
            )
            self._last_mutation_id = event_id

    # ------------------------------------------------------------------
    def _policy_state(self) -> tuple[int, ...]:
        """Return the state tuple used by :class:`SelfImprovementPolicy`."""
        energy = 0.0
        if self.capital_bot:
            try:
                energy = self.capital_bot.energy_score(
                    load=0.0,
                    success_rate=1.0,
                    deploy_eff=1.0,
                    failure_rate=0.0,
                )
            except Exception as exc:
                self.logger.exception("energy_score failed: %s", exc)
                energy = 0.0
        profit = 0.0
        if self.capital_bot:
            try:
                profit = self.capital_bot.profit()
            except Exception as exc:
                self.logger.exception("profit check failed: %s", exc)
                profit = 0.0
        trend = anomaly = patch_rate = 0.0
        if self.data_bot:
            try:
                trend = self.data_bot.long_term_roi_trend(limit=200)
            except Exception as exc:
                self.logger.exception("ROI trend fetch failed: %s", exc)
                trend = 0.0
            try:
                df = self.data_bot.db.fetch(100)
                if hasattr(df, "empty"):
                    if not getattr(df, "empty", True):
                        df["roi"] = df["revenue"] - df["expense"]
                        anomaly = float(len(DataBot.detect_anomalies(df, "roi"))) / len(
                            df
                        )
                elif isinstance(df, list) and df:
                    rois = [
                        float(r.get("revenue", 0.0) - r.get("expense", 0.0)) for r in df
                    ]
                    df_list = [{"roi": r} for r in rois]
                    anomaly = float(
                        len(DataBot.detect_anomalies(df_list, "roi"))
                    ) / len(rois)
            except Exception as exc:
                self.logger.exception("anomaly detection failed: %s", exc)
                anomaly = 0.0
            if getattr(self.data_bot, "patch_db", None):
                try:
                    patch_rate = self.data_bot.patch_db.success_rate()
                except Exception as exc:
                    self.logger.exception("patch success rate lookup failed: %s", exc)
                    patch_rate = 0.0
        avg_roi = avg_complex = revert_rate = 0.0
        module_idx = 0
        module_trend = 0.0
        entropy_flag = 0
        tracker = getattr(self, "tracker", None)
        syn_roi = syn_eff = syn_res = syn_af = 0.0
        if tracker is not None:
            try:
                syn_roi = self._metric_delta("synergy_roi")
                syn_eff = self._metric_delta("synergy_efficiency")
                syn_res = self._metric_delta("synergy_resilience")
                syn_af = self._metric_delta("synergy_antifragility")
            except Exception:
                syn_roi = syn_eff = syn_res = syn_af = 0.0
        syn_roi *= self.synergy_weight_roi
        syn_eff *= self.synergy_weight_efficiency
        syn_res *= self.synergy_weight_resilience
        syn_af *= self.synergy_weight_antifragility
        profit += syn_roi
        energy = max(0.0, energy + syn_eff)
        pdb = self.patch_db or (self.data_bot.patch_db if self.data_bot else None)
        if pdb:
            try:
                repo = _repo_path()
                with pdb._connect() as conn:
                    rows = conn.execute(
                        "SELECT roi_delta, complexity_delta, reverted, filename "
                        "FROM patch_history ORDER BY id DESC LIMIT ?",
                        (10,),
                    ).fetchall()
                if rows:
                    avg_roi = float(sum(r[0] for r in rows) / len(rows))
                    avg_complex = float(sum(r[1] for r in rows) / len(rows))
                    revert_rate = float(sum(1 for r in rows if r[2]) / len(rows))
                    p = Path(rows[0][3])
                    abs_p = p if p.is_absolute() else repo / p
                    try:
                        mod_name = abs_p.resolve().relative_to(repo).as_posix()
                    except Exception:
                        mod_name = p.name
                    module_idx = self.module_index.get(mod_name)
                    mods = [
                        m
                        for m, idx in self.module_clusters.items()
                        if idx == module_idx
                    ]
                    try:
                        if mods:
                            placeholders = ",".join("?" * len(mods))
                            total = conn.execute(
                                f"SELECT SUM(roi_delta) FROM patch_history WHERE filename IN ({placeholders})",
                                [Path(m).name for m in mods],
                            ).fetchone()
                            module_trend = float(total[0] or 0.0)
                        else:
                            module_trend = 0.0
                    except Exception:
                        module_trend = 0.0
                    if self.meta_logger:
                        try:
                            if Path(mod_name).as_posix() in self.entropy_ceiling_modules:
                                entropy_flag = 1
                            if module_trend == 0.0:
                                module_trend = dict(self.meta_logger.rankings()).get(
                                    mod_name, 0.0
                                )
                        except Exception as exc:  # pragma: no cover - best effort
                            self.logger.exception("meta logger stats failed: %s", exc)
            except Exception as exc:
                self.logger.exception("patch metrics failed: %s", exc)
                avg_roi = avg_complex = revert_rate = 0.0
                module_idx = 0
            try:
                kw_count, kw_recent = pdb.keyword_features()
            except Exception as exc:
                self.logger.exception("keyword feature fetch failed: %s", exc)
                kw_count = kw_recent = 0
        else:
            kw_count = kw_recent = 0
        avg_roi_delta = avg_eff = 0.0
        if self.evolution_history:
            try:
                stats = self.evolution_history.averages(limit=5)
                avg_roi_delta = float(stats.get("avg_roi_delta", 0.0))
                avg_eff = float(stats.get("avg_efficiency", 0.0))
            except Exception as exc:  # pragma: no cover - best effort
                self.logger.exception("evolution history stats failed: %s", exc)
                avg_roi_delta = avg_eff = 0.0
        short_avg = 0.0
        if self.raroi_history:
            n = min(len(self.raroi_history), 5)
            short_avg = float(sum(self.raroi_history[-n:]) / n)
        return (
            int(round(profit)),
            int(round(energy * 10)),
            int(round(trend * 10)),
            int(round(anomaly * 10)),
            int(round(patch_rate * 10)),
            int(round(avg_roi * 10)),
            int(round(avg_complex * 10)),
            int(round(revert_rate * 10)),
            int(module_idx),
            int(round(module_trend * 10)),
            int(entropy_flag),
            int(kw_count),
            int(kw_recent),
            int(round(avg_roi_delta * 10)),
            int(round(avg_eff)),
            int(round(short_avg * 10)),
            int(round(self.roi_delta_ema * 10)),
            int(round(syn_roi * 10)),
            int(round(syn_eff * 10)),
            int(round(syn_res * 10)),
            int(round(syn_af * 10)),
        )

    # ------------------------------------------------------------------
    def _collect_action_features(self) -> list[list[float]]:
        """Return recent ROI deltas and their changes for growth prediction.

        Each history entry is evaluated with ``roi_predictor`` when available.
        Entries classified as ``"marginal"`` are skipped unless the optional
        ``allow_marginal_candidates`` flag is set on the engine.  Skipped
        entries are recorded via :meth:`_log_action` for auditability.
        """

        feats: list[list[float]] = []
        history = self.raroi_history[-5:]
        prev = 0.0
        allow_marginal = getattr(self, "allow_marginal_candidates", False)

        for idx, val in enumerate(history):
            feat = [float(val), float(val - prev)]
            roi_est = float(val)
            category = "unknown"
            if getattr(self, "roi_predictor", None):
                try:
                    try:
                        seq, category, _, _ = self.roi_predictor.predict(
                            [feat], horizon=1
                        )
                        roi_est = float(seq[-1]) if seq else 0.0
                    except TypeError:
                        roi_est, category, _, _ = self.roi_predictor.predict([feat])
                        roi_est = float(roi_est)
                except Exception:
                    roi_est, category = 0.0, "unknown"

            if category == "marginal" and not allow_marginal:
                ctx = getattr(self, "_current_context", {}) or {}
                session_id = ""
                vectors: list[tuple[str, str, float]] = []
                if isinstance(ctx, dict):
                    session_id = ctx.get("retrieval_session_id", "")
                    raw_vecs = ctx.get("retrieval_vectors") or []
                    for item in raw_vecs:
                        if isinstance(item, dict):
                            origin = item.get("origin_db") or item.get("origin")
                            vec_id = item.get("vector_id") or item.get("id")
                            score = item.get("score") or item.get("similarity")
                        else:
                            if len(item) == 3:
                                origin, vec_id, score = item
                            elif len(item) == 2:
                                origin, vec_id = item
                                score = 0.0
                            else:
                                continue
                        if origin is not None and vec_id is not None:
                            vectors.append((str(origin), str(vec_id), float(score or 0.0)))
                if getattr(self, "data_bot", None):
                    try:  # pragma: no cover - best effort
                        self.data_bot.db.log_patch_outcome(
                            f"raroi_history_{idx}",
                            False,
                            [(o, v) for o, v, _ in vectors],
                            session_id=session_id,
                            reverted=False,
                        )
                    except Exception:
                        self.logger.exception("failed to log patch outcome")
                pl = getattr(getattr(self, "self_coding_engine", None), "patch_logger", None)
                if pl and vectors and session_id:
                    try:  # pragma: no cover - best effort
                        ids = [(f"{o}:{v}", s) for o, v, s in vectors]
                        pl.track_contributors(ids, False, session_id=session_id)
                    except Exception:
                        self.logger.exception("failed to log patch outcome")
                try:  # pragma: no cover - best effort
                    self._log_action(
                        "skip_candidate", f"raroi_history_{idx}", roi_est, category
                    )
                except Exception:
                    logger.exception("Unhandled exception in self_improvement")
            else:
                feats.append(feat)
            prev = val

        return feats

    def _candidate_features(self, module: str) -> list[list[float]]:
        """Return feature vector aligned with ``adaptive_roi_dataset.build_dataset``.

        The returned sequence follows the column layout used by
        :func:`adaptive_roi_dataset.build_dataset` so the predictor can operate
        on live data. Missing values default to ``0.0`` allowing the engine to
        work even when telemetry is incomplete.  Columns gathered here are::

            [before_metric, after_metric, api_cost_delta, cpu_seconds_delta,
             success_rate_delta, gpt_score, gpt_feedback_score,
             gpt_feedback_tokens, long_term_perf_delta,
             long_term_eval_outcome, resource_cost, resource_cpu_usage,
             resource_gpu_usage, error_count, repair_count]
        """

        # Historical ROI forecast for the module
        hist_roi = 0.0
        if self.pre_roi_bot:
            try:
                res = self.pre_roi_bot.predict_model_roi(module, [])
                hist_roi = float(getattr(res, "roi", 0.0))
            except Exception:
                self.logger.exception("pre ROI forecast failed for %s", module)
        if hist_roi == 0.0 and self.patch_db:
            try:
                conn = self.patch_db.router.get_connection("patch_history")
                row = conn.execute(
                    "SELECT roi_after FROM patch_history WHERE filename=? ORDER BY id DESC LIMIT 1",
                    (module,),
                ).fetchone()
                if row and row[0] is not None:
                    hist_roi = float(row[0])
                    self.logger.info(
                        "historical ROI",
                        extra=log_record(module=module, roi=hist_roi),
                    )
            except Exception:
                self.logger.exception("patch history lookup failed for %s", module)

        # Recent deltas for core metrics
        perf_delta = self._metric_delta("synergy_roi")
        api_cost_delta = self._metric_delta("api_cost")
        cpu_seconds_delta = self._metric_delta("cpu_seconds")
        success_rate_delta = self._metric_delta("success_rate")

        # GPT score from patch scoring backend (if available)
        gpt_score = 0.0
        if self._score_backend:
            try:  # pragma: no cover - backend optional
                rows = self._score_backend.fetch_recent(1)
                if rows:
                    gpt_score = float(rows[0][-1])
            except Exception:
                logger.exception("Unhandled exception in self_improvement")

        # GPT feedback metrics from evaluation history
        gpt_fb_score = 0.0
        gpt_fb_tokens = 0.0
        try:
            eval_db = EvaluationHistoryDB()
            rows = eval_db.history(module, limit=1)
            if rows:
                gpt_fb_score = float(rows[0][0])
            try:
                cur = eval_db.conn.execute(
                    "SELECT gpt_feedback_tokens FROM evaluation_history "
                    "WHERE engine=? ORDER BY ts DESC LIMIT 1",
                    (module,),
                )
                token_row = cur.fetchone()
                if token_row and token_row[0] is not None:
                    gpt_fb_tokens = float(token_row[0])
            except Exception:
                logger.exception("Unhandled exception in self_improvement")
            try:
                eval_db.conn.close()
            except Exception:
                logger.exception("Unhandled exception in self_improvement")
        except Exception:
            logger.exception("Unhandled exception in self_improvement")

        # Resource usage deltas from ROITracker
        res_cost = res_cpu = res_gpu = 0.0
        tracker = getattr(self, "roi_tracker", None)
        if tracker and getattr(tracker, "resource_metrics", None):
            metrics = tracker.resource_metrics
            if len(metrics) >= 2:
                prev = metrics[-2]
                curr = metrics[-1]

                def _extract(row: Sequence[float]) -> tuple[float, float, float]:
                    if len(row) >= 5:
                        cpu, _mem, _disk, cost, gpu = row[:5]
                    elif len(row) == 3:
                        cost, cpu, gpu = row
                    else:
                        cost = cpu = gpu = 0.0
                    return float(cost), float(cpu), float(gpu)

                p_cost, p_cpu, p_gpu = _extract(prev)
                c_cost, c_cpu, c_gpu = _extract(curr)
                res_cost = c_cost - p_cost
                res_cpu = c_cpu - p_cpu
                res_gpu = c_gpu - p_gpu

        # Error counts from tracker metrics
        err_count = rep_count = 0.0
        if tracker:
            try:
                errs = tracker.metrics_history.get("error_count", [])
                reps = tracker.metrics_history.get("repair_count", [])
                if errs:
                    err_count = float(errs[-1])
                if reps:
                    rep_count = float(reps[-1])
            except Exception:
                logger.exception("Unhandled exception in self_improvement")

        before = hist_roi
        after = hist_roi + perf_delta

        feats = [
            before,
            after,
            api_cost_delta,
            cpu_seconds_delta,
            success_rate_delta,
            gpt_score,
            gpt_fb_score,
            gpt_fb_tokens,
            0.0,  # long_term_perf_delta
            0.0,  # long_term_eval_outcome
            res_cost,
            res_cpu,
            res_gpu,
            err_count,
            rep_count,
        ]
        return [feats]

    def _score_modifications(self, modules: Iterable[str]) -> list[tuple[str, float, str, float]]:
        """Score and rank candidate modules for patching.

        Each candidate is weighted by its predicted ROI and an optional
        growth-type multiplier from :class:`AdaptiveROIPredictor`.  If the
        predictor is unavailable or fails, the candidate receives zero weight.
        Confidence scores from the ROI tracker modulate the risk-adjusted ROI
        to produce the final score used for ranking.
        """

        completed = {Path(m).as_posix() for m in self.entropy_ceiling_modules}
        scored: list[tuple[str, float, str, float]] = []
        for mod in modules:
            if Path(mod).as_posix() in completed:
                continue
            features = self._candidate_features(mod)
            roi_est = 0.0
            category = "unknown"
            if self.roi_predictor:
                try:
                    try:
                        seq, category, _, _ = self.roi_predictor.predict(
                            features, horizon=len(features)
                        )
                    except TypeError:
                        val, category, _, _ = self.roi_predictor.predict(features)
                        seq = [float(val)]
                    if isinstance(seq, (list, tuple)) and seq:
                        roi_est = float(seq[-1])
                except Exception:
                    roi_est, category = 0.0, "unknown"
            if self.roi_tracker:
                base_roi, raroi, _ = self.roi_tracker.calculate_raroi(
                    roi_est,
                    workflow_type="standard",
                    metrics={},
                    failing_tests=sts.get_failed_critical_tests(),
                )
            else:
                base_roi, raroi, _ = roi_est, roi_est, []
            tracker = self.roi_tracker
            mult = (
                self.growth_multipliers.get(category, 1.0)
                if self.growth_weighting
                else 1.0
            )
            if tracker:
                final_score, needs_review, confidence = tracker.score_workflow(
                    mod, raroi
                )
            else:
                confidence = 1.0
                final_score, needs_review = raroi, False
            weight = final_score * mult
            threshold_val = self._raroi_threshold()
            if needs_review or raroi < threshold_val:
                reason = "needs_review" if needs_review else "low_raroi"
                label: str | None = None
                recs: Dict[str, str] = {}
                try:
                    cards = tracker.generate_scorecards() if tracker else []
                    label = tracker.workflow_label if tracker else None
                    roi_base = self.baseline_tracker.get("roi_delta")
                    roi_tol = getattr(getattr(settings, "roi", None), "deviation_tolerance", 0.0)
                    recs = {
                        c.scenario: c.recommendation
                        for c in cards
                        if c.recommendation and c.roi_delta < roi_base - roi_tol
                    }
                except Exception:
                    logger.exception("Unhandled exception in self_improvement")
                self.logger.info(
                    "borderline workflow; deferring to review/shadow testing",
                    extra=log_record(
                        module=mod,
                        confidence=confidence,
                        raroi=raroi,
                        weight=weight,
                        threshold=threshold_val,
                        shadow_test=True,
                        reason=reason,
                        workflow_label=label,
                        recommendations=recs,
                    ),
                )
                try:
                    self._log_action("review", mod, weight, category, confidence)
                except Exception:
                    logger.exception("Unhandled exception in self_improvement")
                try:
                    self.borderline_bucket.add_candidate(mod, raroi, confidence, reason)
                    settings = SandboxSettings()
                    if getattr(settings, "micropilot_mode", "") == "auto":
                        try:
                            evaluator = getattr(self, "micro_pilot_evaluator", None)
                            self.borderline_bucket.process(
                                evaluator,
                                raroi_threshold=threshold_val,
                                confidence_threshold=getattr(
                                    self.roi_tracker, "confidence_threshold", 0.0
                                ),
                            )
                        except Exception:
                            logger.exception("Unhandled exception in self_improvement")
                except Exception:
                    logger.exception("Unhandled exception in self_improvement")
                continue
            scored.append((mod, base_roi, category, weight))
            self.logger.debug(
                "scored modification",
                extra=log_record(
                    module=mod,
                    base_roi=base_roi,
                    raroi=raroi,
                    confidence=confidence,
                    weight=weight,
                ),
            )
        if self.use_adaptive_roi:
            scored = [s for s in scored if s[3] > 0]
            scored.sort(key=lambda x: -x[3])
        else:
            scored = [s for s in scored if s[1] > 0]
            scored.sort(key=lambda x: -x[1])
        planner = getattr(self, "action_planner", None)
        if planner:
            try:
                planner.update_priorities({m: w for m, _, _, w in scored})
            except Exception:
                self.logger.exception("priority queue update failed")
        return scored

    def _log_action(
        self,
        action: str,
        target: str,
        roi: float,
        growth: str,
        confidence: float | None = None,
    ) -> None:
        """Persist chosen actions and ROI predictions for auditing."""
        try:
            conn = router.get_connection("action_audit")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS action_audit(
                    ts TEXT,
                    engine TEXT,
                    action TEXT,
                    target TEXT,
                    roi REAL,
                    growth TEXT,
                    confidence REAL
                )
                """
            )
            conn.execute(
                "INSERT INTO action_audit(ts, engine, action, target, roi, growth, confidence) VALUES(?,?,?,?,?,?,?)",
                (
                    datetime.utcnow().isoformat(),
                    self.bot_name,
                    action,
                    target,
                    float(roi),
                    growth,
                    float(confidence or 0.0),
                ),
            )
            conn.commit()
        except Exception as exc:  # pragma: no cover - best effort
            self.logger.exception("failed to log action audit: %s", exc)

    # ------------------------------------------------------------------
    def _should_trigger(self) -> bool:
        """Return ``True`` when a new improvement cycle should run."""

        if getattr(self, "_force_rerun", False):
            self._force_rerun = False
            return True

        self._last_skip_info: dict[str, object] | None = None

        elapsed = time.time() - self.last_run
        should_run = True
        if elapsed < self.interval:
            info = {"reason": "interval_not_elapsed", "metrics": {}}
            self.logger.debug(
                "skipping self-improvement cycle",
                extra=log_record(
                    reason=info["reason"],
                    metrics=info["metrics"],
                    elapsed=elapsed,
                    interval=self.interval,
                ),
            )
            self._last_skip_info = info
            should_run = False
        else:
            # Gather per-metric deltas from the baseline tracker
            history = self.baseline_tracker.to_dict()
            metrics: dict[str, float] = {}
            for name in history.keys():
                if name.endswith("_delta"):
                    continue
                try:
                    metrics[name] = float(self.baseline_tracker.delta(name))
                except Exception:
                    continue

            # Include any available ROI tracker deltas
            roi_tracker = getattr(self, "roi_tracker", None)
            if roi_tracker and hasattr(roi_tracker, "get_scenario_metrics_delta"):
                try:
                    metrics.update(
                        roi_tracker.get_scenario_metrics_delta("overall")
                    )
                except Exception:
                    pass

            # Fetch recent errors and check for critical severity
            events = None
            try:
                eb = getattr(self, "error_bot", None)
                if eb is not None:
                    if hasattr(eb, "recent_errors"):
                        events = eb.recent_errors(limit=5)
                    elif hasattr(eb, "recent_events"):
                        events = eb.recent_events(limit=5)
                    elif hasattr(getattr(eb, "db", None), "recent_errors"):
                        events = eb.db.recent_errors(limit=5)  # type: ignore[attr-defined]
            except Exception:
                events = None

            threshold = getattr(settings, "critical_severity_threshold", 75.0)

            def _severity_to_score(sev: object) -> float | None:
                mapping = {
                    "critical": 100.0,
                    "crit": 100.0,
                    "fatal": 100.0,
                    "high": 75.0,
                    "error": 75.0,
                    "warn": 50.0,
                    "warning": 50.0,
                    "medium": 50.0,
                    "low": 25.0,
                    "info": 0.0,
                }
                if isinstance(sev, str):
                    s = sev.lower()
                    if s in mapping:
                        return mapping[s]
                    try:
                        sev = float(sev)
                    except ValueError:
                        return None
                if isinstance(sev, (int, float)):
                    val = float(sev)
                    if 0 <= val <= 1:
                        return val * 100
                    if 0 <= val <= 5:
                        return val * 20
                    if 0 <= val <= 10:
                        return val * 10
                    return val
                return None

            critical_error = False
            for ev in events or []:
                sev = getattr(getattr(ev, "error_type", None), "severity", None)
                score = _severity_to_score(sev)
                if score is not None and score >= threshold:
                    critical_error = True
                    break

            if metrics and all(v > 0 for v in metrics.values()) and not critical_error:
                info = {"reason": "all_deltas_positive", "metrics": metrics}
                self.logger.debug(
                    "skipping self-improvement cycle",
                    extra=log_record(reason=info["reason"], metrics=info["metrics"]),
                )
                self._last_skip_info = info
                should_run = False

        return should_run

    def _record_state(self) -> None:
        """Store metrics and discrepancies as research items."""
        mdb = self.diagnostics.metrics
        edb = self.diagnostics.error_bot.db
        df = mdb.fetch(20)
        for row in df.itertuples(index=False):
            item = ResearchItem(
                topic="metrics",
                content=str(row._asdict()),
                timestamp=time.time(),
            )
            try:
                self.info_db.add(item)
            except Exception as exc:
                self.logger.exception("failed to record metric item: %s", exc)
        disc = edb.discrepancies()
        if "message" in disc:
            for msg in disc["message"]:
                item = ResearchItem(
                    topic="error",
                    content=str(msg),
                    timestamp=time.time(),
                    tags=["error"],
                )
                try:
                    self.info_db.add(item)
                except Exception as exc:
                    self.logger.exception("failed to record error item: %s", exc)

    def _evaluate_learning(self) -> None:
        """Benchmark the learning engine via cross-validation."""
        if not self.learning_engine:
            return
        try:
            if hasattr(self.learning_engine, "evaluate"):
                result = self.learning_engine.evaluate()
                mean_score = float(result.get("cv_score", 0.0))
                if hasattr(self.learning_engine, "persist_evaluation"):
                    try:
                        self.learning_engine.persist_evaluation(result)
                    except Exception as exc:
                        self.logger.exception("persist_evaluation failed: %s", exc)
            else:
                X, y = self.learning_engine._dataset()  # type: ignore[attr-defined]
                if not X or len(set(y)) < 2:
                    return
                from sklearn.model_selection import cross_val_score

                scores = cross_val_score(self.learning_engine.model, X, y, cv=3)
                mean_score = float(scores.mean())
                if hasattr(self.learning_engine, "persist_evaluation"):
                    try:
                        self.learning_engine.persist_evaluation(
                            {
                                "cv_score": mean_score,
                                "holdout_score": mean_score,
                                "timestamp": time.time(),
                            }
                        )
                    except Exception as exc:
                        self.logger.exception("persist_evaluation failed: %s", exc)
        except Exception as exc:
            self.logger.exception("learning evaluation failed: %s", exc)
            mean_score = 0.0
        item = ResearchItem(
            topic="learning_eval",
            content=str({"cv_score": mean_score}),
            timestamp=time.time(),
        )
        try:
            self.info_db.add(item)
        except Exception as exc:
            self.logger.exception("failed to record learning eval: %s", exc)

    def _evaluate_roi_predictor(self) -> None:
        """Evaluate and periodically retrain the adaptive ROI predictor."""
        if not self.roi_predictor:
            return
        tracker = getattr(self, "tracker", None)
        if tracker is None:
            return
        mae_avg = self.baseline_tracker.get("mae")
        mae_std = self.baseline_tracker.std("mae")
        acc_avg = self.baseline_tracker.get("acc")
        acc_std = self.baseline_tracker.std("acc")
        mae_threshold = mae_avg + self.mae_dev_multiplier * mae_std
        acc_threshold = acc_avg - self.acc_dev_multiplier * acc_std
        try:
            acc, mae = self.roi_predictor.evaluate_model(
                tracker,
                mae_threshold=mae_threshold,
                acc_threshold=acc_threshold,
            )
            self.baseline_tracker.update(mae=mae, acc=acc)
            try:
                self.roi_predictor.record_drift(acc, mae)
            except Exception:
                logger.exception("Unhandled exception in self_improvement")
            self.logger.info(
                "adaptive roi evaluation",
                extra=log_record(accuracy=float(acc), mae=float(mae)),
            )
            drift_metrics = getattr(self.roi_predictor, "drift_metrics", None)
            if drift_metrics:
                try:
                    self.logger.info(
                        "adaptive roi drift",
                        extra=log_record(
                            **{k: float(v) for k, v in drift_metrics.items()}
                        ),
                    )
                except Exception:
                    logger.exception("Unhandled exception in self_improvement")
            try:  # pragma: no cover - best effort telemetry
                prediction_mae.labels(metric="adaptive_roi").set(float(mae))
                prediction_reliability.labels(metric="adaptive_roi").set(float(acc))
            except Exception:
                logger.exception("Unhandled exception in self_improvement")
            if mae > mae_threshold or acc < acc_threshold:
                self.logger.info(
                    "adaptive roi model drift detected",
                    extra=log_record(accuracy=float(acc), mae=float(mae)),
                )
                try:
                    self.roi_predictor.partial_fit()
                except Exception:
                    self.logger.exception("adaptive roi partial_fit failed")
                    try:
                        self.roi_predictor.train()
                    except Exception:
                        self.logger.exception("adaptive roi training failed")
        except Exception as exc:  # pragma: no cover - evaluation failure
            self.logger.exception("adaptive roi evaluation failed: %s", exc)
            acc = mae = 0.0

        now = time.time()
        if (
            len(tracker.roi_history)
            > getattr(self.roi_predictor, "_trained_size", 0)
            and now - self._adaptive_roi_last_train
            >= self.adaptive_roi_train_interval
        ):
            try:
                load_training_data(tracker, router=router)
                self.logger.info("adaptive roi training data loaded")
                selected = getattr(self.roi_predictor, "selected_features", None)
                X, y, g, names = build_dataset(
                    evolution_path="evolution_history.db",
                    roi_path="roi.db",
                    selected_features=selected,
                    return_feature_names=True,
                    router=router,
                )
                dataset = (X, y, g)
                self.roi_predictor.train(
                    dataset,
                    cv=self.roi_predictor.cv,
                    param_grid=self.roi_predictor.param_grid,
                    feature_names=names,
                )
                self._adaptive_roi_last_train = now
                msg = f"adaptive roi model retrained on {len(dataset[0])} samples"
                self.logger.info(msg)
                val_scores = getattr(self.roi_predictor, "validation_scores", {}) or {}
                for name, score in val_scores.items():
                    self.logger.info("validation MAE %s: %.4f", name, score)
                best_params = getattr(self.roi_predictor, "best_params", None)
                best_score = getattr(self.roi_predictor, "best_score", None)
                if best_params and best_score is not None:
                    params = {k: v for k, v in best_params.items() if k != "model"}
                    self.logger.info(
                        "best model %s (MAE=%.4f)",
                        best_params.get("model"),
                        best_score,
                    )
                    if params:
                        self.logger.info("best params: %s", params)
            except Exception as exc:  # pragma: no cover - retrain failure
                self.logger.exception(
                    "adaptive roi scheduled retrain failed: %s", exc
                )

    def fit_truth_adapter(self, X: np.ndarray, y: np.ndarray) -> None:
        """Retrain :class:`TruthAdapter` with new data."""
        try:
            self.truth_adapter.fit(X, y)
            self._truth_adapter_needs_retrain = False
        except Exception:
            self.logger.exception("truth adapter fit failed")

    def _record_warning_summary(
        self, delta: float, warnings: dict[str, list[dict[str, Any]]]
    ) -> None:
        """Store high-severity warnings that coincide with positive ROI."""
        if delta <= 0:
            return
        if warnings.get("ethics") or warnings.get("risk_reward"):
            entry = {"roi_delta": delta, "warnings": warnings}
            self.warning_summary.append(entry)
            self.logger.warning(
                "improvement flagged with high severity warnings",
                extra=log_record(**entry),
            )

    def get_warning_summary(self) -> list[dict[str, Any]]:
        """Return summary entries for high-severity warning improvements."""
        return list(self.warning_summary)

    def _log_improvement_warnings(
        self, warnings: dict[str, list[dict[str, Any]]]
    ) -> None:
        """Write individual improvement warnings to the violation log."""
        for category, entries in warnings.items():
            for idx, warn in enumerate(entries):
                sev = int(warn.get("severity", 1))
                evidence: dict[str, Any] = {
                    "category": category,
                    "file": warn.get("file"),
                    "issue": warn.get("issue") or warn.get("message"),
                }
                if "entry" in warn:
                    evidence["entry"] = warn["entry"]
                if "violations" in warn:
                    evidence["violations"] = warn["violations"]
                if "snippet" in warn:
                    evidence["snippet"] = warn["snippet"]
                if "snippets" in warn:
                    evidence["snippets"] = warn["snippets"]
                log_violation(
                    f"improvement_{self._cycle_count}_{category}_{idx}",
                    "alignment_warning",
                    sev,
                    evidence,
                    alignment_warning=True,
                )

    def _pre_commit_alignment_check(
        self, diff_data: dict[str, dict[str, list[str]]]
    ) -> dict[str, list[dict[str, Any]]]:
        """Run ``flag_improvement`` on proposed changes before commit."""
        settings = SandboxSettings()
        if not getattr(settings, "enable_alignment_flagger", True):
            return {}
        workflow_changes: list[dict[str, Any]] = []
        for file, diff in diff_data.items():
            added = diff.get("added", [])
            removed = diff.get("removed", [])
            diff_text = "\n".join(["+" + ln for ln in added] + ["-" + ln for ln in removed])
            workflow_changes.append(
                {"file": file, "code": "\n".join(added), "diff": diff_text}
            )
        metrics = getattr(self, "_last_scenario_metrics", {})
        warnings: dict[str, list[dict[str, Any]]] = {}
        try:
            warnings = flag_improvement(workflow_changes, metrics, [], None, settings)
        except Exception:
            warnings = {}
        if any(warnings.values()):
            self._log_improvement_warnings(warnings)
            high = any(
                int(w.get("severity", 1)) >= 3
                for cat in warnings.values()
                for w in cat
            )
            if high:
                try:
                    if self._alignment_agent is None:
                        self._alignment_agent = AlignmentReviewAgent()
                        self._alignment_agent.start()
                    self._alignment_agent.auditor.audit({"warnings": warnings})
                except Exception:
                    self.logger.exception("alignment review agent audit failed")
        return warnings

    @radar.track
    def _optimize_self(self) -> tuple[int | None, bool, float]:
        """Apply a patch to this engine via :class:`SelfCodingEngine`."""
        if not self.self_coding_engine:
            return None, False, 0.0
        # Consult ROI predictor before applying a self optimisation patch.
        try:
            roi_est = 0.0
            growth = "unknown"
            weight = 0.0
            predictor = getattr(self, "roi_predictor", None)
            use_adaptive = getattr(self, "use_adaptive_roi", False)
            tracker = getattr(self, "roi_tracker", None)
            bot_name = getattr(self, "bot_name", "")
            confidence = 1.0
            if predictor and use_adaptive:
                try:
                    features = self._candidate_features(bot_name)
                except Exception:
                    features = [[0.0, 0.0, 0.0]]
                try:
                    try:
                        seq, growth, _conf, _ = predictor.predict(
                            features, horizon=len(features)
                        )
                    except TypeError:
                        val, growth, _conf, _ = predictor.predict(features)
                        seq = [float(val)]
                    roi_est = float(seq[-1]) if seq else 0.0
                except Exception:
                    roi_est, growth = 0.0, "unknown"
                if tracker:
                    base_roi, raroi, _ = tracker.calculate_raroi(
                        roi_est,
                        workflow_type="standard",
                        metrics={},
                        failing_tests=sts.get_failed_critical_tests(),
                    )
                else:
                    base_roi, raroi, _ = roi_est, roi_est, []
                if tracker:
                    final_score, needs_review, confidence = tracker.score_workflow(
                        bot_name, raroi
                    )
                else:
                    confidence = 1.0
                    final_score, needs_review = raroi, False
                threshold_val = self._raroi_threshold()
                if needs_review or raroi < threshold_val:
                    reason = "needs_review" if needs_review else "low_raroi"
                    label: str | None = None
                    recs: Dict[str, str] = {}
                    try:
                        cards = tracker.generate_scorecards() if tracker else []
                        label = tracker.workflow_label if tracker else None
                        roi_base = self.baseline_tracker.get("roi_delta")
                        roi_tol = getattr(getattr(settings, "roi", None), "deviation_tolerance", 0.0)
                        recs = {
                            c.scenario: c.recommendation
                            for c in cards
                            if c.recommendation and c.roi_delta < roi_base - roi_tol
                        }
                    except Exception:
                        logger.exception("Unhandled exception in self_improvement")
                    self.logger.info(
                        "self optimisation deferred: borderline",
                        extra=log_record(
                            growth_type=growth,
                            roi_estimate=base_roi,
                            raroi=raroi,
                            confidence=confidence,
                            final_score=final_score,
                            reason=reason,
                            workflow_label=label,
                            recommendations=recs,
                            threshold=threshold_val,
                        ),
                    )
                    try:
                        self._log_action("review", bot_name, final_score, growth, confidence)
                    except Exception:
                        logger.exception("Unhandled exception in self_improvement")
                    try:
                        self.borderline_bucket.add_candidate(bot_name, raroi, confidence, reason)
                        settings = SandboxSettings()
                        if getattr(settings, "micropilot_mode", "") == "auto":
                            try:
                                evaluator = getattr(self, "micro_pilot_evaluator", None)
                                self.borderline_bucket.process(
                                    evaluator,
                                    raroi_threshold=threshold_val,
                                    confidence_threshold=getattr(
                                        self.roi_tracker, "confidence_threshold", 0.0
                                    ),
                                )
                            except Exception:
                                logger.exception("Unhandled exception in self_improvement")
                    except Exception:
                        logger.exception("Unhandled exception in self_improvement")
                    return None, False, 0.0
                mult = (
                    self.growth_multipliers.get(growth, 1.0)
                    if self.growth_weighting
                    else 1.0
                )
                weight = final_score * mult
                if tracker:
                    tracker._next_prediction = base_roi
                    tracker._next_category = growth
                if weight <= 0:
                    self.logger.info(
                        "self optimisation skipped",
                        extra=log_record(
                            growth_type=growth,
                            roi_estimate=base_roi,
                            raroi=raroi,
                            weight=weight,
                            confidence=confidence,
                            final_score=final_score,
                        ),
                    )
                    return None, False, 0.0

            with tempfile.TemporaryDirectory() as before_dir, tempfile.TemporaryDirectory() as after_dir:
                repo = _repo_path().resolve()
                src = resolve_path("self_improvement/engine.py")
                try:
                    module_rel = src.relative_to(repo).as_posix()
                except Exception:
                    module_rel = src.name
                rel = src.name
                before_target = Path(before_dir) / rel
                before_target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, before_target)
                start_time = time.perf_counter()
                patch_id, reverted, delta = self.self_coding_engine.apply_patch(
                    resolve_path("self_improvement/engine.py"),
                    "self_improvement",
                    parent_patch_id=self._last_patch_id,
                    reason="self_improvement",
                    trigger="optimize_self",
                )
                if patch_id is not None and not reverted:
                    after_target = Path(after_dir) / rel
                    after_target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, after_target)
                    diff_data = _collect_diff_data(Path(before_dir), Path(after_dir))
                    self._pre_commit_alignment_check(diff_data)
            before_metric = 0.0
            after_metric = delta
            if self.self_coding_engine.patch_db and patch_id is not None:
                try:
                    with self.self_coding_engine.patch_db._connect() as conn:
                        row = conn.execute(
                            "SELECT roi_before, roi_after FROM patch_history WHERE id=?",
                            (patch_id,),
                        ).fetchone()
                    if row:
                        before_metric = float(row[0])
                        after_metric = float(row[1])
                except Exception:
                    after_metric = before_metric + delta
            else:
                after_metric = before_metric + delta
            with MutationLogger.log_context(
                change=f"self_opt_patch_{patch_id}",
                reason="self_improvement",
                trigger="optimize_self",
                workflow_id=0,
                before_metric=before_metric,
                parent_id=self._last_mutation_id,
            ) as mutation:
                mutation["after_metric"] = after_metric
                mutation["performance"] = delta
                mutation["roi"] = after_metric
            self._last_mutation_id = int(mutation["event_id"])
            self._last_patch_id = patch_id
            if patch_id is not None and not reverted:
                self._alignment_review_last_commit(f"self_opt_patch_{patch_id}")
                self._flag_patch_alignment(
                    patch_id, {"trigger": "optimize_self", "patch_id": patch_id}
                )
            roi_delta = after_metric - before_metric
            if tracker and hasattr(tracker, "update"):
                try:
                    token_metrics: dict[str, float] = {}
                    file_path = _repo_path / module_rel
                    stats, *_rest = _si_metrics._collect_metrics([file_path], _repo_path)
                    info = stats.get(module_rel)
                    if info:
                        token_metrics = {
                            "token_entropy": float(info.get("token_entropy", 0.0)),
                            "token_diversity": float(info.get("token_diversity", 0.0)),
                        }
                    tracker.update(
                        before_metric,
                        after_metric,
                        modules=[module_rel],
                        category=growth,
                        confidence=confidence,
                        metrics=token_metrics or None,
                    )
                except Exception:
                    self.logger.exception("roi tracker update failed")
            if self.metrics_db:
                try:
                    elapsed = time.perf_counter() - start_time
                    self.metrics_db.record(module_rel, elapsed, roi_delta=roi_delta)
                except Exception:
                    self.logger.exception("relevancy metrics record failed")
            if predictor and use_adaptive:
                self._log_action(
                    "self_optimize", bot_name, roi_est, growth, confidence
                )
            return patch_id, reverted, delta
        except Exception as exc:
            self.logger.exception("self optimization failed: %s", exc)
            return None, False, 0.0

    def _on_new_pathway(self, topic: str, payload: object) -> None:
        """Incrementally train when a new pathway is logged."""
        if not self.learning_engine:
            return
        if isinstance(payload, dict):
            try:
                rec = PathwayRecord(
                    actions=payload.get("actions", ""),
                    inputs=payload.get("inputs", ""),
                    outputs=payload.get("outputs", ""),
                    exec_time=float(payload.get("exec_time", 0.0)),
                    resources=payload.get("resources", ""),
                    outcome=Outcome(payload.get("outcome", "FAILURE")),
                    roi=float(payload.get("roi", 0.0)),
                    ts=payload.get("ts", ""),
                )
                self.learning_engine.partial_train(rec)
            except Exception as exc:
                self.logger.exception("failed to process pathway record: %s", exc)

    def _compress_module(self, module_path: Path) -> Path | None:
        """Compress ``module_path`` into a zip archive.

        The resulting archive is stored under ``SANDBOX_DATA_DIR`` in the
        ``compressed_modules`` directory. ``None`` is returned on failure.
        """

        try:
            if not module_path.exists():
                return None
            out_dir = (
                _data_dir()
                / "compressed_modules"
            )
            out_dir.mkdir(parents=True, exist_ok=True)
            archive_base = out_dir / module_path.stem
            archive = shutil.make_archive(
                str(archive_base),
                "zip",
                root_dir=module_path.parent,
                base_dir=module_path.name,
            )
            return Path(archive)
        except Exception:
            return None

    def _record_new_modules(self, modules: Iterable[str]) -> None:
        """Update synergy graph and intent clusters for ``modules``."""

        mods = {m for m in modules if m}
        if not mods:
            return
        repo = _repo_path()
        try:
            grapher = getattr(self, "module_synergy_grapher", None)
            if grapher is None:
                from module_synergy_grapher import ModuleSynergyGrapher

                grapher = ModuleSynergyGrapher(root=repo)
                graph_path = Path(settings.module_synergy_graph_path)
                if not graph_path.is_absolute():
                    graph_path = repo / graph_path
                try:
                    grapher.load(graph_path)
                except Exception:
                    logger.exception("Unhandled exception in self_improvement")
                self.module_synergy_grapher = grapher
            grapher.update_graph(sorted(mods))
        except Exception:
            self.logger.exception("failed to update synergy graph")

        try:
            clusterer = getattr(self, "intent_clusterer", None)
            if clusterer is None:
                from intent_clusterer import IntentClusterer

                data_dir = resolve_path(settings.sandbox_data_dir)
                clusterer = IntentClusterer(
                    local_db_path=data_dir / "intent.db",
                    shared_db_path=data_dir / "intent.db",
                )
                self.intent_clusterer = clusterer
            paths = {resolve_path(f"{m}.py") for m in mods}
            clusterer.index_modules(paths)
        except Exception:
            self.logger.exception("failed to index intent modules")

    # ------------------------------------------------------------------
    def _test_orphan_modules(self, paths: Iterable[str]) -> set[str]:
        """Run self tests for ``paths`` and return modules that succeed.

        Modules are classified via :func:`orphan_analyzer.classify_module` and
        the classification is stored in ``self.orphan_traces`` as well as the
        ``orphan_classifications.json`` cache. Modules classified as
        ``legacy`` or ``redundant`` are skipped unless
        :attr:`SandboxSettings.test_redundant_modules` is enabled. Remaining
        modules are executed via :class:`SelfTestService` with orphan discovery
        enabled. Basic metrics about the run are logged and stored via
        ``data_bot`` when available. Actual integration of passing modules is
        handled by the caller.
        """

        all_modules = [str(p) for p in paths]
        if not all_modules:
            return set()

        import time

        repo = _repo_path()
        data_dir = _data_dir()
        if not data_dir.is_absolute():
            data_dir = repo / data_dir
        meta_path = data_dir / "orphan_classifications.json"
        orphan_path = data_dir / "orphan_modules.json"

        try:  # pragma: no cover - dynamic import
            from self_test_service import SelfTestService as _STS
        except Exception:  # pragma: no cover - service unavailable
            _STS = None

        classifications: dict[str, dict[str, Any]] = {}
        candidates: list[str] = []
        legacy: list[str] = []
        redundant: list[str] = []

        traces = getattr(self, "orphan_traces", None)
        if traces is None:
            traces = {}
            setattr(self, "orphan_traces", traces)

        settings = SandboxSettings()
        multiplier = getattr(settings, "side_effect_dev_multiplier", 1.0)

        for m in all_modules:
            start = time.perf_counter()
            path = Path(m)
            abs_path = path if path.is_absolute() else repo / path
            try:
                res = classify_module(abs_path, include_meta=True)
                cls, meta = res if isinstance(res, tuple) else (res, {})
            except Exception as exc:  # pragma: no cover - best effort
                self.logger.exception("classification failed for %s: %s", abs_path, exc)
                cls, meta = "candidate", {}
            score = float(meta.get("side_effects", 0))
            hist = self.baseline_tracker.to_dict().get("side_effects", [])
            avg = self.baseline_tracker.get("side_effects") if hist else score
            if len(hist) >= 2:
                std = self.baseline_tracker.std("side_effects")
            else:
                std = max(1.0, abs(avg) * 0.1)
            threshold = avg + multiplier * std
            try:
                # record metric for future threshold calculations
                self.baseline_tracker.update(side_effects=score)
            except Exception as exc:
                self.logger.debug(
                    "side effects metric update failed",
                    extra=log_record(module=str(abs_path)),
                    exc_info=exc,
                )
            try:
                rel = abs_path.resolve().relative_to(repo).as_posix()
            except Exception:
                rel = str(abs_path)
            info = traces.setdefault(rel, {"parents": []})
            prev_cls = info.get("classification")
            info["classification"] = cls
            info["redundant"] = cls != "candidate"
            info["side_effects"] = score
            classifications[rel] = {"classification": cls, "side_effects": score}
            elapsed = time.perf_counter() - start
            if getattr(self, "metrics_db", None):
                try:
                    self.metrics_db.record(
                        rel, elapsed, self.module_index, roi_delta=0.0
                    )
                except Exception:
                    self.logger.exception(
                        "relevancy metrics record failed", extra=log_record(module=rel)
                    )
            if score > threshold:
                info["heavy_side_effects"] = True
                self.logger.info(
                    "orphan module skipped due to side effects",
                    extra=log_record(module=rel, side_effects=score),
                )
                try:
                    existing = (
                        json.loads(orphan_path.read_text())
                        if orphan_path.exists()
                        else {}
                    )
                except Exception:
                    existing = {}
                if not isinstance(existing, dict):
                    existing = {}
                entry = existing.get(rel, {})
                entry["reason"] = "heavy_side_effects"
                entry["side_effects"] = score
                existing[rel] = entry
                try:
                    orphan_path.parent.mkdir(parents=True, exist_ok=True)
                    orphan_path.write_text(json.dumps(existing, indent=2))
                except Exception:
                    logger.exception("Unhandled exception in self_improvement")
                continue
            if cls == "legacy":
                legacy.append(rel)
                self.logger.info(
                    "orphan module classified",
                    extra=log_record(module=rel, classification="legacy"),
                )
            elif cls == "redundant":
                redundant.append(rel)
                self.logger.info(
                    "orphan module classified",
                    extra=log_record(module=rel, classification="redundant"),
                )
            else:
                candidates.append(rel)
            try:
                if prev_cls == "legacy" and cls != "legacy":
                    orphan_modules_legacy_total.dec(1)
                elif prev_cls != "legacy" and cls == "legacy":
                    orphan_modules_legacy_total.inc(1)
            except Exception:
                logger.exception("Unhandled exception in self_improvement")

        try:
            existing_meta = (
                json.loads(meta_path.read_text()) if meta_path.exists() else {}
            )
        except Exception:  # pragma: no cover - best effort
            existing_meta = {}
        existing_meta.update(classifications)
        try:
            meta_path.parent.mkdir(parents=True, exist_ok=True)
            meta_path.write_text(json.dumps(existing_meta, indent=2))
        except Exception:  # pragma: no cover - best effort
            self.logger.exception("failed to write orphan classifications")

        try:
            settings = SandboxSettings()
        except Exception:  # pragma: no cover - fallback for tests
            from sandbox_settings import SandboxSettings as _SS  # type: ignore

            settings = _SS()

        past_results: dict[str, dict[str, Any]] = {}
        if _STS is not None:
            try:
                past_results = _STS.orphan_summary()
            except Exception:
                past_results = {}
        if past_results:

            def _ok(m: str) -> bool:
                return past_results.get(m, {}).get("roi_delta", 0.0) >= 0

            candidates[:] = [m for m in candidates if _ok(m)]
            legacy[:] = [m for m in legacy if _ok(m)]
            redundant[:] = [m for m in redundant if _ok(m)]

        radar_flags: dict[str, str] = {}
        try:  # pragma: no cover - optional dependency
            from relevancy_radar import load_usage_stats, evaluate_relevancy

            usage_stats = load_usage_stats()
            if usage_stats:
                radar_flags = evaluate_relevancy(
                    {m: 1 for m in all_modules}, usage_stats
                )
        except Exception:
            radar_flags = {}

        retire_modules: list[str] = []
        compress_modules: list[str] = []
        replace_modules: list[str] = []
        for mod, flag in radar_flags.items():
            info = self.orphan_traces.setdefault(mod, {"parents": []})
            info["radar_flag"] = flag
            if flag == "retire":
                retire_modules.append(mod)
            elif flag == "compress":
                compress_modules.append(mod)
            elif flag == "replace":
                replace_modules.append(mod)

        audit_fn = globals().get("audit_log_event")
        gen_patch = self._patch_generator

        for mod in retire_modules:
            path = repo / mod if not Path(mod).is_absolute() else Path(mod)
            try:
                cls = classify_module(path)
            except Exception:
                cls = "error"
            if audit_fn:
                audit_fn("radar_retire", {"module": mod, "classification": cls})

        for mod in compress_modules:
            path = repo / mod if not Path(mod).is_absolute() else Path(mod)
            archive = None
            compress_fn = getattr(self, "_compress_module", None)
            if callable(compress_fn):
                archive = compress_fn(path)
            if audit_fn:
                audit_fn(
                    "radar_compress",
                    {"module": mod, "archive": str(archive) if archive else None},
                )

        for mod in replace_modules:
            path = repo / mod if not Path(mod).is_absolute() else Path(mod)
            try:
                cls = classify_module(path)
            except Exception:
                cls = "error"
            patch_id = None
            if gen_patch:
                kwargs: dict[str, object] = {}
                try:
                    strat_name = self.next_prompt_strategy()
                except Exception:
                    strat_name = None
                if strat_name:
                    try:
                        kwargs["strategy"] = PromptStrategy(strat_name)
                    except Exception:
                        pass
                try:
                    patch_id = gen_patch(str(path), **kwargs)
                except RuntimeError as exc:
                    self.logger.error("quick_fix_engine unavailable: %s", exc)
                    raise
                except Exception:
                    patch_id = None
            if audit_fn:
                audit_fn(
                    "radar_replace",
                    {"module": mod, "classification": cls, "patch_id": patch_id},
                )

        if not getattr(settings, "test_redundant_modules", False):
            retired = {m for m, f in radar_flags.items() if f == "retire"}
            if retired:
                candidates[:] = [m for m in candidates if m not in retired]
                legacy[:] = [m for m in legacy if m not in retired]
                redundant[:] = [m for m in redundant if m not in retired]

        modules = (
            [*candidates, *legacy, *redundant]
            if getattr(settings, "test_redundant_modules", False)
            else [*candidates]
        )
        self.logger.info(
            "self test start",
            extra=log_record(
                modules=sorted(modules),
                legacy=sorted(legacy),
                redundant=sorted(redundant),
            ),
        )

        if not modules:
            counts = {
                "orphan_modules_tested": 0,
                "orphan_modules_passed": 0,
                "orphan_modules_failed": 0,
                "orphan_modules_reclassified": 0,
                "orphan_modules_redundant": len(redundant),
                "orphan_modules_legacy": len(legacy),
            }
            self._last_orphan_counts = counts
            tracker = getattr(self, "tracker", None)
            if tracker is not None:
                tracker.register_metrics(*counts.keys())
                base = tracker.roi_history[-1] if tracker.roi_history else 0.0
                tracker.update(base, base, metrics=counts)
            try:
                orphan_modules_tested_total.inc(0)
                orphan_modules_passed_total.inc(0)
                orphan_modules_failed_total.inc(0)
                orphan_modules_reclassified_total.inc(0)
                orphan_modules_redundant_total.inc(len(redundant))
                orphan_modules_legacy_total.inc(len(legacy))
            except Exception:
                logger.exception("Unhandled exception in self_improvement")
            return set()

        added_modules: set[str] = set()
        try:
            _tracker, tested = environment.auto_include_modules(
                sorted(modules),
                recursive=True,
                validate=True,
                context_builder=self.self_coding_engine.context_builder,
            )
            added_modules.update(tested.get("added", []))
        except Exception:
            try:
                _tracker, tested = environment.auto_include_modules(
                    sorted(modules),
                    recursive=True,
                    context_builder=self.self_coding_engine.context_builder,
                )
                added_modules.update(tested.get("added", []))
            except Exception:
                tested = {}
        record_new = getattr(self, "_record_new_modules", None)
        if record_new:
            record_new(added_modules)

        reuse_scores: dict[str, float] = {}
        try:
            from module_index_db import ModuleIndexDB
            from task_handoff_bot import WorkflowDB

            idx = getattr(self, "module_index", None) or ModuleIndexDB()
            wf_db = WorkflowDB(Path(SandboxSettings().workflows_db))
            workflows = wf_db.fetch(limit=1000)
            total_wf = len(workflows)
            grp_counts: dict[int, int] = {}
            if total_wf:
                for wf in workflows:
                    seen: set[int] = set()
                    for step in getattr(wf, "workflow", []):
                        mod = step.split(":")[0]
                        file = resolve_module_path(mod)
                        try:
                            gid = idx.get(file.as_posix())
                        except Exception:
                            try:
                                gid = idx.get(mod)
                            except Exception:
                                continue
                        seen.add(gid)
                    for g in seen:
                        grp_counts[g] = grp_counts.get(g, 0) + 1
            for mod in modules:
                try:
                    gid = idx.get(mod)
                except Exception:
                    reuse_scores[mod] = 0.0
                    continue
                reuse_scores[mod] = (
                    grp_counts.get(gid, 0) / total_wf if total_wf else 0.0
                )
        except Exception:
            reuse_scores = {m: 0.0 for m in modules}

        for mod, score in reuse_scores.items():
            info = self.orphan_traces.setdefault(mod, {"parents": []})
            info["reuse_score"] = score

        reuse_threshold = SandboxSettings().orphan_reuse_threshold

        scenario_results: dict[str, dict[str, dict[str, Any]]] = {}
        tracker_wf = None
        try:
            from environment_generator import generate_canonical_presets

            canonical = generate_canonical_presets()
            flat_presets = [p for levels in canonical.values() for p in levels.values()]
            env_map = {m: flat_presets for m in modules}
            tracker_wf, wf_details = environment.run_workflow_simulations(
                env_presets=env_map,
                return_details=True,
                runner_config=self.runner_config,
                context_builder=self.self_coding_engine.context_builder,
            )
            scenario_synergy = getattr(tracker_wf, "scenario_synergy", {})
            for runs in wf_details.values():
                for entry in runs:
                    module = entry.get("module")
                    preset = entry.get("preset", {})
                    scen = preset.get("SCENARIO_NAME")
                    res = entry.get("result", {})
                    if not module or module not in modules or not scen:
                        continue
                    metrics = {
                        k: float(v)
                        for k, v in res.items()
                        if k != "exit_code" and isinstance(v, (int, float))
                    }
                    sy_list = scenario_synergy.get(scen, [])
                    try:
                        scen_roi = (
                            float(sy_list[-1].get("synergy_roi", 0.0))
                            if sy_list
                            else 0.0
                        )
                    except Exception:
                        scen_roi = 0.0
                    scen_map = scenario_results.setdefault(module, {})
                    scen_map[scen] = {
                        "roi": scen_roi,
                        "metrics": metrics,
                        "failed": bool(res.get("exit_code")),
                    }
            for mod, scen_map in scenario_results.items():
                trace = self.orphan_traces.setdefault(mod, {"parents": []})
                trace.setdefault("scenarios", {}).update(scen_map)
                rois = [v.get("roi", 0.0) for v in scen_map.values()]
                trace["workflow_robustness"] = float(min(rois)) if rois else 0.0
                trace["workflow_failed"] = any(v.get("failed") for v in scen_map.values())
        except Exception as exc:  # pragma: no cover - best effort
            self.logger.exception("workflow simulation failed: %s", exc)
            tracker_wf = None

        if _STS is None:
            try:
                from sandbox_runner import run_repo_section_simulations as _sim
            except Exception:  # pragma: no cover - best effort
                return set()
            passed: set[str] = set()
            for m in modules:
                try:
                    tracker_res, details = _sim(
                        str(repo), modules=[m], return_details=True, context_builder=self.self_coding_engine.context_builder
                    )
                except Exception:
                    continue
                scenarios = details.get(m, {})
                scenario_synergy = getattr(tracker_res, "scenario_synergy", {})
                info = self.orphan_traces.setdefault(m, {"parents": []})
                scen_info = info.setdefault("scenarios", {})
                worst_roi = math.inf
                scenario_failed = False
                try:
                    from environment_generator import CANONICAL_PROFILES
                except Exception:  # pragma: no cover - optional dependency
                    CANONICAL_PROFILES = []  # type: ignore[assignment]

                for scen, runs in scenarios.items():
                    names = [s.strip() for s in scen.split("+") if s.strip()]
                    sy_list = scenario_synergy.get(scen, [])
                    try:
                        scen_roi = (
                            float(sy_list[-1].get("synergy_roi", 0.0))
                            if sy_list
                            else 0.0
                        )
                    except Exception:
                        scen_roi = 0.0
                    failed = any(r.get("result", {}).get("exit_code") for r in runs)
                    worst_roi = min(worst_roi, scen_roi)
                    scenario_failed = scenario_failed or failed
                    recorded = False
                    for name in names:
                        if name in CANONICAL_PROFILES:
                            scen_info[name] = {"roi": scen_roi, "failed": failed}
                            recorded = True
                    if not recorded:
                        scen_info[scen] = {"roi": scen_roi, "failed": failed}

                if worst_roi is math.inf:
                    worst_roi = 0.0
                info["robustness"] = float(worst_roi)
                roi_base = self.baseline_tracker.get("roi")
                roi_tol = getattr(getattr(settings, "roi", None), "deviation_tolerance", 0.0)
                roi_delta_base = self.baseline_tracker.get("roi_delta")
                if scenario_failed or worst_roi < roi_base - roi_tol:
                    self.logger.info(
                        "self tests failed",
                        extra=log_record(module=m, robustness=worst_roi),
                    )
                    continue
                roi_total = 0.0
                try:
                    roi_total = sum(tracker_res.module_deltas.get(m, []))
                except Exception:
                    logger.exception("Unhandled exception in self_improvement")
                if roi_total < roi_delta_base - roi_tol:
                    continue
                if reuse_scores.get(m, 0.0) < reuse_threshold:
                    continue
                passed.add(m)
            reclassified = {m for m in passed if m in legacy or m in redundant}
            for name in sorted(reclassified):
                classifications[name] = {"classification": "candidate"}
                info = self.orphan_traces.setdefault(name, {"parents": []})
                info["classification"] = "candidate"
                info["redundant"] = False
                if name in legacy:
                    legacy.remove(name)
                if name in redundant:
                    redundant.remove(name)
            failed_mods = sorted(
                m for m in modules if m not in passed and m not in redundant
            )
            self.logger.info(
                "self test summary",
                extra=log_record(
                    tested=sorted(modules),
                    passed=sorted(passed),
                    failed=failed_mods,
                    redundant=sorted(redundant),
                    legacy=sorted(legacy),
                    reuse_scores={m: reuse_scores.get(m, 0.0) for m in modules},
                ),
            )
            counts = {
                "orphan_modules_tested": len(modules),
                "orphan_modules_passed": len(passed),
                "orphan_modules_failed": len(failed_mods),
                "orphan_modules_reclassified": len(reclassified),
                "orphan_modules_redundant": len(redundant),
                "orphan_modules_legacy": len(legacy),
            }
            self._last_orphan_counts = counts
            tracker = getattr(self, "tracker", None)
            if tracker is not None:
                tracker.register_metrics(*counts.keys())
                base = tracker.roi_history[-1] if tracker.roi_history else 0.0
                tracker.update(base, base, metrics=counts)
            try:
                orphan_modules_tested_total.inc(len(modules))
                orphan_modules_passed_total.inc(len(passed))
                orphan_modules_failed_total.inc(len(failed_mods))
                orphan_modules_reclassified_total.inc(len(reclassified))
                orphan_modules_redundant_total.inc(len(redundant))
                orphan_modules_legacy_total.inc(len(legacy))
            except Exception:
                logger.exception("Unhandled exception in self_improvement")
            try:
                existing_meta = (
                    json.loads(meta_path.read_text()) if meta_path.exists() else {}
                )
            except Exception:  # pragma: no cover - best effort
                existing_meta = {}
            existing_meta.update(classifications)
            try:
                meta_path.parent.mkdir(parents=True, exist_ok=True)
                meta_path.write_text(json.dumps(existing_meta, indent=2))
            except Exception:  # pragma: no cover - best effort
                self.logger.exception("failed to write orphan classifications")
            return passed

        settings = SandboxSettings()
        svc = _STS(
            pytest_args=" ".join(modules),
            include_orphans=True,
            discover_orphans=True,
            discover_isolated=True,
            recursive_orphans=settings.recursive_orphan_scan,
            recursive_isolated=settings.recursive_isolated,
            auto_include_isolated=True,
            include_redundant=settings.test_redundant_modules,
            clean_orphans=True,
            disable_auto_integration=True,
        )

        try:
            asyncio.run(svc._run_once())
        except Exception as exc:  # pragma: no cover - best effort
            self.logger.exception("self test execution failed: %s", exc)
            return set()

        results = svc.results or {}
        integration = results.get("integration")
        if isinstance(integration, dict):
            passing = set(integration.get("integrated", []))
            new_redundant = set(integration.get("redundant", []))
        else:
            passed = set(results.get("orphan_passed", []))
            new_redundant = set(results.get("orphan_redundant", []))
            passing = {p for p in passed if p not in new_redundant}

        passing = {p for p in passing if reuse_scores.get(p, 0.0) >= reuse_threshold}

        filtered: set[str] = set()
        tracker_res = tracker_wf
        for p in passing:
            info = self.orphan_traces.setdefault(p, {"parents": []})
            worst_roi = min(
                float(info.get("robustness", 0.0)),
                float(info.get("workflow_robustness", 0.0)),
            )
            scen_failed = bool(info.get("workflow_failed")) or any(
                v.get("failed") for v in info.get("scenarios", {}).values()
            )
            roi_base = self.baseline_tracker.get("roi")
            roi_tol = getattr(getattr(settings, "roi", None), "deviation_tolerance", 0.0)
            roi_delta_base = self.baseline_tracker.get("roi_delta")
            if scen_failed or worst_roi < roi_base - roi_tol:
                continue
            roi_total = 0.0
            if tracker_res:
                for key, vals in tracker_res.module_deltas.items():
                    if key.startswith(p):
                        roi_total += sum(float(v) for v in vals)
            if roi_total < roi_delta_base - roi_tol:
                continue
            filtered.add(p)
        passing = filtered

        if new_redundant:
            for name in new_redundant:
                classifications[name] = {"classification": "redundant"}
                info = self.orphan_traces.setdefault(name, {"parents": []})
                info["classification"] = "redundant"
                info["redundant"] = True
            redundant.extend(sorted(new_redundant))
            passing -= new_redundant

        reclassified = {m for m in passing if m in legacy or m in redundant}
        for name in sorted(reclassified):
            classifications[name] = {"classification": "candidate"}
            info = self.orphan_traces.setdefault(name, {"parents": []})
            info["classification"] = "candidate"
            info["redundant"] = False
            if name in legacy:
                legacy.remove(name)
            if name in redundant:
                redundant.remove(name)

        failed_mods = [m for m in modules if m not in passing and m not in redundant]
        self.logger.info(
            "self test summary",
            extra=log_record(
                tested=sorted(modules),
                passed=sorted(passing),
                failed=sorted(failed_mods),
                redundant=sorted(redundant),
                legacy=sorted(legacy),
                passed_count=len(passing),
                failed_count=len(failed_mods),
                redundant_count=len(redundant),
                legacy_count=len(legacy),
                reuse_scores={m: reuse_scores.get(m, 0.0) for m in modules},
            ),
        )

        counts = {
            "orphan_modules_tested": len(modules),
            "orphan_modules_passed": len(passing),
            "orphan_modules_failed": len(failed_mods),
            "orphan_modules_reclassified": len(reclassified),
            "orphan_modules_redundant": len(redundant),
            "orphan_modules_legacy": len(legacy),
        }
        self._last_orphan_counts = counts
        tracker = getattr(self, "tracker", None)
        if tracker is not None:
            tracker.register_metrics(*counts.keys())
            base = tracker.roi_history[-1] if tracker.roi_history else 0.0
            tracker.update(base, base, metrics=counts)
        try:
            orphan_modules_tested_total.inc(len(modules))
            orphan_modules_passed_total.inc(len(passing))
            orphan_modules_failed_total.inc(len(failed_mods))
            orphan_modules_reclassified_total.inc(len(reclassified))
            orphan_modules_redundant_total.inc(len(redundant))
            orphan_modules_legacy_total.inc(len(legacy))
        except Exception:
            logger.exception("Unhandled exception in self_improvement")

        if self.data_bot and getattr(self.data_bot, "metrics_db", None):
            try:
                cycle = datetime.utcnow().isoformat()
                db = self.data_bot.metrics_db
                db.log_eval(cycle, "self_test_passed", float(len(passing)))
                if redundant:
                    db.log_eval(cycle, "self_test_redundant", float(len(redundant)))
                if legacy:
                    db.log_eval(cycle, "self_test_legacy", float(len(legacy)))
            except Exception:  # pragma: no cover - best effort
                self.logger.exception("failed to record self test metrics")

        try:
            existing_meta = (
                json.loads(meta_path.read_text()) if meta_path.exists() else {}
            )
        except Exception:  # pragma: no cover - best effort
            existing_meta = {}
        existing_meta.update(classifications)
        try:
            meta_path.parent.mkdir(parents=True, exist_ok=True)
            meta_path.write_text(json.dumps(existing_meta, indent=2))
        except Exception:  # pragma: no cover - best effort
            self.logger.exception("failed to write orphan classifications")

        return passing

    # ------------------------------------------------------------------
    def _integrate_orphans(self, paths: Iterable[str]) -> set[str]:
        """Integrate tested orphan modules and refresh module mappings.

        Candidate modules from ``paths`` are merged with isolates discovered by
        :mod:`scripts.discover_isolated_modules` and expanded through
        :func:`collect_local_dependencies`. The resulting module set is passed to
        :func:`environment.auto_include_modules` with ``recursive=True`` and
        ``validate=True``. After inclusion, ``module_index`` and
        ``module_clusters`` are refreshed, orphan tracking files are cleaned and a
        new orphan discovery round is scheduled so that newly uncovered
        dependencies are evaluated automatically.

        Returns
        -------
        set[str]
            Names of modules that were successfully integrated.
        """

        if not self.module_index:
            return set()

        repo = _repo_path()

        candidates: set[str] = set()
        for p in paths:
            path = Path(p)
            try:
                rel = path.resolve().relative_to(repo).as_posix()
            except Exception:
                rel = path.name
            candidates.add(rel)

        try:
            from scripts.discover_isolated_modules import discover_isolated_modules

            iso = discover_isolated_modules(str(repo), recursive=True)
            candidates.update(iso)
        except Exception as exc:  # pragma: no cover - best effort
            self.logger.exception("isolated module discovery failed: %s", exc)

        try:
            from sandbox_runner.dependency_utils import collect_local_dependencies

            expanded = collect_local_dependencies(sorted(candidates))
        except Exception as exc:  # pragma: no cover - best effort
            self.logger.exception("dependency expansion failed: %s", exc)
            expanded = set(candidates)

        if self.intent_clusterer:
            related: set[str] = set()
            for rel in list(expanded):
                try:
                    matches = self.intent_clusterer.find_modules_related_to(rel)
                    paths = [m.path for m in matches if m.path]
                    clusters = [cid for m in matches for cid in m.cluster_ids]
                    if paths:
                        self.logger.info("intent matches for %s: %s", rel, paths)
                    if clusters:
                        self.logger.info(
                            "intent clusters for %s: %s", rel, clusters
                        )
                    related.update(paths)
                except Exception as exc:  # pragma: no cover - best effort
                    self.logger.exception("intent search failed for %s: %s", rel, exc)
            expanded.update(related)

        try:
            settings = SandboxSettings()
        except Exception:
            try:
                from sandbox_settings import SandboxSettings as _SS  # type: ignore

                settings = _SS()  # type: ignore
            except Exception:  # pragma: no cover - last resort
                settings = type("_SS", (), {"test_redundant_modules": False})()

        traces = getattr(self, "orphan_traces", {})
        mods: set[str] = set()
        legacy = 0
        redundant = 0
        for rel in expanded:
            path = repo / rel
            info = traces.get(rel, {})
            cls = info.get("classification")
            if cls is None:
                try:
                    cls = classify_module(path)
                except Exception as exc:  # pragma: no cover - best effort
                    self.logger.exception("classification failed for %s: %s", path, exc)
                    cls = "candidate"
                info["classification"] = cls
                traces[rel] = info
            if cls == "legacy":
                self.logger.info(
                    "redundant module skipped",
                    extra=log_record(module=rel, classification=cls),
                )
                legacy += 1
                continue
            if cls == "redundant":
                allow = getattr(settings, "test_redundant_modules", False)
                if not allow:
                    self.logger.info(
                        "redundant module skipped",
                        extra=log_record(module=rel, classification=cls),
                    )
                    redundant += 1
                    continue
                redundant += 1
            mods.add(rel)

        try:
            if legacy:
                orphan_modules_legacy_total.inc(legacy)
            if redundant:
                orphan_modules_redundant_total.inc(redundant)
        except Exception:
            logger.exception("Unhandled exception in self_improvement")

        unknown = [m for m in mods if m not in self.module_clusters]
        if not unknown:
            return set()

        try:
            self.module_index.refresh(mods, force=True)
            self.module_index.save()
            self._last_map_refresh = time.time()
            _tracker, tested, updated_wfs, _, _ = self._sandbox_integrate(
                repo,
                modules=sorted(mods),
                logger=self.logger,
                router=GLOBAL_ROUTER,
                context_builder=self.self_coding_engine.context_builder,
            )
            if updated_wfs:
                try:
                    self.logger.info(
                        "workflows updated",
                        extra=log_record(modules=sorted(mods), workflows=updated_wfs),
                    )
                except Exception:
                    logger.exception("Unhandled exception in self_improvement")
                for m in mods:
                    info = traces.setdefault(m, {})
                    info.setdefault("workflows", [])
                    info["workflows"].extend(updated_wfs)
            counts = getattr(self, "_last_orphan_counts", {})
            counts["workflows_updated"] = len(updated_wfs)
            self._last_orphan_counts = counts

            grp_map = {m: self.module_index.get(m) for m in mods}
            for m, idx in grp_map.items():
                self.module_clusters[m] = idx

            try:
                self._update_orphan_modules()
            except Exception as exc:  # pragma: no cover - best effort
                self.logger.exception("recursive orphan update failed: %s", exc)

            data_dir = _data_dir()
            orphan_path = data_dir / "orphan_modules.json"
            meta_path = data_dir / "orphan_classifications.json"
            survivors = [
                m
                for m, info in traces.items()
                if not info.get("redundant") and m not in mods
            ]
            try:
                if orphan_path.exists() or survivors:
                    orphan_path.parent.mkdir(parents=True, exist_ok=True)
                    orphan_path.write_text(json.dumps(sorted(survivors), indent=2))
            except Exception:  # pragma: no cover - best effort
                self.logger.exception("failed to clean orphan modules")
            try:
                if meta_path.exists():
                    meta = json.loads(meta_path.read_text())
                else:
                    meta = {}
                changed = False
                for m in mods:
                    if m in meta:
                        meta.pop(m, None)
                        changed = True
                if changed:
                    if meta:
                        meta_path.write_text(json.dumps(meta, indent=2))
                    else:
                        meta_path.unlink()
            except Exception:  # pragma: no cover - best effort
                self.logger.exception("failed to clean orphan classifications")
            errors: list[str] = []
            try:
                orphan_modules_reintroduced_total.inc(len(mods))
            except Exception as exc:
                self.logger.error(
                    "failed to increment orphan_modules_reintroduced_total: %s",
                    exc,
                    extra=log_record(error=str(exc)),
                )
                errors.append(f"orphan_modules_reintroduced_total: {exc}")
            roi_vals: dict[str, float] = {}
            for m in mods:
                roi_val = 0.0
                pre_bot = getattr(self, "pre_roi_bot", None)
                if pre_bot is not None:
                    try:
                        res = pre_bot.predict_model_roi(m, [])
                        roi_val = float(getattr(res, "roi", 0.0))
                    except Exception:  # pragma: no cover - best effort
                        self.logger.exception("roi prediction failed for %s", m)
                roi_vals[m] = roi_val
                try:
                    data_bot = getattr(self, "data_bot", None)
                    if data_bot and getattr(data_bot, "metrics_db", None):
                        db = data_bot.metrics_db
                        db.log_eval(m, "orphan_module_roi", roi_val)
                        db.log_eval(m, "orphan_module_pass", 1.0)
                        db.log_eval(m, "orphan_module_fail", 0.0)
                except Exception as exc:  # pragma: no cover - best effort
                    self.logger.error(
                        "metrics db logging failed for %s: %s",
                        m,
                        exc,
                        extra=log_record(module=m, error=str(exc)),
                    )
                    errors.append(f"{m}: {exc}")
                try:
                    self.logger.info(
                        "orphan integration stats",
                        extra=log_record(
                            module=m, roi=float(roi_val), passed=True, failed=False
                        ),
                    )
                except Exception as exc:
                    self.logger.error(
                        "failed to log integration stats for %s: %s",
                        m,
                        exc,
                        extra=log_record(module=m, error=str(exc)),
                    )
                    errors.append(f"{m}: {exc}")

            counts = getattr(self, "_last_orphan_counts", {})
            tested = float(counts.get("orphan_modules_tested", len(mods)))
            passed = float(counts.get("orphan_modules_passed", len(mods)))
            pass_rate = passed / tested if tested else 0.0
            try:
                self.baseline_tracker.update(pass_rate=pass_rate)
            except Exception as exc:
                self.logger.debug(
                    "pass rate metric update failed",
                    extra=log_record(pass_rate=pass_rate),
                    exc_info=exc,
                )
            avg_roi = sum(roi_vals.values()) / len(roi_vals) if roi_vals else 0.0
            robust_vals = [
                self.orphan_traces.get(m, {}).get("robustness", 0.0) for m in mods
            ]
            worst_robust = min(robust_vals) if robust_vals else 0.0
            self._last_orphan_metrics = {
                "pass_rate": float(pass_rate),
                "avg_roi": float(avg_roi),
                "worst_scenario_roi": float(worst_robust),
            }

            tracker = getattr(self, "tracker", None)
            if tracker is not None:
                try:
                    tracker.register_metrics(
                        "orphan_pass_rate",
                        "orphan_avg_roi",
                        "orphan_worst_scenario_roi",
                    )
                    base = tracker.roi_history[-1] if tracker.roi_history else 0.0
                    tracker.update(
                        base,
                        base,
                        metrics={
                            "orphan_pass_rate": pass_rate,
                            "orphan_avg_roi": avg_roi,
                            "orphan_worst_scenario_roi": worst_robust,
                        },
                    )
                except Exception as exc:  # pragma: no cover - best effort
                    self.logger.error(
                        "tracker metric update failed: %s",
                        exc,
                        extra=log_record(error=str(exc)),
                    )
                    errors.append(f"tracker: {exc}")
            if errors:
                raise RuntimeError("; ".join(errors))
            return mods
        except Exception as exc:  # pragma: no cover - best effort
            self.logger.exception("orphan integration failed: %s", exc)
            raise

    def _collect_recursive_modules(self, modules: Iterable[str]) -> set[str]:
        """Return ``modules`` plus any local imports they depend on recursively."""
        from pathlib import Path
        from sandbox_runner.dependency_utils import collect_local_dependencies

        repo = _repo_path().resolve()
        traces = getattr(self, "orphan_traces", None)

        roots: list[str] = []
        initial: dict[str, list[str]] = {}
        for m in modules:
            p = Path(m)
            if not p.is_absolute():
                p = repo / p
            try:
                rel = p.resolve().relative_to(repo).as_posix()
            except Exception:
                rel = p.as_posix()
            roots.append(p.as_posix())
            if traces is not None:
                parents = list(traces.get(rel, {}).get("parents", []))
                initial[rel] = parents
                traces.setdefault(rel, {"parents": parents})

        def _on_module(rel: str, _path: Path, parents: list[str]) -> None:
            if traces is None:
                return
            entry = traces.setdefault(rel, {"parents": []})
            if parents:
                entry["parents"] = list(
                    dict.fromkeys(entry.get("parents", []) + parents)
                )

        def _on_dep(dep_rel: str, _parent_rel: str, chain: list[str]) -> None:
            if traces is None:
                return
            entry = traces.setdefault(dep_rel, {"parents": []})
            if chain:
                entry["parents"] = list(dict.fromkeys(entry.get("parents", []) + chain))

        deps = collect_local_dependencies(
            roots,
            initial_parents=initial if traces is not None else None,
            on_module=_on_module if traces is not None else None,
            on_dependency=_on_dep if traces is not None else None,
        )
        return deps | set(initial.keys())

    # ------------------------------------------------------------------
    def _post_round_orphan_scan(self) -> None:
        """Run a post-round orphan discovery and integration pass."""

        repo = _repo_path()
        try:
            added, syn_ok, intent_ok = self._post_round_scan(
                repo,
                logger=self.logger,
                router=GLOBAL_ROUTER,
                context_builder=self.self_coding_engine.context_builder,
            )
        except RuntimeError as exc:
            self.logger.error("post_round_orphan_scan unavailable: %s", exc)
            raise
        except Exception:  # pragma: no cover - best effort
            self.logger.exception("recursive orphan integration failed")
            return

        self.logger.info(
            "post_round_orphan_scan added=%d synergy_ok=%s intent_ok=%s",
            len(added),
            syn_ok,
            intent_ok,
        )

        record_new = getattr(self, "_record_new_modules", None)
        if record_new:
            try:
                record_new(set(added))
            except Exception:  # pragma: no cover - best effort
                self.logger.exception("record new modules failed")

    # ------------------------------------------------------------------
    def _update_orphan_modules(self, modules: Iterable[str] | None = None) -> None:
        """Discover orphan modules and update the tracking file or integrate ``modules``."""
        repo = _repo_path()
        data_dir = _data_dir()
        path = data_dir / "orphan_modules.json"

        if not hasattr(self, "orphan_traces"):
            self.orphan_traces = {}

        if modules:
            modules = list(modules)
            meta_path = data_dir / "orphan_classifications.json"
            try:
                existing_meta = (
                    json.loads(meta_path.read_text()) if meta_path.exists() else {}
                )
            except Exception:  # pragma: no cover - best effort
                existing_meta = {}
            try:
                settings = SandboxSettings()
            except Exception:  # pragma: no cover - fallback for tests
                from sandbox_settings import SandboxSettings as _SS  # type: ignore

                settings = _SS()

            legacy_mods: list[str] = []
            redundant_mods: list[str] = []
            filtered: list[str] = []
            for m in modules:
                p = Path(m)
                try:
                    rel = p.resolve().relative_to(repo).as_posix()
                except Exception:
                    rel = str(p)
                cls = existing_meta.get(rel, {}).get("classification")
                info = self.orphan_traces.setdefault(rel, {"parents": []})
                if cls in {"legacy", "redundant"}:
                    info["classification"] = cls
                    info["redundant"] = True
                    if not getattr(settings, "test_redundant_modules", False):
                        if cls == "legacy":
                            legacy_mods.append(rel)
                        else:
                            redundant_mods.append(rel)
                        continue
                filtered.append(m)
            modules = filtered
            if legacy_mods or redundant_mods:
                try:
                    if legacy_mods:
                        orphan_modules_legacy_total.set(len(legacy_mods))
                    if redundant_mods:
                        orphan_modules_redundant_total.set(len(redundant_mods))
                except Exception:
                    logger.exception("Unhandled exception in self_improvement")
                existing_meta.update(
                    {m: {"classification": "legacy"} for m in legacy_mods}
                )
                existing_meta.update(
                    {m: {"classification": "redundant"} for m in redundant_mods}
                )
                try:
                    meta_path.parent.mkdir(parents=True, exist_ok=True)
                    meta_path.write_text(json.dumps(existing_meta, indent=2))
                except Exception:  # pragma: no cover - best effort
                    self.logger.exception("failed to write orphan classifications")

            try:
                from scripts.discover_isolated_modules import discover_isolated_modules

                iso_mods = discover_isolated_modules(
                    str(repo), recursive=getattr(settings, "recursive_isolated", True)
                )
                modules.extend(sorted(iso_mods))
            except Exception as exc:  # pragma: no cover - best effort
                self.logger.exception("isolated module discovery failed: %s", exc)

            try:
                from sandbox_runner import discover_recursive_orphans as _discover

                trace = _discover(str(repo), module_map=data_dir / "module_map.json")
                for k, v in trace.items():
                    info: dict[str, Any] = {
                        "parents": [
                            str(resolve_path(str(Path(*p.split(".")).with_suffix(".py"))))
                            for p in (v.get("parents") if isinstance(v, dict) else v)
                        ]
                    }
                    if isinstance(v, dict) and "redundant" in v:
                        info["redundant"] = bool(v["redundant"])
                    mod_path = str(resolve_path(str(Path(*k.split(".")).with_suffix(".py"))))
                    self.orphan_traces.setdefault(mod_path, info)
                    modules.append(mod_path)
            except Exception as exc:  # pragma: no cover - best effort
                self.logger.exception("orphan discovery failed: %s", exc)

            try:
                repo_mods = sorted(self._collect_recursive_modules(modules))
            except Exception as exc:  # pragma: no cover - best effort
                self.logger.exception("dependency expansion failed: %s", exc)
                repo_mods = sorted(set(modules))

            passing = self._test_orphan_modules(repo_mods)

            if passing:
                metrics = {
                    m: self.orphan_traces.get(m, {}).get("side_effects", 0)
                    for m in passing
                }
                safe: list[str] = []
                settings = SandboxSettings()
                hist = self.baseline_tracker.to_dict().get("side_effects", [])
                avg = self.baseline_tracker.get("side_effects") if hist else 0.0
                if len(hist) >= 2:
                    std = self.baseline_tracker.std("side_effects")
                else:
                    std = max(1.0, abs(avg) * 0.1)
                threshold = avg + getattr(
                    settings, "side_effect_dev_multiplier", 1.0
                ) * std
                for m in passing:
                    if metrics.get(m, 0) > threshold:
                        info = self.orphan_traces.setdefault(m, {"parents": []})
                        info["heavy_side_effects"] = True
                    else:
                        safe.append(m)
                if safe:
                    added_modules: set[str] = set()
                    try:
                        _tracker, tested = environment.auto_include_modules(
                            sorted(safe),
                            recursive=True,
                            validate=True,
                            context_builder=self.self_coding_engine.context_builder,
                        )
                        added_modules.update(tested.get("added", []))
                        try:
                            kwargs: dict[str, object] = {}
                            try:
                                if (
                                    "side_effects"
                                    in inspect.signature(
                                        environment.try_integrate_into_workflows
                                    ).parameters
                                ):
                                    kwargs["side_effects"] = {
                                        m: metrics.get(m, 0) for m in safe
                                    }
                            except Exception:
                                logger.exception("Unhandled exception in self_improvement")
                            environment.try_integrate_into_workflows(
                                sorted(safe),
                                **kwargs,
                                context_builder=self.self_coding_engine.context_builder,
                            )
                        except Exception as exc:  # pragma: no cover - best effort
                            self.logger.exception(
                                "workflow integration failed: %s", exc
                            )
                            raise
                    except Exception as exc:  # pragma: no cover - best effort
                        self.logger.exception("auto inclusion failed: %s", exc)
                        raise
                    record_new = getattr(self, "_record_new_modules", None)
                    if record_new:
                        record_new(added_modules)

                    abs_paths = [str(repo / p) for p in safe]
                    integrated: set[str] = set()
                    try:
                        integrated = self._integrate_orphans(abs_paths)
                    except Exception as exc:  # pragma: no cover - best effort
                        self.logger.exception("orphan integration failed: %s", exc)
                        raise
                    try:
                        self._refresh_module_map(safe)
                    except Exception as exc:  # pragma: no cover - best effort
                        self.logger.exception("module map refresh failed: %s", exc)
                        raise
                    try:
                        survivors = [
                            m for m in modules if Path(m).name not in integrated
                        ]
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_text(json.dumps(sorted(survivors), indent=2))
                        meta_path = data_dir / "orphan_classifications.json"
                        if meta_path.exists():
                            meta = json.loads(meta_path.read_text())
                            changed = False
                            for m in integrated:
                                if m in meta:
                                    meta.pop(m, None)
                                    changed = True
                            if changed:
                                if meta:
                                    meta_path.write_text(json.dumps(meta, indent=2))
                                else:
                                    meta_path.unlink()
                    except Exception:  # pragma: no cover - best effort
                        self.logger.exception("failed to write orphan modules")
            return

        try:
            settings = SandboxSettings()
        except Exception:  # pragma: no cover - fallback for tests
            from sandbox_settings import SandboxSettings as _SS  # type: ignore

            settings = _SS()
        modules: list[str] = []
        try:
            from scripts.discover_isolated_modules import discover_isolated_modules

            iso_mods = discover_isolated_modules(
                str(repo), recursive=getattr(settings, "recursive_isolated", True)
            )
            modules.extend(sorted(iso_mods))
            for m in iso_mods:
                self.orphan_traces.setdefault(m, {"parents": []})
        except Exception as exc:  # pragma: no cover - best effort
            self.logger.exception("isolated module discovery failed: %s", exc)

        try:
            from sandbox_runner import discover_recursive_orphans as _discover

            trace = _discover(str(repo), module_map=data_dir / "module_map.json")
            for mod, info in trace.items():
                parents = [
                    str(resolve_path(str(Path(*p.split(".")).with_suffix(".py"))))
                    for p in (info.get("parents") if isinstance(info, dict) else info)
                ]
                entry: dict[str, Any] = {"parents": parents}
                if isinstance(info, dict):
                    cls = info.get("classification")
                    if cls is not None:
                        entry["classification"] = cls
                        entry["redundant"] = cls != "candidate"
                    elif "redundant" in info:
                        entry["redundant"] = bool(info["redundant"])
                mod_path = str(resolve_path(str(Path(*mod.split(".")).with_suffix(".py"))))
                self.orphan_traces.setdefault(mod_path, entry).update(entry)
                modules.append(mod_path)
        except Exception as exc:  # pragma: no cover - best effort
            self.logger.exception("orphan discovery failed: %s", exc)

        try:
            existing = json.loads(path.read_text()) if path.exists() else []
        except Exception:  # pragma: no cover - best effort
            existing = []
        if isinstance(existing, list):
            modules.extend(existing)
            for m in existing:
                self.orphan_traces.setdefault(m, {"parents": []})
        elif isinstance(existing, dict):
            for m, info in existing.items():
                modules.append(m)
                if isinstance(info, dict):
                    self.orphan_traces.setdefault(
                        m, {"parents": info.get("parents", [])}
                    )

        if not modules:
            tester = getattr(self, "_test_orphan_modules", None)
            if tester is not None:
                tester([])
            return

        modules = sorted(set(modules))

        filtered: list[str] = []
        skipped: list[str] = []
        for m in modules:
            p = Path(m)
            info = self.orphan_traces.setdefault(m, {"parents": []})
            try:
                cls, _ = classify_module(p)
            except Exception as exc:  # pragma: no cover - best effort
                self.logger.exception("classification failed for %s: %s", p, exc)
                cls = "candidate"
            try:
                redundant = analyze_redundancy(p)
            except Exception:  # pragma: no cover - best effort
                redundant = False
            is_redundant = redundant or cls in {"legacy", "redundant"}
            if is_redundant and not getattr(settings, "test_redundant_modules", False):
                info["classification"] = cls
                info["redundant"] = True
                skipped.append(m)
                self.logger.info(
                    "redundant module skipped",
                    extra=log_record(module=m, classification=cls),
                )
                try:
                    if cls == "legacy":
                        orphan_modules_legacy_total.inc(1)
                    else:
                        orphan_modules_redundant_total.inc(1)
                except Exception:
                    logger.exception("Unhandled exception in self_improvement")
                continue
            if is_redundant:
                info["classification"] = cls
                info["redundant"] = True
            filtered.append(m)

        if skipped:
            self.logger.info(
                "redundant modules skipped", extra=log_record(modules=sorted(skipped))
            )

        if not filtered:
            tester = getattr(self, "_test_orphan_modules", None)
            if tester is not None:
                tester([])
            return

        try:
            if hasattr(self, "_collect_recursive_modules"):
                repo_mods = sorted(self._collect_recursive_modules(filtered))
            else:  # pragma: no cover - fallback
                from sandbox_runner.dependency_utils import collect_local_dependencies

                repo_mods = sorted(collect_local_dependencies(filtered))
        except Exception as exc:  # pragma: no cover - best effort
            self.logger.exception("dependency expansion failed: %s", exc)
            repo_mods = sorted(set(filtered))

        passing = self._test_orphan_modules(repo_mods)
        integrated: set[str] = set()
        if passing:
            metrics = {
                m: self.orphan_traces.get(m, {}).get("side_effects", 0) for m in passing
            }
            safe: list[str] = []
            settings = SandboxSettings()
            hist = self.baseline_tracker.to_dict().get("side_effects", [])
            avg = self.baseline_tracker.get("side_effects") if hist else 0.0
            if len(hist) >= 2:
                std = self.baseline_tracker.std("side_effects")
            else:
                std = max(1.0, abs(avg) * 0.1)
            threshold = avg + getattr(
                settings, "side_effect_dev_multiplier", 1.0
            ) * std
            for m in passing:
                if metrics.get(m, 0) > threshold:
                    info = self.orphan_traces.setdefault(m, {"parents": []})
                    info["heavy_side_effects"] = True
                else:
                    safe.append(m)
            if safe:
                added_modules: set[str] = set()
                try:
                    _tracker, tested = environment.auto_include_modules(
                        sorted(safe),
                        recursive=True,
                        validate=True,
                        context_builder=self.self_coding_engine.context_builder,
                    )
                    added_modules.update(tested.get("added", []))
                    try:
                        kwargs: dict[str, object] = {}
                        try:
                            if (
                                "side_effects"
                                in inspect.signature(
                                    environment.try_integrate_into_workflows
                                ).parameters
                            ):
                                kwargs["side_effects"] = {
                                    m: metrics.get(m, 0) for m in safe
                                }
                        except Exception:
                            logger.exception("Unhandled exception in self_improvement")
                        environment.try_integrate_into_workflows(
                            sorted(safe),
                            **kwargs,
                            context_builder=self.self_coding_engine.context_builder,
                        )
                    except Exception as exc:  # pragma: no cover - best effort
                        self.logger.exception(
                            "workflow integration failed: %s", exc
                        )
                        raise
                except Exception as exc:  # pragma: no cover - best effort
                    self.logger.exception("auto inclusion failed: %s", exc)
                    raise
                record_new = getattr(self, "_record_new_modules", None)
                if record_new:
                    record_new(added_modules)

                repo_paths = [str(repo / p) for p in safe]
                try:
                    integrated = self._integrate_orphans(repo_paths)
                except Exception as exc:  # pragma: no cover - best effort
                    self.logger.exception("orphan integration failed: %s", exc)
                    raise
                try:
                    if hasattr(self, "_refresh_module_map"):
                        self._refresh_module_map(safe)
                except Exception as exc:  # pragma: no cover - best effort
                    self.logger.exception("module map refresh failed: %s", exc)
                    raise

        if SandboxSettings().clean_orphans:
            survivors = [m for m in filtered if m not in passing]
        else:
            survivors = filtered
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(sorted(survivors), indent=2))
            self.logger.info(
                "orphan modules updated", extra=log_record(count=len(survivors))
            )
        except Exception:  # pragma: no cover - best effort
            self.logger.exception("failed to write orphan modules")

    # ------------------------------------------------------------------
    def _load_orphan_candidates(self) -> list[str]:
        """Read orphan module candidates from the tracking file."""
        data_dir = _data_dir()
        path = data_dir / "orphan_modules.json"
        try:
            if path.exists():
                data = json.loads(path.read_text()) or {}
            else:
                return []
        except Exception:  # pragma: no cover - best effort
            self.logger.exception("failed to load orphan candidates")
            return []
        if isinstance(data, list):
            return [str(p) for p in data]
        if isinstance(data, dict):
            return [
                m
                for m, info in data.items()
                if not isinstance(info, dict)
                or info.get("classification", "candidate") == "candidate"
            ]
        return []

    # ------------------------------------------------------------------
    def retest_redundant_modules(self) -> None:
        """Re-run self tests for modules previously classified as redundant.

        Modules listed in ``orphan_modules.json`` that have a corresponding
        classification of ``"redundant"`` in ``orphan_classifications.json``
        are re-executed via :func:`environment.auto_include_modules` with
        ``validate=True`` so they can be reintegrated once they start passing
        their self tests again.
        """

        repo = _repo_path()
        data_dir = _data_dir()
        if not data_dir.is_absolute():
            data_dir = repo / data_dir
        orphan_path = data_dir / "orphan_modules.json"
        meta_path = data_dir / "orphan_classifications.json"
        try:
            modules = (
                json.loads(orphan_path.read_text()) if orphan_path.exists() else []
            )
        except Exception:  # pragma: no cover - best effort
            self.logger.exception("failed to load orphan modules")
            return
        try:
            meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        except Exception:  # pragma: no cover - best effort
            meta = {}

        redundant = [
            m
            for m in modules
            if isinstance(m, str)
            and meta.get(m, {}).get("classification") == "redundant"
        ]
        if not redundant:
            return
        added_modules: set[str] = set()
        try:
            _tracker, tested = environment.auto_include_modules(
                sorted(redundant),
                recursive=True,
                validate=True,
                context_builder=self.self_coding_engine.context_builder,
            )
            added_modules.update(tested.get("added", []))
        except Exception as exc:  # pragma: no cover - best effort
            self.logger.exception("redundant module recheck failed: %s", exc)
        record_new = getattr(self, "_record_new_modules", None)
        if record_new:
            record_new(added_modules)

    # ------------------------------------------------------------------
    def _refresh_module_map(self, modules: Iterable[str] | None = None) -> None:
        """Refresh module grouping when new modules appear.

        Modules accepted for integration are auto-included with recursive
        dependency expansion. Redundant or legacy modules identified by
        :func:`classify_module` are skipped. When
        ``SANDBOX_RECURSIVE_ORPHANS`` is enabled, orphan discovery is executed
        again after integration to traverse any newly uncovered dependencies.
        """
        if modules:
            repo_mods = self._collect_recursive_modules(modules)
            passing = self._test_orphan_modules(repo_mods)
            if passing:
                repo = _repo_path()
                added_modules: set[str] = set()
                try:
                    _tracker, tested = environment.auto_include_modules(
                        sorted(passing),
                        recursive=True,
                        validate=True,
                        context_builder=self.self_coding_engine.context_builder,
                    )
                    added_modules.update(tested.get("added", []))
                except Exception:
                    try:
                        _tracker, tested = auto_include_modules(
                            sorted(passing),
                            recursive=True,
                            validate=True,
                            context_builder=self.self_coding_engine.context_builder,
                        )
                        added_modules.update(tested.get("added", []))
                    except TypeError:
                        _tracker, tested = auto_include_modules(
                            sorted(passing),
                            recursive=True,
                            context_builder=self.self_coding_engine.context_builder,
                        )
                        added_modules.update(tested.get("added", []))
                    except Exception as exc:  # pragma: no cover - best effort
                        self.logger.exception("auto inclusion failed: %s", exc)
                        tested = {}
                record_new = getattr(self, "_record_new_modules", None)
                if record_new:
                    record_new(added_modules)
                integrated: set[str] = set()
                try:
                    abs_paths = [str(repo / p) for p in passing]
                    integrated = self._integrate_orphans(abs_paths)
                except Exception as exc:  # pragma: no cover - best effort
                    self.logger.exception("orphan integration failed: %s", exc)
                try:
                    self._update_orphan_modules()
                except Exception as exc:  # pragma: no cover - best effort
                    self.logger.exception(
                        "post integration orphan update failed: %s", exc
                    )
            return

        if not self.auto_refresh_map or not self.module_index:
            return
        pdb = self.patch_db or (self.data_bot.patch_db if self.data_bot else None)
        if not pdb:
            return
        try:
            iso_ts = datetime.utcfromtimestamp(self._last_map_refresh).isoformat()
            with pdb._connect() as conn:
                rows = conn.execute(
                    "SELECT filename FROM patch_history WHERE ts > ?",
                    (iso_ts,),
                ).fetchall()
        except Exception as exc:  # pragma: no cover - database issues
            self.logger.exception("module refresh query failed: %s", exc)
            return
        repo = _repo_path()
        pending: dict[str, Path] = {}
        for r in rows:
            p = Path(r[0])
            abs_p = p if p.is_absolute() else repo / p
            try:
                rel = abs_p.resolve().relative_to(repo).as_posix()
            except Exception:
                continue
            if rel in self.module_clusters or rel in pending:
                continue
            pending[rel] = abs_p
        new_mods: set[str] = set()
        skipped: set[str] = set()
        for rel, path in pending.items():
            try:
                cls = classify_module(path)
                if cls != "candidate":
                    skipped.add(rel)
                    self.logger.info(
                        "redundant module skipped",
                        extra=log_record(module=rel, classification=cls),
                    )
                    try:
                        if cls == "legacy":
                            orphan_modules_legacy_total.inc(1)
                        else:
                            orphan_modules_redundant_total.inc(1)
                    except Exception:
                        logger.exception("Unhandled exception in self_improvement")
                    continue
            except Exception as exc:  # pragma: no cover - best effort
                self.logger.exception("classification failed for %s: %s", path, exc)
            new_mods.add(rel)
        if skipped:
            self.logger.info(
                "redundant modules skipped", extra=log_record(modules=sorted(skipped))
            )
        if not new_mods:
            return
        try:
            exclude_env = SandboxSettings().exclude_dirs
            exclude = [e for e in exclude_env.split(",") if e] if exclude_env else None
            mapping = build_module_map(repo, ignore=exclude)
            mapping = {
                (
                    str(resolve_path(f"{k}.py"))
                    if not k.endswith(".py")
                    else str(resolve_path(k))
                ): v
                for k, v in mapping.items()
            }
            if skipped:
                for key in list(mapping.keys()):
                    if key in skipped:
                        mapping.pop(key, None)
            self.module_index.merge_groups(mapping)
            self.module_clusters.update(mapping)
            data_dir = _data_dir()
            out = data_dir / "module_map.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            with open(out, "w", encoding="utf-8") as fh:
                json.dump(
                    {
                        "modules": self.module_index._map,
                        "groups": self.module_index._groups,
                    },
                    fh,
                    indent=2,
                )
            self._last_map_refresh = time.time()
            if self.meta_logger and hasattr(self.meta_logger, "audit"):
                try:
                    self.meta_logger.audit.record(
                        {"event": "module_map_refreshed", "modules": sorted(new_mods)}
                    )
                except Exception:
                    logger.exception("Unhandled exception in self_improvement")
            self.logger.info(
                "module map refreshed",
                extra=log_record(modules=sorted(new_mods)),
            )
            try:
                abs_new = [str(pending[m]) for m in new_mods]
                deps = self._collect_recursive_modules(abs_new)
                abs_deps = [str(repo / p) for p in deps]
                try:
                    _tracker, tested = environment.auto_include_modules(
                        sorted(deps),
                        recursive=True,
                        validate=True,
                        context_builder=self.self_coding_engine.context_builder,
                    )
                    record_new = getattr(self, "_record_new_modules", None)
                    if record_new:
                        record_new(set(tested.get("added", [])))
                except Exception as exc:  # pragma: no cover - best effort
                    self.logger.exception("auto inclusion failed: %s", exc)
                deps_integrated: set[str] = set()
                try:
                    deps_integrated = self._integrate_orphans(abs_deps)
                except Exception as exc:  # pragma: no cover - best effort
                    self.logger.exception("orphan integration failed: %s", exc)
                try:
                    self._update_orphan_modules()
                except Exception as exc:  # pragma: no cover - best effort
                    self.logger.exception(
                        "post integration orphan update failed: %s", exc
                    )
            except Exception as exc:  # pragma: no cover - best effort
                self.logger.exception("orphan integration failed: %s", exc)
        except Exception as exc:  # pragma: no cover - runtime issues
            self.logger.exception("module map refresh failed: %s", exc)

    # ------------------------------------------------------------------
    def enqueue_preventative_fixes(self, modules: Iterable[str]) -> None:
        """Queue modules for preventative patch generation."""
        for mod in modules:
            m = str(mod)
            if m and m not in self._preventative_queue:
                self._preventative_queue.append(m)

    def _process_preventative_queue(self) -> None:
        """Generate patches for queued modules."""
        if not self._preventative_queue or not self.self_coding_engine:
            self._preventative_queue.clear()
            return
        queue = list(self._preventative_queue)
        self._preventative_queue.clear()
        scored = self._score_modifications(queue)
        for mod, roi_est, category, weight in scored:
            self.logger.info(
                "patch candidate",
                extra=log_record(
                    module=mod,
                    roi_category=category,
                    roi_estimate=roi_est,
                    weight=weight,
                ),
            )
            try:
                patch_id = None
                with tempfile.TemporaryDirectory() as before_dir, tempfile.TemporaryDirectory() as after_dir:
                    orig = Path(mod)
                    rel = orig.name if orig.is_absolute() else orig
                    src = resolve_path(
                        f"{orig}.py" if orig.suffix == "" else str(orig)
                    )
                    before_target = Path(before_dir) / rel
                    before_target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, before_target)
                    self.logger.info(
                        "gpt_suggestion",
                        extra=log_record(
                            module=mod,
                            suggestion="preventative_patch",
                            tags=[ERROR_FIX],
                        ),
                    )
                    try:
                        log_with_tags(
                            self.gpt_memory,
                            f"preventative_patch:{mod}",
                            "suggested",
                            tags=[f"self_improvement.preventative_patch", FEEDBACK, IMPROVEMENT_PATH, ERROR_FIX, INSIGHT],
                        )
                    except Exception:
                        self.logger.exception(
                            "memory logging failed", extra=log_record(module=mod)
                        )
                    patch_id = self._generate_patch_with_memory(
                        mod, "preventative_patch"
                    )
                    self.logger.info(
                        "patch result",
                        extra=log_record(
                            module=mod,
                            patch_id=patch_id,
                            success=patch_id is not None,
                            tags=["fix_result"],
                        ),
                    )
                    if patch_id is not None:
                        after_target = Path(after_dir) / rel
                        after_target.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(src, after_target)
                        diff_data = _collect_diff_data(Path(before_dir), Path(after_dir))
                        self._pre_commit_alignment_check(diff_data)
                        findings = flag_alignment_issues(diff_data)
                        if findings:
                            log_violation(
                                str(patch_id),
                                "alignment_warning",
                                1,
                                {"findings": findings},
                                alignment_warning=True,
                            )
                        self._alignment_review_last_commit(
                            f"preventative_patch_{patch_id}"
                        )
                        self._flag_patch_alignment(
                            patch_id,
                            {
                                "trigger": "preventative_patch",
                                "module": str(mod),
                                "patch_id": patch_id,
                            },
                        )
                        try:
                            repo = resolve_path(".")
                            self._sandbox_integrate(
                                repo,
                                router=GLOBAL_ROUTER,
                                context_builder=self.self_coding_engine.context_builder,
                            )
                        except Exception:
                            self.logger.exception(
                                "post_patch_orphan_integration_failed"
                            )
                        try:
                            self._post_round_orphan_scan()
                        except Exception:
                            self.logger.exception(
                                "recursive orphan inclusion failed",
                                extra=log_record(module=mod),
                            )
                if self.error_bot and hasattr(self.error_bot, "db"):
                    event_cls = _get_telemetry_event()
                    if event_cls is not None:
                        try:
                            self.error_bot.db.add_telemetry(
                                event_cls(
                                    module=str(mod),
                                    patch_id=patch_id,
                                    resolution_status="attempted",
                                )
                            )
                        except Exception:
                            self.logger.exception(
                                "telemetry record failed", extra=log_record(module=mod)
                            )
                if self.error_predictor:
                    try:
                        self.error_predictor.graph.add_telemetry_event(
                            self.bot_name,
                            "preemptive_patch",
                            str(mod),
                            patch_id=patch_id,
                        )
                    except Exception:
                            self.logger.exception(
                                "graph patch record failed", extra=log_record(module=mod)
                            )
            except Exception:
                self.logger.exception(
                    "preemptive fix failed", extra=log_record(module=mod)
                )
            if self.use_adaptive_roi:
                self._log_action("preventative_patch", mod, roi_est, category)
            try:
                self.logger.info(
                    "post_patch_orphan_discovery",
                    extra=log_record(module=mod),
                )
                repo = resolve_path(".")
                self._sandbox_integrate(
                    repo,
                    router=GLOBAL_ROUTER,
                    context_builder=self.self_coding_engine.context_builder,
                )
            except Exception:
                self.logger.exception(
                    "post_patch_orphan_discovery_failed",
                    extra=log_record(module=mod),
                )

    def _apply_high_risk_patches(self) -> None:
        """Predict high-risk modules and attempt preemptive fixes."""
        if not (self.error_predictor and self.auto_patch_high_risk):
            return
        try:
            high_risk = self.error_predictor.predict_high_risk_modules()
            self.error_predictor.graph.update_error_stats(self.error_bot.db)
            if not high_risk:
                return
            scored = self._score_modifications(high_risk)
            self.logger.info(
                "high risk modules",
                extra=log_record(modules=[m for m, _, _, _ in scored]),
            )
            for mod, roi_est, category, weight in scored:
                self.logger.info(
                    "patch candidate",
                    extra=log_record(
                        module=mod,
                        roi_category=category,
                        roi_estimate=roi_est,
                        weight=weight,
                    ),
                )
                try:
                    patch_id = None
                    with tempfile.TemporaryDirectory() as before_dir, tempfile.TemporaryDirectory() as after_dir:
                        orig = Path(mod)
                        rel = orig.name if orig.is_absolute() else orig
                        src = resolve_path(
                            f"{orig}.py" if orig.suffix == "" else str(orig)
                        )
                        before_target = Path(before_dir) / rel
                        before_target.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(src, before_target)
                        self.logger.info(
                            "gpt_suggestion",
                            extra=log_record(
                                module=mod,
                                suggestion="high_risk_patch",
                                tags=[ERROR_FIX],
                            ),
                        )
                        try:
                            log_with_tags(
                                self.gpt_memory,
                                f"high_risk_patch:{mod}",
                                "suggested",
                                tags=[f"self_improvement.high_risk_patch", FEEDBACK, IMPROVEMENT_PATH, ERROR_FIX, INSIGHT],
                            )
                        except Exception:
                            self.logger.exception(
                                "memory logging failed", extra=log_record(module=mod)
                            )
                        patch_id = self._generate_patch_with_memory(
                            mod, "high_risk_patch"
                        )
                        self.logger.info(
                            "patch result",
                            extra=log_record(
                                module=mod,
                                patch_id=patch_id,
                                success=patch_id is not None,
                                tags=["fix_result"],
                            ),
                        )
                        if patch_id is not None:
                            after_target = Path(after_dir) / rel
                            after_target.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copy2(src, after_target)
                            diff_data = _collect_diff_data(Path(before_dir), Path(after_dir))
                            self._pre_commit_alignment_check(diff_data)
                            findings = flag_alignment_issues(diff_data)
                            if findings:
                                log_violation(
                                    str(patch_id),
                                    "alignment_warning",
                                    1,
                                    {"findings": findings},
                                    alignment_warning=True,
                                )
                            self._alignment_review_last_commit(
                                f"high_risk_patch_{patch_id}"
                            )
                            self._flag_patch_alignment(
                                patch_id,
                                {
                                    "trigger": "high_risk_patch",
                                    "module": str(mod),
                                    "patch_id": patch_id,
                                },
                            )
                            try:
                                repo = resolve_path(".")
                                self._sandbox_integrate(
                                    repo,
                                    router=GLOBAL_ROUTER,
                                    context_builder=self.self_coding_engine.context_builder,
                                )
                            except Exception:
                                self.logger.exception(
                                    "post_patch_orphan_integration_failed"
                                )
                            try:
                                self._post_round_orphan_scan()
                            except Exception:
                                self.logger.exception(
                                    "recursive orphan inclusion failed",
                                    extra=log_record(module=mod),
                                )
                    if self.error_bot and hasattr(self.error_bot, "db"):
                        event_cls = _get_telemetry_event()
                        if event_cls is not None:
                            try:
                                self.error_bot.db.add_telemetry(
                                    event_cls(
                                        module=str(mod),
                                        patch_id=patch_id,
                                        resolution_status="attempted",
                                    )
                                )
                            except Exception:
                                self.logger.exception(
                                    "telemetry record failed", extra=log_record(module=mod)
                                )
                    if self.error_predictor:
                        try:
                            self.error_predictor.graph.add_telemetry_event(
                                self.bot_name,
                                "preemptive_patch",
                                str(mod),
                                patch_id=patch_id,
                            )
                        except Exception:
                            self.logger.exception(
                                "graph patch record failed", extra=log_record(module=mod)
                            )
                except Exception:
                    self.logger.exception(
                        "preemptive fix failed", extra=log_record(module=mod)
                    )
                if self.use_adaptive_roi:
                    self._log_action("high_risk_patch", mod, roi_est, category)
                try:
                    self.logger.info(
                        "post_patch_orphan_discovery",
                        extra=log_record(module=mod),
                    )
                    repo = resolve_path(".")
                    self._sandbox_integrate(
                        repo,
                        router=GLOBAL_ROUTER,
                        context_builder=self.self_coding_engine.context_builder,
                    )
                except Exception:
                    self.logger.exception(
                        "post_patch_orphan_discovery_failed",
                        extra=log_record(module=mod),
                    )
        except Exception as exc:
            self.logger.exception(
                "high risk module prediction failed: %s", exc
            )

    def _evaluate_module_relevance(self) -> None:
        """Evaluate module usage and record relevancy recommendations."""
        try:
            settings = SandboxSettings()
            k = float(getattr(settings, "relevancy_deviation_multiplier", 1.0))
            min_history = int(getattr(settings, "relevancy_history_min_length", 0))
            auto_process = getattr(
                settings, "auto_process_relevancy_flags", True
            )
            metrics_db_path = resolve_path(
                getattr(
                    settings,
                    "relevancy_metrics_db_path",
                    DEFAULT_RELEVANCY_METRICS_DB,
                )
            )
        except Exception:
            k = 1.0
            min_history = 0
            auto_process = True
            metrics_db_path = DEFAULT_RELEVANCY_METRICS_DB

        for mod, counts in getattr(self.relevancy_radar, "_metrics", {}).items():
            impact_val = float(counts.get("impact", 0.0)) + float(
                counts.get("output_impact", 0.0)
            )
            score = (
                float(counts.get("imports", 0.0))
                + float(counts.get("executions", 0.0))
                + impact_val
            )
            metric_name = f"relevancy:{mod}"
            self.baseline_tracker.update(relevancy=score, **{metric_name: score})

        avg = self.baseline_tracker.get("relevancy")
        std = self.baseline_tracker.std("relevancy")
        replace_threshold = max(avg - k * std, 0.0)
        compress_threshold = max(avg - 2 * k * std, 0.0)

        history_len = len(self.baseline_tracker.to_dict().get("relevancy", []))
        if history_len < min_history:
            auto_process = False

        flags: dict[str, str] = dict(self.relevancy_flags)
        try:
            radar_flags = self.relevancy_radar.evaluate_final_contribution(
                compress_threshold, replace_threshold
            )
            flags.update(radar_flags)
        except Exception:
            self.logger.exception("relevancy evaluation failed")
        try:
            metrics_db = resolve_path(metrics_db_path)
            scan_flags = radar_scan(metrics_db)
            if scan_flags:
                flags.update(scan_flags)
        except Exception:
            self.logger.exception("relevancy scan failed")
        if not flags or flags == self.relevancy_flags:
            return
        self.relevancy_flags = flags
        if self.event_bus:
            try:
                self.event_bus.publish("relevancy_flags", flags)
            except Exception:  # pragma: no cover - best effort
                self.logger.exception("relevancy flag publish failed")
        repo = _repo_path()
        for mod, status in flags.items():
            self.logger.info(
                "module flagged", extra=log_record(module=mod, status=status)
            )
            try:
                audit_log_event(
                    "relevancy_flag", {"module": mod, "status": status}
                )
            except Exception:  # pragma: no cover - best effort
                self.logger.exception(
                    "relevancy audit log failed", extra=log_record(module=mod)
                )
            try:
                analyze_redundancy(resolve_module_path(mod))
            except Exception:
                self.logger.exception(
                    "redundancy analysis failed", extra=log_record(module=mod)
                )
            try:
                MutationLogger.log_mutation(
                    change="relevancy",
                    reason=status,
                    trigger="relevancy_radar",
                    performance=0.0,
                    workflow_id=0,
                )
            except Exception:
                self.logger.exception(
                    "mutation logging failed", extra=log_record(module=mod)
                )
        if auto_process:
            try:
                service = ModuleRetirementService(
                    _repo_path()
                )
                results = service.process_flags(flags)
            except Exception:
                self.logger.exception("relevancy flag processing failed")
            else:
                for mod, action in results.items():
                    self.logger.info(
                        "relevancy action",
                        extra=log_record(module=mod, action=action),
                    )
                    if self.event_bus and action != "skipped":
                        try:
                            self.event_bus.publish(
                                f"relevancy:{action}", {"module": mod}
                            )
                        except Exception:
                            self.logger.exception(
                                "relevancy action publish failed",
                                extra=log_record(module=mod, action=action),
                            )
                if self.event_bus:
                    try:
                        self.event_bus.publish("relevancy_actions", results)
                    except Exception:
                        self.logger.exception(
                            "relevancy actions publish failed"
                        )

    def get_relevancy_flags(self) -> dict[str, str]:
        """Return latest module relevancy flags for downstream pipelines."""
        return dict(self.relevancy_flags)

    def _handle_relevancy_flags(
        self,
        flags: dict[str, str],
        *,
        retries: int | None = None,
        delay: float | None = None,
    ) -> None:
        """Process relevancy radar flags and trigger follow-up actions."""
        service = ModuleRetirementService(_repo_path())
        try:
            results = service.process_flags(flags)
        except Exception:
            self.logger.exception("flag processing failed")
            results = {}
        for mod, action in results.items():
            self.logger.info(
                "relevancy action", extra=log_record(module=mod, action=action)
            )
            if self.event_bus and action != "skipped":
                try:
                    self.event_bus.publish(f"relevancy:{action}", {"module": mod})
                except Exception:
                    self.logger.exception(
                        "relevancy action publish failed",
                        extra=log_record(module=mod, action=action),
                    )
        if self.event_bus:
            try:
                self.event_bus.publish("relevancy_actions", results)
            except Exception:
                self.logger.exception("relevancy actions publish failed")
        if append_orphan_classifications:
            repo = _repo_path()
            retire_entries = {
                m: {"classification": "retired"}
                for m, status in flags.items()
                if status == "retire"
            }
            if retire_entries:
                try:
                    append_orphan_classifications(repo, retire_entries)
                except Exception:
                    self.logger.exception("orphan classification failed")
        replace_mods = [m for m, status in flags.items() if status == "replace"]
        try:
            use_synergy = SandboxSettings().use_module_synergy
        except Exception:
            use_synergy = False
        for mod in replace_mods:
            cluster: set[str] = set()
            if use_synergy:
                try:
                    cluster = get_synergy_cluster(mod)
                except Exception:
                    self.logger.exception(
                        "synergy cluster lookup failed",
                        extra=log_record(module=mod),
                    )
            task_id: int | None = None
            if self.self_coding_engine:
                try:
                    context = {"synergy_cluster": list(cluster)} if cluster else None
                    gen_kwargs: dict[str, object] = {}
                    if context is not None:
                        gen_kwargs["context"] = context
                    if retries is not None:
                        gen_kwargs["retries"] = retries
                    if delay is not None:
                        gen_kwargs["delay"] = delay
                    try:
                        strat_name = self.next_prompt_strategy()
                    except Exception:
                        strat_name = None
                    if strat_name:
                        try:
                            gen_kwargs["strategy"] = PromptStrategy(strat_name)
                        except Exception:
                            pass
                    builder = self.self_coding_engine.context_builder
                    try:
                        builder.refresh_db_weights()
                    except Exception:
                        self.logger.exception(
                            "failed to initialise ContextBuilder"
                        )
                        raise
                    gen_kwargs["context_builder"] = builder
                    task_id = self._patch_generator(
                        mod, self.self_coding_engine, **gen_kwargs
                    )
                except RuntimeError as exc:
                    self.logger.error("quick_fix_engine unavailable: %s", exc)
                    raise
                except Exception:
                    self.logger.exception(
                        "replacement generation failed",
                        extra=log_record(module=mod),
                    )
            if self.event_bus:
                event = {"module": mod, "task_id": task_id}
                if cluster:
                    event["synergy_cluster"] = list(cluster)
                try:
                    self.event_bus.publish("relevancy:replace", event)
                except Exception:
                    self.logger.exception(
                        "relevancy replace event publish failed",
                        extra=log_record(module=mod),
                    )
        if self.event_bus:
            try:
                self.event_bus.publish("relevancy:scan", flags)
            except Exception:
                self.logger.exception("relevancy scan event publish failed")

    def _evaluate_workflow_variants(self, seq: str, wf_id: int) -> str:
        """Evolve workflow *seq* using :class:`WorkflowEvolutionManager`."""

        try:
            baseline = self.workflow_evolver.build_callable(seq)
        except Exception:
            self.logger.exception(
                "baseline build failed", extra=log_record(workflow_id=wf_id)
            )
            return "error"

        evolved = self.workflow_evolver.evolve(baseline, wf_id)
        status = "promoted"
        if evolved is baseline:
            status = (
                "stable" if self.workflow_evolver.is_stable(wf_id) else "baseline"
            )

        if self.meta_logger and hasattr(self.meta_logger, "audit"):
            try:
                self.meta_logger.audit.record(
                    {"workflow_id": wf_id, "status": status}
                )
            except Exception:
                self.logger.exception("workflow variant meta logging failed")
        try:
            self._post_round_orphan_scan()
        except Exception:
            self.logger.exception(
                "recursive orphan inclusion failed",
                extra=log_record(workflow_id=wf_id),
            )
        return status

    def _evolve_workflows(self, limit: int = 10) -> dict[int, dict[str, str]]:
        """Evolve stored workflows using :class:`WorkflowEvolutionManager`."""

        workflows: list[tuple[int, str]] = []

        if WorkflowDB is not None and WorkflowRecord is not None:
            try:
                db = WorkflowDB(Path(SandboxSettings().workflows_db))
                for rec in db.fetch_workflows(limit=limit):
                    seq = rec.get("workflow") or []
                    seq_str = "-".join(seq) if isinstance(seq, list) else str(seq)
                    wf_id = int(rec.get("id") or rec.get("wid") or 0)
                    if seq_str:
                        workflows.append((wf_id, seq_str))
            except Exception:  # pragma: no cover - best effort
                self.logger.exception("workflow fetch failed")

        if not workflows and "WorkflowGraph" in globals():
            try:
                graph_obj = WorkflowGraph()
                graph = getattr(graph_obj, "graph", None)
                nodes = []
                if hasattr(graph, "nodes") and callable(getattr(graph, "nodes")):
                    nodes = list(graph.nodes(data=True))  # type: ignore[assignment]
                elif isinstance(graph, dict):
                    nodes = list(graph.get("nodes", {}).items())  # type: ignore[assignment]
                for wid, data in nodes:
                    seq = None
                    if isinstance(data, dict):
                        seq = data.get("sequence") or data.get("workflow")
                    if seq:
                        seq_str = "-".join(seq) if isinstance(seq, list) else str(seq)
                        if seq_str:
                            workflows.append((int(wid), seq_str))
            except Exception:  # pragma: no cover - best effort
                self.logger.exception("workflow graph retrieval failed")

        wf_flags: set[str] = set()
        if getattr(self, "meta_logger", None):
            try:
                flagged = self.meta_logger.diminishing(0.0)
                wf_flags = {f for f in flagged if f.startswith("workflow:")}
                self.meta_logger.flagged_sections.difference_update(wf_flags)
            except Exception:
                logger.exception("Unhandled exception in self_improvement")

        results: dict[int, dict[str, str]] = {}
        for wf_id, seq in workflows:
            wf_key = f"workflow:{wf_id}"
            if wf_key in wf_flags or self.workflow_evolver.is_stable(wf_id):
                continue
            try:
                baseline_callable = self.workflow_evolver.build_callable(seq)
            except Exception:
                self.logger.exception(
                    "baseline build failed", extra=log_record(workflow_id=wf_id)
                )
                continue

            evolved = self.workflow_evolver.evolve(baseline_callable, wf_id)
            status = "promoted"
            if evolved is baseline_callable:
                status = (
                    "stable"
                    if self.workflow_evolver.is_stable(wf_id)
                    else "baseline"
                )
            results[int(wf_id)] = {
                "status": status,
                "parent_id": getattr(evolved, "parent_id", wf_id),
                "mutation_description": getattr(evolved, "mutation_description", ""),
            }
            try:
                self._post_round_orphan_scan()
            except Exception:
                self.logger.exception(
                    "recursive orphan inclusion failed",
                    extra=log_record(workflow_id=wf_id),
                )

        return results

    def _record_snapshot_delta(
        self,
        prompt: object,
        diff: str,
        delta: dict[str, float],
        files: Sequence[Path] | None = None,
        cycle_id: str = "",
    ) -> None:
        """Persist *delta* and log the outcome, updating confidence on success."""

        path = _data_dir() / "snapshots" / "deltas.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(delta) + "\n")

        try:
            import time

            db_path = _data_dir() / "snapshots" / "deltas.db"
            router = DBRouter("snapshot_deltas", str(db_path), str(db_path))
            conn = router.get_connection("deltas")
            conn.execute(
                "CREATE TABLE IF NOT EXISTS deltas (ts REAL, data TEXT)"
            )
            conn.execute(
                "INSERT INTO deltas (ts, data) VALUES (?, ?)",
                (time.time(), json.dumps(delta)),
            )
            conn.commit()
            conn.close()
        except Exception:
            pass

        # Determine whether the snapshot delta represents a regression.  Any
        # negative ROI or sandbox score, entropy increase or failed tests are
        # considered failures and logged with explicit reasons.  When multiple
        # metrics regress we record all of them so downstream analysis can make
        # better decisions about severity.
        failure_reason = None
        failed_metrics: list[str] = []
        tests_passed = delta.get("tests_passed")
        if tests_passed is False or delta.get("tests_failed", 0) > 0:
            failed_metrics.append("tests_failed")
        if delta.get("roi", 0.0) < 0:
            failed_metrics.append("roi")
        if delta.get("sandbox_score", 0.0) < 0:
            failed_metrics.append("sandbox_score")
        if delta.get("entropy", 0.0) > 0:
            failed_metrics.append("entropy")

        if failed_metrics:
            priority = ["tests_failed", "roi", "sandbox_score", "entropy"]
            severe = next((m for m in priority if m in failed_metrics), failed_metrics[0])
            failure_reason = {
                "tests_failed": "tests_failed",
                "roi": "roi_drop",
                "sandbox_score": "score_drop",
                "entropy": "entropy_regression",
            }.get(severe, severe)

        strategy = None
        next_template = None
        if prompt is not None:
            metadata = getattr(prompt, "metadata", {})
            if isinstance(metadata, dict):
                strategy = metadata.get("strategy") or metadata.get("prompt_id")

        success = not failed_metrics
        metrics_payload = dict(delta)
        if not success:
            metrics_payload["failed_metrics"] = failed_metrics
        log_prompt_attempt(
            prompt,
            success=success,
            exec_result={"diff": diff},
            roi_meta=delta,
            failure_reason=failure_reason,
            sandbox_metrics=metrics_payload if not success else None,
        )
        if success:
            for f in files or []:
                try:
                    snapshot_tracker.save_checkpoint(f, cycle_id)
                except Exception:
                    self.logger.exception(
                        "checkpoint save failed", extra=log_record(module=str(f))
                    )
            try:
                current_conf = self.baseline_tracker.get("confidence")
                self.baseline_tracker.update(confidence=current_conf + 1)
            except Exception:
                pass
            updater = getattr(self, "confidence_updater", None)
            if updater:
                try:
                    updater.update(delta)
                except Exception:
                    self.logger.exception("confidence update failed")
        else:
            if strategy:
                try:
                    snapshot_tracker.record_downgrade(str(strategy))
                except Exception:
                    pass
                try:
                    next_template = self.strategy_manager.record_failure(
                        str(strategy),
                        failure_reason,
                        float(delta.get("roi", 0.0)),
                    )
                except Exception:
                    self.logger.exception("strategy rotation failed")
                    next_template = None
                self.pending_strategy = next_template

        if success and strategy and hasattr(self, "strategy_manager"):
            try:
                self.strategy_manager.update(
                    str(strategy), float(delta.get("roi", 0.0)), True
                )
            except Exception:
                self.logger.exception("strategy metrics update failed")

    def next_prompt_strategy(self) -> str | None:
        """Return the next prompt strategy based on historical performance."""

        if hasattr(self, "strategy_analytics"):
            try:
                self.strategy_analytics.refresh_if_stale()
            except Exception:
                self.logger.exception("strategy analytics refresh failed")

        pending = getattr(self, "pending_strategy", None)
        if pending and pending in self.strategy_manager.strategies:
            self.pending_strategy = None
            return pending
        try:
            return self.strategy_manager.next()
        except Exception:
            return None

    @radar.track
    def run_cycle(self, energy: int = 1, *, target_region: "TargetRegion | None" = None) -> AutomationResult:
        """Execute a self-improvement cycle.

        The ``workflow_id`` used for foresight tracking is derived from the
        current sandbox workflow context.
        """
        self._cycle_running = True
        self._cycle_count += 1
        cid = f"cycle-{self._cycle_count}"
        set_correlation_id(cid)
        self._cycle_target_region = target_region
        try:
            repo_path = Path(_repo_path())
            changed_files: list[Path] = []
            snapshot_prompt: object | None = None
            snapshot_diff = ""
            if self._snapshot_tracker:
                all_files = list(repo_path.rglob("*.py"))
                ctx = {
                    "files": all_files,
                    "roi": self.baseline_tracker.current("roi"),
                    "sandbox_score": self.baseline_tracker.current("score"),
                    "prompt": None,
                    "diff": "",
                }
                self._snapshot_tracker.capture("before", ctx, repo_path=repo_path)

            momentum = self.baseline_tracker.momentum
            if self.policy:
                try:
                    self.policy.adjust_for_momentum(momentum)
                except Exception:
                    self.logger.exception("policy momentum adjustment failed")
            workflow_evolution_details: list[dict[str, object]] = []
            evo_allowed = self._should_trigger()
            trigger = evo_allowed
            if not evo_allowed:
                entropy_delta = 0.0
                entropy_anomaly = False
                try:
                    entropy_delta = getattr(self.baseline_tracker, "entropy_delta", 0.0)
                    entropy_anomaly = entropy_delta < 0
                    if not entropy_anomaly and getattr(self, "meta_logger", None):
                        deltas = getattr(self.meta_logger, "module_entropy_deltas", {})
                        for vals in deltas.values():
                            if vals and vals[-1] < 0:
                                entropy_delta = float(vals[-1])
                                entropy_anomaly = True
                                break
                except Exception:
                    entropy_anomaly = False
                error_traces: list[Any] = []
                try:
                    eb = getattr(self, "error_bot", None)
                    if eb:
                        if hasattr(eb, "recent_errors"):
                            error_traces = eb.recent_errors(limit=5)
                        elif hasattr(eb, "recent_events"):
                            error_traces = eb.recent_events(limit=5)
                        elif hasattr(getattr(eb, "db", None), "recent_errors"):
                            error_traces = eb.db.recent_errors(limit=5)  # type: ignore[attr-defined]
                        elif hasattr(getattr(eb, "db", None), "recent_events"):
                            error_traces = eb.db.recent_events(limit=5)  # type: ignore[attr-defined]
                except Exception:
                    error_traces = []
                if entropy_anomaly or error_traces:
                    evo_allowed = True
                    trigger = True
                    self.logger.debug(
                        "fallback trigger activated",
                        extra=log_record(entropy_delta=entropy_delta, error_traces=len(error_traces)),
                    )
            if not evo_allowed:
                info = getattr(self, "_last_skip_info", {})
                self.logger.debug(
                    "self optimisation skipped",
                    extra=log_record(
                        reason=str(info.get("reason", "unknown")),
                        metrics=info.get("metrics"),
                    ),
                )
            planner_chains: list[list[str]] = []
            meta_records: list[dict[str, Any]] = []
            if self._cycle_count % PLANNER_INTERVAL == 0:
                planner_chains = self._plan_cross_domain_chains()
                meta_records = self._discover_meta_workflows()
                if consume_planner_suggestions and planner_chains:
                    try:
                        consume_planner_suggestions(planner_chains)
                    except Exception:
                        self.logger.exception("planner suggestion handling failed")

            def _handle_new_modules() -> None:
                repo = _repo_path()
                pending = self._update_orphan_modules() or []
                while pending:
                    added_modules: set[str] = set()
                    try:
                        _tracker, tested = environment.auto_include_modules(
                            sorted(pending),
                            recursive=True,
                            validate=True,
                            context_builder=self.self_coding_engine.context_builder,
                        )
                        added_modules.update(tested.get("added", []))
                    except Exception as exc:  # pragma: no cover - best effort
                        self.logger.exception("auto inclusion failed: %s", exc)
                    record_new = getattr(self, "_record_new_modules", None)
                    if record_new:
                        record_new(added_modules)
                    abs_paths = [str(repo / p) for p in pending]
                    try:
                        self._integrate_orphans(abs_paths)
                    except Exception as exc:  # pragma: no cover - best effort
                        self.logger.exception("orphan integration failed: %s", exc)
                    try:
                        pending = self._update_orphan_modules() or []
                    except Exception as exc:  # pragma: no cover - best effort
                        self.logger.exception(
                            "successive orphan update failed: %s", exc
                        )
                        break

                try:
                    repo = resolve_path(".")
                    self._sandbox_integrate(
                        repo,
                        router=GLOBAL_ROUTER,
                        context_builder=self.self_coding_engine.context_builder,
                    )
                except Exception as exc:  # pragma: no cover - best effort
                    self.logger.exception(
                        "recursive orphan inclusion failed: %s", exc
                    )

            if self.meta_logger:
                try:
                    settings = SandboxSettings()
                    thr = settings.entropy_ceiling_threshold
                    consecutive = settings.entropy_ceiling_consecutive or 3
                except Exception:
                    thr = None
                    consecutive = 3
                if thr is not None:
                    try:
                        flagged = self.meta_logger.ceiling(thr, consecutive=consecutive)
                    except Exception:
                        flagged = []
                    if flagged:
                        norm = [Path(m).as_posix() for m in flagged]
                        self.entropy_ceiling_modules.update(norm)
                        for m in norm:
                            self.logger.info(
                                "module flagged", extra=log_record(module=m, status="entropy_ceiling")
                            )
                            try:
                                audit_log_event(
                                    "entropy_ceiling",
                                    {"module": m, "status": "entropy_ceiling"},
                                )
                            except Exception:  # pragma: no cover - best effort
                                self.logger.exception(
                                    "entropy ceiling audit log failed",
                                    extra=log_record(module=m),
                                )
                        try:
                            service = ModuleRetirementService(
                                _repo_path()
                            )
                            pending = {m: "retire" for m in flagged}
                            results = service.process_flags(pending)
                            remaining = [
                                m for m, action in results.items() if action == "skipped"
                            ]
                            if remaining:
                                pending = {m: "compress" for m in remaining}
                                results.update(service.process_flags(pending))
                                remaining = [
                                    m
                                    for m in remaining
                                    if results.get(m) == "skipped"
                                ]
                            if remaining:
                                service.process_flags(
                                    {m: "replace" for m in remaining}
                                )
                        except Exception:
                            self.logger.exception(
                                "ceiling flag processing failed",
                                extra=log_record(modules=flagged),
                            )
            try:
                self.retest_redundant_modules()
            except Exception as exc:  # pragma: no cover - best effort
                self.logger.exception("redundant module check failed: %s", exc)
            # refresh orphan data so new modules are considered before policy evaluation
            self._update_orphan_modules()
            orphans = self._load_orphan_candidates()
            if orphans:
                passing = self._test_orphan_modules(orphans)
                if passing:
                    added_modules: set[str] = set()
                    try:
                        _tracker, tested = environment.auto_include_modules(
                            sorted(passing),
                            recursive=True,
                            validate=True,
                            context_builder=self.self_coding_engine.context_builder,
                        )
                        added_modules.update(tested.get("added", []))
                    except Exception as exc:  # pragma: no cover - best effort
                        self.logger.exception("auto inclusion failed: %s", exc)
                    record_new = getattr(self, "_record_new_modules", None)
                    if record_new:
                        record_new(added_modules)
                    repo = _repo_path()
                    abs_paths = [str(repo / p) for p in passing]
                    integrated: set[str] = set()
                    try:
                        integrated = self._integrate_orphans(abs_paths)
                    except Exception as exc:  # pragma: no cover - best effort
                        self.logger.exception("orphan integration failed: %s", exc)
            try:
                self._update_orphan_modules()
            except Exception as exc:  # pragma: no cover - best effort
                self.logger.exception(
                    "post integration orphan update failed: %s", exc
                )
            self._refresh_module_map()
            now = time.time()
            if now - self._last_relevancy_eval >= self.relevancy_eval_interval:
                self._evaluate_module_relevance()
                self._last_relevancy_eval = now
            self._process_preventative_queue()
            if self.error_bot:
                try:
                    predictions = (
                        self.error_bot.predict_errors()
                        if hasattr(self.error_bot, "predict_errors")
                        else []
                    )
                    clusters = (
                        self.error_bot.get_error_clusters()
                        if hasattr(self.error_bot, "get_error_clusters")
                        else {}
                    )
                    baseline = {
                        item.get("error_type", ""): float(item.get("count", 0.0))
                        for item in (
                            self.error_bot.summarize_telemetry(limit=10)
                            if hasattr(self.error_bot, "summarize_telemetry")
                            else []
                        )
                    }
                    affected_modules: set[str] = set()
                    pred_clusters: set[int] = set()
                    for err in predictions:
                        cid = clusters.get(err)
                        if cid is None:
                            continue
                        pred_clusters.add(cid)
                        affected_modules.update(
                            m for m, idx in self.module_clusters.items() if idx == cid
                        )
                    if pred_clusters and affected_modules:
                        self.logger.info(
                            "predicted error clusters",
                            extra=log_record(
                                predicted_clusters=sorted(pred_clusters),
                                modules=sorted(affected_modules),
                            ),
                        )
                        self.error_bot.auto_patch_recurrent_errors()
                        _handle_new_modules()
                        if self.error_predictor:
                            try:
                                self.error_predictor.graph.update_error_stats(
                                    self.error_bot.db
                                )
                            except Exception:
                                self.logger.exception(
                                    "knowledge graph update failed",
                                    extra=log_record(action="auto_patch_recurrent"),
                                )
                        after = {
                            item.get("error_type", ""): float(item.get("count", 0.0))
                            for item in (
                                self.error_bot.summarize_telemetry(limit=10)
                                if hasattr(self.error_bot, "summarize_telemetry")
                                else []
                            )
                        }
                        prevented = [
                            err
                            for err in predictions
                            if baseline.get(err, 0.0) > 0
                            and baseline.get(err) == after.get(err)
                        ]
                        if prevented:
                            self.logger.info(
                                "proactive patch prevented faults",
                                extra=log_record(errors=prevented),
                            )
                except Exception as exc:
                    self.logger.exception(
                        "proactive prediction patching failed: %s", exc
                    )
            self._apply_high_risk_patches()
            _handle_new_modules()
            state = self._policy_state() if self.policy else (0,) * POLICY_STATE_LEN
            predicted = self.policy.score(state) if self.policy else 0.0
            roi_pred: float | None = None
            self.logger.info(
                "cycle start",
                extra=log_record(energy=energy, predicted_roi=predicted, state=state),
            )
            if self.policy:
                self.logger.info(
                    "policy predicted roi",
                    extra=log_record(
                        predicted_roi=predicted,
                        state=state,
                        weights=self.synergy_learner.weights,
                    ),
                )
            before_roi = 0.0
            if self.capital_bot:
                try:
                    before_roi = self.capital_bot.profit()
                    self.logger.info("initial ROI", extra=log_record(roi=before_roi))
                except Exception as exc:
                    self.logger.exception("profit lookup failed: %s", exc)
                    before_roi = 0.0
            if before_roi == 0.0 and self.data_bot:
                try:
                    before_roi = self.data_bot.roi(self.bot_name)
                    self.logger.info(
                        "initial ROI",
                        extra=log_record(roi=before_roi),
                    )
                except Exception as exc:
                    self.logger.exception("ROI lookup failed: %s", exc)
            if self.capital_bot:
                try:
                    energy = int(
                        round(
                            self.capital_bot.energy_score(
                                load=0.0,
                                success_rate=1.0,
                                deploy_eff=1.0,
                                failure_rate=0.0,
                                reward=None,
                            )
                            * 5
                        )
                    )
                    self.logger.info("available energy", extra=log_record(value=energy))
                except Exception as exc:
                    self.logger.exception("energy calculation failed: %s", exc)
                    energy = 1
            if self.policy:
                try:
                    energy = max(1, int(round(energy * (1 + max(0.0, predicted)))))
                    self.logger.info(
                        "policy adjusted energy",
                        extra=log_record(
                            value=energy,
                            predicted_roi=predicted,
                            state=state,
                            weights=self.synergy_learner.weights,
                        ),
                    )
                except Exception as exc:
                    self.logger.exception("policy energy adjustment failed: %s", exc)
            if self.pre_roi_bot:
                try:
                    forecast = self.pre_roi_bot.predict_model_roi(self.bot_name, [])
                    roi_pred = float(getattr(forecast, "roi", 0.0))
                    scale = (
                        1 + max(0.0, roi_pred + self.pre_roi_bias) * self.pre_roi_scale
                    )
                    if self.pre_roi_cap:
                        scale = min(scale, self.pre_roi_cap)
                    energy = max(1, int(round(energy * scale)))
                    self.logger.info(
                        "pre_roi adjusted energy",
                        extra=log_record(value=energy, roi_prediction=roi_pred),
                    )
                except Exception as exc:
                    self.logger.exception("pre ROI energy adjustment failed: %s", exc)
            tracker = getattr(self, "tracker", None)
            if tracker is not None:
                try:
                    syn_adj = self._weighted_synergy_adjustment()
                    self.logger.info(
                        "synergy adjustment",
                        extra=log_record(
                            factor=syn_adj,
                            energy_before=energy,
                            weights=self.synergy_learner.weights,
                        ),
                    )
                    if syn_adj:
                        energy = int(round(energy * (1.0 + syn_adj)))
                        self.logger.info(
                            "synergy adjusted energy",
                            extra=log_record(
                                value=energy, weights=self.synergy_learner.weights
                            ),
                        )
                except Exception as exc:  # pragma: no cover - best effort
                    self.logger.exception("synergy energy adjustment failed: %s", exc)
            try:
                roi_scale = 1.0 + max(0.0, self.roi_delta_ema)
                self.logger.info(
                    "roi ema adjustment",
                    extra=log_record(factor=roi_scale, energy_before=energy),
                )
                energy = int(round(energy * roi_scale))
                self.logger.info(
                    "roi ema adjusted energy", extra=log_record(value=energy)
                )
            except Exception as exc:
                self.logger.exception("roi energy adjustment failed: %s", exc)
            energy = max(1, min(int(energy), 100))
            model_id = bootstrap(
                context_builder=self.aggregator.context_builder
            )
            self.logger.info("model bootstrapped", extra=log_record(model_id=model_id))
            self.info_db.set_current_model(model_id)
            self._record_state()
            if self.learning_engine:
                try:
                    self.logger.info("training learning engine")
                    self.learning_engine.train()
                    self._evaluate_learning()
                except Exception as exc:
                    self.logger.exception("learning engine run failed: %s", exc)
            self.logger.info(
                "pipeline pre-run metrics",
                extra=log_record(
                    predicted_roi=roi_pred if roi_pred is not None else predicted,
                    policy_score=predicted,
                    energy=energy,
                    synergy_weights=self.synergy_learner.weights,
                ),
            )
            self.logger.info(
                "running automation pipeline", extra=log_record(energy=energy)
            )
            result = self.pipeline.run(self.bot_name, energy=energy)
            actions = getattr(getattr(result, "package", None), "actions", None)
            self.logger.info(
                "selected actions",
                extra=log_record(actions=actions, growth_type=self._last_growth_type),
            )
            self.logger.info(
                "pipeline complete",
                extra=log_record(roi=getattr(result.roi, "roi", 0.0)),
            )
            trending_topic = getattr(result, "trending_topic", None)
            patch_id = None
            reverted = False
            if self.self_coding_engine and result.package:
                try:
                    self.logger.info(
                        "applying helper patch",
                        extra=log_record(trending_topic=trending_topic),
                    )
                    mod_path = resolve_path("auto_helpers.py")
                    start_patch = time.perf_counter()
                    patch_id, reverted, delta = self.self_coding_engine.apply_patch(
                        mod_path,
                        "helper",
                        trending_topic=trending_topic,
                        parent_patch_id=self._last_patch_id,
                        reason="helper_patch",
                        trigger="automation_cycle",
                    )
                    before_metric = 0.0
                    after_metric = delta
                    patch_diff = ""
                    if self.self_coding_engine.patch_db and patch_id is not None:
                        try:
                            with self.self_coding_engine.patch_db._connect() as conn:
                                row = conn.execute(
                                    "SELECT roi_before, roi_after, diff FROM patch_history WHERE id=?",
                                    (patch_id,),
                                ).fetchone()
                            if row:
                                before_metric = float(row[0])
                                after_metric = float(row[1])
                                patch_diff = row[2] or ""
                        except Exception:
                            after_metric = before_metric + delta
                    else:
                        after_metric = before_metric + delta
                    snapshot_prompt = getattr(self.self_coding_engine, "_last_prompt", None)
                    snapshot_diff = patch_diff
                    changed_files.append(mod_path)
                    with MutationLogger.log_context(
                        change=f"helper_patch_{patch_id}",
                        reason="self-improvement helper patch",
                        trigger="automation_cycle",
                        workflow_id=0,
                        before_metric=before_metric,
                        parent_id=self._last_mutation_id,
                    ) as mutation:
                        mutation["after_metric"] = after_metric
                        mutation["performance"] = delta
                        mutation["roi"] = after_metric
                    self._last_mutation_id = int(mutation["event_id"])
                    self._last_patch_id = patch_id
                    if patch_id is not None and not reverted:
                        self._alignment_review_last_commit(f"helper_patch_{patch_id}")
                        self._flag_patch_alignment(
                            patch_id,
                            {"trigger": "automation_cycle", "patch_id": patch_id},
                        )
                    roi_delta = after_metric - before_metric
                    tracker = getattr(self, "roi_tracker", None)
                    if tracker and hasattr(tracker, "update"):
                        try:
                            token_metrics: dict[str, float] = {}
                            file_path = _repo_path / mod_path
                            stats, *_rest = _si_metrics._collect_metrics([file_path], _repo_path)
                            info = stats.get(mod_path.as_posix())
                            if info:
                                token_metrics = {
                                    "token_entropy": float(info.get("token_entropy", 0.0)),
                                    "token_diversity": float(info.get("token_diversity", 0.0)),
                                }
                            tracker.update(
                                before_metric,
                                after_metric,
                                modules=[mod_path.as_posix()],
                                metrics=token_metrics or None,
                            )
                        except Exception:
                            self.logger.exception("roi tracker update failed")
                    if self.metrics_db:
                        try:
                            elapsed = time.perf_counter() - start_patch
                            self.metrics_db.record(
                                mod_path.as_posix(),
                                elapsed,
                                roi_delta=roi_delta,
                            )
                        except Exception:
                            self.logger.exception("relevancy metrics record failed")
                    if self.policy:
                        try:
                            self.logger.info(
                                "patch applied",
                                extra=log_record(
                                    patch_id=patch_id,
                                    reverted=reverted,
                                    delta=delta,
                                ),
                            )
                            st = self._policy_state()
                            syn_reward = st[-2] / 10.0 + st[-1] / 10.0
                            self.policy.update(
                                st,
                                delta + syn_reward,
                                synergy_roi_delta=st[-4] / 10.0,
                                synergy_efficiency_delta=st[-3] / 10.0,
                            )
                            if getattr(self.policy, "path", None):
                                try:
                                    self.policy.save()
                                except (
                                    Exception
                                ) as exc:  # pragma: no cover - best effort
                                    self.logger.exception("policy save failed: %s", exc)
                        except Exception as exc:
                            self.logger.exception("policy patch update failed: %s", exc)
                    if self.optimize_self_flag:
                        self._optimize_self()
                except Exception as exc:
                    self.logger.exception("helper patch failed: %s", exc)
                    patch_id = None
                    reverted = False
            _handle_new_modules()
            if self.error_bot:
                try:
                    self.error_bot.auto_patch_recurrent_errors()
                    if self.error_predictor:
                        try:
                            self.error_predictor.graph.update_error_stats(
                                self.error_bot.db
                            )
                        except Exception:
                            self.logger.exception(
                                "knowledge graph update failed",
                                extra=log_record(action="auto_patch_recurrent"),
                            )
                    self.logger.info("error auto-patching complete")
                except Exception as exc:
                    self.logger.exception("auto patch recurrent errors failed: %s", exc)
            _handle_new_modules()
            after_roi = before_roi
            if self.capital_bot:
                try:
                    after_roi = self.capital_bot.profit()
                    self.logger.info(
                        "post-cycle ROI",
                        extra=log_record(before=before_roi, after=after_roi),
                    )
                except Exception as exc:
                    self.logger.exception("post-cycle profit lookup failed: %s", exc)
                    after_roi = before_roi
            roi_value = result.roi.roi if result.roi else 0.0
            roi_realish = roi_value
            pred_realish = predicted
            try:
                features = np.array([[float(roi_value)]], dtype=np.float64)
                drift = self.truth_adapter.check_drift(features)
                preds, low_conf = self.truth_adapter.predict(features)
                roi_realish = float(preds[0])
                if predicted is not None:
                    p_arr = np.array([[float(predicted)]], dtype=np.float64)
                    p_preds, _ = self.truth_adapter.predict(p_arr)
                    pred_realish = float(p_preds[0])
                if drift or low_conf:
                    self.logger.warning(
                        "truth adapter low confidence; scheduling retrain"
                    )
                    self._truth_adapter_needs_retrain = True
            except Exception:
                self.logger.exception("truth adapter predict failed")
            self.logger.info(
                "roi realish", extra=log_record(roi_realish=roi_realish)
            )
            if self.roi_tracker and predicted is not None:
                try:
                    self.roi_tracker.record_roi_prediction(
                        [float(pred_realish)],
                        [float(roi_realish)],
                        predicted_class=self._last_growth_type,
                        workflow_id="self_improvement",
                    )
                except Exception:
                    self.logger.exception("roi tracker record failed")
                self.logger.info(
                    "cycle roi", extra=log_record(predicted=pred_realish, actual=roi_realish)
                )
            if self.roi_tracker:
                try:
                    cards = self.roi_tracker.generate_scorecards()
                except Exception:
                    cards = []
                scorecard = {
                    "decision": "rollback" if reverted else "ship",
                    "raroi_increase": sum(
                        1 for c in cards if getattr(c, "raroi_delta", 0.0) > 0
                    ),
                    "raroi": getattr(self.roi_tracker, "last_raroi", None),
                    "confidence": (
                        self.roi_tracker.confidence_history[-1]
                        if getattr(self.roi_tracker, "confidence_history", [])
                        else None
                    ),
                }
                workflow_id = "self_improvement"
                gov_result: Dict[str, Any] | None = None
                forecast_info: Dict[str, Any] | None = None
                reasons: List[str] = []
                try:
                    wf_ctx = getattr(environment, "current_context", None)
                    try:
                        ctx_obj = wf_ctx() if callable(wf_ctx) else wf_ctx
                        workflow_id = getattr(ctx_obj, "workflow_id", workflow_id)
                    except Exception:
                        logger.exception("Unhandled exception in self_improvement")
                    metrics = {
                        "raroi": scorecard.get("raroi"),
                        "confidence": scorecard.get("confidence"),
                        "sandbox_roi": roi_realish,
                        "adapter_roi": pred_realish,
                    }
                    try:
                        settings = SandboxSettings()
                        agent = HumanAlignmentAgent(settings=settings)
                        logs: list[dict[str, Any]] | None = None
                        try:
                            logs = get_recent_events(limit=20)
                        except Exception as exc:
                            logger.warning("Failed to get recent events: %s", exc)
                        warnings = agent.evaluate_changes(actions, metrics, logs, None)
                        if any(warnings.values()):
                            scorecard["alignment"] = {
                                "status": "fail",
                                "rationale": warnings,
                            }
                        else:
                            scorecard["alignment"] = {
                                "status": "pass",
                                "rationale": "no_warnings",
                            }
                    except Exception:
                        self.logger.exception("alignment evaluation failed")
                        scorecard["alignment"] = {
                            "status": "error",
                            "rationale": "alignment_evaluation_error",
                        }
                    gov_result = deployment_evaluate(
                        scorecard,
                        metrics,
                        patch=str(patch_id) if patch_id is not None else None,
                        foresight_tracker=self.foresight_tracker,
                        workflow_id=workflow_id,
                        borderline_bucket=self.borderline_bucket,
                    )
                except Exception:
                    self.logger.exception("deployment evaluation failed")
                if gov_result:
                    verdict = str(gov_result.get("verdict"))
                    reasons = list(gov_result.get("reasons", []))
                    forecast_info = gov_result.get("foresight")
                    risk: dict[str, object] | None = None
                    try:
                        if self.foresight_tracker and workflow_id:
                            risk = self.foresight_tracker.predict_roi_collapse(workflow_id)
                    except Exception:
                        self.logger.exception("foresight risk check failed")
                    risk_cls = risk.get("risk") if isinstance(risk, Mapping) else None
                    brittle = bool(risk.get("brittle")) if isinstance(risk, Mapping) else False
                    high = bool(risk and (risk_cls != "Stable" or brittle))
                    self.workflow_risk = risk
                    self.workflow_high_risk = high
                    try:
                        audit_log_event(
                            "foresight_risk",
                            {
                                "workflow_id": workflow_id,
                                "risk": risk_cls,
                                "brittle": brittle,
                            },
                        )
                    except Exception:
                        self.logger.exception("risk audit log failed")
                    try:
                        self.logger.info(
                            "foresight risk classification",
                            extra=log_record(
                                workflow_id=workflow_id,
                                risk=risk_cls,
                                brittle=brittle,
                            ),
                        )
                    except Exception:
                        self.logger.exception("risk classification logging failed")
                    if high:
                        verdict = "no_go"
                        if "roi_collapse_risk" not in reasons:
                            reasons.append("roi_collapse_risk")
                        try:
                            self.enqueue_preventative_fixes([workflow_id])
                        except Exception:
                            self.logger.exception("risk queue enqueue failed")
                    if verdict == "promote" and self.foresight_tracker:
                        logger_obj: ForecastLogger | None = None
                        forecast_info: Dict[str, Any] | None = None
                        try:
                            forecaster = UpgradeForecaster(self.foresight_tracker)
                            graph = WorkflowGraph()
                            logger_obj = ForecastLogger(resolve_path("forecast_records/foresight.log"))
                            decision = is_foresight_safe_to_promote(
                                workflow_id,
                                str(patch_id) if patch_id is not None else "",
                                forecaster,
                                graph,
                            )
                            if not isinstance(decision, ForesightDecision):
                                decision = ForesightDecision(*decision)
                            forecast_info = decision.forecast
                            decision_label = (
                                decision.recommendation
                                if not decision.safe
                                else "promote"
                            )
                            log_forecast_record(
                                logger_obj,
                                workflow_id,
                                forecast_info,
                                decision_label,
                                decision.reasons,
                            )
                            if not decision.safe:
                                verdict = decision.recommendation
                                reasons.extend(decision.reasons)
                        except Exception:
                            self.logger.exception("foresight gate check failed")
                        finally:
                            try:
                                if logger_obj is not None:
                                    logger_obj.close()
                            except Exception:
                                logger.exception("Unhandled exception in self_improvement")
                    scorecard["forecast"] = forecast_info
                    scorecard["reasons"] = list(reasons)
                    try:
                        payload = {
                            "verdict": verdict,
                            "reasons": reasons,
                            "forecast": forecast_info,
                        }
                        if verdict in {"borderline", "pilot"}:
                            payload["downgrade_type"] = verdict
                        audit_log_event("deployment_verdict", payload)
                    except Exception:
                        self.logger.exception("audit log failed")
                    if self.event_bus:
                        try:
                            self.event_bus.publish(
                                "deployment_verdict",
                                {
                                    "verdict": verdict,
                                    "reasons": reasons,
                                    "forecast": forecast_info,
                                },
                            )
                        except Exception:
                            self.logger.exception("event bus publish failed")
                    try:
                        self.logger.info(
                            "deployment verdict",
                            extra=log_record(
                                verdict=verdict,
                                reasons=";".join(reasons),
                                forecast=forecast_info,
                            ),
                        )
                    except Exception:
                        self.logger.exception("deployment verdict logging failed")
                    scorecard["deployment_verdict"] = verdict
                    if verdict == "promote":
                        self.workflow_ready = True
                    elif verdict == "demote":
                        self.workflow_ready = False
                        if (
                            patch_id is not None
                            and self.self_coding_engine
                            and not reverted
                        ):
                            try:
                                self.self_coding_engine.rollback_patch(str(patch_id))
                                reverted = True
                                scorecard["decision"] = "rollback"
                            except Exception:
                                self.logger.exception("patch rollback failed")
                    elif verdict in {"micro_pilot", "borderline", "pilot"}:
                        try:
                            self.borderline_bucket.add_candidate(
                                self.bot_name,
                                scorecard.get("raroi"),
                                scorecard.get("confidence"),
                                ";".join(reasons),
                            )
                            settings = SandboxSettings()
                            if getattr(settings, "micropilot_mode", "") == "auto":
                                try:
                                    evaluator = getattr(
                                        self, "micro_pilot_evaluator", None
                                    )
                                    self.borderline_bucket.process(
                                    evaluator,
                                    raroi_threshold=self._raroi_threshold(),
                                    confidence_threshold=getattr(
                                        self.roi_tracker,
                                        "confidence_threshold",
                                        0.0,
                                    ),
                                    )
                                except Exception:
                                    logger.exception("Unhandled exception in self_improvement")
                        except Exception:
                            self.logger.exception("borderline enqueue failed")
                        self.workflow_ready = False
                    else:
                        self.workflow_ready = False
                vetoes: List[str] = []
                try:
                    vetoes = check_veto(scorecard, load_rules())
                except Exception:
                    self.logger.exception("governance check failed")
                try:
                    append_governance_result(scorecard, vetoes, forecast_info, reasons)
                except Exception:
                    self.logger.exception("governance logging failed")
                if vetoes and patch_id is not None and self.self_coding_engine and not reverted:
                    try:
                        self.self_coding_engine.rollback_patch(str(patch_id))
                        reverted = True
                        self.logger.warning(
                            "patch rolled back due to governance veto",
                            extra=log_record(patch_id=patch_id, veto=";".join(vetoes)),
                        )
                    except Exception:
                        self.logger.exception("patch rollback failed")
            if self.evolution_history:
                try:
                    from menace_sandbox.evolution_history_db import EvolutionEvent

                    event_id = self.evolution_history.add(
                        EvolutionEvent(
                            action="self_improvement",
                            before_metric=before_roi,
                            after_metric=after_roi,
                            roi=roi_realish,
                            predicted_roi=pred_realish,
                            trending_topic=trending_topic,
                            reason="self improvement cycle",
                            trigger="run_cycle",
                            performance=after_roi - before_roi,
                            parent_event_id=self._last_mutation_id,
                        )
                    )
                    self._last_mutation_id = event_id
                except Exception as exc:
                    self.logger.exception("evolution history logging failed: %s", exc)
            eff = bottleneck = patch_rate = trend = anomaly = 0.0
            if self.data_bot:
                try:
                    df = self.data_bot.db.fetch(20)
                    if hasattr(df, "empty"):
                        if not getattr(df, "empty", True):
                            eff = float(max(0.0, 100.0 - df["cpu"].mean()))
                            if "errors" in df.columns:
                                bottleneck = float(df["errors"].mean())
                    elif isinstance(df, list) and df:
                        avg_cpu = sum(r.get("cpu", 0.0) for r in df) / len(df)
                        eff = float(max(0.0, 100.0 - avg_cpu))
                        bottleneck = float(
                            sum(r.get("errors", 0.0) for r in df) / len(df)
                        )
                except Exception as exc:
                    self.logger.exception("data fetch failed: %s", exc)
                    eff = bottleneck = 0.0
                if self.self_coding_engine and getattr(
                    self.self_coding_engine, "patch_db", None
                ):
                    try:
                        patch_rate = self.self_coding_engine.patch_db.success_rate()
                    except Exception as exc:
                        self.logger.exception(
                            "self_coding patch rate lookup failed: %s", exc
                        )
                        patch_rate = 0.0
                if getattr(self.data_bot, "patch_db", None) and not patch_rate:
                    try:
                        patch_rate = self.data_bot.patch_db.success_rate()
                    except Exception as exc:
                        self.logger.exception(
                            "data_bot patch rate lookup failed: %s", exc
                        )
                        patch_rate = 0.0
                try:
                    trend = self.data_bot.long_term_roi_trend(limit=200)
                except Exception as exc:
                    self.logger.exception("trend retrieval failed: %s", exc)
                    trend = 0.0
                try:
                    df_anom = self.data_bot.db.fetch(100)
                    if hasattr(df_anom, "empty"):
                        if not getattr(df_anom, "empty", True):
                            df_anom["roi"] = df_anom["revenue"] - df_anom["expense"]
                            anomaly = float(
                                len(DataBot.detect_anomalies(df_anom, "roi"))
                            ) / len(df_anom)
                    elif isinstance(df_anom, list) and df_anom:
                        rois = [
                            float(r.get("revenue", 0.0) - r.get("expense", 0.0))
                            for r in df_anom
                        ]
                        df_list = [{"roi": r} for r in rois]
                        anomaly = float(
                            len(DataBot.detect_anomalies(df_list, "roi"))
                        ) / len(rois)
                except Exception as exc:
                    self.logger.exception("anomaly calculation failed: %s", exc)
                    anomaly = 0.0
                try:
                    self.data_bot.log_evolution_cycle(
                        "self_improvement",
                        before_roi,
                        after_roi,
                          roi_realish,
                        0.0,
                        patch_success=patch_rate,
                        roi_delta=after_roi - before_roi,
                        roi_trend=trend,
                        anomaly_count=anomaly,
                        efficiency=eff,
                        bottleneck=bottleneck,
                        patch_id=patch_id,
                        trending_topic=trending_topic,
                        reverted=reverted,
                        reason="cycle complete",
                        trigger="run_cycle",
                        parent_event_id=self._last_mutation_id,
                    )
                    scenario_metrics: dict[str, float] = {}
                    tracker = getattr(self, "tracker", None)
                    if tracker is not None:
                        db = getattr(self.data_bot, "db", None)
                        for name in (
                            "latency_error_rate",
                            "hostile_failures",
                            "misuse_failures",
                            "concurrency_throughput",
                        ):
                            vals = tracker.metrics_history.get(name)
                            if vals:
                                val = float(vals[-1])
                                scenario_metrics[name] = val
                                if db is not None:
                                    try:
                                        db.log_eval("self_improvement", name, val)
                                    except Exception:
                                        logger.exception("Unhandled exception in self_improvement")
                    if scenario_metrics:
                        frac = self._evaluate_scenario_metrics(scenario_metrics)
                        try:
                            self.baseline_tracker.update(pass_rate=frac, **scenario_metrics)
                        except Exception:
                            self.logger.exception("baseline tracker update failed")
                        self.logger.info(
                            "scenario metrics",
                            extra=log_record(**scenario_metrics),
                        )
                    baseline_roi = self.baseline_tracker.get("roi")
                    self.logger.info(
                        "cycle metrics",
                        extra=log_record(
                            patch_success=patch_rate,
                            roi_delta=after_roi - baseline_roi,
                            roi_trend=trend,
                            anomaly=anomaly,
                        ),
                    )
                    if self.capital_bot:
                        try:
                            self.capital_bot.log_evolution_event(
                                "self_improvement",
                                before_roi,
                                after_roi,
                            )
                        except Exception as exc:
                            self.logger.exception(
                                "capital bot evolution log failed: %s", exc
                            )
                except Exception as exc:
                    self.logger.exception("data_bot evolution logging failed: %s", exc)
            self.last_run = time.time()
            if self._alignment_agent is None:
                try:
                    self._alignment_agent = AlignmentReviewAgent()
                    self._alignment_agent.start()
                except Exception:
                    self.logger.exception("alignment review agent failed to start")
            settings = SandboxSettings()
            delta = after_roi - baseline_roi
            self._check_chain_stagnation()
            warnings: dict[str, list[dict[str, Any]]] = {}
            if delta > 0:
                try:
                    metrics = result.roi.__dict__ if result.roi else None
                    agent = HumanAlignmentAgent(settings=settings)
                    logs: list[dict[str, Any]] | None = getattr(result, "logs", None)
                    if logs is None:
                        try:
                            logs = get_recent_events(limit=20)
                        except Exception as exc:
                            self.logger.warning(
                                "failed to get recent events: %s",
                                exc,
                                extra=log_record(
                                    workflow_id=locals().get("workflow_id", "self_improvement"),
                                    attempt=self._cycle_count,
                                ),
                            )
                            logs = None
                    try:
                        info_proc = subprocess.run(
                            ["git", "show", "-s", "--format=%an,%s"],
                            capture_output=True,
                            text=True,
                            check=True,
                            timeout=60,
                        )
                        out = info_proc.stdout.strip()
                        author, message = out.split(",", 1)
                        commit_info = {"author": author, "message": message}
                    except subprocess.CalledProcessError as exc:
                        self.logger.error(
                            "git show failed",
                            extra=log_record(
                                cmd=exc.cmd, rc=exc.returncode, output=exc.stderr
                            ),
                        )
                        commit_info = None
                    except subprocess.TimeoutExpired as exc:
                        self.logger.error(
                            "git show timed out",
                            extra=log_record(cmd=exc.cmd, timeout=exc.timeout, output=exc.stderr),
                        )
                        commit_info = None
                    except Exception:
                        commit_info = None
                    warnings = agent.evaluate_changes(actions, metrics, logs, commit_info)
                    if any(warnings.values()):
                        result.warnings = warnings
                except Exception as exc:
                    self.logger.exception("improvement flagging failed: %s", exc)
            self.logger.info(
                "roi delta",
                extra=log_record(roi_delta=delta, warnings=warnings),
            )
            self._record_warning_summary(delta, warnings)
            self.roi_delta_ema = (
                1 - self.roi_ema_alpha
            ) * self.roi_delta_ema + self.roi_ema_alpha * delta
            group_idx = None
            if self.patch_db:
                try:
                    repo = _repo_path()
                    with self.patch_db._connect() as conn:
                        row = conn.execute(
                            "SELECT filename FROM patch_history ORDER BY id DESC LIMIT 1"
                        ).fetchone()
                    if row:
                        p = Path(row[0])
                        abs_p = p if p.is_absolute() else repo / p
                        try:
                            mod_name = abs_p.resolve().relative_to(repo).as_posix()
                        except Exception:
                            mod_name = p.name
                        group_idx = self.module_clusters.get(mod_name)
                        if group_idx is None and self.module_index:
                            group_idx = self.module_index.get(mod_name)
                            self.module_clusters[mod_name] = group_idx
                except Exception as exc:
                    self.logger.exception("group index lookup failed: %s", exc)
            if group_idx is not None:
                self.roi_group_history.setdefault(int(group_idx), []).append(delta)
            tracker = getattr(self, "tracker", None)
            raroi_delta = 0.0
            if tracker is not None and len(tracker.raroi_history) >= 2:
                raroi_delta = tracker.raroi_history[-1] - tracker.raroi_history[-2]
            workflow_id = "self_improvement"
            wf_ctx = getattr(environment, "current_context", None)
            try:
                ctx_obj = wf_ctx() if callable(wf_ctx) else wf_ctx
                workflow_id = getattr(ctx_obj, "workflow_id", workflow_id)
            except Exception:
                logger.exception("Unhandled exception in self_improvement")
            try:
                profile_map = getattr(self.foresight_tracker, "workflow_profiles", None)
                if not isinstance(profile_map, Mapping):
                    profile_map = getattr(self.foresight_tracker, "profile_map", {})
                profile = profile_map.get(workflow_id, workflow_id)
                self.foresight_tracker.capture_from_roi(tracker, workflow_id, profile)
            except Exception:
                self.logger.exception("foresight tracker record failed")
            risk_info: dict[str, object] | None = None
            try:
                if self.foresight_tracker:
                    risk_info = self.foresight_tracker.predict_roi_collapse(workflow_id)
            except Exception:
                self.logger.exception("foresight risk check failed")
            prior_high = self.workflow_high_risk
            self.workflow_risk = risk_info
            high_risk = bool(
                risk_info
                and (
                    risk_info.get("risk") in {"Immediate collapse risk", "Volatile"}
                    or bool(risk_info.get("brittle"))
                )
            )
            self.workflow_high_risk = high_risk
            if high_risk:
                self.workflow_ready = False
                if not prior_high:
                    try:
                        self.enqueue_preventative_fixes([workflow_id])
                    except Exception:
                        self.logger.exception("risk queue enqueue failed")
            if result.warnings is None:
                result.warnings = {}
            result.warnings.setdefault("foresight_risk", [])
            if risk_info:
                result.warnings["foresight_risk"].append(risk_info)
            result.warnings["workflow_high_risk"] = [{"value": high_risk}]
            self.roi_history.append(delta)
            self.raroi_history.append(raroi_delta)
            self._save_state()
            self._update_synergy_weights(delta)
            self.logger.info(
                "cycle summary",
                extra=log_record(
                    roi_delta=delta,
                    patch_success=patch_rate,
                    roi_trend=trend,
                    anomaly=anomaly,
                    synergy_weights=self.synergy_learner.weights,
                    growth_type=self._last_growth_type,
                ),
            )
            self._evaluate_roi_predictor()
            if self._score_backend:
                try:
                    self._score_backend.store(
                        {
                            "description": self.bot_name,
                            "result": "success" if delta >= 0 else "decline",
                            "roi_delta": float(delta),
                            "score": float(delta),
                        }
                    )
                except Exception:
                    self.logger.exception("patch score backend store failed")
            if self.policy:
                try:
                    next_state = self._policy_state()
                    syn_reward = next_state[-2] / 10.0 + next_state[-1] / 10.0
                    self.policy.update(
                        state,
                        after_roi - before_roi + syn_reward,
                        next_state,
                        synergy_roi_delta=next_state[-4] / 10.0,
                        synergy_efficiency_delta=next_state[-3] / 10.0,
                    )
                    if getattr(self.policy, "path", None):
                        try:
                            self.policy.save()
                        except Exception as exc:  # pragma: no cover - best effort
                            self.logger.exception("policy save failed: %s", exc)
                    self.logger.info(
                        "policy updated",
                        extra=log_record(
                            reward=after_roi - before_roi,
                            state=state,
                            next_state=next_state,
                            weights=self.synergy_learner.weights,
                        ),
                    )
                except Exception as exc:
                    self.logger.exception("policy update failed: %s", exc)
            if self.policy and getattr(self.policy, "path", None):
                try:
                    self.policy.save()
                except Exception as exc:  # pragma: no cover - best effort
                    self.logger.exception("policy save failed: %s", exc)
            with MutationLogger.log_context(
                change="self_improvement_cycle",
                reason="cycle complete",
                trigger="run_cycle",
                workflow_id=0,
                before_metric=before_roi,
                parent_id=self._last_mutation_id,
            ) as mutation:
                mutation["after_metric"] = after_roi
                mutation["performance"] = delta
                mutation["roi"] = roi_realish
            self._last_mutation_id = int(mutation["event_id"])
            if self.workflow_evolver and evo_allowed:
                try:
                    candidates = self.pathway_db.top_sequences(limit=3)
                except Exception:
                    candidates = []
                for seq, _score in candidates:
                    wf_id = hash(seq) & 0xFFFFFFFF
                    if self.workflow_evolver.is_stable(wf_id):
                        continue
                    try:
                        status = self._evaluate_workflow_variants(seq, wf_id)
                        workflow_evolution_details.append(
                            {"workflow_id": wf_id, "sequence": seq, "status": status}
                        )
                    except Exception:
                        self.logger.exception(
                            "workflow evolution failed",
                            extra=log_record(workflow_id=wf_id),
                        )
                try:
                    evo_results = self._evolve_workflows()
                    for wf_id, summary in evo_results.items():
                        detail = {"workflow_id": wf_id, **summary}
                        workflow_evolution_details.append(detail)
                except Exception:
                    self.logger.exception("workflow evolution layer failed")
                finally:
                    try:
                        self._update_orphan_modules()
                        repo = _repo_path()
                        traces = getattr(self, "orphan_traces", {})
                        if append_orphan_cache:
                            cache_entries = {
                                m: {
                                    "parents": info.get("parents", []),
                                    "classification": info.get("classification", "candidate"),
                                    "redundant": info.get("redundant", False),
                                }
                                for m, info in traces.items()
                            }
                            try:
                                append_orphan_cache(repo, cache_entries)
                            except Exception:
                                self.logger.exception("orphan cache update failed")
                        if append_orphan_classifications:
                            class_entries = {
                                m: {
                                    "parents": info.get("parents", []),
                                    "classification": info.get("classification", "candidate"),
                                    "redundant": info.get("redundant", False),
                                }
                                for m, info in traces.items()
                            }
                            try:
                                append_orphan_classifications(repo, class_entries)
                            except Exception:
                                self.logger.exception("orphan classification update failed")
                        if append_orphan_traces:
                            trace_entries = {
                                m: {
                                    "classification_history": info.get("classification_history", []),
                                    "roi_history": info.get("roi_history", []),
                                }
                                for m, info in traces.items()
                                if info.get("classification_history") or info.get("roi_history")
                            }
                            if trace_entries:
                                try:
                                    append_orphan_traces(repo, trace_entries)
                                except Exception:
                                    self.logger.exception("orphan trace update failed")
                    except Exception:
                        self.logger.exception("post evolution orphan update failed")
            try:
                flags = radar_scan()
                if flags:
                    self._handle_relevancy_flags(flags)
            except Exception:
                self.logger.exception("relevancy radar scan failed")
            if self.foresight_tracker and self.workflow_risk is None:
                workflow_id = "self_improvement"
                wf_ctx = getattr(environment, "current_context", None)
                try:
                    ctx_obj = wf_ctx() if callable(wf_ctx) else wf_ctx
                    workflow_id = getattr(ctx_obj, "workflow_id", workflow_id)
                except Exception:
                    logger.exception("Unhandled exception in self_improvement")
                risk_info: dict[str, object] | None = None
                try:
                    risk_info = self.foresight_tracker.predict_roi_collapse(workflow_id)
                except Exception:
                    self.logger.exception("foresight risk check failed")
                prior_high = self.workflow_high_risk
                self.workflow_risk = risk_info
                high_risk = bool(
                    risk_info
                    and (
                        risk_info.get("risk") in {"Immediate collapse risk", "Volatile"}
                        or bool(risk_info.get("brittle"))
                    )
                )
                self.workflow_high_risk = high_risk
                if high_risk:
                    self.workflow_ready = False
                    if not prior_high:
                        try:
                            self.enqueue_preventative_fixes([workflow_id])
                        except Exception:
                            self.logger.exception("risk queue enqueue failed")
                if result.warnings is None:
                    result.warnings = {}
                result.warnings.setdefault("foresight_risk", [])
                if risk_info:
                    result.warnings["foresight_risk"].append(risk_info)
                result.warnings["workflow_high_risk"] = [{"value": high_risk}]
            if planner_chains:
                workflow_evolution_details.append({"planner_chains": planner_chains})
            if meta_records:
                workflow_evolution_details.append({"meta_workflows": meta_records})
            if workflow_evolution_details:
                result.workflow_evolution = workflow_evolution_details
            if getattr(self, "workflow_scorer", None):
                try:
                    self.workflow_scorer.score_workflow(
                        "self_improvement_cycle",
                        {"cycle": lambda: True},
                    )
                except Exception:
                    self.logger.exception("workflow scoring failed")
            pass_rate = getattr(getattr(result, "roi", None), "success_rate", 0.0)
            repo_files = list(repo_path.rglob("*.py"))
            entropy = _si_metrics.compute_code_entropy(repo_files)
            current_roi = roi_realish
            pass_rate_avg = self.baseline_tracker.get("pass_rate")
            roi_avg = self.roi_baseline.average()
            entropy_avg = self.entropy_baseline.average()
            energy_avg = self.baseline_tracker.get("energy")
            self.baseline_tracker.update(
                roi=current_roi, pass_rate=pass_rate, entropy=entropy
            )
            delta: dict[str, float] = {}
            if self._snapshot_tracker:
                ctx = {
                    "files": repo_files,
                    "roi": self.baseline_tracker.current("roi"),
                    "sandbox_score": self.baseline_tracker.current("score"),
                    "prompt": str(snapshot_prompt) if snapshot_prompt is not None else None,
                    "diff": snapshot_diff,
                }
                self._snapshot_tracker.capture("after", ctx, repo_path=repo_path)
                delta = self._snapshot_tracker.delta()
                self._last_delta = delta
                self._record_snapshot_delta(
                    snapshot_prompt, snapshot_diff, delta, changed_files, cid
                )
            self._check_momentum()
            self._check_delta_score()
            self._check_roi_stagnation()
            self.roi_baseline.append(current_roi)
            self.entropy_baseline.append(entropy)
            pass_rate_delta = pass_rate - pass_rate_avg
            roi_delta = current_roi - roi_avg
            entropy_delta = entropy - entropy_avg
            self.entropy_delta_ema = (
                (1 - self.entropy_weight) * self.entropy_delta_ema
                + self.entropy_weight * entropy_delta
            )
            momentum_delta = self.baseline_tracker.delta("momentum")
            combined_delta = (
                self.roi_weight * roi_delta
                + self.pass_rate_weight * pass_rate_delta
                + self.momentum_weight * momentum_delta
                - self.entropy_weight * entropy_delta
            )
            if combined_delta > 0:
                try:
                    update_alignment_baseline(settings)
                except Exception:
                    self.logger.exception("alignment baseline update failed")
            energy_delta = energy - energy_avg
            roi_db = getattr(self, "roi_db", None)
            if roi_db is None:
                try:
                    roi_db = ROIResultsDB()
                except Exception:
                    roi_db = None
                else:
                    self.roi_db = roi_db
            if roi_db is not None:
                try:
                    roi_db.log_result(
                        workflow_id="self_improvement_cycle",
                        run_id=str(self._cycle_count),
                        runtime=0.0,
                        success_rate=pass_rate,
                        roi_gain=current_roi,
                        workflow_synergy_score=max(0.0, 1.0 - entropy),
                        bottleneck_index=0.0,
                        patchability_score=0.0,
                        code_entropy=entropy,
                        entropy_delta=entropy_delta,
                        module_deltas={},
                    )
                except Exception:
                    self.logger.exception("ROI logging failed")
            self.logger.info(
                "cycle complete",
                extra=log_record(
                    roi=current_roi,
                    predicted_roi=pred_realish,
                    pass_rate_delta=pass_rate_delta,
                    roi_delta=roi_delta,
                    entropy_delta=entropy_delta,
                    energy_delta=energy_delta,
                    momentum_delta=momentum_delta,
                ),
            )
            return result
        except Exception:
            self_improvement_failure_total.labels(reason="run_cycle").inc()
            self.logger.exception(
                "cycle failure", extra=log_record(event="failure")
            )
            raise
        finally:
            self._cycle_running = False
            set_correlation_id(None)
            self.logger.info(
                "cycle shutdown", extra=log_record(event="shutdown")
            )

    async def _schedule_loop(self, energy: int = 1) -> None:
        while not self._stop_event.is_set():
            current_energy = energy
            if self.capital_bot:
                try:
                    current_energy = self.capital_bot.energy_score(
                        load=0.0,
                        success_rate=1.0,
                        deploy_eff=1.0,
                        failure_rate=0.0,
                    )
                except Exception as exc:
                    self.logger.exception("energy check failed: %s", exc)
                    current_energy = energy
            if self.roi_predictor and self.use_adaptive_roi:
                features = self._collect_action_features()
                try:
                    try:
                        seq, growth_type, _, _ = self.roi_predictor.predict(
                            features, horizon=len(features)
                        )
                    except TypeError:
                        val, growth_type, _, _ = self.roi_predictor.predict(features)
                        seq = [float(val)]
                    roi_estimate = float(seq[-1]) if seq else 0.0
                except Exception:
                    roi_estimate, growth_type = 0.0, "unknown"
                self.logger.info(
                    "growth prediction",
                    extra=log_record(
                        growth_type=growth_type,
                        roi_estimate=roi_estimate,
                        features=features,
                    ),
                )
                self._last_growth_type = growth_type
                if growth_type == "exponential":
                    current_energy *= 1.2
                elif growth_type == "marginal":
                    current_energy *= 0.8
            else:
                self._last_growth_type = None
            momentum = self.baseline_tracker.momentum
            if momentum != self._last_momentum:
                self.logger.info(
                    "momentum update",
                    extra=log_record(value=momentum, previous=self._last_momentum),
                )
                self._last_momentum = momentum
            current_energy *= 1 + momentum
            # Fetch recent error telemetry and recent entropy change.  The error
            # count is used as a proxy for overall system health while the
            # entropy delta is examined for potential overfitting.
            (
                traces,
                recent_entropy_delta,
                error_count,
                _entropy_mean,
                _entropy_std,
            ) = meta_planning._recent_error_entropy(
                self.error_bot,
                self.baseline_tracker,
                getattr(SandboxSettings(), "error_window", 5),
            )
            error_count = float(error_count)
            critical_errors = any(
                getattr(getattr(ev, "error_type", None), "severity", None) == "critical"
                for ev in traces
            )
            moving_avg = self.baseline_tracker.get("energy")
            std = self.baseline_tracker.std("energy")
            self.baseline_tracker.update(energy=current_energy, error_count=error_count)
            self._save_state()
            scale = 1.0 + (0.5 - momentum)
            threshold = moving_avg - self.energy_threshold * scale * std
            # Combine weighted deltas from multiple metrics to assess overall
            # system movement.  Entropy increases and momentum drops will
            # reduce the score while ROI and pass rate gains increase it.
            delta_score, components = self._compute_delta_score()
            error_deltas = self.baseline_tracker.delta_history("error_count")
            error_count_delta = error_deltas[-1] if error_deltas else 0.0
            components["error_count_delta"] = error_count_delta
            roi_delta = components.get("roi_delta", 0.0)
            pass_rate_delta = components.get("pass_rate_delta", 0.0)
            momentum_delta = components.get("momentum_delta", 0.0)
            entropy_delta = components.get("entropy_delta", 0.0)
            entropy_std = self.baseline_tracker.std("entropy")
            entropy_threshold = self.entropy_dev_multiplier * entropy_std
            entropy_spike = abs(recent_entropy_delta) > entropy_threshold
            within_baseline = current_energy >= threshold
            metric_values = {
                "roi": self.baseline_tracker.current("roi"),
                "pass_rate": self.baseline_tracker.current("pass_rate"),
                "momentum": self.baseline_tracker.momentum,
                "error_count": error_count,
            }
            error_traces = traces
            overfit_signal = False
            if WorkflowSynergyComparator and hasattr(
                WorkflowSynergyComparator, "analyze_overfitting"
            ):
                spec = getattr(self, "last_workflow_spec", None)
                if spec is not None:
                    try:
                        report = WorkflowSynergyComparator.analyze_overfitting(spec)
                        overfit_signal = getattr(report, "is_overfitting", lambda: False)()
                    except Exception as exc:  # pragma: no cover - best effort
                        self.logger.debug("overfitting analysis failed: %s", exc)
            signals: dict[str, object] = {}
            if error_traces:
                signals["errors"] = len(error_traces)
            if entropy_spike:
                signals["entropy_delta"] = recent_entropy_delta
            if overfit_signal:
                signals["overfitting"] = True
            decision = "escalate"
            run_energy: int | None = None
            log_fields: dict[str, object] = {
                "energy": current_energy,
                "baseline": threshold,
                **metric_values,
                **components,
                "delta_score": delta_score,
            }
            if signals:
                log_fields.update(signals)

            if signals and not self._cycle_running:
                decision = "fallback"
                run_energy = int(round(current_energy * 5))
                msg = "forcing run_cycle due to fallback signals"
            elif (
                roi_delta > 0
                and pass_rate_delta > 0
                and momentum_delta > 0
                and error_count_delta <= 0
                and within_baseline
                and not critical_errors
            ):
                decision = "skip"
                msg = "positive deltas - skipping cycle"
            elif not within_baseline and not self._cycle_running:
                decision = "cycle"
                run_energy = int(round(current_energy * 5))
                msg = "triggering run_cycle due to low energy"
            else:
                self.urgency_tier += 1
                log_fields["tier"] = self.urgency_tier
                msg = "non-positive deltas - escalating urgency"

            log_fields["decision"] = decision
            record = log_record(**log_fields)
            if decision == "escalate":
                self.logger.warning(msg, extra=record)
            else:
                self.logger.info(msg, extra=record)

            if run_energy is not None:
                try:
                    await asyncio.to_thread(self.run_cycle, energy=run_energy)
                except Exception as exc:
                    self.logger.exception(
                        "self improvement run_cycle failed with energy %s: %s",
                        run_energy,
                        exc,
                    )
            await asyncio.sleep(self.interval)

    def schedule(
        self, energy: int = 1, *, loop: asyncio.AbstractEventLoop | None = None
    ) -> asyncio.Task:
        """Start the scheduling loop in the background."""
        if self._schedule_task and not self._schedule_task.done():
            return self._schedule_task
        self.logger.info(
            "scheduling started",
            extra=log_record(energy=energy),
        )
        self._stop_event = asyncio.Event()
        loop = loop or asyncio.get_event_loop()
        self._schedule_task = loop.create_task(self._schedule_loop(energy))
        return self._schedule_task

    async def shutdown_schedule(self) -> None:
        """Stop the scheduler and wait for the task to finish."""
        if self._schedule_task:
            assert self._stop_event is not None
            self.logger.info("schedule shutdown initiated")
            self._stop_event.set()
            try:
                await self._schedule_task
                self.logger.info("schedule task finished")
            finally:
                self._schedule_task = None

    def status(self) -> dict[str, object]:
        """Expose the latest workflow risk evaluation."""
        return {
            "workflow_ready": self.workflow_ready,
            "workflow_high_risk": self.workflow_high_risk,
            "workflow_risk": self.workflow_risk,
        }


from typing import Any, Callable, Optional, Type, Iterable, Sequence, Dict, List


def cli(argv: list[str] | None = None) -> None:
    """Command line interface for synergy utilities."""
    import argparse

    parser = argparse.ArgumentParser(description="Self-improvement utilities")
    parser.add_argument("--config", help="Path to sandbox settings file")
    parser.add_argument(
        "--mutation-rate",
        type=float,
        default=None,
        help="Mutation rate multiplier for meta planning",
    )
    parser.add_argument(
        "--roi-weight",
        type=float,
        default=None,
        help="Weight applied to ROI when composing pipelines",
    )
    parser.add_argument(
        "--domain-penalty",
        type=float,
        default=None,
        help="Penalty for transitioning between workflow domains",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_dash = sub.add_parser("synergy-dashboard", help="start synergy metrics dashboard")
    p_dash.add_argument("--file", default="synergy_history.db")
    p_dash.add_argument("--port", type=int, default=5001)
    p_dash.add_argument("--exporter-host")
    p_dash.add_argument("--exporter-port", type=int, default=8003)
    p_dash.add_argument("--refresh-interval", type=float, default=5.0)
    p_dash.add_argument(
        "--max-history",
        type=int,
        default=1000,
        help="max entries to keep in memory when polling an exporter",
    )
    p_dash.add_argument(
        "--wsgi",
        choices=["flask", "gunicorn", "uvicorn"],
        default="flask",
        help="WSGI/ASGI server to use",
    )

    p_plot = sub.add_parser("plot-synergy", help="plot synergy metrics")
    p_plot.add_argument("history", help="synergy_history.db file")
    p_plot.add_argument("output", help="output PNG file")

    p_fit = sub.add_parser(
        "fit-truth-adapter", help="retrain TruthAdapter with live and shadow data"
    )
    p_fit.add_argument("live", help="NPZ file with live data")
    p_fit.add_argument("shadow", help="NPZ file with shadow data")

    p_update = sub.add_parser(
        "update-truth-adapter",
        help="incrementally update TruthAdapter or reset if retraining required",
    )
    p_update.add_argument("live", help="NPZ file with live data")
    p_update.add_argument("shadow", help="NPZ file with shadow data")

    args = parser.parse_args(argv)
    cfg = load_sandbox_settings(args.config)
    mutation_rate = args.mutation_rate if args.mutation_rate is not None else cfg.meta_mutation_rate
    roi_weight = args.roi_weight if args.roi_weight is not None else cfg.meta_roi_weight
    domain_penalty = (
        args.domain_penalty if args.domain_penalty is not None else cfg.meta_domain_penalty
    )
    os.environ["META_MUTATION_RATE"] = str(mutation_rate)
    os.environ["META_ROI_WEIGHT"] = str(roi_weight)
    os.environ["META_DOMAIN_PENALTY"] = str(domain_penalty)

    if args.cmd == "synergy-dashboard":
        dash = SynergyDashboard(
            args.file,
            exporter_host=args.exporter_host,
            exporter_port=args.exporter_port,
            refresh_interval=args.refresh_interval,
            max_history=args.max_history,
        )
        dash.run(port=args.port, wsgi=args.wsgi)
        return

    if args.cmd == "plot-synergy":
        hist = load_synergy_history(args.history)
        try:
            import matplotlib.pyplot as plt  # type: ignore
        except Exception:
            parser.error("matplotlib is required for plotting")
        labels = list(range(len(hist)))
        metrics = sorted({k for d in hist for k in d})
        for name in metrics:
            vals = [d.get(name, 0.0) for d in hist]
            plt.plot(labels, vals, label=name)
        if metrics:
            plt.legend()
        plt.xlabel("iteration")
        plt.ylabel("value")
        plt.tight_layout()
        plt.savefig(args.output)
        plt.close()
        return

    if args.cmd == "fit-truth-adapter":
        live = np.load(args.live)
        shadow = np.load(args.shadow)
        X = np.vstack([live["X"], shadow["X"]])
        y = np.concatenate([live["y"], shadow["y"]])
        engine = SelfImprovementEngine(context_builder=create_context_builder())
        engine.fit_truth_adapter(X, y)
        return

    if args.cmd == "update-truth-adapter":
        live = np.load(args.live)
        shadow = np.load(args.shadow)
        X = np.vstack([live["X"], shadow["X"]])
        y = np.concatenate([live["y"], shadow["y"]])
        engine = SelfImprovementEngine(context_builder=create_context_builder())
        adapter = engine.truth_adapter
        if adapter.metadata.get("retraining_required"):
            adapter.reset()
            engine.fit_truth_adapter(X, y)
        else:
            adapter.partial_fit(X, y)
        return

    parser.error("unknown command")


def main(argv: list[str] | None = None) -> None:
    cli(argv)


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()
else:  # pragma: no cover - import-time execution path for Codex shim
    if not _MANUAL_LAUNCH_TRIGGERED:
        _MANUAL_LAUNCH_TRIGGERED = True
        _qfe_log("__name__ != '__main__'; forcing manual autonomous launch")
        try:
            runner = importlib.import_module("menace_sandbox.run_autonomous")
        except ImportError:
            try:
                runner = importlib.import_module("run_autonomous")
            except ImportError as exc:
                _qfe_log(
                    "unable to import run_autonomous for manual launch; skipping"
                )
                logger.exception("manual run_autonomous import failed", exc_info=exc)
            else:
                _qfe_log("run_autonomous imported via flat layout; invoking main()")
                try:
                    runner.main([])
                except Exception as exc:  # pragma: no cover - diagnostic only
                    _qfe_log(
                        f"run_autonomous.main() raised {exc!r}; see traceback for details"
                    )
                    logger.exception("manual run_autonomous execution failed", exc_info=exc)
        else:
            _qfe_log("run_autonomous imported; invoking main()")
            try:
                runner.main([])
            except Exception as exc:  # pragma: no cover - diagnostic only
                _qfe_log(
                    f"run_autonomous.main() raised {exc!r}; see traceback for details"
                )
                logger.exception("manual run_autonomous execution failed", exc_info=exc)
