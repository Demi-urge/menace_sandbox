from __future__ import annotations

"""Manage self-coding patches and deployment cycles.

Many operations require a provenance token issued by the active
``EvolutionOrchestrator``.  Call :func:`validate_provenance` to verify that
requests originate from the orchestrator before proceeding.
"""

from pathlib import Path
import sys

try:  # pragma: no cover - allow flat imports
    from .dynamic_path_router import resolve_path, path_for_prompt
except Exception:  # pragma: no cover - fallback for flat layout
    from dynamic_path_router import resolve_path, path_for_prompt  # type: ignore
try:  # pragma: no cover - import module for cache management
    from . import dynamic_path_router as _path_router
except Exception:  # pragma: no cover - fallback for flat layout
    import dynamic_path_router as _path_router  # type: ignore
import logging
import subprocess
import tempfile
import threading
import time
import re
import json
import uuid
import os
import importlib
import shlex
from dataclasses import asdict
from typing import Dict, Any, TYPE_CHECKING, Callable, Iterator, Iterable
from contextlib import contextmanager

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
    from .error_bot import ErrorDB
except Exception:  # pragma: no cover - provide stub when unavailable
    ErrorDB = None  # type: ignore
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
    from .model_automation_pipeline import AutomationResult, ModelAutomationPipeline

_MODEL_AUTOMATION_PIPELINE_CLS: type["ModelAutomationPipeline"] | None = None
_AUTOMATION_RESULT_CLS: type["AutomationResult"] | None = None


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


for _mod_name in ("quick_fix_engine", "menace.quick_fix_engine"):
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


class PatchApprovalPolicy:
    """Run formal verification and tests before patching.

    The test runner command can be customised via ``test_command``.
    When omitted the command is loaded from the ``threshold_service`` or
    per-bot configuration.
    """

    def __init__(
        self,
        *,
        verifier: FormalVerifier | None = None,
        rollback_mgr: AutomatedRollbackManager | None = None,
        bot_name: str = "menace",
        test_command: list[str] | None = None,
        threshold_service: ThresholdService | None = None,
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

    # ------------------------------------------------------------------
    def update_test_command(self, new_cmd: list[str]) -> None:
        """Refresh the command used for running tests."""
        self.test_command = list(new_cmd)

    def approve(self, path: Path) -> bool:
        ok = True
        try:
            if self.verifier and not self.verifier.verify(path):
                ok = False
        except Exception as exc:  # pragma: no cover - verification issues
            self.logger.error("verification failed: %s", exc)
            ok = False
        try:
            subprocess.run(self.test_command, check=True)
        except Exception as exc:  # pragma: no cover - test runner issues
            self.logger.error("self tests failed: %s", exc)
            ok = False
        if ok and self.rollback_mgr:
            try:
                self.rollback_mgr.log_healing_action(
                    self.bot_name, "patch_checks", path_for_prompt(path)
                )
            except Exception as exc:  # pragma: no cover - audit logging issues
                self.logger.exception("failed to log healing action: %s", exc)
        return ok


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
        error_db: ErrorDB | None = None,
        event_bus: UnifiedEventBus | None = None,
        evolution_orchestrator: "EvolutionOrchestrator | None" = None,
        threshold_service: ThresholdService | None = None,
        roi_drop_threshold: float | None = None,
        error_rate_threshold: float | None = None,
    ) -> None:
        if data_bot is None or bot_registry is None:
            raise ValueError("data_bot and bot_registry are required")
        self.engine = self_coding_engine
        self.pipeline = pipeline
        self.bot_name = bot_name
        self.data_bot = data_bot
        self.threshold_service = threshold_service or _DEFAULT_THRESHOLD_SERVICE
        self.approval_policy = approval_policy
        self.logger = logging.getLogger(self.__class__.__name__)
        self._last_patch_id: int | None = None
        self._last_event_id: int | None = None
        self._last_commit_hash: str | None = None
        self._last_validation_summary: Dict[str, Any] | None = None
        thresholds = self.threshold_service.get(bot_name)
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
                baseline_window = getattr(SandboxSettings(), "baseline_window", 5)
            except Exception:
                baseline_window = 5
        baseline_tracker_cls = _get_baseline_tracker_cls()
        self.baseline_tracker = baseline_tracker_cls(
            window=int(baseline_window), metrics=["confidence"]
        )
        try:
            settings = SandboxSettings()
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
                self.bot_registry.register_bot(
                    self.bot_name,
                    manager=self,
                    data_bot=self.data_bot,
                    is_coding_bot=True,
                )
            except Exception:  # pragma: no cover - best effort
                self.logger.exception("failed to register bot in registry")

        try:
            clayer = self.engine.cognition_layer
        except AttributeError as exc:
            raise RuntimeError(
                "engine must provide a cognition_layer with a context_builder"
            ) from exc
        if clayer is None:
            raise RuntimeError(
                "engine.cognition_layer must provide a context_builder"
            )
        try:
            builder = clayer.context_builder
        except AttributeError as exc:
            raise RuntimeError(
                "engine.cognition_layer must provide a context_builder"
            ) from exc
        if builder is None:
            raise RuntimeError(
                "engine.cognition_layer must provide a context_builder"
            )
        self._prepare_context_builder(builder)
        self._init_quick_fix_engine(builder)

        if self.evolution_orchestrator is None:
            try:  # pragma: no cover - optional dependencies
                from .capital_management_bot import CapitalManagementBot
                from .self_improvement.engine import SelfImprovementEngine
                from .system_evolution_manager import SystemEvolutionManager
                from .evolution_orchestrator import EvolutionOrchestrator

                capital = CapitalManagementBot(data_bot=self.data_bot)
                improv = SelfImprovementEngine(
                    context_builder=builder,
                    data_bot=self.data_bot,
                    bot_name=self.bot_name,
                )
                bots = list(getattr(self.bot_registry, "graph", {}).keys())
                evol_mgr = SystemEvolutionManager(bots)
                self.evolution_orchestrator = EvolutionOrchestrator(
                    data_bot=self.data_bot,
                    capital_bot=capital,
                    improvement_engine=improv,
                    evolution_manager=evol_mgr,
                    selfcoding_manager=self,
                    event_bus=self.event_bus,
                )
            except Exception as exc:  # pragma: no cover - best effort
                raise RuntimeError(
                    "EvolutionOrchestrator is required but could not be constructed",
                ) from exc

        if not self.evolution_orchestrator:
            raise RuntimeError(
                "EvolutionOrchestrator is required but could not be constructed",
            )

        try:  # pragma: no cover - best effort
            self.evolution_orchestrator.register_bot(self.bot_name)
        except Exception:
            self.logger.exception(
                "failed to register bot with evolution orchestrator",
            )

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

    def _refresh_thresholds(self) -> None:
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
            t = self.threshold_service.reload(self.bot_name)

            adaptive = False
            try:
                adaptive = getattr(SandboxSettings(), "adaptive_thresholds", False)
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
        db = self.error_db or ErrorDB()
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
            self.logger.error(
                "no workflow tests resolved for bot %s", self.bot_name
            )
            raise RuntimeError(
                f"no workflow tests resolved for {self.bot_name}; cannot run validation"
            )

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

        self.validate_provenance(provenance_token)
        if self.quick_fix is None:
            raise RuntimeError("QuickFixEngine validation unavailable")
        repo_root = Path.cwd().resolve()
        module = Path(module_path)
        if not module.is_absolute():
            module = (repo_root / module).resolve()
        if not module.exists():
            raise FileNotFoundError(f"module path not found: {module}")
        self.refresh_quick_fix_context()
        summary: dict[str, Any] = {}
        ctx_meta = dict(context_meta or {})
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
                        valid, flags = self.quick_fix.validate_patch(
                            str(cloned_module),
                            description,
                            repo_root=clone_root,
                            provenance_token=provenance_token,
                        )
                        summary["quick_fix"] = {
                            "validation_flags": list(flags),
                        }
                        if not valid or flags:
                            raise RuntimeError(
                                "quick fix validation failed"
                            )
                        passed, _pid, apply_flags = self.quick_fix.apply_validated_patch(
                            str(cloned_module),
                            description,
                            ctx_meta,
                            provenance_token=provenance_token,
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
            base_kwargs = dict(svc_kwargs)
            base_pytest_args = base_kwargs.get("pytest_args")
            attempt_records: list[dict[str, Any]] = []
            attempt_count = 0
            current_pytest_args = base_pytest_args
            results: dict[str, Any] = {}
            passed_modules: list[str] = []
            while True:
                run_kwargs = dict(base_kwargs)
                if current_pytest_args is None:
                    run_kwargs.pop("pytest_args", None)
                else:
                    run_kwargs["pytest_args"] = current_pytest_args
                try:
                    service = _SelfTestService(**run_kwargs)
                except FileNotFoundError as exc:
                    raise RuntimeError("SelfTestService initialization failed") from exc
                results, passed_modules = service.run_once()
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
                    ctx_meta_attempt["repair_node_ids"] = list(diagnostics["node_ids"])
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
                    passed, patch_id, apply_flags = self.quick_fix.apply_validated_patch(
                        str(module),
                        repair_desc,
                        ctx_meta_attempt,
                        provenance_token=provenance_token,
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
                    next_pytest_args if next_pytest_args is not None else base_pytest_args
                )
                attempt_count = attempt_index
        except Exception as exc:
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
            raise PermissionError("EvolutionOrchestrator required")
        expected = getattr(orchestrator, "provenance_token", None)
        if not token or token != expected:
            self.logger.warning(
                "patch cycle without valid EvolutionOrchestrator token",
            )
            raise PermissionError("invalid provenance token")

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
        result = self.run_patch(path, description, provenance_token=token, **kwargs)
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
        self.refresh_quick_fix_context()
        if self.approval_policy and not self.approval_policy.approve(path):
            raise RuntimeError("patch approval failed")
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
            except Exception:  # pragma: no cover - best effort
                self.logger.exception(
                    "failed to apply suggestion for %s", prompt_module
                )
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
    manager = SelfCodingManager(
        engine,
        pipeline,
        bot_name=bot_name,
        data_bot=data_bot,
        bot_registry=bot_registry,
        roi_drop_threshold=roi_threshold,
        error_rate_threshold=error_threshold,
        **manager_kwargs,
    )
    if manager.quick_fix is None:
        raise ImportError("QuickFixEngine failed to initialise")
    manager.evolution_orchestrator = evolution_orchestrator
    bot_registry.register_bot(
        bot_name,
        roi_threshold=roi_threshold,
        error_threshold=error_threshold,
        test_failure_threshold=test_failure_threshold,
        manager=manager,
        data_bot=data_bot,
        is_coding_bot=True,
    )
    try:
        data_bot.schedule_monitoring(bot_name)
    except Exception:  # pragma: no cover - best effort
        manager.logger.exception(
            "failed to schedule monitoring for %s", bot_name
        )
    settings = getattr(data_bot, "settings", None)
    thresholds = getattr(settings, "bot_thresholds", {}) if settings else {}
    if bot_name not in thresholds:
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
    if evolution_orchestrator is not None:
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
    event_bus = (
        getattr(manager, "event_bus", None)
        or getattr(evolution_orchestrator, "event_bus", None)
        or getattr(data_bot, "event_bus", None)
    )
    module_path: Path | None = None
    try:
        node = bot_registry.graph.nodes.get(bot_name) if bot_registry else None
        if node:
            module_str = node.get("module")
            if module_str:
                module_path = Path(module_str)
    except Exception:
        module_path = None
    if (module_path is None or not module_path.exists()) and bot_name in getattr(bot_registry, "modules", {}):
        try:
            module_entry = bot_registry.modules.get(bot_name)
            if module_entry:
                module_path = Path(module_entry)
        except Exception:
            module_path = None
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
    provenance_token = None
    if getattr(manager, "evolution_orchestrator", None) is not None:
        provenance_token = getattr(manager.evolution_orchestrator, "provenance_token", None)
    if provenance_token is None and evolution_orchestrator is not None:
        provenance_token = getattr(evolution_orchestrator, "provenance_token", None)
    description = f"internalize:{bot_name}"

    def _emit_failure(reason: str) -> None:
        data_bot_ref = getattr(manager, "data_bot", None)
        if data_bot_ref:
            try:
                data_bot_ref.collect(
                    bot_name,
                    post_patch_cycle_success=0.0,
                    post_patch_cycle_error=reason,
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
        _emit_failure("module_path_missing")
        if not hasattr(manager, "run_post_patch_cycle"):
            return manager
        raise RuntimeError("module path unavailable for internalization")
    if provenance_token is None:
        _emit_failure("missing_provenance")
        raise PermissionError("missing provenance token for post patch validation")
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
    return manager


__all__ = [
    "SelfCodingManager",
    "PatchApprovalPolicy",
    "HelperGenerationError",
    "internalize_coding_bot",
]
