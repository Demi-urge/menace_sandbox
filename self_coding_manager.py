from __future__ import annotations

"""Manage self-coding patches and deployment cycles."""

from pathlib import Path
try:  # pragma: no cover - allow flat imports
    from .dynamic_path_router import resolve_path, path_for_prompt
except Exception:  # pragma: no cover - fallback for flat layout
    from dynamic_path_router import resolve_path, path_for_prompt  # type: ignore
import logging
import subprocess
import tempfile
import threading
import time
import re
import json
import uuid
import os
import shlex
from dataclasses import asdict
from typing import Dict, Any, TYPE_CHECKING

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
from .model_automation_pipeline import ModelAutomationPipeline, AutomationResult
from .data_bot import DataBot
from .error_bot import ErrorDB
from .advanced_error_management import FormalVerifier, AutomatedRollbackManager
from . import mutation_logger as MutationLogger
from .rollback_manager import RollbackManager
from .self_improvement.baseline_tracker import BaselineTracker
from .self_improvement.target_region import TargetRegion
from .sandbox_settings import SandboxSettings
from .patch_attempt_tracker import PatchAttemptTracker
from .self_coding_thresholds import get_thresholds as load_sc_thresholds

try:  # pragma: no cover - optional dependency
    from .quick_fix_engine import QuickFixEngine, generate_patch
except Exception:  # pragma: no cover - optional dependency
    QuickFixEngine = None  # type: ignore
    generate_patch = None  # type: ignore

from context_builder_util import ensure_fresh_weights, create_context_builder

try:  # pragma: no cover - allow flat and package imports
    from .coding_bot_interface import manager_generate_helper as _BASE_MANAGER_GENERATE_HELPER
except Exception:  # pragma: no cover - fallback for flat layout
    from coding_bot_interface import (
        manager_generate_helper as _BASE_MANAGER_GENERATE_HELPER,  # type: ignore
    )

try:  # pragma: no cover - optional dependency for patching helper generation
    from . import quick_fix_engine as _quick_fix_engine_mod
except Exception:  # pragma: no cover - fallback for flat layout
    import quick_fix_engine as _quick_fix_engine_mod  # type: ignore


def _manager_generate_helper_with_builder(manager, description: str, **kwargs: Any) -> str:
    """Create a fresh ``ContextBuilder`` and invoke the base helper generator."""

    builder = create_context_builder()
    ensure_fresh_weights(builder)
    kwargs.setdefault("context_builder", builder)
    try:
        return _BASE_MANAGER_GENERATE_HELPER(manager, description, **kwargs)
    except TypeError:
        kwargs.pop("context_builder", None)
        return _BASE_MANAGER_GENERATE_HELPER(manager, description, **kwargs)


if _quick_fix_engine_mod is not None:  # pragma: no cover - best effort patching
    _quick_fix_engine_mod.manager_generate_helper = _manager_generate_helper_with_builder

try:  # pragma: no cover - allow package/flat imports
    from .patch_suggestion_db import PatchSuggestionDB
except Exception:  # pragma: no cover - fallback for flat layout
    from patch_suggestion_db import PatchSuggestionDB  # type: ignore

try:  # pragma: no cover - allow package/flat imports
    from .bot_registry import BotRegistry
except Exception:  # pragma: no cover - fallback for flat layout
    from bot_registry import BotRegistry  # type: ignore

try:  # pragma: no cover - allow package/flat imports
    from .patch_provenance import record_patch_metadata
except Exception:  # pragma: no cover - fallback for flat layout
    from patch_provenance import record_patch_metadata  # type: ignore

try:  # pragma: no cover - optional dependency
    from .unified_event_bus import UnifiedEventBus
except Exception:  # pragma: no cover - fallback for flat layout
    from unified_event_bus import UnifiedEventBus  # type: ignore

try:  # pragma: no cover - allow package/flat imports
    from .code_database import PatchRecord
except Exception:  # pragma: no cover - fallback for flat layout
    from code_database import PatchRecord  # type: ignore

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from .enhancement_classifier import EnhancementClassifier
    from .evolution_orchestrator import EvolutionOrchestrator


class PatchApprovalPolicy:
    """Run formal verification and tests before patching.

    The test runner command can be customised via ``test_command``.
    """

    def __init__(
        self,
        *,
        verifier: FormalVerifier | None = None,
        rollback_mgr: AutomatedRollbackManager | None = None,
        bot_name: str = "menace",
        test_command: list[str] | None = None,
    ) -> None:
        self.verifier = verifier or FormalVerifier()
        self.rollback_mgr = rollback_mgr
        self.bot_name = bot_name
        self.logger = logging.getLogger(self.__class__.__name__)
        if test_command is None:
            env_cmd = os.getenv("SELF_CODING_TEST_COMMAND")
            if env_cmd:
                test_command = shlex.split(env_cmd)
            else:
                try:
                    test_command = load_sc_thresholds(
                        bot_name, SandboxSettings()
                    ).test_command
                except Exception:
                    test_command = None
        self.test_command = test_command or ["pytest", "-q"]

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
    :class:`ValueError` is raised otherwise.
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
        roi_drop_threshold: float | None = None,
        error_rate_threshold: float | None = None,
    ) -> None:
        if data_bot is None or bot_registry is None:
            raise ValueError("data_bot and bot_registry are required")
        self.engine = self_coding_engine
        self.pipeline = pipeline
        self.bot_name = bot_name
        self.data_bot = data_bot
        self.approval_policy = approval_policy
        self.logger = logging.getLogger(self.__class__.__name__)
        self._last_patch_id: int | None = None
        self._last_event_id: int | None = None
        self._last_commit_hash: str | None = None
        thresholds = data_bot.get_thresholds(bot_name)
        self.roi_drop_threshold = (
            roi_drop_threshold if roi_drop_threshold is not None else thresholds.roi_drop
        )
        self.error_rate_threshold = (
            error_rate_threshold
            if error_rate_threshold is not None
            else thresholds.error_threshold
        )
        self.test_failure_threshold = thresholds.test_failure_threshold
        self._refresh_thresholds()
        self._failure_cache = FailureCache()
        self.suggestion_db = suggestion_db or getattr(self.engine, "patch_suggestion_db", None)
        self.enhancement_classifier = (
            enhancement_classifier or getattr(self.engine, "enhancement_classifier", None)
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
        self.baseline_tracker = BaselineTracker(
            window=int(baseline_window), metrics=["confidence"]
        )
        # ``_forecast_history`` stores predicted metrics so threshold updates
        # can adapt based on recent trends.
        self._forecast_history: Dict[str, list[float]] = {
            "roi": [],
            "errors": [],
            "tests_failed": [],
        }
        if enhancement_classifier and not getattr(self.engine, "enhancement_classifier", None):
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
        self.event_bus = event_bus
        self.evolution_orchestrator = evolution_orchestrator
        if self.bot_registry:
            try:
                self.bot_registry.register_bot(self.bot_name)
            except Exception:  # pragma: no cover - best effort
                self.logger.exception("failed to register bot in registry")

        clayer = getattr(self.engine, "cognition_layer", None)
        builder = getattr(clayer, "context_builder", None) if clayer else None
        if builder is None:
            raise RuntimeError(
                "engine.cognition_layer must provide a context_builder"
            )
        if QuickFixEngine is None and self.quick_fix is None:
            raise RuntimeError(
                "QuickFixEngine is required but could not be imported"
            )
        self._prepare_context_builder(builder)
        self._init_quick_fix_engine(builder)

        if self.evolution_orchestrator is None:
            try:  # pragma: no cover - optional dependencies
                from .capital_management_bot import CapitalManagementBot
                from .self_improvement.engine import SelfImprovementEngine
                from .system_evolution_manager import SystemEvolutionManager
                from .evolution_orchestrator import EvolutionOrchestrator

                if self.data_bot and self.bot_registry:
                    capital = CapitalManagementBot(data_bot=self.data_bot)
                    improv = SelfImprovementEngine(
                        data_bot=self.data_bot, bot_name=self.bot_name
                    )
                    bots = list(getattr(self.bot_registry, "graph", {}).keys())
                    evol_mgr = SystemEvolutionManager(bots)
                    self.evolution_orchestrator = EvolutionOrchestrator(
                        data_bot=self.data_bot,
                        capital_bot=capital,
                        improvement_engine=improv,
                        evolution_manager=evol_mgr,
                        selfcoding_manager=self,
                    )
            except Exception:  # pragma: no cover - best effort
                self.evolution_orchestrator = None

    def register_bot(self, name: str) -> None:
        """Register *name* with the underlying :class:`BotRegistry`."""
        if not self.bot_registry:
            return
        try:
            self.bot_registry.register_bot(name)
        except Exception:  # pragma: no cover - best effort
            self.logger.exception("failed to register bot in registry")

    def _refresh_thresholds(self) -> None:
        """Fetch ROI, error and test-failure thresholds from :class:`DataBot`."""
        if not self.data_bot:
            return
        try:
            # Ensure the DataBot refreshes its view of the configuration so
            # manager decisions use the most recent thresholds.
            t = self.data_bot.reload_thresholds(self.bot_name)
            self.roi_drop_threshold = t.roi_drop
            self.error_rate_threshold = t.error_threshold
            self.test_failure_threshold = t.test_failure_threshold
        except Exception:  # pragma: no cover - best effort
            self.logger.exception("failed to load thresholds for %s", self.bot_name)

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
            self.logger.error(
                "QuickFixEngine is required but could not be imported; "
                "pip install menace[quickfix]",
            )
            raise RuntimeError(
                "QuickFixEngine is required but could not be imported",
            )
        db = self.error_db or ErrorDB()
        self.error_db = db
        try:
            self.quick_fix = QuickFixEngine(db, self, context_builder=builder)
        except Exception as exc:  # pragma: no cover - instantiation errors
            raise RuntimeError(
                "failed to initialise QuickFixEngine",
            ) from exc

    def _ensure_quick_fix_engine(
        self, builder: ContextBuilder | None = None
    ) -> QuickFixEngine:
        """Return an initialised :class:`QuickFixEngine`.

        A fresh *builder* updates the engine's context. When no engine is
        present a new instance is created so patches always undergo
        validation.
        """

        if builder is None:
            clayer = getattr(self.engine, "cognition_layer", None)
            builder = getattr(clayer, "context_builder", None)

        if builder is None:
            builder = ContextBuilder()

        self._init_quick_fix_engine(builder)

        self._prepare_context_builder(builder)
        try:
            self.quick_fix.context_builder = builder
        except Exception:
            self.logger.exception(
                "failed to update QuickFixEngine context builder",
            )
            raise
        return self.quick_fix

    # ------------------------------------------------------------------
    def scan_repo(self) -> None:
        """Invoke the enhancement classifier and queue suggestions."""
        if not self.enhancement_classifier:
            return
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
                            suggestions, key=lambda s: getattr(s, "score", 0.0), reverse=True
                        )[:5]
                    ]
                    event_bus.publish(
                        "enhancement:suggestions",
                        {"count": len(suggestions), "top_scores": top_scores},
                    )
                except Exception:
                    self.logger.exception("failed to publish enhancement suggestions")
        except Exception:
            self.logger.exception("repo scan failed")

    def schedule_repo_scan(self, interval: float = 3600.0) -> None:
        """Run :meth:`scan_repo` on a background scheduler."""
        if not self.enhancement_classifier:
            return

        def _loop() -> None:
            while True:
                time.sleep(interval)
                try:
                    self.scan_repo()
                    db = self.suggestion_db or getattr(self.engine, "patch_suggestion_db", None)
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

        result = self.data_bot.check_degradation(
            self.bot_name, roi, errors, failures
        )

        # ``check_degradation`` adapts thresholds based on the latest metrics;
        # refresh the local cache so subsequent decisions reflect the new
        # values.
        self._refresh_thresholds()
        return result

    # ------------------------------------------------------------------
    def register_patch_cycle(
        self,
        description: str,
        context_meta: Dict[str, Any] | None = None,
    ) -> None:
        """Log baseline metrics for an upcoming patch cycle.

        The baseline ROI and error rates for ``bot_name`` are stored in
        :class:`PatchHistoryDB` and a ``self_coding:cycle_registered`` event is
        emitted on the configured event bus.  The generated record and event
        identifiers are stored for linking with subsequent patch events.
        """

        roi = self.data_bot.roi(self.bot_name) if self.data_bot else 0.0
        errors = (
            self.data_bot.average_errors(self.bot_name) if self.data_bot else 0.0
        )
        patch_db = getattr(self.engine, "patch_db", None)
        patch_id: int | None = None
        if patch_db:
            try:
                rec = PatchRecord(
                    filename=f"{self.bot_name}.cycle",
                    description=description,
                    roi_before=roi,
                    roi_after=roi,
                    errors_before=int(errors),
                    errors_after=int(errors),
                    source_bot=self.bot_name,
                    reason=context_meta.get("reason") if context_meta else None,
                    trigger=context_meta.get("trigger") if context_meta else None,
                )
                patch_id = patch_db.add(rec)
                self._last_patch_id = patch_id
            except Exception:
                self.logger.exception("failed to log patch cycle to DB")
        event_id: int | None = None
        try:
            trigger = (
                context_meta.get("trigger") if context_meta else "degradation"
            )
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
                    "description": description,
                }
                if context_meta:
                    payload.update(context_meta)
                self.event_bus.publish("self_coding:cycle_registered", payload)
            except Exception:
                self.logger.exception("failed to publish cycle_registered event")

    # ------------------------------------------------------------------
    def generate_and_patch(
        self,
        path: Path,
        description: str,
        *,
        context_meta: Dict[str, Any] | None = None,
        context_builder: ContextBuilder | None = None,
        **kwargs: Any,
    ) -> tuple[AutomationResult, str | None]:
        """Patch ``path`` using :meth:`run_patch` with fresh context.

        Returns a tuple of the :class:`AutomationResult` and the commit hash of
        the applied patch, if any.
        """
        builder = context_builder or ContextBuilder()
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
        except Exception as exc:
            if self.event_bus:
                try:
                    self.event_bus.publish(
                        "bot:patch_failed",
                        {"bot": self.bot_name, "reason": str(exc)},
                    )
                except Exception:  # pragma: no cover - best effort
                    self.logger.exception("failed to publish patch_failed event")
            raise RuntimeError("QuickFixEngine validation unavailable") from exc
        result = self.run_patch(
            path,
            description,
            context_meta=context_meta,
            context_builder=builder,
            **kwargs,
        )
        return result, getattr(self, "_last_commit_hash", None)

    # ------------------------------------------------------------------
    def run_patch(
        self,
        path: Path,
        description: str,
        energy: int = 1,
        *,
        context_meta: Dict[str, Any] | None = None,
        context_builder: ContextBuilder | None = None,
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
        customises the VCS command used to clone the repository. When
        ``context_builder`` is supplied the engine and quick fix components use
        it for validation.
        """
        if self.approval_policy and not self.approval_policy.approve(path):
            raise RuntimeError("patch approval failed")
        if self.data_bot:
            self._refresh_thresholds()
        roi = self.data_bot.roi(self.bot_name) if self.data_bot else 0.0
        errors = (
            self.data_bot.average_errors(self.bot_name) if self.data_bot else 0.0
        )
        failures = (
            self.data_bot.average_test_failures(self.bot_name) if self.data_bot else 0.0
        )
        if self.data_bot and not self.data_bot.check_degradation(
            self.bot_name, roi, errors, failures
        ):
            self.logger.info(
                "ROI and error thresholds not met; skipping patch"
            )
            return AutomationResult(None, None)
        before_roi = roi
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
            if context_builder is not None:
                clayer.context_builder = context_builder
                try:
                    self.engine.context_builder = context_builder
                except Exception:
                    self.logger.exception(
                        "failed to refresh engine context builder",
                    )
            builder = getattr(clayer, "context_builder", None)
            if builder is None:
                raise AttributeError(
                    "engine.cognition_layer must provide a context_builder",
                )
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
                raise RuntimeError(
                    "QuickFixEngine validation unavailable"
                ) from exc
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
                                    self.logger.exception("failed to record retry status")
                            raise RuntimeError("similar failure detected")
                if last_fp and self.failure_store:
                    threshold = getattr(self.engine, "failure_similarity_threshold", None)
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
                    func_region = func_region or TargetRegion(
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
                    return AutomationResult(None, None)
                if self.quick_fix is None:
                    raise RuntimeError("QuickFixEngine validation unavailable")
                try:
                    patch_id, flags = generate_patch(
                        module_path,
                        self.engine,
                        self,
                        context_builder=builder,
                        description=desc,
                        context=ctx_meta,
                        return_flags=True,
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
                    try:
                        RollbackManager().rollback(
                            str(patch_id or ""), requesting_bot=self.bot_name
                        )
                    except Exception:  # pragma: no cover - best effort
                        self.logger.exception("rollback failed")
                    raise RuntimeError("quick fix generation failed") from exc
                flags = list(flags or [])
                self._last_risk_flags = flags
                ctx_meta["risk_flags"] = flags
                passed = not flags
                reverted = not passed
                if self.data_bot:
                    try:
                        self.data_bot.record_validation(
                            self.bot_name, module_name, passed, flags or None
                        )
                    except Exception:
                        self.logger.exception("failed to record validation in DataBot")
                if self.bot_registry:
                    try:
                        self.bot_registry.record_validation(self.bot_name, module_name, passed)
                    except Exception:
                        self.logger.exception("failed to record validation in registry")
                if not passed:
                    if self.event_bus:
                        try:
                            self.event_bus.publish(
                                "bot:patch_failed",
                                {
                                    "bot": self.bot_name,
                                    "stage": "risk_validation",
                                    "flags": flags,
                                },
                            )
                        except Exception:  # pragma: no cover - best effort
                            self.logger.exception(
                                "failed to publish patch_failed event",
                            )
                    try:
                        RollbackManager().rollback(
                            str(patch_id or ""), requesting_bot=self.bot_name
                        )
                    except Exception:  # pragma: no cover - best effort
                        self.logger.exception("rollback failed")
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
                        target_region = TargetRegion(
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
                m_err = re.findall(r'([\w.]+(?:Error|Exception):.*)', stack_trace)
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
                    func_region = TargetRegion(
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
                    pass
                try:
                    commit_hash = (
                        subprocess.check_output(
                            ["git", "rev-parse", "HEAD"], cwd=str(clone_root)
                        )
                        .decode()
                        .strip()
                    )
                except Exception:
                    commit_hash = None
            except Exception as exc:  # pragma: no cover - best effort
                self.logger.error("git commit failed: %s", exc)
                try:
                    RollbackManager().rollback(str(patch_id), requesting_bot=self.bot_name)
                except Exception:
                    self.logger.exception("rollback failed")
                raise

            result = self.pipeline.run(self.bot_name, energy=energy)
            after_roi = self.data_bot.roi(self.bot_name) if self.data_bot else 0.0
            roi_delta = after_roi - before_roi
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
                    RollbackManager().rollback(str(patch_id), requesting_bot=self.bot_name)
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
                        RollbackManager().rollback(str(patch_id), requesting_bot=self.bot_name)
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
                self.logger.exception(
                    "failed to log evolution cycle: %s", exc
                )
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
                    resources=(
                        f"hot_swap:{int(time.time())},patch_id:{patch_id}"
                    ),
                )
                self.bot_registry.register_bot(self.bot_name)
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
                            {"bot": self.bot_name, "path": module_path, "patch_id": patch_id, "commit": commit_hash, "reason": "missing_provenance"},
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
                try:
                    self.bot_registry.hot_swap_bot(self.bot_name)
                    self.bot_registry.health_check_bot(self.bot_name, prev_state)
                except Exception:  # pragma: no cover - best effort
                    self.logger.exception(
                        "failed to hot swap bot",
                        extra={"bot": self.bot_name, "module_path": module_path},
                    )
                    raise
                if self.event_bus:
                    try:
                        self.event_bus.publish(
                            "bot:updated",
                            {"bot": self.bot_name, "path": module_path, "patch_id": patch_id, "commit": commit_hash},
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
                            "failed to persist bot registry", extra={"path": str(target)}
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
                }
                self.event_bus.publish("self_coding:patch_applied", payload)
            except Exception:  # pragma: no cover - best effort
                self.logger.exception("failed to publish patch_applied event")
        self.scan_repo()
        try:
            load_failed_tags()
        except Exception:  # pragma: no cover - best effort
            self.logger.exception("failed to refresh failed tags")
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
                        self.logger.exception("failed to record audit log for %s", prompt_module)
                self.run_patch(path, description)
            except Exception:  # pragma: no cover - best effort
                self.logger.exception("failed to apply suggestion for %s", prompt_module)
            finally:
                try:
                    self.suggestion_db.conn.execute(
                        "DELETE FROM suggestions WHERE id=?", (sid,)
                    )
                    self.suggestion_db.conn.commit()
                except Exception:  # pragma: no cover - best effort
                    self.logger.exception("failed to delete suggestion %s", sid)


__all__ = ["SelfCodingManager", "PatchApprovalPolicy", "HelperGenerationError"]
