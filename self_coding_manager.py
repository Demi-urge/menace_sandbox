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
from dataclasses import asdict
from typing import Dict, Any, TYPE_CHECKING

from .error_parser import FailureCache, ErrorReport, ErrorParser
from .failure_fingerprint_store import (
    FailureFingerprint,
    FailureFingerprintStore,
)
from .failure_retry_utils import check_similarity_and_warn, record_failure
try:  # pragma: no cover - optional dependency
    from vector_service.context_builder import (
        record_failed_tags,
        load_failed_tags,
        ContextBuilder,
    )
except Exception as exc:  # pragma: no cover - optional dependency  # noqa: F841

    _ctx_exc = exc

    def _ctx_builder_unavailable(*_a: object, **_k: object) -> None:
        raise RuntimeError(
            "vector_service.ContextBuilder is required but could not be imported"
        ) from _ctx_exc

    class ContextBuilder:  # type: ignore
        """Placeholder when vector service is unavailable."""

        def __init__(self, *a: object, **k: object) -> None:  # noqa: D401 - simple
            _ctx_builder_unavailable()

    def record_failed_tags(_tags: list[str]) -> None:  # type: ignore
        _ctx_builder_unavailable()

    def load_failed_tags() -> set[str]:  # type: ignore
        _ctx_builder_unavailable()

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
from .self_coding_thresholds import get_thresholds
from .patch_attempt_tracker import PatchAttemptTracker

try:  # pragma: no cover - optional dependency
    from . import quick_fix_engine
    from .quick_fix_engine import QuickFixEngine
except Exception as exc:  # pragma: no cover - optional dependency
    quick_fix_engine = None  # type: ignore

    class QuickFixEngine:  # type: ignore
        """Placeholder when :mod:`quick_fix_engine` is unavailable."""

        def __init__(
            self, *a: object, _exc: Exception = exc, **k: object
        ) -> None:  # noqa: D401
            raise RuntimeError(
                "QuickFixEngine is required but could not be imported"
            ) from _exc

from context_builder_util import ensure_fresh_weights

try:  # pragma: no cover - allow package/flat imports
    from .patch_suggestion_db import PatchSuggestionDB
except Exception:  # pragma: no cover - fallback for flat layout
    from patch_suggestion_db import PatchSuggestionDB  # type: ignore

try:  # pragma: no cover - allow package/flat imports
    from .bot_registry import BotRegistry
except Exception:  # pragma: no cover - fallback for flat layout
    from bot_registry import BotRegistry  # type: ignore

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


class PatchApprovalPolicy:
    """Run formal verification and tests before patching."""

    def __init__(
        self,
        *,
        verifier: FormalVerifier | None = None,
        rollback_mgr: AutomatedRollbackManager | None = None,
        bot_name: str = "menace",
    ) -> None:
        self.verifier = verifier or FormalVerifier()
        self.rollback_mgr = rollback_mgr
        self.bot_name = bot_name
        self.logger = logging.getLogger(self.__class__.__name__)

    def approve(self, path: Path) -> bool:
        ok = True
        try:
            if self.verifier and not self.verifier.verify(path):
                ok = False
        except Exception as exc:  # pragma: no cover - verification issues
            self.logger.error("verification failed: %s", exc)
            ok = False
        try:
            subprocess.run(["pytest", "-q"], check=True)
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
    """Apply code patches and redeploy bots."""

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
        event_bus: UnifiedEventBus | None = None,
        quick_fix_engine: QuickFixEngine | None = None,
        roi_drop_threshold: float | None = None,
        error_rate_threshold: float | None = None,
    ) -> None:
        self.engine = self_coding_engine
        self.pipeline = pipeline
        self.bot_name = bot_name
        self.data_bot = data_bot
        self.approval_policy = approval_policy
        self.logger = logging.getLogger(self.__class__.__name__)
        self._last_patch_id: int | None = None
        self._last_event_id: int | None = None
        thresholds = get_thresholds(bot_name)
        self.roi_drop_threshold = (
            roi_drop_threshold
            if roi_drop_threshold is not None
            else thresholds.roi_drop
        )
        self.error_rate_threshold = (
            error_rate_threshold
            if error_rate_threshold is not None
            else thresholds.error_increase
        )
        self.test_failure_threshold = thresholds.test_failure_increase
        self._refresh_thresholds()
        self._last_roi = self.data_bot.roi(self.bot_name) if self.data_bot else 0.0
        self._last_errors = (
            self.data_bot.average_errors(self.bot_name) - self.error_rate_threshold
            if self.data_bot
            else 0.0
        )
        self._last_test_failures = (
            self.data_bot.average_test_failures(self.bot_name)
            if self.data_bot and hasattr(self.data_bot, "average_test_failures")
            else 0.0
        )
        self._failure_cache = FailureCache()
        self.suggestion_db = suggestion_db or getattr(self.engine, "patch_suggestion_db", None)
        self.enhancement_classifier = (
            enhancement_classifier or getattr(self.engine, "enhancement_classifier", None)
        )
        self.failure_store = failure_store
        self.skip_similarity = skip_similarity
        self.quick_fix = quick_fix or quick_fix_engine
        if baseline_window is None:
            try:
                baseline_window = getattr(SandboxSettings(), "baseline_window", 5)
            except Exception:
                baseline_window = 5
        self.baseline_tracker = BaselineTracker(
            window=int(baseline_window), metrics=["confidence"]
        )
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
        if self.bot_registry:
            try:
                self.bot_registry.register_bot(self.bot_name)
            except Exception:  # pragma: no cover - best effort
                self.logger.exception("failed to register bot in registry")

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
            t = self.data_bot.get_thresholds(self.bot_name)
            self.roi_drop_threshold = t.roi_drop
            self.error_rate_threshold = t.error_threshold
            self.test_failure_threshold = t.test_failure_threshold
        except Exception:  # pragma: no cover - best effort
            self.logger.exception("failed to load thresholds for %s", self.bot_name)

    def _ensure_quick_fix_engine(self) -> QuickFixEngine | None:
        """Initialise :class:`QuickFixEngine` on demand.

        Raises
        ------
        RuntimeError
            If the optional ``quick_fix_engine`` dependency is missing or
            initialisation fails.
        """

        if self.quick_fix is not None:
            return self.quick_fix
        if quick_fix_engine is None:
            raise RuntimeError(
                "QuickFixEngine is required but could not be imported"
            )
        clayer = getattr(self.engine, "cognition_layer", None)
        builder = getattr(clayer, "context_builder", None)
        if builder is None:
            raise RuntimeError(
                "engine.cognition_layer must provide a context_builder"
            )
        ensure_fresh_weights(builder)
        self.quick_fix = QuickFixEngine(ErrorDB(), self, context_builder=builder)
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
        failures = (
            self.data_bot.average_test_failures(self.bot_name)
            if hasattr(self.data_bot, "average_test_failures")
            else 0.0
        )
        delta_roi = roi - self._last_roi
        delta_err = errors - self._last_errors
        delta_fail = failures - self._last_test_failures
        self._last_roi = roi
        self._last_errors = errors
        self._last_test_failures = failures
        return (
            delta_roi <= self.roi_drop_threshold
            or delta_err >= self.error_rate_threshold
            or delta_fail >= self.test_failure_threshold
        )

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
    ) -> AutomationResult:
        """Generate a helper then patch ``path`` using :meth:`run_patch`.

        ``context_builder`` is applied to the underlying engine and cognition
        layer before generation to ensure fresh context.  Any additional
        keyword arguments are forwarded to :meth:`run_patch`.
        """

        builder = context_builder or ContextBuilder()
        engine = self.engine
        try:
            engine.context_builder = builder
            clayer = getattr(engine, "cognition_layer", None)
            if clayer is not None:
                clayer.context_builder = builder
        except Exception:
            self.logger.error(
                "context_builder_refresh_failed",
                exc_info=True,
                extra={"path": str(path)},
            )
        try:
            engine.generate_helper(
                description,
                path=path,
                metadata=context_meta,
                strategy=None,
                target_region=None,
            )
        except Exception as exc:  # pragma: no cover - generation failure
            raise HelperGenerationError(str(exc)) from exc

        return self.run_patch(
            path,
            description,
            context_meta=context_meta,
            context_builder=builder,
            **kwargs,
        )

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
    ) -> AutomationResult:
        """Patch *path* then deploy using the automation pipeline.

        ``max_attempts`` controls how many times the patch is retried when tests
        fail.  Context will be rebuilt for each retry excluding tags extracted
        from the failing traceback.  After a successful patch the change is
        committed in a sandbox clone, pushed to ``review_branch`` and merged
        into ``main`` when ``auto_merge`` is ``True`` and the confidence score
        exceeds ``confidence_threshold``.  ``backend`` selects the test
        execution environment; ``"venv"`` uses a virtual environment while
        ``"docker"`` runs tests inside a Docker container.  When
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
        if self.data_bot:
            delta_roi = roi - self._last_roi
            delta_err = errors - self._last_errors
            if delta_roi > self.roi_drop_threshold and delta_err < self.error_rate_threshold:
                self.logger.info(
                    "ROI and error thresholds not met; skipping patch"
                )
                self._last_roi = roi
                self._last_errors = errors
                return AutomationResult(None, None)
        before_roi = roi
        repo_root = Path.cwd().resolve()
        result: AutomationResult | None = None
        after_roi = before_roi
        roi_delta = 0.0
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run(["git", "clone", str(repo_root), tmp], check=True)
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
            self._ensure_quick_fix_engine()
            if self.quick_fix is None:
                raise RuntimeError("QuickFixEngine validation unavailable")
            try:
                self.quick_fix.context_builder = builder
            except Exception:
                self.logger.exception(
                    "failed to update QuickFixEngine context builder",
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
                passed, patch_id = self.quick_fix.apply_validated_patch(
                    module_path,
                    desc,
                    ctx_meta,
                )
                reverted = not passed
                if self.data_bot:
                    try:
                        self.data_bot.record_validation(
                            self.bot_name, module_name, passed, None
                        )
                    except Exception:
                        self.logger.exception("failed to record validation in DataBot")
                if self.bot_registry:
                    try:
                        self.bot_registry.record_validation(self.bot_name, module_name, passed)
                    except Exception:
                        self.logger.exception("failed to record validation in registry")
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
                    ctx, sid = builder.query(desc, exclude_tags=tags)
                    ctx_meta = {
                        "retrieval_context": ctx,
                        "retrieval_session_id": sid,
                        "escalation_level": level,
                    }
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
                subprocess.run(
                    ["git", "commit", "-m", f"patch {patch_id}: {description}"],
                    check=True,
                    cwd=str(clone_root),
                )
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
            if patch_logger is not None:
                try:
                    patch_logger.track_contributors(
                        {},
                        True,
                        patch_id=str(patch_id or ""),
                        contribution=roi_delta,
                    )
                except Exception:
                    self.logger.exception("track_contributors failed")
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
            module_path = path_for_prompt(cloned_path)
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
                        "module": module_path,
                        "version": version,
                    },
                )
            except Exception:  # pragma: no cover - best effort
                self.logger.exception(
                    "failed to update bot registry",
                    extra={"bot": self.bot_name, "module": module_path},
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
        if self.data_bot:
            self._last_roi = self.data_bot.roi(self.bot_name)
            self._last_errors = self.data_bot.average_errors(self.bot_name)
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
