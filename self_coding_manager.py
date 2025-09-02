from __future__ import annotations

"""Manage self-coding patches and deployment cycles."""

from pathlib import Path
import logging
import subprocess
import tempfile
import threading
import time
import re
from typing import Dict, Any, TYPE_CHECKING

from .error_parser import FailureCache, ErrorReport, ErrorParser
from .failure_fingerprint_store import (
    FailureFingerprint,
    FailureFingerprintStore,
)
from .vector_utils import cosine_similarity
try:  # pragma: no cover - optional dependency
    from vector_service.context_builder import record_failed_tags, load_failed_tags
except Exception:  # pragma: no cover - optional dependency

    def record_failed_tags(_tags: list[str]) -> None:  # type: ignore
        return None

    def load_failed_tags() -> set[str]:  # type: ignore
        return set()

from .sandbox_runner.test_harness import run_tests, TestHarnessResult

from .self_coding_engine import SelfCodingEngine
from .model_automation_pipeline import ModelAutomationPipeline, AutomationResult
from .data_bot import DataBot
from .advanced_error_management import FormalVerifier, AutomatedRollbackManager
from . import mutation_logger as MutationLogger
from .rollback_manager import RollbackManager
from .self_improvement.baseline_tracker import BaselineTracker
from .sandbox_settings import SandboxSettings

try:  # pragma: no cover - allow package/flat imports
    from .patch_suggestion_db import PatchSuggestionDB
except Exception:  # pragma: no cover - fallback for flat layout
    from patch_suggestion_db import PatchSuggestionDB  # type: ignore

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
                    self.bot_name, "patch_checks", str(path)
                )
            except Exception as exc:  # pragma: no cover - audit logging issues
                self.logger.exception("failed to log healing action: %s", exc)
        return ok


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
    ) -> None:
        self.engine = self_coding_engine
        self.pipeline = pipeline
        self.bot_name = bot_name
        self.data_bot = data_bot
        self.approval_policy = approval_policy
        self.logger = logging.getLogger(self.__class__.__name__)
        self._last_patch_id: int | None = None
        self._last_event_id: int | None = None
        self._failure_cache = FailureCache()
        self.suggestion_db = suggestion_db or getattr(self.engine, "patch_suggestion_db", None)
        self.enhancement_classifier = (
            enhancement_classifier or getattr(self.engine, "enhancement_classifier", None)
        )
        self.failure_store = failure_store
        self.skip_similarity = skip_similarity
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
    def run_patch(
        self,
        path: Path,
        description: str,
        energy: int = 1,
        *,
        context_meta: Dict[str, Any] | None = None,
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
        ``"docker"`` runs tests inside a Docker container.
        """
        if self.approval_policy and not self.approval_policy.approve(path):
            raise RuntimeError("patch approval failed")
        before_roi = self.data_bot.roi(self.bot_name) if self.data_bot else 0.0
        repo_root = Path.cwd().resolve()
        result: AutomationResult | None = None
        after_roi = before_roi
        roi_delta = 0.0
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run(["git", "clone", str(repo_root), tmp], check=True)
            clone_root = Path(tmp)
            cloned_path = clone_root / path.resolve().relative_to(repo_root)
            attempt = 0
            patch_id: int | None = None
            reverted = False
            ctx_meta = context_meta
            builder = getattr(
                getattr(self.engine, "cognition_layer", None),
                "context_builder",
                None,
            )
            desc = description
            last_fp: FailureFingerprint | None = None

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

            def _run(repo: Path, changed: Path | None) -> TestHarnessResult:
                try:
                    return run_tests(repo, changed, backend=backend)
                except TypeError:
                    return run_tests(repo, changed)

            baseline = _run(clone_root, cloned_path)
            coverage_before = _coverage_ratio(baseline.stdout, baseline.success)
            runtime_before = baseline.duration
            coverage_after = coverage_before
            runtime_after = runtime_before

            while attempt < max_attempts:
                attempt += 1
                self.logger.info("patch attempt %s", attempt)
                if last_fp and self.failure_store:
                    try:
                        matches = self.failure_store.find_similar(last_fp)
                    except Exception:
                        matches = []
                    best = 0.0
                    for m in matches:
                        try:
                            sim = cosine_similarity(last_fp.embedding, m.embedding)
                        except Exception:
                            sim = 0.0
                        if sim > best:
                            best = sim
                    threshold = getattr(
                        self.engine, "failure_similarity_threshold", self.skip_similarity or 0.95
                    )
                    if self.skip_similarity is not None:
                        threshold = max(threshold, self.skip_similarity)
                    if best >= threshold:
                        self.logger.info(
                            "failure fingerprint decision",
                            extra={"action": "skip", "similarity": best},
                        )
                        raise RuntimeError("similar failure detected")
                    warning = f"avoid repeating failure: {last_fp.error_message}"
                    desc = f"{desc}; {warning}"
                    self.logger.info(
                        "failure fingerprint decision",
                        extra={"action": "warning", "similarity": best},
                    )
                patch_id, reverted, _ = self.engine.apply_patch(
                    cloned_path,
                    desc,
                    parent_patch_id=self._last_patch_id,
                    reason=desc,
                    trigger=path.name,
                    context_meta=ctx_meta,
                    baseline_coverage=coverage_before,
                    baseline_runtime=runtime_before,
                )
                harness_result: TestHarnessResult = _run(clone_root, cloned_path)
                if harness_result.success:
                    coverage_after = _coverage_ratio(
                        harness_result.stdout, harness_result.success
                    )
                    runtime_after = harness_result.duration
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
                function_name = ""
                error_msg = ""
                m = re.findall(r'File "[^"]+", line \d+, in ([^\n]+)', stack_trace)
                if m:
                    function_name = m[-1]
                m_err = re.findall(r'([\w.]+(?:Error|Exception):.*)', stack_trace)
                if m_err:
                    error_msg = m_err[-1]
                fingerprint = FailureFingerprint.from_failure(
                    path.name,
                    function_name,
                    stack_trace,
                    error_msg,
                    self.engine.last_prompt_text,
                )
                last_fp = fingerprint
                if self.failure_store:
                    try:
                        self.failure_store.add(fingerprint)
                    except Exception:  # pragma: no cover - best effort
                        self.logger.exception("failed to log failure fingerprint")
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
                    }
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
                    trigger=path.name,
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
                        trigger=path.name,
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
            trigger=path.name,
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
                    trigger=path.name,
                    parent_event_id=self._last_event_id,
                )
            except Exception as exc:
                self.logger.exception(
                    "failed to log evolution cycle: %s", exc
                )
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
            path = Path(module)
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
                                "module": module,
                                "score": round(score, 2),
                                "rationale": rationale,
                            }
                        )
                    except Exception:  # pragma: no cover - best effort
                        self.logger.exception("failed to record audit log for %s", module)
                self.run_patch(path, description)
            except Exception:  # pragma: no cover - best effort
                self.logger.exception("failed to apply suggestion for %s", module)
            finally:
                try:
                    self.suggestion_db.conn.execute(
                        "DELETE FROM suggestions WHERE id=?", (sid,)
                    )
                    self.suggestion_db.conn.commit()
                except Exception:  # pragma: no cover - best effort
                    self.logger.exception("failed to delete suggestion %s", sid)


__all__ = ["SelfCodingManager", "PatchApprovalPolicy"]
