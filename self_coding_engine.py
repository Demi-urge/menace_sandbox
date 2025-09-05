"""Self-coding engine that retrieves code snippets and proposes patches."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Dict, List, Any, Tuple, Mapping
import subprocess
import json
import base64
import logging
import ast
import tempfile
import py_compile
import re
import traceback
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

from .code_database import CodeDB, CodeRecord, PatchHistoryDB, PatchRecord
from .unified_event_bus import UnifiedEventBus
from .trend_predictor import TrendPredictor
try:  # pragma: no cover - allow flat imports
    from .dynamic_path_router import resolve_path, path_for_prompt
except Exception:  # pragma: no cover - fallback for flat layout
    from dynamic_path_router import resolve_path, path_for_prompt  # type: ignore
from gpt_memory_interface import GPTMemoryInterface
from .safety_monitor import SafetyMonitor
try:  # pragma: no cover - optional formal verification dependency
    from .advanced_error_management import FormalVerifier
except Exception:  # pragma: no cover - degrade gracefully when missing
    FormalVerifier = object  # type: ignore[misc,assignment]
from .llm_interface import Prompt, LLMResult, LLMClient
from .llm_router import client_from_settings
try:  # shared GPT memory instance
    from .shared_gpt_memory import GPT_MEMORY_MANAGER
except Exception:  # pragma: no cover - fallback for flat layout
    from shared_gpt_memory import GPT_MEMORY_MANAGER  # type: ignore
try:  # canonical tag constants
    from .log_tags import FEEDBACK, ERROR_FIX, IMPROVEMENT_PATH, INSIGHT
except Exception:  # pragma: no cover - fallback for flat layout
    from log_tags import FEEDBACK, ERROR_FIX, IMPROVEMENT_PATH, INSIGHT  # type: ignore
try:  # pragma: no cover - allow flat imports
    from .gpt_knowledge_service import GPTKnowledgeService
except Exception:  # pragma: no cover - fallback for flat layout
    from gpt_knowledge_service import GPTKnowledgeService  # type: ignore
try:  # pragma: no cover - allow flat imports
    from .knowledge_retriever import (
        get_feedback,
        get_error_fixes,
        recent_feedback,
        recent_error_fix,
        recent_improvement_path,
    )
except Exception:  # pragma: no cover - fallback for flat layout
    from knowledge_retriever import (  # type: ignore
        get_feedback,
        get_error_fixes,
        recent_feedback,
        recent_error_fix,
        recent_improvement_path,
    )
try:  # pragma: no cover - optional rollback support
    from .rollback_manager import RollbackManager
except Exception:  # pragma: no cover - degrade gracefully when missing
    RollbackManager = object  # type: ignore[misc,assignment]
from .audit_trail import AuditTrail
from .access_control import READ, WRITE, check_permission
from .patch_suggestion_db import PatchSuggestionDB, SuggestionRecord
from .patch_attempt_tracker import PatchAttemptTracker
from typing import TYPE_CHECKING
if TYPE_CHECKING:  # pragma: no cover - imported for type hints only
    from .enhancement_classifier import EnhancementClassifier, EnhancementSuggestion
try:  # pragma: no cover - optional dependency
    from .sandbox_runner.workflow_sandbox_runner import WorkflowSandboxRunner
except Exception:  # pragma: no cover - graceful degradation
    WorkflowSandboxRunner = object  # type: ignore[misc]
try:  # pragma: no cover - optional dependency
    from .sandbox_runner.test_harness import run_tests, TestHarnessResult
except Exception:  # pragma: no cover - graceful degradation
    run_tests = None  # type: ignore

    class TestHarnessResult:  # type: ignore[misc]
        success = False
        stdout = ""
from .sandbox_settings import SandboxSettings

try:  # pragma: no cover - optional dependency
    from vector_service import CognitionLayer, PatchLogger, VectorServiceError
except Exception:  # pragma: no cover - defensive fallback
    PatchLogger = object  # type: ignore

    class VectorServiceError(Exception):
        """Raised when the vector service dependency is missing."""

        def __init__(
            self,
            message: str,
            *,
            missing_dependency: str | None = None,
            suggested_fix: str | None = None,
        ) -> None:
            super().__init__(message)
            self.missing_dependency = missing_dependency
            self.suggested_fix = suggested_fix

    class CognitionLayer:  # type: ignore[override]
        """Stub cognition layer used when vector service is unavailable."""

        def __init__(self, *_, **kwargs):
            self.patch_logger = kwargs.get("patch_logger")
            self.roi_tracker = kwargs.get("roi_tracker")
            logging.getLogger(__name__).warning(
                "vector service dependency missing; CognitionLayer disabled"
            )

        def query(self, *_args, **_kwargs):
            raise VectorServiceError(
                "CognitionLayer unavailable",
                missing_dependency="vector_service",
                suggested_fix="install the vector_service package",
            )

        def record_patch_outcome(self, *_args, **_kwargs):
            raise VectorServiceError(
                "CognitionLayer unavailable",
                missing_dependency="vector_service",
                suggested_fix="install the vector_service package",
            )

try:  # pragma: no cover - optional ROI tracking
    from .roi_tracker import ROITracker
except Exception:  # pragma: no cover - degrade gracefully when missing
    ROITracker = object  # type: ignore[misc,assignment]
from .prompt_evolution_memory import PromptEvolutionMemory
try:  # pragma: no cover - optional dependency
    from .patch_provenance import record_patch_metadata
except Exception:  # pragma: no cover - graceful degradation
    def record_patch_metadata(*_a: Any, **_k: Any) -> None:  # type: ignore
        return None
from .prompt_engine import PromptEngine, _ENCODER, diff_within_target_region
from .prompt_memory_trainer import PromptMemoryTrainer
from chunking import split_into_chunks, get_chunk_summaries, summarize_code
try:
    from .prompt_optimizer import PromptOptimizer
except Exception:  # pragma: no cover - fallback for flat layout
    from prompt_optimizer import PromptOptimizer  # type: ignore
from .error_parser import ErrorParser, ErrorReport, parse_failure, FailureCache
try:
    from .target_region import TargetRegion
except Exception:  # pragma: no cover - fallback for direct execution
    from target_region import TargetRegion  # type: ignore
try:
    from .self_improvement.prompt_memory import log_prompt_attempt
except Exception:  # pragma: no cover - fallback for flat layout
    try:
        from self_improvement.prompt_memory import log_prompt_attempt  # type: ignore
    except Exception:  # pragma: no cover - final fallback
        def log_prompt_attempt(*_a: Any, **_k: Any) -> None:  # type: ignore
            return None
from .failure_fingerprint import FailureFingerprint, find_similar, log_fingerprint
from .failure_retry_utils import check_similarity_and_warn, record_failure
try:  # pragma: no cover - optional dependency for metrics
    from . import metrics_exporter as _me
except Exception:  # pragma: no cover - fallback when executed directly
    import metrics_exporter as _me  # type: ignore

_PATCH_ATTEMPTS = _me.Gauge(
    "patch_attempts_total",
    "Patch attempts by region scope",
    labelnames=["scope"],
)
_PATCH_ESCALATIONS = _me.Gauge(
    "patch_escalations_total",
    "Patch escalation events",
    labelnames=["level"],
)
from .self_improvement.baseline_tracker import (  # noqa: E402
    BaselineTracker,
    TRACKER as METRIC_BASELINES,
)
try:  # pragma: no cover - optional dependency
    from .self_improvement.init import FileLock, _atomic_write
except Exception:  # pragma: no cover - fallback for flat layout
    try:
        from self_improvement.init import FileLock, _atomic_write  # type: ignore
    except Exception:  # pragma: no cover - final fallback
        class FileLock:  # type: ignore[misc]
            def __init__(self, *_a, **_k) -> None:
                pass

            def __enter__(self) -> "FileLock":  # pragma: no cover - noop
                return self

            def __exit__(self, *_a) -> bool:  # pragma: no cover - noop
                return False

        def _atomic_write(path: str, data: str, *, mode: str = "w") -> None:  # type: ignore[misc]
            with open(path, mode, encoding="utf-8") as fh:
                fh.write(data)

if TYPE_CHECKING:  # pragma: no cover - type hints
    from .model_automation_pipeline import ModelAutomationPipeline
    from .data_bot import DataBot

# Load prompt configuration from settings instead of environment variables
_settings = SandboxSettings()
VA_PROMPT_TEMPLATE = getattr(_settings, "va_prompt_template", "")
VA_PROMPT_PREFIX = getattr(_settings, "va_prompt_prefix", "")
VA_REPO_LAYOUT_LINES = getattr(_settings, "va_repo_layout_lines", 200)

# Reuse prompt encoder for token counting if available


def _count_tokens(text: str) -> int:
    if _ENCODER is not None:
        try:
            return len(_ENCODER.encode(text))
        except Exception:
            pass
    return len(text.split())


class SelfCodingEngine:
    """Generate new helper code based on existing snippets."""

    def __init__(
        self,
        code_db: CodeDB,
        memory_mgr: GPTMemoryInterface,
        *,
        pipeline: Optional[ModelAutomationPipeline] = None,
        data_bot: Optional[DataBot] = None,
        patch_db: Optional[PatchHistoryDB] = None,
        trend_predictor: Optional[TrendPredictor] = None,
        bot_name: str = "menace",
        safety_monitor: Optional[SafetyMonitor] = None,
        llm_client: Optional[LLMClient] = None,
        rollback_mgr: Optional[RollbackManager] = None,
        formal_verifier: Optional[FormalVerifier] = None,
        patch_suggestion_db: "PatchSuggestionDB" | None = None,
        enhancement_classifier: "EnhancementClassifier" | None = None,
        patch_logger: PatchLogger | None = None,
        cognition_layer: CognitionLayer | None = None,
        bot_roles: Optional[Dict[str, str]] = None,
        audit_trail_path: str | None = None,
        audit_privkey: bytes | None = None,
        event_bus: UnifiedEventBus | None = None,
        gpt_memory: GPTMemoryInterface | None = GPT_MEMORY_MANAGER,
        knowledge_service: GPTKnowledgeService | None = None,
        prompt_memory: PromptMemoryTrainer | None = None,
        prompt_optimizer: PromptOptimizer | None = None,
        prompt_evolution_memory: PromptEvolutionMemory | None = None,
        prompt_tone: str = "neutral",
        token_threshold: int = 3500,
        prompt_chunk_token_threshold: int | None = None,
        chunk_summary_cache_dir: str | Path | None = None,
        prompt_chunk_cache_dir: str | Path | None = None,
        failure_similarity_threshold: float | None = None,
        failure_similarity_limit: int = 3,
        failure_similarity_k: float = 1.0,
        skip_retry_on_similarity: bool = False,
        baseline_window: int | None = None,
        delta_tracker: BaselineTracker | None = None,
        **kwargs: Any,
    ) -> None:
        self.code_db = code_db
        self.memory_mgr = memory_mgr
        self.gpt_memory = gpt_memory or GPT_MEMORY_MANAGER
        self.gpt_memory_manager = self.gpt_memory  # backward compatibility
        self.pipeline = pipeline
        self.data_bot = data_bot
        self.patch_db = patch_db
        self.trend_predictor = trend_predictor
        self.bot_name = bot_name
        if prompt_memory is not None:
            self.prompt_memory = prompt_memory
        else:
            try:
                self.prompt_memory = PromptMemoryTrainer()
            except Exception:
                self.prompt_memory = None
        self.prompt_tone = prompt_tone
        self.token_threshold = token_threshold
        self.chunk_token_threshold = (
            prompt_chunk_token_threshold
            if prompt_chunk_token_threshold is not None
            else _settings.prompt_chunk_token_threshold
        )
        # maintain backward compatibility
        self.prompt_chunk_token_threshold = self.chunk_token_threshold
        cache_dir = (
            chunk_summary_cache_dir
            or prompt_chunk_cache_dir
            or _settings.chunk_summary_cache_dir
        )
        self.chunk_summary_cache_dir = resolve_path(cache_dir)
        # backward compatibility
        self.prompt_chunk_cache_dir = self.chunk_summary_cache_dir
        self._failure_similarity_threshold = failure_similarity_threshold
        self.failure_similarity_limit = failure_similarity_limit
        self.failure_similarity_k = failure_similarity_k
        self.skip_retry_on_similarity = skip_retry_on_similarity
        if baseline_window is None:
            baseline_window = getattr(_settings, "baseline_window", 5)
        self.failure_similarity_tracker = BaselineTracker(
            window=int(baseline_window), metrics=["similarity"]
        )
        self.baseline_tracker = delta_tracker or METRIC_BASELINES
        data_dir = getattr(_settings, "sandbox_data_dir", ".")
        state_candidate = resolve_path(data_dir) / "self_coding_engine_state.json"
        try:
            self._state_path = resolve_path(state_candidate)
        except FileNotFoundError:
            self._state_path = state_candidate
        self.safety_monitor = safety_monitor
        if llm_client is None:
            try:
                llm_client = client_from_settings(_settings)
            except Exception:
                llm_client = None
        self.llm_client = llm_client
        if self.llm_client and getattr(self.llm_client, "gpt_memory", None) is not self.gpt_memory:
            try:
                self.llm_client.gpt_memory = self.gpt_memory  # type: ignore[attr-defined]
            except Exception:
                pass
        self.rollback_mgr = rollback_mgr
        if formal_verifier is None:
            try:
                formal_verifier = FormalVerifier()
            except Exception:  # pragma: no cover - optional dependency missing
                formal_verifier = None
        self.formal_verifier = formal_verifier
        self._active_patches: dict[
            str,
            tuple[
                Path,
                str,
                str,
                List[Tuple[str, str, float]],
                TargetRegion | None,
            ],
        ] = {}
        self.bot_roles: Dict[str, str] = bot_roles or {}
        path_setting = audit_trail_path or _settings.audit_log_path
        try:
            path = resolve_path(path_setting)
        except FileNotFoundError:
            path = Path(path_setting)
        key_b64 = audit_privkey or _settings.audit_privkey
        # Fallback to unsigned logging when no key is provided
        if key_b64:
            priv = base64.b64decode(key_b64) if isinstance(key_b64, str) else key_b64
        else:
            logging.getLogger(__name__).warning(
                "AUDIT_PRIVKEY not set; audit trail entries will not be signed"
            )
            priv = None
        self.audit_trail = AuditTrail(path, priv)
        self.logger = logging.getLogger("SelfCodingEngine")
        self._patch_tracker = PatchAttemptTracker(
            logger=self.logger, escalation_counter=_PATCH_ESCALATIONS
        )
        self.event_bus = event_bus
        self.patch_suggestion_db = patch_suggestion_db
        self.enhancement_classifier = enhancement_classifier
        tracker = ROITracker()
        if patch_logger is not None and getattr(patch_logger, "roi_tracker", None) is None:
            try:
                patch_logger.roi_tracker = tracker  # type: ignore[attr-defined]
            except Exception:
                self.logger.warning(
                    "failed to attach ROI tracker to patch_logger",
                    exc_info=True,
                    extra={"patch_logger": type(patch_logger).__name__},
                )
        if cognition_layer is None:
            try:
                cognition_layer = CognitionLayer(
                    patch_logger=patch_logger, roi_tracker=tracker
                )
            except VectorServiceError as exc:
                self.logger.warning(
                    "cognition layer unavailable during init: %s", exc
                )
                cognition_layer = None
            except Exception:
                cognition_layer = None
        else:
            if getattr(cognition_layer, "roi_tracker", None) is None:
                try:
                    cognition_layer.roi_tracker = tracker  # type: ignore[attr-defined]
                except Exception:
                    self.logger.warning(
                        "failed to attach ROI tracker to cognition_layer",
                        exc_info=True,
                        extra={"cognition_layer": type(cognition_layer).__name__},
                    )
        self.cognition_layer = cognition_layer
        self.patch_logger = patch_logger
        self.roi_tracker = tracker
        self.knowledge_service = knowledge_service
        try:
            success_log_path = resolve_path(_settings.prompt_success_log_path)
        except FileNotFoundError:
            success_log_path = Path(_settings.prompt_success_log_path)
        try:
            failure_log_path = resolve_path(_settings.prompt_failure_log_path)
        except FileNotFoundError:
            failure_log_path = Path(_settings.prompt_failure_log_path)
        if prompt_optimizer is None:
            try:
                prompt_optimizer = PromptOptimizer(
                    success_log_path,
                    failure_log_path,
                )
            except Exception:
                prompt_optimizer = None
        self.prompt_optimizer = prompt_optimizer
        # expose ROI tracker to the prompt engine so retrieved examples can
        # carry risk-adjusted ROI hints when available
        self.prompt_engine = PromptEngine(
            roi_tracker=tracker,
            tone=prompt_tone,
            trainer=self.prompt_memory,
            optimizer=self.prompt_optimizer,
            token_threshold=token_threshold,
            chunk_token_threshold=self.chunk_token_threshold,
            chunk_summary_cache_dir=self.chunk_summary_cache_dir,
            llm=self.llm_client,
        )
        if prompt_evolution_memory is None:
            try:
                prompt_evolution_memory = PromptEvolutionMemory(
                    success_path=success_log_path,
                    failure_path=failure_log_path,
                )
            except Exception:
                prompt_evolution_memory = None
        self.prompt_evolution_memory = prompt_evolution_memory
        self._last_prompt_metadata: Dict[str, Any] = {}
        self._last_prompt: Prompt | None = None
        self.router = kwargs.get("router")
        # store tracebacks from failed attempts for retry prompts
        self._last_retry_trace: str | None = None
        self._failure_cache = FailureCache()
        self._load_state()

    # ------------------------------------------------------------------
    def _load_state(self) -> None:
        lock = FileLock(str(self._state_path) + ".lock")
        try:
            with lock:
                data = json.loads(self._state_path.read_text())
            for v in data.get("similarity", []):
                try:
                    self.failure_similarity_tracker.update(similarity=float(v))
                except Exception:
                    continue
        except FileNotFoundError:
            self._save_state()
        except Exception:
            self.logger.warning("failed to load self-coding engine state")

    def _save_state(self) -> None:
        lock = FileLock(str(self._state_path) + ".lock")
        try:
            payload = {
                "similarity": self.failure_similarity_tracker.to_dict().get(
                    "similarity", []
                )
            }
            _atomic_write(self._state_path, json.dumps(payload), lock=lock)
        except Exception:
            self.logger.warning("failed to persist self-coding engine state")

    @property
    def failure_similarity_threshold(self) -> float:
        avg = self.failure_similarity_tracker.get("similarity")
        std = self.failure_similarity_tracker.std("similarity")
        threshold = avg + self.failure_similarity_k * std
        if self._failure_similarity_threshold is not None:
            threshold = max(self._failure_similarity_threshold, threshold)
        return threshold

    @failure_similarity_threshold.setter
    def failure_similarity_threshold(self, value: float | None) -> None:
        self._failure_similarity_threshold = value

    @property
    def last_prompt_text(self) -> str:
        """Return the text of the last prompt."""
        prompt = self._last_prompt
        return prompt.text if prompt else ""

    # ------------------------------------------------------------------

    def scan_repo(self) -> list["EnhancementSuggestion"]:
        """Run the enhancement classifier and queue suggestions."""
        classifier = getattr(self, "enhancement_classifier", None)
        if not classifier:
            return []
        suggestions: list["EnhancementSuggestion"] = []
        try:
            suggestions = list(classifier.scan_repo())
            if self.patch_suggestion_db:
                self.patch_suggestion_db.queue_enhancement_suggestions(suggestions)
            if self.event_bus:
                try:
                    self.event_bus.publish(
                        "enhancement:suggestions",
                        {
                            "count": len(suggestions),
                            "suggestions": [s.path for s in suggestions],
                        },
                    )
                except Exception:
                    self.logger.exception("event bus publish failed")
        except Exception:
            self.logger.exception("enhancement repo scan failed")
        return suggestions

    def _check_permission(self, action: str, requesting_bot: str | None) -> None:
        if not requesting_bot:
            return
        role = self.bot_roles.get(requesting_bot, READ)
        check_permission(role, action)

    def _log_attempt(self, requesting_bot: str | None, action: str, details: dict) -> None:
        bot = requesting_bot or "unknown"
        ts = datetime.utcnow().isoformat()
        try:
            payload = json.dumps(
                {"timestamp": ts, "bot": bot, "action": action, "details": details},
                sort_keys=True,
            )
            self.audit_trail.record(payload)
        except Exception as exc:
            self.logger.exception("audit trail logging failed for %s", action)
            if self.event_bus:
                try:
                    self.event_bus.publish(
                        "audit:failed", {"action": action, "error": str(exc)}
                    )
                except Exception:
                    self.logger.exception("event bus publish failed")

    def _store_patch_memory(
        self,
        path: Path,
        description: str,
        code: str,
        success: bool,
        roi_delta: float,
        target_region: TargetRegion | None = None,
    ) -> None:
        """Record GPT output and its outcome for later retrieval."""
        status = "success" if success else "failure"
        summary = f"status={status},roi_delta={roi_delta:.4f}"
        if target_region is not None:
            summary += (
                f",lines={target_region.start_line}-{target_region.end_line}"
            )
        try:
            key = f"{path}:{description}"
            if target_region is not None:
                key = f"{key}:{target_region.start_line}-{target_region.end_line}"
            self.gpt_memory.log_interaction(
                key, code.strip(), tags=[ERROR_FIX, IMPROVEMENT_PATH]
            )
            self.gpt_memory.log_interaction(
                f"{key}:result", summary, tags=[FEEDBACK]
            )
        except Exception:
            self.logger.exception("memory logging failed")

    def _record_prompt_metadata(self, success: bool) -> None:
        if not self.prompt_memory:
            return
        if not self._last_prompt_metadata:
            return
        try:
            self.prompt_memory.record(
                tone=self._last_prompt_metadata.get("tone", ""),
                headers=self._last_prompt_metadata.get("headers", []),
                example_order=self._last_prompt_metadata.get("example_order", []),
                success=success,
            )
        except Exception:
            self.logger.exception("failed to store prompt format history")
        finally:
            self._last_prompt_metadata = {}

    def _log_prompt_evolution(
        self,
        patch: str,
        success: bool,
        exec_result: Any | None,
        roi_delta: float,
        coverage: float,
        roi_meta: Mapping[str, Any] | None = None,
        baseline_runtime: float | None = None,
        *,
        module: str = "self_coding_engine",
        action: str = "apply_patch",
    ) -> None:
        """Record prompt execution details via :class:`PromptEvolutionMemory`."""
        if not self.prompt_evolution_memory:
            return
        prompt = getattr(self, "_last_prompt", None)
        if not isinstance(prompt, Prompt):
            prompt = Prompt(getattr(prompt, "text", str(prompt or "")))
        parts = [prompt.system, *prompt.examples, prompt.user]
        flat_prompt = "\n".join([p for p in parts if p])
        runtime = getattr(exec_result, "runtime", None) if exec_result else None
        result: Dict[str, Any] = {"patch": patch}
        runtime_improvement = 0.0
        if isinstance(exec_result, dict):
            result.update(exec_result)
        else:
            result.update(
                {
                    "stdout": getattr(exec_result, "stdout", ""),
                    "stderr": getattr(exec_result, "stderr", ""),
                }
            )
        if runtime is not None:
            result["runtime"] = runtime
            baseline = (
                baseline_runtime
                if baseline_runtime is not None
                else getattr(self, "_prev_runtime", None)
            )
            if baseline is not None:
                runtime_improvement = baseline - runtime
            self._prev_runtime = runtime
        meta = dict(getattr(prompt, "metadata", {}))
        meta.update(getattr(self.prompt_engine, "last_metadata", {}))
        meta.update(getattr(self, "_last_prompt_metadata", {}))
        prompt.metadata = meta
        roi: Dict[str, Any] = {"roi_delta": roi_delta, "coverage": coverage}
        if roi_meta:
            roi.update({k: v for k, v in roi_meta.items() if k != "runtime_improvement"})
        roi["runtime_improvement"] = runtime_improvement
        try:
            self.prompt_evolution_memory.log(
                prompt,
                success,
                result,
                roi,
                format_meta=getattr(self, "_last_prompt_metadata", {}),
                module=module,
                action=action,
                prompt_text=flat_prompt,
            )
        except Exception:
            self.logger.exception("prompt evolution logging failed")
        else:
            if self.prompt_optimizer:
                try:
                    self.prompt_optimizer.refresh()
                except Exception:
                    self.logger.exception("prompt optimizer refresh failed")

    def _track_contributors(
        self,
        session_id: str,
        vectors: Iterable[Tuple[str, str, float] | Tuple[str, str]],
        result: bool,
        patch_id: int | None = None,
        retrieval_metadata: Mapping[str, Mapping[str, Any]] | None = None,
        roi_delta: float | None = None,
        roi_deltas: Mapping[str, float] | None = None,
    ) -> None:
        """Forward vector contribution data to :class:`PatchLogger`.

        Parameters mirror :meth:`vector_service.patch_logger.PatchLogger.track_contributors`.
        This helper exists for historical compatibility and gracefully ignores
        failures from the underlying logger.
        """
        if not self.patch_logger:
            return
        try:
            ids: list[tuple[str, float]] = []
            for item in vectors:
                if len(item) == 3:  # type: ignore[comparison-overlap]
                    origin, vid, score = item  # type: ignore[misc]
                else:
                    origin, vid = item  # type: ignore[misc]
                    score = 0.0
                ids.append((f"{origin}:{vid}", float(score)))
            kwargs: dict[str, Any] = {
                "session_id": session_id or "",
                "retrieval_metadata": retrieval_metadata or {},
            }
            if patch_id is not None:
                kwargs["patch_id"] = str(patch_id)
            if roi_delta is not None:
                kwargs["roi_delta"] = roi_delta
                kwargs["contribution"] = roi_delta
            if roi_deltas:
                kwargs["roi_deltas"] = dict(roi_deltas)
            self.patch_logger.track_contributors(ids, result, **kwargs)
            tracker = getattr(self.patch_logger, "roi_tracker", None)
            if tracker is not None:
                totals: dict[str, float] = {}
                for vid, _ in ids:
                    origin = vid.split(":", 1)[0] if ":" in vid else ""
                    totals[origin] = totals.get(origin, 0.0) + (roi_delta or 0.0)
                if hasattr(tracker, "update_db_metrics"):
                    try:
                        tracker.update_db_metrics({o: {"roi": v} for o, v in totals.items()})
                    except Exception:
                        self.logger.exception(
                            "update_db_metrics failed (roi_delta=%s, ids=%s, totals=%s)",
                            roi_delta,
                            ids,
                            totals,
                        )
                if hasattr(tracker, "metrics"):
                    for o, v in totals.items():
                        tracker.metrics.setdefault(o, {})["roi"] = v
        except Exception:
            self.logger.exception("track_contributors failed")

    # --------------------------------------------------------------
    def suggest_snippets(self, description: str, limit: int = 3) -> Iterable[CodeRecord]:
        """Return code snippets related to the description."""
        try:
            records = self.code_db.search(description)
        except Exception:
            return []
        return [CodeRecord(**r) for r in records[:limit]]

    # --------------------------------------------------------------
    @staticmethod
    def _extract_statements(code: str) -> list[str]:
        """Return top-level statements from *code* or function bodies."""

        try:
            tree = ast.parse(code)
        except Exception:
            return []

        nodes: list[ast.stmt] = []
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                nodes.extend(node.body)
            else:
                nodes.append(node)

        lines: list[str] = []
        for n in nodes:
            try:
                text = ast.unparse(n).strip()
            except Exception:
                continue
            if text:
                lines.extend(text.splitlines())
        return lines

    def _fetch_retry_trace(self, metadata: Dict[str, Any] | None) -> str | None:
        """Return a traceback from metadata, analyzer logs or previous failures."""

        meta = metadata or {}
        trace = meta.get("retry_trace")
        if trace:
            return str(trace)
        log_path = meta.get("analysis_log")
        if log_path:
            try:  # pragma: no cover - best effort
                from codex_output_analyzer import load_analysis_result

                result = load_analysis_result(log_path)
                err = result.get("error")
                if isinstance(err, dict):
                    return err.get("details") or err.get("type")
            except Exception:
                self.logger.debug("failed to load analysis log", exc_info=True)
        return self._last_retry_trace

    @staticmethod
    def _get_repo_layout(limit: int) -> str:
        """Return a short list of top-level Python files in the repo."""
        root = resolve_path(".")
        files = sorted(p.name for p in root.glob("*.py"))
        lines = files[:limit]
        if len(files) > limit:
            lines.append("...")
        return "\n".join(lines)

    def _build_file_context(
        self,
        path: Path,
        chunk_index: int | None = None,
        target_region: TargetRegion | None = None,
    ) -> tuple[str, list[str] | None]:
        """Return either trimmed code or chunk summaries for ``path``.

        When the file's token count exceeds :attr:`chunk_token_threshold` the
        source is summarised via :func:`chunking.get_chunk_summaries`.  When
        ``chunk_index`` is provided the raw source of that chunk is returned and
        the remaining chunks are represented by their summaries.  Otherwise the
        joined summaries are returned.  For smaller files the full source is
        returned truncated to :attr:`token_threshold`. When ``target_region`` is
        supplied only the specified line range plus minimal surrounding context
        is returned.
        """
        path = resolve_path(path)
        try:
            code = path.read_text(encoding="utf-8")
        except Exception:
            return "", None

        if target_region is not None:
            start = target_region.start_line
            end = target_region.end_line
            lines = code.splitlines()
            if not target_region.filename:
                target_region.filename = str(path)
            target_region.original_lines = lines[start - 1:end]
            func_sig = ""
            if target_region.function:
                pat = re.compile(
                    rf"^\s*def\s+{re.escape(target_region.function)}\s*\("
                )
                for ln in lines:
                    if pat.match(ln):
                        func_sig = ln.strip()
                        break
            target_region.func_signature = func_sig or None
            threshold = self.chunk_token_threshold or 0
            header = (
                f"# Region {target_region.function or ''} lines"
                f" {start}-{end}"
            ).strip()
            region_body = "\n".join(target_region.original_lines)
            snippet_lines = ["# start", region_body, "# end"]
            snippet = "\n".join([line for line in snippet_lines if line])
            if func_sig:
                snippet = f"{header}\n{func_sig}\n{snippet}"
            else:
                snippet = f"{header}\n{snippet}"
            if code and threshold and _count_tokens(code) > threshold:
                try:
                    chunks = split_into_chunks(
                        code,
                        threshold,
                        line_ranges=[(start, end)],
                    )
                except Exception:
                    self.logger.exception(
                        "failed to split %s", path_for_prompt(path)
                    )
                    return "", None
                summaries: List[str] = []
                for i, ch in enumerate(chunks):
                    if not (start <= ch.start_line and ch.end_line <= end):
                        summary = summarize_code(ch.text, self.llm_client)
                        summaries.append(f"Chunk {i}: {summary}")
                return snippet, summaries or None
            return snippet, None

        threshold = self.chunk_token_threshold or 0
        if code and threshold and _count_tokens(code) > threshold:
            try:
                summary_entries = get_chunk_summaries(
                    path, threshold, self.llm_client
                )
            except Exception:
                self.logger.exception(
                    "failed to summarise %s", path_for_prompt(path)
                )
                return "", None

            lines = code.splitlines()
            if chunk_index is not None and 0 <= chunk_index < len(summary_entries):
                entry = summary_entries[chunk_index]
                start = int(entry.get("start_line", 1))
                end = int(entry.get("end_line", start))
                selected = "\n".join(lines[start - 1:end])
                context = f"# Chunk {chunk_index} lines {start}-{end}\n{selected}"
                summaries = [
                    f"Chunk {i}: {e.get('summary', '')}"
                    for i, e in enumerate(summary_entries)
                    if i != chunk_index
                ]
            else:
                summaries = [
                    f"Chunk {i}: {e.get('summary', '')}"
                    for i, e in enumerate(summary_entries)
                ]
                context = "\n".join(summaries)
            return context, summaries

        if self.prompt_engine:
            return (
                self.prompt_engine._trim_tokens(code, self.token_threshold),
                None,
            )
        return code, None

    def _apply_prompt_style(self, action: str, module: str | None = None) -> None:
        if not self.prompt_engine:
            return
        try:
            self.prompt_engine.apply_optimizer_format(
                module or "self_coding_engine", action
            )
        except Exception:
            return
        self.prompt_tone = self.prompt_engine.tone
        # store metadata so that failures before prompt construction can be logged
        self._last_prompt_metadata = dict(getattr(self.prompt_engine, "last_metadata", {}))

    def build_visual_agent_prompt(
        self,
        path: str | None,
        description: str,
        context: str,
        retrieval_context: str | None = None,
        repo_layout: str | None = None,
        target_region: TargetRegion | None = None,
        strategy: str | None = None,
    ) -> str:
        """Return a prompt formatted for :class:`VisualAgentClient`.

        When ``target_region`` is provided the line range metadata is embedded in
        the prompt so downstream components can reason about the intended scope.
        ``strategy`` selects an optional instruction block from the strategy
        templates.
        """
        func = f"auto_{description.replace(' ', '_')}"
        repo_layout = repo_layout or self._get_repo_layout(VA_REPO_LAYOUT_LINES)
        resolved = path_for_prompt(path) if path else None
        self._apply_prompt_style(description, module=resolved or "visual_agent")
        retry_trace = self._last_retry_trace
        try:
            prompt_obj = self.prompt_engine.build_prompt(
                description,
                context="\n".join([p for p in (context.strip(), repo_layout) if p]),
                retrieval_context=retrieval_context or "",
                retry_trace=retry_trace,
                tone=self.prompt_tone,
                target_region=target_region,
                strategy=strategy,
            )
        except TypeError:
            prompt_obj = self.prompt_engine.build_prompt(
                description,
                context="\n".join([p for p in (context.strip(), repo_layout) if p]),
                retrieval_context=retrieval_context or "",
                retry_trace=retry_trace,
            )
        self._last_prompt = prompt_obj
        body = prompt_obj.text if isinstance(prompt_obj, Prompt) else str(prompt_obj)
        meta = dict(getattr(self.prompt_engine, "last_metadata", {}))
        meta.update(
            {
                "system": getattr(prompt_obj, "system", ""),
                "examples": getattr(prompt_obj, "examples", []),
            }
        )
        self._last_prompt_metadata = meta
        if VA_PROMPT_TEMPLATE:
            try:
                text = resolve_path(VA_PROMPT_TEMPLATE).read_text()
            except Exception:
                text = VA_PROMPT_TEMPLATE
            data = {
                "path": resolved or "unknown file",
                "description": description,
                "context": context.strip(),
                "retrieval_context": retrieval_context or "",
                "func": func,
                "prompt": body,
            }
            try:
                rendered = text.format(**data)
            except Exception:
                rendered = text
            if "prompt" in text:
                body = rendered
            else:
                if not rendered.endswith("\n"):
                    rendered += "\n"
                body = rendered + body
        prefix = VA_PROMPT_PREFIX
        if prefix:
            if not prefix.endswith("\n"):
                prefix += "\n"
            body = prefix + body
        return body

    def generate_helper(
        self,
        description: str,
        *,
        path: Path | None = None,
        metadata: Dict[str, Any] | None = None,
        chunk_index: int | None = None,
        target_region: TargetRegion | None = None,
        strategy: str | None = None,
    ) -> str:
        """Create helper text using snippet and retrieval context.

        When ``path`` points to a file larger than
        :attr:`prompt_chunk_token_threshold`, the file is summarised via
        :func:`chunking.get_chunk_summaries`.  ``chunk_index`` selects the chunk
        whose raw code is provided to the model; other chunks are represented by
        their summaries.  ``strategy`` allows callers to inject an optional
        instruction snippet from the strategy templates.
        """
        snippets = self.suggest_snippets(description, limit=3)
        snippet_context = "\n\n".join(s.code for s in snippets)
        summaries: List[str] | None = None
        file_context = ""
        if path:
            path = resolve_path(path)
            file_context, summaries = self._build_file_context(
                path, chunk_index, target_region
            )
        context = "\n\n".join(p for p in (file_context, snippet_context) if p)

        def _fallback() -> str:
            """Return a minimal helper implementation."""
            func = f"auto_{description.replace(' ', '_')}"
            body: list[str] = []
            for snip in snippets:
                body = [
                    "    " + ln
                    for ln in self._extract_statements(snip.code)
                    if ln.strip() and ln.strip() != "pass"
                ]
                if body:
                    break
            if not body:
                desc = description.lower()
                if "print" in desc:
                    body = ["    print(*args, **kwargs)"]
                elif "read" in desc and "file" in desc:
                    body = [
                        "    path = args[0] if args else kwargs.get('path')",
                        "    with open(path, 'r', encoding='utf-8') as fh:",
                        "        return fh.read()",
                    ]
                elif "write" in desc and "file" in desc:
                    body = [
                        "    path = args[0] if args else kwargs.get('path')",
                        "    data = args[1] if len(args) > 1 else kwargs.get('data', '')",
                        "    with open(path, 'w', encoding='utf-8') as fh:",
                        "        fh.write(data)",
                    ]
                else:
                    body = [
                        "    return {",
                        f"        'description': '{description}',",
                        "        'args': args,",
                        "        'kwargs': kwargs,",
                        "    }",
                    ]
            skeleton = [
                f"def {func}(*args, **kwargs):",
                f'    """{description}"""',
                *body,
                "",
            ]
            return "\n".join(skeleton)

        if not self.llm_client or not self.prompt_engine:
            return _fallback()
        if metadata is None:
            builder = getattr(self, "context_builder", None)
            if builder:
                try:
                    metadata = {"retrieval_context": builder.build_context(description)}
                except Exception:
                    metadata = None
        repo_layout = self._get_repo_layout(VA_REPO_LAYOUT_LINES)
        context_block = "\n".join([p for p in (context, repo_layout) if p])
        module_name = path_for_prompt(path) if path else "generate_helper"
        self._apply_prompt_style(description, module=module_name)
        retrieval_context = (
            str(metadata.get("retrieval_context", "")) if metadata else ""
        )
        retry_trace = self._fetch_retry_trace(metadata)
        if strategy is None and metadata:
            strategy = (
                metadata.get("strategy")
                or metadata.get("prompt_id")
                or metadata.get("prompt_strategy")
            )
        try:
            prompt_obj = self.prompt_engine.build_prompt(
                description,
                context=context_block,
                retrieval_context=retrieval_context,
                retry_trace=retry_trace,
                tone=self.prompt_tone,
                summaries=summaries,
                target_region=target_region,
                strategy=strategy,
            )
        except TypeError:
            if target_region is not None:
                path_hint = path_for_prompt(path) if path else None
                instr = (
                    f"Modify only lines {target_region.start_line}-{target_region.end_line}"
                )
                if path_hint:
                    instr += f" in {path_hint}"
                instr += " unless dependent code requires changes."
                context_block = (
                    "\n".join([instr, context_block])
                    if context_block
                    else instr
                )
            prompt_obj = self.prompt_engine.build_prompt(
                description,
                context=context_block,
                retrieval_context=retrieval_context,
                retry_trace=retry_trace,
                summaries=summaries,
            )
        except Exception as exc:
            self._last_retry_trace = str(exc)
            self._last_prompt_metadata = {}
            return _fallback()
        self._last_prompt = prompt_obj
        meta = dict(getattr(self.prompt_engine, "last_metadata", {}))
        meta.update(
            {
                "system": getattr(prompt_obj, "system", ""),
                "examples": getattr(prompt_obj, "examples", []),
            }
        )
        self._last_prompt_metadata = meta
        if metadata and metadata.get("retrieval_context"):
            rc = metadata["retrieval_context"]
            if not isinstance(rc, str):
                try:
                    rc_text = json.dumps(rc, indent=2)
                except Exception:
                    rc_text = str(rc)
            else:
                rc_text = rc
            prompt_obj.text += "\n\n### Retrieval context\n" + rc_text

        # Incorporate past patch outcomes from memory
        history = ""
        try:
            entries = get_feedback(self.gpt_memory, description, limit=5)
            if entries:
                hist_summaries: List[str] = []
                for ent in entries:
                    resp = (getattr(ent, "response", "") or "").strip()
                    tag = "success" if "status=success" in resp else "failure"
                    snippet = resp.splitlines()[0]
                    hist_summaries.append(f"{tag}: {snippet}")
                history = "\n".join(hist_summaries)
        except Exception:
            history = ""
        fix_history = ""
        try:
            fixes = get_error_fixes(self.gpt_memory, description, limit=3)
            if fixes:
                snippets = []
                for fix in fixes:
                    resp = (getattr(fix, "response", "") or "").strip()
                    if resp:
                        snippets.append(resp.splitlines()[0])
                if snippets:
                    fix_history = "\n".join(f"fix: {s}" for s in snippets)
        except Exception:
            fix_history = ""
        combined_history = "\n".join([p for p in (history, fix_history) if p])
        insight_lines: List[str] = []
        if self.knowledge_service:
            try:
                insight = recent_feedback(self.knowledge_service)
                if insight:
                    insight_lines.append(f"{FEEDBACK} insight: {insight}")
            except Exception:
                self.logger.warning(
                    "knowledge_service recent_feedback failed",
                    exc_info=True,
                    extra={"description": description},
                )
            try:
                insight = recent_improvement_path(self.knowledge_service)
                if insight:
                    insight_lines.append(f"{IMPROVEMENT_PATH} insight: {insight}")
            except Exception:
                self.logger.warning(
                    "knowledge_service recent_improvement_path failed",
                    exc_info=True,
                    extra={"description": description},
                )
            try:
                insight = recent_error_fix(self.knowledge_service)
                if insight:
                    insight_lines.append(f"{ERROR_FIX} insight: {insight}")
            except Exception:
                self.logger.warning(
                    "knowledge_service recent_error_fix failed",
                    exc_info=True,
                    extra={"description": description},
                )
        if insight_lines:
            insight_block = "\n".join(insight_lines)
            combined_history = "\n".join(
                [p for p in (combined_history, insight_block) if p]
            )
        if combined_history:
            self.logger.info(
                "patch history context",
                extra={
                    "description": description,
                    "history": combined_history,
                    "tags": [INSIGHT],
                },
            )
            prompt_obj.text += "\n\n### Patch history\n" + combined_history

        try:
            result = self.llm_client.generate(prompt_obj)
        except Exception as exc:
            self._last_retry_trace = str(exc)
            result = LLMResult()
        text = result.text.strip()
        if text:
            if self.gpt_memory:
                try:
                    meta_payload = json.dumps({
                        "prompt": prompt_obj.metadata,
                        "raw": result.raw,
                    })
                    self.gpt_memory.log_interaction(
                        prompt_obj.text, text, tags=[ERROR_FIX]
                    )
                    self.gpt_memory.store(
                        f"{description}:metadata", meta_payload, tags=[INSIGHT]
                    )
                except Exception:
                    self.logger.exception("memory logging failed")
            if path is not None and hasattr(self.memory_mgr, "store"):
                try:
                    self.memory_mgr.store(str(path), text, tags="code")
                except Exception:
                    self.logger.exception("memory manager store failed")
            self.logger.info(
                "gpt_suggestion",
                extra={
                    "tags": [ERROR_FIX],
                    "suggestion": text,
                    "description": description,
                    "path": str(path) if path else None,
                },
            )
            return text + ("\n" if not text.endswith("\n") else "")
        return _fallback()

    def patch_file(
        self,
        path: Path,
        description: str,
        *,
        context_meta: Dict[str, Any] | None = None,
        chunk_index: int | None = None,
        target_region: TargetRegion | None = None,
        strategy: str | None = None,
    ) -> tuple[str, bool]:
        """Generate helper code and append it to ``path`` if it passes verification.

        When ``chunk_index`` is provided and the file exceeds
        :attr:`prompt_chunk_token_threshold`, only the selected chunk's raw
        source is included in the prompt.  Other chunks are represented by their
        summaries to keep the prompt size below token limits.  When
        ``target_region`` is supplied, only that snippet and minimal surrounding
        context are presented to the model and the generated code is spliced back
        into the original file at the specified range.  ``strategy`` forwards an
        optional prompt strategy to the underlying :class:`PromptEngine`.
        """
        path = resolve_path(path)
        try:
            code = self.generate_helper(
                description,
                path=path,
                metadata=context_meta,
                chunk_index=chunk_index,
                target_region=target_region,
                strategy=strategy,
            )
        except TypeError:
            code = self.generate_helper(description, strategy=strategy)
        self.logger.info(
            "patch file",
            extra={
                "path": str(path),
                "description": description,
                "tags": [ERROR_FIX],
            },
        )
        verified = True
        if self.formal_verifier:
            with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh:
                fh.write(code)
                tmp_path = resolve_path(fh.name)
            try:
                verified = self.formal_verifier.verify(tmp_path)
            finally:
                try:
                    tmp_path.unlink()
                except Exception:
                    self.logger.exception(
                        "temporary file cleanup failed",
                        extra={"path": str(tmp_path)},
                    )
        else:
            try:
                ast.parse(code)
            except SyntaxError:
                verified = False
        if not verified:
            self.logger.warning("pre-verification failed; patch not applied")
            return "", False
        if target_region is not None:
            original_lines = path.read_text(encoding="utf-8").splitlines()
            if not self._apply_region_patch(path, original_lines, target_region, code):
                return "", False
        else:
            with open(path, "a", encoding="utf-8") as fh:
                fh.write("\n" + code)
        self.memory_mgr.store(str(path), code, tags="code")
        self.logger.info(
            "patch applied",
            extra={
                "path": str(path),
                "description": description,
                "tags": [FEEDBACK],
                "success": True,
            },
        )
        return code, True

    def _apply_region_patch(
        self,
        path: Path,
        original_lines: List[str],
        target_region: TargetRegion,
        patch_text: str,
    ) -> bool:
        """Replace ``target_region`` lines with ``patch_text`` ensuring AST validity."""

        start = max(target_region.start_line - 1, 0)
        end = min(target_region.end_line, len(original_lines))

        indent = ""
        for i in range(start, end):
            if i < len(original_lines):
                m = re.match(r"\s*", original_lines[i])
                indent = m.group(0) if m else ""
                if original_lines[i].strip():
                    break

        patch_lines = patch_text.rstrip().splitlines()
        if indent:
            patch_lines = [
                indent + line if line.strip() else line for line in patch_lines
            ]

        new_lines = original_lines[:start] + patch_lines + original_lines[end:]
        if not diff_within_target_region(original_lines, new_lines, target_region):
            self.logger.warning(
                "patch modified lines outside target region",
                extra={"path": str(path), "tags": [FEEDBACK]},
            )
            return False
        text = "\n".join(new_lines) + "\n"
        try:
            ast.parse(text)
        except SyntaxError:
            return False
        path.write_text(text, encoding="utf-8")
        return True

    def _find_function_region(
        self, lines: List[str], func_name: str
    ) -> TargetRegion | None:
        """Locate ``func_name`` in ``lines`` returning its :class:`TargetRegion`."""

        try:
            tree = ast.parse("\n".join(lines))
        except SyntaxError:
            return None
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == func_name:
                end = getattr(node, "end_lineno", node.lineno)
                return TargetRegion(node.lineno, end, func_name)
        return None

    def _run_ci(self, path: Path | None = None) -> TestHarnessResult:
        """Run linting and unit tests inside isolated environments."""

        file_name = path.name if path else None
        test_data = {file_name: path.read_text(encoding="utf-8")} if path else None

        def workflow() -> bool:
            ok = True
            target = resolve_path(file_name) if file_name else resolve_path(".")
            if self.formal_verifier and path is not None:
                try:
                    if not self.formal_verifier.verify(target):
                        ok = False
                except Exception as exc:  # pragma: no cover - verifier issues
                    self.logger.error("formal verification failed: %s", exc)
                    self._last_retry_trace = str(exc)
                    ok = False
            try:
                py_compile.compile(str(target), doraise=True)
            except Exception as exc:
                self.logger.error("lint failed: %s", exc)
                self._last_retry_trace = str(exc)
                ok = False
            return ok

        runner = WorkflowSandboxRunner()
        metrics = runner.run(workflow, test_data=test_data)
        lint_ok = bool(metrics.modules and metrics.modules[0].result)

        harness_result = run_tests(Path.cwd(), path)
        results = harness_result if isinstance(harness_result, list) else [harness_result]
        success = lint_ok and all(r.success for r in results)
        failure_res = next((r for r in results if not r.success), results[0])
        if success:
            self.logger.info("CI checks succeeded")
        else:
            if not lint_ok:
                self.logger.error("lint failed")
            if not failure_res.success:
                self.logger.error("tests failed")
                self._last_retry_trace = failure_res.stderr or failure_res.stdout
            trace = self._last_retry_trace or ""
            try:
                failure = ErrorParser.parse_failure(trace)
                tag = failure.get("strategy_tag", "")
                if tag and self.patch_suggestion_db:
                    self.patch_suggestion_db.add_failed_strategy(tag)
            except Exception:
                self.logger.exception("failed to store strategy tag")
        return TestHarnessResult(
            success,
            "\n".join(r.stdout for r in results),
            "\n".join(r.stderr for r in results),
            sum(r.duration for r in results) / len(results),
            failure_res.failure,
            failure_res.path,
        )

    def _current_errors(self) -> int:
        """Return the latest recorded error count for the bot."""
        if not self.data_bot:
            return 0
        try:
            rows = self.data_bot.db.fetch(50)
        except Exception:
            return 0
        if hasattr(rows, "empty"):
            df = rows[rows["bot"] == self.bot_name]
            if df.empty:
                return 0
            return int(df.iloc[0]["errors"])
        if isinstance(rows, list):
            for row in rows:
                if row.get("bot") == self.bot_name:
                    return int(row.get("errors", 0))
        return 0

    def _apply_patch_chunked(
        self,
        path: Path,
        description: str,
        *,
        context_meta: Dict[str, Any] | None = None,
        requesting_bot: str | None = None,
        target_region: TargetRegion | None = None,
    ) -> tuple[int | None, bool, float]:
        """Apply patches sequentially to each chunk in a large file.

        When ``target_region`` is provided the patch is generated only for the
        specified line range instead of chunking the entire file.

        For every chunk we craft a prompt that contains the raw source of the
        selected chunk along with summaries of the other chunks.  Generated code
        is spliced back into the original file at the chunk boundary.  Each
        chunk is verified and CI-tested independently; failures trigger an
        immediate rollback of that chunk so processing can continue with the
        remaining chunks.
        """

        # Snapshot of original file to merge patches into
        original_lines = path.read_text(encoding="utf-8").splitlines()
        code = "\n".join(original_lines)

        def _verify(snippet: str) -> bool:
            if self.formal_verifier:
                with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh:
                    fh.write(snippet)
                    tmp_path = resolve_path(fh.name)
                try:
                    return self.formal_verifier.verify(tmp_path)
                finally:
                    try:
                        tmp_path.unlink()
                    except Exception:
                        pass
            try:
                ast.parse(snippet)
                return True
            except SyntaxError:
                return False

        if target_region is not None:
            generated = self.generate_helper(
                description,
                path=path,
                metadata=context_meta,
                target_region=target_region,
            )
            if not generated.strip():
                return None, False, 0.0
                if _verify(generated):
                    if not self._apply_region_patch(
                        path, original_lines, target_region, generated
                    ):
                        if target_region.function:
                            func_region = self._find_function_region(
                                original_lines, target_region.function
                            )
                        else:
                            func_region = None
                    if func_region is None:
                        return None, False, 0.0
                    generated = self.generate_helper(
                        description,
                        path=path,
                        metadata=context_meta,
                        target_region=func_region,
                    )
                    if not generated.strip() or not _verify(generated):
                        return None, False, 0.0
                    if not self._apply_region_patch(path, original_lines, func_region, generated):
                        return None, False, 0.0
                    target_region = func_region
            else:
                if not target_region.function:
                    return None, False, 0.0
                func_region = self._find_function_region(original_lines, target_region.function)
                if func_region is None:
                    return None, False, 0.0
                generated = self.generate_helper(
                    description,
                    path=path,
                    metadata=context_meta,
                    target_region=func_region,
                )
                if not generated.strip() or not _verify(generated):
                    return None, False, 0.0
                if not self._apply_region_patch(path, original_lines, func_region, generated):
                    return None, False, 0.0
                target_region = func_region
            start = max(target_region.start_line - 1, 0)
            end = min(target_region.end_line, len(original_lines))
            patch_key = f"{path}:{description}:{start}-{end}"
            if self.rollback_mgr:
                try:
                    self.rollback_mgr.register_region_patch(
                        patch_key,
                        self.bot_name,
                        str(path),
                        target_region.start_line,
                        target_region.end_line,
                    )
                except Exception:
                    self.logger.exception("failed to register region patch")
            ci_result = self._run_ci(path)
            if not ci_result.success:
                path.write_text("\n".join(original_lines) + "\n", encoding="utf-8")
                if self.rollback_mgr:
                    try:
                        self.rollback_mgr.rollback_region(
                            str(path),
                            target_region.start_line,
                            target_region.end_line,
                            requesting_bot=requesting_bot,
                        )
                    except Exception:
                        self.logger.exception("region rollback failed")
                return None, False, 0.0
            last_patch_id = None
            if self.patch_db:
                try:
                    last_patch_id = self.patch_db.add(
                        PatchRecord(
                            filename=str(path),
                            description=f"{description} [region]",
                            roi_before=0.0,
                            roi_after=0.0,
                            reverted=False,
                        )
                    )
                except Exception:
                    self.logger.exception("failed to record region patch")
            if self.roi_tracker:
                try:
                    self.roi_tracker.update(0.0, 0.0)
                except Exception:
                    pass
            if self.memory_mgr:
                try:
                    self.memory_mgr.store(str(path), generated, tags="code")
                except Exception:
                    self.logger.exception("memory store failed")
            return last_patch_id, True, 0.0

        chunks = split_into_chunks(code, self.chunk_token_threshold)
        offset = 0
        success_any = False
        last_patch_id: int | None = None

        def _generate(idx: int) -> str:
            try:
                return self.generate_helper(
                    description,
                    path=path,
                    metadata=context_meta,
                    chunk_index=idx,
                )
            except Exception:
                self.logger.exception("chunk generation failed", extra={"chunk": idx})
                return ""

        with ThreadPoolExecutor(max_workers=min(len(chunks), 4) or 1) as ex:
            futures = {ex.submit(_generate, i): i for i in range(len(chunks))}
            generated_chunks: Dict[int, str] = {i: "" for i in range(len(chunks))}
            for fut in as_completed(futures):
                idx = futures[fut]
                try:
                    generated_chunks[idx] = fut.result() or ""
                except Exception:
                    self.logger.exception("chunk generation failed", extra={"chunk": idx})
                    generated_chunks[idx] = ""

        for idx, ch in enumerate(chunks):
            generated = generated_chunks.get(idx, "")
            if not generated.strip() or not _verify(generated):
                continue

            insert_at = ch.end_line + offset
            patch_lines = generated.rstrip().splitlines()
            original_lines[insert_at:insert_at] = patch_lines
            path.write_text("\n".join(original_lines) + "\n", encoding="utf-8")

            patch_key = f"{path}:{description}:{idx}"
            if self.rollback_mgr:
                try:
                    self.rollback_mgr.register_patch(patch_key, self.bot_name)
                except Exception:
                    self.logger.exception("failed to register patch")

            ci_result = self._run_ci(path)
            if not ci_result.success:
                del original_lines[insert_at:insert_at + len(patch_lines)]
                path.write_text("\n".join(original_lines) + "\n", encoding="utf-8")
                if self.rollback_mgr:
                    try:
                        self.rollback_mgr.rollback(patch_key, requesting_bot=requesting_bot)
                    except Exception:
                        self.logger.exception("chunk rollback failed")
                if self.patch_db:
                    try:
                        pid = self.patch_db.add(
                            PatchRecord(
                                filename=str(path),
                                description=f"{description} [chunk {idx}]",
                                roi_before=0.0,
                                roi_after=0.0,
                                reverted=True,
                            )
                        )
                        record_patch_metadata(pid, {"chunk": idx}, patch_db=self.patch_db)
                    except Exception:
                        self.logger.exception("failed to record chunk metadata")
                continue

            success_any = True
            offset += len(patch_lines)
            if self.patch_db:
                try:
                    last_patch_id = self.patch_db.add(
                        PatchRecord(
                            filename=str(path),
                            description=f"{description} [chunk {idx}]",
                            roi_before=0.0,
                            roi_after=0.0,
                            reverted=False,
                        )
                    )
                    record_patch_metadata(
                        last_patch_id, {"chunk": idx}, patch_db=self.patch_db
                    )
                except Exception:
                    self.logger.exception("failed to record chunk metadata")

            if self.roi_tracker:
                try:
                    self.roi_tracker.update(0.0, 0.0)
                except Exception:
                    pass

            if self.memory_mgr:
                try:
                    self.memory_mgr.store(str(path), generated, tags="code")
                except Exception:
                    self.logger.exception("memory store failed")

        return last_patch_id, success_any, 0.0

    def apply_patch(
        self,
        path: Path,
        description: str,
        *,
        trending_topic: str | None = None,
        parent_patch_id: int | None = None,
        reason: str | None = None,
        trigger: str | None = None,
        requesting_bot: str | None = None,
        context_meta: Dict[str, Any] | None = None,
        effort_estimate: float | None = None,
        suggestion_id: int | None = None,
        baseline_coverage: float | None = None,
        baseline_runtime: float | None = None,
        target_region: TargetRegion | None = None,
    ) -> tuple[int | None, bool, float]:
        """Patch a file, optionally restricting edits to a target region.

        When ``target_region`` is provided the specified line range along with
        minimal surrounding context is supplied to the language model and the
        generated code is spliced back into the original file.  Unrelated code
        remains untouched and ROI/memory bookkeeping is scoped to the region.

        Returns the rowid of the stored patch if available.
        """
        self.logger.info(
            "apply_patch",
            extra={
                "path": str(path),
                "description": description,
                "tags": [ERROR_FIX],
            },
        )
        self._log_attempt(
            requesting_bot,
            "apply_patch_start",
            {"path": str(path), "description": description},
        )
        try:
            self._check_permission(WRITE, requesting_bot)
        except PermissionError:
            self._log_attempt(requesting_bot, "apply_patch_denied", {"path": str(path)})
            raise
        if reason is None:
            reason = description
        if trigger is None:
            trigger = ""
        before_roi = 0.0
        before_err = self._current_errors()
        pred_before_roi = pred_before_err = 0.0
        if self.trend_predictor:
            try:
                pred = self.trend_predictor.predict_future_metrics(1)
                pred_before_roi = pred.roi
                pred_before_err = pred.errors
            except Exception as exc:
                self.logger.exception("trend prediction failed", exc_info=True)
                if self.event_bus:
                    try:
                        self.event_bus.publish(
                            "patch:error", {"stage": "predict_before", "error": str(exc)}
                        )
                    except Exception:
                        self.logger.exception("event bus publish failed")
                pred_before_roi = pred_before_err = 0.0
        before_complexity = 0.0
        if self.data_bot:
            try:
                before_roi = self.data_bot.roi(self.bot_name)
            except Exception as exc:
                self.logger.error("roi query failed: %s", exc)
                before_roi = 0.0
            try:
                before_complexity = self.data_bot.complexity_score(self.data_bot.db.fetch(20))
            except Exception as exc:
                self.logger.error("complexity query failed: %s", exc)
                before_complexity = 0.0
        if context_meta is None and self.cognition_layer is not None:
            if self.patch_suggestion_db:
                try:
                    tags = self.patch_suggestion_db.failed_strategy_tags()
                    self.cognition_layer.context_builder.exclude_failed_strategies(tags)
                except Exception:
                    self.logger.exception("failed to apply strategy exclusions")
            try:
                ctx, sid = self.cognition_layer.query(description)
                context_meta = {
                    "retrieval_context": ctx,
                    "retrieval_session_id": sid,
                }
            except VectorServiceError as exc:
                self.logger.warning("cognition layer query failed: %s", exc)
                context_meta = {
                    "retrieval_context": "",
                    "retrieval_session_id": "",
                }
            except Exception:
                context_meta = {
                    "retrieval_context": "",
                    "retrieval_session_id": "",
                }
        original = path.read_text(encoding="utf-8")
        if _count_tokens(original) > self.chunk_token_threshold:
            return self._apply_patch_chunked(
                path,
                description,
                context_meta=context_meta,
                requesting_bot=requesting_bot,
                target_region=target_region,
            )
        generated_code, pre_verified = self.patch_file(
            path,
            description,
            context_meta=context_meta,
            target_region=target_region,
        )
        self._log_attempt(
            requesting_bot,
            "patch_verification",
            {"path": str(path), "verified": pre_verified},
        )
        session_id = ""
        if context_meta:
            session_id = context_meta.get("retrieval_session_id", "")
        vectors: List[Tuple[str, str, float]] = []
        retrieval_metadata: Dict[str, Dict[str, Any]] = {}
        if not pre_verified or not generated_code.strip():
            self.logger.info("no code generated; skipping enhancement")
            if target_region is not None:
                cur_lines = path.read_text(encoding="utf-8").splitlines()
                orig_lines = original.splitlines()
                start = max(target_region.start_line - 1, 0)
                end = min(target_region.end_line, len(orig_lines))
                cur_lines[start:end] = orig_lines[start:end]
                path.write_text("\n".join(cur_lines) + "\n", encoding="utf-8")
            else:
                path.write_text(original, encoding="utf-8")
            ci_result = self._run_ci(path)
            self._store_patch_memory(
                path,
                description,
                generated_code,
                False,
                0.0,
                target_region=target_region,
            )
            if self.cognition_layer and session_id:
                try:
                    self.cognition_layer.record_patch_outcome(session_id, False)
                except VectorServiceError as exc:
                    self.logger.warning("cognition layer unavailable: %s", exc)
                except Exception:
                    self.logger.exception("failed to record patch outcome")
            self._log_attempt(
                requesting_bot,
                "apply_patch_result",
                {"path": str(path), "success": False},
            )
            roi_meta = {
                "coverage_delta": (0.0 - (baseline_coverage or 0.0)),
            }
            self._log_prompt_evolution(
                generated_code,
                False,
                ci_result,
                0.0,
                0.0,
                roi_meta,
                baseline_runtime=baseline_runtime,
                module=str(path),
                action=description,
            )
            self._record_prompt_metadata(False)
            return None, False, 0.0
        if self.formal_verifier:
            verified = False
            if target_region is not None:
                with tempfile.NamedTemporaryFile("w", suffix=path.suffix, delete=False) as fh:
                    fh.write(generated_code)
                    tmp_path = resolve_path(fh.name)
                try:
                    verified = self.formal_verifier.verify(tmp_path)
                finally:
                    try:
                        tmp_path.unlink()
                    except Exception:
                        pass
            else:
                verified = self.formal_verifier.verify(path)
            if not verified:
                if target_region is not None:
                    cur_lines = path.read_text(encoding="utf-8").splitlines()
                    orig_lines = original.splitlines()
                    start = max(target_region.start_line - 1, 0)
                    end = min(target_region.end_line, len(orig_lines))
                    cur_lines[start:end] = orig_lines[start:end]
                    path.write_text("\n".join(cur_lines) + "\n", encoding="utf-8")
                else:
                    path.write_text(original, encoding="utf-8")
                ci_result = self._run_ci(path)
                reverted = True
                after_roi = before_roi
                after_err = before_err
                after_complexity = before_complexity
                pred_after_roi = pred_before_roi
                pred_after_err = pred_before_err
                roi_delta = 0.0
                complexity_delta = 0.0
                patch_id: int | None = None
                if self.patch_db:
                    try:
                        patch_id = self.patch_db.add(
                            PatchRecord(
                                filename=str(path),
                                description=description,
                                roi_before=before_roi,
                                roi_after=after_roi,
                                errors_before=before_err,
                                errors_after=after_err,
                                roi_delta=roi_delta,
                                complexity_before=before_complexity,
                                complexity_after=after_complexity,
                                complexity_delta=complexity_delta,
                                predicted_roi=pred_after_roi,
                                predicted_errors=pred_after_err,
                                reverted=True,
                                trending_topic=trending_topic,
                                parent_patch_id=parent_patch_id,
                                reason=reason,
                                trigger=trigger,
                                outcome="FAIL",
                                prompt_headers=json.dumps(
                                    self._last_prompt_metadata.get("headers", [])
                                ),
                                prompt_order=json.dumps(
                                    self._last_prompt_metadata.get("example_order", [])
                                ),
                                prompt_tone=self._last_prompt_metadata.get("tone", ""),
                            )
                        )
                    except Exception:
                        patch_id = None
                patch_key = str(patch_id) if patch_id is not None else description
                if patch_id is not None and self.rollback_mgr:
                    if target_region is not None:
                        self.rollback_mgr.rollback_region(
                            str(path),
                            target_region.start_line,
                            target_region.end_line,
                            requesting_bot=requesting_bot,
                        )
                    else:
                        self.rollback_mgr.rollback(patch_key, requesting_bot=requesting_bot)
                self.logger.info(
                    "patch result",
                    extra={
                        "path": str(path),
                        "patch_id": patch_id,
                        "reverted": True,
                        "roi_delta": roi_delta,
                        "success": False,
                        "tags": [FEEDBACK],
                    },
                )
                self._store_patch_memory(
                    path,
                    description,
                    generated_code,
                    False,
                    roi_delta,
                    target_region=target_region,
                )
                if self.patch_db and session_id and vectors and patch_id is not None:
                    try:
                        self.patch_db.record_vector_metrics(
                            session_id,
                            [(o, v) for o, v, _ in vectors],
                            patch_id=patch_id,
                            contribution=0.0,
                            roi_delta=roi_delta,
                            win=False,
                            regret=True,
                            effort_estimate=effort_estimate,
                        )
                    except Exception:
                        self.logger.exception("failed to log patch outcome")
                if self.data_bot:
                    try:
                        pid = str(patch_id) if patch_id is not None else description
                        self.data_bot.db.log_patch_outcome(
                            pid,
                            False,
                            [(o, v) for o, v, _ in vectors],
                            session_id=session_id,
                            reverted=True,
                        )
                    except Exception:
                        self.logger.exception("failed to log patch outcome")
                self._track_contributors(
                    session_id,
                    vectors,
                    False,
                    patch_id=patch_id,
                    retrieval_metadata=retrieval_metadata,
                    roi_delta=roi_delta,
                )
                self._log_attempt(
                    requesting_bot,
                    "apply_patch_result",
                    {"path": str(path), "success": False, "patch_id": patch_id},
                )
                roi_meta = {
                    "coverage_delta": (0.0 - (baseline_coverage or 0.0)),
                }
                self._log_prompt_evolution(
                    generated_code,
                    False,
                    ci_result,
                    roi_delta,
                    0.0,
                    roi_meta,
                    baseline_runtime=baseline_runtime,
                    module=str(path),
                    action=description,
                )
                self._record_prompt_metadata(False)
                return patch_id, True, roi_delta
        ci_result = self._run_ci(path)
        if not ci_result:
            self.logger.error("CI checks failed; skipping commit")
            if target_region is not None:
                cur_lines = path.read_text(encoding="utf-8").splitlines()
                orig_lines = original.splitlines()
                start = max(target_region.start_line - 1, 0)
                end = min(target_region.end_line, len(orig_lines))
                cur_lines[start:end] = orig_lines[start:end]
                path.write_text("\n".join(cur_lines) + "\n", encoding="utf-8")
            else:
                path.write_text(original, encoding="utf-8")
            self._run_ci(path)
            self._store_patch_memory(
                path,
                description,
                generated_code,
                False,
                0.0,
                target_region=target_region,
            )
            if self.patch_db and session_id and vectors:
                try:
                    self.patch_db.record_vector_metrics(
                        session_id,
                        [(o, v) for o, v, _ in vectors],
                        patch_id=0,
                        contribution=0.0,
                        roi_delta=0.0,
                        win=False,
                        regret=True,
                        effort_estimate=effort_estimate,
                    )
                except Exception:
                    self.logger.exception("failed to log patch outcome")
            if self.data_bot:
                try:
                    self.data_bot.db.log_patch_outcome(
                        description,
                        False,
                        [(o, v) for o, v, _ in vectors],
                        session_id=session_id,
                    )
                except Exception:
                    self.logger.exception("failed to log patch outcome")
            self._track_contributors(
                session_id,
                vectors,
                False,
                retrieval_metadata=retrieval_metadata,
                roi_delta=0.0,
            )
            self._log_attempt(
                requesting_bot,
                "apply_patch_result",
                {"path": str(path), "success": False},
            )
            roi_meta = {
                "coverage_delta": (0.0 - (baseline_coverage or 0.0)),
            }
            self._log_prompt_evolution(
                generated_code,
                False,
                ci_result,
                0.0,
                0.0,
                roi_meta,
                baseline_runtime=baseline_runtime,
                module=str(path),
                action=description,
            )
            self._record_prompt_metadata(False)
            return None, False, 0.0
        if self.safety_monitor and not self.safety_monitor.validate_bot(self.bot_name):
            if target_region is not None:
                cur_lines = path.read_text(encoding="utf-8").splitlines()
                orig_lines = original.splitlines()
                start = max(target_region.start_line - 1, 0)
                end = min(target_region.end_line, len(orig_lines))
                cur_lines[start:end] = orig_lines[start:end]
                path.write_text("\n".join(cur_lines) + "\n", encoding="utf-8")
            else:
                path.write_text(original, encoding="utf-8")
            self._run_ci(path)
            self._store_patch_memory(
                path,
                description,
                generated_code,
                False,
                0.0,
                target_region=target_region,
            )
            if self.patch_db and session_id and vectors:
                try:
                    self.patch_db.record_vector_metrics(
                        session_id,
                        [(o, v) for o, v, _ in vectors],
                        patch_id=0,
                        contribution=0.0,
                        roi_delta=0.0,
                        win=False,
                        regret=True,
                        effort_estimate=effort_estimate,
                    )
                except Exception:
                    self.logger.exception("failed to log patch outcome")
            if self.data_bot:
                try:
                    self.data_bot.db.log_patch_outcome(
                        description,
                        False,
                        [(o, v) for o, v, _ in vectors],
                        session_id=session_id,
                    )
                except Exception:
                    self.logger.exception("failed to log patch outcome")
            self._track_contributors(
                session_id,
                vectors,
                False,
                retrieval_metadata=retrieval_metadata,
                roi_delta=0.0,
            )
            self._log_attempt(
                requesting_bot,
                "apply_patch_result",
                {"path": str(path), "success": False},
            )
            roi_meta = {
                "coverage_delta": (0.0 - (baseline_coverage or 0.0)),
            }
            self._log_prompt_evolution(
                generated_code,
                False,
                ci_result,
                0.0,
                0.0,
                roi_meta,
                baseline_runtime=baseline_runtime,
                module=str(path),
                action=description,
            )
            self._record_prompt_metadata(False)
            return None, False, 0.0
        if self.pipeline:
            try:
                self.pipeline.run(self.bot_name)
            except Exception as exc:
                self.logger.error("pipeline run failed: %s", exc)
        # Extract a simple test coverage ratio from pytest output
        coverage = 0.0
        try:
            passed_match = re.search(r"(\d+)\s+passed", ci_result.stdout)
            failed_match = re.search(r"(\d+)\s+failed", ci_result.stdout)
            passed = int(passed_match.group(1)) if passed_match else 0
            failed = int(failed_match.group(1)) if failed_match else 0
            total = passed + failed
            coverage = passed / total if total else 0.0
        except Exception:
            coverage = 1.0 if ci_result.success else 0.0

        after_roi = before_roi
        after_complexity = before_complexity
        if self.data_bot:
            try:
                after_roi = self.data_bot.roi(self.bot_name)
            except Exception as exc:
                self.logger.error("roi query failed: %s", exc)
            try:
                after_complexity = self.data_bot.complexity_score(self.data_bot.db.fetch(20))
            except Exception as exc:
                self.logger.error("complexity query failed: %s", exc)
                after_complexity = before_complexity
        after_err = self._current_errors()
        pred_after_roi = pred_before_roi
        pred_after_err = pred_before_err
        if self.trend_predictor:
            try:
                self.trend_predictor.train()
                pred = self.trend_predictor.predict_future_metrics(1)
                pred_after_roi = pred.roi
                pred_after_err = pred.errors
            except Exception as exc:
                self.logger.exception("trend prediction failed", exc_info=True)
                if self.event_bus:
                    try:
                        self.event_bus.publish(
                            "patch:error", {"stage": "predict_after", "error": str(exc)}
                        )
                    except Exception:
                        self.logger.exception("event bus publish failed")
                pred_after_roi = pred_before_roi
                pred_after_err = pred_before_err
        roi_delta = after_roi - before_roi
        roi_deltas_map: Mapping[str, float] | None = None
        if self.roi_tracker:
            try:
                self.roi_tracker.update(before_roi, after_roi)
                if getattr(self.roi_tracker, "roi_history", None):
                    roi_delta = float(self.roi_tracker.roi_history[-1])
                roi_deltas_map = self.roi_tracker.origin_db_deltas()
            except Exception:
                self.logger.exception("roi tracker update failed")
        complexity_delta = after_complexity - before_complexity
        err_delta = after_err - before_err
        pred_roi_delta = pred_after_roi - pred_before_roi
        pred_err_delta = pred_after_err - pred_before_err

        # Compare current deltas against historical baselines
        tracker = self.baseline_tracker
        roi_mean = tracker.get("roi_delta")
        roi_std = tracker.std("roi_delta")
        err_mean = tracker.get("error_delta")
        err_std = tracker.std("error_delta")
        comp_mean = tracker.get("complexity_delta")
        comp_std = tracker.std("complexity_delta")
        proi_mean = tracker.get("pred_roi_delta")
        proi_std = tracker.std("pred_roi_delta")
        perr_mean = tracker.get("pred_err_delta")
        perr_std = tracker.std("pred_err_delta")

        roi_drop = roi_delta < (roi_mean - roi_std)
        err_spike = err_delta > (err_mean + err_std)
        comp_spike = complexity_delta > (comp_mean + comp_std)
        proi_drop = pred_roi_delta < (proi_mean - proi_std)
        perr_spike = pred_err_delta > (perr_mean + perr_std)

        reverted = False
        if roi_drop or err_spike or comp_spike or proi_drop or perr_spike:
            if target_region is not None:
                cur_lines = path.read_text(encoding="utf-8").splitlines()
                orig_lines = original.splitlines()
                start = max(target_region.start_line - 1, 0)
                end = min(target_region.end_line, len(orig_lines))
                cur_lines[start:end] = orig_lines[start:end]
                path.write_text("\n".join(cur_lines) + "\n", encoding="utf-8")
            else:
                path.write_text(original, encoding="utf-8")
            self._run_ci(path)
            reverted = True

        # Update baselines after evaluating revert condition
        try:
            tracker.update(
                roi_delta=roi_delta,
                error_delta=err_delta,
                complexity_delta=complexity_delta,
                pred_roi_delta=pred_roi_delta,
                pred_err_delta=pred_err_delta,
            )
        except Exception:
            self.logger.exception("baseline tracker update failed")
        patch_id: int | None = None
        if self.patch_db:
            try:
                patch_id = self.patch_db.add(
                    PatchRecord(
                        filename=str(path),
                        description=description,
                        roi_before=before_roi,
                        roi_after=after_roi,
                        errors_before=before_err,
                        errors_after=after_err,
                        roi_delta=roi_delta,
                        complexity_before=before_complexity,
                        complexity_after=after_complexity,
                        complexity_delta=complexity_delta,
                        predicted_roi=pred_after_roi,
                        predicted_errors=pred_after_err,
                        reverted=reverted,
                        trending_topic=trending_topic,
                        parent_patch_id=parent_patch_id,
                        reason=reason,
                        trigger=trigger,
                        outcome="SUCCESS" if not reverted else "FAIL",
                        prompt_headers=json.dumps(
                            self._last_prompt_metadata.get("headers", [])
                        ),
                        prompt_order=json.dumps(
                            self._last_prompt_metadata.get("example_order", [])
                        ),
                        prompt_tone=self._last_prompt_metadata.get("tone", ""),
                    )
                )
            except Exception as exc:
                self.logger.exception("failed recording patch", exc_info=True)
                if self.event_bus:
                    try:
                        self.event_bus.publish(
                            "patch:error", {"stage": "record", "error": str(exc)}
                        )
                    except Exception:
                        self.logger.exception("event bus publish failed")
                patch_id = None
        patch_key = str(patch_id) if patch_id is not None else description
        branch_name = "main"
        if not reverted:
            if patch_id is not None:
                self._active_patches[patch_key] = (
                    path,
                    original,
                    session_id,
                    list(vectors),
                    target_region,
                )
                if self.rollback_mgr:
                    if target_region is not None:
                        self.rollback_mgr.register_region_patch(
                            patch_key,
                            self.bot_name,
                            str(path),
                            target_region.start_line,
                            target_region.end_line,
                        )
                    else:
                        self.rollback_mgr.register_patch(patch_key, self.bot_name)
            roi_thr = _settings.auto_merge.roi_threshold
            cov_thr = _settings.auto_merge.coverage_threshold
            if roi_delta < roi_thr or coverage < cov_thr:
                if patch_id is not None:
                    branch_name = f"review/{patch_id}"
                    try:
                        subprocess.run(
                            ["git", "push", "origin", f"HEAD:{branch_name}"],
                            check=True,
                        )
                    except Exception:
                        self.logger.exception("review branch update failed")
            else:
                try:
                    subprocess.run(["git", "push", "origin", "HEAD:main"], check=True)
                except Exception as exc:
                    self.logger.exception("git push failed: %s", exc)
            try:
                from sandbox_runner import post_round_orphan_scan
                post_round_orphan_scan(Path.cwd(), router=self.router)
            except Exception:
                self.logger.exception(
                    "post_round_orphan_scan after apply_patch failed"
                )
            if patch_id is not None:
                try:
                    record_patch_metadata(
                        patch_id,
                        {"branch": branch_name, "roi_delta": roi_delta, "coverage": coverage},
                        patch_db=self.patch_db,
                    )
                except Exception:
                    self.logger.exception("failed to record patch metadata")
        elif patch_id is not None and self.rollback_mgr:
            if target_region is not None:
                self.rollback_mgr.rollback_region(
                    str(path),
                    target_region.start_line,
                    target_region.end_line,
                    requesting_bot=requesting_bot,
                )
            else:
                self.rollback_mgr.rollback(patch_key, requesting_bot=requesting_bot)
        if (
            self.patch_suggestion_db
            and not reverted
            and roi_delta > 0
        ):
            try:
                self.patch_suggestion_db.add(
                    SuggestionRecord(
                        module=resolve_path(path).name, description=description
                    )
                )
            except Exception:
                self.logger.exception("failed storing suggestion")
        self.logger.info(
            "patch result",
            extra={
                "path": str(path),
                "patch_id": patch_id,
                "reverted": reverted,
                "roi_delta": roi_delta,
                "success": not reverted,
                "tags": [FEEDBACK],
            },
        )
        self._store_patch_memory(
            path,
            description,
            generated_code,
            not reverted,
            roi_delta,
            target_region=target_region,
        )
        coverage_delta = (
            coverage - baseline_coverage if baseline_coverage is not None else 0.0
        )
        roi_meta = {
            "coverage_delta": coverage_delta,
        }
        if roi_deltas_map:
            roi_meta["roi_deltas"] = roi_deltas_map
        self._log_prompt_evolution(
            generated_code,
            not reverted,
            ci_result,
            roi_delta,
            coverage,
            roi_meta,
            baseline_runtime=baseline_runtime,
            module=str(path),
            action=description,
        )
        try:
            if self.patch_db and session_id and patch_id is not None:
                win_flag = not reverted and roi_delta > 0
                regret_flag = reverted or roi_delta < 0
                self.patch_db.record_vector_metrics(
                    session_id,
                    [(o, v) for o, v, _ in vectors],
                    patch_id=patch_id,
                    contribution=roi_delta,
                    roi_delta=roi_delta,
                    win=win_flag,
                    regret=regret_flag,
                    effort_estimate=effort_estimate,
                )
            if self.data_bot and self.patch_db and patch_id is not None:
                rec = self.patch_db.get(patch_id)
                success = False
                rev_flag = False
                if rec:
                    success = not rec.reverted and rec.roi_delta > 0
                    rev_flag = bool(rec.reverted)
                self.data_bot.db.log_patch_outcome(
                    str(patch_id),
                    success,
                    [(o, v) for o, v, _ in vectors],
                    session_id=session_id,
                    reverted=rev_flag,
                )
        except Exception:
            self.logger.exception("failed to log patch outcome")
        if (
            suggestion_id is not None
            and self.patch_suggestion_db
            and patch_id is not None
        ):
            try:
                error_delta = 0.0
                if self.patch_db:
                    rec = self.patch_db.get(patch_id)
                    if rec:
                        error_delta = float(
                            (rec.errors_after or 0) - (rec.errors_before or 0)
                        )
                self.patch_suggestion_db.log_enhancement_outcome(
                    suggestion_id,
                    patch_id,
                    roi_delta,
                    error_delta,
                )
            except Exception:
                self.logger.exception("failed to log enhancement outcome")
        if self.cognition_layer and session_id:
            try:
                self.cognition_layer.record_patch_outcome(
                    session_id,
                    not reverted,
                    patch_id=str(patch_id or ""),
                    contribution=roi_delta,
                    effort_estimate=effort_estimate,
                )
            except VectorServiceError as exc:
                self.logger.warning("cognition layer unavailable: %s", exc)
            except Exception:
                self.logger.exception("failed to record patch outcome")
        self._track_contributors(
            session_id,
            vectors,
            bool(patch_id) and not reverted,
            patch_id=patch_id,
            retrieval_metadata=retrieval_metadata,
            roi_delta=roi_delta,
            roi_deltas=roi_deltas_map,
        )
        self._log_attempt(
            requesting_bot,
            "apply_patch_result",
            {"path": str(path), "success": not reverted, "patch_id": patch_id},
        )
        # Detailed prompt logging handled by ``_log_prompt_evolution`` above.
        # The unified ``PromptEvolutionMemory`` captures all necessary data so
        # no additional logging is required here.
        self._last_prompt = None
        self._record_prompt_metadata(not reverted)
        if not reverted and getattr(self, "prompt_engine", None):
            try:  # pragma: no cover - best effort refresh
                self.prompt_engine.refresh_trained_config()
                self.prompt_tone = getattr(
                    self.prompt_engine, "tone", self.prompt_tone
                )
            except Exception:
                self.logger.exception("failed to refresh prompt config")
        return patch_id, reverted, roi_delta

    def _build_retry_context(
        self, description: str, report: "ErrorReport" | None
    ) -> Dict[str, Any]:
        """Return retrieval metadata incorporating failure details."""

        meta: Dict[str, Any] = {}
        builder = getattr(self.cognition_layer, "context_builder", None)
        failure_obj = None
        exclude: List[str] | None = None
        if report is not None:
            from types import SimpleNamespace

            failure_obj = SimpleNamespace(
                error_type=report.tags[0] if report.tags else "",
                reproduction_steps=[report.trace],
            )
            exclude = list(report.tags)
            meta["retry_trace"] = report.trace
        if builder is None:
            return meta
        failed_tags: list[str] = []
        if self.patch_suggestion_db:
            try:
                failed_tags = self.patch_suggestion_db.failed_strategy_tags()
                builder.exclude_failed_strategies(failed_tags)
            except Exception:
                self.logger.exception("failed to apply strategy exclusions")
        try:
            ctx, sid = builder.query(
                description,
                exclude_tags=exclude,
                failure=failure_obj,
            )
            meta.update({"retrieval_context": ctx, "retrieval_session_id": sid})
        except Exception:
            meta.update({"retrieval_context": "", "retrieval_session_id": ""})
        return meta

    def _expand_region_to_function(self, path: Path, region: TargetRegion) -> TargetRegion:
        """Return ``region`` expanded to cover the entire function."""
        try:
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source)
            for node in ast.walk(tree):
                if (
                    isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and node.name == region.function
                ):
                    end = getattr(node, "end_lineno", node.lineno)
                    return TargetRegion(
                        start_line=node.lineno,
                        end_line=end,
                        function=region.function,
                    )
        except Exception:
            pass
        return region

    def apply_patch_with_retry(
        self,
        path: Path,
        description: str,
        *,
        max_attempts: int = 1,
        report: "ErrorReport" | None = None,
        **kwargs: Any,
    ) -> tuple[int | None, bool, float]:
        """Retry :meth:`apply_patch` until tests pass or ``max_attempts`` reached."""
        attempts = 0
        current = report
        failures: List[ErrorReport] = []
        context_meta = kwargs.pop("context_meta", None)
        region: TargetRegion | None = kwargs.pop("target_region", None)
        tracker = self._patch_tracker if region is not None else None
        func_region: TargetRegion | None = None
        orig_region = region
        if region is not None:
            region.filename = region.filename or str(path)
            func_region = self._expand_region_to_function(path, region)
            func_region.filename = func_region.filename or str(path)
            assert tracker is not None
            scope_level, active_region = tracker.level_for(region, func_region)
            if scope_level == "module":
                region = None
                active_region = None
                region_scope = "module"
            else:
                region_scope = {
                    "region": "line",
                    "function": "function",
                }[scope_level]
        else:
            active_region = None
            scope_level = "module"
            region_scope = "module"
        last_fp: FailureFingerprint | None = None
        warning = ""
        while attempts < max_attempts:
            warning = ""
            if attempts > 0 and last_fp is not None:
                threshold = self.failure_similarity_threshold
                description, skip, best_sim, matches, _ = check_similarity_and_warn(
                    last_fp,
                    find_similar,
                    threshold,
                    description,
                )
                self.failure_similarity_tracker.update(similarity=best_sim)
                self._save_state()
                if matches:
                    prior = matches[0]
                    warning = (
                        f"Previous similar failure '{prior.error_message}' "
                        f"in {prior.filename}:{prior.function_name}"
                    )
                    if skip or (
                        len(matches) >= self.failure_similarity_limit
                        or self.skip_retry_on_similarity
                    ):
                        details = {
                            "fingerprint_hash": getattr(last_fp, "hash", ""),
                            "similarity": best_sim,
                            "cluster_id": getattr(prior, "cluster_id", None),
                            "reason": "retry_skipped_due_to_similarity",
                        }
                        try:
                            self.audit_trail.record(details)
                        except Exception:
                            self.logger.exception("audit trail logging failed")
                        if self.patch_db:
                            try:
                                conn = self.patch_db.router.get_connection("patch_history")
                                conn.execute(
                                    "INSERT INTO patch_history(filename, description, outcome) "
                                    "VALUES(?,?,?)",
                                    (str(path), json.dumps(details), "retry_skipped"),
                                )
                                conn.commit()
                            except Exception:
                                self.logger.exception("failed to record retry status")
                        break
                    description = description + f"\n\nWARNING: {warning}"
                    try:
                        self.audit_trail.record({"retry_adjusted": warning})
                    except Exception:
                        self.logger.exception("audit trail logging failed")
                    if self.patch_db:
                        try:
                            conn = self.patch_db.router.get_connection("patch_history")
                            conn.execute(
                                "INSERT INTO patch_history(filename, description, outcome) "
                                "VALUES(?,?,?)",
                                (str(path), description, "retry_adjusted"),
                            )
                            conn.commit()
                        except Exception:
                            self.logger.exception("failed to record retry status")
            attempts += 1
            _PATCH_ATTEMPTS.labels(scope=region_scope).inc()
            meta = context_meta if attempts == 1 and context_meta is not None else None
            if current or meta is None:
                meta = self._build_retry_context(description, current)
            if warning:
                meta = {**(meta or {}), "warning": warning}
            try:
                pid, reverted, delta = self.apply_patch(
                    path,
                    description,
                    context_meta=meta,
                    target_region=active_region,
                    **kwargs,
                )
            except Exception as exc:
                trace = traceback.format_exc()
                roi_val = 0.0
                self._log_prompt_evolution(
                    "",
                    False,
                    {"error": str(exc), "trace": trace},
                    0.0,
                    0.0,
                    module=str(path),
                    action=description,
                )
                self._record_prompt_metadata(False)
                if log_prompt_attempt and self.data_bot:
                    try:
                        roi_val = self.data_bot.roi(self.bot_name)
                    except Exception:
                        pass
                if log_prompt_attempt:
                    try:
                        log_prompt_attempt(
                            getattr(self, "_last_prompt", None),
                            False,
                            {"error": str(exc), "trace": trace},
                            {"roi": roi_val, "roi_delta": 0.0},
                            failure_reason=str(exc),
                            sandbox_metrics={"roi": roi_val},
                        )
                    except Exception:
                        self.logger.exception("log_prompt_attempt failed")
                raise
            if pid is not None:
                if failures and self.patch_db:
                    try:
                        self.patch_db.record_vector_metrics(
                            "",
                            [],
                            patch_id=pid,
                            contribution=0.0,
                            roi_delta=delta,
                            win=not reverted,
                            regret=reverted,
                            errors=[{"trace": f.trace, "tags": f.tags} for f in failures],
                            error_trace_count=len(failures),
                        )
                    except Exception:
                        self.logger.exception("failed to record failure traces")
                if log_prompt_attempt:
                    roi_meta: Dict[str, Any] = {"roi_delta": delta}
                    if self.patch_db:
                        try:
                            conn = self.patch_db.router.get_connection("patch_history")
                            row = conn.execute(
                                "SELECT tests_passed FROM patch_history WHERE id=?",
                                (pid,),
                            ).fetchone()
                            if row:
                                roi_meta["tests_passed"] = bool(row[0])
                        except Exception:
                            pass
                    exec_res: Dict[str, Any] = {"patch_id": pid, "reverted": reverted}
                    if failures:
                        exec_res["failures"] = [f.trace for f in failures]
                    try:
                        log_prompt_attempt(
                            getattr(self, "_last_prompt", None),
                            not reverted,
                            exec_res,
                            roi_meta,
                            failure_reason="reverted" if reverted else None,
                            sandbox_metrics=roi_meta if reverted else None,
                        )
                    except Exception:
                        self.logger.exception("log_prompt_attempt failed")
                self.logger.info(
                    "apply_patch_with_retry success",
                    extra={
                        "patch_id": pid,
                        "roi_delta": delta,
                        "reverted": reverted,
                    },
                )
                if tracker and orig_region is not None:
                    tracker.reset(orig_region)
                return pid, reverted, delta
            trace = self._last_retry_trace or ""
            if self._failure_cache.seen(trace):
                break
            current = parse_failure(trace)
            self._failure_cache.add(current)
            failures.append(current)
            try:
                self.audit_trail.record(
                    {"failure_trace": current.trace, "tags": current.tags}
                )
            except Exception:
                self.logger.exception("audit trail logging failed")
            error_msg = ""
            m_err = re.findall(r'([\w.]+(?:Error|Exception):.*)', trace)
            if m_err:
                error_msg = m_err[-1]
            lineno = 0
            m_loc = re.findall(r'File "([^"]+)", line (\d+)', trace)
            if m_loc:
                for fname, ln in m_loc:
                    if Path(fname).name == path.name:
                        lineno = int(ln)
                        break
                else:
                    lineno = int(m_loc[-1][1])
            function_name = "<module>"
            if lineno:
                try:
                    tree = ast.parse(path.read_text(encoding="utf-8"))
                    for node in ast.walk(tree):
                        if isinstance(
                            node, (ast.FunctionDef, ast.AsyncFunctionDef)
                        ) and hasattr(node, "lineno"):
                            end = getattr(node, "end_lineno", node.lineno)
                            if node.lineno <= lineno <= end:
                                function_name = node.name
                                break
                except Exception:
                    pass
            fp = FailureFingerprint.from_failure(
                str(path),
                function_name,
                trace,
                error_msg,
                getattr(self, "_last_prompt", ""),
            )
            record_failure(fp, log_fingerprint)
            last_fp = fp
            if tracker and orig_region is not None and func_region is not None:
                tracker.record_failure(scope_level, orig_region, func_region)
                scope_level, active_region = tracker.level_for(orig_region, func_region)
                if scope_level == "module":
                    region = None
                    active_region = None
                    region_scope = "module"
                else:
                    region_scope = {
                        "region": "line",
                        "function": "function",
                    }[scope_level]
        roi_val = 0.0
        if self.data_bot:
            try:
                roi_val = self.data_bot.roi(self.bot_name)
            except Exception:
                pass
        if log_prompt_attempt:
            exec_res: Dict[str, Any] = {"failures": [f.trace for f in failures]}
            try:
                log_prompt_attempt(
                    getattr(self, "_last_prompt", None),
                    False,
                    exec_res,
                    {"roi": roi_val, "roi_delta": 0.0},
                    failure_reason=None,
                    sandbox_metrics={"roi": roi_val},
                )
            except Exception:
                self.logger.exception("log_prompt_attempt failed")
        self.logger.error(
            "apply_patch_with_retry failed",
            extra={"roi": roi_val, "failures": [f.trace for f in failures]},
        )
        return None, False, 0.0

    def rollback_patch(self, patch_id: str) -> None:
        """Revert a previously applied patch identified by *patch_id*."""
        info = self._active_patches.pop(patch_id, None)
        if not info:
            return
        path, original, session_id, vectors, region = info
        if region is not None:
            cur_lines = path.read_text(encoding="utf-8").splitlines()
            orig_lines = original.splitlines()
            start = max(region.start_line - 1, 0)
            end = min(region.end_line, len(orig_lines))
            cur_lines[start:end] = orig_lines[start:end]
            path.write_text("\n".join(cur_lines) + "\n", encoding="utf-8")
        else:
            path.write_text(original, encoding="utf-8")
        self._run_ci(path)
        try:
            if self.data_bot:
                self.data_bot.db.log_patch_outcome(
                    patch_id,
                    False,
                    [(o, v) for o, v, _ in vectors],
                    session_id=session_id,
                    reverted=True,
                )
        except Exception:
            self.logger.exception("failed to log patch rollback")

    def top_patches(self, limit: int = 5) -> Iterable[PatchRecord]:
        """Return patches with highest ROI improvement."""
        if not self.patch_db:
            return []
        try:
            return self.patch_db.top_patches(limit)
        except Exception as exc:
            self.logger.exception("failed fetching top patches", exc_info=True)
            if self.event_bus:
                try:
                    self.event_bus.publish(
                        "patch:error", {"stage": "top_patches", "error": str(exc)}
                    )
                except Exception:
                    self.logger.exception("event bus publish failed")
            return []

    def refactor_worst_bot(self, data_bot: "DataBot", root_dir: Path | str = ".") -> None:
        """Patch the file of the bot with the highest error rate."""
        bot = data_bot.worst_bot("errors")
        if not bot:
            return
        root_path = resolve_path(str(root_dir))
        try:
            path = resolve_path(root_path / f"{bot}.py")
        except FileNotFoundError:
            return
        self.apply_patch(path, "refactor", reason="refactor", trigger="refactor_worst_bot")


__all__ = ["SelfCodingEngine", "TargetRegion"]
