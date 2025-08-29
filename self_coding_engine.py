"""Self-coding engine that retrieves code snippets and proposes patches."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Dict, List, Any, Tuple, Mapping
import subprocess
import os
import sys
import json
import base64
import logging
import ast
import tempfile
import py_compile
from datetime import datetime

from .code_database import CodeDB, CodeRecord, PatchHistoryDB, PatchRecord
from .unified_event_bus import UnifiedEventBus
from .trend_predictor import TrendPredictor
from gpt_memory_interface import GPTMemoryInterface
from .safety_monitor import SafetyMonitor
from .advanced_error_management import FormalVerifier
from .chatgpt_idea_bot import ChatGPTClient
try:  # pragma: no cover - allow flat imports
    from .memory_aware_gpt_client import ask_with_memory
except Exception:  # pragma: no cover - fallback for flat layout
    from memory_aware_gpt_client import ask_with_memory  # type: ignore
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
from .rollback_manager import RollbackManager
from .audit_trail import AuditTrail
from .access_control import READ, WRITE, check_permission
from .patch_suggestion_db import PatchSuggestionDB, SuggestionRecord
from typing import TYPE_CHECKING
from sandbox_runner.workflow_sandbox_runner import WorkflowSandboxRunner
from .sandbox_settings import SandboxSettings

try:  # pragma: no cover - optional dependency
    from vector_service import CognitionLayer, PatchLogger, VectorServiceError
except Exception:  # pragma: no cover - defensive fallback
    CognitionLayer = None  # type: ignore
    PatchLogger = object  # type: ignore

    class VectorServiceError(Exception):
        """Fallback VectorServiceError when vector service is unavailable."""

        pass

from .roi_tracker import ROITracker

if TYPE_CHECKING:  # pragma: no cover - type hints
    from .model_automation_pipeline import ModelAutomationPipeline
    from .data_bot import DataBot

# Load prompt configuration from settings instead of environment variables
_settings = SandboxSettings()
VA_PROMPT_TEMPLATE = _settings.va_prompt_template
VA_PROMPT_PREFIX = _settings.va_prompt_prefix
VA_REPO_LAYOUT_LINES = _settings.va_repo_layout_lines


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
        llm_client: Optional[ChatGPTClient] = None,
        rollback_mgr: Optional[RollbackManager] = None,
        formal_verifier: Optional[FormalVerifier] = None,
        patch_suggestion_db: "PatchSuggestionDB" | None = None,
        patch_logger: PatchLogger | None = None,
        cognition_layer: CognitionLayer | None = None,
        bot_roles: Optional[Dict[str, str]] = None,
        audit_trail_path: str | None = None,
        audit_privkey: bytes | None = None,
        event_bus: UnifiedEventBus | None = None,
        gpt_memory: GPTMemoryInterface | None = GPT_MEMORY_MANAGER,
        knowledge_service: GPTKnowledgeService | None = None,
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
        self.safety_monitor = safety_monitor
        if llm_client is None and os.getenv("OPENAI_API_KEY"):
            llm_client = ChatGPTClient(
                os.getenv("OPENAI_API_KEY", ""), gpt_memory=self.gpt_memory
            )
        self.llm_client = llm_client
        if self.llm_client:
            self.llm_client.gpt_memory = self.gpt_memory
        self.rollback_mgr = rollback_mgr
        if formal_verifier is None:
            try:
                formal_verifier = FormalVerifier()
            except Exception:  # pragma: no cover - optional dependency missing
                formal_verifier = None
        self.formal_verifier = formal_verifier
        self._active_patches: dict[
            str, tuple[Path, str, str, List[Tuple[str, str, float]]]
        ] = {}
        self.bot_roles: Dict[str, str] = bot_roles or {}
        path = audit_trail_path or os.getenv("AUDIT_LOG_PATH", "audit.log")
        key_b64 = audit_privkey or os.getenv("AUDIT_PRIVKEY")
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
        self.event_bus = event_bus
        self.patch_suggestion_db = patch_suggestion_db
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
                cognition_layer = CognitionLayer(patch_logger=patch_logger, roi_tracker=tracker)
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
        self.router = kwargs.get("router")

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
    ) -> None:
        """Record GPT output and its outcome for later retrieval."""
        status = "success" if success else "failure"
        summary = f"status={status},roi_delta={roi_delta:.4f}"
        try:
            self.gpt_memory.log_interaction(
                f"{path}:{description}", code.strip(), tags=[ERROR_FIX, IMPROVEMENT_PATH]
            )
            self.gpt_memory.log_interaction(
                f"{path}:{description}:result", summary, tags=[FEEDBACK]
            )
        except Exception:
            self.logger.exception("memory logging failed")

    def _track_contributors(self, *args: object, **kwargs: object) -> None:
        """Deprecated helper retained for backward compatibility."""
        return

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

    @staticmethod
    def _get_repo_layout(limit: int) -> str:
        """Return a short list of top-level Python files in the repo."""
        root = Path(__file__).resolve().parent
        files = sorted(p.name for p in root.glob("*.py"))
        lines = files[:limit]
        if len(files) > limit:
            lines.append("...")
        return "\n".join(lines)

    def build_visual_agent_prompt(
        self,
        path: str | None,
        description: str,
        context: str,
        retrieval_context: str | None = None,
        repo_layout: str | None = None,
    ) -> str:
        """Return a prompt formatted for :class:`VisualAgentClient`."""
        func = f"auto_{description.replace(' ', '_')}"

        if VA_PROMPT_TEMPLATE:
            try:
                text = Path(VA_PROMPT_TEMPLATE).read_text()
            except Exception:
                text = VA_PROMPT_TEMPLATE
            try:
                body = text.format(
                    path=path or "unknown file",
                    description=description,
                    context=context.strip(),
                    retrieval_context=(retrieval_context or ""),
                    func=func,
                )
            except Exception:
                body = text
        else:
            lines = [
                "### Introduction",
                f"Add a Python helper to `{path or 'unknown file'}` that {description}.",
                "",
                "### Functions",
                f"- `{func}(*args, **kwargs)`",
                "",
                "### Dependencies",
                "standard library",
                "",
                "### Coding standards",
                (
                    "Follow PEP8 with 4-space indents and <79 character lines. "
                    "Use Google style docstrings and inline comments for complex logic."
                ),
                "",
                "### Repository layout",
                repo_layout or self._get_repo_layout(VA_REPO_LAYOUT_LINES),
                "",
                "### Environment",
                sys.version.split()[0],
                "",
                "### Metadata",
                f"description: {description}",
                "",
                "### Version control",
                "commit all changes to git using descriptive commit messages",
                "",
                "### Testing",
                (
                    "Run `scripts/setup_tests.sh` then execute `pytest --cov`. "
                    "Report any failures."
                ),
            ]
            if retrieval_context:
                try:
                    pretty_ctx = json.dumps(json.loads(retrieval_context), indent=2)
                except Exception:
                    pretty_ctx = retrieval_context
                lines.extend(["", "### Retrieval context", pretty_ctx])
            lines.extend(["", "### Snippet context", context.strip()])
            body = "\n".join(lines).strip() + "\n"

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
    ) -> str:
        """Create helper text by asking an LLM using snippet context and retrieval context."""
        snippets = self.suggest_snippets(description, limit=3)
        context = "\n\n".join(s.code for s in snippets)
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

        if not self.llm_client:
            return _fallback()
        repo_layout = self._get_repo_layout(VA_REPO_LAYOUT_LINES)
        retrieval_context = ""
        if metadata:
            retrieval_context = str(metadata.get("retrieval_context", ""))
        prompt = self.build_visual_agent_prompt(
            str(path) if path else None,
            description,
            context,
            retrieval_context=retrieval_context,
            repo_layout=repo_layout,
        )

        # Incorporate past patch outcomes from memory
        history = ""
        try:
            entries = get_feedback(self.gpt_memory, description, limit=5)
            if entries:
                summaries: List[str] = []
                for ent in entries:
                    resp = (getattr(ent, "response", "") or "").strip()
                    tag = "success" if "status=success" in resp else "failure"
                    snippet = resp.splitlines()[0]
                    summaries.append(f"{tag}: {snippet}")
                history = "\n".join(summaries)
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
            prompt += "\n\n### Patch history\n" + combined_history

        try:
            data = ask_with_memory(
                self.llm_client,
                f"self_coding_engine.generate_helper.{path.name if path else description.replace(' ', '_')}",
                prompt,
                memory=self.gpt_memory,
                tags=[ERROR_FIX, IMPROVEMENT_PATH],
            )
        except Exception:
            data = {}
        text = (
            data.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
            .strip()
        )
        if text:
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
        self, path: Path, description: str, *, context_meta: Dict[str, Any] | None = None
    ) -> tuple[str, bool]:
        """Generate helper code and append it to ``path`` if it passes verification."""
        try:
            code = self.generate_helper(description, path=path, metadata=context_meta)
        except TypeError:
            code = self.generate_helper(description)
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
                tmp_path = Path(fh.name)
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

    def _run_ci(self, path: Path | None = None) -> bool:
        """Run verification and linting inside an isolated sandbox."""

        file_name = path.name if path else None
        test_data = {file_name: path.read_text(encoding="utf-8")} if path else None

        def workflow() -> bool:
            ok = True
            target = Path(file_name) if file_name else Path(".")
            if self.formal_verifier and path is not None:
                try:
                    if not self.formal_verifier.verify(target):
                        ok = False
                except Exception as exc:  # pragma: no cover - verifier issues
                    self.logger.error("formal verification failed: %s", exc)
                    ok = False
            try:
                py_compile.compile(str(target), doraise=True)
            except Exception as exc:
                self.logger.error("lint failed: %s", exc)
                ok = False
            return ok

        runner = WorkflowSandboxRunner()
        metrics = runner.run(workflow, test_data=test_data)
        if metrics.modules:
            result = metrics.modules[0].result
            if result:
                self.logger.info("CI checks succeeded")
            return bool(result)
        return False

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

    def apply_patch(
        self,
        path: Path,
        description: str,
        *,
        threshold: float = 0.0,
        trending_topic: str | None = None,
        parent_patch_id: int | None = None,
        reason: str | None = None,
        trigger: str | None = None,
        requesting_bot: str | None = None,
        context_meta: Dict[str, Any] | None = None,
    ) -> tuple[int | None, bool, float]:
        """Patch file, run CI and benchmark a workflow.

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
            try:
                ctx, sid = self.cognition_layer.query(description)
                context_meta = {
                    "retrieval_context": ctx,
                    "retrieval_session_id": sid,
                }
            except Exception:
                context_meta = {
                    "retrieval_context": "",
                    "retrieval_session_id": "",
                }
        original = path.read_text(encoding="utf-8")
        generated_code, pre_verified = self.patch_file(
            path, description, context_meta=context_meta
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
            path.write_text(original, encoding="utf-8")
            self._run_ci(path)
            self._store_patch_memory(path, description, generated_code, False, 0.0)
            if self.cognition_layer and session_id:
                try:
                    self.cognition_layer.record_patch_outcome(session_id, False)
                except Exception:
                    self.logger.exception("failed to record patch outcome")
            self._log_attempt(
                requesting_bot,
                "apply_patch_result",
                {"path": str(path), "success": False},
            )
            return None, False, 0.0
        if self.formal_verifier and not self.formal_verifier.verify(path):
            path.write_text(original, encoding="utf-8")
            self._run_ci(path)
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
                        )
                    )
                except Exception:
                    patch_id = None
            patch_key = str(patch_id) if patch_id is not None else description
            if patch_id is not None and self.rollback_mgr:
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
            self._store_patch_memory(path, description, generated_code, False, roi_delta)
            if self.patch_db and session_id and vectors and patch_id is not None:
                try:
                    self.patch_db.record_vector_metrics(
                        session_id,
                        [(o, v) for o, v, _ in vectors],
                        patch_id=patch_id,
                        contribution=0.0,
                        win=False,
                        regret=True,
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
            self._track_contributors(session_id, vectors, False, patch_id, retrieval_metadata)
            self._log_attempt(
                requesting_bot,
                "apply_patch_result",
                {"path": str(path), "success": False, "patch_id": patch_id},
            )
            return patch_id, True, roi_delta
        if not self._run_ci(path):
            self.logger.error("CI checks failed; skipping commit")
            path.write_text(original, encoding="utf-8")
            self._run_ci(path)
            self._store_patch_memory(path, description, generated_code, False, 0.0)
            if self.patch_db and session_id and vectors:
                try:
                    self.patch_db.record_vector_metrics(
                        session_id,
                        [(o, v) for o, v, _ in vectors],
                        patch_id=0,
                        contribution=0.0,
                        win=False,
                        regret=True,
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
            self._track_contributors(session_id, vectors, False, retrieval_metadata=retrieval_metadata)
            self._log_attempt(
                requesting_bot,
                "apply_patch_result",
                {"path": str(path), "success": False},
            )
            return None, False, 0.0
        if self.safety_monitor and not self.safety_monitor.validate_bot(self.bot_name):
            path.write_text(original, encoding="utf-8")
            self._run_ci(path)
            self._store_patch_memory(path, description, generated_code, False, 0.0)
            if self.patch_db and session_id and vectors:
                try:
                    self.patch_db.record_vector_metrics(
                        session_id,
                        [(o, v) for o, v, _ in vectors],
                        patch_id=0,
                        contribution=0.0,
                        win=False,
                        regret=True,
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
            self._track_contributors(session_id, vectors, False, retrieval_metadata=retrieval_metadata)
            self._log_attempt(
                requesting_bot,
                "apply_patch_result",
                {"path": str(path), "success": False},
            )
            return None, False, 0.0
        if self.pipeline:
            try:
                self.pipeline.run(self.bot_name)
            except Exception as exc:
                self.logger.error("pipeline run failed: %s", exc)
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
        complexity_delta = after_complexity - before_complexity
        pred_roi_delta = pred_after_roi - pred_before_roi
        pred_err_delta = pred_after_err - pred_before_err
        reverted = False
        if (
            roi_delta < -threshold
            or (after_err - before_err) > threshold
            or complexity_delta > threshold
            or pred_roi_delta < -threshold
            or pred_err_delta > threshold
        ):
            path.write_text(original, encoding="utf-8")
            self._run_ci(path)
            reverted = True
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
        if not reverted:
            if patch_id is not None:
                self._active_patches[patch_key] = (path, original, session_id, list(vectors))
                if self.rollback_mgr:
                    self.rollback_mgr.register_patch(patch_key, self.bot_name)
            try:
                subprocess.run(["./sync_git.sh"], check=True)
            except Exception as exc:
                self.logger.exception("git sync failed", exc_info=True)
                if self.event_bus:
                    try:
                        self.event_bus.publish(
                            "patch:sync_failed", {"error": str(exc)}
                        )
                    except Exception:
                        self.logger.exception("event bus publish failed")
            try:
                from sandbox_runner import post_round_orphan_scan
                post_round_orphan_scan(Path.cwd(), router=self.router)
            except Exception:
                self.logger.exception(
                    "post_round_orphan_scan after apply_patch failed"
                )
        elif patch_id is not None and self.rollback_mgr:
            self.rollback_mgr.rollback(patch_key, requesting_bot=requesting_bot)
        if (
            self.patch_suggestion_db
            and not reverted
            and roi_delta > 0
        ):
            try:
                self.patch_suggestion_db.add(
                    SuggestionRecord(module=Path(path).name, description=description)
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
        self._store_patch_memory(path, description, generated_code, not reverted, roi_delta)
        try:
            if self.patch_db and session_id and patch_id is not None:
                win_flag = not reverted and roi_delta > 0
                regret_flag = reverted or roi_delta < 0
                self.patch_db.record_vector_metrics(
                    session_id,
                    [(o, v) for o, v, _ in vectors],
                    patch_id=patch_id,
                    contribution=roi_delta,
                    win=win_flag,
                    regret=regret_flag,
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
        if self.cognition_layer and session_id:
            try:
                self.cognition_layer.record_patch_outcome(
                    session_id,
                    not reverted,
                    patch_id=str(patch_id or ""),
                    contribution=roi_delta,
                )
            except Exception:
                self.logger.exception("failed to record patch outcome")
        self._track_contributors(session_id, vectors, bool(patch_id) and not reverted, patch_id, retrieval_metadata)
        self._log_attempt(
            requesting_bot,
            "apply_patch_result",
            {"path": str(path), "success": not reverted, "patch_id": patch_id},
        )
        return patch_id, reverted, roi_delta

    def rollback_patch(self, patch_id: str) -> None:
        """Revert a previously applied patch identified by *patch_id*."""
        info = self._active_patches.pop(patch_id, None)
        if not info:
            return
        path, original, session_id, vectors = info
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
        path = Path(root_dir) / f"{bot}.py"
        if not path.exists():
            return
        self.apply_patch(path, "refactor", reason="refactor", trigger="refactor_worst_bot")


__all__ = ["SelfCodingEngine"]
