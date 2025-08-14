"""Self-coding engine that retrieves code snippets and proposes patches."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Dict, List
import subprocess
import os
import sys
import json
import base64
import logging
import ast
from datetime import datetime

from .code_database import CodeDB, CodeRecord, PatchHistoryDB, PatchRecord
from .unified_event_bus import UnifiedEventBus
from .trend_predictor import TrendPredictor
from .menace_memory_manager import MenaceMemoryManager
from .safety_monitor import SafetyMonitor
from .advanced_error_management import FormalVerifier
from .chatgpt_idea_bot import ChatGPTClient
from .gpt_memory import GPTMemoryManager
from .rollback_manager import RollbackManager
from .audit_trail import AuditTrail
from .access_control import READ, WRITE, check_permission
from .patch_suggestion_db import PatchSuggestionDB, SuggestionRecord
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - type hints
    from .model_automation_pipeline import ModelAutomationPipeline
    from .data_bot import DataBot

# Allow overriding the visual agent prompt through environment variables
VA_PROMPT_TEMPLATE = os.getenv("VA_PROMPT_TEMPLATE")
VA_PROMPT_PREFIX = os.getenv("VA_PROMPT_PREFIX", "")
VA_REPO_LAYOUT_LINES = int(os.getenv("VA_REPO_LAYOUT_LINES", "20"))


class SelfCodingEngine:
    """Generate new helper code based on existing snippets."""

    def __init__(
        self,
        code_db: CodeDB,
        memory_mgr: MenaceMemoryManager,
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
        bot_roles: Optional[Dict[str, str]] = None,
        audit_trail_path: str | None = None,
        audit_privkey: bytes | None = None,
        event_bus: UnifiedEventBus | None = None,
        gpt_memory_manager: GPTMemoryManager | None = None,
    ) -> None:
        self.code_db = code_db
        self.memory_mgr = memory_mgr
        self.gpt_memory_manager = gpt_memory_manager or GPTMemoryManager()
        self.pipeline = pipeline
        self.data_bot = data_bot
        self.patch_db = patch_db
        self.trend_predictor = trend_predictor
        self.bot_name = bot_name
        self.safety_monitor = safety_monitor
        if llm_client is None and os.getenv("OPENAI_API_KEY"):
            llm_client = ChatGPTClient(os.getenv("OPENAI_API_KEY", ""))
        self.llm_client = llm_client
        self.rollback_mgr = rollback_mgr
        if formal_verifier is None:
            try:
                formal_verifier = FormalVerifier()
            except Exception:  # pragma: no cover - optional dependency missing
                formal_verifier = None
        self.formal_verifier = formal_verifier
        self._active_patches: dict[str, tuple[Path, str]] = {}
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
        """Record the outcome of a patch operation in GPT memory."""
        outcome = "patch_success" if success else "patch_failure"
        summary = f"roi_delta={roi_delta:.4f}"
        try:
            self.gpt_memory_manager.log_interaction(
                f"{path}:{description}",
                f"{code.strip()}\n\n{summary}",
                tags=[outcome],
            )
        except Exception:
            self.logger.exception("memory logging failed")

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
                "",
                "### Snippet context",
                context.strip(),
            ]
            body = "\n".join(lines).strip() + "\n"

        prefix = VA_PROMPT_PREFIX
        if prefix:
            if not prefix.endswith("\n"):
                prefix += "\n"
            body = prefix + body

        return body

    def generate_helper(self, description: str, *, path: Path | None = None) -> str:
        """Create helper text by asking an LLM using snippet context."""
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
        prompt = self.build_visual_agent_prompt(
            str(path) if path else None,
            description,
            context,
            repo_layout=repo_layout,
        )

        # Incorporate past patch outcomes from memory
        history = ""
        try:
            entries = self.gpt_memory_manager.search_context(
                description,
                tags=["patch_success", "patch_failure"],
                limit=5,
                use_embeddings=False,
            )
            if entries:
                summaries: List[str] = []
                for ent in entries:
                    tag = "patch_success" if "patch_success" in ent.tags else "patch_failure"
                    snippet = (ent.response or "").strip().splitlines()[0]
                    summaries.append(f"{tag}: {snippet}")
                history = "\n".join(summaries)
        except Exception:
            history = ""
        if history:
            prompt += "\n\n### Patch history\n" + history

        try:
            data = self.llm_client.ask(
                [{"role": "user", "content": prompt}],
                memory_manager=self.gpt_memory_manager,
                tags=["code_fix"],
                use_memory=True,
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
                    "tags": ["fix_attempt"],
                    "suggestion": text,
                    "description": description,
                    "path": str(path) if path else None,
                },
            )
            return text + ("\n" if not text.endswith("\n") else "")
        return _fallback()

    def patch_file(self, path: Path, description: str) -> str:
        """Append a generated helper to the given file and return its code."""
        code = self.generate_helper(description, path=path)
        self.logger.info(
            "patch file",
            extra={
                "path": str(path),
                "description": description,
                "tags": ["fix_attempt"],
            },
        )
        with open(path, "a", encoding="utf-8") as fh:
            fh.write("\n" + code)
        self.memory_mgr.store(str(path), code, tags="code")
        self.logger.info(
            "patch applied",
            extra={
                "path": str(path),
                "description": description,
                "tags": ["fix_result"],
                "success": True,
            },
        )
        return code

    def _run_ci(self, path: Path | None = None) -> bool:
        """Run formal verification, linting and tests.

        Return ``True`` if all checks pass.
        """
        ok = True
        if self.formal_verifier and path is not None:
            try:
                if not self.formal_verifier.verify(path):
                    ok = False
            except Exception as exc:  # pragma: no cover - verifier issues
                self.logger.error("formal verification failed: %s", exc)
                ok = False
        try:
            subprocess.run(["ruff", "--quiet", "."], check=True)
        except Exception as exc:  # pragma: no cover - ruff optional
            self.logger.error("ruff failed: %s", exc)
            ok = False
        try:
            pytest_cmd = [
                "pytest",
                "-q",
                "--hypothesis-show-statistics",
                "--cov=menace",
                "--cov-branch",
            ]
            duration = os.getenv("MENACE_TEST_DURATION")
            if duration:
                pytest_cmd.append(f"--hypothesis-max-examples={duration}")
            subprocess.run(pytest_cmd, check=True)
        except Exception as exc:
            self.logger.error("pytest failed: %s", exc)
            ok = False
        if ok:
            self.logger.info("CI checks succeeded")
        return ok

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
    ) -> tuple[int | None, bool, float]:
        """Patch file, run CI and benchmark a workflow.

        Returns the rowid of the stored patch if available.
        """
        self.logger.info(
            "apply_patch",
            extra={
                "path": str(path),
                "description": description,
                "tags": ["fix_attempt"],
            },
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
        original = path.read_text(encoding="utf-8")
        generated_code = self.patch_file(path, description)
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
                    "tags": ["fix_result"],
                },
            )
            self._store_patch_memory(path, description, generated_code, False, roi_delta)
            return patch_id, True, roi_delta
        if not self._run_ci(path):
            self.logger.error("CI checks failed; skipping commit")
            path.write_text(original, encoding="utf-8")
            self._run_ci(path)
            self._store_patch_memory(path, description, generated_code, False, 0.0)
            return None, False, 0.0
        if self.safety_monitor and not self.safety_monitor.validate_bot(self.bot_name):
            path.write_text(original, encoding="utf-8")
            self._run_ci(path)
            self._store_patch_memory(path, description, generated_code, False, 0.0)
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
                self._active_patches[patch_key] = (path, original)
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
                "tags": ["fix_result"],
            },
        )
        self._store_patch_memory(path, description, generated_code, not reverted, roi_delta)
        return patch_id, reverted, roi_delta

    def rollback_patch(self, patch_id: str) -> None:
        """Revert a previously applied patch identified by *patch_id*."""
        info = self._active_patches.pop(patch_id, None)
        if not info:
            return
        path, original = info
        path.write_text(original, encoding="utf-8")
        self._run_ci(path)

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
