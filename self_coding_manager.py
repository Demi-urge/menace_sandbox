from __future__ import annotations

"""Manage self-coding patches and deployment cycles."""

from pathlib import Path
import logging
import subprocess
import tempfile
from typing import Dict, Any

from sandbox_runner.workflow_sandbox_runner import WorkflowSandboxRunner

from .self_coding_engine import SelfCodingEngine
from .model_automation_pipeline import ModelAutomationPipeline, AutomationResult
from .data_bot import DataBot
from .advanced_error_management import FormalVerifier, AutomatedRollbackManager
from . import mutation_logger as MutationLogger


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
    ) -> None:
        self.engine = self_coding_engine
        self.pipeline = pipeline
        self.bot_name = bot_name
        self.data_bot = data_bot
        self.approval_policy = approval_policy
        self.logger = logging.getLogger(self.__class__.__name__)
        self._last_patch_id: int | None = None
        self._last_event_id: int | None = None

    # ------------------------------------------------------------------
    def run_patch(
        self,
        path: Path,
        description: str,
        energy: int = 1,
        *,
        context_meta: Dict[str, Any] | None = None,
    ) -> AutomationResult:
        """Patch *path* then deploy using the automation pipeline."""
        if self.approval_policy and not self.approval_policy.approve(path):
            raise RuntimeError("patch approval failed")
        before_roi = self.data_bot.roi(self.bot_name) if self.data_bot else 0.0
        repo_root = Path.cwd().resolve()
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run(["git", "clone", str(repo_root), tmp], check=True)
            clone_root = Path(tmp)
            cloned_path = clone_root / path.resolve().relative_to(repo_root)
            patch_id, reverted, _ = self.engine.apply_patch(
                cloned_path,
                description,
                parent_patch_id=self._last_patch_id,
                reason=description,
                trigger=path.name,
                context_meta=context_meta,
            )

            def _run_tests() -> None:
                subprocess.run(["pytest", "-q"], check=True, cwd=str(clone_root))

            runner = WorkflowSandboxRunner()
            runner.run(_run_tests, safe_mode=True)
            path.write_text(cloned_path.read_text(encoding="utf-8"), encoding="utf-8")
        result = self.pipeline.run(self.bot_name, energy=energy)
        after_roi = self.data_bot.roi(self.bot_name) if self.data_bot else 0.0
        roi_delta = after_roi - before_roi
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
        return result


__all__ = ["SelfCodingManager", "PatchApprovalPolicy"]
