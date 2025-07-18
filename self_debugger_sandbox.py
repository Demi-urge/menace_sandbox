from __future__ import annotations

"""Self-debugging workflow with sandboxed patch testing."""

import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from datetime import datetime
import json
import io
from coverage import Coverage

from .automated_debugger import AutomatedDebugger
from .self_coding_engine import SelfCodingEngine
from .audit_trail import AuditTrail
from .code_database import PatchHistoryDB, _hash_code
from .self_improvement_policy import SelfImprovementPolicy
from typing import Callable


class SelfDebuggerSandbox(AutomatedDebugger):
    """Extend AutomatedDebugger with sandbox verification."""

    def __init__(
        self,
        telemetry_db: object,
        engine: SelfCodingEngine,
        audit_trail: AuditTrail | None = None,
        policy: SelfImprovementPolicy | None = None,
        state_getter: Callable[[], tuple[int, ...]] | None = None,
    ) -> None:
        super().__init__(telemetry_db, engine)
        self.audit_trail = audit_trail or getattr(engine, "audit_trail", None)
        self.policy = policy
        self.state_getter = state_getter
        self._bad_hashes: set[str] = set()

    # ------------------------------------------------------------------
    def _coverage_percent(self, path: Path, env: dict[str, str] | None = None) -> float:
        """Run tests for *path* under coverage and return the percentage."""
        cov = Coverage()
        buf = io.StringIO()
        cov.start()
        try:
            subprocess.run(["pytest", "-q", str(path)], check=True, env=env)
        finally:
            cov.stop()
        try:
            percent = cov.report(include=[str(path)], file=buf)
        except Exception:
            percent = 0.0
        return float(percent or 0.0)

    # ------------------------------------------------------------------
    def _log_patch(
        self,
        description: str,
        result: str,
        before_cov: float | None = None,
        after_cov: float | None = None,
        *,
        coverage_delta: float | None = None,
        error_delta: float | None = None,
        roi_delta: float | None = None,
    ) -> None:
        if not self.audit_trail:
            return
        try:
            payload = json.dumps(
                {
                    "timestamp": datetime.utcnow().isoformat(),
                    "action": "sandbox_patch",
                    "description": description,
                    "result": result,
                    "coverage_before": before_cov,
                    "coverage_after": after_cov,
                    "coverage_delta": coverage_delta,
                    "error_delta": error_delta,
                    "roi_delta": roi_delta,
                },
                sort_keys=True,
            )
            self.audit_trail.record(payload)
        except Exception:
            self.logger.exception("audit trail logging failed")

    # ------------------------------------------------------------------
    def analyse_and_fix(
        self, patch_db: PatchHistoryDB | None = None, limit: int = 1
    ) -> None:  # type: ignore[override]
        """Analyse telemetry and attempt fixes with retries."""

        for _ in range(max(1, int(limit))):
            logs = list(self._recent_logs())
            if not logs:
                return
            tests = self._generate_tests(logs)
            best: dict[str, object] | None = None
            for code in tests:
                with tempfile.TemporaryDirectory() as tmp:
                    tdir = Path(tmp)
                    repo = tdir / "repo"
                    shutil.copytree(Path("."), repo, dirs_exist_ok=True)
                    test_path = repo / "test_auto.py"
                    test_path.write_text(code)
                    env = os.environ.copy()
                    env["PYTHONPATH"] = str(repo)
                    try:
                        self.engine.patch_file(test_path, "auto_debug")

                        code_hash: str | None = None
                        try:
                            with open(test_path, "rb") as fh:
                                code_hash = _hash_code(fh.read())
                        except Exception:
                            code_hash = None

                        if code_hash:
                            if code_hash in self._bad_hashes:
                                self.logger.info(
                                    "skipping known bad patch", extra={"hash": code_hash}
                                )
                                continue
                            if patch_db:
                                try:
                                    records = patch_db.by_hash(code_hash)
                                except Exception:
                                    records = []
                                if any(r.reverted or r.roi_delta <= 0 for r in records):
                                    self.logger.info(
                                        "skipping patch due to negative history",
                                        extra={"hash": code_hash},
                                    )
                                    self._bad_hashes.add(code_hash)
                                    continue

                        subprocess.run(["pytest", "-q"], cwd=str(repo), check=True, env=env)
                    except Exception:
                        self.logger.exception("sandbox tests failed")
                        continue

                    root_test = Path("test_auto.py")
                    root_test.write_text(code)
                    result = "failed"
                    before_cov = after_cov = None
                    coverage_delta = 0.0
                    error_delta = 0.0
                    roi_delta = 0.0
                    pid = None
                    try:
                        before_cov = self._coverage_percent(root_test)
                        before_err = getattr(self.engine, "_current_errors", lambda: 0)()
                        pid, reverted, roi_delta = self.engine.apply_patch(root_test, "auto_debug")
                        after_cov = self._coverage_percent(root_test)
                        after_err = getattr(self.engine, "_current_errors", lambda: 0)()
                        coverage_delta = (after_cov - before_cov) if after_cov is not None and before_cov is not None else 0.0
                        error_delta = before_err - after_err
                        result = "reverted" if reverted else "success"
                        if (
                            not reverted
                            and pid is not None
                            and getattr(self.engine, "rollback_mgr", None)
                            and (coverage_delta < 0 or error_delta < 0)
                        ):
                            try:
                                self.engine.rollback_mgr.rollback(str(pid))
                            except Exception:
                                self.logger.exception("rollback failed")
                            result = "reverted"
                            reverted = True
                        score = coverage_delta + error_delta + roi_delta if not reverted else float("-inf")
                    except Exception:
                        self.logger.exception("patch failed")
                        score = float("-inf")
                    finally:
                        self._log_patch(
                            "auto_debug_candidate",
                            result,
                            before_cov,
                            after_cov,
                            coverage_delta=coverage_delta,
                            error_delta=error_delta,
                            roi_delta=roi_delta,
                        )
                        if pid is not None and result != "reverted":
                            try:
                                self.engine.rollback_patch(str(pid))
                            except Exception:
                                self.logger.exception("candidate rollback failed")
                        root_test.unlink(missing_ok=True)

                if score > (best["score"] if best else float("-inf")):
                    best = {"score": score, "code": code}

            if not best:
                continue

            code = best["code"]
            root_test = Path("test_auto.py")
            root_test.write_text(code)
            result = "failed"
            before_cov = after_cov = None
            coverage_delta = 0.0
            error_delta = 0.0
            roi_delta = 0.0
            patched = False
            try:
                before_cov = self._coverage_percent(root_test)
                before_err = getattr(self.engine, "_current_errors", lambda: 0)()
                pid, reverted, roi_delta = self.engine.apply_patch(root_test, "auto_debug")
                if self.policy:
                    try:
                        state = self.state_getter() if self.state_getter else ()
                        self.policy.update(state, roi_delta)
                    except Exception as exc:
                        self.logger.exception("policy patch update failed", exc)
                after_cov = self._coverage_percent(root_test)
                after_err = getattr(self.engine, "_current_errors", lambda: 0)()
                coverage_delta = (after_cov - before_cov) if after_cov is not None and before_cov is not None else 0.0
                error_delta = before_err - after_err
                result = "reverted" if reverted else "success"
                if (
                    not reverted
                    and pid is not None
                    and getattr(self.engine, "rollback_mgr", None)
                    and (coverage_delta < 0 or error_delta < 0)
                ):
                    try:
                        self.engine.rollback_mgr.rollback(str(pid))
                    except Exception:
                        self.logger.exception("rollback failed")
                    result = "reverted"
                else:
                    patched = not reverted and coverage_delta >= 0
            except Exception:
                self.logger.exception("patch failed")
            finally:
                self._log_patch(
                    "auto_debug",
                    result,
                    before_cov,
                    after_cov,
                    coverage_delta=coverage_delta,
                    error_delta=error_delta,
                    roi_delta=roi_delta,
                )
                root_test.unlink(missing_ok=True)
            if patched:
                break


__all__ = ["SelfDebuggerSandbox"]
