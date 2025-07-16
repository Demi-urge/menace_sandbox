from __future__ import annotations

"""Basic security auditing helpers with an optional auto-fix loop."""

import logging
import subprocess


class SecurityAuditor:
    """Run static code scans and dependency checks."""

    def __init__(self, base_dir: str = ".") -> None:
        self.base_dir = base_dir
        self.logger = logging.getLogger(self.__class__.__name__)

    # ------------------------------------------------------------------
    def _run(self, cmd: list[str]) -> bool:
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, check=False)
            if res.returncode != 0:
                self.logger.error("%s failed", cmd[0])
                return False
        except FileNotFoundError:  # pragma: no cover - optional tools
            self.logger.warning("%s not installed", cmd[0])
        except Exception as exc:  # pragma: no cover - log only
            self.logger.error("%s failed: %s", cmd[0], exc)
            return False
        return True

    # ------------------------------------------------------------------
    def audit(self) -> bool:
        """Run bandit and safety checks."""
        ok = self._run(["bandit", "-r", self.base_dir, "-q"])
        ok = self._run(["safety", "check", "--full-report"]) and ok
        return ok


def fix_until_safe(auditor: "SecurityAuditor", *, attempts: int = 3) -> bool:
    """Attempt automated fixes until ``auditor`` succeeds."""
    from .automated_debugger import AutomatedDebugger
    from .self_coding_engine import SelfCodingEngine
    from .code_database import CodeDB
    from .menace_memory_manager import MenaceMemoryManager
    from .error_bot import ErrorDB
    import sqlite3

    logger = logging.getLogger("AutoFix")
    error_db = ErrorDB()

    class _Proxy:
        def __init__(self, db: ErrorDB) -> None:
            self.db = db

        def recent_errors(self, limit: int = 5) -> list[str]:
            with sqlite3.connect(self.db.path) as conn:
                rows = conn.execute(
                    "SELECT stack_trace FROM telemetry ORDER BY id DESC LIMIT ?",
                    (limit,),
                ).fetchall()
            return [str(r[0]) for r in rows]

    engine = SelfCodingEngine(CodeDB(), MenaceMemoryManager())
    debugger = AutomatedDebugger(_Proxy(error_db), engine)

    for _ in range(attempts):
        try:
            debugger.analyse_and_fix()
        except Exception as exc:  # pragma: no cover - best effort
            logger.error("auto fix failed: %s", exc)
        if auditor.audit():
            logger.info("codebase secure after auto fix")
            return True
    return False


__all__ = ["SecurityAuditor", "fix_until_safe"]
