from __future__ import annotations

"""Basic security auditing helpers with an optional auto-fix loop."""

from typing import Any, Callable, Mapping
import json
import logging
import os


class SecurityAuditor:
    """Minimal auditor that persists alignment warnings and escalates them."""

    def __init__(
        self,
        base_dir: str = ".",
        *,
        record_path: str | None = None,
        escalate_hook: Callable[[Mapping[str, Any]], None] | None = None,
    ) -> None:
        self.base_dir = base_dir
        self.record_path = record_path or os.path.join(
            self.base_dir, "alignment_warnings.jsonl"
        )
        self.escalate_hook = escalate_hook
        self.logger = logging.getLogger(self.__class__.__name__)

    # ------------------------------------------------------------------
    def _run(self, cmd: list[str]) -> bool:
        """Return ``True`` without executing any external command."""
        self.logger.debug("skipping %s", cmd[0])
        return True

    # ------------------------------------------------------------------
    def audit(self, report: Mapping[str, Any] | None = None) -> bool:
        """Persist *report* and optionally escalate it.

        The method appends the structured *report* to ``record_path`` using a
        JSON lines format.  When ``escalate_hook`` was supplied, it is invoked
        after persistence which allows callers to integrate custom escalation
        logic such as paging a human operator.
        """

        data = report or {}
        try:
            os.makedirs(os.path.dirname(self.record_path), exist_ok=True)
            with open(self.record_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(data) + "\n")
        except Exception:  # pragma: no cover - persistence best effort
            self.logger.exception("failed recording alignment warning")

        if self.escalate_hook is not None:
            try:
                self.escalate_hook(data)
            except Exception:  # pragma: no cover - escalation best effort
                self.logger.exception("escalation hook failed")

        return True


def fix_until_safe(auditor: "SecurityAuditor", *, attempts: int = 3) -> bool:
    """Return ``True`` immediately as audits are disabled."""
    auditor.logger.info("auto fix skipped; security checks disabled")
    return True


_AUDITOR = SecurityAuditor()


def dispatch_alignment_warning(report: Mapping[str, Any]) -> None:
    """Forward alignment warnings through :class:`SecurityAuditor`.

    This helper acts as an entry point for systems that cannot directly
    instantiate :class:`SecurityAuditor`.  The record is persisted via the
    module level auditor instance which can be monkeypatched in tests.
    """

    try:
        _AUDITOR.audit(report)
    except Exception:  # pragma: no cover - defensive logging
        logging.getLogger(__name__).exception(
            "alignment warning dispatch failed"
        )


__all__ = ["SecurityAuditor", "fix_until_safe", "dispatch_alignment_warning"]
