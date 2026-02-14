from __future__ import annotations

"""Human alignment warnings adapter used by self-improvement workflows."""

from time import time
from typing import Any, Optional

from human_alignment_flagger import flag_improvement
from sandbox_settings import SandboxSettings
from violation_logger import log_violation

_DEFAULT_IMPROVEMENT_WARNING_THRESHOLD = 0.5


class HumanAlignmentAgent:
    """Evaluate workflow changes and emit alignment warnings when needed."""

    def __init__(self, settings: Optional[SandboxSettings] = None) -> None:
        self.settings = settings or SandboxSettings()

    def evaluate_changes(
        self,
        workflow_changes,
        metrics,
        logs,
        commit_info=None,
    ) -> dict[str, list[dict[str, Any]]]:
        """Return structured alignment warnings and persist high-severity items."""

        warnings = flag_improvement(
            workflow_changes,
            metrics,
            logs,
            commit_info=commit_info,
            settings=self.settings,
        )

        structured_warnings: dict[str, list[dict[str, Any]]] = {
            "ethics": list(warnings.get("ethics", []) or []),
            "risk_reward": list(warnings.get("risk_reward", []) or []),
            "maintainability": list(warnings.get("maintainability", []) or []),
        }

        threshold = float(
            getattr(
                self.settings,
                "improvement_warning_threshold",
                _DEFAULT_IMPROVEMENT_WARNING_THRESHOLD,
            )
        )

        timestamp = int(time())
        for category, entries in structured_warnings.items():
            for index, warning in enumerate(entries):
                severity = int(warning.get("severity", 1) or 1)
                if severity < threshold:
                    continue
                entry_id = f"improvement_{timestamp}_{category}_{index}"
                evidence = {
                    "category": category,
                    "warning": warning,
                }
                log_violation(
                    entry_id,
                    "alignment_warning",
                    severity,
                    evidence,
                    alignment_warning=True,
                )

        return structured_warnings


__all__ = ["HumanAlignmentAgent"]
