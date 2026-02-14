from __future__ import annotations

"""Human alignment warnings adapter used by self-improvement workflows."""

import sys
from time import time
from typing import Any, Optional

from .human_alignment_flagger import flag_improvement
from .sandbox_settings import SandboxSettings
from . import violation_logger as _violation_logger
from .violation_logger import log_violation

_DEFAULT_IMPROVEMENT_WARNING_THRESHOLD = 0.5


class HumanAlignmentAgent:
    """Evaluate workflow changes and emit alignment warnings when needed."""

    def __init__(
        self,
        settings: Optional[SandboxSettings] = None,
        **kwargs: Any,
    ) -> None:
        # Backward compatibility: some call sites pass configuration under
        # alternate keyword names (for example ``sandbox_settings``).
        if settings is None:
            settings = kwargs.pop("sandbox_settings", None)
        self.settings = settings or SandboxSettings()

    def evaluate_changes(self, *args: Any, **kwargs: Any) -> dict[str, list[dict[str, Any]]]:
        """Return structured alignment warnings and persist high-severity items."""

        workflow_changes = kwargs.pop("workflow_changes", None)
        if workflow_changes is None:
            workflow_changes = kwargs.pop("actions", None)

        metrics = kwargs.pop("metrics", None)
        logs = kwargs.pop("logs", None)

        commit_info = kwargs.pop("commit_info", None)
        if commit_info is None:
            commit_info = kwargs.pop("commit_metadata", None)

        # Consume optional legacy/test-only argument without changing behavior.
        kwargs.pop("reward_override", None)

        if args:
            if workflow_changes is None and len(args) >= 1:
                workflow_changes = args[0]
            if metrics is None and len(args) >= 2:
                metrics = args[1]
            if logs is None and len(args) >= 3:
                logs = args[2]
            if commit_info is None and len(args) >= 4:
                commit_info = args[3]

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
                module_logger = log_violation
                compat_module = sys.modules.get("menace_sandbox.human_alignment_agent")
                if module_logger is _violation_logger.log_violation and compat_module is not None:
                    module_logger = getattr(compat_module, "log_violation", module_logger)

                module_logger(
                    entry_id,
                    "alignment_warning",
                    severity,
                    evidence,
                    alignment_warning=True,
                )

        return structured_warnings


__all__ = [
    "HumanAlignmentAgent",
    "SandboxSettings",
    "flag_improvement",
    "log_violation",
]
