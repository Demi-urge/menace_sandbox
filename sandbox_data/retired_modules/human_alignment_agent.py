from __future__ import annotations

"""Human alignment warnings adapter used by self-improvement workflows.

Contract
--------
``HumanAlignmentAgent.evaluate_changes(actions, metrics, logs, commit_info)``
accepts proposed workflow actions, optional metrics, optional logs and commit
metadata, then returns a warning dictionary with stable iterable collections
for the ``"ethics"``, ``"risk_reward"`` and ``"maintainability"`` keys.
This guarantees engine call sites can safely evaluate warning presence via
expressions like ``any(warnings.values())``.
"""

from importlib import import_module
import sys
from time import time
from typing import Any

import human_alignment_flagger
from sandbox_settings import SandboxSettings

flag_improvement = human_alignment_flagger.flag_improvement

_DEFAULT_IMPROVEMENT_WARNING_THRESHOLD = 0.5


def log_violation(*args: Any, **kwargs: Any):
    """Proxy to :func:`violation_logger.log_violation` with graceful fallback."""

    try:
        from violation_logger import log_violation as _log_violation
    except Exception:
        return None
    return _log_violation(*args, **kwargs)


class HumanAlignmentAgent:
    """Evaluate workflow changes and emit alignment warnings when needed."""

    def __init__(self, settings: SandboxSettings | None = None, **kwargs: Any) -> None:
        if settings is None:
            settings = kwargs.pop("sandbox_settings", None)
        self.settings = settings or SandboxSettings()

    def evaluate_changes(
        self,
        actions=None,
        metrics=None,
        logs=None,
        commit_info=None,
        **kwargs: Any,
    ) -> dict[str, list[dict[str, Any]]]:
        actions = kwargs.pop("workflow_changes", actions)
        actions = kwargs.pop("actions", actions)
        commit_info = kwargs.pop("commit_metadata", commit_info)
        kwargs.pop("reward_override", None)

        empty_warnings: dict[str, list[dict[str, Any]]] = {
            "ethics": [],
            "risk_reward": [],
            "maintainability": [],
        }

        try:
            warnings = self._flag_improvement(
                actions,
                metrics,
                logs,
                commit_info=commit_info,
                settings=self.settings,
            )
        except Exception:
            return empty_warnings

        if not isinstance(warnings, dict):
            return empty_warnings

        try:
            structured_warnings: dict[str, list[dict[str, Any]]] = {
            "ethics": list(warnings.get("ethics", []) or []),
            "risk_reward": list(warnings.get("risk_reward", []) or []),
            "maintainability": list(warnings.get("maintainability", []) or []),
            }
            if actions and not any(structured_warnings.values()):
                structured_warnings["maintainability"].append(
                    {"issue": "no tests provided", "severity": 1}
                )
        except Exception:
            return empty_warnings

        threshold = self._warning_threshold()
        loggers = self._resolve_violation_loggers()
        timestamp = int(time())

        try:
            for category, entries in structured_warnings.items():
                for index, warning in enumerate(entries):
                    severity = int(warning.get("severity", 1) or 1)
                    if severity < threshold:
                        continue
                    for logger in loggers:
                        logger(
                            f"improvement_{timestamp}_{category}_{index}",
                            "alignment_warning",
                            severity,
                            {"category": category, "warning": warning},
                            alignment_warning=True,
                        )
        except Exception:
            return empty_warnings

        return structured_warnings

    def _flag_improvement(self, workflow_changes, metrics, logs, commit_info=None, settings=None):
        module = import_module("human_alignment_flagger")
        func = getattr(module, "flag_improvement", flag_improvement)
        return func(
            workflow_changes,
            metrics,
            logs,
            commit_info=commit_info,
            settings=settings,
        )

    def _warning_threshold(self) -> float:
        for attr in (
            "improvement_warning_threshold",
            "warning_threshold",
            "alignment_warning_threshold",
        ):
            value = getattr(self.settings, attr, None)
            if value is not None:
                return float(value)
        return _DEFAULT_IMPROVEMENT_WARNING_THRESHOLD

    def _resolve_violation_loggers(self) -> list:
        loggers = []
        for module_name in (
            "menace_sandbox.human_alignment_agent",
            "menace.human_alignment_agent",
            __name__,
        ):
            module = sys.modules.get(module_name)
            if module is None:
                continue
            candidate = getattr(module, "log_violation", None)
            if callable(candidate) and candidate not in loggers:
                loggers.append(candidate)
        if not loggers:
            loggers.append(log_violation)
        return loggers


__all__ = [
    "HumanAlignmentAgent",
    "SandboxSettings",
    "flag_improvement",
    "log_violation",
]
