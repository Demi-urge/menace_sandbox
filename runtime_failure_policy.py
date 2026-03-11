from __future__ import annotations

"""Shared runtime failure classification for autonomous services."""

from dataclasses import dataclass


CRITICAL_FAILURES = {
    "self_coding_engine_failure",
    "self_improvement_cycle_failure",
    "self_learning_service_failure",
}

NON_CRITICAL_FAILURES = {
    "bot_or_service_crash",
    "integration_or_api_error",
    "telemetry_failure",
    "optional_dependency_failure",
}


@dataclass(frozen=True)
class RuntimeFailureClassification:
    category: str
    reason: str
    should_exit: bool


def classify_runtime_failure(
    *,
    component: str | None = None,
    error: BaseException | None = None,
    event: str | None = None,
) -> RuntimeFailureClassification:
    """Classify runtime failures into critical/non-critical buckets."""

    haystack = " ".join(
        part.lower()
        for part in (component or "", event or "", str(error or ""))
        if part
    )

    if "self_coding_engine" in haystack or "self-coding engine" in haystack:
        return RuntimeFailureClassification(
            category="critical",
            reason="self_coding_engine_failure",
            should_exit=True,
        )

    if any(
        marker in haystack
        for marker in (
            "self_coding_worker",
            "self-coding worker",
            "self_coding_manager",
            "self-coding manager",
        )
    ):
        return RuntimeFailureClassification(
            category="critical",
            reason="self_coding_engine_failure",
            should_exit=True,
        )

    if "self_learning_service" in haystack or "self-learning service" in haystack:
        return RuntimeFailureClassification(
            category="critical",
            reason="self_learning_service_failure",
            should_exit=True,
        )

    if any(
        marker in haystack
        for marker in (
            "self_improvement_cycle",
            "self-improvement cycle",
            "self improvement cycle",
            "self_improvement",
            "self-improvement",
            "self improvement",
        )
    ):
        return RuntimeFailureClassification(
            category="critical",
            reason="self_improvement_cycle_failure",
            should_exit=True,
        )

    if "telemetry" in haystack or "exporter" in haystack:
        return RuntimeFailureClassification(
            category="non_critical",
            reason="telemetry_failure",
            should_exit=False,
        )

    if "optional" in haystack or "dependency" in haystack:
        return RuntimeFailureClassification(
            category="non_critical",
            reason="optional_dependency_failure",
            should_exit=False,
        )

    if "api" in haystack or "integration" in haystack:
        return RuntimeFailureClassification(
            category="non_critical",
            reason="integration_or_api_error",
            should_exit=False,
        )

    return RuntimeFailureClassification(
        category="non_critical",
        reason="bot_or_service_crash",
        should_exit=False,
    )


__all__ = [
    "CRITICAL_FAILURES",
    "NON_CRITICAL_FAILURES",
    "RuntimeFailureClassification",
    "classify_runtime_failure",
]
