from __future__ import annotations

"""Shared runtime failure classification for autonomous services."""

from dataclasses import dataclass
from enum import Enum
import logging


class RuntimeFailureReason(str, Enum):
    SELF_IMPROVEMENT_WORKER_EXIT = "SELF_IMPROVEMENT_WORKER_EXIT"
    SELF_CODING_ENGINE_CRASH = "SELF_CODING_ENGINE_CRASH"
    SERVICE_PROCESS_EXIT = "SERVICE_PROCESS_EXIT"
    TELEMETRY_FAILURE = "TELEMETRY_FAILURE"
    OPTIONAL_DEPENDENCY_FAILURE = "OPTIONAL_DEPENDENCY_FAILURE"
    INTEGRATION_OR_API_ERROR = "INTEGRATION_OR_API_ERROR"


CRITICAL_FAILURES = {
    RuntimeFailureReason.SELF_CODING_ENGINE_CRASH.value,
    RuntimeFailureReason.SELF_IMPROVEMENT_WORKER_EXIT.value,
}

NON_CRITICAL_FAILURES = {
    RuntimeFailureReason.SERVICE_PROCESS_EXIT.value,
    RuntimeFailureReason.INTEGRATION_OR_API_ERROR.value,
    RuntimeFailureReason.TELEMETRY_FAILURE.value,
    RuntimeFailureReason.OPTIONAL_DEPENDENCY_FAILURE.value,
}


@dataclass(frozen=True)
class RuntimeFailureClassification:
    category: str
    reason: str
    should_exit: bool


def classify_runtime_failure(
    *,
    reason_code: RuntimeFailureReason | str | None = None,
    component: str | None = None,
    error: BaseException | None = None,
    event: str | None = None,
) -> RuntimeFailureClassification:
    """Classify runtime failures into critical/non-critical buckets."""

    logger = logging.getLogger(__name__)
    classifications = {
        RuntimeFailureReason.SELF_CODING_ENGINE_CRASH.value: RuntimeFailureClassification(
            category="critical",
            reason=RuntimeFailureReason.SELF_CODING_ENGINE_CRASH.value,
            should_exit=True,
        ),
        RuntimeFailureReason.SELF_IMPROVEMENT_WORKER_EXIT.value: RuntimeFailureClassification(
            category="critical",
            reason=RuntimeFailureReason.SELF_IMPROVEMENT_WORKER_EXIT.value,
            should_exit=True,
        ),
        RuntimeFailureReason.SERVICE_PROCESS_EXIT.value: RuntimeFailureClassification(
            category="non_critical",
            reason=RuntimeFailureReason.SERVICE_PROCESS_EXIT.value,
            should_exit=False,
        ),
        RuntimeFailureReason.TELEMETRY_FAILURE.value: RuntimeFailureClassification(
            category="non_critical",
            reason=RuntimeFailureReason.TELEMETRY_FAILURE.value,
            should_exit=False,
        ),
        RuntimeFailureReason.OPTIONAL_DEPENDENCY_FAILURE.value: RuntimeFailureClassification(
            category="non_critical",
            reason=RuntimeFailureReason.OPTIONAL_DEPENDENCY_FAILURE.value,
            should_exit=False,
        ),
        RuntimeFailureReason.INTEGRATION_OR_API_ERROR.value: RuntimeFailureClassification(
            category="non_critical",
            reason=RuntimeFailureReason.INTEGRATION_OR_API_ERROR.value,
            should_exit=False,
        ),
    }

    if reason_code is not None:
        normalized_reason = (
            reason_code.value
            if isinstance(reason_code, RuntimeFailureReason)
            else str(reason_code)
        )
        if normalized_reason in classifications:
            return classifications[normalized_reason]
        logger.warning(
            "unknown runtime failure reason_code=%s; using legacy substring fallback",
            normalized_reason,
        )

    haystack = " ".join(
        part.lower()
        for part in (component or "", event or "", str(error or ""))
        if part
    )
    logger.warning(
        "legacy runtime failure substring classification used; please pass explicit reason_code"
    )

    if "self_coding_engine" in haystack or "self-coding engine" in haystack:
        return classifications[RuntimeFailureReason.SELF_CODING_ENGINE_CRASH.value]

    if any(
        marker in haystack
        for marker in (
            "self_coding_worker",
            "self-coding worker",
            "self_coding_manager",
            "self-coding manager",
        )
    ):
        return classifications[RuntimeFailureReason.SELF_CODING_ENGINE_CRASH.value]

    if "self_learning_service" in haystack or "self-learning service" in haystack:
        return classifications[RuntimeFailureReason.SELF_IMPROVEMENT_WORKER_EXIT.value]

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
        return classifications[RuntimeFailureReason.SELF_IMPROVEMENT_WORKER_EXIT.value]

    if "telemetry" in haystack or "exporter" in haystack:
        return classifications[RuntimeFailureReason.TELEMETRY_FAILURE.value]

    if "optional" in haystack or "dependency" in haystack:
        return classifications[RuntimeFailureReason.OPTIONAL_DEPENDENCY_FAILURE.value]

    if "api" in haystack or "integration" in haystack:
        return classifications[RuntimeFailureReason.INTEGRATION_OR_API_ERROR.value]

    return classifications[RuntimeFailureReason.SERVICE_PROCESS_EXIT.value]


__all__ = [
    "CRITICAL_FAILURES",
    "NON_CRITICAL_FAILURES",
    "RuntimeFailureReason",
    "RuntimeFailureClassification",
    "classify_runtime_failure",
]
