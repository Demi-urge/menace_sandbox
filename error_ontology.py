from __future__ import annotations

"""Centralised error taxonomy for Menace."""

from enum import Enum
from typing import Mapping, Type


class ErrorCategory(str, Enum):
    """High level categories for error classification."""

    SemanticBug = "semantic_bug"
    RuntimeFault = "runtime_fault"
    DependencyMismatch = "dependency_mismatch"
    LogicMisfire = "logic_misfire"
    ResourceLimit = "resource_limit"
    Timeout = "timeout"
    ExternalAPI = "external_api"
    Unknown = "unknown"

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value

    # Legacy uppercase aliases for backwards compatibility
    SEMANTIC_BUG = SemanticBug
    RUNTIME_FAULT = RuntimeFault
    DEPENDENCY_MISMATCH = DependencyMismatch
    LOGIC_MISFIRE = LogicMisfire
    RESOURCE_LIMIT = ResourceLimit
    TIMEOUT = Timeout
    EXTERNAL_API = ExternalAPI
    UNKNOWN = Unknown


# Backwards compatibility for legacy imports
ErrorType = ErrorCategory

# Exception, keyword and module mappings for classification
EXCEPTION_TYPE_MAP: Mapping[Type[BaseException], ErrorCategory] = {
    AssertionError: ErrorCategory.LogicMisfire,
    KeyError: ErrorCategory.RuntimeFault,
    IndexError: ErrorCategory.RuntimeFault,
    FileNotFoundError: ErrorCategory.RuntimeFault,
    ImportError: ErrorCategory.DependencyMismatch,
    ModuleNotFoundError: ErrorCategory.DependencyMismatch,
    ZeroDivisionError: ErrorCategory.RuntimeFault,
    AttributeError: ErrorCategory.RuntimeFault,
    MemoryError: ErrorCategory.ResourceLimit,
    TimeoutError: ErrorCategory.Timeout,
    ConnectionError: ErrorCategory.ExternalAPI,
    OSError: ErrorCategory.DependencyMismatch,
    TypeError: ErrorCategory.SemanticBug,
    ValueError: ErrorCategory.SemanticBug,
}

KEYWORD_MAP: Mapping[str, ErrorCategory] = {
    "dependency missing": ErrorCategory.DependencyMismatch,
    "module not found": ErrorCategory.DependencyMismatch,
    "missing dependency": ErrorCategory.DependencyMismatch,
    "no module named": ErrorCategory.DependencyMismatch,
    "cannot import name": ErrorCategory.DependencyMismatch,
    "dependency conflict": ErrorCategory.DependencyMismatch,
    "version conflict": ErrorCategory.DependencyMismatch,
    "not implemented": ErrorCategory.LogicMisfire,
    "assertion failed": ErrorCategory.LogicMisfire,
    "division by zero": ErrorCategory.RuntimeFault,
    "zero division": ErrorCategory.RuntimeFault,
    "attribute error": ErrorCategory.RuntimeFault,
    "attribute not found": ErrorCategory.RuntimeFault,
    "unexpected type": ErrorCategory.SemanticBug,
    "wrong type": ErrorCategory.SemanticBug,
    "out of memory": ErrorCategory.ResourceLimit,
    "memory limit": ErrorCategory.ResourceLimit,
    "timed out": ErrorCategory.Timeout,
    "timeout": ErrorCategory.Timeout,
    "external api": ErrorCategory.ExternalAPI,
    "service unavailable": ErrorCategory.ExternalAPI,
    "connection refused": ErrorCategory.ExternalAPI,
}

MODULE_MAP: Mapping[str, ErrorCategory] = {
    "importlib": ErrorCategory.DependencyMismatch,
    "pkg_resources": ErrorCategory.DependencyMismatch,
    "pip": ErrorCategory.DependencyMismatch,
    "psutil": ErrorCategory.ResourceLimit,
    "asyncio": ErrorCategory.Timeout,
    "requests": ErrorCategory.ExternalAPI,
}


def classify_exception(exc: Exception, stack: str) -> ErrorCategory:
    """Best effort classification of an exception.

    Parameters
    ----------
    exc:
        The raised exception instance.
    stack:
        A formatted stack trace or textual context.
    """

    # Match by explicit exception type
    for etype, category in EXCEPTION_TYPE_MAP.items():
        if isinstance(exc, etype):
            return category

    low = stack.lower()

    # Match by simple keyword search
    for phrase, category in KEYWORD_MAP.items():
        if phrase in low:
            return category

    # Match by module signals
    module = getattr(exc, "__module__", "")
    for mod, category in MODULE_MAP.items():
        if mod in module or mod in low:
            return category

    return ErrorCategory.Unknown


__all__ = [
    "ErrorCategory",
    "ErrorType",
    "classify_exception",
    "EXCEPTION_TYPE_MAP",
    "KEYWORD_MAP",
    "MODULE_MAP",
]
