from __future__ import annotations

"""Centralised error taxonomy for Menace."""

try:  # pragma: no cover - ensure script execution resolves package imports
    from import_compat import bootstrap as _bootstrap
except Exception:  # pragma: no cover - helper unavailable
    _bootstrap = None  # type: ignore
else:  # pragma: no cover - executed only when run as a script
    _bootstrap(__name__, __file__)

from enum import Enum
from typing import Dict, Mapping, Type

from urllib.error import HTTPError

try:  # pragma: no cover - optional dependency
    from requests.exceptions import HTTPError as RequestsHTTPError  # type: ignore
except Exception:  # pragma: no cover - requests may not be installed
    RequestsHTTPError = None  # type: ignore

try:  # pragma: no cover - support running as module or package
    from .sandbox_recovery_manager import SandboxRecoveryError
except Exception:  # pragma: no cover - module not a package
    from sandbox_recovery_manager import SandboxRecoveryError  # type: ignore


class ErrorCategory(str, Enum):
    """High level categories for error classification."""

    SemanticBug = "semantic_bug"
    RuntimeFault = "runtime_fault"
    DependencyMismatch = "dependency_mismatch"
    LogicMisfire = "logic_misfire"
    ResourceLimit = "resource_limit"
    Timeout = "timeout"
    ExternalAPI = "external_api"
    MetricBottleneck = "metric_bottleneck"
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
    METRIC_BOTTLENECK = MetricBottleneck
    UNKNOWN = Unknown


# Backwards compatibility for legacy imports
ErrorType = ErrorCategory

# Exception, keyword and module mappings for classification
EXCEPTION_TYPE_MAP: Dict[Type[BaseException], ErrorCategory] = {
    AssertionError: ErrorCategory.LogicMisfire,
    KeyError: ErrorCategory.RuntimeFault,
    IndexError: ErrorCategory.RuntimeFault,
    FileNotFoundError: ErrorCategory.RuntimeFault,
    ImportError: ErrorCategory.DependencyMismatch,
    ModuleNotFoundError: ErrorCategory.DependencyMismatch,
    ZeroDivisionError: ErrorCategory.RuntimeFault,
    AttributeError: ErrorCategory.RuntimeFault,
    PermissionError: ErrorCategory.RuntimeFault,
    MemoryError: ErrorCategory.ResourceLimit,
    TimeoutError: ErrorCategory.Timeout,
    ConnectionError: ErrorCategory.ExternalAPI,
    HTTPError: ErrorCategory.ExternalAPI,
    OSError: ErrorCategory.DependencyMismatch,
    SandboxRecoveryError: ErrorCategory.RuntimeFault,
    TypeError: ErrorCategory.SemanticBug,
    ValueError: ErrorCategory.SemanticBug,
}

if RequestsHTTPError is not None:
    EXCEPTION_TYPE_MAP[RequestsHTTPError] = ErrorCategory.ExternalAPI  # type: ignore[index]

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
    "permission denied": ErrorCategory.RuntimeFault,
    "access denied": ErrorCategory.RuntimeFault,
    "401 unauthorized": ErrorCategory.ExternalAPI,
    "403 forbidden": ErrorCategory.ExternalAPI,
    "404 not found": ErrorCategory.ExternalAPI,
    "429 too many requests": ErrorCategory.ExternalAPI,
    "500 internal server error": ErrorCategory.ExternalAPI,
    "502 bad gateway": ErrorCategory.ExternalAPI,
    "504 gateway timeout": ErrorCategory.Timeout,
    "cuda error": ErrorCategory.ResourceLimit,
    "gpu error": ErrorCategory.ResourceLimit,
    "gpu out of memory": ErrorCategory.ResourceLimit,
    "accelerator error": ErrorCategory.ResourceLimit,
    "metric bottleneck": ErrorCategory.MetricBottleneck,
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
