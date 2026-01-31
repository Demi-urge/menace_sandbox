from __future__ import annotations

"""Centralised error taxonomy for Menace.

The legacy :class:`ErrorCategory` enum and :func:`classify_exception` helper are
deprecated in favour of :func:`classify_error`, which returns the fixed
``FixedErrorCategory`` taxonomy. A static mapping between the legacy and fixed
categories is provided to keep integrations stable during migration.
"""

try:  # pragma: no cover - ensure script execution resolves package imports
    from import_compat import bootstrap as _bootstrap
except Exception:  # pragma: no cover - helper unavailable
    from pathlib import Path
    import sys

    _bootstrap = None  # type: ignore
    _here = Path(__file__).resolve()
    for _candidate in (_here.parent, *_here.parents):
        compat_path = _candidate / "import_compat.py"
        if compat_path.exists():
            candidate_str = str(_candidate)
            if candidate_str not in sys.path:
                sys.path.insert(0, candidate_str)
            try:
                from import_compat import bootstrap as _bootstrap  # type: ignore
            except Exception:
                _bootstrap = None  # type: ignore
            else:
                break

if "_bootstrap" in globals() and _bootstrap is not None:  # pragma: no cover - script usage
    _bootstrap(__name__, __file__)

from enum import Enum
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple, Union


class FixedErrorCategory(str, Enum):
    """Fixed error taxonomy for classification."""

    SyntaxError = "SyntaxError"
    ImportError = "ImportError"
    TypeErrorMismatch = "TypeError-Mismatch"
    ContractViolation = "ContractViolation"
    EdgeCaseFailure = "EdgeCaseFailure"
    UnhandledException = "UnhandledException"
    InvalidInput = "InvalidInput"
    MissingReturn = "MissingReturn"
    ConfigError = "ConfigError"
    Other = "Other"

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value


class ErrorCategory(str, Enum):
    """Legacy error taxonomy retained for backward compatibility."""

    Unknown = "Unknown"
    RuntimeFault = "RuntimeFault"
    DependencyMismatch = "DependencyMismatch"
    LogicMisfire = "LogicMisfire"
    SemanticBug = "SemanticBug"
    ResourceLimit = "ResourceLimit"
    Timeout = "Timeout"
    ExternalAPI = "ExternalAPI"
    MetricBottleneck = "MetricBottleneck"

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value


ErrorType = ErrorCategory

_ExceptionInput = Union[BaseException, str, Mapping[str, Any]]
_Input = Union[_ExceptionInput, Sequence[_ExceptionInput]]

_EXCEPTION_TYPE_MAP: Tuple[Tuple[type[BaseException], FixedErrorCategory], ...] = (
    (SyntaxError, FixedErrorCategory.SyntaxError),
    (ImportError, FixedErrorCategory.ImportError),
    (ModuleNotFoundError, FixedErrorCategory.ImportError),
    (TypeError, FixedErrorCategory.TypeErrorMismatch),
    (AssertionError, FixedErrorCategory.ContractViolation),
    (ValueError, FixedErrorCategory.InvalidInput),
)

_PHRASE_MAP: Tuple[Tuple[str, FixedErrorCategory], ...] = (
    ("syntax error", FixedErrorCategory.SyntaxError),
    ("no module named", FixedErrorCategory.ImportError),
    ("cannot import", FixedErrorCategory.ImportError),
    ("import error", FixedErrorCategory.ImportError),
    ("typeerror", FixedErrorCategory.TypeErrorMismatch),
    ("type error", FixedErrorCategory.TypeErrorMismatch),
    ("type mismatch", FixedErrorCategory.TypeErrorMismatch),
    ("assertion failed", FixedErrorCategory.ContractViolation),
    ("contract violation", FixedErrorCategory.ContractViolation),
    ("precondition failed", FixedErrorCategory.ContractViolation),
    ("postcondition failed", FixedErrorCategory.ContractViolation),
    ("edge case", FixedErrorCategory.EdgeCaseFailure),
    ("corner case", FixedErrorCategory.EdgeCaseFailure),
    ("unhandled exception", FixedErrorCategory.UnhandledException),
    ("uncaught exception", FixedErrorCategory.UnhandledException),
    ("invalid input", FixedErrorCategory.InvalidInput),
    ("invalid argument", FixedErrorCategory.InvalidInput),
    ("bad request", FixedErrorCategory.InvalidInput),
    ("missing return", FixedErrorCategory.MissingReturn),
    ("did not return", FixedErrorCategory.MissingReturn),
    ("returned none", FixedErrorCategory.MissingReturn),
    ("missing config", FixedErrorCategory.ConfigError),
    ("configuration error", FixedErrorCategory.ConfigError),
    ("config error", FixedErrorCategory.ConfigError),
)


_CATEGORY_PRIORITY: Tuple[FixedErrorCategory, ...] = (
    FixedErrorCategory.SyntaxError,
    FixedErrorCategory.ImportError,
    FixedErrorCategory.TypeErrorMismatch,
    FixedErrorCategory.ContractViolation,
    FixedErrorCategory.EdgeCaseFailure,
    FixedErrorCategory.UnhandledException,
    FixedErrorCategory.InvalidInput,
    FixedErrorCategory.MissingReturn,
    FixedErrorCategory.ConfigError,
    FixedErrorCategory.Other,
)


LEGACY_TO_FIXED_CATEGORY: Mapping[ErrorCategory, FixedErrorCategory] = {
    ErrorCategory.Unknown: FixedErrorCategory.Other,
    ErrorCategory.RuntimeFault: FixedErrorCategory.UnhandledException,
    ErrorCategory.DependencyMismatch: FixedErrorCategory.ImportError,
    ErrorCategory.LogicMisfire: FixedErrorCategory.ContractViolation,
    ErrorCategory.SemanticBug: FixedErrorCategory.TypeErrorMismatch,
    ErrorCategory.ResourceLimit: FixedErrorCategory.UnhandledException,
    ErrorCategory.Timeout: FixedErrorCategory.UnhandledException,
    ErrorCategory.ExternalAPI: FixedErrorCategory.UnhandledException,
    ErrorCategory.MetricBottleneck: FixedErrorCategory.Other,
}

FIXED_TO_LEGACY_CATEGORY: Mapping[FixedErrorCategory, ErrorCategory] = {
    FixedErrorCategory.SyntaxError: ErrorCategory.SemanticBug,
    FixedErrorCategory.ImportError: ErrorCategory.DependencyMismatch,
    FixedErrorCategory.TypeErrorMismatch: ErrorCategory.SemanticBug,
    FixedErrorCategory.ContractViolation: ErrorCategory.LogicMisfire,
    FixedErrorCategory.EdgeCaseFailure: ErrorCategory.RuntimeFault,
    FixedErrorCategory.UnhandledException: ErrorCategory.RuntimeFault,
    FixedErrorCategory.InvalidInput: ErrorCategory.SemanticBug,
    FixedErrorCategory.MissingReturn: ErrorCategory.LogicMisfire,
    FixedErrorCategory.ConfigError: ErrorCategory.DependencyMismatch,
    FixedErrorCategory.Other: ErrorCategory.Unknown,
}


def _mapping_to_text(payload: Mapping[str, Any]) -> str:
    if not payload:
        return ""
    pairs = ", ".join(f"{key!r}: {value!r}" for key, value in payload.items())
    return f"{{{pairs}}}"


def _normalize_inputs(raw: _Input) -> Tuple[List[Union[str, BaseException]], List[str]]:
    errors: List[str] = []
    segments: List[Union[str, BaseException]] = []

    if isinstance(raw, BaseException):
        iterable: Iterable[_ExceptionInput] = (raw,)
    elif isinstance(raw, str):
        iterable = (raw,)
    elif isinstance(raw, Mapping):
        iterable = (raw,)
    elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
        iterable = raw
    else:
        iterable = (raw,)

    for item in iterable:
        if isinstance(item, BaseException):
            segments.append(item)
        elif isinstance(item, str):
            segments.append(item)
        elif isinstance(item, Mapping):
            segments.append(_mapping_to_text(item))
        else:
            errors.append(f"Unsupported input type: {type(item).__name__}")

    return segments, errors


def _segment_to_text(item: Union[str, BaseException]) -> str:
    if isinstance(item, BaseException):
        return f"{item.__class__.__name__}: {item}"
    return item


def _is_empty_input(segments: List[Union[str, BaseException]]) -> bool:
    if not segments:
        return True
    if any(isinstance(item, BaseException) for item in segments):
        return False
    return all(not str(item).strip() for item in segments)


def _classify_from_exception(exc: BaseException) -> Tuple[FixedErrorCategory, str]:
    for etype, category in _EXCEPTION_TYPE_MAP:
        if isinstance(exc, etype):
            return category, f"exception:{etype.__name__}"
    return FixedErrorCategory.Other, "exception:unmatched"


def _classify_from_text(text: str) -> Tuple[FixedErrorCategory, str]:
    if not text.strip():
        return FixedErrorCategory.Other, "text:empty"
    lowered = text.lower()
    for phrase, category in _PHRASE_MAP:
        if phrase in lowered:
            return category, f"phrase:{phrase}"
    return FixedErrorCategory.Other, "text:unmatched"


def _legacy_category_from_module(module_name: str) -> ErrorCategory:
    if not module_name:
        return ErrorCategory.Unknown
    if module_name.startswith(("pkg_resources", "importlib")):
        return ErrorCategory.DependencyMismatch
    if module_name.startswith("requests"):
        return ErrorCategory.ExternalAPI
    return ErrorCategory.Unknown


def _classify_legacy_from_exception(exc: BaseException) -> ErrorCategory:
    try:
        from urllib.error import HTTPError
    except Exception:  # pragma: no cover - stdlib should exist
        HTTPError = None  # type: ignore

    exception_map: Tuple[Tuple[type[BaseException], ErrorCategory], ...] = (
        *((HTTPError, ErrorCategory.ExternalAPI),) if HTTPError else (),
        (MemoryError, ErrorCategory.ResourceLimit),
        (TimeoutError, ErrorCategory.Timeout),
        (KeyError, ErrorCategory.RuntimeFault),
        (IndexError, ErrorCategory.RuntimeFault),
        (FileNotFoundError, ErrorCategory.RuntimeFault),
        (ZeroDivisionError, ErrorCategory.RuntimeFault),
        (AttributeError, ErrorCategory.RuntimeFault),
        (PermissionError, ErrorCategory.RuntimeFault),
        (AssertionError, ErrorCategory.LogicMisfire),
        (NotImplementedError, ErrorCategory.LogicMisfire),
        (TypeError, ErrorCategory.SemanticBug),
        (ValueError, ErrorCategory.SemanticBug),
        (ModuleNotFoundError, ErrorCategory.DependencyMismatch),
        (ImportError, ErrorCategory.DependencyMismatch),
        (OSError, ErrorCategory.DependencyMismatch),
    )

    for etype, category in exception_map:
        if isinstance(exc, etype):
            return category

    if exc.__class__.__name__ == "SandboxRecoveryError":
        return ErrorCategory.RuntimeFault

    module_category = _legacy_category_from_module(exc.__class__.__module__)
    if module_category is not ErrorCategory.Unknown:
        return module_category
    return ErrorCategory.Unknown


def _classify_legacy_from_text(text: str) -> ErrorCategory:
    if not text.strip():
        return ErrorCategory.Unknown
    lowered = text.lower()
    legacy_phrases: Tuple[Tuple[str, ErrorCategory], ...] = (
        ("dependency missing", ErrorCategory.DependencyMismatch),
        ("missing package", ErrorCategory.DependencyMismatch),
        ("module not found", ErrorCategory.DependencyMismatch),
        ("no module named", ErrorCategory.DependencyMismatch),
        ("cannot import name", ErrorCategory.DependencyMismatch),
        ("out of memory", ErrorCategory.ResourceLimit),
        ("cuda error", ErrorCategory.ResourceLimit),
        ("accelerator error", ErrorCategory.ResourceLimit),
        ("timed out", ErrorCategory.Timeout),
        ("timeout", ErrorCategory.Timeout),
        ("external api", ErrorCategory.ExternalAPI),
        ("remote server replied", ErrorCategory.ExternalAPI),
        ("forbidden", ErrorCategory.ExternalAPI),
        ("permission denied", ErrorCategory.RuntimeFault),
    )
    for phrase, category in legacy_phrases:
        if phrase in lowered:
            return category
    return ErrorCategory.Unknown


def _classify_legacy(exc: BaseException, stack: str) -> ErrorCategory:
    exc_category = _classify_legacy_from_exception(exc)
    if exc_category is not ErrorCategory.Unknown:
        return exc_category
    return _classify_legacy_from_text(stack)


def classify_error(raw: _Input) -> Dict[str, Any]:
    """Classify structured or unstructured error inputs.

    Parameters
    ----------
    raw:
        Exception instance, traceback string, log string, or list/tuple of those.
    """

    segments, errors = _normalize_inputs(raw)
    if errors:
        return {
            "status": "error",
            "data": {},
            "errors": errors,
            "meta": {
                "input_length": len(segments),
                "segment_count": len(segments),
            },
        }

    if _is_empty_input(segments):
        return {
            "status": "ok",
            "data": {
                "category": FixedErrorCategory.Other,
                "source": "text",
                "matched_rule": "text:empty",
            },
            "errors": [],
            "meta": {
                "input_length": 0,
                "segment_count": len(segments),
                "empty_input": True,
            },
        }

    classifications: List[Tuple[FixedErrorCategory, str, str]] = []
    for item in segments:
        if isinstance(item, BaseException):
            category, matched_rule = _classify_from_exception(item)
            source = "exception"
        else:
            category, matched_rule = _classify_from_text(str(item))
            source = "text"
        classifications.append((category, matched_rule, source))

    category = FixedErrorCategory.Other
    matched_rule = "unmatched"
    source = "unknown"
    for priority in _CATEGORY_PRIORITY:
        for seg_category, seg_rule, seg_source in classifications:
            if seg_category is priority:
                category = seg_category
                matched_rule = seg_rule
                source = seg_source
                break
        if category is priority:
            break

    meta: Dict[str, Any] = {
        "input_length": sum(len(_segment_to_text(segment)) for segment in segments),
        "segment_count": len(segments),
    }
    if matched_rule.startswith("phrase:"):
        meta["matched_phrase"] = matched_rule.split("phrase:", 1)[1]

    return {
        "status": "ok",
        "data": {
            "category": category,
            "source": source,
            "matched_rule": matched_rule,
        },
        "errors": [],
        "meta": meta,
    }


def classify_exception(exc: Exception, stack: str) -> ErrorCategory:
    """Legacy exception classification that delegates to the fixed taxonomy."""

    result = classify_error([exc, stack])
    data = result.get("data", {})
    fixed_category = data.get("category")
    if isinstance(fixed_category, FixedErrorCategory):
        legacy_category = FIXED_TO_LEGACY_CATEGORY.get(
            fixed_category, ErrorCategory.Unknown
        )
    else:
        legacy_category = ErrorCategory.Unknown

    legacy_override = _classify_legacy(exc, stack)
    if legacy_override is not ErrorCategory.Unknown:
        return legacy_override
    return legacy_category


__all__ = [
    "ErrorCategory",
    "ErrorType",
    "FixedErrorCategory",
    "LEGACY_TO_FIXED_CATEGORY",
    "FIXED_TO_LEGACY_CATEGORY",
    "classify_error",
    "classify_exception",
]
