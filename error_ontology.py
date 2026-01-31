from __future__ import annotations

"""Centralised error taxonomy for Menace."""

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


class ErrorCategory(str, Enum):
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


ErrorType = ErrorCategory

_ExceptionInput = Union[BaseException, str, Mapping[str, Any]]
_Input = Union[_ExceptionInput, Sequence[_ExceptionInput]]

_EXCEPTION_TYPE_MAP: Tuple[Tuple[type[BaseException], ErrorCategory], ...] = (
    (SyntaxError, ErrorCategory.SyntaxError),
    (ImportError, ErrorCategory.ImportError),
    (ModuleNotFoundError, ErrorCategory.ImportError),
    (TypeError, ErrorCategory.TypeErrorMismatch),
    (AssertionError, ErrorCategory.ContractViolation),
    (ValueError, ErrorCategory.InvalidInput),
)

_PHRASE_MAP: Tuple[Tuple[str, ErrorCategory], ...] = (
    ("syntax error", ErrorCategory.SyntaxError),
    ("no module named", ErrorCategory.ImportError),
    ("cannot import", ErrorCategory.ImportError),
    ("import error", ErrorCategory.ImportError),
    ("typeerror", ErrorCategory.TypeErrorMismatch),
    ("type error", ErrorCategory.TypeErrorMismatch),
    ("type mismatch", ErrorCategory.TypeErrorMismatch),
    ("assertion failed", ErrorCategory.ContractViolation),
    ("contract violation", ErrorCategory.ContractViolation),
    ("precondition failed", ErrorCategory.ContractViolation),
    ("postcondition failed", ErrorCategory.ContractViolation),
    ("edge case", ErrorCategory.EdgeCaseFailure),
    ("corner case", ErrorCategory.EdgeCaseFailure),
    ("unhandled exception", ErrorCategory.UnhandledException),
    ("uncaught exception", ErrorCategory.UnhandledException),
    ("invalid input", ErrorCategory.InvalidInput),
    ("invalid argument", ErrorCategory.InvalidInput),
    ("bad request", ErrorCategory.InvalidInput),
    ("missing return", ErrorCategory.MissingReturn),
    ("did not return", ErrorCategory.MissingReturn),
    ("returned none", ErrorCategory.MissingReturn),
    ("missing config", ErrorCategory.ConfigError),
    ("configuration error", ErrorCategory.ConfigError),
    ("config error", ErrorCategory.ConfigError),
)


_CATEGORY_PRIORITY: Tuple[ErrorCategory, ...] = (
    ErrorCategory.SyntaxError,
    ErrorCategory.ImportError,
    ErrorCategory.TypeErrorMismatch,
    ErrorCategory.ContractViolation,
    ErrorCategory.EdgeCaseFailure,
    ErrorCategory.UnhandledException,
    ErrorCategory.InvalidInput,
    ErrorCategory.MissingReturn,
    ErrorCategory.ConfigError,
    ErrorCategory.Other,
)


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


def _classify_from_exception(exc: BaseException) -> Tuple[ErrorCategory, str]:
    for etype, category in _EXCEPTION_TYPE_MAP:
        if isinstance(exc, etype):
            return category, f"exception:{etype.__name__}"
    return ErrorCategory.Other, "exception:unmatched"


def _classify_from_text(text: str) -> Tuple[ErrorCategory, str]:
    if not text.strip():
        return ErrorCategory.Other, "text:empty"
    lowered = text.lower()
    for phrase, category in _PHRASE_MAP:
        if phrase in lowered:
            return category, f"phrase:{phrase}"
    return ErrorCategory.Other, "text:unmatched"


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
                "category": ErrorCategory.Other,
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

    classifications: List[Tuple[ErrorCategory, str, str]] = []
    for item in segments:
        if isinstance(item, BaseException):
            category, matched_rule = _classify_from_exception(item)
            source = "exception"
        else:
            category, matched_rule = _classify_from_text(str(item))
            source = "text"
        classifications.append((category, matched_rule, source))

    category = ErrorCategory.Other
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
    category = data.get("category")
    if isinstance(category, ErrorCategory):
        return category
    return ErrorCategory.Other


__all__ = [
    "ErrorCategory",
    "ErrorType",
    "classify_error",
    "classify_exception",
]
