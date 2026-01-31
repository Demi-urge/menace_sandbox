from __future__ import annotations

"""Centralised error taxonomy for Menace.

:class:`ErrorCategory` defines the fixed error taxonomy required by the
stabilisation pipeline. The legacy taxonomy is preserved as
:class:`LegacyErrorCategory` for backward compatibility. A static mapping
between the two keeps integrations stable during migration.
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
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    Literal,
    Sequence,
    Tuple,
    TypedDict,
    Union,
)


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


FixedErrorCategory = ErrorCategory


class LegacyErrorCategory(str, Enum):
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

ErrorInput = Union[BaseException, str, Mapping[str, Any]]
ErrorInputBundle = Sequence[ErrorInput]
ErrorInputs = Union[ErrorInput, ErrorInputBundle]

_EXCEPTION_TYPE_MAP: Tuple[Tuple[type[BaseException], ErrorCategory], ...] = (
    (SyntaxError, ErrorCategory.SyntaxError),
    (ImportError, ErrorCategory.ImportError),
    (ModuleNotFoundError, ErrorCategory.ImportError),
    (TypeError, ErrorCategory.TypeErrorMismatch),
    (AssertionError, ErrorCategory.ContractViolation),
    (ValueError, ErrorCategory.InvalidInput),
    (KeyError, ErrorCategory.EdgeCaseFailure),
    (IndexError, ErrorCategory.EdgeCaseFailure),
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
    ("keyerror", ErrorCategory.EdgeCaseFailure),
    ("indexerror", ErrorCategory.EdgeCaseFailure),
    ("index out of range", ErrorCategory.EdgeCaseFailure),
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


LEGACY_TO_FIXED_CATEGORY: Mapping[LegacyErrorCategory, ErrorCategory] = {
    LegacyErrorCategory.Unknown: ErrorCategory.Other,
    LegacyErrorCategory.RuntimeFault: ErrorCategory.UnhandledException,
    LegacyErrorCategory.DependencyMismatch: ErrorCategory.ImportError,
    LegacyErrorCategory.LogicMisfire: ErrorCategory.ContractViolation,
    LegacyErrorCategory.SemanticBug: ErrorCategory.TypeErrorMismatch,
    LegacyErrorCategory.ResourceLimit: ErrorCategory.UnhandledException,
    LegacyErrorCategory.Timeout: ErrorCategory.UnhandledException,
    LegacyErrorCategory.ExternalAPI: ErrorCategory.UnhandledException,
    LegacyErrorCategory.MetricBottleneck: ErrorCategory.Other,
}

FIXED_TO_LEGACY_CATEGORY: Mapping[ErrorCategory, LegacyErrorCategory] = {
    ErrorCategory.SyntaxError: LegacyErrorCategory.SemanticBug,
    ErrorCategory.ImportError: LegacyErrorCategory.DependencyMismatch,
    ErrorCategory.TypeErrorMismatch: LegacyErrorCategory.SemanticBug,
    ErrorCategory.ContractViolation: LegacyErrorCategory.LogicMisfire,
    ErrorCategory.EdgeCaseFailure: LegacyErrorCategory.RuntimeFault,
    ErrorCategory.UnhandledException: LegacyErrorCategory.RuntimeFault,
    ErrorCategory.InvalidInput: LegacyErrorCategory.SemanticBug,
    ErrorCategory.MissingReturn: LegacyErrorCategory.LogicMisfire,
    ErrorCategory.ConfigError: LegacyErrorCategory.DependencyMismatch,
    ErrorCategory.Other: LegacyErrorCategory.Unknown,
}


def _mapping_to_text(payload: Mapping[str, Any]) -> str:
    if not payload:
        return ""
    pairs = ", ".join(f"{key!r}: {value!r}" for key, value in payload.items())
    return f"{{{pairs}}}"


def _normalize_inputs(
    raw: ErrorInputs | None,
) -> Tuple[List[Union[str, BaseException]], List[str]]:
    errors: List[str] = []
    segments: List[Union[str, BaseException]] = []

    if raw is None:
        return segments, errors

    if isinstance(raw, BaseException):
        iterable: Iterable[ErrorInput] = (raw,)
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


def _legacy_category_from_module(module_name: str) -> LegacyErrorCategory:
    if not module_name:
        return LegacyErrorCategory.Unknown
    if module_name.startswith(("pkg_resources", "importlib")):
        return LegacyErrorCategory.DependencyMismatch
    if module_name.startswith("requests"):
        return LegacyErrorCategory.ExternalAPI
    return LegacyErrorCategory.Unknown


def _classify_legacy_from_exception(exc: BaseException) -> LegacyErrorCategory:
    try:
        from urllib.error import HTTPError
    except Exception:  # pragma: no cover - stdlib should exist
        HTTPError = None  # type: ignore

    exception_map: Tuple[Tuple[type[BaseException], LegacyErrorCategory], ...] = (
        *((HTTPError, LegacyErrorCategory.ExternalAPI),) if HTTPError else (),
        (MemoryError, LegacyErrorCategory.ResourceLimit),
        (TimeoutError, LegacyErrorCategory.Timeout),
        (KeyError, LegacyErrorCategory.RuntimeFault),
        (IndexError, LegacyErrorCategory.RuntimeFault),
        (FileNotFoundError, LegacyErrorCategory.RuntimeFault),
        (ZeroDivisionError, LegacyErrorCategory.RuntimeFault),
        (AttributeError, LegacyErrorCategory.RuntimeFault),
        (PermissionError, LegacyErrorCategory.RuntimeFault),
        (AssertionError, LegacyErrorCategory.LogicMisfire),
        (NotImplementedError, LegacyErrorCategory.LogicMisfire),
        (TypeError, LegacyErrorCategory.SemanticBug),
        (ValueError, LegacyErrorCategory.SemanticBug),
        (ModuleNotFoundError, LegacyErrorCategory.DependencyMismatch),
        (ImportError, LegacyErrorCategory.DependencyMismatch),
        (OSError, LegacyErrorCategory.DependencyMismatch),
    )

    for etype, category in exception_map:
        if isinstance(exc, etype):
            return category

    if exc.__class__.__name__ == "SandboxRecoveryError":
        return LegacyErrorCategory.RuntimeFault

    module_category = _legacy_category_from_module(exc.__class__.__module__)
    if module_category is not LegacyErrorCategory.Unknown:
        return module_category
    return LegacyErrorCategory.Unknown


def _classify_legacy_from_text(text: str) -> LegacyErrorCategory:
    if not text.strip():
        return LegacyErrorCategory.Unknown
    lowered = text.lower()
    legacy_phrases: Tuple[Tuple[str, LegacyErrorCategory], ...] = (
        ("dependency missing", LegacyErrorCategory.DependencyMismatch),
        ("missing package", LegacyErrorCategory.DependencyMismatch),
        ("module not found", LegacyErrorCategory.DependencyMismatch),
        ("no module named", LegacyErrorCategory.DependencyMismatch),
        ("cannot import name", LegacyErrorCategory.DependencyMismatch),
        ("out of memory", LegacyErrorCategory.ResourceLimit),
        ("cuda error", LegacyErrorCategory.ResourceLimit),
        ("accelerator error", LegacyErrorCategory.ResourceLimit),
        ("timed out", LegacyErrorCategory.Timeout),
        ("timeout", LegacyErrorCategory.Timeout),
        ("external api", LegacyErrorCategory.ExternalAPI),
        ("remote server replied", LegacyErrorCategory.ExternalAPI),
        ("forbidden", LegacyErrorCategory.ExternalAPI),
        ("permission denied", LegacyErrorCategory.RuntimeFault),
    )
    for phrase, category in legacy_phrases:
        if phrase in lowered:
            return category
    return LegacyErrorCategory.Unknown


def _classify_legacy(exc: BaseException, stack: str) -> LegacyErrorCategory:
    exc_category = _classify_legacy_from_exception(exc)
    if exc_category is not LegacyErrorCategory.Unknown:
        return exc_category
    return _classify_legacy_from_text(stack)


class ClassificationData(TypedDict):
    category: str
    source: str
    matched_rule: str


class ClassificationResult(TypedDict):
    status: Literal["ok", "fallback"]
    data: ClassificationData
    errors: List[str]
    meta: Dict[str, Any]


def _select_bundle_category(
    classifications: Sequence[Tuple[ErrorCategory, str, str]],
) -> Tuple[ErrorCategory, str, str]:
    if not classifications:
        return ErrorCategory.Other, "bundle:empty", "bundle"

    non_other = [
        (category, matched_rule, source)
        for category, matched_rule, source in classifications
        if category is not ErrorCategory.Other
    ]
    if not non_other:
        return ErrorCategory.Other, "bundle:unmatched", "bundle"

    unique = {category for category, _, _ in non_other}
    if len(unique) > 1:
        return ErrorCategory.Other, "bundle:ambiguous", "bundle"

    return non_other[0]


def classify_error(raw: ErrorInputs | None) -> ClassificationResult:
    """Classify structured or unstructured error inputs.

    Parameters
    ----------
    raw:
        Exception instance, traceback string, log string, or list/tuple of those.
    """

    segments, errors = _normalize_inputs(raw)

    if _is_empty_input(segments):
        status: Literal["ok", "fallback"] = "fallback"
        data: ClassificationData = {
            "category": ErrorCategory.Other.value,
            "source": "text",
            "matched_rule": "text:empty",
        }
        return {
            "status": status,
            "data": data,
            "errors": errors,
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

    category, matched_rule, source = _select_bundle_category(classifications)

    meta: Dict[str, Any] = {
        "input_length": sum(len(_segment_to_text(segment)) for segment in segments),
        "segment_count": len(segments),
    }
    if len(segments) > 1:
        meta["bundle_size"] = len(segments)
    if matched_rule.startswith("bundle:"):
        meta["bundle_rule"] = matched_rule
    if matched_rule.startswith("phrase:"):
        meta["matched_phrase"] = matched_rule.split("phrase:", 1)[1]
    if matched_rule.startswith("exception:"):
        meta["matched_exception"] = matched_rule.split("exception:", 1)[1]
    if matched_rule == "bundle:ambiguous":
        meta["bundle_categories"] = sorted(
            {
                category.value
                for category, _, _ in classifications
                if category is not ErrorCategory.Other
            }
        )

    status = "ok"
    if errors or category is ErrorCategory.Other:
        status = "fallback"

    return {
        "status": status,
        "data": {
            "category": category.value,
            "source": source,
            "matched_rule": matched_rule,
        },
        "errors": errors,
        "meta": meta,
    }


def classify_exception(exc: Exception, stack: str) -> LegacyErrorCategory:
    """Legacy exception classification that delegates to the fixed taxonomy."""

    result = classify_error([exc, stack])
    data = result.get("data", {})
    fixed_category = data.get("category")
    if isinstance(fixed_category, str):
        try:
            legacy_category = FIXED_TO_LEGACY_CATEGORY.get(
                ErrorCategory(fixed_category), LegacyErrorCategory.Unknown
            )
        except ValueError:
            legacy_category = LegacyErrorCategory.Unknown
    else:
        legacy_category = LegacyErrorCategory.Unknown

    legacy_override = _classify_legacy(exc, stack)
    if legacy_override is not LegacyErrorCategory.Unknown:
        return legacy_override
    return legacy_category


__all__ = [
    "ErrorCategory",
    "ErrorType",
    "FixedErrorCategory",
    "LegacyErrorCategory",
    "LEGACY_TO_FIXED_CATEGORY",
    "FIXED_TO_LEGACY_CATEGORY",
    "classify_error",
    "classify_exception",
]
