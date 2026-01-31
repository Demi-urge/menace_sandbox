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
from typing import Any, Dict, Iterable, List, Literal, Mapping, Sequence, Tuple, TypedDict, Union


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
    """Legacy error taxonomy retained for backward compatibility (not exported)."""

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

ErrorInput = Union[str, BaseException, Mapping[str, object], Sequence[object]]
ErrorInputs = Union[ErrorInput, None]


class ClassificationRule(TypedDict):
    kind: Literal["exception", "phrase"]
    match: Union[type[BaseException], str]
    category: ErrorCategory
    matched_rule: str


_CLASSIFICATION_RULES: Tuple[ClassificationRule, ...] = (
    {
        "kind": "exception",
        "match": SyntaxError,
        "category": ErrorCategory.SyntaxError,
        "matched_rule": "exception:SyntaxError",
    },
    {
        "kind": "exception",
        "match": ImportError,
        "category": ErrorCategory.ImportError,
        "matched_rule": "exception:ImportError",
    },
    {
        "kind": "exception",
        "match": ModuleNotFoundError,
        "category": ErrorCategory.ImportError,
        "matched_rule": "exception:ModuleNotFoundError",
    },
    {
        "kind": "exception",
        "match": TypeError,
        "category": ErrorCategory.TypeErrorMismatch,
        "matched_rule": "exception:TypeError",
    },
    {
        "kind": "exception",
        "match": AssertionError,
        "category": ErrorCategory.ContractViolation,
        "matched_rule": "exception:AssertionError",
    },
    {
        "kind": "exception",
        "match": ValueError,
        "category": ErrorCategory.InvalidInput,
        "matched_rule": "exception:ValueError",
    },
    {
        "kind": "exception",
        "match": KeyError,
        "category": ErrorCategory.EdgeCaseFailure,
        "matched_rule": "exception:KeyError",
    },
    {
        "kind": "exception",
        "match": IndexError,
        "category": ErrorCategory.EdgeCaseFailure,
        "matched_rule": "exception:IndexError",
    },
    {
        "kind": "phrase",
        "match": "syntax error",
        "category": ErrorCategory.SyntaxError,
        "matched_rule": "phrase:syntax error",
    },
    {
        "kind": "phrase",
        "match": "no module named",
        "category": ErrorCategory.ImportError,
        "matched_rule": "phrase:no module named",
    },
    {
        "kind": "phrase",
        "match": "cannot import",
        "category": ErrorCategory.ImportError,
        "matched_rule": "phrase:cannot import",
    },
    {
        "kind": "phrase",
        "match": "import error",
        "category": ErrorCategory.ImportError,
        "matched_rule": "phrase:import error",
    },
    {
        "kind": "phrase",
        "match": "typeerror",
        "category": ErrorCategory.TypeErrorMismatch,
        "matched_rule": "phrase:typeerror",
    },
    {
        "kind": "phrase",
        "match": "type error",
        "category": ErrorCategory.TypeErrorMismatch,
        "matched_rule": "phrase:type error",
    },
    {
        "kind": "phrase",
        "match": "type mismatch",
        "category": ErrorCategory.TypeErrorMismatch,
        "matched_rule": "phrase:type mismatch",
    },
    {
        "kind": "phrase",
        "match": "assertion failed",
        "category": ErrorCategory.ContractViolation,
        "matched_rule": "phrase:assertion failed",
    },
    {
        "kind": "phrase",
        "match": "contract violation",
        "category": ErrorCategory.ContractViolation,
        "matched_rule": "phrase:contract violation",
    },
    {
        "kind": "phrase",
        "match": "precondition failed",
        "category": ErrorCategory.ContractViolation,
        "matched_rule": "phrase:precondition failed",
    },
    {
        "kind": "phrase",
        "match": "postcondition failed",
        "category": ErrorCategory.ContractViolation,
        "matched_rule": "phrase:postcondition failed",
    },
    {
        "kind": "phrase",
        "match": "edge case",
        "category": ErrorCategory.EdgeCaseFailure,
        "matched_rule": "phrase:edge case",
    },
    {
        "kind": "phrase",
        "match": "corner case",
        "category": ErrorCategory.EdgeCaseFailure,
        "matched_rule": "phrase:corner case",
    },
    {
        "kind": "phrase",
        "match": "keyerror",
        "category": ErrorCategory.EdgeCaseFailure,
        "matched_rule": "phrase:keyerror",
    },
    {
        "kind": "phrase",
        "match": "indexerror",
        "category": ErrorCategory.EdgeCaseFailure,
        "matched_rule": "phrase:indexerror",
    },
    {
        "kind": "phrase",
        "match": "index out of range",
        "category": ErrorCategory.EdgeCaseFailure,
        "matched_rule": "phrase:index out of range",
    },
    {
        "kind": "phrase",
        "match": "unhandled exception",
        "category": ErrorCategory.UnhandledException,
        "matched_rule": "phrase:unhandled exception",
    },
    {
        "kind": "phrase",
        "match": "uncaught exception",
        "category": ErrorCategory.UnhandledException,
        "matched_rule": "phrase:uncaught exception",
    },
    {
        "kind": "phrase",
        "match": "invalid input",
        "category": ErrorCategory.InvalidInput,
        "matched_rule": "phrase:invalid input",
    },
    {
        "kind": "phrase",
        "match": "invalid argument",
        "category": ErrorCategory.InvalidInput,
        "matched_rule": "phrase:invalid argument",
    },
    {
        "kind": "phrase",
        "match": "bad request",
        "category": ErrorCategory.InvalidInput,
        "matched_rule": "phrase:bad request",
    },
    {
        "kind": "phrase",
        "match": "missing return",
        "category": ErrorCategory.MissingReturn,
        "matched_rule": "phrase:missing return",
    },
    {
        "kind": "phrase",
        "match": "did not return",
        "category": ErrorCategory.MissingReturn,
        "matched_rule": "phrase:did not return",
    },
    {
        "kind": "phrase",
        "match": "returned none",
        "category": ErrorCategory.MissingReturn,
        "matched_rule": "phrase:returned none",
    },
    {
        "kind": "phrase",
        "match": "missing config",
        "category": ErrorCategory.ConfigError,
        "matched_rule": "phrase:missing config",
    },
    {
        "kind": "phrase",
        "match": "configuration error",
        "category": ErrorCategory.ConfigError,
        "matched_rule": "phrase:configuration error",
    },
    {
        "kind": "phrase",
        "match": "config error",
        "category": ErrorCategory.ConfigError,
        "matched_rule": "phrase:config error",
    },
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


def _normalize_inputs(raw: ErrorInputs) -> Tuple[List[Union[str, BaseException]], List[str]]:
    errors: List[str] = []
    segments: List[Union[str, BaseException]] = []

    if raw is None:
        return segments, errors

    if isinstance(raw, BaseException):
        iterable: Iterable[ErrorInput] = (raw,)
    elif isinstance(raw, str):
        iterable = (raw,)
    elif isinstance(raw, Mapping):
        iterable = tuple(raw.values())
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
        elif isinstance(item, Sequence) and not isinstance(item, (str, bytes, bytearray)):
            segments.append(str(item))
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
    for rule in _CLASSIFICATION_RULES:
        if rule["kind"] != "exception":
            continue
        match = rule["match"]
        if isinstance(match, type) and isinstance(exc, match):
            return rule["category"], rule["matched_rule"]
    return ErrorCategory.Other, "exception:unmatched"


def _classify_from_text(normalized_text: str) -> Tuple[ErrorCategory, str]:
    if not normalized_text.strip():
        return ErrorCategory.Other, "text:empty"
    for rule in _CLASSIFICATION_RULES:
        if rule["kind"] != "phrase":
            continue
        match = rule["match"]
        if isinstance(match, str) and match in normalized_text:
            return rule["category"], rule["matched_rule"]
    return ErrorCategory.Other, "text:unmatched"


class ClassificationData(TypedDict):
    category: str
    source: str
    matched_rule: str | None


class ClassificationResult(TypedDict):
    status: Literal["ok", "fallback"]
    data: ClassificationData
    errors: List[str]
    meta: Dict[str, Any]


def classify_error(raw: ErrorInputs) -> ClassificationResult:
    """Classify structured or unstructured error inputs.

    Parameters
    ----------
    raw:
        Exception instance, traceback string, log string, or list/tuple of those.

    Notes
    -----
    Rule ordering is deterministic: exception rules are evaluated in the order
    listed in ``_CLASSIFICATION_RULES``, followed by phrase rules in that same
    order. Text inputs are normalized once (lowercased) and evaluated using
    literal substring checks (no regex or recursive parsing).

    For multi-error bundles (sequences or mappings with multiple values), the
    merge policy is stable and deterministic: iterate in input order and select
    the first non-``Other`` classification; if none match, return ``Other``.
    """

    segments, errors = _normalize_inputs(raw)

    if _is_empty_input(segments):
        data: ClassificationData = {
            "category": ErrorCategory.Other.value,
            "source": "text",
            "matched_rule": None,
        }
        return {
            "status": "ok",
            "data": data,
            "errors": [],
            "meta": {
                "input_length": 0,
                "segment_count": len(segments),
                "input_kind": "empty",
                "matched_rule": None,
            },
        }

    classifications: List[Tuple[ErrorCategory, str, str]] = []
    for item in segments:
        if isinstance(item, BaseException):
            category, matched_rule = _classify_from_exception(item)
            source = "exception"
        else:
            normalized = str(item).lower()
            category, matched_rule = _classify_from_text(normalized)
            source = "text"
        classifications.append((category, matched_rule, source))

    category = ErrorCategory.Other
    matched_rule: str | None = "bundle:all_other"
    source = "bundle"
    selected_index: int | None = None
    if len(classifications) == 1:
        category, matched_rule, source = classifications[0]
    else:
        for index, (candidate, candidate_rule, candidate_source) in enumerate(
            classifications
        ):
            if candidate is not ErrorCategory.Other:
                category = candidate
                matched_rule = candidate_rule
                source = candidate_source
                selected_index = index
                break

    meta: Dict[str, Any] = {
        "input_length": sum(len(_segment_to_text(segment)) for segment in segments),
        "segment_count": len(segments),
    }
    if len(segments) > 1:
        meta["bundle_size"] = len(segments)
        meta["bundle_selection"] = "first_non_other"
        meta["bundle_selected_index"] = selected_index
        meta["merge_policy"] = "first_non_other_in_order"
    if matched_rule.startswith("bundle:"):
        meta["bundle_rule"] = matched_rule
    if matched_rule and matched_rule.startswith("phrase:"):
        meta["matched_phrase"] = matched_rule.split("phrase:", 1)[1]
    if matched_rule and matched_rule.startswith("exception:"):
        meta["matched_exception"] = matched_rule.split("exception:", 1)[1]
    status = "ok"
    if errors or category is ErrorCategory.Other:
        status = "fallback"
        if category is ErrorCategory.Other and not errors:
            errors = ["Unable to classify input."]

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
    fixed_category = result["data"]["category"]
    try:
        mapped = FIXED_TO_LEGACY_CATEGORY.get(
            ErrorCategory(fixed_category), LegacyErrorCategory.Unknown
        )
    except ValueError:
        mapped = LegacyErrorCategory.Unknown
    return mapped


__all__ = [
    "ErrorCategory",
    "ErrorType",
    "FixedErrorCategory",
    "classify_error",
    "classify_exception",
]
