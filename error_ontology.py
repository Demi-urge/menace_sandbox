from __future__ import annotations

"""Centralised error taxonomy for Menace.

This module defines a non-expanding, fixed taxonomy and a deterministic
classifier. Classification is purely literal: explicit ``isinstance`` checks
for known exception types and literal substring matching for known tokens. The
taxonomy is immutable (no dynamic extension) and the matching order is fixed
and documented to keep outcomes stable across runs.
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

# Fixed, non-expanding taxonomy list used for deterministic ordering.
ERROR_TAXONOMY: Tuple[str, ...] = (
    ErrorCategory.SyntaxError.value,
    ErrorCategory.ImportError.value,
    ErrorCategory.TypeErrorMismatch.value,
    ErrorCategory.ContractViolation.value,
    ErrorCategory.EdgeCaseFailure.value,
    ErrorCategory.UnhandledException.value,
    ErrorCategory.InvalidInput.value,
    ErrorCategory.MissingReturn.value,
    ErrorCategory.ConfigError.value,
    ErrorCategory.Other.value,
)

_TAXONOMY_RANK: Dict[str, int] = {name: index for index, name in enumerate(ERROR_TAXONOMY)}


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

ErrorInput = Union[str, BaseException, type[BaseException], Mapping[str, object], Sequence[object]]
ErrorInputs = Union[ErrorInput, None]


class ClassificationRule(TypedDict):
    kind: Literal["exception", "token"]
    match: Union[type[BaseException], str]
    category: ErrorCategory
    matched_rule: str


_EXCEPTION_RULES: Tuple[ClassificationRule, ...] = (
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
)

# Fixed literal token priority. If multiple matches appear, the first match
# in this list is selected (deterministic ordering). Tokens are lowercase
# literals matched against a lowercased input string.
_TOKEN_RULES: Tuple[ClassificationRule, ...] = (
    {
        "kind": "token",
        "match": "syntax error",
        "category": ErrorCategory.SyntaxError,
        "matched_rule": "token:syntax error",
    },
    {
        "kind": "token",
        "match": "invalid syntax",
        "category": ErrorCategory.SyntaxError,
        "matched_rule": "token:invalid syntax",
    },
    {
        "kind": "token",
        "match": "import error",
        "category": ErrorCategory.ImportError,
        "matched_rule": "token:import error",
    },
    {
        "kind": "token",
        "match": "no module named",
        "category": ErrorCategory.ImportError,
        "matched_rule": "token:no module named",
    },
    {
        "kind": "token",
        "match": "module not found",
        "category": ErrorCategory.ImportError,
        "matched_rule": "token:module not found",
    },
    {
        "kind": "token",
        "match": "type error",
        "category": ErrorCategory.TypeErrorMismatch,
        "matched_rule": "token:type error",
    },
    {
        "kind": "token",
        "match": "type mismatch",
        "category": ErrorCategory.TypeErrorMismatch,
        "matched_rule": "token:type mismatch",
    },
    {
        "kind": "token",
        "match": "unsupported operand",
        "category": ErrorCategory.TypeErrorMismatch,
        "matched_rule": "token:unsupported operand",
    },
    {
        "kind": "token",
        "match": "contract violation",
        "category": ErrorCategory.ContractViolation,
        "matched_rule": "token:contract violation",
    },
    {
        "kind": "token",
        "match": "assertion failed",
        "category": ErrorCategory.ContractViolation,
        "matched_rule": "token:assertion failed",
    },
    {
        "kind": "token",
        "match": "assertion error",
        "category": ErrorCategory.ContractViolation,
        "matched_rule": "token:assertion error",
    },
    {
        "kind": "token",
        "match": "index out of range",
        "category": ErrorCategory.EdgeCaseFailure,
        "matched_rule": "token:index out of range",
    },
    {
        "kind": "token",
        "match": "key error",
        "category": ErrorCategory.EdgeCaseFailure,
        "matched_rule": "token:key error",
    },
    {
        "kind": "token",
        "match": "edge case",
        "category": ErrorCategory.EdgeCaseFailure,
        "matched_rule": "token:edge case",
    },
    {
        "kind": "token",
        "match": "unhandled exception",
        "category": ErrorCategory.UnhandledException,
        "matched_rule": "token:unhandled exception",
    },
    {
        "kind": "token",
        "match": "unexpected exception",
        "category": ErrorCategory.UnhandledException,
        "matched_rule": "token:unexpected exception",
    },
    {
        "kind": "token",
        "match": "invalid input",
        "category": ErrorCategory.InvalidInput,
        "matched_rule": "token:invalid input",
    },
    {
        "kind": "token",
        "match": "bad input",
        "category": ErrorCategory.InvalidInput,
        "matched_rule": "token:bad input",
    },
    {
        "kind": "token",
        "match": "missing return",
        "category": ErrorCategory.MissingReturn,
        "matched_rule": "token:missing return",
    },
    {
        "kind": "token",
        "match": "no return statement",
        "category": ErrorCategory.MissingReturn,
        "matched_rule": "token:no return statement",
    },
    {
        "kind": "token",
        "match": "config error",
        "category": ErrorCategory.ConfigError,
        "matched_rule": "token:config error",
    },
    {
        "kind": "token",
        "match": "configuration error",
        "category": ErrorCategory.ConfigError,
        "matched_rule": "token:configuration error",
    },
    {
        "kind": "token",
        "match": "missing config",
        "category": ErrorCategory.ConfigError,
        "matched_rule": "token:missing config",
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


class ClassificationItem(TypedDict):
    index: int
    status: str
    input_kind: str
    matched_rule_id: str
    normalized: str
    matched_token: str | None


class ClassificationData(TypedDict):
    input_kind: str
    normalized: str
    matched_token: str | None
    matched_rule_id: str
    bundle: List[ClassificationItem] | None


class ClassificationResult(TypedDict):
    status: str
    data: ClassificationData
    errors: List[str]
    meta: Dict[str, Any]


_DICT_KEY_ORDER: Tuple[str, ...] = (
    "exception",
    "traceback",
    "message",
    "error",
    "errors",
)


def _mapping_to_text(payload: Mapping[str, Any]) -> str:
    if not payload:
        return ""
    pairs = ", ".join(f"{key!r}: {value!r}" for key, value in payload.items())
    return f"{{{pairs}}}"


def _safe_text(value: object) -> str:
    return "" if value is None else str(value)


def _extract_from_mapping(payload: Mapping[str, object]) -> object:
    for key in _DICT_KEY_ORDER:
        if key in payload:
            return payload[key]
    return payload


def _normalize_input(raw: ErrorInputs) -> Tuple[str, object | None, List[str]]:
    errors: List[str] = []
    if raw is None:
        return "empty", None, errors
    if isinstance(raw, BaseException):
        return "exception_instance", raw, errors
    if isinstance(raw, type) and issubclass(raw, BaseException):
        return "exception_type", raw, errors
    if isinstance(raw, str):
        return "string", raw, errors
    if isinstance(raw, Mapping):
        extracted = _extract_from_mapping(raw)
        return "mapping", extracted, errors
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
        return "sequence", raw, errors
    errors.append(f"Unsupported input type: {type(raw).__name__}")
    return "other", raw, errors


def _classify_exception(exc: BaseException) -> Tuple[ErrorCategory, str, str]:
    for rule in _EXCEPTION_RULES:
        match = rule["match"]
        if isinstance(match, type) and isinstance(exc, match):
            return rule["category"], rule["matched_rule"], match.__name__
    return ErrorCategory.Other, "exception:unmatched", exc.__class__.__name__


def _classify_exception_type(exc_type: type[BaseException]) -> Tuple[ErrorCategory, str, str]:
    for rule in _EXCEPTION_RULES:
        match = rule["match"]
        if isinstance(match, type) and issubclass(exc_type, match):
            return rule["category"], rule["matched_rule"], match.__name__
    return ErrorCategory.Other, "exception_type:unmatched", exc_type.__name__


def _classify_text(text: str) -> Tuple[ErrorCategory, str, str | None]:
    if not text.strip():
        return ErrorCategory.Other, "text:empty", None
    normalized = text.lower()
    for rule in _TOKEN_RULES:
        token = rule["match"]
        if isinstance(token, str) and token in normalized:
            return rule["category"], rule["matched_rule"], token
    return ErrorCategory.Other, "text:unmatched", None


def _classify_single(value: object) -> Tuple[ErrorCategory, ClassificationItem, List[str]]:
    errors: List[str] = []
    input_kind, normalized, kind_errors = _normalize_input(value)
    errors.extend(kind_errors)

    if input_kind == "empty":
        item: ClassificationItem = {
            "index": 0,
            "status": ErrorCategory.Other.value,
            "input_kind": input_kind,
            "matched_rule_id": "empty:other",
            "normalized": "",
            "matched_token": None,
        }
        return ErrorCategory.Other, item, errors

    if input_kind == "exception_instance" and isinstance(normalized, BaseException):
        category, rule_id, _ = _classify_exception(normalized)
        item = {
            "index": 0,
            "status": category.value,
            "input_kind": input_kind,
            "matched_rule_id": rule_id,
            "normalized": _safe_text(normalized),
            "matched_token": None,
        }
        return category, item, errors

    if input_kind == "exception_type" and isinstance(normalized, type):
        category, rule_id, _ = _classify_exception_type(normalized)
        item = {
            "index": 0,
            "status": category.value,
            "input_kind": input_kind,
            "matched_rule_id": rule_id,
            "normalized": normalized.__name__,
            "matched_token": None,
        }
        return category, item, errors

    if input_kind == "sequence" and isinstance(normalized, Sequence):
        category = ErrorCategory.Other
        item = {
            "index": 0,
            "status": category.value,
            "input_kind": input_kind,
            "matched_rule_id": "sequence:defer",
            "normalized": _safe_text(normalized),
            "matched_token": None,
        }
        return category, item, errors

    if input_kind == "mapping" and isinstance(value, Mapping):
        extracted = _extract_from_mapping(value)
        if isinstance(extracted, BaseException):
            category, rule_id, _ = _classify_exception(extracted)
            item = {
                "index": 0,
                "status": category.value,
                "input_kind": input_kind,
                "matched_rule_id": rule_id,
                "normalized": _safe_text(extracted),
                "matched_token": None,
            }
            return category, item, errors
        if isinstance(extracted, type) and issubclass(extracted, BaseException):
            category, rule_id, _ = _classify_exception_type(extracted)
            item = {
                "index": 0,
                "status": category.value,
                "input_kind": input_kind,
                "matched_rule_id": rule_id,
                "normalized": extracted.__name__,
                "matched_token": None,
            }
            return category, item, errors
        text = _safe_text(extracted)
        category, rule_id, token = _classify_text(text)
        item = {
            "index": 0,
            "status": category.value,
            "input_kind": input_kind,
            "matched_rule_id": rule_id,
            "normalized": text,
            "matched_token": token,
        }
        return category, item, errors

    text = _safe_text(normalized)
    category, rule_id, token = _classify_text(text)
    item = {
        "index": 0,
        "status": category.value,
        "input_kind": input_kind,
        "matched_rule_id": rule_id,
        "normalized": text,
        "matched_token": token,
    }
    if token:
        item["normalized"] = text
    return category, item, errors


def _classify_bundle(
    items: Sequence[object],
) -> Tuple[str, List[ClassificationItem], Dict[str, Any], List[str]]:
    bundle_items: List[ClassificationItem] = []
    bundle_errors: List[str] = []
    selected_index: int | None = None
    selected_rank: int | None = None
    selected_status = ErrorCategory.Other.value

    for index, item in enumerate(items):
        category, data, item_errors = _classify_single(item)
        data["index"] = index
        bundle_items.append(data)
        bundle_errors.extend(item_errors)
        rank = _TAXONOMY_RANK.get(category.value, len(ERROR_TAXONOMY))
        if selected_rank is None or rank < selected_rank:
            selected_rank = rank
            selected_status = category.value
            selected_index = index

    meta: Dict[str, Any] = {
        "bundle_rule": "lowest_taxonomy_rank_then_first",
        "bundle_selected_index": selected_index,
        "bundle_size": len(bundle_items),
    }
    return selected_status, bundle_items, meta, bundle_errors


def classify_error(raw: ErrorInputs) -> ClassificationResult:
    """Classify structured or unstructured error inputs.

    Parameters
    ----------
    raw:
        Exception instance, exception type, traceback string, log string,
        dictionary with conventional keys, or list/tuple of those.

    Notes
    -----
    Matching order is deterministic:
    1) Explicit ``isinstance`` checks for known exception types.
    2) Literal substring matching in the fixed ``_TOKEN_RULES`` list after
       lowercasing the input text.

    For multi-error bundles (lists/tuples), each element is classified in input
    order. The resulting status is selected by severity using the fixed
    taxonomy order (earlier entries are more severe); if multiple entries share
    the same severity, the first in input order is selected.

    For dictionaries, the classifier inspects keys in the fixed order defined
    by ``_DICT_KEY_ORDER`` (``exception``, ``traceback``, ``message``, ``error``,
    ``errors``) and classifies using the first present key.
    """

    input_kind, normalized, errors = _normalize_input(raw)

    if input_kind == "empty" or (isinstance(normalized, str) and not normalized.strip()):
        return {
            "status": ErrorCategory.Other.value,
            "data": {
                "input_kind": input_kind,
                "normalized": "",
                "matched_token": None,
                "matched_rule_id": "empty:other",
                "bundle": None,
            },
            "errors": [],
            "meta": {
                "classifier_version": "1.0",
                "input_kind": input_kind,
                "matched_rule_id": "empty:other",
            },
        }

    if input_kind == "mapping" and isinstance(normalized, Mapping):
        normalized = _extract_from_mapping(normalized)
        if isinstance(normalized, Sequence) and not isinstance(
            normalized, (str, bytes, bytearray)
        ):
            input_kind = "sequence"
        else:
            input_kind = "mapping"

    if input_kind == "sequence" and isinstance(normalized, Sequence):
        status, bundle_items, bundle_meta, bundle_errors = _classify_bundle(normalized)
        matched_rule_id = "bundle:classification"
        return {
            "status": status,
            "data": {
                "input_kind": input_kind,
                "normalized": _safe_text(normalized),
                "matched_token": None,
                "matched_rule_id": matched_rule_id,
                "bundle": bundle_items,
            },
            "errors": errors + bundle_errors,
            "meta": {
                "classifier_version": "1.0",
                "input_kind": input_kind,
                "matched_rule_id": matched_rule_id,
                **bundle_meta,
            },
        }

    if input_kind == "mapping":
        category, item, item_errors = _classify_single(normalized)
        errors.extend(item_errors)
        matched_rule_id = item["matched_rule_id"]
        matched_token = item["matched_token"]
        return {
            "status": category.value,
            "data": {
                "input_kind": input_kind,
                "normalized": item["normalized"],
                "matched_token": matched_token,
                "matched_rule_id": matched_rule_id,
                "bundle": None,
            },
            "errors": errors,
            "meta": {
                "classifier_version": "1.0",
                "input_kind": input_kind,
                "matched_rule_id": matched_rule_id,
            },
        }

    category, item, item_errors = _classify_single(normalized)
    errors.extend(item_errors)
    matched_rule_id = item["matched_rule_id"]
    matched_token = item["matched_token"]
    return {
        "status": category.value,
        "data": {
            "input_kind": input_kind,
            "normalized": item["normalized"],
            "matched_token": matched_token,
            "matched_rule_id": matched_rule_id,
            "bundle": None,
        },
        "errors": errors,
        "meta": {
            "classifier_version": "1.0",
            "input_kind": input_kind,
            "matched_rule_id": matched_rule_id,
        },
    }


def classify_exception(exc: Exception, stack: str) -> LegacyErrorCategory:
    """Legacy exception classification that delegates to the fixed taxonomy."""

    result = classify_error([exc, stack])
    fixed_category = result["status"]
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
    "ERROR_TAXONOMY",
    "classify_error",
    "classify_exception",
]
