from __future__ import annotations

"""Centralised error taxonomy for Menace.

This module defines a non-expanding, fixed taxonomy and a deterministic
classifier. Classification is purely literal: explicit ``isinstance`` checks
for known exception types and literal substring matching for known tokens. The
taxonomy is immutable (no dynamic extension) and the matching order is fixed
and documented to keep outcomes stable across runs.

Script usage intentionally relies on explicit imports instead of path mutation.
For example:

``python -c "from error_ontology import classify_error; print(classify_error('syntax error'))"``
"""

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

_CANONICAL_TOKEN_PHRASES: Tuple[Tuple[ErrorCategory, Tuple[str, ...]], ...] = (
    (ErrorCategory.SyntaxError, ("syntax error",)),
    (ErrorCategory.ImportError, ("import error",)),
    (ErrorCategory.TypeErrorMismatch, ("type mismatch",)),
    (ErrorCategory.ContractViolation, ("contract violation",)),
    (ErrorCategory.EdgeCaseFailure, ("edge case",)),
    (ErrorCategory.UnhandledException, ("unhandled exception",)),
    (ErrorCategory.InvalidInput, ("invalid input",)),
    (ErrorCategory.MissingReturn, ("missing return",)),
    (ErrorCategory.ConfigError, ("config error",)),
    (ErrorCategory.Other, ("other",)),
)


def _dedupe_tokens(tokens: Iterable[str]) -> List[str]:
    seen: set[str] = set()
    deduped: List[str] = []
    for token in tokens:
        normalized = token.strip().lower()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
    return deduped


# Fixed literal token priority. If multiple matches appear, the first match
# in this list is selected (deterministic ordering). Tokens are lowercase
# literals matched against a lowercased input string.
_TOKEN_RULES: Tuple[ClassificationRule, ...] = tuple(
    {
        "kind": "token",
        "match": token,
        "category": category,
        "matched_rule": f"token:{token}",
    }
    for category, phrases in _CANONICAL_TOKEN_PHRASES
    for token in _dedupe_tokens((*phrases, category.value))
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
        normalized_token = token.lower() if isinstance(token, str) else token
        if isinstance(normalized_token, str) and normalized_token in normalized:
            return rule["category"], rule["matched_rule"], normalized_token
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
    selected_status = ErrorCategory.Other.value

    for index, item in enumerate(items):
        category, data, item_errors = _classify_single(item)
        data["index"] = index
        bundle_items.append(data)
        bundle_errors.extend(item_errors)

    meta: Dict[str, Any] = {
        "bundle_rule": "always_other_for_bundle",
        "bundle_selected_index": None,
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
    order and stored in ``bundle``. The top-level status is always ``Other``
    for bundles, regardless of item matches.

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

    if input_kind == "mapping":
        extracted = normalized
        if isinstance(extracted, Sequence) and not isinstance(
            extracted, (str, bytes, bytearray)
        ):
            normalized = extracted
            input_kind = "sequence"
        else:
            normalized = extracted
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
