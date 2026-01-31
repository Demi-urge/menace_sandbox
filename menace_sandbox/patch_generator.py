"""Generate deterministic unified diffs by applying explicit patch rules."""

from __future__ import annotations

import ast
import difflib
import hashlib
import json
import re
from dataclasses import asdict, dataclass, is_dataclass
from typing import Any, Mapping, Sequence

from menace.errors import (
    MenaceError,
    PatchAnchorError,
    PatchConflictError,
    PatchParseError,
    PatchRuleError,
    PatchSyntaxError,
)


@dataclass(frozen=True)
class ReplaceRule:
    """Rule for replacing an anchored target with new content.

    Args:
        rule_id: Unique identifier for the rule.
        description: Human-readable summary of the rule intent.
        anchor: Target text or regex pattern to locate.
        replacement: Replacement content to apply.
        anchor_kind: Whether the anchor is treated as literal or regex.
        meta: Structured metadata associated with the rule.
        count: Optional maximum number of replacements; None means all matches.
        allow_zero_matches: Whether zero matches are allowed without error.
        count_specified: Whether the count key was explicitly provided.
    """

    rule_id: str
    description: str
    anchor: str
    replacement: str
    anchor_kind: str
    meta: Mapping[str, Any] | None
    count: int | None = None
    allow_zero_matches: bool = False
    count_specified: bool = False


@dataclass(frozen=True)
class InsertAfterRule:
    """Rule for inserting content after a resolved anchor.

    Args:
        rule_id: Unique identifier for the rule.
        description: Human-readable summary of the rule intent.
        anchor: Target text or regex pattern to locate.
        content: Content to insert after the anchor.
        anchor_kind: Whether the anchor is treated as literal or regex.
        meta: Structured metadata associated with the rule.
    """

    rule_id: str
    description: str
    anchor: str
    content: str
    anchor_kind: str
    meta: Mapping[str, Any] | None


@dataclass(frozen=True)
class DeleteRegexRule:
    """Rule for deleting text matching a regex pattern.

    Args:
        rule_id: Unique identifier for the rule.
        description: Human-readable summary of the rule intent.
        pattern: Regex pattern to remove from the source.
        flags: Compiled regex flags for the pattern.
        meta: Structured metadata associated with the rule.
        allow_zero_matches: Whether zero matches are allowed without error.
    """

    rule_id: str
    description: str
    pattern: str
    flags: int
    meta: Mapping[str, Any] | None
    allow_zero_matches: bool = False


Rule = ReplaceRule | InsertAfterRule | DeleteRegexRule



@dataclass(frozen=True)
class PatchRuleInput:
    """Schema for patch generation inputs.

    Args:
        source: Original source content to update.
        error_report: Structured metadata about the error context.
        rules: Parsed rules to apply in order.
    """

    source: str
    error_report: Mapping[str, Any]
    rules: list[Rule]


@dataclass(frozen=True)
class ResolvedRule:
    """Resolved edit with concrete spans in the source document.

    Args:
        rule_id: Identifier of the originating rule.
        description: Human-readable summary of the rule intent.
        rule_type: Resolved rule type.
        anchor: Anchor text or pattern.
        anchor_kind: Whether the anchor was literal or regex.
        start: Byte offset start position.
        end: Byte offset end position.
        replacement: Replacement text to apply.
        index: Original rule index in the input list.
        line_start: Starting line number.
        line_end: Ending line number.
        col_start: Starting column number.
        col_end: Ending column number.
    """

    rule_id: str
    description: str
    rule_type: str
    anchor: str
    anchor_kind: str
    start: int
    end: int
    replacement: str
    index: int
    line_start: int
    line_end: int
    col_start: int
    col_end: int


@dataclass(frozen=True)
class PatchChange:
    """Applied change details for auditing patch operations.

    Args:
        rule_id: Identifier of the originating rule.
        description: Human-readable summary of the rule intent.
        rule_type: Resolved rule type.
        start: Byte offset start position in the original source.
        end: Byte offset end position in the original source.
        line_start: Starting line number in the original source.
        line_end: Ending line number in the original source.
        col_start: Starting column number in the original source.
        col_end: Ending column number in the original source.
        before: Source snippet before the change.
        after: Replacement snippet applied for the change.
    """

    rule_id: str
    description: str
    rule_type: str
    start: int
    end: int
    line_start: int
    line_end: int
    col_start: int
    col_end: int
    before: str
    after: str


@dataclass(frozen=True)
class PatchAuditEntry:
    """Minimal audit entry recording ordered rule application."""

    order: int
    rule_id: str
    rule_type: str
    anchor_kind: str


@dataclass(frozen=True)
class PatchResult:
    """Structured output for applied patch rules."""

    content: str
    changes: list[PatchChange]
    audit_trail: list[PatchAuditEntry]
    resolved_rules: list[ResolvedRule]


def validate_rules(rules: list[Rule]) -> None:
    """Validate patch rules for deterministic, schema-safe behavior.

    Args:
        rules: Rule objects to validate.

    Raises:
        PatchRuleError: If a rule violates deterministic constraints.
    """
    if not isinstance(rules, list):
        raise PatchRuleError(
            "rules must be provided as a list",
            details={"field": "rules", "expected": "list", "actual_type": type(rules).__name__},
        )

    allowed_flag_mask = re.IGNORECASE | re.MULTILINE | re.DOTALL
    seen_rule_ids: dict[str, int] = {}
    for index, rule in enumerate(rules):
        if not isinstance(rule, (ReplaceRule, InsertAfterRule, DeleteRegexRule)):
            raise PatchRuleError(
                "rule must be a supported Rule type",
                details=_rule_details(index, rule, expected="Rule"),
            )

        _reject_non_literal_transformations(rule, index)

        rule_id = rule.rule_id
        if not isinstance(rule_id, str) or not rule_id.strip():
            raise PatchRuleError(
                "rule_id is required",
                details=_rule_details(index, rule, field="rule_id"),
            )
        prior_index = seen_rule_ids.get(rule_id)
        if prior_index is not None:
            raise PatchRuleError(
                "duplicate rule_id detected",
                details={
                    "rule_id": rule_id,
                    "rule_index": index,
                    "prior_index": prior_index,
                },
            )
        seen_rule_ids[rule_id] = index
        if not isinstance(rule.description, str) or not rule.description.strip():
            raise PatchRuleError(
                "description is required",
                details=_rule_details(index, rule, rule_id=rule_id, field="description"),
            )
        if rule.meta is None:
            raise PatchRuleError(
                "meta is required",
                details=_rule_details(index, rule, rule_id=rule_id, field="meta"),
            )
        if not isinstance(rule.meta, Mapping):
            raise PatchRuleError(
                "meta must be a mapping",
                details=_rule_details(index, rule, rule_id=rule_id, field="meta"),
            )

        if isinstance(rule, ReplaceRule):
            _require_non_empty_target(rule.anchor, index, rule, rule_id, field="anchor")
            _require_non_empty_target(
                rule.replacement, index, rule, rule_id, field="replacement"
            )
            _validate_anchor(rule.anchor, rule.anchor_kind, index, rule, rule_id)
            if rule.anchor_kind != "literal":
                raise PatchRuleError(
                    "replace rules must use literal anchors",
                    details=_rule_details(
                        index,
                        rule,
                        rule_id=rule_id,
                        field="anchor_kind",
                        anchor_kind=rule.anchor_kind,
                    ),
                )
            _validate_replace_count(rule.count, index, rule, rule_id)
            if not isinstance(rule.count_specified, bool):
                raise PatchRuleError(
                    "count_specified must be a boolean",
                    details=_rule_details(index, rule, rule_id=rule_id, field="count_specified"),
                )
            _validate_allow_zero_matches(rule.allow_zero_matches, index, rule, rule_id)
        elif isinstance(rule, InsertAfterRule):
            _require_non_empty_target(rule.anchor, index, rule, rule_id, field="anchor")
            _require_non_empty_target(rule.content, index, rule, rule_id, field="content")
            _validate_anchor(rule.anchor, rule.anchor_kind, index, rule, rule_id)
        elif isinstance(rule, DeleteRegexRule):
            _require_non_empty_target(rule.pattern, index, rule, rule_id, field="pattern")
            if rule.flags & ~allowed_flag_mask:
                raise PatchRuleError(
                    "regex flags are not supported",
                    details=_rule_details(
                        index, rule, rule_id=rule_id, field="flags", flags=rule.flags
                    ),
                )
            _validate_allow_zero_matches(rule.allow_zero_matches, index, rule, rule_id)


def apply_rules(source: str, rules: list[Rule]) -> PatchResult:
    """Apply deterministic, ordered rules to a source string.

    Rules are applied in the order they are provided to preserve caller intent
    while still enforcing deterministic conflict checks. Anchors are resolved
    against the original source using literal matching unless a rule explicitly
    opts into regex anchors.

    Args:
        source: Original source content to update.
        rules: Ordered rule objects to apply.

    Returns:
        A structured result containing the updated content and audit metadata.

    Raises:
        PatchRuleError: If the inputs or rule definitions are invalid.
        PatchConflictError: If the resolved edits overlap or conflict.
        PatchAnchorError: If a rule anchor fails to resolve.
    """
    if not isinstance(source, str):
        raise PatchRuleError(
            "source must be a string",
            details={"field": "source", "expected": "str", "actual_type": type(source).__name__},
        )
    if not source:
        raise PatchRuleError(
            "source must not be empty",
            details={"field": "source", "expected": "non-empty string"},
        )
    if not rules:
        raise PatchRuleError(
            "rules must not be empty",
            details={"field": "rules"},
        )
    validate_rules(rules)

    ordered_rules = list(rules)
    rule_lookup = {rule.rule_id: rule for rule in ordered_rules}

    line_index = _build_line_index(source)
    resolved_rules: list[ResolvedRule] = []
    for index, rule in enumerate(ordered_rules):
        resolved_rules.extend(_resolve_rule(source, rule, index, line_index))

    ordered_resolved = sorted(resolved_rules, key=_resolved_rule_order_key)
    _raise_on_conflicts(ordered_resolved, rule_lookup)

    updated_source = _apply_edits(source, ordered_resolved)
    changes = _build_patch_changes(source, ordered_resolved)
    audit_trail = [
        PatchAuditEntry(
            order=idx + 1,
            rule_id=rule.rule_id,
            rule_type=rule.rule_type,
            anchor_kind=rule.anchor_kind,
        )
        for idx, rule in enumerate(ordered_resolved)
    ]
    return PatchResult(
        content=updated_source,
        changes=changes,
        audit_trail=audit_trail,
        resolved_rules=list(ordered_resolved),
    )


def _rule_kind(rule: Rule) -> str:
    """Return the canonical rule type string."""
    if isinstance(rule, ReplaceRule):
        return "replace"
    if isinstance(rule, InsertAfterRule):
        return "insert_after"
    if isinstance(rule, DeleteRegexRule):
        return "delete_regex"
    raise PatchRuleError(
        "rule must be a supported Rule type",
        details={"rule": _serialize_rule_payload(rule)},
    )


def _allows_multiple_inserts(rule: InsertAfterRule) -> bool:
    """Check whether a rule explicitly permits multiple inserts at one anchor."""
    if not rule.meta:
        return False
    return bool(rule.meta.get("allow_multiple_inserts"))


def _allows_multiple_matches(rule: ReplaceRule) -> bool:
    """Check whether a replace rule explicitly allows multiple anchor matches."""
    if not rule.meta:
        return False
    return bool(rule.meta.get("allow_multiple_matches"))


def _raise_on_conflicts(
    resolved_rules: Sequence[ResolvedRule],
    rule_lookup: Mapping[str, Rule],
) -> None:
    """Raise PatchConflictError when edits overlap or collide."""
    inserts_by_position: dict[int, list[ResolvedRule]] = {}
    for rule in resolved_rules:
        if rule.start == rule.end:
            inserts_by_position.setdefault(rule.start, []).append(rule)

    for position, rules_at_position in inserts_by_position.items():
        if len(rules_at_position) < 2:
            continue
        allowed = all(
            isinstance(rule_lookup.get(rule.rule_id), InsertAfterRule)
            and _allows_multiple_inserts(rule_lookup[rule.rule_id])  # type: ignore[index]
            for rule in rules_at_position
        )
        if not allowed:
            sample = rules_at_position[0]
            raise PatchConflictError(
                "multiple inserts at the same anchor are not allowed",
                details={
                    "position": position,
                    "line": sample.line_start,
                    "col": sample.col_start,
                    "rule_ids": [rule.rule_id for rule in rules_at_position],
                    "rule_indices": [rule.index for rule in rules_at_position],
                    "allow_multiple_inserts": [
                        bool(
                            isinstance(rule_lookup.get(rule.rule_id), InsertAfterRule)
                            and _allows_multiple_inserts(rule_lookup[rule.rule_id])  # type: ignore[index]
                        )
                        for rule in rules_at_position
                    ],
                },
            )

    ordered = sorted(resolved_rules, key=lambda rule: (rule.start, rule.end, rule.index))
    for idx, rule in enumerate(ordered):
        for other in ordered[idx + 1 :]:
            if other.start >= rule.end and rule.start != rule.end:
                break
            if _spans_overlap(rule, other):
                raise PatchConflictError(
                    "conflicting edits detected",
                details={
                    "rule_id": rule.rule_id,
                    "rule_index": rule.index,
                    "conflicting_rule_id": other.rule_id,
                    "conflicting_rule_index": other.index,
                    "rule_type": rule.rule_type,
                    "conflicting_rule_type": other.rule_type,
                    "span": [rule.start, rule.end],
                    "conflicting_span": [other.start, other.end],
                        "line_offsets": {
                            "start_line": rule.line_start,
                            "start_col": rule.col_start,
                            "end_line": rule.line_end,
                            "end_col": rule.col_end,
                        },
                        "conflicting_line_offsets": {
                            "start_line": other.line_start,
                            "start_col": other.col_start,
                            "end_line": other.line_end,
                            "end_col": other.col_end,
                        },
                    },
                )


def _spans_overlap(first: ResolvedRule, second: ResolvedRule) -> bool:
    """Return True when two spans overlap in the original source."""
    first_is_insert = first.start == first.end
    second_is_insert = second.start == second.end
    if first_is_insert and second_is_insert:
        return first.start == second.start
    if first_is_insert:
        return second.start <= first.start < second.end
    if second_is_insert:
        return first.start <= second.start < first.end
    return max(first.start, second.start) < min(first.end, second.end)


def _build_patch_changes(
    source: str,
    resolved_rules: Sequence[ResolvedRule],
) -> list[PatchChange]:
    """Build a deterministic list of applied patch changes."""
    ordered = sorted(resolved_rules, key=_resolved_rule_order_key)
    return [
        PatchChange(
            rule_id=rule.rule_id,
            description=rule.description,
            rule_type=rule.rule_type,
            start=rule.start,
            end=rule.end,
            line_start=rule.line_start,
            line_end=rule.line_end,
            col_start=rule.col_start,
            col_end=rule.col_end,
            before=source[rule.start : rule.end],
            after=rule.replacement,
        )
        for rule in ordered
    ]


def _resolved_rule_order_key(rule: ResolvedRule) -> tuple[int, str, int, int]:
    """Return deterministic ordering for applied rule metadata."""
    return (rule.index, rule.rule_type, rule.start, rule.end)


def _resolved_rule_apply_key(rule: ResolvedRule) -> tuple[int, int, int, str]:
    """Return deterministic ordering for applying edits to the source."""
    return (rule.start, rule.end, rule.index, rule.rule_type)


def generate_patch(
    source: str,
    error_report: Mapping[str, Any],
    rules: Sequence[Mapping[str, Any]],
    *,
    validate_syntax: bool | None = None,
) -> dict[str, object]:
    """Generate a deterministic patch payload by applying explicit rules.

    Args:
        source: The original source content to modify.
        error_report: Structured metadata about the error context.
        rules: Patch rules to apply. Each rule is a mapping with:
            - type: "replace", "insert_after", or "delete_regex".
            - id: Unique rule identifier.
            - description: Required, non-empty string description.
            - anchor/anchor_kind/replacement: Required for replace.
            - count: Optional integer for replace max replacements, or "all".
            - allow_zero_matches: Optional bool to permit zero matches.
            - anchor/anchor_kind/content: Required for insert_after.
            - pattern/flags: Required for delete_regex (flags are strings such as
              "IGNORECASE", "MULTILINE", or "DOTALL").
            - meta: Required mapping of rule metadata. Set meta["validate_syntax"]
              to True to force syntax validation of the modified source.
        validate_syntax: Explicit override for syntax validation. When None, validation
            runs automatically for supported languages (Python).

    Returns:
        A structured payload containing status, data, errors, and meta fields.
    """
    parsed_rules: list[Rule] = []
    rule_summaries: list[dict[str, Any]] = []
    try:
        _validate_inputs(source, error_report, rules)
        parsed_rules = _coerce_rules(rules)
        validate_rules(parsed_rules)
        rule_summaries = _summarize_rules(parsed_rules)

        if not parsed_rules:
            error = PatchRuleError("rules must not be empty", details={"field": "rules"})
            return _deterministic_error_payload(
                error,
                source=source,
                rule_summaries=rule_summaries,
                notes=["empty_rules"],
            )

        errors: list[dict[str, Any]] = []
        try:
            result = apply_rules(source, parsed_rules)
        except PatchConflictError as exc:
            errors.append(exc.to_dict())
            return _failure_result(
                errors,
                meta=_build_meta(
                    source=source,
                    rule_summaries=rule_summaries,
                    applied_count=0,
                    applied_rules=[],
                    anchor_resolutions=[],
                    syntax_valid=None,
                ),
            )
        except PatchAnchorError as exc:
            raise

        try:
            patch_text = render_patch(result)
        except MenaceError as exc:
            if isinstance(exc, (PatchRuleError, PatchAnchorError)):
                raise
            errors.append(exc.to_dict())
            return _failure_result(
                errors,
                meta=_build_meta(
                    source=source,
                    rule_summaries=rule_summaries,
                    applied_count=0,
                    applied_rules=[],
                    anchor_resolutions=_serialize_anchor_resolutions(result.resolved_rules),
                    syntax_valid=None,
                ),
            )

        syntax_valid: bool | None = None
        should_validate, detected_language = _should_validate_modified_source(
            validate_syntax,
            error_report,
            parsed_rules,
        )
        if should_validate:
            syntax_error = _check_syntax(
                result.content,
                error_report,
                parsed_rules,
                language=detected_language,
            )
            if syntax_error:
                errors.append(syntax_error.to_dict())
                syntax_valid = False
            else:
                syntax_valid = True

        status = "ok" if not errors else "error"
        data: dict[str, object] = {
            "patch_text": patch_text,
            "modified_source": result.content,
            "updated_source": result.content,
            "applied_rules": _serialize_rules(result.resolved_rules),
            "changes": [
                {
                    "rule_id": change.rule_id,
                    "rule_type": change.rule_type,
                    "description": change.description,
                    "spans": {"start": change.start, "end": change.end},
                    "line_offsets": {
                        "start_line": change.line_start,
                        "start_col": change.col_start,
                        "end_line": change.line_end,
                        "end_col": change.col_end,
                    },
                    "before": change.before,
                    "after": change.after,
                }
                for change in result.changes
            ],
            "audit_trail": [asdict(entry) for entry in result.audit_trail],
        }

        return {
            "status": status,
            "data": data,
            "errors": errors,
            "meta": _build_meta(
                source=source,
                rule_summaries=rule_summaries,
                applied_count=len(result.changes),
                applied_rules=_serialize_applied_rules(result.resolved_rules),
                anchor_resolutions=_serialize_anchor_resolutions(result.resolved_rules),
                changed_line_count=_count_changed_lines(patch_text),
                syntax_valid=syntax_valid,
            ),
        }
    except Exception as exc:
        if isinstance(exc, (PatchRuleError, PatchAnchorError)):
            raise
        error = exc if isinstance(exc, MenaceError) else MenaceError(
            "unexpected error during patch generation",
            details={"error_type": type(exc).__name__, "message": str(exc)},
        )
        return _deterministic_error_payload(
            error,
            source=source,
            rule_summaries=rule_summaries,
            applied_rules=[],
        )


def _validate_generate_patch_inputs(
    source: str,
    error_report: Mapping[str, object],
    rules: Sequence[Rule],
) -> None:
    """Validate input types for patch generation."""
    if not isinstance(source, str):
        raise PatchRuleError(
            "source must be a string",
            details={"field": "source", "expected": "str", "actual_type": type(source).__name__},
        )
    if not source:
        raise PatchRuleError(
            "source must not be empty",
            details={"field": "source", "expected": "non-empty string"},
        )
    if not isinstance(error_report, Mapping):
        raise PatchRuleError(
            "error_report must be a mapping",
            details={
                "field": "error_report",
                "expected": "mapping",
                "actual_type": type(error_report).__name__,
            },
        )
    if not isinstance(rules, Sequence) or isinstance(rules, (str, bytes)):
        raise PatchRuleError(
            "rules must be a sequence of Rule objects",
            details={"field": "rules", "expected": "sequence", "actual_type": type(rules).__name__},
        )
    for index, rule in enumerate(rules):
        if not isinstance(rule, (ReplaceRule, InsertAfterRule, DeleteRegexRule)):
            raise PatchRuleError(
                "rule must be a supported Rule type",
                details=_rule_details(index, rule, expected="Rule"),
            )


def _summarize_rules(rules: Sequence[Rule]) -> list[dict[str, Any]]:
    """Create deterministic summaries for patch rules."""
    summaries: list[dict[str, Any]] = []
    for rule in rules:
        summaries.append(
            {
                "id": rule.rule_id,
                "type": _rule_kind(rule),
                "description": rule.description,
            }
        )
    return summaries


def _build_meta(
    *,
    source: str,
    rule_summaries: Sequence[Mapping[str, Any]],
    applied_count: int,
    applied_rules: Sequence[Mapping[str, Any]] | None = None,
    anchor_resolutions: Sequence[Mapping[str, Any]] | None = None,
    changed_line_count: int = 0,
    syntax_valid: bool | None = None,
    notes: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Build deterministic metadata for generate_patch."""
    applied_rules_list = list(applied_rules or [])
    rule_application_order = [
        rule.get("id") for rule in applied_rules_list if isinstance(rule, Mapping) and "id" in rule
    ]
    return {
        "rule_summaries": list(rule_summaries),
        "rule_count": len(rule_summaries),
        "applied_count": applied_count,
        "total_changes": applied_count,
        "applied_rule_ids": [rule["id"] for rule in applied_rules_list if "id" in rule],
        "rule_application_order": rule_application_order,
        "applied_rules": applied_rules_list,
        "anchor_resolutions": list(anchor_resolutions or []),
        "changed_line_count": changed_line_count,
        "syntax_valid": syntax_valid,
        "validation_results": {"syntax_valid": syntax_valid},
        "input_hash": _hash_input(source, rule_summaries),
        "notes": list(notes or []),
    }


def _hash_input(source: str, rule_summaries: Sequence[Mapping[str, Any]]) -> str:
    """Create a deterministic hash of the input source and rule summaries."""
    payload = {
        "source": source,
        "rules": list(rule_summaries),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _failure_result(
    errors: Sequence[dict[str, Any]],
    *,
    meta: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a structured failure payload."""
    return {
        "status": "error",
        "data": _empty_data_payload(),
        "errors": list(errors),
        "meta": dict(meta),
    }


def _deterministic_error_payload(
    error: MenaceError,
    *,
    source: str,
    rule_summaries: Sequence[Mapping[str, Any]],
    applied_count: int = 0,
    applied_rules: Sequence[Mapping[str, Any]] | None = None,
    anchor_resolutions: Sequence[Mapping[str, Any]] | None = None,
    syntax_valid: bool | None = None,
    notes: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Return a deterministic error payload with empty data."""
    return {
        "status": "error",
        "data": _empty_data_payload(),
        "errors": [error.to_dict()],
        "meta": _build_meta(
            source=source,
            rule_summaries=rule_summaries,
            applied_count=applied_count,
            applied_rules=applied_rules,
            anchor_resolutions=anchor_resolutions or [],
            syntax_valid=syntax_valid,
            notes=notes,
        ),
    }


def _empty_data_payload() -> dict[str, Any]:
    """Return a stable empty data payload for error responses."""
    return {
        "patch_text": "",
        "modified_source": "",
        "updated_source": "",
        "applied_rules": [],
        "changes": [],
        "audit_trail": [],
    }


def _serialize_rule_payload(rule: Any) -> Any:
    """Return a JSON-friendly rule payload for error details."""
    if is_dataclass(rule):
        return asdict(rule)
    if isinstance(rule, Mapping):
        return dict(rule)
    return rule


def _rule_details(index: int, rule: Any, *, rule_id: str | None = None, **extra: Any) -> dict[str, Any]:
    """Build structured error details for a rule."""
    details: dict[str, Any] = {
        "rule_index": index,
        "rule": _serialize_rule_payload(rule),
    }
    if rule_id is not None:
        details["rule_id"] = rule_id
    details.update(extra)
    return details


def _validate_inputs(
    source: str,
    error_report: Mapping[str, Any],
    rules: Sequence[Mapping[str, Any]] | Sequence[Rule],
) -> None:
    """Validate the top-level inputs for patch generation."""
    if not isinstance(source, str):
        raise PatchRuleError(
            "source must be a string",
            details={"field": "source", "expected": "str", "actual_type": type(source).__name__},
        )
    if not source:
        raise PatchRuleError(
            "source must not be empty",
            details={"field": "source", "expected": "non-empty string"},
        )
    if not isinstance(error_report, Mapping):
        raise PatchRuleError(
            "error_report must be a mapping",
            details={
                "field": "error_report",
                "expected": "mapping",
                "actual_type": type(error_report).__name__,
            },
        )
    if not isinstance(rules, Sequence) or isinstance(rules, (str, bytes)):
        raise PatchRuleError(
            "rules must be a sequence",
            details={"field": "rules", "expected": "sequence", "actual_type": type(rules).__name__},
        )
    for index, rule in enumerate(rules):
        if isinstance(rule, (ReplaceRule, InsertAfterRule, DeleteRegexRule)):
            continue
        if not isinstance(rule, Mapping):
            raise PatchRuleError(
                "rule must be a mapping or Rule instance",
                details=_rule_details(index, rule, expected="mapping or Rule"),
            )


def _coerce_rules(
    rules: Sequence[Mapping[str, Any]] | Sequence[Rule],
) -> list[ReplaceRule | InsertAfterRule | DeleteRegexRule]:
    """Normalize input rule definitions into concrete Rule objects."""
    if not rules:
        return []
    if all(isinstance(rule, Mapping) for rule in rules):
        return _parse_rules(rules)  # type: ignore[arg-type]
    if all(isinstance(rule, (ReplaceRule, InsertAfterRule, DeleteRegexRule)) for rule in rules):
        return list(rules)  # type: ignore[list-item]
    raise PatchRuleError(
        "rules must be provided as mappings or Rule objects",
        details={"field": "rules", "expected": "mapping or Rule"},
    )


def _parse_rules(
    rules: Sequence[Mapping[str, Any]],
) -> list[ReplaceRule | InsertAfterRule | DeleteRegexRule]:
    """Parse and validate rule definitions."""
    parsed: list[ReplaceRule | InsertAfterRule | DeleteRegexRule] = []
    seen_ids: set[str] = set()
    for index, rule in enumerate(rules):
        if not isinstance(rule, Mapping):
            raise PatchRuleError(
                "rule must be a mapping",
                details=_rule_details(index, rule, expected="mapping"),
            )
        rule_type = _require_non_empty_str(rule, "type", index)
        rule_id = _require_non_empty_str(rule, "id", index)
        if rule_id in seen_ids:
            raise PatchRuleError(
                "rule id must be unique",
                details=_rule_details(index, rule, rule_id=rule_id),
            )
        seen_ids.add(rule_id)
        if rule_type == "replace":
            parsed.append(_parse_replace(rule, index, rule_id))
        elif rule_type == "insert_after":
            parsed.append(_parse_insert_after(rule, index, rule_id))
        elif rule_type == "delete_regex":
            parsed.append(_parse_delete_regex(rule, index, rule_id))
        else:
            raise PatchRuleError(
                "Unknown rule type",
                details=_rule_details(
                    index,
                    rule,
                    rule_id=rule_id,
                    rule_type=rule_type,
                    supported_types=["delete_regex", "insert_after", "replace"],
                ),
            )
    return parsed


def _parse_replace(rule: Mapping[str, Any], index: int, rule_id: str) -> ReplaceRule:
    """Parse a replace rule definition."""
    count_specified = "count" in rule
    _ensure_only_keys(
        rule,
        {
            "type",
            "id",
            "description",
            "anchor",
            "anchor_kind",
            "replacement",
            "count",
            "allow_zero_matches",
            "meta",
        },
        index,
        rule_id,
    )
    description = _parse_description(rule, index, rule_id)
    anchor = _require_non_empty_str(rule, "anchor", index)
    replacement = _require_non_empty_str(rule, "replacement", index)
    anchor_kind = _parse_anchor_kind(rule, index, rule_id)
    count = _parse_replace_count(rule, index, rule_id)
    allow_zero_matches = _parse_allow_zero_matches(rule, index, rule_id)
    meta = _parse_meta(rule, index, rule_id)
    return ReplaceRule(
        rule_id=rule_id,
        description=description,
        anchor=anchor,
        replacement=replacement,
        anchor_kind=anchor_kind,
        count=count,
        allow_zero_matches=allow_zero_matches,
        count_specified=count_specified,
        meta=meta,
    )


def _parse_insert_after(rule: Mapping[str, Any], index: int, rule_id: str) -> InsertAfterRule:
    """Parse an insert-after rule definition."""
    _ensure_only_keys(
        rule,
        {"type", "id", "description", "anchor", "anchor_kind", "content", "meta"},
        index,
        rule_id,
    )
    description = _parse_description(rule, index, rule_id)
    anchor = _require_non_empty_str(rule, "anchor", index)
    content = _require_non_empty_str(rule, "content", index)
    anchor_kind = _parse_anchor_kind(rule, index, rule_id)
    meta = _parse_meta(rule, index, rule_id)
    return InsertAfterRule(
        rule_id=rule_id,
        description=description,
        anchor=anchor,
        content=content,
        anchor_kind=anchor_kind,
        meta=meta,
    )


def _parse_delete_regex(rule: Mapping[str, Any], index: int, rule_id: str) -> DeleteRegexRule:
    """Parse a delete-regex rule definition."""
    _ensure_only_keys(
        rule,
        {"type", "id", "description", "pattern", "flags", "allow_zero_matches", "meta"},
        index,
        rule_id,
    )
    description = _parse_description(rule, index, rule_id)
    pattern = _require_non_empty_str(rule, "pattern", index)
    flags = rule.get("flags", [])
    if not isinstance(flags, Sequence) or isinstance(flags, (str, bytes)):
        raise PatchRuleError(
            "delete_regex flags must be a sequence",
            details=_rule_details(index, rule, rule_id=rule_id, field="flags"),
        )
    compiled_flags = 0
    for flag in flags:
        if not isinstance(flag, str) or not flag.strip():
            raise PatchRuleError(
                "delete_regex flag must be a non-empty string",
                details=_rule_details(index, rule, rule_id=rule_id, flag=flag),
            )
        if flag not in {"IGNORECASE", "MULTILINE", "DOTALL"}:
            raise PatchRuleError(
                "delete_regex flag is invalid",
                details=_rule_details(index, rule, rule_id=rule_id, flag=flag),
            )
        compiled_flags |= _FLAG_MAP[flag]
    try:
        re.compile(pattern, compiled_flags)
    except re.error as exc:
        raise PatchRuleError(
            "delete_regex pattern is invalid",
            details=_rule_details(index, rule, rule_id=rule_id, message=str(exc)),
        ) from exc
    allow_zero_matches = _parse_allow_zero_matches(rule, index, rule_id)
    meta = _parse_meta(rule, index, rule_id)
    return DeleteRegexRule(
        rule_id=rule_id,
        description=description,
        pattern=pattern,
        flags=compiled_flags,
        allow_zero_matches=allow_zero_matches,
        meta=meta,
    )


_FLAG_MAP = {
    "IGNORECASE": re.IGNORECASE,
    "MULTILINE": re.MULTILINE,
    "DOTALL": re.DOTALL,
}


def _ensure_only_keys(
    rule: Mapping[str, Any],
    allowed_keys: set[str],
    index: int,
    rule_id: str,
) -> None:
    """Ensure a rule mapping does not include unknown keys."""
    extra = sorted(key for key in rule.keys() if key not in allowed_keys)
    if extra:
        raise PatchRuleError(
            "rule has unexpected fields",
            details=_rule_details(index, rule, rule_id=rule_id, unexpected=extra),
        )


def _require_non_empty_str(rule: Mapping[str, Any], field: str, index: int) -> str:
    """Extract a non-empty string field from a rule mapping."""
    value = rule.get(field)
    if not isinstance(value, str):
        raise PatchRuleError(
            f"{field} must be a string",
            details=_rule_details(index, rule, field=field),
        )
    if not value.strip():
        raise PatchRuleError(
            f"{field} must be a non-empty string",
            details=_rule_details(index, rule, field=field),
        )
    return value


def _parse_description(rule: Mapping[str, Any], index: int, rule_id: str) -> str:
    """Parse a required description field."""
    if "description" not in rule:
        raise PatchRuleError(
            "description is required",
            details=_rule_details(index, rule, rule_id=rule_id, field="description"),
        )
    description = rule.get("description")
    if not isinstance(description, str):
        raise PatchRuleError(
            "description must be a string",
            details=_rule_details(index, rule, rule_id=rule_id, field="description"),
        )
    if not description.strip():
        raise PatchRuleError(
            "description must be a non-empty string",
            details=_rule_details(index, rule, rule_id=rule_id, field="description"),
        )
    return description


def _require_non_empty_target(
    target: str,
    index: int,
    rule: Rule,
    rule_id: str,
    *,
    field: str,
) -> None:
    """Ensure a rule target string is non-empty."""
    if not isinstance(target, str) or not target.strip():
        raise PatchRuleError(
            f"{field} must be a non-empty string",
            details=_rule_details(index, rule, rule_id=rule_id, field=field),
        )


def _validate_anchor(
    anchor: str,
    anchor_kind: str,
    index: int,
    rule: Rule,
    rule_id: str,
) -> None:
    """Ensure anchors are deterministic and unambiguous."""
    if anchor_kind == "regex":
        try:
            re.compile(anchor)
        except re.error as exc:
            raise PatchRuleError(
                "anchor regex is invalid",
                details=_rule_details(index, rule, rule_id=rule_id, anchor=anchor, message=str(exc)),
            ) from exc


def _parse_anchor_kind(rule: Mapping[str, Any], index: int, rule_id: str) -> str:
    """Parse the anchor kind field."""
    anchor_kind = rule.get("anchor_kind", "literal")
    if not isinstance(anchor_kind, str) or not anchor_kind.strip():
        raise PatchRuleError(
            "anchor_kind must be a non-empty string",
            details=_rule_details(index, rule, rule_id=rule_id, field="anchor_kind"),
        )
    if anchor_kind not in {"literal", "regex"}:
        raise PatchRuleError(
            "anchor_kind must be literal or regex",
            details=_rule_details(index, rule, rule_id=rule_id, anchor_kind=anchor_kind),
        )
    return anchor_kind


def _parse_replace_count(rule: Mapping[str, Any], index: int, rule_id: str) -> int | None:
    """Parse optional count for replace rules."""
    if "count" not in rule:
        return None
    value = rule.get("count")
    if value is None:
        return None
    if isinstance(value, str):
        if value.strip().lower() == "all":
            return None
        raise PatchRuleError(
            "replace count must be an integer or 'all'",
            details=_rule_details(index, rule, rule_id=rule_id, field="count", value=value),
        )
    if isinstance(value, bool) or not isinstance(value, int):
        raise PatchRuleError(
            "replace count must be an integer",
            details=_rule_details(index, rule, rule_id=rule_id, field="count", value=value),
        )
    if value <= 0:
        raise PatchRuleError(
            "replace count must be greater than zero",
            details=_rule_details(index, rule, rule_id=rule_id, field="count", value=value),
        )
    return value


def _parse_allow_zero_matches(rule: Mapping[str, Any], index: int, rule_id: str) -> bool:
    """Parse optional allow_zero_matches field."""
    value = rule.get("allow_zero_matches", False)
    if not isinstance(value, bool):
        raise PatchRuleError(
            "allow_zero_matches must be a boolean",
            details=_rule_details(index, rule, rule_id=rule_id, field="allow_zero_matches"),
        )
    return value


def _validate_replace_count(
    count: int | None,
    index: int,
    rule: Rule,
    rule_id: str,
) -> None:
    """Validate replace count on parsed rules."""
    if count is None:
        return
    if not isinstance(count, int) or count <= 0:
        raise PatchRuleError(
            "replace count must be a positive integer",
            details=_rule_details(index, rule, rule_id=rule_id, field="count", value=count),
        )


def _validate_allow_zero_matches(
    allow_zero_matches: bool,
    index: int,
    rule: Rule,
    rule_id: str,
) -> None:
    """Validate allow_zero_matches on parsed rules."""
    if not isinstance(allow_zero_matches, bool):
        raise PatchRuleError(
            "allow_zero_matches must be a boolean",
            details=_rule_details(
                index,
                rule,
                rule_id=rule_id,
                field="allow_zero_matches",
                value=allow_zero_matches,
            ),
        )


_DISALLOWED_TRANSFORMATION_PATTERN = re.compile(
    r"\b(auto[- ]?format|formatting|reformat|refactor|reindent|lint|prettier|black|isort|"
    r"organize imports|sort imports)\b",
    re.IGNORECASE,
)


def _reject_non_literal_transformations(rule: Rule, index: int) -> None:
    """Reject rule metadata that implies formatting/refactor operations."""
    description = getattr(rule, "description", "")
    if isinstance(description, str) and _DISALLOWED_TRANSFORMATION_PATTERN.search(description):
        raise PatchRuleError(
            "rule description indicates formatting/refactor operation",
            details=_rule_details(index, rule, rule_id=getattr(rule, "rule_id", None)),
        )
    meta = getattr(rule, "meta", None)
    if not isinstance(meta, Mapping):
        return
    for key, value in meta.items():
        if isinstance(key, str) and _DISALLOWED_TRANSFORMATION_PATTERN.search(key):
            if bool(value):
                raise PatchRuleError(
                    "rule metadata indicates formatting/refactor operation",
                    details=_rule_details(
                        index,
                        rule,
                        rule_id=getattr(rule, "rule_id", None),
                        field=key,
                        value=value,
                    ),
                )
        if isinstance(value, str) and _DISALLOWED_TRANSFORMATION_PATTERN.search(value):
            raise PatchRuleError(
                "rule metadata indicates formatting/refactor operation",
                details=_rule_details(
                    index,
                    rule,
                    rule_id=getattr(rule, "rule_id", None),
                    field=key,
                    value=value,
                ),
            )


def _parse_meta(rule: Mapping[str, Any], index: int, rule_id: str) -> Mapping[str, Any]:
    """Parse required meta data for a rule."""
    if "meta" not in rule:
        raise PatchRuleError(
            "meta is required",
            details=_rule_details(index, rule, rule_id=rule_id, field="meta"),
        )
    meta = rule.get("meta")
    if meta is None:
        raise PatchRuleError(
            "meta is required",
            details=_rule_details(index, rule, rule_id=rule_id, field="meta"),
        )
    if not isinstance(meta, Mapping):
        raise PatchRuleError(
            "meta must be a mapping",
            details=_rule_details(index, rule, rule_id=rule_id, field="meta"),
        )
    return meta


def _resolve_rule(
    source: str,
    rule: ReplaceRule | InsertAfterRule | DeleteRegexRule,
    index: int,
    line_index: list[int],
) -> list[ResolvedRule]:
    """Resolve a rule into one or more concrete edits."""
    if isinstance(rule, ReplaceRule):
        spans = _resolve_replace(
            source,
            rule.anchor,
            rule.count,
            rule.allow_zero_matches,
            rule.rule_id,
            rule_index=index,
            rule_payload=rule,
        )
        return [
            _build_resolved_rule(
                rule.rule_id,
                rule.description,
                "replace",
                rule.anchor,
                rule.anchor_kind,
                span_start,
                span_end,
                rule.replacement,
                index,
                line_index,
            )
            for span_start, span_end in spans
        ]
    if isinstance(rule, InsertAfterRule):
        start, end = _resolve_anchor(
            source,
            rule.anchor,
            rule.anchor_kind,
            rule.rule_id,
            rule_index=index,
            rule_payload=rule,
        )
        return [
            _build_resolved_rule(
                rule.rule_id,
                rule.description,
                "insert_after",
                rule.anchor,
                rule.anchor_kind,
                end,
                end,
                rule.content,
                index,
                line_index,
            )
        ]
    if isinstance(rule, DeleteRegexRule):
        spans = _resolve_delete_lines(
            source,
            rule.pattern,
            rule.flags,
            rule.allow_zero_matches,
            rule.rule_id,
            rule_index=index,
            rule_payload=rule,
        )
        return [
            _build_resolved_rule(
                rule.rule_id,
                rule.description,
                "delete_regex",
                rule.pattern,
                "regex",
                span_start,
                span_end,
                "",
                index,
                line_index,
            )
            for span_start, span_end in spans
        ]
    raise PatchRuleError(
        "Unknown rule type",
        details=_rule_details(
            index,
            rule,
            rule_id=getattr(rule, "rule_id", None),
        ),
    )


def _resolve_delete_lines(
    source: str,
    pattern: str,
    flags: int,
    allow_zero_matches: bool,
    rule_id: str,
    *,
    rule_index: int,
    rule_payload: DeleteRegexRule,
) -> list[tuple[int, int]]:
    """Resolve delete rules into line spans matching the regex."""
    compiled = re.compile(pattern, flags)
    spans: list[tuple[int, int]] = []
    line_index = _build_line_index(source)
    for line_number, line_start in enumerate(line_index, start=1):
        line_end = line_index[line_number] if line_number < len(line_index) else len(source)
        line_text = source[line_start:line_end]
        line_content = line_text[:-1] if line_text.endswith("\n") else line_text
        if compiled.search(line_content):
            spans.append((line_start, line_end))
    if not spans and not allow_zero_matches:
        raise PatchAnchorError(
            "delete_regex pattern matched no lines",
            details=_rule_details(
                rule_index,
                rule_payload,
                rule_id=rule_id,
                anchor_search={
                    "pattern": pattern,
                    "flags": flags,
                    "match_count": 0,
                    "lines_checked": len(line_index),
                    "allow_zero_matches": allow_zero_matches,
                },
            ),
        )
    return spans


def _resolve_replace(
    source: str,
    anchor: str,
    count: int | None,
    allow_zero_matches: bool,
    rule_id: str,
    *,
    rule_index: int,
    rule_payload: ReplaceRule,
) -> list[tuple[int, int]]:
    """Resolve literal replace rules into one or more match spans."""
    matches = _find_literal_matches_non_overlapping(source, anchor)
    if not matches:
        if allow_zero_matches:
            return []
        raise PatchAnchorError(
            "anchor not found",
            details=_rule_details(
                rule_index,
                rule_payload,
                rule_id=rule_id,
                anchor_search={
                    "anchor": anchor,
                    "anchor_kind": "literal",
                    "match_count": 0,
                    "matches": [],
                    "allow_zero_matches": allow_zero_matches,
                },
            ),
        )
    if (
        len(matches) > 1
        and count is None
        and not rule_payload.count_specified
        and not _allows_multiple_matches(rule_payload)
    ):
        raise PatchAnchorError(
            "anchor is ambiguous",
            details=_rule_details(
                rule_index,
                rule_payload,
                rule_id=rule_id,
                anchor_search={
                    "anchor": anchor,
                    "anchor_kind": "literal",
                    "match_count": len(matches),
                    "matches": list(matches),
                    "count": count,
                    "count_specified": rule_payload.count_specified,
                    "allow_multiple_matches": _allows_multiple_matches(rule_payload),
                },
            ),
        )
    if count is None:
        return list(matches)
    return list(matches[:count])


def _resolve_anchor(
    source: str,
    anchor: str,
    anchor_kind: str,
    rule_id: str,
    *,
    rule_index: int,
    rule_payload: ReplaceRule | InsertAfterRule,
) -> tuple[int, int]:
    """Resolve an anchor to a single span in the source."""
    if anchor_kind == "literal":
        matches = _find_literal_matches(source, anchor)
    else:
        matches = _find_regex_matches(source, anchor, 0)
    return _select_single_match(
        matches,
        rule_id,
        anchor,
        anchor_kind=anchor_kind,
        rule_index=rule_index,
        rule_payload=rule_payload,
    )


def _find_literal_matches(text: str, anchor: str) -> list[tuple[int, int]]:
    """Return all literal match spans for an anchor."""
    matches: list[tuple[int, int]] = []
    start = 0
    while True:
        idx = text.find(anchor, start)
        if idx == -1:
            break
        matches.append((idx, idx + len(anchor)))
        start = idx + 1
    return matches


def _find_literal_matches_non_overlapping(text: str, anchor: str) -> list[tuple[int, int]]:
    """Return non-overlapping literal match spans for a substring."""
    matches: list[tuple[int, int]] = []
    start = 0
    while True:
        idx = text.find(anchor, start)
        if idx == -1:
            break
        matches.append((idx, idx + len(anchor)))
        start = idx + len(anchor)
    return matches


def _find_regex_matches(text: str, pattern: str, flags: int) -> list[tuple[int, int]]:
    """Return all regex match spans for a pattern."""
    compiled = re.compile(pattern, flags)
    return [(match.start(), match.end()) for match in compiled.finditer(text)]


def _select_single_match(
    matches: Sequence[tuple[int, int]],
    rule_id: str,
    anchor: str,
    *,
    anchor_kind: str,
    rule_index: int,
    rule_payload: ReplaceRule | InsertAfterRule,
) -> tuple[int, int]:
    """Ensure a single match is selected for a rule anchor."""
    if not matches:
        raise PatchAnchorError(
            "anchor not found",
            details=_rule_details(
                rule_index,
                rule_payload,
                rule_id=rule_id,
                anchor_search={
                    "anchor": anchor,
                    "anchor_kind": anchor_kind,
                    "match_count": 0,
                    "matches": [],
                },
            ),
        )
    if len(matches) > 1:
        raise PatchAnchorError(
            "anchor is ambiguous",
            details=_rule_details(
                rule_index,
                rule_payload,
                rule_id=rule_id,
                anchor_search={
                    "anchor": anchor,
                    "anchor_kind": anchor_kind,
                    "match_count": len(matches),
                    "matches": list(matches),
                },
            ),
        )
    return matches[0]


def _build_resolved_rule(
    rule_id: str,
    description: str,
    rule_type: str,
    anchor: str,
    anchor_kind: str,
    start: int,
    end: int,
    replacement: str,
    index: int,
    line_index: list[int],
) -> ResolvedRule:
    """Build a resolved rule with line/column metadata."""
    line_start, col_start = _line_and_column(line_index, start)
    line_end, col_end = _line_and_column(line_index, end)
    return ResolvedRule(
        rule_id=rule_id,
        description=description,
        rule_type=rule_type,
        anchor=anchor,
        anchor_kind=anchor_kind,
        start=start,
        end=end,
        replacement=replacement,
        index=index,
        line_start=line_start,
        line_end=line_end,
        col_start=col_start,
        col_end=col_end,
    )


def _line_and_column(line_index: Sequence[int], offset: int) -> tuple[int, int]:
    """Convert an offset into line and column coordinates."""
    line_number = 1
    for idx, line_start in enumerate(line_index):
        if line_start > offset:
            line_number = idx
            break
        line_number = idx + 1
    line_start = line_index[line_number - 1] if line_number - 1 < len(line_index) else 0
    return line_number, offset - line_start


def _build_line_index(text: str) -> list[int]:
    """Build a list of line start offsets for a source string."""
    indices = [0]
    for idx, char in enumerate(text):
        if char == "\n":
            indices.append(idx + 1)
    return indices


def _detect_conflicts(resolved_rules: Sequence[ResolvedRule]) -> dict[str, Any] | None:
    """Detect conflicting edits across resolved rules."""
    for idx, rule in enumerate(resolved_rules):
        for other in resolved_rules[idx + 1 :]:
            if _rules_conflict(rule, other):
                error = PatchConflictError(
                    "conflicting edits detected",
                    details={
                        "rule_id": rule.rule_id,
                        "conflicting_rule_id": other.rule_id,
                        "span": [rule.start, rule.end],
                        "conflicting_span": [other.start, other.end],
                    },
                )
                return error.to_dict()
    return None


def _rules_conflict(first: ResolvedRule, second: ResolvedRule) -> bool:
    """Determine whether two resolved rules overlap."""
    if first.start == first.end and second.start == second.end:
        return first.start == second.start
    if first.start == first.end:
        return second.start <= first.start <= second.end
    if second.start == second.end:
        return first.start <= second.start <= first.end
    return not (first.end <= second.start or second.end <= first.start)


def _apply_edits(source: str, resolved_rules: Sequence[ResolvedRule]) -> str:
    """Apply resolved edits to the source in deterministic order."""
    ordered = sorted(resolved_rules, key=_resolved_rule_apply_key)
    offset = 0
    updated = source
    for rule in ordered:
        start = rule.start + offset
        end = rule.end + offset
        updated = updated[:start] + rule.replacement + updated[end:]
        offset += len(rule.replacement) - (rule.end - rule.start)
    return updated


def _build_diff(before: str, after: str) -> str:
    """Build a unified diff for the before/after text."""
    before_lines = _lines_for_diff(before)
    after_lines = _lines_for_diff(after)
    diff_lines = difflib.unified_diff(
        before_lines,
        after_lines,
        fromfile="before",
        tofile="after",
        lineterm="",
    )
    return "\n".join(diff_lines)


def _validate_patch_text(patch_text: str, resolved_rules: Sequence[ResolvedRule]) -> None:
    """Ensure the generated patch text is parseable."""
    if not patch_text:
        return
    lines = patch_text.splitlines()
    has_header = any(line.startswith("--- ") for line in lines) and any(
        line.startswith("+++ ") for line in lines
    )
    has_hunk = any(line.startswith("@@") for line in lines)
    if not has_header or not has_hunk:
        raise PatchParseError(
            "Generated patch text is not a valid unified diff",
            details={
                "has_header": has_header,
                "has_hunk": has_hunk,
                "rule_ids": [rule.rule_id for rule in resolved_rules],
                "rule_count": len(resolved_rules),
            },
        )


def _lines_for_diff(text: str) -> list[str]:
    """Prepare lines for unified diff output."""
    lines = text.splitlines()
    if text.endswith("\n"):
        lines.append("")
    return [f"{line}\n" for line in lines]


_PATCH_MAGIC = "MENACE-PATCH 1"
_PATCH_COUNT_PREFIX = "change-count: "
_PATCH_RULE_PREFIX = "@@@ rule "
_PATCH_DESCRIPTION_PREFIX = "@@@ description "
_PATCH_TYPE_PREFIX = "@@@ type "
_PATCH_RANGE_PREFIX = "@@@ range "
_PATCH_HUNK_HEADER = re.compile(r"^@@ -(?P<start>\d+),(?P<count>\d+) \+(?P<start_new>\d+),(?P<count_new>\d+) @@$")
_PATCH_RANGE = re.compile(
    r"^@@@ range bytes (?P<byte_start>\d+)-(?P<byte_end>\d+) "
    r"lines (?P<line_start>\d+)-(?P<line_end>\d+) "
    r"cols (?P<col_start>\d+)-(?P<col_end>\d+)$"
)


def render_patch(result: PatchResult) -> str:
    """Render a deterministic patch format with explicit metadata headers."""
    if not isinstance(result, PatchResult):
        raise PatchRuleError(
            "result must be a PatchResult",
            details={"field": "result", "actual_type": type(result).__name__},
        )
    change_count = len(result.changes)
    lines = [_PATCH_MAGIC, f"{_PATCH_COUNT_PREFIX}{change_count}"]
    for change in result.changes:
        description = change.description
        if not isinstance(description, str):
            raise PatchRuleError(
                "description must be a string",
                details={"rule_id": change.rule_id, "actual_type": type(description).__name__},
            )
        lines.append(f"{_PATCH_RULE_PREFIX}{change.rule_id}")
        lines.append(f"{_PATCH_DESCRIPTION_PREFIX}{description}")
        lines.append(f"{_PATCH_TYPE_PREFIX}{change.rule_type}")
        lines.append(
            f"{_PATCH_RANGE_PREFIX}bytes {change.start}-{change.end} "
            f"lines {change.line_start}-{change.line_end} "
            f"cols {change.col_start}-{change.col_end}"
        )
        before_lines = _split_patch_lines(change.before)
        after_lines = _split_patch_lines(change.after)
        if not before_lines and not after_lines:
            raise PatchRuleError(
                "patch hunks must include at least one change line",
                details={"rule_id": change.rule_id},
            )
        lines.append(
            f"@@ -{change.line_start},{len(before_lines)} "
            f"+{change.line_start},{len(after_lines)} @@"
        )
        lines.extend(f"-{line}" for line in before_lines)
        lines.extend(f"+{line}" for line in after_lines)
    patch_text = "\n".join(lines)
    validate_patch_text(patch_text)
    return patch_text


def validate_patch_text(patch_text: str) -> None:
    """Validate rendered patch text, raising on format violations."""
    if not isinstance(patch_text, str):
        raise PatchRuleError(
            "patch_text must be a string",
            details={"field": "patch_text", "actual_type": type(patch_text).__name__},
        )
    if not patch_text:
        raise PatchRuleError("patch_text must not be empty", details={"field": "patch_text"})
    lines = patch_text.splitlines()
    if not lines or lines[0] != _PATCH_MAGIC:
        raise PatchRuleError(
            "patch_text header is invalid",
            details={"expected": _PATCH_MAGIC, "actual": lines[0] if lines else None},
        )
    if len(lines) < 2 or not lines[1].startswith(_PATCH_COUNT_PREFIX):
        raise PatchRuleError(
            "patch_text missing change-count header",
            details={"expected_prefix": _PATCH_COUNT_PREFIX},
        )
    count_str = lines[1][len(_PATCH_COUNT_PREFIX) :].strip()
    if not count_str.isdigit():
        raise PatchRuleError(
            "change-count must be an integer",
            details={"value": count_str},
        )
    expected_changes = int(count_str)
    idx = 2
    parsed_changes = 0
    while idx < len(lines):
        if not lines[idx].startswith(_PATCH_RULE_PREFIX):
            raise PatchRuleError(
                "patch_text missing rule header",
                details={"line": idx + 1, "value": lines[idx]},
            )
        rule_id = lines[idx][len(_PATCH_RULE_PREFIX) :].strip()
        idx += 1
        if idx >= len(lines) or not lines[idx].startswith(_PATCH_DESCRIPTION_PREFIX):
            raise PatchRuleError(
                "patch_text missing description header",
                details={"line": idx + 1},
            )
        description = lines[idx][len(_PATCH_DESCRIPTION_PREFIX) :]
        idx += 1
        if idx >= len(lines) or not lines[idx].startswith(_PATCH_TYPE_PREFIX):
            raise PatchRuleError(
                "patch_text missing type header",
                details={"line": idx + 1},
            )
        rule_type = lines[idx][len(_PATCH_TYPE_PREFIX) :].strip()
        if not rule_type:
            raise PatchRuleError(
                "type header must not be empty",
                details={"rule_id": rule_id},
            )
        idx += 1
        if idx >= len(lines) or not lines[idx].startswith(_PATCH_RANGE_PREFIX):
            raise PatchRuleError(
                "patch_text missing range header",
                details={"line": idx + 1, "rule_id": rule_id},
            )
        range_line = lines[idx]
        range_match = _PATCH_RANGE.match(range_line)
        if not range_match:
            raise PatchRuleError(
                "range header is invalid",
                details={"rule_id": rule_id, "value": range_line},
            )
        line_start = int(range_match.group("line_start"))
        idx += 1
        if idx >= len(lines):
            raise PatchRuleError(
                "patch_text missing hunk header",
                details={"rule_id": rule_id},
            )
        hunk_match = _PATCH_HUNK_HEADER.match(lines[idx])
        if not hunk_match:
            raise PatchRuleError(
                "hunk header is invalid",
                details={"rule_id": rule_id, "value": lines[idx]},
            )
        hunk_start = int(hunk_match.group("start"))
        before_count = int(hunk_match.group("count"))
        after_count = int(hunk_match.group("count_new"))
        idx += 1
        if before_count == 0 and after_count == 0:
            raise PatchRuleError(
                "hunk must contain at least one change line",
                details={"rule_id": rule_id},
            )
        if hunk_start != line_start:
            raise PatchConflictError(
                "hunk header line start does not match metadata",
                details={"rule_id": rule_id, "metadata_start": line_start, "hunk_start": hunk_start},
            )
        for _ in range(before_count):
            if idx >= len(lines) or not lines[idx].startswith("-"):
                raise PatchConflictError(
                    "hunk deletions do not match header count",
                    details={"rule_id": rule_id, "description": description},
                )
            idx += 1
        for _ in range(after_count):
            if idx >= len(lines) or not lines[idx].startswith("+"):
                raise PatchConflictError(
                    "hunk insertions do not match header count",
                    details={"rule_id": rule_id, "description": description},
                )
            idx += 1
        parsed_changes += 1
    if parsed_changes != expected_changes:
        raise PatchConflictError(
            "change-count does not match parsed hunks",
            details={"expected": expected_changes, "actual": parsed_changes},
        )


def _split_patch_lines(text: str) -> list[str]:
    """Split text into stable patch lines, preserving trailing newline."""
    if not text:
        return []
    lines = text.splitlines()
    if text.endswith("\n"):
        lines.append("")
    return lines


def _serialize_rules(resolved_rules: Sequence[ResolvedRule]) -> list[dict[str, Any]]:
    """Serialize resolved rules into deterministic payloads."""
    ordered = sorted(resolved_rules, key=_resolved_rule_order_key)
    return [
        {
            "id": rule.rule_id,
            "type": rule.rule_type,
            "anchor": rule.anchor,
            "anchor_kind": rule.anchor_kind,
            "span": {"start": rule.start, "end": rule.end},
            "line_offsets": {
                "start_line": rule.line_start,
                "start_col": rule.col_start,
                "end_line": rule.line_end,
                "end_col": rule.col_end,
            },
        }
        for rule in ordered
    ]


def _serialize_anchor_resolutions(resolved_rules: Sequence[ResolvedRule]) -> list[dict[str, Any]]:
    """Serialize anchor resolution metadata for diagnostics."""
    ordered = sorted(resolved_rules, key=_resolved_rule_order_key)
    return [
        {
            "id": rule.rule_id,
            "start": rule.start,
            "end": rule.end,
            "start_line": rule.line_start,
            "end_line": rule.line_end,
        }
        for rule in ordered
    ]


def _serialize_applied_rules(resolved_rules: Sequence[ResolvedRule]) -> list[dict[str, Any]]:
    """Serialize applied rules for deterministic metadata output."""
    ordered = sorted(resolved_rules, key=_resolved_rule_order_key)
    return [
        {
            "id": rule.rule_id,
            "type": rule.rule_type,
            "rationale": rule.description,
            "span": {"start": rule.start, "end": rule.end},
            "line_offsets": {
                "start_line": rule.line_start,
                "start_col": rule.col_start,
                "end_line": rule.line_end,
                "end_col": rule.col_end,
            },
        }
        for rule in ordered
    ]


def _count_changed_lines(patch_text: str) -> int:
    """Count the number of changed lines in a unified diff."""
    count = 0
    for line in patch_text.splitlines():
        if line.startswith("+++") or line.startswith("---") or line.startswith("@@"):
            continue
        if line.startswith("+") or line.startswith("-"):
            count += 1
    return count


def _base_meta(
    source: str,
    resolved_rules: Sequence[ResolvedRule],
    *,
    syntax_valid: bool | None,
    rule_count: int,
) -> dict[str, Any]:
    """Build base metadata for deterministic responses."""
    return {
        "rule_count": rule_count,
        "changed_line_count": 0,
        "anchor_resolutions": _serialize_anchor_resolutions(resolved_rules),
        "syntax_valid": syntax_valid,
        "original_source": source,
    }


def validate_syntax(source: str) -> None:
    """Validate Python syntax for a given source string."""
    try:
        ast.parse(source)
    except SyntaxError as exc:
        raise PatchSyntaxError(
            "Syntax check failed",
            details={
                "error_type": type(exc).__name__,
                "message": exc.msg,
                "line": exc.lineno,
                "column": exc.offset,
                "offset": exc.offset,
                "lineno": exc.lineno,
                "col_offset": exc.offset,
                "msg": exc.msg,
            },
        ) from exc


def _check_syntax(
    source: str,
    error_report: Mapping[str, Any],
    rules: Sequence[ReplaceRule | InsertAfterRule | DeleteRegexRule],
    *,
    language: str | None = None,
) -> MenaceError | None:
    """Optionally validate syntax of the modified source."""
    normalized_language = language.strip().lower() if isinstance(language, str) else None
    if normalized_language is None:
        normalized_language = _detect_language(error_report, rules)
    if normalized_language != "python":
        return None
    try:
        validate_syntax(source)
    except PatchSyntaxError as exc:
        details = dict(exc.details or {})
        details.update(
            {
                "rule_ids": [rule.rule_id for rule in rules],
                "rule_count": len(rules),
            }
        )
        return PatchSyntaxError(exc.message, details=details)
    return None


def _should_validate_modified_source(
    validate_syntax: bool | None,
    error_report: Mapping[str, Any],
    rules: Sequence[ReplaceRule | InsertAfterRule | DeleteRegexRule],
) -> tuple[bool, str | None]:
    """Determine whether to validate modified source syntax."""
    if validate_syntax is False:
        return False, None
    if validate_syntax is True:
        return True, "python"
    for rule in rules:
        meta = rule.meta or {}
        if isinstance(meta, Mapping) and meta.get("validate_syntax") is True:
            return True, "python"
    language = _detect_language(error_report, rules)
    if language and language.strip().lower() == "python":
        return True, language
    return False, None


def _detect_language(
    error_report: Mapping[str, Any],
    rules: Sequence[ReplaceRule | InsertAfterRule | DeleteRegexRule],
) -> str | None:
    """Determine language from error metadata or rule metadata."""
    candidates: list[str] = []
    for key in ("language", "lang"):
        value = error_report.get(key)
        if isinstance(value, str) and value.strip():
            candidates.append(value)
    meta = error_report.get("meta")
    if isinstance(meta, Mapping):
        value = meta.get("language")
        if isinstance(value, str) and value.strip():
            candidates.append(value)
    file_path = error_report.get("file_path")
    if isinstance(file_path, str) and file_path.endswith(".py"):
        candidates.append("python")
    for rule in rules:
        if rule.meta and isinstance(rule.meta.get("language"), str):
            candidates.append(rule.meta["language"])
    for candidate in candidates:
        normalized = candidate.strip().lower()
        if normalized in {"python", "py"}:
            return "python"
    return None
