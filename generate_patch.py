"""Generate deterministic patches by applying explicit rules in-module."""

from __future__ import annotations

import ast
import hashlib
import json
import re
from dataclasses import asdict, dataclass, is_dataclass
from typing import Any, Mapping, Sequence

from menace.errors import (
    MenaceError,
    PatchAnchorError,
    PatchConflictError,
    PatchRuleError,
    PatchSyntaxError,
)


@dataclass(frozen=True)
class ReplaceRule:
    rule_id: str
    description: str
    anchor: str
    replacement: str
    anchor_kind: str
    meta: Mapping[str, Any]
    count: int | None
    allow_zero_matches: bool
    count_specified: bool


@dataclass(frozen=True)
class InsertAfterRule:
    rule_id: str
    description: str
    anchor: str
    content: str
    anchor_kind: str
    meta: Mapping[str, Any]


@dataclass(frozen=True)
class DeleteRegexRule:
    rule_id: str
    description: str
    pattern: str
    flags: int
    meta: Mapping[str, Any]
    allow_zero_matches: bool


Rule = ReplaceRule | InsertAfterRule | DeleteRegexRule


@dataclass(frozen=True)
class ResolvedRule:
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
    rule_id: str
    description: str
    rule_type: str
    anchor: str
    anchor_kind: str
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
    order: int
    rule_id: str
    rule_type: str
    anchor_kind: str


@dataclass(frozen=True)
class PatchResult:
    content: str
    changes: list[PatchChange]
    audit_trail: list[PatchAuditEntry]
    resolved_rules: list[ResolvedRule]


_FLAG_MAP = {
    "IGNORECASE": re.IGNORECASE,
    "MULTILINE": re.MULTILINE,
    "DOTALL": re.DOTALL,
}


def generate_patch(
    source: str,
    error_report: Mapping[str, Any],
    rules: Sequence[Mapping[str, Any]],
    *,
    validate_syntax: bool | None = None,
) -> dict[str, object]:
    """Generate a deterministic patch payload by applying explicit rules."""
    _validate_inputs(source, error_report, rules, validate_syntax)
    parsed_rules = _parse_rules(rules)
    rule_summaries = _summarize_rules(parsed_rules)

    if not parsed_rules:
        return _error_payload(
            PatchRuleError("rules must not be empty", details={"field": "rules"}),
            source=source,
            rule_summaries=rule_summaries,
            notes=["empty_rules"],
        )

    try:
        result = apply_rules(source, parsed_rules)
    except (PatchAnchorError, PatchConflictError) as exc:
        return _error_payload(
            exc,
            source=source,
            rule_summaries=rule_summaries,
        )

    patch_text = render_patch(result)
    errors: list[dict[str, Any]] = []
    syntax_valid: bool | None = None
    if validate_syntax:
        original_error = _syntax_error(source)
        modified_error = _syntax_error(result.content)
        if modified_error:
            details = dict(modified_error.details or {})
            details["rule_ids"] = [rule.rule_id for rule in parsed_rules]
            details["rule_count"] = len(parsed_rules)
            details["introduced_by_edits"] = original_error is None
            error = PatchSyntaxError(modified_error.message, details=details)
            errors.append(error.to_dict())
            syntax_valid = False
        else:
            syntax_valid = True

    data = _build_data_payload(source, result, patch_text)
    meta = _build_meta(
        source=source,
        rule_summaries=rule_summaries,
        applied_rules=_serialize_applied_rules(result.resolved_rules),
        anchor_resolutions=_serialize_anchor_resolutions(result.resolved_rules),
        applied_count=len(result.changes),
        changed_line_count=_count_changed_lines(patch_text),
        syntax_valid=syntax_valid,
        notes=[],
    )
    status = "ok" if not errors else "error"
    return {
        "status": status,
        "data": data,
        "errors": errors,
        "meta": meta,
    }


def apply_rules(source: str, rules: Sequence[Rule]) -> PatchResult:
    """Apply deterministic, ordered rules to a source string."""
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
    validate_rules(rules)

    line_index = _build_line_index(source)
    resolved: list[ResolvedRule] = []
    for index, rule in enumerate(rules):
        resolved.extend(_resolve_rule(source, rule, index, line_index))

    _raise_on_conflicts(resolved, rules)
    updated = _apply_edits(source, resolved)
    changes = _build_patch_changes(source, resolved)
    audit_trail = [
        PatchAuditEntry(
            order=idx + 1,
            rule_id=rule.rule_id,
            rule_type=rule.rule_type,
            anchor_kind=rule.anchor_kind,
        )
        for idx, rule in enumerate(sorted(resolved, key=_resolved_rule_order_key))
    ]
    return PatchResult(
        content=updated,
        changes=changes,
        audit_trail=audit_trail,
        resolved_rules=list(resolved),
    )


def validate_rules(rules: Sequence[Rule]) -> None:
    """Validate parsed rule objects."""
    if not isinstance(rules, Sequence) or isinstance(rules, (str, bytes)):
        raise PatchRuleError(
            "rules must be a sequence",
            details={"field": "rules", "expected": "sequence", "actual_type": type(rules).__name__},
        )
    for index, rule in enumerate(rules):
        if not isinstance(rule, (ReplaceRule, InsertAfterRule, DeleteRegexRule)):
            raise PatchRuleError(
                "rule must be a supported Rule type",
                details=_rule_details(index, rule, expected="Rule"),
            )
        _validate_rule(rule, index)


def render_patch(result: PatchResult) -> str:
    """Render a deterministic patch format with metadata headers."""
    change_count = len(result.changes)
    lines = ["MENACE-PATCH 1", f"change-count: {change_count}"]
    for change in result.changes:
        lines.append(f"@@@ rule {change.rule_id}")
        lines.append(f"@@@ description {change.description}")
        lines.append(f"@@@ type {change.rule_type}")
        lines.append(
            f"@@@ range bytes {change.start}-{change.end} "
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
    _validate_patch_text(patch_text)
    return patch_text


def _validate_inputs(
    source: str,
    error_report: Mapping[str, Any],
    rules: Sequence[Mapping[str, Any]],
    validate_syntax: bool | None,
) -> None:
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
    if validate_syntax is not None and not isinstance(validate_syntax, bool):
        raise PatchRuleError(
            "validate_syntax must be a boolean when provided",
            details={
                "field": "validate_syntax",
                "expected": "bool",
                "actual_type": type(validate_syntax).__name__,
            },
        )
    for index, rule in enumerate(rules):
        if not isinstance(rule, Mapping):
            raise PatchRuleError(
                "rule must be a mapping",
                details=_rule_details(index, rule, expected="mapping"),
            )


def _parse_rules(rules: Sequence[Mapping[str, Any]]) -> list[Rule]:
    parsed: list[Rule] = []
    seen_ids: set[str] = set()
    for index, rule in enumerate(rules):
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
        meta=meta,
        count=count,
        allow_zero_matches=allow_zero_matches,
        count_specified=count_specified,
    )


def _parse_insert_after(rule: Mapping[str, Any], index: int, rule_id: str) -> InsertAfterRule:
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
        if flag not in _FLAG_MAP:
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


def _validate_rule(rule: Rule, index: int) -> None:
    if not isinstance(rule.rule_id, str) or not rule.rule_id.strip():
        raise PatchRuleError(
            "rule_id is required",
            details=_rule_details(index, rule, field="rule_id"),
        )
    if not isinstance(rule.description, str) or not rule.description.strip():
        raise PatchRuleError(
            "description is required",
            details=_rule_details(index, rule, rule_id=rule.rule_id, field="description"),
        )
    if not isinstance(rule.meta, Mapping):
        raise PatchRuleError(
            "meta must be a mapping",
            details=_rule_details(index, rule, rule_id=rule.rule_id, field="meta"),
        )
    if isinstance(rule, ReplaceRule):
        _ensure_non_empty_str(rule.anchor, index, rule, rule.rule_id, field="anchor")
        _ensure_non_empty_str(rule.replacement, index, rule, rule.rule_id, field="replacement")
        _validate_anchor_kind(rule.anchor_kind, index, rule, rule.rule_id)
        _validate_anchor_pattern(rule.anchor, rule.anchor_kind, index, rule, rule.rule_id)
        _validate_replace_count(rule.count, index, rule, rule.rule_id)
    elif isinstance(rule, InsertAfterRule):
        _ensure_non_empty_str(rule.anchor, index, rule, rule.rule_id, field="anchor")
        _ensure_non_empty_str(rule.content, index, rule, rule.rule_id, field="content")
        _validate_anchor_kind(rule.anchor_kind, index, rule, rule.rule_id)
        _validate_anchor_pattern(rule.anchor, rule.anchor_kind, index, rule, rule.rule_id)
    elif isinstance(rule, DeleteRegexRule):
        _ensure_non_empty_str(rule.pattern, index, rule, rule.rule_id, field="pattern")


def _ensure_only_keys(
    rule: Mapping[str, Any],
    allowed_keys: set[str],
    index: int,
    rule_id: str,
) -> None:
    extra = sorted(key for key in rule.keys() if key not in allowed_keys)
    if extra:
        raise PatchRuleError(
            "rule has unexpected fields",
            details=_rule_details(index, rule, rule_id=rule_id, unexpected=extra),
        )


def _require_non_empty_str(rule: Mapping[str, Any], field: str, index: int) -> str:
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
    if "description" not in rule:
        raise PatchRuleError(
            "description is required",
            details=_rule_details(index, rule, rule_id=rule_id, field="description"),
        )
    description = rule.get("description")
    if not isinstance(description, str) or not description.strip():
        raise PatchRuleError(
            "description must be a non-empty string",
            details=_rule_details(index, rule, rule_id=rule_id, field="description"),
        )
    return description


def _parse_anchor_kind(rule: Mapping[str, Any], index: int, rule_id: str) -> str:
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
    value = rule.get("allow_zero_matches", False)
    if not isinstance(value, bool):
        raise PatchRuleError(
            "allow_zero_matches must be a boolean",
            details=_rule_details(index, rule, rule_id=rule_id, field="allow_zero_matches"),
        )
    return value


def _parse_meta(rule: Mapping[str, Any], index: int, rule_id: str) -> Mapping[str, Any]:
    if "meta" not in rule:
        raise PatchRuleError(
            "meta is required",
            details=_rule_details(index, rule, rule_id=rule_id, field="meta"),
        )
    meta = rule.get("meta")
    if meta is None or not isinstance(meta, Mapping):
        raise PatchRuleError(
            "meta must be a mapping",
            details=_rule_details(index, rule, rule_id=rule_id, field="meta"),
        )
    return meta


def _ensure_non_empty_str(
    target: str,
    index: int,
    rule: Rule,
    rule_id: str,
    *,
    field: str,
) -> None:
    if not isinstance(target, str) or not target.strip():
        raise PatchRuleError(
            f"{field} must be a non-empty string",
            details=_rule_details(index, rule, rule_id=rule_id, field=field),
        )


def _validate_anchor_kind(anchor_kind: str, index: int, rule: Rule, rule_id: str) -> None:
    if anchor_kind not in {"literal", "regex"}:
        raise PatchRuleError(
            "anchor_kind must be literal or regex",
            details=_rule_details(index, rule, rule_id=rule_id, anchor_kind=anchor_kind),
        )


def _validate_anchor_pattern(
    anchor: str,
    anchor_kind: str,
    index: int,
    rule: Rule,
    rule_id: str,
) -> None:
    if anchor_kind != "regex":
        return
    try:
        re.compile(anchor)
    except re.error as exc:
        raise PatchRuleError(
            "anchor regex is invalid",
            details=_rule_details(index, rule, rule_id=rule_id, anchor=anchor, message=str(exc)),
        ) from exc


def _validate_replace_count(count: int | None, index: int, rule: Rule, rule_id: str) -> None:
    if count is None:
        return
    if not isinstance(count, int) or count <= 0:
        raise PatchRuleError(
            "replace count must be a positive integer",
            details=_rule_details(index, rule, rule_id=rule_id, field="count", value=count),
        )


def _resolve_rule(
    source: str,
    rule: Rule,
    index: int,
    line_index: list[int],
) -> list[ResolvedRule]:
    if isinstance(rule, ReplaceRule):
        spans = _resolve_replace(
            source,
            rule.anchor,
            rule.anchor_kind,
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
                start,
                end,
                rule.replacement,
                index,
                line_index,
            )
            for start, end in spans
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
                start,
                end,
                "",
                index,
                line_index,
            )
            for start, end in spans
        ]
    raise PatchRuleError(
        "Unknown rule type",
        details=_rule_details(index, rule, rule_id=getattr(rule, "rule_id", None)),
    )


def _resolve_replace(
    source: str,
    anchor: str,
    anchor_kind: str,
    count: int | None,
    allow_zero_matches: bool,
    rule_id: str,
    *,
    rule_index: int,
    rule_payload: ReplaceRule,
) -> list[tuple[int, int]]:
    matches = (
        _find_literal_matches_non_overlapping(source, anchor)
        if anchor_kind == "literal"
        else _find_regex_matches(source, anchor, 0)
    )
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
                    "anchor_kind": anchor_kind,
                    "match_count": 0,
                    "matches": [],
                    "allow_zero_matches": allow_zero_matches,
                },
            ),
        )
    allow_multiple_matches = bool(rule_payload.meta.get("allow_multiple_matches"))
    if len(matches) > 1 and count is None and not rule_payload.count_specified and not allow_multiple_matches:
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
                    "count": count,
                    "count_specified": rule_payload.count_specified,
                    "allow_multiple_matches": allow_multiple_matches,
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
    matches = (
        _find_literal_matches(source, anchor)
        if anchor_kind == "literal"
        else _find_regex_matches(source, anchor, 0)
    )
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


def _find_literal_matches(text: str, anchor: str) -> list[tuple[int, int]]:
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
    compiled = re.compile(pattern, flags)
    return [(match.start(), match.end()) for match in compiled.finditer(text)]


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


def _raise_on_conflicts(resolved_rules: Sequence[ResolvedRule], rules: Sequence[Rule]) -> None:
    inserts_by_position: dict[int, list[ResolvedRule]] = {}
    rule_lookup = {rule.rule_id: rule for rule in rules}
    for rule in resolved_rules:
        if rule.start == rule.end:
            inserts_by_position.setdefault(rule.start, []).append(rule)
    for position, rules_at_position in inserts_by_position.items():
        if len(rules_at_position) < 2:
            continue
        allowed = all(
            isinstance(rule_lookup.get(rule.rule_id), InsertAfterRule)
            and bool(rule_lookup[rule.rule_id].meta.get("allow_multiple_inserts"))
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
                    },
                )


def _spans_overlap(first: ResolvedRule, second: ResolvedRule) -> bool:
    first_is_insert = first.start == first.end
    second_is_insert = second.start == second.end
    if first_is_insert and second_is_insert:
        return first.start == second.start
    if first_is_insert:
        return second.start <= first.start < second.end
    if second_is_insert:
        return first.start <= second.start < first.end
    return max(first.start, second.start) < min(first.end, second.end)


def _apply_edits(source: str, resolved_rules: Sequence[ResolvedRule]) -> str:
    ordered = sorted(resolved_rules, key=_resolved_rule_apply_key)
    offset = 0
    updated = source
    for rule in ordered:
        start = rule.start + offset
        end = rule.end + offset
        updated = updated[:start] + rule.replacement + updated[end:]
        offset += len(rule.replacement) - (rule.end - rule.start)
    return updated


def _build_patch_changes(
    source: str,
    resolved_rules: Sequence[ResolvedRule],
) -> list[PatchChange]:
    ordered = sorted(resolved_rules, key=_resolved_rule_order_key)
    return [
        PatchChange(
            rule_id=rule.rule_id,
            description=rule.description,
            rule_type=rule.rule_type,
            anchor=rule.anchor,
            anchor_kind=rule.anchor_kind,
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
    return (rule.index, rule.rule_type, rule.start, rule.end)


def _resolved_rule_apply_key(rule: ResolvedRule) -> tuple[int, int, int, str]:
    return (rule.start, rule.end, rule.index, rule.rule_type)


def _line_and_column(line_index: Sequence[int], offset: int) -> tuple[int, int]:
    line_number = 1
    for idx, line_start in enumerate(line_index):
        if line_start > offset:
            line_number = idx
            break
        line_number = idx + 1
    line_start = line_index[line_number - 1] if line_number - 1 < len(line_index) else 0
    return line_number, offset - line_start


def _build_line_index(text: str) -> list[int]:
    indices = [0]
    for idx, char in enumerate(text):
        if char == "\n":
            indices.append(idx + 1)
    return indices


def _validate_patch_text(patch_text: str) -> None:
    if not isinstance(patch_text, str):
        raise PatchRuleError(
            "patch_text must be a string",
            details={"field": "patch_text", "actual_type": type(patch_text).__name__},
        )
    if not patch_text:
        raise PatchRuleError("patch_text must not be empty", details={"field": "patch_text"})
    lines = patch_text.splitlines()
    if not lines or lines[0] != "MENACE-PATCH 1":
        raise PatchRuleError(
            "patch_text header is invalid",
            details={"expected": "MENACE-PATCH 1", "actual": lines[0] if lines else None},
        )


def _split_patch_lines(text: str) -> list[str]:
    if not text:
        return []
    lines = text.splitlines()
    if text.endswith("\n"):
        lines.append("")
    return lines


def _build_data_payload(source: str, result: PatchResult, patch_text: str) -> dict[str, Any]:
    return {
        "patch_text": patch_text,
        "modified_source": result.content,
        "updated_source": result.content,
        "applied_rules": _serialize_rules(result.resolved_rules),
        "changes": [
            {
                "rule_id": change.rule_id,
                "rule_type": change.rule_type,
                "description": change.description,
                "anchor": change.anchor,
                "anchor_kind": change.anchor_kind,
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


def _serialize_rules(resolved_rules: Sequence[ResolvedRule]) -> list[dict[str, Any]]:
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


def _serialize_applied_rules(resolved_rules: Sequence[ResolvedRule]) -> list[dict[str, Any]]:
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


def _serialize_anchor_resolutions(resolved_rules: Sequence[ResolvedRule]) -> list[dict[str, Any]]:
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


def _build_meta(
    *,
    source: str,
    rule_summaries: Sequence[Mapping[str, Any]],
    applied_rules: Sequence[Mapping[str, Any]],
    anchor_resolutions: Sequence[Mapping[str, Any]],
    applied_count: int,
    changed_line_count: int,
    syntax_valid: bool | None,
    notes: Sequence[str],
) -> dict[str, Any]:
    applied_rules_list = list(applied_rules)
    return {
        "rule_summaries": list(rule_summaries),
        "rule_count": len(rule_summaries),
        "applied_count": applied_count,
        "total_changes": applied_count,
        "applied_rule_ids": [rule["id"] for rule in applied_rules_list if "id" in rule],
        "rule_application_order": [
            rule["id"] for rule in applied_rules_list if "id" in rule
        ],
        "applied_rules": applied_rules_list,
        "anchor_resolutions": list(anchor_resolutions),
        "changed_line_count": changed_line_count,
        "syntax_valid": syntax_valid,
        "validation_results": {"syntax_valid": syntax_valid},
        "input_hash": _hash_input(source, rule_summaries),
        "notes": list(notes),
    }


def _error_payload(
    error: MenaceError,
    *,
    source: str,
    rule_summaries: Sequence[Mapping[str, Any]],
    notes: Sequence[str] | None = None,
) -> dict[str, Any]:
    return {
        "status": "error",
        "data": _empty_data_payload(),
        "errors": [error.to_dict()],
        "meta": _build_meta(
            source=source,
            rule_summaries=rule_summaries,
            applied_rules=[],
            anchor_resolutions=[],
            applied_count=0,
            changed_line_count=0,
            syntax_valid=None,
            notes=notes or [],
        ),
    }


def _empty_data_payload() -> dict[str, Any]:
    return {
        "patch_text": "",
        "modified_source": "",
        "updated_source": "",
        "applied_rules": [],
        "changes": [],
        "audit_trail": [],
    }


def _summarize_rules(rules: Sequence[Rule]) -> list[dict[str, Any]]:
    return [
        {"id": rule.rule_id, "type": _rule_kind(rule), "description": rule.description}
        for rule in rules
    ]


def _rule_kind(rule: Rule) -> str:
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


def _hash_input(source: str, rule_summaries: Sequence[Mapping[str, Any]]) -> str:
    payload = {"source": source, "rules": list(rule_summaries)}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _count_changed_lines(patch_text: str) -> int:
    count = 0
    for line in patch_text.splitlines():
        if line.startswith("+++ ") or line.startswith("--- ") or line.startswith("@@"):
            continue
        if line.startswith("+") or line.startswith("-"):
            count += 1
    return count


def _syntax_error(source: str) -> PatchSyntaxError | None:
    try:
        ast.parse(source)
    except SyntaxError as exc:
        return PatchSyntaxError(
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
        )
    return None


def _rule_details(index: int, rule: Any, *, rule_id: str | None = None, **extra: Any) -> dict[str, Any]:
    details: dict[str, Any] = {
        "rule_index": index,
        "rule": _serialize_rule_payload(rule),
    }
    if rule_id is not None:
        details["rule_id"] = rule_id
    details.update(extra)
    return details


def _serialize_rule_payload(rule: Any) -> Any:
    if is_dataclass(rule):
        return asdict(rule)
    if isinstance(rule, Mapping):
        return dict(rule)
    return rule


__all__ = [
    "generate_patch",
    "apply_rules",
    "PatchAnchorError",
    "PatchConflictError",
    "PatchRuleError",
    "PatchSyntaxError",
]
