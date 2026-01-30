"""Generate deterministic unified diffs by applying explicit patch rules."""

from __future__ import annotations

import ast
import difflib
import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from menace.errors import MenaceError, ValidationError


@dataclass(frozen=True)
class ReplaceRule:
    rule_id: str
    anchor: str
    replacement: str
    anchor_kind: str
    meta: Mapping[str, Any] | None


@dataclass(frozen=True)
class InsertAfterRule:
    rule_id: str
    anchor: str
    content: str
    anchor_kind: str
    meta: Mapping[str, Any] | None


@dataclass(frozen=True)
class DeleteRegexRule:
    rule_id: str
    pattern: str
    flags: int
    meta: Mapping[str, Any] | None


@dataclass(frozen=True)
class ResolvedRule:
    rule_id: str
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


def generate_patch(
    source: str,
    error_report: Mapping[str, Any],
    rules: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Apply deterministic patch rules and return a structured result."""
    _validate_inputs(source, error_report, rules)
    parsed_rules = _parse_rules(rules)
    line_index = _build_line_index(source)

    resolved_rules = [
        _resolve_rule(source, rule, index, line_index) for index, rule in enumerate(parsed_rules)
    ]

    conflict_error = _detect_conflicts(resolved_rules)
    if conflict_error:
        return _failure_result([conflict_error], meta=_base_meta(resolved_rules, syntax_valid=None))

    updated_source = _apply_edits(source, resolved_rules)
    patch_text = _build_diff(source, updated_source)

    syntax_error = _check_syntax(updated_source, error_report, parsed_rules)
    syntax_valid = syntax_error is None
    errors: list[dict[str, Any]] = []
    if syntax_error:
        errors.append(syntax_error.to_dict())

    data = {
        "patch_text": patch_text,
        "modified_source": updated_source,
        "applied_rules": _serialize_rules(resolved_rules),
    }
    meta = _base_meta(resolved_rules, syntax_valid=syntax_valid)
    meta.update(
        {
            "rule_count": len(parsed_rules),
            "changed_line_count": _count_changed_lines(patch_text),
            "anchor_resolutions": _serialize_anchor_resolutions(resolved_rules),
        }
    )

    status = "ok" if not errors else "error"
    return {
        "status": status,
        "data": data,
        "errors": errors,
        "meta": meta,
    }


def _failure_result(
    errors: Sequence[dict[str, Any]],
    *,
    meta: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "status": "error",
        "data": {
            "patch_text": "",
            "modified_source": "",
            "applied_rules": [],
        },
        "errors": list(errors),
        "meta": dict(meta),
    }


def _validate_inputs(
    source: str,
    error_report: Mapping[str, Any],
    rules: Sequence[Mapping[str, Any]],
) -> None:
    if not isinstance(source, str):
        raise ValidationError(
            "source must be a string",
            details={"field": "source", "expected": "str"},
        )
    if not isinstance(error_report, Mapping):
        raise ValidationError(
            "error_report must be a mapping",
            details={"field": "error_report", "expected": "mapping"},
        )
    if not isinstance(rules, Sequence) or isinstance(rules, (str, bytes)):
        raise ValidationError(
            "rules must be a sequence",
            details={"field": "rules", "expected": "sequence"},
        )


def _parse_rules(rules: Sequence[Mapping[str, Any]]) -> list[ReplaceRule | InsertAfterRule | DeleteRegexRule]:
    parsed: list[ReplaceRule | InsertAfterRule | DeleteRegexRule] = []
    seen_ids: set[str] = set()
    for index, rule in enumerate(rules):
        if not isinstance(rule, Mapping):
            raise ValidationError(
                "rule must be a mapping",
                details={"index": index, "expected": "mapping"},
            )
        rule_type = _require_non_empty_str(rule, "type", index)
        rule_id = _require_non_empty_str(rule, "id", index)
        if rule_id in seen_ids:
            raise ValidationError(
                "rule id must be unique",
                details={"index": index, "id": rule_id},
            )
        seen_ids.add(rule_id)
        if rule_type == "replace":
            parsed.append(_parse_replace(rule, index, rule_id))
        elif rule_type == "insert_after":
            parsed.append(_parse_insert_after(rule, index, rule_id))
        elif rule_type == "delete_regex":
            parsed.append(_parse_delete_regex(rule, index, rule_id))
        else:
            raise ValidationError(
                "Unknown rule type",
                details={
                    "index": index,
                    "rule_type": rule_type,
                    "supported_types": ["delete_regex", "insert_after", "replace"],
                },
            )
    return parsed


def _parse_replace(rule: Mapping[str, Any], index: int, rule_id: str) -> ReplaceRule:
    _ensure_only_keys(rule, {"type", "id", "anchor", "anchor_kind", "replacement", "meta"}, index, rule_id)
    anchor = _require_non_empty_str(rule, "anchor", index)
    replacement = _require_non_empty_str(rule, "replacement", index)
    anchor_kind = _parse_anchor_kind(rule, index, rule_id)
    meta = _parse_meta(rule, index, rule_id)
    return ReplaceRule(rule_id=rule_id, anchor=anchor, replacement=replacement, anchor_kind=anchor_kind, meta=meta)


def _parse_insert_after(rule: Mapping[str, Any], index: int, rule_id: str) -> InsertAfterRule:
    _ensure_only_keys(rule, {"type", "id", "anchor", "anchor_kind", "content", "meta"}, index, rule_id)
    anchor = _require_non_empty_str(rule, "anchor", index)
    content = _require_non_empty_str(rule, "content", index)
    anchor_kind = _parse_anchor_kind(rule, index, rule_id)
    meta = _parse_meta(rule, index, rule_id)
    return InsertAfterRule(rule_id=rule_id, anchor=anchor, content=content, anchor_kind=anchor_kind, meta=meta)


def _parse_delete_regex(rule: Mapping[str, Any], index: int, rule_id: str) -> DeleteRegexRule:
    _ensure_only_keys(rule, {"type", "id", "pattern", "flags", "meta"}, index, rule_id)
    pattern = _require_non_empty_str(rule, "pattern", index)
    flags = rule.get("flags", [])
    if not isinstance(flags, Sequence) or isinstance(flags, (str, bytes)):
        raise ValidationError(
            "delete_regex flags must be a sequence",
            details={"index": index, "id": rule_id, "field": "flags"},
        )
    compiled_flags = 0
    for flag in flags:
        if not isinstance(flag, str) or not flag.strip():
            raise ValidationError(
                "delete_regex flag must be a non-empty string",
                details={"index": index, "id": rule_id, "flag": flag},
            )
        if flag not in {"IGNORECASE", "MULTILINE", "DOTALL"}:
            raise ValidationError(
                "delete_regex flag is invalid",
                details={"index": index, "id": rule_id, "flag": flag},
            )
        compiled_flags |= _FLAG_MAP[flag]
    try:
        re.compile(pattern, compiled_flags)
    except re.error as exc:
        raise ValidationError(
            "delete_regex pattern is invalid",
            details={"index": index, "id": rule_id, "message": str(exc)},
        ) from exc
    meta = _parse_meta(rule, index, rule_id)
    return DeleteRegexRule(rule_id=rule_id, pattern=pattern, flags=compiled_flags, meta=meta)


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
    extra = sorted(key for key in rule.keys() if key not in allowed_keys)
    if extra:
        raise ValidationError(
            "rule has unexpected fields",
            details={"index": index, "id": rule_id, "unexpected": extra},
        )


def _require_non_empty_str(rule: Mapping[str, Any], field: str, index: int) -> str:
    value = rule.get(field)
    if not isinstance(value, str):
        raise ValidationError(
            f"{field} must be a string",
            details={"index": index, "field": field},
        )
    if not value.strip():
        raise ValidationError(
            f"{field} must be a non-empty string",
            details={"index": index, "field": field},
        )
    return value


def _parse_anchor_kind(rule: Mapping[str, Any], index: int, rule_id: str) -> str:
    anchor_kind = rule.get("anchor_kind", "literal")
    if not isinstance(anchor_kind, str) or not anchor_kind.strip():
        raise ValidationError(
            "anchor_kind must be a non-empty string",
            details={"index": index, "id": rule_id, "field": "anchor_kind"},
        )
    if anchor_kind not in {"literal", "regex"}:
        raise ValidationError(
            "anchor_kind must be literal or regex",
            details={"index": index, "id": rule_id, "anchor_kind": anchor_kind},
        )
    return anchor_kind


def _parse_meta(rule: Mapping[str, Any], index: int, rule_id: str) -> Mapping[str, Any] | None:
    if "meta" not in rule:
        return None
    meta = rule.get("meta")
    if not isinstance(meta, Mapping):
        raise ValidationError(
            "meta must be a mapping",
            details={"index": index, "id": rule_id, "field": "meta"},
        )
    return meta


def _resolve_rule(
    source: str,
    rule: ReplaceRule | InsertAfterRule | DeleteRegexRule,
    index: int,
    line_index: list[int],
) -> ResolvedRule:
    if isinstance(rule, ReplaceRule):
        start, end = _resolve_anchor(source, rule.anchor, rule.anchor_kind, rule.rule_id)
        return _build_resolved_rule(
            rule.rule_id,
            "replace",
            rule.anchor,
            rule.anchor_kind,
            start,
            end,
            rule.replacement,
            index,
            line_index,
        )
    if isinstance(rule, InsertAfterRule):
        start, end = _resolve_anchor(source, rule.anchor, rule.anchor_kind, rule.rule_id)
        return _build_resolved_rule(
            rule.rule_id,
            "insert_after",
            rule.anchor,
            rule.anchor_kind,
            end,
            end,
            rule.content,
            index,
            line_index,
        )
    if isinstance(rule, DeleteRegexRule):
        start, end = _resolve_regex(source, rule.pattern, rule.flags, rule.rule_id)
        return _build_resolved_rule(
            rule.rule_id,
            "delete_regex",
            rule.pattern,
            "regex",
            start,
            end,
            "",
            index,
            line_index,
        )
    raise ValidationError(
        "Unknown rule type",
        details={"id": getattr(rule, "rule_id", None)},
    )


def _resolve_anchor(source: str, anchor: str, anchor_kind: str, rule_id: str) -> tuple[int, int]:
    if anchor_kind == "literal":
        matches = _find_literal_matches(source, anchor)
    else:
        matches = _find_regex_matches(source, anchor, 0)
    return _select_single_match(matches, rule_id, anchor)


def _resolve_regex(source: str, pattern: str, flags: int, rule_id: str) -> tuple[int, int]:
    matches = _find_regex_matches(source, pattern, flags)
    return _select_single_match(matches, rule_id, pattern)


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


def _find_regex_matches(text: str, pattern: str, flags: int) -> list[tuple[int, int]]:
    compiled = re.compile(pattern, flags)
    return [(match.start(), match.end()) for match in compiled.finditer(text)]


def _select_single_match(
    matches: Sequence[tuple[int, int]],
    rule_id: str,
    anchor: str,
) -> tuple[int, int]:
    if not matches:
        raise ValidationError(
            "anchor not found",
            details={"id": rule_id, "anchor": anchor},
        )
    if len(matches) > 1:
        raise ValidationError(
            "anchor is ambiguous",
            details={"id": rule_id, "anchor": anchor, "match_count": len(matches)},
        )
    return matches[0]


def _build_resolved_rule(
    rule_id: str,
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


def _detect_conflicts(resolved_rules: Sequence[ResolvedRule]) -> dict[str, Any] | None:
    for idx, rule in enumerate(resolved_rules):
        for other in resolved_rules[idx + 1 :]:
            if _rules_conflict(rule, other):
                error = ValidationError(
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
    if first.start == first.end and second.start == second.end:
        return first.start == second.start
    if first.start == first.end:
        return second.start <= first.start <= second.end
    if second.start == second.end:
        return first.start <= second.start <= first.end
    return not (first.end <= second.start or second.end <= first.start)


def _apply_edits(source: str, resolved_rules: Sequence[ResolvedRule]) -> str:
    ordered = sorted(resolved_rules, key=lambda rule: (rule.start, rule.index))
    offset = 0
    updated = source
    for rule in ordered:
        start = rule.start + offset
        end = rule.end + offset
        updated = updated[:start] + rule.replacement + updated[end:]
        offset += len(rule.replacement) - (rule.end - rule.start)
    return updated


def _build_diff(before: str, after: str) -> str:
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


def _lines_for_diff(text: str) -> list[str]:
    lines = text.splitlines()
    if text.endswith("\n"):
        lines.append("")
    return [f"{line}\n" for line in lines]


def _serialize_rules(resolved_rules: Sequence[ResolvedRule]) -> list[dict[str, Any]]:
    ordered = sorted(resolved_rules, key=lambda rule: rule.index)
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
    ordered = sorted(resolved_rules, key=lambda rule: rule.index)
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


def _count_changed_lines(patch_text: str) -> int:
    count = 0
    for line in patch_text.splitlines():
        if line.startswith("+++") or line.startswith("---") or line.startswith("@@"):
            continue
        if line.startswith("+") or line.startswith("-"):
            count += 1
    return count


def _base_meta(resolved_rules: Sequence[ResolvedRule], *, syntax_valid: bool | None) -> dict[str, Any]:
    return {
        "rule_count": len(resolved_rules),
        "changed_line_count": 0,
        "anchor_resolutions": _serialize_anchor_resolutions(resolved_rules),
        "syntax_valid": syntax_valid,
    }


def _check_syntax(
    source: str,
    error_report: Mapping[str, Any],
    rules: Sequence[ReplaceRule | InsertAfterRule | DeleteRegexRule],
) -> MenaceError | None:
    language = _detect_language(error_report, rules)
    if language != "python":
        return None
    try:
        ast.parse(source)
    except SyntaxError as exc:
        return ValidationError(
            "Syntax check failed",
            details={
                "error_type": "SyntaxError",
                "message": exc.msg,
                "line": exc.lineno,
                "offset": exc.offset,
            },
        )
    return None


def _detect_language(
    error_report: Mapping[str, Any],
    rules: Sequence[ReplaceRule | InsertAfterRule | DeleteRegexRule],
) -> str | None:
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
