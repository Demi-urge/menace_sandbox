"""Generate deterministic unified diffs by applying explicit patch rules."""

from __future__ import annotations

import ast
import difflib
import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from menace.errors.exceptions import MenaceError, ValidationError


@dataclass
class _Line:
    text: str
    modified_by: str | None = None


def generate_patch(
    source: str,
    error_report: Mapping[str, Any],
    rules: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Apply deterministic, line-based patch rules and return a structured result."""

    try:
        _validate_inputs(source, error_report, rules)
        validated_rules = _validate_rules(rules)
        original_lines = source.splitlines()
        trailing_newline = source.endswith("\n")
        line_state = [_Line(text=line) for line in original_lines]

        applied_rule_ids: list[str] = []
        change_summary = {"inserted": 0, "deleted": 0, "replaced": 0}

        for rule in validated_rules:
            _apply_rule(line_state, rule, change_summary)
            applied_rule_ids.append(rule["id"])

        updated_source = _join_lines(line_state, trailing_newline)
        diff_text = _build_diff(source, updated_source)

        syntax_error = _check_syntax(updated_source, error_report, validated_rules)

        errors: list[dict[str, Any]] = []
        status = "success"
        if syntax_error:
            errors.append(syntax_error.to_dict())
            status = "failed"

        return {
            "status": status,
            "data": {
                "diff": diff_text,
                "source": updated_source,
            },
            "errors": errors,
            "meta": {
                "applied_rules": list(applied_rule_ids),
                "line_counts": {
                    "before": len(original_lines),
                    "after": len(line_state),
                },
                "change_summary": dict(change_summary),
            },
        }
    except MenaceError as error:
        return {
            "status": "failed",
            "data": {},
            "errors": [error.to_dict()],
            "meta": {},
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


def _validate_rules(rules: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    validated: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for index, rule in enumerate(rules):
        if not isinstance(rule, Mapping):
            raise ValidationError(
                "rule must be a mapping",
                details={"index": index, "expected": "mapping"},
            )
        rule_type = rule.get("type")
        if rule_type not in {"replace", "insert_after", "insert_before", "delete_regex"}:
            raise ValidationError(
                "Unknown rule type",
                details={
                    "index": index,
                    "rule_type": rule_type,
                    "supported_types": [
                        "delete_regex",
                        "insert_after",
                        "insert_before",
                        "replace",
                    ],
                },
            )
        rule_id = rule.get("id") or f"rule-{index + 1}"
        if not isinstance(rule_id, str):
            raise ValidationError(
                "rule id must be a string",
                details={"index": index, "field": "id"},
            )
        if rule_id in seen_ids:
            raise ValidationError(
                "rule id must be unique",
                details={"index": index, "id": rule_id},
            )
        seen_ids.add(rule_id)
        validator = _RULE_VALIDATORS[rule_type]
        validated_rule = validator(rule, index, rule_id)
        validated.append(validated_rule)
    return validated


def _validate_replace(rule: Mapping[str, Any], index: int, rule_id: str) -> dict[str, Any]:
    _ensure_only_keys(rule, {"type", "id", "line", "content", "meta"}, index, rule_id)
    line = _require_line(rule, index, rule_id)
    content = rule.get("content")
    if not isinstance(content, str):
        raise ValidationError(
            "replace content must be a string",
            details={"index": index, "id": rule_id, "field": "content"},
        )
    return {"type": "replace", "id": rule_id, "line": line, "content": content, "meta": rule.get("meta")}


def _validate_insert(rule: Mapping[str, Any], index: int, rule_id: str) -> dict[str, Any]:
    _ensure_only_keys(rule, {"type", "id", "line", "content", "meta"}, index, rule_id)
    line = _require_line(rule, index, rule_id)
    content = rule.get("content")
    if not isinstance(content, str):
        raise ValidationError(
            "insert content must be a string",
            details={"index": index, "id": rule_id, "field": "content"},
        )
    return {
        "type": rule["type"],
        "id": rule_id,
        "line": line,
        "content": content,
        "meta": rule.get("meta"),
    }


def _validate_delete_regex(rule: Mapping[str, Any], index: int, rule_id: str) -> dict[str, Any]:
    _ensure_only_keys(rule, {"type", "id", "pattern", "flags", "meta"}, index, rule_id)
    pattern = rule.get("pattern")
    if not isinstance(pattern, str):
        raise ValidationError(
            "delete_regex pattern must be a string",
            details={"index": index, "id": rule_id, "field": "pattern"},
        )
    flags = rule.get("flags", [])
    if not isinstance(flags, Sequence) or isinstance(flags, (str, bytes)):
        raise ValidationError(
            "delete_regex flags must be a sequence",
            details={"index": index, "id": rule_id, "field": "flags"},
        )
    compiled_flags = 0
    for flag in flags:
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
    return {
        "type": "delete_regex",
        "id": rule_id,
        "pattern": pattern,
        "flags": compiled_flags,
        "meta": rule.get("meta"),
    }


_RULE_VALIDATORS = {
    "replace": _validate_replace,
    "insert_after": _validate_insert,
    "insert_before": _validate_insert,
    "delete_regex": _validate_delete_regex,
}

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


def _require_line(rule: Mapping[str, Any], index: int, rule_id: str) -> int:
    line = rule.get("line")
    if not isinstance(line, int):
        raise ValidationError(
            "rule line must be an integer",
            details={"index": index, "id": rule_id, "field": "line"},
        )
    if line < 1:
        raise ValidationError(
            "rule line must be >= 1",
            details={"index": index, "id": rule_id, "line": line},
        )
    return line


def _apply_rule(
    lines: list[_Line],
    rule: Mapping[str, Any],
    change_summary: dict[str, int],
) -> None:
    rule_type = rule["type"]
    if rule_type == "replace":
        _apply_replace(lines, rule, change_summary)
    elif rule_type == "insert_after":
        _apply_insert(lines, rule, change_summary, after=True)
    elif rule_type == "insert_before":
        _apply_insert(lines, rule, change_summary, after=False)
    elif rule_type == "delete_regex":
        _apply_delete_regex(lines, rule, change_summary)
    else:
        raise ValidationError(
            "Unknown rule type",
            details={"id": rule.get("id"), "rule_type": rule_type},
        )


def _apply_replace(
    lines: list[_Line],
    rule: Mapping[str, Any],
    change_summary: dict[str, int],
) -> None:
    line_number = rule["line"]
    index = line_number - 1
    if index < 0 or index >= len(lines):
        raise ValidationError(
            "replace line is out of range",
            details={"id": rule["id"], "line": line_number, "line_count": len(lines)},
        )
    line = lines[index]
    if line.modified_by and line.modified_by != rule["id"]:
        raise ValidationError(
            "replace conflicts with previous rule",
            details={
                "id": rule["id"],
                "line": line_number,
                "conflicting_rule_id": line.modified_by,
            },
        )
    replacement_lines = [_Line(text=text, modified_by=rule["id"]) for text in _split_content(rule["content"]) or [""]]
    lines[index : index + 1] = replacement_lines
    change_summary["replaced"] += 1
    if len(replacement_lines) > 1:
        change_summary["inserted"] += len(replacement_lines) - 1


def _apply_insert(
    lines: list[_Line],
    rule: Mapping[str, Any],
    change_summary: dict[str, int],
    *,
    after: bool,
) -> None:
    line_number = rule["line"]
    index = line_number if after else line_number - 1
    if line_number < 1 or line_number > len(lines):
        raise ValidationError(
            "insert anchor is out of range",
            details={"id": rule["id"], "line": line_number, "line_count": len(lines)},
        )
    insertion = [_Line(text=text, modified_by=rule["id"]) for text in _split_content(rule["content"]) or [""]]
    lines[index:index] = insertion
    change_summary["inserted"] += len(insertion)


def _apply_delete_regex(
    lines: list[_Line],
    rule: Mapping[str, Any],
    change_summary: dict[str, int],
) -> None:
    pattern = re.compile(rule["pattern"], rule["flags"])
    conflicts: list[dict[str, Any]] = []
    indices_to_delete: list[int] = []
    for idx, line in enumerate(lines):
        if pattern.search(line.text):
            if line.modified_by and line.modified_by != rule["id"]:
                conflicts.append(
                    {
                        "line": idx + 1,
                        "conflicting_rule_id": line.modified_by,
                    }
                )
            else:
                indices_to_delete.append(idx)
    if conflicts:
        raise ValidationError(
            "delete_regex conflicts with previous rule",
            details={"id": rule["id"], "conflicts": conflicts},
        )
    for idx in reversed(indices_to_delete):
        del lines[idx]
    change_summary["deleted"] += len(indices_to_delete)


def _split_content(content: str) -> list[str]:
    return content.splitlines()


def _join_lines(lines: Iterable[_Line], trailing_newline: bool) -> str:
    text = "\n".join(line.text for line in lines)
    if trailing_newline:
        return text + "\n"
    return text


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


def _check_syntax(
    source: str,
    error_report: Mapping[str, Any],
    rules: Sequence[Mapping[str, Any]],
) -> ValidationError | None:
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
    rules: Sequence[Mapping[str, Any]],
) -> str | None:
    candidates: list[str] = []
    for key in ("language", "lang"):
        value = error_report.get(key)
        if isinstance(value, str):
            candidates.append(value)
    meta = error_report.get("meta")
    if isinstance(meta, Mapping):
        value = meta.get("language")
        if isinstance(value, str):
            candidates.append(value)
    file_path = error_report.get("file_path")
    if isinstance(file_path, str) and file_path.endswith(".py"):
        candidates.append("python")
    for rule in rules:
        meta = rule.get("meta")
        if isinstance(meta, Mapping):
            value = meta.get("language")
            if isinstance(value, str):
                candidates.append(value)
    for candidate in candidates:
        normalized = candidate.strip().lower()
        if normalized in {"python", "py"}:
            return "python"
    return None
