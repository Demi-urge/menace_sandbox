from __future__ import annotations

import ast
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import time
from pathlib import Path
import re

from menace.errors import PatchRuleError, PatchSyntaxError, PatchValidationError, ValidationError

from .response_schemas import normalize_patch_validation

_DIFF_HEADER = re.compile(r"^diff --git a/(?P<left>.+) b/(?P<right>.+)$")
_FILE_OLD = re.compile(r"^--- (?P<path>.+)$")
_FILE_NEW = re.compile(r"^\+\+\+ (?P<path>.+)$")

_DISALLOWED_LITERALS = {
    "GIT binary patch": "binary_patch",
    "Binary files ": "binary_patch",
    "rename from ": "rename_operation",
    "rename to ": "rename_operation",
    "copy from ": "copy_operation",
    "copy to ": "copy_operation",
}

_SUPPORTED_RULE_TYPES = {
    "required_imports",
    "signature_match",
    "forbidden_patterns",
    "mandatory_returns",
    "unchanged_code",
}


@dataclass(frozen=True)
class ValidatedRule:
    rule_type: str
    target: str
    match: str
    severity: str | None
    rule_id: str | None
    rule_index: int
    payload: dict[str, object]


@dataclass(frozen=True)
class PatchValidationLimits:
    max_lines: int = 4000
    max_bytes: int = 400_000
    max_files: int = 50
    max_hunks: int = 400
    allow_new_files: bool = False
    allow_deletes: bool = False


def _iter_disallowed_lines(lines: Iterable[str]) -> Iterable[str]:
    for line in lines:
        for literal, flag in _DISALLOWED_LITERALS.items():
            if literal in line:
                yield flag


def _sanitize_path(raw: str) -> str:
    if raw.startswith(("a/", "b/")):
        return raw[2:]
    return raw


def _validate_path(path: str) -> str | None:
    if not path:
        return "empty_path"
    if path == "/dev/null":
        return None
    if "\x00" in path:
        return "nul_byte_path"
    if Path(path).is_absolute():
        return "absolute_path"
    parts = Path(path).parts
    if any(part in {"..", "."} for part in parts):
        return "traversal_path"
    return None


def validate_patch_text(
    patch_text: str,
    *,
    limits: PatchValidationLimits | None = None,
) -> dict[str, object]:
    """Validate structured patch rules and return a normalized payload.

    Applies size and content limits while rejecting disallowed operations.
    """

    flags: list[str] = []
    context: dict[str, object] = {}

    if not isinstance(patch_text, str):
        return normalize_patch_validation(
            {"valid": False, "flags": ["patch_not_string"], "context": {}}
        )

    if not patch_text.strip():
        return normalize_patch_validation(
            {"valid": False, "flags": ["patch_empty"], "context": {}}
        )

    limits = limits or PatchValidationLimits()

    total_bytes = len(patch_text.encode("utf-8", errors="replace"))
    lines = patch_text.splitlines()
    total_lines = len(lines)

    if total_lines > limits.max_lines:
        flags.append("patch_too_large_lines")
    if total_bytes > limits.max_bytes:
        flags.append("patch_too_large_bytes")

    file_paths: set[str] = set()
    hunk_count = 0
    diff_count = 0
    pending_old = False
    pending_new = False
    current_file: str | None = None

    flags.extend(_iter_disallowed_lines(lines))

    for line in lines:
        header_match = _DIFF_HEADER.match(line)
        if header_match:
            diff_count += 1
            if current_file and (not pending_old or not pending_new):
                flags.append("missing_file_markers")
            left = _sanitize_path(header_match.group("left"))
            right = _sanitize_path(header_match.group("right"))
            for path in (left, right):
                invalid_flag = _validate_path(path)
                if invalid_flag:
                    flags.append(invalid_flag)
                if path and path != "/dev/null":
                    file_paths.add(path)
            current_file = right or left
            pending_old = False
            pending_new = False
            continue

        old_match = _FILE_OLD.match(line)
        if old_match:
            pending_old = True
            path = _sanitize_path(old_match.group("path"))
            if path == "/dev/null" and not limits.allow_new_files:
                flags.append("new_file_disallowed")
            invalid_flag = _validate_path(path)
            if invalid_flag:
                flags.append(invalid_flag)
            if path and path != "/dev/null":
                file_paths.add(path)
            continue

        new_match = _FILE_NEW.match(line)
        if new_match:
            pending_new = True
            path = _sanitize_path(new_match.group("path"))
            if path == "/dev/null" and not limits.allow_deletes:
                flags.append("delete_file_disallowed")
            invalid_flag = _validate_path(path)
            if invalid_flag:
                flags.append(invalid_flag)
            if path and path != "/dev/null":
                file_paths.add(path)
            continue

        if line.startswith("@@"):
            hunk_count += 1

    if diff_count == 0:
        flags.append("missing_diff_header")
    if current_file and (not pending_old or not pending_new):
        flags.append("missing_file_markers")

    if diff_count > limits.max_files:
        flags.append("too_many_files")
    if hunk_count > limits.max_hunks:
        flags.append("too_many_hunks")

    context.update(
        {
            "file_paths": sorted(file_paths),
            "file_count": diff_count,
            "hunk_count": hunk_count,
            "total_lines": total_lines,
            "total_bytes": total_bytes,
        }
    )

    return normalize_patch_validation(
        {
            "valid": not bool(flags),
            "flags": flags,
            "context": context,
        }
    )


def _hash_source(source: str) -> str:
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def _normalize_rules(
    rules: Sequence[Mapping[str, object]] | Mapping[str, object]
) -> tuple[list[dict[str, object]], PatchRuleError | None]:
    if isinstance(rules, Mapping):
        if "rules" in rules:
            candidate = rules.get("rules")
            if isinstance(candidate, Sequence) and not isinstance(candidate, (str, bytes, bytearray)):
                return list(candidate), None
            return (
                [],
                PatchRuleError(
                    "rules must be a list",
                    details={"field": "rules", "actual_type": type(candidate).__name__},
                ),
            )
        if all(isinstance(value, Mapping) for value in rules.values()):
            return [dict(value) for value in rules.values()], None
        return (
            [],
            PatchRuleError(
                "rules must be a list or mapping of rule definitions",
                details={"field": "rules", "actual_type": type(rules).__name__},
            ),
        )
    if isinstance(rules, Sequence) and not isinstance(rules, (str, bytes, bytearray)):
        return list(rules), None
    return (
        [],
        PatchRuleError(
            "rules must be a list or mapping of rule definitions",
            details={"field": "rules", "actual_type": type(rules).__name__},
        ),
    )


def validate_patch(
    original: str,
    patched: str,
    rules: Sequence[Mapping[str, object]] | Mapping[str, object],
    *,
    module_name: str | None = None,
    metadata: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Validate patch updates using deterministic, static AST-based rules."""

    start_time = time.perf_counter()
    errors: list[dict[str, object]] = []
    rule_errors: list[PatchRuleError] = []
    syntax_errors: list[PatchSyntaxError] = []
    original_index = _empty_index()
    patched_index = _empty_index()
    current_rule_id: str | None = None
    current_rule_index: int | None = None

    source_hashes = {
        "original_source_hash": _hash_source(original) if isinstance(original, str) else "",
        "patched_source_hash": _hash_source(patched) if isinstance(patched, str) else "",
    }

    if metadata is None and isinstance(rules, Mapping):
        rules_meta = rules.get("meta")
        if isinstance(rules_meta, Mapping):
            metadata = rules_meta

    rules_list, rules_error = _normalize_rules(rules)
    if rules_error is not None:
        rule_errors.append(rules_error)
        errors.append(rules_error.to_dict())

    if not isinstance(original, str):
        errors.append(
            PatchValidationError(
                "original code must be a string",
                details={"field": "original", "actual_type": type(original).__name__},
            ).to_dict()
        )
    if not isinstance(patched, str):
        errors.append(
            PatchValidationError(
                "patched code must be a string",
                details={"field": "patched", "actual_type": type(patched).__name__},
            ).to_dict()
        )
    if metadata is not None and not isinstance(metadata, Mapping):
        errors.append(
            PatchValidationError(
                "metadata must be a mapping",
                details={"field": "metadata", "actual_type": type(metadata).__name__},
            ).to_dict()
        )

    try:
        if isinstance(original, str) and isinstance(patched, str):
            validated_rules, validation_errors = validate_rules(rules_list)
            rule_errors.extend(validation_errors)

            original_tree, original_index, original_nodes = _build_ast_index(
                original,
                "original",
                module_name=module_name,
                errors=syntax_errors,
            )
            patched_tree, patched_index, patched_nodes = _build_ast_index(
                patched,
                "patched",
                module_name=module_name,
                errors=syntax_errors,
            )

            if syntax_errors:
                errors.extend(error.to_dict() for error in syntax_errors)

            if original_tree and patched_tree:
                for validated_rule in validated_rules:
                    current_rule_id = validated_rule.rule_id
                    current_rule_index = validated_rule.rule_index
                    rule = validated_rule.payload
                    rule_type = validated_rule.rule_type
                    rule_index = validated_rule.rule_index
                    if rule_type == "required_imports":
                        _apply_required_imports(
                            rule,
                            patched_index,
                            rule_errors,
                            rule_index=rule_index,
                        )
                    elif rule_type == "signature_match":
                        _apply_signature_matching(
                            rule,
                            original_nodes,
                            patched_nodes,
                            rule_errors,
                            rule_index=rule_index,
                        )
                    elif rule_type == "forbidden_patterns":
                        _apply_forbidden_patterns(
                            rule,
                            patched_tree,
                            rule_errors,
                            rule_index=rule_index,
                        )
                    elif rule_type == "mandatory_returns":
                        _apply_mandatory_returns(
                            rule,
                            patched_nodes,
                            rule_errors,
                            rule_index=rule_index,
                        )
                    elif rule_type == "unchanged_code":
                        _apply_unchanged_code(
                            rule,
                            original_tree,
                            patched_tree,
                            original_nodes,
                            patched_nodes,
                            rule_errors,
                            rule_index=rule_index,
                        )
                    else:
                        rule_errors.append(
                            PatchRuleError(
                                "unsupported rule type",
                                details={"rule_index": rule_index, "rule_type": rule_type},
                            )
                        )
                    current_rule_id = None
                    current_rule_index = None

            if rule_errors:
                errors.extend(error.to_dict() for error in rule_errors)
    except ValidationError as exc:
        if current_rule_id is not None or current_rule_index is not None:
            details = dict(exc.details or {})
            if current_rule_id is not None:
                details.setdefault("rule_id", current_rule_id)
            if current_rule_index is not None:
                details.setdefault("rule_index", current_rule_index)
            exc.details = details
        errors.append(exc.to_dict())
    except Exception as exc:
        errors.append(
            PatchValidationError(
                "unexpected validation error",
                details={
                    "exception_type": exc.__class__.__name__,
                    "exception_message": str(exc),
                    "rule_id": current_rule_id,
                    "rule_index": current_rule_index,
                    "module": module_name,
                    "stage": "validate_patch",
                },
            ).to_dict()
        )

    data: dict[str, object] = {
        "original_index": original_index,
        "patched_index": patched_index,
    }

    meta: dict[str, object] = {
        "module": module_name,
        "rule_count": len(rules_list),
        "error_count": len(errors),
        "rule_error_count": len(rule_errors),
        "original_function_count": len(original_index["functions"]),
        "patched_function_count": len(patched_index["functions"]),
        "original_class_count": len(original_index["classes"]),
        "patched_class_count": len(patched_index["classes"]),
        "original_import_count": len(original_index["imports"]),
        "patched_import_count": len(patched_index["imports"]),
        "original_hash": original_index["hash"],
        "patched_hash": patched_index["hash"],
        "elapsed_ms": int((time.perf_counter() - start_time) * 1000),
        **source_hashes,
    }
    if metadata is not None and isinstance(metadata, Mapping):
        meta["metadata"] = dict(metadata)

    status = "passed" if not errors else "failed"
    return {"status": status, "data": data, "errors": errors, "meta": meta}


def _build_ast_index(
    source: str,
    label: str,
    *,
    module_name: str | None,
    errors: list[PatchSyntaxError],
) -> tuple[ast.Module | None, dict[str, object], dict[str, ast.AST]]:
    filename = module_name or f"<{label}>"
    captured_error: PatchSyntaxError | None = None
    try:
        compile(source, filename, "exec")
    except SyntaxError as exc:
        captured_error = _syntax_error_payload(exc, label)
        errors.append(captured_error)
    try:
        tree = ast.parse(source, filename=filename, mode="exec")
    except SyntaxError as exc:
        if captured_error is None:
            errors.append(_syntax_error_payload(exc, label))
        return None, _empty_index(), {}
    index, nodes = _index_ast(tree)
    return tree, index, nodes


def _empty_index() -> dict[str, object]:
    return {
        "functions": {},
        "classes": {},
        "imports": [],
        "hash": "",
    }


def _syntax_error_payload(exc: SyntaxError, label: str) -> PatchSyntaxError:
    details: dict[str, object] = {
        "source": label,
        "lineno": exc.lineno,
        "offset": exc.offset,
        "col_offset": exc.offset,
        "text": exc.text,
    }
    return PatchSyntaxError(f"syntax error in {label}", details=details)


def _index_ast(tree: ast.Module) -> tuple[dict[str, object], dict[str, ast.AST]]:
    functions: dict[str, dict[str, object]] = {}
    classes: dict[str, dict[str, object]] = {}
    nodes: dict[str, ast.AST] = {}
    imports: list[dict[str, object]] = []

    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            signature = _signature_key(node)
            functions[node.name] = {
                "name": node.name,
                "lineno": node.lineno,
                "signature": signature,
            }
            nodes[f"function:{node.name}"] = node
        elif isinstance(node, ast.AsyncFunctionDef):
            signature = _signature_key(node)
            functions[node.name] = {
                "name": node.name,
                "lineno": node.lineno,
                "signature": signature,
                "async": True,
            }
            nodes[f"function:{node.name}"] = node
        elif isinstance(node, ast.ClassDef):
            classes[node.name] = {
                "name": node.name,
                "lineno": node.lineno,
                "bases": [_node_dump(base) for base in node.bases],
                "methods": [child.name for child in node.body if isinstance(child, ast.FunctionDef)],
            }
            nodes[f"class:{node.name}"] = node

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(
                    {
                        "kind": "import",
                        "module": alias.name,
                        "name": None,
                        "asname": alias.asname,
                        "level": 0,
                    }
                )
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                imports.append(
                    {
                        "kind": "from",
                        "module": node.module or "",
                        "name": alias.name,
                        "asname": alias.asname,
                        "level": node.level or 0,
                    }
                )

    imports.sort(key=lambda item: (item["kind"], item["module"], item["name"] or "", item["asname"] or "", item["level"]))
    normalized = _node_dump(tree)
    index = {
        "functions": functions,
        "classes": classes,
        "imports": imports,
        "hash": hashlib.sha256(normalized.encode("utf-8")).hexdigest(),
    }
    return index, nodes


def _node_dump(node: ast.AST) -> str:
    return ast.dump(node, include_attributes=False)


def _signature_key(node: ast.FunctionDef | ast.AsyncFunctionDef) -> dict[str, object]:
    args = node.args
    return {
        "posonlyargs": [arg.arg for arg in args.posonlyargs],
        "args": [arg.arg for arg in args.args],
        "vararg": args.vararg.arg if args.vararg else None,
        "kwonlyargs": [arg.arg for arg in args.kwonlyargs],
        "kwarg": args.kwarg.arg if args.kwarg else None,
        "defaults": [_node_dump(value) for value in args.defaults],
        "kw_defaults": [_node_dump(value) if value is not None else None for value in args.kw_defaults],
    }


def validate_rules(
    rules: list[dict[str, object]],
) -> tuple[list[ValidatedRule], list[ValidationError]]:
    """Validate patch rules against a minimal schema."""
    errors: list[PatchRuleError] = []
    validated: list[ValidatedRule] = []

    if not isinstance(rules, list):
        errors.append(
            PatchRuleError(
                "rules must be provided as a list",
                details={"field": "rules", "expected": "list", "actual_type": type(rules).__name__},
            )
        )
        return [], errors

    for rule_index, rule in enumerate(rules):
        if not isinstance(rule, dict):
            errors.append(
                PatchRuleError(
                    "rule must be a dict",
                    details={
                        "rule_index": rule_index,
                        "actual_type": type(rule).__name__,
                    },
                )
            )
            continue

        raw_rule_id = rule.get("rule_id")
        if raw_rule_id is None and "id" in rule:
            raw_rule_id = rule.get("id")

        rule_id_context = raw_rule_id if raw_rule_id is not None else None
        if raw_rule_id is not None and not isinstance(raw_rule_id, str):
            errors.append(
                PatchRuleError(
                    "rule_id must be a string",
                    details=_rule_error_context(rule_index, rule_id_context, field="rule_id"),
                )
            )
            rule_id = None
        else:
            rule_id = raw_rule_id

        rule_type = rule.get("type")
        if not isinstance(rule_type, str):
            errors.append(
                PatchRuleError(
                    "rule type must be a string",
                    details=_rule_error_context(
                        rule_index,
                        rule_id_context,
                        field="type",
                        actual_type=type(rule_type).__name__,
                    ),
                )
            )
        elif rule_type not in _SUPPORTED_RULE_TYPES:
            errors.append(
                PatchRuleError(
                    "unsupported rule type",
                    details=_rule_error_context(rule_index, rule_id_context, rule_type=rule_type),
                )
            )

        target = rule.get("target")
        if not isinstance(target, str):
            errors.append(
                PatchRuleError(
                    "rule target must be a string",
                    details=_rule_error_context(
                        rule_index,
                        rule_id_context,
                        field="target",
                        actual_type=type(target).__name__,
                    ),
                )
            )

        match = rule.get("match")
        if not isinstance(match, str):
            errors.append(
                PatchRuleError(
                    "rule match must be a string",
                    details=_rule_error_context(
                        rule_index,
                        rule_id_context,
                        field="match",
                        actual_type=type(match).__name__,
                    ),
                )
            )

        severity = rule.get("severity")
        if severity is not None and not isinstance(severity, str):
            errors.append(
                PatchRuleError(
                    "severity must be a string",
                    details=_rule_error_context(
                        rule_index,
                        rule_id_context,
                        field="severity",
                        actual_type=type(severity).__name__,
                    ),
                )
            )

        if any(
            isinstance(item, PatchRuleError)
            and item.details.get("rule_index") == rule_index
            for item in errors
        ):
            continue

        validated.append(
            ValidatedRule(
                rule_type=rule_type,
                target=target,
                match=match,
                severity=severity,
                rule_id=rule_id,
                rule_index=rule_index,
                payload=rule,
            )
        )

    return validated, errors


def _rule_error_context(rule_index: int, rule_id: object, **extra: object) -> dict[str, object]:
    details: dict[str, object] = {"rule_index": rule_index}
    if rule_id is not None:
        details["rule_id"] = rule_id
    details.update(extra)
    return details


def _apply_required_imports(
    rule: dict[str, object],
    patched_index: dict[str, object],
    rule_errors: list[PatchRuleError],
    *,
    rule_index: int,
) -> None:
    required = rule.get("imports")
    if not isinstance(required, list):
        rule_errors.append(
            PatchRuleError(
                "required_imports must provide an imports list",
                details={"rule_index": rule_index},
            )
        )
        return
    imports = patched_index.get("imports", [])
    missing: list[dict[str, object]] = []
    for item in required:
        if not isinstance(item, dict):
            missing.append({"error": "invalid_import_spec"})
            continue
        module = item.get("module")
        name = item.get("name")
        level = item.get("level", 0)
        kind = item.get("kind")
        matched = False
        for entry in imports:
            if module is not None and entry.get("module") != module:
                continue
            if name is not None and entry.get("name") != name:
                continue
            if kind is not None and entry.get("kind") != kind:
                continue
            if level is not None and entry.get("level") != level:
                continue
            matched = True
            break
        if not matched:
            missing.append({"module": module, "name": name, "kind": kind, "level": level})
    if missing:
        rule_errors.append(
            PatchRuleError(
                "required imports missing",
                details={"rule_index": rule_index, "missing": missing},
            )
        )


def _apply_signature_matching(
    rule: dict[str, object],
    original_nodes: dict[str, ast.AST],
    patched_nodes: dict[str, ast.AST],
    rule_errors: list[PatchRuleError],
    *,
    rule_index: int,
) -> None:
    functions = rule.get("functions") or rule.get("function")
    if isinstance(functions, str):
        functions = [functions]
    if not isinstance(functions, list) or not functions:
        rule_errors.append(
            PatchRuleError(
                "signature_match requires function names",
                details={"rule_index": rule_index},
            )
        )
        return
    mismatches: list[dict[str, object]] = []
    for name in functions:
        if not isinstance(name, str):
            mismatches.append({"function": name, "error": "invalid_name"})
            continue
        original = original_nodes.get(f"function:{name}")
        patched = patched_nodes.get(f"function:{name}")
        if original is None or patched is None:
            mismatches.append(
                {"function": name, "error": "missing_function", "original": bool(original), "patched": bool(patched)}
            )
            continue
        original_sig = _signature_key(original)  # type: ignore[arg-type]
        patched_sig = _signature_key(patched)  # type: ignore[arg-type]
        if original_sig != patched_sig:
            mismatches.append(
                {"function": name, "original": original_sig, "patched": patched_sig}
            )
    if mismatches:
        rule_errors.append(
            PatchRuleError(
                "function signature mismatch",
                details={"rule_index": rule_index, "mismatches": mismatches},
            )
        )


class _ForbiddenPatternVisitor(ast.NodeVisitor):
    def __init__(
        self,
        *,
        node_types: set[str],
        names: set[str],
        strings: set[str],
        attributes: set[str],
    ) -> None:
        self.node_types = node_types
        self.names = names
        self.strings = strings
        self.attributes = attributes
        self.matches: list[dict[str, object]] = []

    def generic_visit(self, node: ast.AST) -> None:
        node_type = type(node).__name__
        if node_type in self.node_types:
            self.matches.append({"node_type": node_type, "lineno": getattr(node, "lineno", None)})
        super().generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if node.id in self.names:
            self.matches.append({"name": node.id, "lineno": node.lineno})
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr in self.attributes:
            self.matches.append({"attribute": node.attr, "lineno": node.lineno})
        self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant) -> None:
        if isinstance(node.value, str) and node.value in self.strings:
            self.matches.append({"string": node.value, "lineno": node.lineno})
        self.generic_visit(node)


def _apply_forbidden_patterns(
    rule: dict[str, object],
    tree: ast.Module,
    rule_errors: list[PatchRuleError],
    *,
    rule_index: int,
) -> None:
    node_types = rule.get("node_types") or rule.get("types") or []
    names = rule.get("names") or []
    strings = rule.get("strings") or []
    attributes = rule.get("attributes") or []
    if not isinstance(node_types, list) or not isinstance(names, list) or not isinstance(strings, list) or not isinstance(attributes, list):
        rule_errors.append(
            PatchRuleError(
                "forbidden_patterns requires list inputs",
                details={"rule_index": rule_index},
            )
        )
        return
    visitor = _ForbiddenPatternVisitor(
        node_types=set(str(item) for item in node_types),
        names=set(str(item) for item in names),
        strings=set(str(item) for item in strings),
        attributes=set(str(item) for item in attributes),
    )
    visitor.visit(tree)
    if visitor.matches:
        rule_errors.append(
            PatchRuleError(
                "forbidden patterns detected",
                details={"rule_index": rule_index, "matches": visitor.matches},
            )
        )


def _apply_mandatory_returns(
    rule: dict[str, object],
    patched_nodes: dict[str, ast.AST],
    rule_errors: list[PatchRuleError],
    *,
    rule_index: int,
) -> None:
    functions = rule.get("functions") or rule.get("function")
    if isinstance(functions, str):
        functions = [functions]
    require_value = rule.get("require_value", True)
    if not isinstance(functions, list) or not functions:
        rule_errors.append(
            PatchRuleError(
                "mandatory_returns requires function names",
                details={"rule_index": rule_index},
            )
        )
        return
    failures: list[dict[str, object]] = []
    for name in functions:
        node = patched_nodes.get(f"function:{name}")
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            failures.append({"function": name, "error": "missing_function"})
            continue
        ok, details = _all_paths_return(node.body, require_value=require_value)
        if not ok:
            failures.append({"function": name, "details": details})
    if failures:
        rule_errors.append(
            PatchRuleError(
                "mandatory return requirement failed",
                details={"rule_index": rule_index, "failures": failures},
            )
        )


def _all_paths_return(
    statements: list[ast.stmt],
    *,
    require_value: bool,
) -> tuple[bool, dict[str, object]]:
    details: dict[str, object] = {"missing_value": False, "fallthrough": False}
    for statement in statements:
        if isinstance(statement, ast.Return):
            if require_value and statement.value is None:
                details["missing_value"] = True
                return False, details
            return True, details
        if isinstance(statement, ast.If):
            body_ok, body_details = _all_paths_return(statement.body, require_value=require_value)
            orelse_ok, orelse_details = _all_paths_return(statement.orelse, require_value=require_value)
            if body_ok and orelse_ok:
                return True, details
            details["missing_value"] = details["missing_value"] or body_details.get("missing_value") or orelse_details.get("missing_value")
            continue
        if isinstance(statement, ast.Try):
            body_ok, body_details = _all_paths_return(statement.body, require_value=require_value)
            handler_ok = True
            handler_details: list[dict[str, object]] = []
            for handler in statement.handlers:
                handler_return, handler_detail = _all_paths_return(handler.body, require_value=require_value)
                handler_ok = handler_ok and handler_return
                handler_details.append(handler_detail)
            orelse_ok, orelse_details = _all_paths_return(statement.orelse, require_value=require_value)
            if body_ok and handler_ok and orelse_ok:
                return True, details
            details["missing_value"] = (
                details["missing_value"]
                or body_details.get("missing_value")
                or orelse_details.get("missing_value")
                or any(detail.get("missing_value") for detail in handler_details)
            )
            continue
        if isinstance(statement, (ast.For, ast.AsyncFor, ast.While, ast.With, ast.AsyncWith)):
            body_ok, body_details = _all_paths_return(statement.body, require_value=require_value)
            if body_ok:
                return False, body_details
            continue
    details["fallthrough"] = True
    return False, details


def _apply_unchanged_code(
    rule: dict[str, object],
    original_tree: ast.Module,
    patched_tree: ast.Module,
    original_nodes: dict[str, ast.AST],
    patched_nodes: dict[str, ast.AST],
    rule_errors: list[PatchRuleError],
    *,
    rule_index: int,
) -> None:
    scope = rule.get("scope", "module")
    if scope == "module":
        original_hash = hashlib.sha256(_node_dump(original_tree).encode("utf-8")).hexdigest()
        patched_hash = hashlib.sha256(_node_dump(patched_tree).encode("utf-8")).hexdigest()
        if original_hash == patched_hash:
            rule_errors.append(
                PatchRuleError(
                    "patched code is unchanged",
                    details={"rule_index": rule_index, "scope": scope},
                )
            )
        return
    if scope == "functions":
        functions = rule.get("functions")
        if isinstance(functions, str):
            functions = [functions]
        if not isinstance(functions, list) or not functions:
            rule_errors.append(
                PatchRuleError(
                    "unchanged_code requires function names",
                    details={"rule_index": rule_index},
                )
            )
            return
        unchanged: list[str] = []
        for name in functions:
            original = original_nodes.get(f"function:{name}")
            patched = patched_nodes.get(f"function:{name}")
            if not original or not patched:
                continue
            if _node_dump(original) == _node_dump(patched):
                unchanged.append(name)
        if unchanged:
            rule_errors.append(
                PatchRuleError(
                    "patched functions unchanged",
                    details={"rule_index": rule_index, "functions": unchanged},
                )
            )
        return
    rule_errors.append(
        PatchRuleError(
            "unsupported unchanged_code scope",
            details={"rule_index": rule_index, "scope": scope},
        )
    )


__all__ = [
    "PatchValidationLimits",
    "ValidatedRule",
    "validate_patch_text",
    "validate_patch",
    "validate_rules",
]
