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
from .errors import MenaceRuleSchemaError, MenaceValidationError

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
    "syntax",
    "syntax_compile",
    "required_imports",
    "signature_match",
    "forbidden_patterns",
    "static_contracts",
    "mandatory_returns",
}


@dataclass(frozen=True)
class ValidatedRule:
    rule_type: str
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
    schema_errors: list[ValidationError] = []
    syntax_errors: list[PatchSyntaxError] = []
    original_index = _empty_index()
    patched_index = _empty_index()
    rule_summaries: list[dict[str, object]] = []
    current_rule_id: str | None = None
    current_rule_index: int | None = None
    source_hashes = {"original_source_hash": "", "patched_source_hash": ""}
    rules_list: list[Mapping[str, object]] = []
    validated_rules: list[ValidatedRule] = []

    try:
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
            schema_errors.append(rules_error)

        if not schema_errors:
            validated_rules, schema_errors = validate_rules(rules_list)

        if schema_errors:
            errors.extend(error.to_dict() for error in schema_errors)

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

        if isinstance(original, str) and isinstance(patched, str) and not schema_errors:
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
            original_signatures = _extract_signatures(original_tree) if original_tree else _empty_signatures()
            patched_signatures = _extract_signatures(patched_tree) if patched_tree else _empty_signatures()

            if syntax_errors:
                errors.extend(error.to_dict() for error in syntax_errors)

            if original_tree or patched_tree:
                for validated_rule in validated_rules:
                    current_rule_id = validated_rule.rule_id
                    current_rule_index = validated_rule.rule_index
                    rule = validated_rule.payload
                    rule_type = validated_rule.rule_type
                    rule_index = validated_rule.rule_index
                    if rule_type == "syntax_compile":
                        rule_failures, summary = _dispatch_rule(
                            rule_type,
                            rule,
                            rule_index=rule_index,
                            original_tree=original_tree or ast.parse("", mode="exec"),
                            patched_tree=patched_tree or ast.parse("", mode="exec"),
                            original_nodes=original_nodes,
                            patched_nodes=patched_nodes,
                            original_signatures=original_signatures,
                            patched_signatures=patched_signatures,
                            original_index=original_index,
                            patched_index=patched_index,
                            syntax_errors=syntax_errors,
                        )
                    elif original_tree and patched_tree:
                        rule_failures, summary = _dispatch_rule(
                            rule_type,
                            rule,
                            rule_index=rule_index,
                            original_tree=original_tree,
                            patched_tree=patched_tree,
                            original_nodes=original_nodes,
                            patched_nodes=patched_nodes,
                            original_signatures=original_signatures,
                            patched_signatures=patched_signatures,
                            original_index=original_index,
                            patched_index=patched_index,
                            syntax_errors=syntax_errors,
                        )
                    else:
                        rule_failures = [
                            PatchRuleError(
                                "syntax errors prevent rule evaluation",
                                details={"rule_index": rule_index, "code": "syntax_blocking"},
                            )
                        ]
                        summary = {
                            "rule_index": rule_index,
                            "rule_id": rule.get("rule_id") or rule.get("id"),
                            "type": rule_type,
                            "status": "failed",
                            "data": {"reason": "syntax_errors"},
                            "error_count": len(rule_failures),
                        }
                    rule_errors.extend(rule_failures)
                    rule_summaries.append(summary)
                    current_rule_id = None
                    current_rule_index = None

            if rule_errors:
                errors.extend(error.to_dict() for error in rule_errors)
    except Exception as exc:
        errors.append(
            MenaceValidationError(
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
        "rule_summaries": rule_summaries,
    }

    meta: dict[str, object] = {
        "module": module_name,
        "rule_count": len(rules_list),
        "error_count": len(errors),
        "rule_error_count": len(rule_errors) + len(schema_errors),
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
    annotations: dict[str, str] = {}
    returns = _node_dump(node.returns) if node.returns else None
    for arg in [*args.posonlyargs, *args.args, *args.kwonlyargs]:
        if arg.annotation is not None:
            annotations[arg.arg] = _node_dump(arg.annotation)
    if args.vararg and args.vararg.annotation is not None:
        annotations[f"*{args.vararg.arg}"] = _node_dump(args.vararg.annotation)
    if args.kwarg and args.kwarg.annotation is not None:
        annotations[f"**{args.kwarg.arg}"] = _node_dump(args.kwarg.annotation)
    return {
        "posonlyargs": [arg.arg for arg in args.posonlyargs],
        "args": [arg.arg for arg in args.args],
        "vararg": args.vararg.arg if args.vararg else None,
        "kwonlyargs": [arg.arg for arg in args.kwonlyargs],
        "kwarg": args.kwarg.arg if args.kwarg else None,
        "defaults": [_node_dump(value) for value in args.defaults],
        "kw_defaults": [_node_dump(value) if value is not None else None for value in args.kw_defaults],
        "annotations": annotations,
        "returns": returns,
    }


def _extract_signatures(tree: ast.Module) -> dict[str, dict[str, dict[str, object]]]:
    functions: dict[str, dict[str, object]] = {}
    classes: dict[str, dict[str, object]] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions[node.name] = _signature_key(node)
        elif isinstance(node, ast.ClassDef):
            classes[node.name] = _class_signature(node)
    return {"functions": functions, "classes": classes}


def _empty_signatures() -> dict[str, dict[str, dict[str, object]]]:
    return {"functions": {}, "classes": {}}


def _class_signature(node: ast.ClassDef) -> dict[str, object]:
    init_node: ast.FunctionDef | ast.AsyncFunctionDef | None = None
    for child in node.body:
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and child.name == "__init__":
            init_node = child
            break
    if init_node is None:
        signature: dict[str, object] = {
            "posonlyargs": [],
            "args": [],
            "vararg": None,
            "kwonlyargs": [],
            "kwarg": None,
            "defaults": [],
            "kw_defaults": [],
            "annotations": {},
            "returns": None,
        }
    else:
        signature = _signature_key(init_node)
    return {"name": node.name, "signature": signature}


def _signature_payload(signature: dict[str, object], *, include_annotations: bool) -> dict[str, object]:
    payload = {
        "posonlyargs": signature.get("posonlyargs", []),
        "args": signature.get("args", []),
        "vararg": signature.get("vararg"),
        "kwonlyargs": signature.get("kwonlyargs", []),
        "kwarg": signature.get("kwarg"),
        "defaults": signature.get("defaults", []),
        "kw_defaults": signature.get("kw_defaults", []),
    }
    if include_annotations:
        payload["annotations"] = signature.get("annotations", {})
        payload["returns"] = signature.get("returns")
    return payload


def validate_rules(
    rules: list[dict[str, object]],
) -> tuple[list[ValidatedRule], list[ValidationError]]:
    """Validate patch rules against the schema."""
    return _validate_rule_schema(rules)


def _schema_error(
    message: str,
    *,
    rule_index: int,
    rule_id: object | None,
    **details: object,
) -> MenaceRuleSchemaError:
    payload = {"rule_index": rule_index}
    if rule_id is not None:
        payload["rule_id"] = rule_id
    payload.update(details)
    return MenaceRuleSchemaError(message, details=payload)


def _normalize_str_list(
    value: object,
    *,
    field: str,
    rule_index: int,
    rule_id: object | None,
    errors: list[MenaceRuleSchemaError],
) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        invalid = [item for item in value if not isinstance(item, str)]
        if invalid:
            errors.append(
                _schema_error(
                    "rule params must contain string items",
                    rule_index=rule_index,
                    rule_id=rule_id,
                    field=field,
                    invalid_value=invalid[0],
                )
            )
            return []
        return value
    errors.append(
        _schema_error(
            "rule params must be a list",
            rule_index=rule_index,
            rule_id=rule_id,
            field=field,
            actual_type=type(value).__name__,
        )
    )
    return []


def _validate_syntax_compile_params(
    params: Mapping[str, object],
    *,
    rule_index: int,
    rule_id: object | None,
) -> tuple[dict[str, object], list[MenaceRuleSchemaError]]:
    errors: list[MenaceRuleSchemaError] = []
    sources = params.get("sources") if "sources" in params else params.get("source")
    if sources is not None:
        normalized = _normalize_str_list(
            sources,
            field="sources",
            rule_index=rule_index,
            rule_id=rule_id,
            errors=errors,
        )
        if normalized:
            params = {**params, "sources": normalized}
    _validate_require_changes_param(
        params,
        rule_index=rule_index,
        rule_id=rule_id,
        errors=errors,
    )
    return dict(params), errors


def _validate_required_imports_params(
    params: Mapping[str, object],
    *,
    rule_index: int,
    rule_id: object | None,
) -> tuple[dict[str, object], list[MenaceRuleSchemaError]]:
    errors: list[MenaceRuleSchemaError] = []
    imports = params.get("imports")
    if not isinstance(imports, list):
        errors.append(
            _schema_error(
                "required_imports requires an imports list",
                rule_index=rule_index,
                rule_id=rule_id,
                field="imports",
                actual_type=type(imports).__name__,
            )
        )
        return dict(params), errors
    for index, item in enumerate(imports):
        if not isinstance(item, dict):
            errors.append(
                _schema_error(
                    "import spec must be a dict",
                    rule_index=rule_index,
                    rule_id=rule_id,
                    field="imports",
                    import_index=index,
                    actual_type=type(item).__name__,
                )
            )
            continue
        for key in ("module", "name", "kind", "module_pattern", "name_pattern"):
            value = item.get(key)
            if value is not None and not isinstance(value, str):
                errors.append(
                    _schema_error(
                        "import spec fields must be strings",
                        rule_index=rule_index,
                        rule_id=rule_id,
                        field=f"imports.{key}",
                        import_index=index,
                        actual_type=type(value).__name__,
                    )
                )
        level = item.get("level")
        if level is not None and not isinstance(level, int):
            errors.append(
                _schema_error(
                    "import spec level must be an integer",
                    rule_index=rule_index,
                    rule_id=rule_id,
                    field="imports.level",
                    import_index=index,
                    actual_type=type(level).__name__,
                )
            )
    _validate_require_changes_param(
        params,
        rule_index=rule_index,
        rule_id=rule_id,
        errors=errors,
    )
    return dict(params), errors


def _validate_signature_match_params(
    params: Mapping[str, object],
    *,
    rule_index: int,
    rule_id: object | None,
) -> tuple[dict[str, object], list[MenaceRuleSchemaError]]:
    errors: list[MenaceRuleSchemaError] = []
    functions = _normalize_str_list(
        params.get("functions") if "functions" in params else params.get("function"),
        field="functions",
        rule_index=rule_index,
        rule_id=rule_id,
        errors=errors,
    )
    classes = _normalize_str_list(
        params.get("classes") if "classes" in params else params.get("class"),
        field="classes",
        rule_index=rule_index,
        rule_id=rule_id,
        errors=errors,
    )
    if not functions and not classes:
        errors.append(
            _schema_error(
                "signature_match requires functions or classes",
                rule_index=rule_index,
                rule_id=rule_id,
                field="functions",
            )
        )
    require_annotations = params.get("require_annotations")
    if require_annotations is not None and not isinstance(require_annotations, bool):
        errors.append(
            _schema_error(
                "require_annotations must be a boolean",
                rule_index=rule_index,
                rule_id=rule_id,
                field="require_annotations",
                actual_type=type(require_annotations).__name__,
            )
        )
    normalized = dict(params)
    if functions:
        normalized["functions"] = functions
    if classes:
        normalized["classes"] = classes
    _validate_require_changes_param(
        normalized,
        rule_index=rule_index,
        rule_id=rule_id,
        errors=errors,
    )
    return normalized, errors


def _validate_forbidden_patterns_params(
    params: Mapping[str, object],
    *,
    rule_index: int,
    rule_id: object | None,
) -> tuple[dict[str, object], list[MenaceRuleSchemaError]]:
    errors: list[MenaceRuleSchemaError] = []
    normalized: dict[str, object] = dict(params)
    list_fields = {
        "node_types": params.get("node_types") if "node_types" in params else params.get("types"),
        "names": params.get("names"),
        "strings": params.get("strings"),
        "call_names": params.get("call_names"),
        "attributes": params.get("attributes"),
    }
    for field, value in list_fields.items():
        normalized[field] = _normalize_str_list(
            value,
            field=field,
            rule_index=rule_index,
            rule_id=rule_id,
            errors=errors,
        )
    pattern_fields = {
        "name_patterns": params.get("name_patterns"),
        "string_patterns": params.get("string_patterns"),
        "call_patterns": params.get("call_patterns"),
        "attribute_patterns": params.get("attribute_patterns"),
    }
    for field, value in pattern_fields.items():
        normalized[field] = _normalize_str_list(
            value,
            field=field,
            rule_index=rule_index,
            rule_id=rule_id,
            errors=errors,
        )
    _validate_require_changes_param(
        normalized,
        rule_index=rule_index,
        rule_id=rule_id,
        errors=errors,
    )
    return normalized, errors


def _validate_static_contracts_params(
    params: Mapping[str, object],
    *,
    rule_index: int,
    rule_id: object | None,
) -> tuple[dict[str, object], list[MenaceRuleSchemaError]]:
    errors: list[MenaceRuleSchemaError] = []
    functions = _normalize_str_list(
        params.get("functions") if "functions" in params else params.get("function"),
        field="functions",
        rule_index=rule_index,
        rule_id=rule_id,
        errors=errors,
    )
    if not functions:
        errors.append(
            _schema_error(
                "static_contracts requires functions",
                rule_index=rule_index,
                rule_id=rule_id,
                field="functions",
            )
        )
    require_docstring = params.get("require_docstring")
    if require_docstring is not None and not isinstance(require_docstring, bool):
        errors.append(
            _schema_error(
                "require_docstring must be a boolean",
                rule_index=rule_index,
                rule_id=rule_id,
                field="require_docstring",
                actual_type=type(require_docstring).__name__,
            )
        )
    require_annotations = params.get("require_annotations")
    if require_annotations is not None and not isinstance(require_annotations, bool):
        errors.append(
            _schema_error(
                "require_annotations must be a boolean",
                rule_index=rule_index,
                rule_id=rule_id,
                field="require_annotations",
                actual_type=type(require_annotations).__name__,
            )
        )
    must_raise = _normalize_str_list(
        params.get("must_raise"),
        field="must_raise",
        rule_index=rule_index,
        rule_id=rule_id,
        errors=errors,
    )
    normalized = dict(params)
    if functions:
        normalized["functions"] = functions
    if must_raise:
        normalized["must_raise"] = must_raise
    _validate_require_changes_param(
        normalized,
        rule_index=rule_index,
        rule_id=rule_id,
        errors=errors,
    )
    return normalized, errors


def _validate_mandatory_returns_params(
    params: Mapping[str, object],
    *,
    rule_index: int,
    rule_id: object | None,
) -> tuple[dict[str, object], list[MenaceRuleSchemaError]]:
    errors: list[MenaceRuleSchemaError] = []
    functions = _normalize_str_list(
        params.get("functions") if "functions" in params else params.get("function"),
        field="functions",
        rule_index=rule_index,
        rule_id=rule_id,
        errors=errors,
    )
    if not functions:
        errors.append(
            _schema_error(
                "mandatory_returns requires functions",
                rule_index=rule_index,
                rule_id=rule_id,
                field="functions",
            )
        )
    require_terminal = params.get("require_terminal")
    if require_terminal is not None and not isinstance(require_terminal, bool):
        errors.append(
            _schema_error(
                "require_terminal must be a boolean",
                rule_index=rule_index,
                rule_id=rule_id,
                field="require_terminal",
                actual_type=type(require_terminal).__name__,
            )
        )
    require_any = params.get("require_any")
    if require_any is not None and not isinstance(require_any, bool):
        errors.append(
            _schema_error(
                "require_any must be a boolean",
                rule_index=rule_index,
                rule_id=rule_id,
                field="require_any",
                actual_type=type(require_any).__name__,
            )
        )
    normalized = dict(params)
    if functions:
        normalized["functions"] = functions
    _validate_require_changes_param(
        normalized,
        rule_index=rule_index,
        rule_id=rule_id,
        errors=errors,
    )
    return normalized, errors


def _validate_require_changes_param(
    params: Mapping[str, object],
    *,
    rule_index: int,
    rule_id: object | None,
    errors: list[MenaceRuleSchemaError],
) -> None:
    if "require_changes" not in params:
        return
    require_changes = params.get("require_changes")
    if not isinstance(require_changes, bool):
        errors.append(
            _schema_error(
                "require_changes must be a boolean",
                rule_index=rule_index,
                rule_id=rule_id,
                field="require_changes",
                actual_type=type(require_changes).__name__,
            )
        )


def _validate_rule_schema(
    rules: list[dict[str, object]],
) -> tuple[list[ValidatedRule], list[ValidationError]]:
    errors: list[MenaceRuleSchemaError] = []
    validated: list[ValidatedRule] = []

    if not isinstance(rules, list):
        errors.append(
            MenaceRuleSchemaError(
                "rules must be provided as a list",
                details={"field": "rules", "expected": "list", "actual_type": type(rules).__name__},
            )
        )
        return [], errors

    for rule_index, rule in enumerate(rules):
        if not isinstance(rule, dict):
            errors.append(
                MenaceRuleSchemaError(
                    "rule must be a dict",
                    details={
                        "rule_index": rule_index,
                        "actual_type": type(rule).__name__,
                    },
                )
            )
            continue

        raw_rule_id = rule.get("id")
        if "id" not in rule:
            errors.append(
                _schema_error(
                    "rule missing required field",
                    rule_index=rule_index,
                    rule_id=None,
                    field="id",
                )
            )
        elif not isinstance(raw_rule_id, str) or not raw_rule_id.strip():
            errors.append(
                _schema_error(
                    "rule id must be a non-empty string",
                    rule_index=rule_index,
                    rule_id=raw_rule_id,
                    field="id",
                    actual_type=type(raw_rule_id).__name__,
                )
            )

        rule_type = rule.get("type")
        if "type" not in rule:
            errors.append(
                _schema_error(
                    "rule missing required field",
                    rule_index=rule_index,
                    rule_id=raw_rule_id,
                    field="type",
                )
            )
        elif not isinstance(rule_type, str):
            errors.append(
                _schema_error(
                    "rule type must be a string",
                    rule_index=rule_index,
                    rule_id=raw_rule_id,
                    field="type",
                    actual_type=type(rule_type).__name__,
                )
            )
        elif rule_type not in _SUPPORTED_RULE_TYPES:
            errors.append(
                _schema_error(
                    "unsupported rule type",
                    rule_index=rule_index,
                    rule_id=raw_rule_id,
                    rule_type=rule_type,
                )
            )

        params = rule.get("params")
        if "params" not in rule:
            errors.append(
                _schema_error(
                    "rule missing required field",
                    rule_index=rule_index,
                    rule_id=raw_rule_id,
                    field="params",
                )
            )
        elif not isinstance(params, Mapping):
            errors.append(
                _schema_error(
                    "rule params must be a mapping",
                    rule_index=rule_index,
                    rule_id=raw_rule_id,
                    field="params",
                    actual_type=type(params).__name__,
                )
            )

        if any(error.details.get("rule_index") == rule_index for error in errors):
            continue

        params_mapping = params if isinstance(params, Mapping) else {}
        normalized_params: dict[str, object] = dict(params_mapping)
        param_errors: list[MenaceRuleSchemaError] = []
        if rule_type in {"syntax", "syntax_compile"}:
            normalized_params, param_errors = _validate_syntax_compile_params(
                params_mapping, rule_index=rule_index, rule_id=raw_rule_id
            )
        elif rule_type == "required_imports":
            normalized_params, param_errors = _validate_required_imports_params(
                params_mapping, rule_index=rule_index, rule_id=raw_rule_id
            )
        elif rule_type == "signature_match":
            normalized_params, param_errors = _validate_signature_match_params(
                params_mapping, rule_index=rule_index, rule_id=raw_rule_id
            )
        elif rule_type == "forbidden_patterns":
            normalized_params, param_errors = _validate_forbidden_patterns_params(
                params_mapping, rule_index=rule_index, rule_id=raw_rule_id
            )
        elif rule_type == "static_contracts":
            normalized_params, param_errors = _validate_static_contracts_params(
                params_mapping, rule_index=rule_index, rule_id=raw_rule_id
            )
        elif rule_type == "mandatory_returns":
            normalized_params, param_errors = _validate_mandatory_returns_params(
                params_mapping, rule_index=rule_index, rule_id=raw_rule_id
            )

        if param_errors:
            errors.extend(param_errors)
            continue

        payload = dict(normalized_params)
        payload.setdefault("id", raw_rule_id)
        payload.setdefault("rule_id", raw_rule_id)
        payload["type"] = rule_type

        validated.append(
            ValidatedRule(
                rule_type=rule_type,
                rule_id=raw_rule_id,
                rule_index=rule_index,
                payload=payload,
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
    tree: ast.Module,
    rule_errors: list[PatchRuleError],
    *,
    rule_index: int,
) -> dict[str, object]:
    required = rule.get("imports")
    summary: dict[str, object] = {"missing": []}
    if not isinstance(required, list):
        rule_errors.append(
            PatchRuleError(
                "required_imports must provide an imports list",
                details={"rule_index": rule_index, "code": "required_imports_invalid"},
            )
        )
        return summary
    imports = _extract_imports(tree)
    for item in required:
        if not isinstance(item, dict):
            rule_errors.append(
                PatchRuleError(
                    "invalid import spec",
                    details={"rule_index": rule_index, "code": "required_import_invalid_spec"},
                )
            )
            continue
        module = item.get("module")
        name = item.get("name")
        level = item.get("level", 0)
        kind = item.get("kind")
        module_pattern = item.get("module_pattern")
        name_pattern = item.get("name_pattern")
        matched = _import_matches(
            imports,
            module=module,
            name=name,
            kind=kind,
            level=level,
            module_pattern=module_pattern,
            name_pattern=name_pattern,
            rule_index=rule_index,
            rule_errors=rule_errors,
        )
        if not matched:
            missing = {
                "module": module,
                "name": name,
                "kind": kind,
                "level": level,
                "module_pattern": module_pattern,
                "name_pattern": name_pattern,
            }
            summary["missing"].append(missing)
            rule_errors.append(
                PatchRuleError(
                    "required import missing",
                    details={"rule_index": rule_index, "code": "required_import_missing", "missing": missing},
                )
            )
    return summary


def _apply_signature_matching(
    rule: dict[str, object],
    original_signatures: dict[str, dict[str, dict[str, object]]],
    patched_signatures: dict[str, dict[str, dict[str, object]]],
    rule_errors: list[PatchRuleError],
    *,
    rule_index: int,
) -> dict[str, object]:
    functions = rule.get("functions") or rule.get("function")
    classes = rule.get("classes") or rule.get("class")
    if isinstance(functions, str):
        functions = [functions]
    if isinstance(classes, str):
        classes = [classes]
    require_annotations = bool(rule.get("require_annotations"))
    if not isinstance(functions, list):
        functions = []
    if not isinstance(classes, list):
        classes = []
    if not functions and not classes:
        rule_errors.append(
            PatchRuleError(
                "signature_match requires function or class names",
                details={"rule_index": rule_index, "code": "signature_match_invalid"},
            )
        )
        return {"mismatches": []}
    mismatches: list[dict[str, object]] = []
    for name in functions:
        if not isinstance(name, str):
            mismatch = {"symbol": name, "kind": "function", "error": "invalid_name"}
            mismatches.append(mismatch)
            rule_errors.append(
                PatchRuleError(
                    "function signature invalid",
                    details={"rule_index": rule_index, "code": "signature_invalid_name", "mismatch": mismatch},
                )
            )
            continue
        original_sig = original_signatures["functions"].get(name)
        patched_sig = patched_signatures["functions"].get(name)
        if original_sig is None or patched_sig is None:
            mismatch = {
                "symbol": name,
                "kind": "function",
                "error": "missing_symbol",
                "original": original_sig is not None,
                "patched": patched_sig is not None,
            }
            mismatches.append(mismatch)
            rule_errors.append(
                PatchRuleError(
                    "function signature missing",
                    details={"rule_index": rule_index, "code": "signature_missing", "mismatch": mismatch},
                )
            )
            continue
        original_payload = _signature_payload(original_sig, include_annotations=require_annotations)
        patched_payload = _signature_payload(patched_sig, include_annotations=require_annotations)
        if original_payload != patched_payload:
            mismatch = {
                "symbol": name,
                "kind": "function",
                "original": original_payload,
                "patched": patched_payload,
            }
            mismatches.append(mismatch)
            rule_errors.append(
                PatchRuleError(
                    "function signature mismatch",
                    details={"rule_index": rule_index, "code": "signature_mismatch", "mismatch": mismatch},
                )
            )
    for name in classes:
        if not isinstance(name, str):
            mismatch = {"symbol": name, "kind": "class", "error": "invalid_name"}
            mismatches.append(mismatch)
            rule_errors.append(
                PatchRuleError(
                    "class signature invalid",
                    details={"rule_index": rule_index, "code": "signature_invalid_name", "mismatch": mismatch},
                )
            )
            continue
        original_sig = original_signatures["classes"].get(name)
        patched_sig = patched_signatures["classes"].get(name)
        if original_sig is None or patched_sig is None:
            mismatch = {
                "symbol": name,
                "kind": "class",
                "error": "missing_symbol",
                "original": original_sig is not None,
                "patched": patched_sig is not None,
            }
            mismatches.append(mismatch)
            rule_errors.append(
                PatchRuleError(
                    "class signature missing",
                    details={"rule_index": rule_index, "code": "signature_missing", "mismatch": mismatch},
                )
            )
            continue
        original_payload = _signature_payload(original_sig["signature"], include_annotations=require_annotations)
        patched_payload = _signature_payload(patched_sig["signature"], include_annotations=require_annotations)
        if original_payload != patched_payload:
            mismatch = {
                "symbol": name,
                "kind": "class",
                "original": original_payload,
                "patched": patched_payload,
            }
            mismatches.append(mismatch)
            rule_errors.append(
                PatchRuleError(
                    "class signature mismatch",
                    details={"rule_index": rule_index, "code": "signature_mismatch", "mismatch": mismatch},
                )
            )
    return {"mismatches": mismatches}


class _ForbiddenPatternVisitor(ast.NodeVisitor):
    def __init__(
        self,
        *,
        node_types: set[str],
        names: set[str],
        strings: set[str],
        name_patterns: list[re.Pattern[str]],
        string_patterns: list[re.Pattern[str]],
        call_names: set[str],
        call_patterns: list[re.Pattern[str]],
        attributes: set[str],
        attribute_patterns: list[re.Pattern[str]],
    ) -> None:
        self.node_types = node_types
        self.names = names
        self.strings = strings
        self.name_patterns = name_patterns
        self.string_patterns = string_patterns
        self.call_names = call_names
        self.call_patterns = call_patterns
        self.attributes = attributes
        self.attribute_patterns = attribute_patterns
        self.matches: list[dict[str, object]] = []

    def generic_visit(self, node: ast.AST) -> None:
        node_type = type(node).__name__
        if node_type in self.node_types:
            self.matches.append({"node_type": node_type, "lineno": getattr(node, "lineno", None)})
        super().generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if node.id in self.names:
            self.matches.append({"name": node.id, "lineno": node.lineno})
        for pattern in self.name_patterns:
            if pattern.search(node.id):
                self.matches.append({"name": node.id, "pattern": pattern.pattern, "lineno": node.lineno})
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr in self.attributes:
            self.matches.append({"attribute": node.attr, "lineno": node.lineno})
        for pattern in self.attribute_patterns:
            if pattern.search(node.attr):
                self.matches.append(
                    {"attribute": node.attr, "pattern": pattern.pattern, "lineno": node.lineno}
                )
        self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant) -> None:
        if isinstance(node.value, str):
            if node.value in self.strings:
                self.matches.append({"string": node.value, "lineno": node.lineno})
            for pattern in self.string_patterns:
                if pattern.search(node.value):
                    self.matches.append(
                        {"string": node.value, "pattern": pattern.pattern, "lineno": node.lineno}
                    )
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        call_name = _call_name(node.func)
        if call_name and call_name in self.call_names:
            self.matches.append({"call": call_name, "lineno": node.lineno})
        if call_name:
            for pattern in self.call_patterns:
                if pattern.search(call_name):
                    self.matches.append({"call": call_name, "pattern": pattern.pattern, "lineno": node.lineno})
        self.generic_visit(node)


def _apply_forbidden_patterns(
    rule: dict[str, object],
    tree: ast.Module,
    rule_errors: list[PatchRuleError],
    *,
    rule_index: int,
) -> dict[str, object]:
    node_types = rule.get("node_types") or rule.get("types") or []
    names = rule.get("names") or []
    strings = rule.get("strings") or []
    call_names = rule.get("call_names") or []
    attributes = rule.get("attributes") or []
    name_patterns = rule.get("name_patterns") or []
    string_patterns = rule.get("string_patterns") or []
    call_patterns = rule.get("call_patterns") or []
    attribute_patterns = rule.get("attribute_patterns") or []
    if not isinstance(node_types, list) or not isinstance(names, list) or not isinstance(strings, list) or not isinstance(attributes, list) or not isinstance(call_names, list):
        rule_errors.append(
            PatchRuleError(
                "forbidden_patterns requires list inputs",
                details={"rule_index": rule_index, "code": "forbidden_patterns_invalid"},
            )
        )
        return {"matches": []}
    pattern_errors = _validate_pattern_lists(
        rule_errors,
        rule_index=rule_index,
        name_patterns=name_patterns,
        string_patterns=string_patterns,
        call_patterns=call_patterns,
        attribute_patterns=attribute_patterns,
    )
    visitor = _ForbiddenPatternVisitor(
        node_types=set(str(item) for item in node_types),
        names=set(str(item) for item in names),
        strings=set(str(item) for item in strings),
        name_patterns=pattern_errors["name_patterns"],
        string_patterns=pattern_errors["string_patterns"],
        call_names=set(str(item) for item in call_names),
        call_patterns=pattern_errors["call_patterns"],
        attributes=set(str(item) for item in attributes),
        attribute_patterns=pattern_errors["attribute_patterns"],
    )
    visitor.visit(tree)
    for match in visitor.matches:
        rule_errors.append(
            PatchRuleError(
                "forbidden pattern detected",
                details={"rule_index": rule_index, "code": "forbidden_pattern", "match": match},
            )
        )
    return {"matches": visitor.matches}


def _apply_mandatory_returns(
    rule: dict[str, object],
    patched_nodes: dict[str, ast.AST],
    rule_errors: list[PatchRuleError],
    *,
    rule_index: int,
) -> dict[str, object]:
    functions = rule.get("functions") or rule.get("function")
    if isinstance(functions, str):
        functions = [functions]
    require_terminal = bool(rule.get("require_terminal"))
    require_any = rule.get("require_any", True)
    if not isinstance(functions, list) or not functions:
        rule_errors.append(
            PatchRuleError(
                "mandatory_returns requires function names",
                details={"rule_index": rule_index, "code": "mandatory_returns_invalid"},
            )
        )
        return {"failures": []}
    failures: list[dict[str, object]] = []
    for name in functions:
        node = patched_nodes.get(f"function:{name}")
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            failure = {"function": name, "error": "missing_function"}
            failures.append(failure)
            rule_errors.append(
                PatchRuleError(
                    "mandatory return requirement failed",
                    details={"rule_index": rule_index, "code": "mandatory_return_missing", "failure": failure},
                )
            )
            continue
        return_summary = _return_summary(node)
        if require_any and not return_summary["has_return"]:
            failure = {"function": name, "error": "missing_return", "summary": return_summary}
            failures.append(failure)
            rule_errors.append(
                PatchRuleError(
                    "mandatory return requirement failed",
                    details={"rule_index": rule_index, "code": "mandatory_return_missing", "failure": failure},
                )
            )
        if require_terminal and not return_summary["terminal_return"]:
            failure = {"function": name, "error": "missing_terminal_return", "summary": return_summary}
            failures.append(failure)
            rule_errors.append(
                PatchRuleError(
                    "mandatory return requirement failed",
                    details={"rule_index": rule_index, "code": "mandatory_return_terminal", "failure": failure},
                )
            )
    return {"failures": failures}


def _return_summary(node: ast.FunctionDef | ast.AsyncFunctionDef) -> dict[str, object]:
    has_return = any(isinstance(child, ast.Return) for child in ast.walk(node))
    terminal_return = False
    for statement in reversed(node.body):
        if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Constant):
            if isinstance(statement.value.value, str):
                continue
        terminal_return = isinstance(statement, ast.Return)
        break
    return {"has_return": has_return, "terminal_return": terminal_return}


def _apply_static_contracts(
    rule: dict[str, object],
    patched_nodes: dict[str, ast.AST],
    rule_errors: list[PatchRuleError],
    *,
    rule_index: int,
) -> dict[str, object]:
    functions = rule.get("functions") or rule.get("function")
    if isinstance(functions, str):
        functions = [functions]
    require_docstring = bool(rule.get("require_docstring"))
    require_annotations = bool(rule.get("require_annotations"))
    must_raise = rule.get("must_raise") or []
    if not isinstance(functions, list) or not functions:
        rule_errors.append(
            PatchRuleError(
                "static_contracts requires function names",
                details={"rule_index": rule_index, "code": "static_contracts_invalid"},
            )
        )
        return {"failures": []}
    failures: list[dict[str, object]] = []
    for name in functions:
        node = patched_nodes.get(f"function:{name}")
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            failure = {"function": name, "error": "missing_function"}
            failures.append(failure)
            rule_errors.append(
                PatchRuleError(
                    "static contract violation",
                    details={"rule_index": rule_index, "code": "static_contract_missing", "failure": failure},
                )
            )
            continue
        if require_docstring and not ast.get_docstring(node):
            failure = {"function": name, "error": "missing_docstring"}
            failures.append(failure)
            rule_errors.append(
                PatchRuleError(
                    "static contract violation",
                    details={"rule_index": rule_index, "code": "static_contract_docstring", "failure": failure},
                )
            )
        if require_annotations and not _has_full_annotations(node):
            failure = {"function": name, "error": "missing_annotations"}
            failures.append(failure)
            rule_errors.append(
                PatchRuleError(
                    "static contract violation",
                    details={"rule_index": rule_index, "code": "static_contract_annotations", "failure": failure},
                )
            )
        if must_raise:
            raised = _extract_raised_exceptions(node)
            missing = [exc for exc in must_raise if exc not in raised]
            if missing:
                failure = {"function": name, "error": "missing_exception", "missing": missing}
                failures.append(failure)
                rule_errors.append(
                    PatchRuleError(
                        "static contract violation",
                        details={"rule_index": rule_index, "code": "static_contract_raise", "failure": failure},
                    )
                )
    return {"failures": failures}


def _apply_syntax_compile(
    rule: dict[str, object],
    syntax_errors: list[PatchSyntaxError],
    rule_errors: list[PatchRuleError],
    *,
    rule_index: int,
) -> dict[str, object]:
    sources = rule.get("sources") or rule.get("source") or "patched"
    if isinstance(sources, str):
        sources = [sources]
    if not isinstance(sources, list):
        rule_errors.append(
            PatchRuleError(
                "syntax_compile requires source list",
                details={"rule_index": rule_index, "code": "syntax_compile_invalid"},
            )
        )
        return {"errors": []}
    captured: list[dict[str, object]] = []
    for error in syntax_errors:
        details = dict(error.details or {})
        if details.get("source") not in sources:
            continue
        payload = {
            "source": details.get("source"),
            "lineno": details.get("lineno"),
            "offset": details.get("offset"),
            "col_offset": details.get("col_offset"),
            "text": details.get("text"),
        }
        captured.append(payload)
        rule_errors.append(
            PatchRuleError(
                "syntax error",
                details={
                    "rule_index": rule_index,
                    "code": "syntax_error",
                    "line": details.get("lineno"),
                    "column": details.get("col_offset"),
                    "source": details.get("source"),
                    "error": payload,
                },
            )
        )
    return {"errors": captured}


def _dispatch_rule(
    rule_type: str,
    rule: dict[str, object],
    *,
    rule_index: int,
    original_tree: ast.Module,
    patched_tree: ast.Module,
    original_nodes: dict[str, ast.AST],
    patched_nodes: dict[str, ast.AST],
    original_signatures: dict[str, dict[str, dict[str, object]]],
    patched_signatures: dict[str, dict[str, dict[str, object]]],
    original_index: dict[str, object],
    patched_index: dict[str, object],
    syntax_errors: list[PatchSyntaxError],
) -> tuple[list[PatchRuleError], dict[str, object]]:
    rule_errors: list[PatchRuleError] = []
    summary: dict[str, object] = {
        "rule_index": rule_index,
        "rule_id": rule.get("rule_id") or rule.get("id"),
        "type": rule_type,
        "status": "passed",
        "data": {},
    }
    if rule_type in {"syntax", "syntax_compile"}:
        summary["data"] = _apply_syntax_compile(rule, syntax_errors, rule_errors, rule_index=rule_index)
    elif rule_type == "required_imports":
        summary["data"] = _apply_required_imports(rule, patched_tree, rule_errors, rule_index=rule_index)
    elif rule_type == "signature_match":
        summary["data"] = _apply_signature_matching(
            rule,
            original_signatures,
            patched_signatures,
            rule_errors,
            rule_index=rule_index,
        )
    elif rule_type == "forbidden_patterns":
        summary["data"] = _apply_forbidden_patterns(rule, patched_tree, rule_errors, rule_index=rule_index)
    elif rule_type == "static_contracts":
        summary["data"] = _apply_static_contracts(rule, patched_nodes, rule_errors, rule_index=rule_index)
    elif rule_type == "mandatory_returns":
        summary["data"] = _apply_mandatory_returns(rule, patched_nodes, rule_errors, rule_index=rule_index)
    else:
        rule_errors.append(
            PatchRuleError(
                "unsupported rule type",
                details={"rule_index": rule_index, "rule_type": rule_type, "code": "unsupported_rule"},
            )
        )
    change_summary = _apply_change_requirement(
        rule,
        original_tree,
        patched_tree,
        original_nodes,
        patched_nodes,
        original_index,
        patched_index,
        rule_errors,
        rule_index=rule_index,
    )
    if change_summary is not None:
        summary["data"]["change_requirement"] = change_summary
    if rule_errors:
        summary["status"] = "failed"
        summary["error_count"] = len(rule_errors)
    else:
        summary["error_count"] = 0
    return rule_errors, summary


def _apply_change_requirement(
    rule: Mapping[str, object],
    original_tree: ast.Module,
    patched_tree: ast.Module,
    original_nodes: dict[str, ast.AST],
    patched_nodes: dict[str, ast.AST],
    original_index: dict[str, object],
    patched_index: dict[str, object],
    rule_errors: list[PatchRuleError],
    *,
    rule_index: int,
) -> dict[str, object] | None:
    if not rule.get("require_changes"):
        return None
    functions = rule.get("functions") or rule.get("function")
    classes = rule.get("classes") or rule.get("class")
    if isinstance(functions, str):
        functions = [functions]
    if isinstance(classes, str):
        classes = [classes]
    if not isinstance(functions, list):
        functions = []
    if not isinstance(classes, list):
        classes = []
    unchanged: list[dict[str, object]] = []
    missing: list[dict[str, object]] = []
    for name in functions:
        original_node = original_nodes.get(f"function:{name}")
        patched_node = patched_nodes.get(f"function:{name}")
        if original_node is None or patched_node is None:
            missing_item = {"symbol": name, "kind": "function"}
            missing.append(missing_item)
            rule_errors.append(
                PatchRuleError(
                    "required change target missing",
                    details={
                        "rule_index": rule_index,
                        "code": "change_target_missing",
                        "target": missing_item,
                    },
                )
            )
            continue
        if _node_dump(original_node) == _node_dump(patched_node):
            unchanged_item = {"symbol": name, "kind": "function"}
            unchanged.append(unchanged_item)
            rule_errors.append(
                PatchRuleError(
                    "required change not detected",
                    details={
                        "rule_index": rule_index,
                        "code": "unchanged_target",
                        "target": unchanged_item,
                    },
                )
            )
    for name in classes:
        original_node = original_nodes.get(f"class:{name}")
        patched_node = patched_nodes.get(f"class:{name}")
        if original_node is None or patched_node is None:
            missing_item = {"symbol": name, "kind": "class"}
            missing.append(missing_item)
            rule_errors.append(
                PatchRuleError(
                    "required change target missing",
                    details={
                        "rule_index": rule_index,
                        "code": "change_target_missing",
                        "target": missing_item,
                    },
                )
            )
            continue
        if _node_dump(original_node) == _node_dump(patched_node):
            unchanged_item = {"symbol": name, "kind": "class"}
            unchanged.append(unchanged_item)
            rule_errors.append(
                PatchRuleError(
                    "required change not detected",
                    details={
                        "rule_index": rule_index,
                        "code": "unchanged_target",
                        "target": unchanged_item,
                    },
                )
            )
    module_hash_original = original_index.get("hash")
    module_hash_patched = patched_index.get("hash")
    module_changed = module_hash_original != module_hash_patched
    if not functions and not classes:
        if not module_changed:
            rule_errors.append(
                PatchRuleError(
                    "required change not detected",
                    details={
                        "rule_index": rule_index,
                        "code": "unchanged_module",
                        "original_hash": module_hash_original,
                        "patched_hash": module_hash_patched,
                    },
                )
            )
    return {
        "module_changed": module_changed,
        "unchanged": unchanged,
        "missing": missing,
    }


def _extract_imports(tree: ast.Module) -> list[dict[str, object]]:
    imports: list[dict[str, object]] = []
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
    return imports


def _import_matches(
    imports: list[dict[str, object]],
    *,
    module: object,
    name: object,
    kind: object,
    level: object,
    module_pattern: object,
    name_pattern: object,
    rule_index: int,
    rule_errors: list[PatchRuleError],
) -> bool:
    module_regex = _compile_optional_pattern(module_pattern, rule_errors, rule_index, "module_pattern")
    name_regex = _compile_optional_pattern(name_pattern, rule_errors, rule_index, "name_pattern")
    for entry in imports:
        if module is not None and entry.get("module") != module:
            continue
        if module_regex and not module_regex.search(str(entry.get("module"))):
            continue
        if name is not None and entry.get("name") != name:
            continue
        if name_regex and not name_regex.search(str(entry.get("name") or "")):
            continue
        if kind is not None and entry.get("kind") != kind:
            continue
        if level is not None and entry.get("level") != level:
            continue
        return True
    return False


def _validate_pattern_lists(
    rule_errors: list[PatchRuleError],
    *,
    rule_index: int,
    name_patterns: object,
    string_patterns: object,
    call_patterns: object,
    attribute_patterns: object,
) -> dict[str, list[re.Pattern[str]]]:
    return {
        "name_patterns": _compile_patterns(name_patterns, rule_errors, rule_index, "name_patterns"),
        "string_patterns": _compile_patterns(string_patterns, rule_errors, rule_index, "string_patterns"),
        "call_patterns": _compile_patterns(call_patterns, rule_errors, rule_index, "call_patterns"),
        "attribute_patterns": _compile_patterns(attribute_patterns, rule_errors, rule_index, "attribute_patterns"),
    }


def _compile_patterns(
    patterns: object,
    rule_errors: list[PatchRuleError],
    rule_index: int,
    field: str,
) -> list[re.Pattern[str]]:
    if patterns is None:
        return []
    if isinstance(patterns, str):
        patterns = [patterns]
    if not isinstance(patterns, list):
        rule_errors.append(
            PatchRuleError(
                "pattern list must be a list",
                details={"rule_index": rule_index, "code": "invalid_pattern_list", "field": field},
            )
        )
        return []
    compiled: list[re.Pattern[str]] = []
    for pattern in patterns:
        if not isinstance(pattern, str):
            rule_errors.append(
                PatchRuleError(
                    "pattern must be a string",
                    details={"rule_index": rule_index, "code": "invalid_pattern", "field": field},
                )
            )
            continue
        try:
            compiled.append(re.compile(pattern))
        except re.error as exc:
            rule_errors.append(
                PatchRuleError(
                    "invalid regex pattern",
                    details={
                        "rule_index": rule_index,
                        "code": "invalid_regex",
                        "field": field,
                        "pattern": pattern,
                        "error": str(exc),
                    },
                )
            )
    return compiled


def _compile_optional_pattern(
    pattern: object,
    rule_errors: list[PatchRuleError],
    rule_index: int,
    field: str,
) -> re.Pattern[str] | None:
    if pattern is None:
        return None
    if not isinstance(pattern, str):
        rule_errors.append(
            PatchRuleError(
                "pattern must be a string",
                details={"rule_index": rule_index, "code": "invalid_pattern", "field": field},
            )
        )
        return None
    try:
        return re.compile(pattern)
    except re.error as exc:
        rule_errors.append(
            PatchRuleError(
                "invalid regex pattern",
                details={
                    "rule_index": rule_index,
                    "code": "invalid_regex",
                    "field": field,
                    "pattern": pattern,
                    "error": str(exc),
                },
            )
        )
        return None


def _call_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        value = _call_name(node.value)
        if value:
            return f"{value}.{node.attr}"
        return node.attr
    if isinstance(node, ast.Call):
        return _call_name(node.func)
    return None


def _extract_raised_exceptions(node: ast.AST) -> set[str]:
    names: set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Raise) and child.exc is not None:
            exc_name = _call_name(child.exc)
            if exc_name:
                names.add(exc_name)
    return names


def _has_full_annotations(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    args = node.args
    items = [*args.posonlyargs, *args.args, *args.kwonlyargs]
    if any(arg.annotation is None for arg in items):
        return False
    if args.vararg and args.vararg.annotation is None:
        return False
    if args.kwarg and args.kwarg.annotation is None:
        return False
    return node.returns is not None


__all__ = [
    "PatchValidationLimits",
    "ValidatedRule",
    "validate_patch_text",
    "validate_patch",
    "validate_rules",
]
