from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Iterable

from menace.errors import PatchRuleError, PatchSyntaxError, ValidationError


@dataclass(frozen=True)
class SignatureInfo:
    name: str
    posonlyargs: tuple[str, ...]
    args: tuple[str, ...]
    vararg: str | None
    kwonlyargs: tuple[str, ...]
    kwarg: str | None
    defaults: tuple[str, ...]
    kw_defaults: tuple[str | None, ...]
    annotations: dict[str, str]
    returns: str | None


@dataclass(frozen=True)
class ValidatedRule:
    rule_type: str
    rule_id: str | None
    severity: str | None
    rule_index: int
    payload: dict[str, object]


_SUPPORTED_RULE_TYPES = {
    "syntax",
    "required_imports",
    "signature_match",
    "forbidden_patterns",
    "static_contracts",
    "mandatory_return",
}


def parse_module(source: str) -> ast.Module:
    if not isinstance(source, str):
        raise PatchSyntaxError(
            "source must be a string",
            details={"actual_type": type(source).__name__},
        )
    try:
        compile(source, "<module>", "exec")
    except SyntaxError as exc:
        raise PatchSyntaxError(
            "syntax error",
            details=_syntax_details(exc),
        ) from exc
    try:
        return ast.parse(source, filename="<module>", mode="exec")
    except SyntaxError as exc:
        raise PatchSyntaxError(
            "syntax error",
            details=_syntax_details(exc),
        ) from exc


def extract_imports(tree: ast.Module) -> set[tuple[str, str | None]]:
    imports: set[tuple[str, str | None]] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add((alias.name, None))
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                imports.add((module, alias.name))
    return imports


def extract_function_signatures(tree: ast.Module) -> dict[str, SignatureInfo]:
    signatures: dict[str, SignatureInfo] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            signatures[node.name] = _signature_info(node)
    return signatures


def function_guarantees_return(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    return _statements_guarantee_return(node.body)


def validate_patch(
    original: str,
    patched: str,
    rules: list[dict[str, object]],
) -> dict[str, object]:
    errors: list[dict[str, object]] = []
    data: dict[str, object] = {"results": []}
    syntax_errors: list[PatchSyntaxError] = []

    validated_rules, rule_errors = validate_rules(rules)
    errors.extend(error.to_dict() for error in rule_errors)

    original_tree = None
    patched_tree = None
    try:
        original_tree = parse_module(original)
    except PatchSyntaxError as exc:
        syntax_errors.append(_with_source(exc, "original"))
    try:
        patched_tree = parse_module(patched)
    except PatchSyntaxError as exc:
        syntax_errors.append(_with_source(exc, "patched"))

    errors.extend(error.to_dict() for error in syntax_errors)

    original_imports: set[tuple[str, str | None]] = set()
    patched_imports: set[tuple[str, str | None]] = set()
    original_signatures: dict[str, SignatureInfo] = {}
    patched_signatures: dict[str, SignatureInfo] = {}
    original_functions: dict[str, ast.AST] = {}
    patched_functions: dict[str, ast.AST] = {}

    if original_tree is not None:
        original_imports = extract_imports(original_tree)
        original_signatures = extract_function_signatures(original_tree)
        original_functions = _extract_function_nodes(original_tree)

    if patched_tree is not None:
        patched_imports = extract_imports(patched_tree)
        patched_signatures = extract_function_signatures(patched_tree)
        patched_functions = _extract_function_nodes(patched_tree)

    data["original"] = {
        "imports": sorted(original_imports),
        "functions": sorted(original_signatures),
    }
    data["patched"] = {
        "imports": sorted(patched_imports),
        "functions": sorted(patched_signatures),
    }

    for validated_rule in validated_rules:
        result = {
            "rule_index": validated_rule.rule_index,
            "rule_id": validated_rule.rule_id,
            "type": validated_rule.rule_type,
            "status": "pass",
            "details": {},
        }
        rule_payload = validated_rule.payload
        try:
            if validated_rule.rule_type == "syntax":
                if syntax_errors:
                    result["status"] = "fail"
                    result["details"] = {"errors": [err.to_dict() for err in syntax_errors]}
            elif validated_rule.rule_type == "required_imports":
                failures = _check_required_imports(rule_payload, patched_imports)
                if failures:
                    result["status"] = "fail"
                    result["details"] = {"missing": failures}
                    errors.append(
                        PatchRuleError(
                            "required imports missing",
                            details=_rule_error_context(validated_rule, missing=failures),
                        ).to_dict()
                    )
            elif validated_rule.rule_type == "signature_match":
                mismatches = _check_signature_match(
                    rule_payload,
                    original_signatures,
                    patched_signatures,
                )
                if mismatches:
                    result["status"] = "fail"
                    result["details"] = {"mismatches": mismatches}
                    errors.append(
                        PatchRuleError(
                            "signature mismatch",
                            details=_rule_error_context(validated_rule, mismatches=mismatches),
                        ).to_dict()
                    )
            elif validated_rule.rule_type == "forbidden_patterns":
                if patched_tree is None:
                    result["status"] = "fail"
                    result["details"] = {"error": "patched_source_invalid"}
                else:
                    matches = _check_forbidden_patterns(rule_payload, patched_tree)
                    if matches:
                        result["status"] = "fail"
                        result["details"] = {"matches": matches}
                        errors.append(
                            PatchRuleError(
                                "forbidden patterns detected",
                                details=_rule_error_context(validated_rule, matches=matches),
                            ).to_dict()
                        )
            elif validated_rule.rule_type == "static_contracts":
                failures = _check_static_contracts(rule_payload, patched_functions)
                if failures:
                    result["status"] = "fail"
                    result["details"] = {"failures": failures}
                    errors.append(
                        PatchRuleError(
                            "static contract failure",
                            details=_rule_error_context(validated_rule, failures=failures),
                        ).to_dict()
                    )
            elif validated_rule.rule_type == "mandatory_return":
                failures = _check_mandatory_return(rule_payload, patched_functions)
                if failures:
                    result["status"] = "fail"
                    result["details"] = {"failures": failures}
                    errors.append(
                        PatchRuleError(
                            "mandatory return requirement failed",
                            details=_rule_error_context(validated_rule, failures=failures),
                        ).to_dict()
                    )
            else:
                result["status"] = "fail"
                errors.append(
                    PatchRuleError(
                        "unsupported rule type",
                        details=_rule_error_context(validated_rule),
                    ).to_dict()
                )
        except Exception as exc:
            result["status"] = "fail"
            errors.append(
                ValidationError(
                    "unexpected validation error",
                    details={
                        "rule_index": validated_rule.rule_index,
                        "rule_id": validated_rule.rule_id,
                        "exception_type": exc.__class__.__name__,
                        "exception_message": str(exc),
                    },
                ).to_dict()
            )
        data["results"].append(result)

    meta = {
        "rule_count": len(rules),
        "validated_rule_count": len(validated_rules),
        "syntax_error_count": len(syntax_errors),
        "error_count": len(errors),
    }

    status = "pass" if not errors else "fail"
    return {
        "status": status,
        "data": data,
        "errors": errors,
        "meta": meta,
    }


def validate_rules(
    rules: list[dict[str, object]],
) -> tuple[list[ValidatedRule], list[ValidationError]]:
    errors: list[PatchRuleError] = []
    validated: list[ValidatedRule] = []
    if not isinstance(rules, list):
        errors.append(
            PatchRuleError(
                "rules must be a list",
                details={"field": "rules", "actual_type": type(rules).__name__},
            )
        )
        return [], errors

    for rule_index, rule in enumerate(rules):
        if not isinstance(rule, dict):
            errors.append(
                PatchRuleError(
                    "rule must be a dict",
                    details={"rule_index": rule_index, "actual_type": type(rule).__name__},
                )
            )
            continue

        rule_id = _normalize_rule_id(rule)
        rule_type = rule.get("type")
        severity = rule.get("severity")

        if not isinstance(rule_type, str):
            errors.append(
                PatchRuleError(
                    "rule type must be a string",
                    details=_rule_error_context_index(rule_index, rule_id, field="type"),
                )
            )
            continue

        if rule_type not in _SUPPORTED_RULE_TYPES:
            errors.append(
                PatchRuleError(
                    "unsupported rule type",
                    details=_rule_error_context_index(
                        rule_index,
                        rule_id,
                        rule_type=rule_type,
                    ),
                )
            )
            continue

        if severity is not None and not isinstance(severity, str):
            errors.append(
                PatchRuleError(
                    "severity must be a string",
                    details=_rule_error_context_index(rule_index, rule_id, field="severity"),
                )
            )
            continue

        specific_error = _validate_rule_payload(rule_index, rule_id, rule_type, rule)
        if specific_error:
            errors.append(specific_error)
            continue

        validated.append(
            ValidatedRule(
                rule_type=rule_type,
                rule_id=rule_id,
                severity=severity,
                rule_index=rule_index,
                payload=rule,
            )
        )

    return validated, errors


def _syntax_details(exc: SyntaxError) -> dict[str, object]:
    return {
        "lineno": exc.lineno,
        "offset": exc.offset,
        "text": exc.text,
    }


def _with_source(error: PatchSyntaxError, source: str) -> PatchSyntaxError:
    details = dict(error.details or {})
    details["source"] = source
    return PatchSyntaxError(error.message, details=details)


def _extract_function_nodes(tree: ast.Module) -> dict[str, ast.AST]:
    nodes: dict[str, ast.AST] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            nodes[node.name] = node
    return nodes


def _signature_info(node: ast.FunctionDef | ast.AsyncFunctionDef) -> SignatureInfo:
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

    return SignatureInfo(
        name=node.name,
        posonlyargs=tuple(arg.arg for arg in args.posonlyargs),
        args=tuple(arg.arg for arg in args.args),
        vararg=args.vararg.arg if args.vararg else None,
        kwonlyargs=tuple(arg.arg for arg in args.kwonlyargs),
        kwarg=args.kwarg.arg if args.kwarg else None,
        defaults=tuple(_node_dump(value) for value in args.defaults),
        kw_defaults=tuple(_node_dump(value) if value is not None else None for value in args.kw_defaults),
        annotations=annotations,
        returns=returns,
    )


def _node_dump(node: ast.AST) -> str:
    return ast.dump(node, include_attributes=False)


def _normalize_rule_id(rule: dict[str, object]) -> str | None:
    raw = rule.get("rule_id")
    if raw is None and "id" in rule:
        raw = rule.get("id")
    if raw is None:
        return None
    if not isinstance(raw, str):
        return None
    return raw


def _validate_rule_payload(
    rule_index: int,
    rule_id: str | None,
    rule_type: str,
    rule: dict[str, object],
) -> PatchRuleError | None:
    if rule_type == "syntax":
        return None
    if rule_type == "required_imports":
        imports = rule.get("imports")
        if not isinstance(imports, list) or not imports:
            return PatchRuleError(
                "required_imports must include a non-empty imports list",
                details=_rule_error_context_index(rule_index, rule_id, field="imports"),
            )
        for item in imports:
            if not isinstance(item, dict):
                return PatchRuleError(
                    "import spec must be a dict",
                    details=_rule_error_context_index(rule_index, rule_id, field="imports"),
                )
            module = item.get("module")
            if not isinstance(module, str):
                return PatchRuleError(
                    "import spec module must be a string",
                    details=_rule_error_context_index(rule_index, rule_id, field="module"),
                )
            names = item.get("names")
            if names is not None:
                if not isinstance(names, list) or not all(isinstance(name, str) for name in names):
                    return PatchRuleError(
                        "import spec names must be a list of strings",
                        details=_rule_error_context_index(rule_index, rule_id, field="names"),
                    )
        return None
    if rule_type == "signature_match":
        functions = rule.get("functions") or rule.get("function")
        if isinstance(functions, str):
            functions = [functions]
        if not isinstance(functions, list) or not functions or not all(isinstance(name, str) for name in functions):
            return PatchRuleError(
                "signature_match requires function names",
                details=_rule_error_context_index(rule_index, rule_id, field="functions"),
            )
        require_annotations = rule.get("require_annotations")
        if require_annotations is not None and not isinstance(require_annotations, bool):
            return PatchRuleError(
                "require_annotations must be a bool",
                details=_rule_error_context_index(rule_index, rule_id, field="require_annotations"),
            )
        return None
    if rule_type == "forbidden_patterns":
        node_types = rule.get("node_types") or []
        names = rule.get("names") or []
        strings = rule.get("strings") or []
        call_names = rule.get("call_names") or []
        attributes = rule.get("attributes") or []
        if not all(isinstance(item, list) for item in [node_types, names, strings, call_names, attributes]):
            return PatchRuleError(
                "forbidden_patterns expects list parameters",
                details=_rule_error_context_index(rule_index, rule_id),
            )
        if not any([node_types, names, strings, call_names, attributes]):
            return PatchRuleError(
                "forbidden_patterns requires at least one constraint",
                details=_rule_error_context_index(rule_index, rule_id),
            )
        for collection, field in [
            (node_types, "node_types"),
            (names, "names"),
            (strings, "strings"),
            (call_names, "call_names"),
            (attributes, "attributes"),
        ]:
            if not all(isinstance(item, str) for item in collection):
                return PatchRuleError(
                    "forbidden_patterns entries must be strings",
                    details=_rule_error_context_index(rule_index, rule_id, field=field),
                )
        return None
    if rule_type == "static_contracts":
        functions = rule.get("functions") or rule.get("function")
        if isinstance(functions, str):
            functions = [functions]
        if not isinstance(functions, list) or not functions or not all(isinstance(name, str) for name in functions):
            return PatchRuleError(
                "static_contracts requires function names",
                details=_rule_error_context_index(rule_index, rule_id, field="functions"),
            )
        for field in ("require_docstring", "require_annotations"):
            value = rule.get(field)
            if value is not None and not isinstance(value, bool):
                return PatchRuleError(
                    f"{field} must be a bool",
                    details=_rule_error_context_index(rule_index, rule_id, field=field),
                )
        must_raise = rule.get("must_raise")
        if must_raise is not None:
            if not isinstance(must_raise, list) or not all(isinstance(name, str) for name in must_raise):
                return PatchRuleError(
                    "must_raise must be a list of strings",
                    details=_rule_error_context_index(rule_index, rule_id, field="must_raise"),
                )
        return None
    if rule_type == "mandatory_return":
        functions = rule.get("functions") or rule.get("function")
        if isinstance(functions, str):
            functions = [functions]
        if not isinstance(functions, list) or not functions or not all(isinstance(name, str) for name in functions):
            return PatchRuleError(
                "mandatory_return requires function names",
                details=_rule_error_context_index(rule_index, rule_id, field="functions"),
            )
        return None
    return PatchRuleError(
        "unsupported rule type",
        details=_rule_error_context_index(rule_index, rule_id, rule_type=rule_type),
    )


def _rule_error_context(rule: ValidatedRule, **extra: object) -> dict[str, object]:
    details: dict[str, object] = {"rule_index": rule.rule_index}
    if rule.rule_id is not None:
        details["rule_id"] = rule.rule_id
    details.update(extra)
    return details


def _rule_error_context_index(
    rule_index: int,
    rule_id: str | None,
    **extra: object,
) -> dict[str, object]:
    details: dict[str, object] = {"rule_index": rule_index}
    if rule_id is not None:
        details["rule_id"] = rule_id
    details.update(extra)
    return details


def _check_required_imports(
    rule: dict[str, object],
    imports: set[tuple[str, str | None]],
) -> list[dict[str, object]]:
    required = rule.get("imports") or []
    missing: list[dict[str, object]] = []
    for item in required:
        module = item.get("module")
        names = item.get("names")
        if not isinstance(module, str):
            missing.append({"error": "invalid_module", "module": module})
            continue
        if names is None:
            if (module, None) not in imports and not any(entry[0] == module for entry in imports):
                missing.append({"module": module, "names": None})
            continue
        for name in names:
            if (module, name) not in imports:
                missing.append({"module": module, "name": name})
    return missing


def _check_signature_match(
    rule: dict[str, object],
    original: dict[str, SignatureInfo],
    patched: dict[str, SignatureInfo],
) -> list[dict[str, object]]:
    functions = rule.get("functions") or rule.get("function")
    if isinstance(functions, str):
        functions = [functions]
    require_annotations = bool(rule.get("require_annotations"))
    mismatches: list[dict[str, object]] = []
    for name in functions:
        original_sig = original.get(name)
        patched_sig = patched.get(name)
        if original_sig is None or patched_sig is None:
            mismatches.append(
                {
                    "function": name,
                    "error": "missing_function",
                    "original": original_sig is not None,
                    "patched": patched_sig is not None,
                }
            )
            continue
        if not _signatures_match(original_sig, patched_sig, require_annotations=require_annotations):
            mismatches.append(
                {
                    "function": name,
                    "original": _signature_to_payload(original_sig, include_annotations=require_annotations),
                    "patched": _signature_to_payload(patched_sig, include_annotations=require_annotations),
                }
            )
    return mismatches


def _signatures_match(
    original: SignatureInfo,
    patched: SignatureInfo,
    *,
    require_annotations: bool,
) -> bool:
    if _signature_to_payload(original, include_annotations=False) != _signature_to_payload(
        patched, include_annotations=False
    ):
        return False
    if require_annotations:
        return _signature_to_payload(original, include_annotations=True) == _signature_to_payload(
            patched, include_annotations=True
        )
    return True


def _signature_to_payload(signature: SignatureInfo, *, include_annotations: bool) -> dict[str, object]:
    payload = {
        "posonlyargs": signature.posonlyargs,
        "args": signature.args,
        "vararg": signature.vararg,
        "kwonlyargs": signature.kwonlyargs,
        "kwarg": signature.kwarg,
        "defaults": signature.defaults,
        "kw_defaults": signature.kw_defaults,
    }
    if include_annotations:
        payload["annotations"] = signature.annotations
        payload["returns"] = signature.returns
    return payload


class _ForbiddenPatternVisitor(ast.NodeVisitor):
    def __init__(
        self,
        *,
        node_types: set[str],
        names: set[str],
        strings: set[str],
        call_names: set[str],
        attributes: set[str],
    ) -> None:
        self.node_types = node_types
        self.names = names
        self.strings = strings
        self.call_names = call_names
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

    def visit_Call(self, node: ast.Call) -> None:
        call_name = _call_name(node.func)
        if call_name and call_name in self.call_names:
            self.matches.append({"call": call_name, "lineno": node.lineno})
        self.generic_visit(node)


def _check_forbidden_patterns(
    rule: dict[str, object],
    tree: ast.Module,
) -> list[dict[str, object]]:
    visitor = _ForbiddenPatternVisitor(
        node_types=set(rule.get("node_types") or []),
        names=set(rule.get("names") or []),
        strings=set(rule.get("strings") or []),
        call_names=set(rule.get("call_names") or []),
        attributes=set(rule.get("attributes") or []),
    )
    visitor.visit(tree)
    return visitor.matches


def _check_static_contracts(
    rule: dict[str, object],
    functions: dict[str, ast.AST],
) -> list[dict[str, object]]:
    targets = rule.get("functions") or rule.get("function")
    if isinstance(targets, str):
        targets = [targets]
    require_docstring = bool(rule.get("require_docstring"))
    require_annotations = bool(rule.get("require_annotations"))
    must_raise = rule.get("must_raise") or []

    failures: list[dict[str, object]] = []
    for name in targets:
        node = functions.get(name)
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            failures.append({"function": name, "error": "missing_function"})
            continue
        if require_docstring and not ast.get_docstring(node):
            failures.append({"function": name, "error": "missing_docstring"})
        if require_annotations and not _has_full_annotations(node):
            failures.append({"function": name, "error": "missing_annotations"})
        if must_raise:
            raised = _extract_raised_exceptions(node)
            missing = [exc for exc in must_raise if exc not in raised]
            if missing:
                failures.append({"function": name, "error": "missing_exception", "missing": missing})
    return failures


def _check_mandatory_return(
    rule: dict[str, object],
    functions: dict[str, ast.AST],
) -> list[dict[str, object]]:
    targets = rule.get("functions") or rule.get("function")
    if isinstance(targets, str):
        targets = [targets]
    failures: list[dict[str, object]] = []
    for name in targets:
        node = functions.get(name)
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            failures.append({"function": name, "error": "missing_function"})
            continue
        if not function_guarantees_return(node):
            failures.append({"function": name, "error": "missing_guaranteed_return"})
    return failures


def _extract_raised_exceptions(node: ast.AST) -> set[str]:
    names: set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Raise) and child.exc is not None:
            exc_name = _call_name(child.exc)
            if exc_name:
                names.add(exc_name)
    return names


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


def _statements_guarantee_return(statements: Iterable[ast.stmt]) -> bool:
    for statement in statements:
        if _statement_guarantees_return(statement):
            return True
    return False


def _statement_guarantees_return(statement: ast.stmt) -> bool:
    if isinstance(statement, ast.Return):
        return True
    if isinstance(statement, ast.If):
        return _statements_guarantee_return(statement.body) and _statements_guarantee_return(statement.orelse)
    if isinstance(statement, ast.Try):
        if statement.finalbody and _statements_guarantee_return(statement.finalbody):
            return True
        if not _statements_guarantee_return(statement.body):
            return False
        if not _statements_guarantee_return(statement.orelse):
            return False
        for handler in statement.handlers:
            if not _statements_guarantee_return(handler.body):
                return False
        return True
    if isinstance(statement, (ast.With, ast.AsyncWith)):
        return _statements_guarantee_return(statement.body)
    if isinstance(statement, ast.Match):
        if not _match_has_default(statement):
            return False
        return all(_statements_guarantee_return(case.body) for case in statement.cases)
    return False


def _match_has_default(statement: ast.Match) -> bool:
    for case in statement.cases:
        if isinstance(case.pattern, ast.MatchAs) and case.pattern.name is None and case.pattern.pattern is None:
            return True
        if isinstance(case.pattern, ast.MatchWildcard):
            return True
    return False


__all__ = [
    "SignatureInfo",
    "ValidatedRule",
    "parse_module",
    "extract_imports",
    "extract_function_signatures",
    "function_guarantees_return",
    "validate_patch",
    "validate_rules",
]
