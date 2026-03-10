#!/usr/bin/env python3
"""Phase 1 QA checker for invalid runtime stubs in non-test Python files."""
from __future__ import annotations

import argparse
import ast
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[2]

EXCLUDED_PARTS = {
    ".git",
    ".venv",
    "venv",
    "tests",
    "test",
    "fixtures",
    "__pycache__",
}
EXCLUDED_FILES = {"conftest.py"}

# Explicit exceptions for intentional sentinels or compatibility shims.
ALLOWLIST_REASONS: dict[tuple[str, str, str], str] = {}

RUNTIME_SYMBOL_HINTS = (
    "CLIENT",
    "SERVICE",
    "ADAPTER",
    "BACKEND",
    "PROVIDER",
    "BROKER",
    "ENGINE",
    "ROUTER",
    "MANAGER",
)


@dataclass(frozen=True)
class Finding:
    path: str
    line: int
    symbol: str
    rule: str
    message: str


def _allowlisted(path: str, rule: str, symbol: str) -> str | None:
    return ALLOWLIST_REASONS.get((path, rule, symbol))


def _iter_tracked_python_files(paths: Iterable[str]) -> list[Path]:
    tracked = subprocess.run(
        ["git", "ls-files", "*.py"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()

    scopes = [ROOT / p for p in paths]
    files: list[Path] = []
    for rel in tracked:
        rel_path = Path(rel)
        if rel_path.name in EXCLUDED_FILES or any(part in EXCLUDED_PARTS for part in rel_path.parts):
            continue
        abs_path = ROOT / rel_path
        if any(scope == ROOT or abs_path.is_relative_to(scope) for scope in scopes):
            files.append(abs_path)
    return files


class ModuleAnalyzer(ast.NodeVisitor):
    def __init__(self, relpath: str, tree: ast.Module) -> None:
        self.relpath = relpath
        self.tree = tree
        self.findings: list[Finding] = []
        self.exported: set[str] = set()
        self.bad_assignments: dict[str, tuple[int, str]] = {}
        self.container_types: list[dict[str, str]] = [{}]
        self.current_class: ast.ClassDef | None = None
        self.shim_classes: set[str] = set()

    def run(self) -> list[Finding]:
        self._collect_exports()
        self.visit(self.tree)
        for symbol, (line, rule) in self.bad_assignments.items():
            if _allowlisted(self.relpath, rule, symbol):
                continue
            self.findings.append(
                Finding(
                    self.relpath,
                    line,
                    symbol,
                    rule,
                    "Runtime/exported dependency symbol uses invalid placeholder assignment.",
                )
            )
        return sorted(self.findings, key=lambda f: (f.path, f.line, f.rule, f.symbol))

    def _collect_exports(self) -> None:
        for node in self.tree.body:
            if not isinstance(node, ast.Assign):
                continue
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "__all__" and isinstance(node.value, (ast.List, ast.Tuple)):
                    for elt in node.value.elts:
                        if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                            self.exported.add(elt.value)

    def _is_runtime_symbol(self, name: str) -> bool:
        if name in self.exported:
            return True
        if name.startswith("_"):
            return False
        if name.isupper():
            return True
        return any(name.upper().endswith(suffix) for suffix in RUNTIME_SYMBOL_HINTS)

    def _placeholder_rule(self, value: ast.AST) -> str | None:
        if isinstance(value, ast.Constant) and value.value is None:
            return "placeholder_none"
        if isinstance(value, ast.Name) and value.id == "object":
            return "placeholder_object"
        if isinstance(value, ast.Call) and self._is_simple_namespace_ctor(value.func):
            return "placeholder_simple_namespace"
        return None

    @staticmethod
    def _is_simple_namespace_ctor(node: ast.AST) -> bool:
        return (isinstance(node, ast.Name) and node.id == "SimpleNamespace") or (
            isinstance(node, ast.Attribute) and node.attr == "SimpleNamespace"
        )

    def visit_Assign(self, node: ast.Assign) -> None:
        if len(self.container_types) == 1 and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            symbol = node.targets[0].id
            rule = self._placeholder_rule(node.value)
            if rule and self._is_runtime_symbol(symbol):
                self.bad_assignments[symbol] = (node.lineno, rule)

        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            var_name = node.targets[0].id
            container_type = self._container_type(node.value)
            if container_type:
                self.container_types[-1][var_name] = container_type
        self.generic_visit(node)

    def _container_type(self, node: ast.AST) -> str | None:
        if isinstance(node, ast.Dict):
            return "dict"
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "dict":
            return "dict"
        if isinstance(node, (ast.List, ast.Set, ast.Tuple)):
            return type(node).__name__.lower()
        return None

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.container_types.append({})
        in_shim = self.current_class and self.current_class.name in self.shim_classes
        if in_shim and self._is_pass_only_method(node):
            self._add_finding(node.lineno, node.name, "shim_method_pass", "Fallback shim method contains only `pass`.")
        self.generic_visit(node)
        self.container_types.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.visit_FunctionDef(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        prev = self.current_class
        self.current_class = node
        if self._is_shim_class(node):
            self.shim_classes.add(node.name)
        self.generic_visit(node)
        self.current_class = prev

    def _is_shim_class(self, node: ast.ClassDef) -> bool:
        class_name = node.name.lower()
        if any(token in class_name for token in ("shim", "fallback", "stub")):
            return True
        for deco in node.decorator_list:
            if isinstance(deco, ast.Name) and deco.id.lower() in {"shim", "fallback_shim"}:
                return True
            if isinstance(deco, ast.Attribute) and deco.attr.lower() in {"shim", "fallback_shim"}:
                return True
        return False

    @staticmethod
    def _is_pass_only_method(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
        body = node.body
        if len(body) == 1 and isinstance(body[0], ast.Pass):
            return True
        if len(body) == 2 and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str) and isinstance(body[1], ast.Pass):
            return True
        return False

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and node.func.attr == "add":
            symbol = node.func.value.id
            if self._lookup_container(symbol) == "dict":
                self._add_finding(
                    node.lineno,
                    symbol,
                    "container_mismatch_add_on_dict",
                    "Symbol initialized as dict is used with `.add(...)`.",
                )
        self.generic_visit(node)

    def _lookup_container(self, symbol: str) -> str | None:
        for scope in reversed(self.container_types):
            if symbol in scope:
                return scope[symbol]
        return None

    def _add_finding(self, line: int, symbol: str, rule: str, message: str) -> None:
        if _allowlisted(self.relpath, rule, symbol):
            return
        self.findings.append(Finding(self.relpath, line, symbol, rule, message))


def check_file(path: Path) -> list[Finding]:
    abs_path = path if path.is_absolute() else ROOT / path
    rel = abs_path.relative_to(ROOT).as_posix()
    try:
        source = abs_path.read_text(encoding="utf-8")
        tree = ast.parse(source)
    except SyntaxError as exc:
        return [Finding(rel, exc.lineno or 1, "<module>", "syntax_error", f"Unable to parse file: {exc.msg}")]
    return ModuleAnalyzer(rel, tree).run()


def run(paths: Iterable[str]) -> list[Finding]:
    findings: list[Finding] = []
    for path in _iter_tracked_python_files(paths):
        findings.extend(check_file(path))
    return sorted(findings, key=lambda f: (f.path, f.line, f.rule, f.symbol))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", default=["."], help="Optional repository subpaths to scan.")
    args = parser.parse_args()

    findings = run(args.paths)
    if not findings:
        print("check_invalid_stubs: no violations found")
        return 0

    for finding in findings:
        print(f"{finding.path}:{finding.line}: {finding.symbol}: [{finding.rule}] {finding.message}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
