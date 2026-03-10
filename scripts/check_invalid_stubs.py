#!/usr/bin/env python3
"""Detect risky runtime shim/stub placeholders."""
from __future__ import annotations

import argparse
import ast
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parent.parent

# Narrowly scoped exceptions for known-safe bootstrap/compatibility shims.
ALLOWLIST_REASONS: dict[tuple[str, str, str], str] = {
    (
        "self_improvement/engine.py",
        "none_called",
        "TelemetryEvent",
    ): "TelemetryEvent defaults to None when optional telemetry extras are absent; execution paths are guarded and treat None as an explicit opt-out.",
    (
        "sandbox_runner/__init__.py",
        "none_called",
        "_env_simulate_temporal_trajectory",
    ): "The lazy loader raises RuntimeError when the export is missing before invocation, so the fallback None is never invoked unsafely.",
}

EXCLUDED_DIRS = {".git", "docs", "tests", "unit_tests", "venv", ".venv"}
EXCLUDED_FILES = {"conftest.py"}
DEFAULT_RUNTIME_PATHS = (".",)


@dataclass(frozen=True)
class Finding:
    path: str
    line: int
    code: str
    message: str


def _iter_tracked_python_files(runtime_paths: Iterable[str]) -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "*.py"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    scoped_roots = [ROOT / p for p in runtime_paths]
    files: list[Path] = []
    for rel in result.stdout.splitlines():
        rel_path = Path(rel)
        if any(part in EXCLUDED_DIRS for part in rel_path.parts) or rel_path.name in EXCLUDED_FILES:
            continue
        full_path = ROOT / rel_path
        if any(str(root) == str(ROOT) or full_path.is_relative_to(root) for root in scoped_roots):
            files.append(full_path)
    return files


def _is_abstract_method(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    for deco in fn.decorator_list:
        if isinstance(deco, ast.Name) and deco.id == "abstractmethod":
            return True
        if isinstance(deco, ast.Attribute) and deco.attr == "abstractmethod":
            return True
    return False


def _allowlisted(path: str, code: str, symbol: str) -> str | None:
    return ALLOWLIST_REASONS.get((path, code, symbol))


class _ModuleAnalyzer(ast.NodeVisitor):
    def __init__(self, relpath: str, tree: ast.Module) -> None:
        self.relpath = relpath
        self.tree = tree
        self.findings: list[Finding] = []
        self._top_none: dict[str, int] = {}
        self._top_object: dict[str, int] = {}
        self._top_simple_namespace: dict[str, int] = {}
        self._parent_stack: list[ast.AST] = []

    def run(self) -> list[Finding]:
        for stmt in self.tree.body:
            if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
                name = stmt.targets[0].id
                value = stmt.value
                if isinstance(value, ast.Constant) and value.value is None:
                    self._top_none[name] = stmt.lineno
                elif isinstance(value, ast.Name) and value.id == "object":
                    self._top_object[name] = stmt.lineno
                elif isinstance(value, ast.Call) and self._is_simple_namespace_ctor(value.func):
                    self._top_simple_namespace[name] = stmt.lineno

        self.visit(self.tree)

        for name, line in self._top_object.items():
            if _allowlisted(self.relpath, "object_fallback", name) is None:
                self.findings.append(
                    Finding(
                        self.relpath,
                        line,
                        "object_fallback",
                        f"Top-level fallback `{name} = object` is forbidden; use an explicit shim class with deterministic methods.",
                    )
                )

        for name, line in self._top_simple_namespace.items():
            if _allowlisted(self.relpath, "simple_namespace_shim", name) is None:
                self.findings.append(
                    Finding(
                        self.relpath,
                        line,
                        "simple_namespace_shim",
                        f"Top-level SimpleNamespace shim `{name}` is risky; replace with a concrete callable/service class.",
                    )
                )
        return sorted(self.findings, key=lambda f: (f.path, f.line, f.code))

    @property
    def _module_uses_shims(self) -> bool:
        return bool(self._top_none or self._top_object or self._top_simple_namespace)

    def generic_visit(self, node: ast.AST) -> None:
        self._parent_stack.append(node)
        super().generic_visit(node)
        self._parent_stack.pop()

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name) and node.func.id in self._top_none:
            sym = node.func.id
            if not self._is_guarded_none_call(sym) and _allowlisted(self.relpath, "none_called", sym) is None:
                self.findings.append(
                    Finding(
                        self.relpath,
                        self._top_none[sym],
                        "none_called",
                        f"Top-level `{sym} = None` is later called; replace with a callable shim class or guarded injection.",
                    )
                )
        self.generic_visit(node)


    def _is_guarded_none_call(self, symbol: str) -> bool:
        for ancestor in reversed(self._parent_stack[:-1]):
            if isinstance(ancestor, ast.If):
                if self._if_guards_symbol_not_none(ancestor.test, symbol):
                    return True
        return False

    @staticmethod
    def _if_guards_symbol_not_none(test: ast.AST, symbol: str) -> bool:
        if isinstance(test, ast.Name) and test.id == symbol:
            return True
        if isinstance(test, ast.Compare) and isinstance(test.left, ast.Name) and test.left.id == symbol:
            if len(test.ops) == 1 and len(test.comparators) == 1 and isinstance(test.comparators[0], ast.Constant) and test.comparators[0].value is None:
                return isinstance(test.ops[0], ast.IsNot)
        if isinstance(test, ast.BoolOp):
            return any(_ModuleAnalyzer._if_guards_symbol_not_none(value, symbol) for value in test.values)
        if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
            return False
        return False

    def visit_Pass(self, node: ast.Pass) -> None:
        if self._module_uses_shims and self._is_executable_pass(node):
            sym = f"pass@{node.lineno}"
            if _allowlisted(self.relpath, "bare_pass", sym) is None:
                self.findings.append(
                    Finding(
                        self.relpath,
                        node.lineno,
                        "bare_pass",
                        "Bare `pass` in executable branch is forbidden; use explicit behavior (raise, return, or logged no-op).",
                    )
                )
        self.generic_visit(node)

    @staticmethod
    def _is_simple_namespace_ctor(node: ast.AST) -> bool:
        return (isinstance(node, ast.Name) and node.id == "SimpleNamespace") or (
            isinstance(node, ast.Attribute) and node.attr == "SimpleNamespace"
        )

    def _is_executable_pass(self, node: ast.Pass) -> bool:
        if not self._parent_stack:
            return False
        parent = self._parent_stack[-1]
        branch_nodes = (ast.If, ast.For, ast.AsyncFor, ast.While, ast.With, ast.AsyncWith, ast.ExceptHandler, ast.Try)
        if not isinstance(parent, branch_nodes):
            return False
        for ancestor in reversed(self._parent_stack[:-1]):
            if isinstance(ancestor, (ast.FunctionDef, ast.AsyncFunctionDef)):
                return False
        return True


def check_file(path: Path) -> list[Finding]:
    abs_path = path if path.is_absolute() else (ROOT / path)
    rel = abs_path.relative_to(ROOT).as_posix()
    try:
        tree = ast.parse(abs_path.read_text(encoding="utf-8"))
    except SyntaxError as exc:
        return [Finding(rel, exc.lineno or 1, "syntax_error", f"Failed to parse file: {exc.msg}")]
    return _ModuleAnalyzer(rel, tree).run()


def run(runtime_paths: Iterable[str] | None = None) -> list[Finding]:
    files = _iter_tracked_python_files(runtime_paths or DEFAULT_RUNTIME_PATHS)
    findings: list[Finding] = []
    for path in files:
        findings.extend(check_file(path))
    return sorted(findings, key=lambda f: (f.path, f.line, f.code))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="*",
        default=list(DEFAULT_RUNTIME_PATHS),
        help="Runtime paths to scan (default: repository root).",
    )
    args = parser.parse_args()

    findings = run(args.paths)
    if not findings:
        print("check_invalid_stubs: no invalid stubs found")
        return 0

    print("check_invalid_stubs: found invalid stubs")
    for f in findings:
        print(f"{f.path}:{f.line}: [{f.code}] {f.message}")
    print("\nAllowlist entries must be explicit in scripts/check_invalid_stubs.py with a safety rationale.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
