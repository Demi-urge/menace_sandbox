"""Utilities for tracking file and function level changes between code snapshots."""

from __future__ import annotations

import ast
import json
import os
from datetime import datetime
from typing import Dict, List, Tuple


class _DocstringRemover(ast.NodeTransformer):
    """AST transformer that removes docstrings from modules, classes and functions."""

    def visit_Module(self, node: ast.Module) -> ast.Module:  # type: ignore[override]
        self.generic_visit(node)
        if node.body and isinstance(node.body[0], ast.Expr) and isinstance(
            getattr(node.body[0], "value", None), ast.Constant
        ) and isinstance(node.body[0].value.value, str):
            node.body = node.body[1:]
        return node

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.ClassDef:  # type: ignore[override]
        self.generic_visit(node)
        if node.body and isinstance(node.body[0], ast.Expr) and isinstance(
            getattr(node.body[0], "value", None), ast.Constant
        ) and isinstance(node.body[0].value.value, str):
            node.body = node.body[1:]
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:  # type: ignore[override]
        self.generic_visit(node)
        if node.body and isinstance(node.body[0], ast.Expr) and isinstance(
            getattr(node.body[0], "value", None), ast.Constant
        ) and isinstance(node.body[0].value.value, str):
            node.body = node.body[1:]
        return node

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AsyncFunctionDef:  # type: ignore[override]
        self.generic_visit(node)
        if node.body and isinstance(node.body[0], ast.Expr) and isinstance(
            getattr(node.body[0], "value", None), ast.Constant
        ) and isinstance(node.body[0].value.value, str):
            node.body = node.body[1:]
        return node


def _canonical_ast_dump(path: str) -> str:
    """Return a canonical dump of the AST for ``path`` ignoring docstrings."""

    try:
        with open(path, "r", encoding="utf-8") as f:
            src = f.read()
    except FileNotFoundError:
        return ""
    try:
        tree = ast.parse(src)
    except SyntaxError:
        # Fallback: remove blank lines and comments only
        lines = []
        for line in src.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            lines.append(stripped)
        return "\n".join(lines)
    tree = _DocstringRemover().visit(tree)
    ast.fix_missing_locations(tree)
    return ast.dump(tree, include_attributes=False)


def _extract_functions(path: str) -> Dict[str, str]:
    """Return mapping of qualified function names to canonical AST dumps."""

    result: Dict[str, str] = {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            src = f.read()
    except FileNotFoundError:
        return result
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return result
    tree = _DocstringRemover().visit(tree)
    ast.fix_missing_locations(tree)

    def visit(node: ast.AST, parents: List[str]) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                name = ".".join(parents + [child.name])
                result[name] = ast.dump(child, include_attributes=False)
                visit(child, parents + [child.name])
            elif isinstance(child, ast.ClassDef):
                visit(child, parents + [child.name])

    visit(tree, [])
    return result


def track_file_changes(before_dir: str, after_dir: str) -> Dict[str, List[str]]:
    """Return a mapping of added, removed and modified files between two dirs."""

    before_files = {
        os.path.relpath(os.path.join(root, f), start=before_dir)
        for root, _, files in os.walk(before_dir)
        for f in files
        if f.endswith(".py")
    }
    after_files = {
        os.path.relpath(os.path.join(root, f), start=after_dir)
        for root, _, files in os.walk(after_dir)
        for f in files
        if f.endswith(".py")
    }
    added = sorted(after_files - before_files)
    removed = sorted(before_files - after_files)
    common = before_files & after_files

    modified: List[str] = []
    for rel in sorted(common):
        before_dump = _canonical_ast_dump(os.path.join(before_dir, rel))
        after_dump = _canonical_ast_dump(os.path.join(after_dir, rel))
        if before_dump != after_dump:
            modified.append(rel)

    return {"added": added, "removed": removed, "modified": modified}


def track_function_changes(file_before: str | None, file_after: str | None) -> Dict[str, List[str]]:
    """Return function-level changes between two versions of a file."""

    before_funcs = _extract_functions(file_before) if file_before else {}
    after_funcs = _extract_functions(file_after) if file_after else {}

    before_names = set(before_funcs)
    after_names = set(after_funcs)
    added = sorted(after_names - before_names)
    removed = sorted(before_names - after_names)
    modified = sorted(
        name
        for name in before_names & after_names
        if before_funcs[name] != after_funcs[name]
    )
    return {"added": added, "removed": removed, "modified": modified}


def generate_change_report(before_dir: str, after_dir: str, output_path: str) -> None:
    """Compare two snapshots and save a JSON report to ``output_path``."""

    file_changes = track_file_changes(before_dir, after_dir)
    all_files = set(file_changes["added"]) | set(file_changes["removed"]) | set(
        file_changes["modified"]
    )
    report = {
        "generated_at": datetime.utcnow().isoformat(),
        "files": {},
    }
    for rel in sorted(all_files):
        status = (
            "added" if rel in file_changes["added"] else "removed" if rel in file_changes["removed"] else "modified"
        )
        before_path = os.path.join(before_dir, rel) if status != "added" else None
        after_path = os.path.join(after_dir, rel) if status != "removed" else None
        func_changes = track_function_changes(before_path, after_path)
        report["files"][rel] = {
            "status": status,
            "functions": func_changes,
        }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


__all__ = [
    "track_file_changes",
    "track_function_changes",
    "generate_change_report",
]
