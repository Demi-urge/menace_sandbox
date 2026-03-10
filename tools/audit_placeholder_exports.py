#!/usr/bin/env python3
"""Audit module-level placeholder exports used as runtime shims.

Phase 1 policy bans module-level ``None``/``object``/``SimpleNamespace`` shims
unless they are explicitly allowlisted private sentinels or optional dependency
handles that are never dereferenced.
"""
from __future__ import annotations

import argparse
import ast
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]

EXCLUDED_PARTS = {".git", ".venv", "venv", "__pycache__", "tests", "test", "fixtures"}
EXCLUDED_FILES = {"conftest.py"}

# Keep this narrowly scoped: only private sentinels/optional dependency placeholders
# that are never called or dereferenced should be allowlisted.
ALLOWLIST_REASONS: dict[tuple[str, str], str] = {
    (
        "coding_bot_interface.py",
        "_ENGINE_IMPORT_ERROR",
    ): "Optional dependency import error cache; only checked/logged and never dereferenced as a service shim.",
    (
        "self_coding_engine.py",
        "_CODE_DATABASE_IMPORT_ERROR",
    ): "Optional dependency import error cache; used to guard fallback paths and re-raise root cause.",
    (
        "orchestrator_loader.py",
        "_BOOTSTRAP_SENTINEL",
    ): "Private bootstrap sentinel handle propagated between loaders and compared against None only.",
    (
        "research_aggregator_bot.py",
        "_BOOTSTRAP_SENTINEL",
    ): "Private bootstrap sentinel for placeholder wiring; never called or attribute-dereferenced.",
}

BASELINE_PATH = ROOT / "tools" / "qa" / "placeholder_audit_baseline.txt"


@dataclass(frozen=True)
class PlaceholderSymbol:
    symbol: str
    placeholder: str
    line: int


@dataclass(frozen=True)
class Finding:
    path: str
    symbol: str
    assignment_line: int
    placeholder: str
    kind: str
    message: str
    usage_lines: tuple[int, ...] = ()


def _iter_python_files(paths: Iterable[str]) -> list[Path]:
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


def _placeholder_kind(value: ast.AST) -> str | None:
    if isinstance(value, ast.Constant) and value.value is None:
        return "None"
    if isinstance(value, ast.Name) and value.id == "object":
        return "object"
    if isinstance(value, ast.Call):
        fn = value.func
        if isinstance(fn, ast.Name) and fn.id == "SimpleNamespace":
            return "SimpleNamespace"
        if isinstance(fn, ast.Attribute) and fn.attr == "SimpleNamespace":
            return "SimpleNamespace"
    return None


class ModuleAudit(ast.NodeVisitor):
    def __init__(self, path: str, tree: ast.Module) -> None:
        self.path = path
        self.tree = tree
        self.placeholders: dict[str, PlaceholderSymbol] = {}
        self.called_lines: dict[str, set[int]] = {}
        self.attr_lines: dict[str, set[int]] = {}

    def run(self) -> list[Finding]:
        self._collect_module_placeholders()
        self.visit(self.tree)
        findings: list[Finding] = []
        for symbol, info in sorted(self.placeholders.items()):
            called = sorted(self.called_lines.get(symbol, set()))
            attrs = sorted(self.attr_lines.get(symbol, set()))
            usage_lines = tuple(sorted(set(called + attrs)))
            allow_reason = ALLOWLIST_REASONS.get((self.path, symbol))

            if usage_lines:
                findings.append(
                    Finding(
                        path=self.path,
                        symbol=symbol,
                        assignment_line=info.line,
                        placeholder=info.placeholder,
                        kind="dereferenced_placeholder",
                        message=(
                            f"Module-level '{symbol}' assigned to {info.placeholder} is called or has attributes "
                            "accessed/assigned; replace with explicit shim class or guard usage."
                        ),
                        usage_lines=usage_lines,
                    )
                )
                continue

            if allow_reason:
                continue

            findings.append(
                Finding(
                    path=self.path,
                    symbol=symbol,
                    assignment_line=info.line,
                    placeholder=info.placeholder,
                    kind="unallowlisted_placeholder",
                    message=(
                        f"Module-level '{symbol}' assigned to {info.placeholder} is not allowlisted. "
                        "Allowlist only validated private sentinels/optional-dependency placeholders that are never dereferenced."
                    ),
                )
            )
        return findings

    def _collect_module_placeholders(self) -> None:
        for node in self.tree.body:
            if isinstance(node, ast.Assign):
                kind = _placeholder_kind(node.value)
                if not kind:
                    continue
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        self.placeholders[target.id] = PlaceholderSymbol(target.id, kind, node.lineno)
            elif isinstance(node, ast.AnnAssign):
                kind = _placeholder_kind(node.value) if node.value is not None else None
                if kind and isinstance(node.target, ast.Name):
                    symbol = node.target.id
                    self.placeholders[symbol] = PlaceholderSymbol(symbol, kind, node.lineno)

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name) and node.func.id in self.placeholders:
            self.called_lines.setdefault(node.func.id, set()).add(node.lineno)
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if isinstance(node.value, ast.Name) and node.value.id in self.placeholders:
            self.attr_lines.setdefault(node.value.id, set()).add(node.lineno)
        self.generic_visit(node)


def check_file(path: Path) -> list[Finding]:
    rel = path.relative_to(ROOT).as_posix()
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError as exc:
        return [
            Finding(
                path=rel,
                symbol="<module>",
                assignment_line=exc.lineno or 1,
                placeholder="n/a",
                kind="syntax_error",
                message=f"Unable to parse file: {exc.msg}",
            )
        ]
    return ModuleAudit(rel, tree).run()


def run(paths: Iterable[str]) -> list[Finding]:
    findings: list[Finding] = []
    for file_path in _iter_python_files(paths):
        findings.extend(check_file(file_path))
    return sorted(findings, key=lambda f: (f.path, f.assignment_line, f.symbol, f.kind))


def _load_baseline() -> set[str]:
    if not BASELINE_PATH.exists():
        return set()
    entries: set[str] = set()
    for raw in BASELINE_PATH.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        entries.add(line)
    return entries


def _finding_key(finding: Finding) -> str:
    return f"{finding.path}:{finding.symbol}:{finding.kind}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", default=["."], help="Optional repository subpaths to scan.")
    parser.add_argument(
        "--enforce-baseline",
        action="store_true",
        help="Fail only on findings that are not listed in tools/qa/placeholder_audit_baseline.txt.",
    )
    parser.add_argument(
        "--write-baseline",
        action="store_true",
        help="Write current findings to tools/qa/placeholder_audit_baseline.txt and exit.",
    )
    args = parser.parse_args()

    findings = run(args.paths)
    if args.write_baseline:
        lines = ["# Baseline for tools/audit_placeholder_exports.py", "# Format: path:symbol:kind"]
        lines.extend(sorted(_finding_key(f) for f in findings))
        BASELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
        BASELINE_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"audit_placeholder_exports: wrote baseline with {len(findings)} findings")
        return 0

    if args.enforce_baseline:
        baseline = _load_baseline()
        findings = [f for f in findings if _finding_key(f) not in baseline]

    if not findings:
        print("audit_placeholder_exports: no violations found")
        return 0

    for f in findings:
        usage = f"; usage lines={','.join(str(line) for line in f.usage_lines)}" if f.usage_lines else ""
        print(
            f"{f.path}:{f.assignment_line}: {f.symbol}: [{f.kind}] {f.message}{usage}"
        )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
