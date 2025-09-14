from __future__ import annotations

"""Ensure internalized bots specify an evolution orchestrator.

This script scans all ``*_bot.py`` files for calls to
``internalize_coding_bot``.  For every such call the
``evolution_orchestrator`` keyword argument must be present and may not be
set to ``None``.  Offending calls are printed and the script exits with a
non-zero status so the check can be enforced via pre-commit/CI.
"""

import ast
from pathlib import Path


def _missing_orchestrator(path: Path) -> list[int]:
    """Return line numbers of offending ``internalize_coding_bot`` calls."""

    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except Exception:
        return []

    offenders: list[int] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute):
            func_name = func.attr
        elif isinstance(func, ast.Name):
            func_name = func.id
        else:  # pragma: no cover - defensive
            continue
        if func_name != "internalize_coding_bot":
            continue
        has_kw = False
        for kw in node.keywords:
            if kw.arg == "evolution_orchestrator":
                has_kw = True
                if isinstance(kw.value, ast.Constant) and kw.value.value is None:
                    offenders.append(node.lineno)
                break
        if not has_kw:
            offenders.append(node.lineno)
    return offenders


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    failures: list[tuple[Path, int]] = []
    for path in root.rglob("*_bot.py"):
        if "tests" in path.parts or "unit_tests" in path.parts:
            continue
        for line in _missing_orchestrator(path):
            failures.append((path.relative_to(root), line))
    if failures:
        for p, line in failures:
            print(f"{p}:{line} missing evolution_orchestrator")
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover - entry point
    raise SystemExit(main())
