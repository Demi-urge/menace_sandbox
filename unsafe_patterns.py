from __future__ import annotations

"""Simple library of unsafe code patterns for diff scanning."""

import ast
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Iterable, List


@dataclass(frozen=True)
class UnsafePattern:
    """Representation of an unsafe code pattern.

    Attributes
    ----------
    message:
        Human readable description of the pattern.
    call_names:
        Collection of function or attribute names that indicate the pattern
        when present in an AST call expression.
    example:
        Example snippet used for fallback vector similarity matching.
    """

    message: str
    call_names: Iterable[str]
    example: str


# A small catalogue of risky constructs.
UNSAFE_PATTERNS: List[UnsafePattern] = [
    UnsafePattern("use of eval", ["eval"], "eval('data')"),
    UnsafePattern(
        "subprocess with shell",
        ["subprocess.run", "subprocess.call", "subprocess.Popen"],
        "subprocess.run('cmd', shell=True)",
    ),
    UnsafePattern(
        "untrusted pickle load",
        ["pickle.load", "pickle.loads"],
        "pickle.loads(data)",
    ),
    UnsafePattern(
        "yaml load without safe loader",
        ["yaml.load"],
        "yaml.load(data)",
    ),
]


def _extract_call_names(code: str) -> List[str]:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    calls: List[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute):
                base = getattr(func.value, "id", "")
                calls.append(f"{base}.{func.attr}")
            elif isinstance(func, ast.Name):
                calls.append(func.id)
    return calls


def _subprocess_shell_present(code: str) -> bool:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            name = None
            if isinstance(func, ast.Attribute):
                name = f"{getattr(func.value, 'id', '')}.{func.attr}".lower()
            elif isinstance(func, ast.Name):
                name = func.id.lower()
            if name and name.startswith("subprocess"):
                for kw in node.keywords:
                    if (
                        kw.arg == "shell"
                        and isinstance(kw.value, ast.Constant)
                        and bool(kw.value.value)
                    ):
                        return True
    return False


def _vector_match(text: str, pattern: UnsafePattern, threshold: float = 0.8) -> bool:
    return SequenceMatcher(None, text, pattern.example).ratio() >= threshold


def find_matches(text: str) -> List[str]:
    """Return messages for patterns found in *text*.

    The function first attempts AST-based detection using call names and
    special pattern logic (e.g. subprocess with ``shell=True``).  If no
    pattern matches via AST, a fuzzy textual similarity against example
    snippets acts as a lightweight vector similarity heuristic.
    """

    matches: List[str] = []
    call_names = {name.lower() for name in _extract_call_names(text)}
    lowered = text.lower()
    for pat in UNSAFE_PATTERNS:
        pat_calls = {c.lower() for c in pat.call_names}
        if pat.message == "subprocess with shell":
            if _subprocess_shell_present(text):
                matches.append(pat.message)
                continue
        if call_names & pat_calls:
            matches.append(pat.message)
            continue
        if _vector_match(lowered, pat):
            matches.append(pat.message)
    return matches


__all__ = ["UnsafePattern", "UNSAFE_PATTERNS", "find_matches"]
