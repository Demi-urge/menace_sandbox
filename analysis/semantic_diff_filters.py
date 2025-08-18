"""Semantic filters for detecting unsafe constructs in diff hunks.

This module provides lightweight AST-based scanning utilities used by
``codebase_diff_checker`` to flag risky patterns beyond simple keyword
matching. It focuses on identifying the following categories:

* Direct execution via ``eval`` or ``exec``.
* Weak cryptographic hashes such as ``md5`` or ``sha1``.
* Obvious network related calls from modules like ``socket`` or
  ``requests``.
"""

from __future__ import annotations

import ast
from typing import Iterable, List, Tuple

# Mapping of fully qualified call names to messages.
_WEAK_HASH_CALLS = {
    "hashlib.md5": "weak hash function md5",
    "hashlib.sha1": "weak hash function sha1",
}

_NETWORK_PREFIXES: Iterable[str] = (
    "socket.",
    "requests.",
    "urllib.",
    "urllib3.",
    "http.client.",
)

_WEAK_HASH_NAMES = {"md5", "sha1"}


def _call_name(node: ast.Call) -> str:
    """Return dotted name for *node* or empty string."""
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    parts: List[str] = []
    while isinstance(func, ast.Attribute):
        parts.append(func.attr)
        func = func.value
    if isinstance(func, ast.Name):
        parts.append(func.id)
    return ".".join(reversed(parts))


def find_unsafe_nodes(code: str) -> List[Tuple[int, str]]:
    """Return ``(line, message)`` for unsafe constructs in ``code``.

    ``code`` may contain partial snippets; parse errors are ignored and
    result in no matches.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []

    results: List[Tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = _call_name(node)
            if name in {"eval", "exec"}:
                results.append((node.lineno, f"use of {name}"))
            msg = _WEAK_HASH_CALLS.get(name)
            if msg:
                results.append((node.lineno, msg))
            if name == "hashlib.new" and node.args:
                arg = node.args[0]
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    algo = arg.value.lower()
                    if algo in _WEAK_HASH_NAMES:
                        results.append((node.lineno, f"weak hash function {algo}"))
            for prefix in _NETWORK_PREFIXES:
                if name.startswith(prefix):
                    results.append((node.lineno, f"network call via {name}"))
    return results


__all__ = ["find_unsafe_nodes"]
