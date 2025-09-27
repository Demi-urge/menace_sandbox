"""Shared constants and helpers for Stack dataset configuration."""
from __future__ import annotations

from typing import Iterable, List

STACK_LANGUAGE_ALLOWLIST: frozenset[str] = frozenset(
    {
        "assembly",
        "bash",
        "c",
        "c#",
        "c++",
        "clojure",
        "cmake",
        "cobol",
        "coffeescript",
        "cuda",
        "dart",
        "elixir",
        "elm",
        "erlang",
        "fortran",
        "fsharp",
        "go",
        "groovy",
        "haskell",
        "java",
        "javascript",
        "julia",
        "kotlin",
        "lua",
        "matlab",
        "nim",
        "objective-c",
        "ocaml",
        "perl",
        "php",
        "powershell",
        "python",
        "r",
        "ruby",
        "rust",
        "scala",
        "shell",
        "solidity",
        "sql",
        "swift",
        "typescript",
        "vb",
        "zig",
    }
)


def normalise_stack_languages(value: Iterable[str] | str | None) -> List[str]:
    """Return a normalised list of Stack dataset language identifiers."""

    if value is None:
        return []
    if isinstance(value, str):
        candidates = [part.strip() for part in value.split(",")]
    else:
        candidates = [str(part).strip() for part in value]
    seen: list[str] = []
    for candidate in candidates:
        if not candidate:
            continue
        lowered = candidate.lower()
        if lowered not in seen:
            seen.append(lowered)
    return seen
