#!/usr/bin/env python3
"""Fail if SentenceTransformer.encode is used directly.

This check scans Python modules for direct calls to ``.encode`` on
``SentenceTransformer`` instances.  Embedding operations must go through the
``governed_embed`` helper which applies safety checks.  The script exits with a
non-zero status when ungoverned calls are found so CI can flag violations.
"""
from __future__ import annotations

from pathlib import Path
import re
import sys

from dynamic_path_router import get_project_root, resolve_path

# Determine repository root once for use throughout the script
REPO_ROOT = get_project_root()

# Match ``SentenceTransformer(...).encode`` or ``SentenceTransformer.encode``
# directly.  The ``.*`` is non-greedy so multi-line constructions are handled.
DIRECT_ST_ENCODE = re.compile(
    r"SentenceTransformer\s*(?:\([^)]*\)\s*)?\.encode\(", re.DOTALL
)

# Match ``something.encode`` but ignore ``tokenizer.encode`` which is valid
# for token counting.  Additional exclusions can be added as needed.
ENCODE_CALL = re.compile(r"\b(?!tokenizer\.)[A-Za-z_][A-Za-z0-9_]*\.encode\(")


def main() -> int:
    wrapper = resolve_path("governed_embeddings.py").name
    offenders: list[str] = []
    this_file = Path(__file__).resolve()
    for path in REPO_ROOT.rglob("*.py"):
        # Ignore tests, the governed_embeddings module that implements the
        # wrapper around SentenceTransformer.encode, and this check script
        # itself.
        if (
            "tests" in path.parts
            or path.name == wrapper
            or path.resolve() == this_file
        ):
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")

        # Find direct ``SentenceTransformer.encode`` calls including inline
        # instantiations like ``SentenceTransformer("model").encode(...)``.
        flagged_lines: set[int] = set()
        for match in DIRECT_ST_ENCODE.finditer(text):
            lineno = text[: match.start()].count("\n") + 1
            line = text.splitlines()[lineno - 1].strip()
            offenders.append(f"{path.relative_to(REPO_ROOT)}:{lineno}:{line}")
            flagged_lines.add(lineno)

        if "SentenceTransformer" not in text:
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            if lineno in flagged_lines or "encode(" not in line:
                continue
            if ENCODE_CALL.search(line):
                offenders.append(f"{path.relative_to(REPO_ROOT)}:{lineno}:{line.strip()}")
    if offenders:
        print("Ungoverned embedding calls detected:")
        for off in offenders:
            print(off)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
