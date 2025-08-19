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

# Match ``something.encode(`` but ignore ``tokenizer.encode`` which is valid
# for token counting.  Additional exclusions can be added as needed.
ENCODE_CALL = re.compile(r"\b(?!tokenizer\.)[A-Za-z_][A-Za-z0-9_]*\.encode\(")


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    offenders: list[str] = []
    this_file = Path(__file__).resolve()
    for path in root.rglob("*.py"):
        # Ignore tests, the governed_embeddings module that implements the
        # wrapper around SentenceTransformer.encode, and this check script
        # itself.
        if (
            "tests" in path.parts
            or path.name == "governed_embeddings.py"
            or path.resolve() == this_file
        ):
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if "SentenceTransformer" not in text:
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            if "encode(" not in line:
                continue
            if ENCODE_CALL.search(line):
                offenders.append(f"{path.relative_to(root)}:{lineno}:{line.strip()}")
    if offenders:
        print("Ungoverned embedding calls detected:")
        for off in offenders:
            print(off)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
