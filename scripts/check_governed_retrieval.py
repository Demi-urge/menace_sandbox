#!/usr/bin/env python3
"""Fail if retrieval results are used without governance.

This check scans Python modules for direct calls to ``.retrieve`` which may
return raw text from external systems.  Such calls must route results through
``governed_retrieval.govern_retrieval`` or use the high level
``vector_service.retriever.Retriever`` class.  The script exits with a non-zero
status when ungoverned calls are found so CI can flag violations.
"""
from __future__ import annotations

from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Match ``something.retrieve(``
RETRIEVE_CALL = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\.retrieve\(")


def main() -> int:
    from dynamic_path_router import get_project_root

    root = get_project_root()
    offenders: list[str] = []
    for path in root.rglob("*.py"):
        if (
            "tests" in path.parts
            or "docs" in path.parts
            or path.name in {
                "governed_retrieval.py",
                "check_governed_embeddings.py",
                "check_governed_retrieval.py",
                "universal_retriever.py",
            }
            or path.parts[-2:] == ("vector_service", "retriever.py")
        ):
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if "retrieve(" not in text:
            continue
        # Skip files that explicitly handle governance
        if "govern_retrieval" in text or "Retriever" in text:
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            if "retrieve(" not in line:
                continue
            if RETRIEVE_CALL.search(line):
                if any(x in line for x in ("self.retrieve(", "cls.retrieve(", "super().retrieve(")):
                    continue
                offenders.append(f"{path.relative_to(root)}:{lineno}:{line.strip()}")
    if offenders:
        print("Ungoverned retrieval calls detected:")
        for off in offenders:
            print(off)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
