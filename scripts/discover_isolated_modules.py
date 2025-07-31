#!/usr/bin/env python3
"""Discover modules not referenced by tests or other modules."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, List

from scripts.find_orphan_modules import find_orphan_modules

try:
    from sandbox_runner import discover_orphan_modules as _discover_import_orphans
except Exception:  # pragma: no cover - sandbox_runner may not be available
    _discover_import_orphans = None


def discover_isolated_modules(base_dir: str | Path) -> List[str]:
    """Return relative paths of isolated Python modules under *base_dir*."""
    root = Path(base_dir).resolve()

    modules = {str(p) for p in find_orphan_modules(root, recursive=False)}

    if _discover_import_orphans is not None:
        try:
            names = _discover_import_orphans(str(root), recursive=False)
        except Exception:  # pragma: no cover - best effort
            names = []
        for name in names:
            path = root / (name.replace(".", os.sep) + ".py")
            if path.exists():
                modules.add(str(path.relative_to(root)))
    return sorted(modules)


if __name__ == "__main__":  # pragma: no cover - simple CLI
    import json
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", default=".", help="Repository root")
    args = parser.parse_args()
    res = discover_isolated_modules(Path(args.path))
    print(json.dumps(res, indent=2))

