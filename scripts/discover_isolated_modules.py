#!/usr/bin/env python3
"""Discover modules not referenced by tests or other modules."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, List

from orphan_analyzer import analyze_redundancy
from scripts.find_orphan_modules import find_orphan_modules

try:
    from sandbox_runner import (
        discover_orphan_modules as _discover_import_orphans,
        discover_recursive_orphans as _discover_recursive_orphans,
    )
except Exception:  # pragma: no cover - sandbox_runner may not be available
    _discover_import_orphans = None
    _discover_recursive_orphans = None


def discover_isolated_modules(base_dir: str | Path, *, recursive: bool = False) -> List[str]:
    """Return relative paths of isolated Python modules under *base_dir*."""
    import json

    root = Path(base_dir).resolve()

    # ------------------------------------------------------------------
    # Determine modules already known from the module map and ignore them
    known: set[str] = set()
    map_file = root / "sandbox_data" / "module_map.json"
    try:
        data = json.loads(map_file.read_text())
    except Exception:
        data = {}
    if isinstance(data, dict):
        mod_section = data.get("modules") if isinstance(data.get("modules"), dict) else data
        if isinstance(mod_section, dict):
            for key in mod_section.keys():
                p = Path(key)
                if p.suffix != ".py":
                    p = p.with_suffix(".py")
                known.add(p.as_posix())

    modules: set[str] = set()

    def _add(path: Path) -> None:
        rel = path.relative_to(root).as_posix()
        if rel in known:
            return
        if analyze_redundancy(path):
            return
        modules.add(rel)

    for p in find_orphan_modules(root, recursive=False):
        _add(root / p)

    names: Iterable[str] = []
    if _discover_import_orphans is not None:
        try:
            names = _discover_import_orphans(str(root), recursive=False)
        except Exception:  # pragma: no cover - best effort
            names = []
        for name in names:
            path = root / (name.replace(".", os.sep) + ".py")
            if path.exists():
                _add(path)

    if recursive and _discover_recursive_orphans is not None:
        try:
            trace = _discover_recursive_orphans(str(root))
        except Exception:  # pragma: no cover - best effort
            trace = {}
        for name in trace:
            path = root / (name.replace(".", os.sep) + ".py")
            if path.exists():
                _add(path)

    cache = root / "sandbox_data" / "orphan_modules.json"
    try:
        cache.parent.mkdir(exist_ok=True)
        cache.write_text(json.dumps(sorted(modules), indent=2))
    except Exception:  # pragma: no cover - best effort
        pass

    return sorted(modules)


if __name__ == "__main__":  # pragma: no cover - simple CLI
    import json
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", default=".", help="Repository root")
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Include dependencies of isolated modules",
    )
    args = parser.parse_args()
    res = discover_isolated_modules(Path(args.path), recursive=args.recursive)
    print(json.dumps(res, indent=2))

