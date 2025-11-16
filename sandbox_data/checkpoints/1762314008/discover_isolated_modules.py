#!/usr/bin/env python3
"""Discover orphaned modules using :mod:`sandbox_runner.orphan_discovery`.

This tiny wrapper simply invokes :func:`sandbox_runner.orphan_discovery.
discover_recursive_orphans` for the given repository.  The helper persists
classification information to ``sandbox_data/orphan_modules.json`` and
``sandbox_data/orphan_classifications.json`` which this script exposes for
manual inspection.  Non‑redundant module paths are returned and printed as a
JSON list.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

from dynamic_path_router import resolve_path

from sandbox_runner.orphan_discovery import discover_recursive_orphans


def discover_isolated_modules(
    base_dir: str | Path, *, recursive: bool = True
) -> list[str]:
    """Return relative paths of orphan modules under *base_dir*.

    ``discover_recursive_orphans`` performs the heavy lifting and caches the
    resulting classification data.  When ``recursive`` is ``False`` only modules
    without orphan parents are returned.
    """

    repo = resolve_path(str(base_dir))
    mapping = discover_recursive_orphans(str(repo))

    if not recursive:
        mapping = {m: info for m, info in mapping.items() if not info.get("parents")}

    paths: list[str] = []
    for module, info in mapping.items():
        if info.get("redundant"):
            continue
        mod_path = Path(*module.split(".")).with_suffix(".py")
        pkg_init = Path(*module.split(".")) / "__init__.py"
        target = mod_path if (repo / mod_path).exists() else pkg_init
        paths.append(target.as_posix())

    return sorted(paths)


def main(argv: Iterable[str] | None = None) -> None:  # pragma: no cover - CLI
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", default=".", help="Repository root")
    parser.add_argument(
        "--no-recursive",
        action="store_false",
        dest="recursive",
        default=True,
        help="Exclude dependencies of isolated modules",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    res = discover_isolated_modules(args.path, recursive=args.recursive)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
