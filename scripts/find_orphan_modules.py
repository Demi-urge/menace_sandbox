#!/usr/bin/env python3
"""Find Python modules not referenced by tests.

This utility scans all ``.py`` files in a repository (excluding ``tests`` and
optional paths) and checks whether their module names appear anywhere under the
``tests/`` directory. Any modules not referenced are written to
``sandbox_data/orphan_modules.json``.

Pass ``--recursive`` to repeatedly exclude discovered modules and run another
scan until no further orphans are found.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Iterable, List

from dynamic_path_router import resolve_path, get_project_root

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _gather_test_text(tests_root: Path) -> str:
    """Return the concatenated text of all test files."""
    text_parts: List[str] = []
    for path in tests_root.rglob("*.py"):
        try:
            text_parts.append(path.read_text())
        except Exception as exc:  # pragma: no cover - unlikely
            logger.warning("failed to read %s: %s", path, exc)
    return "\n".join(text_parts)


def _should_exclude(rel_path: Path, excludes: Iterable[str]) -> bool:
    path_str = str(rel_path)
    for pattern in excludes:
        if pattern and pattern in path_str:
            return True
    return False


def _find_orphan_modules_once(
    base_dir: Path, *, excludes: Iterable[str] | None = None
) -> List[Path]:
    """Return ``.py`` files whose module names are not referenced in tests."""
    if excludes is None:
        excludes = []
    base_dir = base_dir.resolve()
    tests_root = base_dir / "tests"
    test_text = _gather_test_text(tests_root) if tests_root.exists() else ""
    orphans: List[Path] = []
    for path in base_dir.rglob("*.py"):
        rel = path.relative_to(base_dir)
        if rel.parts[0] == "tests":
            continue
        if _should_exclude(rel, excludes):
            continue
        module_name = rel.stem
        if module_name not in test_text:
            orphans.append(rel)
    return orphans


def find_orphan_modules(
    base_dir: Path,
    *,
    excludes: Iterable[str] | None = None,
    recursive: bool = False,
) -> List[Path]:
    """Return orphan modules under *base_dir*.

    If ``recursive`` is ``True``, repeatedly scan while excluding previously
    discovered modules until no new ones are found.
    """
    if not recursive:
        return _find_orphan_modules_once(base_dir, excludes=excludes)

    collected: List[Path] = []
    excludes = list(excludes or [])
    while True:
        found = _find_orphan_modules_once(base_dir, excludes=excludes)
        new = [p for p in found if p not in collected]
        if not new:
            break
        collected.extend(new)
        excludes.extend(str(p) for p in new)
    return collected


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Find untested Python modules")
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Directory or filename patterns to exclude",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Repeat detection excluding discovered modules until stable",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    root = get_project_root()
    orphan_paths = find_orphan_modules(
        root, excludes=args.exclude, recursive=args.recursive
    )
    output = resolve_path("sandbox_data/orphan_modules.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps([str(p) for p in orphan_paths], indent=2))
    for path in orphan_paths:
        logger.info("orphan module: %s", path)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
