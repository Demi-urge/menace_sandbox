from __future__ import annotations

"""Lightweight workflow discovery for auto‑bootstrapping.

This module inspects the Menace codebase for workflow‑oriented modules and
persists simple workflow specifications derived from them. The goal is to seed
the self‑improvement loop with deterministic workflows even when no curated
specifications are present in configuration or the ``WorkflowDB``.

Discovery currently targets Python modules whose filenames begin with
``workflow_``. Each detected module is mapped to a single‑step workflow that
references the module by its dotted path. The specifications are written to the
``workflows`` directory so downstream loaders can consume them without extra
configuration.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping

from workflow_spec import save_spec, to_spec

try:  # pragma: no cover - allow package-relative execution
    from .dynamic_path_router import resolve_path  # type: ignore
except Exception:  # pragma: no cover - fallback when run as a script
    from dynamic_path_router import resolve_path  # type: ignore


DEFAULT_EXCLUDED_DIRS = {
    ".git",
    "sandbox_data",
    "tests",
    "unit_tests",
    "venv",
    "__pycache__",
    "workflows",
}


def _is_excluded(path: Path, excluded_dirs: set[str]) -> bool:
    """Return ``True`` when ``path`` should be skipped during discovery."""

    return any(part in excluded_dirs for part in path.parts)


def _module_name_from_path(base: Path, path: Path) -> str:
    """Convert ``path`` to a dotted module name relative to ``base``."""

    relative = path.relative_to(base).with_suffix("")
    return ".".join(relative.parts)


def discover_workflow_specs(
    base_path: Path | str | None = None,
    *,
    logger: logging.Logger | None = None,
    excluded_dirs: Iterable[str] = DEFAULT_EXCLUDED_DIRS,
) -> list[Mapping[str, object]]:
    """Discover workflow modules and persist derived specifications.

    Parameters
    ----------
    base_path:
        Root of the repository to scan. When omitted, ``resolve_path(".")`` is
        used to align with existing sandbox path resolution.
    logger:
        Optional logger for status messages; defaults to the module logger.
    excluded_dirs:
        Iterable of directory names to ignore during traversal.
    """

    log = logger or logging.getLogger(__name__)
    root = Path(base_path) if base_path is not None else Path(resolve_path("."))
    exclusions = set(excluded_dirs)

    discovered: list[Mapping[str, object]] = []
    for path in root.rglob("workflow_*.py"):
        if path.name == "__init__.py" or _is_excluded(path, exclusions):
            continue

        try:
            module_name = _module_name_from_path(root, path)
        except ValueError:  # pragma: no cover - defensive path handling
            log.debug("skipping non-relative workflow module", extra={"path": str(path)})
            continue

        spec = to_spec(
            [
                {
                    "module": module_name,
                    "inputs": [],
                    "outputs": [],
                    "files": [],
                    "globals": [],
                }
            ]
        )
        spec["workflow"] = [module_name]
        spec["metadata"] = {
            "workflow_id": module_name,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "parent_id": None,
            "mutation_description": "auto-discovered workflow module",
        }

        filename = f"{module_name.replace('.', '_')}.workflow.json"
        out_path = Path("workflows") / filename

        try:
            saved_path = save_spec(spec, out_path)
            with saved_path.open("r", encoding="utf-8") as fh:
                saved_spec = json.load(fh)
            discovered.append(saved_spec)
            log.debug(
                "discovered workflow module", extra={"module": module_name, "path": str(saved_path)}
            )
        except Exception:
            log.exception("failed to persist discovered workflow", extra={"module": module_name})

    return discovered


__all__ = ["discover_workflow_specs"]
