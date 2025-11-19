from __future__ import annotations

"""Workflow synthesis helpers for Menace service and component modules."""

import ast
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence

try:  # pragma: no cover - allow running as a package
    from .workflow_discovery import DEFAULT_EXCLUDED_DIRS  # type: ignore
except Exception:  # pragma: no cover - fallback when executed directly
    from workflow_discovery import DEFAULT_EXCLUDED_DIRS  # type: ignore

try:  # pragma: no cover - allow package-relative execution
    from .dynamic_path_router import resolve_path  # type: ignore
except Exception:  # pragma: no cover - fallback when executed directly
    from dynamic_path_router import resolve_path  # type: ignore


COMPONENT_FILE_GLOBS: Sequence[str] = (
    "*_service.py",
    "*_manager.py",
    "*_engine.py",
    "*_orchestrator.py",
    "*_router.py",
    "*_scheduler.py",
)

_COMPONENT_TYPE_SUFFIXES: Mapping[str, str] = {
    "_service.py": "service",
    "_manager.py": "manager",
    "_engine.py": "engine",
    "_orchestrator.py": "orchestrator",
    "_router.py": "router",
    "_scheduler.py": "scheduler",
}


def _should_skip(path: Path, excluded_dirs: Iterable[str]) -> bool:
    excluded = set(excluded_dirs)
    return any(part in excluded for part in path.parts)


def _module_name(base: Path, path: Path) -> str:
    relative = path.relative_to(base).with_suffix("")
    return ".".join(relative.parts)


def _doc_summary(path: Path) -> str | None:
    try:
        module = ast.parse(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    doc = ast.get_docstring(module)
    if not doc:
        return None
    summary = doc.strip().splitlines()[0].strip()
    return summary or None


def discover_component_workflows(
    base_path: Path | str | None = None,
    *,
    logger: logging.Logger | None = None,
    excluded_dirs: Iterable[str] | None = None,
    component_globs: Sequence[str] = COMPONENT_FILE_GLOBS,
) -> list[Mapping[str, object]]:
    """Generate workflow specs for Menace component modules."""

    log = logger or logging.getLogger(__name__)
    root = Path(base_path) if base_path is not None else Path(resolve_path("."))
    exclusions = set(excluded_dirs) if excluded_dirs is not None else set(DEFAULT_EXCLUDED_DIRS)

    specs: list[Mapping[str, object]] = []
    seen_ids: set[str] = set()

    for pattern in component_globs:
        for path in root.rglob(pattern):
            if path.name == "__init__.py" or not path.is_file():
                continue
            if _should_skip(path, exclusions):
                continue

            try:
                module_name = _module_name(root, path)
            except ValueError:
                log.debug("skipping non-relative component", extra={"path": str(path)})
                continue

            workflow_id = f"component::{module_name}"
            if workflow_id in seen_ids:
                continue

            seen_ids.add(workflow_id)
            summary = _doc_summary(path)
            component_type = None
            for suffix, label in _COMPONENT_TYPE_SUFFIXES.items():
                if path.name.endswith(suffix):
                    component_type = label
                    break

            specs.append(
                {
                    "workflow": [module_name],
                    "task_sequence": [module_name],
                    "workflow_id": workflow_id,
                    "source": "component_synthesis",
                    "metadata": {
                        "workflow_id": workflow_id,
                        "capability": path.stem,
                        "component_path": str(path.relative_to(root)),
                        "component_type": component_type,
                        "description": summary,
                        "created_at": datetime.now(timezone.utc).isoformat(),
                    },
                }
            )

    log.debug(
        "component workflow synthesis complete",
        extra={"count": len(specs)},
    )
    return specs


__all__ = ["discover_component_workflows"]
