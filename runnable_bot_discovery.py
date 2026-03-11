from __future__ import annotations

"""Discovery helpers for runnable bot startup targets.

Canonical definition:
    "Every bot" means every startup worker callable in ``service_supervisor``
    explicitly marked with ``@runnable_bot_worker``.

This marker-based discovery is intentionally independent from
``PRODUCTION_BOT_MANIFEST`` so tests can detect workers that were added but not
registered in the manifest.
"""

import ast
from pathlib import Path


def discover_runnable_startup_callables() -> tuple[str, ...]:
    """Return startup callable names marked as runnable bot workers."""

    supervisor_path = Path(__file__).with_name("service_supervisor.py")
    module = ast.parse(supervisor_path.read_text(encoding="utf-8"))
    discovered: list[str] = []
    for node in module.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Name) and decorator.id == "runnable_bot_worker":
                discovered.append(node.name)
    return tuple(sorted(discovered))


__all__ = ["discover_runnable_startup_callables"]
