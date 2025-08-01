from __future__ import annotations

"""Helpers for analysing orphan modules before integration."""

from pathlib import Path
from module_graph_analyzer import build_import_graph


def analyze_redundancy(module_path: Path) -> bool:
    """Return ``True`` if ``module_path`` appears redundant.

    A module is considered redundant when it has no import or call relations
    to other modules within the same directory.
    """
    try:
        root = module_path.parent
        graph = build_import_graph(root)
        if module_path.name == "__init__.py":
            mod = module_path.parent.name
        else:
            mod = module_path.stem
        if mod not in graph.nodes:
            return True
        return graph.degree(mod) == 0
    except Exception:
        return False


__all__ = ["analyze_redundancy"]
