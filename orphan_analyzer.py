from __future__ import annotations

"""Helpers for analysing orphan modules before integration."""

from pathlib import Path
from module_graph_analyzer import build_import_graph


LEGACY_MARKERS = {"deprecated", "legacy", "missing_reference"}


def detect_legacy_patterns(module_path: Path) -> bool:
    """Return ``True`` if ``module_path`` contains obvious legacy patterns.

    The check is intentionally lightweight and searches the source text for
    common markers such as ``deprecated`` or ``legacy``.  Errors while reading
    the file simply result in ``False``.
    """
    try:
        text = module_path.read_text(encoding="utf-8", errors="ignore").lower()
    except Exception:
        return False
    return any(marker in text for marker in LEGACY_MARKERS)


def analyze_redundancy(module_path: Path) -> bool:
    """Return ``True`` if ``module_path`` appears redundant or legacy.

    A module is considered redundant when it has no import or call relations
    to other modules within the same directory.  Modules containing clear
    legacy markers (such as the string ``deprecated``) are also treated as
    redundant.  Any analysis failures simply fall back to the legacy pattern
    check, keeping the result deterministic.
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
        redundant = graph.degree(mod) == 0
    except Exception:
        redundant = False
    return redundant or detect_legacy_patterns(module_path)


__all__ = ["analyze_redundancy", "detect_legacy_patterns"]
