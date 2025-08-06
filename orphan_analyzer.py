from __future__ import annotations

"""Helpers for analysing orphan modules before integration."""

from pathlib import Path
from typing import Literal
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


def classify_module(module_path: Path) -> Literal["legacy", "redundant", "candidate"]:
    """Classify ``module_path`` as ``legacy``, ``redundant`` or ``candidate``.

    A module is labelled ``legacy`` when :func:`detect_legacy_patterns` finds
    obvious markers in the source text. If no legacy markers are detected we
    build an import graph for the module's directory. Modules that either do
    not appear in this graph or have no relations to other modules are deemed
    ``redundant``. Any analysis failures simply fall back to ``candidate`` to
    keep the behaviour deterministic.
    """

    try:
        if detect_legacy_patterns(module_path):
            return "legacy"

        root = module_path.parent
        graph = build_import_graph(root)
        if module_path.name == "__init__.py":
            mod = module_path.parent.name
        else:
            mod = module_path.stem
        if mod not in graph.nodes or graph.degree(mod) == 0:
            return "redundant"
    except Exception:
        pass
    return "candidate"


def analyze_redundancy(module_path: Path) -> bool:
    """Backward compatible wrapper returning ``True`` for non-candidates."""

    return classify_module(module_path) != "candidate"


__all__ = ["classify_module", "detect_legacy_patterns", "analyze_redundancy"]
