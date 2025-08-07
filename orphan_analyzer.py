from __future__ import annotations

"""Helpers for analysing orphan modules before integration."""

from pathlib import Path
from typing import Any, Dict, Literal, Tuple
from module_graph_analyzer import build_import_graph
import ast

try:  # optional dependency
    from radon.complexity import cc_visit  # type: ignore
except Exception:  # pragma: no cover - radon missing
    cc_visit = None  # type: ignore


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


def _static_metrics(module_path: Path) -> Dict[str, Any]:
    """Return basic static metrics for ``module_path``.

    The function counts top level functions, records whether a module
    docstring is present and, when :mod:`radon` is available, determines the
    maximum cyclomatic complexity across all functions.
    """

    metrics: Dict[str, Any] = {"functions": 0, "docstring": False, "complexity": 0}
    try:
        tree = ast.parse(module_path.read_text(encoding="utf-8"))
        metrics["functions"] = sum(isinstance(n, ast.FunctionDef) for n in ast.walk(tree))
        metrics["docstring"] = ast.get_docstring(tree) is not None
        if cc_visit is not None:
            try:
                blocks = cc_visit(module_path.read_text(encoding="utf-8"))
                if blocks:
                    metrics["complexity"] = max(b.complexity for b in blocks)
            except Exception:  # pragma: no cover - best effort
                pass
    except Exception:  # pragma: no cover - best effort
        pass
    return metrics


def classify_module(
    module_path: Path, *, include_meta: bool = False
) -> Literal["legacy", "redundant", "candidate"] | Tuple[
    Literal["legacy", "redundant", "candidate"], Dict[str, Any]
]:
    """Classify ``module_path`` as ``legacy``, ``redundant`` or ``candidate``.

    In addition to the previous import-graph based heuristics this variant also
    considers simple static analysis signals: *function count*, *cyclomatic
    complexity* and whether the module defines a top-level docstring. These
    metrics are returned when ``include_meta`` is true so callers can persist
    them alongside the classification.
    """

    metrics = _static_metrics(module_path)
    cls: Literal["legacy", "redundant", "candidate"] = "candidate"
    try:
        if detect_legacy_patterns(module_path):
            cls = "legacy"
        else:
            root = module_path.parent
            graph = build_import_graph(root)
            if module_path.name == "__init__.py":
                mod = module_path.parent.name
            else:
                mod = module_path.stem
            if mod not in graph.nodes or graph.degree(mod) == 0:
                # module is isolated – inspect static metrics to distinguish
                if (
                    metrics["functions"] <= 1
                    and metrics.get("complexity", 0) <= 5
                    and not metrics["docstring"]
                ):
                    cls = "redundant"
                else:
                    cls = "candidate"
    except Exception:  # pragma: no cover - best effort
        cls = "candidate"

    if include_meta:
        return cls, metrics
    return cls


def analyze_redundancy(module_path: Path) -> bool:
    """Backward compatible wrapper returning ``True`` for non-candidates."""

    return classify_module(module_path) != "candidate"


__all__ = ["classify_module", "detect_legacy_patterns", "analyze_redundancy"]
