from __future__ import annotations

"""Helpers for analysing orphan modules before integration."""

from pathlib import Path
from typing import Any, Dict, Iterable, Literal, Tuple, Protocol

import ast

from module_graph_analyzer import build_import_graph

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

    The function counts top level functions and call expressions, records
    whether a module docstring is present and, when :mod:`radon` is available,
    determines the maximum cyclomatic complexity across all functions.  When
    :mod:`radon` is missing a lightweight approximation based on the number of
    branching statements is used instead.
    """

    metrics: Dict[str, Any] = {
        "functions": 0,
        "calls": 0,
        "docstring": False,
        "complexity": 0,
    }
    try:
        text = module_path.read_text(encoding="utf-8")
        tree = ast.parse(text)
        metrics["functions"] = sum(isinstance(n, ast.FunctionDef) for n in ast.walk(tree))
        metrics["calls"] = sum(isinstance(n, ast.Call) for n in ast.walk(tree))
        metrics["docstring"] = ast.get_docstring(tree) is not None
        if cc_visit is not None:
            try:
                blocks = cc_visit(text)
                if blocks:
                    metrics["complexity"] = max(b.complexity for b in blocks)
            except Exception:  # pragma: no cover - best effort
                pass
        else:  # pragma: no cover - executed when radon not installed
            metrics["complexity"] = sum(
                isinstance(n, (ast.If, ast.For, ast.While, ast.Try, ast.With, ast.BoolOp))
                for n in ast.walk(tree)
            )
    except Exception:  # pragma: no cover - best effort
        pass
    return metrics


class Classifier(Protocol):
    """Callable classifying ``module_path`` using ``metrics``."""

    def __call__(
        self, module_path: Path, metrics: Dict[str, Any]
    ) -> Literal["legacy", "redundant"] | None: ...


def _legacy_classifier(module_path: Path, metrics: Dict[str, Any]) -> Literal["legacy"] | None:
    if detect_legacy_patterns(module_path):
        return "legacy"
    return None


def _redundant_classifier(
    module_path: Path, metrics: Dict[str, Any]
) -> Literal["redundant"] | None:
    try:
        root = module_path.parent
        graph = build_import_graph(root)
        if module_path.name == "__init__.py":
            mod = module_path.parent.name
        else:
            mod = module_path.stem
        if mod not in graph.nodes or graph.degree(mod) == 0:
            if (
                metrics.get("functions", 0) <= 1
                and metrics.get("complexity", 0) <= 5
                and metrics.get("calls", 0) == 0
                and not metrics.get("docstring")
            ):
                return "redundant"
    except Exception:  # pragma: no cover - best effort
        pass
    return None


DEFAULT_CLASSIFIERS: Tuple[Classifier, ...] = (
    _legacy_classifier,
    _redundant_classifier,
)


def classify_module(
    module_path: Path,
    *,
    include_meta: bool = False,
    classifiers: Iterable[Classifier] | None = None,
) -> Literal["legacy", "redundant", "candidate"] | Tuple[
    Literal["legacy", "redundant", "candidate"], Dict[str, Any]
]:
    """Classify ``module_path`` as ``legacy``, ``redundant`` or ``candidate``.

    The function delegates to a sequence of classifier strategies.  Each
    classifier receives ``module_path`` and pre-computed ``metrics`` and may
    return a specific classification or ``None`` to defer to the next
    strategy.  When all classifiers defer the module is considered a
    ``candidate``.  ``metrics`` always includes ``functions``, ``calls``,
    ``docstring`` and ``complexity``.
    """

    metrics = _static_metrics(module_path)
    result: Literal["legacy", "redundant"] | None = None
    for classifier in classifiers or DEFAULT_CLASSIFIERS:
        try:
            result = classifier(module_path, metrics)
        except Exception:  # pragma: no cover - best effort
            result = None
        if result is not None:
            break
    cls: Literal["legacy", "redundant", "candidate"] = result or "candidate"

    if include_meta:
        return cls, metrics
    return cls


def analyze_redundancy(
    module_path: Path, *, classifiers: Iterable[Classifier] | None = None
) -> bool:
    """Backward compatible wrapper returning ``True`` for non-candidates."""

    return classify_module(module_path, classifiers=classifiers) != "candidate"


__all__ = [
    "classify_module",
    "detect_legacy_patterns",
    "analyze_redundancy",
    "Classifier",
]
