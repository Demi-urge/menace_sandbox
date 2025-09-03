from __future__ import annotations

"""Utilities for building and clustering module dependency graphs."""

import ast
import json
import logging
from pathlib import Path
from typing import Dict, Iterable, Mapping, Tuple, List

import networkx as nx
from dynamic_path_router import resolve_path


# ---------------------------------------------------------------------------

def _iter_py_files(root: str | Path, ignore: Iterable[str] | None = None) -> Iterable[Path]:
    """Yield all ``.py`` files under ``root`` skipping ignored paths."""
    root = Path(resolve_path(root))
    ignore = set(ignore or [])
    for path in root.rglob("*.py"):
        if not path.is_file():
            continue
        rel = path.relative_to(root)
        if any(rel.match(pat) or pat in rel.parts for pat in ignore):
            continue
        yield path


def build_module_graph(
    repo_path: str | Path,
    *,
    ignore: Iterable[str] | None = None,
    failures: List[Tuple[Path, Exception]] | None = None,
) -> nx.DiGraph:
    """Return a graph with import and call edges between modules.

    Parameters
    ----------
    repo_path:
        Root of the repository to scan.
    ignore:
        Optional iterable of glob-style patterns to skip.
    failures:
        If provided, parse failures will be appended as ``(path, exception)``.
        When ``None`` (default) parse failures raise exceptions.
    """

    logger = logging.getLogger(__name__)

    root = Path(resolve_path(repo_path))
    graph = nx.DiGraph()
    modules: Dict[str, Path] = {}
    for file in _iter_py_files(root, ignore=ignore):
        mod = file.relative_to(root).with_suffix("").as_posix()
        modules[mod] = file
        graph.add_node(mod)

    for mod, file in modules.items():
        try:
            source = file.read_text()
        except OSError as exc:
            logger.error("Failed to read %s: %s", file, exc)
            if failures is not None:
                failures.append((file, exc))
                continue
            raise
        try:
            tree = ast.parse(source)
        except SyntaxError as exc:
            logger.error("Syntax error parsing %s: %s", file, exc)
            if failures is not None:
                failures.append((file, exc))
                continue
            raise
        except Exception as exc:
            logger.error("Failed to parse %s: %s", file, exc)
            if failures is not None:
                failures.append((file, exc))
                continue
            raise
        imports: Dict[str, str] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    name = alias.asname or alias.name.split(".")[0]
                    imports[name] = alias.name
                    target = alias.name
                    if target in modules:
                        graph.add_edge(mod, target)
            elif isinstance(node, ast.ImportFrom):
                pkg_parts = mod.split("/")[:-1]
                if node.level > 1:
                    pkg_parts = pkg_parts[: -(node.level - 1)]
                module = node.module.split(".") if node.module else []
                package_parts = pkg_parts + module
                target_pkg = "/".join(package_parts)
                for alias in node.names:
                    name = alias.asname or alias.name
                    imports[name] = "/".join(package_parts + [alias.name])
                if target_pkg in modules:
                    graph.add_edge(mod, target_pkg)
            elif isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
                    base = func.value.id
                    target_mod = imports.get(base)
                    if target_mod and target_mod in modules:
                        graph.add_edge(mod, target_mod)
                elif isinstance(func, ast.Name):
                    target_mod = imports.get(func.id)
                    if target_mod and target_mod in modules:
                        graph.add_edge(mod, target_mod)
    return graph


# ---------------------------------------------------------------------------

def cluster_modules(graph: nx.DiGraph) -> Dict[int, list[str]]:
    """Group modules using community detection or HDBSCAN."""
    try:
        import hdbscan  # type: ignore
    except Exception:  # pragma: no cover - optional dependency
        hdbscan = None  # type: ignore

    if hdbscan is not None and graph.number_of_nodes() > 1:
        import numpy as np

        data = nx.to_numpy_array(graph)
        clusterer = hdbscan.HDBSCAN(min_cluster_size=2)
        labels = clusterer.fit_predict(data)
        # If clustering produced a single label or all noise, fall back
        if len(set(labels)) > 1 and set(labels) != {-1}:
            mapping: Dict[int, list[str]] = {}
            nodes = list(graph.nodes())
            for node, label in zip(nodes, labels, strict=False):
                mapping.setdefault(int(label), []).append(node)
            return mapping

    from networkx.algorithms.community import greedy_modularity_communities

    communities = greedy_modularity_communities(graph.to_undirected())
    mapping = {i: sorted(list(c)) for i, c in enumerate(communities)}
    return mapping


# ---------------------------------------------------------------------------

def save_module_map(mapping: Mapping, path: str | Path) -> None:
    """Save ``mapping`` to ``path`` as JSON."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(mapping, fh, indent=2)


__all__ = ["build_module_graph", "cluster_modules", "save_module_map"]
