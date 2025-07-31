# -*- coding: utf-8 -*-
"""Utilities for analysing the module dependency graph."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Dict, Iterable

import networkx as nx


# ---------------------------------------------------------------------------

def _iter_py_files(root: Path, ignore: Iterable[str] | None = None) -> Iterable[Path]:
    """Yield all ``.py`` files under ``root`` skipping ignored paths."""
    ignore = set(ignore or [])
    for path in root.rglob("*.py"):
        if not path.is_file():
            continue
        rel = path.relative_to(root)
        posix = rel.as_posix()
        if any(rel.match(pat) or pat in rel.parts for pat in ignore):
            continue
        yield path


def build_import_graph(
    root: Path | str,
    *,
    ignore: Iterable[str] | None = None,
) -> nx.DiGraph:
    """Return a directed graph of imports and cross-module calls."""
    root = Path(root)
    graph = nx.DiGraph()
    modules: Dict[str, Path] = {}
    for file in _iter_py_files(root, ignore=ignore):
        rel = file.relative_to(root)
        if rel.name == "__init__.py":
            mod = rel.parent.as_posix()
        else:
            mod = rel.with_suffix("").as_posix()
        modules[mod] = file
        graph.add_node(mod)

    def _add_edge(a: str, b: str) -> None:
        """Increment the ``weight`` on edge ``a`` -> ``b``."""
        if graph.has_edge(a, b):
            graph[a][b]["weight"] += 1
        else:
            graph.add_edge(a, b, weight=1)

    for mod, file in modules.items():
        try:
            tree = ast.parse(file.read_text())
        except Exception:
            continue
        imports: Dict[str, str] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    name = alias.asname or alias.name.split(".")[0]
                    imports[name] = alias.name
                    target = alias.name
                    if target in modules:
                        _add_edge(mod, target)
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
                    _add_edge(mod, target_pkg)
            elif isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
                    base = func.value.id
                    target_mod = imports.get(base)
                    if target_mod and target_mod in modules:
                        _add_edge(mod, target_mod)
                elif isinstance(func, ast.Name):
                    target_mod = imports.get(func.id)
                    if target_mod and target_mod in modules:
                        _add_edge(mod, target_mod)
    return graph


# ---------------------------------------------------------------------------

def _tokenize_doc(doc: str) -> set[str]:
    return {t.lower() for t in ast.literal_eval(repr(doc)).split() if t.isidentifier()}


def cluster_modules(
    graph: nx.DiGraph,
    *,
    algorithm: str = "greedy",
    threshold: float = 0.1,
    use_semantic: bool = False,
    root: Path | None = None,
) -> Dict[str, int]:
    """Group modules using community detection."""
    # Combine directed edges into an undirected weighted graph
    undirected = nx.Graph()
    for a, b, data in graph.edges(data=True):
        w = data.get("weight", 1)
        if undirected.has_edge(a, b):
            undirected[a][b]["weight"] += w
        elif undirected.has_edge(b, a):
            undirected[b][a]["weight"] += w
        else:
            undirected.add_edge(a, b, weight=w)
    undirected.add_nodes_from(graph.nodes)
    if use_semantic and root is not None:
        docs: Dict[str, set[str]] = {}
        for node in graph.nodes:
            path = (root / (node + ".py")) if root else Path(node)
            try:
                tree = ast.parse(path.read_text())
            except Exception:
                continue
            doc = ast.get_docstring(tree) or ""
            docs[node] = _tokenize_doc(doc)
        for a in graph.nodes:
            for b in graph.nodes:
                if a >= b:
                    continue
                da, db = docs.get(a, set()), docs.get(b, set())
                if not da or not db:
                    continue
                inter = len(da & db)
                union = len(da | db)
                if union and inter / union >= threshold:
                    if undirected.has_edge(a, b):
                        undirected[a][b]["weight"] += 1
                    elif undirected.has_edge(b, a):
                        undirected[b][a]["weight"] += 1
                    else:
                        undirected.add_edge(a, b, weight=1)

    if algorithm == "label":
        from networkx.algorithms.community import asyn_lpa_communities

        communities = list(asyn_lpa_communities(undirected, weight="weight"))
    else:
        from networkx.algorithms.community import greedy_modularity_communities

        communities = list(greedy_modularity_communities(undirected, weight="weight"))

    mapping: Dict[str, int] = {}
    for idx, comm in enumerate(communities):
        for mod in comm:
            mapping[mod] = idx
    return mapping


__all__ = ["build_import_graph", "cluster_modules"]
