from __future__ import annotations

"""Build and cluster a module graph across the repository."""

import ast
import json
from pathlib import Path
from typing import Dict

import networkx as nx


# ---------------------------------------------------------------------------

def _iter_py_files(root: Path) -> list[Path]:
    """Return all Python files under ``root``."""
    return [p for p in root.rglob("*.py") if p.is_file()]


def generate_module_graph(repo_path: str) -> nx.DiGraph:
    """Return a dependency graph including imports and call edges."""
    root = Path(repo_path)
    graph = nx.DiGraph()
    modules: Dict[str, Path] = {}
    for file in _iter_py_files(root):
        mod = file.relative_to(root).with_suffix("").as_posix()
        modules[mod] = file
        graph.add_node(mod)

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
                    if alias.name in modules:
                        graph.add_edge(mod, alias.name)
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
                    target_mod = imports.get(func.value.id)
                    if target_mod and target_mod in modules:
                        graph.add_edge(mod, target_mod)
                elif isinstance(func, ast.Name):
                    target_mod = imports.get(func.id)
                    if target_mod and target_mod in modules:
                        graph.add_edge(mod, target_mod)
    return graph


# ---------------------------------------------------------------------------

def discover_module_groups(graph: nx.DiGraph) -> dict[str, int]:
    """Cluster modules using community detection."""
    try:
        import hdbscan  # type: ignore
    except Exception:
        hdbscan = None  # type: ignore

    if hdbscan is not None and graph.number_of_nodes() > 1:
        import numpy as np

        data = nx.to_numpy_array(graph)
        clusterer = hdbscan.HDBSCAN(min_cluster_size=2)
        labels = clusterer.fit_predict(data)
        if len(set(labels)) > 1 and set(labels) != {-1}:
            nodes = list(graph.nodes())
            mapping: dict[str, int] = {}
            for node, label in zip(nodes, labels, strict=False):
                mapping[node] = int(label)
            return mapping

    from networkx.algorithms.community import greedy_modularity_communities

    communities = greedy_modularity_communities(graph.to_undirected())
    mapping: dict[str, int] = {}
    for idx, comm in enumerate(communities):
        for mod in comm:
            mapping[mod] = idx
    return mapping


# ---------------------------------------------------------------------------

def build_module_map(repo_path: str) -> dict[str, int]:
    """Build and persist a module grouping map."""
    root = Path(repo_path)
    graph = generate_module_graph(root)
    mapping = discover_module_groups(graph)
    out = root / "sandbox_data" / "module_map.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(mapping, fh, indent=2)
    return mapping


__all__ = ["generate_module_graph", "discover_module_groups", "build_module_map"]

