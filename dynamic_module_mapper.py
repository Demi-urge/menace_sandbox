from __future__ import annotations

"""Dynamic module graph discovery and clustering utilities."""

import json
from pathlib import Path
from typing import Dict, Iterable, Mapping

import ast

try:  # networkx is an optional dependency during some tests
    import networkx as nx
except Exception:  # pragma: no cover - fallback stub
    nx = None  # type: ignore


# ---------------------------------------------------------------------------

def _iter_py_files(root: Path) -> Iterable[Path]:
    """Yield all ``.py`` files under ``root``."""
    for path in root.rglob("*.py"):
        if path.is_file():
            yield path


def _tokenize_doc(doc: str) -> set[str]:
    return {t.lower() for t in ast.literal_eval(repr(doc)).split() if t.isidentifier()}


# ---------------------------------------------------------------------------

def build_import_graph(repo_path: str | Path) -> "nx.DiGraph":
    """Return a directed graph of imports and cross-module calls."""
    if nx is None:  # pragma: no cover - networkx missing
        raise RuntimeError("networkx is required to build the module graph")

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

def _cluster_graph(
    graph: "nx.DiGraph",
    *,
    algorithm: str = "greedy",
    threshold: float = 0.1,
    use_semantic: bool = False,
    root: Path | None = None,
) -> dict[str, int]:
    """Return module → cluster assignments."""
    if nx is None:  # pragma: no cover - networkx missing
        raise RuntimeError("networkx is required to cluster modules")

    undirected = graph.to_undirected()

    if use_semantic and root is not None:
        docs: Dict[str, set[str]] = {}
        for node in graph.nodes:
            path = root / (node + ".py")
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
                    undirected.add_edge(a, b)

    mapping: Dict[str, int] = {}
    try:
        import hdbscan  # type: ignore
    except Exception:  # pragma: no cover - optional dependency
        hdbscan = None  # type: ignore

    if hdbscan is not None and graph.number_of_nodes() > 1:
        import numpy as np

        data = nx.to_numpy_array(graph)
        clusterer = hdbscan.HDBSCAN(min_cluster_size=2)
        labels = clusterer.fit_predict(data)
        if len(set(labels)) > 1 and set(labels) != {-1}:
            nodes = list(graph.nodes())
            for node, label in zip(nodes, labels, strict=False):
                mapping[node] = int(label)
            return mapping

    if algorithm == "label":
        from networkx.algorithms.community import asyn_lpa_communities

        communities = list(asyn_lpa_communities(undirected))
    else:
        from networkx.algorithms.community import greedy_modularity_communities

        communities = list(greedy_modularity_communities(undirected))

    for idx, comm in enumerate(communities):
        for mod in comm:
            mapping[mod] = idx

    return mapping


def discover_module_groups(
    repo_path: str | Path,
    *,
    algorithm: str = "greedy",
    threshold: float = 0.1,
    use_semantic: bool = False,
) -> dict[str, list[str]]:
    """Analyse ``repo_path`` and return groups of related modules."""
    mapping = _cluster_graph(
        build_import_graph(repo_path),
        algorithm=algorithm,
        threshold=threshold,
        use_semantic=use_semantic,
        root=Path(repo_path),
    )
    groups: Dict[int, list[str]] = {}
    for mod, cid in mapping.items():
        groups.setdefault(cid, []).append(mod)
    return {str(k): sorted(v) for k, v in groups.items()}


# ---------------------------------------------------------------------------

def build_module_map(
    repo_path: str | Path,
    *,
    algorithm: str = "greedy",
    threshold: float = 0.1,
    use_semantic: bool = False,
) -> dict[str, int]:
    """Build and persist a module grouping map under ``sandbox_data``."""
    root = Path(repo_path)
    mapping = _cluster_graph(
        build_import_graph(root),
        algorithm=algorithm,
        threshold=threshold,
        use_semantic=use_semantic,
        root=root,
    )
    out = root / "sandbox_data" / "module_map.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(mapping, fh, indent=2)
    return mapping


def main(args: list[str] | None = None) -> None:
    """CLI entry point for generating the module map."""
    import argparse

    parser = argparse.ArgumentParser(description="Discover module groups")
    parser.add_argument("repo", nargs="?", default=".", help="Repository root")
    parser.add_argument("--algorithm", default="greedy", choices=["greedy", "label"])
    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument("--semantic", action="store_true", help="Use docstring similarity")
    opts = parser.parse_args(args)

    build_module_map(
        opts.repo,
        algorithm=opts.algorithm,
        threshold=opts.threshold,
        use_semantic=opts.semantic,
    )


if __name__ == "__main__":  # pragma: no cover
    main()


__all__ = ["build_import_graph", "build_module_map", "discover_module_groups"]

