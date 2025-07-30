from __future__ import annotations

"""Dynamic module graph discovery and clustering utilities."""

import ast
import json
from pathlib import Path
from typing import Dict, Iterable, Mapping

try:  # networkx is an optional dependency during some tests
    import networkx as nx
except Exception:  # pragma: no cover - fallback stub
    nx = None  # type: ignore


# ---------------------------------------------------------------------------

def _iter_py_files(root: Path) -> Iterable[Path]:
    """Yield all ``.py`` files below ``root``."""
    for p in root.rglob("*.py"):
        if p.is_file():
            yield p


def build_import_graph(repo_path: str | Path) -> "nx.DiGraph":
    """Return a directed graph of import relationships between modules."""
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
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    target = alias.name.split(".")[0]
                    if target in modules:
                        graph.add_edge(mod, target)
            elif isinstance(node, ast.ImportFrom):
                pkg_parts = mod.split("/")[:-1]
                if node.level > 0:
                    pkg_parts = pkg_parts[: -node.level]
                module = (node.module or "").split(".")
                target = "/".join(pkg_parts + module).rstrip("/")
                if target in modules:
                    graph.add_edge(mod, target)
    return graph


# ---------------------------------------------------------------------------

def _cluster_graph(graph: "nx.DiGraph") -> dict[str, list[str]]:
    """Return communities of ``graph`` using ``networkx`` algorithms."""
    if nx is None:  # pragma: no cover - networkx missing
        raise RuntimeError("networkx is required to cluster modules")

    try:  # prefer greedy modularity
        from networkx.algorithms.community import greedy_modularity_communities

        communities = list(greedy_modularity_communities(graph.to_undirected()))
    except Exception:  # pragma: no cover - extremely small graphs
        communities = [set(c) for c in nx.connected_components(graph.to_undirected())]

    return {str(i): sorted(comm) for i, comm in enumerate(communities)}


def discover_module_groups(repo_path: str | Path) -> dict[str, list[str]]:
    """Analyse ``repo_path`` and return groups of related modules."""
    graph = build_import_graph(repo_path)
    return _cluster_graph(graph)


# ---------------------------------------------------------------------------

def build_module_map(repo_path: str | Path) -> dict[str, list[str]]:
    """Build and persist a module grouping map under ``sandbox_data``."""
    root = Path(repo_path)
    mapping = discover_module_groups(root)
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
    opts = parser.parse_args(args)

    build_module_map(opts.repo)


if __name__ == "__main__":  # pragma: no cover
    main()


__all__ = ["build_import_graph", "discover_module_groups", "build_module_map"]

