from __future__ import annotations

"""Dynamic module graph discovery and clustering utilities."""

import json
from pathlib import Path
from typing import Dict, Mapping

from module_graph_analyzer import build_import_graph as _build_graph, cluster_modules

try:  # networkx is an optional dependency during some tests
    import networkx as nx
except Exception:  # pragma: no cover - fallback stub
    nx = None  # type: ignore


# ---------------------------------------------------------------------------

def build_import_graph(repo_path: str | Path) -> "nx.DiGraph":
    """Return a directed graph of import relationships between modules."""
    if nx is None:  # pragma: no cover - networkx missing
        raise RuntimeError("networkx is required to build the module graph")

    return _build_graph(Path(repo_path))


# ---------------------------------------------------------------------------

def _cluster_graph(
    graph: "nx.DiGraph",
    *,
    algorithm: str = "greedy",
    threshold: float = 0.1,
    use_semantic: bool = False,
    root: Path | None = None,
) -> dict[str, list[str]]:
    """Return communities of ``graph`` using :mod:`module_graph_analyzer`."""
    if nx is None:  # pragma: no cover - networkx missing
        raise RuntimeError("networkx is required to cluster modules")

    mapping = cluster_modules(
        graph,
        algorithm=algorithm,
        threshold=threshold,
        use_semantic=use_semantic,
        root=root,
    )

    groups: Dict[int, list[str]] = {}
    for mod, cid in mapping.items():
        groups.setdefault(cid, []).append(mod)

    return {str(k): sorted(v) for k, v in groups.items()}


def discover_module_groups(
    repo_path: str | Path,
    *,
    algorithm: str = "greedy",
    threshold: float = 0.1,
    use_semantic: bool = False,
) -> dict[str, list[str]]:
    """Analyse ``repo_path`` and return groups of related modules."""
    graph = build_import_graph(repo_path)
    return _cluster_graph(
        graph,
        algorithm=algorithm,
        threshold=threshold,
        use_semantic=use_semantic,
        root=Path(repo_path),
    )


# ---------------------------------------------------------------------------

def build_module_map(
    repo_path: str | Path,
    *,
    algorithm: str = "greedy",
    threshold: float = 0.1,
    use_semantic: bool = False,
) -> dict[str, list[str]]:
    """Build and persist a module grouping map under ``sandbox_data``."""
    root = Path(repo_path)
    mapping = discover_module_groups(
        root,
        algorithm=algorithm,
        threshold=threshold,
        use_semantic=use_semantic,
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


__all__ = ["build_import_graph", "discover_module_groups", "build_module_map"]

