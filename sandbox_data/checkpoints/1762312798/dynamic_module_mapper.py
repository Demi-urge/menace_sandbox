from __future__ import annotations
"""Compatibility wrappers around :mod:`module_graph_analyzer`."""

import json
from pathlib import Path
from typing import Dict, Iterable

from orphan_analyzer import analyze_redundancy

from module_graph_analyzer import build_import_graph, cluster_modules
from dynamic_path_router import resolve_path, resolve_module_path


def discover_module_groups(
    repo_path: str | Path,
    *,
    algorithm: str = "greedy",
    threshold: float = 0.1,
    use_semantic: bool = False,
) -> dict[str, list[str]]:
    """Return groups of related modules under ``repo_path``."""
    root = Path(resolve_path(repo_path))
    mapping = cluster_modules(
        build_import_graph(root),
        algorithm=algorithm,
        threshold=threshold,
        use_semantic=use_semantic,
        root=root,
    )
    groups: Dict[int, list[str]] = {}
    for mod, cid in mapping.items():
        path = resolve_module_path(mod.replace("/", "."))
        if "/" in mod:
            groups.setdefault(cid, []).append(mod)
            continue
        try:
            if analyze_redundancy(path):
                continue
        except Exception:
            pass
        groups.setdefault(cid, []).append(mod)
    return {str(k): sorted(v) for k, v in groups.items()}


def dotify_groups(groups: dict[str, list[str]]) -> dict[str, list[str]]:
    """Convert module paths to dotted import form."""
    return {k: [m.replace("/", ".") for m in mods] for k, mods in groups.items()}


def build_module_map(
    repo_path: str | Path,
    *,
    algorithm: str = "greedy",
    threshold: float = 0.1,
    use_semantic: bool = False,
    ignore: Iterable[str] | None = None,
) -> dict[str, int]:
    """Persist a module grouping map under ``sandbox_data``."""
    root = Path(resolve_path(repo_path))
    mapping = cluster_modules(
        build_import_graph(root, ignore=ignore),
        algorithm=algorithm,
        threshold=threshold,
        use_semantic=use_semantic,
        root=root,
    )
    if mapping and len(mapping) > 1 and algorithm != "hdbscan":
        filtered: Dict[str, int] = {}
        for mod, grp in mapping.items():
            path = resolve_module_path(mod.replace("/", "."))
            if "/" in mod:
                filtered[mod] = grp
                continue
            try:
                if analyze_redundancy(path):
                    continue
            except Exception:
                pass
            filtered[mod] = grp
        mapping = filtered
    out_dir = (root / "sandbox_data").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "module_map.json"
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(mapping, fh, indent=2)
    return mapping


def main(args: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Discover module groups")
    parser.add_argument("repo", nargs="?", default=".", help="Repository root")
    parser.add_argument(
        "--algorithm",
        default="greedy",
        choices=["greedy", "label", "hdbscan"],
    )
    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument("--semantic", action="store_true", help="Use docstring similarity")
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Glob pattern of directories to exclude",
    )
    opts = parser.parse_args(args)

    build_module_map(
        opts.repo,
        algorithm=opts.algorithm,
        threshold=opts.threshold,
        use_semantic=opts.semantic,
        ignore=opts.exclude,
    )


if __name__ == "__main__":  # pragma: no cover
    main()

__all__ = [
    "build_import_graph",
    "build_module_map",
    "discover_module_groups",
    "dotify_groups",
]
