#!/usr/bin/env python3
"""Generate a module grouping map using the dependency graph."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from menace.module_mapper import (
    build_module_graph,
    cluster_modules,
    save_module_map,
)


# ---------------------------------------------------------------------------

def generate_module_map(
    output: Path,
    *,
    root: Path = Path("."),
    algorithm: str = "greedy",
    threshold: float = 0.1,
    semantic: bool = False,
) -> dict[str, int]:
    graph = build_module_graph(root)
    clusters = cluster_modules(graph)
    mapping = {mod: cid for cid, mods in clusters.items() for mod in mods}
    save_module_map(mapping, output)
    return mapping


# ---------------------------------------------------------------------------

def main(args: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate sandbox module map")
    parser.add_argument("--root", default=".", help="Repository root")
    parser.add_argument("--output", default="sandbox_data/module_map.json")
    parser.add_argument("--algorithm", default="greedy", choices=["greedy", "label"])
    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument("--semantic", action="store_true", help="Use docstring similarity")
    opts = parser.parse_args(args)

    generate_module_map(
        output=Path(opts.output),
        root=Path(opts.root),
        algorithm=opts.algorithm,
        threshold=opts.threshold,
        semantic=opts.semantic,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
