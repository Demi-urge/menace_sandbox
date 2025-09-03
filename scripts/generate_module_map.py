#!/usr/bin/env python3
"""Generate a module grouping map using the dynamic module mapper."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dynamic_module_mapper import build_module_map as _build_map
from dynamic_path_router import resolve_path


# ---------------------------------------------------------------------------

def generate_module_map(
    output: Path,
    *,
    root: Path = Path("."),
    algorithm: str = "greedy",
    threshold: float = 0.1,
    semantic: bool = False,
    exclude: list[str] | None = None,
) -> dict[str, int]:
    mapping = _build_map(
        root,
        algorithm=algorithm,
        threshold=threshold,
        use_semantic=semantic,
        ignore=exclude,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(mapping, indent=2))
    return mapping


# ---------------------------------------------------------------------------

def main(args: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build sandbox module map")
    parser.add_argument("repo", nargs="?", default=".", help="Repository path")
    parser.add_argument(
        "--output",
        default=Path(resolve_path("sandbox_data")) / "module_map.json",
    )
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

    generate_module_map(
        output=Path(opts.output),
        root=Path(opts.repo),
        algorithm=opts.algorithm,
        threshold=opts.threshold,
        semantic=opts.semantic,
        exclude=opts.exclude,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
